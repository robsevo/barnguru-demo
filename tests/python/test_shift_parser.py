"""Tests for data.shift_parser — Shift data parser + TOI aggregation.

Covers the full testing contract:
  - Input validation (bad types, missing keys)
  - Known-answer regression (frozen inputs → frozen outputs)
  - Output range (TOI non-negative, secs within period bounds, breakdown sums)
  - Edge cases (empty data, missing fields, 0-second shifts, OT shifts)
  - Integration smoke (to_polars schema, downstream FI engine consumption)
Also covers the new NHLClient.get_shifts endpoint.
"""

import warnings

import httpx
import polars as pl
import pytest
import respx

from data.nhl_client import NHLClient, NHLApiError
from data.pbp_parser import DataMissingWarning
from data.shift_parser import (
    PlayerTOI,
    ShiftParser,
    ShiftRecord,
    _period_offset_secs,
)


# ---------------------------------------------------------------------------
# Fixtures — frozen raw shift dicts
# ---------------------------------------------------------------------------

_GAME_ID = 2024020001

_RAW_SHIFT_MATTHEWS_P1 = {
    "gameId": _GAME_ID,
    "playerId": 8480801,
    "teamId": 10,
    "period": 1,
    "shiftNumber": 1,
    "startTime": "00:00",
    "endTime": "01:23",
    "duration": "01:23",
    "teamAbbrev": "TOR",
    "firstName": "Auston",
    "lastName": "Matthews",
}

_RAW_SHIFT_MATTHEWS_P1_B = {
    "gameId": _GAME_ID,
    "playerId": 8480801,
    "teamId": 10,
    "period": 1,
    "shiftNumber": 2,
    "startTime": "03:00",
    "endTime": "04:15",
    "duration": "01:15",
    "teamAbbrev": "TOR",
    "firstName": "Auston",
    "lastName": "Matthews",
}

_RAW_SHIFT_MATTHEWS_P2 = {
    "gameId": _GAME_ID,
    "playerId": 8480801,
    "teamId": 10,
    "period": 2,
    "shiftNumber": 3,
    "startTime": "02:00",
    "endTime": "03:30",
    "duration": "01:30",
    "teamAbbrev": "TOR",
    "firstName": "Auston",
    "lastName": "Matthews",
}

_RAW_SHIFT_MATTHEWS_P3 = {
    "gameId": _GAME_ID,
    "playerId": 8480801,
    "teamId": 10,
    "period": 3,
    "shiftNumber": 4,
    "startTime": "05:00",
    "endTime": "06:30",
    "duration": "01:30",
    "teamAbbrev": "TOR",
    "firstName": "Auston",
    "lastName": "Matthews",
}

_RAW_SHIFT_MATTHEWS_OT = {
    "gameId": _GAME_ID,
    "playerId": 8480801,
    "teamId": 10,
    "period": 4,
    "shiftNumber": 5,
    "startTime": "00:00",
    "endTime": "02:14",
    "duration": "02:14",
    "teamAbbrev": "TOR",
    "firstName": "Auston",
    "lastName": "Matthews",
}

_RAW_SHIFT_RIELLY_P1 = {
    "gameId": _GAME_ID,
    "playerId": 8480803,
    "teamId": 10,
    "period": 1,
    "shiftNumber": 1,
    "startTime": "00:00",
    "endTime": "02:30",
    "duration": "02:30",
    "teamAbbrev": "TOR",
    "firstName": "Morgan",
    "lastName": "Rielly",
}

_FULL_SHIFTS_RESPONSE = {
    "data": [
        _RAW_SHIFT_MATTHEWS_P1,
        _RAW_SHIFT_MATTHEWS_P1_B,
        _RAW_SHIFT_MATTHEWS_P2,
        _RAW_SHIFT_MATTHEWS_P3,
        _RAW_SHIFT_RIELLY_P1,
    ],
    "total": 5,
}

_SHIFTS_RESPONSE_MOCK = {
    "data": [_RAW_SHIFT_MATTHEWS_P1],
    "total": 1,
}


# ---------------------------------------------------------------------------
# Period offset helper tests
# ---------------------------------------------------------------------------


class TestPeriodOffsetSecs:
    def test_period_1_offset_zero(self):
        assert _period_offset_secs(1) == 0

    def test_period_2_offset_1200(self):
        assert _period_offset_secs(2) == 1200

    def test_period_3_offset_2400(self):
        assert _period_offset_secs(3) == 2400

    def test_period_4_ot_offset_3600(self):
        assert _period_offset_secs(4) == 3600

    def test_period_5_ot_offset_3900(self):
        # second OT in reg season (5-min periods)
        assert _period_offset_secs(5) == 3900


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="Expected dict"):
            ShiftParser().parse("not a dict")  # type: ignore[arg-type]

    def test_missing_data_key_raises(self):
        with pytest.raises(ValueError, match="missing required 'data' key"):
            ShiftParser().parse({"total": 0})

    def test_data_not_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            ShiftParser().parse({"data": "bad"})

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            ShiftParser().parse({})

    def test_data_empty_list_ok(self):
        result = ShiftParser().parse({"data": [], "total": 0})
        assert result == []


# ---------------------------------------------------------------------------
# 2. Known-answer regression tests
# ---------------------------------------------------------------------------


class TestKnownAnswerSingleShift:
    def setup_method(self):
        self.shift = ShiftParser().parse({"data": [_RAW_SHIFT_MATTHEWS_P1]})[0]

    def test_is_shift_record(self):
        assert isinstance(self.shift, ShiftRecord)

    def test_game_id(self):
        assert self.shift.game_id == _GAME_ID

    def test_player_id(self):
        assert self.shift.player_id == 8480801

    def test_team_id(self):
        assert self.shift.team_id == 10

    def test_period(self):
        assert self.shift.period == 1

    def test_shift_number(self):
        assert self.shift.shift_number == 1

    def test_start_time_string(self):
        assert self.shift.start_time == "00:00"

    def test_end_time_string(self):
        assert self.shift.end_time == "01:23"

    def test_duration_secs(self):
        assert self.shift.duration_secs == 83  # 1*60 + 23

    def test_start_time_secs(self):
        assert self.shift.start_time_secs == 0

    def test_end_time_secs(self):
        assert self.shift.end_time_secs == 83

    def test_game_seconds_start_period1(self):
        assert self.shift.game_seconds_start == 0  # offset 0 + 0

    def test_game_seconds_end_period1(self):
        assert self.shift.game_seconds_end == 83

    def test_raw_preserved(self):
        assert self.shift.raw is _RAW_SHIFT_MATTHEWS_P1 or self.shift.raw == _RAW_SHIFT_MATTHEWS_P1


class TestKnownAnswerPeriod2Shift:
    def setup_method(self):
        self.shift = ShiftParser().parse({"data": [_RAW_SHIFT_MATTHEWS_P2]})[0]

    def test_period(self):
        assert self.shift.period == 2

    def test_duration_secs(self):
        assert self.shift.duration_secs == 90  # 1*60+30

    def test_start_time_secs(self):
        assert self.shift.start_time_secs == 120  # 2*60

    def test_game_seconds_start(self):
        # period 2 offset = 1200, start 2:00 = 120s
        assert self.shift.game_seconds_start == 1320

    def test_game_seconds_end(self):
        # 1200 + 120 + 90 = 1410
        assert self.shift.game_seconds_end == 1410


class TestKnownAnswerOTShift:
    def setup_method(self):
        self.shift = ShiftParser().parse({"data": [_RAW_SHIFT_MATTHEWS_OT]})[0]

    def test_period(self):
        assert self.shift.period == 4

    def test_duration_secs(self):
        assert self.shift.duration_secs == 134  # 2*60+14

    def test_game_seconds_start(self):
        # period 4 offset = 3600, start 0:00 → 3600
        assert self.shift.game_seconds_start == 3600

    def test_game_seconds_end(self):
        assert self.shift.game_seconds_end == 3734  # 3600 + 134


class TestKnownAnswerTOI:
    def setup_method(self):
        self.parser = ShiftParser()
        self.shifts = self.parser.parse(_FULL_SHIFTS_RESPONSE)
        self.toi = self.parser.aggregate_toi(self.shifts)
        self.matthews = next(t for t in self.toi if t.player_id == 8480801)
        self.rielly = next(t for t in self.toi if t.player_id == 8480803)

    def test_two_players_in_toi(self):
        assert len(self.toi) == 2

    def test_matthews_total_toi(self):
        # P1: 83s + 75s = 158s, P2: 90s, P3: 90s → total 338s
        assert self.matthews.toi_total_secs == 338

    def test_matthews_p1_toi(self):
        assert self.matthews.toi_p1_secs == 158  # 83 + 75

    def test_matthews_p2_toi(self):
        assert self.matthews.toi_p2_secs == 90

    def test_matthews_p3_toi(self):
        assert self.matthews.toi_p3_secs == 90

    def test_matthews_ot_toi(self):
        assert self.matthews.toi_ot_secs == 0  # no OT shift in fixture

    def test_matthews_shift_count(self):
        assert self.matthews.shift_count == 4

    def test_matthews_avg_shift_secs(self):
        assert self.matthews.avg_shift_secs == pytest.approx(338 / 4)

    def test_rielly_total_toi(self):
        assert self.rielly.toi_total_secs == 150  # 2:30 = 150s

    def test_rielly_p1_toi(self):
        assert self.rielly.toi_p1_secs == 150

    def test_rielly_shift_count(self):
        assert self.rielly.shift_count == 1

    def test_rielly_avg_shift_secs(self):
        assert self.rielly.avg_shift_secs == pytest.approx(150.0)

    def test_toi_breakdown_sums_to_total(self):
        for t in self.toi:
            period_sum = t.toi_p1_secs + t.toi_p2_secs + t.toi_p3_secs + t.toi_ot_secs
            assert period_sum == t.toi_total_secs, (
                f"player_id={t.player_id}: period breakdown {period_sum} ≠ total {t.toi_total_secs}"
            )


class TestKnownAnswerOTInTOI:
    def test_ot_shift_goes_to_toi_ot(self):
        parser = ShiftParser()
        shifts = parser.parse({"data": [_RAW_SHIFT_MATTHEWS_OT]})
        toi = parser.aggregate_toi(shifts)
        assert len(toi) == 1
        assert toi[0].toi_ot_secs == 134
        assert toi[0].toi_p1_secs == 0
        assert toi[0].toi_p2_secs == 0
        assert toi[0].toi_p3_secs == 0
        assert toi[0].toi_total_secs == 134


# ---------------------------------------------------------------------------
# 3. Output range tests
# ---------------------------------------------------------------------------


class TestOutputRanges:
    def setup_method(self):
        parser = ShiftParser()
        self.shifts = parser.parse(_FULL_SHIFTS_RESPONSE)
        self.toi = parser.aggregate_toi(self.shifts)

    def test_duration_secs_non_negative(self):
        for s in self.shifts:
            assert s.duration_secs >= 0

    def test_start_time_secs_non_negative(self):
        for s in self.shifts:
            assert s.start_time_secs >= 0

    def test_game_seconds_start_non_negative(self):
        for s in self.shifts:
            assert s.game_seconds_start >= 0

    def test_game_seconds_end_gte_start(self):
        for s in self.shifts:
            assert s.game_seconds_end >= s.game_seconds_start

    def test_period_times_in_regulation_bounds(self):
        for s in self.shifts:
            if s.period <= 3:
                assert s.start_time_secs <= 1200
                assert s.end_time_secs <= 1200

    def test_toi_total_non_negative(self):
        for t in self.toi:
            assert t.toi_total_secs >= 0

    def test_toi_period_buckets_non_negative(self):
        for t in self.toi:
            assert t.toi_p1_secs >= 0
            assert t.toi_p2_secs >= 0
            assert t.toi_p3_secs >= 0
            assert t.toi_ot_secs >= 0

    def test_shift_count_positive(self):
        for t in self.toi:
            assert t.shift_count > 0

    def test_avg_shift_secs_positive(self):
        for t in self.toi:
            assert t.avg_shift_secs > 0

    def test_toi_realistic_upper_bound(self):
        # No player plays more than 60 minutes per 60-min regulation game
        for t in self.toi:
            assert t.toi_total_secs <= 3900  # reg + 5-min OT max


# ---------------------------------------------------------------------------
# 4. Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_data_returns_empty_list(self):
        result = ShiftParser().parse({"data": []})
        assert result == []

    def test_non_dict_row_emits_warning_and_is_skipped(self):
        raw = {"data": ["bad", _RAW_SHIFT_MATTHEWS_P1]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shifts = ShiftParser().parse(raw)
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert len(shifts) == 1

    def test_missing_player_id_emits_warning_row_skipped(self):
        row = {k: v for k, v in _RAW_SHIFT_MATTHEWS_P1.items() if k != "playerId"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shifts = ShiftParser().parse({"data": [row]})
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert shifts == []

    def test_missing_period_emits_warning_row_skipped(self):
        row = {k: v for k, v in _RAW_SHIFT_MATTHEWS_P1.items() if k != "period"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shifts = ShiftParser().parse({"data": [row]})
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert shifts == []

    def test_missing_duration_emits_warning_derived_from_times(self):
        row = {k: v for k, v in _RAW_SHIFT_MATTHEWS_P1.items() if k != "duration"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shifts = ShiftParser().parse({"data": [row]})
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert len(shifts) == 1
        assert shifts[0].duration_secs == 83  # derived from 01:23 - 00:00

    def test_zero_second_shift_is_valid(self):
        row = {**_RAW_SHIFT_MATTHEWS_P1, "startTime": "05:00", "endTime": "05:00", "duration": "00:00"}
        shifts = ShiftParser().parse({"data": [row]})
        assert len(shifts) == 1
        assert shifts[0].duration_secs == 0

    def test_malformed_time_emits_warning_defaults_zero(self):
        row = {**_RAW_SHIFT_MATTHEWS_P1, "startTime": "bad"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shifts = ShiftParser().parse({"data": [row]})
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert shifts[0].start_time_secs == 0

    def test_aggregate_toi_empty_list(self):
        toi = ShiftParser.aggregate_toi([])
        assert toi == []

    def test_aggregate_toi_zero_shift_count_avg_zero(self):
        # Manually construct a PlayerTOI with 0 shifts (edge case in avg computation)
        # aggregate_toi only produces entries from actual shifts so shift_count >= 1;
        # test the guard via a single 0-second shift instead
        row = {**_RAW_SHIFT_MATTHEWS_P1, "startTime": "05:00", "endTime": "05:00", "duration": "00:00"}
        shifts = ShiftParser().parse({"data": [row]})
        toi = ShiftParser.aggregate_toi(shifts)
        assert len(toi) == 1
        assert toi[0].avg_shift_secs == 0.0
        assert toi[0].shift_count == 1

    def test_multiple_players_same_game(self):
        raw = {"data": [_RAW_SHIFT_MATTHEWS_P1, _RAW_SHIFT_RIELLY_P1]}
        shifts = ShiftParser().parse(raw)
        toi = ShiftParser.aggregate_toi(shifts)
        player_ids = {t.player_id for t in toi}
        assert player_ids == {8480801, 8480803}

    def test_same_player_multiple_shifts_accumulate(self):
        raw = {"data": [_RAW_SHIFT_MATTHEWS_P1, _RAW_SHIFT_MATTHEWS_P1_B]}
        shifts = ShiftParser().parse(raw)
        toi = ShiftParser.aggregate_toi(shifts)
        assert len(toi) == 1
        assert toi[0].toi_total_secs == 83 + 75  # 01:23 + 01:15
        assert toi[0].shift_count == 2

    def test_none_duration_field_derives_from_times(self):
        row = {**_RAW_SHIFT_MATTHEWS_P1, "duration": None}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shifts = ShiftParser().parse({"data": [row]})
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert shifts[0].duration_secs == 83

    def test_null_start_time_treated_as_zero(self):
        row = {**_RAW_SHIFT_MATTHEWS_P1, "startTime": None}
        shifts = ShiftParser().parse({"data": [row]})
        assert shifts[0].start_time_secs == 0

    def test_extra_unknown_fields_in_row_ignored(self):
        row = {**_RAW_SHIFT_MATTHEWS_P1, "unknownField": "whatever"}
        shifts = ShiftParser().parse({"data": [row]})
        assert len(shifts) == 1

    def test_total_field_not_required(self):
        # Some API responses may omit 'total'
        raw = {"data": [_RAW_SHIFT_MATTHEWS_P1]}
        shifts = ShiftParser().parse(raw)
        assert len(shifts) == 1

    def test_period_5_ot_goes_to_ot_bucket(self):
        row = {**_RAW_SHIFT_MATTHEWS_OT, "period": 5}
        shifts = ShiftParser().parse({"data": [row]})
        toi = ShiftParser.aggregate_toi(shifts)
        assert toi[0].toi_ot_secs == toi[0].toi_total_secs
        assert toi[0].toi_p1_secs == 0


# ---------------------------------------------------------------------------
# 5. to_polars tests
# ---------------------------------------------------------------------------


class TestToPolarsShifts:
    def setup_method(self):
        parser = ShiftParser()
        self.shifts = parser.parse(_FULL_SHIFTS_RESPONSE)
        self.df = ShiftParser.to_polars_shifts(self.shifts)

    def test_returns_polars_dataframe(self):
        assert isinstance(self.df, pl.DataFrame)

    def test_row_count(self):
        assert len(self.df) == len(self.shifts)

    def test_expected_columns(self):
        expected = {
            "game_id", "player_id", "team_id", "period", "shift_number",
            "start_time", "end_time", "duration_secs",
            "start_time_secs", "end_time_secs",
            "game_seconds_start", "game_seconds_end",
        }
        assert expected.issubset(set(self.df.columns))

    def test_duration_secs_non_negative(self):
        assert (self.df["duration_secs"] >= 0).all()

    def test_game_seconds_end_gte_start(self):
        assert (self.df["game_seconds_end"] >= self.df["game_seconds_start"]).all()

    def test_empty_shifts_returns_empty_df(self):
        df = ShiftParser.to_polars_shifts([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0


class TestToPolarisTOI:
    def setup_method(self):
        parser = ShiftParser()
        shifts = parser.parse(_FULL_SHIFTS_RESPONSE)
        self.toi = parser.aggregate_toi(shifts)
        self.df = ShiftParser.to_polars_toi(self.toi)

    def test_returns_polars_dataframe(self):
        assert isinstance(self.df, pl.DataFrame)

    def test_row_count(self):
        assert len(self.df) == 2  # Matthews + Rielly

    def test_expected_columns(self):
        expected = {
            "game_id", "player_id", "team_id",
            "toi_total_secs", "toi_p1_secs", "toi_p2_secs", "toi_p3_secs", "toi_ot_secs",
            "shift_count", "avg_shift_secs",
        }
        assert expected.issubset(set(self.df.columns))

    def test_all_toi_non_negative(self):
        for col in ["toi_total_secs", "toi_p1_secs", "toi_p2_secs", "toi_p3_secs", "toi_ot_secs"]:
            assert (self.df[col] >= 0).all()

    def test_period_breakdown_sums_to_total(self):
        computed = (
            self.df["toi_p1_secs"]
            + self.df["toi_p2_secs"]
            + self.df["toi_p3_secs"]
            + self.df["toi_ot_secs"]
        )
        assert (computed == self.df["toi_total_secs"]).all()

    def test_empty_toi_returns_empty_df(self):
        df = ShiftParser.to_polars_toi([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# 6. NHLClient.get_shifts — known-answer + error handling
# ---------------------------------------------------------------------------

_SHIFTS_API_URL = "https://api.nhle.com/stats/rest/en/shiftcharts"


class TestNHLClientGetShifts:
    def test_invalid_game_id_zero_raises(self):
        async def run():
            async with NHLClient() as c:
                with pytest.raises(ValueError):
                    await c.get_shifts(0)

        import asyncio
        asyncio.get_event_loop().run_until_complete(run())

    def test_invalid_game_id_negative_raises(self):
        async def run():
            async with NHLClient() as c:
                with pytest.raises(ValueError):
                    await c.get_shifts(-1)

        import asyncio
        asyncio.get_event_loop().run_until_complete(run())

    @respx.mock
    async def test_get_shifts_returns_dict_with_data(self):
        respx.get(_SHIFTS_API_URL).mock(
            return_value=httpx.Response(200, json=_SHIFTS_RESPONSE_MOCK)
        )
        async with NHLClient() as c:
            result = await c.get_shifts(_GAME_ID)
        assert "data" in result
        assert isinstance(result["data"], list)
        assert result["data"][0]["playerId"] == 8480801

    @respx.mock
    async def test_get_shifts_empty_game_returns_empty_data(self):
        respx.get(_SHIFTS_API_URL).mock(
            return_value=httpx.Response(200, json={"data": [], "total": 0})
        )
        async with NHLClient() as c:
            result = await c.get_shifts(_GAME_ID)
        assert result["data"] == []

    @respx.mock
    async def test_get_shifts_404_raises_nhl_api_error(self):
        respx.get(_SHIFTS_API_URL).mock(return_value=httpx.Response(404))
        async with NHLClient() as c:
            with pytest.raises(NHLApiError) as exc_info:
                await c.get_shifts(_GAME_ID)
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# 7. Integration smoke — downstream FI engine consumption
# ---------------------------------------------------------------------------


class TestIntegrationSmoke:
    def test_shifts_feed_to_toi_and_df(self):
        """Smoke: parse → aggregate → polars pipeline works end-to-end."""
        parser = ShiftParser()
        shifts = parser.parse(_FULL_SHIFTS_RESPONSE)
        toi = parser.aggregate_toi(shifts)
        df = ShiftParser.to_polars_toi(toi)

        # FI engine (3.7) will look up TOI by player_id from a PlayerTOI list
        matthews_toi = next(t for t in toi if t.player_id == 8480801)
        assert matthews_toi.toi_total_secs > 0

        # FI engine can also query the Polars DataFrame
        row = df.filter(pl.col("player_id") == 8480801)
        assert len(row) == 1
        assert row["toi_total_secs"][0] == matthews_toi.toi_total_secs

    def test_shift_df_supports_player_filter(self):
        """Downstream: filter shifts by player for TOI-spike detection (3.7)."""
        parser = ShiftParser()
        shifts = parser.parse(_FULL_SHIFTS_RESPONSE)
        df = ShiftParser.to_polars_shifts(shifts)
        matthews_shifts = df.filter(pl.col("player_id") == 8480801)
        assert len(matthews_shifts) == 4

    def test_toi_total_in_minutes_for_reporting(self):
        """Fatigue engine can express TOI in minutes (TOI/60 is the NHL standard)."""
        parser = ShiftParser()
        shifts = parser.parse(_FULL_SHIFTS_RESPONSE)
        toi = parser.aggregate_toi(shifts)
        for t in toi:
            toi_minutes = t.toi_total_secs / 60.0
            assert 0.0 <= toi_minutes <= 65.0  # sane upper bound

    def test_combined_parse_and_aggregate_with_ot(self):
        """Smoke: full game with OT shift aggregates correctly."""
        parser = ShiftParser()
        all_raw = [
            _RAW_SHIFT_MATTHEWS_P1,
            _RAW_SHIFT_MATTHEWS_P2,
            _RAW_SHIFT_MATTHEWS_P3,
            _RAW_SHIFT_MATTHEWS_OT,
        ]
        shifts = parser.parse({"data": all_raw})
        toi = parser.aggregate_toi(shifts)
        m = toi[0]
        assert m.toi_ot_secs == 134
        assert m.toi_total_secs == 83 + 90 + 90 + 134
        assert m.toi_p1_secs + m.toi_p2_secs + m.toi_p3_secs + m.toi_ot_secs == m.toi_total_secs
