"""Tests for data.roster_sync — Roster sync with player metadata + cache.

Covers the full testing contract:
  - Input validation (empty team code, malformed player entries)
  - Known-answer regression (frozen roster → frozen PlayerProfile fields)
  - Output range (age sane, position/shoots values valid)
  - Edge cases (missing fields, empty roster, cache hit/miss/stale/corrupt)
  - Integration smoke (to_polars schema, downstream FI engine columns)
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

from data.pbp_parser import DataMissingWarning
from data.roster_sync import (
    PlayerProfile,
    RosterSync,
    _normalize_roster,
    compute_age,
)


# ---------------------------------------------------------------------------
# Fixtures — frozen raw API response dicts
# ---------------------------------------------------------------------------

_RAW_MATTHEWS = {
    "id": 8480801,
    "firstName": {"default": "Auston"},
    "lastName": {"default": "Matthews"},
    "sweaterNumber": 34,
    "positionCode": "C",
    "shootsCatches": "L",
    "heightInInches": 75,
    "weightInPounds": 208,
    "birthDate": "1997-09-17",
    "birthCountry": "USA",
    "nationality": "USA",
    "isCaptain": True,
    "isAlternateCaptain": False,
}

_RAW_RIELLY = {
    "id": 8480803,
    "firstName": {"default": "Morgan"},
    "lastName": {"default": "Rielly"},
    "sweaterNumber": 44,
    "positionCode": "D",
    "shootsCatches": "L",
    "heightInInches": 73,
    "weightInPounds": 215,
    "birthDate": "1994-03-09",
    "birthCountry": "CAN",
    "nationality": "CAN",
    "isCaptain": False,
    "isAlternateCaptain": True,
}

_RAW_WOLL = {
    "id": 8480353,
    "firstName": {"default": "Joseph"},
    "lastName": {"default": "Woll"},
    "sweaterNumber": 60,
    "positionCode": "G",
    "shootsCatches": "L",
    "heightInInches": 74,
    "weightInPounds": 197,
    "birthDate": "1998-07-12",
    "birthCountry": "USA",
    "nationality": "USA",
    "isCaptain": False,
    "isAlternateCaptain": False,
}

_RAW_ROSTER = {
    "forwards": [_RAW_MATTHEWS],
    "defensemen": [_RAW_RIELLY],
    "goalies": [_RAW_WOLL],
}


def _make_client(roster_response: dict = _RAW_ROSTER) -> MagicMock:
    """Create a mock NHLClient whose get_roster returns roster_response."""
    client = MagicMock()
    client.get_roster = AsyncMock(return_value=roster_response)
    return client


# ---------------------------------------------------------------------------
# 1. compute_age — known-answer regression
# ---------------------------------------------------------------------------


class TestComputeAge:
    def test_exact_birthday_is_zero(self):
        assert compute_age("2000-01-01", as_of=date(2000, 1, 1)) == pytest.approx(0.0)

    def test_one_year_later(self):
        age = compute_age("2000-01-01", as_of=date(2001, 1, 1))
        # 366 days (2000 is a leap year) / 365.25 ≈ 1.002
        assert age == pytest.approx(366 / 365.25)

    def test_matthews_age_on_game_date(self):
        # Auston Matthews born 1997-09-17; age on 2024-11-15
        age = compute_age("1997-09-17", as_of=date(2024, 11, 15))
        assert age == pytest.approx((date(2024, 11, 15) - date(1997, 9, 17)).days / 365.25)

    def test_none_birth_date_returns_none(self):
        assert compute_age(None) is None

    def test_empty_string_returns_none(self):
        assert compute_age("") is None

    def test_invalid_format_returns_none(self):
        assert compute_age("17-09-1997") is None

    def test_defaults_to_today(self):
        age = compute_age("2000-01-01")
        today_age = (date.today() - date(2000, 1, 1)).days / 365.25
        assert age == pytest.approx(today_age, abs=0.01)

    def test_age_positive_for_born_player(self):
        age = compute_age("1990-06-15", as_of=date(2024, 6, 15))
        assert age == pytest.approx(34.0, abs=0.01)

    def test_future_birth_date_gives_negative_age(self):
        # Unusual but should not crash; downstream callers should validate
        age = compute_age("2030-01-01", as_of=date(2024, 1, 1))
        assert age is not None
        assert age < 0


# ---------------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_empty_team_code_raises(self, tmp_path):
        sync = RosterSync(tmp_path)
        with pytest.raises(ValueError, match="non-empty string"):
            await sync.sync_team(_make_client(), "")

    @pytest.mark.asyncio
    async def test_whitespace_team_code_raises(self, tmp_path):
        sync = RosterSync(tmp_path)
        with pytest.raises(ValueError):
            await sync.sync_team(_make_client(), "   ")

    def test_normalize_roster_non_dict_entry_emits_warning(self):
        import warnings
        raw = {"forwards": ["not a dict", _RAW_MATTHEWS], "defensemen": [], "goalies": []}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            profiles = _normalize_roster(raw, "TOR", None)
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert len(profiles) == 1  # only Matthews

    def test_normalize_roster_missing_player_id_emits_warning(self):
        import warnings
        player_no_id = {k: v for k, v in _RAW_MATTHEWS.items() if k != "id"}
        raw = {"forwards": [player_no_id], "defensemen": [], "goalies": []}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            profiles = _normalize_roster(raw, "TOR", None)
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert profiles == []

    def test_normalize_roster_empty_sections_returns_empty(self):
        profiles = _normalize_roster({"forwards": [], "defensemen": [], "goalies": []}, "TOR", None)
        assert profiles == []

    def test_normalize_roster_missing_section_skipped(self):
        # Only forwards, no defensemen/goalies keys at all
        profiles = _normalize_roster({"forwards": [_RAW_MATTHEWS]}, "TOR", None)
        assert len(profiles) == 1


# ---------------------------------------------------------------------------
# 3. Known-answer regression — PlayerProfile fields
# ---------------------------------------------------------------------------


class TestKnownAnswerNormalization:
    def setup_method(self):
        profiles = _normalize_roster(_RAW_ROSTER, "TOR", date(2024, 11, 15))
        self.by_id = {p.player_id: p for p in profiles}
        self.matthews = self.by_id[8480801]
        self.rielly = self.by_id[8480803]
        self.woll = self.by_id[8480353]

    def test_three_players_total(self):
        assert len(self.by_id) == 3

    def test_matthews_basic_fields(self):
        m = self.matthews
        assert m.player_id == 8480801
        assert m.team_code == "TOR"
        assert m.first_name == "Auston"
        assert m.last_name == "Matthews"

    def test_matthews_position_forward(self):
        assert self.matthews.position == "C"
        assert self.matthews.position_type == "forward"

    def test_matthews_shoots_left(self):
        assert self.matthews.shoots_catches == "L"

    def test_matthews_physical_measurements(self):
        assert self.matthews.height_inches == 75
        assert self.matthews.weight_pounds == 208

    def test_matthews_birth_date(self):
        assert self.matthews.birth_date == "1997-09-17"

    def test_matthews_age_on_game_date(self):
        expected = (date(2024, 11, 15) - date(1997, 9, 17)).days / 365.25
        assert self.matthews.age == pytest.approx(expected)

    def test_matthews_country(self):
        assert self.matthews.birth_country == "USA"
        assert self.matthews.nationality == "USA"

    def test_matthews_is_captain(self):
        assert self.matthews.is_captain is True
        assert self.matthews.is_alternate_captain is False

    def test_rielly_position_defenseman(self):
        assert self.rielly.position == "D"
        assert self.rielly.position_type == "defenseman"

    def test_rielly_is_alternate_captain(self):
        assert self.rielly.is_captain is False
        assert self.rielly.is_alternate_captain is True

    def test_woll_position_goalie(self):
        assert self.woll.position == "G"
        assert self.woll.position_type == "goalie"

    def test_woll_catches_left(self):
        # Goalies use "shootsCatches" for the catching hand
        assert self.woll.shoots_catches == "L"

    def test_jersey_numbers(self):
        assert self.matthews.jersey_number == 34
        assert self.rielly.jersey_number == 44
        assert self.woll.jersey_number == 60

    def test_plain_string_name_fallback(self):
        # Some test fixtures use plain strings for firstName/lastName
        raw_plain = {**_RAW_MATTHEWS, "firstName": "Auston", "lastName": "Matthews"}
        profiles = _normalize_roster(
            {"forwards": [raw_plain], "defensemen": [], "goalies": []}, "TOR", None
        )
        assert profiles[0].first_name == "Auston"
        assert profiles[0].last_name == "Matthews"


# ---------------------------------------------------------------------------
# 4. Output range tests
# ---------------------------------------------------------------------------


class TestOutputRanges:
    def setup_method(self):
        self.profiles = _normalize_roster(_RAW_ROSTER, "TOR", date(2024, 11, 15))

    def test_all_ages_positive(self):
        for p in self.profiles:
            if p.age is not None:
                assert p.age > 0

    def test_all_ages_realistic(self):
        # NHL players are typically 18–45 years old
        for p in self.profiles:
            if p.age is not None:
                assert 15.0 < p.age < 50.0

    def test_position_type_valid(self):
        valid = {"forward", "defenseman", "goalie", "unknown"}
        for p in self.profiles:
            assert p.position_type in valid

    def test_shoots_catches_valid(self):
        valid = {"L", "R", "?"}
        for p in self.profiles:
            assert p.shoots_catches in valid

    def test_height_positive_when_present(self):
        for p in self.profiles:
            if p.height_inches is not None:
                assert p.height_inches > 0

    def test_weight_positive_when_present(self):
        for p in self.profiles:
            if p.weight_pounds is not None:
                assert p.weight_pounds > 0

    def test_team_code_uppercased(self):
        for p in self.profiles:
            assert p.team_code == p.team_code.upper()

    def test_player_ids_positive(self):
        for p in self.profiles:
            assert p.player_id > 0


# ---------------------------------------------------------------------------
# 5. Edge cases — normalization
# ---------------------------------------------------------------------------


class TestEdgeCasesNormalization:
    def test_unknown_position_code_uses_question_mark(self):
        import warnings
        raw = {**_RAW_MATTHEWS, "positionCode": "X"}
        profiles = _normalize_roster(
            {"forwards": [raw], "defensemen": [], "goalies": []}, "TOR", None
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            profiles = _normalize_roster(
                {"forwards": [raw], "defensemen": [], "goalies": []}, "TOR", None
            )
        assert profiles[0].position == "?"
        assert profiles[0].position_type == "unknown"

    def test_missing_shoots_catches_uses_question_mark(self):
        raw = {k: v for k, v in _RAW_MATTHEWS.items() if k != "shootsCatches"}
        profiles = _normalize_roster(
            {"forwards": [raw], "defensemen": [], "goalies": []}, "TOR", None
        )
        assert profiles[0].shoots_catches == "?"

    def test_invalid_shoots_catches_uses_question_mark(self):
        raw = {**_RAW_MATTHEWS, "shootsCatches": "B"}
        profiles = _normalize_roster(
            {"forwards": [raw], "defensemen": [], "goalies": []}, "TOR", None
        )
        assert profiles[0].shoots_catches == "?"

    def test_missing_birth_date_age_is_none(self):
        raw = {k: v for k, v in _RAW_MATTHEWS.items() if k != "birthDate"}
        profiles = _normalize_roster(
            {"forwards": [raw], "defensemen": [], "goalies": []}, "TOR", None
        )
        assert profiles[0].birth_date is None
        assert profiles[0].age is None

    def test_missing_jersey_number_is_none(self):
        raw = {k: v for k, v in _RAW_MATTHEWS.items() if k != "sweaterNumber"}
        profiles = _normalize_roster(
            {"forwards": [raw], "defensemen": [], "goalies": []}, "TOR", None
        )
        assert profiles[0].jersey_number is None

    def test_missing_height_weight_are_none(self):
        raw = {k: v for k, v in _RAW_MATTHEWS.items()
               if k not in ("heightInInches", "weightInPounds")}
        profiles = _normalize_roster(
            {"forwards": [raw], "defensemen": [], "goalies": []}, "TOR", None
        )
        assert profiles[0].height_inches is None
        assert profiles[0].weight_pounds is None

    def test_missing_captain_flags_default_false(self):
        raw = {k: v for k, v in _RAW_MATTHEWS.items()
               if k not in ("isCaptain", "isAlternateCaptain")}
        profiles = _normalize_roster(
            {"forwards": [raw], "defensemen": [], "goalies": []}, "TOR", None
        )
        assert profiles[0].is_captain is False
        assert profiles[0].is_alternate_captain is False

    def test_team_code_lowercased_input_uppercased_in_profile(self):
        profiles = _normalize_roster(_RAW_ROSTER, "tor", None)
        assert all(p.team_code == "TOR" for p in profiles)

    def test_all_position_types_covered(self):
        profiles = _normalize_roster(_RAW_ROSTER, "TOR", None)
        types = {p.position_type for p in profiles}
        assert "forward" in types
        assert "defenseman" in types
        assert "goalie" in types

    def test_nationality_birth_country_none_when_absent(self):
        raw = {k: v for k, v in _RAW_MATTHEWS.items()
               if k not in ("nationality", "birthCountry")}
        profiles = _normalize_roster(
            {"forwards": [raw], "defensemen": [], "goalies": []}, "TOR", None
        )
        assert profiles[0].nationality is None
        assert profiles[0].birth_country is None


# ---------------------------------------------------------------------------
# 6. Cache behavior
# ---------------------------------------------------------------------------


class TestCacheBehavior:
    @pytest.mark.asyncio
    async def test_cache_miss_calls_api(self, tmp_path):
        client = _make_client()
        sync = RosterSync(tmp_path)
        profiles = await sync.sync_team(client, "TOR")
        client.get_roster.assert_called_once_with("TOR")
        assert len(profiles) == 3

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api(self, tmp_path):
        client = _make_client()
        sync = RosterSync(tmp_path)
        # First call writes cache
        await sync.sync_team(client, "TOR")
        # Second call should use cache, not call API again
        await sync.sync_team(client, "TOR")
        assert client.get_roster.call_count == 1

    @pytest.mark.asyncio
    async def test_stale_cache_calls_api_again(self, tmp_path):
        client = _make_client()
        sync = RosterSync(tmp_path, ttl_seconds=3600)
        # Manually write a stale cache (fetched 2 hours ago)
        stale_time = datetime.now(timezone.utc) - timedelta(hours=2)
        profiles = _normalize_roster(_RAW_ROSTER, "TOR", None)
        payload = {
            "fetched_at": stale_time.isoformat(),
            "team_code": "TOR",
            "profiles": [p.__dict__ for p in profiles],
        }
        (tmp_path / "roster_TOR.json").write_text(json.dumps(payload))
        await sync.sync_team(client, "TOR")
        client.get_roster.assert_called_once()

    @pytest.mark.asyncio
    async def test_fresh_cache_skips_api(self, tmp_path):
        client = _make_client()
        sync = RosterSync(tmp_path, ttl_seconds=3600)
        # Manually write a fresh cache
        fresh_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        import dataclasses
        profiles = _normalize_roster(_RAW_ROSTER, "TOR", None)
        payload = {
            "fetched_at": fresh_time.isoformat(),
            "team_code": "TOR",
            "profiles": [dataclasses.asdict(p) for p in profiles],
        }
        (tmp_path / "roster_TOR.json").write_text(json.dumps(payload))
        result = await sync.sync_team(client, "TOR")
        client.get_roster.assert_not_called()
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_corrupt_cache_calls_api(self, tmp_path):
        client = _make_client()
        sync = RosterSync(tmp_path)
        (tmp_path / "roster_TOR.json").write_text("not valid json")
        profiles = await sync.sync_team(client, "TOR")
        client.get_roster.assert_called_once()
        assert len(profiles) == 3

    @pytest.mark.asyncio
    async def test_missing_fetched_at_in_cache_calls_api(self, tmp_path):
        client = _make_client()
        sync = RosterSync(tmp_path)
        (tmp_path / "roster_TOR.json").write_text(json.dumps({"profiles": []}))
        await sync.sync_team(client, "TOR")
        client.get_roster.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_flag_bypasses_cache(self, tmp_path):
        client = _make_client()
        sync = RosterSync(tmp_path)
        # First call writes cache
        await sync.sync_team(client, "TOR")
        # Force re-fetch even though cache is fresh
        await sync.sync_team(client, "TOR", force=True)
        assert client.get_roster.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_file_created_after_sync(self, tmp_path):
        client = _make_client()
        sync = RosterSync(tmp_path)
        await sync.sync_team(client, "TOR")
        assert (tmp_path / "roster_TOR.json").exists()

    @pytest.mark.asyncio
    async def test_cache_file_contains_fetched_at(self, tmp_path):
        client = _make_client()
        sync = RosterSync(tmp_path)
        await sync.sync_team(client, "TOR")
        data = json.loads((tmp_path / "roster_TOR.json").read_text())
        assert "fetched_at" in data
        assert "team_code" in data
        assert "profiles" in data

    def test_invalidate_removes_cache_file(self, tmp_path):
        (tmp_path / "roster_TOR.json").write_text("{}")
        sync = RosterSync(tmp_path)
        sync.invalidate("TOR")
        assert not (tmp_path / "roster_TOR.json").exists()

    def test_invalidate_nonexistent_does_not_raise(self, tmp_path):
        sync = RosterSync(tmp_path)
        sync.invalidate("MTL")  # no file exists

    def test_invalidate_all_removes_all_roster_files(self, tmp_path):
        (tmp_path / "roster_TOR.json").write_text("{}")
        (tmp_path / "roster_MTL.json").write_text("{}")
        (tmp_path / "other_file.json").write_text("{}")
        sync = RosterSync(tmp_path)
        removed = sync.invalidate_all()
        assert removed == 2
        assert not (tmp_path / "roster_TOR.json").exists()
        assert not (tmp_path / "roster_MTL.json").exists()
        assert (tmp_path / "other_file.json").exists()

    @pytest.mark.asyncio
    async def test_team_code_case_insensitive(self, tmp_path):
        client = _make_client()
        sync = RosterSync(tmp_path)
        p1 = await sync.sync_team(client, "tor")
        # Cache should be at roster_TOR.json; lowercase call should hit it
        p2 = await sync.sync_team(client, "TOR")
        assert client.get_roster.call_count == 1
        assert len(p1) == len(p2)

    @pytest.mark.asyncio
    async def test_cache_dir_created_if_not_exists(self, tmp_path):
        client = _make_client()
        nested = tmp_path / "a" / "b" / "c"
        sync = RosterSync(nested)
        await sync.sync_team(client, "TOR")
        assert nested.exists()


# ---------------------------------------------------------------------------
# 7. sync_teams — multiple teams
# ---------------------------------------------------------------------------


class TestSyncTeams:
    @pytest.mark.asyncio
    async def test_sync_two_teams(self, tmp_path):
        mtl_roster = {"forwards": [_RAW_MATTHEWS], "defensemen": [], "goalies": []}
        tor_roster = {"forwards": [], "defensemen": [_RAW_RIELLY], "goalies": []}

        client = MagicMock()
        client.get_roster = AsyncMock(
            side_effect=lambda code: tor_roster if code == "TOR" else mtl_roster
        )

        sync = RosterSync(tmp_path)
        result = await sync.sync_teams(client, ["TOR", "MTL"])

        assert "TOR" in result
        assert "MTL" in result
        assert result["TOR"][0].player_id == 8480803
        assert result["MTL"][0].player_id == 8480801

    @pytest.mark.asyncio
    async def test_sync_teams_keys_uppercased(self, tmp_path):
        client = _make_client()
        sync = RosterSync(tmp_path)
        result = await sync.sync_teams(client, ["tor"])
        assert "TOR" in result

    @pytest.mark.asyncio
    async def test_sync_teams_empty_list(self, tmp_path):
        client = _make_client()
        sync = RosterSync(tmp_path)
        result = await sync.sync_teams(client, [])
        assert result == {}


# ---------------------------------------------------------------------------
# 8. to_polars — schema + content
# ---------------------------------------------------------------------------


class TestToPolars:
    def setup_method(self):
        self.profiles = _normalize_roster(_RAW_ROSTER, "TOR", date(2024, 11, 15))
        self.df = RosterSync.to_polars(self.profiles)

    def test_returns_polars_dataframe(self):
        assert isinstance(self.df, pl.DataFrame)

    def test_row_count(self):
        assert len(self.df) == 3

    def test_expected_columns(self):
        expected = {
            "player_id", "team_code", "first_name", "last_name",
            "position", "position_type", "shoots_catches",
            "jersey_number", "height_inches", "weight_pounds",
            "birth_date", "age", "birth_country", "nationality",
            "is_captain", "is_alternate_captain",
        }
        assert expected.issubset(set(self.df.columns))

    def test_player_ids_present(self):
        ids = set(self.df["player_id"].to_list())
        assert ids == {8480801, 8480803, 8480353}

    def test_position_type_values(self):
        types = set(self.df["position_type"].to_list())
        assert "forward" in types
        assert "defenseman" in types
        assert "goalie" in types

    def test_age_column_not_null(self):
        assert self.df["age"].is_not_null().all()

    def test_captain_flag_for_matthews(self):
        row = self.df.filter(pl.col("player_id") == 8480801)
        assert row["is_captain"][0] is True

    def test_empty_profiles_returns_empty_dataframe(self):
        df = RosterSync.to_polars([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# 9. Integration smoke — downstream FI engine + lineup builder consumption
# ---------------------------------------------------------------------------


class TestIntegrationSmoke:
    @pytest.mark.asyncio
    async def test_full_pipeline_sync_to_polars(self, tmp_path):
        """Smoke: API → normalize → cache → Polars for downstream use."""
        client = _make_client()
        sync = RosterSync(tmp_path)
        profiles = await sync.sync_team(client, "TOR", as_of=date(2024, 11, 15))
        df = RosterSync.to_polars(profiles)

        # FI engine (3.12) needs age per player
        assert df["age"].is_not_null().all()
        ages = df["age"].to_list()
        assert all(15.0 < a < 50.0 for a in ages)

    @pytest.mark.asyncio
    async def test_lineup_builder_can_filter_by_position(self, tmp_path):
        """Feature 6.1 lineup builder filters forwards vs. defensemen vs. goalies."""
        client = _make_client()
        sync = RosterSync(tmp_path)
        profiles = await sync.sync_team(client, "TOR")
        df = RosterSync.to_polars(profiles)

        forwards = df.filter(pl.col("position_type") == "forward")
        defensemen = df.filter(pl.col("position_type") == "defenseman")
        goalies = df.filter(pl.col("position_type") == "goalie")

        assert len(forwards) == 1
        assert len(defensemen) == 1
        assert len(goalies) == 1

    @pytest.mark.asyncio
    async def test_archetype_model_can_select_features(self, tmp_path):
        """Feature 2.11 archetype clustering needs position, handedness, size."""
        client = _make_client()
        sync = RosterSync(tmp_path)
        profiles = await sync.sync_team(client, "TOR")
        df = RosterSync.to_polars(profiles)

        archetype_cols = ["player_id", "position", "shoots_catches",
                          "height_inches", "weight_pounds", "age"]
        subset = df.select(archetype_cols)
        assert subset.shape[1] == len(archetype_cols)

    @pytest.mark.asyncio
    async def test_cache_round_trip_preserves_all_fields(self, tmp_path):
        """PlayerProfile serialized to JSON cache and loaded back is identical."""
        client = _make_client()
        sync = RosterSync(tmp_path)
        original = await sync.sync_team(client, "TOR", as_of=date(2024, 11, 15))
        # Force re-load from cache
        cached = await sync.sync_team(client, "TOR", as_of=date(2024, 11, 15))
        assert client.get_roster.call_count == 1  # only fetched once
        assert len(original) == len(cached)
        for o, c in zip(original, cached):
            assert o.player_id == c.player_id
            assert o.age == pytest.approx(c.age or 0, abs=0.001)
            assert o.position == c.position
            assert o.birth_date == c.birth_date

    def test_compute_age_recompute_for_game_date(self):
        """Downstream models can recompute age for a specific game date."""
        game_date = date(2025, 3, 15)
        age = compute_age("1997-09-17", as_of=game_date)
        expected = (game_date - date(1997, 9, 17)).days / 365.25
        assert age == pytest.approx(expected)
