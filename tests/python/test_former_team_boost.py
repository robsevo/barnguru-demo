"""Tests for Feature 2.27 — Former Team Motivational Boost.

Covers:
  - Departure type base boost values
  - Decay function (years since departure)
  - TOI amplification factor (chip on shoulder)
  - Away game multiplier
  - Boost clamping to [BOOST_MIN, BOOST_MAX]
  - FormerTeamBoostModel.fit() — transaction parsing
  - FormerTeamBoostModel.compute_boost() end-to-end
  - Validation gate (is_validated, VALIDATION_SEASONS_REQUIRED)
  - save() / load() round-trip
  - write_former_team_boost() parquet I/O
  - Edge cases: no transactions, unknown player, fully-decayed boost
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from models.former_team_boost import (
    AWAY_MULTIPLIER,
    BOOST_BY_DEPARTURE,
    BOOST_MAX,
    BOOST_MIN,
    DECAY_SEASONS,
    TOI_MAX_AMPLIFICATION,
    TOI_THRESHOLD_MINUTES,
    VALIDATION_SEASONS_REQUIRED,
    FormerTeamBoostModel,
    _decay_factor,
    _departure_type_from_event,
    _espn_id_to_int,
    _toi_amplification_factor,
    _years_between,
    compute_boost,
    write_former_team_boost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tx_row(
    event_type: str = "trade",
    team: str = "TOR",
    secondary_team: str | None = "MTL",
    player_id_espn: str = "12345",
    player_name: str = "Test Player",
    date: str = "2022-07-01",
) -> dict:
    return {
        "date":                date,
        "event_type":          event_type,
        "team":                team,
        "secondary_team":      secondary_team,
        "player_or_executive": player_name,
        "player_id_espn":      player_id_espn,
        "description":         f"{player_name} {event_type}",
        "notes":               None,
        "fetched_at":          "2025-01-01T00:00:00Z",
        "source":              "espn",
    }


def _tx_df(*rows) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


def _toi_df(player_id: int, team: str, toi_per_game_min: float, gp: int = 50) -> pl.DataFrame:
    return pl.DataFrame({
        "player_id":    [player_id],
        "team":         [team],
        "toi_seconds":  [toi_per_game_min * 60.0 * gp],
        "gp":           [gp],
    })


# ---------------------------------------------------------------------------
# Unit: _departure_type_from_event
# ---------------------------------------------------------------------------

class TestDepartureTypeFromEvent:
    def test_trade(self):
        assert _departure_type_from_event("trade") == "trade"

    def test_waiver_release(self):
        assert _departure_type_from_event("waiver_release") == "waiver_release"

    def test_signing_maps_to_ufa(self):
        assert _departure_type_from_event("signing") == "ufa"

    def test_other_fallback(self):
        assert _departure_type_from_event("activated") == "other"
        assert _departure_type_from_event("call_up") == "other"


# ---------------------------------------------------------------------------
# Unit: _years_between
# ---------------------------------------------------------------------------

class TestYearsBetween:
    def test_same_date_is_zero(self):
        assert _years_between("2022-07-01", "2022-07-01") == pytest.approx(0.0, abs=0.01)

    def test_one_year(self):
        years = _years_between("2022-07-01", "2023-07-01")
        assert years == pytest.approx(1.0, abs=0.02)

    def test_three_years(self):
        years = _years_between("2021-07-01", "2024-07-01")
        assert years == pytest.approx(3.0, abs=0.02)

    def test_bad_date_returns_zero(self):
        assert _years_between("not-a-date", "2024-01-01") == 0.0

    def test_later_date_gives_positive(self):
        assert _years_between("2020-01-01", "2025-01-01") > 0.0


# ---------------------------------------------------------------------------
# Unit: _decay_factor
# ---------------------------------------------------------------------------

class TestDecayFactor:
    def test_zero_years_is_full(self):
        assert _decay_factor(0.0) == pytest.approx(1.0)

    def test_decay_at_half_life(self):
        assert _decay_factor(DECAY_SEASONS / 2) == pytest.approx(0.5, abs=0.01)

    def test_at_decay_seasons_is_zero(self):
        assert _decay_factor(DECAY_SEASONS) == pytest.approx(0.0, abs=1e-9)

    def test_beyond_decay_seasons_is_zero(self):
        assert _decay_factor(DECAY_SEASONS + 1.0) == 0.0
        assert _decay_factor(10.0) == 0.0

    def test_negative_years_clamps_to_one(self):
        # Edge: departure in future — treat as 0 elapsed
        assert _decay_factor(-0.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Unit: _toi_amplification_factor
# ---------------------------------------------------------------------------

class TestTOIAmplificationFactor:
    def test_at_threshold_is_one(self):
        assert _toi_amplification_factor(TOI_THRESHOLD_MINUTES) == pytest.approx(1.0)

    def test_above_threshold_is_one(self):
        assert _toi_amplification_factor(TOI_THRESHOLD_MINUTES + 5.0) == pytest.approx(1.0)

    def test_zero_toi_is_max(self):
        factor = _toi_amplification_factor(0.0)
        assert factor == pytest.approx(TOI_MAX_AMPLIFICATION)

    def test_half_threshold_is_amplified(self):
        factor = _toi_amplification_factor(TOI_THRESHOLD_MINUTES / 2)
        assert 1.0 < factor <= TOI_MAX_AMPLIFICATION

    def test_capped_at_max_amplification(self):
        # Even with very low TOI, never exceeds cap
        factor = _toi_amplification_factor(-100.0)
        assert factor <= TOI_MAX_AMPLIFICATION


# ---------------------------------------------------------------------------
# Unit: _espn_id_to_int
# ---------------------------------------------------------------------------

class TestEspnIdToInt:
    def test_numeric_string(self):
        assert _espn_id_to_int("3900169") == 3900169

    def test_empty_string_is_zero(self):
        assert _espn_id_to_int("") == 0

    def test_none_is_zero(self):
        assert _espn_id_to_int(None) == 0

    def test_non_numeric_is_zero(self):
        assert _espn_id_to_int("abc") == 0


# ---------------------------------------------------------------------------
# Unit: boost constants
# ---------------------------------------------------------------------------

class TestBoostConstants:
    def test_waived_is_highest(self):
        assert BOOST_BY_DEPARTURE["waiver_release"] == BOOST_MAX

    def test_rfa_is_lowest(self):
        assert BOOST_BY_DEPARTURE["rfa"] == BOOST_MIN + BOOST_BY_DEPARTURE["rfa"]
        assert BOOST_BY_DEPARTURE["rfa"] <= min(BOOST_BY_DEPARTURE.values()) + 0.01

    def test_all_boosts_in_range(self):
        for dep, val in BOOST_BY_DEPARTURE.items():
            assert BOOST_MIN <= val <= BOOST_MAX, f"{dep}: {val} out of range"

    def test_validation_seasons_required_is_3(self):
        assert VALIDATION_SEASONS_REQUIRED == 3


# ---------------------------------------------------------------------------
# FormerTeamBoostModel.fit()
# ---------------------------------------------------------------------------

class TestModelFit:
    def test_empty_transactions_returns_empty_df(self):
        model = FormerTeamBoostModel()
        df = model.fit(pl.DataFrame())
        assert len(df) == 0
        assert model.player_count() == 0

    def test_trade_creates_former_team_entry(self):
        model = FormerTeamBoostModel()
        # Player ESPN ID=12345 traded FROM MTL to TOR on 2022-07-01
        tx = _tx_df(_tx_row(event_type="trade", team="TOR", secondary_team="MTL",
                             player_id_espn="12345"))
        df = model.fit(tx)
        assert len(df) == 1
        assert df["former_team"][0] == "MTL"
        assert df["departure_type"][0] == "trade"

    def test_waiver_creates_former_team_entry(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row(event_type="waiver_release", team="PIT", secondary_team=None,
                             player_id_espn="99"))
        df = model.fit(tx)
        assert len(df) == 1
        assert df["former_team"][0] == "PIT"
        assert df["departure_type"][0] == "waiver_release"
        assert df["base_boost"][0] == pytest.approx(BOOST_BY_DEPARTURE["waiver_release"])

    def test_signing_creates_former_team_entry(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row(event_type="signing", team="EDM", secondary_team=None,
                             player_id_espn="777"))
        df = model.fit(tx)
        assert len(df) == 1
        assert df["former_team"][0] == "EDM"
        assert df["departure_type"][0] == "ufa"

    def test_irrelevant_events_ignored(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(
            _tx_row(event_type="call_up", team="BOS"),
            _tx_row(event_type="activated", team="NYR"),
            _tx_row(event_type="ltir", team="CGY"),
        )
        df = model.fit(tx)
        assert len(df) == 0

    def test_multiple_events_for_same_player(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(
            _tx_row(event_type="trade", team="TOR", secondary_team="MTL",
                    player_id_espn="12345", date="2020-07-01"),
            _tx_row(event_type="trade", team="VAN", secondary_team="TOR",
                    player_id_espn="12345", date="2022-07-01"),
        )
        df = model.fit(tx)
        assert len(df) == 2
        assert model.player_count() == 1
        teams = set(df["former_team"].to_list())
        assert teams == {"MTL", "TOR"}

    def test_output_schema_complete(self):
        from models.former_team_boost import FORMER_TEAM_BOOST_SCHEMA
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        df = model.fit(tx)
        for col in FORMER_TEAM_BOOST_SCHEMA:
            assert col in df.columns, f"Missing column: {col}"

    def test_model_version_set(self):
        from models.former_team_boost import MODEL_VERSION
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        df = model.fit(tx)
        assert df["model_version"][0] == MODEL_VERSION


# ---------------------------------------------------------------------------
# FormerTeamBoostModel.compute_boost()
# ---------------------------------------------------------------------------

class TestComputeBoost:
    def _fit_trade_model(
        self,
        player_id: str = "12345",
        former_team: str = "MTL",
        departure_date: str = "2024-07-01",
        toi_df: pl.DataFrame | None = None,
    ) -> FormerTeamBoostModel:
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row(event_type="trade", team="TOR", secondary_team=former_team,
                             player_id_espn=player_id, date=departure_date))
        model.fit(tx, player_toi_df=toi_df)
        return model

    def test_no_former_team_returns_zero(self):
        model = self._fit_trade_model()
        assert model.compute_boost(12345, "EDM", "2025-01-01") == 0.0

    def test_unknown_player_returns_zero(self):
        model = self._fit_trade_model()
        assert model.compute_boost(99999, "MTL", "2025-01-01") == 0.0

    def test_former_team_match_gives_nonzero_boost(self):
        model = self._fit_trade_model(departure_date="2024-07-01")
        boost = model.compute_boost(12345, "MTL", "2025-01-15")
        assert boost > 0.0

    def test_boost_within_range(self):
        model = self._fit_trade_model(departure_date="2024-07-01")
        boost = model.compute_boost(12345, "MTL", "2025-01-15")
        assert BOOST_MIN <= boost <= BOOST_MAX

    def test_away_gives_higher_boost(self):
        model = self._fit_trade_model(departure_date="2024-07-01")
        home_boost = model.compute_boost(12345, "MTL", "2025-01-15", is_away=False)
        away_boost = model.compute_boost(12345, "MTL", "2025-01-15", is_away=True)
        assert away_boost > home_boost
        assert away_boost == pytest.approx(min(home_boost * AWAY_MULTIPLIER, BOOST_MAX), abs=1e-6)

    def test_decay_reduces_boost(self):
        model = self._fit_trade_model(departure_date="2022-01-01")
        boost_near   = model.compute_boost(12345, "MTL", "2022-06-01")
        boost_far    = model.compute_boost(12345, "MTL", "2024-06-01")
        assert boost_near > boost_far

    def test_fully_decayed_returns_zero(self):
        model = self._fit_trade_model(departure_date="2020-01-01")
        # 5+ years later — fully decayed
        boost = model.compute_boost(12345, "MTL", "2025-06-01")
        assert boost == 0.0

    def test_waived_boost_higher_than_trade(self):
        model_waive = FormerTeamBoostModel()
        tx_waive = _tx_df(_tx_row(event_type="waiver_release", team="MTL",
                                   secondary_team=None, player_id_espn="1",
                                   date="2024-07-01"))
        model_waive.fit(tx_waive)

        model_trade = FormerTeamBoostModel()
        tx_trade = _tx_df(_tx_row(event_type="trade", team="TOR", secondary_team="MTL",
                                   player_id_espn="2", date="2024-07-01"))
        model_trade.fit(tx_trade)

        boost_waive = model_waive.compute_boost(1, "MTL", "2025-01-01")
        boost_trade = model_trade.compute_boost(2, "MTL", "2025-01-01")
        assert boost_waive > boost_trade

    def test_toi_amplification_increases_boost(self):
        """Player with very low TOI gets higher boost than one with full TOI."""
        # Low TOI: 8 min/game (well below threshold of 14)
        low_toi_df = _toi_df(player_id=12345, team="MTL", toi_per_game_min=8.0)
        model_low = self._fit_trade_model(departure_date="2024-07-01", toi_df=low_toi_df)

        # Full TOI: 20 min/game (above threshold)
        full_toi_df = _toi_df(player_id=12345, team="MTL", toi_per_game_min=20.0)
        model_full = self._fit_trade_model(departure_date="2024-07-01", toi_df=full_toi_df)

        boost_low  = model_low.compute_boost(12345, "MTL", "2025-01-01")
        boost_full = model_full.compute_boost(12345, "MTL", "2025-01-01")
        assert boost_low > boost_full

    def test_case_insensitive_team_matching(self):
        model = self._fit_trade_model(former_team="MTL", departure_date="2024-07-01")
        boost_upper = model.compute_boost(12345, "MTL", "2025-01-01")
        boost_lower = model.compute_boost(12345, "mtl", "2025-01-01")
        assert boost_upper == pytest.approx(boost_lower)

    def test_takes_best_boost_across_multiple_former_teams(self):
        """If player has two former-team entries for same team, take highest."""
        model = FormerTeamBoostModel()
        # Two events for same player vs same team (e.g., traded then bought out)
        tx = _tx_df(
            _tx_row(event_type="trade", team="TOR", secondary_team="MTL",
                    player_id_espn="55", date="2024-01-01"),
            _tx_row(event_type="waiver_release", team="MTL", secondary_team=None,
                    player_id_espn="55", date="2024-06-01"),
        )
        model.fit(tx)
        boost = model.compute_boost(55, "MTL", "2025-01-01")
        # Should be the waiver boost (higher) not just the trade boost
        expected_waiver = BOOST_BY_DEPARTURE["waiver_release"] * _decay_factor(
            _years_between("2024-06-01", "2025-01-01")
        )
        assert boost == pytest.approx(min(expected_waiver, BOOST_MAX), abs=0.01)


# ---------------------------------------------------------------------------
# FormerTeamBoostModel.has_former_team_matchup()
# ---------------------------------------------------------------------------

class TestHasFormerTeamMatchup:
    def test_returns_true_on_active_matchup(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row(event_type="trade", team="TOR", secondary_team="EDM",
                             player_id_espn="5", date="2024-07-01"))
        model.fit(tx)
        assert model.has_former_team_matchup(5, "EDM", "2025-01-01") is True

    def test_returns_false_on_wrong_opponent(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row(event_type="trade", team="TOR", secondary_team="EDM",
                             player_id_espn="5", date="2024-07-01"))
        model.fit(tx)
        assert model.has_former_team_matchup(5, "MTL", "2025-01-01") is False

    def test_returns_false_after_decay(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row(event_type="trade", team="TOR", secondary_team="EDM",
                             player_id_espn="5", date="2020-01-01"))
        model.fit(tx)
        assert model.has_former_team_matchup(5, "EDM", "2026-01-01") is False


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------

class TestValidationGate:
    def test_not_validated_without_seasons(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        model.fit(tx, validated_seasons=None)
        assert model.is_validated is False

    def test_not_validated_with_fewer_than_required(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        model.fit(tx, validated_seasons=[2022, 2023])
        assert model.is_validated is False

    def test_validated_with_three_seasons(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        model.fit(tx, validated_seasons=[2022, 2023, 2024])
        assert model.is_validated is True

    def test_validated_with_more_than_three(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        model.fit(tx, validated_seasons=[2021, 2022, 2023, 2024])
        assert model.is_validated is True

    def test_validated_flag_in_output_df(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        df = model.fit(tx, validated_seasons=[2022, 2023, 2024])
        assert df["validated"][0] is True

    def test_unvalidated_flag_in_output_df(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        df = model.fit(tx, validated_seasons=[])
        assert df["validated"][0] is False


# ---------------------------------------------------------------------------
# former_teams() helper
# ---------------------------------------------------------------------------

class TestFormerTeams:
    def test_returns_correct_teams(self):
        model = FormerTeamBoostModel()
        tx = _tx_df(
            _tx_row(event_type="trade", team="TOR", secondary_team="MTL",
                    player_id_espn="8", date="2022-01-01"),
            _tx_row(event_type="waiver_release", team="VAN",
                    player_id_espn="8", date="2024-01-01"),
        )
        model.fit(tx)
        teams = model.former_teams(8)
        assert set(teams) == {"MTL", "VAN"}

    def test_unknown_player_empty_list(self):
        model = FormerTeamBoostModel()
        model.fit(pl.DataFrame())
        assert model.former_teams(99999) == []


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_preserves_boost(self, tmp_path):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row(event_type="waiver_release", team="PIT",
                             player_id_espn="999", date="2024-07-01"))
        model.fit(tx, validated_seasons=[2022, 2023, 2024])

        path = tmp_path / "ftb_model.pkl"
        model.save(path)

        loaded = FormerTeamBoostModel.load(path)
        boost_orig   = model.compute_boost(999, "PIT", "2025-01-01")
        boost_loaded = loaded.compute_boost(999, "PIT", "2025-01-01")
        assert boost_orig == pytest.approx(boost_loaded)

    def test_roundtrip_preserves_validation_state(self, tmp_path):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        model.fit(tx, validated_seasons=[2022, 2023, 2024])

        path = tmp_path / "ftb.pkl"
        model.save(path)
        loaded = FormerTeamBoostModel.load(path)
        assert loaded.is_validated is True

    def test_roundtrip_unvalidated(self, tmp_path):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        model.fit(tx)

        path = tmp_path / "ftb2.pkl"
        model.save(path)
        loaded = FormerTeamBoostModel.load(path)
        assert loaded.is_validated is False


# ---------------------------------------------------------------------------
# write_former_team_boost / parquet I/O
# ---------------------------------------------------------------------------

class TestParquetIO:
    def test_write_creates_file(self, tmp_path):
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        df = model.fit(tx)
        path = write_former_team_boost(df, tmp_path, season=2025)
        assert path.exists()
        assert path.name == "former_team_boost_2025.parquet"

    def test_written_parquet_readable(self, tmp_path):
        import polars as pl
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        df = model.fit(tx)
        path = write_former_team_boost(df, tmp_path, season=2025)
        df2 = pl.read_parquet(path)
        assert len(df2) == len(df)

    def test_empty_df_writes_cleanly(self, tmp_path):
        model = FormerTeamBoostModel()
        df = model.fit(pl.DataFrame())
        path = write_former_team_boost(df, tmp_path, season=2025)
        assert path.exists()

    def test_output_has_correct_columns(self, tmp_path):
        from models.former_team_boost import FORMER_TEAM_BOOST_SCHEMA
        model = FormerTeamBoostModel()
        tx = _tx_df(_tx_row())
        df = model.fit(tx)
        path = write_former_team_boost(df, tmp_path, season=2025)
        df2 = pl.read_parquet(path)
        for col in FORMER_TEAM_BOOST_SCHEMA:
            assert col in df2.columns


# ---------------------------------------------------------------------------
# Stateless compute_boost helper
# ---------------------------------------------------------------------------

class TestStatelessComputeBoost:
    def test_returns_nonzero_for_matching_entry(self):
        from models.former_team_boost import _FormerTeamEntry
        entry = _FormerTeamEntry(
            player_id=1, player_name="Test", former_team="MTL",
            departure_date="2024-07-01", departure_type="trade",
            toi_per_game_min=15.0,
            base_boost=BOOST_BY_DEPARTURE["trade"],
        )
        result = compute_boost(
            player_id=1, opponent_team="MTL", game_date="2025-01-01",
            is_away=False, former_teams_entries=[entry],
        )
        assert result > 0.0

    def test_returns_zero_for_wrong_team(self):
        from models.former_team_boost import _FormerTeamEntry
        entry = _FormerTeamEntry(
            player_id=1, player_name="Test", former_team="MTL",
            departure_date="2024-07-01", departure_type="trade",
            toi_per_game_min=15.0,
            base_boost=BOOST_BY_DEPARTURE["trade"],
        )
        result = compute_boost(
            player_id=1, opponent_team="EDM", game_date="2025-01-01",
            is_away=False, former_teams_entries=[entry],
        )
        assert result == 0.0
