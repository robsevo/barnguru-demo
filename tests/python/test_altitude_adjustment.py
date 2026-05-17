"""Tests for Feature 3.6 — AltitudeAdjustmentModel."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.altitude_adjustment import (
    ALTITUDE_SCHEMA,
    AltitudeAdjustmentModel,
    HIGH_ALTITUDE_THRESHOLD_FT,
    MAX_PENALTY,
    NHL_ARENA_ELEVATION_FT,
    altitude_penalty,
    write_altitude_adjustment,
)
from models.rapm_model import DataMissingWarning


def _sched(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "game_id":   0,
        "game_date": "2026-01-15",
        "home_team": "COL",
        "away_team": "NYR",
    }
    filled = [{**defaults, **r} for r in rows]
    for i, r in enumerate(filled):
        if r.get("game_id") in (None, 0):
            r["game_id"] = 1_000 + i
    if not filled:
        return pl.DataFrame(
            schema={
                "game_id":   pl.Int64,
                "game_date": pl.Utf8,
                "home_team": pl.Utf8,
                "away_team": pl.Utf8,
            }
        )
    return pl.DataFrame(filled)


# ---------------------------------------------------------------------------
# Elevation table sanity
# ---------------------------------------------------------------------------

class TestElevationTable:
    def test_denver_above_threshold(self):
        assert NHL_ARENA_ELEVATION_FT["COL"] > HIGH_ALTITUDE_THRESHOLD_FT

    def test_calgary_above_threshold(self):
        assert NHL_ARENA_ELEVATION_FT["CGY"] > HIGH_ALTITUDE_THRESHOLD_FT

    def test_utah_above_threshold(self):
        assert NHL_ARENA_ELEVATION_FT["UTA"] > HIGH_ALTITUDE_THRESHOLD_FT

    def test_edmonton_below_threshold(self):
        # EDM is high-ish but below the aerobic-penalty threshold.
        assert NHL_ARENA_ELEVATION_FT["EDM"] < HIGH_ALTITUDE_THRESHOLD_FT

    def test_sea_level_teams_under_300ft(self):
        for team in ("FLA", "MTL", "NJD", "PHI"):
            assert NHL_ARENA_ELEVATION_FT[team] < 300

    def test_all_32_current_teams_present(self):
        current = {
            "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
            "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
            "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
            "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
        }
        assert current - set(NHL_ARENA_ELEVATION_FT.keys()) == set()


# ---------------------------------------------------------------------------
# Penalty formula
# ---------------------------------------------------------------------------

class TestPenaltyFormula:
    def test_home_team_zero_penalty(self):
        # Even at COL, the home team is acclimatized → 0.
        p = altitude_penalty(5280, 5280, is_home=True)
        assert p == pytest.approx(0.0)

    def test_low_altitude_venue_zero_penalty(self):
        # NYR (50 ft) visiting BOS (20 ft): no aerobic hit.
        p = altitude_penalty(20, 50, is_home=False)
        assert p == pytest.approx(0.0)

    def test_sea_level_visitor_at_denver_max_hit(self):
        # NYR at COL: full acclimatization factor, big elevation excess.
        p = altitude_penalty(5280, 50, is_home=False)
        # venue_excess = 2.28, raw = 0.0684, acclim ≈ 0.983 → ~0.067
        assert 0.05 < p < MAX_PENALTY + 1e-9

    def test_calgary_visiting_denver_smaller_hit(self):
        # CGY (3400 ft) at COL (5280 ft): CGY's home is above threshold so
        # acclim factor = 0 → penalty = 0.
        p = altitude_penalty(5280, 3400, is_home=False)
        assert p == pytest.approx(0.0)

    def test_penalty_capped_at_max(self):
        # Crank up to ridiculous numbers; the cap must hold.
        p = altitude_penalty(15_000, 0, is_home=False)
        assert p == pytest.approx(MAX_PENALTY)

    def test_penalty_monotonic_in_venue_height(self):
        p_low  = altitude_penalty(3500, 50, is_home=False)
        p_high = altitude_penalty(5280, 50, is_home=False)
        assert p_high > p_low

    def test_penalty_monotonic_in_visitor_home_height(self):
        # The lower the visitor's home rink, the *bigger* the penalty.
        p_low_home  = altitude_penalty(5280,    0, is_home=False)
        p_high_home = altitude_penalty(5280, 2000, is_home=False)
        assert p_low_home > p_high_home


# ---------------------------------------------------------------------------
# Validation / empty
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = AltitudeAdjustmentModel().compute(bad)
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_returns_empty_schema(self):
        out = AltitudeAdjustmentModel().compute(_sched([]))
        for col in ALTITUDE_SCHEMA:
            assert col in out.columns
        assert len(out) == 0

    def test_output_columns(self):
        out = AltitudeAdjustmentModel().compute(
            _sched([{"home_team": "COL", "away_team": "NYR"}])
        )
        assert set(out.columns) == set(ALTITUDE_SCHEMA.keys())


# ---------------------------------------------------------------------------
# End-to-end semantics on team-games
# ---------------------------------------------------------------------------

class TestComputeSemantics:
    def test_high_altitude_flag_set_for_denver_games(self):
        out = AltitudeAdjustmentModel().compute(
            _sched([{"home_team": "COL", "away_team": "NYR"}])
        )
        for row in out.to_dicts():
            assert row["is_high_altitude"] is True

    def test_low_altitude_flag_for_florida_games(self):
        out = AltitudeAdjustmentModel().compute(
            _sched([{"home_team": "FLA", "away_team": "BOS"}])
        )
        for row in out.to_dicts():
            assert row["is_high_altitude"] is False

    def test_home_penalty_zero_visitor_penalty_positive(self):
        out = AltitudeAdjustmentModel().compute(
            _sched([{"home_team": "COL", "away_team": "NYR"}])
        )
        col = out.filter(pl.col("team") == "COL").to_dicts()[0]
        nyr = out.filter(pl.col("team") == "NYR").to_dicts()[0]
        assert col["altitude_penalty"] == pytest.approx(0.0)
        assert nyr["altitude_penalty"] > 0.0

    def test_elevation_delta_sign(self):
        # NYR at COL: venue (5280) − home (50) = +5230.
        # COL at NYR: venue (50) − home (5280) = -5230.
        out = AltitudeAdjustmentModel().compute(
            pl.DataFrame(
                [
                    {"game_id": 1, "game_date": "2026-01-15",
                     "home_team": "COL", "away_team": "NYR"},
                    {"game_id": 2, "game_date": "2026-01-20",
                     "home_team": "NYR", "away_team": "COL"},
                ]
            )
        )
        nyr_at_col = out.filter(
            (pl.col("team") == "NYR") & (pl.col("game_id") == 1)
        ).to_dicts()[0]
        col_at_nyr = out.filter(
            (pl.col("team") == "COL") & (pl.col("game_id") == 2)
        ).to_dicts()[0]
        assert nyr_at_col["elevation_delta_ft"] > 0
        assert col_at_nyr["elevation_delta_ft"] < 0
        # COL at sea level → no aerobic penalty for downhill travel.
        assert col_at_nyr["altitude_penalty"] == pytest.approx(0.0)

    def test_unknown_venue_warns_does_not_crash(self):
        rows = [
            {"game_date": "2026-01-15", "home_team": "ZZZ", "away_team": "NYR"},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = AltitudeAdjustmentModel().compute(_sched(rows))
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert len(out) > 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        out = AltitudeAdjustmentModel().compute(
            _sched([{"home_team": "COL", "away_team": "NYR"}])
        )
        path = write_altitude_adjustment(out, tmp_path, "2026-01-15")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in ALTITUDE_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        out = AltitudeAdjustmentModel().compute(
            _sched([{"home_team": "COL", "away_team": "NYR"}])
        )
        path = write_altitude_adjustment(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name


# ---------------------------------------------------------------------------
# Multi-team
# ---------------------------------------------------------------------------

class TestMultiTeam:
    def test_both_teams_emit_rows(self):
        out = AltitudeAdjustmentModel().compute(
            _sched([{"home_team": "COL", "away_team": "NYR"}])
        )
        teams = set(out["team"].to_list())
        assert teams == {"COL", "NYR"}


# ---------------------------------------------------------------------------
# Custom elevation injection
# ---------------------------------------------------------------------------

class TestCustomElevation:
    def test_custom_elevation_override(self):
        # Force a contrived high-altitude venue with a sea-level visitor.
        custom = {"A": 0, "B": 10000}
        out = AltitudeAdjustmentModel(elevation_map=custom).compute(
            _sched([{"home_team": "B", "away_team": "A"}])
        )
        a = out.filter(pl.col("team") == "A").to_dicts()[0]
        # Crazy excess but capped at MAX_PENALTY.
        assert a["altitude_penalty"] == pytest.approx(MAX_PENALTY)
