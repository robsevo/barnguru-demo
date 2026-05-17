"""Tests for Feature 3.1 — ScheduleDensityModel."""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import polars as pl
import pytest

from models.rapm_model import DataMissingWarning
from models.schedule_density import (
    SCHEDULE_DENSITY_SCHEMA,
    ScheduleDensityModel,
    write_schedule_density,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sched(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal schedule DataFrame with defaults filled in."""
    defaults = {
        "game_id": 0,
        "game_date": "2026-03-01",
        "home_team": "EDM",
        "away_team": "TOR",
    }
    filled = [{**defaults, **r} for r in rows]
    for i, r in enumerate(filled):
        if r.get("game_id") in (None, 0):
            r["game_id"] = 1_000 + i
    if not filled:
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64,
                "game_date": pl.Utf8,
                "home_team": pl.Utf8,
                "away_team": pl.Utf8,
            }
        )
    return pl.DataFrame(filled)


def _series_of_games(team: str, opp: str, dates: list[str], home: list[bool]) -> pl.DataFrame:
    """Build a schedule of N games for ``team`` against ``opp`` on given dates."""
    assert len(dates) == len(home)
    rows = []
    for d, is_h in zip(dates, home):
        if is_h:
            rows.append({"game_date": d, "home_team": team, "away_team": opp})
        else:
            rows.append({"game_date": d, "home_team": opp, "away_team": team})
    return _sched(rows)


# ---------------------------------------------------------------------------
# Validation / empty inputs
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns_and_returns_empty(self):
        bad = pl.DataFrame({"description": ["foo"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ScheduleDensityModel().compute(bad)
        assert len(result) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_df_returns_empty_schema(self):
        empty = _sched([])
        result = ScheduleDensityModel().compute(empty)
        assert len(result) == 0
        for col in SCHEDULE_DENSITY_SCHEMA:
            assert col in result.columns

    def test_output_schema_columns(self):
        result = ScheduleDensityModel().compute(_sched([{"game_date": "2026-03-01"}]))
        assert set(result.columns) == set(SCHEDULE_DENSITY_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Single-team density
# ---------------------------------------------------------------------------

class TestSingleTeamDensity:
    def test_first_game_has_no_rest(self):
        df = _series_of_games("EDM", "TOR", ["2026-03-01"], [True])
        out = ScheduleDensityModel().compute_team(df, "EDM")
        row = out.to_dicts()[0]
        assert row["rest_days"] == -1
        assert row["is_b2b"] is False
        assert row["is_3_in_4"] is False
        assert row["games_last_7d"] == 1
        assert row["games_last_14d"] == 1

    def test_b2b_detected(self):
        df = _series_of_games(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-02"],
            [True, False],
        )
        out = ScheduleDensityModel().compute_team(df, "EDM").sort("game_date")
        rows = out.to_dicts()
        assert rows[0]["is_b2b"] is False
        assert rows[1]["is_b2b"] is True
        assert rows[1]["rest_days"] == 1

    def test_one_day_off_not_b2b(self):
        df = _series_of_games(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-03"],
            [True, True],
        )
        out = ScheduleDensityModel().compute_team(df, "EDM").sort("game_date")
        rows = out.to_dicts()
        assert rows[1]["is_b2b"] is False
        assert rows[1]["rest_days"] == 2

    def test_three_in_four_flag(self):
        # Games on day 0, 1, 3 → on day-3 game, 3 games in last 4 days inclusive.
        df = _series_of_games(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-02", "2026-03-04"],
            [True, True, True],
        )
        out = ScheduleDensityModel().compute_team(df, "EDM").sort("game_date")
        rows = out.to_dicts()
        assert rows[0]["is_3_in_4"] is False
        assert rows[1]["is_3_in_4"] is False
        assert rows[2]["is_3_in_4"] is True

    def test_three_in_five_not_flagged(self):
        # Games on day 0, 2, 4 — only 2 of those fall within 4 days of each.
        df = _series_of_games(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-03", "2026-03-05"],
            [True, True, True],
        )
        out = ScheduleDensityModel().compute_team(df, "EDM").sort("game_date")
        assert all(row["is_3_in_4"] is False for row in out.to_dicts())

    def test_games_last_7d_inclusive(self):
        # Seven games on consecutive days → final game sees 7 games in last 7d.
        dates = [(date(2026, 3, 1) + timedelta(days=i)).isoformat() for i in range(7)]
        df = _series_of_games("EDM", "TOR", dates, [True] * 7)
        out = ScheduleDensityModel().compute_team(df, "EDM").sort("game_date")
        rows = out.to_dicts()
        assert rows[-1]["games_last_7d"] == 7
        assert rows[0]["games_last_7d"] == 1

    def test_games_last_14d_outside_window(self):
        # Game 15 days before the second game is well outside the 14-day window.
        df = _series_of_games(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-16"],
            [True, True],
        )
        out = ScheduleDensityModel().compute_team(df, "EDM").sort("game_date")
        rows = out.to_dicts()
        assert rows[1]["games_last_14d"] == 1  # only the current game counts

    def test_games_last_14d_boundary_inclusive(self):
        # "Last 14 days" = today + 13 prior calendar days. A game exactly
        # 13 days earlier still counts; a game 14 days earlier does not.
        df = _series_of_games(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-14"],   # 13 days apart
            [True, True],
        )
        out = ScheduleDensityModel().compute_team(df, "EDM").sort("game_date")
        rows = out.to_dicts()
        assert rows[1]["games_last_14d"] == 2

    def test_games_last_14d_boundary_exclusive_at_14(self):
        df = _series_of_games(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-15"],   # 14 days apart
            [True, True],
        )
        out = ScheduleDensityModel().compute_team(df, "EDM").sort("game_date")
        rows = out.to_dicts()
        assert rows[1]["games_last_14d"] == 1

    def test_games_last_7d_boundary_inclusive(self):
        # "Last 7 days" = today + 6 prior. A game 6 days earlier counts.
        df = _series_of_games(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-07"],   # 6 days apart
            [True, True],
        )
        out = ScheduleDensityModel().compute_team(df, "EDM").sort("game_date")
        rows = out.to_dicts()
        assert rows[1]["games_last_7d"] == 2

    def test_games_last_7d_boundary_exclusive_at_7(self):
        df = _series_of_games(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-08"],   # 7 days apart
            [True, True],
        )
        out = ScheduleDensityModel().compute_team(df, "EDM").sort("game_date")
        rows = out.to_dicts()
        assert rows[1]["games_last_7d"] == 1


# ---------------------------------------------------------------------------
# Multi-team independence
# ---------------------------------------------------------------------------

class TestMultiTeam:
    def test_both_teams_get_rows(self):
        df = _sched([{"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"}])
        out = ScheduleDensityModel().compute(df)
        teams = set(out["team"].to_list())
        assert teams == {"EDM", "TOR"}

    def test_teams_indexed_independently(self):
        # EDM has a b2b on Mar 1–2; TOR has only Mar 2. TOR should not see b2b.
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "VAN"},
            {"game_date": "2026-03-02", "home_team": "EDM", "away_team": "TOR"},
        ]
        out = ScheduleDensityModel().compute(_sched(rows))
        edm_mar2 = out.filter((pl.col("team") == "EDM") & (pl.col("game_date") == "2026-03-02")).to_dicts()[0]
        tor_mar2 = out.filter((pl.col("team") == "TOR") & (pl.col("game_date") == "2026-03-02")).to_dicts()[0]
        assert edm_mar2["is_b2b"] is True
        assert tor_mar2["is_b2b"] is False
        assert tor_mar2["rest_days"] == -1

    def test_home_flag_per_team(self):
        rows = [{"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"}]
        out = ScheduleDensityModel().compute(_sched(rows))
        edm = out.filter(pl.col("team") == "EDM").to_dicts()[0]
        tor = out.filter(pl.col("team") == "TOR").to_dicts()[0]
        assert edm["is_home"] is True
        assert tor["is_home"] is False


# ---------------------------------------------------------------------------
# latest_for_team
# ---------------------------------------------------------------------------

class TestLatestForTeam:
    def test_returns_none_when_no_games(self):
        model = ScheduleDensityModel()
        assert model.latest_for_team(_sched([]), "EDM") is None

    def test_returns_last_game_by_default(self):
        df = _series_of_games(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-10"],
            [True, True],
        )
        latest = ScheduleDensityModel().latest_for_team(df, "EDM")
        assert latest is not None
        assert latest["game_date"] == "2026-03-10"

    def test_as_of_date_caps_result(self):
        df = _series_of_games(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-10"],
            [True, True],
        )
        latest = ScheduleDensityModel().latest_for_team(df, "EDM", as_of_date="2026-03-05")
        assert latest is not None
        assert latest["game_date"] == "2026-03-01"

    def test_as_of_before_first_game_returns_none(self):
        df = _series_of_games("EDM", "TOR", ["2026-03-10"], [True])
        latest = ScheduleDensityModel().latest_for_team(df, "EDM", as_of_date="2026-03-01")
        assert latest is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_write_schedule_density(self, tmp_path):
        df = _series_of_games("EDM", "TOR", ["2026-03-01"], [True])
        result = ScheduleDensityModel().compute(df)
        path = write_schedule_density(result, tmp_path, "2026-03-01")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in SCHEDULE_DENSITY_SCHEMA:
            assert col in loaded.columns

    def test_write_filename_contains_date(self, tmp_path):
        df = _series_of_games("EDM", "TOR", ["2026-03-01"], [True])
        result = ScheduleDensityModel().compute(df)
        path = write_schedule_density(result, tmp_path, "2026-04-15")
        assert "2026-04-15" in path.name
