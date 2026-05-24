"""Unit tests for models/goalie_pull.py — Feature 4.5."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from models.goalie_pull import (
    DataMissingWarning,
    GOALIE_PULL_SCHEMA,
    MODEL_VERSION,
    PULL_EVENTS_SCHEMA,
    aggregate_pulls,
    compute_goalie_pull,
    detect_pulls,
    read_goalie_pull,
    write_goalie_pull,
)


# ---------------------------------------------------------------------------
# Synthetic PBP factory
# ---------------------------------------------------------------------------


def _pbp_with_pull(
    *,
    pull_period: int = 3,
    pull_at_secs: int = 1100,
    pull_back_at_secs: int = 1170,
    pulling_side: str = "away",       # "home" | "away"
    home_score: int = 2,
    away_score: int = 1,
    home_team_id: int = 8,
    away_team_id: int = 10,
) -> pl.DataFrame:
    """Build a minimal PBP with a single goalie-pull window."""
    rows = []
    # Pre-pull state: 5v5 EV
    rows.append({
        "game_id":              1,
        "sort_order":           1,
        "period":               pull_period,
        "time_in_period_secs":  pull_at_secs - 30,
        "home_skaters":         5,
        "away_skaters":         5,
        "home_score":           home_score,
        "away_score":           away_score,
        "home_team_id":         home_team_id,
        "away_team_id":         away_team_id,
    })
    # Pull on
    hs = 6 if pulling_side == "home" else 5
    aw = 6 if pulling_side == "away" else 5
    rows.append({
        "game_id":              1,
        "sort_order":           2,
        "period":               pull_period,
        "time_in_period_secs":  pull_at_secs,
        "home_skaters":         hs,
        "away_skaters":         aw,
        "home_score":           home_score,
        "away_score":           away_score,
        "home_team_id":         home_team_id,
        "away_team_id":         away_team_id,
    })
    # Pull resolved (goalie back / goal)
    rows.append({
        "game_id":              1,
        "sort_order":           3,
        "period":               pull_period,
        "time_in_period_secs":  pull_back_at_secs,
        "home_skaters":         5,
        "away_skaters":         5,
        "home_score":           home_score,
        "away_score":           away_score,
        "home_team_id":         home_team_id,
        "away_team_id":         away_team_id,
    })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detect_pulls_finds_away_team_pull() -> None:
    pbp = _pbp_with_pull(pulling_side="away", pull_at_secs=1100, pull_back_at_secs=1170,
                          home_score=2, away_score=1)
    events = detect_pulls(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    assert len(events) == 1
    r = events.row(0, named=True)
    assert r["team"] == "TOR"
    assert r["period"] == 3
    assert r["deficit"] == 1
    assert r["t_pull_secs"] == 3600.0 - 100.0  # period 3 → (3-1)*1200 + 1100 = 3500
    # 3 periods regulation = 3600s; tr = 3600 − 3500 = 100s
    assert r["time_remaining_secs"] == pytest.approx(100.0, abs=1.0)


def test_detect_pulls_skips_short_flicker() -> None:
    # Delayed-penalty flicker — pull lasts only 2 seconds
    pbp = _pbp_with_pull(pull_at_secs=1100, pull_back_at_secs=1102,
                          home_score=2, away_score=1)
    events = detect_pulls(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    assert events.is_empty()


def test_detect_pulls_ignores_early_periods() -> None:
    pbp = _pbp_with_pull(pull_period=1, pull_at_secs=500, pull_back_at_secs=560,
                          home_score=2, away_score=1)
    events = detect_pulls(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    assert events.is_empty()


def test_detect_pulls_clamps_deficit_at_3() -> None:
    pbp = _pbp_with_pull(pulling_side="away", pull_at_secs=1100, pull_back_at_secs=1170,
                          home_score=5, away_score=1)
    events = detect_pulls(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    assert len(events) == 1
    assert events.row(0, named=True)["deficit"] == 3


def test_aggregate_pulls_groups_by_team_and_deficit() -> None:
    pbp_g1 = _pbp_with_pull(pulling_side="away", pull_at_secs=1100, pull_back_at_secs=1170,
                            home_score=2, away_score=1)
    pbp_g2 = _pbp_with_pull(pulling_side="away", pull_at_secs=1050, pull_back_at_secs=1170,
                            home_score=3, away_score=1)
    # Rewrite g2's game_id and adjust IDs to keep it as TOR pulling
    pbp_g2 = pbp_g2.with_columns(pl.col("game_id").map_elements(lambda _: 2, return_dtype=pl.Int64))
    pbp = pl.concat([pbp_g1, pbp_g2])
    events = detect_pulls(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    assert len(events) == 2
    summary = aggregate_pulls(events, pbp, team_lookup={8: "MTL", 10: "TOR"})
    assert not summary.is_empty()
    assert set(summary.columns) == set(GOALIE_PULL_SCHEMA.keys())

    # TOR pulled twice, once with 1-goal deficit, once with 2-goal deficit
    tor = summary.filter(pl.col("team") == "TOR")
    assert tor["n_pulls"].sum() == 2


def test_aggregate_pulls_warns_on_empty() -> None:
    with pytest.warns(DataMissingWarning):
        df = aggregate_pulls(pl.DataFrame(schema=PULL_EVENTS_SCHEMA), pl.DataFrame())
    assert df.is_empty()


def test_compute_goalie_pull_one_shot() -> None:
    pbp = _pbp_with_pull(pulling_side="away", pull_at_secs=1100, pull_back_at_secs=1170,
                          home_score=2, away_score=1)
    summary, events = compute_goalie_pull(pbp, season=2025,
                                           team_lookup={8: "MTL", 10: "TOR"})
    assert not events.is_empty()
    assert not summary.is_empty()


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    pbp = _pbp_with_pull(pulling_side="away", pull_at_secs=1100, pull_back_at_secs=1170,
                          home_score=2, away_score=1)
    summary, events = compute_goalie_pull(pbp, season=2025,
                                           team_lookup={8: "MTL", 10: "TOR"})
    write_goalie_pull(summary, events, tmp_path / "goalie_pull", season=2025)
    rt = read_goalie_pull(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(summary)
