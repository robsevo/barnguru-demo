"""Unit tests for models/line_deployment.py — Feature 4.1.

All tests use synthetic data — no live API calls or on-disk parquet files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.line_deployment import (
    LINE_DEPLOYMENT_SCHEMA,
    MODEL_VERSION,
    N_DEFENSE_PAIRS,
    N_FORWARD_LINES,
    build_position_lookup,
    compute_line_deployment,
    read_line_deployment,
    write_line_deployment,
)


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def _build_synth_inputs(
    n_games: int = 30,
    seed:    int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Construct shifts + pbp + shots for two teams (HOME=8, AWAY=10) so
    that each team has clearly separable lines.

    Team layout:
        Forwards 1–12 (lines: 1-2-3, 4-5-6, 7-8-9, 10-11-12)
        Defense  13–18 (pairs: 13-14, 15-16, 17-18)
        Goalie   19 (excluded from shifts)
    Away team players are offset by 100.
    """
    rng = np.random.default_rng(seed)

    home_id = 8  # MTL
    away_id = 10 # TOR

    home_forwards = list(range(1, 13))
    home_defense  = list(range(13, 19))
    away_forwards = [p + 100 for p in home_forwards]
    away_defense  = [p + 100 for p in home_defense]

    # Lines (groups of 3 F)
    home_F_lines = [home_forwards[i:i + 3] for i in range(0, 12, 3)]
    away_F_lines = [away_forwards[i:i + 3] for i in range(0, 12, 3)]
    # D-pairs
    home_D_pairs = [home_defense[i:i + 2] for i in range(0, 6, 2)]
    away_D_pairs = [away_defense[i:i + 2] for i in range(0, 6, 2)]

    # Target TOI shares: L1 30%, L2 25%, L3 25%, L4 20%; D1 35%, D2 35%, D3 30%
    f_weights = [0.30, 0.25, 0.25, 0.20]
    d_weights = [0.35, 0.35, 0.30]

    shifts_rows: list[dict] = []
    pbp_rows:    list[dict] = []

    for g_idx in range(n_games):
        game_id = 2025020000 + g_idx + 1
        # 3 periods × 20 minutes = 3600 game seconds
        # Generate ~60 shifts per team in a simple alternating pattern
        cur_t = 0.0
        period = 1
        shift_no = 0

        # PBP event marking 5v5 at start of game
        pbp_rows.append({
            "game_id":            game_id,
            "period":             1,
            "time_in_period_secs": 0,
            "home_skaters":       5,
            "away_skaters":       5,
            "home_team_id":       home_id,
            "away_team_id":       away_id,
        })

        while cur_t < 3600 and period <= 3:
            # Pick which line/pair is on by weights
            h_f_idx = int(rng.choice(N_FORWARD_LINES, p=f_weights))
            a_f_idx = int(rng.choice(N_FORWARD_LINES, p=f_weights))
            h_d_idx = int(rng.choice(N_DEFENSE_PAIRS, p=d_weights))
            a_d_idx = int(rng.choice(N_DEFENSE_PAIRS, p=d_weights))

            shift_dur = float(rng.integers(40, 70))   # 40-70 sec shifts

            t_start = cur_t
            t_end   = min(cur_t + shift_dur, 3600.0)
            for p in home_F_lines[h_f_idx]:
                shifts_rows.append({
                    "game_id":             game_id,
                    "player_id":           p,
                    "team_id":             home_id,
                    "period":              ((int(t_start) // 1200) + 1),
                    "shift_number":        shift_no,
                    "duration_secs":       t_end - t_start,
                    "game_seconds_start":  t_start,
                    "game_seconds_end":    t_end,
                })
            for p in home_D_pairs[h_d_idx]:
                shifts_rows.append({
                    "game_id":             game_id,
                    "player_id":           p,
                    "team_id":             home_id,
                    "period":              ((int(t_start) // 1200) + 1),
                    "shift_number":        shift_no,
                    "duration_secs":       t_end - t_start,
                    "game_seconds_start":  t_start,
                    "game_seconds_end":    t_end,
                })
            for p in away_F_lines[a_f_idx]:
                shifts_rows.append({
                    "game_id":             game_id,
                    "player_id":           p,
                    "team_id":             away_id,
                    "period":              ((int(t_start) // 1200) + 1),
                    "shift_number":        shift_no,
                    "duration_secs":       t_end - t_start,
                    "game_seconds_start":  t_start,
                    "game_seconds_end":    t_end,
                })
            for p in away_D_pairs[a_d_idx]:
                shifts_rows.append({
                    "game_id":             game_id,
                    "player_id":           p,
                    "team_id":             away_id,
                    "period":              ((int(t_start) // 1200) + 1),
                    "shift_number":        shift_no,
                    "duration_secs":       t_end - t_start,
                    "game_seconds_start":  t_start,
                    "game_seconds_end":    t_end,
                })

            cur_t = t_end
            shift_no += 1
            if cur_t >= 1200 * period:
                period += 1

    # Synth shots: assign positions for each player
    shots_rows: list[dict] = []
    for p in home_forwards + away_forwards:
        pos = "C" if p % 3 == 1 else ("L" if p % 3 == 2 else "R")
        shots_rows.append({"shooter_id": p, "player_position": pos})
    for p in home_defense + away_defense:
        shots_rows.append({"shooter_id": p, "player_position": "D"})

    shifts_df = pl.DataFrame(shifts_rows)
    pbp_df    = pl.DataFrame(pbp_rows)
    shots_df  = pl.DataFrame(shots_rows)
    return shifts_df, pbp_df, shots_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_position_lookup_classifies_F_and_D() -> None:
    df = pl.DataFrame({
        "shooter_id":      [1, 1, 2, 3, 4, 5],
        "player_position": ["C", "C", "L", "R", "D", "D"],
    })
    pos = build_position_lookup(df)
    assert pos[1] == "F"
    assert pos[2] == "F"
    assert pos[3] == "F"
    assert pos[4] == "D"
    assert pos[5] == "D"


def test_build_position_lookup_handles_empty() -> None:
    df = pl.DataFrame(schema={"shooter_id": pl.Int64, "player_position": pl.Utf8})
    assert build_position_lookup(df) == {}


def test_compute_line_deployment_recovers_lines_and_pairs() -> None:
    shifts_df, pbp_df, shots_df = _build_synth_inputs(n_games=30, seed=7)

    df = compute_line_deployment(
        shifts_df, pbp_df, shots_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )

    # Schema check
    assert set(df.columns) == set(LINE_DEPLOYMENT_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()

    # Two teams, each has 4 F lines + 3 D pairs
    teams = df["team"].unique().to_list()
    assert set(teams) == {"MTL", "TOR"}

    mtl_f = df.filter((pl.col("team") == "MTL") & (pl.col("line_type") == "F"))
    mtl_d = df.filter((pl.col("team") == "MTL") & (pl.col("line_type") == "D"))
    assert len(mtl_f) == N_FORWARD_LINES
    assert len(mtl_d) == N_DEFENSE_PAIRS

    # Verify line ranks are 1..4 / 1..3
    assert sorted(mtl_f["line_rank"].to_list()) == [1, 2, 3, 4]
    assert sorted(mtl_d["line_rank"].to_list()) == [1, 2, 3]

    # Forward lines for MTL should be subsets of {1..12}; D pairs of {13..18}
    for r in mtl_f.iter_rows(named=True):
        assert set(filter(lambda x: x is not None,
                          [r["player_1"], r["player_2"], r["player_3"]])).issubset(set(range(1, 13)))
    for r in mtl_d.iter_rows(named=True):
        pair = {r["player_1"], r["player_2"]}
        assert pair.issubset(set(range(13, 19)))

    # L1 should have the highest share among forward lines
    l1_share = mtl_f.filter(pl.col("line_rank") == 1)["share_of_team_toi"][0]
    l4_share = mtl_f.filter(pl.col("line_rank") == 4)["share_of_team_toi"][0]
    assert l1_share >= l4_share

    # Greedy assembly recovers exact line membership (no shuffling synth)
    actual_lines = set()
    for r in mtl_f.iter_rows(named=True):
        actual_lines.add(frozenset([r["player_1"], r["player_2"], r["player_3"]]))
    expected = {frozenset(range(i, i + 3)) for i in range(1, 13, 3)}
    assert actual_lines == expected


def test_compute_line_deployment_min_games_threshold() -> None:
    """Teams with too few games should be filtered out."""
    shifts_df, pbp_df, shots_df = _build_synth_inputs(n_games=3, seed=1)
    df = compute_line_deployment(
        shifts_df, pbp_df, shots_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    # MIN_TEAM_GAMES = 5 → 3 games should produce no output
    assert df.is_empty()


def test_compute_line_deployment_empty_inputs() -> None:
    df = compute_line_deployment(
        pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), season=2025,
    )
    assert df.is_empty()
    assert set(df.columns) == set(LINE_DEPLOYMENT_SCHEMA.keys())


def test_compute_line_deployment_shares_sum_to_expected_total() -> None:
    """Forward and defense shares should each lie in [0, 1] and the 4
    lines / 3 pairs cover roughly the full 5v5 deployment."""
    shifts_df, pbp_df, shots_df = _build_synth_inputs(n_games=30, seed=11)
    df = compute_line_deployment(
        shifts_df, pbp_df, shots_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    for team in ("MTL", "TOR"):
        f_share = df.filter((pl.col("team") == team) & (pl.col("line_type") == "F"))["share_of_team_toi"].sum()
        d_share = df.filter((pl.col("team") == team) & (pl.col("line_type") == "D"))["share_of_team_toi"].sum()
        assert 0.5 <= f_share <= 1.5  # may sum >1 because line members overlap on ice with others
        assert 0.5 <= d_share <= 1.5


def test_line_toi_per_game_and_cohesion_fields() -> None:
    """line_toi_per_game must be >= trio_toi_per_game (each member plays
    at least as much as the trio plays together) and cohesion_pct ∈ [0, 1].
    Synth has perfectly locked lines, so cohesion should be ~1.0."""
    shifts_df, pbp_df, shots_df = _build_synth_inputs(n_games=30, seed=17)
    df = compute_line_deployment(
        shifts_df, pbp_df, shots_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    assert (df["line_toi_per_game"] >= df["trio_toi_per_game"] - 1e-6).all()
    assert (df["cohesion_pct"] >= 0.0).all()
    assert (df["cohesion_pct"] <= 1.0 + 1e-9).all()
    # Synth deploys lines as locked trios — cohesion should be near 1.0
    # for the highest-rank assembled units
    top_lines = df.filter(pl.col("line_rank") == 1)
    assert (top_lines["cohesion_pct"] > 0.90).all(), (
        f"locked synth lines should have cohesion ≈ 1.0; got {top_lines['cohesion_pct'].to_list()}"
    )


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    shifts_df, pbp_df, shots_df = _build_synth_inputs(n_games=30, seed=3)
    df = compute_line_deployment(
        shifts_df, pbp_df, shots_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    assert not df.is_empty()

    out_dir = tmp_path / "line_deployment"
    write_line_deployment(df, out_dir, season=2025)

    rt = read_line_deployment(tmp_path, season=2025)
    assert rt is not None
    assert len(rt) == len(df)
    assert set(rt.columns) == set(df.columns)

    # Latest-available read
    latest = read_line_deployment(tmp_path)
    assert latest is not None and len(latest) == len(df)
