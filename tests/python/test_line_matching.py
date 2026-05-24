"""Unit tests for models/line_matching.py — Feature 4.2.

All tests use synthetic data — no live API calls or on-disk parquet files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from models.line_matching import (
    LINE_MATCHING_SCHEMA,
    MIN_PAIR_SECS,
    MODEL_VERSION,
    compute_line_matching,
    read_line_matching,
    team_matchup_profile,
    write_line_matching,
)


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------


def _build_inputs(
    n_games: int = 20,
    seed:    int = 7,
    last_change_advantage: float = 0.5,
):
    """Generate synthetic shifts + pbp + line-deployment for one matchup
    (HOME=8 MTL vs AWAY=10 TOR).

    `last_change_advantage` controls how aggressively the home team
    matches its L1 against the road's L1 (0 = no matching, 1 = perfect).
    """
    rng = np.random.default_rng(seed)

    home_id = 8
    away_id = 10

    home_F = [list(range(1, 4)), list(range(4, 7)),
              list(range(7, 10)), list(range(10, 13))]
    home_D = [list(range(13, 15)), list(range(15, 17)), list(range(17, 19))]
    away_F = [[p + 100 for p in line] for line in home_F]
    away_D = [[p + 100 for p in pair] for pair in home_D]

    shifts_rows: list[dict] = []
    pbp_rows:    list[dict] = []

    for g_idx in range(n_games):
        game_id = 2025020000 + g_idx + 1
        pbp_rows.append({
            "game_id":             game_id,
            "period":              1,
            "time_in_period_secs": 0,
            "home_skaters":        5,
            "away_skaters":        5,
            "home_team_id":        home_id,
            "away_team_id":        away_id,
        })

        cur_t = 0.0
        shift_no = 0
        while cur_t < 3600:
            # Away picks line uniformly
            a_f_idx = int(rng.integers(0, 4))
            a_d_idx = int(rng.integers(0, 3))

            # Home matches AWAY L1 with HOME L1 with probability
            # `last_change_advantage`; otherwise random
            if a_f_idx == 0 and rng.random() < last_change_advantage:
                h_f_idx = 0
            else:
                h_f_idx = int(rng.integers(0, 4))
            h_d_idx = int(rng.integers(0, 3))

            dur = float(rng.integers(40, 70))
            t_start = cur_t
            t_end   = min(cur_t + dur, 3600.0)

            for p in home_F[h_f_idx]:
                shifts_rows.append({
                    "game_id":            game_id,
                    "player_id":          p,
                    "team_id":            home_id,
                    "period":             1,
                    "shift_number":       shift_no,
                    "duration_secs":      t_end - t_start,
                    "game_seconds_start": t_start,
                    "game_seconds_end":   t_end,
                })
            for p in home_D[h_d_idx]:
                shifts_rows.append({
                    "game_id":            game_id,
                    "player_id":          p,
                    "team_id":            home_id,
                    "period":             1,
                    "shift_number":       shift_no,
                    "duration_secs":      t_end - t_start,
                    "game_seconds_start": t_start,
                    "game_seconds_end":   t_end,
                })
            for p in away_F[a_f_idx]:
                shifts_rows.append({
                    "game_id":            game_id,
                    "player_id":          p,
                    "team_id":            away_id,
                    "period":             1,
                    "shift_number":       shift_no,
                    "duration_secs":      t_end - t_start,
                    "game_seconds_start": t_start,
                    "game_seconds_end":   t_end,
                })
            for p in away_D[a_d_idx]:
                shifts_rows.append({
                    "game_id":            game_id,
                    "player_id":          p,
                    "team_id":            away_id,
                    "period":             1,
                    "shift_number":       shift_no,
                    "duration_secs":      t_end - t_start,
                    "game_seconds_start": t_start,
                    "game_seconds_end":   t_end,
                })
            cur_t = t_end
            shift_no += 1

    # Build line-deployment frame from the actual lines we used
    ldep_rows: list[dict] = []
    for rank, line in enumerate(home_F, start=1):
        ldep_rows.append({
            "team": "MTL", "season": 2025, "line_type": "F",
            "line_rank": rank, "player_1": line[0], "player_2": line[1], "player_3": line[2],
            "chemistry_toi_secs": 1.0, "projected_toi_per_game": 1.0,
            "share_of_team_toi": 0.25, "team_gp": n_games, "model_version": "line_deployment_v1",
        })
    for rank, pair in enumerate(home_D, start=1):
        ldep_rows.append({
            "team": "MTL", "season": 2025, "line_type": "D",
            "line_rank": rank, "player_1": pair[0], "player_2": pair[1], "player_3": None,
            "chemistry_toi_secs": 1.0, "projected_toi_per_game": 1.0,
            "share_of_team_toi": 0.3, "team_gp": n_games, "model_version": "line_deployment_v1",
        })
    for rank, line in enumerate(away_F, start=1):
        ldep_rows.append({
            "team": "TOR", "season": 2025, "line_type": "F",
            "line_rank": rank, "player_1": line[0], "player_2": line[1], "player_3": line[2],
            "chemistry_toi_secs": 1.0, "projected_toi_per_game": 1.0,
            "share_of_team_toi": 0.25, "team_gp": n_games, "model_version": "line_deployment_v1",
        })
    for rank, pair in enumerate(away_D, start=1):
        ldep_rows.append({
            "team": "TOR", "season": 2025, "line_type": "D",
            "line_rank": rank, "player_1": pair[0], "player_2": pair[1], "player_3": None,
            "chemistry_toi_secs": 1.0, "projected_toi_per_game": 1.0,
            "share_of_team_toi": 0.3, "team_gp": n_games, "model_version": "line_deployment_v1",
        })

    return (
        pl.DataFrame(shifts_rows),
        pl.DataFrame(pbp_rows),
        pl.DataFrame(ldep_rows),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_schema() -> None:
    shifts_df, pbp_df, ldep_df = _build_inputs(n_games=20, seed=1, last_change_advantage=0.8)
    df = compute_line_matching(
        shifts_df, pbp_df, ldep_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    assert set(df.columns) == set(LINE_MATCHING_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()


def test_share_sums_to_one_per_opp_line() -> None:
    """For each (focal_team, opponent, line_type, opp_line_rank, venue) the
    share_of_opp_line values across own_line_rank should sum to ~1."""
    shifts_df, pbp_df, ldep_df = _build_inputs(n_games=30, seed=2, last_change_advantage=0.7)
    df = compute_line_matching(
        shifts_df, pbp_df, ldep_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    assert not df.is_empty()

    grouped = (
        df.group_by(["team", "opponent", "line_type", "opp_line_rank", "venue"])
        .agg(pl.col("share_of_opp_line").sum().alias("sum"))
    )
    for s in grouped["sum"].to_list():
        assert 0.99 < s < 1.01, f"share sum {s} not ~1"


def test_last_change_advantage_drives_matching() -> None:
    """When home team has high last-change advantage, its L1's share of
    away L1's TOI should clearly exceed its L1's share of away L4's TOI."""
    shifts_df, pbp_df, ldep_df = _build_inputs(n_games=80, seed=3, last_change_advantage=0.9)
    df = compute_line_matching(
        shifts_df, pbp_df, ldep_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    # Filter to MTL @home vs TOR's L1
    sub = df.filter(
        (pl.col("team") == "MTL")
        & (pl.col("opponent") == "TOR")
        & (pl.col("line_type") == "F")
        & (pl.col("venue") == "home")
        & (pl.col("opp_line_rank") == 1)
        & (pl.col("own_line_rank") == 1)
    )
    assert not sub.is_empty()
    mtl_l1_share_vs_l1 = float(sub["share_of_opp_line"][0])

    sub_vs_l4 = df.filter(
        (pl.col("team") == "MTL")
        & (pl.col("opponent") == "TOR")
        & (pl.col("line_type") == "F")
        & (pl.col("venue") == "home")
        & (pl.col("opp_line_rank") == 4)
        & (pl.col("own_line_rank") == 1)
    )
    mtl_l1_share_vs_l4 = float(sub_vs_l4["share_of_opp_line"][0]) if not sub_vs_l4.is_empty() else 0.0

    assert mtl_l1_share_vs_l1 > mtl_l1_share_vs_l4 + 0.15, (
        f"home L1 matched against L1 ({mtl_l1_share_vs_l1:.3f}) "
        f"should clearly exceed home L1 vs L4 ({mtl_l1_share_vs_l4:.3f})"
    )


def test_empty_inputs() -> None:
    df = compute_line_matching(
        pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), season=2025,
    )
    assert df.is_empty()
    assert set(df.columns) == set(LINE_MATCHING_SCHEMA.keys())


def test_min_pair_secs_threshold() -> None:
    """Pairs with very little TOI should be filtered out."""
    shifts_df, pbp_df, ldep_df = _build_inputs(n_games=20, seed=4, last_change_advantage=0.5)
    df = compute_line_matching(
        shifts_df, pbp_df, ldep_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    assert (df["matchup_toi_secs"] >= MIN_PAIR_SECS).all()


def test_team_matchup_profile() -> None:
    shifts_df, pbp_df, ldep_df = _build_inputs(n_games=30, seed=5, last_change_advantage=0.7)
    df = compute_line_matching(
        shifts_df, pbp_df, ldep_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    prof = team_matchup_profile(df, "MTL", line_type="F")
    assert not prof.is_empty()
    # Individual cells must lie in [0, 1]
    assert (prof["weighted_share"] >= 0.0).all()
    assert (prof["weighted_share"] <= 1.0 + 1e-9).all()
    # The four-row column for any (opp_rank, venue) must sum to ~1.0 —
    # this regression-locks the share-normalization fix.
    sums = (
        prof.group_by(["opp_line_rank", "venue"])
        .agg(pl.col("weighted_share").sum().alias("s"))
    )
    for s in sums["s"].to_list():
        assert 0.99 < s < 1.01, f"profile column sum {s} should be ~1.0"


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    shifts_df, pbp_df, ldep_df = _build_inputs(n_games=20, seed=6, last_change_advantage=0.5)
    df = compute_line_matching(
        shifts_df, pbp_df, ldep_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    write_line_matching(df, tmp_path / "line_matching", season=2025)
    rt = read_line_matching(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
