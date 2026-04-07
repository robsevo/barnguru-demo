"""compute_hot_hand — Feature 2.28 script.

Loads per-player per-game shot/goal data (from MoneyPuck or NHL PBP
parquets) and computes the 5-game burst hot-hand score.

The hot-hand score is a Z-score of (goals − xG) over a 5-game rolling
window, Bayesian-shrunk by career games played.  It complements the
20-game EWMA form metric (Feature 2.13).

Output files written to ``<data_dir>/hot_hand/``:
    hot_hand_{season}.parquet          — per-game detail
    hot_hand_summary_{season}.parquet  — one row per player (latest snapshot)

Usage::

    uv run python scripts/compute_hot_hand.py
    uv run python scripts/compute_hot_hand.py --seasons 2022 2023 2024 2025
    uv run python scripts/compute_hot_hand.py --season 2025 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from models.hot_hand_model import (
    BURST_WINDOW,
    FULL_SIGNAL_GP,
    MIN_SEASON_GP,
    HotHandModel,
    write_hot_hand,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR   = Path.home() / ".gretzky" / "data"
DEFAULT_SEASONS    = [2022, 2023, 2024, 2025]
HOT_HAND_SUBDIR    = "hot_hand"
RAW_SUBDIR         = "raw"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_player_game_stats(data_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Load per-player per-game stats from MoneyPuck or NHL PBP parquets.

    Accepted filename patterns (checked in order):
      - ``moneypuck_skaters_{season}.parquet``
      - ``player_game_stats_{season}.parquet``
      - ``pbp_player_stats_{season}.parquet``

    Required output columns: player_id, game_id, goals, xg_expected.
    Optional columns:        player_name, career_gp.
    """
    raw_dir = data_dir / RAW_SUBDIR
    frames: list[pl.DataFrame] = []

    for season in seasons:
        loaded = False
        for pattern in (
            f"moneypuck_skaters_{season}.parquet",
            f"player_game_stats_{season}.parquet",
            f"pbp_player_stats_{season}.parquet",
        ):
            f = raw_dir / pattern
            if f.exists():
                try:
                    df = pl.read_parquet(f)
                    df = df.with_columns(pl.lit(season).cast(pl.Int64).alias("_src_season"))
                    frames.append(df)
                    print(f"  Loaded: {f.name}  ({len(df):,} rows)")
                    loaded = True
                    break
                except Exception as e:
                    print(f"  Warning: could not load {f.name}: {e}")
        if not loaded:
            print(f"  [warn] No player-game stats found for season {season} — skipping.")

    if not frames:
        # ── PBP fallback: derive per-player per-game goal counts from PBP ──
        print("  [info] Falling back to PBP-derived per-game stats…")
        pbp_frames: list[pl.DataFrame] = []
        for season in seasons:
            pbp_path = raw_dir / f"pbp_{season}.parquet"
            if not pbp_path.exists():
                continue
            try:
                pbp = pl.read_parquet(pbp_path)
                goals = pbp.filter(pl.col("event_type") == "goal")
                if "scorer_id" not in goals.columns:
                    continue
                # Goals scored per player per game
                scored = (
                    goals
                    .filter(pl.col("scorer_id").is_not_null())
                    .group_by(["game_id", "scorer_id"])
                    .agg(pl.len().alias("goals"))
                    .rename({"scorer_id": "player_id"})
                )
                # Assists (weight 0 for goals, but needed for game appearances)
                a1 = (
                    goals
                    .filter(pl.col("assist1_id").is_not_null())
                    .group_by(["game_id", "assist1_id"])
                    .agg(pl.len().alias("assists"))
                    .rename({"assist1_id": "player_id"})
                )
                # All players who appeared (scored or assisted) in each game
                all_players = pl.concat([
                    scored.select(["game_id", "player_id"]),
                    a1.select(["game_id", "player_id"]),
                ]).unique(["game_id", "player_id"])

                derived = (
                    all_players
                    .join(scored, on=["game_id", "player_id"], how="left")
                    .with_columns([
                        pl.col("goals").fill_null(0).cast(pl.Int64),
                        # No xG from PBP alone — use 0.3 × goals as rough proxy
                        (pl.col("goals").fill_null(0).cast(pl.Float64) * 0.3).alias("xg_expected"),
                        pl.lit(season).cast(pl.Int64).alias("_src_season"),
                    ])
                )
                pbp_frames.append(derived)
                print(f"  Derived from PBP: pbp_{season}.parquet  ({len(derived):,} player-game rows)")
            except Exception as e:
                print(f"  [warn] Could not derive from pbp_{season}.parquet: {e}")

        if not pbp_frames:
            return pl.DataFrame()
        return pl.concat(pbp_frames, how="diagonal")

    return pl.concat(frames, how="diagonal")


def _normalise_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename common alternative column names to the canonical schema.

    MoneyPuck column names differ from the internal canonical names.
    This function bridges the gap without requiring upstream changes.
    """
    renames: dict[str, str] = {}

    # player_id alternatives
    for alt in ("playerId", "player_id_nhl", "nhl_player_id"):
        if alt in df.columns and "player_id" not in df.columns:
            renames[alt] = "player_id"

    # game_id alternatives
    for alt in ("gameId", "game_id_nhl"):
        if alt in df.columns and "game_id" not in df.columns:
            renames[alt] = "game_id"

    # goals alternatives
    for alt in ("I_F_goals", "goals_for", "g"):
        if alt in df.columns and "goals" not in df.columns:
            renames[alt] = "goals"

    # xg alternatives
    for alt in ("I_F_xGoals", "xGoals", "xg", "i_f_xgoals", "xgoals"):
        if alt in df.columns and "xg_expected" not in df.columns:
            renames[alt] = "xg_expected"

    # player_name alternatives
    for alt in ("name", "playerName", "player_name_full"):
        if alt in df.columns and "player_name" not in df.columns:
            renames[alt] = "player_name"

    if renames:
        df = df.rename(renames)

    return df


def _build_career_gp(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``career_gp`` column if absent.

    Approximates career GP as the total number of game rows per player
    across all loaded seasons combined.  This is a conservative lower bound
    (some seasons may be missing); the Bayesian shrinkage handles uncertainty
    gracefully — underestimating career_gp just means more shrinkage toward
    zero, which is safe.

    If ``career_gp`` is already present, it is left unchanged.
    """
    if "career_gp" in df.columns:
        return df

    gp_per_player = (
        df
        .group_by("player_id")
        .agg(pl.len().alias("career_gp"))
    )
    return df.join(gp_per_player, on="player_id", how="left")


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_leaderboard(summary_df: pl.DataFrame, n: int = 15) -> None:
    """Print top/bottom players by hot_hand_score."""
    if summary_df.is_empty():
        return

    top = summary_df.sort("hot_hand_score", descending=True).head(n)
    print(f"\n  Top-{n} hot hands (highest burst score):")
    print(
        f"  {'Player':<25}  {'GP':>4}  {'G (5g)':>7}  "
        f"{'xG (5g)':>8}  {'G−xG (5g)':>10}  {'Score':>6}  {'Shrink':>7}"
    )
    print(f"  {'─'*25}  {'─'*4}  {'─'*7}  {'─'*8}  {'─'*10}  {'─'*6}  {'─'*7}")
    for row in top.to_dicts():
        name = (row.get("player_name") or "")[:24] or f"ID:{row['player_id']}"
        print(
            f"  {name:<25}  {row['games_played']:>4}  "
            f"{row['goals_5g']:>7.2f}  {row['xg_5g']:>8.2f}  "
            f"{row['goals_vs_xg_5g']:>+10.2f}  "
            f"{row['hot_hand_score']:>+6.2f}  "
            f"{row['shrinkage_factor']:>7.3f}"
        )

    bottom = summary_df.sort("hot_hand_score", descending=False).head(n)
    print(f"\n  Bottom-{n} (coldest burst):")
    print(
        f"  {'Player':<25}  {'GP':>4}  {'Score':>6}"
    )
    print(f"  {'─'*25}  {'─'*4}  {'─'*6}")
    for row in bottom.to_dicts():
        name = (row.get("player_name") or "")[:24] or f"ID:{row['player_id']}"
        print(f"  {name:<25}  {row['games_played']:>4}  {row['hot_hand_score']:>+6.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 5-game burst hot-hand signal per player (Feature 2.28)."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        metavar="YEAR",
        help=f"Seasons to load player-game data for (default: {DEFAULT_SEASONS}).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        metavar="YEAR",
        help="Target season for output file naming (default: max of --seasons).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root data directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output parquets.",
    )
    args = parser.parse_args()

    seasons: list[int] = sorted(args.seasons)
    target_season: int = args.season if args.season is not None else max(seasons)
    data_dir: Path     = args.data_dir

    print(f"[hot-hand] Feature 2.28 — Short-Burst Hot Hand Signal")
    print(f"  Burst window : {BURST_WINDOW} games")
    print(f"  Shrinkage GP : {FULL_SIGNAL_GP} (full signal at ≥ {FULL_SIGNAL_GP} career GP)")
    print(f"  Min games    : {MIN_SEASON_GP} before non-zero score")
    print(f"  Seasons      : {seasons}")
    print(f"  Target season: {target_season}")
    print(f"  Data dir     : {data_dir}")

    # ── Check output ─────────────────────────────────────────────────────────
    out_dir      = data_dir / HOT_HAND_SUBDIR
    out_game     = out_dir / f"hot_hand_{target_season}.parquet"
    out_summary  = out_dir / f"hot_hand_summary_{target_season}.parquet"

    if out_game.exists() and out_summary.exists() and not args.force:
        print(f"\n[hot-hand] Output already exists: {out_game}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    # ── Load data ────────────────────────────────────────────────────────────
    raw_dir = data_dir / RAW_SUBDIR
    if not raw_dir.exists():
        print(f"\n[hot-hand] Raw data directory not found: {raw_dir}")
        print("  Run 'uv run python scripts/gretzky.py sync' or 'ingest' first.")
        sys.exit(1)

    print(f"\n[hot-hand] Loading player-game stats for seasons: {seasons}")
    df = _load_player_game_stats(data_dir, seasons)

    if df.is_empty():
        print("[hot-hand] No player-game data loaded.")
        print(
            "  Expected files under:\n"
            f"    {raw_dir}/\n"
            "  e.g. moneypuck_skaters_{season}.parquet or player_game_stats_{season}.parquet"
        )
        print("  Run 'uv run python scripts/gretzky.py sync' first.")
        sys.exit(1)

    print(f"  {len(df):,} rows loaded across all seasons")

    # ── Normalise columns ─────────────────────────────────────────────────────
    df = _normalise_columns(df)

    required = {"player_id", "game_id", "goals", "xg_expected"}
    missing  = required - set(df.columns)
    if missing:
        print(f"\n[hot-hand] ERROR: Required columns missing after normalisation: {sorted(missing)}")
        print(f"  Available columns: {df.columns}")
        sys.exit(1)

    # ── Build career_gp if absent ─────────────────────────────────────────────
    df = _build_career_gp(df)

    # ── Filter to target season only (for this run's output) ─────────────────
    if "_src_season" in df.columns:
        df_season = df.filter(pl.col("_src_season") == target_season)
        if df_season.is_empty():
            print(f"\n[hot-hand] No rows found for target season {target_season}.")
            print(f"  Loaded seasons: {seasons}")
            sys.exit(1)
        n_players = df_season["player_id"].n_unique()
        n_games   = df_season["game_id"].n_unique()
        print(f"\n[hot-hand] Season {target_season}: {n_players:,} players × ~{n_games:,} games")
    else:
        df_season = df
        n_players = df["player_id"].n_unique()
        print(f"\n[hot-hand] {n_players:,} unique players")

    # ── Compute ───────────────────────────────────────────────────────────────
    print(f"\n[hot-hand] Computing hot-hand scores…")
    model    = HotHandModel(window=BURST_WINDOW)
    game_df  = model.compute(df_season, season=target_season)
    summ_df  = model.summary(df_season, season=target_season)

    # ── Print summary ─────────────────────────────────────────────────────────
    n_positive = int((summ_df["hot_hand_score"] > 0.5).sum())
    n_negative = int((summ_df["hot_hand_score"] < -0.5).sum())
    print(f"  Players with score > +0.5 (hot):    {n_positive:,}")
    print(f"  Players with score < -0.5 (cold):   {n_negative:,}")
    print(f"  Players scored:                      {len(summ_df):,}")

    _print_leaderboard(summ_df)

    # ── Save ──────────────────────────────────────────────────────────────────
    p_game, p_summary = write_hot_hand(game_df, summ_df, out_dir, target_season)
    print(f"\n[hot-hand] Per-game detail written : {p_game}")
    print(f"[hot-hand] Player summary written  : {p_summary}")
    print(f"\n[hot-hand] Done.")


if __name__ == "__main__":
    main()
