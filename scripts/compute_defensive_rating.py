"""Compute Composite Defensive Rating (CDR) — Feature 2.26.

Loads Evolving Hockey RAPM data + historical PBP player stats, computes the
CDR per player, and writes a parquet to ~/.gretzky/data/cdr/.

Usage:
    uv run python scripts/compute_defensive_rating.py
    uv run python scripts/compute_defensive_rating.py --season 2025
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))

import polars as pl

from models.defensive_rating import CompositeDefensiveRating, write_cdr


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _data_dir() -> Path:
    default = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
    return Path(os.environ.get("GRETZKY_DATA_DIR", str(default)))


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_rapm(data_dir: Path, season: int) -> pl.DataFrame:
    """Load internally computed RAPM parquet for *season*.

    Evolving Hockey was removed — it moved behind a paywall.
    RAPM is computed from NHL shift + MoneyPuck shot data via gretzky train-rapm.
    """
    path = data_dir / "rapm" / f"rapm_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"RAPM parquet not found: {path}\n"
            f"Run: uv run python scripts/gretzky.py ingest && "
            f"uv run python scripts/gretzky.py train-rapm"
        )
    df = pl.read_parquet(path)
    # Map internal schema → CDR-expected schema:
    #   toi_ev       → toi
    #   rapm_ev_def  → rapm_xga  (sign flip: internal lower=better → CDR higher=better)
    #   xga_60       → 0.0 (not in internal RAPM; component will be neutral/equal for all)
    df = df.rename({"toi_ev": "toi"}).with_columns(
        (-pl.col("rapm_ev_def")).alias("rapm_xga"),
    )
    # xga_60 is computed from on-ice xG exposure in train_rapm; fall back to 0.0
    # for parquets trained before this column was added.
    if "xga_60" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("xga_60"))
    return df


def _load_pbp_stats(data_dir: Path, season: int) -> pl.DataFrame:
    """Load historical ingestor player stats (T/G counts) for *season*.

    Falls back to empty DataFrame if not yet available — CDR will use league
    medians for the T/G component instead of per-player values.
    """
    stats_dir = data_dir / "raw"
    path = stats_dir / f"player_stats_{season}.parquet"
    if not path.exists():
        print(
            f"  WARNING: {path} not found — T/G component will use league medians.\n"
            f"  Run: uv run python scripts/run_phase1_sync.py to populate player stats."
        )
        return pl.DataFrame()
    return pl.read_parquet(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Composite Defensive Rating (Feature 2.26)."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season to compute CDR for (default: current NHL season)",
    )
    args = parser.parse_args()

    season = args.season or _current_nhl_season()
    data_dir = _data_dir()
    cdr_dir = data_dir / "cdr"

    print("=" * 60)
    print("  GRTZKY — Composite Defensive Rating (Feature 2.26)")
    print("=" * 60)
    print(f"  Season   : {season}")
    print(f"  Data dir : {data_dir}")
    print()

    # ── 1. Load RAPM ──────────────────────────────────────────────────────────
    print("[ 1/3 ] Loading internally computed RAPM data…")
    try:
        rapm_df = _load_rapm(data_dir, season)
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        print("  Run: uv run python scripts/gretzky.py ingest && "
              "uv run python scripts/gretzky.py train-rapm")
        sys.exit(0)
    print(f"  Loaded {len(rapm_df):,} RAPM records")

    # ── 2. Load PBP player stats (T/G) ────────────────────────────────────────
    print("[ 2/3 ] Loading player stats (giveaways / takeaways)…")
    pbp_stats_df = _load_pbp_stats(data_dir, season)
    if not pbp_stats_df.is_empty():
        n_with_tg = pbp_stats_df.filter(
            pl.col("takeaway_count").is_not_null() | pl.col("giveaway_count").is_not_null()
        ).height if "takeaway_count" in pbp_stats_df.columns else 0
        print(f"  Loaded {len(pbp_stats_df):,} player stat rows  ({n_with_tg} with T/G data)")
    else:
        print("  No player stats available — using league medians for T/G component")

    # ── 3. Compute CDR ────────────────────────────────────────────────────────
    print("[ 3/3 ] Computing CDR…")
    try:
        cdr = CompositeDefensiveRating()
        cdr_df = cdr.compute(rapm_df, pbp_stats_df, season)
    except ValueError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    out_path = write_cdr(cdr_df, cdr_dir, season)

    n_players = len(cdr_df)
    top5    = cdr_df.head(5)
    bottom5 = cdr_df.tail(5).sort("cdr", descending=False)

    print(f"  Players computed : {n_players}")
    print(f"  Parquet saved   → {out_path}")
    print()

    print("  Top 5 defensive (highest CDR):")
    for row in top5.to_dicts():
        print(
            f"    {row['player_name']:<24s}  CDR {row['cdr']:+.2f}"
            f"  (RAPM {row['rapm_xga']:+.2f}, TK/GV {row['tk_gv_ratio']:.2f},"
            f" xGA/60 {row['xga_60']:.2f})"
        )

    print()
    print("  Bottom 5 defensive (lowest CDR):")
    for row in bottom5.to_dicts():
        print(
            f"    {row['player_name']:<24s}  CDR {row['cdr']:+.2f}"
            f"  (RAPM {row['rapm_xga']:+.2f}, TK/GV {row['tk_gv_ratio']:.2f},"
            f" xGA/60 {row['xga_60']:.2f})"
        )

    print()
    print("=" * 60)
    print("  Done. Refresh /phase2 to see CDR on player cards.")
    print("=" * 60)


if __name__ == "__main__":
    main()
