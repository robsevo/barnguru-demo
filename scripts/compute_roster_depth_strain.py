"""compute_roster_depth_strain — Feature 3.16 driver script.

Reads the active roster cache + the latest injury-status snapshot
(Feature 3.13 output) and writes per-skater roster-depth-strain rows:
for each team, how much of the IR'd players' baseline TOI gets
redistributed across the healthy skaters.

Baseline TOI per player is the *mean game TOI* from the most recent
``toi_*.parquet`` ingest. Players with no TOI history get a
``DEFAULT_BASELINE_SECS`` placeholder so the team's strain math still
captures their absence — though their contribution is small by design.

Usage::

    uv run python scripts/gretzky.py roster-depth-strain
    uv run python scripts/gretzky.py roster-depth-strain -- --date 2026-05-17
    uv run python scripts/gretzky.py roster-depth-strain -- --force
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from data.data_store import DataStore
from models.injury_status_integrator import STATUS_OUT
from models.roster_depth_strain import (
    RosterDepthStrain,
    write_roster_depth_strain,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
ROSTER_DEPTH_STRAIN_SUBDIR = "roster_depth_strain"
INJURY_STATUS_SUBDIR       = "injury_status"

# Fallback per-game TOI in seconds for players with no recorded shift data
# (typical 4th-line forward ≈ 10 min).
DEFAULT_BASELINE_SECS = 600


def _baseline_toi(store: DataStore) -> pl.DataFrame:
    """Per-player mean TOI seconds from the most-recent toi parquet."""
    toi = store.toi()
    if len(toi) == 0 or "toi_total_secs" not in toi.columns:
        return pl.DataFrame(schema={"player_id": pl.Int64,
                                    "baseline_toi_secs": pl.Int64})
    return (
        toi.filter(pl.col("player_id").is_not_null())
           .group_by("player_id")
           .agg(pl.col("toi_total_secs").mean().alias("baseline_toi_secs"))
           .with_columns(
               pl.col("baseline_toi_secs").cast(pl.Int64),
           )
    )


def _latest_injury_status(args, as_of: str) -> pl.DataFrame:
    """Latest injury-status snapshot from Feature 3.13 output."""
    inj_dir = args.data_dir / INJURY_STATUS_SUBDIR
    target  = inj_dir / f"injury_status_{as_of}.parquet"
    if target.exists():
        return pl.read_parquet(target)
    # Fall back to whatever the most recent snapshot is.
    if not inj_dir.exists():
        return pl.DataFrame(schema={"player_id": pl.Int64, "status": pl.Utf8})
    candidates = sorted(inj_dir.glob("injury_status_*.parquet"))
    if not candidates:
        return pl.DataFrame(schema={"player_id": pl.Int64, "status": pl.Utf8})
    return pl.read_parquet(candidates[-1])


def _build_roster_df(store: DataStore, injury: pl.DataFrame) -> pl.DataFrame:
    """Stitch active roster × baseline TOI × IR flag into the model input."""
    # Active roster
    raw = store.roster()
    if len(raw) == 0:
        print("[roster-depth-strain] DataStore.roster() returned no players.")
        sys.exit(1)

    cols = raw.columns
    name_col = "full_name" if "full_name" in cols else "player_name"
    team_col = "team_abbrev" if "team_abbrev" in cols else "team"
    pos_col  = "position" if "position" in cols else (
        "position_code" if "position_code" in cols else None
    )
    if "player_id" not in cols or team_col not in cols or pos_col is None:
        print("[roster-depth-strain] Roster cache missing required cols.")
        sys.exit(1)

    roster = (
        raw.select(["player_id", team_col, pos_col])
           .rename({team_col: "team", pos_col: "position"})
           .with_columns(
               pl.col("team").cast(pl.Utf8).str.to_uppercase(),
               pl.col("position").cast(pl.Utf8),
           )
           .unique(subset=["player_id"])
    )

    # IR membership from injury_status snapshot
    if len(injury) > 0 and "status" in injury.columns:
        ir_set = set(
            injury.filter(pl.col("status") == STATUS_OUT)["player_id"].to_list()
        )
    else:
        ir_set = set()

    # Baseline TOI per player
    baseline = _baseline_toi(store)

    roster = roster.join(baseline, on="player_id", how="left").with_columns(
        pl.col("baseline_toi_secs").fill_null(DEFAULT_BASELINE_SECS).cast(pl.Int64),
        pl.col("player_id").is_in(list(ir_set)).alias("is_on_ir"),
    )
    return roster


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-skater roster-depth strain (Feature 3.16)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None,
                        help="as-of date. Default: today (UTC).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / ROSTER_DEPTH_STRAIN_SUBDIR
    out_path = out_dir / f"roster_depth_strain_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[roster-depth-strain] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[roster-depth-strain] Computing roster strain as of {as_of}…")
    with DataStore(args.data_dir) as store:
        injury = _latest_injury_status(args, as_of)
        roster = _build_roster_df(store, injury)

    n_total = len(roster)
    n_ir    = int(roster["is_on_ir"].sum() or 0)
    print(f"  {n_total:,} skater rows  ({n_ir} on IR)")

    result = RosterDepthStrain().compute(roster, as_of_date=as_of)
    n_rows = len(result)
    print(f"  {n_rows:,} per-skater rows produced.")

    if n_rows > 0:
        team_top = (
            result.filter(~pl.col("is_on_ir"))
                  .group_by("team")
                  .agg(
                      pl.col("team_strain_score").first(),
                      pl.col("ir_skater_count").first(),
                      pl.col("extra_per_healthy_secs").first(),
                  )
                  .sort("team_strain_score", descending=True)
                  .head(10)
        )
        if len(team_top) > 0:
            print("\n  Top 10 most-strained teams:")
            print(f"  {'Team':<5}  {'IR':>3}  {'Extra/sec':>9}  {'Strain':>6}")
            print(f"  {'─'*5}  {'─'*3}  {'─'*9}  {'─'*6}")
            for r in team_top.to_dicts():
                print(
                    f"  {r['team']:<5}  {r['ir_skater_count']:>3d}  "
                    f"{r['extra_per_healthy_secs']:>9.1f}  "
                    f"{r['team_strain_score']:>6.3f}"
                )

    path = write_roster_depth_strain(result, out_dir, as_of)
    print(f"\n[roster-depth-strain] Written: {path}")


if __name__ == "__main__":
    main()
