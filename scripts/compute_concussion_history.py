"""compute_concussion_history — Feature 3.14 driver script.

Walks every cached ESPN injury parquet (``injuries_*.parquet`` written by
``data.injury_sync``), keeps only the rows whose injury text matches a
concussion/head keyword, joins to NHL ``player_id`` via the roster cache,
and writes a per-player career summary: episode count, last episode
date, days-since, and the fatigue sensitivity multiplier.

Usage::

    uv run python scripts/gretzky.py concussion-history
    uv run python scripts/gretzky.py concussion-history -- --date 2026-05-17
    uv run python scripts/gretzky.py concussion-history -- --force
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
from models.concussion_history import (
    ConcussionHistoryFlag,
    write_concussion_history,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
CONCUSSION_HISTORY_SUBDIR = "concussion_history"


def _injury_history(store: DataStore) -> pl.DataFrame:
    """Daily injury rows, joined to NHL player_id, with text fields kept."""
    raw = store.injuries(date=None)
    empty = pl.DataFrame(schema={
        "player_id":     pl.Int64,
        "observed_date": pl.Utf8,
        "injury_type":   pl.Utf8,
        "injury_detail": pl.Utf8,
        "status_raw":    pl.Utf8,
    })
    if len(raw) == 0:
        return empty

    roster = store.roster()
    if len(roster) == 0 or "player_id" not in roster.columns:
        return empty
    name_col = "full_name" if "full_name" in roster.columns else "player_name"
    if name_col not in roster.columns:
        return empty

    bridge = (
        roster.select(["player_id", "team_abbrev", name_col])
        .rename({"team_abbrev": "team_code", name_col: "player_name"})
        .with_columns(
            pl.col("team_code").cast(pl.Utf8).str.to_uppercase(),
            pl.col("player_name").cast(pl.Utf8).str.to_lowercase(),
        )
        .unique(subset=["team_code", "player_name"])
    )

    cols = ["player_name", "team_code", "fetched_at"]
    for c in ("injury_type", "injury_detail", "status_raw"):
        if c in raw.columns:
            cols.append(c)

    joined = (
        raw.select(cols)
        .with_columns(
            pl.col("team_code").cast(pl.Utf8).str.to_uppercase(),
            pl.col("player_name").cast(pl.Utf8).str.to_lowercase(),
            pl.col("fetched_at").cast(pl.Utf8).str.slice(0, 10).alias("observed_date"),
        )
        .drop("fetched_at")
        .join(bridge, on=["team_code", "player_name"], how="inner")
    )

    # Ensure all text columns exist downstream.
    for c in ("injury_type", "injury_detail", "status_raw"):
        if c not in joined.columns:
            joined = joined.with_columns(pl.lit(None).cast(pl.Utf8).alias(c))

    return joined.select(
        ["player_id", "observed_date", "injury_type", "injury_detail", "status_raw"]
    ).unique()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-player prior-concussion summary (Feature 3.14)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None,
                        help="as-of date. Default: today (UTC).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / CONCUSSION_HISTORY_SUBDIR
    out_path = out_dir / f"concussion_history_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[concussion-history] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[concussion-history] Computing concussion history as of {as_of}…")
    with DataStore(args.data_dir) as store:
        history = _injury_history(store)
    print(f"  {len(history):,} injury observation rows")

    result = ConcussionHistoryFlag().compute(history, as_of_date=as_of)
    n_rows = len(result)
    print(f"  {n_rows:,} player rows produced.")

    if n_rows > 0:
        top = result.sort("prior_concussion_count", descending=True).head(10)
        print("\n  Top 10 prior-concussion counts:")
        print(f"  {'Player':<10}  {'Count':>5}  {'Mult':>5}  {'Last':<10}")
        print(f"  {'─'*10}  {'─'*5}  {'─'*5}  {'─'*10}")
        for r in top.to_dicts():
            print(
                f"  {r['player_id']:<10}  {r['prior_concussion_count']:>5d}  "
                f"{r['fatigue_sensitivity_multiplier']:>5.2f}  "
                f"{r['last_concussion_date']:<10}"
            )

    path = write_concussion_history(result, out_dir, as_of)
    print(f"\n[concussion-history] Written: {path}")


if __name__ == "__main__":
    main()
