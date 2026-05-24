#!/usr/bin/env python3
"""Compute Day-of-Week Signature + Broadcast Context — Feature 4.21."""
from __future__ import annotations
import argparse, os, sys, warnings
from datetime import datetime, timezone
from pathlib import Path
_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path: sys.path.insert(0, str(_REPO))
import polars as pl
_DD = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))

def _s() -> int:
    n = datetime.now(timezone.utc); return n.year if n.month >= 10 else n.year - 1

def main() -> None:
    p = argparse.ArgumentParser(description="DOW Signature + Broadcast Context (4.21)")
    p.add_argument("--season", type=int, default=_s()); p.add_argument("--force", action="store_true")
    p.add_argument("--data-dir", type=Path, default=_DD)
    a = p.parse_args()
    from models.dow_signature import (
        fetch_game_dates, compute_dow_signature, compute_broadcast_context,
        write_dow_signature, write_broadcast_context,
    )
    from models.rapm_model import _NHL_TEAM_IDS

    od = a.data_dir / "dow_signature"
    op = od / f"dow_signature_{a.season}.parquet"
    if op.exists() and not a.force: print(f"  exists, skipping"); return

    print(f"  Fetching game dates for {a.season} (cached if available)…")
    game_dates = fetch_game_dates(a.season, a.data_dir / "game_dates")
    print(f"  {len(game_dates)} games with dates")
    if game_dates.is_empty():
        print("  [WARN] No game dates — DOW signature cannot be computed.", file=sys.stderr)
        return

    pbp = pl.read_parquet(a.data_dir / "raw" / f"pbp_{a.season}.parquet")
    print(f"  Computing DOW signature for {a.season}…")
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        df = compute_dow_signature(pbp, game_dates, a.season)
    for w in c: print(f"  [WARN] {w.message}", file=sys.stderr)

    path = write_dow_signature(df, od, a.season)
    print(f"  Saved {path}  ({len(df)} player DOW profiles)")

    # Top Saturday performers (Caufield anchor)
    if not df.is_empty():
        print(f"\n  ── Saturday leaders ──")
        sat = df.sort("sat_z", descending=True).head(10)
        for r in sat.iter_rows(named=True):
            pid = r["player_id"]
            print(f"    {pid:<10}  Sat z={r['sat_z']:+.3f}  best={r['best_day']} ({r['best_day_zscore']:+.3f})  GP {r['career_gp']}")

    # Broadcast context
    print(f"\n  Computing broadcast context…")
    bc = compute_broadcast_context(game_dates, _NHL_TEAM_IDS, pbp, a.season)
    bc_path = write_broadcast_context(bc, od, a.season)
    rivalries = bc.filter(pl.col("is_rivalry") == True)
    print(f"  Saved {bc_path}  ({len(bc)} games, {len(rivalries)} rivalry matchups)")
    print("\nDone.")

if __name__ == "__main__": main()
