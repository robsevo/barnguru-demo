#!/usr/bin/env python3
"""Compute Playoff Elimination Fatigue — Feature 4.20."""
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
    p = argparse.ArgumentParser(description="Playoff Elimination Fatigue (4.20)")
    p.add_argument("--season", type=int, default=_s()); p.add_argument("--force", action="store_true")
    p.add_argument("--data-dir", type=Path, default=_DD)
    a = p.parse_args()
    from models.playoff_elimination import compute_playoff_elimination, write_playoff_elimination
    from models.rapm_model import _NHL_TEAM_IDS
    od = a.data_dir / "playoff_elimination"
    op = od / f"playoff_elimination_{a.season}.parquet"
    if op.exists() and not a.force: print(f"  exists, skipping"); return
    ts = pl.read_parquet(a.data_dir / "raw" / f"team_stats_{a.season}.parquet")
    print(f"  Computing playoff elimination for {a.season}…")
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        df = compute_playoff_elimination(ts, _NHL_TEAM_IDS, a.season)
    for w in c: print(f"  [WARN] {w.message}", file=sys.stderr)
    path = write_playoff_elimination(df, od, a.season)
    print(f"  Saved {path}  ({len(df)} rows)")
    dragged = df.filter(pl.col("elimination_drag") > 0)
    if not dragged.is_empty():
        print(f"\n  Teams with elimination drag:")
        for r in dragged.iter_rows(named=True):
            print(f"    {r['team']:<4}  prob {r['playoff_prob']:.3f}  drag {r['elimination_drag']:.3f}  eff ×{r['efficiency_multiplier']:.3f}  {r['games_remaining']}g left")
    else:
        print(f"  No teams with elimination drag (all either in playoff picture or season over).")
    print("\nDone.")

if __name__ == "__main__": main()
