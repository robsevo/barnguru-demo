#!/usr/bin/env python3
"""Compute Venue Atmosphere / Scare Factor — Feature 4.19."""
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
    p = argparse.ArgumentParser(description="Venue Atmosphere (4.19)")
    p.add_argument("--season", type=int, default=_s()); p.add_argument("--force", action="store_true")
    p.add_argument("--data-dir", type=Path, default=_DD)
    a = p.parse_args()
    from models.venue_atmosphere import compute_venue_atmosphere, write_venue_atmosphere
    from models.rapm_model import _NHL_TEAM_IDS
    od = a.data_dir / "venue_atmosphere"
    op = od / f"venue_atmosphere_{a.season}.parquet"
    if op.exists() and not a.force: print(f"  exists, skipping"); return
    pbp = pl.read_parquet(a.data_dir / "raw" / f"pbp_{a.season}.parquet")
    sp = a.data_dir / "shots" / f"shots_{a.season}.parquet"
    shots = pl.read_parquet(sp) if sp.exists() else pl.DataFrame()
    print(f"  Computing venue atmosphere for {a.season}…")
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        df = compute_venue_atmosphere(pbp, shots, _NHL_TEAM_IDS, a.season)
    for w in c: print(f"  [WARN] {w.message}", file=sys.stderr)
    path = write_venue_atmosphere(df, od, a.season)
    print(f"  Saved {path}  ({len(df)} rows)")
    print(f"\n  ── Scariest buildings ──")
    for r in df.head(10).iter_rows(named=True):
        print(f"    {r['team']:<4}  scare {r['scare_factor']:+.3f}  rank {r['scare_rank']:.2f}  "
              f"vSV%Δ {r['visiting_sv_delta']:+.4f}  vFOW%Δ {r['visiting_fow_delta']:+.3f}  "
              f"ppΔ {r['ref_pp_delta']:+.3f}")
    print("\nDone.")

if __name__ == "__main__": main()
