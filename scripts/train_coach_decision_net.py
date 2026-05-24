#!/usr/bin/env python3
"""Train Per-Coach Decision Network — Feature 4.17.

V1: consolidates Phase 4 outputs (4.1–4.6) into unified per-coach
decision probability profiles.

Requires: Phase 4 parquets (timeout_usage, goalie_pull, line_deployment,
           st_deployment, penalty_tendency, line_matching) + coaches.json

Outputs:  ~/.gretzky/data/coach_decision_net/coach_decision_net_{season}.parquet
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import polars as pl

_DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
_COACHES_JSON = _REPO / "data" / "coaches.json"

def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1

def _load_parquet(data_dir: Path, subdir: str, season: int) -> pl.DataFrame:
    p = data_dir / subdir / f"{subdir}_{season}.parquet"
    return pl.read_parquet(p) if p.exists() else pl.DataFrame()

def _load_parquet_glob(data_dir: Path, subdir: str, glob_pat: str) -> pl.DataFrame:
    d = data_dir / subdir
    if not d.exists():
        return pl.DataFrame()
    files = sorted(d.glob(glob_pat))
    return pl.read_parquet(files[-1]) if files else pl.DataFrame()

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Coach Decision Net (Feature 4.17)")
    parser.add_argument("--season", type=int, default=_current_nhl_season())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from models.coach_decision_net import compute_coach_decision_net, write_coach_decision_net

    output_dir = args.data_dir / "coach_decision_net"
    out_path = output_dir / f"coach_decision_net_{args.season}.parquet"
    if out_path.exists() and not args.force:
        print(f"  Season {args.season}: exists, skipping (use --force)")
        return

    coaches = json.loads(_COACHES_JSON.read_text()).get("coaches", []) if _COACHES_JSON.exists() else []
    print(f"  Loaded {len(coaches)} coaches")

    timeout_df  = _load_parquet(args.data_dir, "timeout_usage",    args.season)
    pull_df     = _load_parquet_glob(args.data_dir, "goalie_pull", f"goalie_pull_{args.season}*.parquet")
    deploy_df   = _load_parquet(args.data_dir, "line_deployment",  args.season)
    st_df       = _load_parquet(args.data_dir, "st_deployment",    args.season)
    penalty_df  = _load_parquet(args.data_dir, "penalty_tendency", args.season)
    match_df    = _load_parquet(args.data_dir, "line_matching",    args.season)

    print(f"  Phase 4 inputs: timeout={len(timeout_df)} pull={len(pull_df)} deploy={len(deploy_df)} st={len(st_df)} penalty={len(penalty_df)} match={len(match_df)}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = compute_coach_decision_net(
            coaches=coaches, timeout_df=timeout_df, goalie_pull_df=pull_df,
            line_deploy_df=deploy_df, st_deploy_df=st_df,
            penalty_df=penalty_df, line_match_df=match_df, season=args.season,
        )
    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    path = write_coach_decision_net(df, output_dir, args.season)
    print(f"  Saved {path}  ({len(df)} rows)")
    print(f"\n  ── Top 10 most aggressive coaches ──")
    for r in df.head(10).iter_rows(named=True):
        print(f"    {r['team']:<4} {r['coach_name']:<24}  agg {r['overall_aggression']:.2f}  "
              f"TO {r['timeout_aggression']:.2f}  pull {r['pull_aggression']:.2f}  "
              f"shelter {r['line_shelter_score']:.2f}  match {r['matching_intensity']:.2f}")
    print("\nDone.")

if __name__ == "__main__":
    main()
