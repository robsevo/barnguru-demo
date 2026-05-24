#!/usr/bin/env python3
"""Train PP Coordinator Model — Feature 4.9.

Requires::
    ~/.gretzky/data/raw/pbp_{season}.parquet
    ~/.gretzky/data/shots/shots_{season}.parquet
    ~/.gretzky/data/st_deployment/st_deployment_{season}.parquet

Usage::

    uv run python scripts/train_pp_coordinator.py
    uv run python scripts/train_pp_coordinator.py --season 2025 --force

Outputs::
    ~/.gretzky/data/pp_coordinator/pp_coordinator_{season}.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

_DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _build_position_map(data_dir: Path, season: int) -> dict[int, str]:
    """player_id → position from the MoneyPuck shots parquet."""
    shots_path = data_dir / "shots" / f"shots_{season}.parquet"
    if not shots_path.exists():
        return {}
    df = (
        pl.read_parquet(shots_path, columns=["shooter_id", "player_position"])
        .filter(pl.col("player_position").is_not_null())
        .unique(subset=["shooter_id"])
    )
    return {int(r["shooter_id"]): str(r["player_position"]) for r in df.iter_rows(named=True)}


def _build_name_map(data_dir: Path) -> dict[int, str]:
    """player_id → name, harvested from the broadest parquet sources.

    Mirrors ``_build_name_lookup`` in ``dashboard/api/main.py`` so the
    script-time map matches what the API reports back to the frontend.
    """
    lookup: dict[int, str] = {}

    def _ingest(glob_iter, only_if_missing: bool = False) -> None:
        for p in sorted(glob_iter):
            try:
                df = pl.read_parquet(p, columns=["player_id", "player_name"])
            except Exception:
                continue
            for r in df.iter_rows(named=True):
                pid  = r.get("player_id")
                name = r.get("player_name") or ""
                if not pid or not name or name.startswith("player_"):
                    continue
                key = int(pid)
                if only_if_missing and key in lookup:
                    continue
                lookup[key] = name

    _ingest((data_dir / "bayes_ratings").glob("*.parquet"))
    _ingest((data_dir / "skating_baseline").glob("*.parquet"), only_if_missing=True)
    _ingest((data_dir / "war").glob("*.parquet"), only_if_missing=True)
    _ingest((data_dir / "special_teams").glob("*.parquet"), only_if_missing=True)
    _ingest((data_dir / "rapm").glob("rapm_*.parquet"), only_if_missing=True)

    # Optional JSON cache last (least reliable; usually only ~1 entry)
    cache = data_dir / "player_name_cache.json"
    if cache.exists():
        import json
        try:
            blob = json.loads(cache.read_text())
            for k, v in blob.items():
                if not str(k).lstrip("-").isdigit():
                    continue
                key = int(k)
                if key not in lookup and v and not v.startswith("player_"):
                    lookup[key] = str(v)
        except json.JSONDecodeError:
            pass
    return lookup


def _print_top(df: pl.DataFrame, n: int = 10) -> None:
    print("\n  ── Top PP systems by xG/60 ──")
    for r in df.head(n).iter_rows(named=True):
        print(
            f"    {r['team']:<4}  "
            f"xG/60 {r['pp_xg_per_60']:.2f}  "
            f"Sh/60 {r['pp_shots_per_60']:.1f}  "
            f"G/60 {r['pp_goals_per_60']:.2f}  "
            f"xG/sh {r['pp_xg_per_shot']:.3f}  "
            f"carry% {r['pp_carry_pct']:.2f}  "
            f"QB {r['pp1_qb_name'] or '—':<22}  "
            f"share {r['pp1_qb_share']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PP Coordinator Model (Feature 4.9)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--season", type=int, default=_current_nhl_season(), metavar="YEAR")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from models.pp_coordinator import compute_pp_coordinator, write_pp_coordinator
    from models.rapm_model    import _NHL_TEAM_IDS

    output_dir = args.data_dir / "pp_coordinator"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"pp_coordinator_{args.season}.parquet"
    if out_path.exists() and not args.force:
        print(f"  Season {args.season}: already exists, skipping (use --force)")
        return

    pbp_path   = args.data_dir / "raw" / f"pbp_{args.season}.parquet"
    shots_path = args.data_dir / "shots" / f"shots_{args.season}.parquet"
    st_path    = args.data_dir / "st_deployment" / f"st_deployment_{args.season}.parquet"

    for p in (pbp_path, shots_path, st_path):
        if not p.exists():
            print(f"[pp-coordinator] required file missing: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"  Loading PBP, shots, st_deployment for {args.season}…")
    pbp_df   = pl.read_parquet(pbp_path)
    shots_df = pl.read_parquet(shots_path)
    st_df    = pl.read_parquet(st_path)

    position_map = _build_position_map(args.data_dir, args.season)
    name_lookup  = _build_name_map(args.data_dir)
    print(f"  Position map: {len(position_map)} players · name lookup: {len(name_lookup)}")

    print(f"  Computing PP coordinator signature for {args.season}…")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = compute_pp_coordinator(
            pbp_df       = pbp_df,
            shots_df     = shots_df,
            st_df        = st_df,
            position_map = position_map,
            name_lookup  = name_lookup,
            season       = args.season,
            team_lookup  = _NHL_TEAM_IDS,
        )
    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    path = write_pp_coordinator(df, output_dir, args.season)
    print(f"  Saved {path}  ({len(df)} rows)")
    _print_top(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
