#!/usr/bin/env python3
"""Train Roster Fit Score — Feature 4.12.

Requires::
    ~/.gretzky/data/coaching_style/coaching_style_{season}.parquet
    ~/.gretzky/data/archetypes/archetype_assignments_{season}.parquet
    ~/.gretzky/data/rapm/rapm_{season}.parquet

Usage::

    uv run python scripts/train_roster_fit.py
    uv run python scripts/train_roster_fit.py --season 2025 --force

Outputs::
    ~/.gretzky/data/roster_fit/roster_fit_{season}.parquet
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


def _print_top(df: pl.DataFrame, n: int = 10) -> None:
    print("\n  ── Top fits ──")
    for r in df.head(n).iter_rows(named=True):
        archs = r["archetypes"] or []
        shares = r["archetype_shares"] or []
        breakdown = ", ".join(
            f"{a} {s*100:.0f}%" for a, s in list(zip(archs, shares))[:3]
        )
        print(
            f"    {r['team']:<4}  fit {r['fit_score']:.3f}  "
            f"top {r['archetype_top']:<18}  "
            f"weak dim: {r['mismatch_dim'] or '—':<22} ({r['mismatch_support']:.2f})\n"
            f"       {breakdown}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Roster Fit Score (Feature 4.12)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--season", type=int, default=_current_nhl_season(), metavar="YEAR")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from models.roster_fit import compute_roster_fit, write_roster_fit
    from models.rapm_model  import _NHL_TEAM_IDS

    output_dir = args.data_dir / "roster_fit"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"roster_fit_{args.season}.parquet"
    if out_path.exists() and not args.force:
        print(f"  Season {args.season}: already exists, skipping (use --force)")
        return

    style_path   = args.data_dir / "coaching_style"      / f"coaching_style_{args.season}.parquet"
    arch_path    = args.data_dir / "archetypes"          / f"archetype_assignments_{args.season}.parquet"
    rapm_path    = args.data_dir / "rapm"                / f"rapm_{args.season}.parquet"

    for p in (style_path, arch_path, rapm_path):
        if not p.exists():
            print(f"[roster-fit] required file missing: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"  Loading style + archetype + rapm for {args.season}…")
    style_df = pl.read_parquet(style_path)
    arch_df  = pl.read_parquet(arch_path)
    rapm_df  = pl.read_parquet(rapm_path)

    print(f"  Computing roster fit for {args.season}…")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = compute_roster_fit(
            style_df     = style_df,
            archetype_df = arch_df,
            rapm_df      = rapm_df,
            team_lookup  = _NHL_TEAM_IDS,
            season       = args.season,
        )
    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    path = write_roster_fit(df, output_dir, args.season)
    print(f"  Saved {path}  ({len(df)} rows)")
    _print_top(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
