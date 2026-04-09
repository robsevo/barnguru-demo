"""compute_roster_disruption — Feature 2.19 training script.

Loads the full transaction log, computes Roster Disruption Index for every
team as of a given date, and writes the result to parquet.

Usage::

    uv run python scripts/compute_roster_disruption.py
    uv run python scripts/compute_roster_disruption.py --date 2025-03-15
    uv run python scripts/compute_roster_disruption.py --teams EDM TOR BOS
    uv run python scripts/compute_roster_disruption.py --force
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
import os

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from models.roster_disruption import (
    RosterDisruptionModel,
    write_roster_disruption,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
TX_SUBDIR        = "transactions"
RDI_SUBDIR       = "roster_disruption"
MODEL_SUBDIR     = "roster_disruption"


def _latest_transactions_parquet(tx_dir: Path) -> Path | None:
    """Return the most recently modified transactions parquet."""
    files = sorted(tx_dir.glob("transactions_*.parquet"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Roster Disruption Index for all teams (Feature 2.19)."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="As-of date in YYYY-MM-DD format (default: today UTC).",
    )
    parser.add_argument(
        "--teams",
        nargs="+",
        default=None,
        metavar="TEAM",
        help="Restrict to specific team abbreviations (default: all teams in transaction log).",
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
        help="Overwrite existing output parquet for today's date.",
    )
    args = parser.parse_args()

    as_of_date = args.date or datetime.now(timezone.utc).date().isoformat()
    data_dir: Path = args.data_dir

    # ── Load transactions ─────────────────────────────────────────────────
    tx_dir = data_dir / TX_SUBDIR
    tx_path = _latest_transactions_parquet(tx_dir)

    if tx_path is None:
        print(f"[roster-disruption] No transactions parquet found in {tx_dir}.")
        print("  Run 'uv run python scripts/gretzky.py sync' first.")
        sys.exit(1)

    print(f"[roster-disruption] Loading transactions from {tx_path}")
    transactions_df = pl.read_parquet(tx_path)
    print(f"  {len(transactions_df):,} transaction rows")

    # ── Compute disruption ────────────────────────────────────────────────
    rdi_dir = data_dir / RDI_SUBDIR
    out_path = rdi_dir / f"roster_disruption_{as_of_date}.parquet"

    if out_path.exists() and not args.force:
        print(f"[roster-disruption] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    model = RosterDisruptionModel()
    print(f"[roster-disruption] Computing disruption index as of {as_of_date}…")
    result_df = model.compute_all(
        transactions_df,
        as_of_date=as_of_date,
        teams=args.teams,
    )

    n_teams = len(result_df)
    print(f"  Computed for {n_teams} team(s).")

    # ── Print top disrupted teams ─────────────────────────────────────────
    if n_teams > 0:
        top = (
            result_df
            .sort("disruption_index", descending=True)
            .head(10)
        )
        print("\n  Top disrupted teams:")
        print(f"  {'Team':<6}  {'Disruption':>10}  {'n_moves_30d':>11}  {'n_moves_72h':>11}  {'chaos':>6}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*11}  {'─'*11}  {'─'*6}")
        for row in top.to_dicts():
            chaos_flag = " !" if row["chaos_multiplier"] > 1.0 else ""
            print(
                f"  {row['team']:<6}  {row['disruption_index']:>10.4f}  "
                f"{row['n_moves_30d']:>11}  {row['n_moves_72h']:>11}  "
                f"{row['chaos_multiplier']:>5.1f}{chaos_flag}"
            )

    # ── Save ──────────────────────────────────────────────────────────────
    path = write_roster_disruption(result_df, rdi_dir, as_of_date)
    print(f"\n[roster-disruption] Written: {path}")

    # ── Save model config ─────────────────────────────────────────────────
    model_dir = data_dir / MODEL_SUBDIR
    model_path = model_dir / "rdi_model.pkl"
    model.save(model_path)
    print(f"[roster-disruption] Model saved: {model_path}")


if __name__ == "__main__":
    main()
