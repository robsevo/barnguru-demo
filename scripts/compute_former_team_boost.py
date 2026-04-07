"""compute_former_team_boost — Feature 2.27 training script.

Loads ESPN transaction parquets and optional player TOI stats, then computes
per-player former-team motivational boost scores.

The signal validation gate (``is_validated``) is set when
``--validated-seasons`` supplies at least 3 confirmed seasons.
Until then the parquet is written with ``validated=False`` so downstream
consumers (the Rust engine) know not to apply the boost in production.

Usage::

    uv run python scripts/compute_former_team_boost.py
    uv run python scripts/compute_former_team_boost.py --seasons 2022 2023 2024 2025
    uv run python scripts/compute_former_team_boost.py --validated-seasons 2022 2023 2024
    uv run python scripts/compute_former_team_boost.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from models.former_team_boost import (
    BOOST_BY_DEPARTURE,
    VALIDATION_SEASONS_REQUIRED,
    FormerTeamBoostModel,
    write_former_team_boost,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR       = Path.home() / ".gretzky" / "data"
DEFAULT_SEASONS        = [2022, 2023, 2024, 2025]
TRANSACTIONS_SUBDIR    = "raw"
PLAYER_STATS_SUBDIR    = "raw"
FORMER_TEAM_SUBDIR     = "former_team_boost"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_transactions(data_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Load ESPN transaction parquets for the given seasons.

    Accepts files named:
      - ``espn_transactions_{season}.parquet``
      - ``transactions_{season}.parquet``
      - ``transactions.parquet``  (single unified file)
    """
    raw_dir = data_dir / TRANSACTIONS_SUBDIR
    frames: list[pl.DataFrame] = []

    for season in seasons:
        for pattern in (
            f"espn_transactions_{season}.parquet",
            f"transactions_{season}.parquet",
        ):
            f = raw_dir / pattern
            if f.exists():
                try:
                    frames.append(pl.read_parquet(f))
                    print(f"  Loaded: {f.name}")
                except Exception as e:
                    print(f"  Warning: could not load {f.name}: {e}")
                break

    # Fallback: single unified transactions file in raw/
    if not frames:
        fallback = raw_dir / "transactions.parquet"
        if fallback.exists():
            try:
                frames.append(pl.read_parquet(fallback))
                print(f"  Loaded: {fallback.name}")
            except Exception as e:
                print(f"  Warning: could not load {fallback.name}: {e}")

    # Fallback: per-date files written by TransactionSyncer in data/transactions/
    if not frames:
        tx_dir = data_dir / "transactions"
        if tx_dir.exists():
            per_date = sorted(tx_dir.glob("transactions_*.parquet"))
            if per_date:
                loaded = 0
                for f in per_date:
                    try:
                        frames.append(pl.read_parquet(f))
                        loaded += 1
                    except Exception:
                        pass
                if loaded:
                    print(f"  Loaded {loaded} per-date transaction files from {tx_dir}")

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def _load_player_toi(data_dir: Path, seasons: list[int]) -> pl.DataFrame | None:
    """Load player TOI stats (optional — used for chip-on-shoulder amplification).

    Looks for files named:
      - ``player_stats_{season}.parquet``
    Silently returns None if nothing is found.
    """
    raw_dir = data_dir / PLAYER_STATS_SUBDIR
    frames: list[pl.DataFrame] = []

    for season in seasons:
        f = raw_dir / f"player_stats_{season}.parquet"
        if f.exists():
            try:
                df = pl.read_parquet(f)
                frames.append(df)
                print(f"  Loaded TOI: {f.name}")
            except Exception as e:
                print(f"  Warning: could not load {f.name}: {e}")

    if not frames:
        return None

    df = pl.concat(frames, how="diagonal")

    # We need: player_id, team, toi_seconds, gp
    required = {"player_id", "team", "toi_seconds", "gp"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        print(f"  [warn] player_stats missing columns {missing}; TOI factor disabled.")
        return None

    return df


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_top_matchups(boost_df: pl.DataFrame, n: int = 10) -> None:
    """Print the top N former-team matchups by base_boost."""
    if len(boost_df) == 0:
        return

    top = boost_df.sort("base_boost", descending=True).head(n)
    print(f"\n  Top-{n} former-team matchups (highest base boost):")
    print(
        f"  {'Player':<25}  {'Former Team':<11}  "
        f"{'Departure':<16}  {'TOI/G':>6}  {'Base Boost':>10}"
    )
    print(f"  {'─'*25}  {'─'*11}  {'─'*16}  {'─'*6}  {'─'*10}")
    for row in top.to_dicts():
        print(
            f"  {row['player_name']:<25}  {row['former_team']:<11}  "
            f"{row['departure_type']:<16}  "
            f"{row['toi_per_game_min']:>6.1f}  "
            f"{row['base_boost']:>+10.3f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Former Team Motivational Boost scores (Feature 2.27)."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        metavar="YEAR",
        help=f"Seasons to pool for transaction data (default: {DEFAULT_SEASONS}).",
    )
    parser.add_argument(
        "--validated-seasons",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        dest="validated_seasons",
        help=(
            f"Seasons already back-tested against NHL PBP. "
            f"Requires ≥{VALIDATION_SEASONS_REQUIRED} seasons to set validated=True."
        ),
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
        help="Overwrite existing output parquet.",
    )
    args = parser.parse_args()

    seasons: list[int] = sorted(args.seasons)
    target_season = max(seasons)
    data_dir: Path = args.data_dir

    # ── Check output ──────────────────────────────────────────────────────────
    out_dir  = data_dir / FORMER_TEAM_SUBDIR
    out_path = out_dir / f"former_team_boost_{target_season}.parquet"
    if out_path.exists() and not args.force:
        print(f"[former-team-boost] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    # ── Load transaction data ─────────────────────────────────────────────────
    tx_dir = data_dir / TRANSACTIONS_SUBDIR
    if not tx_dir.exists():
        print(f"[former-team-boost] Transactions directory not found: {tx_dir}")
        print("  Run 'uv run python scripts/gretzky.py sync' first.")
        sys.exit(1)

    print(f"[former-team-boost] Loading transactions for seasons: {seasons}")
    tx_df = _load_transactions(data_dir, seasons)

    if len(tx_df) == 0:
        print("[former-team-boost] No transaction data found.")
        print(
            "  Ensure espn_transactions_{season}.parquet files exist under "
            f"{tx_dir}/"
        )
        print("  Run 'uv run python scripts/gretzky.py sync' first.")
        sys.exit(1)

    print(f"  {len(tx_df):,} transaction rows loaded")

    # ── Load optional player TOI ──────────────────────────────────────────────
    print(f"\n[former-team-boost] Loading player TOI stats (optional)…")
    toi_df = _load_player_toi(data_dir, seasons)
    if toi_df is None:
        print("  No TOI stats found — chip-on-shoulder TOI amplification disabled.")

    # ── Fit model ─────────────────────────────────────────────────────────────
    validated_seasons = args.validated_seasons or []
    print(f"\n[former-team-boost] Computing former-team boosts…")
    print(f"  Validated seasons: {validated_seasons or '(none — gate closed)'}")

    model = FormerTeamBoostModel()
    boost_df = model.fit(
        transactions_df=tx_df,
        player_toi_df=toi_df,
        validated_seasons=validated_seasons,
    )

    n_players  = model.player_count()
    n_entries  = len(boost_df)
    validated  = model.is_validated
    print(f"  Players with former-team entries: {n_players:,}")
    print(f"  Total (player, former_team) pairs: {n_entries:,}")
    print(f"  Signal validated: {validated} "
          f"({'≥3 seasons confirmed' if validated else 'gate CLOSED — do not apply in production'})")

    # ── Print summary ─────────────────────────────────────────────────────────
    if n_entries > 0:
        _print_top_matchups(boost_df)

        # Per departure-type breakdown
        print("\n  Breakdown by departure type:")
        for dep_type, base in sorted(BOOST_BY_DEPARTURE.items(),
                                     key=lambda kv: kv[1], reverse=True):
            n = len(boost_df.filter(pl.col("departure_type") == dep_type))
            if n > 0:
                print(f"    {dep_type:<20} base={base:.2f}  count={n}")

    if not validated:
        print(
            "\n  [GATE] Signal is NOT validated. To apply boosts in production:\n"
            f"    1. Back-test against ≥{VALIDATION_SEASONS_REQUIRED} seasons of NHL PBP\n"
            "    2. Re-run with --validated-seasons <year1> <year2> <year3>"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    path = write_former_team_boost(boost_df, out_dir, target_season)
    print(f"\n[former-team-boost] Boost table written: {path}")

    model_path = out_dir / "former_team_boost_model.pkl"
    model.save(model_path)
    print(f"[former-team-boost] Model saved: {model_path}")


if __name__ == "__main__":
    main()
