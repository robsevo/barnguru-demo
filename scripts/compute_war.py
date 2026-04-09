"""compute_war — Feature 2.25 script.

Loads RAPM parquets (and optionally xG finishing + cap data) and computes
per-player WAR and contract efficiency for the target season.

Usage::

    uv run python scripts/compute_war.py
    uv run python scripts/compute_war.py --seasons 2023 2024 2025
    uv run python scripts/compute_war.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from models.war_model import WARModel, write_war


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR   = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
DEFAULT_SEASONS   = [2023, 2024, 2025]
RAPM_SUBDIR       = "rapm"
FINISHING_SUBDIR  = "xg_finishing"
CAP_SUBDIR        = "cap"
WAR_SUBDIR        = "war"


def _latest_parquet(directory: Path, glob_pattern: str) -> Path | None:
    files = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_rapm(rapm_dir: Path, seasons: list[int]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for season in seasons:
        f = _latest_parquet(rapm_dir, f"rapm_{season}*.parquet")
        if f:
            frames.append(pl.read_parquet(f))
            print(f"  Loaded RAPM: {f.name}")
        else:
            print(f"  No RAPM file found for season {season}")
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def _load_finishing(finishing_dir: Path, seasons: list[int]) -> pl.DataFrame | None:
    frames: list[pl.DataFrame] = []
    for season in seasons:
        f = _latest_parquet(finishing_dir, f"xg_finishing_{season}*.parquet")
        if not f:
            f = _latest_parquet(finishing_dir, f"finishing_{season}*.parquet")
        if f:
            frames.append(pl.read_parquet(f))
            print(f"  Loaded finishing: {f.name}")
    if not frames:
        return None
    return pl.concat(frames, how="diagonal")


def _load_cap(cap_dir: Path, seasons: list[int]) -> pl.DataFrame | None:
    frames: list[pl.DataFrame] = []
    for season in seasons:
        f = _latest_parquet(cap_dir, f"cap_{season}*.parquet")
        if f:
            frames.append(pl.read_parquet(f))
            print(f"  Loaded cap: {f.name}")
    if not frames:
        return None
    return pl.concat(frames, how="diagonal")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute WAR + Contract Efficiency (Feature 2.25)."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        metavar="YEAR",
        help=f"Seasons to compute (default: {DEFAULT_SEASONS}).",
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
    data_dir: Path = args.data_dir
    target_season = max(seasons)

    # ── Check output ──────────────────────────────────────────────────────
    war_dir  = data_dir / WAR_SUBDIR
    out_path = war_dir / f"war_{target_season}.parquet"
    if out_path.exists() and not args.force:
        print(f"[war] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    # ── Load RAPM (required) ──────────────────────────────────────────────
    rapm_dir = data_dir / RAPM_SUBDIR
    if not rapm_dir.exists():
        print(f"[war] RAPM directory not found: {rapm_dir}")
        print("  Run 'uv run python scripts/gretzky.py train-rapm' first.")
        sys.exit(1)

    print(f"[war] Loading RAPM data for seasons: {seasons}")
    rapm_df = _load_rapm(rapm_dir, seasons)

    if len(rapm_df) == 0:
        print("[war] No RAPM data found.")
        print("  Run 'uv run python scripts/gretzky.py train-rapm' first.")
        sys.exit(1)
    print(f"  {len(rapm_df):,} player-season rows loaded")

    # ── Load finishing (optional) ─────────────────────────────────────────
    finishing_dir = data_dir / FINISHING_SUBDIR
    finishing_df  = None
    if finishing_dir.exists():
        print(f"[war] Loading xG finishing data…")
        finishing_df = _load_finishing(finishing_dir, seasons)
        if finishing_df is None:
            print("  No finishing data found — finishing GAR will be 0.")
    else:
        print("[war] No finishing directory — finishing GAR will be 0.")

    # ── Load cap data (optional) ──────────────────────────────────────────
    cap_dir = data_dir / CAP_SUBDIR
    cap_df  = None
    if cap_dir.exists():
        print(f"[war] Loading cap data…")
        cap_df = _load_cap(cap_dir, seasons)
        if cap_df is None:
            print("  No cap data found — contract efficiency will be null.")
    else:
        print("[war] No cap directory — contract efficiency will be null.")

    # ── Compute WAR ───────────────────────────────────────────────────────
    print(f"\n[war] Computing WAR for season {target_season}…")
    model = WARModel()
    war_df = model.fit(rapm_df, finishing_df=finishing_df, cap_df=cap_df)

    n_players = len(war_df)
    print(f"  WAR computed for {n_players:,} player(s).")

    # ── Summary ───────────────────────────────────────────────────────────
    if n_players > 0:
        avg_war = war_df["war"].mean()
        print(f"\n  League mean WAR: {avg_war:.2f}")

        top_war = war_df.sort("war", descending=True).head(10)
        print(f"\n  Top-10 players by WAR:")
        has_cap = war_df["contract_efficiency"].is_not_null().any()
        if has_cap:
            print(f"  {'Player':<22}  {'Pos':<3}  {'TOI_EV':>6}  {'GAR':>6}  {'WAR':>5}  {'Cap ($M)':>8}  {'Eff':>5}")
            print(f"  {'─'*22}  {'─'*3}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*5}")
        else:
            print(f"  {'Player':<22}  {'Pos':<3}  {'TOI_EV':>6}  {'GAR':>6}  {'WAR':>5}")
            print(f"  {'─'*22}  {'─'*3}  {'─'*6}  {'─'*6}  {'─'*5}")
        for row in top_war.to_dicts():
            cap  = row.get("cap_hit")
            eff  = row.get("contract_efficiency")
            line = (
                f"  {row['player_name']:<22}  {row['position']:<3}  "
                f"{row['toi_ev']:>6.0f}  {row['gar_total']:>6.1f}  {row['war']:>5.2f}"
            )
            if has_cap:
                cap_str = f"{cap:>8.2f}" if cap is not None else f"{'—':>8}"
                eff_str = f"{eff:>5.2f}" if eff is not None else f"{'—':>5}"
                line += f"  {cap_str}  {eff_str}"
            print(line)

        # Value contracts (efficiency > 1.5, WAR > 1.0)
        value_contracts = war_df.filter(
            (pl.col("contract_efficiency") > 1.5) & (pl.col("war") > 1.0)
        )
        if len(value_contracts) > 0:
            print(f"\n  Value contracts (efficiency > 1.5×, WAR > 1.0): {len(value_contracts)}")

    # ── Save ──────────────────────────────────────────────────────────────
    path = write_war(war_df, war_dir, target_season)
    print(f"\n[war] WAR written: {path}")

    model_path = war_dir / "war_model.pkl"
    model.save(model_path)
    print(f"[war] Model saved: {model_path}")


if __name__ == "__main__":
    main()
