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
PBP_SUBDIR        = "raw"  # pbp_{season}.parquet lives here


def _latest_parquet(directory: Path, glob_pattern: str) -> Path | None:
    files = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _pick_for_context(directory: Path, season: int, base: str, season_type: str) -> Path | None:
    """Pick exact playoff/regular parquet for a season.

    Playoff: prefer ``{base}{season}_playoffs.parquet``, fall back to season.
    Regular: prefer ``{base}{season}.parquet`` (without _playoffs suffix).
    """
    if season_type == "playoffs":
        po = directory / f"{base}{season}_playoffs.parquet"
        if po.exists():
            return po
        # Don't silently load season parquet here — caller handles the warning.
    rg = directory / f"{base}{season}.parquet"
    return rg if rg.exists() else None


def _load_rapm(
    rapm_dir: Path, seasons: list[int], season_type: str = "regular"
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for season in seasons:
        f = _pick_for_context(rapm_dir, season, "rapm_", season_type)
        if f:
            frames.append(pl.read_parquet(f))
            print(f"  Loaded RAPM: {f.name}")
        else:
            print(f"  No RAPM file found for season {season} ({season_type})")
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def _load_finishing(
    finishing_dir: Path, seasons: list[int], season_type: str = "regular"
) -> pl.DataFrame | None:
    frames: list[pl.DataFrame] = []
    for season in seasons:
        f = _pick_for_context(finishing_dir, season, "xg_finishing_", season_type)
        if not f:
            f = _pick_for_context(finishing_dir, season, "finishing_", season_type)
        if f:
            frames.append(pl.read_parquet(f))
            print(f"  Loaded finishing: {f.name}")
    if not frames:
        return None
    return pl.concat(frames, how="diagonal")


def _load_penalties_from_pbp(
    pbp_dir: Path, seasons: list[int], season_type: str = "regular"
) -> pl.DataFrame | None:
    """Aggregate per-player penalty minutes taken and drawn from our PBP parquets.

    Returns a DataFrame with: player_id, season, pim_taken, pim_drawn.
    Returns None if no PBP data is found.
    """
    frames: list[pl.DataFrame] = []
    for season in seasons:
        path = pbp_dir / f"pbp_{season}.parquet"
        if not path.exists():
            continue
        try:
            schema = pl.read_parquet_schema(path)
        except Exception:
            continue

        # Need at minimum committed_by_id or drawn_by_id + duration_minutes
        has_committed = "committed_by_id" in schema
        has_drawn     = "drawn_by_id"     in schema
        has_duration  = "duration_minutes" in schema
        has_event     = any(c in schema for c in ("event_type", "event"))

        if not has_event or not (has_committed or has_drawn):
            continue

        # Lazy + filter-pushdown: penalty rows are <1% of PBP, so collecting
        # only those rows cuts peak from ~250 MB/season to a few MB per season.
        event_col = "event_type" if "event_type" in schema else ("event" if "event" in schema else None)
        if event_col is None:
            continue

        want = [c for c in (
            event_col,
            "committed_by_id", "drawn_by_id",
            "duration_minutes",
            "game_id",
        ) if c in schema]

        try:
            lf = pl.scan_parquet(path).select(want)
        except Exception:
            continue

        # Filter by game type (game_id encodes it as the TT digits) — pushed into scan.
        if "game_id" in want:
            gt = (pl.col("game_id") % 1_000_000) // 10_000
            if season_type == "playoffs":
                lf = lf.filter(gt == 3)
            elif season_type == "regular":
                lf = lf.filter(gt == 2)

        # Penalty-only filter — pushed into scan so non-penalty rows never materialise.
        lf = lf.filter(pl.col(event_col).str.to_lowercase() == "penalty")

        try:
            pen_df = lf.collect(streaming=True)
        except Exception:
            continue

        if pen_df.is_empty():
            continue

        # Duration defaults to 2 if missing
        if "duration_minutes" not in pen_df.columns:
            pen_df = pen_df.with_columns(pl.lit(2.0).alias("duration_minutes"))
        else:
            pen_df = pen_df.with_columns(
                pl.col("duration_minutes").cast(pl.Float64).fill_null(2.0)
            )

        season_frames: list[pl.DataFrame] = []

        if has_committed and "committed_by_id" in pen_df.columns:
            taken = (
                pen_df
                .filter(pl.col("committed_by_id").is_not_null())
                .group_by("committed_by_id")
                .agg(pl.col("duration_minutes").sum().alias("pim_taken"))
                .rename({"committed_by_id": "player_id"})
                .with_columns(pl.col("player_id").cast(pl.Int64))
            )
            season_frames.append(taken)

        if has_drawn and "drawn_by_id" in pen_df.columns:
            drawn = (
                pen_df
                .filter(pl.col("drawn_by_id").is_not_null())
                .group_by("drawn_by_id")
                .agg(pl.col("duration_minutes").sum().alias("pim_drawn"))
                .rename({"drawn_by_id": "player_id"})
                .with_columns(pl.col("player_id").cast(pl.Int64))
            )
            season_frames.append(drawn)

        if not season_frames:
            continue

        combined = season_frames[0]
        for f in season_frames[1:]:
            combined = combined.join(f, on="player_id", how="outer_coalesce")

        for col in ("pim_taken", "pim_drawn"):
            if col not in combined.columns:
                combined = combined.with_columns(pl.lit(0.0).alias(col))
            else:
                combined = combined.with_columns(pl.col(col).fill_null(0.0))

        combined = combined.with_columns(pl.lit(season).cast(pl.Int64).alias("season"))
        frames.append(combined)
        print(f"  Loaded penalties from PBP: {path.name} — {len(combined)} players")

    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed")


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
    parser.add_argument(
        "--season-type",
        choices=["regular", "playoffs"],
        default="regular",
        help="'regular' writes war_{year}.parquet (default). "
             "'playoffs' reads playoff RAPM/xG/penalties and writes "
             "war_{year}_playoffs.parquet.",
    )
    args = parser.parse_args()

    seasons: list[int] = sorted(args.seasons)
    data_dir: Path = args.data_dir
    target_season = max(seasons)
    season_type: str = args.season_type
    out_suffix       = "_playoffs" if season_type == "playoffs" else ""

    # ── Check output ──────────────────────────────────────────────────────
    war_dir  = data_dir / WAR_SUBDIR
    out_path = war_dir / f"war_{target_season}{out_suffix}.parquet"
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

    print(f"[war] Loading RAPM data for seasons: {seasons} ({season_type})")
    rapm_df = _load_rapm(rapm_dir, seasons, season_type=season_type)

    if len(rapm_df) == 0:
        print("[war] No RAPM data found.")
        print("  Run 'uv run python scripts/gretzky.py train-rapm' first.")
        sys.exit(1)
    print(f"  {len(rapm_df):,} player-season rows loaded")

    # ── Strip goalies from skater WAR ─────────────────────────────────────
    # Goalies appear in the RAPM parquet because they share ice time, but their
    # "EV TOI" is 40-60 min/game (vs ~15-20 for skaters), which wildly inflates
    # their WAR.  Identify goalies from the goalie_ratings parquet and exclude.
    goalie_dir = data_dir / "goalie_ratings"
    goalie_ids: set[int] = set()
    if goalie_dir.exists():
        for gp in sorted(goalie_dir.glob("goalie_ratings_*.parquet")):
            try:
                gdf = pl.read_parquet(gp, columns=["player_id"])
                goalie_ids.update(int(x) for x in gdf["player_id"].drop_nulls().to_list())
            except Exception:
                pass
    if goalie_ids:
        before = len(rapm_df)
        rapm_df = rapm_df.filter(~pl.col("player_id").is_in(list(goalie_ids)))
        print(f"  Excluded {before - len(rapm_df)} goalie-season rows ({len(goalie_ids)} unique goalies)")
    else:
        print("  [warn] No goalie_ratings data found; goalies may inflate WAR leaderboard.")

    # Drop prior-season RAPM rows — model.fit() only operates on target_season,
    # and _compute_replacement_level computes from the frame passed to fit().
    # Frees ~100-200 MB on multi-season runs. No-op on single-season nightly runs.
    if "season" in rapm_df.columns:
        rapm_df = rapm_df.filter(pl.col("season") == target_season)

    # ── Load finishing (optional) ─────────────────────────────────────────
    finishing_dir = data_dir / FINISHING_SUBDIR
    finishing_df  = None
    if finishing_dir.exists():
        print(f"[war] Loading xG finishing data…")
        finishing_df = _load_finishing(finishing_dir, seasons, season_type=season_type)
        if finishing_df is None:
            print("  No finishing data found — finishing GAR will be 0.")
    else:
        print("[war] No finishing directory — finishing GAR will be 0.")

    # ── Load penalty data from our own PBP parquets (optional) ───────────
    pbp_dir = data_dir / PBP_SUBDIR
    penalty_df = None
    if pbp_dir.exists():
        print(f"[war] Loading penalty data from PBP parquets…")
        penalty_df = _load_penalties_from_pbp(pbp_dir, seasons, season_type=season_type)
        if penalty_df is None:
            print("  No PBP penalty data found — penalty GAR will be 0.")
    else:
        print("[war] No PBP directory — penalty GAR will be 0.")

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

    # ── Compute WAR — target season only ─────────────────────────────────
    print(f"\n[war] Computing WAR for season {target_season}…")
    model = WARModel()
    target_rapm_df = (
        rapm_df.filter(pl.col("season") == target_season)
        if "season" in rapm_df.columns
        else rapm_df
    )
    # Playoffs have ~10% of the regular-season TOI; drop the floor accordingly
    # so the leaderboard isn't empty.
    war_min_toi = 20.0 if season_type == "playoffs" else None
    war_df = model.fit(
        target_rapm_df,
        finishing_df=finishing_df,
        penalty_df=penalty_df,
        cap_df=cap_df,
        min_toi_ev_output=war_min_toi,
    )

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
    path = write_war(war_df, war_dir, target_season, season_type=season_type)
    print(f"\n[war] WAR written: {path}")

    model_path = war_dir / "war_model.pkl"
    model.save(model_path)
    print(f"[war] Model saved: {model_path}")


if __name__ == "__main__":
    main()
