#!/usr/bin/env python3
"""Train the RAPM model (Feature 2.2) with daisy-chain seasonal priors.

Requires:
  - NHL shift data:  ~/.gretzky/data/raw/shifts_{season}.parquet
                     (run: uv run python scripts/run_phase1_sync.py to populate)
  - MoneyPuck shots: ~/.gretzky/data/shots/shots_{season}.parquet
                     (already populated if phase 1 sync has run)

Usage::

    uv run python scripts/train_rapm_model.py
    uv run python scripts/train_rapm_model.py --seasons 2023 2024 2025
    uv run python scripts/train_rapm_model.py --chain-from 2015 --seasons 2025 --force

Outputs:
    ~/.gretzky/data/rapm/rapm_{season}.parquet   — per-player RAPM estimates
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
import os

from datetime import datetime, timezone

import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _default_seasons() -> list[int]:
    cur = _current_nhl_season()
    return list(range(2023, cur + 1))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _game_type_from_id(col: str = "game_id") -> pl.Expr:
    """NHL game_id encodes game type as the 5th-6th digits.

    {YYYY}0{TT}{GGGG}  where TT = 02 (regular) or 03 (playoffs).
    Returns the integer game type (2 or 3).
    """
    return (pl.col(col) % 1_000_000) // 10_000


def _load_season(
    data_dir: Path,
    season: int,
    season_type: str = "regular",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load shifts + shots for a season. Returns (shifts_df, shots_df).

    season_type filters both DataFrames to the matching game type:
      "regular"  → shifts.game_type == 2, shots.is_playoff == False (default)
      "playoffs" → shifts.game_type == 3, shots.is_playoff == True

    Raises DataMissingWarning (returns empty DFs) if files are absent.
    """
    from models.rapm_model import DataMissingWarning

    shifts_path = data_dir / "raw" / f"shifts_{season}.parquet"
    shots_path  = data_dir / "shots" / f"shots_{season}.parquet"

    if not shifts_path.exists():
        warnings.warn(
            f"Season {season}: shifts file not found at {shifts_path}.\n"
            "  Run: uv run python scripts/run_phase1_sync.py",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(), pl.DataFrame()

    if not shots_path.exists():
        warnings.warn(
            f"Season {season}: shots file not found at {shots_path}.\n"
            "  Run: uv run python scripts/run_phase1_sync.py",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(), pl.DataFrame()

    print(f"  Loading season {season} ({season_type}): {shifts_path.name}, {shots_path.name}")
    shifts_df = pl.read_parquet(shifts_path)
    shots_df  = pl.read_parquet(shots_path)

    if season_type == "playoffs":
        if "game_id" in shifts_df.columns:
            shifts_df = shifts_df.filter(_game_type_from_id() == 3)
        if "is_playoff" in shots_df.columns:
            shots_df = shots_df.filter(pl.col("is_playoff") == True)  # noqa: E712
    elif season_type == "regular":
        if "game_id" in shifts_df.columns:
            shifts_df = shifts_df.filter(_game_type_from_id() == 2)
        if "is_playoff" in shots_df.columns:
            shots_df = shots_df.filter(pl.col("is_playoff") == False)  # noqa: E712

    return shifts_df, shots_df


def _print_leaders(result: pl.DataFrame, season: int) -> None:
    """Print top 10 EV offensive and defensive RAPM leaders."""
    print(f"\n  ── Season {season} EV Offensive RAPM leaders ──")
    top_off = (
        result
        .filter(pl.col("toi_ev") >= 3600)  # min 60 min EV TOI
        .sort("rapm_ev_off", descending=True)
        .head(10)
        .select(["player_name", "team", "rapm_ev_off", "toi_ev"])
    )
    for r in top_off.to_dicts():
        toi_min = r["toi_ev"] / 60.0
        print(f"    {r['player_name']:<22} {r['team']}  EV_Off: {r['rapm_ev_off']:+.3f}  TOI: {toi_min:.0f}m")

    print(f"\n  ── Season {season} EV Defensive RAPM leaders ──")
    top_def = (
        result
        .filter(pl.col("toi_ev") >= 3600)
        .sort("rapm_ev_def", descending=False)   # lower xGA = better defender
        .head(10)
        .select(["player_name", "team", "rapm_ev_def", "toi_ev"])
    )
    for r in top_def.to_dicts():
        toi_min = r["toi_ev"] / 60.0
        print(f"    {r['player_name']:<22} {r['team']}  EV_Def: {r['rapm_ev_def']:+.3f}  TOI: {toi_min:.0f}m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RAPM model with daisy-chain priors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_default_seasons(),
        metavar="YEAR",
        help="Season years to fit (oldest → newest). "
             "Each season uses the previous season's posterior as its prior. "
             "Defaults to the 5 most recent NHL seasons.",
    )
    parser.add_argument(
        "--chain-from",
        type=int,
        default=None,
        metavar="YEAR",
        help="Extend the daisy chain back to this year to warm up priors "
             "(seasons before --seasons are fit but not saved).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if output parquets already exist.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="PATH",
        help="Root data directory.",
    )
    parser.add_argument(
        "--season-type",
        choices=["regular", "playoffs"],
        default="regular",
        help="'regular' writes rapm_{season}.parquet (default). "
             "'playoffs' filters shifts+shots to playoff games and writes "
             "rapm_{season}_playoffs.parquet.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir = data_dir / "rapm"
    output_dir.mkdir(parents=True, exist_ok=True)
    season_type = args.season_type
    out_suffix = "_playoffs" if season_type == "playoffs" else ""

    # Check if all outputs already exist
    target_seasons: list[int] = sorted(args.seasons)
    if not args.force:
        all_exist = all(
            (output_dir / f"rapm_{s}{out_suffix}.parquet").exists()
            for s in target_seasons
        )
        if all_exist:
            print(
                f"All RAPM parquets already exist for {target_seasons}{out_suffix}. "
                "Use --force to retrain."
            )
            return

    # Build full season list (warmup + target)
    warmup_seasons: list[int] = []
    if args.chain_from is not None:
        warmup_seasons = sorted(
            s for s in range(args.chain_from, min(target_seasons))
            if s not in target_seasons
        )

    all_seasons = warmup_seasons + target_seasons
    print(f"Daisy chain: {all_seasons}")
    print(f"Saving outputs for: {target_seasons}")
    print(f"Output dir: {output_dir}\n")

    # Verify at least one season has shifts data
    has_shifts = any(
        (data_dir / "raw" / f"shifts_{s}.parquet").exists()
        for s in target_seasons
    )
    if not has_shifts:
        print(
            "ERROR: No shift data found for any target season.\n"
            "  Shift data is required to build the stints matrix.\n\n"
            "  Run the historical ingestor first:\n"
            "    uv run python scripts/gretzky.py ingest\n\n"
            "  This will populate:\n"
            f"    {data_dir / 'raw' / 'shifts_<season>.parquet'}\n"
            f"    {data_dir / 'raw' / 'pbp_<season>.parquet'}",
            file=sys.stderr,
        )
        sys.exit(1)

    from models.rapm_model import RAPMModel

    model = RAPMModel()

    # Build a cross-season player name lookup. Priority (highest to lowest):
    #   1. NHL Stats API bulk skater summary — covers every skater with ≥1 GP
    #   2. EDGE skating parquets — all skaters with EDGE qualifying minutes
    #   3. Goalie ratings parquets — goalies (not in EDGE)
    #   4. MoneyPuck shot data — shooters only fallback
    name_fallback: dict[int, str] = {}

    # Layer 1: shots (cheapest, load first so higher-quality sources overwrite)
    for _ns in (2023, 2024, 2025):
        _shots_path = data_dir / "shots" / f"shots_{_ns}.parquet"
        if _shots_path.exists():
            _s = pl.read_parquet(_shots_path, columns=["shooter_id", "shooter_name"])
            for _r in _s.drop_nulls("shooter_id").unique("shooter_id").to_dicts():
                name_fallback[int(_r["shooter_id"])] = str(_r["shooter_name"])

    # Layer 2: EDGE skating
    edge_dir = data_dir / "edge"
    if edge_dir.exists():
        for _ep in sorted(edge_dir.glob("edge_skating_*.parquet")):
            try:
                _e = pl.read_parquet(_ep, columns=["player_id", "player_name"])
                for _r in _e.drop_nulls("player_id").unique("player_id").to_dicts():
                    if _r["player_name"]:
                        name_fallback[int(_r["player_id"])] = str(_r["player_name"])
            except Exception:
                pass

    # Layer 3: goalie ratings (goalies aren't in EDGE skating)
    goalie_dir = data_dir / "goalie_ratings"
    if goalie_dir.exists():
        for _gp in sorted(goalie_dir.glob("goalie_ratings_*.parquet")):
            try:
                _g = pl.read_parquet(_gp, columns=["player_id", "player_name"])
                for _r in _g.drop_nulls("player_id").unique("player_id").to_dicts():
                    if _r["player_name"]:
                        name_fallback[int(_r["player_id"])] = str(_r["player_name"])
            except Exception:
                pass

    # Layer 4: NHL Stats API bulk skater + goalie summary — highest quality,
    # covers all players incl. low-TOI players not in EDGE. Overwrites above.
    try:
        import asyncio as _asyncio
        from data.nhl_client import NHLClient as _NHLClient

        async def _fetch_names() -> dict[int, str]:
            result: dict[int, str] = {}
            async with _NHLClient() as _c:
                for _season_id in ("20222023", "20232024", "20242025"):
                    for _endpoint, _name_key in [
                        ("/stats/rest/en/skater/summary", "skaterFullName"),
                        ("/stats/rest/en/goalie/summary",  "goalieFullName"),
                    ]:
                        start = 0
                        while True:
                            try:
                                _r = await _c._get(_c._stats, _endpoint, params={
                                    "isAggregate": "false", "isGame": "false",
                                    "factCayenneExp": "gamesPlayed>=1",
                                    "cayenneExp": f"gameTypeId=2 and seasonId={_season_id}",
                                    "start": start, "limit": 100,
                                })
                            except Exception:
                                break
                            rows = _r.get("data", [])
                            for row in rows:
                                pid = row.get("playerId")
                                name = row.get(_name_key) or ""
                                if pid and name:
                                    result[int(pid)] = name
                            total = _r.get("total", 0)
                            start += len(rows)
                            if start >= total or not rows:
                                break
            return result

        _api_names = _asyncio.run(_fetch_names())
        name_fallback.update(_api_names)
        print(f"  [rapm] NHL API name lookup: {len(_api_names)} players")
    except Exception as _e:
        print(f"  [rapm] NHL API name lookup failed (non-fatal): {_e}")

    model.name_fallback = name_fallback

    def loader(season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        return _load_season(data_dir, season, season_type=season_type)

    print("Training RAPM daisy chain...\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = model.fit_daisy_chain(
            seasons=all_seasons,
            data_loader_fn=loader,
        )

    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    # Save target seasons only
    saved = 0
    for season in target_seasons:
        if season not in results:
            print(f"  Season {season}: no result (data missing?)")
            continue

        result = results[season]
        out_path = output_dir / f"rapm_{season}{out_suffix}.parquet"

        if out_path.exists() and not args.force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            continue

        result.write_parquet(out_path)
        print(f"  Saved {out_path}  ({len(result)} players)")
        _print_leaders(result, season)
        saved += 1

    print(f"\nDone. {saved}/{len(target_seasons)} seasons saved.")

    # Save model object
    model_path = data_dir / "rapm" / "rapm_model.pkl"
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
