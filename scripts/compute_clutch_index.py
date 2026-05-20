"""compute_clutch_index — Feature 2.29 script.

Computes the per-player Clutch Index (Win Probability Added) from NHL
play-by-play data.

The script:
  1. Loads goal events and player TOI from PBP parquets
  2. (Optional) Fits the win-probability logistic regression on historical
     game states and outcomes — or falls back to the calibrated sigmoid
     approximation when data is insufficient
  3. Computes per-player WPA, clutch index, and Bayesian-shrunk clutch index
     for each season
  4. Writes per-season and career-weighted parquets to <data_dir>/clutch_index/

Output files:
    <data_dir>/clutch_index/clutch_index_{season}.parquet
    <data_dir>/clutch_index/clutch_index_career.parquet
    <data_dir>/clutch_index/clutch_index_model.pkl

Usage::

    uv run python scripts/compute_clutch_index.py
    uv run python scripts/compute_clutch_index.py --seasons 2022 2023 2024 2025
    uv run python scripts/compute_clutch_index.py --validated-seasons 2022 2023 2024
    uv run python scripts/compute_clutch_index.py --force
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

from models.clutch_index import (
    ASSIST_WEIGHT,
    FULL_SIGNAL_GP,
    SEASON_DECAY,
    VALIDATION_SEASONS,
    ClutchIndexModel,
    WinProbabilityModel,
    write_clutch_index,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR   = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
DEFAULT_SEASONS    = [2022, 2023, 2024, 2025]
RAW_SUBDIR         = "raw"
CLUTCH_SUBDIR      = "clutch_index"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_goals(data_dir: Path, season: int, season_type: str = "regular") -> pl.DataFrame:
    """Load goal events for a season from PBP parquets.

    Accepted filename patterns:
      - ``pbp_events_{season}.parquet``
      - ``pbp_{season}.parquet``

    Filters to event_type == "goal".  Adds home_score_before / away_score_before
    from home_score / away_score if pre-computed columns are absent (assumes
    scores in the PBP represent the state *after* the event, so we subtract 1
    from the scoring team to get the state before).
    """
    raw_dir = data_dir / RAW_SUBDIR
    df: pl.DataFrame | None = None

    for pattern in (
        f"pbp_events_{season}.parquet",
        f"pbp_{season}.parquet",
    ):
        f = raw_dir / pattern
        if f.exists():
            try:
                df = pl.read_parquet(f)
                print(f"  Loaded PBP: {f.name}  ({len(df):,} rows)")
                break
            except Exception as e:
                print(f"  Warning: could not load {f.name}: {e}")

    if df is None:
        return pl.DataFrame()

    # Filter by game type (game_id encodes it: SSSS{TT}{GGGG} where TT=02|03)
    if "game_id" in df.columns:
        gt = (pl.col("game_id") % 1_000_000) // 10_000
        if season_type == "playoffs":
            df = df.filter(gt == 3)
        elif season_type == "regular":
            df = df.filter(gt == 2)

    # Filter to goal events
    if "event_type" in df.columns:
        df = df.filter(pl.col("event_type") == "goal")
    elif "event_type_raw" in df.columns:
        df = df.filter(pl.col("event_type_raw").str.to_lowercase().str.contains("goal"))

    if len(df) == 0:
        return pl.DataFrame()

    # Build home_score_before / away_score_before if absent
    if "home_score_before" not in df.columns and "home_score" in df.columns:
        # home_score / away_score in PBP represent state *after* the event
        if "event_owner_team_id" in df.columns and "home_team_id" in df.columns:
            df = df.with_columns([
                pl.when(pl.col("event_owner_team_id") == pl.col("home_team_id"))
                  .then(pl.col("home_score") - 1)
                  .otherwise(pl.col("home_score"))
                  .alias("home_score_before"),
                pl.when(pl.col("event_owner_team_id") != pl.col("home_team_id"))
                  .then(pl.col("away_score") - 1)
                  .otherwise(pl.col("away_score"))
                  .alias("away_score_before"),
            ])
        else:
            df = df.with_columns([
                (pl.col("home_score") - 1).alias("home_score_before"),
                pl.col("away_score").alias("away_score_before"),
            ])

    # Rename time columns if needed
    if "time_remaining_secs" not in df.columns and "time_remaining_secs" in df.columns:
        pass   # already correct
    elif "time_remaining_secs" not in df.columns and "time_in_period_secs" in df.columns:
        # time_in_period_secs = elapsed; convert to remaining
        period_secs = 20 * 60
        df = df.with_columns(
            (pl.lit(period_secs) - pl.col("time_in_period_secs")).alias("time_remaining_secs")
        )

    # Ensure scorer_id column
    if "scorer_id" not in df.columns:
        for alt in ("goal_player_id", "player_id"):
            if alt in df.columns:
                df = df.rename({alt: "scorer_id"})
                break

    print(f"    → {len(df):,} goal events for season {season}")
    return df


def _load_toi(data_dir: Path, season: int) -> pl.DataFrame:
    """Load player TOI data for a season.

    Accepted filename patterns:
      - ``player_toi_{season}.parquet``
      - ``player_game_stats_{season}.parquet``
      - ``moneypuck_skaters_{season}.parquet``
    """
    raw_dir = data_dir / RAW_SUBDIR
    df: pl.DataFrame | None = None

    for pattern in (
        f"toi_{season}.parquet",
        f"player_toi_{season}.parquet",
        f"player_game_stats_{season}.parquet",
        f"moneypuck_skaters_{season}.parquet",
    ):
        f = raw_dir / pattern
        if f.exists():
            try:
                df = pl.read_parquet(f)
                print(f"  Loaded TOI: {f.name}  ({len(df):,} rows)")
                break
            except Exception as e:
                print(f"  Warning: could not load {f.name}: {e}")

    if df is None:
        return pl.DataFrame()

    # Normalise column names
    renames: dict[str, str] = {}
    for alt in ("playerId", "player_id_nhl"):
        if alt in df.columns and "player_id" not in df.columns:
            renames[alt] = "player_id"
    for alt in ("toi_total_secs", "icetime", "toi", "toi_60", "I_F_icetime"):
        if alt in df.columns and "toi_seconds" not in df.columns:
            # If the value looks like minutes, convert to seconds
            renames[alt] = "_toi_raw"

    if renames:
        df = df.rename(renames)

    if "toi_seconds" not in df.columns and "_toi_raw" in df.columns:
        # Heuristic: if median value < 30, it's in minutes; else seconds
        med = df["_toi_raw"].drop_nulls().cast(pl.Float64).median() or 0.0
        if med < 30:
            df = df.with_columns((pl.col("_toi_raw") * 60.0).alias("toi_seconds"))
        else:
            df = df.with_columns(pl.col("_toi_raw").cast(pl.Float64).alias("toi_seconds"))

    if "player_id" not in df.columns or "toi_seconds" not in df.columns:
        print(f"  [warn] TOI file missing required columns (player_id, toi_seconds); skipping.")
        return pl.DataFrame()

    return df


def _load_game_outcomes(data_dir: Path, season: int) -> pl.DataFrame:
    """Load game outcomes (home_win) for WP model fitting.

    Accepted patterns:
      - ``game_results_{season}.parquet``
      - ``schedule_{season}.parquet``
    Returns empty DataFrame if not found (WP model falls back to analytical).
    """
    raw_dir = data_dir / RAW_SUBDIR
    for pattern in (f"game_results_{season}.parquet", f"schedule_{season}.parquet"):
        f = raw_dir / pattern
        if f.exists():
            try:
                df = pl.read_parquet(f)
                if "home_win" not in df.columns:
                    if "home_score" in df.columns and "away_score" in df.columns:
                        df = df.with_columns(
                            (pl.col("home_score") > pl.col("away_score")).alias("home_win")
                        )
                    else:
                        return pl.DataFrame()
                if "game_id" not in df.columns:
                    return pl.DataFrame()
                print(f"  Loaded outcomes: {f.name}  ({len(df):,} games)")
                return df.select(["game_id", "home_win"])
            except Exception as e:
                print(f"  Warning: could not load {f.name}: {e}")
    return pl.DataFrame()


# ---------------------------------------------------------------------------
# WP model state builder (for optional logistic regression training)
# ---------------------------------------------------------------------------

def _build_wp_states(goals_df: pl.DataFrame, outcomes_df: pl.DataFrame) -> pl.DataFrame:
    """Build per-goal game-state rows for WP model training.

    Each row is a game state observation that will be labelled with the
    ultimate game outcome to train P(home_win | state).
    """
    if goals_df.is_empty() or outcomes_df.is_empty():
        return pl.DataFrame()

    required = {"game_id", "period", "time_remaining_secs", "home_score_before", "away_score_before"}
    if not required.issubset(set(goals_df.columns)):
        return pl.DataFrame()

    from models.clutch_index import _time_elapsed_fraction

    rows = []
    for r in goals_df.to_dicts():
        try:
            period = int(r["period"])
            tr     = int(r["time_remaining_secs"])
            te     = _time_elapsed_fraction(period, tr)
            ld     = int(r["home_score_before"]) - int(r["away_score_before"])
            rows.append({
                "game_id":      int(r["game_id"]),
                "lead_diff":    ld,
                "period":       period,
                "time_elapsed": te,
            })
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_leaderboard(df: pl.DataFrame, n: int = 15) -> None:
    """Print top and bottom clutch players."""
    if df.is_empty():
        return

    top = df.sort("clutch_index_shrunk", descending=True).head(n)
    print(f"\n  Top-{n} clutch performers (highest WPA above expectation):")
    print(
        f"  {'Player':<25}  {'Team':>4}  {'GP':>4}  "
        f"{'Act WPA/60':>10}  {'Exp WPA/60':>10}  {'Clutch':>7}  {'Shrink':>7}"
    )
    print(f"  {'─'*25}  {'─'*4}  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*7}")
    for row in top.to_dicts():
        name = (row.get("player_name") or "")[:24] or f"ID:{row['player_id']}"
        print(
            f"  {name:<25}  {str(row.get('team',''))[:4]:>4}  "
            f"{row.get('career_gp', 0):>4}  "
            f"{row['actual_wpa_per60']:>+10.4f}  "
            f"{row['expected_wpa_per60']:>+10.4f}  "
            f"{row['clutch_index_shrunk']:>+7.4f}  "
            f"{row['shrinkage_factor']:>7.3f}"
        )

    bottom = df.sort("clutch_index_shrunk", descending=False).head(5)
    print(f"\n  Bottom-5 (most anti-clutch):")
    print(f"  {'Player':<25}  {'Clutch':>7}")
    print(f"  {'─'*25}  {'─'*7}")
    for row in bottom.to_dicts():
        name = (row.get("player_name") or "")[:24] or f"ID:{row['player_id']}"
        print(f"  {name:<25}  {row['clutch_index_shrunk']:>+7.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Clutch Index (Win Probability Added) per player (Feature 2.29)."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        metavar="YEAR",
        help=f"Seasons to process (default: {DEFAULT_SEASONS}).",
    )
    parser.add_argument(
        "--validated-seasons",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        dest="validated_seasons",
        help=(
            f"Seasons already back-tested. "
            f"Requires ≥{VALIDATION_SEASONS} to mark signal as validated."
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
        help="Overwrite existing output parquets.",
    )
    parser.add_argument(
        "--season-type",
        choices=["regular", "playoffs"],
        default="regular",
        help="'regular' writes clutch_index_{year}.parquet (default). "
             "'playoffs' filters PBP to playoff games and writes "
             "clutch_index_{year}_playoffs.parquet.",
    )
    args = parser.parse_args()

    seasons: list[int] = sorted(args.seasons)
    target_season: int = max(seasons)
    data_dir: Path     = args.data_dir
    season_type: str   = args.season_type
    out_suffix         = "_playoffs" if season_type == "playoffs" else ""

    print(f"[clutch-index] Feature 2.29 — Clutch Index (Win Probability Added)")
    print(f"  Assist weight  : {ASSIST_WEIGHT}")
    print(f"  Full-signal GP : {FULL_SIGNAL_GP} career GP")
    print(f"  Season decay   : {SEASON_DECAY}^(seasons_ago)")
    print(f"  Seasons        : {seasons}")
    print(f"  Target season  : {target_season}")
    print(f"  Data dir       : {data_dir}")

    # ── Check output ──────────────────────────────────────────────────────────
    out_dir    = data_dir / CLUTCH_SUBDIR
    out_season = out_dir / f"clutch_index_{target_season}{out_suffix}.parquet"
    out_career = out_dir / f"clutch_index_career{out_suffix}.parquet"

    if out_season.exists() and out_career.exists() and not args.force:
        print(f"\n[clutch-index] Output already exists: {out_season}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    raw_dir = data_dir / RAW_SUBDIR
    if not raw_dir.exists():
        print(f"\n[clutch-index] Raw data directory not found: {raw_dir}")
        print("  Run 'uv run python scripts/gretzky.py sync' or 'ingest' first.")
        sys.exit(1)

    # ── Instantiate model ─────────────────────────────────────────────────────
    model = ClutchIndexModel()
    validated_seasons = args.validated_seasons or []

    # ── Build player name lookup from NHL Stats API ───────────────────────────
    name_lookup: dict[int, str] = {}
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

        name_lookup = _asyncio.run(_fetch_names())
        print(f"  [clutch-index] NHL API name lookup: {len(name_lookup)} players")
    except Exception as _e:
        print(f"  [clutch-index] NHL API name lookup failed (non-fatal): {_e}")

    player_meta_df = pl.DataFrame(
        [{"player_id": pid, "player_name": name} for pid, name in name_lookup.items()]
    ) if name_lookup else None

    # ── Process each season ───────────────────────────────────────────────────
    season_dfs: list[pl.DataFrame] = []

    for season in seasons:
        print(f"\n[clutch-index] Processing season {season}…")

        goals_df    = _load_goals(data_dir, season, season_type=season_type)
        toi_df      = _load_toi(data_dir, season)
        outcomes_df = _load_game_outcomes(data_dir, season)

        if goals_df.is_empty():
            print(f"  [warn] No goal events for season {season} — skipping.")
            season_dfs.append(pl.DataFrame())
            continue
        if toi_df.is_empty():
            print(f"  [warn] No TOI data for season {season} — skipping.")
            season_dfs.append(pl.DataFrame())
            continue

        # Optional: train WP logistic regression if game outcomes available
        if not outcomes_df.is_empty() and not model._wp_model.is_fitted:
            print(f"  Fitting win-probability model on season {season} data…")
            states_df = _build_wp_states(goals_df, outcomes_df)
            if not states_df.is_empty():
                try:
                    model.fit_wp_model(states_df, outcomes_df)
                    print(f"  WP model fitted on {len(states_df):,} goal-state observations.")
                except Exception as e:
                    print(f"  [warn] WP model fitting failed: {e}; using analytical fallback.")
            else:
                print(f"  [warn] Could not build WP states; using analytical fallback.")
        elif not model._wp_model.is_fitted:
            print(f"  No game outcomes found; using analytical WP fallback.")

        # Compute clutch index for this season
        n_goals  = len(goals_df)
        n_players = toi_df["player_id"].n_unique() if "player_id" in toi_df.columns else 0
        print(f"  Computing WPA for {n_goals:,} goals × {n_players:,} players…")

        try:
            season_df = model.compute_season(
                goals_df=goals_df,
                toi_df=toi_df,
                season=season,
                player_meta_df=player_meta_df,
                validated_seasons=validated_seasons if season == target_season else None,
            )
        except Exception as e:
            print(f"  [error] compute_season failed for {season}: {e}")
            season_dfs.append(pl.DataFrame())
            continue

        n_scored  = len(season_df)
        n_pos     = int((season_df["clutch_index_shrunk"] > 0).sum()) if n_scored > 0 else 0
        n_neg     = int((season_df["clutch_index_shrunk"] < 0).sum()) if n_scored > 0 else 0
        print(f"  Players scored: {n_scored:,}  (positive: {n_pos:,}  negative: {n_neg:,})")

        if season == target_season and not season_df.is_empty():
            _print_leaderboard(season_df)

        season_dfs.append(season_df)

    # ── Career weighted average ───────────────────────────────────────────────
    non_empty = [(df, s) for df, s in zip(season_dfs, seasons) if not df.is_empty()]
    if non_empty:
        career_dfs, career_seasons = zip(*non_empty)
        print(f"\n[clutch-index] Computing career-weighted index ({len(career_seasons)} seasons)…")
        career_df = model.compute_career(list(career_dfs), list(career_seasons))
        print(f"  Career rows: {len(career_df):,}")
    else:
        print(f"\n[clutch-index] No valid seasons to build career table.")
        career_df = pl.DataFrame()

    # Determine the season_df for output (target season or last non-empty)
    target_idx = seasons.index(target_season)
    out_season_df = season_dfs[target_idx]
    if out_season_df.is_empty():
        # Fall back to most recent non-empty
        for df in reversed(season_dfs):
            if not df.is_empty():
                out_season_df = df
                break

    if out_season_df.is_empty() and career_df.is_empty():
        print("\n[clutch-index] No data produced — check that PBP and TOI parquets exist.")
        print(f"  Expected under: {raw_dir}/")
        # Playoffs variant is a no-op when playoff PBP hasn't been ingested
        # yet (pre-playoff regular season, or pbp_*.parquet ingested before
        # the game_types=2,3 default was set). Don't fail the whole pipeline.
        if season_type == "playoffs":
            sys.exit(0)
        sys.exit(1)

    # ── Validation gate ───────────────────────────────────────────────────────
    validated = model.is_validated
    print(f"\n  Signal validated: {validated} "
          f"({'≥3 seasons confirmed' if validated else 'gate CLOSED — do not apply in production'})")
    if not validated:
        print(
            "\n  [GATE] Signal is NOT validated. To apply clutch index in production:\n"
            f"    1. Back-test against ≥{VALIDATION_SEASONS} seasons of NHL PBP\n"
            "    2. Re-run with --validated-seasons <year1> <year2> <year3>"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    if not out_season_df.is_empty() or not career_df.is_empty():
        out_dir.mkdir(parents=True, exist_ok=True)

        if not out_season_df.is_empty():
            p_season, p_career = write_clutch_index(
                out_season_df, career_df, out_dir, target_season, season_type=season_type
            )
            print(f"\n[clutch-index] Season parquet written  : {p_season}")
            print(f"[clutch-index] Career parquet written  : {p_career}")
        else:
            if not career_df.is_empty():
                p_career = out_dir / f"clutch_index_career{out_suffix}.parquet"
                career_df.write_parquet(p_career)
                print(f"\n[clutch-index] Career parquet written  : {p_career}")

        # Save model
        model_path = out_dir / "clutch_index_model.pkl"
        model.save(model_path)
        print(f"[clutch-index] Model saved             : {model_path}")

    print(f"\n[clutch-index] Done.")


if __name__ == "__main__":
    main()
