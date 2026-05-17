"""compute_trade_integration — Feature 3.20 driver script.

Loads ESPN transaction parquets, derives per-(player, trade) records,
then runs ``TradeIntegrationModel`` over a set of game-date observations
(by default the next 30 days of scheduled games) and writes the per-game
integration modifier table.

Phase 4 (roster fit, 4.12) is not yet built, so this script runs in
travel-only mode unless an external fit-score parquet is provided via
``--fit-scores PATH`` with columns ``(player_id, team, fit_score)``.

Usage::

    uv run python scripts/gretzky.py trade-integration
    uv run python scripts/gretzky.py trade-integration -- --date 2026-05-17
    uv run python scripts/gretzky.py trade-integration -- --force
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from data.schedule_sync import latest_schedule_parquet
from models.trade_integration import (
    TradeIntegrationModel,
    write_trade_integration,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
TRADE_INTEGRATION_SUBDIR = "trade_integration"
SCHEDULE_SUBDIR          = "schedule"
TRANSACTIONS_SUBDIR      = "raw"
HORIZON_DAYS_DEFAULT     = 30


def _load_transactions(data_dir: Path, seasons: list[int]) -> pl.DataFrame:
    raw_dir = data_dir / TRANSACTIONS_SUBDIR
    frames: list[pl.DataFrame] = []
    for season in seasons:
        for pattern in (
            f"espn_transactions_{season}.parquet",
            f"transactions_{season}.parquet",
        ):
            p = raw_dir / pattern
            if p.exists():
                try:
                    frames.append(pl.read_parquet(p))
                    print(f"  Loaded: {p.name}")
                except Exception as e:
                    print(f"  Warning: could not load {p.name}: {e}")
                break
    if not frames:
        fallback = raw_dir / "transactions.parquet"
        if fallback.exists():
            try:
                frames.append(pl.read_parquet(fallback))
                print(f"  Loaded: {fallback.name}")
            except Exception as e:
                print(f"  Warning: could not load {fallback.name}: {e}")
    if not frames:
        tx_dir = data_dir / "transactions"
        if tx_dir.exists():
            per_date = sorted(tx_dir.glob("transactions_*.parquet"))
            for f in per_date:
                try:
                    frames.append(pl.read_parquet(f))
                except Exception:
                    pass
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def _trades_from_transactions(tx_df: pl.DataFrame) -> pl.DataFrame:
    """Reshape ESPN ``trade`` rows into the model's input schema."""
    required = {"event_type", "team", "secondary_team", "date",
                "player_or_executive", "player_id_espn"}
    missing = required - set(tx_df.columns)
    if missing:
        print(f"  [warn] transactions missing columns {missing}; "
              "trade-integration trade extraction will yield 0 rows.")
        return pl.DataFrame(schema={
            "player_id":   pl.Int64,
            "trade_date":  pl.Utf8,
            "new_team":    pl.Utf8,
            "old_team":    pl.Utf8,
            "position":    pl.Utf8,
        })

    rows: list[dict] = []
    for r in tx_df.to_dicts():
        if r.get("event_type") != "trade":
            continue
        new_team = (r.get("team") or "").strip()
        old_team = (r.get("secondary_team") or "").strip()
        date     = r.get("date") or ""
        if not new_team or not old_team or not date:
            continue
        try:
            pid = int(r.get("player_id_espn"))
        except (TypeError, ValueError):
            continue
        rows.append({
            "player_id":   pid,
            "trade_date":  date,
            "new_team":    new_team.upper(),
            "old_team":    old_team.upper(),
            "position":    "",   # ESPN doesn't include position here
        })

    if not rows:
        return pl.DataFrame(schema={
            "player_id":   pl.Int64,
            "trade_date":  pl.Utf8,
            "new_team":    pl.Utf8,
            "old_team":    pl.Utf8,
            "position":    pl.Utf8,
        })
    return pl.DataFrame(rows, schema={
        "player_id":   pl.Int64,
        "trade_date":  pl.Utf8,
        "new_team":    pl.Utf8,
        "old_team":    pl.Utf8,
        "position":    pl.Utf8,
    })


def _build_observations(
    data_dir:  Path,
    trades_df: pl.DataFrame,
    as_of:     str,
    horizon:   int,
) -> pl.DataFrame:
    """One row per (player_id, game_date) for the next ``horizon`` days."""
    sched_dir  = data_dir / SCHEDULE_SUBDIR
    sched_path = latest_schedule_parquet(sched_dir)
    if sched_path is None:
        print(f"[trade-integration] No schedule parquet in {sched_dir}.")
        return pl.DataFrame(schema={
            "player_id": pl.Int64,
            "game_id":   pl.Int64,
            "game_date": pl.Utf8,
        })
    schedule = pl.read_parquet(sched_path).select(
        ["game_id", "game_date", "home_team", "away_team"]
    )
    start = as_of
    end   = (datetime.strptime(as_of, "%Y-%m-%d")
             + timedelta(days=int(horizon))).date().isoformat()
    schedule = schedule.filter(
        (pl.col("game_date") >= start) & (pl.col("game_date") <= end)
    )

    # For each trade, emit observations across the schedule window where
    # the player's *new* team is playing.
    obs_rows: list[dict] = []
    sched_dicts = schedule.to_dicts()
    for t in trades_df.to_dicts():
        pid = t["player_id"]
        team = t["new_team"]
        for g in sched_dicts:
            if g["home_team"] == team or g["away_team"] == team:
                obs_rows.append({
                    "player_id": pid,
                    "game_id":   int(g["game_id"]),
                    "game_date": g["game_date"],
                })

    if not obs_rows:
        return pl.DataFrame(schema={
            "player_id": pl.Int64,
            "game_id":   pl.Int64,
            "game_date": pl.Utf8,
        })
    return pl.DataFrame(obs_rows, schema={
        "player_id": pl.Int64,
        "game_id":   pl.Int64,
        "game_date": pl.Utf8,
    })


def _load_fit_scores(path: Path | None) -> dict[tuple[int, str], float] | None:
    if path is None:
        return None
    if not path.exists():
        print(f"[trade-integration] fit-scores file not found: {path}")
        return None
    df = pl.read_parquet(path)
    required = {"player_id", "team", "fit_score"}
    if not required.issubset(set(df.columns)):
        print(f"[trade-integration] fit-scores parquet missing columns; "
              f"need {required}.")
        return None
    lookup: dict[tuple[int, str], float] = {}
    for r in df.to_dicts():
        lookup[(int(r["player_id"]), str(r["team"]).upper())] = float(r["fit_score"])
    return lookup


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute per-(player, game) trade integration "
                    "modifier (Feature 3.20)."
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--date", type=str, default=None,
                   help="as-of date. Default: today (UTC).")
    p.add_argument("--horizon-days", type=int, default=HORIZON_DAYS_DEFAULT,
                   help="Schedule window forward from as-of date.")
    p.add_argument("--seasons", nargs="+", type=int,
                   default=[2024, 2025, 2026],
                   help="Seasons to pool for transaction data.")
    p.add_argument("--fit-scores", type=Path, default=None,
                   help="Optional parquet with (player_id, team, fit_score).")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / TRADE_INTEGRATION_SUBDIR
    out_path = out_dir / f"trade_integration_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[trade-integration] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[trade-integration] Loading transactions for seasons "
          f"{sorted(args.seasons)}…")
    tx_df = _load_transactions(args.data_dir, sorted(args.seasons))
    print(f"  {len(tx_df):,} transaction rows loaded")

    trades_df = _trades_from_transactions(tx_df)
    print(f"  {len(trades_df):,} trade rows extracted")

    print(f"[trade-integration] Building observation window ({args.horizon_days}d)…")
    obs_df = _build_observations(args.data_dir, trades_df, as_of, args.horizon_days)
    print(f"  {len(obs_df):,} (player, game) observation rows")

    fit_scores = _load_fit_scores(args.fit_scores)
    if fit_scores is None:
        print("  No fit_scores supplied — running travel-only "
              "(fit_delta defaults to 0).")
    else:
        print(f"  Loaded {len(fit_scores)} fit_score entries")

    result = TradeIntegrationModel().compute(
        trades_df, obs_df, as_of_date=as_of, fit_scores=fit_scores,
    )
    n = len(result)
    print(f"  {n:,} integration rows produced")

    if n > 0:
        print("\n  Top-10 by |integration_factor|:")
        top = (
            result.with_columns(pl.col("integration_factor").abs().alias("_a"))
                  .sort("_a", descending=True)
                  .drop("_a")
                  .head(10)
        )
        print(f"  {'pid':>10}  {'date':<10}  {'new':<4}  {'pos':<3}  "
              f"{'gs':>3}  {'fit_d':>6}  {'factor':>7}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*4}  {'─'*3}  "
              f"{'─'*3}  {'─'*6}  {'─'*7}")
        for r in top.to_dicts():
            print(
                f"  {r['player_id']:>10}  {r['game_date']:<10}  "
                f"{r['new_team']:<4}  {r['position']:<3}  "
                f"{r['games_since_trade']:>3d}  "
                f"{r['fit_delta']:>+6.2f}  "
                f"{r['integration_factor']:>+7.3f}"
            )

    path = write_trade_integration(result, out_dir, as_of)
    print(f"\n[trade-integration] Written: {path}")


if __name__ == "__main__":
    main()
