"""compute_seasonal_performance — Feature 3.22 driver script.

Reads the latest composite-FI parquet (3.17) for the (player, game)
row set, joins roster age + games-played-to-date, optionally pulls
playoff_prob from a standings projector parquet (10.3 — falls back to
0.5 when missing), and writes the seasonal motivation factor per row.

Usage::

    uv run python scripts/gretzky.py seasonal-performance
    uv run python scripts/gretzky.py seasonal-performance -- --date 2026-05-17
    uv run python scripts/gretzky.py seasonal-performance -- --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from models.seasonal_performance import (
    SeasonalPerformanceFactor,
    write_seasonal_performance,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
FI_SUBDIR        = "composite_fi"
ROSTER_SUBDIR    = "raw"
PLAYOFF_PROB_PARQUET = "playoff_prob.parquet"
PLAYER_STATS_SUBDIR  = "raw"
OUT_SUBDIR       = "seasonal_performance"


def _latest_fi_parquet(fi_dir: Path) -> Path | None:
    if not fi_dir.exists():
        return None
    parquets = sorted(fi_dir.glob("composite_fi_*.parquet"))
    return parquets[-1] if parquets else None


def _load_roster_age(data_dir: Path) -> dict[int, float]:
    """Build a {player_id: age_years} map from cached roster JSON files."""
    roster_dir = data_dir / ROSTER_SUBDIR
    today = datetime.now(timezone.utc).date()
    ages: dict[int, float] = {}
    if not roster_dir.exists():
        return ages
    for f in sorted(roster_dir.glob("roster_*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for skater in data.get("forwards", []) + data.get("defensemen", []) + data.get("goalies", []):
            pid = skater.get("id")
            bdate = skater.get("birthDate") or skater.get("birth_date")
            if pid is None or not bdate:
                continue
            try:
                b = datetime.strptime(bdate[:10], "%Y-%m-%d").date()
                pid_i = int(pid)
                ages[pid_i] = (today - b).days / 365.25
            except (ValueError, TypeError):
                continue
    return ages


def _load_player_gp(data_dir: Path, season: int) -> dict[int, int]:
    """{player_id: GP_so_far_this_season} from player_stats parquet."""
    stats_path = data_dir / PLAYER_STATS_SUBDIR / f"player_stats_{season}.parquet"
    if not stats_path.exists():
        return {}
    df = pl.read_parquet(stats_path)
    if "player_id" not in df.columns:
        return {}
    # If there is a per-game row format, count rows per player.
    if "game_id" in df.columns:
        agg = df.group_by("player_id").agg(pl.count().alias("gp_ytd"))
    elif "gp" in df.columns:
        agg = df.group_by("player_id").agg(pl.col("gp").sum().alias("gp_ytd"))
    else:
        return {}
    return {int(r["player_id"]): int(r["gp_ytd"]) for r in agg.to_dicts()}


def _load_playoff_prob(data_dir: Path) -> dict[str, float]:
    """{team_abbrev: playoff_prob} — empty when 10.3 hasn't run yet."""
    p = data_dir / PLAYOFF_PROB_PARQUET
    if not p.exists():
        return {}
    df = pl.read_parquet(p)
    if not {"team", "playoff_prob"}.issubset(set(df.columns)):
        return {}
    return {str(r["team"]).upper(): float(r["playoff_prob"]) for r in df.to_dicts()}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute per-(player, game) seasonal motivation factor "
                    "(Feature 3.22)."
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--date", type=str, default=None,
                   help="as-of date. Default: today (UTC).")
    p.add_argument("--season", type=int, default=None,
                   help="NHL season year (start year) for GP lookup.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / OUT_SUBDIR
    out_path = out_dir / f"seasonal_performance_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[seasonal-performance] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    # Pull the FI parquet for the (player, game) set.
    fi_dir  = args.data_dir / FI_SUBDIR
    fi_path = fi_dir / f"composite_fi_{as_of}.parquet"
    if not fi_path.exists():
        latest = _latest_fi_parquet(fi_dir)
        if latest is None:
            print(f"[seasonal-performance] No composite-FI parquet in {fi_dir}.")
            print("  Run `gretzky composite-fi` first.")
            sys.exit(1)
        fi_path = latest
    print(f"[seasonal-performance] Reading FI rows from {fi_path}")
    fi_df = pl.read_parquet(fi_path)
    base_cols = [c for c in ("player_id", "game_id", "game_date") if c in fi_df.columns]
    if "player_id" not in base_cols or "game_date" not in base_cols:
        print("[seasonal-performance] FI parquet missing required cols.")
        sys.exit(1)
    fi_df = fi_df.select(base_cols).unique()
    print(f"  {len(fi_df):,} unique (player, game) rows")

    season = args.season or datetime.strptime(as_of, "%Y-%m-%d").year
    print(f"[seasonal-performance] Loading roster ages…")
    ages = _load_roster_age(args.data_dir)
    print(f"  {len(ages):,} player ages")

    print(f"[seasonal-performance] Loading player GP-to-date for season {season}…")
    gp_lookup = _load_player_gp(args.data_dir, season)
    print(f"  {len(gp_lookup):,} player GP rows")

    print(f"[seasonal-performance] Loading playoff_prob projections (optional)…")
    playoff_lookup = _load_playoff_prob(args.data_dir)
    print(f"  {len(playoff_lookup):,} team playoff probabilities")

    rows = fi_df.to_dicts()
    enriched: list[dict] = []
    for r in rows:
        pid = int(r["player_id"])
        # Playoff prob lookup requires a team — we don't have it from
        # composite_fi directly; default to 0.5 until 10.3 wires through.
        enriched.append({
            "player_id":         pid,
            "game_id":           int(r.get("game_id") or 0),
            "game_date":         str(r["game_date"]),
            "playoff_prob":      0.5,
            "age":               float(ages.get(pid, 27.0)),
            "games_played_ytd":  int(gp_lookup.get(pid, 41)),
        })
    inputs_df = pl.DataFrame(enriched, schema={
        "player_id":         pl.Int64,
        "game_id":           pl.Int64,
        "game_date":         pl.Utf8,
        "playoff_prob":      pl.Float64,
        "age":               pl.Float64,
        "games_played_ytd":  pl.Int64,
    })

    model = SeasonalPerformanceFactor()
    print("\n  Per-month base effects:")
    for m in range(1, 13):
        print(f"    Month {m:>2d}: {model.base_month[m]:+.4f}")

    result = model.predict(inputs_df, as_of_date=as_of)
    n = len(result)
    print(f"\n  {n:,} (player, game) rows produced")

    if n > 0:
        print("\n  Distribution by month:")
        by_month = (
            result.group_by("month_of_season")
                  .agg([
                      pl.col("seasonal_motivation_factor").mean().alias("mean_factor"),
                      pl.count().alias("n"),
                  ])
                  .sort("month_of_season")
        )
        for r in by_month.to_dicts():
            print(f"    Month {r['month_of_season']:>2d}:  "
                  f"n={r['n']:>5d}  mean_factor={r['mean_factor']:+.4f}")

    path = write_seasonal_performance(result, out_dir, as_of)
    print(f"\n[seasonal-performance] Written: {path}")

    model_path = out_dir / "seasonal_performance_model.pkl"
    model.save(model_path)
    print(f"[seasonal-performance] Model saved: {model_path}")


if __name__ == "__main__":
    main()
