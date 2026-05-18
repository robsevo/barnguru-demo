"""compute_composite_fi — Feature 3.17 driver script.

Joins every Phase 3 fatigue signal already on disk into a single
(player, game) signal frame, runs ``CompositeFatigueIndex``, and writes
the composite-FI parquet.

This script does not invent inputs. If a feature parquet is missing the
corresponding column is left at its no-effect default and the composite
FI absorbs whatever signals are present.

Usage::

    uv run python scripts/gretzky.py composite-fi
    uv run python scripts/gretzky.py composite-fi -- --date 2026-05-17
    uv run python scripts/gretzky.py composite-fi -- --force
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from data.schedule_sync import latest_schedule_parquet
from models.composite_fi import (
    CompositeFatigueIndex,
    write_composite_fi,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)

COMPOSITE_FI_SUBDIR     = "composite_fi"
TOI_LOAD_SUBDIR         = "toi_load"
ST_LOAD_SUBDIR          = "special_teams_load"
CONTACT_LOAD_SUBDIR     = "physical_contact_load"
OT_FATIGUE_SUBDIR       = "overtime_fatigue"
FIGHT_FATIGUE_SUBDIR    = "fight_fatigue"
SCHEDULE_DENSITY_SUBDIR = "schedule_density"
TRAVEL_DISTANCE_SUBDIR  = "travel_distance"
TZ_CROSSING_SUBDIR      = "time_zone_crossing"
CIRCADIAN_SUBDIR        = "circadian_alignment"
ALTITUDE_SUBDIR         = "altitude_adjustment"
AGE_RECOVERY_SUBDIR     = "age_recovery"
INJURY_STATUS_SUBDIR    = "injury_status"
CONCUSSION_SUBDIR       = "concussion_history"
PRIOR_PLAYOFF_SUBDIR    = "prior_playoff_load"
PLAYOFF_FATIGUE_SUBDIR  = "playoff_fatigue"
ROSTER_STRAIN_SUBDIR    = "roster_depth_strain"


def _latest_parquet(dirpath: Path, prefix: str) -> Path | None:
    if not dirpath.exists():
        return None
    candidates = sorted(dirpath.glob(f"{prefix}_*.parquet"))
    return candidates[-1] if candidates else None


def _read_or_empty(dirpath: Path, prefix: str) -> pl.DataFrame:
    path = _latest_parquet(dirpath, prefix)
    if path is None:
        return pl.DataFrame()
    return pl.read_parquet(path)


def _build_signals(args, as_of: str) -> pl.DataFrame:
    """Stitch every Phase 3 output into a (player_id, game_id, game_date) frame."""
    d = args.data_dir

    # TOI load is the spine: per (player, game).
    toi = _read_or_empty(d / TOI_LOAD_SUBDIR, "toi_load")
    if len(toi) == 0:
        print(f"[composite-fi] No toi_load parquet under {d / TOI_LOAD_SUBDIR}.")
        print("  Composite FI needs TOI load as its (player, game) spine.")
        sys.exit(1)

    base = toi.select([
        pl.col("player_id"),
        pl.col("game_id"),
        pl.col("game_date"),
        pl.col("team_id"),
        pl.col("toi_spike_z"),
    ])

    # Per-team schedule signals (3.1, 3.3, 3.4, 3.5, 3.6). Joined on
    # (team_id, game_id) but those parquets typically key on team abbrev;
    # leave them out when the spine has team_id only.
    # In practice these are computed elsewhere; we leave the columns at
    # their no-effect defaults if not provided.

    # Special teams load (3.8)
    st = _read_or_empty(d / ST_LOAD_SUBDIR, "special_teams_load")
    if len(st) > 0 and "st_intensity_score" in st.columns:
        base = base.join(
            st.select(["player_id", "game_id", "st_intensity_score"]),
            on=["player_id", "game_id"], how="left",
        )

    # Physical contact load (3.9)
    contact = _read_or_empty(d / CONTACT_LOAD_SUBDIR, "physical_contact_load")
    if len(contact) > 0 and "contact_load_score" in contact.columns:
        base = base.join(
            contact.select(["player_id", "game_id", "contact_load_score"]),
            on=["player_id", "game_id"], how="left",
        )

    # OT fatigue (3.10)
    ot = _read_or_empty(d / OT_FATIGUE_SUBDIR, "overtime_fatigue")
    if len(ot) > 0 and "ot_equiv_toi_secs" in ot.columns:
        base = base.join(
            ot.select(["player_id", "game_id", "ot_equiv_toi_secs"]),
            on=["player_id", "game_id"], how="left",
        )

    # Fight fatigue (3.11)
    fight = _read_or_empty(d / FIGHT_FATIGUE_SUBDIR, "fight_fatigue")
    if len(fight) > 0 and "fight_score" in fight.columns:
        base = base.join(
            fight.select(["player_id", "game_id", "fight_score"]),
            on=["player_id", "game_id"], how="left",
        )

    # Per-player static signals (broadcast onto every game row).
    age = _read_or_empty(d / AGE_RECOVERY_SUBDIR, "age_recovery")
    if len(age) > 0 and "recovery_coefficient" in age.columns:
        base = base.join(
            age.select(["player_id", "recovery_coefficient"]),
            on="player_id", how="left",
        )

    inj = _read_or_empty(d / INJURY_STATUS_SUBDIR, "injury_status")
    if len(inj) > 0 and "rust_factor" in inj.columns:
        base = base.join(
            inj.select(["player_id", "rust_factor"]),
            on="player_id", how="left",
        )

    conc = _read_or_empty(d / CONCUSSION_SUBDIR, "concussion_history")
    if len(conc) > 0 and "fatigue_sensitivity_multiplier" in conc.columns:
        base = base.join(
            conc.select(["player_id", "fatigue_sensitivity_multiplier"]),
            on="player_id", how="left",
        )

    pp = _read_or_empty(d / PRIOR_PLAYOFF_SUBDIR, "prior_playoff_load")
    if len(pp) > 0 and "playoff_load_penalty" in pp.columns:
        base = base.join(
            pp.select(["player_id", "playoff_load_penalty"]),
            on="player_id", how="left",
        )

    # 3.23 — current-playoff-run fatigue, per (player, game)
    pf = _read_or_empty(d / PLAYOFF_FATIGUE_SUBDIR, "playoff_fatigue")
    if len(pf) > 0 and "playoff_fatigue_score" in pf.columns:
        base = base.join(
            pf.select(["player_id", "game_id", "playoff_fatigue_score"]),
            on=["player_id", "game_id"], how="left",
        )

    # 3.1 — schedule density, per (team, game). rest_days feeds the recovery
    # patch (Part 2): a multi-day gap boosts the per-player recovery_coefficient
    # before the model applies it as a divisor.
    sd = _read_or_empty(d / SCHEDULE_DENSITY_SUBDIR, "schedule_density")
    if len(sd) > 0 and "rest_days" in sd.columns and "team" in sd.columns:
        # team_id on the spine vs team abbrev on schedule_density — join via
        # game_id alone is too loose (one row per team per game), so we
        # broadcast per game_id using the home/away team's rest_days. v1
        # assumption: every player gets the rest_days of one of the two teams;
        # this is wrong-cheap until we route via roster. Pick the row matching
        # the player's team_id via a simple per-game pick (max over both
        # teams' rest_days favors the rested side, which is conservative
        # for the fatigue boost direction).
        sd_max = (
            sd.group_by("game_id")
              .agg(pl.col("rest_days").max().alias("rest_days"))
        )
        base = base.join(sd_max, on="game_id", how="left")

    strain = _read_or_empty(d / ROSTER_STRAIN_SUBDIR, "roster_depth_strain")
    if len(strain) > 0 and "player_strain_score" in strain.columns:
        base = base.join(
            strain.select([
                pl.col("player_id"),
                pl.col("player_strain_score").alias("team_strain_score"),
            ]),
            on="player_id", how="left",
        )

    # Join game_type from schedule (2 = regular, 3 = playoffs) so the
    # backend can filter playoff-context views without a follow-up join.
    sched_path = latest_schedule_parquet(d / "schedule")
    if sched_path is not None:
        try:
            sched = pl.read_parquet(sched_path).select(["game_id", "game_type"])
            base = base.join(sched.unique(subset=["game_id"]),
                             on="game_id", how="left")
        except Exception:
            pass

    # Part 2 — Rest-day recovery patch.
    # The age-driven recovery_coefficient was static. Boost it by up to +25%
    # when the team has multi-day rest before this game so that fatigue
    # actually bleeds off between games. rest_bonus = min(0.25, (rest_days-1) × 0.05).
    # Symmetric across ages in v1; age-conditioned recovery is a v2 refinement.
    if "rest_days" in base.columns and "recovery_coefficient" in base.columns:
        base = base.with_columns(
            (
                pl.col("recovery_coefficient")
                * (
                    1.0
                    + pl.min_horizontal(
                        pl.lit(0.25),
                        pl.max_horizontal(
                            pl.lit(0.0),
                            (pl.col("rest_days").fill_null(1) - 1) * 0.05,
                        ),
                    )
                )
            ).alias("recovery_coefficient")
        )

    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute composite Fatigue Index (Feature 3.17)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None,
                        help="as-of date. Default: today (UTC).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / COMPOSITE_FI_SUBDIR
    out_path = out_dir / f"composite_fi_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[composite-fi] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[composite-fi] Stitching Phase 3 signals as of {as_of}…")
    signals = _build_signals(args, as_of)
    print(f"  {len(signals):,} (player, game) signal rows")

    model = CompositeFatigueIndex()
    print(f"  Weights (sum={model.total_weight:.3f}):")
    for name, w in sorted(model.weights.items(), key=lambda x: -x[1]):
        print(f"    {name:<22s}  {w:.3f}")

    result = model.compute(signals, as_of_date=as_of)
    n_rows = len(result)
    print(f"\n  {n_rows:,} composite-FI rows produced.")

    if n_rows > 0:
        top = result.sort("fatigue_index", descending=True).head(10)
        print("\n  Top 10 fatigued (player, game) rows:")
        print(f"  {'Player':<10}  {'Game':<10}  {'Date':<10}  {'FI':>5}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*5}")
        for r in top.to_dicts():
            print(
                f"  {r['player_id']:<10}  {r['game_id']:<10}  "
                f"{r['game_date']:<10}  {r['fatigue_index']:>5.2f}"
            )

    path = write_composite_fi(result, out_dir, as_of)
    print(f"\n[composite-fi] Written: {path}")


if __name__ == "__main__":
    main()
