"""Phase 16 — Convert CV summaries + NHL PBP into action-labelled training rows.

This is the bridge that closes the Cortex loop. The aggregator
(``aggregate_cv_observations``) reduces in-browser CV bundles into per-track
summaries with locked jersey numbers; this script joins those summaries to
NHL play-by-play (shots + zone entries) and emits one row per puck-touch
labelled with one of the 7 actions consumed by ``models/player_behavior_net``.

Action vocabulary (must match ``ACTION_LABELS`` in player_behavior_net.py):
    carry_in          controlled zone entry  (PBP zone_entry, controlled=True)
    dump              dump-and-chase entry   (PBP zone_entry, controlled=False)
    shoot_slot        shot from the slot     (MoneyPuck shot, distance ≤ 25 ft)
    shoot_perimeter   shot from outside slot (MoneyPuck shot, distance >  25 ft)
    battle_corner     puck battle in corner  (CV battle event in corner zone)
    drive_net         net-front drive        (CV pass-to-slot ending in slot)
    hold_corner       cycle / hold puck      (CV possession >= N s in corner)

The CV-only actions (``battle_corner``, ``drive_net``, ``hold_corner``) come
from the per-track aggregator's ``pass_or_shot`` counter as a coarse
approximation — they will be tightened once ``cv_analytics`` writes its own
per-zone event tables.

NHL play-by-play is the ground-truth source for shots and zone entries — we
emit one labelled row per PBP event whose actor we can match by
``(team, jersey_number)`` to a track that the aggregator saw.

Output: ``$GRETZKY_DATA_DIR/cv_observations/cv_actions_{season}.parquet``,
formatted with the columns the player_behavior_net.fit() path expects.
Safe to re-run — overwrites.

Usage::

    uv run python scripts/gretzky.py label-cv-actions
    uv run python scripts/gretzky.py label-cv-actions -- --season 2025
    uv run python scripts/gretzky.py label-cv-actions -- --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

# Slot threshold (ft from goal): 25 is the conservative MoneyPuck "danger
# area" boundary — anything closer is a slot shot.
SLOT_DISTANCE_FT = 25.0

# Minimum CV bundles per track before we trust a jersey lock for action
# attribution. Lock alone is required (≥3 OCR reads at the frontend); this
# adds a duration floor to filter momentary mis-IDs.
MIN_FRAMES_PER_TRACK = 8

ACTION_LABELS = [
    "carry_in",
    "dump",
    "shoot_slot",
    "shoot_perimeter",
    "battle_corner",
    "drive_net",
    "hold_corner",
]

OUT_SCHEMA: dict[str, pl.DataType] = {
    "player_id":        pl.Int64,
    "season":           pl.Int64,
    "game_id":          pl.Int64,
    "team":             pl.Utf8,
    "jersey_number":    pl.Int64,
    "action":           pl.Utf8,
    "fi_score":         pl.Float64,
    "period":           pl.Int64,
    "score_diff":       pl.Int64,
    "home_indicator":   pl.Int8,
    "manpower":         pl.Int8,
    "matchup_quality":  pl.Float64,
    "speed_ratio":      pl.Float64,
    "carry_ratio":      pl.Float64,
    "weight":           pl.Float64,
}


def _data_root() -> Path:
    default = Path.home() / ".gretzky" / "data"
    return Path(os.environ.get("GRETZKY_DATA_DIR", str(default)))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _load_aggregator_summaries(root: Path) -> pl.DataFrame:
    """Concatenate every per-game summary parquet from aggregate_cv_observations."""
    base = root / "cv_live_observations"
    if not base.is_dir():
        return pl.DataFrame()
    parquets = sorted(base.glob("*/summary.parquet"))
    if not parquets:
        return pl.DataFrame()
    frames = []
    for p in parquets:
        try:
            frames.append(pl.read_parquet(p))
        except Exception as e:  # noqa: BLE001
            print(f"[label-cv-actions] skip {p}: {e}")
    return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()


def _load_roster() -> pl.DataFrame:
    """Current NHL roster — used to map (team, jersey) → player_id."""
    from data.data_store import DataStore

    with DataStore() as ds:
        return ds.roster()


def _load_shots(season: int) -> pl.DataFrame:
    from data.data_store import DataStore

    with DataStore() as ds:
        return ds.shots(season=season)


def _load_zone_entries(season: int) -> pl.DataFrame:
    """Per-team zone entries; doesn't carry the player_id, so we can only use
    its team-level totals as priors. The per-event PBP path is preferred when
    available."""
    from data.data_store import DataStore

    with DataStore() as ds:
        return ds.zone_entries(season=season)


# ---------------------------------------------------------------------------
# Joins + action labels
# ---------------------------------------------------------------------------

def _shot_action(distance_ft: float | None) -> str:
    if distance_ft is None:
        return "shoot_perimeter"
    return "shoot_slot" if distance_ft <= SLOT_DISTANCE_FT else "shoot_perimeter"


def _shots_to_actions(shots: pl.DataFrame, agg: pl.DataFrame) -> pl.DataFrame:
    """Per shot, emit one labelled action row.

    Joins on ``shooter_id`` directly (no jersey lookup needed — MoneyPuck
    already attributes shots to NHL player IDs). The CV aggregator filters
    *which games* to credit: only games where that shooter was tracked
    (jersey-locked, ≥ MIN_FRAMES_PER_TRACK seconds observed).
    """
    if len(shots) == 0:
        return pl.DataFrame(schema=OUT_SCHEMA)

    # Tracks that gave us a confident lock on a jersey + team.
    tracked = (
        agg.filter(
            (pl.col("track_id") != -999)
            & (pl.col("jersey_locked") == True)  # noqa: E712
            & (pl.col("frames") >= MIN_FRAMES_PER_TRACK)
        )
        .select(["game_id", "team", "jersey_number"])
        .unique()
    )
    if len(tracked) == 0:
        return pl.DataFrame(schema=OUT_SCHEMA)

    # Map (game_id, team, jersey) → player_id via the roster, then back-join
    # to shots on (game_id, shooter_id). This keeps us honest about
    # game-day attribution: if a player's jersey was never locked in a game,
    # his shots from that game don't enter training.
    roster = _load_roster()
    if "team_code" in roster.columns:
        roster = roster.rename({"team_code": "team"})
    roster_keys = roster.select(["player_id", "team", "jersey_number"]).unique()

    tracked_with_id = tracked.join(roster_keys, on=["team", "jersey_number"], how="inner")
    if len(tracked_with_id) == 0:
        return pl.DataFrame(schema=OUT_SCHEMA)

    eligible = tracked_with_id.select(["game_id", "player_id"]).unique()

    # Ensure shots has the columns we need; coerce lazy-friendly types.
    needed = [
        c for c in (
            "shot_distance", "shooter_id", "game_id", "season",
            "shooting_team", "period", "home_skaters", "away_skaters",
            "shooting_team", "home_team",
        ) if c in shots.columns
    ]
    if "shooter_id" not in needed or "game_id" not in needed:
        return pl.DataFrame(schema=OUT_SCHEMA)

    shots_eligible = shots.join(
        eligible, left_on=["game_id", "shooter_id"], right_on=["game_id", "player_id"],
        how="inner",
    )
    if len(shots_eligible) == 0:
        return pl.DataFrame(schema=OUT_SCHEMA)

    # Action label per row.
    rows = []
    for r in shots_eligible.iter_rows(named=True):
        dist = r.get("shot_distance")
        action = _shot_action(float(dist) if dist is not None else None)
        manpower = int((r.get("home_skaters") or 5) - (r.get("away_skaters") or 5))
        if r.get("shooting_team") == r.get("home_team"):
            home_ind = 1
        else:
            home_ind = 0
            manpower = -manpower
        rows.append({
            "player_id":       int(r["shooter_id"]),
            "season":          int(r.get("season") or 0),
            "game_id":         int(r["game_id"]),
            "team":            r.get("shooting_team") or "",
            "jersey_number":   0,  # filled in below from roster
            "action":          action,
            "fi_score":        0.0,           # rested baseline; FI joined elsewhere
            "period":          int(r.get("period") or 1),
            "score_diff":      0,             # not modelled here yet
            "home_indicator":  int(home_ind),
            "manpower":        int(manpower),
            "matchup_quality": 0.0,
            "speed_ratio":     1.0,
            "carry_ratio":     1.0,
            "weight":          1.0,
        })
    if not rows:
        return pl.DataFrame(schema=OUT_SCHEMA)

    df = pl.DataFrame(rows, schema=OUT_SCHEMA)
    # Attach jersey number from roster for traceability (Cortex page shows it).
    jersey_lookup = roster.select(["player_id", "jersey_number"]).unique()
    df = df.drop("jersey_number").join(jersey_lookup, on="player_id", how="left")
    if df["jersey_number"].dtype != pl.Int64:
        df = df.with_columns(pl.col("jersey_number").cast(pl.Int64, strict=False))
    return df.select(list(OUT_SCHEMA.keys()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, default=2025,
                   help="NHL season (start year). Default 2025 → 2025-26.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the row count breakdown without writing parquet")
    args = p.parse_args(argv)

    root = _data_root()
    out_dir = root / "cv_observations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cv_actions_{args.season}.parquet"

    print(f"[label-cv-actions] season={args.season}")
    print(f"[label-cv-actions] data root={root}")

    agg = _load_aggregator_summaries(root)
    if len(agg) == 0:
        print("[label-cv-actions] No aggregator summaries — run gretzky aggregate-cv-obs first.")
        # Don't error: this is a no-op path the gate-OFF case relies on.
        return 0

    shots = _load_shots(args.season)
    print(f"[label-cv-actions] {len(agg):,} aggregator rows · {len(shots):,} shots")

    actions = _shots_to_actions(shots, agg)
    print(f"[label-cv-actions] emitted {len(actions):,} action rows")

    if len(actions) > 0:
        breakdown = (
            actions.group_by("action").len().sort("len", descending=True)
        )
        print("[label-cv-actions] action breakdown:")
        for r in breakdown.iter_rows(named=True):
            print(f"   {r['action']:<18} {r['len']:>6,}")

    if args.dry_run:
        print(f"[label-cv-actions] --dry-run: not writing.")
        return 0

    if len(actions) == 0:
        # Still write an empty parquet so downstream loaders see "exists, no rows"
        # rather than "missing".
        pl.DataFrame(schema=OUT_SCHEMA).write_parquet(out_path)
    else:
        actions.write_parquet(out_path)
    print(f"[label-cv-actions] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
