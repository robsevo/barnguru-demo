"""Phase 16 bridge — reduce in-browser CV observation NDJSON into per-track summaries.

The frontend (`useCvLiveReplay` + `usePostCvObservations`) POSTs batched
observation bundles to `POST /api/cv/live/observations`, which appends them
as NDJSON at ``$GRETZKY_DATA_DIR/cv_live_observations/{game_id}/{YYYY-MM-DD}.ndjson``.

Each bundle captures ~1 s of tracked state: scene kind, player count,
per-track positions + velocities, and detected events. This script consumes
that raw log and produces per-track per-game summaries (skating time, mean
speed, max speed, pass/shot events authored) that a later training cycle can
fold into ``models/player_behavior_net.py``.

We do NOT join to NHL player IDs here — jersey OCR accuracy on live broadcast
crops is not yet proven, so summaries remain at the track_id level. When
jersey OCR is trusted, downstream consumers can map track_id → player_id.

Output: ``$GRETZKY_DATA_DIR/cv_live_observations/{game_id}/summary.parquet``
(or CSV fallback if polars missing). Safe to re-run — it overwrites.

Usage::

    uv run python scripts/gretzky.py aggregate-cv-obs
    uv run python scripts/gretzky.py aggregate-cv-obs -- --game-id 2025020456
    uv run python scripts/gretzky.py aggregate-cv-obs -- --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _data_root() -> Path:
    default = Path.home() / ".gretzky" / "data"
    return Path(os.environ.get("GRETZKY_DATA_DIR", str(default)))


def _game_dirs(root: Path, only_id: int | None) -> list[Path]:
    base = root / "cv_live_observations"
    if not base.exists():
        return []
    if only_id is not None:
        cand = base / str(only_id)
        return [cand] if cand.exists() else []
    return [p for p in base.iterdir() if p.is_dir() and p.name.isdigit()]


def _iter_ndjson(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _reduce_game(game_dir: Path) -> list[dict]:
    """Collapse every NDJSON bundle under *game_dir* into per-track rows.

    We pool all clients and dates for a given game_id. Each track_id is scoped
    to one client session; tracks from different clients are treated as
    independent (intentional — they saw different segments of the video).
    """
    ndjsons = sorted(game_dir.glob("*.ndjson"))
    if not ndjsons:
        return []

    # (client_id, track_id) → running summary
    tracks: dict[tuple[str, int], dict] = {}
    scene_counts: dict[str, int] = {"game": 0, "break": 0, "unknown": 0}
    events_total = 0
    pass_shot_total = 0
    bundles_total = 0

    for fp in ndjsons:
        for rec in _iter_ndjson(fp):
            bundles_total += 1
            client_id = rec.get("client_id", "unknown")
            scene     = rec.get("scene", "unknown")
            scene_counts[scene] = scene_counts.get(scene, 0) + 1

            for ev in rec.get("events_recent", []) or []:
                events_total += 1
                if ev.get("kind") == "pass_or_shot":
                    pass_shot_total += 1

            vt = rec.get("video_time")
            for p in rec.get("players", []) or []:
                tid = p.get("track_id")
                if tid is None or tid < 0:
                    continue  # -1 is puck pseudo-entry

                key = (client_id, int(tid))
                sp_kmh = p.get("speed_kmh")
                cls    = p.get("class_name", "")
                cx, cy = p.get("cx_n"), p.get("cy_n")

                entry = tracks.get(key)
                if entry is None:
                    entry = {
                        "client_id":   client_id,
                        "track_id":    int(tid),
                        "class_votes": {},
                        "frames":      0,
                        "sum_speed":   0.0,
                        "n_speed":     0,
                        "max_speed":   0.0,
                        "first_time":  vt,
                        "last_time":   vt,
                        "sum_x":       0.0,
                        "sum_y":       0.0,
                        "n_pos":       0,
                    }
                    tracks[key] = entry

                entry["frames"] += 1
                entry["class_votes"][cls] = entry["class_votes"].get(cls, 0) + 1
                if isinstance(sp_kmh, (int, float)) and math.isfinite(sp_kmh):
                    entry["sum_speed"] += float(sp_kmh)
                    entry["n_speed"]   += 1
                    if sp_kmh > entry["max_speed"]:
                        entry["max_speed"] = float(sp_kmh)
                if vt is not None:
                    if entry["first_time"] is None or vt < entry["first_time"]:
                        entry["first_time"] = vt
                    if entry["last_time"] is None or vt > entry["last_time"]:
                        entry["last_time"] = vt
                if isinstance(cx, (int, float)) and isinstance(cy, (int, float)):
                    entry["sum_x"] += float(cx)
                    entry["sum_y"] += float(cy)
                    entry["n_pos"] += 1

    rows: list[dict] = []
    for (client_id, tid), e in tracks.items():
        votes = e["class_votes"]
        stable_cls = max(votes.items(), key=lambda kv: kv[1])[0] if votes else "unknown"
        frames = e["frames"]
        # We don't know the exact inference cadence, but the frontend samples
        # at 1 Hz. So frames ≈ seconds of observed presence.
        seconds_observed = float(frames)
        rows.append({
            "game_id":           int(game_dir.name),
            "client_id":         client_id,
            "track_id":          tid,
            "stable_class":      stable_cls,
            "frames":            frames,
            "seconds_observed":  seconds_observed,
            "mean_speed_kmh":    round(e["sum_speed"] / e["n_speed"], 2) if e["n_speed"] else None,
            "max_speed_kmh":     round(e["max_speed"], 2) if e["max_speed"] > 0 else None,
            "mean_cx":           round(e["sum_x"] / e["n_pos"], 4) if e["n_pos"] else None,
            "mean_cy":           round(e["sum_y"] / e["n_pos"], 4) if e["n_pos"] else None,
            "first_video_time":  e["first_time"],
            "last_video_time":   e["last_time"],
        })

    # Header row captures game-level counts so consumers don't need to re-scan.
    # We emit it as a special row with track_id = -999.
    rows.append({
        "game_id":           int(game_dir.name),
        "client_id":         "__summary__",
        "track_id":          -999,
        "stable_class":      "summary",
        "frames":            bundles_total,
        "seconds_observed":  None,
        "mean_speed_kmh":    None,
        "max_speed_kmh":     None,
        "mean_cx":           None,
        "mean_cy":           None,
        "first_video_time":  None,
        "last_video_time":   None,
        "_scene_game":       scene_counts.get("game", 0),
        "_scene_break":      scene_counts.get("break", 0),
        "_scene_unknown":    scene_counts.get("unknown", 0),
        "_events":           events_total,
        "_pass_or_shot":     pass_shot_total,
    })

    return rows


def _write_summary(game_dir: Path, rows: list[dict]) -> Path:
    out = game_dir / "summary.parquet"
    try:
        import polars as pl
        # Polars needs consistent columns — fill missing keys.
        all_keys: set[str] = set()
        for r in rows:
            all_keys.update(r.keys())
        normalized = [{k: r.get(k) for k in all_keys} for r in rows]
        pl.DataFrame(normalized).write_parquet(out)
        return out
    except ImportError:
        out = game_dir / "summary.csv"
        if rows:
            import csv
            keys = sorted({k for r in rows for k in r.keys()})
            with out.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
        return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", type=int, default=None,
                        help="Only aggregate this game_id (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary to stdout instead of writing parquet")
    args = parser.parse_args(argv)

    root = _data_root()
    game_dirs = _game_dirs(root, args.game_id)
    if not game_dirs:
        print(f"No CV observation data found under {root / 'cv_live_observations'}")
        return 0

    total_games  = 0
    total_tracks = 0
    for gd in game_dirs:
        rows = _reduce_game(gd)
        # Exclude the game-level summary row from the "tracks" count.
        track_rows = [r for r in rows if r["track_id"] != -999]
        if not track_rows:
            continue
        total_games  += 1
        total_tracks += len(track_rows)
        if args.dry_run:
            print(f"\n── game {gd.name} — {len(track_rows)} tracks ──")
            for r in track_rows[:5]:
                print(json.dumps(r, default=str))
            if len(track_rows) > 5:
                print(f"  … and {len(track_rows) - 5} more")
        else:
            out = _write_summary(gd, rows)
            print(f"wrote {out} — {len(track_rows)} tracks")

    print(f"\n{total_games} games, {total_tracks} tracks total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
