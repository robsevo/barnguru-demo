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
    arena_counts: dict[str, int] = {}
    arena_first: str | None = None
    events_total = 0
    pass_shot_total = 0
    bundles_total = 0

    for fp in ndjsons:
        for rec in _iter_ndjson(fp):
            bundles_total += 1
            client_id = rec.get("client_id", "unknown")
            scene     = rec.get("scene", "unknown")
            arena     = rec.get("arena") or None
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
            if arena:
                arena_counts[arena] = arena_counts.get(arena, 0) + 1
                if arena_first is None:
                    arena_first = arena

            # Per-track event attribution. Pass/shot events carry from_track_id;
            # we credit the originator track so the per-(jersey, team) action
            # counts the labeler emits downstream are actually sourced.
            per_track_events: dict[int, int] = {}
            for ev in rec.get("events_recent", []) or []:
                events_total += 1
                if ev.get("kind") == "pass_or_shot":
                    pass_shot_total += 1
                    src = ev.get("from_track_id")
                    if isinstance(src, int) and src >= 0:
                        per_track_events[src] = per_track_events.get(src, 0) + 1

            vt = rec.get("video_time")
            for p in rec.get("players", []) or []:
                tid = p.get("track_id")
                if tid is None or tid < 0:
                    continue  # -1 is puck pseudo-entry

                key = (client_id, int(tid))
                sp_kmh = p.get("speed_kmh")
                cls    = p.get("class_name", "")
                cx, cy = p.get("cx_n"), p.get("cy_n")
                jersey = p.get("jersey_number")
                jersey_locked = bool(p.get("jersey_locked", False))
                team   = p.get("team", "") or ""

                entry = tracks.get(key)
                if entry is None:
                    entry = {
                        "client_id":     client_id,
                        "track_id":      int(tid),
                        "class_votes":   {},
                        "frames":        0,
                        "sum_speed":     0.0,
                        "n_speed":       0,
                        "max_speed":     0.0,
                        "first_time":    vt,
                        "last_time":     vt,
                        "sum_x":         0.0,
                        "sum_y":         0.0,
                        "n_pos":         0,
                        # Jersey lock: take the first locked (jersey_number, team)
                        # we see for this track. Locked status implies ≥3
                        # consistent OCR reads at the frontend, so flipping it is
                        # rare; but we deliberately never overwrite once set so a
                        # flicker can't poison the attribution.
                        "jersey_number": jersey if jersey_locked else None,
                        "jersey_locked": jersey_locked,
                        "team":          team,
                        "pass_or_shot":  0,
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
                # Jersey lock — first definitive lock wins.
                if (not entry["jersey_locked"]) and jersey_locked and jersey is not None:
                    entry["jersey_number"] = int(jersey)
                    entry["jersey_locked"] = True
                # Team attribution: keep the dominant non-empty value.
                if team and not entry["team"]:
                    entry["team"] = team
                # Credit pass/shot events whose from_track_id matches this track.
                if int(tid) in per_track_events:
                    entry["pass_or_shot"] += per_track_events[int(tid)]

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
            "jersey_number":     e["jersey_number"],
            "jersey_locked":     e["jersey_locked"],
            "team":              e["team"] or None,
            "pass_or_shot":      e["pass_or_shot"],
        })

    # Most-frequent arena wins, since a single game is always in one building.
    dominant_arena = max(arena_counts.items(), key=lambda kv: kv[1])[0] if arena_counts else arena_first

    # Header row captures game-level counts so consumers don't need to re-scan.
    # We emit it as a special row with track_id = -999.
    rows.append({
        "game_id":           int(game_dir.name),
        "client_id":         "__summary__",
        "track_id":          -999,
        "stable_class":      "summary",
        "arena":             dominant_arena,
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

    # Also stamp every track row with the arena so downstream per-arena joins
    # don't need the summary row.
    for r in rows:
        if r["track_id"] != -999 and "arena" not in r:
            r["arena"] = dominant_arena

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
    # Per-arena aggregate accumulator; keys are arena names, values are
    # {games, tracks, bundles, scene_game, scene_break, events, pass_or_shot}.
    per_arena: dict[str, dict] = {}

    for gd in game_dirs:
        rows = _reduce_game(gd)
        # Exclude the game-level summary row from the "tracks" count.
        track_rows   = [r for r in rows if r["track_id"] != -999]
        summary_rows = [r for r in rows if r["track_id"] == -999]
        if not track_rows:
            continue
        total_games  += 1
        total_tracks += len(track_rows)

        # Accumulate per-arena totals from the summary row (one per game).
        if summary_rows:
            s = summary_rows[0]
            arena = s.get("arena") or "__unknown__"
            entry = per_arena.setdefault(arena, {
                "arena": arena, "games": 0, "tracks": 0, "bundles": 0,
                "scene_game": 0, "scene_break": 0, "events": 0, "pass_or_shot": 0,
            })
            entry["games"]        += 1
            entry["tracks"]       += len(track_rows)
            entry["bundles"]      += s.get("frames", 0) or 0
            entry["scene_game"]   += s.get("_scene_game", 0) or 0
            entry["scene_break"]  += s.get("_scene_break", 0) or 0
            entry["events"]       += s.get("_events", 0) or 0
            entry["pass_or_shot"] += s.get("_pass_or_shot", 0) or 0

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

    # Global per-arena roll-up — answers "how much training signal do we have
    # per building?" Written once at the root, not per-game.
    if per_arena and not args.dry_run:
        arena_rows = sorted(per_arena.values(), key=lambda e: -e["bundles"])
        arena_out  = root / "cv_live_observations" / "summary_by_arena.parquet"
        try:
            import polars as pl
            pl.DataFrame(arena_rows).write_parquet(arena_out)
            print(f"wrote {arena_out} — {len(arena_rows)} arenas")
        except ImportError:
            import csv
            arena_out = arena_out.with_suffix(".csv")
            keys = sorted({k for r in arena_rows for k in r.keys()})
            with arena_out.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in arena_rows:
                    w.writerow(r)
            print(f"wrote {arena_out} — {len(arena_rows)} arenas")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
