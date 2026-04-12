#!/usr/bin/env python3
"""Compute Puck Battle / Scrum Proxy ratings (Feature 2.18).

Estimates per-player and per-team board battle win rates from hits, blocked
shots, zone entry carry%, and net-front shot frequency.  The per-team
controlled_entry_prob feeds directly into the Rust game engine as P(controlled
zone entry) per shift.

Requires:
    ~/.gretzky/data/raw/pbp_{season}.parquet     (NHL PBP — hits, blocks)
    ~/.gretzky/data/shots/shots_{season}.parquet  (MoneyPuck shots — net-front)
    Optional: ~/.gretzky/data/edge/edge_{season}.parquet  (EDGE carry-entry%)

Usage::

    uv run python scripts/compute_puck_battles.py
    uv run python scripts/compute_puck_battles.py --seasons 2025
    uv run python scripts/compute_puck_battles.py --force

Outputs:
    ~/.gretzky/data/battles/puck_battle_{season}.parquet  (player-level)
    ~/.gretzky/data/battles/team_battle_{season}.parquet  (team-level)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
import os

import polars as pl

_DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _default_seasons() -> list[int]:
    cur = _current_nhl_season()
    return list(range(2023, cur + 1))


def _load_pbp(data_dir: Path, season: int) -> pl.DataFrame:
    from models.rapm_model import DataMissingWarning

    path = data_dir / "raw" / f"pbp_{season}.parquet"
    if not path.exists():
        warnings.warn(
            f"Season {season}: PBP file not found at {path}. "
            "Run: uv run python scripts/gretzky.py ingest",
            DataMissingWarning, stacklevel=2,
        )
        return pl.DataFrame()
    return pl.read_parquet(path)


def _load_shots(data_dir: Path, season: int) -> pl.DataFrame:
    from models.rapm_model import DataMissingWarning

    path = data_dir / "shots" / f"shots_{season}.parquet"
    if not path.exists():
        warnings.warn(
            f"Season {season}: shots file not found at {path}.",
            DataMissingWarning, stacklevel=2,
        )
        return pl.DataFrame()
    return pl.read_parquet(path)


def _load_edge(data_dir: Path, season: int) -> pl.DataFrame | None:
    path = data_dir / "edge" / f"edge_{season}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _build_player_stats_from_pbp(pbp_df: pl.DataFrame, season: int) -> pl.DataFrame:
    """Aggregate per-player hits and blocked shots from NHL PBP data.

    Uses dedicated hitter_id / blocker_id columns written by the updated PBP
    parser (pbp_parser.py >= 2026-04). Falls back gracefully when those columns
    are absent (old parquets).

    Returns DataFrame with: player_id, team, hits, blocked_shots, toi_ev, season.
    """
    if len(pbp_df) == 0:
        return pl.DataFrame()

    event_col = next((c for c in ["event_type", "event"] if c in pbp_df.columns), None)
    frames: list[pl.DataFrame] = []

    # ── Hits via dedicated hitter_id column (new parser) ──────────────────
    if "hitter_id" in pbp_df.columns and event_col:
        hit_events = pbp_df.filter(
            pl.col(event_col).str.to_lowercase() == "hit"
        ).filter(pl.col("hitter_id").is_not_null())
        if len(hit_events) > 0:
            hits_agg = (
                hit_events
                .group_by("hitter_id")
                .agg(pl.len().alias("hits"))
                .rename({"hitter_id": "player_id"})
                .with_columns(pl.col("player_id").cast(pl.Int64))
            )
            frames.append(hits_agg)

    # ── Blocked shots via dedicated blocker_id column (new parser) ─────────
    if "blocker_id" in pbp_df.columns and event_col:
        block_events = pbp_df.filter(
            pl.col(event_col).str.to_lowercase() == "shot"
        ).filter(pl.col("blocker_id").is_not_null())
        if len(block_events) > 0:
            blocks_agg = (
                block_events
                .group_by("blocker_id")
                .agg(pl.len().alias("blocked_shots"))
                .rename({"blocker_id": "player_id"})
                .with_columns(pl.col("player_id").cast(pl.Int64))
            )
            frames.append(blocks_agg)

    if not frames:
        print("    [warn] No hitter_id/blocker_id columns found — PBP may need re-ingest.")
        return pl.DataFrame()

    # Join hits and blocks
    result = frames[0]
    for f in frames[1:]:
        result = result.join(f, on="player_id", how="outer_coalesce")

    for col in ["hits", "blocked_shots"]:
        if col not in result.columns:
            result = result.with_columns(pl.lit(0.0).alias(col))
        else:
            result = result.with_columns(pl.col(col).cast(pl.Float64).fill_null(0.0))

    # Default TOI (30 min); no reliable per-player TOI in PBP without shifts join
    result = result.with_columns(pl.lit(1800.0).alias("toi_ev"))

    # Add team — use event_owner_team_id from hit events
    if "hitter_id" in pbp_df.columns and "event_owner_team_id" in pbp_df.columns and event_col:
        hit_team = (
            pbp_df
            .filter(pl.col(event_col).str.to_lowercase() == "hit")
            .filter(pl.col("hitter_id").is_not_null())
            .group_by("hitter_id")
            .agg(pl.col("event_owner_team_id").first())
            .rename({"hitter_id": "player_id", "event_owner_team_id": "_team_id"})
            .with_columns(pl.col("player_id").cast(pl.Int64))
        )
        result = result.join(hit_team, on="player_id", how="left")
        result = result.with_columns(
            pl.col("_team_id").cast(pl.Utf8).fill_null("UNK").alias("team")
        ).drop("_team_id")
    else:
        result = result.with_columns(pl.lit("UNK").alias("team"))

    result = result.with_columns(pl.lit(season).cast(pl.Int64).alias("season"))
    # Drop rows where player_id is null (shouldn't happen but guard)
    result = result.filter(pl.col("player_id").is_not_null())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Puck Battle / Scrum Proxy ratings (Feature 2.18).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_default_seasons(),
        metavar="YEAR",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-compute even if output already exists.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="PATH",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir = data_dir / "battles"
    output_dir.mkdir(parents=True, exist_ok=True)

    from models.puck_battle_model import (
        PuckBattleModel, write_player_battle, write_team_battle,
    )

    model = PuckBattleModel()
    seasons = sorted(args.seasons)
    processed = 0

    for season in seasons:
        player_path = output_dir / f"puck_battle_{season}.parquet"
        team_path   = output_dir / f"team_battle_{season}.parquet"

        if player_path.exists() and not args.force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            continue

        print(f"\n  Season {season}:")

        # Load data
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pbp_df  = _load_pbp(data_dir, season)
            shots_df = _load_shots(data_dir, season)
        for w in caught:
            print(f"    [WARN] {w.message}", file=sys.stderr)

        edge_df = _load_edge(data_dir, season)
        if edge_df is not None:
            print(f"    EDGE data: {len(edge_df)} rows (carry entry% available)")
        else:
            print("    EDGE data: not found (carry_entry_pct will default to NaN)")

        # Build player-level stats from PBP
        player_stats = _build_player_stats_from_pbp(pbp_df, season)
        if len(player_stats) == 0:
            print(f"    No PBP data; skipping season {season}")
            continue

        print(f"    PBP players: {len(player_stats)}, shots: {len(shots_df)}")

        # Compute battle scores
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            player_df = model.fit(
                player_stats_df = player_stats,
                season          = season,
                shots_df        = shots_df if len(shots_df) > 0 else None,
                carry_entry_df  = edge_df,
            )
        for w in caught:
            print(f"    [WARN] {w.message}", file=sys.stderr)

        if len(player_df) == 0:
            print(f"    No battle data produced; skipping season {season}")
            continue

        # Save player ratings
        write_player_battle(player_df, output_dir, season)
        print(f"    Player battle ratings: {len(player_df)} players → {player_path}")

        # Compute and save team ratings
        team_df = model.team_battle_ratings(player_df, season=season)
        write_team_battle(team_df, output_dir, season)
        print(f"    Team battle ratings: {len(team_df)} teams → {team_path}")

        # Print top battlers
        if "battle_score" in player_df.columns:
            top = player_df.sort("battle_score", descending=True).head(5)
            print("    Top battlers by battle_score:")
            for row in top.to_dicts():
                pid   = row.get("player_id", "?")
                score = row.get("battle_score", float("nan"))
                pct   = row.get("battle_percentile", float("nan"))
                cep   = row.get("controlled_entry_prob", float("nan"))
                team  = row.get("team", "UNK")
                cep_s   = f"{cep:.3f}"    if cep   is not None else "N/A"
                pct_s   = f"{pct:.0f}"    if pct   is not None else "N/A"
                score_s = f"{score:+.3f}" if score is not None else "N/A"
                pid_s   = str(pid)        if pid   is not None else "?"
                team_s  = str(team)       if team  is not None else "UNK"
                print(f"      pid={pid_s:<8} team={team_s:<5} score={score_s}  pct={pct_s}  P(ctrl_entry)={cep_s}")

        # Print team controlled entry probs
        if len(team_df) > 0:
            print("    Team controlled entry probs (top 5):")
            top_teams = team_df.sort("controlled_entry_prob", descending=True).head(5)
            for row in top_teams.to_dicts():
                team = row.get("team", "?")
                cep  = row.get("controlled_entry_prob", float("nan"))
                score = row.get("team_battle_score", float("nan"))
                print(f"      {team:<5}  score={score:+.3f}  P(ctrl_entry)={cep:.3f}")

        processed += 1

    print(f"\nDone. {processed}/{len(seasons)} seasons processed.")


if __name__ == "__main__":
    main()
