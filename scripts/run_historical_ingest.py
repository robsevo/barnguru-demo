"""Run the NHL historical ingestor — downloads PBP + shift data for one or more seasons.

Uses the free NHL Stats API (no auth required). Each game's play-by-play and
shift chart are fetched concurrently (default concurrency=5) and written to
Parquet. A manifest tracks completed games so re-runs skip already-ingested games.

Outputs (written to GRETZKY_DATA_DIR/raw/):
  shifts_{season}.parquet      ← required by train-rapm
  pbp_{season}.parquet
  toi_{season}.parquet
  player_stats_{season}.parquet

Run before training RAPM:
  uv run python scripts/gretzky.py ingest
  uv run python scripts/gretzky.py train-rapm

Usage:
  uv run python scripts/run_historical_ingest.py
  uv run python scripts/run_historical_ingest.py --seasons 2023 2024 2025
  uv run python scripts/run_historical_ingest.py --season 2025 --force
  uv run python scripts/run_historical_ingest.py --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))


def _data_dir() -> Path:
    default = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
    return Path(os.environ.get("GRETZKY_DATA_DIR", str(default)))


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


async def _ingest_season(
    base: Path,
    season: int,
    *,
    concurrency: int,
    force: bool,
) -> None:
    from data.historical_ingestor import HistoricalIngestor
    from data.nhl_client import NHLClient

    ingestor = HistoricalIngestor(base / "raw", concurrency=concurrency)

    if force:
        cleared = ingestor.clear_errors()
        if cleared:
            print(f"  Cleared {cleared} error entries from manifest (--force)")

    print(f"  Fetching game IDs for season {season}...")
    async with NHLClient() as client:
        summary = await ingestor.ingest_season(client, season)

    if summary.skipped == summary.total_games and summary.total_games > 0:
        status = "cached"
    elif summary.successful > 0:
        status = "ok"
    else:
        status = "empty"
    icon = {"ok": "✓", "cached": "●", "empty": "○"}.get(status, "✗")
    print(
        f"  {icon} season {season}  [{status}]"
        f"  games {summary.successful}/{summary.total_games} ok"
        f"  skipped {summary.skipped}"
        f"  failed {summary.failed}"
        f"  shifts {summary.total_shifts:,}"
        f"  plays {summary.total_plays:,}"
    )
    if summary.failed > 0:
        print(f"    {summary.failed} games failed — re-run to retry (manifest tracks state)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest NHL historical PBP + shift data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        help="Season start-years to ingest (default: current NHL season only)",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        metavar="YEAR",
        help="Single season shorthand (overridden by --seasons if both given)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        metavar="N",
        help="Number of concurrent game fetches",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear error entries from manifest so they are re-fetched",
    )
    args = parser.parse_args()

    base = _data_dir()
    base.mkdir(parents=True, exist_ok=True)

    if args.seasons:
        seasons = sorted(args.seasons)
    elif args.season:
        seasons = [args.season]
    else:
        # Default to 3-season window starting no earlier than 2023 (Jan 1, 2024 cutoff)
        cur = _current_nhl_season()
        seasons = list(range(max(cur - 2, 2023), cur + 1))

    print(f"Historical ingest — seasons {seasons}  (GRETZKY_DATA_DIR={base})")
    print(f"  Output dir : {base / 'raw'}")
    print(f"  Concurrency: {args.concurrency}")
    print()

    for season in seasons:
        print(f"[ Season {season} ]")
        try:
            asyncio.run(
                _ingest_season(
                    base,
                    season,
                    concurrency=args.concurrency,
                    force=args.force,
                )
            )
        except Exception as exc:
            print(f"  ✗ season {season} failed: {exc}", file=sys.stderr)

    print()
    print("Done. Run 'gretzky train-rapm' to train the RAPM model.")


if __name__ == "__main__":
    main()
