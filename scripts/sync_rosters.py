"""sync_rosters — Fetch current NHL rosters for all 32 teams.

Writes ``roster_{TEAM}.json`` files to ``{GRETZKY_DATA_DIR}/raw/`` so
``DataStore.roster()`` can read them. Needed by Phase 3 features
3.12 (age recovery) and 3.16 (roster depth strain).

The script bypasses ``RosterSync.sync_team`` because that method calls
``NHLClient.get_roster()`` → ``/v1/roster/{TEAM}/current``, which the
NHL API now serves as a 307 redirect to the season-keyed URL. The
httpx clients in this codebase don't follow redirects (would affect
many other endpoints), so we call ``get_roster_season(team, season)``
directly with the current season and reuse the normalize-and-cache
helpers from ``data.roster_sync``.

Usage::

    uv run python scripts/gretzky.py sync-rosters
    uv run python scripts/gretzky.py sync-rosters -- --force
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.nhl_client import NHLClient
from data.roster_sync import RosterSync, _normalize_roster


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
ROSTER_SUBDIR = "raw"   # DataStore.roster() reads from {base}/raw/

# Current 32-team list. ARI is omitted (relocated to UTA in 2024-25).
TEAMS: list[str] = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]


def _current_nhl_season() -> int:
    """NHL season start-year. Oct–Sep boundary."""
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


async def _sync_team(
    client: NHLClient,
    team: str,
    season: int,
    cache_dir: Path,
    force: bool,
) -> int | None:
    """Fetch one team's roster + write to cache. Returns player count or None on failure."""
    cache_path = cache_dir / f"roster_{team}.json"
    sync = RosterSync(cache_dir=cache_dir)

    # Honor RosterSync's TTL semantics if not --force: a fresh-enough
    # cache file lets us skip the API call entirely.
    if not force and cache_path.exists():
        cached = sync._load_cache(team)
        if cached is not None:
            return len(cached)

    raw = await client.get_roster_season(team, season)
    profiles = _normalize_roster(raw, team, as_of=date.today())
    # Re-use RosterSync's internal writer for an identical on-disk shape.
    sync._write_cache(team, profiles)
    return len(profiles)


async def _sync_all(data_dir: Path, force: bool) -> tuple[int, int]:
    """Sync rosters for every team. Returns (ok_count, fail_count)."""
    cache_dir = data_dir / ROSTER_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    season = _current_nhl_season()
    print(f"  using season={season} (current NHL season start-year)")

    ok = 0
    failed = 0
    async with NHLClient() as client:
        for team in TEAMS:
            try:
                n = await _sync_team(client, team, season, cache_dir, force)
                if n is None:
                    print(f"  ✗ {team}: roster returned None")
                    failed += 1
                else:
                    print(f"  ✓ {team}: {n:>3} players")
                    ok += 1
            except Exception as e:
                print(f"  ✗ {team}: {e}")
                failed += 1
    return ok, failed


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sync NHL rosters for all 32 teams to "
                    "{GRETZKY_DATA_DIR}/raw/roster_{TEAM}.json."
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--force", action="store_true",
                   help="Bypass the 6-hour TTL and re-fetch every team.")
    args = p.parse_args()

    print(f"[sync-rosters] Writing to {args.data_dir / ROSTER_SUBDIR}/")
    print(f"[sync-rosters] force={args.force}  ({len(TEAMS)} teams)")
    ok, failed = asyncio.run(_sync_all(args.data_dir, args.force))
    print(f"\n[sync-rosters] Done — ok={ok}  failed={failed}")
    if failed and ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
