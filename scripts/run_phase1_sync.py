"""Run all Phase 1 syncs that have live free-API support.

Pulls real data from ESPN and NHL Stats API and persists to ~/.gretzky/data/.
No fake data. No seeds. Real API calls or nothing.

Modules synced:
  injuries          ← ESPN injury feed          (free, no auth)
  transactions      ← ESPN transactions API     (free, no auth)
  edge              ← NHL Stats API (async)     (free, no auth)
  goalie_stats      ← MoneyPuck goalie CSV      (free, no auth)
  historical_ingest ← NHL Stats API PBP+shifts  (free, no auth) — slow, ~1300 games/season

Modules NOT synced here:
  evolving_hockey  → paywall (site moved RAPM/zone-entry behind subscription)
  morning_skate    → will show not_run
  press_conference → will show not_run

Usage:
  uv run python scripts/run_phase1_sync.py
  uv run python scripts/run_phase1_sync.py --date 2026-03-21 --season 2025
  uv run python scripts/run_phase1_sync.py --days 30          # backfill 30 days of transactions
  uv run python scripts/run_phase1_sync.py --force
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make repo root importable
_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))


def _data_dir() -> Path:
    default = Path.home() / ".gretzky" / "data"
    return Path(os.environ.get("GRETZKY_DATA_DIR", str(default)))


def _current_nhl_season() -> int:
    """Current NHL season start-year. NHL runs Oct–Sep: March 2026 → 2025 (2025-26)."""
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _print_result(module: str, status: str, count: int, extra: str = "") -> None:
    icon = {"ok": "✓", "skipped": "–", "empty": "○"}.get(status, "✗")
    count_str = f"{count} records" if count else "0 records"
    msg = f"  {icon} {module:20s} [{status:12s}] {count_str}"
    if extra:
        msg += f"  ({extra})"
    print(msg)


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------


def sync_injuries(base: Path, *, force: bool = False) -> None:
    from data.injury_client import InjuryClient
    from data.injury_sync import InjurySync

    sync = InjurySync(base / "injuries")
    with InjuryClient() as client:
        result = sync.sync(client)
    _print_result("injuries", result.status, result.record_count)


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


def sync_transactions(base: Path, date: str, *, days: int = 7, force: bool = False) -> None:
    """Sync transactions for the last `days` days."""
    from datetime import date as _date, timedelta

    from data.transaction_client import TransactionClient
    from data.transaction_parser import TransactionParser
    from data.transaction_sync import TransactionSyncer

    sync = TransactionSyncer(base / "transactions")
    end = _date.fromisoformat(date)
    start = end - timedelta(days=days - 1)

    total = 0
    statuses = []
    with TransactionClient() as client:
        current = start
        while current <= end:
            d = current.isoformat()
            try:
                result = sync.sync(client, TransactionParser, d, force=force)
                if result.status == "skipped":
                    # skipped result always has event_count=0; read stored count from manifest
                    stored = sync.get_manifest().get("dates", {}).get(d, {})
                    total += stored.get("event_count", 0)
                else:
                    total += result.event_count
                statuses.append(result.status)
            except Exception:
                statuses.append("error")
            current += timedelta(days=1)

    final_status = "ok" if total > 0 else ("empty" if all(s == "empty" for s in statuses) else "skipped")
    _print_result("transactions", final_status, total, f"{start} → {date}")


# ---------------------------------------------------------------------------
# Goalie stats (MoneyPuck)
# ---------------------------------------------------------------------------


def sync_goalie_stats(base: Path, season: int, *, force: bool = False) -> None:
    from data.moneypuck_client import MoneyPuckClient
    from data.moneypuck_goalie_sync import GoalieStatsSyncer

    syncer = GoalieStatsSyncer(base / "goalie_stats")
    with MoneyPuckClient() as client:
        result = syncer.sync_season(client, season, force=force)
    _print_result(
        "goalie_stats",
        result.status,
        result.goalie_count,
        f"season={season}  rows={result.row_count}",
    )
    if result.error_message:
        print(f"    error: {result.error_message}")


# ---------------------------------------------------------------------------
# EDGE (async)
# ---------------------------------------------------------------------------


async def sync_edge(base: Path, season: int, *, force: bool = False) -> None:
    from data.edge_sync import EdgeSync
    from data.nhl_client import NHLClient

    sync = EdgeSync(base / "edge")
    async with NHLClient() as client:
        result = await sync.sync_season(client, season, force=force)
    total = result.skating_count + result.shot_count
    _print_result(
        "edge",
        result.status,
        total,
        f"season={season}  skating={result.skating_count}  shots={result.shot_count}",
    )
    if result.error_message:
        print(f"    error: {result.error_message}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 live syncs against real APIs."
    )
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date for injuries + transactions (YYYY-MM-DD, default: today UTC)",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=_current_nhl_season(),
        help="EDGE season start-year (default: current NHL season)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of transactions to sync (default: 7, use 30 to backfill)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass TTL / skip-if-ok and force re-fetch",
    )
    args = parser.parse_args()

    base = _data_dir()
    base.mkdir(parents=True, exist_ok=True)

    print(f"Phase 1 sync — {args.date}  (GRETZKY_DATA_DIR={base})")
    print()

    # --- Injuries ---
    try:
        sync_injuries(base, force=args.force)
    except Exception as exc:
        print(f"  ✗ injuries              [error       ] {exc}")

    # --- Transactions ---
    try:
        sync_transactions(base, args.date, days=args.days, force=args.force)
    except Exception as exc:
        print(f"  ✗ transactions          [error       ] {exc}")

    # --- Goalie stats ---
    try:
        sync_goalie_stats(base, args.season, force=args.force)
    except Exception as exc:
        print(f"  ✗ goalie_stats          [error       ] {exc}")

    # --- EDGE (async) ---
    try:
        asyncio.run(sync_edge(base, args.season, force=args.force))
    except Exception as exc:
        print(f"  ✗ edge                  [error       ] {exc}")

    # --- Evolving Hockey — PAYWALL: site moved behind subscription ---
    print(f"  · {'evolving_hockey':20s} [not_run     ] paywall — use internal RAPM (gretzky train-rapm)")

    # --- Historical ingest (PBP + shifts — required for RAPM) ---
    try:
        from scripts.run_historical_ingest import _ingest_season as _hist
        print(f"  · historical_ingest    [running...  ] season={args.season} — this may take 10–30 min")
        asyncio.run(_hist(base, args.season, concurrency=5, force=args.force))
    except Exception as exc:
        print(f"  ✗ historical_ingest     [error       ] {exc}")

    # --- Not yet built ---
    print(f"  · {'morning_skate':20s} [not_run     ] requires LLM fetcher (not yet built)")
    print(f"  · {'press_conference':20s} [not_run     ] requires LLM fetcher (not yet built)")

    print()
    print("Done. Refresh http://localhost:3000/phase1 to see real data.")


if __name__ == "__main__":
    main()
