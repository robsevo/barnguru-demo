"""
scripts/check_iptv.py
Test each URL in iptv_paid_channels.json and write iptv_status.json
(server-side only — not committed to git).

Prints a full report with every channel name, URL, and working/broken status.
Run: uv run python scripts/gretzky.py check-iptv
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Both files are GENERATED ON THE PROD BOX (extract-iptv / this script) and are
# gitignored, so they never exist in a fresh CI checkout — resolving them relative to
# this script made the nightly's check-iptv step fail every run ("No channel file
# found at /home/user/…"), and left the /iptv-channel-status endpoint serving stale
# data. Overridable so the workflow can point at the deployed copies.
_CHANNELS_PATH = Path(
    os.environ.get("IPTV_PAID_CHANNELS_FILE")
    or Path(__file__).parent.parent / "dashboard" / "api" / "iptv_paid_channels.json"
)
_STATUS_PATH = Path(
    os.environ.get("IPTV_STATUS_FILE")
    or Path(__file__).parent.parent / "dashboard" / "api" / "iptv_status.json"
)

_TIMEOUT = 8.0
_CONCURRENCY = 10  # parallel checks


async def _check_url(
    client: httpx.AsyncClient, url: str
) -> tuple[bool, int]:
    """Return (alive, http_status). -1 = timeout, 0 = connection error."""
    try:
        if url.lower().split("?")[0].endswith(".m3u8"):
            r = await client.head(url)
        else:
            r = await client.get(url)
        return r.status_code < 400, r.status_code
    except httpx.TimeoutException:
        return False, -1
    except Exception:
        return False, 0


async def _check_all(channels: list[dict]) -> list[dict]:
    sem = asyncio.Semaphore(_CONCURRENCY)
    checked_at = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []

    async def _worker(ch: dict) -> dict:
        async with sem:
            alive, status = await _check_url(client, ch["url"])
        return {
            "title": ch.get("title", ""),
            "url": ch.get("url", ""),
            "account": ch.get("account", ""),
            "working": alive,
            "http_status": status,
            "last_checked": checked_at,
        }

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=_TIMEOUT,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    ) as client:
        results = list(await asyncio.gather(*[_worker(ch) for ch in channels]))

    return results


def _status_label(http_status: int) -> str:
    if http_status == -1:
        return "timeout"
    if http_status == 0:
        return "conn error"
    return str(http_status)


def main() -> None:
    if not _CHANNELS_PATH.exists():
        print(f"No channel file found at {_CHANNELS_PATH}")
        print("Run 'uv run python scripts/gretzky.py extract-iptv' first.")
        sys.exit(1)

    channels: list[dict] = json.loads(_CHANNELS_PATH.read_text())
    if not channels:
        print("Channel file is empty — nothing to check.")
        sys.exit(1)

    print(f"Testing {len(channels)} channels (up to {_CONCURRENCY} in parallel)...\n")
    results = asyncio.run(_check_all(channels))

    now = datetime.now(timezone.utc).isoformat()
    working  = [r for r in results if r["working"]]
    broken   = [r for r in results if not r["working"]]

    # Write status JSON
    status_doc = {
        "checked_at": now,
        "summary": {
            "total":   len(results),
            "working": len(working),
            "broken":  len(broken),
        },
        "channels": results,
    }
    _STATUS_PATH.write_text(json.dumps(status_doc, indent=2))

    # ── Human-readable report ─────────────────────────────────────────────────
    print(f"IPTV Channel Status — {now[:10]}")
    print(f"Working: {len(working)} / {len(results)}\n")

    by_account: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_account[r["account"]].append(r)

    for account in sorted(by_account):
        chs = sorted(by_account[account], key=lambda x: x["title"])
        print(f"{account}:")
        for ch in chs:
            mark = "\u2713" if ch["working"] else "\u2717"
            extra = "" if ch["working"] else f"  [{_status_label(ch['http_status'])}]"
            print(f"  {mark} {ch['title']:<35} {ch['url']}{extra}")
        print()

    if broken:
        print(f"BROKEN ({len(broken)}):")
        for ch in broken:
            print(
                f"  \u2717 {ch['title']} ({ch['account']}) "
                f"— {_status_label(ch['http_status'])}"
            )
        print()

    print(f"Status written to {_STATUS_PATH}")


if __name__ == "__main__":
    main()
