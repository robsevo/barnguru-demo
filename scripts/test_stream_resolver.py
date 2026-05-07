"""Live verification harness for the stream resolver.

Runs the resolver against a fixed set of well-known TMDB ids — 5 movies
and 5 series episodes. For each: calls resolve_*, fetches the returned
m3u8 manifest, asserts it parses as a real HLS playlist with at least
one segment line.

Pass criterion: ≥80% (8/10) resolve through ANY enabled provider, manifest
fetches successfully.

Usage:
    TMDB_API_KEY=... uv run python scripts/test_stream_resolver.py
    TMDB_API_KEY=... STREAM_RESOLVER_ENABLED=1 uv run python scripts/test_stream_resolver.py

Exit codes:
    0  - pass criterion met
    1  - some titles failed but resolver ran
    2  - resolver itself failed to start
    3  - no providers enabled
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import httpx

# Make the package importable when run from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.api.stream_resolver import init_resolver, shutdown_resolver  # noqa: E402


CANARY_MOVIES = [
    ("Inception",        27205),
    ("The Dark Knight",  155),
    ("Parasite",         496243),
    ("John Wick",        245891),
    ("The Matrix",       603),
]

CANARY_EPISODES = [
    ("Severance S01E01",          95396, 1, 1),
    ("The Bear S01E01",           136315, 1, 1),
    ("Succession S01E01",         75003, 1, 1),
    ("House of the Dragon S01E01", 94997, 1, 1),
    ("Andor S01E01",              83867, 1, 1),
]


def _unwrap_proxy(url: str) -> str:
    """The resolver wraps URLs as `/lounge/vod-stream-proxy?url=<encoded>`.
    For testing we want the raw upstream URL so we can fetch it directly.
    Falls back to the original URL if not wrapped."""
    if "/lounge/vod-stream-proxy?" not in url:
        return url
    qs = parse_qs(urlparse(url).query)
    return qs.get("url", [url])[0]


async def _fetch_manifest(url: str, referer: str | None) -> tuple[bool, str]:
    """Fetch the m3u8 and return (is_valid_manifest, reason)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0.0.0",
    }
    if referer:
        headers["Referer"] = referer
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as cli:
            r = await cli.get(url, headers=headers)
        if r.status_code != 200:
            return False, f"http_{r.status_code}"
        body = r.text
        if "#EXTM3U" not in body:
            return False, "no_extm3u_header"
        if "#EXTINF" not in body and "#EXT-X-STREAM-INF" not in body:
            return False, "no_segments_or_variants"
        return True, "ok"
    except Exception as e:
        return False, f"fetch_err:{type(e).__name__}"


async def _test_one(resolver, label: str, args: tuple, *, kind: str) -> tuple[bool, str]:
    t0 = time.monotonic()
    try:
        if kind == "movie":
            res = await resolver.resolve_movie(args[0], budget_s=20.0)
        else:
            res = await resolver.resolve_episode(args[0], args[1], args[2], budget_s=20.0)
    except Exception as e:
        return False, f"resolve_err:{type(e).__name__}:{e}"
    elapsed = time.monotonic() - t0
    if not res.stream_urls:
        tried = ",".join(a.provider_id for a in res.providers_tried) or "none"
        return False, f"no_streams ({elapsed:.1f}s, tried={tried})"
    raw = _unwrap_proxy(res.stream_urls[0])
    referer = None
    if "&referer=" in res.stream_urls[0]:
        referer = parse_qs(urlparse(res.stream_urls[0]).query).get("referer", [""])[0]
    ok, reason = await _fetch_manifest(raw, referer)
    if not ok:
        return False, f"manifest_failed:{reason} url={raw[:80]}"
    return True, f"ok ({elapsed:.1f}s, host={urlparse(raw).hostname})"


async def main() -> int:
    if not os.environ.get("TMDB_API_KEY"):
        print("ERROR: TMDB_API_KEY environment variable required", file=sys.stderr)
        return 2

    try:
        resolver = await init_resolver()
    except Exception as e:
        print(f"resolver init failed: {e}", file=sys.stderr)
        return 2

    if not resolver.providers:
        print("no providers enabled — patterns.json all _enabled:false", file=sys.stderr)
        await shutdown_resolver()
        return 3

    print(f"=== test_stream_resolver — {len(resolver.providers)} provider(s): "
          f"{list(resolver.providers.keys())} ===")
    print()

    results: list[tuple[str, bool, str]] = []
    print("-- movies --")
    for label, tmdb_id in CANARY_MOVIES:
        ok, reason = await _test_one(resolver, label, (tmdb_id,), kind="movie")
        results.append((label, ok, reason))
        mark = "✓" if ok else "✗"
        print(f"  {mark} {label}: {reason}")

    print()
    print("-- series --")
    for label, tmdb_id, season, episode in CANARY_EPISODES:
        ok, reason = await _test_one(
            resolver, label, (tmdb_id, season, episode), kind="series",
        )
        results.append((label, ok, reason))
        mark = "✓" if ok else "✗"
        print(f"  {mark} {label}: {reason}")

    wins = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    rate = wins / total if total else 0.0
    print()
    print(f"=== {wins}/{total} resolved ({rate:.0%}) ===")

    # Show health snapshot
    print()
    print("-- provider health --")
    for pid, h in resolver.health.all().items():
        print(f"  {pid}: success_rate={h.success_rate:.2f} "
              f"({h.rolling_successes}/{h.rolling_successes + h.rolling_failures}) "
              f"silent_empty={h.consecutive_silent_empty}")

    await shutdown_resolver()
    return 0 if rate >= 0.80 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
