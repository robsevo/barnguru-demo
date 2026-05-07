"""Memory-profile harness — runs N sequential resolves and asserts RSS
stays under a budget. Catches resource leaks before they hit production.

Usage:
    TMDB_API_KEY=... uv run python scripts/resolver_memprof.py
    TMDB_API_KEY=... RESOLVER_MEMPROF_N=20 uv run python scripts/resolver_memprof.py

Env:
    RESOLVER_MEMPROF_N           — number of resolves (default 20)
    RESOLVER_MEMPROF_RSS_MB_MAX  — assert RSS never exceeds this (default 600)

Exit codes:
    0  - within budget
    1  - exceeded RSS budget
    2  - resolver init failed
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.api.stream_resolver import init_resolver, shutdown_resolver  # noqa: E402

# Cycle through a varied list so we don't just hit cache.
CANARY_TMDB_IDS = [27205, 155, 603, 245891, 496243, 872585, 693134, 597, 24428, 122]


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback: parse /proc/self/status
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        return kb / 1024
        except Exception:
            pass
        return 0.0


async def main() -> int:
    if not os.environ.get("TMDB_API_KEY"):
        print("ERROR: TMDB_API_KEY required", file=sys.stderr)
        return 2

    n_resolves = int(os.environ.get("RESOLVER_MEMPROF_N", "20"))
    rss_max_mb = float(os.environ.get("RESOLVER_MEMPROF_RSS_MB_MAX", "600"))

    try:
        resolver = await init_resolver()
    except Exception as e:
        print(f"init failed: {e}", file=sys.stderr)
        return 2

    rss_baseline = _rss_mb()
    rss_peak = rss_baseline
    print(f"baseline RSS: {rss_baseline:.0f} MB")
    print(f"running {n_resolves} resolves...")

    for i in range(n_resolves):
        tmdb = CANARY_TMDB_IDS[i % len(CANARY_TMDB_IDS)]
        try:
            await resolver.resolve_movie(tmdb, budget_s=15.0)
        except Exception as e:
            print(f"  iter {i}: resolve raised {type(e).__name__}: {e}")
        rss = _rss_mb()
        rss_peak = max(rss_peak, rss)
        if i % 5 == 0:
            print(f"  iter {i}: RSS={rss:.0f} MB peak={rss_peak:.0f} MB")
        if rss > rss_max_mb:
            print(f"FAIL: RSS={rss:.0f} MB exceeded budget {rss_max_mb:.0f} MB at iter {i}")
            await shutdown_resolver()
            return 1

    print()
    print(f"=== peak RSS: {rss_peak:.0f} MB (budget {rss_max_mb:.0f} MB) ===")
    print(f"=== delta from baseline: {rss_peak - rss_baseline:.0f} MB ===")

    await shutdown_resolver()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
