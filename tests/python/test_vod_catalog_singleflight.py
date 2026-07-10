"""Offline tests for the VOD catalog cold-start hardening in main.py.

Locks in the fix for the evening 504 meltdowns: after an API restart the
VOD catalog cache was cold and every concurrent /lounge/vod/catalog and
/lounge/vod/details request kicked its OWN full TMDB build (no lock, no
persistence) — a build stampede on the 2 GB box.

main.py is too heavy to import in tests, so the helpers under test are
AST-extracted and exec'd against stubbed globals — same pattern as the
box's EPG merge tests.
"""
from __future__ import annotations

import ast
import asyncio
import json
import time as _time
from pathlib import Path

import pytest

_MAIN = Path(__file__).parents[2] / "dashboard" / "api" / "main.py"

_WANTED = {
    "_load_vod_catalog_snapshot_sync",
    "_save_vod_catalog_snapshot_sync",
    "_ensure_vod_catalog_restored",
    "_kick_vod_catalog_build",
    "_vod_warming_response",
}


def _extract_sources() -> list[str]:
    src = _MAIN.read_text()
    tree = ast.parse(src)
    pieces = [
        ast.get_source_segment(src, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in _WANTED
    ]
    assert len(pieces) == len(_WANTED), (
        f"extracted {len(pieces)}/{len(_WANTED)} helpers from main.py"
    )
    return pieces


_SOURCES = _extract_sources()

_BUILD_RESULT = {
    "by_service": {"Netflix": {"movies": [{"t": "x"}], "series": []}},
    "movies_by_id": {},
    "series_by_id": {},
    "metadata_source": "tmdb",
    "fetched_at": 0,
}


@pytest.fixture()
def vod_env(tmp_path: Path) -> dict:
    """Fresh exec'd globals per test: cold cache, tmp snapshot path, and a
    swappable `_build_vod_catalog` stub (rebind g["_build_vod_catalog"])."""
    calls = {"n": 0}

    async def _default_build() -> dict:
        calls["n"] += 1
        await asyncio.sleep(0.2)
        return json.loads(json.dumps(_BUILD_RESULT))

    g: dict = {
        "asyncio": asyncio,
        "json": json,
        "_time": _time,
        "Path": Path,
        "print": print,
        "_GRETZKY_DATA_DIR": tmp_path,
        "_VOD_CATALOG_SNAPSHOT": tmp_path / "lounge_vod_catalog.json",
        "_LOUNGE_VOD_TTL": 24 * 3600,
        "_lounge_vod_cache": {"data": None, "ts": 0.0},
        "_vod_catalog_build_task": None,
        "_build_vod_catalog": _default_build,
        "_build_calls": calls,
    }
    for src in _SOURCES:
        exec(compile(src, "<extracted>", "exec"), g)
    return g


async def test_single_flight_shares_one_build(vod_env: dict) -> None:
    tasks = [vod_env["_kick_vod_catalog_build"]() for _ in range(10)]
    assert len({id(t) for t in tasks}) == 1, "concurrent kicks must share one task"
    await tasks[0]
    assert vod_env["_build_calls"]["n"] == 1
    assert vod_env["_lounge_vod_cache"]["data"] is not None


async def test_snapshot_roundtrip_restores_cold_cache(vod_env: dict) -> None:
    cache = vod_env["_lounge_vod_cache"]
    await vod_env["_kick_vod_catalog_build"]()
    ts_after_build = cache["ts"]
    snap = vod_env["_VOD_CATALOG_SNAPSHOT"]
    assert snap.exists(), "successful build must write the snapshot"
    assert not snap.with_suffix(".json.tmp").exists(), "tmp file left behind"

    cache["data"], cache["ts"] = None, 0.0  # simulate a restart
    await vod_env["_ensure_vod_catalog_restored"]()
    assert cache["data"] is not None
    assert abs(cache["ts"] - ts_after_build) < 1


async def test_empty_build_keeps_last_good_and_retries_soon(vod_env: dict) -> None:
    cache = vod_env["_lounge_vod_cache"]
    await vod_env["_kick_vod_catalog_build"]()
    good = cache["data"]

    async def _empty_build() -> dict:
        return {"by_service": {"Netflix": {"movies": [], "series": []}}}

    vod_env["_build_vod_catalog"] = _empty_build
    cache["ts"] = 0.0  # force stale
    await vod_env["_kick_vod_catalog_build"]()
    assert cache["data"] is good, "empty build must not clobber a good catalog"
    retry_in = cache["ts"] - (_time.time() - vod_env["_LOUNGE_VOD_TTL"])
    assert 500 < retry_in < 700, f"retry window {retry_in:.0f}s, want ~600"


async def test_waiter_timeout_does_not_cancel_shared_build(vod_env: dict) -> None:
    task = vod_env["_kick_vod_catalog_build"]()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.01)
    assert not task.cancelled(), "waiter timeout cancelled the shared build"
    await task
    assert vod_env["_lounge_vod_cache"]["data"] is not None


async def test_failed_build_leaves_cache_untouched(vod_env: dict) -> None:
    cache = vod_env["_lounge_vod_cache"]
    await vod_env["_kick_vod_catalog_build"]()
    keep = cache["data"]

    async def _boom() -> dict:
        raise RuntimeError("tmdb down")

    vod_env["_build_vod_catalog"] = _boom
    cache["ts"] = 0.0
    await vod_env["_kick_vod_catalog_build"]()  # must not raise
    assert cache["data"] is keep


def test_warming_response_shape(vod_env: dict) -> None:
    r = vod_env["_vod_warming_response"]()
    assert r.status_code == 503
    body = json.loads(r.body)
    assert body["error"] == "vod_warming"
    assert r.headers["retry-after"] == "20"
