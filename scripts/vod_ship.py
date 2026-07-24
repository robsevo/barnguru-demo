#!/usr/bin/env python3
"""vod_ship.py — build the upstream VOD *stream index* off-box and ship it to the box.

WHY: the box maps a title to a playable stream via `_vod_stream_index`
({tmdb_id|name: [relay /vod URLs]}), built from every VOD account's
`get_vod_streams`. Those payloads are large — an upstream host alone is ~24 MB / 76k
entries — so building the index across ~16 accounts is an account-scaling memory
spike, and it's part of what OOM'd the 2 GB box (2026-07-24). This runs the SAME
build here (16 GB, residential IP the panels don't block), writes a compact
`data/vod_stream_index.json`, uploads it as the `vod-index-latest` GitHub Release
asset, and triggers `ship-vod.yml` which installs it on the box. The box then
loads the shipped index instead of fetching every account itself.

Pure optimization, exactly like epg_ship.py: if this never runs (PC off/stale),
the box self-builds the index in the background — nothing hard-depends on this PC,
least of all streaming.

HOW the logic stays in sync with the box WITHOUT drift: we `ast`-load the EXACT
build functions out of dashboard/api/main.py's SOURCE and exec them here — no
`import` of the heavy FastAPI module. Unlike epg_ship's hand-listed _NEEDED, the
VOD build's dependency graph is larger, so we resolve the transitive closure of
first-party names automatically from a small seed. A self-test aborts the ship if
the extracted logic misbehaves.

URLs are shipped RAW (unwrapped) — exactly like the EPG needs no special config.
The box applies its OWN relay wrap (_relay_wrap_vod, using its IPTV_LOCAL_PROXY_URL)
when it loads the file, so the result is byte-identical to what the box builds
itself. This script deliberately clears IPTV_LOCAL_PROXY_URL so it can never
double-wrap. The only real input it needs is data/dynamic_upstream_accounts.json
(the scraped-account list; hardcoded panels are always included).

Usage:  make vod-ship                 # build + publish + trigger box pull
        make vod-ship ARGS=--dry-run  # build + write file, no upload/trigger
Flags:  --dry-run     build + write data/vod_stream_index.json only
        --no-trigger  upload the asset but don't dispatch ship-vod.yml
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_MAIN = _REPO / "dashboard" / "api" / "main.py"
_OUT = _REPO / "data" / "vod_stream_index.json"
_RELEASE_TAG = "vod-index-latest"
_SHIP_WORKFLOW = "ship-vod.yml"

# Small seed; _closure() pulls in everything these transitively reference that is
# defined at main.py's top level (functions + module globals).
_SEED = [
    "_build_vod_stream_index",
    "_build_series_loc_index",
    "_VOD_ACCOUNTS",
    "_VOD_MAX_ACCOUNTS",
    "_upstream_ACCOUNTS",
    "_load_dynamic_upstream_accounts",
]

# A near-empty index means the panels were unreachable — don't overwrite the box's
# working index with junk (mirrors epg_ship's don't-ship-good-with-empty guard).
_MIN_KEYS = 50


def _base_ns() -> dict:
    """stdlib / third-party names the extracted code references, under the SAME
    aliases main.py imports them as. First-party names come from the closure."""
    import collections  # noqa: F401
    import re
    import time
    import urllib.parse as _u
    import httpx  # noqa: F401  (used by the extracted fetchers at call time)

    return {
        "__file__": str(_MAIN),  # so _load_dynamic_upstream_accounts resolves data/
        "asyncio": asyncio,
        "json": json,
        "os": os,
        "re": re,
        "_re_bc": re,
        "_re_iptv": re,
        "_re": re,
        "_re_backup": re,
        "time": time,
        "_time": time,
        "_t_iptv": time,
        "httpx": httpx,
        "_httpx_backup": httpx,
        "_httpx": httpx,
        "collections": collections,
        "Path": Path,
        "cast": lambda _t, v: v,  # typing.cast — identity at runtime
        "urlparse": _u.urlparse,
        "urljoin": _u.urljoin,
        "quote": _u.quote,
        "urlunparse": _u.urlunparse,
    }


def _index_top_level(src: str) -> dict[str, ast.AST]:
    """name -> top-level node (functions + simple assignments) from main.py."""
    tree = ast.parse(src)
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name and name not in out:
            out[name] = node
    return out


def _closure(seed: list[str], by_name: dict[str, ast.AST]) -> dict[str, ast.AST]:
    """Transitive closure: every first-party top-level name the seed reaches."""
    included: dict[str, ast.AST] = {}
    stack = list(seed)
    while stack:
        name = stack.pop()
        if name in included:
            continue
        node = by_name.get(name)
        if node is None:
            continue  # not first-party (import alias / builtin / provided in _base_ns)
        included[name] = node
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id not in included:
                stack.append(sub.id)
    return included


def _load_box_core() -> dict:
    """Namespace with the box's exact VOD-index build functions, exec'd from
    main.py source (no import), plus the FULL account list merged in."""
    src = _MAIN.read_text()
    by_name = _index_top_level(src)
    missing = [s for s in _SEED if s not in by_name]
    if missing:
        sys.exit(f"vod-ship: seed names not in main.py: {missing} — refusing to ship.")
    included = _closure(_SEED, by_name)

    ns = _base_ns()
    for node in sorted(included.values(), key=lambda n: n.lineno):
        seg = ast.get_source_segment(src, node)
        try:
            exec(seg, ns)  # noqa: S102 — trusted first-party source
        except Exception as e:  # noqa: BLE001
            label = getattr(node, "name", None) or "<assignment>"
            sys.exit(f"vod-ship: failed to exec '{label}': {type(e).__name__}: {e}\n"
                     f"  (a name it references is missing from _base_ns or the closure)")

    # main.py's DYNAMIC-account merge is top-level statements (a for-loop), not a
    # named node, so ast-extraction can't lift it. Replicate it here so the build
    # uses the full pool (hardcoded + scraped), UNCAPPED — this PC has the RAM the
    # box doesn't, which is the whole point.
    accounts = list(ns["_upstream_ACCOUNTS"])  # hardcoded literal
    seen = {(h, u) for _, h, _p, u, _pw in accounts}
    try:
        for row in ns["_load_dynamic_upstream_accounts"]():
            if (row[1], row[3]) not in seen:
                accounts.append(row)
                seen.add((row[1], row[3]))
    except Exception as e:  # noqa: BLE001
        print(f"vod-ship: WARNING — dynamic account load failed ({e}); "
              f"building from hardcoded panels only.", file=sys.stderr)
    ns["_upstream_ACCOUNTS"] = accounts
    # VOD keeps its quality cap (movies empty out past ~_VOD_MAX_ACCOUNTS) but off
    # the FULL pool, not the box's memory-capped one.
    ns["_VOD_ACCOUNTS"] = accounts[: ns["_VOD_MAX_ACCOUNTS"]]
    print(f"vod-ship: {len(accounts)} total accounts, "
          f"{len(ns['_VOD_ACCOUNTS'])} used for VOD.")
    return ns


def _self_test(ns: dict) -> None:
    nrm = ns.get("_norm_series_name")
    if not callable(nrm):
        sys.exit("vod-ship: _norm_series_name missing from closure — refusing to ship.")
    checks = [
        ("norm year+tag", nrm("Breaking Bad (2008) HD") == "breakingbad"),
        ("norm punctuation", nrm("Spider-Man: No Way Home") == "spidermannowayhome"),
        ("accounts present", len(ns["_upstream_ACCOUNTS"]) >= 4),
    ]
    failed = [n for n, ok in checks if not ok]
    if failed:
        sys.exit(f"vod-ship: self-test FAILED ({failed}) — extracted logic looks wrong.")


def _publish(no_trigger: bool) -> None:
    if subprocess.run(["gh", "release", "view", _RELEASE_TAG],
                      cwd=_REPO, capture_output=True).returncode != 0:
        subprocess.run(
            ["gh", "release", "create", _RELEASE_TAG, "--title", "VOD index snapshots",
             "--notes", "Off-box VOD stream index shipped to the box (scripts/vod_ship.py)."],
            cwd=_REPO, check=True)
    subprocess.run(["gh", "release", "upload", _RELEASE_TAG, str(_OUT), "--clobber"],
                   cwd=_REPO, check=True)
    print(f"vod-ship: uploaded {_OUT.name} to release {_RELEASE_TAG}")
    if no_trigger:
        print("vod-ship: --no-trigger set; skipping ship-vod.yml dispatch.")
        return
    subprocess.run(["gh", "workflow", "run", _SHIP_WORKFLOW], cwd=_REPO, check=True)
    print(f"vod-ship: dispatched {_SHIP_WORKFLOW} — the box will pull + refresh (no restart).")


def main() -> None:
    ap = argparse.ArgumentParser(prog="vod-ship")
    ap.add_argument("--dry-run", action="store_true", help="build + write the file only")
    ap.add_argument("--no-trigger", action="store_true", help="upload but don't dispatch the workflow")
    args, _ = ap.parse_known_args()

    # Ship RAW urls — the box wraps them with its own relay on load. Clearing these
    # guarantees the build can't accidentally relay-wrap (which would double-wrap on
    # the box). Same philosophy as epg_ship: this machine produces pure data.
    os.environ.pop("IPTV_LOCAL_PROXY_URL", None)
    os.environ.pop("IPTV_RELAY_TOKEN", None)

    ns = _load_box_core()
    _self_test(ns)

    print("vod-ship: building VOD stream index from all panels …")
    idx: dict = asyncio.run(ns["_build_vod_stream_index"]())
    print(f"vod-ship: built {len(idx)} index keys (tmdb_id + name).")
    if len(idx) < _MIN_KEYS:
        sys.exit(f"vod-ship: only {len(idx)} keys (< {_MIN_KEYS}) — panels likely "
                 f"unreachable; NOT shipping (box keeps its own index).")

    doc = {"built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "index": idx}
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = _OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False))
    os.replace(tmp, _OUT)
    print(f"vod-ship: wrote {_OUT} ({_OUT.stat().st_size // 1024} KB, "
          f"built_utc {doc['built_utc']}).")

    if args.dry_run:
        print("vod-ship: --dry-run set; not publishing.")
        return
    _publish(args.no_trigger)


if __name__ == "__main__":
    main()
