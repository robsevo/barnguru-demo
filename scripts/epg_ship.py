#!/usr/bin/env python3
"""epg_ship.py — build the merged cable EPG on a laptop and publish it to the box.

WHY: fetching xmltv.php from the upstream panels (18–41 MB each) and merging it is
the heaviest thing the 2 GB API box does, and the historical OOM cause. This runs
the SAME build off-box (laptop: more RAM, a residential IP the panels don't block),
writes data/lounge_epg.json (the cable guide only — the box folds in live SPORTS
itself), uploads it as the `epg-latest` GitHub Release asset, and triggers
ship-epg.yml which rsyncs it onto the box. The box then reads the file instead of
doing the fetch (dashboard/api/main.py: _load_shipped_lounge_epg / _build_lounge_epg).
Pure optimization: if this is skipped, the box self-builds — nothing depends on the
laptop, least of all streaming (the relay + live-channel build are untouched here).

HOW the logic stays in sync with the box WITHOUT drift or a fragile hand-copy:
rather than duplicate ~200 lines of unicode-heavy matching regexes, we load the
EXACT function bodies out of dashboard/api/main.py's SOURCE with `ast` and exec them
here — no `import` of the heavy FastAPI module (no server side effects), and no edit
to it (zero risk to the live/stream path). A self-test (below) aborts the ship if the
extracted logic misbehaves, so a reformat/refactor of main.py can never publish a
wrong guide — the box just self-builds until this is fixed.

Usage:  make epg-ship            # build + publish + trigger box pull
        make epg-ship ARGS=--dry-run     # build + write file, no publish
Flags:  --dry-run     build + write data/lounge_epg.json only (no upload/trigger)
        --no-trigger  upload the release asset but don't dispatch ship-epg.yml
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re as _re_bc
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_MAIN = _REPO / "dashboard" / "api" / "main.py"
_OUT = _REPO / "data" / "lounge_epg.json"
_RELEASE_TAG = "epg-latest"
_SHIP_WORKFLOW = "ship-epg.yml"

# Top-level names lifted verbatim from main.py (in source order — assignment deps
# like _LOUNGE_CHANNEL_NAMES -> _BARNCENTRE_CHANNEL_NAMES resolve naturally).
_NEEDED = [
    "_BROWSER_HEADERS",
    "_upstream_ACCOUNTS",            # the hardcoded literal only (no dynamic scraped merge)
    "_BARNCENTRE_CHANNEL_NAMES",
    "_normalize_ch",
    "_SHORT_FR_DISPLAYS",
    "_FOREIGN_FEED_RE",
    "_ch_matches",
    "_LOUNGE_CHANNEL_NAMES",
    "_fetch_upstream_xmltv",
    "_EPG_ALLOWED_LANGS",
    "_EPG_FOREIGN_HINT_RE",
    "_guide_is_foreign",
    "_self_build_cable_epg",
]

# Minimum channels that must carry an UPCOMING programme for the build to be worth
# shipping — the "don't ship good-with-empty" guard (mirrors iptv_freshness.py and
# the box's own degraded-build notion). A near-empty build means the panels were
# unreachable; skip the ship and let the box keep serving what it has.
_MIN_FUTURE_CHANNELS = 10


def _load_box_epg_core() -> dict:
    """Return a namespace holding the box's exact EPG-merge functions, exec'd from
    main.py's source (no import of the running module)."""
    src = _MAIN.read_text()
    tree = ast.parse(src)
    nodes: dict[str, ast.AST] = {}
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name and name in _NEEDED and name not in nodes:
            nodes[name] = node
    missing = [n for n in _NEEDED if n not in nodes]
    if missing:
        sys.exit(f"epg-ship: could not locate in main.py: {missing} — refusing to ship.")
    ns: dict = {"_re_bc": _re_bc, "asyncio": asyncio, "re": _re_bc}
    for node in sorted(nodes.values(), key=lambda n: n.lineno):
        exec(ast.get_source_segment(src, node), ns)  # noqa: S102 — trusted first-party source
    return ns


def _self_test(ns: dict) -> None:
    """Abort the ship if the extracted matching logic doesn't behave as expected —
    a reformat of main.py that broke extraction must never publish a wrong guide."""
    nrm, chm, frn = ns["_normalize_ch"], ns["_ch_matches"], ns["_guide_is_foreign"]
    checks = [
        ("normalize pipe", nrm("US | Fox Sports 1 HD") == "fs1"),
        ("normalize dash", nrm("CA - SPORTSNET EAST") == "sportsnet east"),
        ("match hd suffix", chm("HBO HD", "HBO") is True),
        ("reject foreign feed", chm("ESPN Brasil", "ESPN") is False),
        ("foreign guide pl", frn([{"title": "Odcinek", "lang": "pl"}] * 10) is True),
        ("english guide ok", frn([{"title": "Evening News", "lang": "en"}] * 10) is False),
        ("accounts present", len(ns["_upstream_ACCOUNTS"]) >= 4),
        ("channels present", len(ns["_LOUNGE_CHANNEL_NAMES"]) >= 50),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        sys.exit(f"epg-ship: self-test FAILED ({failed}) — extracted logic looks wrong, refusing to ship.")


def _future_channel_count(channels: dict) -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n = 0
    for progs in channels.values():
        if any((p.get("stop_utc") or "") > now for p in progs):
            n += 1
    return n


def _publish(no_trigger: bool) -> None:
    if subprocess.run(["gh", "release", "view", _RELEASE_TAG],
                      cwd=_REPO, capture_output=True).returncode != 0:
        subprocess.run(
            ["gh", "release", "create", _RELEASE_TAG, "--title", "EPG snapshots",
             "--notes", "Daily off-box EPG build shipped to the box (scripts/epg_ship.py)."],
            cwd=_REPO, check=True)
    subprocess.run(["gh", "release", "upload", _RELEASE_TAG, str(_OUT), "--clobber"],
                   cwd=_REPO, check=True)
    print(f"epg-ship: uploaded {_OUT.name} to release {_RELEASE_TAG}")
    if no_trigger:
        print("epg-ship: --no-trigger set; skipping ship-epg.yml dispatch.")
        return
    subprocess.run(["gh", "workflow", "run", _SHIP_WORKFLOW], cwd=_REPO, check=True)
    print(f"epg-ship: dispatched {_SHIP_WORKFLOW} — the box will pull + refresh (no restart).")


def main() -> None:
    ap = argparse.ArgumentParser(prog="epg-ship")
    ap.add_argument("--dry-run", action="store_true", help="build + write the file only; no upload/trigger")
    ap.add_argument("--no-trigger", action="store_true", help="upload the asset but don't dispatch ship-epg.yml")
    # gretzky passes verb args through after `--`; tolerate unknown extras.
    args, _ = ap.parse_known_args()

    ns = _load_box_epg_core()
    _self_test(ns)

    print("epg-ship: building cable EPG from the hardcoded panels …")
    channels: dict = asyncio.run(ns["_self_build_cable_epg"]())
    total = len(channels)
    future = _future_channel_count(channels)
    print(f"epg-ship: built {total} channels, {future} with an upcoming programme.")
    if future < _MIN_FUTURE_CHANNELS:
        sys.exit(f"epg-ship: only {future} channels have upcoming programmes "
                 f"(< {_MIN_FUTURE_CHANNELS}) — panels likely unreachable; NOT shipping "
                 f"(the box keeps serving its own guide).")

    doc = {"built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "channels": channels}
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = _OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False))
    os.replace(tmp, _OUT)
    print(f"epg-ship: wrote {_OUT} ({_OUT.stat().st_size // 1024} KB, built_utc {doc['built_utc']}).")

    if args.dry_run:
        print("epg-ship: --dry-run set; not publishing.")
        return
    _publish(args.no_trigger)


if __name__ == "__main__":
    main()
