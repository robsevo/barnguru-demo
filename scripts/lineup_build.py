#!/usr/bin/env python3
"""lineup_build.py — precompute the lounge channel lineup OFF the API event loop.

WHY: `_derive_lounge_lineup` matches every curated lounge name against every entry
in the shipped pool — O(names x ~110k) — which is 5-12 minutes of solid Python. It
used to run inside the API's own background refresh, and because `_build_one` was
`async def` with zero awaits, `gather()` never yielded: once an hour (_LOUNGE_TTL)
the single uvicorn worker went dead and the API answered NOTHING, not even /health.
Measured 2026-07-31: 45 blackouts in 4 days, 6h41m, lengthening as the pool grew.
That is what "movies and series hang" actually was.

This runs the SAME derivation in a SEPARATE PROCESS and writes data/lounge_lineup.json,
which the box then serves in milliseconds (`_load_shipped_lounge_lineup`).

Runs ON THE BOX, unlike epg_ship / pool_ship / vod_ship. It has to: the pool ships
with RAW urls and the box relay-wraps them on load with a token that never leaves
the box, and the lineup's ranking/dedup operate on the WRAPPED urls. Building it
anywhere else would produce a lineup that doesn't match what the box would build.
It is a separate short-lived process, so its CPU competes with the API through the
OS scheduler instead of starving one event loop — the API stays responsive.

Pure optimization, exactly like the other ships: if this never runs, a missing or
too-old file falls through to the box's own build (now threaded, so still no
blackout). Nothing hard-depends on it.

HOW the logic stays in sync with the box: `ast`-load the exact functions out of
dashboard/api/main.py's SOURCE and exec them — no import of the FastAPI module (no
server side effects), and no second copy of the matching rules to drift.

Usage:  python3 scripts/lineup_build.py              # build + install atomically
        python3 scripts/lineup_build.py --dry-run    # build + report, write nothing
        python3 scripts/lineup_build.py --out PATH   # write somewhere else
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

_REPO = Path(__file__).resolve().parents[1]
_MAIN = _REPO / "dashboard" / "api" / "main.py"
_OUT = Path(os.environ.get(
    "LOUNGE_LINEUP_SHIP_FILE", str(_REPO / "data" / "lounge_lineup.json")))

# Everything the derivation needs; the closure walker pulls in the rest (matchers,
# blocklists, capacity ranking, logo + recode helpers) automatically.
_SEED = [
    "_derive_lounge_lineup",
    "_load_shipped_iptv_pool",
    "_is_tvpass",
]

# A lineup this small means the pool was missing or unreadable — don't overwrite a
# working file with junk (mirrors the _MIN_CHANNELS guard in pool_ship.py).
_MIN_CHANNELS = int(os.environ.get("MIN_LINEUP_CHANNELS", "40"))


def _index_top_level(src: str) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in ast.parse(src).body:
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
    """Seed names plus everything they transitively reference that main.py defines
    at top level. Names we don't own (builtins, import aliases) simply aren't in
    `by_name` and are expected to come from the base namespace."""
    included: dict[str, ast.AST] = {}
    stack = list(seed)
    while stack:
        name = stack.pop()
        if name in included:
            continue
        node = by_name.get(name)
        if node is None:
            continue  # not first-party — provided by the base namespace
        included[name] = node
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id not in included:
                stack.append(sub.id)
    return included


def _base_namespace(src: str) -> dict:
    """Re-create main.py's module-level IMPORT aliases (`_re_bc`, `_time_iptv`, …).

    Resolved from main.py's own import statements rather than hand-listed, so a new
    alias on the box never turns into a NameError here. Imports that fail (heavy
    server-only deps) are skipped — the closure only ever touches a handful, and a
    genuinely missing one still surfaces as a clear exec error below.
    """
    import importlib
    ns: dict = {
        # main.py's path, so the data-file constants it derives from
        # `Path(__file__).resolve().parents[2]` land on the same repo root the
        # box uses — not on this script's directory.
        "__file__": str(_MAIN),
        "__name__": "barn_lineup_core",
        "__builtins__": __builtins__,
    }
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            for a in node.names:
                try:
                    ns[a.asname or a.name.split(".")[0]] = importlib.import_module(a.name)
                except Exception:  # noqa: BLE001, S110 — optional server-only dep
                    pass
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                if a.name == "*":
                    continue
                try:
                    mod = importlib.import_module(node.module)
                    ns[a.asname or a.name] = getattr(mod, a.name)
                except Exception:  # noqa: BLE001, S110 — optional server-only dep
                    pass
    return ns


def _load_box_lineup_core() -> dict:
    """A namespace holding the box's exact lineup-derivation functions, exec'd from
    main.py's source (no import of the running module)."""
    src = _MAIN.read_text()
    by_name = _index_top_level(src)
    missing = [n for n in _SEED if n not in by_name]
    if missing:
        sys.exit(f"lineup-build: could not locate in main.py: {missing} — refusing to build.")
    ns = _base_namespace(src)
    for node in sorted(_closure(_SEED, by_name).values(), key=lambda n: n.lineno):
        seg = ast.get_source_segment(src, node)
        try:
            exec(seg, ns)  # noqa: S102 — trusted first-party source
        except Exception as e:  # noqa: BLE001
            label = getattr(node, "name", None) or "<assignment>"
            sys.exit(f"lineup-build: failed to exec '{label}': {type(e).__name__}: {e}\n"
                     f"  (a name it references is missing from the base namespace)")
    return ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build + report, write nothing")
    ap.add_argument("--out", default=str(_OUT), help="destination file")
    args = ap.parse_args()

    if not os.environ.get("IPTV_RELAY_TOKEN"):
        # Without the token _rewrite_iptv_url can't wrap, so the lineup would rank
        # and dedup raw upstream urls and NOT match what the box builds. Refuse
        # rather than ship a subtly-wrong lineup.
        sys.exit("lineup-build: IPTV_RELAY_TOKEN is not set — refusing to build a "
                 "lineup whose urls wouldn't match the box's.")

    ns = _load_box_lineup_core()
    t0 = time.time()

    pool = ns["_load_shipped_iptv_pool"]()
    if not pool:
        sys.exit("lineup-build: no usable shipped pool (missing/empty/too old) — "
                 "nothing to derive from.")
    all_channels = [ch for ch in pool if not ns["_is_tvpass"](ch)]
    t_pool = time.time() - t0

    t1 = time.time()
    lineup = ns["_derive_lounge_lineup"](all_channels)
    t_derive = time.time() - t1

    playable = [ch for ch in lineup if ch.get("primary_url")]
    print(f"pool      {len(all_channels):>7} entries   ({t_pool:.1f}s)")
    print(f"derived   {len(lineup):>7} channels  ({t_derive:.1f}s)  <- this is the "
          f"work that used to block the event loop")
    print(f"playable  {len(playable):>7} channels")

    if len(lineup) < _MIN_CHANNELS:
        sys.exit(f"lineup-build: only {len(lineup)} channels (< {_MIN_CHANNELS}) — "
                 f"refusing to install; the box keeps what it has.")
    if not playable:
        sys.exit("lineup-build: nothing playable — refusing to install; the box "
                 "keeps what it has.")

    doc = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "derive_seconds": round(t_derive, 1),
        "pool_entries": len(all_channels),
        "channels": lineup,
    }
    if args.dry_run:
        print("dry-run — nothing written")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic install: write beside the target then rename, so the box's
    # mtime-cached reader never observes a half-written file.
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(out)
    print(f"installed {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
