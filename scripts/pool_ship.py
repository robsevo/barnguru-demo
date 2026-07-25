#!/usr/bin/env python3
"""pool_ship.py — build the live IPTV channel pool off-box and ship it to the box.

WHY: `_build_iptv_channels` downloads every panel's m3u_plus playlist and splits
it before filtering — an upstream host alone is ~554k lines (~150-200MB) — so building
the pool across all accounts is the account-scaling memory spike that OOM'd the
2GB box (2026-07-24). This runs the SAME build here (16GB, residential IP), writes
a compact `data/iptv_pool.json` (~7MB, the deduped ~7k-channel pool), uploads it as
the `iptv-pool-latest` GitHub Release asset, and triggers `ship-pool.yml` which
installs it on the box. The box loads the pool instead of parsing every playlist.

Pure optimization, exactly like epg_ship / vod_ship: if this never runs (PC
off/stale) the box self-builds — nothing hard-depends on the PC, streaming is
untouched.

URLs are shipped RAW. The relay is token-gated, so — unlike VOD — the box must do
the relay-wrap itself (the token stays on the box, never on this PC): it rewrites
each URL with `_rewrite_iptv_url` on load, byte-identical to a self-build. This
script clears IPTV_LOCAL_PROXY_URL so it can never wrap. Only real input: a
reasonably fresh data/dynamic_upstream_accounts.json (hardcoded panels always in).

HOW the logic stays in sync with the box: `ast`-load the exact build functions out
of dashboard/api/main.py's SOURCE and exec them (no import of the FastAPI module),
resolving the transitive first-party closure automatically from a small seed.

Usage:  make pool-ship                 # build + publish + trigger box pull
        make pool-ship ARGS=--dry-run  # build + write file, no upload/trigger
Flags:  --dry-run     build + write data/iptv_pool.json only
        --no-trigger  upload the asset but don't dispatch ship-pool.yml
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
_OUT = _REPO / "data" / "iptv_pool.json"
_RELEASE_TAG = "iptv-pool-latest"
_SHIP_WORKFLOW = "ship-pool.yml"

_SEED = [
    "_build_iptv_channels",
    "_LIVE_ACCOUNTS",
    "_upstream_ACCOUNTS",
    "_load_dynamic_upstream_accounts",
    "_verify_stream_alive",    # liveness probe (the check the box can't afford)
]

# Sources die at the PANEL (host) level — a down panel returns nothing for ALL its
# channels (an upstream host = 0 ch today). So we don't probe every candidate; we sample a
# few URLs per host, decide host live/dead, and drop every source from a dead host.
# ~16 hosts x this many samples = a tiny, fast, robust probe.
_HOST_SAMPLES = int(os.environ.get("HOST_SAMPLES", "8"))

# A near-empty pool means the panels were unreachable — don't overwrite the box's
# working pool with junk (mirrors epg_ship / vod_ship).
_MIN_CHANNELS = 500
# How many sources to verify at once on this PC. The box's own _VERIFY_SEM is 20
# (to protect its 2GB); we have RAM + a residential IP, so go wider.
_VERIFY_CONCURRENCY = int(os.environ.get("VERIFY_CONCURRENCY", "120"))


def _base_ns() -> dict:
    import collections  # noqa: F401
    import re
    import time
    import urllib.parse as _u
    import httpx  # noqa: F401  (used by the extracted fetchers at call time)

    return {
        "__file__": str(_MAIN),  # so _load_dynamic_upstream_accounts resolves data/
        "asyncio": asyncio, "json": json, "os": os, "re": re,
        "_re_bc": re, "_re_iptv": re, "_re": re, "_re_backup": re,
        "time": time, "_time": time, "_t_iptv": time, "_time_iptv": time,
        "httpx": httpx, "_httpx_backup": httpx, "_httpx": httpx,
        "collections": collections, "Path": Path,
        "cast": lambda _t, v: v,
        "urlparse": _u.urlparse, "urljoin": _u.urljoin,
        "quote": _u.quote, "urlunparse": _u.urlunparse,
    }


def _index_top_level(src: str) -> dict[str, ast.AST]:
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
    src = _MAIN.read_text()
    by_name = _index_top_level(src)
    missing = [s for s in _SEED if s not in by_name]
    if missing:
        sys.exit(f"pool-ship: seed names not in main.py: {missing} — refusing to ship.")
    included = _closure(_SEED, by_name)

    ns = _base_ns()
    for node in sorted(included.values(), key=lambda n: n.lineno):
        seg = ast.get_source_segment(src, node)
        try:
            exec(seg, ns)  # noqa: S102 — trusted first-party source
        except Exception as e:  # noqa: BLE001
            label = getattr(node, "name", None) or "<assignment>"
            sys.exit(f"pool-ship: failed to exec '{label}': {type(e).__name__}: {e}\n"
                     f"  (a name it references is missing from _base_ns or the closure)")

    # main.py's dynamic-account merge is top-level statements, not a named node, so
    # replicate it — UNCAPPED (this PC has the RAM the box doesn't).
    hardcoded = list(ns["_upstream_ACCOUNTS"])  # the literal, before the scraped merge
    accounts = list(hardcoded)
    seen = {(h, u) for _, h, _p, u, _pw in accounts}
    try:
        for row in ns["_load_dynamic_upstream_accounts"]():
            if (row[1], row[3]) not in seen:
                accounts.append(row)
                seen.add((row[1], row[3]))
    except Exception as e:  # noqa: BLE001
        print(f"pool-ship: WARNING — dynamic account load failed ({e}); "
              f"hardcoded panels only.", file=sys.stderr)
    # Hosts from the HARDCODED panels (before the scraped merge) are the box's
    # curated primaries (an upstream host TSN, bgdc, …). They're often served only through
    # the relay — a direct probe from here false-negatives — and the box already
    # handles their liveness (demotion + client rotation). Never host-drop them; we
    # only prune dead SCRAPED panels.
    ns["_HARDCODED_HOSTS"] = {str(r[1]).lower() for r in hardcoded}
    ns["_upstream_ACCOUNTS"] = accounts
    ns["_LIVE_ACCOUNTS"] = accounts  # live uses the FULL pool
    print(f"pool-ship: {len(accounts)} accounts ({len(ns['_HARDCODED_HOSTS'])} hardcoded hosts).")
    return ns


def _self_test(ns: dict) -> None:
    rw = ns.get("_rewrite_iptv_url")
    if not callable(rw):
        sys.exit("pool-ship: _rewrite_iptv_url missing from closure — refusing to ship.")
    checks = [
        # No tunnel env ⇒ rewrite is identity (we ship raw; the box wraps).
        ("rewrite is raw", rw("http://example.com/live/x/y/1.m3u8") == "http://example.com/live/x/y/1.m3u8"),
        ("accounts present", len(ns["_upstream_ACCOUNTS"]) >= 4),
    ]
    failed = [n for n, ok in checks if not ok]
    if failed:
        sys.exit(f"pool-ship: self-test FAILED ({failed}) — extracted logic looks wrong.")


def _publish(no_trigger: bool) -> None:
    if subprocess.run(["gh", "release", "view", _RELEASE_TAG],
                      cwd=_REPO, capture_output=True).returncode != 0:
        subprocess.run(
            ["gh", "release", "create", _RELEASE_TAG, "--title", "IPTV pool snapshots",
             "--notes", "Off-box live IPTV pool shipped to the box (scripts/pool_ship.py)."],
            cwd=_REPO, check=True)
    subprocess.run(["gh", "release", "upload", _RELEASE_TAG, str(_OUT), "--clobber"],
                   cwd=_REPO, check=True)
    print(f"pool-ship: uploaded {_OUT.name} to release {_RELEASE_TAG}")
    if no_trigger:
        print("pool-ship: --no-trigger set; skipping ship-pool.yml dispatch.")
        return
    subprocess.run(["gh", "workflow", "run", _SHIP_WORKFLOW], cwd=_REPO, check=True)
    print(f"pool-ship: dispatched {_SHIP_WORKFLOW} — the box will pull + refresh (no restart).")


def _host_of(url: str) -> str:
    from urllib.parse import urlparse
    return (urlparse(url or "").hostname or "").lower()


# ── relay-based verification (preferred) ─────────────────────────────────────
# Probing panels DIRECTLY from this PC is unreliable: hosts that block or
# rate-limit this IP false-negative (an upstream host read "live" one run, "dead" the
# next), and the box serves through the relay anyway. So ask the relay's
# /upstream-json to hit each account's player_api.php from its RESIDENTIAL IP —
# the authoritative "is this account online, and is it busy?" answer, and a plain
# JSON fetch (no ffmpeg session, no stream bandwidth). Creds come from
# `make sync-relay-token`; without them we fall back to direct host-sampling.
_RELAY_ENV = Path.home() / ".config" / "grtzky" / "relay.env"
_RELAY_CONCURRENCY = int(os.environ.get("RELAY_VERIFY_CONCURRENCY", "6"))


def _relay_creds() -> "tuple[str, str] | None":
    try:
        kv = dict(
            ln.split("=", 1) for ln in _RELAY_ENV.read_text().splitlines()
            if "=" in ln and not ln.lstrip().startswith("#")
        )
    except OSError:
        return None
    base = (kv.get("IPTV_LOCAL_PROXY_URL") or "").strip().rstrip("/")
    token = (kv.get("IPTV_RELAY_TOKEN") or "").strip()
    return (base, token) if base and token else None


def _acct_key_from_url(url: str) -> "tuple[str, str | None]":
    """(host, username) for an upstream stream URL. Username is only trusted when the
    path follows the /<kind>/<user>/<pass>/<id> convention; hosts like an upstream host use
    /play/<token>/ts with no user, which yields None (host-level decision instead)."""
    from urllib.parse import urlparse
    p = urlparse(url or "")
    seg = [s for s in (p.path or "").split("/") if s]
    user = seg[1] if len(seg) >= 3 and seg[0] in ("live", "movie", "series") else None
    return (p.hostname or "").lower(), user


async def _verify_accounts_via_relay(accounts: list, relay: str, token: str) -> dict:
    """Per-account verdicts from the relay's residential IP.
    Returns {"dead": {(host,user), …}, "alive_hosts": {…}, "probed_hosts": {…}, "busy": [...]}"""
    import httpx
    from urllib.parse import quote

    sem = asyncio.Semaphore(_RELAY_CONCURRENCY)   # relay also serves live TV — be gentle

    async def _one(label, host, port, user, pw):
        api = f"http://{host}:{port}/player_api.php?username={user}&password={pw}"
        probe = f"{relay}/upstream-json?u={quote(api, safe='')}&t={quote(token, safe='')}"
        async with sem:
            try:
                async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as cl:
                    r = await asyncio.wait_for(cl.get(probe), timeout=25.0)
            except Exception:  # noqa: BLE001 — relay/tunnel hiccup ⇒ inconclusive
                return (label, host, user, None, None)
        if r.status_code == 400:
            return (label, host, user, None, None)   # host not relay-allowlisted ⇒ unknown
        if r.status_code != 200:
            return (label, host, user, None, None)   # 502 etc ⇒ inconclusive, never drop
        try:
            info = (r.json() or {}).get("user_info") or {}
        except Exception:  # noqa: BLE001 — non-JSON (panel error page) ⇒ inconclusive
            return (label, host, user, None, None)
        if not info:
            return (label, host, user, None, None)
        alive = int(info.get("auth") or 0) == 1 and str(info.get("status", "")).lower() == "active"
        try:
            act, mx = int(info.get("active_cons") or 0), int(info.get("max_connections") or 0)
        except (TypeError, ValueError):
            act, mx = 0, 0
        return (label, host, user, alive, (act, mx))

    print(f"pool-ship: verifying {len(accounts)} accounts through the relay "
          f"(residential IP, concurrency {_RELAY_CONCURRENCY}) …")
    results = await asyncio.gather(*[_one(*a) for a in accounts])

    dead, alive_hosts, probed_hosts, busy, unknown = set(), set(), set(), [], 0
    capacity: dict[str, int] = {}
    contradictory: list[str] = []
    for label, host, user, alive, cons in results:
        h = str(host).lower()
        if alive is None:
            unknown += 1
            continue
        probed_hosts.add(h)
        if alive:
            alive_hosts.add(h)
            if cons:
                act, mx = cons
                # TRUST max_connections only when the panel's own numbers are
                # self-consistent. an upstream host reported active=25 against max=1 — a
                # 1-connection account cannot serve 25, so that field is misreported
                # and must NOT be shipped as "worst capacity" (it would demote a
                # source that demonstrably serves many viewers). Contradictory hosts
                # are simply omitted, leaving the box's neutral ranking in place.
                if mx > 0 and act > mx:
                    contradictory.append(f"{label}({act}/{mx})")
                else:
                    # Keep the most permissive capacity seen for a host (0 = unlimited).
                    prev = capacity.get(h)
                    if prev is None or mx == 0 or (prev != 0 and mx > prev):
                        capacity[h] = mx
                    if mx > 0 and act >= mx:
                        busy.append(f"{label}({act}/{mx})")
        else:
            dead.add((h, str(user)))
    print(f"pool-ship: accounts — {len(alive_hosts)} hosts with a live account, "
          f"{len(dead)} dead accounts, {unknown} inconclusive (kept).")
    if busy:
        print(f"pool-ship: genuinely at connection limit: {', '.join(busy)}")
    if contradictory:
        print(f"pool-ship: misreported max_connections (active > max ⇒ field ignored): "
              f"{', '.join(contradictory)}")
    single = sorted(h for h, mx in capacity.items() if mx == 1)
    if single:
        print(f"pool-ship: 1-connection hosts (one viewer at a time): {single}")
    return {"dead": dead, "alive_hosts": alive_hosts, "probed_hosts": probed_hosts,
            "capacity": capacity}


def _prune_by_dead_accounts(channels: list, verdict: dict, protected: set) -> list:
    """Drop sources belonging to accounts the relay says are dead. A source whose
    username isn't in its URL is dropped only when EVERY probed account on its host
    is dead. Inconclusive verdicts never drop anything."""
    dead, alive_hosts, probed = verdict["dead"], verdict["alive_hosts"], verdict["probed_hosts"]
    kept = []
    for ch in channels:
        if ch.get("source") in {"tvpass"}:
            kept.append(ch)
            continue
        host, user = _acct_key_from_url(ch.get("url", ""))
        if host in protected:
            kept.append(ch)
            continue
        if user is not None:
            if (host, user) in dead:
                continue
        elif host in probed and host not in alive_hosts:
            continue  # no user in URL and every account on this host is dead
        kept.append(ch)
    return kept


async def _verify_and_prune(ns: dict, channels: list) -> list:
    """Sample-probe each upstream HOST and drop every source from a dead panel, so
    what ships is only sources whose panel is actually serving. This is what fixes
    "busy/offline sources": a down panel (an upstream host=0ch today) is dropped wholesale,
    and channels fall through to the live panels. The 2GB box can't afford any of
    this; this PC can. Fast + robust: ~16 hosts x a few samples, not thousands."""
    verify = ns["_verify_stream_alive"]
    ns["_VERIFY_SEM"] = asyncio.Semaphore(_VERIFY_CONCURRENCY)  # box caps at 20; we have RAM

    # Never host-drop: tvpass (token flow a raw probe can't replicate) and the
    # hardcoded/curated panels (an upstream host etc. — often relay-only, so a direct probe
    # false-negatives; the box handles their liveness). We only prune dead SCRAPED
    # panels, which is where the busy/dead junk actually is.
    _PROTECTED_SOURCES = {"tvpass"}
    hardcoded_hosts = ns.get("_HARDCODED_HOSTS", set())

    by_host: dict[str, list] = {}
    for ch in channels:
        if ch.get("source") in _PROTECTED_SOURCES:
            continue
        h = _host_of(ch.get("url", ""))
        if h and h not in hardcoded_hosts:
            by_host.setdefault(h, []).append(ch)

    # Evenly-spaced samples per host (spread across its channel list).
    sample_host: dict[str, str] = {}   # sample url -> host
    for h, chans in by_host.items():
        step = max(1, len(chans) // _HOST_SAMPLES)
        for c in chans[::step][:_HOST_SAMPLES]:
            u = c.get("url")
            if u:
                sample_host[u] = h

    print(f"pool-ship: probing {len(by_host)} hosts ({len(sample_host)} sample sources) …")

    async def _probe(u: str):
        try:
            # Hard cap — _verify_stream_alive GETs the body and a live .ts trickles
            # forever, so a source that won't answer promptly counts as not working.
            ok = await asyncio.wait_for(verify(u, timeout=3.0), timeout=4.5)
        except Exception:  # noqa: BLE001  (incl. asyncio.TimeoutError)
            ok = False
        return u, ok

    results = await asyncio.gather(*[_probe(u) for u in sample_host])
    live_host: dict[str, bool] = {}
    for u, ok in results:
        h = sample_host[u]
        live_host[h] = live_host.get(h, False) or ok  # host live if ANY sample works
    dead = {h for h, live in live_host.items() if not live}
    live = [h for h, ok in live_host.items() if ok]
    print(f"pool-ship: {len(live)} live hosts, {len(dead)} dead panels dropped: {sorted(dead)}")
    if not dead:
        return channels
    return [ch for ch in channels
            if ch.get("source") in _PROTECTED_SOURCES
            or _host_of(ch.get("url", "")) in hardcoded_hosts
            or _host_of(ch.get("url", "")) not in dead]


def main() -> None:
    ap = argparse.ArgumentParser(prog="pool-ship")
    ap.add_argument("--dry-run", action="store_true", help="build + write the file only")
    ap.add_argument("--no-trigger", action="store_true", help="upload but don't dispatch the workflow")
    args, _ = ap.parse_known_args()

    # Ship RAW urls — the box relay-wraps on load with its own (secret) token.
    os.environ.pop("IPTV_LOCAL_PROXY_URL", None)
    os.environ.pop("IPTV_RELAY_TOKEN", None)

    ns = _load_box_core()
    _self_test(ns)

    print("pool-ship: building live IPTV pool from all sources …")

    async def _run() -> tuple:
        chans = await ns["_build_iptv_channels"]()
        print(f"pool-ship: built {len(chans)} channels.")
        if len(chans) < _MIN_CHANNELS:
            return chans, {}  # let main() abort below
        creds = _relay_creds()
        if creds:
            # Authoritative path: per-account verdicts from the relay's residential
            # IP. This DOES verify the curated panels too (an upstream host et al.) — the
            # thing direct probing couldn't do — so nothing is blanket-protected
            # here except tvpass (token flow, no account to query).
            relay, token = creds
            verdict = await _verify_accounts_via_relay(ns["_LIVE_ACCOUNTS"], relay, token)
            before = len(chans)
            chans = _prune_by_dead_accounts(chans, verdict, set())
            print(f"pool-ship: dropped {before - len(chans)} sources from dead accounts.")
            return chans, verdict.get("capacity") or {}
        print("pool-ship: no relay creds at ~/.config/grtzky/relay.env "
              "(run `make sync-relay-token`) — falling back to direct host-sampling.",
              file=sys.stderr)
        return await _verify_and_prune(ns, chans), {}

    channels, capacity = asyncio.run(_run())
    if len(channels) < _MIN_CHANNELS:
        sys.exit(f"pool-ship: only {len(channels)} channels (< {_MIN_CHANNELS}) — panels "
                 f"likely unreachable; NOT shipping (box keeps its own pool).")
    print(f"pool-ship: {len(channels)} channels after source verification.")

    # `capacity` = {host: max_connections} measured through the relay (0 = unlimited).
    # The box feeds it into _upstream_CAPACITY so _account_cap_rank puts never-busy
    # accounts first when picking a primary — the fix for "this source is busy".
    doc = {"built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "capacity": capacity,
           "channels": channels}
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = _OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False))
    os.replace(tmp, _OUT)
    print(f"pool-ship: wrote {_OUT} ({_OUT.stat().st_size // 1024} KB, "
          f"built_utc {doc['built_utc']}).")

    if args.dry_run:
        print("pool-ship: --dry-run set; not publishing.")
        return
    _publish(args.no_trigger)


if __name__ == "__main__":
    main()
