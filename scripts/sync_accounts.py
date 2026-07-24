#!/usr/bin/env python3
"""sync_accounts.py — pull the box's CURRENT upstream accounts onto this PC.

The off-box builders (vod_ship.py / pool_ship.py) build from
data/dynamic_upstream_accounts.json. That file is maintained on the BOX by the
nightly freshness pipeline; the PC's copy goes stale. This dispatches
push-accounts.yml (which reads the box's file and uploads it as an artifact),
waits for it, and downloads it here — so a subsequent `make {vod,pool}-ship`
builds from current accounts.

Usage:  make sync-accounts
        make refresh-all      # sync-accounts + epg-ship + vod-ship + pool-ship
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "data" / "dynamic_upstream_accounts.json"
_WF = "push-accounts.yml"


def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], cwd=_REPO, capture_output=True, text=True)


def main() -> None:
    r = _gh("workflow", "run", _WF)
    if r.returncode != 0:
        sys.exit(f"sync-accounts: dispatch failed: {r.stderr.strip()}")
    print("sync-accounts: dispatched push-accounts.yml; waiting for the box …")
    time.sleep(8)

    rid = _gh("run", "list", "--workflow", _WF, "--limit", "1",
              "--json", "databaseId", "-q", ".[0].databaseId").stdout.strip()
    if not rid:
        sys.exit("sync-accounts: could not find the dispatched run.")

    for _ in range(40):  # ~4 min ceiling
        st = _gh("run", "view", rid, "--json", "status", "-q", ".status").stdout.strip()
        if st == "completed":
            break
        time.sleep(6)
    concl = _gh("run", "view", rid, "--json", "conclusion", "-q", ".conclusion").stdout.strip()
    if concl != "success":
        sys.exit(f"sync-accounts: run {rid} ended {concl or 'incomplete'!r} — box unreachable?")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temp dir (gh run download won't overwrite an existing file),
    # validate, then atomically replace the local copy.
    with tempfile.TemporaryDirectory() as td:
        r = _gh("run", "download", rid, "--name", "accounts", "--dir", td)
        if r.returncode != 0:
            sys.exit(f"sync-accounts: artifact download failed: {r.stderr.strip()}")
        src = Path(td) / "dynamic_upstream_accounts.json"
        try:
            data = json.loads(src.read_text())
            assert isinstance(data, list) and data
        except Exception as e:  # noqa: BLE001
            sys.exit(f"sync-accounts: downloaded file invalid ({e}) — leaving prior copy.")
        shutil.move(str(src), str(_OUT))  # shutil handles the /tmp→repo cross-device case
    print(f"sync-accounts: wrote {_OUT} ({len(data)} fresh accounts).")


if __name__ == "__main__":
    main()
