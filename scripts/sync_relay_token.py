#!/usr/bin/env python3
"""sync_relay_token.py — pull the relay's base URL + token onto this PC.

The off-box ships verify sources THROUGH the relay (the residential path the box
serves from), which needs the relay's token. This dispatches push-relay-token.yml
(the box reads iptv.env and uploads a 1-day artifact), waits, and writes the creds
to ~/.config/grtzky/relay.env with mode 600 — NEVER into the repo.

pool_ship.py reads that file; without it, it falls back to direct host-sampling.

Usage:  make sync-relay-token
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_OUT = Path.home() / ".config" / "grtzky" / "relay.env"
_WF = "push-relay-token.yml"


def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], cwd=_REPO, capture_output=True, text=True)


def main() -> None:
    r = _gh("workflow", "run", _WF)
    if r.returncode != 0:
        sys.exit(f"sync-relay-token: dispatch failed: {r.stderr.strip()}")
    print("sync-relay-token: dispatched push-relay-token.yml; waiting for the box …")
    time.sleep(8)

    rid = _gh("run", "list", "--workflow", _WF, "--limit", "1",
              "--json", "databaseId", "-q", ".[0].databaseId").stdout.strip()
    if not rid:
        sys.exit("sync-relay-token: could not find the dispatched run.")

    for _ in range(40):  # ~4 min ceiling
        if _gh("run", "view", rid, "--json", "status", "-q", ".status").stdout.strip() == "completed":
            break
        time.sleep(6)
    concl = _gh("run", "view", rid, "--json", "conclusion", "-q", ".conclusion").stdout.strip()
    if concl != "success":
        sys.exit(f"sync-relay-token: run {rid} ended {concl or 'incomplete'!r} — box unreachable?")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        r = _gh("run", "download", rid, "--name", "relay-creds", "--dir", td)
        if r.returncode != 0:
            sys.exit(f"sync-relay-token: artifact download failed: {r.stderr.strip()}")
        src = Path(td) / "relay.env"
        text = src.read_text()
        if "IPTV_RELAY_TOKEN=" not in text or "IPTV_LOCAL_PROXY_URL=" not in text:
            sys.exit("sync-relay-token: artifact missing expected keys — leaving prior copy.")
        shutil.move(str(src), str(_OUT))  # shutil handles the /tmp->home cross-device case
    os.chmod(_OUT, stat.S_IRUSR | stat.S_IWUSR)  # 600 — creds, not repo content
    url = next((ln.split("=", 1)[1] for ln in text.splitlines()
                if ln.startswith("IPTV_LOCAL_PROXY_URL=")), "?")
    print(f"sync-relay-token: wrote {_OUT} (mode 600) for relay {url}")


if __name__ == "__main__":
    main()
