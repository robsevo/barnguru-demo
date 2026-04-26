"""
scripts/extract_iptv.py
Extract NHL channels from paid IPTV M3U playlists and save to
dashboard/api/iptv_paid_channels.json.

Reads credentials from IPTV{n}_URL / IPTV{n}_USER / IPTV{n}_PASS (n=1..6).
Run: uv run python scripts/gretzky.py extract-iptv
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

_NHL_KEYWORDS = [
    "espn", "tnt", "tbs", "nhl", "sportsnet", "tsn", "msg",
    "foxsports", "fs1", "fs2", "abc", "nbc", "tva", "rds",
    "fanduel", "bally", "victory", "nesn", "nbcs", "nbcsp",
    "spectrum sportsnet", "nhln",
]

_OUT_PATH = Path(__file__).parent.parent / "dashboard" / "api" / "iptv_paid_channels.json"
_ENV_FILE = Path(__file__).parent.parent / "dashboard" / "api" / "iptv.env"


def _load_iptv_env_file() -> None:
    """Seed os.environ from dashboard/api/iptv.env if present.

    Mirrors the loader in dashboard/api/main.py so this script works in
    both contexts: CI runner (where the sidecar file is freshly written
    from the IPTV_ENV_BLOCK secret) and local dev (where the user has
    set the env vars in their shell or written the file by hand).
    Existing environment values win over the file so a manual override
    still works.
    """
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_iptv_env_file()


def _account_label(base_url: str) -> str:
    """Derive a short label from the service hostname."""
    host = base_url.split("//")[-1].split(":")[0]
    # Use first segment of hostname (e.g. "an upstream host" from "an upstream host.ddns.net")
    return host.split(".")[0]


def _is_nhl(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in _NHL_KEYWORDS)


def _fetch_and_parse(client: httpx.Client, m3u_url: str, account: str) -> list[dict]:
    """Fetch one M3U playlist and return NHL channel entries."""
    try:
        r = client.get(m3u_url, timeout=20.0, follow_redirects=True)
        if r.status_code != 200:
            print(f"  [{account}] HTTP {r.status_code} — skipping")
            return []
        text = r.text.lstrip("\ufeff")  # strip BOM if present
        lines = text.splitlines()
    except Exception as e:
        print(f"  [{account}] Fetch error: {e}")
        return []

    channels: list[dict] = []
    i = 0
    while i < len(lines) - 1:
        line = lines[i].strip()
        if line.startswith("#EXTINF"):
            m = re.search(r'tvg-name="([^"]+)"', line)
            title = m.group(1).strip() if m else line.split(",")[-1].strip()
            url_line = lines[i + 1].strip()
            if _is_nhl(title) and url_line.startswith("http"):
                channels.append({
                    "title": title,
                    "url": url_line,
                    "source": "paid_iptv",
                    "account": account,
                    "feed": "iptv",
                    "priority": 2,
                    "embed_only": False,
                })
            i += 2
        else:
            i += 1

    return channels


def main() -> None:
    extracted_at = datetime.now(timezone.utc).isoformat()
    all_channels: list[dict] = []
    seen_urls: set[str] = set()

    print("Extracting NHL channels from paid IPTV M3U playlists...\n")

    with httpx.Client(timeout=20.0) as client:
        for i in range(1, 7):
            base_url = os.environ.get(f"IPTV{i}_URL", "").rstrip("/")
            user     = os.environ.get(f"IPTV{i}_USER", "")
            pw       = os.environ.get(f"IPTV{i}_PASS", "")
            if not (base_url and user and pw):
                continue

            m3u_url = f"{base_url}/get.php?username={user}&password={pw}&type=m3u_plus&output=hls"
            account = _account_label(base_url)
            print(f"  [{account}] Fetching {base_url} ...")

            raw = _fetch_and_parse(client, m3u_url, account)
            new = 0
            for ch in raw:
                ch["extracted_at"] = extracted_at
                if ch["url"] not in seen_urls:
                    seen_urls.add(ch["url"])
                    all_channels.append(ch)
                    new += 1

            print(f"  [{account}] {new} unique NHL channels extracted")

    if not all_channels:
        print(
            "\nNo channels extracted. Set IPTV{n}_URL / IPTV{n}_USER / IPTV{n}_PASS"
            " env vars and re-run."
        )
        sys.exit(1)

    _OUT_PATH.write_text(json.dumps(all_channels, indent=2))
    print(f"\nSaved {len(all_channels)} channels → {_OUT_PATH}\n")

    counts = Counter(ch["account"] for ch in all_channels)
    print("Breakdown by account:")
    for acct, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {acct:20s} {count} channels")


if __name__ == "__main__":
    main()
