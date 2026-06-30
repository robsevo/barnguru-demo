#!/usr/bin/env python3
"""IPTV freshness scraper — finds fresh, working upstream accounts and feeds them
to the channel builder + relay allowlist.

Pipeline:
  1. Scrape r/REDACTED_SOURCE RSS for base64 blobs (configurable day window).
  2. Decode them: plain base64 → M3U URLs, or paste.sh → AES-decrypt → M3U URLs.
  3. Parse upstream (server/user/pass) credentials from the M3U URLs.
  4. Verify each account via player_api.php (auth=1, Active, has live streams).
  5. Write two outputs (consumed by main.py + iptv_relay.py):
       data/dynamic_upstream_accounts.json  -> merged into _upstream_ACCOUNTS
       data/relay_allowed_hosts.json      -> relay host allowlist

Runs on the self-hosted nightly runner (residential IP) so provider hosts that
block VPS IPs are reachable, and so it can write the files the API + relay read.

Usage:
  uv run python scripts/iptv_freshness.py [--days N] [--max-accounts N] [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

REDDIT_RSS = "https://old.reddit.com/r/REDACTED_SOURCE/new/.rss"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
BASE64_RE = re.compile(rb"[A-Za-z0-9+/=]{40,}")
M3U_URL_RE = re.compile(r"https?://[^\s\"'<>|]+get\.php\?[^\s\"'<>|]+", re.I)
PASTESH_RE = re.compile(r"^https?://paste\.sh/([A-Za-z0-9_-]+)(?:#([A-Za-z0-9_-]+))?")
ATOM = "{http://www.w3.org/2005/Atom}"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ACCOUNTS_OUT = DATA_DIR / "dynamic_upstream_accounts.json"
ALLOWLIST_OUT = DATA_DIR / "relay_allowed_hosts.json"


# ── 1. Reddit scrape ────────────────────────────────────────────────────────
def scrape_reddit(days: int) -> list[dict]:
    import xml.etree.ElementTree as ET
    max_age = days * 86400
    try:
        r = httpx.get(REDDIT_RSS, headers={"User-Agent": UA,
                      "Accept": "application/atom+xml, application/xml, */*"},
                      timeout=20.0, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        print(f"[reddit] fetch failed: {e}", file=sys.stderr)
        return []

    posts: list[dict] = []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        print(f"[reddit] XML parse failed: {e}", file=sys.stderr)
        return []

    now = datetime.now(timezone.utc).timestamp()
    for entry in root.iter(f"{ATOM}entry"):
        pub = entry.findtext(f"{ATOM}published")
        if not pub:
            continue
        try:
            age = now - datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        if age > max_age:
            continue
        post_id = (entry.findtext(f"{ATOM}id") or "")
        m = re.search(r"t3_(\w+)", post_id)
        post_id = m.group(1) if m else post_id[:20]
        content = entry.findtext(f"{ATOM}content") or ""
        blobs = {b.decode() for b in BASE64_RE.findall(content.encode())}
        if blobs:
            posts.append({"post_id": post_id, "blobs": list(blobs)})
    print(f"[reddit] {len(posts)} posts with base64 in last {days}d", file=sys.stderr)
    return posts


# ── 2. Decode credentials (plain base64 + paste.sh) ─────────────────────────
def try_decode(b64: str) -> str | None:
    try:
        decoded = base64.b64decode(b64, validate=False).decode("utf-8", "replace")
    except Exception:
        return None
    # mostly-binary garbage → reject (printable chars should dominate)
    nonprint = re.sub(r"[\x20-\x7e\n\r\t]", "", decoded)
    if len(nonprint) > len(decoded) * 0.1:
        return None
    return decoded.strip()


def decrypt_pastesh(enc_b64: str, paste_id: str, fragment: str) -> str | None:
    """paste.sh v3: OpenSSL Salted__ + AES-256-CBC, key/iv via PBKDF2-HMAC-SHA512."""
    try:
        raw = base64.b64decode(enc_b64)
        salt, ciphertext = raw[8:16], raw[16:]
        password = (paste_id + fragment + "https://paste.sh").encode()
        derived = hashlib.pbkdf2_hmac("sha512", password, salt, 1, 48)
        key, iv = derived[:32], derived[32:48]
        dec = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
        padded = dec.update(ciphertext) + dec.finalize()
        return padded[: -padded[-1]].decode("utf-8", "replace")  # strip PKCS7 pad
    except Exception:
        return None


def fetch_pastesh(url: str, client: httpx.Client) -> list[str]:
    m = PASTESH_RE.match(url)
    if not m:
        return []
    paste_id, fragment = m.group(1), (m.group(2) or "")
    try:
        r = client.get(url.split("#")[0], headers={"User-Agent": UA}, timeout=15.0)
        if r.status_code != 200:
            return []
        cm = re.search(r'name=["\']?content["\']?\s+value=["\']([^"\']+)["\']', r.text)
        if not cm:
            return []
        text = decrypt_pastesh(cm.group(1), paste_id, fragment)
        return M3U_URL_RE.findall(text) if text else []
    except Exception:
        return []


def parse_upstream(url: str) -> tuple[str, int, str, str] | None:
    try:
        p = urlparse(url)
        qs = dict(x.split("=", 1) for x in p.query.split("&") if "=" in x)
        user, pw = qs.get("username", ""), qs.get("password", "")
        if not user or not pw or not p.hostname:
            return None
        return (p.hostname.lower(), p.port or (443 if p.scheme == "https" else 80), user, pw)
    except Exception:
        return None


def decode_credentials(posts: list[dict]) -> list[tuple[str, int, str, str]]:
    seen_blob: set[str] = set()
    accounts: dict[tuple, None] = {}
    with httpx.Client(follow_redirects=True) as client:
        for post in posts:
            for b64 in post["blobs"]:
                if b64 in seen_blob:
                    continue
                seen_blob.add(b64)
                decoded = try_decode(b64)
                if not decoded:
                    continue
                for line in decoded.splitlines():
                    line = line.strip()
                    urls: list[str] = []
                    if PASTESH_RE.match(line):
                        urls = fetch_pastesh(line, client)
                    elif M3U_URL_RE.match(line):
                        urls = [line]
                    for u in urls:
                        acct = parse_upstream(u)
                        if acct:
                            accounts[acct] = None
    print(f"[decode] {len(accounts)} distinct upstream accounts", file=sys.stderr)
    return list(accounts)


# ── 3. Verify accounts via player_api ───────────────────────────────────────
def verify_account(acct: tuple[str, int, str, str], client: httpx.Client) -> dict | None:
    host, port, user, pw = acct
    scheme = "https" if port == 443 else "http"
    base = f"{scheme}://{host}:{port}"
    try:
        r = client.get(f"{base}/player_api.php",
                       params={"username": user, "password": pw},
                       headers={"User-Agent": "VLC/3.0.20"}, timeout=12.0)
        info = r.json().get("user_info", {})
        if info.get("auth") != 1 or str(info.get("status", "")).lower() != "active":
            return None
        # has capacity?
        try:
            act, mx = int(info.get("active_cons") or 0), int(info.get("max_connections") or 0)
            if mx and act >= mx:
                return None
        except (TypeError, ValueError):
            pass
        # has live streams?
        lr = client.get(f"{base}/player_api.php",
                        params={"username": user, "password": pw, "action": "get_live_streams"},
                        headers={"User-Agent": "VLC/3.0.20"}, timeout=20.0)
        streams = lr.json()
        n = len(streams) if isinstance(streams, list) else 0
        if n == 0:
            return None
        return {"host": host, "port": port, "username": user, "password": pw, "live_streams": n}
    except Exception:
        return None


def verify_accounts(accounts: list[tuple], max_keep: int, max_per_host: int = 3) -> list[dict]:
    """Keep up to max_keep working accounts, but no more than max_per_host on any
    single host — diversity across providers beats many creds on one box (which
    Origin's per-host backup cap would collapse anyway, and which all die together
    if that one host goes down)."""
    good: list[dict] = []
    per_host: dict[str, int] = {}
    with httpx.Client(follow_redirects=True) as client:
        for acct in accounts:
            host = acct[0]
            if per_host.get(host, 0) >= max_per_host:
                continue
            res = verify_account(acct, client)
            if res:
                good.append(res)
                per_host[host] = per_host.get(host, 0) + 1
                print(f"[verify] ✅ {res['host']}:{res['port']} ({res['live_streams']} live)", file=sys.stderr)
                if len(good) >= max_keep:
                    break
            else:
                print(f"[verify] ✗ {acct[0]}:{acct[1]}", file=sys.stderr)
    return good


# ── 4. Write outputs ────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--max-accounts", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    posts = scrape_reddit(args.days)
    accounts = decode_credentials(posts) if posts else []

    # ── Account accumulation ──────────────────────────────────────────────
    # Reddit creds rotate, so each scrape finds a DIFFERENT set; replacing the
    # account file every run made working VOD providers (e.g. thebiddox.xyz)
    # vanish the next night ("it worked before, now it's gone"). Instead, merge
    # the newly-scraped accounts with the already-DEPLOYED ones (the runner IS
    # the prod box) and re-verify the whole union — working accounts persist,
    # dead ones drop, new ones get added. Bounded by the keep cap below.
    DEPLOYED_ACCOUNTS = Path("/app/gretzky/data/dynamic_upstream_accounts.json")
    prior: list[tuple] = []
    for src in (DEPLOYED_ACCOUNTS, ACCOUNTS_OUT):
        try:
            for row in json.loads(src.read_text()):
                # rows are [label, host, port, user, pw]; verify wants (host,port,user,pw)
                if isinstance(row, list) and len(row) >= 5:
                    prior.append((row[1], int(row[2]), row[3], row[4]))
        except (OSError, ValueError, TypeError):
            continue
    # Dedup the union. PRIOR accounts go FIRST: verify_accounts stops at the keep
    # cap, so verifying known-good accumulated accounts first guarantees they
    # persist run-to-run; freshly-scraped accounts then fill the remaining slots.
    seen, pool = set(), []
    for acct in prior + accounts:
        key = (acct[0], acct[1], acct[2], acct[3])
        if key not in seen:
            seen.add(key)
            pool.append(acct)
    print(f"[accumulate] {len(prior)} prior + {len(accounts)} scraped = {len(pool)} to verify",
          file=sys.stderr)
    if not pool:
        print("No accounts to verify — nothing to do.", file=sys.stderr)
        return 0
    # Keep more than a single run would (so accumulated good accounts survive).
    # The VOD-index build no longer bounds this: the API indexes VOD from a capped
    # subset (_VOD_ACCOUNTS = _upstream_ACCOUNTS[:_VOD_MAX_ACCOUNTS], dashboard/api/
    # main.py) while live + EPG use the FULL pool. So extra accounts deepen live
    # per-channel failover without re-bloating the VOD warm that emptied movies at
    # ~20 accounts. Bounded only to keep the file + relay allowlist sane. If you
    # raise this, keep _VOD_MAX_ACCOUNTS under ~20.
    keep = max(args.max_accounts, 24)
    good = verify_accounts(pool, keep)
    print(f"\n[result] {len(good)} working accounts of {len(pool)} (scraped+accumulated)", file=sys.stderr)

    # Account tuples for _upstream_ACCOUNTS: (label, host, port, username, password)
    acct_rows = [[f"fresh-{g['host']}", g["host"], g["port"], g["username"], g["password"]]
                 for g in good]
    hosts = sorted({g["host"] for g in good})

    if args.dry_run:
        print(json.dumps({"accounts": acct_rows, "hosts": hosts}, indent=2))
        return 0

    if not good:
        print("No working accounts — leaving existing files untouched (don't replace good with empty).",
              file=sys.stderr)
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ACCOUNTS_OUT.write_text(json.dumps(acct_rows, indent=2) + "\n")
    # Merge hosts with any existing allowlist (union — don't drop hosts other runs added).
    existing: set[str] = set()
    if ALLOWLIST_OUT.exists():
        try:
            existing = {str(h).lower() for h in json.loads(ALLOWLIST_OUT.read_text())}
        except (OSError, ValueError):
            pass
    merged = sorted(existing | set(hosts))
    ALLOWLIST_OUT.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"[write] {ACCOUNTS_OUT.name}: {len(acct_rows)} accounts; "
          f"{ALLOWLIST_OUT.name}: {len(merged)} hosts", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
