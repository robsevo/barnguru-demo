"""
IPTV residential relay — bypasses VPS IP blocking on upstream providers.

Run on a machine with a residential IP (laptop, later Pi). Expose publicly via
ngrok or Cloudflare Tunnel. The localhost:8000 backend rewrites upstream URLs
through here when IPTV_LOCAL_PROXY_URL is set, so the provider sees a
residential IP instead of our VPS.

Caches HLS segments briefly (30s / ~300 MB). 7 viewers on the same channel
therefore cost 1 upstream fetch per segment — protects both your residential
upload budget and the provider's per-account concurrent-session cap.

Usage:
    uv run python scripts/iptv_relay.py                # or set IPTV_RELAY_PORT
    ngrok http 9000                                     # pin the tunnel URL
    # on localhost:8000:
    #   systemctl edit Origin-guru-api.service
    #   Environment="IPTV_LOCAL_PROXY_URL=https://abc.ngrok-free.app"
    #   Environment="IPTV_RELAY_TOKEN=<same_as_below>"    (optional)
"""

from __future__ import annotations

import os
import re
import time
from collections import OrderedDict
from urllib.parse import quote, unquote, urljoin, urlparse

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

PORT          = int(os.environ.get("IPTV_RELAY_PORT", "9000"))
TOKEN         = os.environ.get("IPTV_RELAY_TOKEN") or None
CACHE_BYTES   = int(os.environ.get("IPTV_RELAY_CACHE_MB", "300")) * 1024 * 1024
CACHE_TTL_S   = 30.0
FETCH_TIMEOUT = 20.0

# Defence-in-depth: tunnel is publicly reachable, so refuse any URL not
# pointing at one of the upstream hosts wired in dashboard/api/main.py.
# Some providers 302-redirect to sub-hosts for CDNs / VOD shards
# (e.g. vod.tv14s.xyz, cdn.an upstream host.ddns.net), so we accept any host
# that either equals the base domain or ends with a dotted suffix of it.
ALLOWED_HOSTS = {
    "an upstream host.ddns.net",
    "tv14s.xyz",
    "ampztl.xyz",
    "lunar.pm",
}
ALLOWED_SUFFIXES = tuple(f".{h}" for h in ALLOWED_HOSTS)


class _LRU:
    def __init__(self, max_bytes: int) -> None:
        self._d: OrderedDict[str, tuple[float, bytes, str]] = OrderedDict()
        self._bytes = 0
        self._max = max_bytes

    def get(self, k: str) -> tuple[bytes, str] | None:
        v = self._d.get(k)
        if v is None:
            return None
        ts, data, ct = v
        if time.monotonic() - ts > CACHE_TTL_S:
            self._d.pop(k, None)
            self._bytes -= len(data)
            return None
        self._d.move_to_end(k)
        return data, ct

    def put(self, k: str, data: bytes, content_type: str) -> None:
        old = self._d.pop(k, None)
        if old is not None:
            self._bytes -= len(old[1])
        self._d[k] = (time.monotonic(), data, content_type)
        self._bytes += len(data)
        while self._bytes > self._max and self._d:
            _, (_, old_data, _) = self._d.popitem(last=False)
            self._bytes -= len(old_data)


app = FastAPI()
seg_cache = _LRU(CACHE_BYTES)


def _check_token(request: Request) -> None:
    if TOKEN is None:
        return
    provided = request.query_params.get("t") or request.headers.get("x-relay-token")
    if provided != TOKEN:
        raise HTTPException(status_code=403, detail="bad token")


def _check_host(url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    if host in ALLOWED_HOSTS or host.endswith(ALLOWED_SUFFIXES):
        return
    raise HTTPException(status_code=400, detail=f"host not allowed: {host}")


def _tunnel_base(request: Request) -> str:
    # ngrok + Cloudflare Tunnel both set x-forwarded-host/proto.
    host  = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    proto = request.headers.get("x-forwarded-proto", "https")
    return f"{proto}://{host}"


_KEY_URI_RE = re.compile(r'URI="([^"]+)"')


def _rewrite_m3u8(body: str, base_url: str, tunnel_base: str, token_q: str) -> str:
    """Rewrite every segment / sub-playlist / key URI back through this relay."""
    out: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#EXT-X-KEY") and "URI=" in stripped:
            m = _KEY_URI_RE.search(stripped)
            if m:
                key_abs = urljoin(base_url, m.group(1))
                new_uri = f"{tunnel_base}/ts?u={quote(key_abs, safe='')}{token_q}"
                line = stripped.replace(m.group(0), f'URI="{new_uri}"')
            out.append(line)
        elif stripped and not stripped.startswith("#"):
            abs_url = urljoin(base_url, stripped)
            endpoint = "m3u8" if ".m3u8" in abs_url.lower().split("?", 1)[0] else "ts"
            out.append(f"{tunnel_base}/{endpoint}?u={quote(abs_url, safe='')}{token_q}")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


_CORS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
}


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "cached_bytes": seg_cache._bytes, "cached_items": len(seg_cache._d)}


@app.get("/m3u8")
async def proxy_m3u8(u: str, request: Request) -> Response:
    _check_token(request)
    url = unquote(u)
    _check_host(url)

    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True) as cl:
            r = await cl.get(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:200])

    token_q = f"&t={TOKEN}" if TOKEN else ""
    rewritten = _rewrite_m3u8(r.text, str(r.url), _tunnel_base(request), token_q)
    return Response(
        content=rewritten,
        media_type="application/vnd.apple.mpegurl",
        headers={**_CORS, "Cache-Control": "no-cache"},
    )


@app.get("/ts")
async def proxy_ts(u: str, request: Request) -> Response:
    # Segments are served from upstream-controlled CDN hosts (e.g. lb58.xxip9.top
    # for tv14s). The host allowlist is only enforced on /m3u8 (where the URL is
    # user-supplied); on /ts the token — or lack of a publicly-reachable tunnel —
    # is the access control.
    _check_token(request)
    url = unquote(u)

    # Some providers chain .m3u8s through paths that this relay can't guess
    # at rewrite time. If the URL is actually a playlist, re-route (and restore
    # host check — that path does fetch a manifest).
    if ".m3u8" in url.lower().split("?", 1)[0]:
        return await proxy_m3u8(u=u, request=request)

    cached = seg_cache.get(url)
    if cached is not None:
        data, ct = cached
        return Response(content=data, media_type=ct, headers={**_CORS, "X-Cache": "HIT"})

    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True) as cl:
            r = await cl.get(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code)

    data = r.content
    ct   = r.headers.get("content-type") or "video/mp2t"
    seg_cache.put(url, data, ct)
    return Response(content=data, media_type=ct, headers={**_CORS, "X-Cache": "MISS"})


if __name__ == "__main__":
    print(f"[iptv-relay] port {PORT} — token: {'SET' if TOKEN else 'none (set IPTV_RELAY_TOKEN for auth)'}")
    print(f"[iptv-relay] cache: {CACHE_BYTES // 1024 // 1024} MB / {CACHE_TTL_S}s TTL")
    print(f"[iptv-relay] allowed hosts: {sorted(ALLOWED_HOSTS)}")
    print(f"[iptv-relay] next step: `ngrok http {PORT}` (or `cloudflared tunnel …`)")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
