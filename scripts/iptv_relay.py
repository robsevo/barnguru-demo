"""
IPTV residential relay — bypasses VPS IP blocking on upstream providers.

Run on a machine with a residential IP (laptop, later Pi). Expose publicly via
a Cloudflare Tunnel (cloudflared). The localhost:8000 backend rewrites upstream
URLs through here when IPTV_LOCAL_PROXY_URL is set, so the provider sees a
residential IP instead of our VPS.

Caches HLS segments briefly (30s / ~300 MB). 7 viewers on the same channel
therefore cost 1 upstream fetch per segment — protects both your residential
upload budget and the provider's per-account concurrent-session cap.

Usage:
    uv run python scripts/iptv_relay.py                # or set IPTV_RELAY_PORT
    cloudflared tunnel run iptv-relay                  # public URL via localhost:3000
    # IPTV_LOCAL_PROXY_URL + IPTV_RELAY_TOKEN are deployed via the
    # IPTV_ENV_BLOCK GitHub secret → dashboard/api/iptv.env (see
    # .github/workflows/nightly.yml). Update the secret + redeploy if the
    # tunnel URL ever changes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlparse

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

PORT          = int(os.environ.get("IPTV_RELAY_PORT", "9000"))
TOKEN         = os.environ.get("IPTV_RELAY_TOKEN") or None
CACHE_BYTES   = int(os.environ.get("IPTV_RELAY_CACHE_MB", "300")) * 1024 * 1024
CACHE_TTL_S   = 30.0
FETCH_TIMEOUT = 20.0

# TS→HLS transmux: ampztl and other upstream accounts that only serve raw MPEG-TS
# can't be played by hls.js directly. /hls spawns ffmpeg to repackage the TS
# into a rolling HLS playlist on disk; the browser consumes the manifest and
# segments from this relay.
#
# Quality tiers: /hls?q=720p|480p re-encodes to a capped bitrate so users on
# weak connections don't sit on buffer-empty spinners. q=passthrough (or
# missing q) keeps the original `-c copy` behavior. Software libx264 veryfast
# costs ~30-50% of one core per 720p stream; hardware encoders (NVENC, QSV,
# VideoToolbox) drop that to <5%. The startup log names the chosen encoder so
# you can sanity-check which path is in use before relying on it for many
# concurrent feeds.
HLS_WORKDIR            = Path("/tmp/iptv_relay_hls")
# 6s segments (was 3s): on a loaded shared box, longer segments give ffmpeg
# HALF as many hard real-time deadlines to hit per minute and the player half
# as many segment fetches + manifest refreshes — each of which is a place a
# stall can start. Resilience-first (Apple's authoring-spec default is 6s
# too); the cost is ~3s more channel-change latency since the first segment
# takes longer to finalize. GOP=60 (≈2s) still divides 6s cleanly (3 GOPs).
HLS_SEGMENT_SECONDS    = 6
# 18 segments × 6s = 108s manifest window — unchanged window, so the player's
# liveSyncDuration=36 (seconds-based, segment-size-agnostic) still sits 36s
# behind live with ~72s of look-back room so a recovery seek can't fall off
# the back of the deleted-segments cliff. Disk cost ~36 MB per session.
HLS_LIST_SIZE          = 18
# Tab-switch / ad-break tolerance: 30s was killing sessions when users
# briefly looked away, forcing a full ffmpeg respawn on resume. 90s is still
# tight enough that abandoned chips don't pile up but covers normal viewer
# behavior (phone call, brief tab switch, talking with someone in the room).
# OOM protection comes from HLS_MAX_SESSIONS below, not this idle timeout —
# 12 × ~80 MB resident keeps the relay under 1 GB regardless of how long
# each session lives.
HLS_IDLE_TIMEOUT_S     = 90.0
# 20s (was 15s): a 6s-segment session needs ~6-8s of wall time to finalize its
# first segment before a manifest exists, so a 15s budget left little margin on
# a busy box and risked a premature "ffmpeg did not produce manifest in time"
# kill → respawn loop. 20s keeps abandoned-chip cleanup tight while giving the
# longer first segment room to land.
HLS_STARTUP_TIMEOUT_S  = 20.0
# Hard cap on concurrent ffmpeg sessions. Each ffmpeg holds 60-80 MB
# resident; on a 2 GB VPS even 25 sessions saturates RAM and the OOM
# killer takes the relay (or worse, a sibling service) down. Capping
# at 12 leaves headroom for the API workers + python heap and lets
# the LRU drop stale sessions instead of accumulating until OOM.
# When the cap is reached, the LEAST-recently-used session is killed
# before the new one starts. Active viewers are unaffected; only
# abandoned channel-hop / verifier-probe leftovers get evicted.
HLS_MAX_SESSIONS       = 12

# (width, height, video_bitrate, video_bufsize, audio_bitrate)
_QUALITY_TIERS: dict[str, tuple[int, int, str, str, str]] = {
    "720p": (1280, 720, "2500k", "5000k", "128k"),
    "480p": (854,  480, "1200k", "2400k", "96k"),
}
_VALID_QUALITIES = frozenset({"720p", "480p", "passthrough"})

_ENCODER: str | None = None


def _probe_encoder(enc: str) -> bool:
    """Run a 1-frame test encode to confirm the encoder actually works.

    Hardware encoders often appear in `ffmpeg -encoders` because they are
    compiled in, but fail at runtime when the matching driver/runtime isn't
    installed (e.g. h264_nvenc with no libcuda.so.1, h264_qsv with no iHD
    driver). A 1-frame null-mux is the cheapest way to find that out.
    """
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=black:s=64x64:d=0.05",
             "-c:v", enc, "-frames:v", "1",
             "-f", "null", "-"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _detect_encoder() -> str:
    """Pick the fastest *actually-working* H.264 encoder. libx264 always works."""
    if not shutil.which("ffmpeg"):
        return "libx264"
    for enc in ("h264_videotoolbox", "h264_nvenc", "h264_qsv"):
        if _probe_encoder(enc):
            return enc
    return "libx264"


def _get_encoder() -> str:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = _detect_encoder()
    return _ENCODER


def _encode_args(quality: str, encoder: str) -> list[str]:
    """ffmpeg args between `-i <url>` and the HLS muxer for the given tier.

    Audio is always re-encoded to AAC — stream-copying audio while re-encoding
    video drifts apart on long-running live streams. `-sc_threshold` is
    libx264-only; nvenc/qsv reject it and refuse to start, so each encoder
    branch picks its own scene-cut control.
    """
    if quality == "passthrough":
        # Video pass-through, audio normalized to AAC. Some upstream TS feeds
        # ship audio (MP2, AC3, ADTS-AAC variants) that MediaSource refuses to
        # append, surfacing as bufferAppendError in hls.js. AAC re-encode is
        # cheap (~2-5% of one core) and guarantees MSE-compatible audio.
        return ["-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-ac", "2"]

    width, height, vbr, vbufsize, abr = _QUALITY_TIERS[quality]
    common_audio = ["-c:a", "aac", "-b:a", abr, "-ac", "2"]
    common_rate  = ["-b:v", vbr, "-maxrate", vbr, "-bufsize", vbufsize]
    scale        = ["-vf", f"scale={width}:{height}"]
    # GOP = 60 frames ≈ 2s at 30fps (2.4s at 25fps). With HLS_SEGMENT_SECONDS=6
    # this puts 3 keyframes inside every segment and divides evenly, so segment
    # boundaries stay keyframe-aligned. Was -g 96 (3.2s GOP) which frequently
    # misaligned with the segment boundary and added 0.5-1.5s of first-paint
    # latency.
    gop          = ["-g", "60", "-keyint_min", "60"]

    if encoder == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-profile:v", "main",
                *gop, *scale, *common_rate, *common_audio]
    if encoder == "h264_nvenc":
        # -bf 0 + -rc-lookahead 0 mirror libx264's -tune zerolatency: zero
        # B-frames and zero lookahead cut first-segment latency to roughly
        # match the libx264 path. -preset p4 -tune ll already targets
        # low-latency, but B-frames remain on by default.
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll",
                "-profile:v", "main", "-no-scenecut", "1",
                "-bf", "0", "-rc-lookahead", "0",
                *gop, *scale, *common_rate, *common_audio]
    if encoder == "h264_qsv":
        # Same zero-latency posture as nvenc — qsv accepts -bf 0 and
        # -look_ahead 0 (qsv-specific name for lookahead).
        return ["-c:v", "h264_qsv", "-preset", "veryfast",
                "-profile:v", "main",
                "-bf", "0", "-look_ahead", "0",
                *gop, *scale, *common_rate, *common_audio]
    # libx264 fallback. -sc_threshold 0 stops scene-cut from breaking
    # constant-GOP, which keeps HLS segment boundaries aligned. -tune
    # zerolatency removes B-frames + tightens lookahead so the first
    # segment finalizes ~30-50 ms sooner — meaningful when the manifest
    # poll loop in /hls is waiting for it on cold start.
    return ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
            "-profile:v", "main",
            "-sc_threshold", "0",
            *gop, *scale, *common_rate, *common_audio]

# Defence-in-depth: tunnel is publicly reachable, so refuse any URL not
# pointing at one of the upstream hosts wired in dashboard/api/main.py.
# Some providers 302-redirect to sub-hosts for CDNs / VOD shards
# (e.g. vod.tv14s.xyz, cdn.an upstream host.ddns.net), so we accept any host
# that either equals the base domain or ends with a dotted suffix of it.
ALLOWED_HOSTS = {
    "an upstream host.ddns.net",
    "ampztl.xyz",
    "tvpass.org",
    "thetvapp.to",
    "kstv.us",
    "bgdc.live",
    "an upstream host.an upstream host.co",
}
ALLOWED_SUFFIXES = tuple(f".{h}" for h in ALLOWED_HOSTS)

# Dynamic allowlist: the link-freshness pipeline verifies fresh upstream hosts and
# writes them here, so newly-scraped providers can be relayed without a code edit
# + redeploy. Still token-gated (_check_token), so this only widens which hosts a
# token-bearing caller may proxy — not an open proxy. File is a JSON list of
# hostnames: ["mundo2.pro", "host2.tld", ...]. Reloaded on mtime change (cheap).
DYNAMIC_HOSTS_FILE = Path(
    os.environ.get(
        "IPTV_RELAY_DYNAMIC_HOSTS",
        str(Path(__file__).resolve().parent.parent / "data" / "relay_allowed_hosts.json"),
    )
)
_dyn_cache: tuple[float, frozenset[str], tuple[str, ...]] = (0.0, frozenset(), ())

# Standard HTTP status codes Starlette/HTTPException will accept. Upstream IPTV
# panels sometimes return non-standard codes (e.g. hottest.plus 456 = conn-limit)
# that crash HTTPException; we map anything not in here to 502.
from http import HTTPStatus as _HTTPStatus
_VALID_HTTP = frozenset(s.value for s in _HTTPStatus)


def _dynamic_hosts() -> tuple[frozenset[str], tuple[str, ...]]:
    """(hosts, suffixes) from the pipeline-written allowlist, cached by mtime."""
    global _dyn_cache
    try:
        mtime = DYNAMIC_HOSTS_FILE.stat().st_mtime
    except OSError:
        return frozenset(), ()
    if mtime != _dyn_cache[0]:
        try:
            raw = json.loads(DYNAMIC_HOSTS_FILE.read_text())
            hosts = frozenset(str(h).lower().strip() for h in raw if h)
            suffixes = tuple(f".{h}" for h in hosts)
            _dyn_cache = (mtime, hosts, suffixes)
        except (OSError, ValueError):
            return _dyn_cache[1], _dyn_cache[2]
    return _dyn_cache[1], _dyn_cache[2]


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
# CORS: hls.js fetches segments from this relay across origins (the player
# runs on localhost:3000 / localhost:8000, the relay sits at localhost:8000 — a
# different origin). Allow all so cross-origin segment reads work.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)
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
    dyn_hosts, dyn_suffixes = _dynamic_hosts()
    if host in dyn_hosts or (dyn_suffixes and host.endswith(dyn_suffixes)):
        return
    raise HTTPException(status_code=400, detail=f"host not allowed: {host}")


def _tunnel_base(request: Request) -> str:
    # Cloudflare Tunnel sets x-forwarded-host/proto so we can reconstruct
    # the public URL even when the relay binds to localhost.
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

_UPSTREAM_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# Pooled httpx client for /m3u8 and /ts proxying. The previous code opened
# a fresh AsyncClient inside every handler, paying a TCP+TLS handshake
# (~100-300 ms) on EVERY segment fetch from the upstream upstream/CDN. With
# 3-second segments and concurrent viewers, that handshake cost was draining
# the player buffer the same way q=720p re-encoding used to. Native IPTV
# players (XCIPTV) don't have this problem because they hold a persistent
# connection. Pooled keep-alive matches that behaviour.
_UPSTREAM_HTTP: httpx.AsyncClient | None = None


# --- KSTV proxy rotation -----------------------------------------------------
# kstv firewalls our VPS IP at TCP level. Need a residential/3rd-party-datacenter
# proxy to reach the auth host — but kstv aggressively blocks individual proxy
# IPs once they're seen doing upstream auth. Rotation buys lifetime: hold a list,
# round-robin per request, on connection-error / 502 / 503 mark the IP cooled
# off for KSTV_PROXY_COOLDOWN_S seconds and try the next one. When all IPs are
# cooling, fall back to direct (will fail, but no worse than no proxy).
#
# KSTV_PROXY_LIST format: comma-separated `http://USER:PASS@HOST:PORT` URLs.
# Backward-compatible: KSTV_PROXY_URL (singular) still works as a 1-item list.
# Only the kstv.us *auth host* needs the proxy; the CDN it redirects to is
# reachable direct from the VPS, so segments stream without proxy bandwidth.
KSTV_PROXY_COOLDOWN_S = 300.0
_kstv_proxy_state: dict[str, float] = {}    # proxy_url -> cool-until-timestamp
_kstv_proxy_idx = 0
_kstv_transports: dict[str, httpx.AsyncHTTPTransport] = {}


def _kstv_proxy_pool() -> list[str]:
    """Parse KSTV_PROXY_LIST (or fall back to KSTV_PROXY_URL)."""
    raw = os.environ.get("KSTV_PROXY_LIST") or os.environ.get("KSTV_PROXY_URL") or ""
    return [p.strip() for p in raw.split(",") if p.strip()]


def _kstv_get_transport() -> httpx.AsyncHTTPTransport | None:
    """Pick the next not-cooling proxy and return a cached transport for it.
    Returns None when no proxy is configured or all are cooling.
    """
    global _kstv_proxy_idx
    pool = _kstv_proxy_pool()
    if not pool:
        return None
    now = time.monotonic()
    # Walk the list once from the rotation index; first non-cooling wins.
    for offset in range(len(pool)):
        idx = (_kstv_proxy_idx + offset) % len(pool)
        proxy = pool[idx]
        cool_until = _kstv_proxy_state.get(proxy, 0.0)
        if now >= cool_until:
            _kstv_proxy_idx = (idx + 1) % len(pool)
            t = _kstv_transports.get(proxy)
            if t is None:
                t = httpx.AsyncHTTPTransport(proxy=proxy)
                _kstv_transports[proxy] = t
            return t
    return None  # all cooling


def _kstv_mark_bad(transport: httpx.AsyncHTTPTransport) -> None:
    """Find which proxy URL backs this transport and put it on cooldown."""
    for url, t in _kstv_transports.items():
        if t is transport:
            _kstv_proxy_state[url] = time.monotonic() + KSTV_PROXY_COOLDOWN_S
            return


class _KstvRotatingTransport(httpx.AsyncBaseTransport):
    """A transport that pulls a fresh upstream transport from the rotating
    pool on every request and demotes it on failure. Mounted only on
    http(s)://kstv.us via httpx mounts so other hosts are unaffected.
    """
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        last_exc: Exception | None = None
        # Try every still-available proxy once before giving up.
        for _ in range(max(1, len(_kstv_proxy_pool()))):
            transport = _kstv_get_transport()
            if transport is None:
                break
            try:
                resp = await transport.handle_async_request(request)
                # 502/503 from the proxy itself (Webshare returns these when
                # the destination is blocked at the egress) → demote and retry.
                if resp.status_code in (502, 503):
                    await resp.aclose()
                    _kstv_mark_bad(transport)
                    continue
                return resp
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.RemoteProtocolError) as e:
                _kstv_mark_bad(transport)
                last_exc = e
                continue
        if last_exc is not None:
            raise last_exc
        # All proxies cooling and no exception — surface a 502 so the caller
        # logs an error rather than hanging.
        raise httpx.ConnectError("all kstv proxies on cooldown")

    async def aclose(self) -> None:
        for t in list(_kstv_transports.values()):
            try:
                await t.aclose()
            except Exception:
                pass
        _kstv_transports.clear()


async def _get_upstream_http() -> httpx.AsyncClient:
    global _UPSTREAM_HTTP
    if _UPSTREAM_HTTP is None:
        mounts = None
        if _kstv_proxy_pool():
            rotating = _KstvRotatingTransport()
            mounts = {
                "http://kstv.us": rotating,
                "https://kstv.us": rotating,
            }
        _UPSTREAM_HTTP = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=4.0, read=FETCH_TIMEOUT, write=FETCH_TIMEOUT, pool=5.0),
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=100,
                keepalive_expiry=30.0,
            ),
            follow_redirects=True,
            headers={"User-Agent": _UPSTREAM_UA},
            mounts=mounts,
        )
    return _UPSTREAM_HTTP


@app.on_event("startup")
async def _start_upstream_http() -> None:
    await _get_upstream_http()


@app.on_event("shutdown")
async def _stop_upstream_http() -> None:
    global _UPSTREAM_HTTP
    if _UPSTREAM_HTTP is not None:
        await _UPSTREAM_HTTP.aclose()
        _UPSTREAM_HTTP = None


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "cached_bytes": seg_cache._bytes, "cached_items": len(seg_cache._d)}


@app.get("/upstream-json")
async def proxy_upstream_json(u: str, request: Request) -> Response:
    """Proxy an upstream Codes JSON catalog call (player_api.php, get.php).

    Some upstream panels (kstv.us today, possibly others tomorrow) firewall
    the VPS IP at their edge — channel enumeration fails with
    "All connection attempts failed" even though stream playback through
    the relay works fine. This endpoint lets localhost:8000 fetch the JSON
    catalog from the relay's residential IP, just like it already routes
    playback URLs through /m3u8 and /hls. Body is returned as-is; the API
    parses it (no rewriting needed since stream URLs in the catalog get
    rewritten via _rewrite_iptv_url after parsing).
    """
    _check_token(request)
    url = unquote(u)
    _check_host(url)

    try:
        cl = await _get_upstream_http()
        r = await cl.get(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream: {e}")
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type") or "application/json",
        headers={**_CORS, "Cache-Control": "no-cache"},
    )


@app.get("/m3u8")
async def proxy_m3u8(u: str, request: Request) -> Response:
    _check_token(request)
    url = unquote(u)
    _check_host(url)

    try:
        cl = await _get_upstream_http()
        r = await cl.get(url)
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
        return Response(
            content=data,
            media_type=ct,
            headers={**_CORS, "X-Cache": "HIT", "Cache-Control": "public, max-age=60, immutable"},
        )

    try:
        cl = await _get_upstream_http()
        r = await cl.get(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code)

    data = r.content
    ct   = r.headers.get("content-type") or "video/mp2t"
    seg_cache.put(url, data, ct)
    return Response(
        content=data,
        media_type=ct,
        headers={**_CORS, "X-Cache": "MISS", "Cache-Control": "public, max-age=60, immutable"},
    )


@app.get("/vod-debug")
async def vod_debug() -> Response:
    """TEMP: report what the relay actually sees for the dynamic allowlist."""
    hosts, suffixes = _dynamic_hosts()
    info = {
        "dynamic_hosts_file": str(DYNAMIC_HOSTS_FILE),
        "file_exists": DYNAMIC_HOSTS_FILE.exists(),
        "dynamic_hosts": sorted(hosts),
        "static_allowed": sorted(ALLOWED_HOSTS),
        "hottest_allowed": "hottest.plus" in hosts or "hottest.plus" in ALLOWED_HOSTS,
    }
    return Response(content=json.dumps(info, indent=2), media_type="application/json")


@app.get("/vod")
async def proxy_vod(u: str, request: Request) -> Response:
    """Range-aware passthrough for VOD movie/episode files (mp4/mkv/avi) from
    allowlisted upstream hosts — same relay path as live TV, so films play in the
    browser over https without the VPS-block / mixed-content problems. Streams
    bytes (no full-file buffering) and forwards the client's Range header so the
    player can seek and start instantly. Token-gated + host-allowlisted."""
    try:
        return await _proxy_vod_impl(u, request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"vod upstream: {e}")


async def _proxy_vod_impl(u: str, request: Request) -> Response:
    _check_token(request)
    url = unquote(u)
    _check_host(url)

    fwd = {"User-Agent": _UPSTREAM_UA, "Accept-Encoding": "identity"}
    rng = request.headers.get("range")
    if rng:
        fwd["Range"] = rng

    # Dedicated client per VOD stream (NOT the shared _get_upstream_http, whose
    # short read-timeout is tuned for tiny live segments and would abort a movie,
    # and whose pooled connection can close when this handler returns). The
    # client + open stream are kept alive by opening them INSIDE the body
    # generator and closing in its finally — so the connection lives exactly as
    # long as the StreamingResponse consumes it.
    client = httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0),
    )

    # Open upstream first so we can surface a clean status before streaming.
    try:
        req = client.build_request("GET", url, headers=fwd)
        upstream = await client.send(req, stream=True)
    except Exception as e:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"upstream: {e}")

    if upstream.status_code not in (200, 206):
        code = upstream.status_code
        await upstream.aclose()
        await client.aclose()
        # Map non-standard upstream codes (e.g. hottest.plus returns 456 for
        # connection-limit/geo-block) to 502 — Starlette rejects status codes
        # outside the standard set, which was itself crashing into a 500.
        safe = code if 400 <= code <= 599 and code in _VALID_HTTP else 502
        raise HTTPException(status_code=safe, detail=f"upstream {code}")

    passthru = {**_CORS, "Accept-Ranges": "bytes"}
    for hh in ("content-type", "content-length", "content-range"):
        if hh in upstream.headers:
            passthru[hh] = upstream.headers[hh]

    async def _body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        _body(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type") or "video/mp4",
        headers=passthru,
    )


# ---------------------------------------------------------------------------
# /hls — TS→HLS transmux for accounts that only serve raw MPEG-TS
# ---------------------------------------------------------------------------

class _HLSSession:
    __slots__ = ("id", "url", "workdir", "proc", "last_access", "started_at",
                 "has_served_segment", "mode")

    def __init__(self, sid: str, url: str, workdir: Path, mode: str = "live") -> None:
        self.id          = sid
        self.url         = url
        self.workdir     = workdir
        self.proc: subprocess.Popen[bytes] | None = None
        self.last_access = time.monotonic()
        self.started_at  = time.monotonic()
        # Flip True once /hls-seg actually serves a fragment from this session.
        # Warmup-only sessions (hover spawn, never clicked) stay False and are
        # the first to be LRU-evicted under capacity pressure.
        self.has_served_segment = False
        # "live" (channel transmux) or "vod" (remux fallback). LIVE ALWAYS WINS:
        # vod sessions have their own low cap and are evicted first — a movie
        # fallback must never starve or displace someone's live channel.
        self.mode = mode


_HLS_SESSIONS:  dict[str, _HLSSession] = {}
_HLS_LOCK       = asyncio.Lock()
_HLS_REAPER: asyncio.Task[None] | None = None
_SEG_NAME_RE    = re.compile(r"^seg\d{4,}\.ts$")
_BROWSER_UA     = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# sid → (url, quality, mode). Outlives the _HLS_SESSIONS entry so that /hls-seg can
# lazy-respawn a session that was LRU-evicted while the player still had its
# segment URLs queued. Without this, an evicted session forces hls.js to wait
# for the next manifest poll (3-6s) before recovering — the lazy respawn cuts
# that to ~1-2s on next segment request. Bounded so a long-running relay
# doesn't accumulate every URL it has ever seen.
_HLS_URL_BY_SID: "OrderedDict[str, tuple[str, str, str]]" = OrderedDict()
_HLS_URL_MAP_MAX = 200


def _remember_hls_url(sid: str, url: str, quality: str, mode: str = "live", start: float = 0.0) -> None:
    _HLS_URL_BY_SID[sid] = (url, quality, mode, start)
    _HLS_URL_BY_SID.move_to_end(sid)
    while len(_HLS_URL_BY_SID) > _HLS_URL_MAP_MAX:
        _HLS_URL_BY_SID.popitem(last=False)


def _hls_session_id(url: str, quality: str = "passthrough", mode: str = "live", start: float = 0.0, aidx: "int | None" = None) -> str:
    # Live sids keep the historical tag shape so in-flight sessions survive a
    # relay restart/deploy without a key change; vod gets its own namespace
    # (keyed by start offset AND chosen audio track — different resume points or
    # languages are distinct sessions, so switching audio spawns a fresh remux).
    tag = f"{quality}|{url}" if mode == "live" else f"{mode}|{quality}|{start:g}|a{aidx if aidx is not None else 'auto'}|{url}"
    return hashlib.sha1(tag.encode()).hexdigest()[:16]


async def _ensure_hls_reaper() -> None:
    global _HLS_REAPER
    if _HLS_REAPER is None or _HLS_REAPER.done():
        _HLS_REAPER = asyncio.create_task(_hls_reaper())


async def _hls_reaper() -> None:
    while True:
        await asyncio.sleep(10)
        try:
            now = time.monotonic()
            expired: list[_HLSSession] = []
            async with _HLS_LOCK:
                for sid, sess in list(_HLS_SESSIONS.items()):
                    if now - sess.last_access > HLS_IDLE_TIMEOUT_S:
                        expired.append(sess)
                        del _HLS_SESSIONS[sid]
            for sess in expired:
                _kill_session(sess)
        except Exception:
            # Reaper must never die — keep looping
            continue


def _kill_session(sess: _HLSSession) -> None:
    if sess.proc is not None and sess.proc.poll() is None:
        sess.proc.terminate()
        try:
            sess.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            sess.proc.kill()
    shutil.rmtree(sess.workdir, ignore_errors=True)


def _probe_video_codec(url: str) -> "str | None":
    """codec_name of the first video stream (h264/hevc/…), None on failure.
    Bounded: reads only the container header over HTTP."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0",
             "-user_agent", _BROWSER_UA, url],
            capture_output=True, timeout=10,
        )
        lines = r.stdout.decode(errors="replace").strip().splitlines()
        return lines[0].strip() or None if lines else None
    except Exception:
        return None


_ENGLISH_LANG_TAGS = {"eng", "en", "english", "en-us", "en-gb"}


# Audio/duration probe results by URL. The player now asks for this on EVERY
# remux play (duration is what enables seeking), and English auto-detect wants
# the same data, so an uncached probe would mean two header reads against a slow
# panel per play. A file's track layout and runtime don't change under us.
_AUDIO_META_CACHE: "dict[str, tuple[float, dict]]" = {}
_AUDIO_META_TTL = 3600.0
_AUDIO_META_MAX = 200


def _probe_audio_meta(url: str) -> dict:
    hit = _AUDIO_META_CACHE.get(url)
    if hit is not None and time.monotonic() - hit[0] <= _AUDIO_META_TTL:
        return hit[1]
    meta = _probe_audio_meta_uncached(url)
    # Don't cache a total failure — a panel hiccup shouldn't pin "no tracks,
    # no duration" (i.e. no seeking) for an hour.
    if meta["tracks"] or meta["duration"] is not None:
        if len(_AUDIO_META_CACHE) >= _AUDIO_META_MAX:
            for k in sorted(_AUDIO_META_CACHE, key=lambda k: _AUDIO_META_CACHE[k][0])[: _AUDIO_META_MAX // 4]:
                _AUDIO_META_CACHE.pop(k, None)
        _AUDIO_META_CACHE[url] = (time.monotonic(), meta)
    return meta


def _probe_audio_meta_uncached(url: str) -> dict:
    """Audio streams + container duration in ONE header read:
        {"tracks": [{"rel": 0, "lang": "ger", ...}], "duration": 5423.2|None}
    `rel` is the AUDIO-RELATIVE index — exactly what `-map 0:a:<rel>` wants.
    Duration is what lets the player treat the rolling remux as seekable
    (seek = re-request with &start=), so it rides the same ffprobe rather
    than costing a second one.

    Bounded like _probe_video_codec: reads only the container header over HTTP.
    Returns empty/None fields on any failure, which callers treat as "no info".
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index,channels:stream_tags=language,title:format=duration",
             "-of", "json", "-user_agent", _BROWSER_UA, url],
            capture_output=True, timeout=12,
        )
        data = json.loads(r.stdout.decode(errors="replace") or "{}") or {}
        streams = data.get("streams") or []
        out = []
        for rel, st in enumerate(streams):
            tags = {str(k).lower(): str(v) for k, v in (st.get("tags") or {}).items()}
            out.append({
                "rel": rel,
                "lang": (tags.get("language") or "").strip().lower(),
                "title": (tags.get("title") or "").strip(),
                "channels": st.get("channels") or 0,
            })
        duration: "float | None" = None
        try:
            raw = (data.get("format") or {}).get("duration")
            if raw is not None:
                duration = float(raw)
                if not (duration > 0):
                    duration = None
        except (TypeError, ValueError):
            duration = None
        return {"tracks": out, "duration": duration}
    except Exception:
        return {"tracks": [], "duration": None}


def _probe_audio_tracks(url: str) -> list:
    return _probe_audio_meta(url)["tracks"]


def _probe_english_audio_index(url: str) -> "int | None":
    """Audio-relative index of the first ENGLISH audio stream, or None.

    The auto-default: these panels serve multi-audio rips (titles tagged
    "MULTI"), and ffmpeg's default picks the highest-channel track — routinely a
    French/German 5.1 sitting ahead of English stereo, so the remux comes out in
    the wrong language. None means "no opinion" → ffmpeg's default is left alone.
    A caller can override this entirely by passing an explicit aidx.
    """
    for t in _probe_audio_tracks(url):
        if t["lang"] in _ENGLISH_LANG_TAGS or "english" in t["title"].lower():
            return t["rel"]
    return None


# Per-URL probe results (video codec + English audio index). Seeking a remux
# re-requests /remux.m3u8 with a new &start=, which is a NEW session id for the
# SAME file — without this cache every seek pays the two ffprobe header reads
# again (several seconds each against a slow panel) before ffmpeg even spawns.
# Keyed by URL only: the file's codec/track layout can't change under us.
_VOD_PROBE_CACHE: "dict[str, tuple[float, str | None, int | None]]" = {}
_VOD_PROBE_TTL = 3600.0
_VOD_PROBE_MAX = 200


def _vod_probe_cached(url: str) -> "tuple[str | None, int | None] | None":
    hit = _VOD_PROBE_CACHE.get(url)
    if hit is None or time.monotonic() - hit[0] > _VOD_PROBE_TTL:
        return None
    return (hit[1], hit[2])


def _vod_probe_store(url: str, vcodec: "str | None", eng_aidx: "int | None") -> None:
    if len(_VOD_PROBE_CACHE) >= _VOD_PROBE_MAX:
        # Drop the oldest entries; a fixed small cap, not an LRU — churn here is
        # a few titles a night, the cap is a leak guard, not a tuning knob.
        for k in sorted(_VOD_PROBE_CACHE, key=lambda k: _VOD_PROBE_CACHE[k][0])[: _VOD_PROBE_MAX // 4]:
            _VOD_PROBE_CACHE.pop(k, None)
    _VOD_PROBE_CACHE[url] = (time.monotonic(), vcodec, eng_aidx)


async def _start_or_get_hls_session(url: str, quality: str = "passthrough", mode: str = "live", start: float = 0.0, aidx: "int | None" = None) -> _HLSSession:
    sid = _hls_session_id(url, quality, mode, start, aidx)

    # VOD: probe the codec BEFORE taking the lock (ffprobe can take seconds and
    # must not stall live session ops). ONLY H.264 is remuxable: copy-mode
    # ffmpeg costs ~nothing, like the live passthrough. TRANSCODE IS DISABLED —
    # a libx264 encode eats a full core of this 2-core box and STARVED THE LIVE
    # TRANSMUX SESSIONS (live channels visibly skipped, 2026-07-02). HEVC/VP9
    # rips get a fast 415 so the player fails over immediately instead of
    # spinning on a stream the box can't afford to produce.
    vod_vcodec: "str | None" = None
    vod_eng_audio: "int | None" = None
    if mode == "vod":
        async with _HLS_LOCK:
            existing = _HLS_SESSIONS.get(sid)
            if existing is not None and existing.proc is not None and existing.proc.poll() is None:
                existing.last_access = time.monotonic()
                return existing
        cached_probe = _vod_probe_cached(url)
        if cached_probe is not None:
            vod_vcodec, cached_eng = cached_probe
        else:
            vod_vcodec = await asyncio.to_thread(_probe_video_codec, url)
            cached_eng = None if vod_vcodec is not None and vod_vcodec != "h264" \
                else await asyncio.to_thread(_probe_english_audio_index, url)
            _vod_probe_store(url, vod_vcodec, cached_eng)
        if vod_vcodec is not None and vod_vcodec != "h264":
            raise HTTPException(status_code=415, detail=f"unsupported vod codec: {vod_vcodec} (transcode disabled)")
        # An explicit aidx (user picked a language) wins outright. Otherwise fall
        # back to the (cached) auto-detected English index.
        vod_eng_audio = aidx if aidx is not None else cached_eng

    async with _HLS_LOCK:
        existing = _HLS_SESSIONS.get(sid)
        if existing is not None and existing.proc is not None and existing.proc.poll() is None:
            existing.last_access = time.monotonic()
            return existing

        # VOD sub-cap: at most 3 remux sessions, evicting the oldest VOD
        # session (never a live one) to make room. Keeps movie fallbacks from
        # ever crowding the live budget.
        if mode == "vod":
            vod_sessions = [s for s in _HLS_SESSIONS.values() if s.mode == "vod"]
            if len(vod_sessions) >= 3:
                vod_sessions.sort(key=lambda s: (s.has_served_segment, s.last_access))
                for victim in vod_sessions[: len(vod_sessions) - 3 + 1]:
                    _HLS_SESSIONS.pop(victim.id, None)
                    _kill_session(victim)

        # LRU eviction: enforce HLS_MAX_SESSIONS BEFORE starting a new
        # ffmpeg. Without this the session table can balloon during
        # channel-hop / verifier-probe bursts (we observed 35+ live
        # ffmpegs on a 2 GB VPS, OOM-killing the relay).
        #
        # Sort key: VOD sessions go first (live always wins), then warmup-only
        # sessions (has_served_segment=False) before any session that has
        # actually streamed a fragment — that protects the active viewer from
        # being evicted by a flurry of chip hovers. Within each group, oldest
        # last_access goes first.
        if len(_HLS_SESSIONS) >= HLS_MAX_SESSIONS:
            victims = sorted(
                _HLS_SESSIONS.values(),
                key=lambda s: (s.mode != "vod", s.has_served_segment, s.last_access),
            )
            for victim in victims[: len(_HLS_SESSIONS) - HLS_MAX_SESSIONS + 1]:
                _HLS_SESSIONS.pop(victim.id, None)
                _kill_session(victim)

        workdir = HLS_WORKDIR / sid
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True, exist_ok=True)

        encode = _encode_args(quality, _get_encoder())
        # `-fflags +discardcorrupt -err_detect ignore_err` keeps ffmpeg alive
        # through transient TS packet corruption from flaky upstream upstreams.
        # Copy-mode tolerated this implicitly; once we re-encode the decoder
        # gets stricter and otherwise exits early on the first bad MB.
        #
        # Native IPTV-player parity: the gap between us and XCIPTV/VLC for live
        # smoothness comes down to how aggressively the input layer hides
        # upstream jitter and timestamp glitches.
        #
        # `-rtbufsize 32M` — 32 MB real-time input buffer so brief upstream
        # jitter (upstream CDN backpressure, ISP microbursts) is absorbed
        # instead of forcing packet drops + reconnects.
        #
        # `-thread_queue_size 1024` — input demuxer queue. Default is 8
        # packets, which fills almost instantly on a TS stream and forces
        # ffmpeg to throttle the upstream read. Native players keep deep
        # internal queues for exactly this reason.
        #
        # `-probesize 1000000 -analyzeduration 1000000` — cap input probing
        # at 1 MB / 1 s instead of the 5 MB / 5 s default. We KNOW the
        # upstream is H.264+AAC TS; ffmpeg doesn't need to read 5 s of stream
        # to confirm that. Cuts cold-start latency proportionally on every
        # session respawn (idle timeout, reconnect, fresh viewer).
        #
        # `+genpts` (added to fflags) — regenerate PTS on the fly when the
        # upstream drops timestamp markers. Without it, missing PTS causes
        # ffmpeg to pause output (no segment produced) until the next clean
        # marker arrives. Native players synthesize timestamps as a matter
        # of course; we should too.
        if mode == "vod":
            # VOD container remux (mkv/ts movie files over HTTP) → the SAME
            # rolling live-style HLS as the live path. `-re` paces input at
            # native speed so the rolling window tracks the viewer instead of
            # racing to EOF in minutes; the cost is NO seeking — it plays like
            # a live channel. This is the tvspot resolver's LAST-RESORT
            # fallback when a title has no browser-playable mp4 anywhere.
            # H.264 video copies straight through; HEVC/VP9 (x265 rips — TLOU
            # was hevc Main10, undecodable in HLS-TS on every real player)
            # re-encode at 720p via the shared live tier. Audio always → AAC
            # (rips carry AC3/EAC3/DTS that browsers can't decode).
            # h264 (or probe-inconclusive benefit-of-the-doubt): cheap copy.
            # Non-h264 was already rejected with 415 before the lock.
            enc_args = ["-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ac", "2"]
            # Pin the English track when the file identifies one. Without an
            # explicit -map, ffmpeg's default picks the highest-channel audio,
            # which on MULTI rips is usually not English. No opinion → no -map,
            # so untagged files keep exactly the old behaviour.
            map_args = (
                ["-map", "0:v:0", "-map", f"0:a:{vod_eng_audio}"]
                if vod_eng_audio is not None else []
            )
            args = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning",
                # Input seek BEFORE -i: fast keyframe seek, so a resume/failover
                # can pick up mid-file instead of always restarting at 0:00.
                *(["-ss", f"{start:g}"] if start > 0 else []),
                "-re",
                "-user_agent", _BROWSER_UA,
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "5",
                "-i", url,
                *map_args,
                *enc_args,
                "-max_muxing_queue_size", "4096",
                "-f", "hls",
                "-hls_time", str(HLS_SEGMENT_SECONDS),
                "-hls_list_size", str(HLS_LIST_SIZE),
                "-hls_flags", "delete_segments+omit_endlist+independent_segments+discont_start",
                "-hls_segment_filename", str(workdir / "seg%04d.ts"),
                str(workdir / "live.m3u8"),
            ]
        else:
            args = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning",
                "-fflags", "+discardcorrupt+genpts",
                "-rtbufsize", "32M",
                "-thread_queue_size", "1024",
                "-probesize", "1000000",
                "-analyzeduration", "1000000",
                "-err_detect", "ignore_err",
                "-user_agent", _BROWSER_UA,
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "5",
                "-i", url,
                *encode,
                # Larger muxer queue tolerates 5-10s of upstream backpressure
                # before ffmpeg drops packets — pairs with the deep player buffer.
                "-max_muxing_queue_size", "4096",
                "-f", "hls",
                "-hls_time", str(HLS_SEGMENT_SECONDS),
                "-hls_list_size", str(HLS_LIST_SIZE),
                # discont_start: signals to hls.js that this session is fresh and
                # any PTS continuity from a prior session is broken. Combined with
                # the player's maxBufferHole=1.5s, hls.js flushes the MSE source
                # buffer cleanly at the marker instead of stalling on PTS jumps
                # after an LRU-evict + respawn for the same channel.
                "-hls_flags", "delete_segments+omit_endlist+independent_segments+discont_start",
                "-hls_segment_filename", str(workdir / "seg%04d.ts"),
                str(workdir / "live.m3u8"),
            ]
        proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Protect the stream from the API. The relay and the API share this one
        # 2 GB box, so when the API is CPU-busy (EPG/catalog builds) ffmpeg can
        # miss its `-re` real-time deadline and the segment pipeline stalls —
        # the browser then buffers even though the upstream is fine. The relay
        # runs as root, so nudge each ffmpeg to a higher scheduling priority
        # (nice -5) than the API workers (nice 0): under contention ffmpeg is
        # scheduled first and keeps producing segments. Pure scheduling bias —
        # it never kills anything, and passthrough `-c copy` uses little CPU, so
        # this can't starve the API; it just stops the API from starving streams.
        # Best-effort: a platform without setpriority / lacking privilege is a
        # no-op, not a failure.
        try:
            os.setpriority(os.PRIO_PROCESS, proc.pid, -5)
        except (OSError, AttributeError, PermissionError):
            pass
        sess = _HLSSession(sid, url, workdir, mode)
        sess.proc = proc
        _HLS_SESSIONS[sid] = sess

    manifest = sess.workdir / "live.m3u8"
    # VOD gets a longer runway: `-re` paces the input, so the first segment
    # can't exist before ~HLS_SEGMENT_SECONDS of wall time + container probe.
    startup_s = HLS_STARTUP_TIMEOUT_S if mode == "live" else 25.0
    deadline = time.monotonic() + startup_s
    while time.monotonic() < deadline:
        if manifest.exists() and manifest.stat().st_size > 0:
            _remember_hls_url(sid, url, quality, mode, start)
            return sess
        if sess.proc is not None and sess.proc.poll() is not None:
            stderr_tail = b""
            if sess.proc.stderr is not None:
                try:
                    stderr_tail = sess.proc.stderr.read() or b""
                except Exception:
                    pass
            async with _HLS_LOCK:
                _HLS_SESSIONS.pop(sid, None)
            _kill_session(sess)
            stderr_text = stderr_tail.decode(errors="replace")[:400]
            # Distinguish permanent upstream failures (auth, account-locked,
            # forbidden) from transient ones. hls.js retries 5xx automatically,
            # which on a dead chip wastes 10-20s before fail-over fires —
            # returning 4xx tells hls.js to give up immediately so StreamPanel's
            # onError advances to the next chip without delay.
            if any(tok in stderr_text for tok in ("401 Unauthorized", "403 Forbidden", "authorization failed")):
                raise HTTPException(status_code=410, detail=f"upstream auth failed: {stderr_text}")
            raise HTTPException(status_code=502, detail=f"ffmpeg exited early: {stderr_text}")
        await asyncio.sleep(0.25)

    async with _HLS_LOCK:
        _HLS_SESSIONS.pop(sid, None)
    _kill_session(sess)
    raise HTTPException(status_code=504, detail="ffmpeg did not produce manifest in time")


@app.get("/hls")
async def proxy_hls(u: str, request: Request, q: str = "passthrough") -> Response:
    """Transmux a raw MPEG-TS upstream into a rolling HLS playlist.

    Shared by session id = sha1(quality|url)[:16], so N viewers on the same
    (channel, quality) cost one ffmpeg process, not N. Segments are served via
    /hls-seg/<sid>/<f>. q=720p|480p re-encode for buffering relief; default
    passthrough keeps the original `-c copy` behavior.
    """
    _check_token(request)
    url = unquote(u)
    _check_host(url)
    if q not in _VALID_QUALITIES:
        raise HTTPException(status_code=400, detail=f"unknown q: {q}")

    await _ensure_hls_reaper()

    sess = await _start_or_get_hls_session(url, q)
    sess.last_access = time.monotonic()
    return _session_manifest_response(sess, request)


def _session_manifest_response(sess: _HLSSession, request: Request) -> Response:
    """Serve a session's live.m3u8 with segment lines rewritten to /hls-seg."""
    try:
        body = (sess.workdir / "live.m3u8").read_text()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="manifest not yet available")

    tunnel_base = _tunnel_base(request)
    token_q     = f"&t={quote(TOKEN, safe='')}" if TOKEN else ""
    out: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            out.append(f"{tunnel_base}/hls-seg/{sess.id}/{stripped}?k=1{token_q}")
        else:
            out.append(line)
    rewritten = "\n".join(out) + "\n"

    return Response(
        content=rewritten,
        media_type="application/vnd.apple.mpegurl",
        headers={**_CORS, "Cache-Control": "no-cache"},
    )


# upstream VOD file path: /movie|series/<user>/<pass>/<stream_id>.<container>
_VOD_PATH_RE = re.compile(r"^/(movie|series)/[^/]+/[^/]+/[^/]+\.(mkv|avi|mp4|ts)$", re.IGNORECASE)


def _check_vod_url(url: str) -> None:
    """Remux inputs aren't limited to the live-host allowlist — VOD panels are
    discovered nightly by the tvspot pipeline and rotate too often to sync into
    data/relay_allowed_hosts.json. Instead the URL must LOOK like an upstream VOD
    file path; that plus the relay token keeps this from being an open proxy
    for arbitrary web content."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise HTTPException(status_code=400, detail="bad url")
    if not _VOD_PATH_RE.match(p.path or ""):
        raise HTTPException(status_code=400, detail="not an upstream VOD path")


@app.get("/audio-tracks")
async def audio_tracks(u: str, request: Request) -> Response:
    """List a VOD file's audio tracks so the player can offer a language menu.

    Same token + upstream-path gate as the remux. The player fetches this once per
    title, shows the languages, and re-requests remux.m3u8 with &aidx=<rel> for
    the chosen one. Bounded (ffprobe reads only the header); [] on any failure,
    which the player renders as "no choice available, English default stands".
    """
    _check_token(request)
    url = unquote(u)
    _check_vod_url(url)
    meta = await asyncio.to_thread(_probe_audio_meta, url)
    # `duration` (seconds, null when unknown) is what makes the rolling remux
    # seekable client-side: the player clamps seek targets to it and re-requests
    # remux.m3u8 with &start=<target>.
    return JSONResponse(meta, headers=_CORS)


@app.get("/remux.m3u8")
async def remux_vod(u: str, request: Request, start: float = 0, aidx: "int | None" = None) -> Response:
    """Remux a VOD container browsers can't play (mkv/avi) into rolling HLS.

    Live-style output: no seeking, plays like a live channel. The tvspot VOD
    resolver appends these as LAST-RESORT fallback sources when a title has no
    browser-playable mp4 on any panel. Session lifecycle (idle reap, LRU cap,
    lazy respawn via /hls-seg) is shared with the live /hls path — a movie
    occupies one of the HLS_MAX_SESSIONS slots for its runtime. The ".m3u8"
    route suffix is load-bearing: the tvspot player picks its HLS engine by
    URL substring.
    """
    _check_token(request)
    url = unquote(u)
    _check_vod_url(url)

    await _ensure_hls_reaper()

    sess = await _start_or_get_hls_session(url, "passthrough", "vod", max(0.0, start), aidx)
    sess.last_access = time.monotonic()
    return _session_manifest_response(sess, request)


@app.get("/hls-seg/{session_id}/{segment}")
async def proxy_hls_segment(session_id: str, segment: str, request: Request) -> Response:
    _check_token(request)

    if not _SEG_NAME_RE.match(segment):
        raise HTTPException(status_code=400, detail="invalid segment name")

    sess = _HLS_SESSIONS.get(session_id)
    if sess is None:
        # Session was LRU-evicted or reaped while the player still had segment
        # URLs queued. Look up the upstream URL we recorded when /hls last
        # served this sid and respawn the ffmpeg transparently. The respawned
        # session starts at seg0000, so the player's stale segNNNN.ts request
        # will 404 below — that's recoverable (hls.js refetches the manifest
        # on its next poll) instead of the 410 cliff we used to return.
        info = _HLS_URL_BY_SID.get(session_id)
        if info is None:
            raise HTTPException(status_code=410, detail="session expired")
        url, quality, mode, start = info
        try:
            sess = await _start_or_get_hls_session(url, quality, mode, start)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"respawn failed: {e}")

    sess.last_access = time.monotonic()
    sess.has_served_segment = True

    path = sess.workdir / segment
    if not path.exists():
        raise HTTPException(status_code=404, detail="segment not ready")

    # Off-loop disk read: ffmpeg writes ~1 MB segments and FastAPI's worker
    # threadpool absorbs the read without blocking the event loop. With 5+
    # concurrent BarnCentre tabs (the chip warmup chain is wide), serial
    # path.read_bytes() inside the async handler would queue segment fetches
    # and surface as cross-tab stalls.
    data = await asyncio.to_thread(path.read_bytes)
    return Response(
        content=data,
        media_type="video/mp2t",
        # Segments are immutable: ffmpeg writes seg00NN.ts once and never
        # rewrites it. Cloudflare Tunnel + browser cache will reuse them, so
        # the relay disk stops servicing repeat fetches (back-buffer reads,
        # second viewer of the same chip).
        headers={**_CORS, "Cache-Control": "public, max-age=60, immutable"},
    )


if __name__ == "__main__":
    HLS_WORKDIR.mkdir(parents=True, exist_ok=True)
    if not shutil.which("ffmpeg"):
        print("[iptv-relay] WARNING: ffmpeg not found on PATH — /hls endpoint will 502")
    print(f"[iptv-relay] h264 encoder: {_get_encoder()}")
    print(f"[iptv-relay] port {PORT} — token: {'SET' if TOKEN else 'none (set IPTV_RELAY_TOKEN for auth)'}")
    print(f"[iptv-relay] cache: {CACHE_BYTES // 1024 // 1024} MB / {CACHE_TTL_S}s TTL")
    print(f"[iptv-relay] allowed hosts: {sorted(ALLOWED_HOSTS)}")
    print(f"[iptv-relay] hls workdir: {HLS_WORKDIR} (idle timeout {HLS_IDLE_TIMEOUT_S}s)")
    print(f"[iptv-relay] next step: `cloudflared tunnel run iptv-relay` (publishes via localhost:8000)")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
