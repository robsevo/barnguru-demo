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
from fastapi.responses import Response

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
HLS_SEGMENT_SECONDS    = 3
# 36 segments × 3s = 108s manifest window. Final plateau per user request —
# matches player's liveSyncDuration=36 (36s behind live) with ~72s of look-back
# room so a recovery seek can't fall off the back of the deleted-segments
# cliff. Disk cost ~36 MB per session (passthrough TS), trivial against the
# 300 MB cache budget.
HLS_LIST_SIZE          = 36
# Tab-switch / ad-break tolerance: 30s was killing sessions when users
# briefly looked away, forcing a full ffmpeg respawn on resume. 90s is still
# tight enough that abandoned chips don't pile up but covers normal viewer
# behavior (phone call, brief tab switch, talking with someone in the room).
# OOM protection comes from HLS_MAX_SESSIONS below, not this idle timeout —
# 12 × ~80 MB resident keeps the relay under 1 GB regardless of how long
# each session lives.
HLS_IDLE_TIMEOUT_S     = 90.0
HLS_STARTUP_TIMEOUT_S  = 15.0
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
        return ["-c", "copy"]

    width, height, vbr, vbufsize, abr = _QUALITY_TIERS[quality]
    common_audio = ["-c:a", "aac", "-b:a", abr, "-ac", "2"]
    common_rate  = ["-b:v", vbr, "-maxrate", vbr, "-bufsize", vbufsize]
    scale        = ["-vf", f"scale={width}:{height}"]
    # GOP = 60 frames ≈ 2s at 30fps (2.4s at 25fps). With HLS_SEGMENT_SECONDS=3
    # this guarantees a keyframe inside the first segment instead of forcing
    # hls.js to wait an extra GOP on cold start. Was -g 96 (3.2s GOP) which
    # frequently misaligned with the segment boundary and added 0.5-1.5s of
    # first-paint latency.
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


# ---------------------------------------------------------------------------
# /hls — TS→HLS transmux for accounts that only serve raw MPEG-TS
# ---------------------------------------------------------------------------

class _HLSSession:
    __slots__ = ("id", "url", "workdir", "proc", "last_access", "started_at")

    def __init__(self, sid: str, url: str, workdir: Path) -> None:
        self.id          = sid
        self.url         = url
        self.workdir     = workdir
        self.proc: subprocess.Popen[bytes] | None = None
        self.last_access = time.monotonic()
        self.started_at  = time.monotonic()


_HLS_SESSIONS:  dict[str, _HLSSession] = {}
_HLS_LOCK       = asyncio.Lock()
_HLS_REAPER: asyncio.Task[None] | None = None
_SEG_NAME_RE    = re.compile(r"^seg\d{4,}\.ts$")
_BROWSER_UA     = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def _hls_session_id(url: str, quality: str = "passthrough") -> str:
    return hashlib.sha1(f"{quality}|{url}".encode()).hexdigest()[:16]


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


async def _start_or_get_hls_session(url: str, quality: str = "passthrough") -> _HLSSession:
    sid = _hls_session_id(url, quality)

    async with _HLS_LOCK:
        existing = _HLS_SESSIONS.get(sid)
        if existing is not None and existing.proc is not None and existing.proc.poll() is None:
            existing.last_access = time.monotonic()
            return existing

        # LRU eviction: enforce HLS_MAX_SESSIONS BEFORE starting a new
        # ffmpeg. Without this the session table can balloon during
        # channel-hop / verifier-probe bursts (we observed 35+ live
        # ffmpegs on a 2 GB VPS, OOM-killing the relay). Drop the
        # least-recently-used session whose pid is still alive.
        if len(_HLS_SESSIONS) >= HLS_MAX_SESSIONS:
            victims = sorted(_HLS_SESSIONS.values(), key=lambda s: s.last_access)
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
            "-f", "hls",
            "-hls_time", str(HLS_SEGMENT_SECONDS),
            "-hls_list_size", str(HLS_LIST_SIZE),
            "-hls_flags", "delete_segments+omit_endlist+independent_segments",
            "-hls_segment_filename", str(workdir / "seg%04d.ts"),
            str(workdir / "live.m3u8"),
        ]
        proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        sess = _HLSSession(sid, url, workdir)
        sess.proc = proc
        _HLS_SESSIONS[sid] = sess

    manifest = sess.workdir / "live.m3u8"
    deadline = time.monotonic() + HLS_STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if manifest.exists() and manifest.stat().st_size > 0:
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


@app.get("/hls-seg/{session_id}/{segment}")
async def proxy_hls_segment(session_id: str, segment: str, request: Request) -> Response:
    _check_token(request)

    if not _SEG_NAME_RE.match(segment):
        raise HTTPException(status_code=400, detail="invalid segment name")

    sess = _HLS_SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=410, detail="session expired")
    sess.last_access = time.monotonic()

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
