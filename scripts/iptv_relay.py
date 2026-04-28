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
HLS_LIST_SIZE          = 6
HLS_IDLE_TIMEOUT_S     = 30.0
HLS_STARTUP_TIMEOUT_S  = 15.0

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
    gop          = ["-g", "96", "-keyint_min", "96"]

    if encoder == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-profile:v", "main",
                *gop, *scale, *common_rate, *common_audio]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll",
                "-profile:v", "main", "-no-scenecut", "1",
                *gop, *scale, *common_rate, *common_audio]
    if encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-preset", "veryfast",
                "-profile:v", "main",
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

        workdir = HLS_WORKDIR / sid
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True, exist_ok=True)

        encode = _encode_args(quality, _get_encoder())
        # `-fflags +discardcorrupt -err_detect ignore_err` keeps ffmpeg alive
        # through transient TS packet corruption from flaky upstream upstreams.
        # Copy-mode tolerated this implicitly; once we re-encode the decoder
        # gets stricter and otherwise exits early on the first bad MB.
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-fflags", "+discardcorrupt",
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

    data = path.read_bytes()
    return Response(
        content=data,
        media_type="video/mp2t",
        headers={**_CORS, "Cache-Control": "no-cache"},
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
