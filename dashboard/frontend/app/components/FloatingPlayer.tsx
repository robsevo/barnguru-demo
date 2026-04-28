"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePiP } from "@/utils/pipContext";

function proxyUrl(raw: string) {
  return `/api/stream-proxy?url=${encodeURIComponent(raw)}`;
}

export default function FloatingPlayer() {
  const { stream, deactivate } = usePiP();

  // ── Responsive width: 480px on desktop, 320px on mobile ────────────────────
  const [pipW, setPipW] = useState(320);
  const [pipH, setPipH] = useState<number | null>(null); // null = auto 16:9
  const snapToBottomRight = useCallback((w: number) => ({
    x: Math.max(8, window.innerWidth  - w - 16),
    y: Math.max(8, window.innerHeight - Math.round(w * 9 / 16) - 60),
  }), []);

  // ── Position — initialised client-side so video element always mounts ──────
  const [pos, setPos] = useState({ x: 0, y: 0 });
  useLayoutEffect(() => {
    const w = window.innerWidth >= 768 ? 520 : 320;
    setPipW(w);
    setPos(snapToBottomRight(w));
  }, [snapToBottomRight]);
  // Re-snap to bottom-right whenever a new stream activates
  useEffect(() => {
    if (!stream) return;
    const w = window.innerWidth >= 768 ? 520 : 320;
    setPipW(w);
    setPipH(null);
    setPos(snapToBottomRight(w));
  }, [stream?.url, snapToBottomRight]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Drag — document-level listeners, reliable across all child elements ─────
  const posRef   = useRef(pos);
  posRef.current = pos;
  const pipWRef  = useRef(pipW);
  pipWRef.current = pipW;
  const pipHRef  = useRef<number | null>(pipH);
  pipHRef.current = pipH;
  const dragging = useRef<{ ox: number; oy: number; px: number; py: number } | null>(null);
  const resizing = useRef<{
    corner: "br" | "bl" | "tr" | "tl";
    ox: number; oy: number;
    pw: number; ph: number;
    px: number; py: number;
  } | null>(null);

  const makeResizeHandler = (corner: "br" | "bl" | "tr" | "tl") => (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const currentH = pipHRef.current ?? Math.round(pipWRef.current * 9 / 16);
    resizing.current = {
      corner,
      ox: e.clientX, oy: e.clientY,
      pw: pipWRef.current, ph: currentH,
      px: posRef.current.x, py: posRef.current.y,
    };
    const MIN_W = 200;
    const onMove = (ev: PointerEvent) => {
      if (!resizing.current) return;
      const { corner: c, ox, oy, pw, ph, px, py } = resizing.current;
      const dx = ev.clientX - ox;
      const dy = ev.clientY - oy;
      const MAX_W = window.innerWidth * 0.95;
      let newW = pw, newH = ph, newX = px, newY = py;
      if (c === "br") {
        newW = Math.max(MIN_W, Math.min(MAX_W, pw + dx));
        newH = Math.max(112, ph + dy);
      } else if (c === "bl") {
        newW = Math.max(MIN_W, Math.min(MAX_W, pw - dx));
        newH = Math.max(112, ph + dy);
        newX = px + (pw - newW);
      } else if (c === "tr") {
        newW = Math.max(MIN_W, Math.min(MAX_W, pw + dx));
        newH = Math.max(112, ph - dy);
        newY = py + (ph - newH);
      } else {
        newW = Math.max(MIN_W, Math.min(MAX_W, pw - dx));
        newH = Math.max(112, ph - dy);
        newX = px + (pw - newW);
        newY = py + (ph - newH);
      }
      setPipW(newW);
      setPipH(newH);
      setPos({
        x: Math.max(0, Math.min(window.innerWidth - newW, newX)),
        y: Math.max(0, Math.min(window.innerHeight - 40, newY)),
      });
    };
    const onUp = () => {
      resizing.current = null;
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  };

  const onHeaderPointerDown = useCallback((e: React.PointerEvent) => {
    // Only drag on primary button, ignore close/link clicks
    if (e.button !== 0) return;
    e.preventDefault();
    dragging.current = {
      ox: e.clientX, oy: e.clientY,
      px: posRef.current.x, py: posRef.current.y,
    };
    const onMove = (ev: PointerEvent) => {
      if (!dragging.current) return;
      const { ox, oy, px, py } = dragging.current;
      setPos({
        x: Math.max(0, Math.min(window.innerWidth  - pipW, px + ev.clientX - ox)),
        y: Math.max(0, Math.min(window.innerHeight - 40,   py + ev.clientY - oy)),
      });
    };
    const onUp = () => {
      dragging.current = null;
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup",   onUp);
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup",   onUp);
  }, []);

  // ── HLS — callback ref so effect re-runs the moment the video element mounts
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const videoRef = useCallback((el: HTMLVideoElement | null) => setVideoEl(el), []);

  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState<string | null>(null);
  const [isIframe,  setIsIframe]  = useState(false);
  const [pipActive, setPipActive] = useState(false);
  const pipSupported = typeof document !== "undefined" && !!document.pictureInPictureEnabled;

  // Auto-fallback to 480p — see StreamPanel for the full rationale. Keeps
  // PiP and the main game-page player symmetric.
  const lowQualityRef  = useRef(false);
  const resolvedUrlRef = useRef<string | null>(null);
  const [reattachToken,  setReattachToken]  = useState(0);
  const [downgradeNotice, setDowngradeNotice] = useState(false);

  // Reset on stream change so a new pinned stream starts at 720p again.
  useEffect(() => {
    lowQualityRef.current = false;
    resolvedUrlRef.current = null;
    setDowngradeNotice(false);
  }, [stream?.url]);

  useEffect(() => {
    if (!stream || !videoEl) return;

    setLoading(true);
    setError(null);
    setIsIframe(false);
    setPipActive(false);

    let hlsInstance: import("hls.js").default | null = null;
    let cancelled = false;

    const attach = async () => {
      // Embed-only streams: skip server-side resolve (server IP is blocked by these sites).
      // Load directly in the browser iframe — browser's residential IP won't be blocked.
      if (stream.embedOnly) {
        setIsIframe(true);
        setLoading(false);
        return;
      }

      try {
        const res  = await fetch(`/api/stream-resolve?url=${encodeURIComponent(stream.url)}`);
        const data = await res.json() as { type: string; url: string };
        if (cancelled) return;

        if (data.type === "embed") {
          setIsIframe(true);
          setLoading(false);
          return;
        }
        if (data.type !== "m3u8") {
          setError("stream unavailable");
          setLoading(false);
          return;
        }

        const effective = lowQualityRef.current
          ? data.url.replace("q=720p", "q=480p")
          : data.url;
        resolvedUrlRef.current = effective;
        const src    = proxyUrl(effective);
        const HlsMod = await import("hls.js");
        const Hls    = HlsMod.default;
        if (cancelled) return;

        if (!Hls.isSupported()) {
          if (videoEl.canPlayType("application/vnd.apple.mpegurl")) {
            videoEl.muted = true;
            videoEl.src = src;
            setLoading(false);
            videoEl.play().catch(() => {});
          } else {
            setError("format not supported");
            setLoading(false);
          }
          return;
        }

        // React doesn't reflect the `muted` prop to the DOM attribute — set it
        // explicitly so browsers allow autoplay without user interaction.
        videoEl.muted = true;

        // See StreamPanel: IPTV manifests aren't LL-HLS, so lowLatencyMode's
        // tighter window amplifies drift on flaky upstream feeds and presents as
        // looping. Tighter live tuning + fast manifest failure for snappier
        // initial load and recovery.
        hlsInstance = new Hls({
          enableWorker: true,
          lowLatencyMode: false,
          liveDurationInfinity: true,
          liveSyncDuration: 4,
          liveMaxLatencyDuration: 8,
          backBufferLength: 10,
          // Cold-start budget — see StreamPanel.tsx for the full rationale.
          // Relay ffmpeg spin-up + tunnel hop + re-encode startup can blow past
          // the old 6s window on a fresh chip click and fail-over before the
          // first segment lands.
          manifestLoadingTimeOut: 14000,
          manifestLoadingMaxRetry: 2,
          // Buffer caps — see StreamPanel.tsx. Keeps PiP behavior symmetric
          // with the main game-page player.
          maxBufferLength: 20,
          maxMaxBufferLength: 45,
          fragLoadingTimeOut: 8000,
          fragLoadingMaxRetry: 2,
          startLevel: 0,
          capLevelToPlayerSize: true,
        });
        hlsInstance.loadSource(src);
        hlsInstance.attachMedia(videoEl);
        hlsInstance.on(Hls.Events.MANIFEST_PARSED, () => {
          if (cancelled) return;
          setLoading(false);
          videoEl.muted = true;
          const buf = videoEl.buffered;
          if (buf.length > 0) {
            videoEl.currentTime = Math.max(0, buf.end(buf.length - 1) - 2);
          }
          videoEl.play().catch(() => {});
        });
        hlsInstance.on(Hls.Events.ERROR, (_e, d) => {
          if (d.fatal && !cancelled) { setError("stream error"); setLoading(false); }
        });
      } catch {
        if (!cancelled) { setError("could not connect"); setLoading(false); }
      }
    };

    attach();
    return () => {
      cancelled = true;
      hlsInstance?.destroy();
    };
  }, [stream?.url, videoEl, reattachToken]); // eslint-disable-line react-hooks/exhaustive-deps

  // Live-stream watchdog (see StreamPanel for full rationale): catch backward
  // jumps + drift behind live edge on flaky IPTV and snap to ~4s behind buffered
  // live edge. Capped at 3 resyncs so a genuinely broken feed doesn't get
  // bounced forever — the existing error UI takes over after that.
  useEffect(() => {
    if (!videoEl || isIframe || error || loading) return;
    const LIVE_DRIFT_RESYNC = 10;
    const MAX_RESYNCS = 3;
    let lastTime = -1;
    let resyncs = 0;

    const tryDowngrade = (): boolean => {
      if (lowQualityRef.current) return false;
      const r = resolvedUrlRef.current;
      if (!r || !r.includes("q=720p")) return false;
      lowQualityRef.current = true;
      setDowngradeNotice(true);
      setTimeout(() => setDowngradeNotice(false), 2_500);
      setReattachToken((n) => n + 1);
      return true;
    };

    const tick = setInterval(() => {
      if (videoEl.paused || videoEl.ended || videoEl.readyState < 2) {
        lastTime = videoEl.currentTime;
        return;
      }
      const t = videoEl.currentTime;
      const buf = videoEl.buffered;
      const liveEdge = buf.length > 0 ? buf.end(buf.length - 1) : t;
      const looped  = lastTime > 0 && t < lastTime - 3;
      const drifted = liveEdge - t > LIVE_DRIFT_RESYNC;
      if ((looped || drifted) && resyncs < MAX_RESYNCS && buf.length > 0) {
        if (resyncs >= 1 && tryDowngrade()) return;
        videoEl.currentTime = Math.max(0, buf.end(buf.length - 1) - 4);
        resyncs += 1;
        lastTime = videoEl.currentTime;
        return;
      }
      lastTime = t;
    }, 1000);
    return () => clearInterval(tick);
  }, [videoEl, isIframe, error, loading, reattachToken]);

  // Track native browser PiP state
  useEffect(() => {
    if (!videoEl) return;
    const enter = () => setPipActive(true);
    const leave = () => setPipActive(false);
    videoEl.addEventListener("enterpictureinpicture", enter);
    videoEl.addEventListener("leavepictureinpicture",  leave);
    return () => {
      videoEl.removeEventListener("enterpictureinpicture", enter);
      videoEl.removeEventListener("leavepictureinpicture",  leave);
    };
  }, [videoEl]);

  // Background playback: PiP on visibility change + MediaSession for lock screen controls
  useEffect(() => {
    if (!videoEl) return;

    // MediaSession: tell the OS this is live media, override pause to keep playing
    if ("mediaSession" in navigator) {
      navigator.mediaSession.metadata = new MediaMetadata({ title: stream?.title || "NHL Live", artist: "GRTZKY" });
      const noop = () => {};
      navigator.mediaSession.setActionHandler("play",  () => { videoEl.play().catch(noop); });
      navigator.mediaSession.setActionHandler("pause", () => { /* keep playing in background */ });
      navigator.mediaSession.setActionHandler("stop",  null);
    }

    // Auto-PiP when user backgrounds the app / switches tabs (Android Chrome + Safari 14+)
    if (!pipSupported) return;
    const onVisibility = () => {
      if (document.hidden && !videoEl.paused && document.pictureInPictureElement !== videoEl) {
        videoEl.requestPictureInPicture().catch(() => {});
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      if ("mediaSession" in navigator) {
        navigator.mediaSession.setActionHandler("play",  null);
        navigator.mediaSession.setActionHandler("pause", null);
      }
    };
  }, [videoEl, pipSupported, stream?.title]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggleBrowserPiP = useCallback(async () => {
    if (!videoEl) return;
    try {
      if (document.pictureInPictureElement === videoEl) {
        await document.exitPictureInPicture();
      } else {
        await videoEl.requestPictureInPicture();
      }
    } catch { /* cancelled or unsupported */ }
  }, [videoEl]);

  if (!stream) return null;

  return (
    <div
      className="fixed z-[9999] select-none"
      style={{ left: pos.x, top: pos.y, width: pipW }}
    >
      {/* ── Corner resize handles ──────────────────────────────────────── */}
      {/* top-left */}
      <div className="absolute -top-1 -left-1 w-5 h-5 z-10 cursor-nw-resize group/tl"
        onPointerDown={makeResizeHandler("tl")}>
        <div className="absolute top-1 left-1 w-3 h-3 opacity-0 group-hover/tl:opacity-100 transition-opacity duration-150">
          <div className="absolute top-0 left-0 w-full h-[2px] bg-[#C9A84C]/50 rounded" />
          <div className="absolute top-0 left-0 h-full w-[2px] bg-[#C9A84C]/50 rounded" />
        </div>
      </div>
      {/* top-right */}
      <div className="absolute -top-1 -right-1 w-5 h-5 z-10 cursor-ne-resize group/tr"
        onPointerDown={makeResizeHandler("tr")}>
        <div className="absolute top-1 right-1 w-3 h-3 opacity-0 group-hover/tr:opacity-100 transition-opacity duration-150">
          <div className="absolute top-0 right-0 w-full h-[2px] bg-[#C9A84C]/50 rounded" />
          <div className="absolute top-0 right-0 h-full w-[2px] bg-[#C9A84C]/50 rounded" />
        </div>
      </div>
      {/* bottom-left */}
      <div className="absolute -bottom-1 -left-1 w-5 h-5 z-10 cursor-sw-resize group/bl"
        onPointerDown={makeResizeHandler("bl")}>
        <div className="absolute bottom-1 left-1 w-3 h-3 opacity-0 group-hover/bl:opacity-100 transition-opacity duration-150">
          <div className="absolute bottom-0 left-0 w-full h-[2px] bg-[#C9A84C]/50 rounded" />
          <div className="absolute bottom-0 left-0 h-full w-[2px] bg-[#C9A84C]/50 rounded" />
        </div>
      </div>
      {/* bottom-right */}
      <div className="absolute -bottom-1 -right-1 w-5 h-5 z-10 cursor-se-resize group/br"
        onPointerDown={makeResizeHandler("br")}>
        <div className="absolute bottom-1 right-1 w-3 h-3 opacity-0 group-hover/br:opacity-100 transition-opacity duration-150">
          <div className="absolute bottom-0 right-0 w-full h-[2px] bg-[#C9A84C]/50 rounded" />
          <div className="absolute bottom-0 right-0 h-full w-[2px] bg-[#C9A84C]/50 rounded" />
        </div>
      </div>

      <div className="rounded-xl overflow-hidden border border-[#C9A84C]/[0.30] bg-[#08090b]
        shadow-[0_16px_48px_rgba(0,0,0,0.90),0_4px_16px_rgba(0,0,0,0.70),inset_0_1px_0_rgba(201,168,76,0.12)]">

        {/* ── Drag handle / header ───────────────────────────────────────── */}
        <div
          className="flex items-center justify-between px-3 py-2 cursor-grab active:cursor-grabbing bg-gradient-to-r from-[#0d0f13] to-[#0a0c10] border-b border-[#C9A84C]/[0.12]"
          style={{ touchAction: "none" }}
          onPointerDown={onHeaderPointerDown}
        >
          <div className="flex items-center gap-2 min-w-0 pointer-events-none">
            {/* Grip dots */}
            <div className="flex flex-col gap-[3px] shrink-0 opacity-40">
              <div className="flex gap-[3px]">
                <span className="w-[3px] h-[3px] rounded-full bg-[#C9A84C]" />
                <span className="w-[3px] h-[3px] rounded-full bg-[#C9A84C]" />
              </div>
              <div className="flex gap-[3px]">
                <span className="w-[3px] h-[3px] rounded-full bg-[#C9A84C]" />
                <span className="w-[3px] h-[3px] rounded-full bg-[#C9A84C]" />
              </div>
            </div>
            <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-white/40 truncate">
              {stream.title || "Live Stream"}
            </span>
          </div>

          {/* Buttons — stop propagation so they don't trigger drag */}
          <div className="flex items-center gap-1.5 shrink-0 pointer-events-auto">
            <Link
              href={`/game/${stream.gameId}`}
              className="text-[8px] font-black uppercase tracking-[0.14em] px-2 py-0.5 rounded border border-[#C9A84C]/20 text-[#C9A84C]/50 hover:text-[#C9A84C] hover:border-[#C9A84C]/40 transition-colors"
              onPointerDown={e => e.stopPropagation()}
            >
              game ↗
            </Link>
            <button
              onClick={deactivate}
              onPointerDown={e => e.stopPropagation()}
              className="w-7 h-7 flex items-center justify-center rounded-md text-[14px] text-white/40 hover:text-white hover:bg-[#f87171]/20 border border-transparent hover:border-[#f87171]/30 transition-all duration-150"
              title="Close"
            >
              ✕
            </button>
          </div>
        </div>

        {/* ── Video / iframe area ────────────────────────────────────────── */}
        <div className="relative bg-black" style={pipH !== null ? { height: pipH } : { aspectRatio: "16/9" }}>
          {isIframe ? (
            <iframe
              src={`/api/stream-embed?url=${encodeURIComponent(stream.url)}`}
              className="absolute inset-0 w-full h-full border-0"
              allowFullScreen
              allow="autoplay; fullscreen; picture-in-picture; encrypted-media"
            />
          ) : (
            <>
              <video
                ref={videoRef}
                className="w-full h-full"
                controls
                autoPlay
                muted
                playsInline
                // Android Chrome: auto-enter browser PiP when tab is hidden
                {...({ autopictureinpicture: "true" } as object)}
                onLoadedData={() => setLoading(false)}
                onError={() => { setError("stream unavailable"); setLoading(false); }}
              />

              {downgradeNotice && !loading && !error && (
                <div className="absolute top-2 left-1/2 -translate-x-1/2 px-2.5 py-1 rounded bg-black/70 pointer-events-none">
                  <span className="text-[9px] font-mono text-white/60 uppercase tracking-[0.18em]">switched to 480p</span>
                </div>
              )}

              {loading && !error && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/70 pointer-events-none">
                  <span className="text-[10px] font-mono text-white/35 animate-pulse">connecting…</span>
                </div>
              )}

              {error && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/80">
                  <span className="text-[10px] font-mono text-[#f87171]/60">{error}</span>
                </div>
              )}

              {/* Pop-out button */}
              {!loading && !error && pipSupported && (
                <button
                  onClick={toggleBrowserPiP}
                  title={pipActive ? "Exit picture-in-picture" : "Pop out (OS picture-in-picture)"}
                  className={`absolute bottom-2 left-2 text-[8px] font-black uppercase tracking-[0.12em] px-2 py-1 rounded border transition-all ${
                    pipActive
                      ? "border-[#C9A84C]/50 bg-[#C9A84C]/20 text-[#C9A84C]"
                      : "border-white/20 bg-black/60 text-white/40 hover:text-white/80 hover:border-white/40"
                  }`}
                >
                  {pipActive ? "⊡ in PiP" : "⊡ pop out"}
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
