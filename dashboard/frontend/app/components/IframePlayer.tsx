"use client";

import { useEffect } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "";

function reportStreamSuccess(rawUrl: string) {
  if (!API_BASE) return;
  fetch(`${API_BASE}/stream-success?url=${encodeURIComponent(rawUrl)}`, {
    method: "POST",
    keepalive: true,
  }).catch(() => {});
}

export default function IframePlayer({ url, onFail, priority }: {
  url: string;
  onFail: () => void;
  priority?: number;
}) {
  // Route through the Next.js /api/stream-embed wrapper (same origin as the
  // page). Hitting localhost:8000 directly was failing for users whose network
  // / browser refused that hostname ("localhost:8000 refused to connect"); the
  // Vercel-hosted wrapper sidesteps that since the page already loads from
  // Vercel. The wrapper HTML is tiny (single <iframe> + ad-blocker shim), so
  // the egress hit is negligible — the actual video bytes still come from the
  // third-party stream domain inside the inner iframe.
  const src =
    `/api/stream-embed?url=${encodeURIComponent(url)}` +
    (priority != null ? `&priority=${priority}` : "");

  useEffect(() => {
    let reportedSuccess = false;
    const handler = (e: MessageEvent) => {
      if (e.data?.url !== url) return;
      const t = e.data?.type;
      if (t === "Origin-stream-killed") {
        onFail();
      } else if (t === "Origin-frame-progress") {
        if (!reportedSuccess) {
          reportedSuccess = true;
          reportStreamSuccess(url);
        }
      }
    };
    window.addEventListener("message", handler);
    return () => {
      window.removeEventListener("message", handler);
    };
  }, [url, onFail]);

  return (
    <div className="relative w-full overflow-hidden bg-black" style={{ aspectRatio: "16/9" }}>
      <iframe
        src={src}
        className="w-full h-full border-0"
        allow="autoplay; fullscreen; picture-in-picture; encrypted-media"
      />
      <button
        onClick={onFail}
        className="absolute top-2 right-2 text-[8px] font-semibold uppercase tracking-[0.14em] px-2 py-1 rounded border border-white/20 bg-black/60 text-white/40 hover:text-white/80 hover:border-white/40 transition-colors pointer-events-auto"
      >
        try next ›
      </button>
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="sm:hidden absolute bottom-2 left-2 text-[8px] font-semibold uppercase tracking-[0.14em] px-2 py-1 rounded border border-white/20 bg-black/60 text-white/40"
      >
        open ↗ for background
      </a>
    </div>
  );
}
