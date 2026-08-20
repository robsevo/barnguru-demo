import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Every deploy gets a new buildId so clients can never serve stale HTML that
  // points at a chunk hash which still references last week's model URL. Next
  // writes `_next/static/<buildId>/...` into every HTML response; with a fresh
  // buildId, the browser is forced to re-fetch HTML + JS on the next visit.
  // VERCEL_GIT_COMMIT_SHA is injected by Vercel; falls back to GITHUB_SHA or
  // "dev" in local dev.
  generateBuildId: async () =>
    process.env.VERCEL_GIT_COMMIT_SHA || process.env.GITHUB_SHA || "dev",
  // Hard cache policy: HTML short-lived (so a new deploy propagates fast),
  // static chunks + CV model files immutable (their names already contain
  // content hashes / explicit version tags like `.v2.`). This replaces the
  // default Vercel cache-control which caches HTML for 10 seconds — long
  // enough to pin a full session on a stale bundle for some users.
  async headers() {
    return [
      {
        // All Next-generated HTML responses (pages + Server Components output).
        source: "/:path*",
        headers: [
          // Stale-while-revalidate = 0 forces Vercel's edge to treat dynamic
          // HTML as always revalidate. Users still load fast because the
          // rest of the bundle is on long-cache, but they never pin to a
          // stale index.html after a deploy.
          { key: "Cache-Control", value: "public, max-age=0, must-revalidate" },
          // Cross-origin isolation for multi-threaded WASM (2-4× faster CV
          // inference on machines without WebGPU). `credentialless` keeps
          // cross-origin iframes (localhost:8000 stream-embed, Brightcove
          // highlight player) loadable — they'd fail outright under
          // `require-corp` because those upstreams don't set CORP headers.
          // Trade-off: cross-origin requests from this page drop cookies.
          // We don't need cookies on those paths (stream URLs use their own
          // auth; model files are public Cloudflare Pages).
          { key: "Cross-Origin-Opener-Policy",   value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
        ],
      },
    ];
  },
};

export default nextConfig;
