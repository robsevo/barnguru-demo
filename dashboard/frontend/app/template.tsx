"use client";

/**
 * Route-transition wrapper — App Router re-mounts a template on every
 * navigation, so wrapping children here gives us a clean fade-in animation
 * on each route swap without touching individual pages. CSS-only so it
 * works under SSR + respects prefers-reduced-motion.
 */
export default function Template({ children }: { children: React.ReactNode }) {
  return <div className="route-transition">{children}</div>;
}
