"use client";

import { usePathname } from "next/navigation";
import ClientLayout from "./ClientLayout";

/**
 * The shell every page renders inside.
 *
 * Deliberately thin. It used to stack a dev banner, a live scoreboard ticker, a
 * goal feed and a score-hiding toggle above every single route — four
 * always-mounted widgets, each polling on its own timer, before you saw any of
 * the page you actually asked for. This build is a statistics site, so the page
 * gets the screen.
 */
export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  if (pathname === "/login") {
    return <>{children}</>;
  }

  return (
    <ClientLayout>
      {/* Keyed wrapper re-mounts on route change so .page-enter fires a
          fresh fade-in transition between pages. */}
      <div key={pathname} className="page-enter">
        {children}
      </div>
    </ClientLayout>
  );
}
