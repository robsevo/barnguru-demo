"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import ClientLayout from "./ClientLayout";
import ScoreboardBar from "./ScoreboardBar";
import GoalFeed from "./GoalFeed";

function DevBanner() {
  const [banner, setBanner] = useState<{ active: boolean; message: string } | null>(null);

  useEffect(() => {
    const load = () =>
      fetch("/api/dev/banner")
        .then(r => r.json())
        .then(d => setBanner(d))
        .catch(() => {});
    load();
    const t = setInterval(load, 30_000);
    return () => clearInterval(t);
  }, []);

  if (!banner?.active) return null;

  return (
    <div className="sticky top-0 z-[60] w-full px-3 py-2 flex items-center justify-center gap-2
      bg-[#fbbf24]/[0.12] border-b border-[#fbbf24]/30
      backdrop-blur-sm shadow-[0_1px_0_rgba(251,191,36,0.15)]">
      <span className="w-1.5 h-1.5 rounded-full bg-[#fbbf24] shadow-[0_0_6px_rgba(251,191,36,0.9)] animate-pulse shrink-0" />
      <p className="text-[10px] font-semibold text-[#fbbf24]/80 tracking-wide text-center leading-snug">
        {banner.message}
      </p>
    </div>
  );
}

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  if (pathname === "/login") {
    return <>{children}</>;
  }

  return (
    <ClientLayout>
      <DevBanner />
      <ScoreboardBar />
      <GoalFeed />
      {children}
    </ClientLayout>
  );
}
