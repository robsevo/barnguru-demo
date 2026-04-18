"use client";

import { useState } from "react";
import Link from "next/link";
import { logoUrl } from "@/utils/nhl";

/**
 * Universal team logo — always links to /teams/{abbrev}.
 * Hover: scale up + glow.  Active/press: scale down.
 * Supports optional smSize for responsive two-size rendering.
 */
export default function TeamLogoLink({
  abbrev,
  size = 28,
  smSize,
  className = "",
}: {
  abbrev: string;
  size?: number;
  smSize?: number;
  className?: string;
}) {
  const [err, setErr] = useState(false);

  if (err || !abbrev) {
    return (
      <span
        className={`inline-flex items-center justify-center text-[10px] font-mono text-white/30 shrink-0 ${className}`}
        style={{ width: smSize ?? size, height: smSize ?? size }}
      >
        {abbrev || "—"}
      </span>
    );
  }

  const makeImg = (imgSize: number) => (
    <img
      src={logoUrl(abbrev)}
      alt={abbrev}
      width={imgSize}
      height={imgSize}
      onError={() => setErr(true)}
      className="object-contain"
      style={{ width: imgSize, height: imgSize }}
      draggable={false}
    />
  );

  const content = smSize ? (
    <>
      <span className="sm:hidden inline-flex">{makeImg(size)}</span>
      <span className="hidden sm:inline-flex">{makeImg(smSize)}</span>
    </>
  ) : makeImg(size);

  return (
    <Link
      href={`/teams/${abbrev}`}
      onClick={e => e.stopPropagation()}
      className={`team-logo-link inline-flex shrink-0 items-center justify-center ${className}`}
      title={`${abbrev} team page`}
    >
      {content}
    </Link>
  );
}
