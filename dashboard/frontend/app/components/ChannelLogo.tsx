// Self-hosted channel logos for IPTV chips. Used by /barncentre and the
// /game/[id] StreamPanel so the same network always renders the same way.
// Files live under public/logos/.
export const CH_LOGO: Record<string, string> = {
  TSN1: "/logos/tsn.svg",  TSN2: "/logos/tsn.svg",
  TSN3: "/logos/tsn.svg",  TSN4: "/logos/tsn.svg",  TSN5: "/logos/tsn.svg",
  "Sportsnet East":    "/logos/sportsnet.svg",
  "Sportsnet Ontario": "/logos/sportsnet.svg",
  "Sportsnet West":    "/logos/sportsnet.svg",
  "Sportsnet Pacific": "/logos/sportsnet.svg",
  "Sportsnet 360":     "/logos/sportsnet.svg",
  "Sportsnet One":     "/logos/sportsnet.svg",
  Sportsnet:           "/logos/sportsnet.svg",
  ESPN:         "/logos/espn.svg",
  ESPN2:        "/logos/espn2.png",
  "ESPN+":      "/logos/espn.svg",
  "NHL Network":"/logos/nhlnetwork.svg",
  FS1:          "/logos/fs1.svg",
  FS2:          "/logos/fs2.svg",
  NESN:         "/logos/nesn.png",
  FanDuel:      "/logos/fanduel.svg",
  "Bally Sports":       "/logos/bally.svg",
  "CBS Sports":         "/logos/cbssports.svg",
  "CBS Sports Network": "/logos/cbssportsnet.svg",
  RDS:          "/logos/rds.svg",
  "RDS 2":      "/logos/rds2.svg",
  "RDS INFO":   "/logos/rdsinfo.svg",
  "RDS Info":   "/logos/rdsinfo.svg",
  "TVA Sports":   "/logos/tvasports.svg",
  "TVA Sports 2": "/logos/tvasports.svg",
  TNT:          "/logos/tnt.svg",
  TBS:          "/logos/tbs.svg",
};

// SVGs with transparent backgrounds need white-silhouette treatment for dark UI.
// Sportsnet and NHL Network have their own coloured backgrounds — show as-is.
export const CH_LOGO_INVERT = new Set([
  "TSN1","TSN2","TSN3","TSN4","TSN5",
  "ESPN","ESPN+","FS1",
]);

// Brand logos with dark elements (black text, navy fills) that need a light
// backdrop to stay legible on the dark UI. The chip below gets a white pill.
export const CH_LOGO_LIGHT_BG = new Set([
  "RDS", "RDS 2", "RDS INFO", "RDS Info",
  "TVA Sports", "TVA Sports 2",
  "Sportsnet East", "Sportsnet Ontario", "Sportsnet West", "Sportsnet Pacific",
  "Sportsnet 360", "Sportsnet One", "Sportsnet",
  "CBS Sports", "CBS Sports Network",
]);

export function ChannelLogo({ name, height = 26 }: { name: string; height?: number }) {
  const src = CH_LOGO[name];
  if (!src) return null;
  const invert  = CH_LOGO_INVERT.has(name);
  const lightBg = CH_LOGO_LIGHT_BG.has(name);
  const img = (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt={name}
      style={{
        height: `${height}px`,
        width: "auto",
        maxWidth: `${height * 3.5}px`,
        objectFit: "contain",
        filter: invert ? "brightness(0) invert(1)" : undefined,
        opacity: invert ? 0.88 : 0.92,
        display: "block",
      }}
      onError={e => { (e.target as HTMLImageElement).style.display = "none"; }}
    />
  );
  if (lightBg) {
    return (
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#ffffff",
          borderRadius: `${Math.max(4, height * 0.18)}px`,
          padding: `${Math.max(2, height * 0.1)}px ${Math.max(3, height * 0.18)}px`,
        }}
      >
        {img}
      </span>
    );
  }
  return img;
}
