import type { Metadata, Viewport } from "next";
import { Barlow, Barlow_Condensed } from "next/font/google";
import "./globals.css";
import AppShell from "./components/AppShell";

const barlow = Barlow({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800", "900"],
  variable: "--font-sans",
  display: "swap",
});

const barlowCondensed = Barlow_Condensed({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800", "900"],
  variable: "--font-condensed",
  display: "swap",
});

export const metadata: Metadata = {
  title: "GRTZKY",
  description: "Bayesian Analytics and Rating Network",
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
    ],
    apple: "/apple-touch-icon.png",
  },
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "GRTZKY",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // Open the TLS connection to the stream proxy before the user clicks any chip
  // so the first manifest fetch skips the ~80-150 ms TCP+TLS handshake.
  const apiOrigin = process.env.NEXT_PUBLIC_API_URL || "";
  return (
    <html lang="en">
      <head>
        {apiOrigin ? (
          <>
            <link rel="preconnect" href={apiOrigin} crossOrigin="anonymous" />
            <link rel="dns-prefetch" href={apiOrigin} />
          </>
        ) : null}
      </head>
      <body className={`${barlow.variable} ${barlowCondensed.variable}`}>
        <AppShell>
          {children}
        </AppShell>
      </body>
    </html>
  );
}
