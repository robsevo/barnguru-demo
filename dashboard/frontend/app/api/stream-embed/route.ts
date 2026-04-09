import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 15;

/**
 * Stream embed proxy — dual-mode.
 *
 * DESKTOP: wrapper mode
 *   Return a thin HTML wrapper with a plain <iframe src="original-url">.
 *   Player runs at its native origin, passes all self-checks, plays.
 *   Desktop Chrome/Safari/Firefox block popup ads without needing sandbox.
 *
 * MOBILE: proxy mode
 *   Mobile browsers allow more popups. We fetch the embed HTML server-side,
 *   strip ad scripts and sandbox-detection code, and return the stripped HTML.
 *   A dynamic spoof script overrides window.top/parent, document.referrer, and
 *   window.open so the page believes it's running at its native origin.
 */

const AD_DOMAINS = "aclib\\.net|popads\\.net|popcash\\.net|propellerads\\.com|exoclick\\.com|trafficjunky\\.net|tsyndicate\\.com|adspyglass\\.com|adtelligent\\.com|newcpmagate\\.com|realsrv\\.com|trafficstars\\.com|hilltopads\\.net|plugrush\\.com|adcash\\.com|juicyads\\.com|bidvertiser\\.com|adsterra\\.com|doubleclick\\.net|googlesyndication\\.com|amazon-adsystem\\.com|adnxs\\.com|rubiconproject\\.com|pubmatic\\.com|openx\\.net|criteo\\.com|moonicorn\\.com|seedr\\.cc|trafficforce\\.com|smartadserver\\.com|advertising\\.com|media\\.net|taboola\\.com|outbrain\\.com|revcontent\\.com|mgid\\.com|content\\.ad|sharethrough\\.com|triplelift\\.com|33across\\.com|sovrn\\.com|lijit\\.com|appnexus\\.com|indexexchange\\.com|spotxchange\\.com|spotx\\.tv|springserve\\.com|adskeeper\\.co\\.uk|adblade\\.com|conversantmedia\\.com|undertone\\.com|rhythmone\\.com|yieldmo\\.com";
const AD_SRC_RE = new RegExp(`src=(["'])(https?:)?//[^"']*(?:${AD_DOMAINS})[^"']*\\1`, "gi");

const AD_INLINE_RE = [
  /<script[^>]*>[\s\S]*?aclib\.runPop\s*\([\s\S]*?<\/script>/gi,
  /<script[^>]*>[\s\S]*?\.runPop\s*\([\s\S]*?<\/script>/gi,
  /<script[^>]*>[\s\S]*?popunder[\s\S]*?<\/script>/gi,
  /<script[^>]*src=(["'])[^"']*(?:aclib|popads|popcash|propellerads|exoclick|trafficjunky|newcpmagate|realsrv)\.[^"']*\1[^>]*>[\s\S]*?<\/script>/gi,
  /<script[^>]*src=(["'])[^"']*(?:aclib|popads|popcash|propellerads|exoclick|trafficjunky|newcpmagate|realsrv)\.[^"']*\1[^>]*\/>/gi,
];

const SANDBOX_DETECT_RE = [
  /\$\(["']body["']\)\.append\s*\([^)]*application\/pdf[^)]*\)\s*;?/g,
  /const\s+sandDetect\s*=\s*function[^;]+;/g,
  /const\s+bodyMsg\s*=\s*function[\s\S]*?return\s+true\s*\}\s*;/g,
];

// Static auto-click inject (desktop wrapper)
const AUTO_CLICK_INJECT = `<script>
(function(){
  try{window.open=function(){return{closed:true,focus:function(){}}};}catch(e){}
  var C=['[class*="close" i]','[class*="dismiss" i]','[class*="skip" i]','[id*="close" i]',
    '[id*="dismiss" i]','[aria-label*="close" i]','.ad-close','#ad-close','.skip-ad',
    '.fc-cta-consent','#didomi-notice-agree-button','.qc-cmp2-summary-buttons button'],
    P=['button.jw-icon-display','.vjs-big-play-button','[class*="play-btn" i]',
    '[aria-label*="play" i]','button[class*="play" i]','.plyr__control--overlaid'];
  function click(s){s.forEach(function(q){try{document.querySelectorAll(q).forEach(function(el){if(el&&el.offsetParent!==null)el.click();});}catch(e){}});}
  function run(){click(C);setTimeout(function(){click(P);},200);}
  run();
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',function(){run();setTimeout(run,500);});
  else{setTimeout(run,300);setTimeout(run,800);}
  var t=setInterval(run,1500);setTimeout(function(){clearInterval(t);},30000);
})();
</script>`;

// Dynamic mobile inject — spoofs iframe detection + referrer so stream pages
// believe they're running at their own origin, not embedded in localhost:3000.
function mobileInject(streamOrigin: string): string {
  const safeOrigin = streamOrigin.replace(/"/g, "");
  return `<script>
(function(){
  var O=${JSON.stringify(safeOrigin)};
  try{Object.defineProperty(window,'top',{get:function(){return window;}});
  Object.defineProperty(window,'parent',{get:function(){return window;}});
  Object.defineProperty(window,'frameElement',{get:function(){return null;}});}catch(e){}
  try{Object.defineProperty(document,'referrer',{get:function(){return O+'/';}});}catch(e){}
  try{window.open=function(){return{closed:true,focus:function(){}}};}catch(e){}
  var C=['[class*="close" i]','[class*="dismiss" i]','[class*="skip" i]','[id*="close" i]',
    '[id*="dismiss" i]','[aria-label*="close" i]','.ad-close','#ad-close','.skip-ad',
    '.fc-cta-consent','#didomi-notice-agree-button','.qc-cmp2-summary-buttons button'],
    P=['button.jw-icon-display','.vjs-big-play-button','[class*="play-btn" i]',
    '[aria-label*="play" i]','button[class*="play" i]','.plyr__control--overlaid'];
  function click(s){s.forEach(function(q){try{document.querySelectorAll(q).forEach(function(el){if(el&&el.offsetParent!==null)el.click();});}catch(e){}});}
  function run(){click(C);setTimeout(function(){click(P);},200);}
  run();
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',function(){run();setTimeout(run,500);});
  else{setTimeout(run,300);setTimeout(run,800);}
  var t=setInterval(run,1500);setTimeout(function(){clearInterval(t);},30000);
})();
</script>`;
}

function isMobileUA(ua: string): boolean {
  return /Mobile|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua);
}

export async function GET(request: NextRequest) {
  const url = request.nextUrl.searchParams.get("url");
  if (!url) return new NextResponse("Missing url", { status: 400 });

  let parsedUrl: URL;
  try {
    parsedUrl = new URL(url);
    if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
      return new NextResponse("Invalid url", { status: 400 });
    }
  } catch {
    return new NextResponse("Invalid url", { status: 400 });
  }

  const cors = {
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "no-cache",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=(), payment=(), interest-cohort=()",
  };
  const ua = request.headers.get("user-agent") ?? "";

  // ── DESKTOP: wrapper — direct iframe at native origin, browser popup blocker ──
  if (!isMobileUA(ua)) {
    const safeUrl = parsedUrl.href.replace(/"/g, "%22").replace(/'/g, "%27");
    const html = `<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Permissions-Policy" content="geolocation=(), camera=(), microphone=(), payment=()">
<style>*{margin:0;padding:0;box-sizing:border-box}html,body{width:100%;height:100%;background:#000;overflow:hidden}iframe{position:absolute;inset:0;width:100%;height:100%;border:0}</style>
<script>(function(){try{window.open=function(){return{closed:true,focus:function(){}};}}catch(e){}window.addEventListener('blur',function(){setTimeout(function(){window.focus&&window.focus();},0);},true);})();</script>
${AUTO_CLICK_INJECT}
</head><body>
<iframe src="${safeUrl}" allow="autoplay; fullscreen; picture-in-picture; encrypted-media" allowfullscreen referrerpolicy="no-referrer-when-downgrade"></iframe>
</body></html>`;
    return new NextResponse(html, {
      headers: { ...cors, "Content-Type": "text/html; charset=utf-8" },
    });
  }

  // ── MOBILE: proxy — fetch + strip ads + spoof iframe/referrer detection ──────
  try {
    const origin = new URL(url).origin;
    // Preserve the mobile UA so the stream site serves mobile-compatible player HTML
    const fetchUA = ua || "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1";
    const res = await fetch(url, {
      headers: {
        "User-Agent": fetchUA,
        "Referer": origin + "/",
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
      },
      redirect: "follow",
      signal: AbortSignal.timeout(12000),
    });

    if (!res.ok) {
      return new NextResponse(`upstream ${res.status}`, { status: 502, headers: cors });
    }

    let html = await res.text();

    // Absolutify relative URLs
    html = html.replace(/(src|href|action)=(["'])\/\/([^"']+)\2/g, `$1=$2https://$3$2`);
    html = html.replace(/(src|href|action)=(["'])\/([^/"'#][^"']*)\2/g, `$1=$2${origin}/$3$2`);

    // Strip ad scripts
    for (const re of AD_INLINE_RE) html = html.replace(re, "");
    html = html.replace(AD_SRC_RE, 'src=$1/dev/null$1');

    // Strip sandbox/iframe detection code
    for (const re of SANDBOX_DETECT_RE) html = html.replace(re, "");

    // Inject spoof (window.top, document.referrer) + auto-clicker
    const inject = mobileInject(origin);
    html = html.replace(/<\/head>/i, `${inject}</head>`);
    if (!html.includes(inject)) html = inject + html;

    return new NextResponse(html, {
      headers: { ...cors, "Content-Type": "text/html; charset=utf-8" },
    });
  } catch (e) {
    return new NextResponse(String(e), { status: 502, headers: cors });
  }
}
