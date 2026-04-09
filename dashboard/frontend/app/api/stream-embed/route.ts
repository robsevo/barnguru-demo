import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 15;

/**
 * Stream embed proxy — wrapper mode (all clients).
 *
 * Returns a thin HTML wrapper with a plain <iframe src="original-url">.
 * The player runs at its native origin so all self-checks pass, HLS/XHR
 * requests go directly to the stream domain (no CORS issues), and the
 * stream plays normally on both desktop and mobile.
 *
 * Ad blocking is handled in the wrapper context (not the stream iframe):
 *   - window.open overridden to a no-op → kills popup/popunder tabs
 *   - blur/focus trap → refocuses the page if a popunder steals focus
 *   - auto-clicker → dismisses consent overlays and ad close buttons
 *   - Permissions-Policy header → blocks dangerous browser feature requests
 *
 * Note: ad scripts baked into the stream player itself are not strippable
 * via this approach (would require proxy mode which breaks CORS). The
 * popup/overlay blocking above covers the most disruptive ad formats.
 */

// Injected into the wrapper page (not the stream iframe).
// Kills popups, refocuses on blur (popunder defence), auto-clicks overlays.
const WRAPPER_INJECT = `<script>
(function(){
  // Block all window.open calls (popup / popunder ads)
  try{window.open=function(){return{closed:true,focus:function(){},location:{href:''}}};}catch(e){}
  // Refocus immediately if anything steals focus (popunder pattern)
  window.addEventListener('blur',function(){setTimeout(function(){try{window.focus();}catch(e){}},0);},true);
  // Auto-dismiss consent dialogs and ad close buttons; auto-click play buttons
  var C=['[class*="close" i]','[class*="dismiss" i]','[class*="skip" i]','[id*="close" i]',
    '[id*="dismiss" i]','[aria-label*="close" i]','.ad-close','#ad-close','.skip-ad',
    '.fc-cta-consent','#didomi-notice-agree-button','.qc-cmp2-summary-buttons button',
    '[class*="cookie" i] button','[id*="cookie" i] button','[class*="consent" i] button'],
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

  const safeUrl = parsedUrl.href.replace(/"/g, "%22").replace(/'/g, "%27");

  const html = `<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Permissions-Policy" content="geolocation=(), camera=(), microphone=(), payment=()">
<style>*{margin:0;padding:0;box-sizing:border-box}html,body{width:100%;height:100%;background:#000;overflow:hidden}iframe{position:absolute;inset:0;width:100%;height:100%;border:0}</style>
${WRAPPER_INJECT}
</head><body>
<iframe src="${safeUrl}" allow="autoplay; fullscreen; picture-in-picture; encrypted-media" allowfullscreen referrerpolicy="no-referrer-when-downgrade"></iframe>
</body></html>`;

  return new NextResponse(html, {
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Access-Control-Allow-Origin": "*",
      "Cache-Control": "no-cache",
      "Permissions-Policy": "geolocation=(), camera=(), microphone=(), payment=(), interest-cohort=()",
    },
  });
}
