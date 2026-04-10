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
// All selectors/observers run in the wrapper document context only —
// cross-origin stream iframe content is inaccessible by design.
const WRAPPER_INJECT = `<script>
(function(){
  // 1. Block window.open in the wrapper context (catches ads injected here)
  try{window.open=function(){return{closed:true,focus:function(){},location:{href:''}}};}catch(e){}

  // 2. Refocus on blur — kills popunder focus-steal pattern
  window.addEventListener('blur',function(){setTimeout(function(){try{window.focus();}catch(e){}},0);},true);

  // 3. Block target="_blank" / "_new" / "_top" anchor clicks in wrapper context
  document.addEventListener('click',function(e){
    var el=e.target;
    while(el&&el.tagName!=='A')el=el.parentElement;
    if(el&&el.target&&/^(_blank|_new|_top)$/i.test(el.target)){
      e.preventDefault();e.stopImmediatePropagation();
    }
  },true);

  // 4. MutationObserver — remove ad iframes/high-z overlays injected into wrapper doc
  var streamIframe=null;
  function killAdNode(node){
    if(!node||!node.tagName)return;
    // Any iframe that isn't the stream player → kill it
    if(node.tagName==='IFRAME'&&node!==streamIframe){try{node.remove();}catch(e){}return;}
    // Fixed/absolute divs with suspiciously high z-index → kill them
    if(/^(DIV|SECTION|ASIDE|FIGURE)$/.test(node.tagName)){
      try{
        var st=window.getComputedStyle(node);
        var z=parseInt(st.zIndex,10);
        if((st.position==='fixed'||st.position==='absolute')&&z>9000){node.remove();}
      }catch(e){}
    }
  }
  var obs=new MutationObserver(function(muts){
    muts.forEach(function(m){m.addedNodes.forEach(killAdNode);});
  });

  // 5. Auto-dismiss consent dialogs / ad overlays; auto-click play buttons
  var C=['[class*="close" i]','[class*="dismiss" i]','[class*="skip" i]','[id*="close" i]',
    '[id*="dismiss" i]','[aria-label*="close" i]','.ad-close','#ad-close','.skip-ad',
    '.fc-cta-consent','#didomi-notice-agree-button','.qc-cmp2-summary-buttons button',
    '[class*="cookie" i] button','[id*="cookie" i] button','[class*="consent" i] button'],
    P=['button.jw-icon-display','.vjs-big-play-button','[class*="play-btn" i]',
    '[aria-label*="play" i]','button[class*="play" i]','.plyr__control--overlaid'];
  function click(s){s.forEach(function(q){try{document.querySelectorAll(q).forEach(function(el){if(el&&el.offsetParent!==null)el.click();});}catch(e){}});}
  function run(){click(C);setTimeout(function(){click(P);},200);}

  function init(){
    streamIframe=document.querySelector('iframe');
    obs.observe(document.body,{childList:true,subtree:false});
    run();
    setTimeout(run,300);setTimeout(run,800);
  }

  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',function(){init();});
  else init();
  var t=setInterval(run,1500);setTimeout(function(){clearInterval(t);},30000);
})();
</script>`;

function isMobile(ua: string): boolean {
  return /android|iphone|ipad|ipod|mobile|tablet/i.test(ua);
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

  const safeUrl = parsedUrl.href.replace(/"/g, "%22").replace(/'/g, "%27");
  const ua = request.headers.get("user-agent") ?? "";
  const priorityParam = request.nextUrl.searchParams.get("priority");
  const priority = priorityParam != null ? parseInt(priorityParam, 10) : 3;

  // Sandbox policy:
  //   mobile (all streams): always sandbox — popups disruptive on mobile
  //   desktop priority 0-1 (direct m3u8, known-clean embeds): sandbox — these providers
  //     don't check for sandbox, and they're the ones most worth protecting
  //   desktop priority 2-3 (aggregators, iframe-only): no sandbox — these providers
  //     actively detect and refuse to load under sandbox
  const useSandbox = isMobile(ua) || priority <= 1;
  const sandboxAttr = useSandbox
    ? `sandbox="allow-scripts allow-same-origin allow-forms allow-presentation allow-orientation-lock allow-modals" `
    : "";

  const html = `<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Permissions-Policy" content="geolocation=(), camera=(), microphone=(), payment=()">
<style>*{margin:0;padding:0;box-sizing:border-box}html,body{width:100%;height:100%;background:#000;overflow:hidden}iframe{position:absolute;inset:0;width:100%;height:100%;border:0}</style>
${WRAPPER_INJECT}
</head><body>
<iframe src="${safeUrl}" ${sandboxAttr}allow="autoplay; fullscreen; picture-in-picture; encrypted-media" allowfullscreen referrerpolicy="no-referrer-when-downgrade"></iframe>
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
