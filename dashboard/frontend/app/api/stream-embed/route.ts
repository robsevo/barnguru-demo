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
 *   - window.location overridden → blocks redirect-based ads
 *   - fetch/XHR patched → blocks requests to known ad domains
 *   - document.createElement patched → blocks injected ad scripts
 *   - blur/focus trap → refocuses the page if a popunder steals focus
 *   - MutationObserver → removes injected ad iframes and overlays
 *   - auto-clicker → dismisses consent overlays and ad close buttons
 *   - Permissions-Policy header → blocks dangerous browser feature requests
 *   - Referrer-Policy: no-referrer → denies referrer-based ad targeting
 *
 * Note: ad scripts baked into the stream player itself are not strippable
 * via this approach (would require proxy mode which breaks CORS). The
 * popup/overlay/redirect blocking above covers the most disruptive ad formats.
 */

// Injected into the wrapper page (not the stream iframe).
// All selectors/observers run in the wrapper document context only —
// cross-origin stream iframe content is inaccessible by design.
const WRAPPER_INJECT = `<script>
(function(){
  // 1. Block window.open in the wrapper context (catches ads injected here)
  try{window.open=function(){return{closed:true,focus:function(){},location:{href:''}}};}catch(e){}

  // 2. Block window.location redirects — ad scripts use location.href/assign/replace
  //    to navigate away from the stream. Override them to no-ops.
  try{
    var _locRef=window.location;
    var _noop=function(){};
    try{
      Object.defineProperty(window,'location',{get:function(){return _locRef;},set:_noop,configurable:true});
    }catch(e){}
    try{_locRef.assign=_noop;}catch(e){}
    try{_locRef.replace=_noop;}catch(e){}
    try{window.history.pushState=_noop;}catch(e){}
    try{window.history.replaceState=_noop;}catch(e){}
  }catch(e){}

  // 3. Block fetch/XHR to known ad networks — drop the request entirely.
  var AD=new RegExp(
    'popads\\.net|exoclick\\.com|trafficjunky|adnxs\\.com|doubleclick\\.net|'+
    'popunder|clickunder|trafficstars|adsterra|hilltopads|juicyads|propellerads|'+
    'monetizer|adcash|revcontent|outbrain|taboola|popcash|plugrush|ero-advertising|'+
    'fuckingfast\\.co|cdn77ads|adspyglass','i'
  );
  try{
    var _oFetch=window.fetch;
    window.fetch=function(input,init){
      var u=typeof input==='string'?input:(input&&input.url?input.url:'');
      if(AD.test(u))return new Promise(function(){});
      return _oFetch.apply(this,arguments);
    };
  }catch(e){}
  try{
    var _oOpen=XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open=function(m,url){
      if(AD.test(url||'')){this._adBlocked=true;return;}
      return _oOpen.apply(this,arguments);
    };
    var _oSend=XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.send=function(){
      if(this._adBlocked)return;
      return _oSend.apply(this,arguments);
    };
  }catch(e){}

  // 4. Block injected ad <script> elements by intercepting createElement
  try{
    var _oCreate=document.createElement.bind(document);
    document.createElement=function(tag){
      var el=_oCreate(tag);
      if((tag+'').toLowerCase()==='script'){
        var _desc=Object.getOwnPropertyDescriptor(HTMLScriptElement.prototype,'src');
        if(_desc&&_desc.set){
          Object.defineProperty(el,'src',{
            set:function(v){if(!AD.test(v||''))_desc.set.call(this,v);},
            get:function(){return _desc.get?_desc.get.call(this):'';},
            configurable:true
          });
        }
      }
      return el;
    };
  }catch(e){}

  // 5. Refocus on blur — kills popunder focus-steal pattern
  window.addEventListener('blur',function(){setTimeout(function(){try{window.focus();}catch(e){}},0);},true);

  // 6. Block target="_blank" / "_new" / "_top" anchor clicks in wrapper context
  document.addEventListener('click',function(e){
    var el=e.target;
    while(el&&el.tagName!=='A')el=el.parentElement;
    if(el&&el.target&&/^(_blank|_new|_top)$/i.test(el.target)){
      e.preventDefault();e.stopImmediatePropagation();
    }
  },true);

  // 7. MutationObserver — remove ad iframes/high-z overlays/suspicious anchors injected into wrapper doc
  var streamIframe=null;
  function killAdNode(node){
    if(!node||!node.tagName)return;
    var tag=node.tagName;
    // Any iframe that isn't the stream player → kill it
    if(tag==='IFRAME'&&node!==streamIframe){try{node.remove();}catch(e){}return;}
    // Fixed/absolute elements with suspiciously high z-index → kill them
    if(/^(DIV|SECTION|ASIDE|FIGURE|SPAN|A)$/.test(tag)){
      try{
        var st=window.getComputedStyle(node);
        var z=parseInt(st.zIndex,10);
        if((st.position==='fixed'||st.position==='absolute')&&z>9000){node.remove();return;}
      }catch(e){}
    }
    // Anchor tags added to body pointing to ad domains → kill them
    if(tag==='A'){
      try{if(AD.test(node.href||''))node.remove();}catch(e){}
    }
    // Script tags pointing to ad domains injected into the wrapper → block src
    if(tag==='SCRIPT'){
      try{if(AD.test(node.src||'')){node.src='';node.remove();}}catch(e){}
    }
  }
  var obs=new MutationObserver(function(muts){
    muts.forEach(function(m){m.addedNodes.forEach(killAdNode);});
  });

  // 8. Auto-dismiss consent dialogs / ad overlays; auto-click play buttons
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
    obs.observe(document.body,{childList:true,subtree:true});
    run();
    setTimeout(run,300);setTimeout(run,800);

    // 9. Sandbox failure detection — only when _BG_SANDBOX flag is set.
    if(typeof _BG_SANDBOX!=='undefined'&&_BG_SANDBOX&&streamIframe){
      var _playerActive=false;
      window.addEventListener('message',function(e){
        if(streamIframe&&e.source===streamIframe.contentWindow)_playerActive=true;
      });
      document.addEventListener('click',function(){_playerActive=true;},{once:true});
      streamIframe.addEventListener('load',function(){
        setTimeout(function(){
          if(!_playerActive){
            try{window.parent.postMessage({type:'Origin-sandbox-fail',url:typeof _BG_URL!=='undefined'?_BG_URL:''},'*');}catch(e){}
          }
        },3000);
      });
    }
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
  //   Mobile: always sandbox — blocks popup ads. If the provider detects sandbox
  //     and refuses to load, the user can tap "Open / Cast ↗" in native browser.
  //   Desktop priority 0-1: sandbox — these providers don't check for it.
  //   Desktop priority 2-3: no sandbox — these providers detect and reject it,
  //     and on desktop we can try yt-dlp extraction as a fallback.
  const mobile = isMobile(ua);
  const useSandbox = mobile || priority <= 1;
  const sandboxAttr = useSandbox
    ? `sandbox="allow-scripts allow-same-origin allow-forms allow-presentation allow-orientation-lock allow-modals" `
    : "";

  // On mobile: sandbox is forced for popup blocking but disable the 3s auto-skip
  // (_BG_SANDBOX) — streams take longer to initialise on mobile.
  const bgVars = useSandbox && !mobile
    ? `<script>var _BG_SANDBOX=1,_BG_URL=${JSON.stringify(safeUrl)};</script>`
    : "";

  const html = `<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Permissions-Policy" content="geolocation=(), camera=(), microphone=(), payment=()">
<style>*{margin:0;padding:0;box-sizing:border-box}html,body{width:100%;height:100%;background:#000;overflow:hidden}iframe{position:absolute;inset:0;width:100%;height:100%;border:0}</style>
${bgVars}${WRAPPER_INJECT}
</head><body>
<iframe src="${safeUrl}" ${sandboxAttr}allow="autoplay; fullscreen; picture-in-picture; encrypted-media" allowfullscreen referrerpolicy="no-referrer-when-downgrade"></iframe>
</body></html>`;

  return new NextResponse(html, {
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Access-Control-Allow-Origin": "*",
      "Cache-Control": "no-cache",
      "Referrer-Policy": "no-referrer",
      "Permissions-Policy": "geolocation=(), camera=(), microphone=(), payment=(), interest-cohort=()",
    },
  });
}
