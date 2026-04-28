import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 15;

/**
 * Stream embed proxy — server-side passthrough mode.
 *
 * Why we can't just iframe the third-party host:
 *   uBlock / AdGuard / Brave / Pi-hole have rules that block third-party
 *   iframes from known popup/ad hosts (embedsports.top, dlstreams.com,
 *   abcsport.top, ...). The user's browser refuses the iframe load — Chrome
 *   shows "host refused to connect" inside the chip — even though the host
 *   is reachable on direct navigation. Bob's report: every backup chip
 *   shows that error in-page; tapping "Open / Cast ↗" works.
 *
 * Fix: fetch the host HTML server-side and serve it from /api/stream-embed
 * (localhost:3000). The browser only sees a same-origin iframe load, so the
 * ad-blocker's third-party-iframe rule no longer triggers. Trade-offs:
 *   - Players that check window.location will see localhost:3000 and may refuse
 *     to initialise. Auto-skip is gone, so the user just clicks "try next ›".
 *   - The host's relative URLs (jquery, bundle-jw.js, m3u8, ...) need to
 *     resolve to the original host, not localhost:3000 — handled with <base href>.
 *   - Origin-absolute paths (/api/foo) won't be caught by <base href>; we
 *     intercept fetch/XHR client-side and rewrite them to the original host.
 *   - Ad scripts injected by the host still try to fire — stripped server-
 *     side and blocked again client-side via the same WRAPPER_INJECT shim
 *     used previously.
 */

const WRAPPER_INJECT = `<script>
(function(){
  var _BG_HOST=(typeof _BG_HOST!=='undefined')?_BG_HOST:'';

  // 1. Block popup/popunder windows
  try{window.open=function(){return{closed:true,focus:function(){},location:{href:''}}};}catch(e){}

  // 2. Trap top-frame navigation attempts
  try{
    var _locRef=window.location;
    var _noop=function(){};
    try{Object.defineProperty(window,'location',{get:function(){return _locRef;},set:_noop,configurable:true});}catch(e){}
    try{_locRef.assign=_noop;}catch(e){}
    try{_locRef.replace=_noop;}catch(e){}
  }catch(e){}

  // 3. Ad-network blocklist (used in 4 + 5 + DOM observer)
  var AD=new RegExp('popads\\\\.net|exoclick\\\\.com|trafficjunky|adnxs\\\\.com|doubleclick\\\\.net|popunder|clickunder|trafficstars|adsterra|hilltopads|juicyads|propellerads|monetizer|adcash|revcontent|outbrain|taboola|popcash|plugrush|ero-advertising|fuckingfast\\\\.co|cdn77ads|adspyglass|aclib','i');

  // 4. Rewrite origin-absolute fetch/XHR back to the original host. Players
  //    that hard-code "/api/..." or "/stream/..." would 404 against localhost:3000;
  //    rewriting to "<host>/api/..." routes them back to the real upstream.
  //    Also drops requests to ad networks entirely.
  function rewriteUrl(u){
    if(typeof u!=='string')return u;
    if(AD.test(u))return null;
    if(u.charAt(0)==='/' && u.charAt(1)!=='/' && _BG_HOST){return _BG_HOST+u;}
    return u;
  }
  try{
    var _oFetch=window.fetch;
    window.fetch=function(input,init){
      var u=typeof input==='string'?input:(input&&input.url?input.url:'');
      var rw=rewriteUrl(u);
      if(rw===null)return new Promise(function(){});
      if(rw!==u && typeof input==='string')input=rw;
      return _oFetch.apply(this,[input,init]);
    };
  }catch(e){}
  try{
    var _oOpen=XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open=function(m,url){
      var rw=rewriteUrl(url||'');
      if(rw===null){this._adBlocked=true;return;}
      var args=[].slice.call(arguments);args[1]=rw;
      return _oOpen.apply(this,args);
    };
    var _oSend=XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.send=function(){
      if(this._adBlocked)return;
      return _oSend.apply(this,arguments);
    };
  }catch(e){}

  // 5. Force every runtime <iframe>/<frame> src through /api/stream-embed and
  //    block ad-host <script> srcs. Three injection paths to cover:
  //    a) document.createElement('iframe').src = X
  //    b) el.innerHTML = '<iframe src=X>' (parser path; createElement bypassed)
  //    c) someIframe.setAttribute('src', X) (sets attribute, not the JS prop)
  //    The earlier per-element hook covered (a) only — ad networks routinely
  //    use innerHTML to inject their player iframes, which slipped past it
  //    and the user saw "exposestrat.com refused to connect" when their
  //    browser refused the X-Frame-Options-DENY load.
  //    Fix: hook the prototype-level src setter + setAttribute, so EVERY
  //    iframe in the page proxifies regardless of construction path. Same
  //    pattern for <script> with the ad-host blocklist.
  function proxify(raw){
    if(typeof raw!=='string'||!raw)return raw;
    if(/^\\/api\\/stream-embed\\?/.test(raw))return raw;
    try{
      var abs=new URL(raw,_BG_HOST||document.baseURI).href;
      if(!/^https?:/i.test(abs))return raw;
      return '/api/stream-embed?url='+encodeURIComponent(abs);
    }catch(e){return raw;}
  }
  // Prototype-level src hook — catches createElement, innerHTML-parsed,
  // cloned, document.write'd iframes; anything that ends up calling the
  // .src setter goes through proxify.
  try{
    var _ifDesc=Object.getOwnPropertyDescriptor(HTMLIFrameElement.prototype,'src');
    if(_ifDesc&&_ifDesc.set){
      Object.defineProperty(HTMLIFrameElement.prototype,'src',{
        set:function(v){_ifDesc.set.call(this,proxify(v));},
        get:_ifDesc.get,
        configurable:true
      });
    }
    if(window.HTMLFrameElement){
      var _frDesc=Object.getOwnPropertyDescriptor(HTMLFrameElement.prototype,'src');
      if(_frDesc&&_frDesc.set){
        Object.defineProperty(HTMLFrameElement.prototype,'src',{
          set:function(v){_frDesc.set.call(this,proxify(v));},
          get:_frDesc.get,
          configurable:true
        });
      }
    }
  }catch(e){}
  // <script> src hook — same prototype-level pattern; blocks ad-host scripts
  // injected via innerHTML or appendChild.
  try{
    var _scDesc=Object.getOwnPropertyDescriptor(HTMLScriptElement.prototype,'src');
    if(_scDesc&&_scDesc.set){
      Object.defineProperty(HTMLScriptElement.prototype,'src',{
        set:function(v){var rw=rewriteUrl(v||'');if(rw!==null)_scDesc.set.call(this,rw);},
        get:_scDesc.get,
        configurable:true
      });
    }
  }catch(e){}
  // setAttribute hook — covers iframe.setAttribute('src', X), the third
  // injection path. Element.setAttribute('src',...) on iframes/scripts goes
  // here BEFORE the parser-driven src reflection runs.
  try{
    var _oSetAttr=Element.prototype.setAttribute;
    Element.prototype.setAttribute=function(name,value){
      if(name && (name+'').toLowerCase()==='src' && this.tagName){
        var T=this.tagName;
        if(T==='IFRAME'||T==='FRAME')value=proxify(value);
        else if(T==='SCRIPT'){var rw=rewriteUrl(value||'');if(rw===null)return;value=rw;}
      }
      return _oSetAttr.call(this,name,value);
    };
  }catch(e){}

  // 6. Block automatic fullscreen
  try{
    var _noFS=function(){return Promise.resolve();};
    Element.prototype.requestFullscreen=_noFS;
    if('webkitRequestFullscreen' in Element.prototype)Element.prototype.webkitRequestFullscreen=_noFS;
    if(window.HTMLVideoElement&&'webkitEnterFullscreen' in HTMLVideoElement.prototype){
      HTMLVideoElement.prototype.webkitEnterFullscreen=function(){};
      HTMLVideoElement.prototype.webkitExitFullscreen=function(){};
    }
  }catch(e){}

  // 7. Refocus on blur
  window.addEventListener('blur',function(){setTimeout(function(){try{window.focus();}catch(e){}},0);},true);

  // 8. Block target=_blank/_top anchor clicks
  document.addEventListener('click',function(e){
    var el=e.target;
    while(el&&el.tagName!=='A')el=el.parentElement;
    if(el&&el.target&&/^(_blank|_new|_top)$/i.test(el.target)){
      e.preventDefault();e.stopImmediatePropagation();
    }
  },true);

  // 9. MutationObserver — last line of defence:
  //    a) strip ad iframes (matches AD blocklist) and high-z overlays
  //    b) re-proxify any iframe whose src ended up un-proxied. The setter
  //       hook above catches most paths, but iframes parsed from innerHTML
  //       can occasionally race — by the time the observer fires the load
  //       has been queued; rewriting src here cancels the in-flight load
  //       and re-issues it through /api/stream-embed.
  function killAdNode(node){
    if(!node||!node.tagName)return;
    var tag=node.tagName;
    if(tag==='IFRAME'||tag==='FRAME'){
      try{
        var s=node.getAttribute('src')||'';
        if(AD.test(s)){node.remove();return;}
        if(s && /^https?:/i.test(s) && !/^\\/api\\/stream-embed\\?/.test(s)){
          node.setAttribute('src',proxify(s));
        }
      }catch(e){}
      return;
    }
    if(/^(DIV|SECTION|ASIDE|FIGURE|SPAN|A)$/.test(tag)){
      try{
        var st=window.getComputedStyle(node);
        var z=parseInt(st.zIndex,10);
        if((st.position==='fixed'||st.position==='absolute')&&z>9000){node.remove();return;}
      }catch(e){}
    }
    if(tag==='SCRIPT'){
      try{if(AD.test(node.src||'')){node.src='';node.remove();}}catch(e){}
    }
  }
  var obs=new MutationObserver(function(muts){muts.forEach(function(m){m.addedNodes.forEach(killAdNode);});});

  function init(){
    obs.observe(document.body,{childList:true,subtree:true});
    // Heartbeat — tells the parent the wrapper rendered, used for chip-success ranking
    var _streamUrl=(typeof _BG_URL!=='undefined')?_BG_URL:'';
    setInterval(function(){
      try{window.parent.postMessage({type:'Origin-frame-progress',url:_streamUrl},'*');}catch(e){}
    },1500);
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);
  else init();
})();
</script>`;

const AD_HOST_RE = /(popads\.net|exoclick\.com|trafficjunky|adnxs\.com|doubleclick\.net|popunder|clickunder|trafficstars|adsterra|hilltopads|juicyads|propellerads|monetizer|adcash|revcontent|outbrain|taboola|popcash|plugrush|ero-advertising|fuckingfast\.co|cdn77ads|adspyglass|aclib|al5sm\.com|kzt2afc1rp52\.com|beggingmusty\.com)/i;

function stripAdScripts(html: string): string {
  // Remove <script src="..."> pointing to ad networks
  html = html.replace(/<script[^>]*\ssrc=["'][^"']*["'][^>]*>\s*<\/script>/gi, (m) =>
    AD_HOST_RE.test(m) ? "" : m,
  );
  // Remove inline <script>...</script> that calls aclib.runPop / popads / etc.
  html = html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, (m) =>
    /aclib\.runpop|aclib\.runautopop|popads|popunder|adsterra|propellerads/i.test(m) ? "" : m,
  );
  return html;
}

// The host page often has <iframe src="https://other-aggregator.tld/..."> —
// that nested third-party iframe gets blocked by the user's ad-blocker the
// same way the outer one was, defeating our proxy. Rewrite every absolute-
// URL iframe/frame src to recurse through /api/stream-embed so the inner
// load is also same-origin.
function rewriteNestedIframes(html: string, baseHref: string): string {
  const proxyOf = (raw: string): string => {
    try {
      const abs = new URL(raw, baseHref).href;
      if (!/^https?:/i.test(abs)) return raw;
      return `/api/stream-embed?url=${encodeURIComponent(abs)}`;
    } catch {
      return raw;
    }
  };
  return html.replace(
    /(<(?:iframe|frame)\b[^>]*?\bsrc=)(["'])(.*?)\2/gi,
    (_m, prefix, q, src) => `${prefix}${q}${proxyOf(src)}${q}`,
  );
}

function injectIntoHead(html: string, snippet: string): string {
  // Prefer an existing <head> open tag.
  if (/<head[^>]*>/i.test(html)) {
    return html.replace(/<head[^>]*>/i, (m) => `${m}${snippet}`);
  }
  // No <head> — many of these hosts ship `<!doctype><html><meta ...>` and
  // rely on the parser to auto-create a head. Slot our snippet right after
  // <html> so it lives in the implicit head rather than dangling above
  // <!doctype>.
  if (/<html[^>]*>/i.test(html)) {
    return html.replace(/<html[^>]*>/i, (m) => `${m}<head>${snippet}</head>`);
  }
  if (/<!doctype[^>]*>/i.test(html)) {
    return html.replace(/<!doctype[^>]*>/i, (m) => `${m}<head>${snippet}</head>`);
  }
  return `<head>${snippet}</head>${html}`;
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

  const baseHref = `${parsedUrl.protocol}//${parsedUrl.host}/`;
  const hostOrigin = `${parsedUrl.protocol}//${parsedUrl.host}`;

  // Server-side fetch the host HTML. Onhockey is the canonical referer for
  // most of these aggregator pages; without it many of them serve a 403 or
  // a "direct access denied" body.
  let upstreamHtml: string;
  try {
    const upstream = await fetch(parsedUrl.href, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        Referer: "https://onhockey.tv/",
        Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
      },
      // 8 s ceiling — aggregator pages should respond fast; if they don't,
      // the chip is dead and we want to bail rather than blow the 15 s
      // serverless cap.
      signal: AbortSignal.timeout(8000),
      redirect: "follow",
    });
    if (!upstream.ok) {
      return new NextResponse(`Upstream ${upstream.status}`, { status: 502 });
    }
    // Some hosts (onhockey-derivatives) ship windows-1251; most ship utf-8.
    // We let the browser sort out odd charsets — Node fetch returns text via
    // the Content-Type charset hint, defaulting to utf-8 when none is given.
    upstreamHtml = await upstream.text();
  } catch {
    return new NextResponse("Upstream fetch failed", { status: 502 });
  }

  // Server-side ad strip first, then inject our preamble in the head.
  // Order matters: <base href> first so subresource URLs resolve correctly,
  // then context vars + dark style, then WRAPPER_INJECT last so its shims
  // are installed before any inline script in the body runs.
  const safeOrigin = JSON.stringify(hostOrigin);
  const safeUrl = JSON.stringify(parsedUrl.href);
  const baseTag = `<base href="${baseHref}">`;
  const ctxVars = `<script>var _BG_HOST=${safeOrigin};var _BG_URL=${safeUrl};</script>`;
  // Defensive dark-bg style so a host that forgets to set a body color
  // doesn't show as a white screen for the brief load handshake.
  const darkStyle = `<style>html,body{background:#000 !important;color-scheme:dark}body{margin:0}</style>`;
  const preamble = `${baseTag}${ctxVars}${darkStyle}${WRAPPER_INJECT}`;

  let html = stripAdScripts(upstreamHtml);
  html = rewriteNestedIframes(html, baseHref);
  html = injectIntoHead(html, preamble);

  return new NextResponse(html, {
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Access-Control-Allow-Origin": "*",
      // No-store — these are short-lived live streams; caching a token-
      // signed page would serve stale URLs to the next viewer.
      "Cache-Control": "no-store",
      "Referrer-Policy": "no-referrer",
      "Permissions-Policy": "geolocation=(), camera=(), microphone=(), payment=()",
    },
  });
}
