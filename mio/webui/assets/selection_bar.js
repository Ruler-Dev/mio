// selection_bar.js — selection-contextual action bar for main-chat
// artifacts (not just Design Mode).
//
// Monkey-patches HTMLIFrameElement.prototype.srcdoc so that any new
// .artifact-iframe gets a tiny click-capture script injected before
// it boots. On click inside the iframe, the iframe posts a
// {__mioSelect, selector, tag, textHint, outer, rect} message. The
// parent renders a floating action bar at the click coord with:
//   Extract to prompt  — inserts outerHTML into the composer
//   Copy outerHTML     — clipboard write
//   Regenerate just this — seeds a scoped prompt into the composer
//   Ask about this      — prepends "What does this do?" + the HTML
//
// Only fires for new iframes created after this module loads, which
// in practice means "all of them" because the module lives in the
// safe-loader.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.selectionBar) return;

  // 1) Monkey-patch srcdoc setter so we can inject our capture script.
  try {
    const proto = HTMLIFrameElement.prototype;
    const orig = Object.getOwnPropertyDescriptor(proto, "srcdoc");
    if (orig && orig.set) {
      Object.defineProperty(proto, "srcdoc", {
        configurable: true,
        get: orig.get,
        set(val) {
          try {
            if (this.classList?.contains("artifact-iframe") && typeof val === "string") {
              val = inject(val);
            }
          } catch {}
          return orig.set.call(this, val);
        },
      });
    }
  } catch { /* older engine — just no selection bar */ }

  function inject(html) {
    const script = `
<script>
(function(){
  // Opt-out: allow the artifact to block selection by setting a flag.
  if (window.__mioNoSelect) return;
  function cssPath(el){
    if (!(el instanceof Element)) return '';
    const parts = [];
    while (el && el.nodeType === 1 && parts.length < 6) {
      let name = el.nodeName.toLowerCase();
      if (el.id) { name += '#' + el.id; parts.unshift(name); break; }
      const cls = (el.className||'').toString().trim().split(/\\s+/).filter(Boolean).slice(0,2);
      if (cls.length) name += '.' + cls.join('.');
      const parent = el.parentElement;
      if (parent) {
        const sibs = [...parent.children].filter(c => c.nodeName === el.nodeName);
        if (sibs.length > 1) name += ':nth-of-type(' + (sibs.indexOf(el) + 1) + ')';
      }
      parts.unshift(name);
      el = parent;
    }
    return parts.join(' > ');
  }
  // Alt+click picks an element. Plain click still works as designed.
  document.addEventListener('click', function(e){
    if (!e.altKey) return;
    var t = e.target;
    if (!(t instanceof Element)) return;
    e.preventDefault(); e.stopPropagation();
    var rect = t.getBoundingClientRect();
    try {
      parent.postMessage({
        __mioSelect: true,
        selector: cssPath(t),
        tag: t.tagName.toLowerCase(),
        textHint: (t.innerText || '').trim().slice(0, 80),
        outer: (t.outerHTML || '').slice(0, 2000),
        rect: {x: rect.left, y: rect.top, w: rect.width, h: rect.height},
      }, '*');
    } catch (_) {}
  }, true);
})();
</script>`;
    if (/<\/head>/i.test(html)) return html.replace(/<\/head>/i, script + "</head>");
    if (/<body/i.test(html))    return html.replace(/<body([^>]*)>/i, "<body$1>" + script);
    return script + html;
  }

  // 2) Listen for the postMessage and render the floating bar.
  let currentBar = null;
  window.addEventListener("message", (e) => {
    if (!e.data || e.data.__mioSelect !== true) return;
    // Figure out which iframe the message came from to position the bar.
    const frames = document.querySelectorAll("iframe.artifact-iframe");
    let origin = null;
    for (const f of frames) if (f.contentWindow === e.source) { origin = f; break; }
    if (!origin) return;
    showBar(origin, e.data);
  });

  function showBar(iframe, pick) {
    closeBar();
    const frameRect = iframe.getBoundingClientRect();
    const bar = document.createElement("div");
    bar.className = "mio-select-bar";
    bar.style.left = Math.max(8, frameRect.left + pick.rect.x) + "px";
    bar.style.top  = Math.max(40, frameRect.top  + pick.rect.y + pick.rect.h + 8) + "px";
    bar.innerHTML = `
      <div class="mio-select-tag">${escapeHtml(pick.tag)}${pick.textHint ? " · " + escapeHtml(pick.textHint) : ""}</div>
      <button data-act="regen"   title="Draft a scoped regen prompt">Regenerate just this</button>
      <button data-act="ask"     title="Ask Mio about this element">Ask about this</button>
      <button data-act="extract" title="Insert HTML into composer">Extract</button>
      <button data-act="copy"    title="Copy outerHTML to clipboard">Copy</button>
      <button data-act="close"   aria-label="Close">×</button>
    `;
    document.body.appendChild(bar);
    currentBar = bar;
    bar.querySelector('[data-act="close"]').addEventListener("click", closeBar);
    bar.querySelector('[data-act="regen"]').addEventListener("click", () => {
      toInput(`Regenerate only the ${pick.tag} element${pick.textHint ? ` "${pick.textHint}"` : ""} at \`${pick.selector}\`. Keep everything else exactly as it is.`);
      closeBar();
    });
    bar.querySelector('[data-act="ask"]').addEventListener("click", () => {
      toInput(`What does this element do and how could I improve it?\n\n\`\`\`html\n${pick.outer}\n\`\`\``);
      closeBar();
    });
    bar.querySelector('[data-act="extract"]').addEventListener("click", () => {
      toInput("```html\n" + pick.outer + "\n```\n");
      closeBar();
    });
    bar.querySelector('[data-act="copy"]').addEventListener("click", () => {
      navigator.clipboard?.writeText(pick.outer || "");
      bar.querySelector('[data-act="copy"]').textContent = "✓ copied";
      setTimeout(closeBar, 500);
    });
    // Dismiss on outside click
    setTimeout(() => {
      const h = (ev) => { if (!bar.contains(ev.target)) { closeBar(); document.removeEventListener("mousedown", h, true); } };
      document.addEventListener("mousedown", h, true);
    }, 50);
  }

  function closeBar() {
    currentBar?.remove();
    currentBar = null;
  }

  function toInput(text) {
    const input = document.querySelector("textarea#messageInput, textarea#input, textarea.input, textarea");
    if (!input) return;
    input.value = (input.value ? input.value + "\n\n" : "") + text;
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }

  window.Mio.selectionBar = { closeBar };
})();
