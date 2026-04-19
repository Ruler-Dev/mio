// view_design.js — Design Mode.
//
// Two-pane canvas inspired by Claude's artifact mode + Google Stitch:
//   left:  prompt composer + vibe chips + history log (compact)
//   right: artifact preview with version scrubber and Preview / Code tabs
//
// Each prompt fires the model via /v1/chat/completions with a design-
// focused system prompt that strongly biases toward a single
// <antArtifact> containing a React/Tailwind page. The response is
// captured as a new numbered version and rendered in a sandboxed
// iframe. Versions stack up — scrub back through any of them without
// losing what came after.

(function () {
  window.Mio = window.Mio || {};
  const ready = () => {
    if (!window.Mio.views) return setTimeout(ready, 50);
    window.Mio.views.register("design", {
      title: "Design",
      mount(host) { renderRoot(host); },
    });
  };
  ready();

  // --- State -----------------------------------------------------------

  const STORAGE_KEY = "mio.design.session";
  const DEFAULT_VIBES = [
    "minimal", "dark mode", "playful", "premium B2B",
    "brutalist", "glassmorphism", "retro 80s", "dense dashboard",
    "editorial", "warm earthy", "neon arcade", "monochrome",
  ];

  const SYSTEM_PROMPT = `You are a senior UI/UX engineer. The user is designing a web page or component.

Always respond with a SHORT intro sentence, then a SINGLE <antArtifact> tag containing a fully self-contained HTML document the user can preview in an iframe. No external build step: use React + Tailwind via CDN with Babel standalone, or plain HTML/CSS if that's enough. Keep code tight and polished; animations OK when they help.

Artifact template:
<antArtifact identifier="design-v{N}" type="text/html" title="Short title of this design">
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
</head>
<body class="bg-neutral-50">
<div id="root"></div>
<script type="text/babel">
// your React component here
const App = () => (<div className="…">…</div>);
ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
</script>
</body>
</html>
</antArtifact>

No explanations after the artifact. The artifact IS the design.`;

  const state = loadSession();

  function loadSession() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) return JSON.parse(raw);
    } catch {}
    return { versions: [], history: [], activeVersion: -1 };
  }
  function saveSession() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch {}
  }

  // --- Render ----------------------------------------------------------

  function renderRoot(host) {
    host.innerHTML = `
      <div class="view-design">
        <aside class="design-left">
          <header class="design-left-head">
            <h1>Design Mode</h1>
            <button class="btn-ghost" data-action="reset">New session</button>
          </header>
          <div class="design-history" id="design-history"></div>
          <div class="design-composer">
            <div class="design-vibes" id="design-vibes"></div>
            <div class="design-refs" id="design-refs"></div>
            <textarea class="design-input" id="design-input" rows="3" placeholder="Describe what you want to design… Paste a screenshot to use as visual reference."></textarea>
            <div class="design-composer-foot">
              <div class="design-scope" id="design-scope" title="Patch: surgical edit, fast, keeps unrelated parts. Rewrite: full regenerate from scratch.">
                <button class="design-scope-btn" data-scope="patch">Patch</button>
                <button class="design-scope-btn" data-scope="rewrite">Rewrite</button>
              </div>
              <label class="design-check" title="Fire 3 parallel generations with temperature jitter, pick the best">
                <input type="checkbox" id="design-variants"> 3 variants
              </label>
              <div style="flex:1"></div>
              <button class="btn-ghost design-generate" data-action="generate">Generate</button>
            </div>
          </div>
        </aside>
        <main class="design-right">
          <header class="design-right-head">
            <div class="design-tabs" role="tablist">
              <button class="design-tab active" data-tab="preview" role="tab">Preview</button>
              <button class="design-tab"        data-tab="code"    role="tab">Code</button>
              <button class="design-tab"        data-tab="diff"    role="tab">Diff</button>
            </div>
            <div class="design-widths" role="group" aria-label="Preview width">
              <button class="design-width" data-width="mobile"  title="Mobile · 375 px"  aria-label="Mobile">📱</button>
              <button class="design-width" data-width="tablet"  title="Tablet · 768 px"  aria-label="Tablet">📲</button>
              <button class="design-width" data-width="desktop" title="Desktop · 1280 px" aria-label="Desktop">🖥️</button>
              <button class="design-width active" data-width="fit" title="Fit pane" aria-label="Fit">⛶</button>
            </div>
            <div class="design-version-label" id="design-version-label">No design yet</div>
            <div style="flex:1"></div>
            <button class="btn-ghost" data-action="inspect" title="Click an element in the preview to edit it">Inspect</button>
            <button class="btn-ghost" data-action="tokens" title="Tweak colors, radii, fonts live (no model call)">Tokens</button>
            <button class="btn-ghost" data-action="fork" title="Fork a variant from this version">Fork</button>
            <button class="btn-ghost" data-action="copy">Copy HTML</button>
            <button class="btn-ghost" data-action="download">Download</button>
          </header>
          <div class="design-canvas" id="design-canvas"></div>
          <footer class="design-scrubber" id="design-scrubber"></footer>
        </main>
      </div>
    `;
    renderVibes(host);
    renderHistory(host);
    renderVersions(host);
    wireHandlers(host);
  }

  // --- Reference images (paste / drop) ---------------------------------
  // Held in memory only (not persisted) — kept on `state._refs` for the
  // current render cycle; cleared after Generate so they don't bloat
  // every subsequent turn.
  state._refs = state._refs || [];

  async function addReference(host, file) {
    // Downscale to <= 1024 px longest edge + compress to JPEG so we
    // don't blow through the model's pixel budget on a 5-MB screenshot.
    try {
      const dataUrl = await compressImage(file, 1024, 0.86);
      state._refs.push({ name: file.name || "pasted.png", dataUrl, bytes: approxBytes(dataUrl) });
      renderReferences(host);
    } catch (e) {
      console.warn("[design] image compress failed:", e);
    }
  }

  function renderReferences(host) {
    const wrap = host.querySelector("#design-refs");
    if (!wrap) return;
    if (!state._refs.length) { wrap.innerHTML = ""; wrap.hidden = true; return; }
    wrap.hidden = false;
    wrap.innerHTML = state._refs.map((r, i) => `
      <div class="design-ref-chip" data-idx="${i}">
        <img src="${r.dataUrl}" alt="">
        <div class="design-ref-chip-meta">
          <span class="design-ref-chip-name">${escapeHtml(r.name)}</span>
          <span class="design-ref-chip-size">${kib(r.bytes)}</span>
        </div>
        <button class="design-ref-chip-close" aria-label="Remove">×</button>
      </div>
    `).join("");
    wrap.querySelectorAll(".design-ref-chip-close").forEach((btn, i) => {
      btn.addEventListener("click", () => {
        state._refs.splice(i, 1);
        renderReferences(host);
      });
    });
  }

  function compressImage(file, maxEdge, quality) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const img = new Image();
        img.onload = () => {
          const ratio = Math.min(1, maxEdge / Math.max(img.width, img.height));
          const w = Math.round(img.width * ratio);
          const h = Math.round(img.height * ratio);
          const canvas = document.createElement("canvas");
          canvas.width = w; canvas.height = h;
          canvas.getContext("2d").drawImage(img, 0, 0, w, h);
          resolve(canvas.toDataURL("image/jpeg", quality));
        };
        img.onerror = reject;
        img.src = reader.result;
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  function approxBytes(dataUrl) {
    const b64 = dataUrl.split(",")[1] || "";
    return Math.round(b64.length * 0.75);
  }

  function kib(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
    return `${(n / 1024 / 1024).toFixed(1)} MiB`;
  }

  // --- Scope classifier (Patch vs Rewrite) -----------------------------
  // Patch keywords favour surgical edits; Rewrite keywords favour full
  // regeneration. When neither matches, default to Patch when there's an
  // existing active version (iterating), Rewrite otherwise (first turn).
  const PATCH_HINTS = /\b(change|tweak|adjust|edit|update|fix|make\s+(the|it|this|them)|swap|replace\s+(only|just)|rename|recolor|increase|decrease|reduce|bigger|smaller|darker|lighter|move|shift|shrink|grow|hide|show|add\s+(a|an|one|the))\b/i;
  const REWRITE_HINTS = /\b(redo|redesign|rewrite|from\s+scratch|start\s+over|new\s+(layout|design)|completely|entirely\s+different|reimagine|totally\s+different|scrap\s+it)\b/i;

  function classifyScope(prompt) {
    if (REWRITE_HINTS.test(prompt)) return "rewrite";
    if (PATCH_HINTS.test(prompt))   return "patch";
    return state.versions.length > 0 ? "patch" : "rewrite";
  }

  function applyScopeChip(host, scope) {
    state._scope = scope;
    saveSession();
    host.querySelectorAll(".design-scope-btn").forEach((b) =>
      b.classList.toggle("active", b.dataset.scope === scope));
  }

  function onPromptInput(host) {
    if (state._scopeLocked) return;
    const input = host.querySelector("#design-input");
    applyScopeChip(host, classifyScope(input.value || ""));
  }

  function renderVibes(host) {
    const wrap = host.querySelector("#design-vibes");
    wrap.innerHTML = "";
    for (const v of DEFAULT_VIBES) {
      const chip = document.createElement("button");
      chip.className = "design-vibe";
      chip.textContent = v;
      chip.addEventListener("click", () => {
        const input = host.querySelector("#design-input");
        const sep = input.value && !input.value.endsWith(" ") ? " " : "";
        input.value += `${sep}[${v}] `;
        input.focus();
      });
      wrap.appendChild(chip);
    }
  }

  function renderHistory(host) {
    const wrap = host.querySelector("#design-history");
    if (!state.history.length) {
      wrap.innerHTML = `<div class="muted" style="padding:10px 14px;font-size:12px">No messages yet. Try a vibe chip.</div>`;
      return;
    }
    wrap.innerHTML = state.history.map((m, i) => `
      <div class="design-msg design-msg-${m.role}">
        <div class="design-msg-role">${m.role}</div>
        <div class="design-msg-text">${escapeHtml(m.text || "")}</div>
      </div>
    `).join("");
    wrap.scrollTop = wrap.scrollHeight;
  }

  function renderVersions(host) {
    const scrubber = host.querySelector("#design-scrubber");
    const label = host.querySelector("#design-version-label");
    scrubber.innerHTML = "";
    if (!state.versions.length) {
      label.textContent = "No design yet";
      host.querySelector("#design-canvas").innerHTML = `
        <div class="design-empty">
          <p>Describe a page, click a vibe, hit Generate.</p>
        </div>
      `;
      return;
    }
    for (let i = 0; i < state.versions.length; i++) {
      const v = state.versions[i];
      const dot = document.createElement("button");
      dot.className = "design-version" + (i === state.activeVersion ? " active" : "");
      dot.textContent = "v" + (i + 1);
      dot.title = v.title || "";
      dot.addEventListener("click", () => switchVersion(host, i));
      scrubber.appendChild(dot);
    }
    const active = state.versions[state.activeVersion] || state.versions[state.versions.length - 1];
    label.textContent = `v${state.activeVersion + 1} · ${active.title || "design"}`;
    showVersion(host, active);
  }

  function showVersion(host, v) {
    const canvas = host.querySelector("#design-canvas");
    const tab = host.querySelector(".design-tab.active").dataset.tab;

    // Variant comparison takes precedence over the normal view — once
    // the user has fired "3 variants", show them side-by-side until
    // they click one to keep.
    if (state._pendingCompare) {
      const group = state._pendingCompare;
      const variants = state.versions.filter((ver) => ver.variantGroup === group);
      if (variants.length > 1) {
        renderVariantGrid(host, canvas, variants);
        return;
      }
      // Single-member group; clear the flag and fall through.
      state._pendingCompare = null;
    }
    if (tab === "code") {
      canvas.innerHTML = `<pre class="design-code"><code>${escapeHtml(v.html || "")}</code></pre>`;
    } else if (tab === "diff") {
      const idx = state.versions.indexOf(v);
      const prev = idx > 0 ? state.versions[idx - 1] : null;
      if (!prev) {
        canvas.innerHTML = `<div class="design-empty"><p>No earlier version to diff against.</p></div>`;
      } else {
        canvas.innerHTML = `<div class="design-diff">${renderLineDiff(prev.html || "", v.html || "")}</div>`;
      }
    } else {
      canvas.innerHTML = "";
      // Preview frame — wrapped in a sizer so we can constrain the iframe
      // to real device widths (375 / 768 / 1280) or let it fill ("fit").
      const sizer = document.createElement("div");
      sizer.className = "design-sizer";
      canvas.appendChild(sizer);

      const errorBar = document.createElement("div");
      errorBar.className = "design-error-bar";
      errorBar.hidden = true;
      canvas.appendChild(errorBar);

      const iframe = document.createElement("iframe");
      iframe.className = "design-iframe";
      iframe.sandbox = "allow-scripts";
      iframe.srcdoc = injectRuntimeHelpers(v.html || "", { inspect: !!state._inspect });
      sizer.appendChild(iframe);
      applyWidth(sizer, state._width || "fit");

      // Messages from the iframe — errors + element clicks in inspect mode.
      const handler = (e) => {
        if (!e.data || e.source !== iframe.contentWindow) return;
        if (e.data.__mioDesignError === true) {
          showError(host, errorBar, e.data);
        } else if (e.data.__mioDesignPick === true) {
          onElementPicked(host, e.data);
        }
      };
      window.addEventListener("message", handler);
      iframe.addEventListener("load", () => {
        errorBar.hidden = true;
        // Re-apply any sticky token overrides to the freshly loaded iframe
        const overrides = state._tokenOverrides || {};
        for (const [name, value] of Object.entries(overrides)) {
          applyTokenOverride(host, name, value);
        }
      });
    }
  }

  function renderVariantGrid(host, canvas, variants) {
    canvas.innerHTML = `
      <div class="design-compare">
        <div class="design-compare-head">
          <strong>Pick a variant</strong>
          <span class="muted">${variants.length} generated · click "Keep" to continue with that one</span>
          <button class="btn-ghost" data-action="skip-compare">Skip (keep all)</button>
        </div>
        <div class="design-compare-grid" style="grid-template-columns: repeat(${variants.length}, 1fr);"></div>
      </div>
    `;
    const grid = canvas.querySelector(".design-compare-grid");
    variants.forEach((v, i) => {
      const cell = document.createElement("div");
      cell.className = "design-compare-cell";
      cell.innerHTML = `
        <div class="design-compare-label">${escapeHtml(v.title)}</div>
        <div class="design-compare-frame-wrap"></div>
        <div class="design-compare-actions">
          <button class="btn-ghost design-compare-keep" data-keep="${state.versions.indexOf(v)}">Keep this one</button>
        </div>
      `;
      const wrap = cell.querySelector(".design-compare-frame-wrap");
      const iframe = document.createElement("iframe");
      iframe.className = "design-compare-frame";
      iframe.sandbox = "allow-scripts";
      iframe.srcdoc = injectRuntimeHelpers(v.html || "", { inspect: false });
      wrap.appendChild(iframe);
      cell.querySelector(".design-compare-keep").addEventListener("click", () => {
        state._pendingCompare = null;
        state.activeVersion = parseInt(cell.querySelector(".design-compare-keep").dataset.keep, 10);
        saveSession();
        renderVersions(host);
      });
      grid.appendChild(cell);
    });
    canvas.querySelector('[data-action="skip-compare"]').addEventListener("click", () => {
      state._pendingCompare = null;
      saveSession();
      renderVersions(host);
    });
  }

  function applyWidth(sizer, width) {
    const map = { mobile: 375, tablet: 768, desktop: 1280 };
    if (width === "fit" || !map[width]) {
      sizer.style.width = "100%";
      sizer.style.maxWidth = "none";
    } else {
      sizer.style.width = map[width] + "px";
      sizer.style.maxWidth = "100%";
    }
  }

  // Injects small helper scripts into the artifact HTML:
  //   - error reporter (always on) — forwards JS errors to parent
  //   - inspect mode (opt-in) — on click, outlines the element and
  //     posts a selector + outerHTML snippet back to parent
  // The iframe has sandbox="allow-scripts" only, so postMessage is
  // the only channel back.
  function injectRuntimeHelpers(html, { inspect = false } = {}) {
    const errorScript = `
<script>
(function(){
  function post(err){ try { parent.postMessage({__mioDesignError:true, message:String(err.message||err), stack:String(err.stack||''), source:String(err.filename||''), line:err.lineno||0, col:err.colno||0}, '*'); } catch(_){} }
  window.addEventListener('error', (e) => post({message: e.message, filename: e.filename, lineno: e.lineno, colno: e.colno, stack: e.error?.stack}));
  window.addEventListener('unhandledrejection', (e) => post({message: 'Unhandled rejection: ' + (e.reason?.message || e.reason), stack: e.reason?.stack}));
  // Live token patcher — parent posts {__mioDesignTokenSet, name, value}
  // and we rewrite that CSS custom property on :root.
  window.addEventListener('message', (e) => {
    if (!e.data || e.data.__mioDesignTokenSet !== true) return;
    try { document.documentElement.style.setProperty(e.data.name, e.data.value); } catch(_) {}
  });
})();
</script>`;
    const inspectScript = !inspect ? "" : `
<style>
  html.__mio_insp, html.__mio_insp * { cursor: crosshair !important; }
  html.__mio_insp *:hover { outline: 2px dashed #7aa2f7 !important; outline-offset: 2px; }
  .__mio_insp_picked { outline: 2px solid #7aa2f7 !important; outline-offset: 2px; box-shadow: 0 0 0 4px rgba(122,162,247,0.25) !important; }
</style>
<script>
(function(){
  document.documentElement.classList.add('__mio_insp');
  function cssPath(el){
    if (!(el instanceof Element)) return '';
    const parts = [];
    while (el && el.nodeType === 1 && parts.length < 6) {
      let name = el.nodeName.toLowerCase();
      if (el.id) { name += '#' + el.id; parts.unshift(name); break; }
      const cls = (el.className || '').toString().trim().split(/\\s+/).filter(Boolean).slice(0,2);
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
  document.addEventListener('click', function(e){
    e.preventDefault(); e.stopPropagation();
    const t = e.target;
    document.querySelectorAll('.__mio_insp_picked').forEach(n => n.classList.remove('__mio_insp_picked'));
    try { t.classList.add('__mio_insp_picked'); } catch(_){}
    const outer = (t.outerHTML || '').slice(0, 800);
    const textHint = (t.innerText || '').trim().slice(0, 80);
    try {
      parent.postMessage({__mioDesignPick:true, selector: cssPath(t), tag: t.tagName.toLowerCase(), textHint, outer}, '*');
    } catch(_) {}
  }, true);
})();
</script>`;
    const head = errorScript + inspectScript;
    if (/<\/head>/i.test(html)) return html.replace(/<\/head>/i, head + "</head>");
    if (/<body/i.test(html))    return html.replace(/<body([^>]*)>/i, "<body$1>" + head);
    return head + html;
  }

  function onElementPicked(host, pick) {
    const input = host.querySelector("#design-input");
    const hint = pick.textHint ? ` "${pick.textHint}"` : "";
    const seed = `Change this specific element (${pick.tag}${hint}) at \`${pick.selector}\` so it `;
    input.value = seed;
    input.focus();
    // Park the caret at the end so the user can type the change.
    input.setSelectionRange(input.value.length, input.value.length);
    // Seed the next gen with current HTML so we iterate in place.
    const v = state.versions[state.activeVersion];
    if (v) state._forkSeed = v.html;
    // Turn inspect off after a pick so accidental clicks don't recur.
    toggleInspect(host, false);
  }

  function toggleInspect(host, on) {
    state._inspect = !!on;
    saveSession();
    const btn = host.querySelector('[data-action="inspect"]');
    if (btn) btn.classList.toggle("active", !!on);
    // Re-render so the iframe gets the helper script refreshed.
    const v = state.versions[state.activeVersion];
    if (v) showVersion(host, v);
  }

  function showError(host, bar, err) {
    bar.hidden = false;
    bar.innerHTML = `
      <div class="design-error-left">
        <strong>Iframe error:</strong>
        <span class="design-error-msg">${escapeHtml(err.message || "")}</span>
      </div>
      <button class="design-error-fix" data-action="try-fix">Try fixing with Mio</button>
      <button class="design-error-close" data-action="close" aria-label="Close">×</button>
    `;
    bar.querySelector('[data-action="close"]').addEventListener("click", () => { bar.hidden = true; });
    bar.querySelector('[data-action="try-fix"]').addEventListener("click", () => {
      const input = host.querySelector("#design-input");
      const trace = err.stack ? err.stack.slice(0, 600) : "";
      input.value = `Fix this error in the current design:\n\n${err.message}\n\n${trace}`.trim();
      input.focus();
      // Seed the fork so we pass the current HTML back as context
      const v = state.versions[state.activeVersion];
      if (v) state._forkSeed = v.html;
    });
  }

  // Minimal line-level diff — no external deps. Good enough for
  // spotting "changed this section" at a glance. Not an LCS-optimal
  // diff; we just mark lines that don't appear in the other version.
  function renderLineDiff(a, b) {
    const aLines = a.split("\n");
    const bLines = b.split("\n");
    const aSet = new Set(aLines);
    const bSet = new Set(bLines);
    const out = [];
    let i = 0, j = 0;
    while (i < aLines.length || j < bLines.length) {
      const la = aLines[i];
      const lb = bLines[j];
      if (i < aLines.length && j < bLines.length && la === lb) {
        out.push({ k: " ", l: la });
        i++; j++;
      } else if (i < aLines.length && !bSet.has(la)) {
        out.push({ k: "-", l: la });
        i++;
      } else if (j < bLines.length && !aSet.has(lb)) {
        out.push({ k: "+", l: lb });
        j++;
      } else if (i < aLines.length && j < bLines.length) {
        // Both present but misaligned; advance both
        out.push({ k: "-", l: la });
        out.push({ k: "+", l: lb });
        i++; j++;
      } else if (i < aLines.length) {
        out.push({ k: "-", l: la });
        i++;
      } else {
        out.push({ k: "+", l: lb });
        j++;
      }
    }
    return out.map((r) =>
      `<div class="design-diff-line design-diff-${r.k === '+' ? 'add' : r.k === '-' ? 'del' : 'eq'}"><span class="design-diff-mark">${r.k}</span>${escapeHtml(r.l)}</div>`
    ).join("");
  }

  function switchVersion(host, i) {
    state.activeVersion = i;
    saveSession();
    renderVersions(host);
  }

  // --- Wiring ----------------------------------------------------------

  function wireHandlers(host) {
    const input = host.querySelector("#design-input");
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        generate(host);
      }
    });
    input.addEventListener("input", () => onPromptInput(host));
    // Paste image → add as reference chip
    input.addEventListener("paste", async (e) => {
      const files = Array.from(e.clipboardData?.items || [])
        .filter((it) => it.kind === "file" && it.type.startsWith("image/"))
        .map((it) => it.getAsFile())
        .filter(Boolean);
      if (!files.length) return;
      e.preventDefault();
      for (const f of files) await addReference(host, f);
    });
    // Drag-drop image onto composer
    const composer = host.querySelector(".design-composer");
    ["dragover", "dragenter"].forEach((ev) => {
      composer.addEventListener(ev, (e) => { e.preventDefault(); composer.classList.add("drop-hover"); });
    });
    ["dragleave", "drop"].forEach((ev) => {
      composer.addEventListener(ev, (e) => composer.classList.remove("drop-hover"));
    });
    composer.addEventListener("drop", async (e) => {
      e.preventDefault();
      const files = Array.from(e.dataTransfer?.files || []).filter((f) => f.type.startsWith("image/"));
      for (const f of files) await addReference(host, f);
    });
    renderReferences(host);
    // Scope chip — click to override the auto-classification
    host.querySelectorAll(".design-scope-btn").forEach((b) => {
      b.addEventListener("click", () => {
        state._scopeLocked = true;
        applyScopeChip(host, b.dataset.scope);
      });
    });
    // Initial scope (auto-classify empty prompt → defaults)
    onPromptInput(host);
    host.querySelector('[data-action="generate"]').addEventListener("click", () => generate(host));
    host.querySelector('[data-action="reset"]').addEventListener("click", () => {
      if (!confirm("Clear this design session? Versions will be lost.")) return;
      state.versions = []; state.history = []; state.activeVersion = -1;
      saveSession();
      renderRoot(host);
    });
    host.querySelector('[data-action="copy"]').addEventListener("click", () => {
      const v = state.versions[state.activeVersion];
      if (!v) return;
      navigator.clipboard?.writeText(v.html || "");
    });
    host.querySelector('[data-action="download"]').addEventListener("click", () => {
      const v = state.versions[state.activeVersion];
      if (!v) return;
      const blob = new Blob([v.html || ""], { type: "text/html" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = (v.title || "design") + ".html";
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    });
    host.querySelector('[data-action="fork"]').addEventListener("click", () => {
      const v = state.versions[state.activeVersion];
      if (!v) return;
      const input = host.querySelector("#design-input");
      const prior = v.prompt ? `\n(forked from v${state.activeVersion + 1}: "${v.prompt}")` : "";
      input.value = `Start from this design and ` + prior;
      input.focus();
      // Cursor between "and " and the prior-note so the user can type
      const pos = "Start from this design and ".length;
      input.setSelectionRange(pos, pos);
      // Seed the next generation with the active version's HTML as context
      state._forkSeed = v.html;
    });
    host.querySelectorAll(".design-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        host.querySelectorAll(".design-tab").forEach((t) => t.classList.toggle("active", t === tab));
        const v = state.versions[state.activeVersion];
        if (v) showVersion(host, v);
      });
    });
    // Width toggles (mobile / tablet / desktop / fit)
    host.querySelectorAll(".design-width").forEach((btn) => {
      btn.addEventListener("click", () => {
        state._width = btn.dataset.width;
        saveSession();
        host.querySelectorAll(".design-width").forEach((b) =>
          b.classList.toggle("active", b === btn));
        const sizer = host.querySelector(".design-sizer");
        if (sizer) applyWidth(sizer, state._width);
      });
    });
    // Restore persisted width on mount
    if (state._width) {
      host.querySelectorAll(".design-width").forEach((b) =>
        b.classList.toggle("active", b.dataset.width === state._width));
    }
    // Inspect toggle
    host.querySelector('[data-action="inspect"]').addEventListener("click", () => {
      toggleInspect(host, !state._inspect);
    });
    if (state._inspect) {
      host.querySelector('[data-action="inspect"]')?.classList.add("active");
    }
    // Tokens panel toggle
    host.querySelector('[data-action="tokens"]').addEventListener("click", () => {
      toggleTokens(host);
    });
  }

  // --- Design tokens panel ---------------------------------------------
  // Pulls any --var: value declaration from the active version's HTML,
  // lets the user tweak them with live swatches/sliders, and posts a
  // setProperty patch to the iframe. Changes are zero-model-call; when
  // the user hits "Bake into prompt", the overrides become a system
  // nudge on the next generation so they persist.

  const COLOR_RE = /(#[0-9a-fA-F]{3,8}\b|rgb[a]?\([^)]+\)|hsl[a]?\([^)]+\))/;
  const NUMERIC_RE = /(-?[0-9]*\.?[0-9]+)(px|rem|em|%)?\s*$/;

  function extractTokens(html) {
    if (!html) return [];
    // Match `--name: value;` declarations in style blocks or style attrs.
    const re = /--([a-zA-Z0-9_-]+)\s*:\s*([^;}\n]+?)\s*[;}]/g;
    const out = new Map();
    let m;
    while ((m = re.exec(html)) !== null) {
      const name = "--" + m[1];
      const value = m[2].trim();
      // De-dup: keep first (order matters in cascades, first usually wins at :root)
      if (!out.has(name)) out.set(name, value);
    }
    return Array.from(out, ([name, value]) => ({ name, value }));
  }

  function tokenKind(value) {
    if (COLOR_RE.test(value)) return "color";
    if (NUMERIC_RE.test(value)) return "numeric";
    return "text";
  }

  function toggleTokens(host) {
    const existing = host.querySelector(".design-tokens-panel");
    if (existing) { existing.remove(); return; }
    const v = state.versions[state.activeVersion];
    if (!v) return;
    const tokens = extractTokens(v.html);
    const panel = document.createElement("aside");
    panel.className = "design-tokens-panel";
    if (!tokens.length) {
      panel.innerHTML = `
        <header><strong>Design tokens</strong><button data-action="close" aria-label="Close">×</button></header>
        <div class="design-tokens-empty">
          <p>No CSS custom properties (<code>--name: value</code>) found in this design.</p>
          <p class="muted">Ask the model to "define your colors as CSS variables" to unlock live tweaking.</p>
        </div>
      `;
    } else {
      panel.innerHTML = `
        <header>
          <strong>Design tokens</strong>
          <span class="muted" style="font-size:11px">${tokens.length} found</span>
          <div style="flex:1"></div>
          <button data-action="close" aria-label="Close">×</button>
        </header>
        <div class="design-tokens-list" id="design-tokens-list"></div>
        <footer>
          <button data-action="reset">Reset</button>
          <button data-action="bake" title="Include these overrides in the next Generate so they persist">Bake into prompt</button>
        </footer>
      `;
    }
    host.querySelector(".design-right").appendChild(panel);
    panel.querySelector('[data-action="close"]').addEventListener("click", () => panel.remove());
    if (!tokens.length) return;

    const list = panel.querySelector("#design-tokens-list");
    state._tokenOverrides = state._tokenOverrides || {};
    for (const t of tokens) {
      list.appendChild(renderTokenRow(host, t));
    }
    panel.querySelector('[data-action="reset"]').addEventListener("click", () => {
      state._tokenOverrides = {};
      saveSession();
      // Clear any applied style in the iframe by re-rendering
      showVersion(host, state.versions[state.activeVersion]);
      toggleTokens(host); toggleTokens(host); // re-open fresh
    });
    panel.querySelector('[data-action="bake"]').addEventListener("click", () => {
      const overrides = state._tokenOverrides || {};
      if (!Object.keys(overrides).length) return;
      const input = host.querySelector("#design-input");
      const list = Object.entries(overrides).map(([k, v]) => `  ${k}: ${v}`).join("\n");
      input.value = `Keep the current design but use these token values at :root:\n${list}\n\n`;
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
      onPromptInput(host);
    });
  }

  function renderTokenRow(host, token) {
    const row = document.createElement("div");
    row.className = "design-token-row";
    const kind = tokenKind(token.value);
    const current = (state._tokenOverrides?.[token.name]) ?? token.value;
    let control = "";
    if (kind === "color") {
      const hex = toHex(current) || "#888888";
      control = `<input type="color" value="${hex}" data-kind="color">`;
    } else if (kind === "numeric") {
      const m = current.match(NUMERIC_RE);
      const num = m ? parseFloat(m[1]) : 0;
      const unit = (m && m[2]) || "px";
      control = `<input type="number" step="0.25" value="${num}" data-kind="numeric" data-unit="${unit}">`;
    } else {
      control = `<input type="text" value="${escapeAttr(current)}" data-kind="text">`;
    }
    row.innerHTML = `
      <code class="design-token-name" title="${escapeAttr(token.name)}">${escapeHtml(token.name)}</code>
      ${control}
      <span class="design-token-value" aria-hidden="true">${escapeHtml(current)}</span>
    `;
    const input = row.querySelector("input");
    input.addEventListener("input", () => {
      const kind = input.dataset.kind;
      let newVal;
      if (kind === "numeric") newVal = input.value + (input.dataset.unit || "");
      else newVal = input.value;
      state._tokenOverrides[token.name] = newVal;
      row.querySelector(".design-token-value").textContent = newVal;
      applyTokenOverride(host, token.name, newVal);
      saveSession();
    });
    return row;
  }

  function applyTokenOverride(host, name, value) {
    // postMessage to the iframe, which runs a tiny setProperty script.
    const iframe = host.querySelector(".design-iframe");
    if (!iframe || !iframe.contentWindow) return;
    iframe.contentWindow.postMessage(
      { __mioDesignTokenSet: true, name, value },
      "*",
    );
  }

  function toHex(color) {
    // Accepts #rgb / #rrggbb / rgb() / rgba(). Returns #rrggbb.
    if (!color) return null;
    const s = color.trim();
    if (/^#([0-9a-f]{6})$/i.test(s)) return s;
    if (/^#([0-9a-f]{3})$/i.test(s)) {
      return "#" + s[1] + s[1] + s[2] + s[2] + s[3] + s[3];
    }
    const m = s.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
    if (m) {
      const h = (n) => Number(n).toString(16).padStart(2, "0");
      return "#" + h(m[1]) + h(m[2]) + h(m[3]);
    }
    return null;
  }

  async function generate(host) {
    const input = host.querySelector("#design-input");
    const prompt = input.value.trim();
    if (!prompt) return;
    const variants = host.querySelector("#design-variants").checked ? 3 : 1;

    state.history.push({ role: "user", text: prompt });
    saveSession();
    renderHistory(host);
    input.value = "";
    const genBtn = host.querySelector('[data-action="generate"]');
    genBtn.disabled = true; genBtn.textContent = variants > 1 ? "Generating 3…" : "Generating…";

    try {
      // Use the existing OpenAI-compatible endpoint. Model pick-up:
      // whatever the server has loaded as default (mio-large-moe).
      const messages = [
        { role: "system", content: SYSTEM_PROMPT },
        ...state.history
          .filter((m) => m.role === "user" || m.role === "assistant")
          .slice(-10)
          .map((m) => ({ role: m.role, content: m.text || "" })),
      ];
      // Replace the latest user message (already pushed above) with the
      // current prompt if it's not already the tail. When the user has
      // pasted one or more reference images, use the multimodal
      // content-list format so the VL model sees them alongside text.
      if (messages[messages.length - 1].role !== "user") {
        messages.push({ role: "user", content: prompt });
      }
      if (state._refs && state._refs.length) {
        // Re-pack the last user message in multimodal form.
        const textContent = typeof messages[messages.length - 1].content === "string"
          ? messages[messages.length - 1].content
          : prompt;
        const parts = [{ type: "text", text: textContent + "\n\n(Use the reference images above as visual inspiration — do not try to pixel-match.)" }];
        for (const r of state._refs) {
          parts.push({ type: "image_url", image_url: { url: r.dataUrl } });
        }
        messages[messages.length - 1] = { role: "user", content: parts };
      }
      // If this generation is a fork from an existing version, hand the
      // model the full HTML of the source so it can iterate rather than
      // start from scratch.
      const activeV = state.versions[state.activeVersion];
      const seedHtml = state._forkSeed || (state._scope === "patch" && activeV ? activeV.html : null);
      if (seedHtml) {
        const scopeNote = state._scope === "patch"
          ? `SCOPE: PATCH. The user is making a targeted edit to the existing design. Produce the FULL updated HTML in the artifact, but change ONLY the parts implied by the request. Preserve layout, colors, fonts, and all unrelated elements exactly.`
          : `The user is iterating on an existing design. Build on it rather than starting from scratch.`;
        messages.unshift({
          role: "system",
          content: `${scopeNote}\n\nHere is the full HTML of the current design:\n\n\`\`\`html\n${seedHtml.slice(0, 40000)}\n\`\`\``,
        });
        state._forkSeed = null;
      }
      // Unlock auto-classifier for next turn once the user has sent one.
      state._scopeLocked = false;

      const runs = [];
      for (let i = 0; i < variants; i++) {
        const temp = variants === 1 ? 0.7 : 0.5 + (i * 0.25);
        runs.push(runOne(messages, temp));
      }
      const results = await Promise.all(runs);
      const variantGroup = variants > 1 ? Date.now() : null;
      const startIdx = state.versions.length;
      for (let i = 0; i < results.length; i++) {
        const text = results[i];
        const html = extractArtifact(text);
        const versionNum = state.versions.length + 1;
        state.versions.push({
          n:     versionNum,
          title: `v${versionNum}` + (variantGroup ? ` (variant ${String.fromCharCode(65 + i)})` : ""),
          html:  html || renderErrorHTML(text),
          prompt,
          ts:    Date.now(),
          variantGroup,
        });
      }
      state.activeVersion = state.versions.length - 1;
      // When we have variants, stage them for comparison on the next
      // showVersion — the user gets a 3-up grid until they pick one.
      state._pendingCompare = variantGroup;
      state.history.push({
        role: "assistant",
        text: variantGroup ? `Generated ${variants} variants — pick one to keep (the others stay in the scrubber).`
                           : `Generated v${state.versions.length}.`,
      });
      // Drop references — they were one-shot seeds for this turn.
      state._refs = [];
      renderReferences(host);
      saveSession();
      renderHistory(host);
      renderVersions(host);
    } catch (e) {
      state.history.push({ role: "assistant", text: "Error: " + e.message });
      saveSession();
      renderHistory(host);
    } finally {
      genBtn.disabled = false; genBtn.textContent = "Generate";
    }
  }

  async function runOne(messages, temperature) {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "mio-auto",
        messages,
        temperature,
        max_tokens: 4096,
        stream: false,
      }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    return data.choices?.[0]?.message?.content || "";
  }

  function extractArtifact(text) {
    // Grab the contents of the first <antArtifact …>…</antArtifact>.
    const m = text.match(/<antArtifact[^>]*>([\s\S]*?)<\/antArtifact>/);
    if (!m) return null;
    return m[1].trim();
  }

  function renderErrorHTML(rawReply) {
    return `<!doctype html><html><body style="margin:0;padding:24px;font-family:-apple-system,system-ui,sans-serif;color:#333;background:#fff"><h2 style="margin:0 0 10px">No &lt;antArtifact&gt; in the reply</h2><p style="color:#666;font-size:13px;margin:0 0 12px">The model didn't wrap its output in the expected tag. Raw reply:</p><pre style="background:#f5f5f5;padding:12px;border-radius:6px;font-size:12px;white-space:pre-wrap;overflow:auto;max-height:60vh">${escapeHtml(rawReply)}</pre></body></html>`;
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }
})();
