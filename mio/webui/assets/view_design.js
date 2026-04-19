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

  // --- Output kinds -----------------------------------------------------
  // A second axis alongside platform: what *kind* of artifact the model
  // should produce. Each kind has its own system-prompt addendum that
  // biases the generation toward a specific stack. `page` is the
  // default (existing behaviour).
  const KINDS = {
    page:   { label: "Page",      icon: "🖼",  addendum: "" },
    scene:  { label: "3D Scene",  icon: "🧊",  addendum: SCENE_ADDENDUM() },
    ar:     { label: "AR",        icon: "📦",  addendum: AR_ADDENDUM() },
    shader: { label: "Shader",    icon: "🌈",  addendum: SHADER_ADDENDUM() },
    game:   { label: "Game",      icon: "🎮",  addendum: GAME_ADDENDUM() },
    cad:    { label: "CAD",       icon: "📐",  addendum: CAD_ADDENDUM() },
    blender:{ label: "Blender",   icon: "🟠",  addendum: BLENDER_ADDENDUM() },
  };

  function SCENE_ADDENDUM() {
    return `\n\nOUTPUT KIND: 3D SCENE.
Use Three.js via UMD from https://cdn.jsdelivr.net/npm/three@0.162 plus OrbitControls
(https://cdn.jsdelivr.net/npm/three@0.162/examples/jsm/controls/OrbitControls.js via importmap). Ship a full-screen canvas that fills the viewport. Required:
- Real geometry (TorusKnot / IcosahedronGeometry / a glTF load from https://cdn.jsdelivr.net/npm/three@0.162/examples/models/gltf/DamagedHelmet/glTF/DamagedHelmet.gltf — or similar well-hosted asset)
- HDRI environment via RGBELoader + PMREMGenerator (Poly Haven 1K HDRI, e.g. https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/kloofendal_48d_partly_cloudy_puresky_1k.hdr)
- At least one PBR material (MeshStandardMaterial with metalness/roughness and the HDRI as envMap)
- Shadow-casting directional light + ambient + soft background
- OrbitControls enabled, damping on
- One subtle post effect (FilmPass or dust particles) — optional but welcome
- renderer.outputColorSpace = THREE.SRGBColorSpace; ACES tone mapping
NO rotating cube on grey. Produce a scene worth looking at.`;
  }

  function AR_ADDENDUM() {
    return `\n\nOUTPUT KIND: AR-READY MODEL VIEWER.
Use Google's <model-viewer> web component via
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>.
Single <model-viewer> element taking the full viewport, with these attributes:
  src="<glb url>"
  ios-src="<usdz url>"
  ar ar-modes="scene-viewer quick-look webxr"
  camera-controls
  shadow-intensity="1"
  auto-rotate
  environment-image="neutral"
Prefer assets from https://modelviewer.dev/shared-assets/models/ or the Khronos glTF sample models at https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/. If the user hasn't given a specific model, pick one relevant to the prompt (Horse, Astronaut, Duck, DamagedHelmet, RobotExpressive, SciFiHelmet). Add a translucent info pill at the top with the model's name + a "View in AR" prompt on mobile. Include a small loading poster image.`;
  }

  function SHADER_ADDENDUM() {
    return `\n\nOUTPUT KIND: SHADER ART.
Ship a single <canvas> filling the viewport + a fullscreen triangle vertex shader + a fragment shader that reads classic ShaderToy-style uniforms:
  iTime (seconds since start)
  iResolution (vec3)
  iMouse (vec4)
Write the fragment shader in GLSL ES 3.00 with \`out vec4 fragColor;\`. Use WebGL2. Include time-animated distance fields, fbm noise, domain warping, polar mappings — produce art, not a solid color. No external libraries. If the user mentions a ShaderToy ID (e.g. "ShaderToy XsXXDn") or pastes a \`mainImage(fragColor, fragCoord)\` function, wrap it in the WebGL2 boilerplate as-is — preserving their logic.`;
  }

  function BLENDER_ADDENDUM() {
    return `\n\nOUTPUT KIND: BLENDER (bpy).
Produce a short intro, then a SINGLE <antArtifact identifier="blender-v{N}" type="application/vnd.pimio.blender" title="Short title">python code</antArtifact>
containing ready-to-run bpy code for Blender 4.2+. The artifact panel will render a "▶ Send to Blender" button that POSTs the code to the user's running Blender via the blender-mcp addon (localhost:9876).

Conventions:
- Start with \`import bpy\` and, if needed, \`import bmesh, math, random\`.
- Clear the default cube only when the user asked to "start clean":
    for o in list(bpy.data.objects):
        if o.type == 'MESH' and o.name.startswith('Cube'):
            bpy.data.objects.remove(o, do_unlink=True)
- Use \`bpy.data.objects.remove()\` NOT the 4.x-removed context-override \`bpy.ops.object.delete(...)\`.
- Prefer additive construction: primitive_add → modifiers (Subdivision / Bevel / Array / Mirror) → materials (Principled BSDF with named nodes) → light / camera setup.
- For materials, use \`node_tree.nodes\` / \`node_tree.links\`, not the legacy \`mat.diffuse_color\` shortcut.
- At the end, print a one-line summary via \`print(...)\` so the user sees progress in stdout.
- NEVER call \`bpy.ops.wm.save_as_mainfile\` or anything that overwrites files unless the user explicitly asked.

If the user asks to SEE the result, recommend calling the \`blender_snapshot\` skill afterwards.`;
  }

  function CAD_ADDENDUM() {
    return `\n\nOUTPUT KIND: PARAMETRIC CAD.
Produce a single-file interactive CAD viewer using JSCAD (functional parametric modeling).
Import core modules via esm.sh:
  <script type="importmap">
  { "imports": {
      "@jscad/modeling": "https://esm.sh/@jscad/modeling@2",
      "@jscad/regl-renderer": "https://esm.sh/@jscad/regl-renderer@2"
  }}
  </script>
Layout:
  - Full-viewport <canvas> on the right for the regl-renderer viewport
    (orbit + pan + zoom). Dark canvas bg (#18181b), grid-on-floor, axes.
  - Left 280 px parameter panel with real <input> controls (range, number,
    checkbox, color) bound to the scene. Each slider re-runs the model
    and updates the viewport live. Use native <input> + a tiny evented
    pattern — no frameworks.
  - Toolbar on top: [Export STL] [Export 3MF] [Reset camera] [Wireframe].
Model building:
  - Write the CAD as one function \`build(params)\` returning a single
    geometry (primitives + booleans + transforms + extrudeLinear/Rotate).
  - Prefer additive composition (union / subtract / intersect) with
    clear named sub-parts.
  - Export via \`stlSerializer.serialize({ binary: true }, geom)\` from
    @jscad/io (also via esm.sh) and trigger a Blob download.
  - Parameters must include realistic defaults so the viewer shows a
    finished object on load, not an empty scene.
Prefer JSCAD over OpenSCAD-WASM unless the user explicitly asks for
.scad syntax — JSCAD is lighter, pure JS, and renders instantly.`;
  }

  function GAME_ADDENDUM() {
    return `\n\nOUTPUT KIND: PLAYABLE GAME PROTOTYPE.
Use kaboom.js from https://unpkg.com/kaboom@3000/dist/kaboom.js — smallest viable game engine. Or Phaser 3 for larger scopes. Ship a fullscreen canvas with:
- A game loop (real update/render split)
- Pointer-lock or WASD keyboard input (Pointer is better for touch)
- A real objective (collect / avoid / score) — not just "move around"
- Score / timer / restart UI overlay
- Forgiving physics (no frame-rate-dependent deltas)
- Post-game recap (final score + restart button)
Use kaboom's component idioms (pos, area, body, sprite) — don't hand-roll everything. For sprites use CC0 kit assets from Kay Lousberg / Quaternius when the model knows a relevant GLB; otherwise draw with kaboom primitives.`;
  }

  // --- Platform system prompts ----------------------------------------
  // Each platform has its own spec for what "good" looks like.
  // Prompts are deliberately specific (components, token names, font
  // stacks) so a generic LLM reliably produces platform-authentic output.

  const PLATFORMS = {
    web: {
      label: "Web",
      viewport: { w: null, h: null, scale: 1 }, // fills canvas
      frame: null,
      systemPrompt: `You are a senior web UI engineer. Output ONE <antArtifact type="text/html"> with a fully self-contained responsive HTML document using:
- Tailwind via <script src="https://cdn.tailwindcss.com"></script>
- React 18 + ReactDOM via unpkg
- Babel Standalone for JSX in <script type="text/babel">
Rules: modern restrained aesthetic, 2 type sizes, generous whitespace, one accent. Animations only where they aid meaning. No external build step. No explanations after the artifact.`,
    },
    ios: {
      label: "iOS",
      viewport: { w: 402, h: 874, scale: 1 }, // iPhone 16 Pro logical points
      frame: "iphone",
      systemPrompt: `You are a senior iOS engineer producing an iPhone-shaped web mock that LOOKS and FEELS like an iOS 18+ app built with SwiftUI (rendered in HTML for preview).

Required conventions:
- Viewport: 393×852 (iPhone 16 Pro). Use a full-bleed layout inside a <body> styled background: #000 outside safe areas.
- Safe areas: respect Dynamic Island top inset (~54px) and home-indicator bottom (~34px) via padding-top: env(safe-area-inset-top, 54px); padding-bottom: env(safe-area-inset-bottom, 34px).
- Typography: system UI stack — font-family: -apple-system, "SF Pro Text", "SF Pro Display", system-ui; text styles mirror iOS (Large Title 34/41, Title1 28, Headline 17/22 semibold, Body 17/22, Subheadline 15, Footnote 13, Caption 12).
- Controls: use iOS idioms — UINavigationBar-style large title that collapses on scroll; UITabBar at the bottom with 4-5 items + SF-Symbol-style icons; UISwitch / UISegmentedControl / UIStepper look; rounded sheets / modals with a top grabber.
- Colors: iOS system palette — blue #007AFF, green #34C759, red #FF3B30, orange #FF9500, gray fill #F2F2F7 (light) / #1C1C1E (dark).
- Gestures: swipe-to-go-back on nav stacks; pull-to-refresh.
- Use SF Symbol equivalents via inline SVG or unicode glyphs (⌂ ◎ ⚙ ⋯) — don't invent icon fonts.
- No Tailwind — use semantic CSS with custom properties so the Tokens panel surfaces them: --system-blue, --system-gray-6, --label, --secondary-label, --system-background, --secondary-system-background, --radius-card (12px), --radius-sheet (14px).

Output ONE <antArtifact type="text/html"> with <!doctype html> and viewport meta "width=device-width, initial-scale=1, viewport-fit=cover". No explanations after.`,
    },
    android: {
      label: "Android",
      viewport: { w: 448, h: 992, scale: 1 }, // Pixel 9 Pro logical dp
      frame: "pixel",
      systemPrompt: `You are a senior Android engineer producing a Pixel-shaped web mock that LOOKS and FEELS like a Material 3 Expressive app (rendered in HTML for preview).

Required conventions:
- Viewport: 412×915 (Pixel 9 Pro dp). Edge-to-edge; respect a 28px top status-bar area and a 48px bottom system-bar / NavigationBar area.
- Typography: Roboto Flex via <link href="https://fonts.googleapis.com/css2?family=Roboto+Flex:opsz,wght@8..144,400;8..144,500;8..144,700&display=swap" rel="stylesheet">. Text roles: Display Large 57/64, Headline Large 32/40, Title Large 22/28, Body Large 16/24, Body Medium 14/20, Label Large 14/20.
- Material You tokens as CSS vars (so Tokens panel surfaces them):
  --md-primary, --md-on-primary, --md-primary-container, --md-on-primary-container,
  --md-secondary, --md-tertiary,
  --md-surface, --md-surface-variant, --md-on-surface, --md-on-surface-variant,
  --md-outline, --md-outline-variant,
  --md-background, --md-error,
  --md-shape-sm (4px), --md-shape-md (12px), --md-shape-lg (16px), --md-shape-xl (28px).
- Light default palette: primary #6750A4, on-primary #FFFFFF, surface #FFFBFE, on-surface #1C1B1F, surface-variant #E7E0EC.
- Components: TopAppBar with centered or left-aligned title; BottomNavigationBar with 3-5 destinations (outlined icon idle, filled icon + label pill when active); extended FAB for primary action; Cards with elevated/outlined variants; rounded Chips; ripple-able Buttons (filled, tonal, outlined, text); BottomSheet with drag handle.
- Dynamic color: if a wallpaper color is mentioned, derive primary from it (60% lightness target).
- Use Material Symbols via <span class="material-symbols-outlined"> via the stylesheet <link href="https://fonts.googleapis.com/icon?family=Material+Symbols+Outlined" rel="stylesheet">.
- No Tailwind.

Output ONE <antArtifact type="text/html"> with <!doctype html> and viewport meta "width=device-width, initial-scale=1, viewport-fit=cover". No explanations after.`,
    },
    ipad: {
      label: "iPad",
      viewport: { w: 1024, h: 768, scale: 0.9 }, // iPad Pro 11" landscape
      frame: "ipad",
      systemPrompt: `You are a senior iOS engineer producing an iPad-shaped web mock that LOOKS and FEELS like an iPadOS 18+ app.

Required conventions:
- Viewport: 1024×768 (iPad Pro 11" landscape).
- Use a split-view layout (sidebar + detail) typical of iPadOS master-detail apps. Sidebar 320 px with groupings; detail scrolls.
- Respect the same iOS type scale, colors, and control idioms as the iOS prompt (shared CSS custom properties).
- Font stack: -apple-system, "SF Pro Text", "SF Pro Display", system-ui.
- Toolbar above the detail with large title + trailing toolbar buttons (gear, share, compose).

Output ONE <antArtifact type="text/html"> with <!doctype html> and viewport meta. No explanations after.`,
    },
  };

  const SYSTEM_PROMPT = PLATFORMS.web.systemPrompt; // legacy alias

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

  // --- Shortcuts: only active while Design view is mounted ----------
  function bindDesignShortcuts(host) {
    if (host._designKb) return;
    host._designKb = (e) => {
      // Never hijack typing
      const t = e.target;
      const inText = t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT" || t.isContentEditable);
      if (inText && e.key !== "Enter") return; // Enter handled elsewhere
      if (!document.querySelector(".view-design")) return;
      if (!(e.metaKey || e.ctrlKey)) return;
      if (e.shiftKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "enter") { e.preventDefault(); generate(host); return; }
      if (k === "i") { e.preventDefault(); toggleInspect(host, !state._inspect); return; }
      if (k === "e") { e.preventDefault(); toggleEdit(host, !state._edit);       return; }
      if (k === "t") { e.preventDefault(); toggleTokens(host);                    return; }
      if (k === "r") {
        // Only if user has a last prompt in history
        const lastUser = [...state.history].reverse().find((m) => m.role === "user");
        if (lastUser) {
          e.preventDefault();
          const input = host.querySelector("#design-input");
          if (input) { input.value = lastUser.text; generate(host); }
        }
        return;
      }
    };
    window.addEventListener("keydown", host._designKb);
  }

  // --- Onboarding coachmarks (first open) ---------------------------
  const ONBOARDING_KEY = "mio.design.onboarded.v1";
  function maybeShowCoachmarks(host) {
    try { if (localStorage.getItem(ONBOARDING_KEY)) return; } catch {}
    const ov = document.createElement("div");
    ov.className = "design-coach";
    ov.innerHTML = `
      <div class="design-coach-card">
        <h2>Design Mode</h2>
        <p>A focused canvas for iterating on UIs, 3D scenes, AR models, shaders, games, parametric CAD, and Blender scripts — all in one place.</p>
        <ul>
          <li><b>Platform row</b> → picks your target shell (Web / iOS / Android / iPad) and wraps the preview in a real device frame.</li>
          <li><b>Kind row</b> → sets what the model emits: Page · 3D · AR · Shader · Game · CAD · Blender.</li>
          <li><b>Vibe chips</b> → quick style tokens that prepend into your prompt.</li>
          <li><b>Shortcuts</b>: ⌘⏎ Generate · ⌘I Inspect · ⌘E Edit · ⌘T Tokens · ⌘R Re-run last.</li>
          <li><b>📎 Paste a screenshot</b> in the composer to use it as visual reference (VL input).</li>
          <li><b>.zip</b> exports every version + README + prompt log.</li>
        </ul>
        <div class="design-coach-actions">
          <button data-act="tour-skip">Skip</button>
          <button data-act="tour-ok" class="primary">Got it</button>
        </div>
      </div>
    `;
    host.appendChild(ov);
    const dismiss = () => {
      try { localStorage.setItem(ONBOARDING_KEY, "1"); } catch {}
      ov.remove();
    };
    ov.querySelector('[data-act="tour-skip"]').addEventListener("click", dismiss);
    ov.querySelector('[data-act="tour-ok"]').addEventListener("click", dismiss);
  }

  function renderRoot(host) {
    host.innerHTML = `
      <div class="view-design">
        <aside class="design-left">
          <header class="design-left-head">
            <h1>Design Mode</h1>
            <button class="btn-ghost" data-action="reset">New session</button>
          </header>
          <div class="design-platforms" role="tablist" aria-label="Target platform">
            <button class="design-platform" data-platform="web"     title="Web">Web</button>
            <button class="design-platform" data-platform="ios"     title="iOS · HIG · iPhone 16 Pro">iOS</button>
            <button class="design-platform" data-platform="android" title="Android · Material 3 · Pixel 9 Pro">Android</button>
            <button class="design-platform" data-platform="ipad"    title="iPad Pro · HIG · landscape">iPad</button>
          </div>
          <div class="design-kinds" role="tablist" aria-label="Output kind">
            <button class="design-kind" data-kind="page"   title="Regular page / component">🖼 Page</button>
            <button class="design-kind" data-kind="scene"  title="Three.js 3D scene with HDRI + PBR">🧊 3D</button>
            <button class="design-kind" data-kind="ar"     title="&lt;model-viewer&gt; — iOS Quick Look + Android Scene Viewer">📦 AR</button>
            <button class="design-kind" data-kind="shader" title="Full-screen ShaderToy-style fragment shader">🌈 Shader</button>
            <button class="design-kind" data-kind="game"   title="Playable kaboom.js game prototype">🎮 Game</button>
            <button class="design-kind" data-kind="cad"    title="Parametric CAD (JSCAD) with sliders + STL export">📐 CAD</button>
            <button class="design-kind" data-kind="blender" title="Blender bpy code — runs in your open Blender via the blender-mcp addon">🟠 Blender</button>
          </div>
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
              <button class="btn-ghost" data-action="research" title="Do a web search for inspiration, then generate">🔎 Research + Generate</button>
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
            <button class="btn-ghost" data-action="inspect" title="Click an element → draft a prompt to change it">Inspect</button>
            <button class="btn-ghost" data-action="edit" title="Click any element and edit its text + styles directly (no model call)">Edit</button>
            <button class="btn-ghost" data-action="tokens" title="Tweak colors, radii, fonts live (no model call)">Tokens</button>
            <button class="btn-ghost" data-action="console" title="Show console + network activity from the preview"><span id="design-console-badge" hidden></span>Console</button>
            <button class="btn-ghost" data-action="fork" title="Fork a variant from this version">Fork</button>
            <button class="btn-ghost" data-action="copy">Copy HTML</button>
            <button class="btn-ghost" data-action="download" title="Download active version as index.html">HTML</button>
            <button class="btn-ghost" data-action="export-zip" title="Download full session: every version, README, prompt history">.zip</button>
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
    bindDesignShortcuts(host);
    maybeShowCoachmarks(host);
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
        <div class="design-msg-text">${renderMsgMd(m.text || "")}</div>
      </div>
    `).join("");
    wrap.scrollTop = wrap.scrollHeight;
    // Let Prism colourise any fenced code blocks.
    try { window.Prism?.highlightAllUnder?.(wrap); } catch {}
  }

  function renderMsgMd(text) {
    // Render the chat-side message as markdown so fenced blocks
    // (```html…```) turn into real code boxes instead of a wall of
    // escaped text. Falls back to escaped text if marked.js isn't
    // loaded yet.
    if (!window.marked?.parse) return escapeHtml(text);
    try {
      // Force escape of raw HTML so the artifact body doesn't render
      // as live DOM inside the sidebar.
      return window.marked.parse(text, { breaks: true, gfm: true, mangle: false, headerIds: false });
    } catch {
      return escapeHtml(text);
    }
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
      // Detect language from the body so Prism can colour it.
      const text = v.html || "";
      const lang = /^\s*<!doctype|<html/i.test(text) ? "markup"
                 : /^[\s\S]*?(import\s+\w|const\s+\w|function\s+\w)/.test(text) ? "javascript"
                 : /^[\s\S]*?(import\s+bpy|def\s+\w)/.test(text) ? "python"
                 : "markup";
      canvas.innerHTML = `<pre class="design-code language-${lang}"><code class="language-${lang}">${escapeHtml(text)}</code></pre>`;
      // Re-run Prism if it loaded with the main chat surface.
      try { window.Prism?.highlightAllUnder?.(canvas); } catch {}
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
      // Preview frame — wrapped in a sizer (and optionally a device
      // frame) so we can constrain the iframe to real device widths or
      // chrome when a platform is chosen.
      const platform = state._platform || "web";
      const pInfo = PLATFORMS[platform] || PLATFORMS.web;

      const sizer = document.createElement("div");
      sizer.className = "design-sizer";
      if (pInfo.viewport.w) {
        sizer.style.width  = pInfo.viewport.w + "px";
        sizer.style.height = pInfo.viewport.h + "px";
        sizer.style.flex   = "0 0 auto";
      }
      canvas.appendChild(sizer);

      const errorBar = document.createElement("div");
      errorBar.className = "design-error-bar";
      errorBar.hidden = true;
      canvas.appendChild(errorBar);

      const iframe = document.createElement("iframe");
      iframe.className = "design-iframe";
      iframe.sandbox = "allow-scripts";
      iframe.srcdoc = injectRuntimeHelpers(v.html || "", { inspect: !!state._inspect, edit: !!state._edit });

      // Wrap with a device frame when platform != web
      if (pInfo.frame) {
        const frame = buildDeviceFrame(pInfo.frame, iframe);
        sizer.appendChild(frame);
      } else {
        sizer.appendChild(iframe);
      }

      if (!pInfo.viewport.w) applyWidth(sizer, state._width || "fit");

      // Messages from the iframe — errors + element clicks in inspect/edit mode.
      const handler = (e) => {
        if (!e.data || e.source !== iframe.contentWindow) return;
        if (e.data.__mioDesignError === true) {
          showError(host, errorBar, e.data);
        } else if (e.data.__mioDesignPick === true) {
          onElementPicked(host, e.data);
        } else if (e.data.__mioDesignEditPick === true) {
          onEditPicked(host, iframe, e.data);
        } else if (e.data.__mioDesignEditText === true) {
          onEditText(host, e.data);
        } else if (e.data.__mioDesignConsole === true) {
          onConsoleEvent(host, { kind: "log", ...e.data });
        } else if (e.data.__mioDesignNet === true) {
          onConsoleEvent(host, { kind: "net", ...e.data });
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

  // --- Device frames (iPhone / Pixel / iPad) ----------------------------
  // SVG + CSS chrome around the iframe. Content bleeds through the
  // cutout (notch / Dynamic Island / punch-hole) — the chrome sits
  // above. A live status bar shows time and a battery glyph.

  function buildDeviceFrame(kind, iframe) {
    const wrap = document.createElement("div");
    wrap.className = "device-frame device-frame-" + kind;
    let body = "";
    if (kind === "iphone") {
      body = `
        <div class="device-shell">
          <div class="device-screen"></div>
          <div class="device-island"></div>
          <div class="device-statusbar">
            <span class="device-time"></span>
            <span class="device-status-right">
              <span class="device-signal">●●●</span>
              <span class="device-wifi">⌇</span>
              <span class="device-batt"><span class="device-batt-fill"></span></span>
            </span>
          </div>
          <div class="device-home-indicator"></div>
        </div>
      `;
    } else if (kind === "pixel") {
      body = `
        <div class="device-shell">
          <div class="device-screen"></div>
          <div class="device-punch"></div>
          <div class="device-statusbar">
            <span class="device-time"></span>
            <span class="device-status-right">
              <span class="device-wifi">◢</span>
              <span class="device-signal">▤</span>
              <span class="device-batt"><span class="device-batt-fill"></span></span>
            </span>
          </div>
          <div class="device-gesture-bar"></div>
        </div>
      `;
    } else if (kind === "ipad") {
      body = `
        <div class="device-shell device-shell-ipad">
          <div class="device-screen"></div>
          <div class="device-home-indicator"></div>
        </div>
      `;
    }
    wrap.innerHTML = body;
    const screen = wrap.querySelector(".device-screen");
    if (screen) screen.appendChild(iframe);
    // Live time + battery
    refreshDeviceStatus(wrap);
    const intervalId = setInterval(() => refreshDeviceStatus(wrap), 30_000);
    // Clean up when frame is removed
    const observer = new MutationObserver(() => {
      if (!wrap.isConnected) {
        clearInterval(intervalId);
        observer.disconnect();
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    return wrap;
  }

  function refreshDeviceStatus(wrap) {
    const t = wrap.querySelector(".device-time");
    if (t) {
      const d = new Date();
      t.textContent = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    const fill = wrap.querySelector(".device-batt-fill");
    if (fill && navigator.getBattery) {
      navigator.getBattery().then((b) => {
        fill.style.width = Math.max(8, Math.round(b.level * 100)) + "%";
        fill.style.background = b.level < 0.2 ? "#FF3B30" : "#fff";
      }).catch(() => {
        fill.style.width = "72%";
      });
    } else if (fill) {
      fill.style.width = "72%";
    }
  }

  // --- Console drawer --------------------------------------------------
  // A bottom drawer that accumulates log + net events from the iframe.
  // State is in-memory per Design-Mode session (not persisted — logs
  // are ephemeral debug info).

  state._consoleEvents = state._consoleEvents || [];
  const MAX_CONSOLE_EVENTS = 200;

  function onConsoleEvent(host, evt) {
    state._consoleEvents.push(evt);
    if (state._consoleEvents.length > MAX_CONSOLE_EVENTS) {
      state._consoleEvents.splice(0, state._consoleEvents.length - MAX_CONSOLE_EVENTS);
    }
    // If drawer open, append live
    const log = host.querySelector(".design-console-log");
    if (log) appendConsoleLine(log, evt);
    // Badge on the button
    const badge = host.querySelector("#design-console-badge");
    if (badge && !host.querySelector(".design-console")) {
      const count = state._consoleEvents.length;
      badge.hidden = false;
      badge.textContent = count > 99 ? "99+" : String(count);
    }
  }

  function toggleConsole(host) {
    const existing = host.querySelector(".design-console");
    if (existing) { existing.remove(); return; }
    const panel = document.createElement("div");
    panel.className = "design-console";
    panel.innerHTML = `
      <header>
        <strong>Console</strong>
        <span class="muted">${state._consoleEvents.length} events</span>
        <div style="flex:1"></div>
        <button data-action="clear-console">Clear</button>
        <button data-action="send-console" title="Insert a summary into the composer">Send to chat</button>
        <button data-action="close-console" aria-label="Close">×</button>
      </header>
      <div class="design-console-log"></div>
    `;
    host.querySelector(".design-right").appendChild(panel);
    const log = panel.querySelector(".design-console-log");
    for (const evt of state._consoleEvents) appendConsoleLine(log, evt);
    panel.querySelector('[data-action="close-console"]').addEventListener("click", () => panel.remove());
    panel.querySelector('[data-action="clear-console"]').addEventListener("click", () => {
      state._consoleEvents = [];
      log.innerHTML = "";
      const badge = host.querySelector("#design-console-badge");
      if (badge) { badge.hidden = true; badge.textContent = ""; }
    });
    panel.querySelector('[data-action="send-console"]').addEventListener("click", () => {
      const input = host.querySelector("#design-input");
      const summary = state._consoleEvents.slice(-20).map((e) => {
        if (e.kind === "net") return `[${e.method} ${e.status || "ERR"}] ${e.url} ${e.ms}ms`;
        return `[${e.level}] ${e.message}`;
      }).join("\n");
      input.value = `Here's the recent console activity from the preview — investigate + fix anything off:\n\n${summary}\n\n`;
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
      const v = state.versions[state.activeVersion];
      if (v) state._forkSeed = v.html;
    });
    // Hide badge while open
    const badge = host.querySelector("#design-console-badge");
    if (badge) { badge.hidden = true; badge.textContent = ""; }
  }

  function appendConsoleLine(log, evt) {
    const ln = document.createElement("div");
    ln.className = "design-console-line " + (evt.kind || "");
    if (evt.kind === "net") {
      const status = evt.status || "ERR";
      ln.classList.add("net-" + (status >= 400 || status === 0 ? "err" : "ok"));
      ln.innerHTML = `
        <span class="c-tag">${escapeHtml(evt.method || "")}</span>
        <span class="c-status">${status}</span>
        <span class="c-msg" title="${escapeAttr(evt.url || "")}">${escapeHtml(evt.url || "")}</span>
        <span class="c-ms">${evt.ms ?? ""}ms</span>
      `;
    } else {
      ln.classList.add("level-" + (evt.level || "log"));
      ln.innerHTML = `
        <span class="c-tag">${escapeHtml(evt.level || "log")}</span>
        <span class="c-msg">${escapeHtml(evt.message || "")}</span>
      `;
    }
    log.appendChild(ln);
    log.scrollTop = log.scrollHeight;
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
  function injectRuntimeHelpers(html, { inspect = false, edit = false } = {}) {
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
  // Mirror console.* and failed fetches back to the parent for the
  // Design Mode console drawer.
  function mioLog(level, args){
    try {
      var msg = args.map(function(a){ try { return typeof a === 'string' ? a : JSON.stringify(a); } catch(_) { return String(a); } }).join(' ');
      parent.postMessage({__mioDesignConsole:true, level:level, message:msg.slice(0, 800), ts: Date.now()}, '*');
    } catch(_){}
  }
  ['log','info','warn','error','debug'].forEach(function(level){
    var orig = console[level].bind(console);
    console[level] = function(){ mioLog(level, [].slice.call(arguments)); orig.apply(console, arguments); };
  });
  // Network capture (fetch only; XHR skipped for brevity)
  var origFetch = window.fetch && window.fetch.bind(window);
  if (origFetch) {
    window.fetch = function(input, init){
      var url = typeof input === 'string' ? input : (input && input.url) || '';
      var method = (init && init.method) || (input && input.method) || 'GET';
      var t0 = performance.now();
      return origFetch(input, init).then(function(r){
        try { parent.postMessage({__mioDesignNet:true, url:url, method:method, status:r.status, ms: Math.round(performance.now()-t0)}, '*'); } catch(_){}
        return r;
      }).catch(function(err){
        try { parent.postMessage({__mioDesignNet:true, url:url, method:method, status:0, ms: Math.round(performance.now()-t0), error: String(err)}, '*'); } catch(_){}
        throw err;
      });
    };
  }
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
    const editScript = !edit ? "" : `
<style>
  html.__mio_edit, html.__mio_edit * { cursor: pointer !important; }
  html.__mio_edit *:hover { outline: 1px dashed #34C759 !important; outline-offset: 1px; }
  .__mio_edit_target {
    outline: 2px solid #34C759 !important;
    outline-offset: 2px;
    box-shadow: 0 0 0 6px rgba(52, 199, 89, 0.20) !important;
  }
  .__mio_edit_target[contenteditable="true"] { cursor: text !important; }
</style>
<script>
(function(){
  document.documentElement.classList.add('__mio_edit');
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
  let current = null;
  function unselect() {
    if (current) { current.classList.remove('__mio_edit_target'); current.removeAttribute('contenteditable'); current = null; }
  }
  document.addEventListener('click', function(e){
    if (e.target.closest('.__mio_edit_toolbar')) return; // allow toolbar clicks
    const t = e.target;
    if (!(t instanceof Element)) return;
    e.preventDefault(); e.stopPropagation();
    unselect();
    current = t;
    t.classList.add('__mio_edit_target');
    t.setAttribute('contenteditable', 'true');
    const rect = t.getBoundingClientRect();
    const styles = getComputedStyle(t);
    try {
      parent.postMessage({
        __mioDesignEditPick: true,
        selector: cssPath(t),
        tag: t.tagName.toLowerCase(),
        rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height },
        styles: {
          color:           styles.color,
          backgroundColor: styles.backgroundColor,
          fontSize:        styles.fontSize,
          fontWeight:      styles.fontWeight,
          fontStyle:       styles.fontStyle,
          textDecoration:  styles.textDecorationLine || styles.textDecoration,
        },
        text: (t.innerText || '').slice(0, 500),
      }, '*');
    } catch(_) {}
  }, true);
  // Forward text edits on blur so parent can bake innerHTML back into source.
  document.addEventListener('blur', function(e){
    const t = e.target;
    if (!(t instanceof Element) || !t.classList.contains('__mio_edit_target')) return;
    try {
      parent.postMessage({
        __mioDesignEditText: true,
        selector: cssPath(t),
        newText: t.innerText || '',
      }, '*');
    } catch(_) {}
  }, true);
  // Parent → iframe commands: apply live style changes
  window.addEventListener('message', (e) => {
    if (!e.data) return;
    if (e.data.__mioDesignStyleSet === true && current) {
      for (const [k, v] of Object.entries(e.data.styles || {})) {
        current.style[k] = v;
      }
    }
    if (e.data.__mioDesignUnselect === true) unselect();
  });
})();
</script>`;
    const head = errorScript + inspectScript + editScript;
    if (/<\/head>/i.test(html)) return html.replace(/<\/head>/i, head + "</head>");
    if (/<body/i.test(html))    return html.replace(/<body([^>]*)>/i, "<body$1>" + head);
    return head + html;
  }

  function onElementPicked(host, pick) {
    // Ask: edit-in-place OR extract-as-component?
    const menu = document.createElement("div");
    menu.className = "design-pick-menu";
    menu.innerHTML = `
      <div class="design-pick-hdr">${escapeHtml(pick.tag + (pick.textHint ? " · " + pick.textHint : ""))}</div>
      <button data-act="edit"   title="Edit this specific element">Change this specific element</button>
      <button data-act="extract" title="Save to the per-session component shelf for reuse">Extract as component</button>
      <button data-act="regen"  title="Regenerate just this element">Regenerate just this</button>
      <button data-act="cancel">Cancel</button>
    `;
    document.body.appendChild(menu);
    // Position near the clicked element
    const canvas = host.querySelector("#design-canvas");
    const rect = canvas?.getBoundingClientRect();
    if (rect && pick.rect) {
      menu.style.left = (rect.left + pick.rect.x) + "px";
      menu.style.top  = (rect.top + pick.rect.y + pick.rect.h + 6) + "px";
    }
    const close = () => menu.remove();
    menu.querySelector('[data-act="cancel"]').addEventListener("click", close);
    menu.querySelector('[data-act="edit"]').addEventListener("click", () => {
      const input = host.querySelector("#design-input");
      const hint = pick.textHint ? ` "${pick.textHint}"` : "";
      input.value = `Change this specific element (${pick.tag}${hint}) at \`${pick.selector}\` so it `;
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
      const v = state.versions[state.activeVersion];
      if (v) state._forkSeed = v.html;
      toggleInspect(host, false);
      close();
    });
    menu.querySelector('[data-act="extract"]').addEventListener("click", () => {
      saveComponent(host, pick);
      toggleInspect(host, false);
      close();
    });
    menu.querySelector('[data-act="regen"]').addEventListener("click", () => {
      const input = host.querySelector("#design-input");
      const hint = pick.textHint ? ` "${pick.textHint}"` : "";
      input.value = `Regenerate ONLY the ${pick.tag} element${hint} (at \`${pick.selector}\`). Keep everything else exactly as it is.`;
      input.focus();
      const v = state.versions[state.activeVersion];
      if (v) state._forkSeed = v.html;
      toggleInspect(host, false);
      close();
    });
    // Auto-dismiss on outside click
    setTimeout(() => {
      const onClick = (e) => {
        if (!menu.contains(e.target)) { close(); document.removeEventListener("click", onClick); }
      };
      document.addEventListener("click", onClick);
    }, 50);
  }

  // --- Component shelf: reusable snippets per session --------------
  function saveComponent(host, pick) {
    state._components = state._components || [];
    state._components.push({
      id: "c" + Date.now(),
      tag: pick.tag,
      selector: pick.selector,
      textHint: pick.textHint,
      outer: pick.outer,
      ts: Date.now(),
    });
    saveSession();
    // Tiny toast
    const tip = document.createElement("div");
    tip.className = "design-extract-toast";
    tip.textContent = `Extracted ${pick.tag}${pick.textHint ? " · " + pick.textHint : ""} — reusable via /components in the composer`;
    document.body.appendChild(tip);
    setTimeout(() => tip.remove(), 2200);
  }

  function toggleInspect(host, on) {
    state._inspect = !!on;
    if (on) state._edit = false;
    saveSession();
    const btn = host.querySelector('[data-action="inspect"]');
    if (btn) btn.classList.toggle("active", !!on);
    const editBtn = host.querySelector('[data-action="edit"]');
    if (editBtn) editBtn.classList.remove("active");
    hideEditToolbar(host);
    // Re-render so the iframe gets the helper script refreshed.
    const v = state.versions[state.activeVersion];
    if (v) showVersion(host, v);
  }

  function toggleEdit(host, on) {
    state._edit = !!on;
    if (on) state._inspect = false;
    saveSession();
    host.querySelector('[data-action="edit"]')?.classList.toggle("active", !!on);
    host.querySelector('[data-action="inspect"]')?.classList.remove("active");
    hideEditToolbar(host);
    const v = state.versions[state.activeVersion];
    if (v) showVersion(host, v);
  }

  // --- Edit-mode state & handlers --------------------------------------
  // state._edits: { [selector]: { text?, styles: {prop:value} } }

  function onEditPicked(host, iframe, msg) {
    state._editCurrent = { iframe, selector: msg.selector };
    showEditToolbar(host, iframe, msg);
  }

  function onEditText(host, msg) {
    if (!msg.selector) return;
    state._edits = state._edits || {};
    state._edits[msg.selector] = state._edits[msg.selector] || { styles: {} };
    state._edits[msg.selector].text = msg.newText;
    markDirty(host);
  }

  function setEditStyle(host, prop, value) {
    const cur = state._editCurrent;
    if (!cur) return;
    state._edits = state._edits || {};
    state._edits[cur.selector] = state._edits[cur.selector] || { styles: {} };
    state._edits[cur.selector].styles[prop] = value;
    // Live-apply in iframe
    const payload = {};
    payload[prop] = value;
    try {
      cur.iframe.contentWindow?.postMessage({ __mioDesignStyleSet: true, styles: payload }, "*");
    } catch {}
    markDirty(host);
  }

  function markDirty(host) {
    const btn = host.querySelector(".design-edit-bake");
    if (!btn) return;
    const count = Object.keys(state._edits || {}).length;
    btn.hidden = count === 0;
    btn.textContent = count ? `Bake ${count} edit${count === 1 ? "" : "s"} → new version` : "";
  }

  function showEditToolbar(host, iframe, msg) {
    hideEditToolbar(host);
    const canvas = host.querySelector("#design-canvas");
    if (!canvas || !msg.rect) return;
    const iframeRect = iframe.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    // Position relative to canvas, above the clicked element.
    const x = iframeRect.left - canvasRect.left + msg.rect.x;
    const y = iframeRect.top  - canvasRect.top  + msg.rect.y;
    const bar = document.createElement("div");
    bar.className = "design-edit-toolbar";
    bar.style.left = Math.max(10, x) + "px";
    bar.style.top  = Math.max(10, y - 44) + "px";
    const s = msg.styles || {};
    const sizePx = parseFloat(s.fontSize) || 16;
    bar.innerHTML = `
      <label class="design-edit-slot" title="Text color">
        <span>A</span><input type="color" value="${toHex(s.color) || "#000000"}" data-prop="color">
      </label>
      <label class="design-edit-slot" title="Background">
        <span>▦</span><input type="color" value="${toHex(s.backgroundColor) || "#ffffff"}" data-prop="backgroundColor">
      </label>
      <span class="design-edit-sep"></span>
      <div class="design-edit-size">
        <button data-delta="-2" aria-label="Smaller">A-</button>
        <input type="number" min="8" max="120" step="1" value="${Math.round(sizePx)}" data-prop="fontSize-px">
        <button data-delta="2" aria-label="Bigger">A+</button>
      </div>
      <span class="design-edit-sep"></span>
      <button class="design-edit-b" data-tog="bold"   title="Bold">B</button>
      <button class="design-edit-i" data-tog="italic" title="Italic">I</button>
      <button class="design-edit-u" data-tog="underline" title="Underline">U</button>
      <span class="design-edit-sep"></span>
      <button class="design-edit-close" aria-label="Done">Done</button>
    `;
    // Reflect current state on toggles
    if ((s.fontWeight || "") >= 600)            bar.querySelector(".design-edit-b").classList.add("on");
    if ((s.fontStyle || "") === "italic")       bar.querySelector(".design-edit-i").classList.add("on");
    if ((s.textDecoration || "").includes("underline")) bar.querySelector(".design-edit-u").classList.add("on");
    canvas.appendChild(bar);

    bar.querySelector('[data-prop="color"]').addEventListener("input", (e) =>
      setEditStyle(host, "color", e.target.value));
    bar.querySelector('[data-prop="backgroundColor"]').addEventListener("input", (e) =>
      setEditStyle(host, "backgroundColor", e.target.value));
    const sizeInput = bar.querySelector('[data-prop="fontSize-px"]');
    sizeInput.addEventListener("input", (e) =>
      setEditStyle(host, "fontSize", e.target.value + "px"));
    bar.querySelectorAll("[data-delta]").forEach((b) => {
      b.addEventListener("click", () => {
        const next = Math.max(8, Math.min(120, parseFloat(sizeInput.value) + parseFloat(b.dataset.delta)));
        sizeInput.value = next;
        setEditStyle(host, "fontSize", next + "px");
      });
    });
    bar.querySelectorAll("[data-tog]").forEach((b) => {
      b.addEventListener("click", () => {
        const prop = b.dataset.tog;
        const on = !b.classList.contains("on");
        b.classList.toggle("on", on);
        if (prop === "bold")      setEditStyle(host, "fontWeight", on ? "700" : "400");
        if (prop === "italic")    setEditStyle(host, "fontStyle",  on ? "italic" : "normal");
        if (prop === "underline") setEditStyle(host, "textDecoration", on ? "underline" : "none");
      });
    });
    bar.querySelector(".design-edit-close").addEventListener("click", () => {
      try { iframe.contentWindow?.postMessage({ __mioDesignUnselect: true }, "*"); } catch {}
      hideEditToolbar(host);
      state._editCurrent = null;
    });
  }

  function hideEditToolbar(host) {
    host.querySelector(".design-edit-toolbar")?.remove();
  }

  function bakeEditsIntoNewVersion(host) {
    const edits = state._edits || {};
    if (!Object.keys(edits).length) return;
    const v = state.versions[state.activeVersion];
    if (!v) return;
    let html = v.html || "";

    // 1) Append a style block at end of <head> with selector-scoped
    //    overrides for any style changes.
    const styleRules = [];
    for (const [selector, entry] of Object.entries(edits)) {
      if (!entry.styles) continue;
      const decls = Object.entries(entry.styles)
        .map(([p, val]) => `${p.replace(/([A-Z])/g, "-$1").toLowerCase()}: ${val} !important;`)
        .join(" ");
      if (decls) styleRules.push(`${selector} { ${decls} }`);
    }
    if (styleRules.length) {
      const block = `\n<style id="__mio-baked-edits">\n${styleRules.join("\n")}\n</style>`;
      if (/<\/head>/i.test(html)) html = html.replace(/<\/head>/i, block + "\n</head>");
      else html = block + html;
    }

    // 2) For text edits, append a small script that rewrites elements
    //    by selector at load. This survives React hydration since it
    //    runs after ReactDOM.render() in the <body> order we injected.
    const textEdits = [];
    for (const [selector, entry] of Object.entries(edits)) {
      if (typeof entry.text === "string" && entry.text) {
        textEdits.push([selector, entry.text]);
      }
    }
    if (textEdits.length) {
      const payload = JSON.stringify(textEdits);
      const script = `
<script>
(function(){
  function apply(){
    var edits = ${payload};
    for (var i = 0; i < edits.length; i++) {
      try {
        var el = document.querySelector(edits[i][0]);
        if (el) el.innerText = edits[i][1];
      } catch(_){}
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => setTimeout(apply, 50));
  else setTimeout(apply, 50);
})();
</script>
`;
      if (/<\/body>/i.test(html)) html = html.replace(/<\/body>/i, script + "</body>");
      else html = html + script;
    }

    // Push as a new version
    const n = state.versions.length + 1;
    state.versions.push({
      n,
      title: `v${n} (edits)`,
      html,
      prompt: `(local edits: ${Object.keys(edits).length} element${Object.keys(edits).length === 1 ? "" : "s"})`,
      ts: Date.now(),
    });
    state.activeVersion = state.versions.length - 1;
    state._edits = {};
    state._editCurrent = null;
    saveSession();
    hideEditToolbar(host);
    renderVersions(host);
    markDirty(host);
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
    host.querySelector('[data-action="research"]').addEventListener("click", () => generate(host, { research: true }));
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
    host.querySelector('[data-action="export-zip"]').addEventListener("click", async () => {
      if (!state.versions.length) return;
      const btn = host.querySelector('[data-action="export-zip"]');
      btn.disabled = true; btn.textContent = "Zipping…";
      try {
        const title = (state.versions[state.activeVersion]?.title || "mio-design")
          .replace(/^v\d+(\s*\(.*?\))?\s*-?\s*/i, "") || "mio-design";
        const res = await fetch("/ui/api/design/export", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title,
            platform: state._platform || "web",
            versions: state.versions,
            active:   state.activeVersion,
            history:  state.history,
          }),
        });
        if (!res.ok) throw new Error("HTTP " + res.status);
        const blob = await res.blob();
        const cd = res.headers.get("Content-Disposition") || "";
        const m = cd.match(/filename="([^"]+)"/);
        const filename = m ? m[1] : (title + ".zip");
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      } catch (e) {
        alert("Export failed: " + e.message);
      } finally {
        btn.disabled = false; btn.textContent = ".zip";
      }
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
    // Edit-mode toggle
    host.querySelector('[data-action="edit"]').addEventListener("click", () => {
      toggleEdit(host, !state._edit);
    });
    if (state._edit) {
      host.querySelector('[data-action="edit"]')?.classList.add("active");
    }
    // Tokens panel toggle
    host.querySelector('[data-action="tokens"]').addEventListener("click", () => {
      toggleTokens(host);
    });
    // Console drawer toggle
    host.querySelector('[data-action="console"]').addEventListener("click", () => {
      toggleConsole(host);
    });
    // Floating "Bake edits" bar — shown whenever there are pending
    // local edits. Positioned above the scrubber.
    if (!host.querySelector(".design-edit-bake")) {
      const bakeBtn = document.createElement("button");
      bakeBtn.className = "design-edit-bake";
      bakeBtn.hidden = true;
      bakeBtn.addEventListener("click", () => bakeEditsIntoNewVersion(host));
      host.querySelector(".design-right").appendChild(bakeBtn);
      markDirty(host);
    }
    // Kind picker
    const currentKind = state._kind || "page";
    host.querySelectorAll(".design-kind").forEach((b) => {
      b.classList.toggle("active", b.dataset.kind === currentKind);
      b.addEventListener("click", () => {
        state._kind = b.dataset.kind;
        saveSession();
        host.querySelectorAll(".design-kind").forEach((x) =>
          x.classList.toggle("active", x.dataset.kind === state._kind));
      });
    });
    // Platform picker
    const currentPlatform = state._platform || "web";
    host.querySelectorAll(".design-platform").forEach((b) => {
      b.classList.toggle("active", b.dataset.platform === currentPlatform);
      b.addEventListener("click", () => {
        state._platform = b.dataset.platform;
        saveSession();
        host.querySelectorAll(".design-platform").forEach((x) =>
          x.classList.toggle("active", x.dataset.platform === state._platform));
        // Apply viewport that matches this platform
        const p = PLATFORMS[state._platform] || PLATFORMS.web;
        const sizer = host.querySelector(".design-sizer");
        if (sizer) {
          if (p.viewport.w) {
            sizer.style.width  = p.viewport.w + "px";
            sizer.style.height = p.viewport.h + "px";
            sizer.style.flex   = "0 0 auto";
          } else {
            sizer.style.width  = "100%";
            sizer.style.height = "";
            sizer.style.flex   = "";
          }
        }
        // Re-render the active version so the device frame wraps
        const v = state.versions[state.activeVersion];
        if (v) showVersion(host, v);
      });
    });
  }

  // --- Platform token presets ------------------------------------------
  // Click any preset to seed the current design's token overrides with
  // a known-good palette for that platform. Overrides apply live via
  // the token panel's existing setProperty pipeline, and "Bake into
  // prompt" carries them into the next generation.

  const TOKEN_PRESETS = {
    "iOS System Light": {
      "--system-blue":              "#007AFF",
      "--system-green":              "#34C759",
      "--system-indigo":              "#5856D6",
      "--system-orange":              "#FF9500",
      "--system-pink":                "#FF2D55",
      "--system-red":                 "#FF3B30",
      "--system-background":          "#FFFFFF",
      "--secondary-system-background": "#F2F2F7",
      "--label":                      "#000000",
      "--secondary-label":            "#3C3C4399",
      "--separator":                  "#3C3C4349",
      "--radius-card":                "12px",
      "--radius-sheet":               "14px",
    },
    "iOS System Dark": {
      "--system-blue":              "#0A84FF",
      "--system-green":              "#30D158",
      "--system-indigo":              "#5E5CE6",
      "--system-orange":              "#FF9F0A",
      "--system-pink":                "#FF375F",
      "--system-red":                 "#FF453A",
      "--system-background":          "#000000",
      "--secondary-system-background": "#1C1C1E",
      "--label":                      "#FFFFFF",
      "--secondary-label":            "#EBEBF599",
      "--separator":                  "#54545899",
      "--radius-card":                "12px",
      "--radius-sheet":               "14px",
    },
    "Material 3 Light": {
      "--md-primary":              "#6750A4",
      "--md-on-primary":            "#FFFFFF",
      "--md-primary-container":     "#EADDFF",
      "--md-on-primary-container":  "#21005D",
      "--md-secondary":             "#625B71",
      "--md-tertiary":              "#7D5260",
      "--md-surface":               "#FFFBFE",
      "--md-on-surface":            "#1C1B1F",
      "--md-surface-variant":       "#E7E0EC",
      "--md-outline":               "#79747E",
      "--md-background":            "#FFFBFE",
      "--md-error":                 "#B3261E",
      "--md-shape-sm":              "4px",
      "--md-shape-md":              "12px",
      "--md-shape-lg":              "16px",
      "--md-shape-xl":              "28px",
    },
    "Material 3 Dark": {
      "--md-primary":              "#D0BCFF",
      "--md-on-primary":            "#381E72",
      "--md-primary-container":     "#4F378B",
      "--md-on-primary-container":  "#EADDFF",
      "--md-secondary":             "#CCC2DC",
      "--md-tertiary":              "#EFB8C8",
      "--md-surface":               "#1C1B1F",
      "--md-on-surface":            "#E6E1E5",
      "--md-surface-variant":       "#49454F",
      "--md-outline":               "#938F99",
      "--md-background":            "#1C1B1F",
      "--md-error":                 "#F2B8B5",
      "--md-shape-sm":              "4px",
      "--md-shape-md":              "12px",
      "--md-shape-lg":              "16px",
      "--md-shape-xl":              "28px",
    },
    "Tailwind Slate": {
      "--color-accent":   "#0ea5e9",
      "--color-bg":       "#ffffff",
      "--color-surface":  "#f8fafc",
      "--color-text":     "#0f172a",
      "--color-muted":    "#475569",
      "--color-border":   "#e2e8f0",
      "--radius-sm":      "4px",
      "--radius-md":      "8px",
      "--radius-lg":      "12px",
    },
    "Tailwind Zinc Dark": {
      "--color-accent":   "#f472b6",
      "--color-bg":       "#09090b",
      "--color-surface":  "#18181b",
      "--color-text":     "#fafafa",
      "--color-muted":    "#a1a1aa",
      "--color-border":   "#27272a",
      "--radius-sm":      "4px",
      "--radius-md":      "8px",
      "--radius-lg":      "12px",
    },
  };

  function renderTokenPresets(host, wrap) {
    if (!wrap) return;
    const platform = state._platform || "web";
    const suggested = new Set();
    if (platform === "ios" || platform === "ipad") {
      suggested.add("iOS System Light"); suggested.add("iOS System Dark");
    } else if (platform === "android") {
      suggested.add("Material 3 Light"); suggested.add("Material 3 Dark");
    } else {
      suggested.add("Tailwind Slate"); suggested.add("Tailwind Zinc Dark");
    }
    const names = Object.keys(TOKEN_PRESETS).sort(
      (a, b) => (suggested.has(b) - suggested.has(a)));
    wrap.innerHTML = names.map((n) =>
      `<button class="design-preset-chip${suggested.has(n) ? " suggested" : ""}" data-preset="${escapeAttr(n)}">${escapeHtml(n)}</button>`
    ).join("");
    wrap.querySelectorAll(".design-preset-chip").forEach((btn) => {
      btn.addEventListener("click", () => applyTokenPreset(host, btn.dataset.preset));
    });
  }

  function applyTokenPreset(host, name) {
    const preset = TOKEN_PRESETS[name];
    if (!preset) return;
    state._tokenOverrides = { ...(state._tokenOverrides || {}), ...preset };
    saveSession();
    for (const [k, v] of Object.entries(preset)) applyTokenOverride(host, k, v);
    // Re-open the tokens panel so the user sees the new rows
    const panel = host.querySelector(".design-tokens-panel");
    if (panel) { panel.remove(); toggleTokens(host); }
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
    const fromHtml = extractTokens(v.html);
    // Also include any tokens the user has seeded via a preset but that
    // the HTML doesn't declare yet, so they appear as editable rows.
    const overrides = state._tokenOverrides || {};
    const seen = new Set(fromHtml.map((t) => t.name));
    const extras = Object.entries(overrides)
      .filter(([name]) => !seen.has(name))
      .map(([name, value]) => ({ name, value }));
    const tokens = fromHtml.concat(extras);
    const panel = document.createElement("aside");
    panel.className = "design-tokens-panel";
    if (!tokens.length) {
      panel.innerHTML = `
        <header><strong>Design tokens</strong><button data-action="close" aria-label="Close">×</button></header>
        <div class="design-tokens-empty">
          <p>No CSS custom properties (<code>--name: value</code>) found in this design.</p>
          <p class="muted">Seed a token preset to start tweaking:</p>
          <div class="design-token-presets" id="design-token-presets"></div>
          <p class="muted" style="margin-top:14px">…or ask the model to "define colors as CSS variables" and regenerate.</p>
        </div>
      `;
      renderTokenPresets(host, panel.querySelector("#design-token-presets"));
    } else {
      panel.innerHTML = `
        <header>
          <strong>Design tokens</strong>
          <span class="muted" style="font-size:11px">${tokens.length} found</span>
          <div style="flex:1"></div>
          <button data-action="close" aria-label="Close">×</button>
        </header>
        <div class="design-token-presets-bar" id="design-token-presets-bar"></div>
        <div class="design-tokens-list" id="design-tokens-list"></div>
        <footer>
          <button data-action="reset" title="Drop all overrides">Reset</button>
          <button data-action="bake-local" title="Regex-rewrite the CSS variables in the source HTML and create a new version — no model call">Apply locally</button>
          <button data-action="bake" title="Include these overrides in the next Generate so they persist">Bake into prompt</button>
        </footer>
      `;
      renderTokenPresets(host, panel.querySelector("#design-token-presets-bar"));
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
    panel.querySelector('[data-action="bake-local"]').addEventListener("click", () => bakeTokensLocally(host));
  }

  // Checklist I — regex-rewrite tokens in the source HTML, create a
  // new version. No model call. Handles three cases per variable:
  //   1. `--name: value;` inside any CSS rule → rewrite value
  //   2. If the token doesn't appear anywhere → inject `:root {...}`
  //      rule in an override style block at end of <head>.
  function bakeTokensLocally(host) {
    const overrides = state._tokenOverrides || {};
    if (!Object.keys(overrides).length) return;
    const v = state.versions[state.activeVersion];
    if (!v) return;
    let html = v.html || "";

    const missing = [];
    for (const [name, value] of Object.entries(overrides)) {
      const bare = name.startsWith("--") ? name.slice(2) : name;
      // Escape regex specials in the variable name
      const escN = bare.replace(/[-\\^$*+?.()|[\]{}]/g, "\\$&");
      const re = new RegExp(`(--${escN}\\s*:\\s*)[^;}\\n]+`, "g");
      const next = html.replace(re, `$1${value}`);
      if (next === html) missing.push([name, value]);
      html = next;
    }

    // Any vars that didn't exist in source — inject :root overrides.
    if (missing.length) {
      const rule = `:root { ${missing.map(([k, v]) => `${k}: ${v};`).join(" ")} }`;
      const block = `\n<style id="__mio-baked-tokens">\n${rule}\n</style>`;
      if (/<\/head>/i.test(html)) html = html.replace(/<\/head>/i, block + "\n</head>");
      else html = block + html;
    }

    const n = state.versions.length + 1;
    const count = Object.keys(overrides).length;
    state.versions.push({
      n,
      title: `v${n} (tokens)`,
      html,
      prompt: `(local token bake: ${count} variable${count === 1 ? "" : "s"})`,
      ts: Date.now(),
    });
    state.activeVersion = state.versions.length - 1;
    state._tokenOverrides = {};
    saveSession();
    // Close panel + re-render
    host.querySelector(".design-tokens-panel")?.remove();
    renderVersions(host);
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

  async function generate(host, { research = false } = {}) {
    const input = host.querySelector("#design-input");
    const prompt = input.value.trim();
    if (!prompt) return;
    const variants = host.querySelector("#design-variants").checked ? 3 : 1;

    state.history.push({ role: "user", text: prompt });
    saveSession();
    renderHistory(host);
    input.value = "";
    const genBtn = host.querySelector('[data-action="generate"]');
    const researchBtn = host.querySelector('[data-action="research"]');
    [genBtn, researchBtn].forEach((b) => { if (b) b.disabled = true; });
    genBtn.textContent = variants > 1 ? "Generating 3…" : "Generating…";

    // Optional research pass — fires web_search + search_images via
    // the existing skill dispatch and injects the results as a system
    // message. Total budget: ~5 seconds, both in parallel.
    let researchContext = null;
    if (research) {
      genBtn.textContent = "Researching…";
      try {
        researchContext = await doResearch(prompt, state._platform || "web");
        state.history.push({ role: "assistant", text: `Research: ${researchContext.summary}` });
        renderHistory(host);
      } catch (e) {
        console.warn("[design] research failed:", e);
      }
      genBtn.textContent = variants > 1 ? "Generating 3…" : "Generating…";
    }

    try {
      // Use the existing OpenAI-compatible endpoint. Model pick-up:
      // whatever the server has loaded as default (mio-large-moe).
      const platform = state._platform || "web";
      const kind     = state._kind || "page";
      const sysPrompt = (PLATFORMS[platform] || PLATFORMS.web).systemPrompt
                      + ((KINDS[kind] || KINDS.page).addendum || "");
      const messages = [
        { role: "system", content: sysPrompt },
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
      // Inject research findings as a system-message prelude.
      if (researchContext) {
        messages.unshift({
          role: "system",
          content: `Research findings for inspiration (do NOT copy — use as directional signal):\n\n${researchContext.text}`,
        });
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

      // Single-variant path streams incrementally into the chat history
      // panel so the user sees tokens arriving. Variants stay
      // non-streaming (rendering 3 streams live would be noise).
      let results;
      if (variants === 1) {
        // Seed a live assistant entry in the chat log; we update its
        // text on each delta.
        const liveIdx = state.history.push({ role: "assistant", text: "", streaming: true }) - 1;
        renderHistory(host);
        const onDelta = (_delta, full) => {
          state.history[liveIdx].text = streamPreview(full);
          // Throttle DOM writes: render at most every 3 chunks
          if (!onDelta._t) onDelta._t = true;
          if (onDelta._pending) return;
          onDelta._pending = true;
          requestAnimationFrame(() => {
            onDelta._pending = false;
            renderHistory(host);
          });
        };
        const text = await runOne(messages, 0.7, onDelta);
        // Finalise the live entry: replace streamed preview with a
        // terse final summary ("Generated v{N}") once the artifact
        // parses; the actual HTML is in state.versions.
        state.history[liveIdx].streaming = false;
        state.history[liveIdx].text = `Generating v${state.versions.length + 1}…`;
        results = [text];
      } else {
        const runs = [];
        for (let i = 0; i < variants; i++) {
          const temp = 0.5 + (i * 0.25);
          runs.push(runOne(messages, temp));
        }
        results = await Promise.all(runs);
      }
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
      if (researchBtn) researchBtn.disabled = false;
    }
  }

  // --- Web research --------------------------------------------------
  // Uses existing skill dispatch — no new backend needed.
  async function doResearch(prompt, platform) {
    const q = `${platform === "ios" ? "iOS app " : platform === "android" ? "Android app " : "web "}UI design ${prompt}`;
    const [searchRes, imageRes] = await Promise.allSettled([
      fetch("/ui/api/skills/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: "web_search", arguments: { query: q, limit: 5 } }),
      }).then((r) => r.json()),
      fetch("/ui/api/skills/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: "search_images", arguments: { query: q, limit: 6 } }),
      }).then((r) => r.json()),
    ]);

    const lines = [];
    const summaryBits = [];
    if (searchRes.status === "fulfilled" && searchRes.value?.results?.length) {
      lines.push("### Reference articles");
      for (const r of searchRes.value.results.slice(0, 5)) {
        const t = r.title || r.url || "";
        const u = r.url || "";
        const snip = (r.snippet || r.summary || "").slice(0, 140).replace(/\s+/g, " ");
        lines.push(`- **${t}** — ${u}${snip ? `\n  ${snip}` : ""}`);
      }
      summaryBits.push(`${searchRes.value.results.length} articles`);
    }
    if (imageRes.status === "fulfilled" && imageRes.value?.results?.length) {
      lines.push("\n### Reference imagery");
      for (const r of imageRes.value.results.slice(0, 6)) {
        lines.push(`- ${r.title || r.source || "image"} — ${r.url || r.image || ""}`);
      }
      summaryBits.push(`${imageRes.value.results.length} images`);
    }
    return {
      text: lines.join("\n") || "(no research results)",
      summary: summaryBits.join(" + ") || "nothing found",
    };
  }

  async function runOne(messages, temperature, onDelta) {
    // Streaming via OpenAI-spec SSE. Caller gets incremental deltas
    // through onDelta(chunk). The full accumulated content is returned
    // on finish. If onDelta is omitted we still stream (same cost) but
    // nothing surfaces live.
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "mio-auto",
        messages,
        temperature,
        max_tokens: 4096,
        stream: true,
      }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let full = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // Parse any complete SSE events (data: …\n\n).
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const evt = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of evt.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === "[DONE]") continue;
          try {
            const j = JSON.parse(payload);
            const delta = j.choices?.[0]?.delta?.content || "";
            if (delta) {
              full += delta;
              if (onDelta) onDelta(delta, full);
            }
          } catch { /* ignore malformed line */ }
        }
      }
    }
    return full;
  }

  // Shorten the streamed body for preview in the chat log: when the
  // model is emitting the big HTML/code body, collapse it to a short
  // "📄 …writing artifact…" marker so the sidebar doesn't vomit
  // thousands of tokens of code. Handles three emission styles:
  //   1. <antArtifact …>…</antArtifact>
  //   2. ```html / ```HTML / ```xml / ```svg / ```python  code fence
  //   3. a bare <!doctype html> / <html> document body
  function streamPreview(full) {
    let t = full;
    // 1. <antArtifact>
    const aOpen = t.indexOf("<antArtifact");
    if (aOpen >= 0) {
      const aClose = t.indexOf("</antArtifact>", aOpen);
      if (aClose >= 0) {
        t = t.slice(0, aOpen) + "📄 …artifact ready…" + t.slice(aClose + "</antArtifact>".length);
      } else {
        return t.slice(0, aOpen).trim() + " 📄 …writing artifact…";
      }
    }
    // 2. Fenced code block (html / xml / svg / python / plain ``` )
    const fenceOpen = t.search(/```(?:html|HTML|xml|svg|python|py|js|javascript|jsx|tsx)?/);
    if (fenceOpen >= 0) {
      // Look for the closing fence after it
      const after = t.indexOf("\n```", fenceOpen + 3);
      if (after >= 0) {
        t = t.slice(0, fenceOpen) + "📄 …artifact ready…" + t.slice(after + 4);
      } else {
        return t.slice(0, fenceOpen).trim() + " 📄 …writing artifact…";
      }
    }
    // 3. Bare <!doctype html> or <html> in the middle of prose
    const docStart = t.search(/<!doctype\s+html|<html\b/i);
    if (docStart >= 0) {
      const docEnd = t.toLowerCase().indexOf("</html>", docStart);
      if (docEnd >= 0) {
        t = t.slice(0, docStart) + "📄 …artifact ready…" + t.slice(docEnd + "</html>".length);
      } else {
        return t.slice(0, docStart).trim() + " 📄 …writing artifact…";
      }
    }
    // Keep the tail (latest words) rather than the head.
    if (t.length > 600) t = "…" + t.slice(t.length - 600);
    return t;
  }

  function extractArtifact(text) {
    // Preferred shape: <antArtifact …>…</antArtifact>
    let attrs = "", body = null;
    const m = text.match(/<antArtifact([^>]*)>([\s\S]*?)<\/antArtifact>/);
    if (m) {
      attrs = m[1] || "";
      body  = m[2].trim();
    } else {
      // Fallback 1: ```html … ``` / ```HTML … ``` code fence.
      const fence = text.match(/```(?:html|HTML|xml|svg)?\s*\n([\s\S]*?)\n```/);
      if (fence) body = fence[1].trim();
      // Fallback 2: a bare <!doctype html>…</html> or <html>…</html>.
      if (!body) {
        const html = text.match(/<!doctype html[\s\S]*?<\/html>/i)
                  || text.match(/<html[\s\S]*?<\/html>/i);
        if (html) body = html[0].trim();
      }
    }
    if (!body) return null;

    // Blender kind: repackage python code in a mini viewer.
    if (/type\s*=\s*"application\/vnd\.pimio\.blender"/i.test(attrs) ||
        (state._kind === "blender")) {
      return buildBlenderViewer(body);
    }
    return body;
  }

  function buildBlenderViewer(code) {
    // Self-contained HTML that shows the bpy code and POSTs it to the
    // blender_exec skill when the user hits the button.
    const escaped = code.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const json = JSON.stringify(code);
    return `<!doctype html>
<html><head><meta charset="utf-8"><style>
  :root { color-scheme: dark; }
  body { margin:0; font-family: -apple-system, system-ui, sans-serif; background: #0f1115; color: #e8e9ec; }
  header { display:flex; align-items:center; gap:10px; padding: 10px 14px; border-bottom: 1px solid #2a2e38; background: #171a21; }
  header strong { font-size: 13px; }
  header .muted { color: #8a8f98; font-size: 11px; flex: 1; }
  #send { background: #E87D0D; color: #fff; border: 0; padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 500; cursor: pointer; }
  #send:hover { filter: brightness(1.08); }
  #send:disabled { opacity: 0.6; cursor: wait; }
  #status { font-size: 11px; padding: 6px 14px; min-height: 16px; }
  #status.ok { color: #7fd07f; } #status.err { color: #f47b7b; }
  pre { margin: 0; padding: 14px 18px; font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; line-height: 1.55; overflow: auto; background: #0d1117; height: calc(100vh - 96px); }
</style></head>
<body>
  <header>
    <strong>🟠 Blender</strong>
    <span class="muted">bpy code — runs in your open Blender via the blender-mcp addon (localhost:9876)</span>
    <button id="send">▶ Send to Blender</button>
    <button id="snap">📸 Snapshot</button>
  </header>
  <div id="status"></div>
  <pre><code>${escaped}</code></pre>
<script>
const code = ${json};
async function run(skill, args){
  const r = await fetch("/ui/api/skills/run", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({name: skill, arguments: args || {}})
  });
  return r.json();
}
function status(msg, cls){ const el = document.getElementById('status'); el.textContent = msg; el.className = cls||""; }
document.getElementById('send').addEventListener('click', async () => {
  const btn = document.getElementById('send');
  btn.disabled = true; btn.textContent = 'Sending…';
  status("Running in Blender…");
  try {
    const data = await run("blender_exec", {code});
    if (data.error) { status("Error: " + data.error + (data.hint ? " — " + data.hint : ""), "err"); }
    else if (!data.ok) { status("Blender reported: " + (data.stdout || "failed"), "err"); }
    else { status("✓ Done. " + (data.stdout || "").slice(0, 200), "ok"); }
  } catch (e) { status("Failed: " + e.message, "err"); }
  finally { btn.disabled = false; btn.textContent = '▶ Send to Blender'; }
});
document.getElementById('snap').addEventListener('click', async () => {
  status("Rendering viewport…");
  const data = await run("blender_snapshot", {});
  if (data.url) {
    const img = new Image(); img.src = data.url; img.style.maxWidth = '100%'; img.style.marginTop = '8px';
    const s = document.getElementById('status'); s.textContent = ""; s.className=""; s.appendChild(img);
  } else {
    status("Snapshot failed: " + (data.error || "unknown"), "err");
  }
});
</script>
</body></html>`;
  }

  function renderErrorHTML(rawReply) {
    return `<!doctype html><html><body style="margin:0;padding:24px;font-family:-apple-system,system-ui,sans-serif;color:#333;background:#fff"><h2 style="margin:0 0 10px">No &lt;antArtifact&gt; in the reply</h2><p style="color:#666;font-size:13px;margin:0 0 12px">The model didn't wrap its output in the expected tag. Raw reply:</p><pre style="background:#f5f5f5;padding:12px;border-radius:6px;font-size:12px;white-space:pre-wrap;overflow:auto;max-height:60vh">${escapeHtml(rawReply)}</pre></body></html>`;
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }
})();
