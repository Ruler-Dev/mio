// views.js — view registry + router.
//
// Mio has a single-shell SPA. The "Chat" surface (.sidebar + .main) is
// considered the baseline view and owns most of the existing DOM. Other
// views (Workspaces, Docs, Design, Obsidian, Settings) register here and
// mount into a sibling overlay container `#view-stage`.
//
// Contract:
//   Mio.views.register(name, {
//     title:      "Workspaces",
//     icon:       "<svg …>",       // rail icon (inline SVG string)
//     mount(el):  attach your DOM into `el` (first activation; lazy init)
//     activate(): called every time the view becomes visible
//     deactivate(): called when leaving the view
//   });
//   Mio.views.switch("docs");
//
// Persists the active view in localStorage so reloads restore it.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.views) return;

  const STORAGE_KEY = "mio.activeView";
  const DEFAULT_VIEW = "chat";
  const registry = new Map();
  let activeView = null;
  let stageEl = null;
  let chatEls = null; // cached references to chat surface we hide/show

  function getStage() {
    if (stageEl && stageEl.isConnected) return stageEl;
    stageEl = document.getElementById("view-stage");
    if (!stageEl) {
      stageEl = document.createElement("div");
      stageEl.id = "view-stage";
      stageEl.className = "view-stage";
      stageEl.style.display = "none";
      const app = document.getElementById("app");
      (app || document.body).appendChild(stageEl);
    }
    return stageEl;
  }

  function cacheChatEls() {
    if (chatEls) return chatEls;
    chatEls = {
      sidebar: document.querySelector(".app > .sidebar"),
      main:    document.querySelector(".app > .main"),
    };
    return chatEls;
  }

  function showChat(show) {
    const { sidebar, main } = cacheChatEls();
    const v = show ? "" : "none";
    if (sidebar) sidebar.style.display = v;
    if (main)    main.style.display    = v;
  }

  function register(name, spec) {
    if (!name || typeof name !== "string") return;
    registry.set(name, { mounted: false, ...spec });
  }

  function list() {
    return Array.from(registry.entries()).map(([name, v]) => ({
      name,
      title: v.title || name,
      icon:  v.icon  || null,
      badge: v.badge || null,
    }));
  }

  function getActive() { return activeView; }

  function _switch(name) {
    if (name === activeView) return;
    const spec = registry.get(name);
    if (!spec && name !== "chat") {
      console.warn("[Mio.views] unknown view:", name);
      return;
    }

    // Deactivate previous
    if (activeView && activeView !== "chat") {
      const prev = registry.get(activeView);
      try { prev?.deactivate?.(); } catch (e) { console.warn(e); }
    }

    activeView = name;
    try { localStorage.setItem(STORAGE_KEY, name); } catch {}

    // Update rail active state
    document.querySelectorAll(".nav-rail-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.view === name);
    });

    // Chat is baseline — hide the stage, show the existing chat surface.
    if (name === "chat") {
      getStage().style.display = "none";
      showChat(true);
      return;
    }

    // Other views — hide the chat, show the stage with this view's DOM.
    const stage = getStage();
    showChat(false);
    stage.innerHTML = "";
    stage.style.display = "flex";

    // Lazy-mount the view the first time it's opened
    const host = document.createElement("div");
    host.className = "view view-" + name;
    host.style.flex = "1";
    host.style.display = "flex";
    host.style.minWidth = "0";
    stage.appendChild(host);
    if (!spec.mounted) {
      try { spec.mount?.(host); spec.mounted = true; }
      catch (e) { console.error("[Mio.views] mount failed for " + name, e); }
    } else {
      // Re-mount into the fresh host (we recreated the DOM on switch)
      try { spec.mount?.(host); }
      catch (e) { console.error("[Mio.views] re-mount failed for " + name, e); }
    }
    try { spec.activate?.(); } catch (e) { console.warn(e); }
  }

  function initialView() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved && (saved === "chat" || registry.has(saved))) return saved;
    } catch {}
    return DEFAULT_VIEW;
  }

  // Boot: after DOM is ready, switch to whatever the user had open last.
  function boot() {
    // Make sure the chat surface is visible while the rail decides.
    showChat(true);
    _switch(initialView());
  }

  window.Mio.views = {
    register,
    list,
    switch: _switch,
    getActive,
    _boot: boot,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    // The deterministic loader exposes one promise for the complete module
    // registry. Wait for it so a persisted non-chat view is registered before
    // initialView() checks it.
    (window.Mio.modulesReady || Promise.resolve()).then(boot);
  }
})();
