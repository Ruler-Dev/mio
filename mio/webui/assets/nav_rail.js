// nav_rail.js — 48px vertical icon rail on the far left.
//
// Owns the rail DOM, reads the view registry (Mio.views), and wires the
// click handlers. Settings at the bottom is not a view — it opens the
// existing openSettings() modal.
//
// Rail item order is fixed here rather than in Mio.views because the
// rail is the product surface, not the registry's concern.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.navRail) return;

  const ITEMS = [
    { view: "chat",        title: "Chat",       icon: iconChat(),       shortcut: "⌘1" },
    { view: "workspaces",  title: "Workspaces", icon: iconWorkspaces(), shortcut: "⌘2" },
    { view: "docs",        title: "Docs & RAG", icon: iconDocs(),       shortcut: "⌘3" },
    { view: "design",      title: "Design",     icon: iconDesign(),     shortcut: "⌘4" },
    { view: "obsidian",    title: "Obsidian",   icon: iconObsidian(),   shortcut: "⌘5" },
    { view: "dashboards",  title: "Dashboards", icon: iconDashboards(), shortcut: "⌘6" },
    { view: "flow",        title: "Flow",       icon: iconFlow(),       shortcut: "⌘7" },
    { view: "journal",     title: "Journal",    icon: iconJournal(),    shortcut: "⌘8" },
    { view: "notebook",    title: "Notebook",   icon: iconNotebook(),   shortcut: "⌘9" },
  ];
  // Bottom section (below a spacer) — settings + dashboard link
  const BOTTOM = [
    { title: "Dashboard", icon: iconDashboard(), href: "/ui/dashboard" },
    { title: "Playground", icon: iconPlayground(), href: "/ui/playground" },
    { title: "Stats", icon: iconStats(), href: "/ui/stats" },
    { title: "Settings", icon: iconSettings(), onclick: () => {
        if (typeof openSettings === "function") openSettings();
    } },
  ];

  function mount() {
    const app = document.getElementById("app");
    if (!app) return;
    if (document.querySelector(".nav-rail")) return; // idempotent
    const rail = document.createElement("div");
    rail.className = "nav-rail";
    rail.innerHTML = `
      <div class="nav-rail-logo" title="Mio">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" width="20" height="20">
          <circle cx="12" cy="12" r="9" opacity="0.25"/>
          <path d="M12 3 C 16 8, 16 16, 12 21 C 8 16, 8 8, 12 3 Z"/>
        </svg>
      </div>
      <div class="nav-rail-group" data-group="top"></div>
      <div class="nav-rail-spacer"></div>
      <div class="nav-rail-group" data-group="bottom"></div>
    `;
    app.prepend(rail);

    const top = rail.querySelector('[data-group="top"]');
    for (const it of ITEMS) {
      top.appendChild(button(it));
    }
    const bot = rail.querySelector('[data-group="bottom"]');
    for (const it of BOTTOM) {
      bot.appendChild(button(it));
    }
  }

  function button(it) {
    const b = document.createElement("button");
    b.className = "nav-rail-btn";
    b.title = it.title;
    b.setAttribute("aria-label", it.title);
    if (it.view) {
      b.dataset.view = it.view;
      b.addEventListener("click", () => {
        if (window.Mio?.views?.switch) {
          window.Mio.views.switch(it.view);
        }
      });
    } else if (it.href) {
      b.addEventListener("click", () => {
        window.location.href = it.href;
      });
    } else if (it.onclick) {
      b.addEventListener("click", it.onclick);
    }
    const tip = escapeHtml(it.title) + (it.shortcut ? `<span class="nav-rail-tip-kbd">${escapeHtml(it.shortcut)}</span>` : "");
    b.innerHTML = `
      <span class="nav-rail-icon">${it.icon}</span>
      <span class="nav-rail-badge" data-badge-for="${it.view || ""}" hidden></span>
      <span class="nav-rail-tip">${tip}</span>
    `;
    return b;
  }

  // --- Keyboard shortcuts ------------------------------------------------

  function bindShortcuts() {
    window.addEventListener("keydown", (e) => {
      // Only plain ⌘/Ctrl + 1..5 — respect input focus so we don't eat
      // the user's own "type 1 in my prompt" situation.
      if (!(e.metaKey || e.ctrlKey) || e.shiftKey || e.altKey) return;
      const match = {
        Digit1: "chat", Digit2: "workspaces", Digit3: "docs",
        Digit4: "design", Digit5: "obsidian", Digit6: "dashboards",
        Digit7: "flow", Digit8: "journal", Digit9: "notebook",
      }[e.code];
      if (!match) return;
      e.preventDefault();
      window.Mio?.views?.switch?.(match);
    });
  }

  // --- Badges -----------------------------------------------------------

  async function refreshBadges() {
    const docsBadge = document.querySelector('[data-badge-for="docs"]');
    if (docsBadge) {
      try {
        const r = await fetch("/ui/api/ingest?limit=500");
        const { items = [] } = await r.json();
        if (items.length > 0) {
          docsBadge.textContent = items.length > 99 ? "99+" : String(items.length);
          docsBadge.hidden = false;
        }
      } catch { /* quietly skip */ }
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c])
    );
  }

  // --- Inline SVGs (sized 20x20, stroke currentColor) -------------------

  function svg(inner) {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" width="20" height="20">${inner}</svg>`;
  }
  function iconChat()       { return svg(`<path d="M21 12a8 8 0 0 1-11.5 7.2L4 21l1.8-5.5A8 8 0 1 1 21 12Z"/>`); }
  function iconWorkspaces() { return svg(`<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>`); }
  function iconDocs()       { return svg(`<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h6"/>`); }
  function iconDesign()     { return svg(`<path d="M12 3l1.8 5 5.2 1.8L15 13.8 16 19l-4-2.6L8 19l1-5.2-4-4.2L10.2 8z"/>`); }
  function iconObsidian()   { return svg(`<path d="M12 2l8 6-3 11H7L4 8z"/><path d="M12 2l-5 6 5 6 5-6z" opacity="0.5"/>`); }
  function iconDashboards() { return svg(`<rect x="3" y="3" width="8" height="5" rx="1.5"/><rect x="13" y="3" width="8" height="9" rx="1.5"/><rect x="3" y="10" width="8" height="11" rx="1.5"/><rect x="13" y="14" width="8" height="7" rx="1.5"/>`); }
  function iconFlow()       { return svg(`<circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="18" r="2"/><path d="M7 7l3 9M17 7l-3 9"/>`); }
  function iconJournal()    { return svg(`<path d="M6 3h10a2 2 0 0 1 2 2v15l-3-2-3 2-3-2-3 2V5a2 2 0 0 1 2-2z"/><path d="M9 7h6M9 11h6"/>`); }
  function iconNotebook()   { return svg(`<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 10h16M10 4v16"/>`); }
  function iconDashboard()  { return svg(`<path d="M3 12L12 4l9 8"/><path d="M5 10v10h14V10"/>`); }
  function iconPlayground() { return svg(`<circle cx="12" cy="12" r="9"/><path d="M10 9l5 3-5 3z" fill="currentColor"/>`); }
  function iconStats()      { return svg(`<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>`); }
  function iconSettings()   { return svg(`<circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M2 12h4M18 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8"/>`); }

  // Initial mount — as early as possible so it doesn't flash.
  function boot() {
    mount();
    bindShortcuts();
    refreshBadges();
    // Refresh badges when user returns to the tab (likely did something)
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) refreshBadges();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }

  window.Mio.navRail = { mount, refreshBadges };
})();
