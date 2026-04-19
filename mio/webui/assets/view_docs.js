// view_docs.js — Docs & RAG view.
//
// Real functional minimum today: list everything ingested via the Mio
// Clip browser extension (GET /ui/api/ingest), let the user delete
// entries, and show basic metadata. Upload / folder-indexing UI lands
// in the next iteration.

(function () {
  window.Mio = window.Mio || {};
  const ready = () => {
    if (!window.Mio.views) return setTimeout(ready, 50);
    window.Mio.views.register("docs", {
      title: "Docs & RAG",
      mount(host) {
        host.innerHTML = `
          <div class="view-docs">
            <header class="view-header">
              <div>
                <h1>Docs &amp; RAG</h1>
                <p class="muted">Clipped from the browser extension and indexed for local search.</p>
              </div>
              <div class="view-header-actions">
                <button class="btn-ghost" data-action="refresh">Refresh</button>
              </div>
            </header>
            <div class="view-body">
              <div class="docs-grid" id="docs-grid">
                <div class="muted">Loading…</div>
              </div>
            </div>
          </div>
        `;
        const grid = host.querySelector("#docs-grid");
        host.querySelector('[data-action="refresh"]').addEventListener("click", () => load(grid));
        load(grid);
      },
    });
  };
  ready();

  async function load(grid) {
    grid.innerHTML = `<div class="muted">Loading…</div>`;
    try {
      const res = await fetch("/ui/api/ingest");
      const { items = [] } = await res.json();
      if (!items.length) {
        grid.innerHTML = `
          <div class="docs-empty">
            <h2>Nothing clipped yet</h2>
            <p>Install the Mio Clip Safari / Chrome extension from
              <code>browser-extension/</code>, then hit the toolbar button
              on any page. It lands here and in your local RAG index.</p>
          </div>
        `;
        return;
      }
      grid.innerHTML = "";
      for (const it of items) {
        grid.appendChild(card(it));
      }
    } catch (e) {
      grid.innerHTML = `<div class="docs-empty"><p>Failed to load: ${escapeHtml(String(e))}</p></div>`;
    }
  }

  function card(it) {
    const el = document.createElement("div");
    el.className = "docs-card";
    const when = new Date(it.mtime * 1000).toLocaleString();
    el.innerHTML = `
      <div class="docs-card-title">${escapeHtml(it.title || it.id)}</div>
      <a class="docs-card-url" href="${escapeAttr(it.url)}" target="_blank" rel="noopener">${escapeHtml(it.url || "")}</a>
      <div class="docs-card-meta">
        <span>${kib(it.size)}</span>
        <span>•</span>
        <span>${escapeHtml(when)}</span>
      </div>
      <div class="docs-card-actions">
        <button data-act="mention">@-mention in chat</button>
        <button data-act="delete" class="danger">Delete</button>
      </div>
    `;
    el.querySelector('[data-act="mention"]').addEventListener("click", () => {
      // Insert a @-mention token into the chat input. The Chat view's
      // existing code handles the actual lookup via search_local_folder.
      if (typeof window.insertIntoInput === "function") {
        window.insertIntoInput(`@doc:${it.id} `);
      }
      if (window.Mio?.views?.switch) window.Mio.views.switch("chat");
    });
    el.querySelector('[data-act="delete"]').addEventListener("click", async () => {
      if (!confirm(`Delete "${it.title || it.id}"?`)) return;
      await fetch(`/ui/api/ingest/${encodeURIComponent(it.id)}`, { method: "DELETE" });
      el.remove();
    });
    return el;
  }

  function kib(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }
  function escapeAttr(s) { return escapeHtml(s); }
})();
