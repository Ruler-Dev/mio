// view_docs.js — Docs & RAG view.
//
// Three tabs:
//   • Clipped   — markdown files under ~/.mio/ingest/ (browser extension)
//                 with tag filter, delete, @-mention
//   • Folders   — indexed folders (SQLite FTS5): add / re-index / drop
//   • Search    — full-text query across everything, preview snippets

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
                <p class="muted">Clipped web pages + indexed folders, all searchable from chat.</p>
              </div>
              <div class="view-header-actions">
                <button class="btn-ghost" data-action="refresh">Refresh</button>
              </div>
            </header>
            <div class="docs-tabs" role="tablist">
              <button class="docs-tab active" data-tab="clipped" role="tab">Clipped</button>
              <button class="docs-tab"        data-tab="folders" role="tab">Folders</button>
              <button class="docs-tab"        data-tab="search"  role="tab">Search</button>
            </div>
            <div class="view-body">
              <div class="docs-panel" data-panel="clipped"></div>
              <div class="docs-panel" data-panel="folders" hidden></div>
              <div class="docs-panel" data-panel="search"  hidden></div>
            </div>
          </div>
        `;
        const tabs = host.querySelectorAll(".docs-tab");
        tabs.forEach((t) => t.addEventListener("click", () => select(host, t.dataset.tab)));
        host.querySelector('[data-action="refresh"]').addEventListener("click", () => select(host, current(host), true));
        renderClipped(panel(host, "clipped"));
      },
    });
  };
  ready();

  function panel(host, name) { return host.querySelector(`[data-panel="${name}"]`); }
  function current(host) { return host.querySelector(".docs-tab.active")?.dataset.tab || "clipped"; }
  function select(host, name, force = false) {
    if (!force && current(host) === name) return;
    host.querySelectorAll(".docs-tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    host.querySelectorAll(".docs-panel").forEach((p) => p.hidden = p.dataset.panel !== name);
    const p = panel(host, name);
    if (name === "clipped") renderClipped(p);
    else if (name === "folders") renderFolders(p);
    else if (name === "search") renderSearch(p);
  }

  // ---- Clipped panel ----------------------------------------------------

  async function renderClipped(panel) {
    panel.innerHTML = `<div class="muted">Loading…</div>`;
    const res = await fetch("/ui/api/ingest");
    const { items = [], tags = [] } = await res.json();
    if (!items.length) {
      panel.innerHTML = `
        <div class="docs-empty">
          <h2>Nothing clipped yet</h2>
          <p>Install the Mio Clip Safari / Chrome extension from
            <code>browser-extension/</code>, then hit the toolbar button on any page.
            Clippings land here and are auto-indexed for chat search.</p>
        </div>`;
      return;
    }
    panel.innerHTML = `
      <div class="docs-toolbar">
        <div class="docs-chips">
          <button class="docs-chip active" data-tag="">All (${items.length})</button>
          ${tags.map(t => `<button class="docs-chip" data-tag="${escapeAttr(t)}">${escapeHtml(t)}</button>`).join("")}
        </div>
      </div>
      <div class="docs-grid" id="docs-grid"></div>
    `;
    const grid = panel.querySelector("#docs-grid");
    for (const it of items) grid.appendChild(card(it));
    panel.querySelectorAll(".docs-chip").forEach((chip) => {
      chip.addEventListener("click", async () => {
        panel.querySelectorAll(".docs-chip").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        const tag = chip.dataset.tag;
        const r = await fetch("/ui/api/ingest" + (tag ? "?tag=" + encodeURIComponent(tag) : ""));
        const { items: filtered = [] } = await r.json();
        grid.innerHTML = "";
        for (const it of filtered) grid.appendChild(card(it));
      });
    });
  }

  function card(it) {
    const el = document.createElement("div");
    el.className = "docs-card";
    const when = new Date(it.mtime * 1000).toLocaleString();
    el.innerHTML = `
      <div class="docs-card-title">${escapeHtml(it.title || it.id)}</div>
      <a class="docs-card-url" href="${escapeAttr(it.url)}" target="_blank" rel="noopener">${escapeHtml(it.url || "")}</a>
      <div class="docs-card-tags">${(it.tags || []).map(t => `<span class="docs-tag-pill">${escapeHtml(t)}</span>`).join("")}</div>
      <div class="docs-card-meta"><span>${kib(it.size)}</span><span>•</span><span>${escapeHtml(when)}</span></div>
      <div class="docs-card-actions">
        <button data-act="mention">@-mention</button>
        <button data-act="delete" class="danger">Delete</button>
      </div>
    `;
    el.querySelector('[data-act="mention"]').addEventListener("click", () => mentionInChat(it));
    el.querySelector('[data-act="delete"]').addEventListener("click", async () => {
      if (!confirm(`Delete "${it.title || it.id}"?`)) return;
      await fetch(`/ui/api/ingest/${encodeURIComponent(it.id)}`, { method: "DELETE" });
      el.remove();
    });
    return el;
  }

  function mentionInChat(it) {
    const input = document.querySelector("#input, textarea#messageInput, textarea.input, textarea");
    if (input) {
      const token = `@doc:${it.id} `;
      input.value = (input.value || "") + token;
      input.focus();
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
    if (window.Mio?.views?.switch) window.Mio.views.switch("chat");
  }

  // ---- Folders panel ---------------------------------------------------

  async function renderFolders(panel) {
    panel.innerHTML = `<div class="muted">Loading…</div>`;
    const res = await fetch("/ui/api/rag/indexes");
    const { indexes = [], error } = await res.json();
    panel.innerHTML = `
      <div class="docs-toolbar">
        <form class="docs-add-folder" id="docs-add-folder">
          <input name="path"  type="text" placeholder="Absolute folder path (e.g. /Users/me/notes)" required>
          <input name="label" type="text" placeholder="Label (optional)">
          <button type="submit">Index folder</button>
        </form>
      </div>
      ${error ? `<div class="docs-empty"><p>Error loading indexes: ${escapeHtml(error)}</p></div>` : ""}
      <div class="docs-grid" id="idx-grid"></div>
    `;
    const grid = panel.querySelector("#idx-grid");
    if (!indexes.length) {
      grid.innerHTML = `
        <div class="docs-empty">
          <h2>No indexed folders</h2>
          <p>Point Mio at any folder on disk — it'll full-text-index the contents into
            <code>~/.mio/rag.sqlite</code> (SQLite FTS5, no embeddings, no GPU).</p>
        </div>`;
    } else {
      for (const idx of indexes) grid.appendChild(idxCard(idx, grid));
    }
    panel.querySelector("#docs-add-folder").addEventListener("submit", async (e) => {
      e.preventDefault();
      const form = e.target;
      const body = {
        path:  form.elements.path.value.trim(),
        label: form.elements.label.value.trim() || null,
      };
      const btn = form.querySelector("button");
      btn.disabled = true; btn.textContent = "Indexing…";
      try {
        const r = await fetch("/ui/api/rag/index", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await r.json();
        if (data.error) alert("Error: " + data.error);
        renderFolders(panel);
      } finally {
        btn.disabled = false; btn.textContent = "Index folder";
      }
    });
  }

  function idxCard(idx, grid) {
    const el = document.createElement("div");
    el.className = "docs-card";
    el.innerHTML = `
      <div class="docs-card-title">${escapeHtml(idx.label || idx.path.split("/").filter(Boolean).pop() || idx.path)}</div>
      <div class="docs-card-url">${escapeHtml(idx.path)}</div>
      <div class="docs-card-meta">
        <span>${idx.file_count} files</span>
        <span>•</span>
        <span>indexed ${escapeHtml(idx.indexed_at || "")}</span>
      </div>
      <div class="docs-card-actions">
        <button data-act="reindex">Re-index</button>
        <button data-act="drop" class="danger">Drop</button>
      </div>
    `;
    el.querySelector('[data-act="reindex"]').addEventListener("click", async () => {
      const btn = el.querySelector('[data-act="reindex"]');
      btn.disabled = true; btn.textContent = "Re-indexing…";
      await fetch("/ui/api/rag/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: idx.path, label: idx.label }),
      });
      renderFolders(el.parentElement.parentElement);
    });
    el.querySelector('[data-act="drop"]').addEventListener("click", async () => {
      if (!confirm(`Drop index "${idx.label || idx.path}"?`)) return;
      await fetch(`/ui/api/rag/index/${idx.id}`, { method: "DELETE" });
      renderFolders(el.parentElement.parentElement);
    });
    return el;
  }

  // ---- Search panel ----------------------------------------------------

  function renderSearch(panel) {
    panel.innerHTML = `
      <div class="docs-toolbar">
        <form class="docs-search-form" id="docs-search-form">
          <input name="q" type="text" placeholder="Full-text query across clipped docs + indexed folders" autofocus>
          <button type="submit">Search</button>
        </form>
      </div>
      <div id="docs-search-results" class="docs-search-results"></div>
    `;
    panel.querySelector("#docs-search-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const q = e.target.elements.q.value.trim();
      if (!q) return;
      const results = panel.querySelector("#docs-search-results");
      results.innerHTML = `<div class="muted">Searching…</div>`;
      const r = await fetch("/ui/api/rag/search?q=" + encodeURIComponent(q) + "&limit=20");
      const data = await r.json();
      if (data.error) {
        results.innerHTML = `<div class="docs-empty"><p>${escapeHtml(data.error)}</p></div>`;
        return;
      }
      const hits = data.results || [];
      if (!hits.length) {
        results.innerHTML = `<div class="docs-empty"><p>No matches.</p></div>`;
        return;
      }
      results.innerHTML = hits.map(renderHit).join("");
    });
  }

  function renderHit(h) {
    const snippet = (h.snippet || "").replace(/</g, "&lt;");
    return `
      <div class="docs-hit">
        <div class="docs-hit-head">
          <strong>${escapeHtml(h.title || h.filename || h.path || "")}</strong>
          <span class="docs-hit-path">${escapeHtml(h.path || "")}</span>
        </div>
        <div class="docs-hit-snippet">${snippet}</div>
      </div>
    `;
  }

  // ---- helpers ---------------------------------------------------------

  function kib(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
  }
  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }
  function escapeAttr(s) { return escapeHtml(s); }
})();
