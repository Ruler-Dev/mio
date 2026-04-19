// view_workspaces.js — Workspaces view.
//
// A workspace = reusable bundle of (name, description, system prompt,
// tier, context window, caveman level, pinned prompts, files). Backed
// by /ui/api/projects.
//
// Grid of workspace cards + "+ New workspace". Clicking a workspace
// opens its detail editor; "Open chat" on a workspace switches to the
// Chat view with that workspace selected in the existing project
// selector (so the sidebar filters to just its chats).

(function () {
  window.Mio = window.Mio || {};
  const ready = () => {
    if (!window.Mio.views) return setTimeout(ready, 50);
    window.Mio.views.register("workspaces", {
      title: "Workspaces",
      mount(host) {
        host.innerHTML = `
          <div class="view-workspaces">
            <header class="view-header">
              <div>
                <h1>Workspaces</h1>
                <p class="muted">Reusable bundles — model, context, system prompt, pinned prompts, all in one.</p>
              </div>
              <div class="view-header-actions">
                <button class="btn-ghost" data-action="new">+ New workspace</button>
                <button class="btn-ghost" data-action="refresh">Refresh</button>
              </div>
            </header>
            <div class="view-body">
              <div class="workspaces-grid" id="ws-grid">
                <div class="muted">Loading…</div>
              </div>
            </div>
          </div>
        `;
        const grid = host.querySelector("#ws-grid");
        host.querySelector('[data-action="new"]').addEventListener("click", () => openEditor(null, grid));
        host.querySelector('[data-action="refresh"]').addEventListener("click", () => load(grid));
        load(grid);
      },
    });
  };
  ready();

  async function load(grid) {
    grid.innerHTML = `<div class="muted">Loading…</div>`;
    try {
      const [projRes, sessRes] = await Promise.all([
        fetch("/ui/api/projects"),
        fetch("/ui/api/sessions"),
      ]);
      const { projects = [] } = await projRes.json();
      const { sessions = [] } = await sessRes.json();
      const countByPid = {};
      for (const s of sessions) {
        if (s.project_id) countByPid[s.project_id] = (countByPid[s.project_id] || 0) + 1;
      }
      grid.innerHTML = "";
      if (!projects.length) {
        grid.appendChild(emptyStateCard(grid));
      }
      for (const p of projects) grid.appendChild(card(p, countByPid[p.id] || 0, grid));
      grid.appendChild(newCard(grid));
    } catch (e) {
      grid.innerHTML = `<div class="muted">Failed to load: ${escapeHtml(String(e))}</div>`;
    }
  }

  function emptyStateCard(grid) {
    const el = document.createElement("div");
    el.className = "ws-empty";
    el.innerHTML = `
      <h2>No workspaces yet</h2>
      <p>Create one to bundle a system prompt, a model tier, a context window, and your pinned prompts.</p>
    `;
    return el;
  }

  function card(p, chatCount, grid) {
    const el = document.createElement("div");
    el.className = "ws-card";
    el.style.setProperty("--ws-accent", p.color || "#3b82f6");
    el.innerHTML = `
      <div class="ws-card-stripe"></div>
      <div class="ws-card-body">
        <div class="ws-card-icon">${escapeHtml(p.icon || "◉")}</div>
        <div class="ws-card-title">${escapeHtml(p.name || "Untitled")}</div>
        <div class="ws-card-desc">${escapeHtml(p.description || "")}</div>
        <div class="ws-card-meta">
          <span>${escapeHtml(p.tier || "any tier")}</span>
          <span>•</span>
          <span>${p.context_window ? humanCtx(p.context_window) : "default ctx"}</span>
          <span>•</span>
          <span>${chatCount} chat${chatCount === 1 ? "" : "s"}</span>
        </div>
      </div>
      <div class="ws-card-actions">
        <button data-act="open">Open chat</button>
        <button data-act="edit">Edit</button>
        <button data-act="delete" class="danger">Delete</button>
      </div>
    `;
    el.querySelector('[data-act="open"]').addEventListener("click", () => openInChat(p));
    el.querySelector('[data-act="edit"]').addEventListener("click", () => openEditor(p, grid));
    el.querySelector('[data-act="delete"]').addEventListener("click", async () => {
      if (!confirm(`Delete workspace "${p.name}"?`)) return;
      await fetch(`/ui/api/projects/${encodeURIComponent(p.id)}`, { method: "DELETE" });
      load(grid);
    });
    return el;
  }

  function newCard(grid) {
    const el = document.createElement("div");
    el.className = "ws-card ws-card-new";
    el.innerHTML = `
      <div class="ws-card-body" style="align-items:center;justify-content:center;gap:8px">
        <div style="font-size:28px;opacity:0.5">+</div>
        <div style="color:var(--text-secondary)">New workspace</div>
      </div>
    `;
    el.addEventListener("click", () => openEditor(null, grid));
    return el;
  }

  function openInChat(p) {
    // Tell the existing Chat view to select this project in the top-bar
    // project dropdown, then switch views.
    const select = document.getElementById("projectSelect");
    if (select) {
      select.value = p.id;
      if (typeof window.setActiveProject === "function") {
        try { window.setActiveProject(p.id); } catch {}
      }
    }
    if (window.Mio?.views?.switch) window.Mio.views.switch("chat");
    // Best-effort tier/context application if the workspace pins them
    if (p.tier && typeof window.switchTier === "function") {
      try { window.switchTier(p.tier); } catch {}
    }
  }

  function openEditor(existing, grid) {
    const backdrop = document.createElement("div");
    backdrop.className = "ws-editor-backdrop";
    const dlg = document.createElement("div");
    dlg.className = "ws-editor";
    const isNew = !existing;
    const p = existing || { color: "#3b82f6" };
    dlg.innerHTML = `
      <h2>${isNew ? "New workspace" : "Edit workspace"}</h2>
      <label><span>Name</span><input id="ws-f-name" type="text" value="${escapeAttr(p.name || "")}" placeholder="Team handbook / research Q3 / …"></label>
      <label><span>Description</span><input id="ws-f-desc" type="text" value="${escapeAttr(p.description || "")}" placeholder="What this workspace is for"></label>
      <label><span>System prompt (optional)</span><textarea id="ws-f-sys" rows="3" placeholder="Applied to every chat opened in this workspace">${escapeHtml(p.system_prompt || "")}</textarea></label>
      <div class="ws-editor-row">
        <label><span>Tier</span>
          <select id="ws-f-tier">
            <option value="">(default)</option>
            <option value="small">small</option>
            <option value="medium">medium</option>
            <option value="large">large</option>
            <option value="large-moe">large-moe</option>
          </select>
        </label>
        <label><span>Context</span>
          <select id="ws-f-ctx">
            <option value="">(default)</option>
            <option value="8192">8K</option>
            <option value="16384">16K</option>
            <option value="32768">32K</option>
            <option value="65536">64K</option>
            <option value="131072">128K</option>
            <option value="262144">256K</option>
          </select>
        </label>
        <label><span>Caveman</span>
          <select id="ws-f-cave">
            <option value="">(default)</option>
            <option value="off">off</option>
            <option value="lite">lite</option>
            <option value="full">full</option>
            <option value="ultra">ultra</option>
          </select>
        </label>
      </div>
      <div class="ws-editor-row">
        <label><span>Icon (emoji)</span><input id="ws-f-icon" type="text" value="${escapeAttr(p.icon || "")}" placeholder="📚 🧪 🔭 …" maxlength="4"></label>
        <label><span>Accent</span><input id="ws-f-color" type="color" value="${escapeAttr(p.color || "#3b82f6")}"></label>
      </div>
      <div class="ws-editor-actions">
        <button data-act="cancel">Cancel</button>
        <button data-act="save" class="primary">${isNew ? "Create" : "Save"}</button>
      </div>
    `;
    backdrop.appendChild(dlg);
    document.body.appendChild(backdrop);
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) backdrop.remove();
    });

    // Pre-fill select values (assignable after render).
    const set = (id, v) => { const el = dlg.querySelector(id); if (el && v != null) el.value = String(v); };
    set("#ws-f-tier", p.tier);
    set("#ws-f-ctx",  p.context_window);
    set("#ws-f-cave", p.caveman_level);

    dlg.querySelector('[data-act="cancel"]').addEventListener("click", () => backdrop.remove());
    dlg.querySelector('[data-act="save"]').addEventListener("click", async () => {
      const body = {
        id:              p.id || undefined,
        name:            dlg.querySelector("#ws-f-name").value.trim() || "Untitled",
        description:     dlg.querySelector("#ws-f-desc").value.trim(),
        system_prompt:   dlg.querySelector("#ws-f-sys").value.trim(),
        tier:            dlg.querySelector("#ws-f-tier").value || null,
        context_window:  parseInt(dlg.querySelector("#ws-f-ctx").value, 10) || null,
        caveman_level:   dlg.querySelector("#ws-f-cave").value || null,
        icon:            dlg.querySelector("#ws-f-icon").value.trim(),
        color:           dlg.querySelector("#ws-f-color").value || "#3b82f6",
        files:           p.files || [],
        pinned_prompts:  p.pinned_prompts || [],
      };
      const res = await fetch("/ui/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        backdrop.remove();
        load(grid);
      }
    });
  }

  function humanCtx(n) {
    if (n >= 1024) {
      const k = n / 1024;
      return (Number.isInteger(k) ? k : k.toFixed(1)) + "K";
    }
    return String(n);
  }
  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }
  function escapeAttr(s) { return escapeHtml(s); }
})();
