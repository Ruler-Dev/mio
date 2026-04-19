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

  const TEMPLATES = [
    {
      icon: "🔬", color: "#0ea5e9",
      name: "Research Assistant",
      description: "Summarise papers, synthesise findings, cite sources.",
      system_prompt: "You are a research assistant. For every question: (1) think step-by-step, (2) cite sources when answering from retrieved context, (3) flag uncertainty explicitly, (4) offer one follow-up question. Prefer concise prose over bullet dumps.",
      tier: "large-moe", context_window: 131072, caveman_level: "lite",
    },
    {
      icon: "🛠", color: "#10b981",
      name: "Coding Agent",
      description: "Plan, write, test. Respects your project's conventions.",
      system_prompt: "You are a coding agent. Before writing code, restate the task in one line. Make minimal changes, match surrounding style, add tests where they exist. When uncertain, ask ONE clarifying question before editing.",
      tier: "large-moe", context_window: 131072, caveman_level: "full",
    },
    {
      icon: "🎨", color: "#ec4899",
      name: "UI Designer",
      description: "Iterate on interfaces. Tailwind-first, components clean.",
      system_prompt: "You are a UI/UX designer. Default to Tailwind CSS + React. Produce self-contained HTML artifacts. Favour restraint: 2 type sizes, 1 accent, generous whitespace. When asked to change something, change only that — don't redo the whole page.",
      tier: "large-moe", context_window: 32768, caveman_level: "full",
    },
    {
      icon: "✍️", color: "#a855f7",
      name: "Writing Editor",
      description: "Tighten prose. Preserve voice. Flag weak sentences.",
      system_prompt: "You are a copy editor. Preserve the author's voice. Your priorities: clarity, rhythm, economy — in that order. When editing, show the revised passage and list the specific changes you made and why. Don't rewrite — edit.",
      tier: "medium", context_window: 32768, caveman_level: "off",
    },
    {
      icon: "📓", color: "#f59e0b",
      name: "Daily Journal",
      description: "Rubber-duck + gentle self-reflection partner.",
      system_prompt: "You are a journaling partner. Match the user's register — if they're casual, be casual. Ask open-ended questions, never give unsolicited advice. Reflect feelings back. Close each exchange with a single sentence that summarises what the user said.",
      tier: "medium", context_window: 32768, caveman_level: "off",
    },
    {
      icon: "🎓", color: "#6366f1",
      name: "Study Buddy",
      description: "Explain concepts, quiz back, adjust to confusion.",
      system_prompt: "You are a patient tutor. When asked to explain: (1) give an intuition in one sentence, (2) a concrete example, (3) a common misconception to avoid. After each concept, ask ONE question to check understanding. Adjust depth based on the answer.",
      tier: "medium", context_window: 32768, caveman_level: "lite",
    },
  ];

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
      // Empty state: show starter templates the user can one-click create.
      if (!projects.length) {
        grid.appendChild(emptyStateHeader());
        for (const t of TEMPLATES) grid.appendChild(templateCard(t, grid));
        grid.appendChild(newCard(grid));
        return;
      }
      for (const p of projects) grid.appendChild(card(p, countByPid[p.id] || 0, grid));
      grid.appendChild(newCard(grid));
    } catch (e) {
      grid.innerHTML = `<div class="muted">Failed to load: ${escapeHtml(String(e))}</div>`;
    }
  }

  function emptyStateHeader() {
    const el = document.createElement("div");
    el.className = "ws-empty";
    el.innerHTML = `
      <h2>Start with a template</h2>
      <p>Pick one to scaffold a workspace. You can edit everything after.</p>
    `;
    return el;
  }

  function templateCard(tmpl, grid) {
    const el = document.createElement("div");
    el.className = "ws-card ws-card-template";
    el.style.setProperty("--ws-accent", tmpl.color);
    el.innerHTML = `
      <div class="ws-card-stripe"></div>
      <div class="ws-card-body">
        <div class="ws-card-icon">${escapeHtml(tmpl.icon)}</div>
        <div class="ws-card-title">${escapeHtml(tmpl.name)}</div>
        <div class="ws-card-desc">${escapeHtml(tmpl.description)}</div>
        <div class="ws-card-meta">
          <span>${escapeHtml(tmpl.tier)}</span>
          <span>•</span>
          <span>${humanCtx(tmpl.context_window)}</span>
          <span>•</span>
          <span>caveman ${escapeHtml(tmpl.caveman_level)}</span>
        </div>
      </div>
      <div class="ws-card-actions">
        <button data-act="use" style="flex:1;background:var(--ws-accent);color:#fff;border-color:var(--ws-accent)">Use this template</button>
      </div>
    `;
    el.querySelector('[data-act="use"]').addEventListener("click", async () => {
      const res = await fetch("/ui/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: tmpl.name, description: tmpl.description,
          system_prompt: tmpl.system_prompt, icon: tmpl.icon, color: tmpl.color,
          tier: tmpl.tier, context_window: tmpl.context_window,
          caveman_level: tmpl.caveman_level, files: [], pinned_prompts: [],
        }),
      });
      if (res.ok) load(grid);
    });
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
