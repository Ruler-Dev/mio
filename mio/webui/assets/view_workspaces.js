// view_workspaces.js — Workspaces view.
//
// A workspace = reusable bundle of (name, description, project-scoped system
// prompt/files, tier, minimum context capacity, prompt policy). Backed by
// /ui/api/projects. Activation is a checked transaction: unsupported legacy
// fields are reported, never silently treated as active.
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
                <p class="muted">Project context, model tier and prompt policy — activated as one checked profile.</p>
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

  const PROMPT_MODES = ["none", "caveman", "ponytail"];
  const PROMPT_LEVELS = ["lite", "full", "ultra"];

  function workspacePromptPolicy(workspace) {
    const modernMode = PROMPT_MODES.includes(workspace?.prompt_mode)
      ? workspace.prompt_mode
      : null;
    let mode = modernMode;
    let level = PROMPT_LEVELS.includes(workspace?.prompt_level)
      ? workspace.prompt_level
      : null;

    // Projects created before PromptPolicy stored Caveman in one field.
    // Read that shape without writing it back on the next save.
    if (!mode) {
      const legacy = String(workspace?.caveman_level ?? "").toLowerCase();
      if (legacy === "off") mode = "none";
      else if (PROMPT_LEVELS.includes(legacy)) {
        mode = "caveman";
        level = legacy;
      }
    }
    if (!mode) return null;
    if (mode === "none") return { prompt_mode: "none", prompt_level: null };
    return { prompt_mode: mode, prompt_level: level || "full" };
  }

  function workspacePolicyValue(workspace) {
    const policy = workspacePromptPolicy(workspace);
    if (!policy) return "";
    return policy.prompt_mode === "none"
      ? "none"
      : `${policy.prompt_mode}/${policy.prompt_level}`;
  }

  function workspacePolicyLabel(workspace) {
    const policy = workspacePromptPolicy(workspace);
    if (!policy) return "prompt inherited";
    if (policy.prompt_mode === "none") return "prompt none";
    const mode = policy.prompt_mode === "ponytail" ? "Ponytail" : "Caveman";
    const level = policy.prompt_level.charAt(0).toUpperCase() + policy.prompt_level.slice(1);
    return `${mode} ${level}`;
  }

  function policyFromEditorValue(value) {
    if (!value) return null;
    if (value === "none") return { prompt_mode: "none", prompt_level: null };
    const match = /^(caveman|ponytail)\/(lite|full|ultra)$/.exec(value);
    return match ? { prompt_mode: match[1], prompt_level: match[2] } : null;
  }

  const TEMPLATES = [
    {
      icon: "🔬", color: "#0ea5e9",
      name: "Research Assistant",
      description: "Summarise papers, synthesise findings, cite sources.",
      system_prompt: "You are a research assistant. For every question: (1) think step-by-step, (2) cite sources when answering from retrieved context, (3) flag uncertainty explicitly, (4) offer one follow-up question. Prefer concise prose over bullet dumps.",
      tier: "large-moe", context_window: 131072,
      prompt_mode: "caveman", prompt_level: "lite",
    },
    {
      icon: "🛠", color: "#10b981",
      name: "Coding Agent",
      description: "Plan, write, test. Respects your project's conventions.",
      system_prompt: "You are a coding agent. Before writing code, restate the task in one line. Make minimal changes, match surrounding style, add tests where they exist. When uncertain, ask ONE clarifying question before editing.",
      tier: "large-moe", context_window: 131072,
      prompt_mode: "ponytail", prompt_level: "full",
    },
    {
      icon: "🎨", color: "#ec4899",
      name: "UI Designer",
      description: "Iterate on interfaces. Tailwind-first, components clean.",
      system_prompt: "You are a UI/UX designer. Default to Tailwind CSS + React. Produce self-contained HTML artifacts. Favour restraint: 2 type sizes, 1 accent, generous whitespace. When asked to change something, change only that — don't redo the whole page.",
      tier: "large-moe", context_window: 32768,
      prompt_mode: "ponytail", prompt_level: "lite",
    },
    {
      icon: "✍️", color: "#a855f7",
      name: "Writing Editor",
      description: "Tighten prose. Preserve voice. Flag weak sentences.",
      system_prompt: "You are a copy editor. Preserve the author's voice. Your priorities: clarity, rhythm, economy — in that order. When editing, show the revised passage and list the specific changes you made and why. Don't rewrite — edit.",
      tier: "medium", context_window: 16384,
      prompt_mode: "none", prompt_level: null,
    },
    {
      icon: "📓", color: "#f59e0b",
      name: "Daily Journal",
      description: "Rubber-duck + gentle self-reflection partner.",
      system_prompt: "You are a journaling partner. Match the user's register — if they're casual, be casual. Ask open-ended questions, never give unsolicited advice. Reflect feelings back. Close each exchange with a single sentence that summarises what the user said.",
      tier: "medium", context_window: 16384,
      prompt_mode: "none", prompt_level: null,
    },
    {
      icon: "🎓", color: "#6366f1",
      name: "Study Buddy",
      description: "Explain concepts, quiz back, adjust to confusion.",
      system_prompt: "You are a patient tutor. When asked to explain: (1) give an intuition in one sentence, (2) a concrete example, (3) a common misconception to avoid. After each concept, ask ONE question to check understanding. Adjust depth based on the answer.",
      tier: "medium", context_window: 16384,
      prompt_mode: "caveman", prompt_level: "lite",
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
          <span>${humanCtx(tmpl.context_window)} min ctx</span>
          <span>•</span>
          <span>${escapeHtml(workspacePolicyLabel(tmpl))}</span>
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
          prompt_mode: tmpl.prompt_mode, prompt_level: tmpl.prompt_level, files: [],
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
          <span>${p.context_window ? humanCtx(p.context_window) + " min ctx" : "tier context"}</span>
          <span>•</span>
          <span>${escapeHtml(workspacePolicyLabel(p))}</span>
          <span>•</span>
          <span>${chatCount} chat${chatCount === 1 ? "" : "s"}</span>
        </div>
      </div>
      <div class="ws-card-actions">
        <button data-act="open">Open chat</button>
        <button data-act="edit">Edit</button>
        <button data-act="delete" class="danger">Delete</button>
      </div>
      <div class="ws-activation-feedback muted" role="status" aria-live="polite" hidden></div>
    `;
    const openButton = el.querySelector('[data-act="open"]');
    openButton.addEventListener("click", () => openInChat(p, openButton, el));
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

  async function openInChat(p, button, cardElement) {
    const feedback = cardElement?.querySelector(".ws-activation-feedback");
    const showFeedback = (message, isError) => {
      if (!feedback) return;
      feedback.hidden = false;
      feedback.textContent = message;
      feedback.style.color = isError ? "var(--danger, #ef4444)" : "";
    };
    if (typeof window.setActiveProject !== "function") {
      const message = "Workspace not activated: chat project selection is unavailable.";
      showFeedback(message, true);
      if (window.toast) window.toast(message, 6000);
      return;
    }

    const originalLabel = button?.textContent || "Open chat";
    if (button) {
      button.disabled = true;
      button.textContent = "Activating…";
    }
    showFeedback("Validating model, context and prompt policy…", false);

    let runtimeCommitted = false;
    try {
      const res = await fetch(`/ui/api/projects/${encodeURIComponent(p.id)}/activate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      let activation = {};
      try { activation = await res.json(); } catch (_) {}
      if (!res.ok || !activation.ok) {
        throw new Error(activation.detail || activation.error || `HTTP ${res.status}`);
      }
      runtimeCommitted = true;

      // The backend has committed every supported runtime change. Only now
      // publish the browser-local project id that is sent with chat requests.
      if (typeof window.loadProjects === "function") await window.loadProjects();
      await window.setActiveProject(p.id);
      const select = document.getElementById("projectSelect");
      if (select) select.value = p.id;
      if (typeof window.loadConfig === "function") await window.loadConfig();

      const runtime = activation.runtime || {};
      const summaryParts = [`Workspace active: ${activation.workspace?.name || p.name || "Untitled"}`];
      if (runtime.tier) summaryParts.push(`tier ${runtime.tier}`);
      if (runtime.prompt_policy) {
        const policyLabel = workspacePolicyLabel({ prompt_mode: runtime.prompt_policy.split("/")[0], prompt_level: runtime.prompt_policy.split("/")[1] });
        summaryParts.push(
          runtime.prompt_policy_pinned ? policyLabel : `${policyLabel} (inherited)`
        );
      }
      if (activation.context_requirement) {
        summaryParts.push(
          `${humanCtx(activation.context_requirement.requested)} context verified`
        );
      }
      const summary = summaryParts.join(" · ");
      showFeedback(summary, false);

      const warnings = Array.isArray(activation.warnings)
        ? activation.warnings.map((item) => item?.message).filter(Boolean)
        : [];
      if (window.Mio?.views?.switch) window.Mio.views.switch("chat");
      if (warnings.length) {
        const warning = `Workspace active with limitations: ${warnings.join("; ")}`;
        if (window.appendSystemMessage) window.appendSystemMessage(warning);
        if (window.toast) window.toast(warning, 9000);
      } else if (window.toast) {
        window.toast(summary, 5000);
      }
    } catch (error) {
      const prefix = runtimeCommitted
        ? "Runtime profile applied, but chat selection failed"
        : "Workspace activation failed";
      const message = `${prefix}: ${error?.message || String(error)}`;
      showFeedback(message, true);
      if (window.toast) window.toast(message, 7000);
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = originalLabel;
      }
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
      <label><span>System prompt (optional)</span><textarea id="ws-f-sys" rows="3" placeholder="Added to requests sent with this workspace">${escapeHtml(p.system_prompt || "")}</textarea></label>
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
        <label><span>Minimum context</span>
          <select id="ws-f-ctx">
            <option value="">(default)</option>
            <option value="8192">8K</option>
            <option value="16384">16K</option>
            <option value="32768">32K</option>
            <option value="65536">64K</option>
            <option value="131072">128K</option>
            <option value="262144">256K</option>
          </select>
          <small class="muted">Verified against tier capacity; this does not resize a loaded model.</small>
        </label>
        <label><span>Prompt policy</span>
          <select id="ws-f-policy" aria-label="Workspace prompt policy">
            <option value="">(inherit current)</option>
            <option value="none">None</option>
            <optgroup label="Caveman">
              <option value="caveman/lite">Caveman · Lite</option>
              <option value="caveman/full">Caveman · Full</option>
              <option value="caveman/ultra">Caveman · Ultra</option>
            </optgroup>
            <optgroup label="Ponytail">
              <option value="ponytail/lite">Ponytail · Lite</option>
              <option value="ponytail/full">Ponytail · Full</option>
              <option value="ponytail/ultra">Ponytail · Ultra</option>
            </optgroup>
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
    set("#ws-f-policy", workspacePolicyValue(p));

    dlg.querySelector('[data-act="cancel"]').addEventListener("click", () => backdrop.remove());
    dlg.querySelector('[data-act="save"]').addEventListener("click", async () => {
      const promptPolicy = policyFromEditorValue(dlg.querySelector("#ws-f-policy").value);
      const body = {
        id:              p.id || undefined,
        name:            dlg.querySelector("#ws-f-name").value.trim() || "Untitled",
        description:     dlg.querySelector("#ws-f-desc").value.trim(),
        system_prompt:   dlg.querySelector("#ws-f-sys").value.trim(),
        tier:            dlg.querySelector("#ws-f-tier").value || null,
        context_window:  parseInt(dlg.querySelector("#ws-f-ctx").value, 10) || null,
        prompt_mode:     promptPolicy?.prompt_mode || null,
        prompt_level:    promptPolicy?.prompt_mode === "none" ? null : promptPolicy?.prompt_level || null,
        icon:            dlg.querySelector("#ws-f-icon").value.trim(),
        color:           dlg.querySelector("#ws-f-color").value || "#3b82f6",
        files:           p.files || [],
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
