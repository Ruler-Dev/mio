// view_flow.js — Flow Mode (visual agent builder).
//
// A Drawflow-based graph editor where nodes are llm_call / skill_call /
// http_fetch / if_else / iterate / user_input / output. Graphs persist
// as JSON under ~/.mio/flows/<id>.json via /ui/api/flows.
//
// This is the editor only — the server-side runner lands in the next
// iteration. For now the "Run" button stubs with a placeholder message.

(function () {
  window.Mio = window.Mio || {};
  const ready = () => {
    if (!window.Mio.views) return setTimeout(ready, 50);
    window.Mio.views.register("flow", {
      title: "Flow",
      mount(host) { renderRoot(host); },
    });
  };
  ready();

  const STATE = { currentId: null, currentName: "", editor: null };

  const NODE_TYPES = [
    { type: "llm_call",   label: "LLM call",   color: "#6366f1", io: ["in","out"],
      desc: "Chat completion on the loaded model" },
    { type: "skill_call", label: "Skill",      color: "#0ea5e9", io: ["in","out"],
      desc: "Run any Mio skill (web_search, generate_pdf_report, etc.)" },
    { type: "http_fetch", label: "HTTP fetch", color: "#64748b", io: ["in","out"],
      desc: "GET/POST any URL" },
    { type: "if_else",    label: "If / Else",  color: "#f59e0b", io: ["in","true","false"],
      desc: "Route on a boolean expression" },
    { type: "iterate",    label: "Iterate",    color: "#10b981", io: ["in","out"],
      desc: "Run the downstream subgraph for each item in a list" },
    { type: "user_input", label: "User input", color: "#a855f7", io: ["out"],
      desc: "Pause for user response" },
    { type: "output",     label: "Output",     color: "#ec4899", io: ["in"],
      desc: "Surface as chat message or artifact" },
  ];

  async function ensureDrawflow() {
    if (window.Drawflow) return window.Drawflow;
    await Promise.all([
      new Promise((res, rej) => {
        const s = document.createElement("script");
        s.src = "https://cdn.jsdelivr.net/npm/drawflow@0.0.59/dist/drawflow.min.js";
        s.onload = res; s.onerror = rej;
        document.head.appendChild(s);
      }),
      new Promise((res) => {
        const l = document.createElement("link");
        l.rel = "stylesheet";
        l.href = "https://cdn.jsdelivr.net/npm/drawflow@0.0.59/dist/drawflow.min.css";
        l.onload = res;
        document.head.appendChild(l);
      }),
    ]);
    return window.Drawflow;
  }

  async function renderRoot(host) {
    host.innerHTML = `
      <div class="view-flow">
        <header class="view-header">
          <div>
            <h1>Flow <span id="flow-name-display" class="muted" style="font-family:var(--font-mono);font-size:13px"></span></h1>
            <p class="muted">Visual agent graphs — save, run, expose as a skill.</p>
          </div>
          <div class="view-header-actions">
            <button class="btn-ghost" data-action="new">New</button>
            <button class="btn-ghost" data-action="open">Open…</button>
            <button class="btn-ghost" data-action="save">Save</button>
            <button class="btn-ghost" data-action="run" style="background:var(--accent);color:#fff;border-color:var(--accent)">▶ Run</button>
          </div>
        </header>
        <div class="flow-split">
          <aside class="flow-palette">
            <header><strong>Nodes</strong></header>
            <div class="flow-nodes" id="flow-nodes"></div>
            <footer class="muted" style="font-size:10px;padding:10px 12px">
              Drag nodes onto the canvas. Connect output ports to input ports.
            </footer>
          </aside>
          <main class="flow-canvas-wrap">
            <div id="flow-canvas" class="flow-canvas"></div>
          </main>
        </div>
      </div>
    `;
    await ensureDrawflow();
    const id = "flow-canvas";
    const container = host.querySelector("#" + id);
    const editor = new window.Drawflow(container);
    editor.reroute = true;
    editor.reroute_fix_curvature = true;
    editor.start();
    STATE.editor = editor;

    renderPalette(host);
    wireHeaderActions(host);
    refreshName(host);
    bindDragToCanvas(host, editor);
  }

  function refreshName(host) {
    const el = host.querySelector("#flow-name-display");
    if (!el) return;
    el.textContent = STATE.currentName ? "· " + STATE.currentName : "· untitled";
  }

  function renderPalette(host) {
    const wrap = host.querySelector("#flow-nodes");
    wrap.innerHTML = "";
    for (const n of NODE_TYPES) {
      const card = document.createElement("div");
      card.className = "flow-node-card";
      card.draggable = true;
      card.style.borderLeft = "3px solid " + n.color;
      card.innerHTML = `
        <div class="flow-node-card-title">${n.label}</div>
        <div class="flow-node-card-desc">${n.desc}</div>
      `;
      card.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("text/plain", n.type);
        e.dataTransfer.effectAllowed = "copy";
      });
      wrap.appendChild(card);
    }
  }

  function bindDragToCanvas(host, editor) {
    const canvas = host.querySelector(".flow-canvas");
    canvas.addEventListener("dragover", (e) => { e.preventDefault(); });
    canvas.addEventListener("drop", (e) => {
      e.preventDefault();
      const type = e.dataTransfer.getData("text/plain");
      if (!type) return;
      const def = NODE_TYPES.find((n) => n.type === type);
      if (!def) return;
      // Convert drop coord to editor-space
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left + canvas.scrollLeft - (editor.precanvas?.getBoundingClientRect().left - rect.left || 0);
      const y = e.clientY - rect.top  + canvas.scrollTop  - (editor.precanvas?.getBoundingClientRect().top  - rect.top  || 0);
      const inputs = def.io.includes("in") ? 1 : 0;
      const outputs = def.io.filter((p) => p !== "in").length;
      const data = defaultData(def.type);
      const html = `
        <div style="border-left:3px solid ${def.color};padding-left:6px">
          <div style="font-weight:500;font-size:11px">${def.label}</div>
          <div style="font-size:10px;color:#888" class="df-label">${escapeHtml(data._hint || "")}</div>
        </div>
      `;
      editor.addNode(def.type, inputs, outputs, x, y, def.type, data, html);
    });
  }

  function defaultData(type) {
    if (type === "llm_call")   return { prompt: "Hello {{input}}", system: "", _hint: "prompt…" };
    if (type === "skill_call") return { skill: "web_search", args: "{\"query\": \"{{input}}\"}", _hint: "web_search" };
    if (type === "http_fetch") return { method: "GET", url: "https://example.com", _hint: "GET example.com" };
    if (type === "if_else")    return { expr: "value == true", _hint: "value == true" };
    if (type === "iterate")    return { list_expr: "{{input}}", _hint: "over {{input}}" };
    if (type === "user_input") return { label: "Enter value", _hint: "prompt user" };
    if (type === "output")     return { mode: "chat", _hint: "→ chat" };
    return {};
  }

  function wireHeaderActions(host) {
    host.querySelector('[data-action="new"]').addEventListener("click", () => {
      if (!confirm("Discard current graph and start a new flow?")) return;
      STATE.editor.clearModuleSelected();
      STATE.currentId = null;
      STATE.currentName = "";
      refreshName(host);
    });
    host.querySelector('[data-action="save"]').addEventListener("click", async () => {
      const name = STATE.currentName || prompt("Name this flow:", "my-flow");
      if (!name) return;
      STATE.currentName = name;
      const raw = STATE.editor.export();
      const body = {
        id:    STATE.currentId || undefined,
        name,
        nodes: raw?.drawflow?.Home?.data || {},
        edges: [],
      };
      const r = await fetch("/ui/api/flows", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (data.ok) {
        STATE.currentId = data.id;
        refreshName(host);
        flash(host, "Saved.");
      } else {
        alert("Save failed: " + (data.error || "unknown"));
      }
    });
    host.querySelector('[data-action="open"]').addEventListener("click", async () => {
      const r = await fetch("/ui/api/flows");
      const { flows = [] } = await r.json();
      if (!flows.length) { alert("No saved flows yet."); return; }
      const pick = prompt("Open which?\n\n" + flows.map((f, i) => `${i+1}. ${f.name}  (${f.nodes} nodes)`).join("\n"));
      const idx = parseInt(pick, 10) - 1;
      if (!Number.isFinite(idx) || idx < 0 || idx >= flows.length) return;
      const f = flows[idx];
      const data = await fetch("/ui/api/flows/" + encodeURIComponent(f.id)).then((r) => r.json());
      // Re-inflate
      STATE.editor.clear();
      const raw = { drawflow: { Home: { data: data.nodes || {} } } };
      STATE.editor.import(raw);
      STATE.currentId = f.id;
      STATE.currentName = data.name || f.name;
      refreshName(host);
    });
    host.querySelector('[data-action="run"]').addEventListener("click", () => runFlow(host));
  }

  async function runFlow(host) {
    if (!STATE.currentId) {
      alert("Save the flow first.");
      return;
    }
    // Clear any prior status overlays
    host.querySelectorAll(".flow-node-status").forEach((n) => n.remove());
    const drawer = ensureDrawer(host);
    drawer.log.innerHTML = "";
    drawer.appendEvent({ type: "starting" });
    const r = await fetch(`/ui/api/flows/${STATE.currentId}/run`, { method: "POST" });
    const { run_id, error } = await r.json();
    if (error) { drawer.appendEvent({ type: "error", error }); return; }
    const es = new EventSource(`/ui/api/flows/runs/${run_id}/events`);
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        drawer.appendEvent(data);
        markNodeStatus(host, data);
        if (data.type === "run_finished") es.close();
      } catch {}
    };
    es.onerror = () => es.close();
  }

  function markNodeStatus(host, evt) {
    if (!evt.node_id) return;
    const nodeEl = host.querySelector(`[id="node-${evt.node_id}"]`);
    if (!nodeEl) return;
    let dot = nodeEl.querySelector(".flow-node-status");
    if (!dot) {
      dot = document.createElement("div");
      dot.className = "flow-node-status";
      nodeEl.appendChild(dot);
    }
    dot.className = "flow-node-status " + (evt.type || "");
  }

  function ensureDrawer(host) {
    let drawer = host.querySelector(".flow-drawer");
    if (drawer) return drawer._api;
    drawer = document.createElement("div");
    drawer.className = "flow-drawer";
    drawer.innerHTML = `
      <header>
        <strong>Run log</strong>
        <button class="flow-drawer-close" aria-label="Close">×</button>
      </header>
      <div class="flow-drawer-log"></div>
    `;
    host.querySelector(".view-flow").appendChild(drawer);
    drawer.querySelector(".flow-drawer-close").addEventListener("click", () => drawer.remove());
    const log = drawer.querySelector(".flow-drawer-log");
    drawer._api = {
      log,
      appendEvent(evt) {
        const ln = document.createElement("div");
        ln.className = "flow-drawer-line " + (evt.type || "");
        const icon = {
          starting:        "⏳",
          run_started:     "🟢",
          node_started:    "▶",
          node_finished:   "✓",
          node_error:      "✗",
          run_finished:    "✅",
          error:           "✗",
        }[evt.type] || "·";
        const body = evt.output !== undefined
          ? (typeof evt.output === "string" ? evt.output : JSON.stringify(evt.output))
          : (evt.error || evt.class || evt.node_order?.join(" → ") || "");
        ln.textContent = `${icon} ${(evt.node_id ? (evt.node_id + " · ") : "")}${evt.class || ""} ${body || ""}`.trim();
        log.appendChild(ln);
        log.scrollTop = log.scrollHeight;
      },
    };
    return drawer._api;
  }

  function flash(host, msg) {
    const t = document.createElement("div");
    t.className = "flow-toast";
    t.textContent = msg;
    host.appendChild(t);
    setTimeout(() => t.remove(), 1600);
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }
})();
