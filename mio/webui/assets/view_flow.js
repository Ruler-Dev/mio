// view_flow.js — Flow Mode (visual agent builder).
//
// A Drawflow-based graph editor where nodes are llm_call / skill_call /
// http_fetch / if_else / iterate / user_input / output. Graphs persist
// as JSON under ~/.mio/flows/<id>.json via /ui/api/flows.
//
// The server-side DAG runner streams node state back over SSE. User-input
// nodes are collected before launch and supplied in the run environment.

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

  const STATE = {
    currentId: null,
    currentName: "",
    editor: null,
    selectedNodeId: null,
    exposed: false,
    skillName: "",
  };

  const NODE_TYPES = [
    // --- Core (existing) ---
    { type: "llm_call",      label: "LLM call",       color: "#6366f1", io: ["in","out"],
      desc: "Chat completion on the loaded model" },
    { type: "skill_call",    label: "Skill",          color: "#0ea5e9", io: ["in","out"],
      desc: "Run any Mio skill (web_search, generate_pdf_report, …)" },
    { type: "http_fetch",    label: "HTTP fetch",     color: "#64748b", io: ["in","out"],
      desc: "GET/POST any URL" },
    { type: "if_else",       label: "If / Else",      color: "#f59e0b", io: ["in","true","false"],
      desc: "Route on a boolean expression" },
    { type: "iterate",       label: "Iterate / map",  color: "#10b981", io: ["in","out"],
      desc: "Normalize a list and optionally map a template over each item" },
    { type: "user_input",    label: "User input",     color: "#a855f7", io: ["out"],
      desc: "Pause for user response" },
    { type: "output",        label: "Output",         color: "#ec4899", io: ["in"],
      desc: "Return the final value in the Flow run result" },

    // --- New: data shaping ---
    { type: "constant",      label: "Constant",       color: "#94a3b8", io: ["out"],
      desc: "Static value / seed string. Great start node." },
    { type: "template",      label: "Template",       color: "#60a5fa", io: ["in","out"],
      desc: "Mustache-style {{input}} / {{n.out}} / {{env.X}} interpolation" },
    { type: "parse_json",    label: "Parse JSON",     color: "#22d3ee", io: ["in","out"],
      desc: "JSON.parse the input" },
    { type: "to_json",       label: "To JSON",        color: "#22d3ee", io: ["in","out"],
      desc: "JSON.stringify the input (pretty)" },
    { type: "regex_extract", label: "Regex extract",  color: "#fb923c", io: ["in","out"],
      desc: "First capture group of a regex run over the input" },
    { type: "split",         label: "Split",          color: "#fbbf24", io: ["in","out"],
      desc: "Split a string by delimiter → list" },
    { type: "join",          label: "Join",           color: "#fbbf24", io: ["in","out"],
      desc: "Join a list by delimiter → string" },

    // --- New: memory + time ---
    { type: "mem_get",       label: "Memory get",     color: "#14b8a6", io: ["in","out"],
      desc: "Read Mio's persistent Flow memory by key" },
    { type: "mem_set",       label: "Memory set",     color: "#14b8a6", io: ["in","out"],
      desc: "Write a value under a key (persistent across runs)" },
    { type: "delay",         label: "Delay",          color: "#d946ef", io: ["in","out"],
      desc: "Wait N ms, then pass input through" },
    { type: "clock",         label: "Clock",          color: "#d946ef", io: ["out"],
      desc: "Emit the current ISO timestamp" },
    { type: "random",        label: "Random pick",    color: "#f43f5e", io: ["in","out"],
      desc: "Pick a random item from a list input" },

    // --- New: knowledge + surface ---
    { type: "rag_search",    label: "RAG search",     color: "#84cc16", io: ["in","out"],
      desc: "Full-text search across indexed folders + clipped docs" },
    { type: "artifact_emit", label: "Emit artifact",  color: "#ec4899", io: ["in","out"],
      desc: "Add the input to Mio's artifact gallery and open its panel" },
  ];

  // Every node type has an explicit inspector definition. Empty definitions
  // are intentional: those nodes are pure transforms with no parameters.
  const NODE_FIELDS = {
    llm_call: [
      { key: "prompt", label: "Prompt", kind: "textarea", rows: 5 },
      { key: "system", label: "System prompt", kind: "textarea", rows: 4 },
      { key: "tier", label: "Model tier", kind: "text", placeholder: "first loaded tier" },
      { key: "temperature", label: "Temperature", kind: "number", min: 0, max: 2, step: 0.05 },
      { key: "max_tokens", label: "Max tokens", kind: "number", min: 1, max: 32768, step: 1 },
    ],
    skill_call: [
      { key: "skill", label: "Mio skill", kind: "text" },
      { key: "args", label: "Arguments (JSON)", kind: "textarea", rows: 6 },
    ],
    http_fetch: [
      { key: "method", label: "Method", kind: "select", options: ["GET", "POST"] },
      { key: "url", label: "URL", kind: "text" },
      { key: "body", label: "POST body", kind: "textarea", rows: 5 },
    ],
    if_else: [{ key: "expr", label: "Expression", kind: "text" }],
    iterate: [
      { key: "list_expr", label: "List expression", kind: "textarea", rows: 3 },
      { key: "template", label: "Per-item template", kind: "textarea", rows: 4 },
      { key: "parse_json", label: "Parse mapped JSON", kind: "checkbox" },
    ],
    user_input: [
      { key: "label", label: "Prompt label", kind: "text" },
      { key: "key", label: "Stable input key", kind: "text" },
      { key: "default", label: "Default value", kind: "text", removeWhenEmpty: true },
    ],
    output: [],
    constant: [{ key: "value", label: "Value", kind: "textarea", rows: 5 }],
    template: [{ key: "template", label: "Template", kind: "textarea", rows: 6 }],
    parse_json: [],
    to_json: [{ key: "indent", label: "Indent", kind: "number", min: 0, max: 8, step: 1 }],
    regex_extract: [
      { key: "pattern", label: "Pattern", kind: "textarea", rows: 3 },
      { key: "flags", label: "Flags (i, m, s)", kind: "text" },
    ],
    split: [{ key: "delim", label: "Delimiter", kind: "text" }],
    join: [{ key: "delim", label: "Delimiter", kind: "text" }],
    mem_get: [{ key: "key", label: "Memory key", kind: "text" }],
    mem_set: [{ key: "key", label: "Memory key", kind: "text" }],
    delay: [{ key: "ms", label: "Milliseconds", kind: "number", min: 0, max: 300000, step: 100 }],
    clock: [],
    random: [],
    rag_search: [
      { key: "query", label: "Query", kind: "textarea", rows: 3 },
      { key: "limit", label: "Result limit", kind: "number", min: 1, max: 50, step: 1 },
    ],
    artifact_emit: [
      { key: "type", label: "Artifact MIME type", kind: "text" },
      { key: "title", label: "Title", kind: "text" },
    ],
  };

  async function ensureDrawflow() {
    if (window.Drawflow) return window.Drawflow;
    await Promise.all([
      new Promise((res, rej) => {
        const s = document.createElement("script");
        s.src = "https://cdn.jsdelivr.net/npm/drawflow@0.0.59/dist/drawflow.min.js";
        s.integrity = "sha384-AwUJfl4ROgOxbeFXwuFb9a5iT0jo4xQ2irCGirX4Z1aZbaIDD10j/nY/+RVcTg5E";
        s.crossOrigin = "anonymous";
        s.onload = res; s.onerror = rej;
        document.head.appendChild(s);
      }),
      new Promise((res) => {
        const l = document.createElement("link");
        l.rel = "stylesheet";
        l.href = "https://cdn.jsdelivr.net/npm/drawflow@0.0.59/dist/drawflow.min.css";
        l.integrity = "sha384-IFh+Q6zh+LRcTjqVmAKetdGY59dT485vtvWT5DAKQy8iv5+fYWHXisHP7mFKcFqV";
        l.crossOrigin = "anonymous";
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
            <button class="btn-ghost" data-action="expose" disabled>Expose as skill</button>
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
          <aside class="flow-inspector" aria-label="Node inspector">
            <header><strong>Inspector</strong></header>
            <div class="flow-inspector-body">
              <p class="muted">Select a node to configure it.</p>
            </div>
          </aside>
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

    editor.on("nodeSelected", (nodeId) => {
      STATE.selectedNodeId = String(nodeId);
      renderInspector(host, STATE.selectedNodeId);
    });
    editor.on("nodeUnselected", () => {
      STATE.selectedNodeId = null;
      renderInspector(host, null);
    });
    editor.on("nodeRemoved", (nodeId) => {
      if (String(nodeId) === STATE.selectedNodeId) {
        STATE.selectedNodeId = null;
        renderInspector(host, null);
      }
    });

    renderPalette(host);
    wireHeaderActions(host);
    refreshName(host);
    refreshExposeButton(host);
    bindDragToCanvas(host, editor);
  }

  function refreshName(host) {
    const el = host.querySelector("#flow-name-display");
    if (!el) return;
    el.textContent = STATE.currentName ? "· " + STATE.currentName : "· untitled";
  }

  function refreshExposeButton(host) {
    const button = host.querySelector('[data-action="expose"]');
    if (!button) return;
    button.disabled = !STATE.currentId;
    button.textContent = STATE.exposed ? `Unexpose ${STATE.skillName}` : "Expose as skill";
    button.classList.toggle("active", STATE.exposed);
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
      const html = nodeHtml(def.type, data);
      editor.addNode(def.type, inputs, outputs, x, y, def.type, data, html);
    });
  }

  function defaultData(type) {
    if (type === "llm_call")      return { prompt: "Hello {{input}}", system: "", _hint: "prompt…" };
    if (type === "skill_call")    return { skill: "web_search", args: "{\"query\": \"{{input}}\"}", _hint: "web_search" };
    if (type === "http_fetch")    return { method: "GET", url: "https://example.com", _hint: "GET example.com" };
    if (type === "if_else")       return { expr: "{{input}} == true", _hint: "input == true" };
    if (type === "iterate")       return { list_expr: "{{input}}", template: "", _hint: "over {{input}}" };
    if (type === "user_input")    return { label: "Enter value", key: "", _hint: "prompt user" };
    if (type === "output")        return { _hint: "→ run result" };
    if (type === "constant")      return { value: "hello world", _hint: "\"hello world\"" };
    if (type === "template")      return { template: "You said: {{input}}", _hint: "{{input}}" };
    if (type === "parse_json")    return { _hint: "JSON → obj" };
    if (type === "to_json")       return { indent: 2, _hint: "obj → JSON" };
    if (type === "regex_extract") return { pattern: "\\b([A-Za-z]+)\\b", flags: "i", _hint: "first group" };
    if (type === "split")         return { delim: ",", _hint: "by \",\"" };
    if (type === "join")          return { delim: ", ", _hint: "with \", \"" };
    if (type === "mem_get")       return { key: "last_run", _hint: "mem[last_run]" };
    if (type === "mem_set")       return { key: "last_run", _hint: "mem[last_run] ←" };
    if (type === "delay")         return { ms: 500, _hint: "500 ms" };
    if (type === "clock")         return { _hint: "now()" };
    if (type === "random")        return { _hint: "pick one" };
    if (type === "rag_search")    return { query: "{{input}}", limit: 5, _hint: "top-5" };
    if (type === "artifact_emit") return { type: "text/html", title: "Flow output", _hint: "→ gallery + panel" };
    return {};
  }

  function hintFor(type, data) {
    const value = (key, fallback = "") => {
      const raw = data?.[key];
      if (raw === undefined || raw === null || raw === "") return fallback;
      return typeof raw === "string" ? raw : JSON.stringify(raw);
    };
    const hints = {
      llm_call: () => value("prompt", "prompt…"),
      skill_call: () => value("skill", "choose skill"),
      http_fetch: () => `${value("method", "GET")} ${value("url", "URL")}`,
      if_else: () => value("expr", "condition"),
      iterate: () => `over ${value("list_expr", "{{input}}")}`,
      user_input: () => value("label", value("key", "user input")),
      output: () => "→ run result",
      constant: () => value("value", "constant"),
      template: () => value("template", "{{input}}"),
      parse_json: () => "JSON → object",
      to_json: () => `object → JSON (${value("indent", "compact")})`,
      regex_extract: () => value("pattern", "pattern"),
      split: () => `split by ${value("delim", "separator")}`,
      join: () => `join with ${value("delim", "separator")}`,
      mem_get: () => `read ${value("key", "key")}`,
      mem_set: () => `write ${value("key", "key")}`,
      delay: () => `${value("ms", "0")} ms`,
      clock: () => "now()",
      random: () => "pick one",
      rag_search: () => value("query", "local search"),
      artifact_emit: () => `→ ${value("title", "artifact")}`,
    };
    const hint = (hints[type] || (() => type))();
    return String(hint).replace(/\s+/g, " ").trim().slice(0, 80);
  }

  function nodeHtml(type, data) {
    const def = NODE_TYPES.find((item) => item.type === type);
    const label = def?.label || type || "Unknown node";
    const color = def?.color || "#64748b";
    const hint = hintFor(type, data || {});
    return `
      <div style="border-left:3px solid ${color};padding-left:6px">
        <div style="font-weight:500;font-size:11px">${escapeHtml(label)}</div>
        <div style="font-size:10px;color:#888" class="df-label">${escapeHtml(hint)}</div>
      </div>
    `;
  }

  function normalizeImportedGraph(nodes) {
    const normalized = (nodes && typeof nodes === "object") ? nodes : {};
    for (const node of Object.values(normalized)) {
      if (!node || typeof node !== "object") continue;
      const type = String(node.class || node.name || "");
      node.data = (node.data && typeof node.data === "object") ? node.data : {};
      node.data._hint = hintFor(type, node.data);
      // Drawflow imports persisted HTML verbatim. Rebuilding it from Mio's
      // constants closes the stored-XSS boundary for edited flow JSON files.
      node.html = nodeHtml(type, node.data);
    }
    return normalized;
  }

  function renderInspector(host, nodeId) {
    const body = host.querySelector(".flow-inspector-body");
    if (!body) return;
    body.replaceChildren();
    if (!nodeId || !STATE.editor) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "Select a node to configure it.";
      body.appendChild(empty);
      return;
    }

    const node = STATE.editor.getNodeFromId(nodeId);
    if (!node) return renderInspector(host, null);
    const type = String(node.class || node.name || "");
    const def = NODE_TYPES.find((item) => item.type === type);
    const heading = document.createElement("div");
    heading.className = "flow-inspector-title";
    const title = document.createElement("strong");
    title.textContent = def?.label || type || "Unknown node";
    const identifier = document.createElement("span");
    identifier.textContent = `Node ${nodeId}`;
    heading.append(title, identifier);
    body.appendChild(heading);

    const fields = NODE_FIELDS[type];
    if (!Array.isArray(fields)) {
      const warning = document.createElement("p");
      warning.className = "muted";
      warning.textContent = "This node type is unsupported and will fail at runtime.";
      body.appendChild(warning);
      return;
    }
    if (!fields.length) {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = "This node has no configurable parameters.";
      body.appendChild(note);
      return;
    }

    const data = { ...(node.data || {}) };
    for (const field of fields) {
      const row = document.createElement("label");
      row.className = `flow-inspector-field${field.kind === "checkbox" ? " checkbox" : ""}`;
      const caption = document.createElement("span");
      caption.textContent = field.label;
      let control;
      if (field.kind === "textarea") {
        control = document.createElement("textarea");
        control.rows = field.rows || 3;
      } else if (field.kind === "select") {
        control = document.createElement("select");
        for (const optionValue of field.options || []) {
          const option = document.createElement("option");
          option.value = optionValue;
          option.textContent = optionValue;
          control.appendChild(option);
        }
      } else {
        control = document.createElement("input");
        control.type = field.kind === "checkbox" ? "checkbox" : field.kind;
      }
      control.dataset.field = field.key;
      if (field.placeholder) control.placeholder = field.placeholder;
      if (field.min !== undefined) control.min = String(field.min);
      if (field.max !== undefined) control.max = String(field.max);
      if (field.step !== undefined) control.step = String(field.step);
      if (field.kind === "checkbox") {
        control.checked = Boolean(data[field.key]);
      } else {
        const existing = data[field.key];
        control.value = existing === undefined || existing === null
          ? ""
          : (typeof existing === "string" ? existing : JSON.stringify(existing));
      }
      control.addEventListener("input", () => {
        const latest = STATE.editor.getNodeFromId(nodeId);
        if (!latest) return;
        const next = { ...(latest.data || {}) };
        if (field.removeWhenEmpty && control.value === "") {
          delete next[field.key];
        } else if (field.kind === "checkbox") {
          next[field.key] = control.checked;
        } else if (field.kind === "number") {
          const parsed = Number(control.value);
          next[field.key] = Number.isFinite(parsed) ? parsed : "";
        } else {
          next[field.key] = control.value;
        }
        next._hint = hintFor(type, next);
        STATE.editor.updateNodeDataFromId(nodeId, next);
        const label = host.querySelector(`[id="node-${CSS.escape(String(nodeId))}"] .df-label`);
        if (label) label.textContent = next._hint;
      });
      if (field.kind === "checkbox") row.append(control, caption);
      else row.append(caption, control);
      body.appendChild(row);
    }
  }

  function wireHeaderActions(host) {
    host.querySelector('[data-action="new"]').addEventListener("click", () => {
      if (!confirm("Discard current graph and start a new flow?")) return;
      STATE.editor.clearModuleSelected();
      STATE.currentId = null;
      STATE.currentName = "";
      STATE.selectedNodeId = null;
      STATE.exposed = false;
      STATE.skillName = "";
      refreshName(host);
      refreshExposeButton(host);
      renderInspector(host, null);
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
        refreshExposeButton(host);
        flash(host, "Saved.");
      } else {
        alert("Save failed: " + (data.error || "unknown"));
      }
    });
    host.querySelector('[data-action="open"]').addEventListener("click", async () => {
      const r = await fetch("/ui/api/flows");
      const { flows = [] } = await r.json();
      if (!flows.length) {
        flash(host, "No saved flows yet.");
        return;
      }
      const pick = prompt("Open which?\n\n" + flows.map((f, i) => `${i+1}. ${f.name}  (${f.nodes} nodes)`).join("\n"));
      const idx = parseInt(pick, 10) - 1;
      if (!Number.isFinite(idx) || idx < 0 || idx >= flows.length) return;
      const f = flows[idx];
      const data = await fetch("/ui/api/flows/" + encodeURIComponent(f.id)).then((r) => r.json());
      // Re-inflate
      STATE.editor.clear();
      const raw = { drawflow: { Home: { data: normalizeImportedGraph(data.nodes || {}) } } };
      STATE.editor.import(raw);
      STATE.currentId = f.id;
      STATE.currentName = data.name || f.name;
      STATE.selectedNodeId = null;
      STATE.exposed = Boolean(data.skill?.exposed);
      STATE.skillName = STATE.exposed ? String(data.skill?.name || "") : "";
      refreshName(host);
      refreshExposeButton(host);
      renderInspector(host, null);
    });
    host.querySelector('[data-action="expose"]').addEventListener("click", () => toggleExposure(host));
    host.querySelector('[data-action="run"]').addEventListener("click", () => runFlow(host));
  }

  async function toggleExposure(host) {
    if (!STATE.currentId) return;
    const endpoint = `/ui/api/flows/${encodeURIComponent(STATE.currentId)}/expose`;
    if (STATE.exposed) {
      if (!confirm(`Remove ${STATE.skillName} from Mio skills?`)) return;
      const response = await fetch(endpoint, { method: "DELETE" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || !payload.ok) {
        alert("Unexpose failed: " + (payload.detail || payload.error || response.statusText));
        return;
      }
      STATE.exposed = false;
      STATE.skillName = "";
      refreshExposeButton(host);
      flash(host, "Flow removed from Mio skills.");
      return;
    }

    const suggested = (STATE.currentName || STATE.currentId)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 60) || `flow_${STATE.currentId}`;
    const safeSuggestion = /^[a-z]/.test(suggested) ? suggested : `flow_${suggested}`;
    const name = prompt("Mio skill name (lowercase letters, digits, underscore):", safeSuggestion);
    if (!name) return;
    const description = prompt(
      "Describe when Mio should use this flow:",
      `Run the ${STATE.currentName || name} flow`,
    );
    if (description === null) return;
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) {
      alert("Expose failed: " + (payload.detail || payload.error || response.statusText));
      return;
    }
    STATE.exposed = true;
    STATE.skillName = payload.skill?.name || name;
    refreshExposeButton(host);
    flash(host, `Published as ${STATE.skillName}.`);
  }

  async function runFlow(host) {
    if (!STATE.currentId) {
      flash(host, "Save the flow first.");
      return;
    }
    // Clear any prior status overlays
    host.querySelectorAll(".flow-node-status").forEach((n) => n.remove());
    const drawer = ensureDrawer(host);
    drawer.log.innerHTML = "";
    drawer.appendEvent({ type: "starting" });

    // Resolve user_input nodes explicitly instead of silently substituting an
    // empty string. Values are keyed by both node id and an optional stable key.
    const graph = STATE.editor.export()?.drawflow?.Home?.data || {};
    const saveResponse = await fetch("/ui/api/flows", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: STATE.currentId, name: STATE.currentName, nodes: graph, edges: [] }),
    });
    if (!saveResponse.ok) {
      drawer.appendEvent({ type: "error", error: "Could not save the current graph before running" });
      return;
    }
    const userInput = {};
    for (const [nodeId, node] of Object.entries(graph)) {
      if ((node.class || node.name) !== "user_input") continue;
      const nodeData = node.data || {};
      const label = nodeData.label || nodeData.key || `Value for node ${nodeId}`;
      const value = prompt(label + ":", nodeData.default ?? "");
      if (value === null) {
        drawer.appendEvent({ type: "run_cancelled", error: "User cancelled input" });
        return;
      }
      userInput[nodeId] = value;
      if (nodeData.key) userInput[nodeData.key] = value;
    }

    const r = await fetch(`/ui/api/flows/${STATE.currentId}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ env: { user_input: userInput } }),
    });
    const { run_id, error } = await r.json();
    if (error) { drawer.appendEvent({ type: "error", error }); return; }
    const es = new EventSource(`/ui/api/flows/runs/${run_id}/events`);
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        drawer.appendEvent(data);
        markNodeStatus(host, data);
        if (data.type === "artifact_emitted") consumeArtifactEvent(data, drawer);
        if (data.type === "run_finished") es.close();
      } catch {}
    };
    es.onerror = () => es.close();
  }

  function consumeArtifactEvent(evt, drawer) {
    const api = window.Mio?.artifacts;
    if (!api || typeof api.ingestAndOpen !== "function") {
      drawer.appendEvent({
        type: "error",
        error: "Mio artifact surface is unavailable",
      });
      return;
    }
    try {
      api.ingestAndOpen(evt.artifact);
    } catch (error) {
      drawer.appendEvent({
        type: "error",
        error: `Could not open artifact: ${error?.message || error}`,
      });
    }
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
        ln.className = "flow-drawer-line " + (evt.type || "")
          + (evt.type === "run_finished" && evt.ok === false ? " failed" : "");
        const icon = {
          starting:        "⏳",
          run_started:     "🟢",
          node_started:    "▶",
          node_finished:   "✓",
          artifact_emitted:"▣",
          node_error:      "✗",
          node_skipped:    "↷",
          run_finished:    evt.ok === false ? "❌" : "✅",
          run_cancelled:   "■",
          error:           "✗",
        }[evt.type] || "·";
        const artifactLabel = evt.type === "artifact_emitted" && evt.artifact
          ? `${evt.artifact.title || "Artifact"} (${evt.artifact.type || "unknown"})`
          : "";
        const body = artifactLabel || (evt.output !== undefined
          ? (typeof evt.output === "string" ? evt.output : JSON.stringify(evt.output))
          : (evt.error || evt.class || evt.node_order?.join(" → ") || ""));
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
