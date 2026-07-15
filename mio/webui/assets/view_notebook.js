// view_notebook.js — Notebook Mode.
//
// Cell-based interactive canvas. Four cell types:
//   python   — runs in Pyodide (Web Worker); outputs stdout / return repr
//   markdown — rendered prose
//   chat     — one-shot question to the loaded Mio model, output is text
//   skill    — calls /ui/api/skills/run, output is JSON
//
// Notebooks persist in localStorage (mio.notebook.v1) as a flat list
// of cells. Execution order is linear for now (no reactive cell-DAG
// yet — that lands next iteration once we have deterministic AST
// extraction of Python names).

(function () {
  window.Mio = window.Mio || {};
  const ready = () => {
    if (!window.Mio.views) return setTimeout(ready, 50);
    window.Mio.views.register("notebook", {
      title: "Notebook",
      mount(host) { renderRoot(host); },
    });
  };
  ready();

  const STORAGE_KEY = "mio.notebook.v1";
  let pyodideLoading = null;

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) return JSON.parse(raw);
    } catch {}
    return {
      cells: [
        { id: cid(), type: "markdown", src: "# Mio Notebook\n\nDrop cells, run them, chain them. Python runs via Pyodide in a Web Worker — the kernel lives in your browser." },
        { id: cid(), type: "python",   src: "import sys\nprint('Pyodide ready:', sys.version)" },
      ],
    };
  }
  function saveState(state) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch {}
  }
  function cid() { return "c" + Math.random().toString(36).slice(2, 9); }

  function renderRoot(host) {
    const state = loadState();
    host.innerHTML = `
      <div class="view-notebook">
        <header class="view-header">
          <div>
            <h1>Notebook</h1>
            <p class="muted">Polyglot cells — python · markdown · chat · skill. Python runs in Pyodide.</p>
          </div>
          <div class="view-header-actions">
            <button class="btn-ghost" data-action="run-all">▶ Run all</button>
            <button class="btn-ghost" data-action="clear">Clear outputs</button>
            <button class="btn-ghost" data-action="reset">New notebook</button>
          </div>
        </header>
        <div class="view-body" style="padding:16px 24px">
          <div id="nb-cells" class="nb-cells"></div>
          <div class="nb-add-row">
            <button data-add="python">+ Python</button>
            <button data-add="markdown">+ Markdown</button>
            <button data-add="chat">+ Chat</button>
            <button data-add="skill">+ Skill</button>
          </div>
        </div>
      </div>
    `;
    renderCells(host, state);
    wireHeader(host, state);
  }

  function renderCells(host, state) {
    const wrap = host.querySelector("#nb-cells");
    wrap.innerHTML = "";
    state.cells.forEach((cell, i) => wrap.appendChild(buildCell(host, state, cell, i)));
  }

  function buildCell(host, state, cell, idx) {
    const el = document.createElement("div");
    el.className = "nb-cell nb-cell-" + cell.type;
    el.dataset.id = cell.id;
    el.innerHTML = `
      <header class="nb-cell-head">
        <span class="nb-cell-kind">${cell.type}</span>
        <span class="nb-cell-id">${cell.id}</span>
        <div style="flex:1"></div>
        <button data-act="run"    title="Run this cell">▶</button>
        <button data-act="up"     title="Move up">↑</button>
        <button data-act="down"   title="Move down">↓</button>
        <button data-act="delete" title="Delete">×</button>
      </header>
      <textarea class="nb-cell-src" spellcheck="false">${escapeHtml(cell.src || "")}</textarea>
      <div class="nb-cell-out"></div>
    `;
    const ta = el.querySelector(".nb-cell-src");
    const out = el.querySelector(".nb-cell-out");
    autoSize(ta);
    ta.addEventListener("input", () => {
      cell.src = ta.value;
      autoSize(ta);
      saveState(state);
    });
    ta.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        runCell(cell, out, state);
      }
    });
    el.querySelector('[data-act="run"]').addEventListener("click", () => runCell(cell, out, state));
    el.querySelector('[data-act="up"]').addEventListener("click", () => {
      if (idx <= 0) return;
      state.cells.splice(idx, 1); state.cells.splice(idx - 1, 0, cell);
      saveState(state); renderCells(host, state);
    });
    el.querySelector('[data-act="down"]').addEventListener("click", () => {
      if (idx >= state.cells.length - 1) return;
      state.cells.splice(idx, 1); state.cells.splice(idx + 1, 0, cell);
      saveState(state); renderCells(host, state);
    });
    el.querySelector('[data-act="delete"]').addEventListener("click", () => {
      state.cells.splice(idx, 1);
      saveState(state); renderCells(host, state);
    });
    // If the cell has a persisted output, re-render it
    if (cell.lastOutput) {
      renderOutput(out, cell.type, cell.lastOutput);
    }
    return el;
  }

  function wireHeader(host, state) {
    host.querySelectorAll("[data-add]").forEach((b) => {
      b.addEventListener("click", () => {
        const t = b.dataset.add;
        const defaults = {
          python:   "# Python cell — runs in Pyodide\nx = 10\nprint('x =', x)",
          markdown: "# Section\n\nWrite notes in **markdown**.",
          chat:     "Ask Mio a question…",
          skill:    JSON.stringify({ skill: "web_search", args: { query: "local LLM tips" } }, null, 2),
        };
        state.cells.push({ id: cid(), type: t, src: defaults[t] || "" });
        saveState(state);
        renderCells(host, state);
      });
    });
    host.querySelector('[data-action="run-all"]').addEventListener("click", async () => {
      for (const cell of state.cells) {
        const el = host.querySelector(`[data-id="${cell.id}"] .nb-cell-out`);
        await runCell(cell, el, state);
      }
    });
    host.querySelector('[data-action="clear"]').addEventListener("click", () => {
      for (const cell of state.cells) delete cell.lastOutput;
      saveState(state);
      renderCells(host, state);
    });
    host.querySelector('[data-action="reset"]').addEventListener("click", () => {
      if (!confirm("Discard all cells and start a new notebook?")) return;
      localStorage.removeItem(STORAGE_KEY);
      renderRoot(host);
    });
  }

  function autoSize(ta) {
    ta.style.height = "auto";
    ta.style.height = Math.min(600, ta.scrollHeight + 2) + "px";
  }

  // --- Cell runners -----------------------------------------------------

  async function runCell(cell, outEl, state) {
    if (!outEl) return;
    outEl.innerHTML = `<div class="muted" style="padding:8px">running…</div>`;
    let result;
    try {
      if (cell.type === "python")        result = await runPython(cell.src);
      else if (cell.type === "markdown") result = { html: renderMarkdown(cell.src) };
      else if (cell.type === "chat")     result = { text: await runChat(cell.src) };
      else if (cell.type === "skill")    result = await runSkill(cell.src);
      else                               result = { error: "unknown cell type" };
    } catch (e) {
      result = { error: String(e?.message || e) };
    }
    cell.lastOutput = result;
    saveState(state);
    renderOutput(outEl, cell.type, result);
  }

  function renderOutput(outEl, type, result) {
    if (!result) { outEl.innerHTML = ""; return; }
    if (result.error) {
      outEl.innerHTML = `<pre class="nb-out-err">${escapeHtml(result.error)}</pre>`;
      return;
    }
    if (type === "markdown") {
      outEl.innerHTML = `<div class="nb-md">${result.html || ""}</div>`;
      return;
    }
    if (type === "python") {
      const stdout = result.stdout || "";
      const ret = result.result;
      let html = "";
      if (stdout) html += `<pre class="nb-out">${escapeHtml(stdout)}</pre>`;
      if (ret !== undefined && ret !== null && ret !== "") {
        html += `<pre class="nb-out-return">${escapeHtml(typeof ret === "string" ? ret : JSON.stringify(ret, null, 2))}</pre>`;
      }
      outEl.innerHTML = html || `<div class="muted" style="padding:4px 8px;font-size:11px">(no output)</div>`;
      return;
    }
    if (type === "chat") {
      outEl.innerHTML = `<div class="nb-out-chat">${escapeHtml(result.text || "")}</div>`;
      return;
    }
    if (type === "skill") {
      outEl.innerHTML = `<pre class="nb-out">${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
      return;
    }
  }

  async function runPython(src) {
    const py = await ensurePyodide();
    // Capture stdout/stderr
    py.setStdout({ batched: (s) => { buf.stdout += s; } });
    py.setStderr({ batched: (s) => { buf.stdout += s; } });
    const buf = { stdout: "" };
    let result;
    try {
      result = await py.runPythonAsync(src);
      if (result !== undefined && result !== null) {
        try { result = result.toJs ? result.toJs() : result; } catch {}
      }
    } catch (e) {
      return { stdout: buf.stdout, error: String(e?.message || e) };
    }
    return { stdout: buf.stdout, result };
  }

  async function ensurePyodide() {
    if (window.pyodide) return window.pyodide;
    if (pyodideLoading) return pyodideLoading;
    pyodideLoading = (async () => {
      if (!window.loadPyodide) {
        await new Promise((res, rej) => {
          const s = document.createElement("script");
          s.src = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
          s.integrity = "sha384-i3R37b3tF+HWudsUf1VSEOY2YxwSNMqY8DQa9Z0O3xh+NkJ9o+yjcGyIi5huj+nB";
          s.crossOrigin = "anonymous";
          s.onload = res; s.onerror = rej;
          document.head.appendChild(s);
        });
      }
      const py = await window.loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/" });
      window.pyodide = py;
      return py;
    })();
    return pyodideLoading;
  }

  async function runChat(prompt) {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "mio-auto",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.6, max_tokens: 600, stream: false,
      }),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    return data.choices?.[0]?.message?.content || "";
  }

  async function runSkill(srcJSON) {
    let body;
    try { body = JSON.parse(srcJSON); } catch (e) { throw new Error("invalid JSON: " + e.message); }
    const r = await fetch("/ui/api/skills/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: body.skill, arguments: body.args || {} }),
    });
    return await r.json();
  }

  function renderMarkdown(src) {
    // If marked is already on the page (main chat uses it) reuse it;
    // otherwise do a tiny subset fallback.
    if (window.marked?.parse) {
      const rendered = window.marked.parse(src);
      return window.Mio?.sanitizeHtml ? window.Mio.sanitizeHtml(rendered) : escapeHtml(src);
    }
    return escapeHtml(src).replace(/\n/g, "<br>");
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }
})();
