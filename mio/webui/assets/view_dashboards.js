// view_dashboards.js — Dashboard Mode (MVP).
//
// Two-pane editor: data dock on the left, panel grid on the right.
// User pastes CSV / uploads a file → registers a datasource;
// types a prompt → the model emits a <antDashboardPanel> JSON block
// that we add to the grid; clicks a panel → edit its spec.
//
// This MVP keeps the grid simple (CSS grid with resize, no gridstack
// dep yet) and renders via Chart.js. Richer ECharts / DuckDB-WASM
// sql cells land in the next iteration.

(function () {
  window.Mio = window.Mio || {};
  const ready = () => {
    if (!window.Mio.views) return setTimeout(ready, 50);
    window.Mio.views.register("dashboards", {
      title: "Dashboards",
      mount(host) { renderRoot(host); },
    });
  };
  ready();

  const STORAGE_KEY = "mio.dashboards.session";

  function loadSession() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) return JSON.parse(raw);
    } catch {}
    return { sources: [], panels: [], nextPanelId: 1, nextSourceId: 1 };
  }
  function saveSession(state) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch {}
  }
  function parseCSV(text) {
    // Dependency-free CSV parse. Handles quoted fields + \r\n.
    const rows = [];
    let cur = [], field = "", inQ = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (inQ) {
        if (c === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; }
          else inQ = false;
        } else { field += c; }
      } else {
        if (c === '"') inQ = true;
        else if (c === ",") { cur.push(field); field = ""; }
        else if (c === "\n") { cur.push(field); rows.push(cur); cur = []; field = ""; }
        else if (c === "\r") { /* swallow */ }
        else field += c;
      }
    }
    if (field.length || cur.length) { cur.push(field); rows.push(cur); }
    if (!rows.length) return { headers: [], rows: [] };
    const [headers, ...rest] = rows;
    return { headers, rows: rest };
  }

  function renderRoot(host) {
    const state = loadSession();
    host.innerHTML = `
      <div class="view-dashboards">
        <header class="view-header">
          <div>
            <h1>Dashboards</h1>
            <p class="muted">Drop data, build panels, nothing leaves your Mac.</p>
          </div>
          <div class="view-header-actions">
            <button class="btn-ghost" data-action="add-panel">+ Panel</button>
            <button class="btn-ghost" data-action="clear">Clear</button>
          </div>
        </header>
        <div class="dash-split">
          <aside class="dash-dock">
            <header><strong>Data sources</strong></header>
            <div class="dash-source-list" id="dash-sources"></div>
            <footer>
              <label class="dash-upload">
                <input type="file" accept=".csv,.tsv,.json,.txt" style="display:none">
                <span>Upload file</span>
              </label>
              <button class="btn-ghost" data-action="paste">Paste CSV</button>
            </footer>
          </aside>
          <main class="dash-grid" id="dash-grid"></main>
        </div>
      </div>
    `;
    renderSources(host, state);
    renderPanels(host, state);
    wireHandlers(host, state);
  }

  function renderSources(host, state) {
    const wrap = host.querySelector("#dash-sources");
    if (!wrap) return;
    if (!state.sources.length) {
      wrap.innerHTML = `<div class="muted" style="padding:10px 14px;font-size:12px">Paste or upload a CSV / JSON to get started.</div>`;
      return;
    }
    wrap.innerHTML = state.sources.map((s, i) => `
      <div class="dash-source" data-idx="${i}">
        <div>
          <div class="dash-source-name">${escapeHtml(s.name)}</div>
          <div class="dash-source-meta">${s.rows.length} rows · ${s.headers.length} cols · <code>${escapeHtml(s.id)}</code></div>
        </div>
        <button class="dash-source-del" aria-label="Remove">×</button>
      </div>
    `).join("");
    wrap.querySelectorAll(".dash-source-del").forEach((btn, i) => {
      btn.addEventListener("click", () => {
        state.sources.splice(i, 1);
        saveSession(state);
        renderSources(host, state);
      });
    });
  }

  function renderPanels(host, state) {
    const grid = host.querySelector("#dash-grid");
    if (!grid) return;
    if (!state.panels.length) {
      grid.innerHTML = `
        <div class="dash-empty">
          <h2>No panels yet</h2>
          <p>Hit <b>+ Panel</b> to add a chart spec, or ask Mio in chat<br>
          "show sales by region from ${state.sources[0]?.id || "ds_1"}" and a panel will drop in here.</p>
        </div>
      `;
      return;
    }
    grid.innerHTML = "";
    for (const panel of state.panels) grid.appendChild(renderPanel(host, state, panel));
  }

  function renderPanel(host, state, panel) {
    const card = document.createElement("div");
    card.className = "dash-panel";
    card.style.gridColumn = `span ${panel.w || 6}`;
    card.style.gridRow    = `span ${panel.h || 1}`;
    card.innerHTML = `
      <header>
        <span class="dash-panel-title">${escapeHtml(panel.title || "Untitled panel")}</span>
        <div class="dash-panel-actions">
          <button data-act="edit">Edit</button>
          <button data-act="delete">×</button>
        </div>
      </header>
      <div class="dash-panel-body"></div>
    `;
    const body = card.querySelector(".dash-panel-body");
    drawPanel(panel, state, body);
    card.querySelector('[data-act="edit"]').addEventListener("click", () => editPanel(host, state, panel));
    card.querySelector('[data-act="delete"]').addEventListener("click", () => {
      state.panels = state.panels.filter((p) => p.id !== panel.id);
      saveSession(state);
      renderPanels(host, state);
    });
    return card;
  }

  async function ensureChartJs() {
    if (window.Chart) return window.Chart;
    await new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js";
      s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
    return window.Chart;
  }

  async function drawPanel(panel, state, body) {
    if (panel.type === "kpi") {
      const val = resolveValue(panel, state);
      body.innerHTML = `
        <div class="dash-kpi">
          <div class="dash-kpi-value">${escapeHtml(String(val ?? "—"))}</div>
          <div class="dash-kpi-label">${escapeHtml(panel.label || "")}</div>
        </div>
      `;
      return;
    }
    if (panel.type === "table") {
      body.innerHTML = renderTable(resolveTable(panel, state));
      return;
    }
    if (panel.type === "chart") {
      await ensureChartJs();
      const canvas = document.createElement("canvas");
      body.innerHTML = "";
      body.appendChild(canvas);
      const data = resolveChart(panel, state);
      new window.Chart(canvas, {
        type: panel.chart_type || "bar",
        data: {
          labels: data.labels,
          datasets: [{
            label: panel.title || "",
            data: data.values,
            backgroundColor: ["#3b82f6","#0ea5e9","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899"],
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
        },
      });
    }
  }

  function resolveTable(panel, state) {
    const s = state.sources.find((x) => x.id === panel.source);
    if (!s) return { headers: [], rows: [] };
    return { headers: s.headers, rows: s.rows.slice(0, panel.limit || 12) };
  }

  function resolveChart(panel, state) {
    const s = state.sources.find((x) => x.id === panel.source);
    if (!s) return { labels: panel.labels || [], values: panel.values || [] };
    const labelCol = Math.max(0, s.headers.indexOf(panel.x));
    const valueCol = Math.max(0, s.headers.indexOf(panel.y));
    return {
      labels: s.rows.map((r) => r[labelCol]),
      values: s.rows.map((r) => parseFloat(r[valueCol]) || 0),
    };
  }

  function resolveValue(panel, state) {
    if (panel.value !== undefined) return panel.value;
    const s = state.sources.find((x) => x.id === panel.source);
    if (!s) return "—";
    const col = Math.max(0, s.headers.indexOf(panel.column));
    const nums = s.rows.map((r) => parseFloat(r[col]) || 0);
    if (panel.agg === "sum")    return nums.reduce((a, b) => a + b, 0).toFixed(2);
    if (panel.agg === "avg")    return (nums.reduce((a, b) => a + b, 0) / (nums.length || 1)).toFixed(2);
    if (panel.agg === "count")  return String(nums.length);
    if (panel.agg === "min")    return String(Math.min(...nums));
    if (panel.agg === "max")    return String(Math.max(...nums));
    return "—";
  }

  function renderTable({ headers, rows }) {
    if (!headers.length) return `<div class="muted" style="padding:10px">No data</div>`;
    return `<div class="dash-table-wrap"><table class="dash-table"><thead><tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead><tbody>${
      rows.map((r) => `<tr>${r.map((c) => `<td>${escapeHtml(c)}</td>`).join("")}</tr>`).join("")
    }</tbody></table></div>`;
  }

  function editPanel(host, state, panel) {
    const spec = prompt("Edit panel spec (JSON):", JSON.stringify(panel, null, 2));
    if (!spec) return;
    try {
      const next = JSON.parse(spec);
      Object.assign(panel, next);
      saveSession(state);
      renderPanels(host, state);
    } catch (e) {
      alert("Invalid JSON: " + e.message);
    }
  }

  function addPanel(host, state) {
    const id = state.nextPanelId++;
    const source = state.sources[0];
    state.panels.push({
      id, w: 6, h: 1,
      title: `Panel ${id}`,
      type: "chart", chart_type: "bar",
      source: source?.id || null,
      x: source?.headers[0] || "",
      y: source?.headers[1] || "",
    });
    saveSession(state);
    renderPanels(host, state);
  }

  async function importFile(host, state, file) {
    const text = await file.text();
    let parsed;
    if (file.name.toLowerCase().endsWith(".json")) {
      try {
        const data = JSON.parse(text);
        if (Array.isArray(data) && data.length && typeof data[0] === "object") {
          parsed = { headers: Object.keys(data[0]), rows: data.map((o) => Object.values(o).map(String)) };
        } else {
          alert("JSON must be an array of objects.");
          return;
        }
      } catch (e) { alert("JSON parse error: " + e.message); return; }
    } else {
      parsed = parseCSV(text);
    }
    const id = `ds_${state.nextSourceId++}`;
    state.sources.push({ id, name: file.name, ...parsed });
    saveSession(state);
    renderSources(host, state);
  }

  function wireHandlers(host, state) {
    host.querySelector('[data-action="add-panel"]').addEventListener("click", () => {
      if (!state.sources.length) { alert("Add a data source first (paste or upload a CSV)."); return; }
      addPanel(host, state);
    });
    host.querySelector('[data-action="clear"]').addEventListener("click", () => {
      if (!confirm("Clear all panels + sources in this dashboard?")) return;
      state.sources = []; state.panels = []; state.nextPanelId = 1; state.nextSourceId = 1;
      saveSession(state);
      renderSources(host, state); renderPanels(host, state);
    });
    host.querySelector('[data-action="paste"]').addEventListener("click", async () => {
      const txt = prompt("Paste CSV content:");
      if (!txt) return;
      const name = prompt("Name this source:", "pasted");
      const parsed = parseCSV(txt);
      const id = `ds_${state.nextSourceId++}`;
      state.sources.push({ id, name: name || "pasted", ...parsed });
      saveSession(state);
      renderSources(host, state);
    });
    const fileInput = host.querySelector(".dash-upload input[type=file]");
    fileInput.addEventListener("change", async (e) => {
      for (const f of Array.from(e.target.files || [])) await importFile(host, state, f);
      e.target.value = "";
    });
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }
})();
