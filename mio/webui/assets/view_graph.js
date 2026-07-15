// view_graph.js — Knowledge-graph view.
//
// Renders nodes + edges from /ui/api/graph using Cytoscape.js (CDN).
// Node types: session / artifact / project / doc / note — each with
// its own accent. Edges carry a rel ("in", "emitted").
//
// Filter chips at the top toggle node types. Clicking a session
// node jumps to that chat.

(function () {
  window.Mio = window.Mio || {};
  const ready = () => {
    if (!window.Mio.views) return setTimeout(ready, 50);
    window.Mio.views.register("graph", {
      title: "Graph",
      mount(host) { renderRoot(host); },
    });
  };
  ready();

  const TYPE_STYLE = {
    session:  { color: "#3b82f6", label: "Chats" },
    artifact: { color: "#10b981", label: "Artifacts" },
    project:  { color: "#f59e0b", label: "Workspaces" },
    doc:      { color: "#a855f7", label: "Clipped" },
    note:     { color: "#ec4899", label: "Notes" },
  };

  async function ensureCy() {
    if (window.cytoscape) return window.cytoscape;
    await new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/dist/cytoscape.min.js";
      s.integrity = "sha384-H3uzGzTfGHUAumB8+s4GEdfFwzAceN9wCCndN8AXubWKFIPuBSWKKtWDx7RhSf/z";
      s.crossOrigin = "anonymous";
      s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
    return window.cytoscape;
  }

  async function renderRoot(host) {
    host.innerHTML = `
      <div class="view-graph">
        <header class="view-header">
          <div>
            <h1>Graph</h1>
            <p class="muted">Your chats, artifacts, workspaces, clipped docs and Obsidian notes, wired up.</p>
          </div>
          <div class="view-header-actions">
            <div class="graph-filters" id="graph-filters"></div>
            <button class="btn-ghost" data-action="refresh">Refresh</button>
            <button class="btn-ghost" data-action="fit">Fit</button>
          </div>
        </header>
        <div id="graph-canvas" class="graph-canvas"></div>
      </div>
    `;
    const filtersEl = host.querySelector("#graph-filters");
    for (const [k, v] of Object.entries(TYPE_STYLE)) {
      const chip = document.createElement("button");
      chip.className = "graph-filter active";
      chip.dataset.type = k;
      chip.innerHTML = `<span class="dot" style="background:${v.color}"></span>${v.label}`;
      chip.addEventListener("click", () => {
        chip.classList.toggle("active");
        applyFilter(host);
      });
      filtersEl.appendChild(chip);
    }
    await ensureCy();
    const { nodes = [], edges = [] } = await fetch("/ui/api/graph").then((r) => r.json());
    const cy = window.cytoscape({
      container: host.querySelector("#graph-canvas"),
      elements: [
        ...nodes.map((n) => ({ data: { id: n.id, label: n.label, type: n.type } })),
        ...edges
          .filter((e) => nodes.some((n) => n.id === e.source) && nodes.some((n) => n.id === e.target))
          .map((e, i) => ({ data: { id: "e" + i, source: e.source, target: e.target, rel: e.rel } })),
      ],
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            "label": "data(label)",
            "font-size": 10,
            "color": "#c9d1d9",
            "text-outline-width": 2,
            "text-outline-color": "#0d1117",
            "width": 24,
            "height": 24,
            "text-valign": "bottom",
            "text-margin-y": 4,
          },
        },
        {
          selector: "edge",
          style: {
            "width": 1,
            "line-color": "#30363d",
            "curve-style": "bezier",
            "target-arrow-color": "#30363d",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.6,
            "opacity": 0.7,
          },
        },
        ...Object.entries(TYPE_STYLE).map(([type, { color }]) => ({
          selector: `node[type="${type}"]`,
          style: { "background-color": color },
        })),
        {
          selector: "node:selected",
          style: { "border-width": 3, "border-color": "#58a6ff" },
        },
      ],
      layout: { name: "cose", nodeRepulsion: 8000, idealEdgeLength: 70, animate: false },
      wheelSensitivity: 0.2,
    });
    host._cy = cy;
    cy.on("tap", "node", (evt) => {
      const n = evt.target.data();
      if (n.id?.startsWith("session:")) {
        const sid = n.id.slice("session:".length);
        window.Mio?.views?.switch?.("chat");
        setTimeout(() => window.loadSession?.(sid), 100);
      }
    });
    host.querySelector('[data-action="refresh"]').addEventListener("click", () => renderRoot(host));
    host.querySelector('[data-action="fit"]').addEventListener("click", () => cy.fit());
    applyFilter(host);
  }

  function applyFilter(host) {
    const cy = host._cy;
    if (!cy) return;
    const active = new Set(
      Array.from(host.querySelectorAll(".graph-filter.active")).map((c) => c.dataset.type)
    );
    cy.nodes().forEach((n) => {
      const show = active.has(n.data("type"));
      n.style("display", show ? "element" : "none");
    });
    cy.edges().forEach((e) => {
      const visible = e.source().style("display") !== "none" && e.target().style("display") !== "none";
      e.style("display", visible ? "element" : "none");
    });
  }
})();
