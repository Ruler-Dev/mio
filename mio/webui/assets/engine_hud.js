// engine_hud.js — floating engine status widget.
//
// Bottom-left pill showing the loaded tier, last tok/s, and
// context-window utilisation as a filled progress bar. Click to
// toggle details (prompt tok/s, acceptance, VRAM) and swap tier
// via the existing switchTier() global.
//
// Polls /ui/api/model-info every 5 s while visible, and reads the
// most recent metrics from window.lastMetrics (set by the chat
// pipeline when the main UI exposes it).

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.engineHud) return;

  let hud = null;
  let poll = null;
  let expanded = false;
  let info = null;

  function mount() {
    if (hud) return;
    hud = document.createElement("div");
    hud.className = "mio-engine-hud";
    hud.innerHTML = `
      <div class="mio-engine-row">
        <span class="mio-engine-tier">…</span>
        <span class="mio-engine-tps">— tok/s</span>
        <div class="mio-engine-ctx"><div class="mio-engine-ctx-fill"></div></div>
      </div>
      <div class="mio-engine-detail" hidden></div>
    `;
    document.body.appendChild(hud);
    hud.addEventListener("click", () => {
      expanded = !expanded;
      hud.querySelector(".mio-engine-detail").hidden = !expanded;
      if (expanded) refresh();
    });
    refresh();
    poll = setInterval(refreshLight, 5000);
  }

  async function refresh() {
    try {
      const r = await fetch("/ui/api/model-info");
      info = await r.json();
      render();
    } catch {}
  }
  function refreshLight() {
    // Update tok/s live without hitting the server every 5s (unless
    // the panel's expanded).
    const tps = window.lastMetrics?.generation_tps
              ?? window.lastMetrics?.tps
              ?? info?.last_tps
              ?? null;
    if (hud && tps != null) {
      hud.querySelector(".mio-engine-tps").textContent = Number(tps).toFixed(1) + " tok/s";
    }
    if (expanded) refresh();
  }

  function render() {
    if (!hud || !info) return;
    const tier = info.tier || info.active_tier || "…";
    const tps = window.lastMetrics?.generation_tps || info.last_tps || 0;
    const ctxUsed = info.context_used || window.lastContextUsed || 0;
    const ctxWin  = info.context_window || info.ctx_window || 32768;
    const pct = ctxWin ? Math.min(1, ctxUsed / ctxWin) : 0;
    hud.querySelector(".mio-engine-tier").textContent = tier;
    hud.querySelector(".mio-engine-tps").textContent  = tps ? Number(tps).toFixed(1) + " tok/s" : "idle";
    hud.querySelector(".mio-engine-ctx-fill").style.width = (pct * 100).toFixed(1) + "%";
    hud.querySelector(".mio-engine-ctx-fill").style.background =
      pct > 0.85 ? "#dc2626" : pct > 0.65 ? "#f59e0b" : "var(--accent)";

    if (expanded) {
      const d = hud.querySelector(".mio-engine-detail");
      d.innerHTML = `
        <div class="mio-engine-grid">
          <span>tier</span><code>${escapeHtml(tier)}</code>
          <span>ctx</span><code>${humanInt(ctxUsed)} / ${humanInt(ctxWin)} (${(pct * 100).toFixed(1)}%)</code>
          <span>gen</span><code>${tps ? Number(tps).toFixed(1) : "–"} tok/s</code>
          <span>prompt</span><code>${fmtNum(window.lastMetrics?.prompt_tps)} tok/s</code>
          <span>accept</span><code>${fmtNum(window.lastMetrics?.avg_acceptance_length, 2)}</code>
          <span>VRAM</span><code>${fmtNum(window.lastMetrics?.peak_memory_gb, 1)} GB</code>
        </div>
        <div class="mio-engine-actions">
          ${(info.tiers || ["small","medium","large","large-moe"]).map(
              (t) => `<button data-tier="${escapeAttr(t)}"${t === tier ? " disabled" : ""}>${escapeHtml(t)}</button>`
          ).join("")}
        </div>
      `;
      d.querySelectorAll("button[data-tier]").forEach((b) => {
        b.addEventListener("click", (e) => {
          e.stopPropagation();
          const t = b.dataset.tier;
          if (typeof window.switchTier === "function") {
            try { window.switchTier(t); } catch {}
          }
        });
      });
    }
  }

  function fmtNum(n, dp = 1) {
    if (typeof n !== "number" || !isFinite(n)) return "–";
    return n.toFixed(dp);
  }
  function humanInt(n) {
    if (n == null) return "?";
    if (n >= 1024) return (n / 1024).toFixed(1) + "K";
    return String(n);
  }
  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }
  function escapeAttr(s) { return escapeHtml(s); }

  // Boot only when the main chat surface has had a chance to set up;
  // otherwise just delay.
  setTimeout(mount, 600);

  window.Mio.engineHud = {
    mount, refresh,
    unmount() { clearInterval(poll); hud?.remove(); hud = null; },
  };
})();
