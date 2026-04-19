// sovereignty.js — persistent local-first footer bar.
//
// Signals what users care about when their data lives on-device:
//   · Where the data actually is (~/.mio)
//   · Whether the tab has made outbound network calls this session
//   · One-click reveal in Finder
//   · One-click workspace export as zip
//
// Zero-telemetry by design. The network counter fires on fetch() to
// non-origin hosts only — local /ui/* and /v1/* calls don't count.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.sovereignty) return;

  let netOutCount = 0;
  let netHosts = new Set();
  let barEl = null;

  // Monkey-patch fetch to count off-origin hits.
  const origFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    try {
      const url = typeof input === "string" ? input : input.url;
      if (url) {
        const u = new URL(url, location.href);
        if (u.origin !== location.origin) {
          netOutCount++;
          netHosts.add(u.host);
          updateBar();
        }
      }
    } catch { /* ignore */ }
    return origFetch(input, init);
  };

  function mount() {
    if (document.querySelector(".mio-sovereignty")) return;
    const bar = document.createElement("div");
    bar.className = "mio-sovereignty";
    bar.innerHTML = `
      <div class="mio-sov-seg mio-sov-local" title="Everything runs on this Mac">
        <span class="mio-sov-dot"></span>
        <span>Local</span>
      </div>
      <div class="mio-sov-sep"></div>
      <div class="mio-sov-seg mio-sov-net" data-action="net"
           title="Off-origin fetches this session (web skills, HF pulls). Click to see hosts.">
        <span class="mio-sov-icon">↗</span>
        <span class="mio-sov-net-count">0</span>
        <span>network calls</span>
      </div>
      <div class="mio-sov-sep"></div>
      <div class="mio-sov-seg mio-sov-data" data-action="reveal"
           title="Reveal the data folder in Finder">
        <span class="mio-sov-icon">⌂</span>
        <span>~/.mio</span>
      </div>
      <div class="mio-sov-spacer"></div>
      <button class="mio-sov-btn" data-action="export" title="Export everything as a zip">Export workspace</button>
    `;
    document.body.appendChild(bar);
    barEl = bar;

    bar.querySelector('[data-action="net"]').addEventListener("click", showNetworkPopover);
    bar.querySelector('[data-action="reveal"]').addEventListener("click", revealInFinder);
    bar.querySelector('[data-action="export"]').addEventListener("click", exportWorkspace);
    updateBar();
  }

  function updateBar() {
    if (!barEl) return;
    const countEl = barEl.querySelector(".mio-sov-net-count");
    if (countEl) countEl.textContent = netOutCount;
    barEl.classList.toggle("mio-sov-warm", netOutCount > 0);
  }

  function showNetworkPopover() {
    const existing = document.querySelector(".mio-sov-popover");
    if (existing) { existing.remove(); return; }
    const pop = document.createElement("div");
    pop.className = "mio-sov-popover";
    const hosts = Array.from(netHosts).sort();
    pop.innerHTML = `
      <header>Off-origin hosts this session</header>
      ${hosts.length
        ? `<ul>${hosts.map((h) => `<li><code>${escapeHtml(h)}</code></li>`).join("")}</ul>`
        : `<p class="muted">None. No data has left this machine.</p>`}
      <footer>
        <button data-action="reset-net">Reset counter</button>
        <button data-action="close">Close</button>
      </footer>
    `;
    document.body.appendChild(pop);
    pop.querySelector('[data-action="close"]').addEventListener("click", () => pop.remove());
    pop.querySelector('[data-action="reset-net"]').addEventListener("click", () => {
      netOutCount = 0; netHosts.clear(); updateBar(); pop.remove();
    });
    // Dismiss on outside click
    setTimeout(() => {
      document.addEventListener("click", function onClick(e) {
        if (!pop.contains(e.target) && !e.target.closest(".mio-sov-net")) {
          pop.remove();
          document.removeEventListener("click", onClick);
        }
      });
    }, 50);
  }

  async function revealInFinder() {
    try {
      const r = await fetch("/ui/api/reveal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: "~/.mio" }),
      });
      const data = await r.json();
      if (data.error) alert("Reveal failed: " + data.error);
    } catch (e) {
      alert("Reveal failed: " + e.message);
    }
  }

  async function exportWorkspace() {
    const btn = barEl.querySelector('[data-action="export"]');
    if (!btn) return;
    btn.disabled = true; btn.textContent = "Zipping…";
    try {
      const r = await fetch("/ui/api/export-workspace", { method: "POST" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const blob = await r.blob();
      const cd = r.headers.get("Content-Disposition") || "";
      const m = cd.match(/filename="([^"]+)"/);
      const filename = m ? m[1] : "mio-workspace.zip";
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    } catch (e) {
      alert("Export failed: " + e.message);
    } finally {
      btn.disabled = false; btn.textContent = "Export workspace";
    }
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount, { once: true });
  } else {
    mount();
  }

  window.Mio.sovereignty = {
    mount,
    reset() { netOutCount = 0; netHosts.clear(); updateBar(); },
  };
})();
