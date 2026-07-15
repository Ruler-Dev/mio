// onboarding_sovereignty.js — first-launch local-first card.
//
// Shown once per browser profile. A three-button card explaining:
//   1. Where the data lives  (~/.mio, with Reveal button)
//   2. That nothing leaves the machine until you invoke a web skill
//      (with a pointer to the footer network counter)
//   3. How to wipe / export
//
// Dismissal stored in localStorage (mio.sovereignty.onboarded.v1).

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.sovereigntyOnboarding) return;
  const KEY = "mio.sovereignty.onboarded.v1";

  function show() {
    if (localStorage.getItem(KEY)) return;
    const ov = document.createElement("div");
    ov.className = "mio-sov-onboard";
    ov.innerHTML = `
      <div class="mio-sov-onboard-card">
        <div class="mio-sov-onboard-badge">🟢 Local</div>
        <h1>Welcome to Mio</h1>
        <p class="mio-sov-onboard-lead">A local AI workstation that runs entirely on this Mac. Nothing leaves your machine unless you ask it to.</p>
        <div class="mio-sov-onboard-grid">
          <div>
            <div class="mio-sov-onboard-ico">⌂</div>
            <strong>Your data</strong>
            <p>Chats, artifacts, projects, notes, RAG index, and journals all live under <code>~/.mio</code>. Plain files, no lock-in.</p>
            <button data-act="reveal">Reveal in Finder</button>
          </div>
          <div>
            <div class="mio-sov-onboard-ico">↗</div>
            <strong>Network calls</strong>
            <p>When a web skill needs the internet (search, fetch, HuggingFace), the footer bar lights up and logs the host. You can audit every call.</p>
            <button data-act="network">Open network monitor</button>
          </div>
          <div>
            <div class="mio-sov-onboard-ico">📦</div>
            <strong>Export / wipe</strong>
            <p>"Export workspace" zips everything for backup. Delete <code>~/.mio</code> and Mio starts fresh, no residue.</p>
            <button data-act="export">Export now</button>
          </div>
        </div>
        <div class="mio-sov-onboard-actions">
          <label class="mio-sov-restricted">
            <input type="checkbox" id="mio-sov-restricted"> Start in Restricted mode (built-in skills only)
          </label>
          <div style="flex:1"></div>
          <button data-act="done" class="primary">Got it</button>
        </div>
      </div>
    `;
    document.body.appendChild(ov);
    // Click-outside and Esc dismiss — prevents "invisible card" from
    // locking the user out if the card's background blends into the
    // overlay (dark mode edge case).
    const dismiss = () => {
      localStorage.setItem(KEY, "1");
      ov.remove();
      window.removeEventListener("keydown", onEsc);
    };
    const onEsc = (e) => { if (e.key === "Escape") dismiss(); };
    window.addEventListener("keydown", onEsc);
    ov.addEventListener("click", (e) => {
      if (e.target === ov) dismiss();
    });
    ov.querySelector('[data-act="done"]').addEventListener("click", () => {
      const restricted = ov.querySelector("#mio-sov-restricted").checked;
      if (restricted) localStorage.setItem("mio.restricted", "1");
      dismiss();
    });
    ov.querySelector('[data-act="reveal"]').addEventListener("click", async () => {
      try { await fetch("/ui/api/reveal", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: "~/.mio" }) }); } catch {}
    });
    ov.querySelector('[data-act="network"]').addEventListener("click", () => {
      dismiss();
      // The persistent sovereignty footer owns the audited host list. Opening
      // its real control keeps onboarding and the live monitor in sync.
      requestAnimationFrame(() => {
        const networkMonitor = document.querySelector(".mio-sovereignty .mio-sov-net");
        if (networkMonitor) networkMonitor.click();
      });
    });
    ov.querySelector('[data-act="export"]').addEventListener("click", async () => {
      try {
        const r = await fetch("/ui/api/export-workspace", { method: "POST" });
        const blob = await r.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "mio-workspace.zip";
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      } catch (e) { alert("Export failed: " + e.message); }
    });
  }

  // Delay to let the rest of the UI boot.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(show, 400), { once: true });
  } else {
    setTimeout(show, 400);
  }

  window.Mio.sovereigntyOnboarding = { show, reset() { localStorage.removeItem(KEY); } };
})();
