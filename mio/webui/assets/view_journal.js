// view_journal.js — Daily Note view.
//
// Simple full-width markdown editor for today's ~/.mio/journal/<date>.md
// with a "yesterday's tail" block pinned above it. Auto-save every 2s
// of idle after edit; a persistent settings toggle (mio.journal.landing)
// — if on, switching to Chat falls back to opening the Journal once per
// day.

(function () {
  window.Mio = window.Mio || {};
  const ready = () => {
    if (!window.Mio.views) return setTimeout(ready, 50);
    window.Mio.views.register("journal", {
      title: "Journal",
      mount(host) { renderRoot(host); },
    });
  };
  ready();

  const LANDING_KEY = "mio.journal.landing";

  async function renderRoot(host) {
    host.innerHTML = `<div class="muted" style="padding:28px">Loading today's journal…</div>`;
    const data = await fetch("/ui/api/journal/today").then((r) => r.json());
    host.innerHTML = `
      <div class="view-journal">
        <header class="view-header">
          <div>
            <h1>Journal — ${data.date}</h1>
            <p class="muted"><code style="font-family:var(--font-mono);font-size:11px">${escapeHtml(data.path)}</code></p>
          </div>
          <div class="view-header-actions">
            <label class="journal-landing">
              <input type="checkbox" id="journal-landing">
              <span>Open on launch</span>
            </label>
            <button class="btn-ghost" data-action="save">Save now</button>
          </div>
        </header>
        <div class="view-body" style="padding:0;height:calc(100vh - 86px - 28px)">
          ${data.yesterday_tail ? `
            <div class="journal-yesterday">
              <header>Yesterday · ${data.yesterday_date}</header>
              <pre>${escapeHtml(data.yesterday_tail)}</pre>
            </div>
          ` : ""}
          <textarea class="journal-editor" id="journal-text" spellcheck="false" placeholder="Today's thoughts, questions, plans, rubber-duck log…">${escapeHtml(data.content)}</textarea>
          <div class="journal-foot muted" id="journal-foot"></div>
        </div>
      </div>
    `;
    const ta = host.querySelector("#journal-text");
    const foot = host.querySelector("#journal-foot");
    const landingCb = host.querySelector("#journal-landing");
    landingCb.checked = localStorage.getItem(LANDING_KEY) === "1";
    landingCb.addEventListener("change", () => {
      localStorage.setItem(LANDING_KEY, landingCb.checked ? "1" : "0");
    });

    let t = null;
    ta.addEventListener("input", () => {
      foot.textContent = "Unsaved — auto-saving in 2s…";
      clearTimeout(t);
      t = setTimeout(() => save(ta, foot), 2000);
    });
    host.querySelector('[data-action="save"]').addEventListener("click", () => save(ta, foot, true));
    updateFoot(ta, foot);
  }

  async function save(ta, foot, immediate) {
    try {
      const r = await fetch("/ui/api/journal/today", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: ta.value }),
      });
      await r.json();
      if (immediate) foot.textContent = `Saved · ${ta.value.length} chars`;
      else           updateFoot(ta, foot);
    } catch (e) {
      foot.textContent = "Save failed: " + e.message;
    }
  }
  function updateFoot(ta, foot) {
    foot.textContent = `${ta.value.length} chars · ${ta.value.split("\n").length} lines`;
  }

  // Landing-on-launch: if the user has the toggle on and the view
  // router is restoring to Chat, pivot them to the journal on first
  // open of the day (per localStorage).
  document.addEventListener("DOMContentLoaded", () => {
    const on = localStorage.getItem(LANDING_KEY) === "1";
    if (!on) return;
    const lastOpen = localStorage.getItem("mio.journal.lastOpenDay");
    const today = new Date().toISOString().slice(0, 10);
    if (lastOpen === today) return;
    localStorage.setItem("mio.journal.lastOpenDay", today);
    // Wait until the views router has booted
    setTimeout(() => {
      if (window.Mio?.views?.switch) window.Mio.views.switch("journal");
    }, 200);
  });

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }
})();
