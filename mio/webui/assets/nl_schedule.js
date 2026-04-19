// nl_schedule.js — natural-language scheduling from the composer.
//
// Recognises patterns like:
//   "remind me tomorrow at 9am to water the plants"
//   "every weekday at 8:30am summarise my inbox"
//   "in 30 minutes stretch"
//   "at 17:00 kick off the build"
//
// When detected and the user hits ⌘⏎, a small chip replaces the
// composer text and offers "Create schedule" → POST /ui/api/schedules
// without firing the usual chat send. Escape cancels back to a
// normal prompt.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.nlSchedule) return;

  const PATTERNS = [
    // "every {day} at {HH:MM}? {am/pm}? {task}"
    { kind: "recurring", re: /^every\s+(day|weekday|weekend|mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)(?:\s+at)?\s+([0-9]{1,2})(?::([0-9]{2}))?\s*(am|pm)?\s+(.+)$/i },
    // "tomorrow at 9am do X"
    { kind: "once",      re: /^(tomorrow|today)(?:\s+at)?\s+([0-9]{1,2})(?::([0-9]{2}))?\s*(am|pm)?\s+(.+)$/i },
    // "in N {min|hours} X"
    { kind: "delay",     re: /^in\s+(\d+)\s+(minute|minutes|min|hour|hours|hr|hrs)\s+(.+)$/i },
    // "at HH[:MM] {am/pm}? X"   (today)
    { kind: "once-today", re: /^at\s+([0-9]{1,2})(?::([0-9]{2}))?\s*(am|pm)?\s+(.+)$/i },
    // "remind me (every … | tomorrow … | in …) to X"
    { kind: "remind",    re: /^remind\s+me(?:\s+to)?\s+(.+)$/i },
  ];

  function parse(text) {
    text = (text || "").trim();
    if (!text) return null;

    // Peel off "remind me to" prefix
    let stripped = text.replace(/^remind\s+me\s+(?:to\s+)?/i, "");
    const secondary = stripped !== text;

    for (const p of PATTERNS) {
      if (p.kind === "remind") continue;
      const m = (secondary ? stripped : text).match(p.re);
      if (!m) continue;

      if (p.kind === "delay") {
        const n = parseInt(m[1], 10);
        const unit = m[2].toLowerCase();
        const mins = unit.startsWith("h") ? n * 60 : n;
        return {
          kind: "delay",
          minutes: mins,
          task: m[3].trim(),
          display: `in ${n} ${unit}`,
        };
      }
      if (p.kind === "once" || p.kind === "once-today") {
        const when = p.kind === "once-today" ? "today" : m[1].toLowerCase();
        const offset = m[0].length - m[m.length - 1].length;
        const h0 = parseInt(p.kind === "once-today" ? m[1] : m[2], 10);
        const mn = parseInt(p.kind === "once-today" ? (m[2] || "0") : (m[3] || "0"), 10);
        const ampm = (p.kind === "once-today" ? m[3] : m[4] || "").toLowerCase();
        let h = h0 % 24;
        if (ampm === "pm" && h < 12) h += 12;
        if (ampm === "am" && h === 12) h = 0;
        return {
          kind: "once",
          when,
          hour: h, minute: mn,
          task: (p.kind === "once-today" ? m[4] : m[5]).trim(),
          display: `${when} ${String(h).padStart(2, "0")}:${String(mn).padStart(2, "0")}`,
        };
      }
      if (p.kind === "recurring") {
        const day = m[1].toLowerCase();
        const h0 = parseInt(m[2], 10);
        const mn = parseInt(m[3] || "0", 10);
        const ampm = (m[4] || "").toLowerCase();
        let h = h0 % 24;
        if (ampm === "pm" && h < 12) h += 12;
        if (ampm === "am" && h === 12) h = 0;
        return {
          kind: "recurring",
          day,
          hour: h, minute: mn,
          task: m[5].trim(),
          display: `every ${day} ${String(h).padStart(2, "0")}:${String(mn).padStart(2, "0")}`,
        };
      }
    }
    return null;
  }

  function toCadence(parsed) {
    // Maps to the scheduler.py schema. Keep it simple — the scheduler
    // accepts whatever shape we send; create_schedule normalises.
    if (parsed.kind === "delay") {
      const at = new Date(Date.now() + parsed.minutes * 60_000);
      return { type: "once", at: at.toISOString() };
    }
    if (parsed.kind === "once") {
      const d = new Date();
      if (parsed.when === "tomorrow") d.setDate(d.getDate() + 1);
      d.setHours(parsed.hour, parsed.minute, 0, 0);
      return { type: "once", at: d.toISOString() };
    }
    if (parsed.kind === "recurring") {
      const dayMap = { mon: 1, tue: 2, wed: 3, thu: 4, fri: 5, sat: 6, sun: 0 };
      return {
        type: "weekly",
        day: parsed.day === "day" || parsed.day === "weekday" || parsed.day === "weekend"
          ? parsed.day : parsed.day.slice(0, 3),
        hour: parsed.hour, minute: parsed.minute,
      };
    }
    return null;
  }

  let currentChip = null;

  function findInput() {
    return document.querySelector(
      "textarea#messageInput, textarea#inputArea, textarea#input, textarea.input, textarea[data-role='chat-input']",
    ) || document.querySelector("textarea");
  }

  function showChip(input, parsed) {
    hideChip();
    const rect = input.getBoundingClientRect();
    const chip = document.createElement("div");
    chip.className = "mio-nl-schedule";
    chip.style.left = (rect.left + 10) + "px";
    chip.style.top  = (rect.top - 40) + "px";
    chip.innerHTML = `
      <span class="mio-nl-ico">⏰</span>
      <span class="mio-nl-disp">${escapeHtml(parsed.display)}</span>
      <span class="mio-nl-task">${escapeHtml(parsed.task.slice(0, 40))}</span>
      <button data-act="create">Create schedule ⌘⏎</button>
      <button data-act="cancel" aria-label="Cancel">×</button>
    `;
    document.body.appendChild(chip);
    currentChip = { el: chip, parsed, input };
    chip.querySelector('[data-act="create"]').addEventListener("click", () => create(parsed, input));
    chip.querySelector('[data-act="cancel"]').addEventListener("click", hideChip);
  }

  function hideChip() {
    currentChip?.el?.remove();
    currentChip = null;
  }

  async function create(parsed, input) {
    try {
      const cadence = toCadence(parsed);
      const r = await fetch("/ui/api/schedules", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: parsed.task.slice(0, 60),
          prompt: parsed.task,
          cadence,
          enabled: true,
        }),
      });
      const data = await r.json();
      if (data?.id || data?.ok || data?.schedule) {
        input.value = "";
        input.dispatchEvent(new Event("input", { bubbles: true }));
        toast("Scheduled: " + parsed.display);
      } else {
        toast("Schedule create failed");
      }
    } catch (e) {
      toast("Schedule failed: " + e.message);
    }
    hideChip();
  }

  function toast(msg) {
    const t = document.createElement("div");
    t.className = "mio-nl-toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2200);
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }

  function attach() {
    const input = findInput();
    if (!input || input._nlBound) return;
    input._nlBound = true;
    input.addEventListener("input", () => {
      const p = parse(input.value);
      if (p) showChip(input, p); else hideChip();
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && currentChip) { e.preventDefault(); hideChip(); return; }
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && currentChip) {
        e.preventDefault();
        create(currentChip.parsed, input);
      }
    });
    input.addEventListener("blur", () => setTimeout(hideChip, 200));
  }

  function boot() {
    attach();
    const obs = new MutationObserver(() => attach());
    obs.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }

  window.Mio.nlSchedule = { parse };
})();
