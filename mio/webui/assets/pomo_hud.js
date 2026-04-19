// pomo_hud.js — minimal Pomodoro timer HUD.
//
// ⌘⇧P toggles. Shows a small floating pill with MM:SS + play/pause/
// reset. Alternates 25-min focus and 5-min break. On each state
// change a ding (beep via AudioContext). Optional browser
// notification when a cycle ends (user consent prompted first).
//
// Persists: work length, break length, cycles completed.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.pomoHud) return;

  const DEFAULT_WORK  = 25 * 60;
  const DEFAULT_BREAK =  5 * 60;
  let hud = null;
  let remaining = DEFAULT_WORK;
  let total     = DEFAULT_WORK;
  let phase     = "work";
  let running   = false;
  let tick      = null;
  let cycles    = parseInt(localStorage.getItem("mio.pomo.cycles") || "0", 10);

  function open() {
    if (hud) return;
    hud = document.createElement("div");
    hud.className = "mio-pomo";
    hud.innerHTML = `
      <div class="mio-pomo-ring">
        <svg viewBox="0 0 36 36"><circle class="bg" cx="18" cy="18" r="15"/><circle class="fg" cx="18" cy="18" r="15"/></svg>
        <div class="mio-pomo-time"></div>
      </div>
      <div class="mio-pomo-ctrls">
        <span class="mio-pomo-phase">Focus</span>
        <button data-act="start">▶</button>
        <button data-act="reset">↺</button>
        <button data-act="close" aria-label="Close">×</button>
      </div>
      <div class="mio-pomo-cycles">🍅 ${cycles}</div>
    `;
    document.body.appendChild(hud);
    hud.querySelector('[data-act="start"]').addEventListener("click", toggleRun);
    hud.querySelector('[data-act="reset"]').addEventListener("click", reset);
    hud.querySelector('[data-act="close"]').addEventListener("click", close);
    render();
  }

  function close() {
    stop();
    hud?.remove();
    hud = null;
  }

  function toggleRun() {
    running ? stop() : start();
  }
  function start() {
    running = true;
    tick = setInterval(step, 1000);
    render();
  }
  function stop() {
    running = false;
    clearInterval(tick);
    render();
  }
  function step() {
    remaining--;
    if (remaining <= 0) {
      ding();
      if (phase === "work") {
        cycles++;
        localStorage.setItem("mio.pomo.cycles", String(cycles));
        phase = "break"; total = remaining = DEFAULT_BREAK;
        notify("Time for a break 🫖", "5 min");
      } else {
        phase = "work"; total = remaining = DEFAULT_WORK;
        notify("Back to focus 🍅", "25 min");
      }
    }
    render();
  }
  function reset() {
    stop();
    phase = "work"; total = remaining = DEFAULT_WORK;
    render();
  }

  function render() {
    if (!hud) return;
    const m = Math.floor(remaining / 60), s = remaining % 60;
    hud.querySelector(".mio-pomo-time").textContent = `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
    hud.querySelector(".mio-pomo-phase").textContent = phase === "work" ? "Focus" : "Break";
    hud.querySelector('[data-act="start"]').textContent = running ? "⏸" : "▶";
    hud.classList.toggle("break", phase === "break");
    const pct = 1 - remaining / total;
    const fg = hud.querySelector("svg .fg");
    const circ = 2 * Math.PI * 15;
    fg.setAttribute("stroke-dasharray", circ);
    fg.setAttribute("stroke-dashoffset", circ * (1 - pct));
    hud.querySelector(".mio-pomo-cycles").textContent = `🍅 ${cycles}`;
  }

  function ding() {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.type = "sine"; o.frequency.value = phase === "work" ? 660 : 440;
      g.gain.setValueAtTime(0.0001, ctx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.45);
      o.connect(g); g.connect(ctx.destination);
      o.start(); o.stop(ctx.currentTime + 0.5);
    } catch {}
  }

  function notify(title, body) {
    if ("Notification" in window && Notification.permission === "granted") {
      try { new Notification(title, { body }); } catch {}
    }
  }

  // Global shortcut
  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "p") {
      e.preventDefault();
      hud ? close() : open();
      if ("Notification" in window && Notification.permission === "default") {
        Notification.requestPermission().catch(() => {});
      }
    }
  });

  window.Mio.pomoHud = { open, close, toggle: () => (hud ? close() : open()) };
})();
