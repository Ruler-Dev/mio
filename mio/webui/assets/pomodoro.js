// /pomodoro — opens a pomodoro timer in a new window/popup. Self-contained
// page with start/pause/reset controls and audible end-of-session beep.
(function () {
  const NS = (window.Mio = window.Mio || {});

  function open(focusMin = 25, breakMin = 5) {
    const html = `<!doctype html><html><head><meta charset="utf-8">
<title>Pomodoro · Mio</title>
<style>
  body { margin: 0; font-family: -apple-system, sans-serif; background: #0a0a14; color: #eee; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; }
  h1 { font-size: 72px; font-weight: 300; margin: 0 0 8px; font-variant-numeric: tabular-nums; }
  h2 { font-size: 14px; color: #8b9cb8; letter-spacing: 3px; text-transform: uppercase; margin: 0 0 40px; }
  .btns { display: flex; gap: 12px; margin-top: 24px; }
  button { background: #3b82f6; color: #fff; border: 0; padding: 10px 22px; border-radius: 999px; font-size: 14px; cursor: pointer; }
  button.ghost { background: transparent; border: 1px solid #2a2a38; color: #9ca3af; }
  .phase { padding: 3px 14px; border-radius: 999px; font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 16px; }
  .phase.focus { background: rgba(59,130,246,0.2); color: #60a5fa; }
  .phase.break { background: rgba(16,185,129,0.2); color: #34d399; }
</style></head>
<body>
<div class="phase focus" id="phase">Focus</div>
<h1 id="display">${String(focusMin).padStart(2,'0')}:00</h1>
<h2 id="round">Round 1 of ∞</h2>
<div class="btns">
  <button id="startBtn">▶ Start</button>
  <button class="ghost" id="pauseBtn" disabled>Pause</button>
  <button class="ghost" id="resetBtn">Reset</button>
</div>
<script>
let FOCUS = ${focusMin * 60}, BREAK = ${breakMin * 60};
let remaining = FOCUS;
let phase = 'focus';
let round = 1;
let timerId = null;
let running = false;
function fmt(s) { const m = Math.floor(s/60), sec = s%60; return String(m).padStart(2,'0') + ':' + String(sec).padStart(2,'0'); }
function render() {
  document.getElementById('display').textContent = fmt(remaining);
  const ph = document.getElementById('phase');
  ph.textContent = phase === 'focus' ? 'Focus' : 'Break';
  ph.className = 'phase ' + phase;
  document.getElementById('round').textContent = 'Round ' + round + ' · ' + (phase === 'focus' ? 'Focus' : 'Break');
  document.title = fmt(remaining) + ' · ' + phase + ' · Mio';
}
function tick() {
  remaining--;
  if (remaining < 0) {
    beep();
    if (phase === 'focus') { phase = 'break'; remaining = BREAK; }
    else { phase = 'focus'; remaining = FOCUS; round++; }
  }
  render();
}
function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 880;
    osc.connect(gain); gain.connect(ctx.destination);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.6);
    osc.start(); osc.stop(ctx.currentTime + 0.6);
  } catch (e) {}
}
document.getElementById('startBtn').onclick = () => {
  if (running) return;
  running = true;
  timerId = setInterval(tick, 1000);
  document.getElementById('startBtn').disabled = true;
  document.getElementById('pauseBtn').disabled = false;
};
document.getElementById('pauseBtn').onclick = () => {
  running = false;
  clearInterval(timerId);
  document.getElementById('startBtn').disabled = false;
  document.getElementById('pauseBtn').disabled = true;
};
document.getElementById('resetBtn').onclick = () => {
  running = false;
  clearInterval(timerId);
  remaining = FOCUS; phase = 'focus'; round = 1;
  document.getElementById('startBtn').disabled = false;
  document.getElementById('pauseBtn').disabled = true;
  render();
};
render();
</script>
</body></html>`;
    const w = window.open('', 'pomodoro', 'width=420,height=520');
    if (!w) { if (window.toast) window.toast('Popup blocked'); return; }
    w.document.open();
    w.document.write(html);
    w.document.close();
  }

  NS.pomodoro = { open };
})();
