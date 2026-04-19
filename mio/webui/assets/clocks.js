// /worldclock — multi-timezone clock popup
// /zen         — ambient breathing-guide timer
(function () {
  const NS = (window.Mio = window.Mio || {});

  function worldclock() {
    const zones = [
      ['SF', 'America/Los_Angeles'],
      ['NYC', 'America/New_York'],
      ['LDN', 'Europe/London'],
      ['BER', 'Europe/Berlin'],
      ['TYO', 'Asia/Tokyo'],
      ['SYD', 'Australia/Sydney'],
      ['DXB', 'Asia/Dubai'],
      ['SIN', 'Asia/Singapore'],
    ];
    const html = `<!doctype html><html><head><meta charset="utf-8">
<title>World clock · Mio</title>
<style>
  body { margin: 0; font-family: -apple-system, sans-serif; background: #0a0a14; color: #eee; min-height: 100vh; padding: 40px; }
  h1 { font-size: 14px; color: #9ca3af; letter-spacing: 2px; text-transform: uppercase; margin: 0 0 30px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; max-width: 900px; }
  .card { background: #161621; border: 1px solid #2a2a38; border-radius: 14px; padding: 20px; }
  .city { font-size: 12px; color: #9ca3af; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px; }
  .time { font-size: 42px; font-variant-numeric: tabular-nums; font-weight: 300; line-height: 1; }
  .date { font-size: 11px; color: #9ca3af; margin-top: 8px; font-family: ui-monospace,monospace; }
  .off { font-size: 10px; color: #60a5fa; margin-top: 4px; }
  .night { background: linear-gradient(135deg, #0f1729, #1a1a2e); }
  .day { background: linear-gradient(135deg, #161621, #1d1d2e); }
</style></head>
<body>
<h1>World clock</h1>
<div class="grid">
${zones.map(([city, tz]) => `<div class="card" data-tz="${tz}">
  <div class="city">${city}</div>
  <div class="time">--:--</div>
  <div class="date">—</div>
  <div class="off">${tz}</div>
</div>`).join('')}
</div>
<script>
function tick() {
  document.querySelectorAll('.card').forEach(card => {
    const tz = card.dataset.tz;
    try {
      const now = new Date();
      const fmt = new Intl.DateTimeFormat('en-US', { timeZone: tz, hour: '2-digit', minute: '2-digit', hour12: false });
      const dfmt = new Intl.DateTimeFormat('en-US', { timeZone: tz, weekday: 'short', month: 'short', day: '2-digit' });
      const hourFmt = new Intl.DateTimeFormat('en-US', { timeZone: tz, hour: '2-digit', hour12: false });
      card.querySelector('.time').textContent = fmt.format(now);
      card.querySelector('.date').textContent = dfmt.format(now);
      const hr = parseInt(hourFmt.format(now));
      card.classList.toggle('night', hr < 6 || hr >= 20);
      card.classList.toggle('day', hr >= 6 && hr < 20);
    } catch (e) {}
  });
}
tick(); setInterval(tick, 1000);
</script></body></html>`;
    const w = window.open('', 'worldclock', 'width=900,height=600');
    if (!w) { if (window.toast) window.toast('Popup blocked'); return; }
    w.document.open(); w.document.write(html); w.document.close();
  }

  function zen(minutes = 5) {
    const html = `<!doctype html><html><head><meta charset="utf-8">
<title>Zen · Mio</title>
<style>
  body { margin: 0; font-family: -apple-system, sans-serif; background: radial-gradient(circle at center, #1a2339 0%, #050814 60%); color: #eee; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; overflow: hidden; }
  .orb { width: 200px; height: 200px; border-radius: 50%; background: radial-gradient(circle, #60a5fa, #3b82f6); box-shadow: 0 0 80px rgba(59,130,246,0.45); animation: breath 8s infinite ease-in-out; }
  @keyframes breath { 0%,100% { transform: scale(0.65); } 50% { transform: scale(1.1); } }
  .word { margin-top: 40px; font-size: 18px; color: #9bb0d4; letter-spacing: 4px; text-transform: uppercase; }
  .timer { margin-top: 24px; font-size: 40px; font-variant-numeric: tabular-nums; font-weight: 200; }
  button { margin-top: 40px; background: transparent; border: 1px solid #334155; color: #9bb0d4; padding: 8px 20px; border-radius: 999px; cursor: pointer; font-size: 12px; letter-spacing: 2px; text-transform: uppercase; }
  button:hover { border-color: #60a5fa; color: #fff; }
</style></head>
<body>
<div class="orb"></div>
<div class="word" id="word">Inhale</div>
<div class="timer" id="timer">${String(minutes).padStart(2,'0')}:00</div>
<button onclick="window.close()">End session</button>
<script>
let total = ${minutes * 60};
let state = 'inhale';
let phase = 0;
setInterval(() => {
  phase = (phase + 1) % 8;
  if (phase < 4) { state = 'inhale'; document.getElementById('word').textContent = 'Inhale'; }
  else           { state = 'exhale'; document.getElementById('word').textContent = 'Exhale'; }
}, 1000);
setInterval(() => {
  total--;
  if (total <= 0) { document.getElementById('word').textContent = 'Done'; document.getElementById('timer').textContent = '00:00'; return; }
  const m = Math.floor(total/60), s = total%60;
  document.getElementById('timer').textContent = String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
}, 1000);
</script></body></html>`;
    const w = window.open('', 'zen', 'width=600,height=600');
    if (!w) { if (window.toast) window.toast('Popup blocked'); return; }
    w.document.open(); w.document.write(html); w.document.close();
  }

  NS.clocks = { worldclock, zen };
})();
