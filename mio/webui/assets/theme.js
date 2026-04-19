// Accent color theme chooser — swaps the CSS --accent variable + a few
// derived tones. Persists to localStorage.
(function () {
  const NS = (window.Mio = window.Mio || {});
  const KEY = 'mio-accent';

  const ACCENTS = [
    { name: 'blue',    hex: '#3b82f6', subtle: 'rgba(59,130,246,0.08)' },
    { name: 'indigo',  hex: '#4f46e5', subtle: 'rgba(79,70,229,0.08)' },
    { name: 'violet',  hex: '#7c3aed', subtle: 'rgba(124,58,237,0.08)' },
    { name: 'pink',    hex: '#ec4899', subtle: 'rgba(236,72,153,0.08)' },
    { name: 'rose',    hex: '#f43f5e', subtle: 'rgba(244,63,94,0.08)' },
    { name: 'red',     hex: '#ef4444', subtle: 'rgba(239,68,68,0.08)' },
    { name: 'orange',  hex: '#f97316', subtle: 'rgba(249,115,22,0.08)' },
    { name: 'amber',   hex: '#f59e0b', subtle: 'rgba(245,158,11,0.08)' },
    { name: 'emerald', hex: '#10b981', subtle: 'rgba(16,185,129,0.08)' },
    { name: 'teal',    hex: '#14b8a6', subtle: 'rgba(20,184,166,0.08)' },
    { name: 'cyan',    hex: '#06b6d4', subtle: 'rgba(6,182,212,0.08)' },
    { name: 'slate',   hex: '#475569', subtle: 'rgba(71,85,105,0.08)' },
  ];

  function apply(name) {
    const a = ACCENTS.find(x => x.name === name) || ACCENTS[0];
    document.documentElement.style.setProperty('--accent', a.hex);
    document.documentElement.style.setProperty('--accent-subtle', a.subtle);
    localStorage.setItem(KEY, a.name);
  }

  function current() { return localStorage.getItem(KEY) || 'blue'; }

  function open() {
    close();
    const cur = current();
    const overlay = document.createElement('div');
    overlay.id = 'themeOverlay';
    overlay.className = 'theme-overlay';
    overlay.innerHTML = `
      <div class="theme-modal">
        <h2>Accent color</h2>
        <p>Tap a swatch to apply instantly.</p>
        <div class="theme-grid">
          ${ACCENTS.map(a => `
            <button class="theme-swatch ${cur === a.name ? 'active' : ''}"
              style="background:${a.hex}"
              title="${a.name}"
              onclick="Mio.theme.apply('${a.name}'); Mio.theme.open();"></button>
          `).join('')}
        </div>
        <div class="theme-actions">
          <button class="theme-close" onclick="Mio.theme.close()">Done</button>
        </div>
      </div>
    `;
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
    document.body.appendChild(overlay);
  }

  function close() {
    const o = document.getElementById('themeOverlay');
    if (o) o.remove();
  }

  function injectCSS() {
    if (document.getElementById('theme-css')) return;
    const css = document.createElement('style');
    css.id = 'theme-css';
    css.textContent = `
      .theme-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); backdrop-filter: blur(4px); z-index: 1650; display: flex; align-items: center; justify-content: center; }
      .theme-modal { background: var(--bg-chat); border: 1px solid var(--border); border-radius: 14px; padding: 20px; width: min(360px, 100%); }
      .theme-modal h2 { margin: 0 0 8px; font-size: 16px; }
      .theme-modal p { color: var(--text-muted); font-size: 12px; margin: 0 0 14px; }
      .theme-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
      .theme-swatch { width: 100%; aspect-ratio: 1; border: 2px solid var(--border); border-radius: 50%; cursor: pointer; transition: transform 120ms, box-shadow 120ms; }
      .theme-swatch:hover { transform: scale(1.08); }
      .theme-swatch.active { border-color: #fff; box-shadow: 0 0 0 3px var(--accent); }
      .theme-actions { display: flex; justify-content: flex-end; margin-top: 16px; }
      .theme-close { background: var(--accent); color: #fff; border: 0; padding: 7px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; }
    `;
    document.head.appendChild(css);
  }

  injectCSS();
  apply(current());

  NS.theme = { open, close, apply, current, accents: ACCENTS };
})();
