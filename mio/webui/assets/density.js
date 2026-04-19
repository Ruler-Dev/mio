// Density toggle — compact vs comfortable vs cozy spacing for the chat.
// Saved to localStorage; applied by toggling a class on <html>.
(function () {
  const NS = (window.Mio = window.Mio || {});
  const KEY = 'mio-density';
  const MODES = ['comfortable', 'compact', 'cozy'];

  function apply(mode) {
    document.documentElement.setAttribute('data-density', mode || 'comfortable');
    localStorage.setItem(KEY, mode);
  }

  function get() {
    return localStorage.getItem(KEY) || 'comfortable';
  }

  function cycle() {
    const cur = get();
    const next = MODES[(MODES.indexOf(cur) + 1) % MODES.length];
    apply(next);
    if (window.toast) window.toast('Density: ' + next);
  }

  function injectCSS() {
    if (document.getElementById('density-css')) return;
    const css = document.createElement('style');
    css.id = 'density-css';
    css.textContent = `
      /* comfortable is the default — no overrides needed */
      html[data-density="compact"] .message { padding: 8px 20px; }
      html[data-density="compact"] .msg-content { font-size: 13px; line-height: 1.5; }
      html[data-density="compact"] .msg-body { gap: 4px; }
      html[data-density="compact"] .msg-actions { padding-top: 4px; margin-top: 4px; }
      html[data-density="compact"] .messages-inner { gap: 4px; padding: 8px 0; }
      html[data-density="compact"] .tool-panel { margin: 4px 0; }
      html[data-density="compact"] .artifact-card { padding: 8px 12px; }

      html[data-density="cozy"] .message { padding: 24px 20px; }
      html[data-density="cozy"] .msg-content { font-size: 15.5px; line-height: 1.8; }
      html[data-density="cozy"] .messages-inner { gap: 18px; padding: 24px 0; }
    `;
    document.head.appendChild(css);
  }

  injectCSS();
  apply(get());

  NS.density = { apply, get, cycle, modes: MODES };
})();
