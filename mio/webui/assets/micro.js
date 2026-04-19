// Small polish features bundled — scroll helpers, time-of-day greeting,
// offline indicator, user-defined slash aliases.
(function () {
  const NS = (window.Mio = window.Mio || {});

  // ---- Scroll-to-top / scroll-to-bottom buttons ----
  function initScroll() {
    const scroll = document.getElementById('messagesScroll');
    if (!scroll) return setTimeout(initScroll, 300);
    if (document.getElementById('scrollHelpers')) return;
    const wrap = document.createElement('div');
    wrap.id = 'scrollHelpers';
    wrap.className = 'scroll-helpers';
    wrap.innerHTML = `
      <button class="sh-btn" title="Top" onclick="Mio.micro.top()">↑</button>
      <button class="sh-btn" title="Bottom" onclick="Mio.micro.bottom()">↓</button>
    `;
    scroll.parentElement.appendChild(wrap);
    const update = () => {
      const tooLong = scroll.scrollHeight > scroll.clientHeight + 300;
      wrap.style.display = tooLong ? 'flex' : 'none';
    };
    scroll.addEventListener('scroll', update);
    new MutationObserver(update).observe(scroll, { childList: true, subtree: true });
    update();
  }
  function top() {
    const s = document.getElementById('messagesScroll');
    if (s) s.scrollTo({ top: 0, behavior: 'smooth' });
  }
  function bottom() {
    const s = document.getElementById('messagesScroll');
    if (s) s.scrollTo({ top: s.scrollHeight, behavior: 'smooth' });
  }

  // ---- Time-of-day greeting ----
  function greeting() {
    const h = new Date().getHours();
    if (h < 5) return 'Still up?';
    if (h < 12) return 'Good morning';
    if (h < 17) return 'Good afternoon';
    if (h < 22) return 'Good evening';
    return 'Burning the midnight oil?';
  }
  function applyGreeting() {
    const welcomeSub = document.querySelector('.welcome-sub');
    if (!welcomeSub) return;
    if (welcomeSub.dataset.greetingApplied) return;
    welcomeSub.dataset.greetingApplied = '1';
    welcomeSub.insertAdjacentHTML('afterbegin', `<div class="tod-greeting">${greeting()}.</div>`);
  }

  // ---- Offline / reconnecting banner ----
  let _wsDisconnectedAt = null;
  function pollWS() {
    const ws = window.ws;
    const state = ws ? ws.readyState : 3;
    let bar = document.getElementById('offlineBar');
    if (state === 1) {
      if (bar) bar.remove();
      _wsDisconnectedAt = null;
    } else {
      _wsDisconnectedAt = _wsDisconnectedAt || Date.now();
      if (!bar) {
        bar = document.createElement('div');
        bar.id = 'offlineBar';
        bar.className = 'offline-bar';
        document.body.appendChild(bar);
      }
      const secs = Math.floor((Date.now() - _wsDisconnectedAt) / 1000);
      bar.innerHTML = `<span class="off-dot"></span>Reconnecting to server… (${secs}s)`;
    }
  }
  setInterval(pollWS, 1000);

  // ---- User-defined slash aliases ----
  const ALIASES_KEY = 'mio-slash-aliases';
  function loadAliases() {
    try { return JSON.parse(localStorage.getItem(ALIASES_KEY) || '{}'); } catch (e) { return {}; }
  }
  function saveAliases(obj) { localStorage.setItem(ALIASES_KEY, JSON.stringify(obj)); }

  function aliasRun(text) {
    // /alias add <name> <template>
    // /alias rm <name>
    // /alias list
    const m = text.match(/^\/alias\s+(add|rm|list)(?:\s+(\S+))?\s*([\s\S]*)$/);
    if (!m) return '/alias add <name> <template>\n/alias rm <name>\n/alias list';
    const cmd = m[1];
    const name = m[2];
    const rest = (m[3] || '').trim();
    const aliases = loadAliases();
    if (cmd === 'add') {
      if (!name || !rest) { if (window.toast) window.toast('Usage: /alias add <name> <template>'); return; }
      aliases[name] = rest;
      saveAliases(aliases);
      if (window.toast) window.toast('Alias saved: /' + name);
      // Inject into SLASH_TEMPLATES for autocomplete
      if (window.SLASH_TEMPLATES) window.SLASH_TEMPLATES[name] = rest;
      return;
    }
    if (cmd === 'rm') {
      delete aliases[name];
      saveAliases(aliases);
      if (window.SLASH_TEMPLATES) delete window.SLASH_TEMPLATES[name];
      if (window.toast) window.toast('Alias removed: /' + name);
      return;
    }
    if (cmd === 'list') {
      const keys = Object.keys(aliases);
      const body = keys.length
        ? keys.map(k => `• /${k} → ${aliases[k].slice(0, 80)}${aliases[k].length > 80 ? '…' : ''}`).join('\n')
        : 'No user-defined aliases yet. Try `/alias add greet "Say hello to {{ARG}}"`.';
      if (window.appendSystemMessage) window.appendSystemMessage('**User aliases:**\n\n' + body);
      return;
    }
  }

  function applyAliasesToTemplates() {
    if (!window.SLASH_TEMPLATES) return setTimeout(applyAliasesToTemplates, 200);
    const aliases = loadAliases();
    for (const [k, v] of Object.entries(aliases)) {
      window.SLASH_TEMPLATES[k] = v;
    }
  }

  function injectCSS() {
    if (document.getElementById('micro-css')) return;
    const css = document.createElement('style');
    css.id = 'micro-css';
    css.textContent = `
      .scroll-helpers { position: absolute; bottom: 100px; right: 16px; display: none; flex-direction: column; gap: 6px; z-index: 5; }
      .sh-btn { width: 34px; height: 34px; background: var(--bg-chat); border: 1px solid var(--border); color: var(--text-secondary); border-radius: 50%; cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.15); font-size: 14px; }
      .sh-btn:hover { background: var(--bg-hover); color: var(--text-primary); border-color: var(--accent); }
      .offline-bar { position: fixed; top: 10px; left: 50%; transform: translateX(-50%); background: rgba(239,68,68,0.95); color: #fff; padding: 8px 16px; border-radius: 999px; font-size: 12px; z-index: 1950; display: flex; align-items: center; gap: 8px; box-shadow: 0 4px 18px rgba(239,68,68,0.35); }
      .off-dot { width: 8px; height: 8px; background: #fff; border-radius: 50%; animation: offpulse 1.2s infinite; }
      @keyframes offpulse { 0%,100% {opacity:1} 50%{opacity:0.25} }
      .tod-greeting { color: var(--accent); font-weight: 500; margin-bottom: 4px; font-size: 15px; }
    `;
    document.head.appendChild(css);
  }

  injectCSS();
  applyGreeting();
  initScroll();
  applyAliasesToTemplates();

  // Re-apply greeting on newChat. applyGreeting is idempotent via the
  // dataset flag, but we still check here so we don't burn CPU on
  // every single DOM mutation.
  new MutationObserver(() => {
    const sub = document.querySelector('.welcome-sub');
    if (sub && !sub.dataset.greetingApplied) applyGreeting();
  }).observe(document.body, { childList: true, subtree: true });

  NS.micro = { top, bottom, aliasRun, loadAliases };
})();
