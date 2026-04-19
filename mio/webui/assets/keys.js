// /keys — full-screen searchable list of every keyboard shortcut +
// slash command. Bound to Cmd+/ or Ctrl+/ for instant access.
(function () {
  const NS = (window.Mio = window.Mio || {});

  const SHORTCUTS = [
    { keys: '⌘K', action: 'Open command palette' },
    { keys: '⌘N', action: 'Start a new chat' },
    { keys: '⌘,', action: 'Open Settings' },
    { keys: '⌘⇧V', action: 'Paste clipboard as hidden context' },
    { keys: '⌘/', action: 'Show this shortcuts sheet' },
    { keys: '?', action: 'Open the cheatsheet' },
    { keys: 'Enter', action: 'Send message' },
    { keys: 'Shift+Enter', action: 'Insert a newline' },
    { keys: 'Tab', action: 'Accept highlighted slash command' },
    { keys: 'Esc', action: 'Close any overlay' },
    { keys: 'Drop file', action: 'Attach a PDF / doc / image' },
    { keys: 'Drop image', action: 'Attach as multimodal input' },
    { keys: 'Paste image', action: 'Attach clipboard image' },
  ];

  function escapeHTML(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function open() {
    close();
    const overlay = document.createElement('div');
    overlay.className = 'keys-overlay';
    overlay.id = 'keysOverlay';
    overlay.innerHTML = `
      <div class="keys-modal">
        <div class="keys-head">
          <h2>Keyboard shortcuts & slash commands</h2>
          <input id="keysFilter" placeholder="Filter…" oninput="Mio.keys.filter()">
          <button class="keys-close" onclick="Mio.keys.close()">×</button>
        </div>
        <div class="keys-body" id="keysBody"></div>
      </div>
    `;
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
    document.body.appendChild(overlay);
    renderList();
    setTimeout(() => document.getElementById('keysFilter')?.focus(), 50);
  }

  function close() {
    const o = document.getElementById('keysOverlay');
    if (o) o.remove();
  }

  function renderList() {
    const q = (document.getElementById('keysFilter')?.value || '').toLowerCase();
    const body = document.getElementById('keysBody');
    if (!body) return;
    const templates = window.SLASH_TEMPLATES || {};
    const slashList = Object.entries(templates).map(([k, v]) => ({
      keys: '/' + k, action: v.replace('{{ARG}}', '<arg>'),
    }));
    const all = [...SHORTCUTS.map(x => ({...x, kind: 'shortcut'})),
                 ...slashList.map(x => ({...x, kind: 'slash'}))];
    const filtered = q ? all.filter(x => x.keys.toLowerCase().includes(q) || x.action.toLowerCase().includes(q)) : all;
    const shortcuts = filtered.filter(x => x.kind === 'shortcut');
    const slashes = filtered.filter(x => x.kind === 'slash');
    body.innerHTML = `
      ${shortcuts.length ? `
        <h3>Shortcuts</h3>
        <table><tbody>${shortcuts.map(s =>
          `<tr><td><kbd>${escapeHTML(s.keys)}</kbd></td><td>${escapeHTML(s.action)}</td></tr>`
        ).join('')}</tbody></table>` : ''}
      ${slashes.length ? `
        <h3>Slash commands (${slashes.length})</h3>
        <table><tbody>${slashes.map(s =>
          `<tr><td><code>${escapeHTML(s.keys)}</code></td><td>${escapeHTML(s.action)}</td></tr>`
        ).join('')}</tbody></table>` : ''}
      ${!shortcuts.length && !slashes.length ? '<div class="keys-empty">No matches.</div>' : ''}
    `;
  }

  function filter() { renderList(); }

  function injectCSS() {
    if (document.getElementById('keys-css')) return;
    const css = document.createElement('style');
    css.id = 'keys-css';
    css.textContent = `
      .keys-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); backdrop-filter: blur(4px); z-index: 1500; display: flex; align-items: center; justify-content: center; padding: 40px; }
      .keys-modal { background: var(--bg-chat); border: 1px solid var(--border); border-radius: 14px; width: min(760px, 100%); max-height: 80vh; display: flex; flex-direction: column; overflow: hidden; }
      .keys-head { padding: 16px 20px; border-bottom: 1px solid var(--border-subtle); display: flex; align-items: center; gap: 12px; }
      .keys-head h2 { font-size: 16px; margin: 0; flex: 1; }
      .keys-head input { flex: 0 0 220px; background: var(--bg-input); border: 1px solid var(--border); color: var(--text-primary); padding: 6px 10px; border-radius: 6px; font-size: 13px; }
      .keys-close { background: transparent; border: 0; color: var(--text-muted); font-size: 20px; cursor: pointer; width: 28px; height: 28px; border-radius: 4px; }
      .keys-close:hover { color: var(--text-primary); background: var(--bg-hover); }
      .keys-body { overflow-y: auto; padding: 14px 20px; }
      .keys-body h3 { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.6px; margin: 14px 0 8px; }
      .keys-body table { width: 100%; border-collapse: collapse; }
      .keys-body td { padding: 6px 8px; border-bottom: 1px solid var(--border-subtle); font-size: 13px; }
      .keys-body td:first-child { width: 180px; white-space: nowrap; }
      .keys-body kbd, .keys-body code { background: var(--bg-input); border: 1px solid var(--border); padding: 2px 8px; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 11.5px; color: var(--accent); }
      .keys-empty { padding: 40px; text-align: center; color: var(--text-muted); }
    `;
    document.head.appendChild(css);
  }

  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === '/') {
      e.preventDefault();
      open();
    }
    if (e.key === 'Escape') close();
  });

  injectCSS();
  NS.keys = { open, close, filter };
})();
