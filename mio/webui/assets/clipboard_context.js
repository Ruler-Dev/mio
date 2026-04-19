// Clipboard as context — Cmd+Shift+V pastes the clipboard as a hidden
// system-context message, not a visible user prompt. Useful for "look at
// this article, then answer my next question". A chip above the input
// shows active context with preview + × to clear.
(function () {
  const NS = (window.Mio = window.Mio || {});

  let _context = null;  // { text, addedAt }

  async function capture() {
    try {
      const text = await navigator.clipboard.readText();
      if (!text.trim()) {
        if (window.toast) window.toast('Clipboard is empty');
        return;
      }
      _context = { text: text.trim(), addedAt: Date.now() };
      renderChip();
      if (window.toast) window.toast('Clipboard attached as context (' + text.length.toLocaleString() + ' chars)');
    } catch (e) {
      if (window.toast) window.toast('Clipboard read failed — check browser permission');
    }
  }

  function clear() {
    _context = null;
    renderChip();
  }

  function getContext() {
    return _context ? _context.text : null;
  }

  // Called by mio_ui.html's sendMessage path to consume context for this
  // send; the chip disappears automatically after send.
  function consume() {
    const c = _context ? _context.text : null;
    _context = null;
    renderChip();
    return c;
  }

  function renderChip() {
    let bar = document.getElementById('clipboardCtxChip');
    if (!_context) { if (bar) bar.remove(); return; }
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'clipboardCtxChip';
      bar.className = 'clipctx-chip';
      const inputArea = document.querySelector('.input-area');
      if (inputArea) inputArea.insertBefore(bar, inputArea.firstChild);
    }
    const preview = _context.text.replace(/\s+/g, ' ').slice(0, 90);
    bar.innerHTML = `
      <span class="clipctx-icon">📋</span>
      <span class="clipctx-label">Clipboard attached</span>
      <span class="clipctx-preview">${escapeHTML(preview)}${_context.text.length > 90 ? '…' : ''}</span>
      <span class="clipctx-size">${_context.text.length.toLocaleString()} chars</span>
      <button class="clipctx-clear" onclick="Mio.clipboardContext.clear()" title="Clear">×</button>
    `;
  }

  function escapeHTML(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function injectCSS() {
    if (document.getElementById('clipctx-css')) return;
    const css = document.createElement('style');
    css.id = 'clipctx-css';
    css.textContent = `
      .clipctx-chip { display: flex; align-items: center; gap: 8px; background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.35); padding: 6px 12px; border-radius: 10px; font-size: 12px; color: var(--text-primary); margin-bottom: 8px; }
      .clipctx-icon { font-size: 14px; }
      .clipctx-label { font-weight: 600; color: var(--accent); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
      .clipctx-preview { flex: 1; color: var(--text-muted); font-family: var(--font-mono); font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .clipctx-size { color: var(--text-muted); font-family: var(--font-mono); font-size: 10.5px; }
      .clipctx-clear { background: transparent; border: 0; color: var(--text-muted); font-size: 14px; cursor: pointer; padding: 0 4px; border-radius: 4px; }
      .clipctx-clear:hover { color: var(--text-primary); background: rgba(255,255,255,0.08); }
    `;
    document.head.appendChild(css);
  }

  injectCSS();

  // Cmd+Shift+V / Ctrl+Shift+V global hotkey
  document.addEventListener('keydown', (e) => {
    const metaOrCtrl = e.metaKey || e.ctrlKey;
    if (metaOrCtrl && e.shiftKey && e.key.toLowerCase() === 'v') {
      e.preventDefault();
      capture();
    }
  });

  NS.clipboardContext = { capture, clear, get: getContext, consume, render: renderChip };
})();
