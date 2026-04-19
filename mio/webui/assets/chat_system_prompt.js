// Per-chat system prompt override — each session can set its own SP
// that wins over the global settings. Persisted into the session
// payload under `chat_system_prompt`.
(function () {
  const NS = (window.Mio = window.Mio || {});

  function open() {
    close();
    const cur = window.currentChatSystemPrompt || '';
    const overlay = document.createElement('div');
    overlay.id = 'csOverlay';
    overlay.className = 'cs-overlay';
    overlay.innerHTML = `
      <div class="cs-modal">
        <h2>System prompt for this chat</h2>
        <p>Leave empty to fall back to the global setting. Any active persona will be merged in front of this on send.</p>
        <textarea id="csText" placeholder="You are an expert…">${esc(cur)}</textarea>
        <div class="cs-actions">
          <button class="cs-ghost" onclick="Mio.chatSystemPrompt.close()">Cancel</button>
          <button class="cs-ghost cs-danger" onclick="Mio.chatSystemPrompt.clear()">Clear</button>
          <button class="cs-primary" onclick="Mio.chatSystemPrompt.save()">Save</button>
        </div>
      </div>
    `;
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
    document.body.appendChild(overlay);
    setTimeout(() => document.getElementById('csText')?.focus(), 60);
  }

  function close() {
    const o = document.getElementById('csOverlay');
    if (o) o.remove();
  }

  function save() {
    const v = document.getElementById('csText')?.value || '';
    window.currentChatSystemPrompt = v.trim();
    if (window.autoSave) window.autoSave();
    close();
    if (window.toast) window.toast(v.trim() ? 'Chat system prompt saved' : 'Chat system prompt cleared');
  }

  function clear() {
    window.currentChatSystemPrompt = '';
    if (window.autoSave) window.autoSave();
    close();
    if (window.toast) window.toast('Chat system prompt cleared');
  }

  function esc(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function injectCSS() {
    if (document.getElementById('cs-css')) return;
    const css = document.createElement('style');
    css.id = 'cs-css';
    css.textContent = `
      .cs-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); backdrop-filter: blur(4px); z-index: 1650; display: flex; align-items: center; justify-content: center; padding: 40px; }
      .cs-modal { background: var(--bg-chat); border: 1px solid var(--border); border-radius: 14px; width: min(640px, 100%); padding: 20px; }
      .cs-modal h2 { font-size: 16px; margin: 0 0 8px; }
      .cs-modal p { color: var(--text-muted); font-size: 12px; margin: 0 0 12px; }
      .cs-modal textarea { width: 100%; min-height: 200px; background: var(--bg-input); border: 1px solid var(--border); color: var(--text-primary); padding: 10px 12px; border-radius: 8px; font-size: 13px; font-family: ui-monospace,monospace; resize: vertical; }
      .cs-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 14px; }
      .cs-ghost, .cs-primary { background: transparent; border: 1px solid var(--border); color: var(--text-secondary); padding: 7px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; }
      .cs-ghost.cs-danger { color: #ef4444; border-color: rgba(239,68,68,0.3); }
      .cs-primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    `;
    document.head.appendChild(css);
  }

  injectCSS();
  NS.chatSystemPrompt = { open, close, save, clear };
})();
