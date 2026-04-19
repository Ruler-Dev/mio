// Chat templates — save the current conversation as a reusable template
// and load it into a new chat.
(function () {
  const NS = (window.Mio = window.Mio || {});

  async function save() {
    const msgs = (window.chatMessages || []).filter(m => m.role !== 'system');
    if (!msgs.length) { if (window.toast) window.toast('Nothing to save'); return; }
    const name = prompt('Template name:');
    if (!name) return;
    const description = prompt('Short description (optional):') || '';
    const r = await fetch('/ui/api/chat-templates', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        name,
        description,
        messages: msgs.map(m => ({ role: m.role, content: m.content })),
      }),
    }).then(r => r.json());
    if (r.error) { if (window.toast) window.toast('Save failed: ' + r.error); return; }
    if (window.toast) window.toast('Template saved: ' + name);
  }

  async function list() {
    const { templates = [] } = await fetch('/ui/api/chat-templates').then(r => r.json());
    open(templates);
  }

  function open(templates) {
    close();
    const overlay = document.createElement('div');
    overlay.id = 'tplOverlay';
    overlay.className = 'tpl-overlay';
    overlay.innerHTML = `
      <div class="tpl-modal">
        <div class="tpl-head">
          <h2>Chat templates</h2>
          <button class="tpl-close" onclick="Mio.templates.close()">×</button>
        </div>
        <div class="tpl-body">
          ${templates.length ? templates.map(t => `
            <div class="tpl-item">
              <div class="tpl-item-body">
                <div class="tpl-name">${esc(t.name)}</div>
                <div class="tpl-meta">${t.messages.length} messages · ${esc(t.description || '')}</div>
              </div>
              <div class="tpl-actions">
                <button onclick="Mio.templates.loadOne('${t.id}')">Load</button>
                <button class="danger" onclick="Mio.templates.del('${t.id}')">Delete</button>
              </div>
            </div>
          `).join('') : '<div class="tpl-empty">No templates saved yet. Run <code>/save-template</code> to create one.</div>'}
        </div>
      </div>
    `;
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
    document.body.appendChild(overlay);
  }

  function close() {
    const o = document.getElementById('tplOverlay');
    if (o) o.remove();
  }

  async function loadOne(id) {
    const { templates = [] } = await fetch('/ui/api/chat-templates').then(r => r.json());
    const t = templates.find(x => x.id === id);
    if (!t) return;
    // Append into current chat; user can continue from there
    for (const m of t.messages) {
      window.chatMessages = window.chatMessages || [];
      window.chatMessages.push(m);
    }
    if (window.renderAllMessages) window.renderAllMessages();
    close();
    if (window.toast) window.toast('Loaded template: ' + t.name);
  }

  async function del(id) {
    if (!confirm('Delete this template?')) return;
    await fetch('/ui/api/chat-templates/' + id, { method: 'DELETE' });
    list();
  }

  function esc(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function injectCSS() {
    if (document.getElementById('tpl-css')) return;
    const css = document.createElement('style');
    css.id = 'tpl-css';
    css.textContent = `
      .tpl-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); backdrop-filter: blur(4px); z-index: 1700; display: flex; align-items: center; justify-content: center; padding: 40px; }
      .tpl-modal { background: var(--bg-chat); border: 1px solid var(--border); border-radius: 14px; width: min(640px, 100%); max-height: 80vh; display: flex; flex-direction: column; }
      .tpl-head { padding: 16px 20px; border-bottom: 1px solid var(--border-subtle); display: flex; justify-content: space-between; align-items: center; }
      .tpl-head h2 { font-size: 16px; margin: 0; }
      .tpl-close { background: transparent; border: 0; color: var(--text-muted); font-size: 20px; cursor: pointer; }
      .tpl-body { overflow-y: auto; padding: 14px 20px; }
      .tpl-item { display: flex; gap: 12px; align-items: center; padding: 12px 0; border-bottom: 1px solid var(--border-subtle); }
      .tpl-item-body { flex: 1; min-width: 0; }
      .tpl-name { font-weight: 600; color: var(--text-primary); }
      .tpl-meta { color: var(--text-muted); font-size: 12px; margin-top: 3px; }
      .tpl-actions { display: flex; gap: 6px; }
      .tpl-actions button { background: var(--accent); color: #fff; border: 0; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; }
      .tpl-actions button.danger { background: transparent; border: 1px solid #ef4444; color: #ef4444; }
      .tpl-empty { padding: 24px; text-align: center; color: var(--text-muted); }
    `;
    document.head.appendChild(css);
  }

  injectCSS();
  NS.templates = { save, list, loadOne, del, close };
})();
