// Pinned messages — renders a sticky bar at the top of the chat showing
// every message flagged with `pinned: true` in chatMessages. The pin
// state itself is toggled by the existing pinMessage() in mio_ui.html
// (which also persists via autoSave), so this module is rendering-only.
(function () {
  const NS = (window.Mio = window.Mio || {});

  function pinned() {
    const msgs = window.chatMessages || [];
    return msgs
      .map((m, i) => ({ m, i }))
      .filter(({ m }) => m && m.pinned);
  }

  function render() {
    const items = pinned();
    let bar = document.getElementById('pinnedBar');
    if (!items.length) { if (bar) bar.remove(); return; }
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'pinnedBar';
      bar.className = 'pinned-bar';
      const scroll = document.getElementById('messagesScroll');
      if (scroll) scroll.parentElement.insertBefore(bar, scroll);
    }
    bar.innerHTML =
      '<div class="pinned-bar-head">📌 Pinned</div>' +
      items.map(({ m, i }) => {
        const snippet = (m.content || '')
          .replace(/<antArtifact[\s\S]*?<\/antArtifact>/g, '[artifact]')
          .replace(/<[^>]+>/g, '')
          .replace(/\s+/g, ' ')
          .slice(0, 120);
        return `<div class="pinned-item">
          <span class="pinned-role">${m.role === 'user' ? 'You' : 'Mio'}</span>
          <span class="pinned-text" onclick="Mio.pinned.scrollTo(${i})">${escapeHTML(snippet)}</span>
          <button class="pinned-rm" onclick="pinMessage(${i})" title="Unpin">×</button>
        </div>`;
      }).join('');
  }

  function scrollTo(msgIdx) {
    const all = document.querySelectorAll('#messages > .message');
    const target = all[msgIdx];
    if (!target) return;
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    target.style.transition = 'background 600ms';
    const prev = target.style.background;
    target.style.background = 'rgba(245,158,11,0.14)';
    setTimeout(() => { target.style.background = prev; }, 1600);
  }

  function escapeHTML(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function injectCSS() {
    if (document.getElementById('pinned-css')) return;
    const css = document.createElement('style');
    css.id = 'pinned-css';
    css.textContent = `
      .pinned-bar { position: sticky; top: 0; z-index: 10; background: var(--bg-chat); backdrop-filter: blur(8px); border-bottom: 1px solid var(--border-subtle); padding: 8px 20px 10px; max-width: 780px; margin: 0 auto; font-size: 12px; }
      .pinned-bar-head { color: #f59e0b; font-weight: 600; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px; }
      .pinned-item { display: flex; gap: 8px; align-items: center; padding: 4px 8px; border-radius: 6px; }
      .pinned-item:hover { background: var(--bg-hover); }
      .pinned-role { font-weight: 600; color: var(--text-secondary); font-size: 10px; width: 32px; flex: 0 0 auto; }
      .pinned-text { flex: 1; color: var(--text-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: pointer; }
      .pinned-text:hover { text-decoration: underline; }
      .pinned-rm { background: transparent; border: 0; color: var(--text-muted); font-size: 14px; cursor: pointer; padding: 0 4px; border-radius: 4px; }
      .pinned-rm:hover { color: #ef4444; background: rgba(239,68,68,0.1); }
    `;
    document.head.appendChild(css);
  }

  NS.pinned = { render, scrollTo, init: injectCSS };
  injectCSS();
})();
