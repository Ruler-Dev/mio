// Message reactions — each message can carry a bag of emoji reactions.
// Click-to-toggle. Data lives on the chatMessage and is persisted via
// autoSave. A compact picker opens on the "+" affordance.
(function () {
  const NS = (window.Mio = window.Mio || {});

  const EMOJI = ["👍", "❤️", "🔖", "🎯", "🔥", "🤯", "🤔", "😂", "👀", "✅"];

  function toggle(idx, emoji) {
    const m = window.chatMessages && window.chatMessages[idx];
    if (!m) return;
    m.reactions = m.reactions || {};
    m.reactions[emoji] = !m.reactions[emoji];
    if (!m.reactions[emoji]) delete m.reactions[emoji];
    if (window.autoSave) window.autoSave();
    // Re-render just the one message if possible
    if (window.renderAllMessages) window.renderAllMessages();
  }

  function renderOn(msgEl, idx) {
    const m = window.chatMessages && window.chatMessages[idx];
    if (!m) return;
    const body = msgEl.querySelector('.msg-body');
    if (!body || body.querySelector('.reactions')) return;
    const container = document.createElement('div');
    container.className = 'reactions';
    const rx = m.reactions || {};
    const active = Object.keys(rx).filter(k => rx[k]);
    const chips = active.map(e =>
      `<span class="rx-chip active" onclick="Mio.reactions.toggle(${idx}, '${e}')">${e}</span>`
    ).join('');
    const picker = `<span class="rx-add" onclick="Mio.reactions.openPicker(${idx}, this)" title="Add reaction">＋</span>`;
    container.innerHTML = chips + picker;
    body.appendChild(container);
  }

  function openPicker(idx, anchor) {
    closePicker();
    const pop = document.createElement('div');
    pop.className = 'rx-popup';
    pop.id = 'rxPopup';
    pop.innerHTML = EMOJI.map(e =>
      `<span class="rx-choice" onclick="Mio.reactions.toggle(${idx}, '${e}'); Mio.reactions.closePicker();">${e}</span>`
    ).join('');
    document.body.appendChild(pop);
    const r = anchor.getBoundingClientRect();
    pop.style.top = (r.top - pop.offsetHeight - 8) + 'px';
    pop.style.left = r.left + 'px';
    setTimeout(() => {
      document.addEventListener('click', _dismissOnOutside, true);
    }, 0);
  }
  function _dismissOnOutside(e) {
    if (!e.target.closest('#rxPopup') && !e.target.classList.contains('rx-add')) {
      closePicker();
    }
  }
  function closePicker() {
    const p = document.getElementById('rxPopup');
    if (p) p.remove();
    document.removeEventListener('click', _dismissOnOutside, true);
  }

  function injectCSS() {
    if (document.getElementById('rx-css')) return;
    const css = document.createElement('style');
    css.id = 'rx-css';
    css.textContent = `
      .reactions { display: flex; gap: 4px; margin-top: 6px; flex-wrap: wrap; }
      .rx-chip, .rx-add { display: inline-flex; align-items: center; justify-content: center; min-width: 28px; height: 24px; padding: 0 8px; border-radius: 12px; border: 1px solid var(--border); background: var(--bg-chat); font-size: 12px; cursor: pointer; transition: all 120ms; }
      .rx-chip:hover, .rx-add:hover { border-color: var(--accent); background: var(--bg-hover); }
      .rx-chip.active { border-color: var(--accent); background: rgba(59,130,246,0.12); }
      .rx-add { color: var(--text-muted); font-size: 14px; line-height: 1; }
      .rx-popup { position: fixed; background: var(--bg-chat); border: 1px solid var(--border); border-radius: 10px; padding: 6px; z-index: 100; display: flex; gap: 2px; box-shadow: 0 8px 30px rgba(0,0,0,0.3); }
      .rx-choice { font-size: 18px; padding: 4px 6px; border-radius: 4px; cursor: pointer; }
      .rx-choice:hover { background: var(--bg-hover); }
    `;
    document.head.appendChild(css);
  }

  // Auto-render on each msg-actions appearance
  const obs = new MutationObserver(muts => {
    for (const mu of muts) {
      for (const node of mu.addedNodes) {
        if (node.nodeType !== 1) continue;
        const arr = node.matches && node.matches('.msg-actions') ? [node]
          : (node.querySelectorAll ? Array.from(node.querySelectorAll('.msg-actions')) : []);
        for (const act of arr) {
          const msgEl = act.closest('.message');
          if (!msgEl) continue;
          const siblings = document.querySelectorAll('#messages > .message');
          const idx = Array.from(siblings).indexOf(msgEl);
          if (idx >= 0) renderOn(msgEl, idx);
        }
      }
    }
  });
  obs.observe(document.body, { childList: true, subtree: true });

  injectCSS();
  NS.reactions = { toggle, openPicker, closePicker, render: renderOn };
})();
