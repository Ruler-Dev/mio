// Branching — each regenerate keeps the prior reply as a sibling
// branch; arrow buttons on the message navigate between branches
// without losing earlier versions.
//
// Data model: chatMessages[idx] may have
//   m.branches = [ "first reply", "second reply", "third reply" ]
//   m.branchIdx = 2   // which branch is currently shown as m.content
// When branchIdx changes we set m.content = m.branches[branchIdx] and
// re-render from that index onward.
(function () {
  const NS = (window.Mio = window.Mio || {});

  function snapshotBefore(idx) {
    const m = window.chatMessages && window.chatMessages[idx];
    if (!m || m.role !== 'assistant' || !m.content) return;
    m.branches = m.branches || [m.content];
    if (!m.branches.includes(m.content)) m.branches.push(m.content);
  }

  function appendNewBranch(idx, newContent) {
    const m = window.chatMessages && window.chatMessages[idx];
    if (!m) return;
    m.branches = m.branches || [];
    if (!m.branches.includes(newContent)) m.branches.push(newContent);
    m.branchIdx = m.branches.length - 1;
  }

  function switchBranch(idx, delta) {
    const m = window.chatMessages && window.chatMessages[idx];
    if (!m || !m.branches || m.branches.length < 2) return;
    const cur = (m.branchIdx == null) ? m.branches.indexOf(m.content) : m.branchIdx;
    const n = m.branches.length;
    const next = (cur + delta + n) % n;
    m.branchIdx = next;
    m.content = m.branches[next];
    if (window.renderAllMessages) window.renderAllMessages();
    if (window.autoSave) window.autoSave();
  }

  function maybeRenderArrows(msgEl, idx) {
    const m = window.chatMessages && window.chatMessages[idx];
    if (!m || m.role !== 'assistant' || !m.branches || m.branches.length < 2) return;
    const actions = msgEl.querySelector('.msg-actions');
    if (!actions || actions.querySelector('.branch-nav')) return;
    const cur = (m.branchIdx == null) ? m.branches.indexOf(m.content) : m.branchIdx;
    const nav = document.createElement('div');
    nav.className = 'branch-nav';
    nav.innerHTML = `
      <button class="msg-action-btn" onclick="Mio.branching.switch(${idx}, -1)">◀</button>
      <span class="branch-pos">${cur + 1} / ${m.branches.length}</span>
      <button class="msg-action-btn" onclick="Mio.branching.switch(${idx}, 1)">▶</button>
    `;
    actions.appendChild(nav);
  }

  function injectCSS() {
    if (document.getElementById('branch-css')) return;
    const css = document.createElement('style');
    css.id = 'branch-css';
    css.textContent = `
      .branch-nav { display: inline-flex; align-items: center; gap: 4px; margin-left: 8px; padding-left: 8px; border-left: 1px solid var(--border-subtle); }
      .branch-nav .msg-action-btn { padding: 2px 8px; font-size: 11px; }
      .branch-pos { font-size: 11px; color: var(--text-muted); font-family: ui-monospace, monospace; min-width: 28px; text-align: center; }
    `;
    document.head.appendChild(css);
  }

  // Hook appendMsgActions (called inside renderAllMessages) without
  // monkey-patching: run a MutationObserver to catch new .msg-actions
  // nodes and attach arrows when applicable.
  const obs = new MutationObserver((muts) => {
    for (const mu of muts) {
      for (const node of mu.addedNodes) {
        if (node.nodeType !== 1) continue;
        const actionsList = node.matches && node.matches('.msg-actions') ? [node]
          : (node.querySelectorAll ? Array.from(node.querySelectorAll('.msg-actions')) : []);
        for (const act of actionsList) {
          const msgEl = act.closest('.message');
          if (!msgEl) continue;
          // Message index is its position among .message siblings under #messages
          const siblings = document.querySelectorAll('#messages > .message');
          const idx = Array.from(siblings).indexOf(msgEl);
          if (idx >= 0) maybeRenderArrows(msgEl, idx);
        }
      }
    }
  });
  obs.observe(document.body, { childList: true, subtree: true });

  injectCSS();

  NS.branching = { snapshot: snapshotBefore, append: appendNewBranch, switch: switchBranch };
})();
