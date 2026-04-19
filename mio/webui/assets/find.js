// In-chat find bar — Cmd+F opens a floating search input that highlights
// and navigates matches within the current chat messages.
(function () {
  const NS = (window.Mio = window.Mio || {});

  let _matches = [];
  let _idx = 0;

  function open() {
    if (document.getElementById('findBar')) {
      document.getElementById('findInput')?.focus();
      return;
    }
    const bar = document.createElement('div');
    bar.id = 'findBar';
    bar.className = 'find-bar';
    bar.innerHTML = `
      <input id="findInput" placeholder="Find in chat…" oninput="Mio.find.search()">
      <span id="findCount">0 / 0</span>
      <button onclick="Mio.find.prev()" title="Previous">↑</button>
      <button onclick="Mio.find.next()" title="Next">↓</button>
      <button onclick="Mio.find.close()" title="Close">×</button>
    `;
    document.body.appendChild(bar);
    setTimeout(() => document.getElementById('findInput')?.focus(), 50);
  }

  function close() {
    const b = document.getElementById('findBar');
    if (b) b.remove();
    _clearHighlights();
  }

  function _clearHighlights() {
    document.querySelectorAll('.find-hit').forEach(el => {
      const parent = el.parentNode;
      if (!parent) return;
      parent.replaceChild(document.createTextNode(el.textContent), el);
      parent.normalize();
    });
    _matches = [];
    _idx = 0;
    const cnt = document.getElementById('findCount');
    if (cnt) cnt.textContent = '0 / 0';
  }

  function search() {
    _clearHighlights();
    const q = (document.getElementById('findInput')?.value || '').trim();
    if (!q) return;
    const re = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
    const nodes = document.querySelectorAll('#messages .msg-content');
    nodes.forEach(n => highlightIn(n, re));
    _matches = Array.from(document.querySelectorAll('.find-hit'));
    _idx = 0;
    if (_matches.length) _focus(_idx);
    const cnt = document.getElementById('findCount');
    if (cnt) cnt.textContent = (_matches.length ? '1' : '0') + ' / ' + _matches.length;
  }

  function highlightIn(root, re) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const targets = [];
    let node;
    while ((node = walker.nextNode())) {
      if (re.test(node.nodeValue)) targets.push(node);
      re.lastIndex = 0;
    }
    for (const n of targets) {
      const frag = document.createDocumentFragment();
      let last = 0;
      const v = n.nodeValue;
      re.lastIndex = 0;
      let m;
      while ((m = re.exec(v))) {
        if (m.index > last) frag.appendChild(document.createTextNode(v.slice(last, m.index)));
        const span = document.createElement('mark');
        span.className = 'find-hit';
        span.textContent = m[0];
        frag.appendChild(span);
        last = m.index + m[0].length;
        if (re.lastIndex === m.index) re.lastIndex++;
      }
      if (last < v.length) frag.appendChild(document.createTextNode(v.slice(last)));
      n.parentNode.replaceChild(frag, n);
    }
  }

  function _focus(i) {
    _matches.forEach((m, j) => m.classList.toggle('find-active', i === j));
    if (_matches[i]) _matches[i].scrollIntoView({ behavior: 'smooth', block: 'center' });
    const cnt = document.getElementById('findCount');
    if (cnt) cnt.textContent = (i + 1) + ' / ' + _matches.length;
  }

  function next() { if (!_matches.length) return; _idx = (_idx + 1) % _matches.length; _focus(_idx); }
  function prev() { if (!_matches.length) return; _idx = (_idx - 1 + _matches.length) % _matches.length; _focus(_idx); }

  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'f') {
      if (document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)
          && document.activeElement.id !== 'findInput') {
        // Browser native find is fine in input fields
        return;
      }
      e.preventDefault();
      open();
    }
    if (e.key === 'Escape' && document.getElementById('findBar')) {
      close();
      e.preventDefault();
    }
    if (document.getElementById('findBar') && e.key === 'Enter') {
      e.preventDefault();
      if (e.shiftKey) prev(); else next();
    }
  });

  function injectCSS() {
    if (document.getElementById('find-css')) return;
    const css = document.createElement('style');
    css.id = 'find-css';
    css.textContent = `
      .find-bar { position: fixed; top: 12px; right: 12px; background: var(--bg-chat); border: 1px solid var(--border); border-radius: 10px; padding: 6px 8px; display: flex; gap: 6px; align-items: center; z-index: 1900; box-shadow: 0 4px 18px rgba(0,0,0,0.35); }
      .find-bar input { background: var(--bg-input); border: 1px solid var(--border); color: var(--text-primary); padding: 5px 10px; border-radius: 6px; font-size: 13px; width: 220px; outline: 0; }
      .find-bar span { color: var(--text-muted); font-family: ui-monospace,monospace; font-size: 11px; min-width: 48px; text-align: center; }
      .find-bar button { background: transparent; border: 0; color: var(--text-secondary); width: 26px; height: 26px; border-radius: 4px; cursor: pointer; font-size: 14px; }
      .find-bar button:hover { background: var(--bg-hover); color: var(--text-primary); }
      mark.find-hit { background: rgba(245,158,11,0.35); color: inherit; border-radius: 2px; padding: 0 1px; }
      mark.find-active { background: #f59e0b; color: #111; }
    `;
    document.head.appendChild(css);
  }

  injectCSS();
  NS.find = { open, close, search, next, prev };
})();
