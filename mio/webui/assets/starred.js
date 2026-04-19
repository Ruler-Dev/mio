// Starred sessions — stores a set of session IDs in localStorage and
// decorates matching rows in the sidebar with a ⭐. A filter toggle
// restricts the list to starred only.
(function () {
  const NS = (window.Mio = window.Mio || {});
  const KEY = 'mio-starred-sessions';
  const FILTER_KEY = 'mio-show-starred-only';

  function load() {
    try { return new Set(JSON.parse(localStorage.getItem(KEY) || '[]')); }
    catch (e) { return new Set(); }
  }
  function save(set) { localStorage.setItem(KEY, JSON.stringify([...set])); }

  function toggle(id) {
    const s = load();
    if (s.has(id)) s.delete(id);
    else s.add(id);
    save(s);
    redecorate();
  }

  function isStarred(id) { return load().has(id); }

  function redecorate() {
    const starred = load();
    const filterOn = localStorage.getItem(FILTER_KEY) === '1';
    document.querySelectorAll('.chat-item').forEach(row => {
      const id = row.getAttribute('data-id') || row.dataset.id;
      if (!id) return;
      let star = row.querySelector('.session-star');
      if (!star) {
        star = document.createElement('span');
        star.className = 'session-star';
        star.onclick = (e) => { e.stopPropagation(); toggle(id); };
        row.appendChild(star);
      }
      // IDEMPOTENT: only write DOM when the value actually differs.
      // Otherwise textContent=... fires a childList mutation every
      // observer tick → MutationObserver loop → page freeze.
      const want = starred.has(id) ? '★' : '☆';
      if (star.textContent !== want) star.textContent = want;
      const shouldStar = starred.has(id);
      if (star.classList.contains('starred') !== shouldStar)
        star.classList.toggle('starred', shouldStar);
      const wantDisplay = (filterOn && !starred.has(id)) ? 'none' : '';
      if (row.style.display !== wantDisplay) row.style.display = wantDisplay;
    });
    const toggleBtn = document.getElementById('starFilterBtn');
    if (toggleBtn) {
      const wantActive = filterOn;
      if (toggleBtn.classList.contains('active') !== wantActive)
        toggleBtn.classList.toggle('active', wantActive);
    }
  }

  function toggleFilter() {
    const on = localStorage.getItem(FILTER_KEY) === '1';
    localStorage.setItem(FILTER_KEY, on ? '0' : '1');
    redecorate();
  }

  function addFilterButton() {
    const sidebarActions = document.querySelector('.sidebar-header-actions');
    if (!sidebarActions || document.getElementById('starFilterBtn')) return;
    const btn = document.createElement('button');
    btn.id = 'starFilterBtn';
    btn.className = 'icon-btn';
    btn.title = 'Show starred only';
    btn.innerHTML = '⭐';
    btn.style.cssText = 'font-size:16px';
    btn.onclick = toggleFilter;
    sidebarActions.insertBefore(btn, sidebarActions.firstChild);
  }

  function injectCSS() {
    if (document.getElementById('starred-css')) return;
    const css = document.createElement('style');
    css.id = 'starred-css';
    css.textContent = `
      .session-star { position: absolute; right: 30px; top: 8px; color: var(--text-muted); font-size: 14px; cursor: pointer; opacity: 0; transition: all 120ms; padding: 2px 4px; z-index: 2; }
      .chat-item:hover .session-star, .session-star.starred { opacity: 1; }
      .session-star.starred { color: #f59e0b; }
      .chat-item { position: relative; }
      #starFilterBtn.active { background: rgba(245,158,11,0.2); border-color: #f59e0b; }
    `;
    document.head.appendChild(css);
  }

  injectCSS();
  addFilterButton();

  // Re-decorate whenever the sidebar list is re-rendered
  const mo = new MutationObserver(() => {
    if (document.querySelector('.chat-item')) {
      redecorate();
      addFilterButton();
    }
  });
  mo.observe(document.body, { childList: true, subtree: true });

  NS.starred = { toggle, isStarred, redecorate, toggleFilter };
})();
