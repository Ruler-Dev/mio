// Presentation mode — cycles through every artifact in the current chat
// as if they were slides. Arrow keys / spacebar advance. Escape exits.
(function () {
  const NS = (window.Mio = window.Mio || {});

  let _active = false;
  let _idx = 0;
  let _ids = [];

  function _orderedArtifacts() {
    // Render artifacts in chat-message insertion order; fall back to
    // allArtifacts.values() if we can't infer.
    const msgs = window.chatMessages || [];
    const seen = new Set();
    const ids = [];
    for (const m of msgs) {
      const txt = m.content || '';
      const matches = txt.matchAll(/<antArtifact[^>]*\bidentifier="([^"]+)"/g);
      for (const match of matches) {
        if (!seen.has(match[1])) { seen.add(match[1]); ids.push(match[1]); }
      }
      const ph = txt.matchAll(/\[\[ARTIFACT:([\w-]+)\]\]/g);
      for (const match of ph) {
        if (!seen.has(match[1])) { seen.add(match[1]); ids.push(match[1]); }
      }
    }
    // Any auto_artifacts not tagged in messages still show up in allArtifacts
    if (window.allArtifacts) {
      for (const id of Object.keys(window.allArtifacts)) {
        if (!seen.has(id)) { seen.add(id); ids.push(id); }
      }
    }
    return ids;
  }

  function open() {
    _ids = _orderedArtifacts();
    if (!_ids.length) { if (window.toast) window.toast('No artifacts in this chat'); return; }
    _idx = 0;
    _render();
    _active = true;
    document.body.classList.add('pres-active');
  }

  function close() {
    _active = false;
    document.body.classList.remove('pres-active');
    const root = document.getElementById('presRoot');
    if (root) root.remove();
  }

  function _render() {
    let root = document.getElementById('presRoot');
    if (!root) {
      root = document.createElement('div');
      root.id = 'presRoot';
      root.className = 'pres-root';
      root.innerHTML = `
        <div class="pres-body" id="presBody"></div>
        <div class="pres-bar">
          <button class="pres-nav" onclick="Mio.presentation.prev()">←</button>
          <span class="pres-pos" id="presPos"></span>
          <span class="pres-title" id="presTitle"></span>
          <button class="pres-nav" onclick="Mio.presentation.next()">→</button>
          <button class="pres-close" onclick="Mio.presentation.close()" title="Exit (Esc)">×</button>
        </div>`;
      document.body.appendChild(root);
    }
    const id = _ids[_idx];
    const art = window.allArtifacts && window.allArtifacts[id];
    const body = document.getElementById('presBody');
    body.innerHTML = '';
    if (art && window.renderArtifactPreview) {
      window.renderArtifactPreview(body, art);
    } else {
      body.innerHTML = '<div style="padding:40px;color:#ef4444">Artifact ' + id + ' not loaded in this chat.</div>';
    }
    document.getElementById('presPos').textContent = (_idx + 1) + ' / ' + _ids.length;
    document.getElementById('presTitle').textContent = art ? art.title : '';
  }

  function next() { if (!_active) return; _idx = (_idx + 1) % _ids.length; _render(); }
  function prev() { if (!_active) return; _idx = (_idx - 1 + _ids.length) % _ids.length; _render(); }

  document.addEventListener('keydown', (e) => {
    if (!_active) return;
    if (e.key === 'Escape') { close(); e.preventDefault(); return; }
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') { next(); e.preventDefault(); return; }
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') { prev(); e.preventDefault(); return; }
    if (e.key === 'f') { document.documentElement.requestFullscreen?.(); }
  });

  function injectCSS() {
    if (document.getElementById('pres-css')) return;
    const css = document.createElement('style');
    css.id = 'pres-css';
    css.textContent = `
      .pres-root { position: fixed; inset: 0; z-index: 2000; background: #0a0a14; display: flex; flex-direction: column; }
      .pres-body { flex: 1; overflow: hidden; padding: 40px; display: flex; align-items: stretch; justify-content: center; }
      .pres-body > * { flex: 1; max-width: 1400px; border-radius: 14px; overflow: hidden; box-shadow: 0 30px 80px rgba(0,0,0,0.7); }
      .pres-body iframe { width: 100%; height: 100%; border: 0; background: #fff; }
      .pres-bar { display: flex; align-items: center; gap: 14px; padding: 12px 24px; background: rgba(0,0,0,0.55); color: #fff; font-size: 13px; border-top: 1px solid rgba(255,255,255,0.08); }
      .pres-nav { background: rgba(255,255,255,0.1); border: 0; color: #fff; width: 36px; height: 36px; border-radius: 50%; cursor: pointer; font-size: 16px; }
      .pres-nav:hover { background: rgba(255,255,255,0.2); }
      .pres-pos { font-family: ui-monospace,monospace; font-size: 12px; color: #aaa; min-width: 60px; }
      .pres-title { flex: 1; color: #eee; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .pres-close { background: transparent; border: 0; color: #aaa; font-size: 22px; cursor: pointer; width: 36px; height: 36px; border-radius: 50%; }
      .pres-close:hover { background: rgba(239,68,68,0.2); color: #fff; }
      body.pres-active { overflow: hidden; }
    `;
    document.head.appendChild(css);
  }

  injectCSS();

  NS.presentation = { open, close, next, prev };
})();
