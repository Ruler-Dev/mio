// First-run onboarding tour — 6 steps highlighting artifacts, slash
// commands, personas, dashboard, keyboard shortcuts, and cache. Shown
// automatically on first visit; replayable via /tour.
(function () {
  const NS = (window.Mio = window.Mio || {});
  const SEEN_KEY = 'mio-onboarded-v1';
  const SOVEREIGNTY_SEEN_KEY = 'mio.sovereignty.onboarded.v1';

  const STEPS = [
    {
      title: "Welcome to Mio",
      body: "Mio runs 100% locally on Apple Silicon via MLX. No keys, no cloud. Let's walk through the powers.",
      icon: "👋",
    },
    {
      title: "Slash commands",
      body: "Type / in the message box to search Mio's live command and template catalog for documents, diagrams, 3D scenes, artifacts, and more.",
      icon: "/",
    },
    {
      title: "Personas",
      body: "Try /as teacher, /as skeptic, /as haiku, /as pirate — 27 distinct voices swap the system prompt for the current chat. /as-list for all.",
      icon: "🎭",
    },
    {
      title: "Artifacts",
      body: "Anything visual or interactive opens in the artifact panel, including native MLX benchmarks, model cards, inference traces, speculative acceptance atlases, React, charts, Mermaid, maps, and 3D scenes.",
      icon: "✨",
    },
    {
      title: "Dashboard",
      body: "/dashboard manages scheduled prompts, webhook triggers, and indexed local folders (for RAG over your notes or code).",
      icon: "🗂️",
    },
    {
      title: "You're set",
      body: "Press ⌘/ any time to see every shortcut. Click Dismiss — or replay later with /tour.",
      icon: "🚀",
    },
  ];

  function shouldShow() {
    return !localStorage.getItem(SEEN_KEY);
  }

  function markSeen() {
    localStorage.setItem(SEEN_KEY, String(Date.now()));
  }

  let _step = 0;

  function open() {
    _step = 0;
    close();
    const overlay = document.createElement('div');
    overlay.id = 'tourOverlay';
    overlay.className = 'tour-overlay';
    overlay.innerHTML = `
      <div class="tour-card">
        <div class="tour-progress" id="tourProgress"></div>
        <div class="tour-body" id="tourBody"></div>
        <div class="tour-actions">
          <button class="tour-btn tour-skip" onclick="Mio.tour.dismiss()">Skip</button>
          <button class="tour-btn tour-prev" onclick="Mio.tour.prev()">Back</button>
          <button class="tour-btn tour-next tour-primary" onclick="Mio.tour.next()">Next</button>
        </div>
      </div>
    `;
    overlay.addEventListener('click', (e) => { if (e.target === overlay) dismiss(); });
    document.body.appendChild(overlay);
    renderStep();
  }

  function close() {
    const o = document.getElementById('tourOverlay');
    if (o) o.remove();
  }

  function dismiss() {
    markSeen();
    close();
  }

  function next() {
    if (_step >= STEPS.length - 1) { dismiss(); return; }
    _step++;
    renderStep();
  }

  function prev() {
    if (_step <= 0) return;
    _step--;
    renderStep();
  }

  function renderStep() {
    const s = STEPS[_step];
    const body = document.getElementById('tourBody');
    if (!body) return;
    body.innerHTML = `
      <div class="tour-icon">${s.icon}</div>
      <h2>${escapeHTML(s.title)}</h2>
      <p>${escapeHTML(s.body)}</p>
    `;
    const prog = document.getElementById('tourProgress');
    prog.innerHTML = STEPS.map((_, i) =>
      `<span class="tour-dot ${i === _step ? 'active' : ''} ${i < _step ? 'done' : ''}"></span>`
    ).join('');
    const nextBtn = document.querySelector('.tour-next');
    if (nextBtn) nextBtn.textContent = _step === STEPS.length - 1 ? 'Done' : 'Next';
    const prevBtn = document.querySelector('.tour-prev');
    if (prevBtn) prevBtn.style.visibility = _step === 0 ? 'hidden' : '';
  }

  function escapeHTML(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function injectCSS() {
    if (document.getElementById('tour-css')) return;
    const css = document.createElement('style');
    css.id = 'tour-css';
    css.textContent = `
      .tour-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); backdrop-filter: blur(6px); z-index: 1800; display: flex; align-items: center; justify-content: center; padding: 40px; }
      .tour-card { background: var(--bg-chat); border: 1px solid var(--border); border-radius: 16px; width: min(520px, 100%); padding: 28px; text-align: center; box-shadow: 0 30px 80px rgba(0,0,0,0.5); }
      .tour-progress { display: flex; gap: 6px; justify-content: center; margin-bottom: 24px; }
      .tour-dot { width: 26px; height: 4px; border-radius: 2px; background: var(--border); transition: all 220ms; }
      .tour-dot.active { background: var(--accent); width: 36px; }
      .tour-dot.done { background: var(--text-secondary); }
      .tour-icon { font-size: 48px; margin-bottom: 14px; }
      .tour-body h2 { font-size: 22px; margin: 0 0 10px; color: var(--text-primary); }
      .tour-body p { color: var(--text-secondary); font-size: 14px; line-height: 1.6; margin: 0 0 6px; max-width: 420px; margin-left: auto; margin-right: auto; }
      .tour-actions { display: flex; gap: 10px; justify-content: center; margin-top: 24px; }
      .tour-btn { background: transparent; border: 1px solid var(--border); color: var(--text-secondary); padding: 8px 18px; border-radius: 8px; cursor: pointer; font-size: 13px; }
      .tour-btn:hover { background: var(--bg-hover); color: var(--text-primary); }
      .tour-primary { background: var(--accent); border-color: var(--accent); color: #fff; }
      .tour-primary:hover { background: #2563eb; color: #fff; }
      .tour-skip { margin-right: auto; color: var(--text-muted); border: none; }
    `;
    document.head.appendChild(css);
  }

  injectCSS();

  // Sequence the two first-run surfaces. The sovereignty card explains the
  // local boundary first; the feature tour starts only after that card has
  // been acknowledged, never underneath it.
  if (shouldShow()) {
    const maybeOpen = () => {
      if (shouldShow() && localStorage.getItem(SOVEREIGNTY_SEEN_KEY)) open();
    };
    if (localStorage.getItem(SOVEREIGNTY_SEEN_KEY)) {
      setTimeout(maybeOpen, 600);
    } else {
      window.addEventListener('mio:sovereignty-onboarded', () => setTimeout(maybeOpen, 180), { once: true });
    }
  }

  NS.tour = { open, close, next, prev, dismiss };
})();
