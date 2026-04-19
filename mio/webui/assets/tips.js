// "Did you know?" — rotating tips surfaced in the welcome screen and as
// occasional bottom-corner toasts after long idle periods.
(function () {
  const NS = (window.Mio = window.Mio || {});

  const TIPS = [
    "Type / in the message box to see 90+ slash commands.",
    "Cmd+K opens a palette with every command and every artifact in this chat.",
    "Drag a PDF onto the chat to analyze it — the text is auto-extracted.",
    "Drop an image and the model will read it (all MLX tiers are multimodal).",
    "Type /as teacher to switch personas. /as-list shows all 27.",
    "Click ⭐ on a message to pin it to the top of the chat.",
    "Cycle chat density with /density — compact fits twice as much on screen.",
    "Hold the mic button to talk — the model transcribes with Whisper-MLX.",
    "Toggle focus mode with /focus for zen-style full-screen chat.",
    "Every artifact opens in a side panel — click it again to re-open later.",
    "Presets auto-pick themselves for documents — just say 'create a flyer'.",
    "Say 'same but in emerald' and only the palette swaps — not the layout.",
    "Mio has 94 artifact types. Type /gallery to see them all in this chat.",
    "The cache lives in ~/.mio/ — Settings → Cache lets you wipe any kind.",
    "Projects group chats with shared memory and attached files. /projects.",
    "Type /workspace to let Mio read and write in a folder you pick.",
    "Ask 'explain visually' and Mio builds an interactive artifact, not a PDF.",
    "Tandem mode routes simple queries to fast tiers, hard ones to large-moe.",
    "⌘⇧V pastes the clipboard as hidden context, not as a visible message.",
    "The artifact panel has Preview / Source / Diff — flip through revisions.",
    "Fork a conversation from any point with the Fork button on a message.",
    "/search goes to the web; find_anime goes to MAL. Mio picks the right one.",
    "Export the whole chat as Markdown via /export.",
    "/gallery browses every artifact in the current chat with thumbnails.",
    "When streaming, the expandable pill shows tool calls and artifact progress.",
    "Images attached to a chat are cached locally — they survive page reloads.",
    "The weather skill auto-renders an animated widget; no need to prompt it.",
    "Voice responses are off by default. Toggle in Settings → Voice.",
    "Try /convo for hands-free voice conversation mode.",
    "The model name and PQ bit-depth are shown at the top of the chat.",
    "Hit Cmd+N to start a new chat instantly.",
    "/random picks a random anime, manga, movie or game to show you.",
    "The sidebar has live chat search — type in the box to filter.",
    "Pressing Tab in the slash popup autocompletes the highlighted command.",
    "Every artifact template is </script>-safe — paste any text without breakage.",
  ];

  function pickTip() {
    // Deterministic per-day cycle so the welcome screen tip is stable
    // but different across days.
    const day = Math.floor(Date.now() / 86400000);
    return TIPS[day % TIPS.length];
  }

  function renderWelcomeTip() {
    const container = document.getElementById('welcome');
    if (!container) return;
    // IDEMPOTENT: if the tip already exists, do nothing. Otherwise the
    // innerHTML re-set below would trigger the MutationObserver that
    // called us → infinite render loop that thrashes the page.
    if (container.querySelector('.welcome-tip')) return;
    const tip = document.createElement('div');
    tip.className = 'welcome-tip';
    container.appendChild(tip);
    tip.innerHTML = `
      <span class="welcome-tip-icon">💡</span>
      <span class="welcome-tip-label">Did you know?</span>
      <span class="welcome-tip-text">${escapeHTML(pickTip())}</span>
      <button class="welcome-tip-next" onclick="Mio.tips.rotate()" title="Next tip">→</button>
    `;
  }

  let _tipIdx = -1;
  function rotate() {
    _tipIdx = (_tipIdx + 1) % TIPS.length;
    const el = document.querySelector('.welcome-tip-text');
    if (el) el.textContent = TIPS[_tipIdx];
  }

  function injectCSS() {
    if (document.getElementById('tips-css')) return;
    const css = document.createElement('style');
    css.id = 'tips-css';
    css.textContent = `
      .welcome-tip { margin: 28px auto 0; padding: 12px 16px; max-width: 640px; background: linear-gradient(135deg, rgba(59,130,246,0.06), rgba(139,92,246,0.06)); border: 1px solid var(--border-subtle); border-radius: 12px; display: flex; align-items: center; gap: 12px; font-size: 12px; color: var(--text-secondary); }
      .welcome-tip-icon { font-size: 18px; }
      .welcome-tip-label { font-weight: 600; color: var(--accent); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
      .welcome-tip-text { flex: 1; color: var(--text-primary); }
      .welcome-tip-next { background: transparent; border: 1px solid var(--border); color: var(--text-secondary); border-radius: 50%; width: 26px; height: 26px; font-size: 14px; cursor: pointer; }
      .welcome-tip-next:hover { background: var(--bg-hover); color: var(--text-primary); }
    `;
    document.head.appendChild(css);
  }

  function escapeHTML(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  injectCSS();

  // Auto-render on welcome screen presence. The render function itself
  // is idempotent, but we still guard here to avoid pointless work.
  const obs = new MutationObserver(() => {
    const w = document.getElementById('welcome');
    if (w && !w.querySelector('.welcome-tip')) renderWelcomeTip();
  });
  obs.observe(document.body, { childList: true, subtree: true });
  // Initial render if the welcome screen is already in the DOM
  renderWelcomeTip();

  NS.tips = { pick: pickTip, rotate, render: renderWelcomeTip, list: () => TIPS.slice() };
})();
