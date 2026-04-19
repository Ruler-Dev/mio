// Small utilities: /surprise (random prompt + random persona), persistent
// input drafts keyed by session.
(function () {
  const NS = (window.Mio = window.Mio || {});

  function surprise() {
    const bank = window.SUGGESTION_BANK || [];
    const personas = window.PERSONAS || {};
    const personaKeys = Object.keys(personas);
    if (!bank.length) {
      if (window.toast) window.toast('No suggestions loaded yet');
      return;
    }
    const prompt = bank[Math.floor(Math.random() * bank.length)];
    const pKey = personaKeys.length && Math.random() > 0.4
      ? personaKeys[Math.floor(Math.random() * personaKeys.length)]
      : null;
    const input = document.getElementById('inputArea');
    if (!input) return;
    const prefix = pKey ? '/as ' + pKey + ' ' : '';
    input.value = prefix + prompt;
    if (window.toast) window.toast('Surprise! ' + (pKey ? 'Persona: ' + pKey : 'Solo prompt'));
    // Send immediately
    if (window.sendMessage) window.sendMessage();
  }

  // ---- Persistent drafts ----
  const DRAFT_PREFIX = 'mio-draft-';

  function saveDraft() {
    const input = document.getElementById('inputArea');
    if (!input) return;
    const sid = window.currentSessionId || 'global';
    const v = input.value;
    if (v && v.trim()) {
      localStorage.setItem(DRAFT_PREFIX + sid, v);
    } else {
      localStorage.removeItem(DRAFT_PREFIX + sid);
    }
  }

  function loadDraft() {
    const input = document.getElementById('inputArea');
    if (!input) return;
    const sid = window.currentSessionId || 'global';
    const v = localStorage.getItem(DRAFT_PREFIX + sid);
    if (v && !input.value) input.value = v;
  }

  function clearDraftForSession() {
    const sid = window.currentSessionId || 'global';
    localStorage.removeItem(DRAFT_PREFIX + sid);
  }

  function attachDraftWatcher() {
    const input = document.getElementById('inputArea');
    if (!input) return setTimeout(attachDraftWatcher, 300);
    input.addEventListener('input', saveDraft);
    loadDraft();
    // Reload whenever session id changes — poll cheaply.
    let lastSid = window.currentSessionId;
    setInterval(() => {
      if (window.currentSessionId !== lastSid) {
        lastSid = window.currentSessionId;
        if (input.value === '') loadDraft();
      }
    }, 500);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attachDraftWatcher);
  } else {
    attachDraftWatcher();
  }

  // Clear the draft for the current session after a successful send
  const origSend = window.sendMessage;
  if (origSend) {
    window.sendMessage = function () {
      const result = origSend.apply(this, arguments);
      clearDraftForSession();
      return result;
    };
  }

  NS.extras = { surprise, saveDraft, loadDraft };
})();
