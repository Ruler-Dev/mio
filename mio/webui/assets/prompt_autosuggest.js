// prompt_autosuggest.js — dim-text autosuggest from recent prompts.
//
// Records every committed user message into a ring buffer in
// localStorage. When the composer is focused with a non-empty prefix,
// the most-recent matching entry appears as ghost text after the
// caret. → / Tab accepts; anything else dismisses.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.promptAutosuggest) return;

  const KEY = "mio.prompt-history.v1";
  const MAX = 200;

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch { return []; }
  }
  function save(list) {
    try { localStorage.setItem(KEY, JSON.stringify(list.slice(-MAX))); } catch {}
  }
  function push(text) {
    const t = (text || "").trim();
    if (!t || t.length > 2000) return;
    const list = load();
    // Dedupe: drop any exact prior match so the new one floats to the end.
    for (let i = list.length - 1; i >= 0; i--) if (list[i] === t) list.splice(i, 1);
    list.push(t);
    save(list);
  }

  // Find a matching history entry prefix-first (case insensitive)
  function suggest(prefix) {
    if (!prefix || prefix.length < 3) return null;
    const list = load();
    const lower = prefix.toLowerCase();
    for (let i = list.length - 1; i >= 0; i--) {
      const h = list[i];
      if (h.toLowerCase().startsWith(lower) && h.length > prefix.length) {
        return h.slice(prefix.length);
      }
    }
    return null;
  }

  // --- UI ---------------------------------------------------------
  let ghostEl = null;
  let boundInput = null;

  function mountGhost(input) {
    if (ghostEl || !input) return;
    ghostEl = document.createElement("div");
    ghostEl.className = "mio-ghost";
    ghostEl.style.position = "fixed";
    ghostEl.style.pointerEvents = "none";
    ghostEl.style.whiteSpace = "pre-wrap";
    ghostEl.style.overflow = "hidden";
    document.body.appendChild(ghostEl);
  }

  function computeGhostPosition(input) {
    if (!ghostEl) return;
    const rect = input.getBoundingClientRect();
    const style = getComputedStyle(input);
    ghostEl.style.left      = rect.left + "px";
    ghostEl.style.top       = rect.top  + "px";
    ghostEl.style.width     = rect.width  + "px";
    ghostEl.style.height    = rect.height + "px";
    ghostEl.style.padding   = style.padding;
    ghostEl.style.border    = style.border;
    ghostEl.style.font      = style.font;
    ghostEl.style.lineHeight = style.lineHeight;
    ghostEl.style.letterSpacing = style.letterSpacing;
    ghostEl.style.boxSizing = style.boxSizing;
  }

  function updateGhost(input) {
    if (!input) return;
    mountGhost(input);
    const text = input.value;
    const caret = input.selectionStart ?? text.length;
    // Only suggest at end-of-field (caret at the tail)
    if (caret !== text.length) { hideGhost(); return; }
    const s = suggest(text);
    if (!s) { hideGhost(); return; }
    computeGhostPosition(input);
    // Render: visible `text` (invisible via color:transparent) then the
    // suggestion in muted color.
    ghostEl.innerHTML =
      `<span style="color:transparent">${escapeHtml(text)}</span>` +
      `<span class="mio-ghost-tail">${escapeHtml(s)}</span>`;
    ghostEl._pending = s;
  }

  function hideGhost() {
    if (ghostEl) { ghostEl.style.display = "none"; ghostEl._pending = null; }
  }
  function showGhost() {
    if (ghostEl) ghostEl.style.display = "";
  }

  function onInput(e) {
    updateGhost(e.target);
  }
  function onKeydown(e) {
    if (!ghostEl || !ghostEl._pending) return;
    if (e.key === "ArrowRight" || e.key === "Tab") {
      // Accept only if caret is at the end
      const input = e.target;
      if (input.selectionStart === input.value.length) {
        e.preventDefault();
        input.value += ghostEl._pending;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        hideGhost();
      }
    } else if (e.key === "Escape") {
      hideGhost();
    }
  }

  // Capture a prompt when it gets sent — listens for Enter that's NOT
  // shift+enter (the standard send shortcut in the main chat).
  function attach() {
    const input = document.querySelector(
      "textarea#messageInput, textarea#input, textarea.input, textarea[data-role='chat-input']"
    ) || document.querySelector("textarea");
    if (!input || input === boundInput) return;
    boundInput = input;
    input.addEventListener("input", onInput);
    input.addEventListener("keydown", (e) => {
      onKeydown(e);
      if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.ctrlKey && input.value.trim()) {
        push(input.value);
      }
    });
    input.addEventListener("blur", hideGhost);
    input.addEventListener("focus", () => updateGhost(input));
    // Reposition on scroll / resize
    window.addEventListener("scroll", () => updateGhost(input), true);
    window.addEventListener("resize", () => updateGhost(input));
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }

  function boot() {
    attach();
    const obs = new MutationObserver(() => attach());
    obs.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }

  window.Mio.promptAutosuggest = { push, clear: () => localStorage.removeItem(KEY) };
})();
