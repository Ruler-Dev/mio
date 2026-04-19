// rerun.js — first-class re-runnable user messages.
//
// Every user message in the main chat gains three shortcuts while
// hovered / focused:
//   R        — re-run the prompt with the current model + settings
//   E        — edit the prompt inline, then Enter to re-run
//   ⌘⇧.      — fork to a new chat starting from that message
//
// The UI doesn't depend on the main mio_ui.html internals — it finds
// user-message nodes by their existing `.message.user` class and
// attaches a small action rail on hover, plus listens for global
// shortcut keys when a user message has focus.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.rerun) return;

  const USER_MSG_SELECTORS = [
    ".message.user",
    ".msg-user",
    "[data-role='user']",
    ".chat-message--user",
  ];
  const sel = USER_MSG_SELECTORS.join(", ");

  // --- Plumbing: inject per-message controls idempotently -------------

  function ensureControls(msgEl) {
    if (msgEl.querySelector(".rerun-rail")) return;
    const rail = document.createElement("div");
    rail.className = "rerun-rail";
    rail.innerHTML = `
      <button data-act="rerun" title="Re-run (R)">↻</button>
      <button data-act="edit"  title="Edit &amp; re-run (E)">✎</button>
      <button data-act="fork"  title="Fork from here (⌘⇧.)">⑂</button>
    `;
    msgEl.appendChild(rail);
    rail.querySelector('[data-act="rerun"]').addEventListener("click", (e) => {
      e.stopPropagation(); rerunMessage(msgEl);
    });
    rail.querySelector('[data-act="edit"]').addEventListener("click", (e) => {
      e.stopPropagation(); editMessage(msgEl);
    });
    rail.querySelector('[data-act="fork"]').addEventListener("click", (e) => {
      e.stopPropagation(); forkFromMessage(msgEl);
    });
    msgEl.setAttribute("tabindex", "0");
  }

  // --- Action implementations -----------------------------------------
  // We go through the existing `sendMessage` / `setInput` / `loadChat`
  // globals that mio_ui.html exposes, but we degrade gracefully if
  // those aren't present.

  function messageText(el) {
    // Prefer a known text node; fall back to innerText minus children.
    const txt = el.querySelector(".msg-text, .message-content, .content");
    if (txt) return (txt.innerText || txt.textContent || "").trim();
    // Strip rail buttons before reading text.
    const clone = el.cloneNode(true);
    clone.querySelectorAll(".rerun-rail").forEach((n) => n.remove());
    return (clone.innerText || "").trim();
  }

  function rerunMessage(msgEl) {
    const text = messageText(msgEl);
    if (!text) return;
    setInput(text);
    flash(msgEl, "Re-running…");
    // The existing chat UI calls sendMessage on Enter — do it directly
    // if the global helper is exposed.
    if (typeof window.sendMessage === "function") {
      try { window.sendMessage(); return; } catch {}
    }
    // Fallback: dispatch Enter on the input.
    const input = findInput();
    if (!input) return;
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  }

  function editMessage(msgEl) {
    const text = messageText(msgEl);
    setInput(text);
    const input = findInput();
    if (input) {
      input.focus();
      // Park the caret at the end so the user can keep typing.
      if ("setSelectionRange" in input) {
        input.setSelectionRange(input.value.length, input.value.length);
      }
    }
    flash(msgEl, "Loaded into composer. Edit then Enter to re-run.");
  }

  async function forkFromMessage(msgEl) {
    const text = messageText(msgEl);
    if (!text) return;
    flash(msgEl, "Forking to a new chat…");
    // Create a new chat, park the prompt in the composer.
    if (typeof window.newChat === "function") {
      try { window.newChat(); } catch {}
    }
    setTimeout(() => {
      setInput(text);
      flash(msgEl, "");
    }, 100);
  }

  function setInput(text) {
    const input = findInput();
    if (!input) return;
    input.value = text;
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function findInput() {
    return document.querySelector(
      "textarea#messageInput, textarea#input, textarea.input, textarea[data-role='chat-input']",
    ) || document.querySelector("textarea");
  }

  function flash(el, msg) {
    if (!msg) {
      el.querySelector(".rerun-flash")?.remove();
      return;
    }
    let node = el.querySelector(".rerun-flash");
    if (!node) {
      node = document.createElement("span");
      node.className = "rerun-flash";
      el.appendChild(node);
    }
    node.textContent = msg;
    clearTimeout(node._t);
    node._t = setTimeout(() => node.remove(), 2000);
  }

  // --- Keyboard shortcuts (while a user message is focused/hovered) ---

  function currentTarget() {
    // Active element (if it's a user message) beats hover, else hover.
    const ae = document.activeElement;
    if (ae && ae.matches?.(sel)) return ae;
    return document.querySelector(sel + ":hover");
  }

  function onKeyDown(e) {
    // Respect inputs / editable targets — never hijack typing.
    const t = e.target;
    const inText = t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT" || t.isContentEditable);
    if (inText) return;

    const target = currentTarget();
    if (!target) return;

    if (e.key === "r" || e.key === "R") {
      e.preventDefault();
      rerunMessage(target);
    } else if (e.key === "e" || e.key === "E") {
      e.preventDefault();
      editMessage(target);
    } else if (e.key === "." && e.metaKey && e.shiftKey) {
      e.preventDefault();
      forkFromMessage(target);
    }
  }

  // --- Boot -----------------------------------------------------------

  function scan() {
    document.querySelectorAll(sel).forEach(ensureControls);
  }

  function boot() {
    scan();
    const obs = new MutationObserver(scan);
    obs.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("keydown", onKeyDown);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }

  window.Mio.rerun = { rerunMessage, editMessage, forkFromMessage };
})();
