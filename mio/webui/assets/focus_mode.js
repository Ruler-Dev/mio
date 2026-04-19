// focus_mode.js — immersive writing mode.
//
// ⌘⇧F (or the command palette "Focus" entry) hides the nav rail,
// sovereignty bar, voice FAB, and all other chrome. The chat
// surface expands to fill the viewport. Press the same shortcut
// again (or Esc) to exit.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.focusMode) return;

  const CLASS = "mio-focus";
  const KEY = "mio.focus";

  function setFocus(on) {
    document.documentElement.classList.toggle(CLASS, !!on);
    try { localStorage.setItem(KEY, on ? "1" : "0"); } catch {}
    if (on) showToast("Focus mode · ⌘⇧F or Esc to exit");
  }

  function toggle() { setFocus(!document.documentElement.classList.contains(CLASS)); }

  function showToast(msg) {
    const t = document.createElement("div");
    t.className = "mio-focus-toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 1800);
  }

  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "f") {
      e.preventDefault();
      toggle();
    } else if (e.key === "Escape" && document.documentElement.classList.contains(CLASS)) {
      setFocus(false);
    }
  });

  // Restore last state
  if (localStorage.getItem(KEY) === "1") {
    document.documentElement.classList.add(CLASS);
  }

  window.Mio.focusMode = { toggle, on: () => setFocus(true), off: () => setFocus(false) };
})();
