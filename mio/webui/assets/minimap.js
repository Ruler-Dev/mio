// minimap.js — scroll mini-map for long chats.
//
// Thin vertical strip on the right of the main chat surface. Each
// message gets a tick; assistant turns that emitted an artifact get
// a brighter dot + a "◇" marker. Dragging the strip scrolls the
// chat; clicking any dot jumps to that message.
//
// Zero dependencies, idempotent, self-updating via MutationObserver.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.minimap) return;

  const CHAT_SELECTORS = [
    ".chat-messages", "#messages", ".messages", ".chat-body",
  ];

  function findChat() {
    for (const sel of CHAT_SELECTORS) {
      const el = document.querySelector(sel);
      if (el) return el;
    }
    return null;
  }

  let strip = null;
  let chatEl = null;
  let rebuildTimer = null;

  function mount() {
    chatEl = findChat();
    if (!chatEl) return;
    if (strip) return;
    strip = document.createElement("div");
    strip.className = "mio-minimap";
    chatEl.parentElement.appendChild(strip);
    rebuild();
    chatEl.addEventListener("scroll", updateIndicator, { passive: true });
    const obs = new MutationObserver(() => {
      clearTimeout(rebuildTimer);
      rebuildTimer = setTimeout(rebuild, 250);
    });
    obs.observe(chatEl, { childList: true, subtree: true });
    window.addEventListener("resize", rebuild);
  }

  function rebuild() {
    if (!strip || !chatEl) return;
    const msgs = Array.from(chatEl.querySelectorAll(".message, .msg, [data-role]"));
    if (!msgs.length) { strip.innerHTML = ""; strip.style.display = "none"; return; }
    // Only show if we have enough content to bother
    if (chatEl.scrollHeight <= chatEl.clientHeight * 1.5) {
      strip.style.display = "none";
      return;
    }
    strip.style.display = "";
    strip.innerHTML = `<div class="mio-minimap-indicator"></div>`;
    const total = chatEl.scrollHeight;
    for (const m of msgs) {
      const top = (m.offsetTop / total) * 100;
      const role = m.classList.contains("user") || m.dataset.role === "user" ? "user"
                 : m.classList.contains("assistant") || m.dataset.role === "assistant" ? "assistant"
                 : "other";
      const hasArtifact = !!m.querySelector("iframe.artifact-iframe, .artifact-card, [data-artifact-id]");
      const dot = document.createElement("div");
      dot.className = "mio-minimap-dot mio-minimap-" + role + (hasArtifact ? " artifact" : "");
      dot.style.top = top + "%";
      dot.addEventListener("click", (e) => {
        e.stopPropagation();
        m.scrollIntoView({ behavior: "smooth", block: "start" });
      });
      strip.appendChild(dot);
    }
    updateIndicator();
  }

  function updateIndicator() {
    if (!strip || !chatEl) return;
    const ind = strip.querySelector(".mio-minimap-indicator");
    if (!ind) return;
    const vis = chatEl.clientHeight / chatEl.scrollHeight;
    const pos = chatEl.scrollTop / chatEl.scrollHeight;
    ind.style.top    = (pos * 100) + "%";
    ind.style.height = (vis * 100) + "%";
  }

  // Delay mount so the chat DOM exists
  setTimeout(mount, 800);
  // Re-attempt on body mutations in case the chat is lazily injected
  const attachObs = new MutationObserver(() => { if (!strip) mount(); });
  attachObs.observe(document.body, { childList: true, subtree: true });

  window.Mio.minimap = { rebuild, unmount() { strip?.remove(); strip = null; } };
})();
