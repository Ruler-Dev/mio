// scratchpad.js — always-available local scratch note.
//
// Press ⌘⇧S (or click the ✎ pill in the sovereignty bar) to open a
// floating notepad. Contents persist in localStorage (mio.scratchpad)
// and on the server as ~/.mio/scratchpad.md so it's also reachable
// from a terminal. Auto-saves on blur + every 4 s of idle typing.
//
// Markdown preview toggle. "Send to chat" inserts the content
// into the composer. "Save to Obsidian" writes it as a note in
// the configured vault (if any).

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.scratchpad) return;

  const LS_KEY = "mio.scratchpad";
  let pad = null;
  let syncTimer = null;

  function open() {
    if (pad) return;
    pad = document.createElement("div");
    pad.className = "mio-pad";
    pad.innerHTML = `
      <header>
        <strong>Scratchpad</strong>
        <span class="muted">saves to <code>~/.mio/scratchpad.md</code></span>
        <div style="flex:1"></div>
        <button data-act="preview" title="Toggle preview">👁</button>
        <button data-act="chat"    title="Send to chat">→ chat</button>
        <button data-act="note"    title="Save as Obsidian note">→ note</button>
        <button data-act="close"   aria-label="Close">×</button>
      </header>
      <textarea class="mio-pad-src" spellcheck="false"></textarea>
      <div class="mio-pad-preview" hidden></div>
      <footer id="mio-pad-foot"></footer>
    `;
    document.body.appendChild(pad);
    const src = pad.querySelector(".mio-pad-src");
    const prev = pad.querySelector(".mio-pad-preview");
    const foot = pad.querySelector("#mio-pad-foot");

    // Load — prefer server copy, fall back to localStorage, then empty.
    fetch("/ui/api/scratchpad").then((r) => r.json()).then((d) => {
      const text = (d && d.content) || localStorage.getItem(LS_KEY) || "";
      src.value = text;
      updateFoot();
    }).catch(() => {
      src.value = localStorage.getItem(LS_KEY) || "";
      updateFoot();
    });

    src.addEventListener("input", () => {
      localStorage.setItem(LS_KEY, src.value);
      clearTimeout(syncTimer);
      syncTimer = setTimeout(saveToServer, 4000);
      updateFoot();
    });
    src.addEventListener("blur", saveToServer);

    pad.querySelector('[data-act="close"]').addEventListener("click", close);
    pad.querySelector('[data-act="preview"]').addEventListener("click", () => {
      const showing = !prev.hidden;
      if (showing) { prev.hidden = true;  src.hidden = false; return; }
      const rendered = window.marked?.parse
        ? window.marked.parse(src.value)
        : escapeHtml(src.value).replace(/\n/g, "<br>");
      prev.innerHTML = window.Mio?.sanitizeHtml
        ? window.Mio.sanitizeHtml(rendered)
        : escapeHtml(src.value).replace(/\n/g, "<br>");
      prev.hidden = false; src.hidden = true;
    });
    pad.querySelector('[data-act="chat"]').addEventListener("click", () => {
      const input = document.querySelector("textarea#inputArea, textarea#messageInput, textarea#input, textarea.input, textarea");
      if (input) {
        input.value += (input.value ? "\n\n" : "") + src.value;
        input.focus();
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
      close();
    });
    pad.querySelector('[data-act="note"]').addEventListener("click", async () => {
      const name = prompt("Save scratchpad to Obsidian vault at path (relative):", `scratchpad/${new Date().toISOString().slice(0,10)}.md`);
      if (!name) return;
      try {
        const r = await fetch("/ui/api/obsidian/note", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: name, content: src.value }),
        });
        const d = await r.json();
        if (d.error) alert("Save failed: " + d.error);
        else foot.textContent = `Saved to vault: ${name}`;
      } catch (e) { alert("Save failed: " + e.message); }
    });

    function updateFoot() {
      foot.textContent = `${src.value.length} chars · ${src.value.split("\n").length} lines`;
    }

    async function saveToServer() {
      try {
        await fetch("/ui/api/scratchpad", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content: src.value }),
        });
      } catch {}
    }
  }

  function close() {
    if (!pad) return;
    pad.remove();
    pad = null;
  }

  // Global shortcut: ⌘⇧S (or Ctrl+Shift+S)
  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "s") {
      const t = e.target;
      const inText = t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT");
      // Still allow — user explicitly asked for the scratchpad.
      e.preventDefault();
      pad ? close() : open();
    }
  });

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }

  window.Mio.scratchpad = { open, close };
})();
