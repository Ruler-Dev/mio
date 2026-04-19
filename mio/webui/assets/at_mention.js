// at_mention.js — @-autocomplete for the chat composer.
//
// Type `@` anywhere in the chat input and a dropdown appears with
// three groups: clipped docs, Obsidian notes, workspaces. Filters
// live as you keep typing. Arrow keys / Tab / Enter select; Esc
// dismisses. Selection inserts the right token:
//
//    @doc:<id>    — clipped doc (browser extension)
//    @note:<path> — Obsidian note
//    @ws:<id>     — workspace
//
// The backend doesn't need to know about these tokens specifically —
// the existing RAG skills already match on text inside ~/.mio/ingest
// and vault files, and the token survives as-is in the prompt so the
// model treats it as a pointer.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.atMention) return;

  const MAX_PER_GROUP = 6;
  let dropdown = null;
  let lastFetch = 0;
  let cache = null;
  let activeIdx = 0;
  let items = [];
  let input = null;
  let trigger = null; // { start, input }

  function findInput() {
    return document.querySelector(
      "textarea#messageInput, textarea#input, textarea.input, textarea[data-role='chat-input']",
    ) || document.querySelector("textarea");
  }

  function attach() {
    const el = findInput();
    if (!el || el === input) return;
    input = el;
    input.addEventListener("input", onInput);
    input.addEventListener("keydown", onKeyDown);
    input.addEventListener("blur", () => setTimeout(close, 150));
  }

  function onInput() {
    const caret = input.selectionStart;
    const text = input.value.slice(0, caret);
    // Look for the last @ that isn't preceded by a word character.
    const m = text.match(/(^|[\s(])@([a-zA-Z0-9._\-\/]*)$/);
    if (!m) { close(); return; }
    const start = caret - m[2].length - 1; // position of the @
    trigger = { start, query: m[2].toLowerCase() };
    open();
    render();
  }

  function onKeyDown(e) {
    if (!dropdown) return;
    if (e.key === "ArrowDown") { e.preventDefault(); activeIdx = (activeIdx + 1) % items.length; render(); return; }
    if (e.key === "ArrowUp")   { e.preventDefault(); activeIdx = (activeIdx - 1 + items.length) % items.length; render(); return; }
    if (e.key === "Enter" || e.key === "Tab") {
      if (items.length) { e.preventDefault(); choose(items[activeIdx]); return; }
    }
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
  }

  function open() {
    if (dropdown) return;
    dropdown = document.createElement("div");
    dropdown.className = "at-mention-dropdown";
    document.body.appendChild(dropdown);
    positionDropdown();
    window.addEventListener("scroll", positionDropdown, true);
    window.addEventListener("resize", positionDropdown);
  }

  function close() {
    if (!dropdown) return;
    dropdown.remove();
    dropdown = null;
    trigger = null;
    items = [];
    activeIdx = 0;
    window.removeEventListener("scroll", positionDropdown, true);
    window.removeEventListener("resize", positionDropdown);
  }

  function positionDropdown() {
    if (!dropdown || !input) return;
    const rect = input.getBoundingClientRect();
    dropdown.style.left = (rect.left + 8) + "px";
    dropdown.style.bottom = (window.innerHeight - rect.top + 6) + "px";
    dropdown.style.maxWidth = Math.max(280, rect.width - 40) + "px";
  }

  async function ensureCache() {
    const now = Date.now();
    if (cache && (now - lastFetch < 30_000)) return cache;
    lastFetch = now;
    try {
      const [ing, notes, wsList] = await Promise.all([
        fetch("/ui/api/ingest?limit=200").then((r) => r.json()).catch(() => ({ items: [] })),
        fetch("/ui/api/obsidian/tree").then((r) => r.json()).catch(() => ({ tree: [] })),
        fetch("/ui/api/projects").then((r) => r.json()).catch(() => ({ projects: [] })),
      ]);
      const docs = (ing.items || []).map((x) => ({
        kind: "doc", group: "Clipped", label: x.title || x.id, hint: x.url || x.id,
        token: `@doc:${x.id}`,
      }));
      const noteItems = flattenNotes(notes.tree || []).map((n) => ({
        kind: "note", group: "Obsidian", label: n.name, hint: n.path,
        token: `@note:${n.path}`,
      }));
      const workspaces = (wsList.projects || []).map((p) => ({
        kind: "ws", group: "Workspace", label: p.name, hint: p.description || "",
        token: `@ws:${p.id}`,
      }));
      cache = { docs, noteItems, workspaces };
    } catch {
      cache = { docs: [], noteItems: [], workspaces: [] };
    }
    return cache;
  }

  function flattenNotes(tree) {
    const out = [];
    const walk = (nodes) => {
      for (const n of nodes) {
        if (n.type === "note") out.push(n);
        else if (n.children) walk(n.children);
      }
    };
    walk(tree);
    return out;
  }

  async function render() {
    if (!dropdown) return;
    const c = await ensureCache();
    const q = (trigger?.query || "").toLowerCase();
    const filter = (xs) => xs.filter((x) => !q || x.label.toLowerCase().includes(q) || x.hint.toLowerCase().includes(q)).slice(0, MAX_PER_GROUP);
    const groups = [
      ["Clipped",   filter(c.docs)],
      ["Obsidian",  filter(c.noteItems)],
      ["Workspaces", filter(c.workspaces)],
    ].filter(([_, xs]) => xs.length);

    items = [];
    for (const [, xs] of groups) items.push(...xs);
    activeIdx = Math.min(activeIdx, Math.max(0, items.length - 1));

    if (!items.length) {
      dropdown.innerHTML = `<div class="at-mention-empty">No matches${q ? ` for "${escapeHtml(q)}"` : ""}</div>`;
      return;
    }

    let html = "";
    let idx = 0;
    for (const [g, xs] of groups) {
      html += `<div class="at-mention-group">${escapeHtml(g)}</div>`;
      for (const it of xs) {
        const active = idx === activeIdx;
        html += `
          <button class="at-mention-item${active ? " active" : ""}" data-idx="${idx}">
            <span class="at-mention-kind at-mention-kind-${it.kind}">${kindIcon(it.kind)}</span>
            <span class="at-mention-label">${highlight(it.label, q)}</span>
            <span class="at-mention-hint">${escapeHtml(it.hint || "")}</span>
          </button>
        `;
        idx++;
      }
    }
    dropdown.innerHTML = html;
    dropdown.querySelectorAll(".at-mention-item").forEach((el) => {
      el.addEventListener("mousedown", (e) => {
        e.preventDefault();
        const i = parseInt(el.dataset.idx, 10);
        if (Number.isFinite(i) && items[i]) choose(items[i]);
      });
    });
  }

  function choose(it) {
    if (!trigger) return;
    const before = input.value.slice(0, trigger.start);
    const after = input.value.slice(input.selectionStart);
    const insertion = it.token + " ";
    input.value = before + insertion + after;
    const pos = before.length + insertion.length;
    input.setSelectionRange(pos, pos);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    close();
    input.focus();
  }

  function kindIcon(kind) {
    if (kind === "doc")  return "📎";
    if (kind === "note") return "📝";
    if (kind === "ws")   return "⚑";
    return "•";
  }

  function highlight(text, q) {
    const t = escapeHtml(text);
    if (!q) return t;
    const idx = text.toLowerCase().indexOf(q);
    if (idx < 0) return t;
    return escapeHtml(text.slice(0, idx))
      + `<mark>${escapeHtml(text.slice(idx, idx + q.length))}</mark>`
      + escapeHtml(text.slice(idx + q.length));
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }

  // Attach now + keep re-attaching whenever the chat input is replaced
  // (e.g. when a chat gets loaded and the DOM is rebuilt).
  function watch() {
    attach();
    const obs = new MutationObserver(() => attach());
    obs.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", watch, { once: true });
  } else {
    watch();
  }

  window.Mio.atMention = { attach, close };
})();
