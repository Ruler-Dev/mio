// view_obsidian.js — Obsidian vault integration view.
//
// Configure vault path → browse folder tree → open any note in a
// read + edit pane → save back → @-mention in chat. Reindex button
// pushes the vault into the local RAG store so the model can search
// notes as a normal tool call.

(function () {
  window.Mio = window.Mio || {};
  const ready = () => {
    if (!window.Mio.views) return setTimeout(ready, 50);
    window.Mio.views.register("obsidian", {
      title: "Obsidian",
      mount(host) { renderRoot(host); },
    });
  };
  ready();

  async function renderRoot(host) {
    host.innerHTML = `<div class="muted" style="padding:28px">Loading vault…</div>`;
    const cfg = await fetch("/ui/api/obsidian/config").then((r) => r.json());
    if (!cfg.vault_path) {
      renderConfigure(host, "");
      return;
    }
    if (!cfg.vault_exists) {
      renderConfigure(host, cfg.vault_path,
        `The configured path doesn't exist any more: ${cfg.vault_path}`);
      return;
    }
    renderBrowser(host, cfg.vault_path);
  }

  function renderConfigure(host, currentPath, errorMsg) {
    host.innerHTML = `
      <div class="view-obsidian-configure">
        <div class="view-empty-inner" style="max-width:560px">
          <h1>Point Mio at your Obsidian vault</h1>
          <p>Absolute path to the vault's root folder. Mio will list notes, open them,
             write new ones, and full-text-index the vault for chat search.</p>
          ${errorMsg ? `<p style="color:#dc2626">${escapeHtml(errorMsg)}</p>` : ""}
          <form id="obs-cfg-form" style="margin-top:18px">
            <input name="path" type="text" value="${escapeAttr(currentPath || "")}" placeholder="/Users/you/Documents/Obsidian/MyVault" style="width:100%;padding:9px 12px;border:1px solid var(--border);background:var(--bg-sidebar);color:var(--text-primary);border-radius:6px;font-size:13px;font-family:inherit">
            <div style="margin-top:12px;display:flex;gap:8px;justify-content:flex-end">
              <button type="submit" class="btn-ghost" style="background:var(--accent);color:#fff;border-color:var(--accent)">Save &amp; connect</button>
            </div>
          </form>
        </div>
      </div>
    `;
    host.querySelector("#obs-cfg-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const path = e.target.elements.path.value.trim();
      if (!path) return;
      const res = await fetch("/ui/api/obsidian/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vault_path: path }),
      });
      const data = await res.json();
      if (data.error) {
        renderConfigure(host, path, data.error);
      } else {
        renderBrowser(host, data.vault_path);
      }
    });
  }

  async function renderBrowser(host, vaultPath) {
    host.innerHTML = `
      <div class="view-obsidian">
        <header class="view-header">
          <div>
            <h1>Obsidian</h1>
            <p class="muted"><code style="font-family:var(--font-mono);font-size:11px">${escapeHtml(vaultPath)}</code></p>
          </div>
          <div class="view-header-actions">
            <button class="btn-ghost" data-action="new-note">+ Note</button>
            <button class="btn-ghost" data-action="reindex">Reindex for RAG</button>
            <button class="btn-ghost" data-action="reconfigure">Change vault</button>
          </div>
        </header>
        <div class="obs-split">
          <aside class="obs-tree" id="obs-tree"><div class="muted">Loading tree…</div></aside>
          <main class="obs-editor" id="obs-editor">
            <div class="view-empty-inner" style="max-width:400px;margin:60px auto;text-align:center">
              <h2 style="font-size:15px;color:var(--text-secondary)">Pick a note</h2>
              <p class="muted" style="font-size:12px">Select any note from the tree on the left to open it.</p>
            </div>
          </main>
        </div>
      </div>
    `;
    host.querySelector('[data-action="reconfigure"]').addEventListener("click", () => renderConfigure(host, vaultPath));
    host.querySelector('[data-action="reindex"]').addEventListener("click", async (e) => {
      e.target.disabled = true; e.target.textContent = "Reindexing…";
      const r = await fetch("/ui/api/obsidian/reindex", { method: "POST" });
      const data = await r.json();
      e.target.disabled = false; e.target.textContent = "Reindex for RAG";
      if (data.error) alert("Reindex failed: " + data.error);
      else alert(`Indexed ${data.file_count || data.files || "?"} files.`);
    });
    host.querySelector('[data-action="new-note"]').addEventListener("click", () => {
      const name = prompt("Note path (relative to vault, e.g. 'inbox/new-thought.md'):");
      if (!name) return;
      openEditor(host, { path: name, name: name.split("/").pop(), content: "" }, true);
    });
    const { tree = [], error } = await fetch("/ui/api/obsidian/tree").then((r) => r.json());
    if (error) {
      host.querySelector("#obs-tree").innerHTML = `<div class="muted">${escapeHtml(error)}</div>`;
      return;
    }
    renderTree(host, tree);
  }

  function renderTree(host, tree) {
    const container = host.querySelector("#obs-tree");
    container.innerHTML = "";
    container.appendChild(buildTree(host, tree));
  }

  function buildTree(host, nodes, depth = 0) {
    const frag = document.createDocumentFragment();
    for (const n of nodes) {
      const row = document.createElement("div");
      row.className = "obs-row obs-" + n.type;
      row.style.paddingLeft = (8 + depth * 14) + "px";
      if (n.type === "folder") {
        let open = depth < 1;
        row.innerHTML = `<span class="obs-caret">${open ? "▾" : "▸"}</span><span class="obs-name">${escapeHtml(n.name)}</span>`;
        const children = document.createElement("div");
        children.style.display = open ? "" : "none";
        children.appendChild(buildTree(host, n.children || [], depth + 1));
        row.addEventListener("click", () => {
          open = !open;
          row.querySelector(".obs-caret").textContent = open ? "▾" : "▸";
          children.style.display = open ? "" : "none";
        });
        frag.appendChild(row);
        frag.appendChild(children);
      } else {
        row.innerHTML = `<span class="obs-caret"> </span><span class="obs-name">${escapeHtml(n.name)}</span>`;
        row.addEventListener("click", async () => {
          const res = await fetch("/ui/api/obsidian/note?path=" + encodeURIComponent(n.path));
          const data = await res.json();
          if (data.error) return alert(data.error);
          openEditor(host, data, false);
          host.querySelectorAll(".obs-row.active").forEach((r) => r.classList.remove("active"));
          row.classList.add("active");
        });
        frag.appendChild(row);
      }
    }
    return frag;
  }

  function openEditor(host, note, isNew) {
    const editor = host.querySelector("#obs-editor");
    editor.innerHTML = `
      <div class="obs-editor-header">
        <div class="obs-editor-path" title="${escapeAttr(note.path)}">${escapeHtml(note.path || note.name || "")}</div>
        <div style="display:flex;gap:6px">
          <button class="btn-ghost" data-act="mention">@-mention in chat</button>
          <button class="btn-ghost" data-act="save" style="background:var(--accent);color:#fff;border-color:var(--accent)">Save</button>
        </div>
      </div>
      <textarea class="obs-editor-text" spellcheck="false">${escapeHtml(note.content || "")}</textarea>
      <div class="obs-editor-foot muted" id="obs-editor-foot"></div>
    `;
    const ta = editor.querySelector(".obs-editor-text");
    const foot = editor.querySelector("#obs-editor-foot");
    const updateFoot = () => {
      const chars = ta.value.length;
      const lines = ta.value.split("\n").length;
      foot.textContent = `${chars} chars · ${lines} lines`;
    };
    ta.addEventListener("input", updateFoot);
    updateFoot();

    editor.querySelector('[data-act="save"]').addEventListener("click", async () => {
      const res = await fetch("/ui/api/obsidian/note", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: note.path, content: ta.value }),
      });
      const data = await res.json();
      if (data.error) alert("Save failed: " + data.error);
      else {
        foot.textContent = `Saved · ${data.size} bytes`;
        if (isNew) {
          // Refresh the tree so the new note shows up
          const host2 = editor.closest(".view");
          if (host2) {
            const { tree = [] } = await fetch("/ui/api/obsidian/tree").then((r) => r.json());
            renderTree(host2, tree);
          }
        }
      }
    });
    editor.querySelector('[data-act="mention"]').addEventListener("click", () => {
      const input = document.querySelector("#input, textarea#messageInput, textarea.input, textarea");
      if (input) {
        input.value = (input.value || "") + `@note:${note.path} `;
        input.focus();
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
      if (window.Mio?.views?.switch) window.Mio.views.switch("chat");
    });
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }
  function escapeAttr(s) { return escapeHtml(s); }
})();
