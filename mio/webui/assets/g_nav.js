// g_nav.js — vim-style `G <letter>` navigation.
//
// Press `G` then a letter within 1.5 s to jump:
//   G C  Chat
//   G W  Workspaces
//   G D  Docs & RAG
//   G S  Design (Studio)
//   G O  Obsidian
//   G B  Dashboards
//   G A  Attachments page
//   G P  Playground page
//   G T  Stats page
//   G H  Dashboard (scheduler / webhooks) page
//
// Only fires when no input/textarea/contentEditable has focus so it
// never eats keystrokes during prompting.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.gNav) return;

  const VIEW_MAP = {
    c: { kind: "view", target: "chat",       label: "Chat" },
    w: { kind: "view", target: "workspaces", label: "Workspaces" },
    d: { kind: "view", target: "docs",       label: "Docs & RAG" },
    s: { kind: "view", target: "design",     label: "Design" },
    o: { kind: "view", target: "obsidian",   label: "Obsidian" },
    b: { kind: "view", target: "dashboards", label: "Dashboards" },
    f: { kind: "view", target: "flow",       label: "Flow" },
    j: { kind: "view", target: "journal",    label: "Journal" },
    n: { kind: "view", target: "notebook",   label: "Notebook" },
    a: { kind: "href", target: "/ui/attachments", label: "Attachments" },
    p: { kind: "href", target: "/ui/playground",  label: "Playground" },
    t: { kind: "href", target: "/ui/stats",       label: "Stats" },
    h: { kind: "href", target: "/ui/dashboard",   label: "Dashboard" },
  };

  let armed = false;
  let armedAt = 0;
  let toast = null;

  function showToast(msg) {
    if (!toast) {
      toast = document.createElement("div");
      toast.className = "mio-g-toast";
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => toast.classList.remove("show"), 1500);
  }

  function onKey(e) {
    const t = e.target;
    const inText = t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT" || t.isContentEditable);
    if (inText) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    // Arm on plain `g`
    if (!armed && (e.key === "g" || e.key === "G")) {
      e.preventDefault();
      armed = true;
      armedAt = Date.now();
      showToast("G → press a letter (C W D S O B A P T H)");
      setTimeout(() => { if (Date.now() - armedAt >= 1500) { armed = false; if (toast) toast.classList.remove("show"); } }, 1500);
      return;
    }

    // Second key while armed
    if (armed) {
      armed = false;
      const key = e.key.toLowerCase();
      const dest = VIEW_MAP[key];
      if (!dest) { if (toast) toast.classList.remove("show"); return; }
      e.preventDefault();
      if (dest.kind === "view") {
        if (window.Mio?.views?.switch) window.Mio.views.switch(dest.target);
      } else {
        window.location.href = dest.target;
      }
      showToast("→ " + dest.label);
    }
  }

  window.addEventListener("keydown", onKey);
  window.Mio.gNav = { showToast };
})();
