// smart_paste.js — context-aware paste suggestions.
//
// When the user pastes into the chat composer, inspect the content
// and offer a one-click smart action via a floating tooltip that
// fades after 6 s. Triggers that beat "boring paste":
//
//   URL (http(s)://…)            → "Fetch + summarize"  → web_search
//                                 → "Clip to Docs"      → /ui/api/ingest
//   JSON                         → "Pretty-print"
//   JWT  (3 b64 segments)         → "Decode header + payload"
//   CSV-ish (comma lines)         → "Open in Dashboards"
//   Error stack (…Error|Traceback)→ "Explain + fix"
//   Long (> 2000 chars)           → "Summarize this first"

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.smartPaste) return;

  function findInput() {
    return document.querySelector(
      "textarea#inputArea, textarea#messageInput, textarea#input, textarea.input, textarea[data-role='chat-input']",
    ) || document.querySelector("textarea");
  }

  function detect(text) {
    text = text.trim();
    if (!text) return null;
    const actions = [];

    if (/^https?:\/\/\S+$/i.test(text)) {
      actions.push({ label: "Fetch + summarize", act: () => chat(`Fetch this URL and summarize the key points:\n${text}`) });
      actions.push({ label: "Clip to Docs", act: () => clipUrl(text) });
    }
    if (/^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*$/.test(text.split("\n")[0])) {
      actions.push({ label: "Decode JWT", act: () => decodeJwt(text) });
    }
    const tr = text.trim();
    if ((tr.startsWith("{") && tr.endsWith("}")) || (tr.startsWith("[") && tr.endsWith("]"))) {
      actions.push({ label: "Pretty-print JSON", act: () => prettyJson(tr) });
    }
    // CSV-ish: at least two commas on the first line + multiple lines
    const firstLine = text.split("\n")[0] || "";
    if ((firstLine.match(/,/g) || []).length >= 2 && text.includes("\n")) {
      actions.push({ label: "Open in Dashboards", act: () => importCsv(text) });
    }
    if (/Traceback \(most recent call last\)|Error:|Exception:|\bat\s+[A-Za-z]/.test(text)) {
      actions.push({ label: "Explain + fix", act: () => chat(`Explain this error and suggest a fix:\n\n\`\`\`\n${text}\n\`\`\``) });
    }
    if (text.length > 2000) {
      actions.push({ label: "Summarize first", act: () => chat(`Here's a long paste — give me a tight summary (key points, not a rehash):\n\n${text.slice(0, 8000)}${text.length > 8000 ? "\n\n…(truncated)" : ""}`) });
    }
    return actions.length ? actions : null;
  }

  // --- Actions -------------------------------------------------------

  function chat(prompt) {
    const input = findInput();
    if (!input) return;
    input.value = prompt;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.focus();
    hideTip();
  }

  async function clipUrl(url) {
    try {
      const r = await fetch("/ui/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, title: url, text: "(to be fetched)", target: "rag" }),
      });
      const d = await r.json();
      if (d.error) alert("Clip failed: " + d.error);
      else toast("Clipped to Docs: " + (d.title || url));
    } catch (e) { alert("Clip failed: " + e.message); }
    hideTip();
  }

  function decodeJwt(tok) {
    try {
      const [h, p] = tok.split(".").slice(0, 2);
      const dec = (s) => JSON.parse(atob(s.replace(/-/g, "+").replace(/_/g, "/") + "===".slice(0, (4 - s.length % 4) % 4)));
      const out = { header: dec(h), payload: dec(p) };
      chat("Here's a decoded JWT:\n\n```json\n" + JSON.stringify(out, null, 2) + "\n```");
    } catch (e) { alert("Not a valid JWT."); }
  }

  function prettyJson(src) {
    try {
      const parsed = JSON.parse(src);
      const pretty = JSON.stringify(parsed, null, 2);
      const input = findInput();
      if (!input) return;
      input.value = "```json\n" + pretty + "\n```";
      input.dispatchEvent(new Event("input", { bubbles: true }));
    } catch (e) { alert("Not valid JSON."); }
    hideTip();
  }

  function importCsv(text) {
    // Stash the CSV in localStorage; pivot to Dashboards which reads
    // it as a pasted source.
    try {
      const key = "mio.smartPaste.pendingCsv";
      localStorage.setItem(key, text);
      toast("Switching to Dashboards… paste is ready in the Data dock.");
      setTimeout(() => window.Mio?.views?.switch?.("dashboards"), 600);
    } catch (e) { alert("Failed: " + e.message); }
    hideTip();
  }

  // --- Tooltip -------------------------------------------------------

  let tip = null, tipTimer = null;
  function showTip(anchor, actions) {
    hideTip();
    const rect = anchor.getBoundingClientRect();
    tip = document.createElement("div");
    tip.className = "mio-smart-paste";
    tip.style.left = Math.min(rect.left + 10, window.innerWidth - 340) + "px";
    tip.style.top  = Math.max(40, rect.top - 44) + "px";
    tip.innerHTML = `
      <span class="mio-smart-lbl">💡 Smart paste</span>
      ${actions.map((a, i) => `<button data-i="${i}">${escapeHtml(a.label)}</button>`).join("")}
      <button class="mio-smart-dismiss" aria-label="Dismiss">×</button>
    `;
    document.body.appendChild(tip);
    tip.querySelectorAll("button[data-i]").forEach((b) => {
      b.addEventListener("click", () => actions[parseInt(b.dataset.i, 10)].act());
    });
    tip.querySelector(".mio-smart-dismiss").addEventListener("click", hideTip);
    tipTimer = setTimeout(hideTip, 6000);
  }
  function hideTip() {
    clearTimeout(tipTimer);
    tip?.remove(); tip = null;
  }

  function toast(msg) {
    const t = document.createElement("div");
    t.className = "mio-smart-toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2200);
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  }

  function attach() {
    const input = findInput();
    if (!input || input._smartPaste) return;
    input._smartPaste = true;
    input.addEventListener("paste", (e) => {
      const text = e.clipboardData?.getData("text/plain") || "";
      if (!text) return;
      // Let the paste happen normally, then surface the suggestion.
      setTimeout(() => {
        const actions = detect(text);
        if (actions) showTip(input, actions);
      }, 20);
    });
  }

  function boot() {
    attach();
    new MutationObserver(attach).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else { boot(); }

  window.Mio.smartPaste = { detect };
})();
