// Mio Clip — background service worker.
// Owns the context-menu entries, dispatches ingest requests to the local
// Mio instance, and surfaces success/failure via browser notifications.

const DEFAULT_ENDPOINT = "http://localhost:9090/ui/api/ingest";

async function endpoint() {
  const { mioEndpoint } = await chrome.storage.sync.get("mioEndpoint");
  return (mioEndpoint || DEFAULT_ENDPOINT).replace(/\/$/, "");
}

function notify(title, message, type = "basic") {
  chrome.notifications?.create?.(
    { type, iconUrl: "icons/icon-128.png", title, message },
  );
}

async function ingest(payload) {
  const url = await endpoint();
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      notify("Mio Clip failed", `HTTP ${res.status} from ${url}`);
      return null;
    }
    const data = await res.json();
    notify(
      "Sent to Mio",
      `"${data.title || payload.title || payload.url}" — ${data.chars || payload.text?.length || 0} chars`,
    );
    return data;
  } catch (e) {
    notify("Mio unreachable", `Is Mio running at ${url}?`);
    return null;
  }
}

async function extractFromTab(tab, { selectionOnly = false } = {}) {
  const [result] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: (selectionOnly) => {
      const selection = window.getSelection()?.toString() || "";
      if (selectionOnly) {
        return {
          url: location.href,
          title: document.title,
          selection,
          text: selection,
          html: "",
        };
      }
      // Readability-lite: pull the largest text-dense element.
      const candidates = Array.from(document.querySelectorAll("article, main, #content, .content, body"));
      let best = candidates[0] || document.body;
      let bestLen = (best?.innerText || "").length;
      for (const el of candidates) {
        const l = (el.innerText || "").length;
        if (l > bestLen) { best = el; bestLen = l; }
      }
      return {
        url: location.href,
        title: document.title,
        selection,
        text: (best?.innerText || "").slice(0, 200_000),
        html: (best?.innerHTML || "").slice(0, 400_000),
      };
    },
    args: [selectionOnly],
  });
  return result?.result;
}

async function clipActive({ selectionOnly = false, tags = [] } = {}) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return null;
  const extracted = await extractFromTab(tab, { selectionOnly });
  if (!extracted) return null;
  return ingest({ ...extracted, tags, target: "rag" });
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "mio-clip-page",
    title: "Send this page to Mio",
    contexts: ["page"],
  });
  chrome.contextMenus.create({
    id: "mio-clip-selection",
    title: "Send selection to Mio",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId === "mio-clip-page") {
    await clipActive({ selectionOnly: false });
  } else if (info.menuItemId === "mio-clip-selection") {
    await clipActive({ selectionOnly: true });
  }
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "mio-clip") {
    clipActive({
      selectionOnly: !!msg.selectionOnly,
      tags: msg.tags || [],
    }).then(sendResponse);
    return true; // async response
  }
  if (msg?.type === "mio-endpoint-get") {
    endpoint().then(sendResponse);
    return true;
  }
  if (msg?.type === "mio-endpoint-set") {
    chrome.storage.sync.set({ mioEndpoint: msg.endpoint || DEFAULT_ENDPOINT })
      .then(() => sendResponse({ ok: true }));
    return true;
  }
});
