// popup.js — controls the extension popup.
// Reads endpoint from storage, wires the two action buttons.

const $ = (id) => document.getElementById(id);

async function init() {
  // Load current endpoint.
  const endpoint = await new Promise((r) =>
    chrome.runtime.sendMessage({ type: "mio-endpoint-get" }, r),
  );
  $("endpoint").value = endpoint || "http://localhost:9090/ui/api/ingest";

  // Persist endpoint on blur.
  $("endpoint").addEventListener("change", async () => {
    const ep = $("endpoint").value.trim();
    await new Promise((r) =>
      chrome.runtime.sendMessage({ type: "mio-endpoint-set", endpoint: ep }, r),
    );
    showStatus("Endpoint saved", "ok");
  });

  // Clip buttons.
  $("clip-page").addEventListener("click", () => clip({ selectionOnly: false }));
  $("clip-selection").addEventListener("click", () => clip({ selectionOnly: true }));
}

async function clip({ selectionOnly }) {
  const tags = $("tags").value
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);

  showStatus(selectionOnly ? "Clipping selection…" : "Clipping page…", "");
  const result = await new Promise((r) =>
    chrome.runtime.sendMessage({ type: "mio-clip", selectionOnly, tags }, r),
  );
  if (!result) {
    showStatus("Failed — is Mio running?", "err");
    return;
  }
  showStatus(`Saved: ${result.title || result.url} (${result.chars} chars)`, "ok");
  setTimeout(() => window.close(), 900);
}

function showStatus(msg, cls) {
  const el = $("status");
  el.className = "status" + (cls ? " " + cls : "");
  el.textContent = msg;
}

init();
