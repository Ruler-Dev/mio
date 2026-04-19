// refine_prompt.js — one-click "improve my prompt" helper.
//
// Adds a small ✨ button that floats at the bottom-right of the
// composer whenever its text is ≥10 chars. Clicking sends the
// current text to the model with a meta-prompt that asks for a
// crisper rewrite. The composer text is replaced with the result
// (original backed up in localStorage for undo with ⌘Z-like semantics
// — click the button's ↶ variant to restore).

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.refinePrompt) return;

  const META = `You are a prompt editor. The user gave you a draft they intend to send to a coding/design assistant. Return a single improved version that is:
- Specific (concrete nouns, exact output format)
- Scoped (one task, not a wishlist)
- Contextful (preserve any domain terms they used)

Rules: no preamble, no explanation, no quotes around the output. Just the rewritten prompt, 1-4 sentences. If the draft is already good, output it unchanged.`;

  function findInput() {
    return document.querySelector(
      "textarea#inputArea, textarea#messageInput, textarea#input, textarea.input, textarea[data-role='chat-input']",
    ) || document.querySelector("textarea");
  }

  let btn = null;
  let lastOriginal = null;

  function attach() {
    const input = findInput();
    if (!input || input._refinePrompt) return;
    input._refinePrompt = true;
    if (!btn) {
      btn = document.createElement("button");
      btn.className = "mio-refine";
      btn.title = "Refine this prompt (⌘⇧/)";
      btn.textContent = "✨";
      btn.hidden = true;
      document.body.appendChild(btn);
      btn.addEventListener("click", refine);
    }
    const check = () => {
      const show = (input.value || "").trim().length >= 10 && document.activeElement === input;
      if (!show) { btn.hidden = true; return; }
      const rect = input.getBoundingClientRect();
      btn.style.right = (window.innerWidth - rect.right + 8) + "px";
      btn.style.bottom = (window.innerHeight - rect.bottom + 8) + "px";
      btn.hidden = false;
    };
    input.addEventListener("input", check);
    input.addEventListener("focus", check);
    input.addEventListener("blur", () => setTimeout(() => {
      // Allow click events on the button to fire before hiding
      if (document.activeElement !== btn) btn.hidden = true;
    }, 120));
    window.addEventListener("scroll", check, true);
    window.addEventListener("resize", check);
    window.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && (e.key === "/" || e.key === "?")) {
        e.preventDefault();
        refine();
      }
    });
  }

  async function refine() {
    const input = findInput();
    if (!input) return;
    const original = input.value.trim();
    if (!original) return;
    // Undo-toggle: if the user just refined and didn't type since,
    // second click restores the original.
    if (lastOriginal && lastOriginal.refined === input.value) {
      input.value = lastOriginal.original;
      lastOriginal = null;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      setBtnState("idle");
      return;
    }
    setBtnState("busy");
    try {
      const res = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "mio-auto",
          messages: [
            { role: "system", content: META },
            { role: "user",   content: original },
          ],
          temperature: 0.3, max_tokens: 300, stream: false,
        }),
      });
      const data = await res.json();
      const refined = (data.choices?.[0]?.message?.content || "").trim();
      if (refined && refined !== original) {
        lastOriginal = { original, refined };
        input.value = refined;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        setBtnState("restored");
      } else {
        setBtnState("idle");
      }
    } catch (e) {
      setBtnState("idle");
    }
  }

  function setBtnState(s) {
    if (!btn) return;
    if (s === "busy")      btn.textContent = "⋯";
    else if (s === "restored") btn.textContent = "↶"; // second click to undo
    else                    btn.textContent = "✨";
  }

  function boot() {
    attach();
    new MutationObserver(attach).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else { boot(); }

  window.Mio.refinePrompt = { refine };
})();
