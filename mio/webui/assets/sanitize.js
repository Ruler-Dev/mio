// sanitize.js — local, dependency-free HTML sanitizer for rendered Markdown.
//
// Artifacts intentionally run code, but only inside sandboxed iframes. Chat,
// notebook, design-history and scratchpad Markdown stay in the parent document,
// so they use this strict allowlist before assigning to innerHTML.

(function () {
  "use strict";

  const NS = (window.Mio = window.Mio || {});
  if (NS.sanitizeHtml) return;

  const ALLOWED_TAGS = new Set([
    "A", "B", "BLOCKQUOTE", "BR", "CODE", "DEL", "DETAILS", "EM", "H1",
    "H2", "H3", "H4", "H5", "H6", "HR", "I", "IMG", "KBD", "LI",
    "MARK", "OL", "P", "PRE", "S", "SAMP", "SMALL", "SPAN", "STRONG",
    "SUB", "SUMMARY", "SUP", "TABLE", "TBODY", "TD", "TFOOT", "TH",
    "THEAD", "TR", "U", "UL",
  ]);

  // Contents of these elements are discarded instead of unwrapped. This keeps
  // script/style source and form controls out of the visible document too.
  const DROP_WITH_CONTENTS = new Set([
    "BASE", "BUTTON", "CANVAS", "EMBED", "FORM", "IFRAME", "INPUT", "LINK",
    "MATH", "META", "OBJECT", "OPTION", "SCRIPT", "SELECT", "STYLE", "SVG",
    "TEXTAREA", "VIDEO", "AUDIO", "SOURCE",
  ]);

  const SAFE_PROTOCOLS = new Set(["http:", "https:", "mailto:", "tel:"]);

  function safeUrl(raw, allowMail) {
    const value = String(raw || "").trim().replace(/[\u0000-\u001f\u007f]/g, "");
    if (!value) return "";
    if (value.startsWith("#") || (value.startsWith("/") && !value.startsWith("//"))) {
      return value;
    }
    try {
      const parsed = new URL(value, window.location.href);
      if (!SAFE_PROTOCOLS.has(parsed.protocol)) return "";
      if (!allowMail && (parsed.protocol === "mailto:" || parsed.protocol === "tel:")) return "";
      return value;
    } catch {
      return "";
    }
  }

  function sanitizeAttributes(el) {
    const tag = el.tagName;
    for (const attr of Array.from(el.attributes)) {
      const name = attr.name.toLowerCase();
      let keep = false;

      if (tag === "A" && name === "href") {
        const url = safeUrl(attr.value, true);
        if (url) { el.setAttribute("href", url); keep = true; }
      } else if (tag === "IMG" && name === "src") {
        const url = safeUrl(attr.value, false);
        if (url) { el.setAttribute("src", url); keep = true; }
      } else if (tag === "A" && name === "title") {
        keep = true;
      } else if (tag === "IMG" && ["alt", "title"].includes(name)) {
        keep = true;
      } else if (["TD", "TH"].includes(tag) && ["colspan", "rowspan"].includes(name)) {
        const count = Number.parseInt(attr.value, 10);
        if (Number.isInteger(count) && count >= 1 && count <= 100) {
          el.setAttribute(name, String(count));
          keep = true;
        }
      } else if (["CODE", "PRE", "SPAN"].includes(tag) && name === "class") {
        // Prism emits only language-* and token classes. Do not allow arbitrary
        // application classes that could visually escape the message surface.
        const classes = attr.value.split(/\s+/).filter((value) =>
          /^(?:language-[a-z0-9_-]+|token|keyword|string|number|comment|operator|punctuation|function|class-name|boolean|property|tag|attr-name|attr-value)$/i.test(value)
        );
        if (classes.length) {
          el.setAttribute("class", classes.join(" "));
          keep = true;
        }
      }

      if (!keep) el.removeAttribute(attr.name);
    }

    if (tag === "A" && el.hasAttribute("href")) {
      el.setAttribute("rel", "noopener noreferrer nofollow");
      el.setAttribute("target", "_blank");
    }
    if (tag === "IMG") {
      el.setAttribute("loading", "lazy");
      el.setAttribute("referrerpolicy", "no-referrer");
    }
  }

  function sanitizeHtml(input) {
    const template = document.createElement("template");
    template.innerHTML = String(input ?? "");
    const walker = document.createTreeWalker(template.content, NodeFilter.SHOW_ELEMENT);
    const elements = [];
    while (walker.nextNode()) elements.push(walker.currentNode);

    // Process children before parents so unwrapping an unknown wrapper cannot
    // bypass validation of anything nested inside it.
    for (const el of elements.reverse()) {
      if (DROP_WITH_CONTENTS.has(el.tagName)) {
        el.remove();
      } else if (!ALLOWED_TAGS.has(el.tagName)) {
        el.replaceWith(...Array.from(el.childNodes));
      } else {
        sanitizeAttributes(el);
      }
    }
    return template.innerHTML;
  }

  NS.sanitizeHtml = sanitizeHtml;
})();
