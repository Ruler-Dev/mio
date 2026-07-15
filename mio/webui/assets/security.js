// Same-origin request helpers for Mio's local WebUI security boundary.
(function () {
  "use strict";
  window.Mio = window.Mio || {};
  if (window.Mio.security) return;

  const unsafeMethods = new Set(["POST", "PUT", "PATCH", "DELETE"]);
  const nativeFetch = window.fetch.bind(window);

  function cookie(name) {
    const prefix = encodeURIComponent(name) + "=";
    for (const part of document.cookie.split(";")) {
      const item = part.trim();
      if (item.startsWith(prefix)) return decodeURIComponent(item.slice(prefix.length));
    }
    return "";
  }

  function csrfToken() {
    return cookie("mio_csrf");
  }

  function requestUrl(input) {
    try {
      return new URL(typeof input === "string" || input instanceof URL ? input : input.url, location.href);
    } catch {
      return null;
    }
  }

  window.fetch = function mioSecureFetch(input, init) {
    const options = Object.assign({}, init || {});
    const method = String(options.method || (input instanceof Request ? input.method : "GET")).toUpperCase();
    const url = requestUrl(input);
    if (unsafeMethods.has(method) && url && url.origin === location.origin) {
      const headers = new Headers(input instanceof Request ? input.headers : undefined);
      new Headers(options.headers || {}).forEach((value, key) => headers.set(key, value));
      const token = csrfToken();
      if (token) headers.set("X-Mio-CSRF-Token", token);
      options.headers = headers;
      if (!options.credentials) options.credentials = "same-origin";
    }
    return nativeFetch(input, options);
  };

  function openWebSocket(path) {
    const url = new URL(path, location.href);
    url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const protocols = ["mio-ui"];
    const token = csrfToken();
    if (token) protocols.push("mio-csrf." + token);
    return new WebSocket(url, protocols);
  }

  async function runSkill(name, args, options) {
    const confirmed = Boolean(options && options.confirmSensitive);
    const headers = new Headers({ "Content-Type": "application/json" });
    if (confirmed) headers.set("X-Mio-Dangerous-Action", name);
    return window.fetch("/ui/api/skills/run", {
      method: "POST",
      headers,
      body: JSON.stringify({
        name,
        args: args || {},
        confirm_sensitive: confirmed,
      }),
    });
  }

  window.Mio.security = Object.freeze({ csrfToken, openWebSocket, runSkill });
})();
