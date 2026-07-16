// Shared client for Mio's local WebUI JSON APIs.
(function () {
  "use strict";

  window.Mio = window.Mio || {};
  if (window.Mio.api?.runSkill) return;

  class MioApiError extends Error {
    constructor(message, { status = 0, payload = null, cause = null } = {}) {
      super(message);
      this.name = "MioApiError";
      this.status = status;
      this.payload = payload;
      if (cause) this.cause = cause;
    }
  }

  function errorText(payload) {
    if (!payload || typeof payload !== "object") return "";
    if (typeof payload.error === "string") return payload.error;
    const detail = payload.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object") {
      const parts = [detail.error, detail.reason].filter((part) => typeof part === "string" && part);
      if (parts.length) return parts.join(": ");
    }
    return "";
  }

  async function readJson(response) {
    try {
      return await response.json();
    } catch (cause) {
      throw new MioApiError("Skill API returned invalid JSON", {
        status: Number(response?.status) || 0,
        cause,
      });
    }
  }

  function directSkillRequest(name, args, options) {
    const confirmed = Boolean(options?.confirmSensitive);
    const headers = new Headers({ "Content-Type": "application/json" });
    if (confirmed) headers.set("X-Mio-Dangerous-Action", name);
    return window.fetch("/ui/api/skills/run", {
      method: "POST",
      headers,
      credentials: "same-origin",
      body: JSON.stringify({
        name,
        args,
        confirm_sensitive: confirmed,
      }),
    });
  }

  async function runSkill(name, args = {}, options = {}) {
    if (typeof name !== "string" || !name.trim()) {
      throw new TypeError("Skill name must be a non-empty string");
    }
    if (!args || typeof args !== "object" || Array.isArray(args)) {
      throw new TypeError("Skill arguments must be an object");
    }

    const transport = window.Mio.security?.runSkill;
    let response;
    try {
      response = transport
        ? await transport(name, args, options)
        : await directSkillRequest(name, args, options);
    } catch (cause) {
      throw new MioApiError(`Skill API request failed: ${String(cause?.message || cause)}`, { cause });
    }
    const payload = await readJson(response);
    const status = Number(response?.status) || 0;
    const message = errorText(payload);

    if (!response?.ok) {
      throw new MioApiError(
        `Skill API HTTP ${status || "error"}${message ? `: ${message}` : ""}`,
        { status, payload },
      );
    }
    if (!payload || typeof payload !== "object" || payload.ok !== true) {
      throw new MioApiError(message || "Skill execution failed", { status, payload });
    }
    if (!Object.prototype.hasOwnProperty.call(payload, "result")) {
      throw new MioApiError("Skill API response is missing result", { status, payload });
    }
    return payload.result;
  }

  window.Mio.api = Object.freeze({ ...(window.Mio.api || {}), MioApiError, runSkill });
})();
