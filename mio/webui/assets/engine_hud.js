// engine_hud.js — floating status for the real local MLX engine.
//
// Runtime measurements come from /ui/api/model-info. Configured and loaded
// tiers come from /ui/api/config; this file intentionally has no tier list or
// context-window fallback of its own.

(function () {
  "use strict";

  window.Mio = window.Mio || {};
  if (window.Mio.engineHud) return;

  let hud = null;
  let elements = null;
  let poll = null;
  let expanded = false;
  let info = null;
  let config = null;
  let allTiers = [];
  let loadedTiers = [];
  let status = "loading";
  let statusMessage = "Checking the local engine…";
  let switchTarget = "";
  let inFlight = null;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  function mount() {
    if (hud) return;

    hud = element("div", "mio-engine-hud");
    hud.tabIndex = 0;
    hud.setAttribute("role", "button");
    hud.setAttribute("aria-label", "MLX engine status");
    hud.setAttribute("aria-expanded", "false");

    const row = element("div", "mio-engine-row");
    const tier = element("span", "mio-engine-tier", "engine");
    const tps = element("span", "mio-engine-tps", "checking…");
    const context = element("div", "mio-engine-ctx");
    context.setAttribute("aria-hidden", "true");
    const contextFill = element("div", "mio-engine-ctx-fill");
    context.appendChild(contextFill);
    row.append(tier, tps, context);

    const detail = element("div", "mio-engine-detail");
    detail.hidden = true;
    hud.append(row, detail);
    document.body.appendChild(hud);
    elements = { tier, tps, contextFill, detail };

    hud.addEventListener("click", toggleExpanded);
    hud.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggleExpanded();
      }
    });

    render();
    refresh();
    poll = setInterval(() => {
      if (!document.hidden) refresh();
    }, 5000);
    document.addEventListener("visibilitychange", handleVisibility);
  }

  function toggleExpanded() {
    expanded = !expanded;
    hud.setAttribute("aria-expanded", String(expanded));
    elements.detail.hidden = !expanded;
    render();
    if (expanded) refresh();
  }

  function handleVisibility() {
    if (!document.hidden) refresh();
  }

  async function refresh() {
    if (inFlight) return inFlight;
    inFlight = refreshFromBackend();
    try {
      await inFlight;
    } finally {
      inFlight = null;
    }
  }

  async function refreshFromBackend() {
    if (!info && status !== "switching") {
      status = "loading";
      statusMessage = "Checking the local engine…";
      render();
    }

    try {
      const [modelResponse, configResponse] = await Promise.all([
        fetch("/ui/api/model-info"),
        fetch("/ui/api/config"),
      ]);
      const [modelData, configData] = await Promise.all([
        modelResponse.json(),
        configResponse.json(),
      ]);

      if (!configResponse.ok || !configData || typeof configData !== "object" || configData.error) {
        throw new Error("The engine configuration endpoint is unavailable.");
      }

      config = configData;
      loadedTiers = cleanTiers(configData.loaded_tiers);
      allTiers = uniqueTiers([...cleanTiers(configData.all_tiers), ...loadedTiers]);

      if (!loadedTiers.length) {
        info = null;
        status = "empty";
        statusMessage = allTiers.length
          ? "No model is loaded. Choose a configured tier below."
          : "No MLX tiers are configured. Run Mio setup first.";
        render();
        return;
      }

      if (!modelResponse.ok || !modelData || typeof modelData !== "object" || modelData.error) {
        throw new Error("A loaded model did not return engine metrics.");
      }

      info = modelData;
      status = "ready";
      statusMessage = "Local MLX engine ready.";
      render();
    } catch (error) {
      status = "error";
      statusMessage = error instanceof Error && error.message
        ? error.message
        : "The local engine is unavailable.";
      render();
    }
  }

  function render() {
    if (!hud || !elements) return;
    hud.dataset.state = status;
    hud.title = statusMessage;

    const active = activeTier();
    const tps = finiteNumber(info && info.last_gen_tps);
    const contextUsed = finiteNumber(window.lastContextUsed)
      ?? finiteNumber(info && info.last_prompt_tokens)
      ?? 0;
    const contextWindow = finiteNumber(info && info.context_window) ?? 0;
    const ratio = contextWindow > 0 ? Math.min(1, contextUsed / contextWindow) : 0;

    if (status === "loading") {
      elements.tier.textContent = "engine";
      elements.tps.textContent = "checking…";
    } else if (status === "switching") {
      elements.tier.textContent = switchTarget || "engine";
      elements.tps.textContent = "switching…";
    } else if (status === "empty") {
      elements.tier.textContent = "no model";
      elements.tps.textContent = "idle";
    } else if (status === "error") {
      elements.tier.textContent = active || "engine unavailable";
      elements.tps.textContent = "retry";
    } else {
      elements.tier.textContent = active || "MLX";
      elements.tps.textContent = tps !== null && tps > 0 ? tps.toFixed(1) + " tok/s" : "idle";
    }

    elements.contextFill.style.width = (ratio * 100).toFixed(1) + "%";
    elements.contextFill.style.background = ratio > 0.85
      ? "#dc2626"
      : ratio > 0.65 ? "#f59e0b" : "var(--accent)";

    if (expanded) renderDetails({ active, tps, contextUsed, contextWindow, ratio });
  }

  function renderDetails(metrics) {
    const detail = elements.detail;
    detail.replaceChildren();

    const grid = element("div", "mio-engine-grid");
    appendMetric(grid, "state", statusMessage);
    appendMetric(grid, "model", info && info.model_name ? info.model_name : "–");
    appendMetric(grid, "loaded", loadedTiers.length ? loadedTiers.join(", ") : "none");
    appendMetric(grid, "ctx", metrics.contextWindow > 0
      ? `${humanInt(metrics.contextUsed)} / ${humanInt(metrics.contextWindow)} (${(metrics.ratio * 100).toFixed(1)}%)`
      : "–");
    appendMetric(grid, "gen", metrics.tps !== null && metrics.tps > 0 ? metrics.tps.toFixed(1) + " tok/s" : "idle");
    appendMetric(grid, "prompt", info && finiteNumber(info.last_prompt_tokens) !== null
      ? humanInt(Number(info.last_prompt_tokens)) + " tokens"
      : "–");
    appendMetric(grid, "VRAM", info && finiteNumber(info.vram_gb) !== null
      ? Number(info.vram_gb).toFixed(1) + " GB"
      : "–");
    detail.appendChild(grid);

    const actions = element("div", "mio-engine-actions");
    if (!allTiers.length) {
      actions.appendChild(element("span", "", "No configured tiers"));
    } else {
      allTiers.forEach(tier => {
        const loaded = loadedTiers.includes(tier);
        const button = element("button", "", tier + (loaded ? " • loaded" : ""));
        button.type = "button";
        button.dataset.tier = tier;
        button.disabled = status === "loading" || status === "switching" || tier === metrics.active;
        button.title = tier === metrics.active ? `${tier} is active` : `Switch to ${tier}`;
        button.addEventListener("click", event => {
          event.stopPropagation();
          switchToTier(tier);
        });
        actions.appendChild(button);
      });
    }
    detail.appendChild(actions);
  }

  function appendMetric(grid, label, value) {
    grid.append(element("span", "", label), element("code", "", value));
  }

  async function switchToTier(tier) {
    if (!allTiers.includes(tier) || status === "switching" || tier === activeTier()) return;
    if (typeof window.switchTier !== "function") {
      status = "error";
      statusMessage = "Tier switching is unavailable on this page.";
      render();
      return;
    }

    status = "switching";
    switchTarget = tier;
    statusMessage = `Switching the local engine to ${tier}…`;
    render();
    try {
      await window.switchTier(tier);
      info = null;
      await refresh();
      if (activeTier() !== tier) {
        status = "error";
        statusMessage = `Mio could not activate ${tier}.`;
        render();
      }
    } catch (error) {
      status = "error";
      statusMessage = `Mio could not activate ${tier}.`;
      render();
    } finally {
      switchTarget = "";
    }
  }

  function activeTier() {
    const configured = typeof config?.active_tier === "string" ? config.active_tier : "";
    if (configured && loadedTiers.includes(configured)) return configured;
    const measured = typeof info?.tier === "string" ? info.tier : "";
    if (measured && loadedTiers.includes(measured)) return measured;
    return loadedTiers[0] || "";
  }

  function cleanTiers(value) {
    if (!Array.isArray(value)) return [];
    return uniqueTiers(value.filter(tier => typeof tier === "string").map(tier => tier.trim()).filter(Boolean));
  }

  function uniqueTiers(value) {
    return [...new Set(value)];
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === "" || typeof value === "boolean") return null;
    const number = Number(value);
    return Number.isFinite(number) && number >= 0 ? number : null;
  }

  function humanInt(value) {
    if (!Number.isFinite(value)) return "–";
    if (value >= 1024) return (value / 1024).toFixed(1) + "K";
    return Math.round(value).toString();
  }

  function snapshot() {
    return Object.freeze({
      status,
      message: statusMessage,
      active_tier: activeTier() || null,
      all_tiers: [...allTiers],
      loaded_tiers: [...loadedTiers],
      last_gen_tps: finiteNumber(info && info.last_gen_tps),
    });
  }

  function unmount() {
    clearInterval(poll);
    poll = null;
    document.removeEventListener("visibilitychange", handleVisibility);
    if (hud) hud.remove();
    hud = null;
    elements = null;
  }

  setTimeout(mount, 600);

  window.Mio.engineHud = Object.freeze({ mount, refresh, snapshot, unmount });
})();
