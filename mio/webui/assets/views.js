// views.js — view registry + router.
//
// Mio has a single-shell SPA. The "Chat" surface (.sidebar + .main) is
// considered the baseline view and owns most of the existing DOM. Other
// views (Workspaces, Docs, Design, Obsidian, Settings) register here and
// mount into a sibling overlay container `#view-stage`.
//
// Contract:
//   Mio.views.register(name, {
//     title:      "Workspaces",
//     icon:       "<svg …>",       // rail icon (inline SVG string)
//     mount(el, context): attach DOM; may return a Promise and/or cleanup fn
//     render(el, context): async-compatible alias used when mount is absent
//     activate(context): called after mount/render; may be async
//     deactivate(context): called and awaited before leaving the view
//     cleanup(el, context): final async-compatible teardown hook
//   });
//   await Mio.views.switch("docs");
//
// Persists the active view in localStorage so reloads restore it.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.views) return;

  const STORAGE_KEY = "mio.activeView";
  const DEFAULT_VIEW = "chat";
  const registry = new Map();
  let activeView = null;
  let stageEl = null;
  let chatEls = null; // cached references to chat surface we hide/show
  let currentSession = null;
  let pendingRequest = null;
  let transitionTail = Promise.resolve();
  let navigationToken = 0;
  let booted = false;

  class NavigationCancelled extends Error {
    constructor() {
      super("View navigation was superseded");
      this.name = "NavigationCancelled";
    }
  }

  function getStage() {
    if (stageEl && stageEl.isConnected) return stageEl;
    stageEl = document.getElementById("view-stage");
    if (!stageEl) {
      stageEl = document.createElement("div");
      stageEl.id = "view-stage";
      stageEl.className = "view-stage";
      stageEl.style.display = "none";
      const app = document.getElementById("app");
      (app || document.body).appendChild(stageEl);
    }
    return stageEl;
  }

  function cacheChatEls() {
    if (chatEls) return chatEls;
    chatEls = {
      sidebar: document.querySelector(".app > .sidebar"),
      main:    document.querySelector(".app > .main"),
    };
    return chatEls;
  }

  function showChat(show) {
    const { sidebar, main } = cacheChatEls();
    const v = show ? "" : "none";
    if (sidebar) sidebar.style.display = v;
    if (main)    main.style.display    = v;
  }

  function register(name, spec) {
    if (!name || typeof name !== "string") return;
    const normalized = spec && typeof spec === "object" ? spec : {};
    registry.set(name, { mounted: false, ...normalized });
  }

  function list() {
    return Array.from(registry.entries()).map(([name, v]) => ({
      name,
      title: v.title || name,
      icon:  v.icon  || null,
      badge: v.badge || null,
    }));
  }

  function getActive() { return activeView; }

  function setRailActive(name) {
    document.querySelectorAll(".nav-rail-btn").forEach((button) => {
      button.classList.toggle("active", button.dataset.view === name);
    });
  }

  function commitActive(name) {
    activeView = name;
    setRailActive(name);
    try { localStorage.setItem(STORAGE_KEY, name); } catch {}
  }

  function statusPanel(kind, name, error, retry) {
    const spec = registry.get(name);
    const title = spec?.title || name;
    const panel = document.createElement("div");
    panel.className = "view-router-state view-router-" + kind;
    panel.dataset.state = kind;
    panel.setAttribute("role", kind === "error" ? "alert" : "status");
    panel.setAttribute("aria-live", kind === "error" ? "assertive" : "polite");

    const heading = document.createElement("strong");
    heading.className = "view-router-state-title";
    heading.textContent = kind === "error" ? "Could not open " + title : "Loading " + title + "…";
    panel.appendChild(heading);

    if (kind === "error") {
      const detail = document.createElement("p");
      detail.className = "view-router-state-detail";
      detail.textContent = error instanceof Error ? error.message : String(error || "Unknown error");
      panel.appendChild(detail);

      const button = document.createElement("button");
      button.type = "button";
      button.className = "view-router-retry";
      button.textContent = "Retry";
      button.addEventListener("click", () => {
        button.disabled = true;
        void _switch(name);
      }, { once: true });
      panel.appendChild(button);
      if (typeof retry === "function") retry(button);
    }
    return panel;
  }

  function showLoading(name, host) {
    const stage = getStage();
    showChat(false);
    stage.style.display = "flex";
    stage.setAttribute("aria-busy", "true");
    host.hidden = true;
    stage.replaceChildren(statusPanel("loading", name), host);
    setRailActive(name);
  }

  function showError(name, error) {
    const stage = getStage();
    showChat(false);
    stage.style.display = "flex";
    stage.setAttribute("aria-busy", "false");
    stage.replaceChildren(statusPanel("error", name, error));
    setRailActive(name);
  }

  function showMounted(host) {
    const stage = getStage();
    host.hidden = false;
    stage.setAttribute("aria-busy", "false");
    stage.replaceChildren(host);
  }

  function waitForAbort(promise, signal) {
    if (signal.aborted) return Promise.reject(new NavigationCancelled());
    return new Promise((resolve, reject) => {
      const cancel = () => reject(new NavigationCancelled());
      signal.addEventListener("abort", cancel, { once: true });
      promise.then(
        (value) => {
          signal.removeEventListener("abort", cancel);
          resolve(value);
        },
        (error) => {
          signal.removeEventListener("abort", cancel);
          reject(error);
        },
      );
    });
  }

  function invoke(fn, receiver, args) {
    if (typeof fn !== "function") return Promise.resolve(undefined);
    try {
      return Promise.resolve(fn.apply(receiver, args));
    } catch (error) {
      return Promise.reject(error);
    }
  }

  function cleanupFunction(result) {
    if (typeof result === "function") return result;
    if (!result || typeof result !== "object") return null;
    for (const name of ["cleanup", "dispose", "destroy", "unmount"]) {
      if (typeof result[name] === "function") return result[name].bind(result);
    }
    return null;
  }

  async function runReturnedCleanup(result, label) {
    const cleanup = cleanupFunction(result);
    if (!cleanup) return;
    try {
      await cleanup();
    } catch (error) {
      console.warn("[Mio.views] " + label + " cleanup failed", error);
    }
  }

  async function awaitLifecycle(session, fn, args, label) {
    const lifecycle = invoke(fn, session.spec, args);
    try {
      const result = await waitForAbort(lifecycle, session.context.signal);
      const cleanup = cleanupFunction(result);
      if (cleanup) session.cleanups.push({ cleanup, label });
      return result;
    } catch (error) {
      if (error instanceof NavigationCancelled) {
        // A lifecycle that ignores AbortSignal must not block a newer route.
        // If it later yields a cleanup handle, execute it off the stale host.
        lifecycle.then(
          (result) => runReturnedCleanup(result, label + " (late)"),
          (lateError) => {
            if (!session.context.signal.aborted) {
              console.warn("[Mio.views] " + label + " failed", lateError);
            }
          },
        );
      }
      throw error;
    }
  }

  async function disposeSession(session, deactivate) {
    if (!session) return;
    if (session.disposePromise) return session.disposePromise;
    session.disposePromise = (async () => {
      if (!session.context.signal.aborted) session.controller.abort();

      if (deactivate && typeof session.spec.deactivate === "function") {
        try {
          const result = await invoke(
            session.spec.deactivate,
            session.spec,
            [session.context],
          );
          await runReturnedCleanup(result, session.name + " deactivate");
        } catch (error) {
          console.warn("[Mio.views] deactivate failed for " + session.name, error);
        }
      }

      for (let index = session.cleanups.length - 1; index >= 0; index -= 1) {
        const entry = session.cleanups[index];
        try {
          await entry.cleanup();
        } catch (error) {
          console.warn("[Mio.views] " + entry.label + " cleanup failed", error);
        }
      }

      if (session.mountStarted && typeof session.spec.cleanup === "function") {
        try {
          await session.spec.cleanup(session.host, session.context);
        } catch (error) {
          console.warn("[Mio.views] cleanup failed for " + session.name, error);
        }
      }
      session.spec.mounted = false;
      session.host.remove();
    })();
    return session.disposePromise;
  }

  function createSession(name, spec, navigation) {
    const host = document.createElement("div");
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, "-");
    host.className = "view view-" + safeName;
    host.dataset.view = name;
    host.style.flex = "1";
    host.style.display = "flex";
    host.style.minWidth = "0";
    const context = {
      view: name,
      token: navigation.id,
      signal: navigation.controller.signal,
      isCurrent: () => (
        navigation.id === navigationToken && !navigation.controller.signal.aborted
      ),
    };
    return {
      name,
      spec,
      host,
      context,
      controller: navigation.controller,
      cleanups: [],
      mountStarted: false,
      activationStarted: false,
      disposePromise: null,
    };
  }

  async function performSwitch(navigation) {
    const { id, name, controller } = navigation;
    const isCurrent = () => id === navigationToken && !controller.signal.aborted;
    if (!isCurrent()) return false;

    if (currentSession) {
      const previous = currentSession;
      await disposeSession(previous, true);
      if (currentSession === previous) currentSession = null;
      activeView = null;
    } else if (activeView === "chat") {
      activeView = null;
    }
    if (!isCurrent()) return false;

    if (name === "chat") {
      const stage = getStage();
      stage.style.display = "none";
      stage.setAttribute("aria-busy", "false");
      stage.replaceChildren();
      showChat(true);
      commitActive("chat");
      return true;
    }

    const spec = registry.get(name);
    const session = createSession(name, spec, navigation);
    showLoading(name, session.host);

    try {
      const renderer = spec.mount || spec.render;
      session.mountStarted = typeof renderer === "function";
      await awaitLifecycle(
        session,
        renderer,
        [session.host, session.context],
        name + " mount",
      );
      spec.mounted = true;
      if (!isCurrent()) throw new NavigationCancelled();

      session.activationStarted = typeof spec.activate === "function";
      await awaitLifecycle(
        session,
        spec.activate,
        [session.context],
        name + " activate",
      );
      if (!isCurrent()) throw new NavigationCancelled();

      showMounted(session.host);
      currentSession = session;
      commitActive(name);
      return true;
    } catch (error) {
      const superseded = error instanceof NavigationCancelled || id !== navigationToken;
      await disposeSession(session, session.activationStarted);
      if (!superseded && id === navigationToken) showError(name, error);
      return false;
    }
  }

  function _switch(name) {
    const spec = registry.get(name);
    if (!spec && name !== "chat") {
      console.warn("[Mio.views] unknown view:", name);
      return Promise.resolve(false);
    }
    if (!pendingRequest && name === activeView) return Promise.resolve(true);
    if (pendingRequest?.name === name && !pendingRequest.controller.signal.aborted) {
      return pendingRequest.promise;
    }

    if (pendingRequest) pendingRequest.controller.abort();
    if (currentSession && !currentSession.context.signal.aborted) {
      currentSession.controller.abort();
    }

    const navigation = {
      id: ++navigationToken,
      name,
      controller: new AbortController(),
      promise: null,
    };
    const run = transitionTail.then(() => performSwitch(navigation));
    transitionTail = run.then(() => undefined, () => undefined);
    navigation.promise = run;
    pendingRequest = navigation;
    run.then(
      () => { if (pendingRequest?.id === navigation.id) pendingRequest = null; },
      () => { if (pendingRequest?.id === navigation.id) pendingRequest = null; },
    );
    return run;
  }

  function initialView() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved && (saved === "chat" || registry.has(saved))) return saved;
    } catch {}
    return DEFAULT_VIEW;
  }

  // Boot: after DOM is ready, switch to whatever the user had open last.
  function boot() {
    if (booted) return pendingRequest?.promise || Promise.resolve(true);
    booted = true;
    // Make sure the chat surface is visible while the rail decides.
    showChat(true);
    return _switch(initialView());
  }

  window.Mio.views = {
    register,
    list,
    switch: _switch,
    getActive,
    _boot: boot,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    // The deterministic loader exposes one promise for the complete module
    // registry. Wait for it so a persisted non-chat view is registered before
    // initialView() checks it.
    void Promise.resolve(window.Mio.modulesReady || undefined).then(
      boot,
      (error) => {
        console.warn("[Mio.views] module registry did not finish cleanly", error);
        return boot();
      },
    ).catch((error) => console.error("[Mio.views] boot failed", error));
  }
})();
