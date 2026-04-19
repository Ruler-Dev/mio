// view_workspaces.js — placeholder for the Workspaces view.
//
// Full implementation lands in the next iteration. Today: a clean empty
// state that hints at what's coming so switching to this view is an
// honest no-op rather than a blank screen or a hang.

(function () {
  window.Mio = window.Mio || {};
  const ready = () => {
    if (!window.Mio.views) return setTimeout(ready, 50);
    window.Mio.views.register("workspaces", {
      title: "Workspaces",
      mount(host) {
        host.innerHTML = `
          <div class="view-empty">
            <div class="view-empty-inner">
              <h1>Workspaces</h1>
              <p>
                Bundle a model, context size, system prompt, pinned prompts
                and a set of chats into one reusable "workspace." Switch
                between them as one click — no need to re-configure every
                time you change modes.
              </p>
              <p class="muted">Landing in the next commit.</p>
            </div>
          </div>
        `;
      },
    });
  };
  ready();
})();
