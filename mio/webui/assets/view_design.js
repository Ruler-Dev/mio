// view_design.js — placeholder for Design Mode.
//
// Full Claude/Stitch-inspired canvas lands next: chat left, artifact
// panel right, Preview/Code/Diff tabs, version scrubber, parallel-
// variant generation with the tandem router, vibe chips.

(function () {
  window.Mio = window.Mio || {};
  const ready = () => {
    if (!window.Mio.views) return setTimeout(ready, 50);
    window.Mio.views.register("design", {
      title: "Design",
      mount(host) {
        host.innerHTML = `
          <div class="view-empty">
            <div class="view-empty-inner">
              <h1>Design Mode</h1>
              <p>
                A focused canvas for chatting about UI designs. The model
                emits React / HTML artifacts you can preview, diff, and
                scrub through prior versions. Uses the tandem router to
                generate multiple design variants in parallel.
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
