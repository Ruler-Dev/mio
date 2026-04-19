// view_obsidian.js — placeholder for the Obsidian integration view.
//
// Planned for the next iteration:
//   - Point Mio at your Obsidian vault path (persisted in ~/.mio/config)
//   - Browse the vault tree, open notes in an inline editor
//   - @-mention notes in chat; Mio reads the note as context
//   - Write notes back from chat with a "save as note" button on artifacts
//   - Respect existing links (wikilinks / markdown links)

(function () {
  window.Mio = window.Mio || {};
  const ready = () => {
    if (!window.Mio.views) return setTimeout(ready, 50);
    window.Mio.views.register("obsidian", {
      title: "Obsidian",
      mount(host) {
        host.innerHTML = `
          <div class="view-empty">
            <div class="view-empty-inner">
              <h1>Obsidian</h1>
              <p>
                First-class Obsidian vault integration — browse notes,
                @-mention them in chat, write new notes from chat output.
                Not just RAG: real bidirectional link awareness.
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
