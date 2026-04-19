Mio UI — modular assets
=======================

Files in this directory are served at `/ui/assets/<filename>` and loaded
by `mio_ui.html` via `<script src>` / `<link>` tags.

New features should land here as standalone modules rather than growing
the monolithic `mio_ui.html`. Each module:

1. Exposes a small public API on a namespace (`window.Mio.<feature>`).
2. Reads/writes shared state through the `window.Mio` object rather
   than touching random globals.
3. Registers its own slash commands / keyboard shortcuts / settings UI
   in an `init()` function called from `mio_ui.html`'s bootstrap.

Keep each module under ~300 lines — split further if it grows.
