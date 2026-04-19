# Mio Clip — Safari web extension

Tiny browser extension that clips the current page, your selection, or a readability-cleaned copy of the article, and sends it to a running Mio instance via `POST /ui/api/ingest`. Mio stores the document under `~/.mio/ingest/` as a timestamped markdown file with YAML front-matter and auto-indexes it for local RAG, so you can `@`-reference it in chat right after clipping.

The code is a plain Web Extension (MV3). It's written with a `browser.*`-first adapter, so the same files also load unchanged in Chrome / Edge / Arc / Brave if you want.

**The repo ships source only — nothing is pre-installed or signed.**

---

## What's in this folder

```
browser-extension/
├── manifest.json         # MV3 manifest with browser_specific_settings.safari
├── background.js         # service worker: context menus + ingest dispatch
├── popup.html            # toolbar popup UI
├── popup.js              # popup controller
├── icons/                # drop 16/32/48/128 PNGs here before installing
└── README.md             # this file
```

---

## Install — Safari (macOS)

Safari Web Extensions need to be wrapped in a tiny Xcode host app. Apple ships the conversion tool:

```bash
# From the repo root. This does NOT modify the browser-extension/ files,
# it only generates a wrapper Xcode project next to them.
xcrun safari-web-extension-converter browser-extension/ \
    --project-location ~/Desktop/MioClipXcode \
    --app-name "Mio Clip" \
    --bundle-identifier dev.mio.clip \
    --no-open
```

Then in Xcode:

1. Open the generated `Mio Clip.xcodeproj`.
2. Build + run the host app once (⌘R).
3. In Safari: *Settings → Extensions*, enable **Mio Clip**.
4. If Safari rejects the unsigned extension, open *Develop menu → Allow Unsigned Extensions* first (Safari 16.4+).
5. Pin the extension to the toolbar.

Any edits you make under `browser-extension/` are picked up in Safari by rebuilding the host app (⌘B in Xcode, then toggle the extension off/on in Safari settings).

---

## Install — Chrome / Edge / Arc / Brave (optional)

The same folder loads directly in any Chromium browser:

1. `chrome://extensions` → enable *Developer mode* (top right).
2. *Load unpacked* → select `browser-extension/`.
3. Pin the extension to the toolbar.

---

## Use

- **Toolbar button** → popup with *Whole page* / *Selection only* and a tag input.
- **Right-click a page** → *Send this page to Mio*.
- **Right-click highlighted text** → *Send selection to Mio*.

A success notification confirms the ingest, including character count.

---

## Endpoint

Default target is `http://localhost:9090/ui/api/ingest`. Change it in the popup's *Mio endpoint* field — the value persists via `browser.storage.sync`.

---

## Payload

Whatever the extension POSTs ends up in the JSON body:

```json
{
  "url": "https://example.com/article",
  "title": "Article title",
  "text": "readability-extracted plain text",
  "html": "raw-ish article HTML",
  "selection": "only present when the user sent a selection",
  "tags": ["optional"],
  "target": "rag"
}
```

The server answers with `{id, path, url, title, summary, chars, indexed}`. The `id` matches the file stem under `~/.mio/ingest/` so you can reference it from the Mio UI's Docs view.

---

## Privacy

The extension only talks to the host you configure. Default is `localhost:9090`, so nothing leaves your machine unless you point it somewhere else. Page / selection content is only sent when you explicitly click *Clip*, never on page load.
