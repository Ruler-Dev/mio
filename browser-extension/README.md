# Mio Clip — browser extension

Tiny Chrome/Chromium/Edge extension that clips the current page, a selection, or a readability-cleaned copy of the article, and sends it to a running Mio instance via `POST /ui/api/ingest`. The document is stored under `~/.mio/ingest/` as a timestamped markdown file and auto-indexed for local RAG, so you can `@` it in chat right after clipping.

## Install

1. Open `chrome://extensions` (or `edge://extensions`).
2. Enable **Developer mode** (top right).
3. Click **Load unpacked** and point it at this directory (`browser-extension/`).
4. (Optional) Pin the extension to your toolbar.

Icons: the repo ships `icons/` empty — drop any 16/32/48/128 PNGs in there before loading, or Chrome will use a default.

## Use

- **Toolbar button** → popup with *Whole page* / *Selection only* and a tag input.
- **Right-click a page** → *Send this page to Mio*.
- **Right-click highlighted text** → *Send selection to Mio*.

A success notification confirms the ingest, including character count.

## Endpoint

Default target is `http://localhost:9090/ui/api/ingest`. Change it in the popup's *Mio endpoint* field — the value persists via `chrome.storage.sync`.

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

The server answers with `{id, path, url, title, summary, chars, indexed}`. The `id` matches the file stem under `~/.mio/ingest/` so you can reference it from the UI's Docs view.

## Privacy

The extension only talks to the host you configure. Default is `localhost:9090`, so nothing leaves your machine unless you point it somewhere else. Page/selection content is only sent when you explicitly click *Clip*, never on page load.
