"""Local folder RAG — index .md/.txt/.py/.html/.json files under a folder
into a SQLite FTS5 index, then query it with `search_local_folder`.

Self-contained, zero external dependencies beyond the stdlib sqlite3.
Good enough for a personal knowledge base up to ~100k files. For
embedding-based semantic search, swap out FTS5 for a later pass.
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path

RAG_DB = Path.home() / ".mio" / "rag.sqlite"
RAG_DB.parent.mkdir(parents=True, exist_ok=True)

_TEXT_EXTS = {
    ".md", ".markdown", ".txt", ".rst",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".sh", ".bash",
    ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".xml",
    ".html", ".css", ".scss", ".sql", ".org", ".tex",
}

_SKIP_DIRS = {
    "node_modules", ".git", ".venv", "venv", "env", "__pycache__",
    "dist", "build", ".cache", ".next", "target", ".svelte-kit",
    "site-packages",
}

_MAX_FILE_BYTES = 2_000_000  # 2 MB per file


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(RAG_DB))
    con.executescript("""
        CREATE TABLE IF NOT EXISTS indexes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            label TEXT,
            indexed_at INTEGER,
            file_count INTEGER DEFAULT 0
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(
            index_id UNINDEXED,
            path,
            title,
            body,
            tokenize = 'porter unicode61'
        );
    """)
    return con


def _iter_text_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() not in _TEXT_EXTS:
                continue
            try:
                if p.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield p


def index_folder(path: str, label: str | None = None, replace: bool = True) -> dict:
    """Scan `path` recursively for text files and add them to the local
    RAG index. If `replace=True`, prior content for this folder is wiped.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        return {"skill": "index_folder", "error": f"not a directory: {root}"}
    label = label or root.name
    con = _connect()
    cur = con.cursor()
    # Upsert the index row
    cur.execute("INSERT INTO indexes (path, label, indexed_at) VALUES (?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET label = excluded.label, "
                "indexed_at = excluded.indexed_at",
                (str(root), label, int(time.time())))
    cur.execute("SELECT id FROM indexes WHERE path = ?", (str(root),))
    index_id = cur.fetchone()[0]
    if replace:
        cur.execute("DELETE FROM documents WHERE index_id = ?", (index_id,))
    count = 0
    for p in _iter_text_files(root):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        title = p.relative_to(root).as_posix()
        cur.execute("INSERT INTO documents (index_id, path, title, body) VALUES (?, ?, ?, ?)",
                    (index_id, str(p), title, text))
        count += 1
    cur.execute("UPDATE indexes SET file_count = ?, indexed_at = ? WHERE id = ?",
                (count, int(time.time()), index_id))
    con.commit()
    con.close()
    return {
        "skill": "index_folder",
        "index_id": index_id,
        "path": str(root),
        "label": label,
        "file_count": count,
    }


def list_indexes() -> dict:
    con = _connect()
    cur = con.cursor()
    rows = cur.execute(
        "SELECT id, path, label, file_count, indexed_at FROM indexes ORDER BY indexed_at DESC"
    ).fetchall()
    con.close()
    return {
        "skill": "list_indexes",
        "indexes": [
            {"id": r[0], "path": r[1], "label": r[2], "file_count": r[3], "indexed_at": r[4]}
            for r in rows
        ],
    }


def search_local_folder(
    query: str,
    limit: int = 8,
    index_label: str | None = None,
    snippet_chars: int = 280,
) -> dict:
    """Full-text search against the local RAG index. Returns file paths,
    titles, and a centered snippet around the match.
    """
    if not query or not query.strip():
        return {"skill": "search_local_folder", "error": "empty query"}
    con = _connect()
    cur = con.cursor()
    # Restrict to a single index label if requested
    args: list = []
    sql = (
        "SELECT d.path, d.title, snippet(documents, 3, '«', '»', ' … ', 12) "
        "FROM documents d JOIN indexes i ON i.id = d.index_id "
        "WHERE documents MATCH ?"
    )
    # Escape FTS5 special tokens naively
    safe_q = re.sub(r'[\"\*\^\(\)]', ' ', query).strip()
    args.append(safe_q)
    if index_label:
        sql += " AND i.label = ?"
        args.append(index_label)
    sql += " LIMIT ?"
    args.append(int(limit))
    try:
        rows = cur.execute(sql, args).fetchall()
    except sqlite3.OperationalError as e:
        return {"skill": "search_local_folder", "error": f"fts5 error: {e}"}
    con.close()
    results = [
        {"path": r[0], "title": r[1], "snippet": (r[2] or "")[:snippet_chars]}
        for r in rows
    ]
    return {
        "skill": "search_local_folder",
        "query": query,
        "count": len(results),
        "results": results,
    }


def drop_index(index_id: int) -> dict:
    con = _connect()
    cur = con.cursor()
    cur.execute("DELETE FROM documents WHERE index_id = ?", (index_id,))
    cur.execute("DELETE FROM indexes WHERE id = ?", (index_id,))
    con.commit()
    con.close()
    return {"skill": "drop_index", "id": index_id, "ok": True}
