"""Local cumulative LLM Wiki MCP server owned by Mio.

The design follows Andrej Karpathy's LLM Wiki research direction: durable,
inspectable pages accumulate evidence instead of re-deriving every answer.
Original proposal: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

All content is confined to ``~/.mio/wiki`` by default. The server has no
network dependency and does not invoke a model by itself.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from mio.paths import mio_home

MAX_REQUEST_BYTES = 1024 * 1024
MAX_PAGE_BYTES = 2 * 1024 * 1024
_LINK_RE = re.compile(r"\[\[([a-zA-Z0-9][a-zA-Z0-9_-]{0,99})\]\]")


class WikiError(ValueError):
    pass


def default_wiki_root() -> Path:
    override = os.environ.get("MIO_WIKI_ROOT")
    return Path(override).expanduser() if override else mio_home() / "wiki"


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.casefold()).strip("-")[:100]
    if not slug:
        raise WikiError("title/slug must contain at least one letter or digit")
    return slug


def _strings(value: Iterable[Any] | None, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise WikiError(f"{field} must be an array of strings")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise WikiError(f"{field} must contain non-empty strings")
        result.append(item.strip())
    return list(dict.fromkeys(result))


class WikiStore:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or default_wiki_root()).expanduser().resolve()
        self.pages_dir = self.root / "pages"

    def _path(self, slug: str) -> Path:
        safe_slug = slugify(slug)
        path = (self.pages_dir / f"{safe_slug}.json").resolve()
        if self.pages_dir.resolve() not in path.parents:
            raise WikiError("page path escapes wiki root")
        return path

    def _load_path(self, path: Path) -> dict[str, Any]:
        try:
            if path.stat().st_size > MAX_PAGE_BYTES:
                raise WikiError(f"wiki page {path.stem!r} exceeds 2 MiB")
            page = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise WikiError(f"wiki page {path.stem!r} does not exist") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise WikiError(f"cannot read wiki page {path.stem!r}: {exc}") from exc
        if not isinstance(page, dict):
            raise WikiError(f"wiki page {path.stem!r} is malformed")
        return page

    def read(self, slug: str) -> dict[str, Any]:
        return self._load_path(self._path(slug))

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        if not self.pages_dir.exists():
            return []
        pages = []
        for path in sorted(self.pages_dir.glob("*.json")):
            try:
                page = self._load_path(path)
            except WikiError:
                continue
            pages.append(
                {
                    "slug": page.get("slug", path.stem),
                    "title": page.get("title", path.stem),
                    "tags": page.get("tags", []),
                    "updated_at": page.get("updated_at"),
                    "source_count": len(page.get("sources", [])),
                }
            )
        pages.sort(key=lambda page: str(page.get("updated_at") or ""), reverse=True)
        return pages[:limit]

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        terms = [term for term in re.findall(r"[\w-]+", query.casefold()) if term]
        if not terms:
            raise WikiError("search query cannot be empty")
        hits = []
        for metadata in self.list(limit=1000):
            page = self.read(metadata["slug"])
            title = str(page.get("title", "")).casefold()
            content = str(page.get("content", "")).casefold()
            tags = " ".join(page.get("tags", [])).casefold()
            score = sum(8 * title.count(term) + 3 * tags.count(term) + content.count(term) for term in terms)
            if score:
                snippet_at = min((content.find(term) for term in terms if term in content), default=0)
                raw_content = str(page.get("content", ""))
                start = max(0, snippet_at - 80)
                hits.append(
                    {
                        **metadata,
                        "score": score,
                        "snippet": raw_content[start : start + 320].strip(),
                    }
                )
        hits.sort(key=lambda hit: (-hit["score"], str(hit["slug"])))
        return hits[: max(1, min(int(limit), 100))]

    def write(
        self,
        *,
        title: str,
        content: str,
        sources: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        slug: str | None = None,
    ) -> dict[str, Any]:
        title = title.strip()
        content = content.strip()
        if not title or not content:
            raise WikiError("title and content are required")
        if len(content.encode("utf-8")) > MAX_PAGE_BYTES:
            raise WikiError("wiki page exceeds 2 MiB")
        safe_slug = slugify(slug or title)
        path = self._path(safe_slug)
        now = datetime.now(timezone.utc).isoformat()
        created_at = now
        revision = 1
        if path.exists():
            old = self._load_path(path)
            created_at = str(old.get("created_at") or now)
            revision = int(old.get("revision", 0)) + 1
        page = {
            "slug": safe_slug,
            "title": title,
            "content": content,
            "sources": _strings(sources, "sources"),
            "tags": _strings(tags, "tags"),
            "revision": revision,
            "created_at": created_at,
            "updated_at": now,
        }
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.pages_dir.chmod(0o700)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{safe_slug}-", suffix=".tmp", dir=self.pages_dir)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(page, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return page

    def ingest(
        self,
        *,
        title: str,
        content: str,
        source: str,
        tags: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        source = source.strip()
        if not source:
            raise WikiError("ingest requires a source")
        if not content.strip():
            raise WikiError("ingest requires content")
        slug = slugify(title)
        if self._path(slug).exists():
            old = self.read(slug)
            merged_content = f"{old.get('content', '').rstrip()}\n\n{content.strip()}".strip()
            sources = [*old.get("sources", []), source]
            merged_tags = [*old.get("tags", []), *(tags or [])]
        else:
            merged_content = content
            sources = [source]
            merged_tags = list(tags or [])
        return self.write(
            title=title,
            slug=slug,
            content=merged_content,
            sources=sources,
            tags=merged_tags,
        )

    def lint(self, slug: str | None = None) -> dict[str, Any]:
        slugs = [slugify(slug)] if slug else [page["slug"] for page in self.list(limit=1000)]
        known = {page["slug"] for page in self.list(limit=1000)}
        issues = []
        checked = 0
        for page_slug in slugs:
            try:
                page = self.read(page_slug)
            except WikiError as exc:
                issues.append({"slug": page_slug, "code": "unreadable", "message": str(exc)})
                continue
            checked += 1
            if not page.get("sources"):
                issues.append({"slug": page_slug, "code": "missing-sources", "message": "page has no provenance"})
            if len(str(page.get("content", ""))) < 80:
                issues.append({"slug": page_slug, "code": "thin-content", "message": "page has fewer than 80 characters"})
            for target in _LINK_RE.findall(str(page.get("content", ""))):
                normalized = slugify(target)
                if normalized not in known:
                    issues.append(
                        {"slug": page_slug, "code": "broken-link", "message": f"missing [[{normalized}]]"}
                    )
        return {"checked": checked, "issue_count": len(issues), "issues": issues}


def tool_definitions() -> list[dict[str, Any]]:
    common_page = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
            "tags": {"type": "array", "items": {"type": "string"}},
            "slug": {"type": "string"},
        },
        "required": ["title", "content"],
        "additionalProperties": False,
    }
    return [
        {
            "name": "llm_wiki_list",
            "description": "List local cumulative wiki pages.",
            "inputSchema": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 1000}},
                "additionalProperties": False,
            },
        },
        {
            "name": "llm_wiki_search",
            "description": "Search local wiki titles, tags, and page content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "llm_wiki_read",
            "description": "Read one page by slug.",
            "inputSchema": {
                "type": "object",
                "properties": {"slug": {"type": "string"}},
                "required": ["slug"],
                "additionalProperties": False,
            },
        },
        {
            "name": "llm_wiki_write",
            "description": "Create or replace a local wiki page; this is an explicit write.",
            "inputSchema": common_page,
        },
        {
            "name": "llm_wiki_ingest",
            "description": "Append sourced evidence to a cumulative page.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "source": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "content", "source"],
                "additionalProperties": False,
            },
        },
        {
            "name": "llm_wiki_lint",
            "description": "Check provenance, thin pages, and broken internal links.",
            "inputSchema": {
                "type": "object",
                "properties": {"slug": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    ]


def call_tool(store: WikiStore, name: str, arguments: Mapping[str, Any]) -> Any:
    if name == "llm_wiki_list":
        return store.list(limit=arguments.get("limit", 100))
    if name == "llm_wiki_search":
        return store.search(str(arguments.get("query", "")), limit=arguments.get("limit", 20))
    if name == "llm_wiki_read":
        return store.read(str(arguments.get("slug", "")))
    if name == "llm_wiki_write":
        return store.write(
            title=str(arguments.get("title", "")),
            content=str(arguments.get("content", "")),
            sources=arguments.get("sources"),
            tags=arguments.get("tags"),
            slug=arguments.get("slug"),
        )
    if name == "llm_wiki_ingest":
        return store.ingest(
            title=str(arguments.get("title", "")),
            content=str(arguments.get("content", "")),
            source=str(arguments.get("source", "")),
            tags=arguments.get("tags"),
        )
    if name == "llm_wiki_lint":
        return store.lint(arguments.get("slug"))
    raise WikiError(f"unknown wiki tool {name!r}")


def _tool_result(value: Any, *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
        "isError": is_error,
        "structuredContent": value if isinstance(value, (dict, list)) else {"value": value},
    }


def handle_request(message: Mapping[str, Any], store: WikiStore) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        return None
    try:
        if method == "initialize":
            result: Any = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "mio-llm-wiki", "version": "0.1.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": tool_definitions()}
        elif method == "tools/call":
            params = message.get("params") or {}
            if not isinstance(params, Mapping):
                raise WikiError("tools/call params must be an object")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, Mapping):
                raise WikiError("tool arguments must be an object")
            result = _tool_result(call_tool(store, str(params.get("name", "")), arguments))
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"method {method!r} not found"},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except (WikiError, TypeError, ValueError) as exc:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": _tool_result({"error": str(exc)}, is_error=True),
        }


def main() -> None:
    store = WikiStore()
    while True:
        raw_line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not raw_line:
            break
        if len(raw_line) > MAX_REQUEST_BYTES:
            while raw_line and not raw_line.endswith(b"\n"):
                raw_line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
            continue
        try:
            message = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(message, dict):
            continue
        response = handle_request(message, store)
        if response is not None:
            sys.stdout.write(json.dumps(response, separators=(",", ":"), ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
