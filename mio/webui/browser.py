"""agent-browser Python wrapper for web search and browsing.

Wraps the agent-browser CLI (Rust binary) via subprocess.
Provides web search by navigating to search engines, extracting results,
and returning structured data.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    domain: str


def _run(args: list[str], timeout: int = 30) -> dict:
    """Run an agent-browser command and parse JSON output."""
    cmd = ["agent-browser"] + args + ["--json"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return {"success": False, "error": result.stderr.strip()}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"success": True, "data": {"text": result.stdout}}
    except FileNotFoundError:
        return {"success": False, "error": "agent-browser not installed. Run: npm install -g @anthropic/agent-browser"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Browser operation timed out"}


def is_available() -> bool:
    """Check if agent-browser is installed."""
    try:
        result = subprocess.run(
            ["agent-browser", "--version"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def search_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the web using DuckDuckGo via agent-browser.

    Falls back to a simple HTTP request if agent-browser is not available.
    """
    if not is_available():
        return _search_fallback(query, max_results)

    # Navigate to DuckDuckGo
    _run(["open", f"https://duckduckgo.com/?q={_urlencode(query)}"])
    # Wait for results
    _run(["wait", "--load", "networkidle"])
    # Get page text
    resp = _run(["get", "text", "body"])

    if not resp.get("success"):
        return _search_fallback(query, max_results)

    text = resp.get("data", {}).get("text", "")
    results = _parse_ddg_results(text, max_results)

    # Close the browser tab
    _run(["close"])

    return results if results else _search_fallback(query, max_results)


def fetch_page(url: str) -> str:
    """Fetch a page's text content via agent-browser or urllib."""
    if is_available():
        _run(["open", url])
        _run(["wait", "--load", "networkidle"])
        resp = _run(["get", "text", "body"])
        _run(["close"])
        if resp.get("success"):
            text = resp.get("data", {}).get("text", "")
            if text.strip():
                return text

    text = _fetch_fallback(url)
    if text.strip():
        return text

    # Reddit subreddit JSON endpoints are case-sensitive and return empty
    # for the wrong case. Try to resolve the canonical name and retry.
    canonical = _reddit_resolve_case(url)
    if canonical and canonical != url:
        return _fetch_fallback(canonical)

    return ""


def _reddit_resolve_case(url: str) -> str | None:
    """If `url` targets a reddit subreddit with wrong casing, return the
    canonical URL. Otherwise return None.
    """
    import re, json as _json, ssl, urllib.parse, urllib.request

    m = re.match(r"^(https?://(?:www\.|old\.)?reddit\.com)/r/([^/?#]+)(/.*)?$", url)
    if not m:
        return None
    base, name, rest = m.group(1), m.group(2), m.group(3) or ""
    search_url = (
        "https://www.reddit.com/subreddits/search.json"
        f"?q={urllib.parse.quote(name)}&limit=5"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    req = urllib.request.Request(search_url, headers=headers)
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    try:
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = _json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    # The user-typed name may be wrong-case (hitting a banned/empty shell
    # sub) or a near-miss spelling (e.g. "locallama" → "LocalLLaMA"). Trust
    # Reddit's own search ranking and pick the highest-subscriber result;
    # Reddit already biases toward sub-name prefix matches.
    best = None
    best_subs = -1
    for child in data.get("data", {}).get("children", []):
        d = child.get("data", {})
        dn = d.get("display_name") or ""
        subs = d.get("subscribers") or 0
        if subs > best_subs:
            best = dn
            best_subs = subs
    if not best or best == name or best_subs < 1000:
        return None
    return f"{base}/r/{best}{rest}"


def _search_fallback(query: str, max_results: int) -> list[SearchResult]:
    """Fallback web search using DuckDuckGo HTML (no JS needed)."""
    import ssl
    import urllib.request
    import urllib.parse
    import re

    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)

    # Handle macOS Python SSL certificate issues
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
            html = response.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    results = []
    # Parse DuckDuckGo HTML results
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html, re.DOTALL
    )

    for href, title, snippet in blocks[:max_results]:
        # Clean HTML tags
        title = re.sub(r"<[^>]+>", "", title).strip()
        snippet = re.sub(r"<[^>]+>", "", snippet).strip()
        # Extract actual URL from DDG redirect
        actual_url = href
        if "uddg=" in href:
            match = re.search(r"uddg=([^&]+)", href)
            if match:
                actual_url = urllib.parse.unquote(match.group(1))
        domain = re.sub(r"https?://([^/]+).*", r"\1", actual_url)
        results.append(SearchResult(
            title=title, url=actual_url, snippet=snippet, domain=domain
        ))

    return results


def _fetch_fallback(url: str) -> str:
    """Fetch page content with urllib.

    Uses a realistic Chrome User-Agent plus Accept/Accept-Language headers so
    sites that sniff minimal UAs (Forbes, TechCrunch, Yahoo Finance, etc.) are
    less likely to return an anti-bot wall. Follows redirects and extracts
    text from <article>/<main> when present for better signal-to-noise.
    """
    import ssl
    import urllib.request

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity",
        "Cache-Control": "no-cache",
    }
    req = urllib.request.Request(url, headers=headers)
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=20, context=ctx) as response:
            html = response.read().decode("utf-8", errors="replace")
        import re
        # Drop noise
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
        html = re.sub(r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.DOTALL)
        # Prefer article/main body when present
        body = None
        for tag in ("article", "main"):
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html, flags=re.DOTALL | re.IGNORECASE)
            if m and len(m.group(1)) > 400:
                body = m.group(1)
                break
        source = body if body else html
        text = re.sub(r"<[^>]+>", " ", source)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:8000]
    except Exception:
        return ""


def _parse_ddg_results(text: str, max_results: int) -> list[SearchResult]:
    """Parse DuckDuckGo search results from page text."""
    # This is a best-effort parser for the accessibility tree text
    results = []
    lines = text.split("\n")
    i = 0
    while i < len(lines) and len(results) < max_results:
        line = lines[i].strip()
        if line and "http" in line.lower() and i + 1 < len(lines):
            url = line
            title = lines[i + 1].strip() if i + 1 < len(lines) else ""
            snippet = lines[i + 2].strip() if i + 2 < len(lines) else ""
            import re
            domain = re.sub(r"https?://([^/]+).*", r"\1", url)
            results.append(SearchResult(
                title=title or domain, url=url, snippet=snippet, domain=domain
            ))
            i += 3
        else:
            i += 1
    return results


def _urlencode(s: str) -> str:
    import urllib.parse
    return urllib.parse.quote_plus(s)
