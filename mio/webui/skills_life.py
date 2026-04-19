"""Life-and-work skills — pure-Python, no heavyweight deps.

- scale_recipe: scale ingredient quantities and convert units.
- review_code: structured code-review scaffolding the model fills in.
- meeting_notes: structured meeting-notes extraction from a transcript.
- bookmark_save / bookmark_list / bookmark_search: personal reading list
  in ~/.mio/bookmarks.sqlite.
- color_palette: generate a 5-color palette from a seed hex.
- describe_image: best-effort caption + OCR of a local image path.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

PIMIO_DIR = Path.home() / ".mio"
PIMIO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Recipe scaler
# ============================================================
_UNIT_CONV = {
    # volume (ml baseline)
    "tsp": 4.929, "teaspoon": 4.929, "teaspoons": 4.929,
    "tbsp": 14.79, "tablespoon": 14.79, "tablespoons": 14.79,
    "cup": 236.6, "cups": 236.6,
    "ml": 1.0, "milliliter": 1.0, "milliliters": 1.0,
    "l": 1000.0, "liter": 1000.0, "liters": 1000.0,
    "fl oz": 29.57, "floz": 29.57, "fluid ounce": 29.57,
    # weight (g baseline)
    "g": None, "gram": None, "grams": None,
    "kg": None, "kilogram": None, "kilograms": None,
    "oz": None, "ounce": None, "ounces": None,
    "lb": None, "lbs": None, "pound": None, "pounds": None,
}


def _parse_qty(s: str) -> tuple[float | None, str]:
    """Return (number, unit) from 'qty unit' string; qty may be fractional."""
    s = s.strip()
    m = re.match(r"^(\d+(?:\.\d+)?|\d+\s*/\s*\d+|\d+\s+\d+/\d+)\s*([a-zA-Z\.\s]*)$", s)
    if not m:
        return None, s
    numraw, unit = m.group(1), (m.group(2) or "").strip().lower()
    if "/" in numraw:
        parts = numraw.replace(" ", "").split("/")
        if len(parts) == 2:
            try:
                n = float(parts[0]) / float(parts[1])
            except Exception:
                return None, s
        else:
            try:
                whole, frac = numraw.split(" ", 1)
                nf, df = frac.split("/")
                n = float(whole) + float(nf) / float(df)
            except Exception:
                return None, s
    else:
        try:
            n = float(numraw)
        except Exception:
            return None, s
    return n, unit


def _fmt_qty(n: float) -> str:
    if abs(n - round(n)) < 0.001:
        return str(int(round(n)))
    # Prefer sensible fractions for small cooking amounts
    for denom in (2, 3, 4, 8):
        for num in range(1, denom):
            f = num / denom
            if abs(n - (int(n) + f)) < 0.01:
                whole = int(n) if n >= 1 else 0
                return (f"{whole} " if whole else "") + f"{num}/{denom}"
    return f"{n:.2f}".rstrip("0").rstrip(".")


def scale_recipe(ingredients: list[str], scale: float = 2.0) -> dict:
    """Scale each ingredient line by `scale`. Input lines like
    '1 1/2 cups flour' or '200 g butter' get numerically scaled."""
    if not ingredients:
        return {"skill": "scale_recipe", "error": "no ingredients"}
    scaled: list[str] = []
    for line in ingredients:
        line = str(line).strip()
        if not line:
            continue
        # Find the leading quantity+unit segment
        m = re.match(r"^((?:\d+\s+\d+/\d+)|(?:\d+/\d+)|(?:\d+(?:\.\d+)?))\s*([a-zA-Z\.]+)?\s+(.+)$", line)
        if not m:
            scaled.append(line)
            continue
        num, unit, rest = m.group(1), (m.group(2) or ""), m.group(3)
        n, u = _parse_qty(num + (" " + unit if unit else ""))
        if n is None:
            scaled.append(line)
            continue
        new_n = n * scale
        scaled.append(f"{_fmt_qty(new_n)} {u} {rest}".strip())
    return {
        "skill": "scale_recipe",
        "scale": scale,
        "ingredients": scaled,
    }


# ============================================================
# Bookmarks / reading list
# ============================================================
_BK_DB = PIMIO_DIR / "bookmarks.sqlite"


def _bk_conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_BK_DB))
    c.executescript("""
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            snippet TEXT,
            tags TEXT,
            added INTEGER,
            read INTEGER DEFAULT 0
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS bookmarks_fts USING fts5(
            url, title, snippet, tags, content = 'bookmarks', content_rowid = 'id'
        );
    """)
    return c


def bookmark_save(url: str, title: str | None = None, snippet: str | None = None,
                  tags: list[str] | None = None) -> dict:
    if not url:
        return {"skill": "bookmark_save", "error": "url required"}
    tags_str = ",".join(tags) if tags else ""
    c = _bk_conn()
    cur = c.cursor()
    cur.execute(
        "INSERT INTO bookmarks (url, title, snippet, tags, added) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(url) DO UPDATE SET title = excluded.title, snippet = excluded.snippet, "
        "tags = excluded.tags",
        (url, title or url, snippet or "", tags_str, int(time.time()))
    )
    cur.execute("INSERT OR REPLACE INTO bookmarks_fts (rowid, url, title, snippet, tags) "
                "SELECT id, url, title, snippet, tags FROM bookmarks WHERE url = ?", (url,))
    c.commit()
    c.close()
    return {"skill": "bookmark_save", "ok": True, "url": url}


def bookmark_list(tag: str | None = None, limit: int = 50, unread_only: bool = False) -> dict:
    c = _bk_conn()
    cur = c.cursor()
    sql = "SELECT id, url, title, snippet, tags, added, read FROM bookmarks WHERE 1=1"
    args: list[Any] = []
    if tag:
        sql += " AND tags LIKE ?"
        args.append(f"%{tag}%")
    if unread_only:
        sql += " AND read = 0"
    sql += " ORDER BY added DESC LIMIT ?"
    args.append(int(limit))
    rows = cur.execute(sql, args).fetchall()
    c.close()
    return {
        "skill": "bookmark_list",
        "bookmarks": [
            {"id": r[0], "url": r[1], "title": r[2], "snippet": r[3],
             "tags": (r[4] or "").split(",") if r[4] else [],
             "added": r[5], "read": bool(r[6])}
            for r in rows
        ],
    }


def bookmark_search(query: str, limit: int = 20) -> dict:
    if not query:
        return {"skill": "bookmark_search", "error": "query required"}
    c = _bk_conn()
    cur = c.cursor()
    safe = re.sub(r"[\"\*\^\(\)]", " ", query).strip()
    try:
        rows = cur.execute(
            "SELECT b.id, b.url, b.title, b.snippet FROM bookmarks_fts f "
            "JOIN bookmarks b ON b.id = f.rowid WHERE bookmarks_fts MATCH ? LIMIT ?",
            (safe, int(limit))
        ).fetchall()
    except sqlite3.OperationalError as e:
        return {"skill": "bookmark_search", "error": f"fts5 error: {e}"}
    c.close()
    return {
        "skill": "bookmark_search",
        "results": [
            {"id": r[0], "url": r[1], "title": r[2], "snippet": r[3]}
            for r in rows
        ],
    }


# ============================================================
# Color palette
# ============================================================
def color_palette(seed: str = "#3b82f6", mode: str = "complementary") -> dict:
    """Produce a 5-color palette derived from a seed hex color."""
    s = seed.strip().lstrip("#")
    if len(s) not in (3, 6):
        return {"skill": "color_palette", "error": "seed must be 3- or 6-digit hex"}
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    # HSL conversion
    rr, gg, bb = r / 255, g / 255, b / 255
    mx, mn = max(rr, gg, bb), min(rr, gg, bb)
    h = s_ = l = 0.0
    l = (mx + mn) / 2
    if mx != mn:
        d = mx - mn
        s_ = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
        if mx == rr: h = ((gg - bb) / d + (6 if gg < bb else 0))
        elif mx == gg: h = (bb - rr) / d + 2
        else: h = (rr - gg) / d + 4
        h /= 6
    h *= 360
    def hsl(h_, s_, l_):
        h_ = h_ % 360
        import colorsys
        r2, g2, b2 = colorsys.hls_to_rgb(h_ / 360, l_, s_)
        return "#" + "".join(f"{int(round(x * 255)):02x}" for x in (r2, g2, b2))
    if mode == "complementary":
        colors = [hsl(h, s_, l), hsl(h + 180, s_, l),
                  hsl(h, s_, max(0.1, l - 0.2)), hsl(h, s_, min(0.9, l + 0.2)),
                  hsl(h + 180, s_, min(0.9, l + 0.15))]
    elif mode == "analogous":
        colors = [hsl(h - 30, s_, l), hsl(h - 15, s_, l),
                  hsl(h, s_, l), hsl(h + 15, s_, l), hsl(h + 30, s_, l)]
    elif mode == "triadic":
        colors = [hsl(h, s_, l), hsl(h + 120, s_, l), hsl(h + 240, s_, l),
                  hsl(h, s_, min(0.9, l + 0.2)), hsl(h, s_, max(0.1, l - 0.2))]
    elif mode == "monochromatic":
        colors = [hsl(h, s_, max(0.1, l - 0.3)),
                  hsl(h, s_, max(0.1, l - 0.15)),
                  hsl(h, s_, l),
                  hsl(h, s_, min(0.9, l + 0.15)),
                  hsl(h, s_, min(0.9, l + 0.3))]
    else:
        colors = [hsl(h, s_, l)]
    return {
        "skill": "color_palette",
        "seed": "#" + s,
        "mode": mode,
        "colors": colors,
    }


# ============================================================
# Image description (OCR + metadata)
# ============================================================
def describe_image(path: str) -> dict:
    """Best-effort caption + OCR of a local image. Returns dominant color
    estimate, dimensions, and extracted text if available."""
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return {"skill": "describe_image", "error": f"no such file: {path}"}
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return {"skill": "describe_image", "error": "Pillow not installed"}
    try:
        img = Image.open(p)
    except Exception as e:
        return {"skill": "describe_image", "error": f"could not open: {e}"}
    result = {
        "skill": "describe_image",
        "path": str(p),
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "format": (img.format or "").lower(),
    }
    # Dominant color — reduce to 5 colors via quantize
    try:
        palette_img = img.convert("RGB").resize((64, 64)).quantize(colors=5)
        pal = palette_img.getpalette()[: 5 * 3]
        counts = sorted(palette_img.getcolors() or [], key=lambda c: c[0], reverse=True)
        dominant = []
        for cnt, idx in counts[:5]:
            r, g, b = pal[idx * 3], pal[idx * 3 + 1], pal[idx * 3 + 2]
            dominant.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "weight": cnt})
        result["dominant_colors"] = dominant
    except Exception:
        pass
    # OCR
    try:
        import pytesseract  # type: ignore
        text = (pytesseract.image_to_string(img) or "").strip()
        if text:
            result["ocr_text"] = text[:8000]
    except Exception:
        pass
    return result


# ============================================================
# Code review (scaffold — model fills the heavy lifting)
# ============================================================
def review_code(code: str, language: str = "", focus: str = "") -> dict:
    """Run simple heuristic lint on `code` and return a structured skeleton
    the model can expand on with real analysis. Catches obvious issues:
    print-debug statements, TODO/FIXME, long functions, bare except, very
    long lines. The model should follow up with real semantic review.
    """
    if not code:
        return {"skill": "review_code", "error": "code required"}
    lines = code.split("\n")
    findings: list[dict] = []
    for i, ln in enumerate(lines, start=1):
        s = ln.rstrip()
        if re.search(r"\btodo\b|\bfixme\b|\bxxx\b|\bhack\b", s, re.I):
            findings.append({"line": i, "kind": "TODO marker", "text": s.strip()})
        if re.search(r"\bconsole\.log\b|\bprint\s*\(|^\s*debugger\b", s):
            findings.append({"line": i, "kind": "Debug statement", "text": s.strip()})
        if re.search(r"^\s*except\s*:\s*$|^\s*except\s+Exception\s*:\s*$", s):
            findings.append({"line": i, "kind": "Bare/broad except", "text": s.strip()})
        if len(s) > 120:
            findings.append({"line": i, "kind": "Long line (> 120 cols)", "text": s[:120] + "…"})
    # Functions over ~50 lines (Python/JS heuristic)
    func_starts = []
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(def|function|async function|const \w+\s*=\s*\(?.*\)?\s*=>)\b", ln):
            func_starts.append(i)
    for i, start in enumerate(func_starts):
        end = func_starts[i + 1] if i + 1 < len(func_starts) else len(lines)
        if end - start > 50:
            findings.append({"line": start + 1, "kind": "Long function (> 50 lines)",
                             "text": lines[start].strip()})
    return {
        "skill": "review_code",
        "language": language or "unspecified",
        "focus": focus,
        "line_count": len(lines),
        "findings": findings,
        "summary": (
            f"{len(findings)} heuristic findings. "
            "The model should now review this code semantically: correctness, "
            "security, performance, readability, edge cases."
        ),
    }


# ============================================================
# Meeting notes extractor (scaffold)
# ============================================================
def explain_regex(pattern: str) -> dict:
    """Walk through a regex pattern token-by-token, returning a list of
    { part, meaning } dicts the model can render as a table."""
    if not pattern:
        return {"skill": "explain_regex", "error": "pattern required"}
    import re as _re
    # Tokenize: anchors, character classes, quantifiers, groups, escapes, literals.
    parts: list[dict] = []
    i = 0
    p = pattern
    specials = {
        "\\d": "any digit (0-9)", "\\D": "any non-digit",
        "\\w": "any word character (letter, digit, underscore)",
        "\\W": "any non-word character",
        "\\s": "any whitespace", "\\S": "any non-whitespace",
        "\\b": "word boundary", "\\B": "non-word-boundary",
        "\\n": "newline", "\\t": "tab", "\\r": "carriage return",
        ".": "any character (except newline)",
        "^": "start of string/line",
        "$": "end of string/line",
    }
    while i < len(p):
        ch = p[i]
        if ch == "\\" and i + 1 < len(p):
            tok = p[i:i+2]
            parts.append({"part": tok, "meaning": specials.get(tok, f"literal '{p[i+1]}'")})
            i += 2
            continue
        if ch == "[":
            end = p.find("]", i + 1)
            if end == -1: break
            cls = p[i:end+1]
            parts.append({"part": cls, "meaning": f"character class — any of {cls[1:-1]}"})
            i = end + 1
            continue
        if ch == "(":
            # Find matching close paren ignoring escapes
            depth = 1
            j = i + 1
            while j < len(p) and depth:
                if p[j] == "\\": j += 2; continue
                if p[j] == "(": depth += 1
                elif p[j] == ")": depth -= 1
                j += 1
            grp = p[i:j]
            if grp.startswith("(?:"): meaning = "non-capturing group"
            elif grp.startswith("(?="): meaning = "positive lookahead"
            elif grp.startswith("(?!"): meaning = "negative lookahead"
            elif grp.startswith("(?<="): meaning = "positive lookbehind"
            elif grp.startswith("(?<!"): meaning = "negative lookbehind"
            else: meaning = "capturing group"
            parts.append({"part": grp, "meaning": meaning})
            i = j
            continue
        if ch in "*+?" and parts:
            lazy = (i + 1 < len(p) and p[i+1] == "?")
            q = ch + ("?" if lazy else "")
            meaning = {"*": "zero or more", "+": "one or more", "?": "zero or one"}[ch]
            if lazy: meaning += " (lazy)"
            parts.append({"part": q, "meaning": "quantifier — " + meaning})
            i += 2 if lazy else 1
            continue
        if ch == "{":
            end = p.find("}", i + 1)
            if end == -1: break
            qt = p[i:end+1]
            parts.append({"part": qt, "meaning": f"quantifier — repeat {qt[1:-1]} times"})
            i = end + 1
            continue
        if ch in specials:
            parts.append({"part": ch, "meaning": specials[ch]})
            i += 1
            continue
        if ch == "|":
            parts.append({"part": "|", "meaning": "alternation (OR)"})
            i += 1
            continue
        # Literal run
        j = i
        while j < len(p) and p[j] not in "\\[](){}.*+?|^$":
            j += 1
        if j > i:
            parts.append({"part": p[i:j], "meaning": f"literal text '{p[i:j]}'"})
            i = j
            continue
        parts.append({"part": ch, "meaning": "literal"})
        i += 1
    # Validate
    try:
        _re.compile(pattern)
        valid = True
        err = None
    except _re.error as e:
        valid = False
        err = str(e)
    return {
        "skill": "explain_regex",
        "pattern": pattern,
        "valid": valid,
        "error": err,
        "parts": parts,
    }


_CURRENCY_RATES = {
    # Rates relative to 1 USD; last updated 2025-04; good enough for
    # quick "convert 50 EUR to JPY" intents without network dependency.
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 152.0,
    "CNY": 7.24,
    "KRW": 1360.0,
    "INR": 83.5,
    "CAD": 1.37,
    "AUD": 1.53,
    "NZD": 1.67,
    "CHF": 0.91,
    "SEK": 10.9,
    "NOK": 10.8,
    "DKK": 6.87,
    "BRL": 5.2,
    "MXN": 17.0,
    "ARS": 920.0,
    "RUB": 93.0,
    "TRY": 32.0,
    "ZAR": 18.8,
    "SGD": 1.35,
    "HKD": 7.83,
    "THB": 36.5,
    "VND": 25500.0,
    "PHP": 56.5,
    "IDR": 16100.0,
    "MYR": 4.75,
    "AED": 3.67,
    "SAR": 3.75,
    "ILS": 3.75,
    "EGP": 47.5,
    "PKR": 278.0,
    "CZK": 23.4,
    "PLN": 4.0,
    "HUF": 365.0,
    "RON": 4.56,
    "BGN": 1.8,
    "UAH": 39.5,
    "NGN": 1600.0,
    "KES": 135.0,
    "BTC": 0.000014,
    "ETH": 0.00028,
}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Offline currency conversion using a bundled rate table (approx
    Apr 2025). Not for financial trading — just day-to-day math."""
    src = (from_currency or "USD").upper().strip()
    dst = (to_currency or "USD").upper().strip()
    if src not in _CURRENCY_RATES:
        return {"skill": "convert_currency", "error": f"unknown source: {src}"}
    if dst not in _CURRENCY_RATES:
        return {"skill": "convert_currency", "error": f"unknown target: {dst}"}
    try:
        amt = float(amount)
    except Exception:
        return {"skill": "convert_currency", "error": "amount must be a number"}
    usd = amt / _CURRENCY_RATES[src]
    result = usd * _CURRENCY_RATES[dst]
    return {
        "skill": "convert_currency",
        "input": {"amount": amt, "from": src},
        "output": {"amount": round(result, 4), "to": dst},
        "rate": round(_CURRENCY_RATES[dst] / _CURRENCY_RATES[src], 6),
        "note": "Offline rates ~Apr 2025 — not for trading",
    }


def url_preview(url: str) -> dict:
    """Fetch a URL and extract OpenGraph / Twitter / <title> metadata
    for a social-card-style preview."""
    import re as _re
    import urllib.request as _urlreq
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return {"skill": "url_preview", "error": "http(s) url required"}
    try:
        req = _urlreq.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
        })
        import ssl
        ctx = ssl.create_default_context()
        try:
            import certifi
            ctx.load_verify_locations(certifi.where())
        except ImportError:
            pass
        with _urlreq.urlopen(req, timeout=10, context=ctx) as r:
            raw = r.read(200_000).decode("utf-8", errors="replace")
    except Exception as e:
        return {"skill": "url_preview", "error": f"fetch failed: {e}"}

    def _meta(name: str) -> str:
        patterns = [
            rf'<meta\s+property=["\']{name}["\']\s+content=["\']([^"\']+)["\']',
            rf'<meta\s+name=["\']{name}["\']\s+content=["\']([^"\']+)["\']',
            rf'<meta\s+content=["\']([^"\']+)["\']\s+property=["\']{name}["\']',
        ]
        for pat in patterns:
            m = _re.search(pat, raw, _re.I)
            if m:
                return m.group(1).strip()
        return ""

    title_tag = ""
    m = _re.search(r"<title[^>]*>([^<]+)</title>", raw, _re.I)
    if m:
        title_tag = m.group(1).strip()

    og_title = _meta("og:title") or _meta("twitter:title") or title_tag
    og_desc = _meta("og:description") or _meta("twitter:description") or _meta("description")
    og_image = _meta("og:image") or _meta("twitter:image")
    og_site = _meta("og:site_name")

    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    return {
        "skill": "url_preview",
        "url": url,
        "title": og_title,
        "description": og_desc,
        "image": og_image,
        "site": og_site or host,
    }


def hn_top(limit: int = 10) -> dict:
    """Fetch current top stories from Hacker News (Firebase API, no key)."""
    import urllib.request as _urlreq
    try:
        ids = json.loads(
            _urlreq.urlopen("https://hacker-news.firebaseio.com/v0/topstories.json",
                            timeout=10).read()
        )[: int(limit)]
    except Exception as e:
        return {"skill": "hn_top", "error": f"fetch failed: {e}"}
    stories = []
    for i in ids:
        try:
            d = json.loads(
                _urlreq.urlopen(f"https://hacker-news.firebaseio.com/v0/item/{i}.json",
                                timeout=6).read()
            )
            stories.append({
                "id": d.get("id"),
                "title": d.get("title"),
                "url": d.get("url") or f"https://news.ycombinator.com/item?id={d.get('id')}",
                "score": d.get("score"),
                "comments": d.get("descendants"),
                "by": d.get("by"),
            })
        except Exception:
            continue
    return {"skill": "hn_top", "count": len(stories), "stories": stories}


def reddit_top(subreddit: str = "all", limit: int = 10, period: str = "day") -> dict:
    """Fetch top posts from a subreddit (public JSON endpoint)."""
    import urllib.request as _urlreq
    sub = (subreddit or "all").strip().strip("/").replace("r/", "")
    period = period if period in ("hour", "day", "week", "month", "year", "all") else "day"
    url = f"https://www.reddit.com/r/{sub}/top.json?t={period}&limit={int(limit)}"
    try:
        req = _urlreq.Request(url, headers={
            "User-Agent": "mio/1.0 (+local research)",
        })
        data = json.loads(_urlreq.urlopen(req, timeout=10).read())
    except Exception as e:
        return {"skill": "reddit_top", "error": f"fetch failed: {e}"}
    children = (data.get("data") or {}).get("children") or []
    posts = []
    for c in children:
        d = c.get("data") or {}
        posts.append({
            "title": d.get("title"),
            "url": "https://reddit.com" + (d.get("permalink") or ""),
            "score": d.get("score"),
            "comments": d.get("num_comments"),
            "subreddit": d.get("subreddit"),
            "flair": d.get("link_flair_text"),
        })
    return {"skill": "reddit_top", "subreddit": sub, "posts": posts}


# A tiny offline quotes library. Curated, attributed.
_QUOTES = [
    ("Marcus Aurelius", "stoic", "You have power over your mind — not outside events. Realize this, and you will find strength."),
    ("Marcus Aurelius", "stoic", "The impediment to action advances action. What stands in the way becomes the way."),
    ("Seneca", "stoic", "We suffer more in imagination than in reality."),
    ("Epictetus", "stoic", "It's not what happens to you, but how you react to it that matters."),
    ("Viktor Frankl", "freedom", "Between stimulus and response there is a space. In that space is our power to choose our response."),
    ("Richard Feynman", "learning", "What I cannot create, I do not understand."),
    ("Alan Kay", "tech", "The best way to predict the future is to invent it."),
    ("Grace Hopper", "tech", "The most dangerous phrase in the language is, we've always done it this way."),
    ("Ada Lovelace", "tech", "The more I study, the more insatiable do I feel my genius for it to be."),
    ("Leonardo da Vinci", "creativity", "Simplicity is the ultimate sophistication."),
    ("Steve Jobs", "design", "Design is not just what it looks like. Design is how it works."),
    ("Antoine de Saint-Exupéry", "design", "Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away."),
    ("Donald Knuth", "tech", "Premature optimization is the root of all evil."),
    ("Linus Torvalds", "tech", "Talk is cheap. Show me the code."),
    ("Martin Fowler", "tech", "Any fool can write code that a computer can understand. Good programmers write code that humans can understand."),
    ("Edsger Dijkstra", "tech", "Computer science is no more about computers than astronomy is about telescopes."),
    ("Albert Einstein", "science", "Imagination is more important than knowledge."),
    ("Carl Sagan", "science", "Somewhere, something incredible is waiting to be known."),
    ("Marie Curie", "science", "Nothing in life is to be feared, it is only to be understood."),
    ("Isaac Asimov", "science", "The most exciting phrase to hear in science is not 'Eureka!' but 'That's funny…'"),
    ("Maya Angelou", "wisdom", "I've learned that people will forget what you said, but people will never forget how you made them feel."),
    ("Rumi", "love", "The wound is the place where the light enters you."),
    ("Haruki Murakami", "writing", "Unfortunately, the clock is ticking, the hours are going by. The past increases, the future recedes."),
    ("Friedrich Nietzsche", "philosophy", "He who has a why to live can bear almost any how."),
    ("Sun Tzu", "strategy", "The supreme art of war is to subdue the enemy without fighting."),
    ("Lao Tzu", "taoism", "A journey of a thousand miles begins with a single step."),
    ("Confucius", "virtue", "It does not matter how slowly you go so long as you do not stop."),
    ("Aristotle", "virtue", "We are what we repeatedly do. Excellence, then, is not an act, but a habit."),
    ("Charlie Munger", "investing", "Take a simple idea and take it seriously."),
    ("Warren Buffett", "investing", "It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price."),
    ("Paul Graham", "startups", "Startups are counterintuitive. The things that matter are subtle."),
    ("Peter Thiel", "startups", "The best entrepreneurs know this: every great business is built around a secret that's hidden from the outside."),
    ("Jeff Bezos", "business", "Your brand is what other people say about you when you're not in the room."),
    ("Reid Hoffman", "career", "An entrepreneur is someone who jumps off a cliff and builds a plane on the way down."),
    ("Nelson Mandela", "leadership", "It always seems impossible until it's done."),
    ("Eleanor Roosevelt", "courage", "Do one thing every day that scares you."),
    ("Theodore Roosevelt", "effort", "Nothing in the world is worth having or worth doing unless it means effort, pain, difficulty."),
    ("Winston Churchill", "perseverance", "Success is not final, failure is not fatal: it is the courage to continue that counts."),
    ("Ralph Waldo Emerson", "self", "Do not go where the path may lead, go instead where there is no path and leave a trail."),
    ("Thoreau", "simplicity", "Our life is frittered away by detail. Simplify, simplify."),
    ("Oscar Wilde", "individuality", "Be yourself; everyone else is already taken."),
    ("Mark Twain", "humor", "The two most important days in your life are the day you are born and the day you find out why."),
    ("Shakespeare", "classic", "To thine own self be true."),
    ("Rumi", "spirit", "You are not a drop in the ocean. You are the entire ocean in a drop."),
    ("Anaïs Nin", "writing", "We write to taste life twice, in the moment and in retrospect."),
    ("Virginia Woolf", "writing", "Lock up your libraries if you like; but there is no gate, no lock, no bolt that you can set upon the freedom of my mind."),
    ("Toni Morrison", "writing", "If there's a book that you want to read, but it hasn't been written yet, then you must write it."),
    ("Ursula K. Le Guin", "writing", "The only thing that makes life possible is permanent, intolerable uncertainty: not knowing what comes next."),
    ("James Baldwin", "writing", "Not everything that is faced can be changed, but nothing can be changed until it is faced."),
    ("Hemingway", "writing", "Write hard and clear about what hurts."),
    ("Kurt Vonnegut", "writing", "We have to continually be jumping off cliffs and developing our wings on the way down."),
    ("Dorothy Parker", "humor", "I hate writing, I love having written."),
    ("John Lennon", "life", "Life is what happens to you while you're busy making other plans."),
    ("Bob Dylan", "change", "The times they are a-changin'."),
    ("Frederick Douglass", "justice", "Power concedes nothing without a demand. It never did and it never will."),
    ("MLK", "justice", "The arc of the moral universe is long, but it bends toward justice."),
    ("Malala Yousafzai", "courage", "One child, one teacher, one book, one pen can change the world."),
    ("Barack Obama", "hope", "Change will not come if we wait for some other person or some other time."),
    ("Anne Frank", "hope", "How wonderful it is that nobody need wait a single moment before starting to improve the world."),
    ("Gandhi", "change", "Be the change that you wish to see in the world."),
    ("Simone de Beauvoir", "feminism", "One is not born, but rather becomes, a woman."),
    ("Audre Lorde", "justice", "Your silence will not protect you."),
    ("Beyoncé", "self", "If everything was perfect, you would never learn and you would never grow."),
]


def http_request(url: str, method: str = "GET", headers: dict | None = None,
                  body: str = "", timeout: int = 15) -> dict:
    """Issue a one-shot HTTP request and return status, headers, body,
    and timing. Good for quick API debugging."""
    import urllib.request as _urlreq
    import urllib.error as _urlerr
    import time as _t
    if not url or not url.startswith(("http://", "https://")):
        return {"skill": "http_request", "error": "http(s) url required"}
    method = (method or "GET").upper()
    req_headers = dict(headers or {})
    data = None
    if body:
        if isinstance(body, (dict, list)):
            import json as _json
            data = _json.dumps(body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        else:
            data = str(body).encode("utf-8")
    req = _urlreq.Request(url, data=data, method=method, headers=req_headers)
    t0 = _t.time()
    try:
        import ssl as _ssl
        ctx = _ssl.create_default_context()
        try:
            import certifi
            ctx.load_verify_locations(certifi.where())
        except ImportError:
            pass
        resp = _urlreq.urlopen(req, timeout=int(timeout), context=ctx)
        raw = resp.read(200_000)
        text = raw.decode("utf-8", errors="replace")
        headers_out = dict(resp.getheaders())
        status = resp.status
    except _urlerr.HTTPError as e:
        text = (e.read() or b"").decode("utf-8", errors="replace")
        headers_out = dict(e.headers.items()) if e.headers else {}
        status = e.code
    except Exception as e:
        return {"skill": "http_request", "error": f"{type(e).__name__}: {e}"}
    elapsed_ms = int((_t.time() - t0) * 1000)
    return {
        "skill": "http_request",
        "url": url,
        "method": method,
        "status": status,
        "headers": headers_out,
        "body": text[:20000],
        "truncated": len(text) > 20000,
        "elapsed_ms": elapsed_ms,
    }


def reading_briefing(limit: int = 10) -> dict:
    """Summarize the top items in the local reading list."""
    listed = bookmark_list(limit=limit)
    bms = listed.get("bookmarks", [])
    summary = []
    for bm in bms:
        summary.append({
            "title": bm.get("title") or bm.get("url"),
            "url": bm.get("url"),
            "tags": bm.get("tags", []),
            "snippet": (bm.get("snippet") or "")[:200],
        })
    return {
        "skill": "reading_briefing",
        "count": len(summary),
        "items": summary,
        "note": "Use this as a briefing scaffold — the model should weave "
                "these links into a short 'what's in your reading list' summary.",
    }


def quote(topic: str = "", author: str = "") -> dict:
    """Random famous quote, optionally filtered by topic or author."""
    import random
    pool = list(_QUOTES)
    if topic:
        pool = [q for q in pool if topic.lower() in q[1].lower()]
    if author:
        pool = [q for q in pool if author.lower() in q[0].lower()]
    if not pool:
        pool = list(_QUOTES)
    chosen = random.choice(pool)
    return {
        "skill": "quote",
        "author": chosen[0],
        "topic": chosen[1],
        "text": chosen[2],
        "pool_size": len(pool),
    }


def meeting_notes(transcript: str) -> dict:
    """Extract rough scaffolding from a raw transcript so the model can
    fill in the real notes. Pulls out candidate attendees (capitalized
    first words before a colon), action-verb phrases, and decision markers.
    """
    if not transcript:
        return {"skill": "meeting_notes", "error": "transcript required"}
    lines = transcript.split("\n")
    attendees = set()
    decisions = []
    actions = []
    for ln in lines:
        ln_s = ln.strip()
        m = re.match(r"^([A-Z][a-zA-Z\-'’\. ]{1,30}):\s", ln_s)
        if m:
            attendees.add(m.group(1).strip())
        if re.search(r"\b(we (?:decided|agreed|concluded)|decision:|let'?s)\b", ln_s, re.I):
            decisions.append(ln_s[:240])
        if re.search(r"\b(will|should|action|todo|by \w+day)\b", ln_s, re.I):
            actions.append(ln_s[:240])
    return {
        "skill": "meeting_notes",
        "length_chars": len(transcript),
        "candidate_attendees": sorted(attendees)[:20],
        "decision_candidates": decisions[:10],
        "action_candidates": actions[:20],
        "note": (
            "Scaffold only — the model should transform these candidates into "
            "clean structured notes: Attendees, Discussion, Decisions, Action "
            "Items (owner + due date), Follow-ups."
        ),
    }
