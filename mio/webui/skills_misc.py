"""Miscellaneous utility skills (Phase B).

- generate_qr_code       → qrcode
- generate_ical          → icalendar
- generate_csv           → csv stdlib
- generate_sqlite_db     → sqlite3 stdlib
- generate_resume        → reportlab templated
- generate_invoice       → reportlab templated
- extract_pdf_text       → pdfplumber
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path


def _output_path(filename: str | None, ext: str) -> Path:
    fn = filename or f"mio-{int(time.time())}{ext}"
    if not fn.endswith(ext):
        fn = fn + ext
    p = Path.home() / "Downloads" / fn
    p.parent.mkdir(exist_ok=True)
    return p


# ============================================================
# QR code
# ============================================================
def generate_qr_code(data: str, filename: str | None = None, size: int = 10) -> dict:
    try:
        import qrcode
    except ImportError:
        return {"skill": "generate_qr_code", "error": "qrcode not installed"}
    out = _output_path(filename, ".png")
    img = qrcode.make(data, box_size=size, border=2)
    img.save(str(out))
    return {"skill": "generate_qr_code", "path": str(out), "filename": out.name}


# ============================================================
# iCal
# ============================================================
def generate_ical(
    title: str,
    events: list[dict],
    filename: str | None = None,
) -> dict:
    """`events` = [{summary, start, end, location?, description?, ...}]
    `start`/`end` can be ISO 8601 datetime strings or date-only strings."""
    try:
        from icalendar import Calendar, Event
        from datetime import datetime, date
    except ImportError:
        return {"skill": "generate_ical", "error": "icalendar not installed"}

    def _parse(s):
        if not s:
            return None
        if isinstance(s, (datetime, date)):
            return s
        if len(s) == 10:
            return datetime.strptime(s, "%Y-%m-%d").date()
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    cal = Calendar()
    cal.add("prodid", "-//Mio//EN")
    cal.add("version", "2.0")
    cal.add("x-wr-calname", title)
    for e in events or []:
        ev = Event()
        ev.add("summary", e.get("summary") or e.get("title") or "Event")
        if e.get("description"):
            ev.add("description", e["description"])
        if e.get("location"):
            ev.add("location", e["location"])
        start = _parse(e.get("start") or e.get("dtstart"))
        end = _parse(e.get("end") or e.get("dtend"))
        if start:
            ev.add("dtstart", start)
        if end:
            ev.add("dtend", end)
        cal.add_component(ev)

    out = _output_path(filename, ".ics")
    with open(out, "wb") as f:
        f.write(cal.to_ical())
    return {"skill": "generate_ical", "path": str(out), "filename": out.name,
            "event_count": len(events or [])}


# ============================================================
# CSV
# ============================================================
def generate_markdown(
    title: str,
    content: str,
    filename: str | None = None,
    tags: list[str] | None = None,
    vault_path: str | None = None,
    frontmatter: dict | None = None,
) -> dict:
    """Write a Markdown file with optional YAML frontmatter (Obsidian-friendly).

    - `title` becomes the filename stem if no `filename` given, and the first H1.
    - `tags` populate `tags:` in the frontmatter.
    - `vault_path` overrides the default ~/Downloads location; point this at
      an Obsidian vault path to drop the note directly into your vault.
    - `frontmatter` is merged with {title, tags, created} and written as YAML.
    """
    import datetime as _dt
    from pathlib import Path as _Path

    stem = (filename or title or "mio-note").strip()
    if not stem.endswith(".md"):
        stem = stem + ".md"
    # Sanitize slashes / path separators
    stem = stem.replace("/", "-").replace("\\", "-").strip()
    root = _Path(vault_path).expanduser() if vault_path else _Path.home() / "Downloads"
    root.mkdir(parents=True, exist_ok=True)
    out = root / stem

    fm = {
        "title": title,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if tags:
        fm["tags"] = list(tags) if isinstance(tags, (list, tuple)) else [str(tags)]
    if frontmatter:
        fm.update(frontmatter)

    def _yaml_val(v):
        if isinstance(v, list):
            return "[" + ", ".join(_yaml_val(x) for x in v) + "]"
        if isinstance(v, (int, float, bool)) or v is None:
            return str(v) if v is not None else ""
        s = str(v)
        return s if not any(c in s for c in ":#[]{},&*?|>'\"") else '"' + s.replace('"', '\\"') + '"'

    fm_lines = ["---"] + [f"{k}: {_yaml_val(v)}" for k, v in fm.items()] + ["---", ""]
    body = content or ""
    if not body.lstrip().startswith("#") and title:
        body = f"# {title}\n\n" + body
    out.write_text("\n".join(fm_lines) + body, encoding="utf-8")
    return {
        "skill": "generate_markdown",
        "path": str(out),
        "filename": out.name,
        "vault": vault_path or "~/Downloads",
        "size": out.stat().st_size,
    }


def generate_csv(
    headers: list[str],
    rows: list[list],
    filename: str | None = None,
) -> dict:
    out = _output_path(filename, ".csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if headers:
            w.writerow(headers)
        for r in rows or []:
            w.writerow(r if isinstance(r, (list, tuple)) else [r])
    return {"skill": "generate_csv", "path": str(out), "filename": out.name,
            "row_count": len(rows or [])}


# ============================================================
# SQLite DB
# ============================================================
def generate_sqlite_db(
    table: str,
    headers: list[str],
    rows: list[list],
    filename: str | None = None,
) -> dict:
    import sqlite3
    import re
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        return {"skill": "generate_sqlite_db", "error": f"invalid table name: {table}"}
    for h in headers:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", h):
            return {"skill": "generate_sqlite_db", "error": f"invalid column: {h}"}
    out = _output_path(filename, ".sqlite")
    if out.exists():
        out.unlink()
    conn = sqlite3.connect(str(out))
    cols = ", ".join(f'"{h}" TEXT' for h in headers)
    conn.execute(f'CREATE TABLE "{table}" ({cols})')
    conn.executemany(
        f'INSERT INTO "{table}" VALUES ({",".join("?" * len(headers))})',
        [[str(x) for x in r] for r in rows or []],
    )
    conn.commit()
    conn.close()
    return {"skill": "generate_sqlite_db", "path": str(out), "filename": out.name,
            "table": table, "row_count": len(rows or [])}


# ============================================================
# Resume
# ============================================================
def generate_resume(profile: dict, filename: str | None = None) -> dict:
    """`profile` = {name, title, contact:{email,phone,location,website},
    summary, experience:[{role, company, dates, bullets:[...]}],
    education:[{degree, school, dates}], skills:[...]}"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        )
    except ImportError:
        return {"skill": "generate_resume", "error": "reportlab not installed"}

    ACCENT = colors.HexColor("#3b82f6")
    MUTED = colors.HexColor("#6f6f6f")
    TEXT = colors.HexColor("#171717")

    out = _output_path(filename, ".pdf")
    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )

    name_style = ParagraphStyle("Name", fontName="Helvetica-Bold",
                                fontSize=26, leading=30, textColor=TEXT)
    role_style = ParagraphStyle("Role", fontName="Helvetica",
                                fontSize=13, leading=17, textColor=ACCENT, spaceAfter=4)
    contact_style = ParagraphStyle("Contact", fontName="Helvetica",
                                   fontSize=9, leading=13, textColor=MUTED, spaceAfter=14)
    section_style = ParagraphStyle("Sec", fontName="Helvetica-Bold",
                                   fontSize=11, leading=16, textColor=ACCENT,
                                   spaceBefore=12, spaceAfter=4,
                                   borderPadding=(0, 0, 3, 0),
                                   borderColor=ACCENT, borderWidth=0)
    job_style = ParagraphStyle("Job", fontName="Helvetica-Bold",
                               fontSize=10.5, leading=14, textColor=TEXT)
    meta_style = ParagraphStyle("Meta", fontName="Helvetica-Oblique",
                                fontSize=9, leading=12, textColor=MUTED, spaceAfter=3)
    body_style = ParagraphStyle("Body", fontName="Helvetica",
                                fontSize=10, leading=14, textColor=TEXT,
                                leftIndent=12, bulletIndent=2, spaceAfter=2)

    story = []
    story.append(Paragraph(profile.get("name") or "Name", name_style))
    if profile.get("title"):
        story.append(Paragraph(profile["title"], role_style))
    contact = profile.get("contact") or {}
    contact_parts = [v for v in [contact.get("email"), contact.get("phone"),
                                 contact.get("location"), contact.get("website")] if v]
    if contact_parts:
        story.append(Paragraph(" · ".join(contact_parts), contact_style))

    if profile.get("summary"):
        story.append(Paragraph("SUMMARY", section_style))
        story.append(Paragraph(profile["summary"], body_style))

    exp = profile.get("experience") or []
    if exp:
        story.append(Paragraph("EXPERIENCE", section_style))
        for j in exp:
            role = j.get("role") or ""
            company = j.get("company") or ""
            dates = j.get("dates") or ""
            story.append(Paragraph(f"{role} — {company}", job_style))
            if dates:
                story.append(Paragraph(dates, meta_style))
            for b in j.get("bullets") or []:
                story.append(Paragraph(f"• {b}", body_style))
            story.append(Spacer(1, 3))

    edu = profile.get("education") or []
    if edu:
        story.append(Paragraph("EDUCATION", section_style))
        for e in edu:
            story.append(Paragraph(
                f"{e.get('degree','')} — {e.get('school','')}", job_style))
            if e.get("dates"):
                story.append(Paragraph(e["dates"], meta_style))

    skills = profile.get("skills") or []
    if skills:
        story.append(Paragraph("SKILLS", section_style))
        story.append(Paragraph(" · ".join(str(s) for s in skills), body_style))

    doc.build(story)
    return {"skill": "generate_resume", "path": str(out), "filename": out.name}


# ============================================================
# Invoice
# ============================================================
def generate_invoice(
    issuer: dict,
    recipient: dict,
    items: list[dict],
    invoice_number: str = "",
    invoice_date: str = "",
    due_date: str = "",
    currency: str = "EUR",
    tax_rate: float = 0.0,
    notes: str = "",
    filename: str | None = None,
) -> dict:
    """`items` = [{description, qty, unit_price}]"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        )
    except ImportError:
        return {"skill": "generate_invoice", "error": "reportlab not installed"}

    ACCENT = colors.HexColor("#3b82f6")
    HEAD = colors.HexColor("#1F4E78")
    SOFT = colors.HexColor("#F3F6FB")
    TEXT = colors.HexColor("#171717")
    MUTED = colors.HexColor("#6f6f6f")

    out = _output_path(filename, ".pdf")
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            leftMargin=22 * mm, rightMargin=22 * mm,
                            topMargin=22 * mm, bottomMargin=22 * mm)
    big = ParagraphStyle("Big", fontName="Helvetica-Bold", fontSize=28,
                         leading=32, textColor=TEXT)
    muted = ParagraphStyle("Muted", fontName="Helvetica", fontSize=9,
                           leading=13, textColor=MUTED)
    label = ParagraphStyle("Label", fontName="Helvetica-Bold", fontSize=9,
                           leading=12, textColor=ACCENT,
                           spaceBefore=4, spaceAfter=2)
    val = ParagraphStyle("Val", fontName="Helvetica", fontSize=10, leading=14,
                         textColor=TEXT)

    story = [Paragraph("INVOICE", big), Spacer(1, 4)]
    if invoice_number:
        story.append(Paragraph(f"Invoice #{invoice_number}", muted))
    if invoice_date:
        story.append(Paragraph(f"Issued: {invoice_date}", muted))
    if due_date:
        story.append(Paragraph(f"Due: {due_date}", muted))
    story.append(Spacer(1, 12))

    def _party_block(d):
        lines = [d.get("name", ""),
                 d.get("address", ""),
                 d.get("city", ""),
                 d.get("country", ""),
                 d.get("tax_id", ""),
                 d.get("email", "")]
        return [Paragraph(ln, val) for ln in lines if ln]

    data = [[
        [Paragraph("FROM", label), *_party_block(issuer)],
        [Paragraph("BILL TO", label), *_party_block(recipient)],
    ]]
    t = Table(data, colWidths=[85 * mm, 85 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    rows = [["#", "Description", "Qty", "Unit", "Total"]]
    subtotal = 0.0
    for i, it in enumerate(items or [], start=1):
        qty = float(it.get("qty", 1) or 1)
        unit = float(it.get("unit_price", 0) or 0)
        total = qty * unit
        subtotal += total
        rows.append([
            str(i),
            it.get("description", ""),
            f"{qty:g}",
            f"{unit:.2f} {currency}",
            f"{total:.2f} {currency}",
        ])
    tax = subtotal * (tax_rate / 100.0)
    grand = subtotal + tax

    tbl = Table(rows, colWidths=[12 * mm, 85 * mm, 18 * mm, 25 * mm, 30 * mm],
                repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
        ("FONTSIZE", (0, 1), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SOFT, colors.white]),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    totals = [
        ["Subtotal", f"{subtotal:.2f} {currency}"],
        [f"Tax ({tax_rate:g}%)", f"{tax:.2f} {currency}"],
        ["Total", f"{grand:.2f} {currency}"],
    ]
    tt = Table(totals, colWidths=[55 * mm, 35 * mm], hAlign="RIGHT")
    tt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (-1, 0), (-1, -1), "RIGHT"),
        ("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, -1), (-1, -1), 12),
        ("TEXTCOLOR", (0, -1), (-1, -1), ACCENT),
        ("TOPPADDING", (0, -1), (-1, -1), 6),
    ]))
    story.append(tt)

    if notes:
        story.append(Spacer(1, 18))
        story.append(Paragraph("NOTES", label))
        story.append(Paragraph(notes, val))

    doc.build(story)
    return {"skill": "generate_invoice", "path": str(out), "filename": out.name,
            "subtotal": round(subtotal, 2), "tax": round(tax, 2),
            "total": round(grand, 2), "currency": currency}


# ============================================================
# Translation (MyMemory free API — no key required)
# ============================================================
def translate_text(text: str, target: str, source: str = "auto") -> dict:
    """Translate `text` into `target` language (ISO code, e.g. 'es', 'ja').
    Uses MyMemory free endpoint — no key, rate limited to anon quota."""
    import urllib.parse
    import urllib.request
    import ssl
    import json as _json

    def _ctx():
        c = ssl.create_default_context()
        try:
            import certifi
            c.load_verify_locations(certifi.where())
        except ImportError:
            pass
        return c

    if not text.strip():
        return {"skill": "translate_text", "error": "empty text"}
    q = urllib.parse.quote(text[:500])
    langpair = f"{source if source != 'auto' else 'autodetect'}|{target}"
    url = f"https://api.mymemory.translated.net/get?q={q}&langpair={urllib.parse.quote(langpair)}"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 Mio/1.0",
        })
        with urllib.request.urlopen(req, timeout=15, context=_ctx()) as r:
            data = _json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        return {"skill": "translate_text", "error": str(e)}
    rd = data.get("responseData") or {}
    return {
        "skill": "translate_text",
        "translated": rd.get("translatedText", ""),
        "source": source,
        "target": target,
        "match": rd.get("match", 0),
    }


# ============================================================
# Media databases (no API keys)
#   - Jikan (MyAnimeList) for anime / manga
#   - TVmaze for TV / movies
#   - Wikipedia REST for games / fallback + images
# ============================================================
def _http_json(url: str, timeout: int = 15) -> dict:
    import ssl as _ssl, urllib.request as _ur, json as _json
    req = _ur.Request(url, headers={
        "User-Agent": "Mio/1.0 (+https://github.com/Ruler-Dev/mio)",
        "Accept": "application/json",
    })
    ctx = _ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    with _ur.urlopen(req, timeout=timeout, context=ctx) as r:
        return _json.loads(r.read().decode("utf-8", errors="replace"))


def _trim(s: str, n: int = 600) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def find_anime(query: str = "", genre: str = "", count: int = 3, random: bool = False) -> dict:
    """Search Jikan (unofficial MyAnimeList API). If random=True returns a
    single random anime; otherwise searches by query/genre and returns `count`.
    Genre examples: action, romance, comedy, sci-fi, fantasy, horror, slice-of-life.
    """
    import urllib.parse as _up
    try:
        if random:
            data = _http_json("https://api.jikan.moe/v4/random/anime")
            rows = [data.get("data", {})]
        else:
            if genre and not query:
                # Genre-only: list top by score in that genre
                gmap = {}  # lazy-load genre ids when needed
                try:
                    gdata = _http_json("https://api.jikan.moe/v4/genres/anime")
                    gmap = {g["name"].lower(): g["mal_id"] for g in gdata.get("data", [])}
                except Exception:
                    pass
                gid = gmap.get(genre.lower())
                url = f"https://api.jikan.moe/v4/anime?order_by=score&sort=desc&limit={count}"
                if gid:
                    url += f"&genres={gid}"
                rows = _http_json(url).get("data", [])[:count]
            else:
                # When searching by a specific title, let Jikan rank by
                # relevance (omitting order_by). When it's a vague query
                # (short keyword) we may still want the score ordering,
                # but the default is relevance — model asked for a
                # title, it should get that title first.
                q = _up.quote(query or "anime")
                url = f"https://api.jikan.moe/v4/anime?q={q}&limit={max(count, 5)}"
                rows = _http_json(url).get("data", [])[:count]
    except Exception as e:
        return {"skill": "find_anime", "error": str(e)}

    results = []
    for a in rows:
        if not a:
            continue
        img = (a.get("images") or {}).get("jpg") or (a.get("images") or {}).get("webp") or {}
        # Extract YouTube trailer ID if present
        trailer_embed = (a.get("trailer") or {}).get("embed_url", "") or ""
        trailer_id = None
        import re as _re_t
        m_yt = _re_t.search(r"/embed/([\w-]{11})", trailer_embed)
        if m_yt: trailer_id = m_yt.group(1)

        results.append({
            "title": a.get("title") or a.get("title_english") or "?",
            "title_english": a.get("title_english"),
            "title_japanese": a.get("title_japanese"),
            "synopsis": _trim(a.get("synopsis", ""), 700),
            "episodes": a.get("episodes"),
            "type": a.get("type"),
            "status": a.get("status"),
            "year": a.get("year"),
            "score": a.get("score"),
            "genres": [g.get("name") for g in (a.get("genres") or [])],
            "studios": [s.get("name") for s in (a.get("studios") or [])],
            "image": img.get("large_image_url") or img.get("image_url"),
            "url": a.get("url"),
            "trailer_id": trailer_id,
            "kind": "anime",
        })
    return {"skill": "find_anime", "count": len(results), "results": results}


def find_manga(query: str = "", genre: str = "", count: int = 3, random: bool = False) -> dict:
    """Same as find_anime but for manga via Jikan."""
    import urllib.parse as _up
    try:
        if random:
            data = _http_json("https://api.jikan.moe/v4/random/manga")
            rows = [data.get("data", {})]
        else:
            q = _up.quote(query or genre or "manga")
            url = f"https://api.jikan.moe/v4/manga?q={q}&order_by=score&sort=desc&limit={count}"
            rows = _http_json(url).get("data", [])[:count]
    except Exception as e:
        return {"skill": "find_manga", "error": str(e)}

    results = []
    for a in rows:
        if not a:
            continue
        img = (a.get("images") or {}).get("jpg") or {}
        results.append({
            "title": a.get("title") or a.get("title_english") or "?",
            "synopsis": _trim(a.get("synopsis", ""), 700),
            "chapters": a.get("chapters"),
            "volumes": a.get("volumes"),
            "type": a.get("type"),
            "status": a.get("status"),
            "score": a.get("score"),
            "genres": [g.get("name") for g in (a.get("genres") or [])],
            "authors": [x.get("name") for x in (a.get("authors") or [])],
            "image": img.get("large_image_url") or img.get("image_url"),
            "url": a.get("url"),
            "kind": "manga",
        })
    return {"skill": "find_manga", "count": len(results), "results": results}


def find_movie_tv(query: str, type: str = "any", count: int = 3) -> dict:
    """Search TVmaze for TV shows (free, no API key). Movies are limited —
    TVmaze is TV-focused but covers many mini-series and specials. Use
    query keyword + type: 'tv', 'movie', or 'any'."""
    import urllib.parse as _up
    try:
        url = f"https://api.tvmaze.com/search/shows?q={_up.quote(query)}"
        rows = _http_json(url)[:count]
    except Exception as e:
        return {"skill": "find_movie_tv", "error": str(e)}

    results = []
    for hit in rows:
        s = hit.get("show") or {}
        img = (s.get("image") or {}) or {}
        results.append({
            "title": s.get("name"),
            "synopsis": _trim((s.get("summary") or "").replace("<p>", "").replace("</p>", "\n").replace("<b>","").replace("</b>",""), 700),
            "language": s.get("language"),
            "genres": s.get("genres", []),
            "status": s.get("status"),
            "premiered": s.get("premiered"),
            "ended": s.get("ended"),
            "rating": (s.get("rating") or {}).get("average"),
            "network": (s.get("network") or {}).get("name") or (s.get("webChannel") or {}).get("name"),
            "runtime": s.get("runtime") or s.get("averageRuntime"),
            "episodes": None,
            "image": img.get("original") or img.get("medium"),
            "url": s.get("url"),
            "kind": "tv",
        })
    return {"skill": "find_movie_tv", "count": len(results), "results": results}


def find_game(query: str, count: int = 3) -> dict:
    """Search Wikipedia for a video game — returns title, extract, thumbnail.
    No API key; commercial-grade databases need keys."""
    import urllib.parse as _up
    results = []
    try:
        # Wikipedia search
        q = _up.quote(query + " video game")
        search = _http_json(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&format=json&srlimit={count}")
        hits = (search.get("query") or {}).get("search") or []
        for h in hits[:count]:
            title = h.get("title")
            tq = _up.quote(title.replace(" ", "_"))
            try:
                page = _http_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{tq}")
            except Exception:
                continue
            results.append({
                "title": page.get("title"),
                "synopsis": _trim(page.get("extract", ""), 700),
                "image": (page.get("thumbnail") or {}).get("source") or (page.get("originalimage") or {}).get("source"),
                "url": (page.get("content_urls", {}).get("desktop") or {}).get("page"),
                "kind": "game",
            })
    except Exception as e:
        return {"skill": "find_game", "error": str(e)}
    return {"skill": "find_game", "count": len(results), "results": results}


def search_youtube(query: str, count: int = 5) -> dict:
    """Scrape YouTube search results (no API key). Returns video IDs +
    titles + channel + thumbnails. Extracts videoRenderer blocks from
    the ytInitialData JSON embedded in the results page."""
    import urllib.parse as _up, urllib.request as _ur, re as _re, ssl as _ssl
    try:
        req = _ur.Request(
            f"https://www.youtube.com/results?search_query={_up.quote(query)}",
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        ctx = _ssl.create_default_context()
        try:
            import certifi
            ctx.load_verify_locations(certifi.where())
        except ImportError:
            pass
        html = _ur.urlopen(req, timeout=15, context=ctx).read().decode("utf-8", errors="replace")
    except Exception as e:
        return {"skill": "search_youtube", "error": str(e)}

    # Find every videoRenderer chunk and extract fields within each
    results: list[dict] = []
    seen = set()
    positions = [m.start() for m in _re.finditer(r'"videoRenderer":\{', html)]
    # Add end sentinel
    positions.append(len(html))
    for i in range(len(positions) - 1):
        chunk = html[positions[i]:positions[i + 1]]
        vid_m = _re.search(r'"videoId":"([\w-]{11})"', chunk)
        if not vid_m: continue
        vid = vid_m.group(1)
        if vid in seen: continue
        title_m = _re.search(r'"title":\{"runs":\[\{"text":"([^"]+)"', chunk)
        if not title_m: continue
        seen.add(vid)
        def _u(s):
            try:
                return s.encode('utf-8').decode('unicode_escape')
            except Exception:
                return s
        ch = _re.search(r'"longBylineText":\{"runs":\[\{"text":"([^"]+)"', chunk)
        dur = _re.search(r'"lengthText":\{"simpleText":"([^"]+)"', chunk)
        views = _re.search(r'"viewCountText":\{"simpleText":"([^"]+)"', chunk)
        results.append({
            "id": vid,
            "title": _u(title_m.group(1)),
            "channel": _u(ch.group(1)) if ch else "",
            "duration": dur.group(1) if dur else "",
            "views": views.group(1) if views else "",
            "url": f"https://www.youtube.com/watch?v={vid}",
            "embed": f"https://www.youtube.com/embed/{vid}",
            "thumbnail": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
        })
        if len(results) >= count: break
    return {"skill": "search_youtube", "count": len(results), "results": results, "query": query}


def search_images(query: str, count: int = 6) -> dict:
    """Return image URLs + thumbnails for a query. Uses Wikimedia Commons
    which is no-key and generally safe for work. For broader coverage the
    model can call fetch_url on an image-search page instead."""
    import urllib.parse as _up
    try:
        q = _up.quote(query)
        data = _http_json(
            "https://commons.wikimedia.org/w/api.php?action=query&format=json"
            f"&generator=search&gsrsearch={q}&gsrlimit={count}&gsrnamespace=6"
            "&prop=imageinfo&iiprop=url|size&iiurlwidth=600"
        )
        pages = (data.get("query") or {}).get("pages") or {}
        results = []
        for p in pages.values():
            info = (p.get("imageinfo") or [{}])[0]
            if not info.get("url"):
                continue
            results.append({
                "title": p.get("title"),
                "url": info.get("thumburl") or info.get("url"),
                "source": info.get("url"),
                "width": info.get("thumbwidth") or info.get("width"),
                "height": info.get("thumbheight") or info.get("height"),
            })
        return {"skill": "search_images", "count": len(results), "results": results}
    except Exception as e:
        return {"skill": "search_images", "error": str(e)}


# ============================================================
# PDF text extraction
# ============================================================
def extract_pdf_text(path: str, max_pages: int = 30) -> dict:
    try:
        import pdfplumber
    except ImportError:
        return {"skill": "extract_pdf_text", "error": "pdfplumber not installed"}

    # Path sandbox: if it's just a filename, look in ~/Downloads
    p = Path(path)
    if not p.is_absolute():
        p = Path.home() / "Downloads" / path
    if not p.exists():
        return {"skill": "extract_pdf_text", "error": f"not found: {p}"}

    pages = []
    try:
        with pdfplumber.open(str(p)) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                text = (page.extract_text() or "").strip()
                pages.append({"page": i + 1, "text": text[:6000]})
    except Exception as e:
        return {"skill": "extract_pdf_text", "error": str(e)}

    return {
        "skill": "extract_pdf_text",
        "path": str(p),
        "page_count": len(pages),
        "pages": pages,
    }
