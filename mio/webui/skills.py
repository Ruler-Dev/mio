"""Mio UI Skills — web search, PDF generation, and everyday tasks.

Skills are invoked by the model when it detects user intent. They run
as tool calls and return structured results for the UI to render.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path


# ===== Web Search Skill =====

def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web and return structured results."""
    from mio.webui.browser import search_web

    results = search_web(query, max_results=max_results)
    return {
        "skill": "web_search",
        "query": query,
        "results": [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "domain": r.domain,
            }
            for r in results
        ],
    }


def fetch_url(url: str) -> dict:
    """Fetch and extract text from a URL, with on-disk cache.

    First checks ~/.mio/web-cache/<sha1>.json. On miss, fetches via
    agent-browser / urllib and persists. Cache is user-wipeable from
    Settings → Cache.
    """
    from mio.webui.browser import fetch_page
    from mio.webui.router import web_cache_get, web_cache_put

    cached = web_cache_get(url)
    if cached is not None:
        return {
            "skill": "fetch_url",
            "url": url,
            "content": cached[:6000],
            "truncated": len(cached) > 6000,
            "cached": True,
        }

    text = fetch_page(url)
    if not text.strip():
        return {
            "skill": "fetch_url",
            "url": url,
            "content": "",
            "truncated": False,
            "error": (
                "fetch_failed: page returned no readable text "
                "(likely anti-bot wall, paywall, or JS-only content). "
                "Do not fabricate a summary — tell the user this URL is unavailable."
            ),
        }
    web_cache_put(url, text)
    return {
        "skill": "fetch_url",
        "url": url,
        "content": text[:6000],
        "truncated": len(text) > 6000,
        "cached": False,
    }


# ===== PDF Generation Skill =====

_UNICODE_FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    # Linux (DejaVu is commonly installed)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


def _find_unicode_font() -> tuple[str, str] | None:
    """Return (regular_path, bold_path_or_regular) for a Unicode TTF, or None."""
    import os
    for p in _UNICODE_FONT_CANDIDATES:
        if os.path.exists(p):
            bold = p.replace(".ttf", " Bold.ttf")
            if not os.path.exists(bold):
                # DejaVu variant
                alt = p.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
                bold = alt if os.path.exists(alt) else p
            return p, bold
    return None


def generate_pdf(
    title: str,
    content: str,
    filename: str | None = None,
    preset: str | None = None,
    color: str | None = None,
    theme: str | None = None,
    **_ignored,
) -> dict:
    """Generate a PDF document from markdown-like content.

    Registers a Unicode TTF (Arial Unicode on macOS, DejaVu on Linux) when
    available so characters like en-dash, smart quotes, and accented letters
    render correctly. Falls back to Helvetica with ASCII transliteration when
    no Unicode font is found.

    If `preset`, `color`, or `theme` is passed, the call is delegated to
    `generate_pdf_report` (which has the full 64-preset × 39-color styling
    system). That keeps the tool DWIM — the model sometimes picks
    `generate_pdf` by name but passes styling kwargs that only the rich
    variant understands.
    """
    if preset or color or theme:
        try:
            from mio.webui.skills_docs import generate_pdf_report
        except ImportError:
            pass
        else:
            return generate_pdf_report(
                title=title,
                content=content,
                filename=filename,
                preset=preset or theme or "auto",
                color=color,
            )

    try:
        from fpdf import FPDF
    except ImportError:
        return {
            "skill": "generate_pdf",
            "error": "fpdf2 not installed. Run: pip install fpdf2",
        }

    filename = filename or f"mio-{int(time.time())}.pdf"
    output_dir = Path.home() / "Downloads"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    font_paths = _find_unicode_font()
    if font_paths:
        reg, bold = font_paths
        pdf.add_font("Uni", style="", fname=reg)
        pdf.add_font("Uni", style="B", fname=bold)
        font_family = "Uni"

        def sanitize(s: str) -> str:
            return s
    else:
        font_family = "Helvetica"

        def sanitize(s: str) -> str:
            # fpdf2 with core fonts only supports latin-1; transliterate anything
            # outside that range so generation never fails.
            replacements = {
                "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
                "\u201C": '"', "\u201D": '"', "\u2026": "...", "\u2022": "*",
                "\u00A0": " ",
            }
            for k, v in replacements.items():
                s = s.replace(k, v)
            return s.encode("latin-1", "replace").decode("latin-1")

    # Title
    pdf.set_font(font_family, "B", 20)
    pdf.cell(0, 14, sanitize(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_font(font_family, "", 11)
    for line in content.split("\n"):
        stripped = sanitize(line.strip())
        if stripped.startswith("# "):
            pdf.ln(6)
            pdf.set_font(font_family, "B", 16)
            pdf.cell(0, 10, stripped[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(font_family, "", 11)
        elif stripped.startswith("## "):
            pdf.ln(4)
            pdf.set_font(font_family, "B", 13)
            pdf.cell(0, 8, stripped[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(font_family, "", 11)
        elif stripped.startswith("### "):
            pdf.ln(3)
            pdf.set_font(font_family, "B", 11)
            pdf.cell(0, 7, stripped[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(font_family, "", 11)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            pdf.cell(8)
            pdf.cell(0, 6, "\u2022 " + stripped[2:], new_x="LMARGIN", new_y="NEXT")
        elif stripped == "":
            pdf.ln(3)
        else:
            pdf.multi_cell(0, 6, stripped)

    pdf.output(str(output_path))

    return {
        "skill": "generate_pdf",
        "path": str(output_path),
        "filename": filename,
        "pages": pdf.pages_count,
    }


# ===== Chart Generation Skill =====

def generate_chart(
    chart_type: str,
    title: str,
    labels: list,
    values: list,
    filename: str | None = None,
    xlabel: str = "",
    ylabel: str = "",
) -> dict:
    """Generate a PNG chart. chart_type ∈ {bar, hbar, line, pie}."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "skill": "generate_chart",
            "error": "matplotlib not installed. Run: pip install matplotlib",
        }

    if not labels or not values or len(labels) != len(values):
        return {
            "skill": "generate_chart",
            "error": "labels and values must be non-empty and equal length",
        }

    filename = filename or f"mio-chart-{int(time.time())}.png"
    output_path = Path.home() / "Downloads" / filename
    output_path.parent.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    ct = chart_type.lower()
    if ct == "bar":
        ax.bar(labels, values, color="#3b82f6")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    elif ct == "hbar":
        ax.barh(labels, values, color="#3b82f6")
        ax.invert_yaxis()
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    elif ct == "line":
        ax.plot(labels, values, marker="o", color="#3b82f6", linewidth=2)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    elif ct == "pie":
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
    else:
        plt.close(fig)
        return {
            "skill": "generate_chart",
            "error": f"unknown chart_type '{chart_type}' (use bar, hbar, line, or pie)",
        }

    ax.set_title(title, fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "skill": "generate_chart",
        "path": str(output_path),
        "filename": filename,
        "chart_type": ct,
    }


# ===== Execute Code Skill =====

def execute_python(code: str, timeout: int = 30) -> dict:
    """Execute Python code in a subprocess and return output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        script_path = f.name

    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(Path.home() / "Downloads"),
        )
        return {
            "skill": "execute_python",
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:2000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "skill": "execute_python",
            "error": f"Execution timed out after {timeout}s",
        }
    finally:
        os.unlink(script_path)


# ===== Skill registry =====

SKILLS = {
    "web_search": {
        "function": web_search,
        "description": (
            "Search the web. Returns a list of result URLs with short snippets. "
            "Snippets alone are NOT enough to answer questions about specific "
            "facts, dates, or recent events — after searching you MUST call "
            "fetch_url on the 2-3 most relevant results to read the actual "
            "page content before answering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
    "fetch_url": {
        "function": fetch_url,
        "description": (
            "Fetch and extract text content from a URL. Use this after "
            "web_search to read the actual articles. Always fetch before "
            "citing specific facts, dates, numbers, or quotes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
            },
            "required": ["url"],
        },
    },
    "generate_pdf": {
        "function": generate_pdf,
        "description": "Generate a PDF document with Unicode support",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title"},
                "content": {"type": "string", "description": "Document content in markdown"},
                "filename": {"type": "string", "description": "Output filename"},
            },
            "required": ["title", "content"],
        },
    },
    "generate_chart": {
        "function": generate_chart,
        "description": (
            "Generate a bar / hbar / line / pie chart as a PNG image. "
            "Use this when the user asks for a chart, graph, diagram, or "
            "visualization of numeric data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "description": "One of: bar, hbar, line, pie",
                },
                "title": {"type": "string", "description": "Chart title"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Category labels",
                },
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Numeric values, same length as labels",
                },
                "xlabel": {"type": "string", "description": "X-axis label (optional)"},
                "ylabel": {"type": "string", "description": "Y-axis label (optional)"},
                "filename": {"type": "string", "description": "Output filename (optional)"},
            },
            "required": ["chart_type", "title", "labels", "values"],
        },
    },
    "execute_python": {
        "function": execute_python,
        "description": "Execute Python code and return the output",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        },
    },
}


# Attach document-generation + weather skills (mirrors Anthropic's Claude
# Skills library choices: python-docx, openpyxl, python-pptx, reportlab).
from mio.webui.skills_docs import (
    generate_docx as _g_docx,
    generate_xlsx as _g_xlsx,
    generate_pptx as _g_pptx,
    generate_pdf_report as _g_pdf_report,
    generate_letter as _g_letter,
    generate_certificate as _g_certificate,
    generate_flyer as _g_flyer,
    generate_menu as _g_menu,
    generate_brochure as _g_brochure,
    generate_newsletter as _g_newsletter,
    generate_business_card as _g_bcard,
    PDF_PRESETS as _PDF_PRESETS,
    OFFICE_PRESETS as _OFFICE_PRESETS,
    COLOR_PALETTES as _COLOR_PALETTES,
)

# Color-override parameter — works alongside `preset` to produce variations
# ("same layout, different color"). Accepts 40+ named palettes plus common
# aliases ("blue", "dark green", "hot pink", "gold", "cyan", etc).
_COLOR_PARAM = {
    "type": "string",
    "description": (
        "Color override. Overrides ONLY the color palette (accent / header "
        "/ soft background) while KEEPING the preset's layout, fonts, and "
        "decoration. Use this when the user asks for the 'same document "
        "but in emerald / red / dark blue / etc.' — don't change preset, "
        "just pass color. Accepts palette names (" +
        ", ".join(_COLOR_PALETTES[:12]) + ", etc.) and common aliases "
        "('blue', 'dark green', 'hot pink', 'gold', 'cyan')."
    ),
}

_OFFICE_PRESET_PARAM = {
    "type": "string",
    "enum": ("auto",) + _OFFICE_PRESETS,
    "description": (
        "Visual preset for the Office file (color palette + font pairing). "
        "Leave as 'auto' (default) — the skill picks by content keywords. "
        "Name one only if the user asked for a different look."
    ),
}

# Preset spec shared by every PDF skill. 'auto' is the default — the skill
# picks a fitting look based on the document kind + content keywords. The
# model should ONLY name a specific preset when the user has asked for a
# different look after seeing the first draft.
_PRESET_PARAM = {
    "type": "string",
    "enum": ("auto",) + _PDF_PRESETS,
    "description": (
        "Visual preset. Leave as 'auto' (default) — the skill picks a look "
        "that fits the document type and content. ONLY pass a named preset "
        "when the user asked to change style. There are ~60 presets grouped "
        "by mood: corporate (navy, oxford, embassy), editorial (vogue, "
        "broadsheet, manuscript), tech (carbon, blueprint, terminal, lab), "
        "dark (midnight, obsidian, eclipse, nocturne), warm/festive "
        "(sunset, ember, bubblegum, tropical, peach, neon), nature "
        "(forest, sage, mint, emerald), minimal (minimal, bone, whisper, "
        "duotone), classic (walnut, parchment, brass, cambridge, claret)."
    ),
}

# Surgical color overrides. The model should pass ONLY the ones the user
# explicitly asked for — everything else stays from the preset. These win
# over both `preset` and `color` when set.
_BG_COLOR_PARAM = {
    "type": "string",
    "description": (
        "Override the page background. Accepts a color name ('black', "
        "'white', 'light green', 'mint', 'cream', 'peach', 'lavender') or "
        "a hex ('#09090b'). Use when the user asks for a specific "
        "background, e.g. 'background light green', 'sfondo chiaro'."
    ),
}
_TEXT_COLOR_PARAM = {
    "type": "string",
    "description": (
        "Override the body text color. Same accepted formats as "
        "background_color. Use when the user asks for a specific text "
        "color, e.g. 'text black', 'testo bianco'."
    ),
}
_ACCENT_COLOR_PARAM = {
    "type": "string",
    "description": (
        "Override the accent color (headings / accent bar / links). Same "
        "accepted formats as background_color."
    ),
}

from mio.webui.weather import get_weather as _g_weather


SKILLS["generate_docx"] = {
    "function": _g_docx,
    "description": (
        "Generate a Microsoft Word document (.docx). Accepts markdown-like "
        "content (headings with #/##/###, bullets with -, blockquotes with >, "
        "fenced code blocks). Visual preset is auto-selected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Document title"},
            "content": {"type": "string", "description": "Markdown-style body"},
            "filename": {"type": "string", "description": "Output filename (optional)"},
            "author": {"type": "string", "description": "Document author (optional)"},
            "preset": _OFFICE_PRESET_PARAM,
            "color": _COLOR_PARAM,
            "accent_color": _ACCENT_COLOR_PARAM,
            "text_color": _TEXT_COLOR_PARAM,
        },
        "required": ["title", "content"],
    },
}

SKILLS["generate_xlsx"] = {
    "function": _g_xlsx,
    "description": (
        "Generate an Excel spreadsheet (.xlsx) with styled header row, "
        "zebra-striped rows, auto-sized columns, and frozen header. "
        "Visual preset is auto-selected based on title/headers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Title shown above table (optional)"},
            "headers": {
                "type": "array", "items": {"type": "string"},
                "description": "Column headers",
            },
            "rows": {
                "type": "array",
                "items": {"type": "array"},
                "description": "Data rows, each inner array is one row",
            },
            "filename": {"type": "string", "description": "Output filename (optional)"},
            "sheet_name": {"type": "string", "description": "Sheet name (optional)"},
            "preset": _OFFICE_PRESET_PARAM,
            "color": _COLOR_PARAM,
            "accent_color": _ACCENT_COLOR_PARAM,
            "text_color": _TEXT_COLOR_PARAM,
        },
        "required": ["headers", "rows"],
    },
}

SKILLS["generate_pptx"] = {
    "function": _g_pptx,
    "description": (
        "Generate a PowerPoint deck (.pptx) with a colored left accent bar "
        "and footer strip per slide. `slides` is a list of "
        "{title, content, notes?} dicts; content uses '-' prefixed bullets. "
        "Visual preset is auto-selected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Deck title (cover slide)"},
            "subtitle": {"type": "string", "description": "Cover subtitle (optional)"},
            "slides": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Array of {title, content, notes?} per slide",
            },
            "filename": {"type": "string", "description": "Output filename (optional)"},
            "preset": _OFFICE_PRESET_PARAM,
            "color": _COLOR_PARAM,
            "accent_color": _ACCENT_COLOR_PARAM,
            "text_color": _TEXT_COLOR_PARAM,
        },
        "required": ["title", "slides"],
    },
}

SKILLS["generate_pdf_report"] = {
    "function": _g_pdf_report,
    "description": (
        "Generate a professionally-styled PDF report (blue accent bar, "
        "footer, colored headings, styled tables, optional charts). "
        "EASIEST usage: pass `content` as a markdown string with #/##/### "
        "headings, paragraphs, bulleted lists (- item), and GFM pipe tables. "
        "Example content:\n"
        "  # Summary\n\n  Overview paragraph here.\n\n  ## Details\n\n"
        "  - Point one\n  - Point two\n\n"
        "  | Region | Revenue |\n  |---|---|\n  | EMEA | 1.2M |\n\n"
        "For charts or images, additionally pass `sections` with explicit "
        "block dicts (kind ∈ heading|paragraph|bullets|table|chart|image|"
        "pagebreak). Use this instead of generate_pdf for anything with "
        "charts, tables, or multiple sections."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Report title (shown big on page 1 and in footer)"},
            "content": {
                "type": "string",
                "description": (
                    "Markdown body (headings, paragraphs, bullets, pipe "
                    "tables). Easiest way to fill the report."
                ),
            },
            "sections": {
                "type": "array",
                "items": {"type": "object"},
                "description": (
                    "Optional structured blocks appended after `content`. "
                    "Use for charts/images/page-breaks."
                ),
            },
            "filename": {"type": "string", "description": "Output filename (optional)"},
            "author": {"type": "string", "description": "Author name (optional)"},
            "preset": _PRESET_PARAM,
            "color": _COLOR_PARAM,
            "background_color": _BG_COLOR_PARAM,
            "text_color": _TEXT_COLOR_PARAM,
            "accent_color": _ACCENT_COLOR_PARAM,
            "background_color": {
                "type": "string",
                "description": (
                    "Override the page background. Accepts a color name "
                    "('black', 'light green', 'mint', 'cream') or a hex "
                    "('#09090b'). Use when the user asks for a specific "
                    "background, e.g. 'make the background light green'."
                ),
            },
            "text_color": {
                "type": "string",
                "description": (
                    "Override the body text color. Same accepted formats as "
                    "background_color. Use when the user asks for a specific "
                    "text color, e.g. 'make the text black'."
                ),
            },
            "accent_color": {
                "type": "string",
                "description": (
                    "Override the accent color (headings / accent bar / "
                    "links). Same accepted formats as background_color."
                ),
            },
        },
        "required": ["title"],
    },
}

SKILLS["generate_letter"] = {
    "function": _g_letter,
    "description": (
        "Generate a formal business letter PDF: letterhead, date, recipient "
        "address block, salutation, body paragraphs, closing, signature line. "
        "Use for cover letters, complaint letters, resignation, references, "
        "any formal correspondence."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recipient_name": {"type": "string"},
            "recipient_address": {"type": "string", "description": "Multi-line address (use \\n between lines)"},
            "body": {"type": "string", "description": "Letter body; blank lines separate paragraphs"},
            "sender_name": {"type": "string"},
            "sender_address": {"type": "string", "description": "Multi-line return address (optional)"},
            "subject": {"type": "string", "description": "Re: line (optional)"},
            "salutation": {"type": "string", "description": "Defaults to 'Dear {recipient_name},'"},
            "closing": {"type": "string", "description": "e.g. 'Sincerely,' 'Best regards,'"},
            "date": {"type": "string", "description": "Defaults to today"},
            "filename": {"type": "string"},
            "preset": _PRESET_PARAM,
            "color": _COLOR_PARAM,
            "background_color": _BG_COLOR_PARAM,
            "text_color": _TEXT_COLOR_PARAM,
            "accent_color": _ACCENT_COLOR_PARAM,
        },
        "required": ["recipient_name", "body"],
    },
}

SKILLS["generate_certificate"] = {
    "function": _g_certificate,
    "description": (
        "Generate a landscape certificate PDF with ornate borders, centered "
        "recipient name, achievement text, issue date, and signature lines. "
        "Use for awards, diplomas, completion certificates, recognition."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Person being honored"},
            "achievement": {"type": "string", "description": "What they accomplished (1–2 sentences)"},
            "issuer": {"type": "string", "description": "Organization or person issuing it"},
            "date": {"type": "string", "description": "Issue date (defaults to today)"},
            "signatures": {
                "type": "array", "items": {"type": "object"},
                "description": "List of {name, role} dicts for signature lines",
            },
            "filename": {"type": "string"},
            "preset": _PRESET_PARAM,
            "color": _COLOR_PARAM,
            "background_color": _BG_COLOR_PARAM,
            "text_color": _TEXT_COLOR_PARAM,
            "accent_color": _ACCENT_COLOR_PARAM,
            "orientation": {"type": "string", "enum": ["landscape", "portrait"]},
        },
        "required": ["recipient", "achievement"],
    },
}

SKILLS["generate_flyer"] = {
    "function": _g_flyer,
    "description": (
        "Generate a single-page poster/flyer PDF with a hero color block, "
        "big title, optional image, body copy, and a call-to-action pill. "
        "Use for events, promotions, announcements, recruiting posters."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "subtitle": {"type": "string"},
            "body": {"type": "string", "description": "Main copy (kept short)"},
            "call_to_action": {"type": "string", "description": "CTA pill text, e.g. 'Register at example.com'"},
            "footer": {"type": "string", "description": "Fine-print footer"},
            "image_path": {"type": "string", "description": "Optional hero image file path"},
            "filename": {"type": "string"},
            "preset": _PRESET_PARAM,
            "color": _COLOR_PARAM,
            "background_color": _BG_COLOR_PARAM,
            "text_color": _TEXT_COLOR_PARAM,
            "accent_color": _ACCENT_COLOR_PARAM,
        },
        "required": ["title"],
    },
}

SKILLS["generate_menu"] = {
    "function": _g_menu,
    "description": (
        "Generate a restaurant menu PDF: restaurant name, tagline, two-column "
        "grid of category sections (each with items: name/description/price)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "restaurant_name": {"type": "string"},
            "tagline": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of {name, items: [{name, description, price}]}",
            },
            "footer": {"type": "string"},
            "filename": {"type": "string"},
            "preset": _PRESET_PARAM,
            "color": _COLOR_PARAM,
            "background_color": _BG_COLOR_PARAM,
            "text_color": _TEXT_COLOR_PARAM,
            "accent_color": _ACCENT_COLOR_PARAM,
        },
        "required": ["restaurant_name", "sections"],
    },
}

SKILLS["generate_brochure"] = {
    "function": _g_brochure,
    "description": (
        "Generate a landscape tri-fold brochure PDF with three panels "
        "(heading + body + bullet points per panel). Pass exactly 3 panels; "
        "the right-most panel prints as the cover."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "panels": {
                "type": "array", "items": {"type": "object"},
                "description": "Three {heading, body, bullets} dicts",
            },
            "footer": {"type": "string"},
            "filename": {"type": "string"},
            "preset": _PRESET_PARAM,
            "color": _COLOR_PARAM,
            "background_color": _BG_COLOR_PARAM,
            "text_color": _TEXT_COLOR_PARAM,
            "accent_color": _ACCENT_COLOR_PARAM,
        },
        "required": ["title", "panels"],
    },
}

SKILLS["generate_newsletter"] = {
    "function": _g_newsletter,
    "description": (
        "Generate a newsletter PDF: masthead (title + issue/date), full-width "
        "lead story, then a two-column grid of shorter articles."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Newsletter name (masthead)"},
            "issue": {"type": "string", "description": "e.g. 'Issue 42 · March 2026'"},
            "lead_headline": {"type": "string"},
            "lead_body": {"type": "string"},
            "articles": {
                "type": "array", "items": {"type": "object"},
                "description": "List of {heading, body} for two-column articles",
            },
            "footer": {"type": "string"},
            "filename": {"type": "string"},
            "preset": _PRESET_PARAM,
            "color": _COLOR_PARAM,
            "background_color": _BG_COLOR_PARAM,
            "text_color": _TEXT_COLOR_PARAM,
            "accent_color": _ACCENT_COLOR_PARAM,
        },
        "required": ["title"],
    },
}

SKILLS["generate_business_card"] = {
    "function": _g_bcard,
    "description": (
        "Generate a single business card (3.5\"×2\") centered on an A4 PDF. "
        "Good for 'make me a business card' requests."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "role": {"type": "string"},
            "company": {"type": "string"},
            "email": {"type": "string"},
            "phone": {"type": "string"},
            "website": {"type": "string"},
            "address": {"type": "string"},
            "filename": {"type": "string"},
            "preset": _PRESET_PARAM,
            "color": _COLOR_PARAM,
            "background_color": _BG_COLOR_PARAM,
            "text_color": _TEXT_COLOR_PARAM,
            "accent_color": _ACCENT_COLOR_PARAM,
        },
        "required": ["name"],
    },
}

from mio.webui.skills_misc import (
    generate_qr_code as _g_qr,
    generate_ical as _g_ical,
    generate_csv as _g_csv,
    generate_sqlite_db as _g_sqlite,
    generate_resume as _g_resume,
    generate_invoice as _g_invoice,
    generate_markdown as _g_md,
    extract_pdf_text as _g_extract_pdf,
    translate_text as _g_translate,
    find_anime as _g_anime,
    find_manga as _g_manga,
    find_movie_tv as _g_tv,
    find_game as _g_game,
    search_images as _g_imgs,
    search_youtube as _g_yt,
)

SKILLS["generate_markdown"] = {
    "function": _g_md,
    "description": (
        "Save a note as a Markdown (.md) file with optional YAML frontmatter "
        "(title/tags/created), Obsidian-compatible. Pass `vault_path` to "
        "drop it directly into an Obsidian vault; otherwise it lands in "
        "~/Downloads. Use for 'save to Obsidian', 'write a note', "
        "'export as markdown', knowledge capture, journal entries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Note title; becomes H1 and filename stem"},
            "content": {"type": "string", "description": "Markdown body"},
            "filename": {"type": "string", "description": "Override filename (without .md suffix is fine)"},
            "tags": {"type": "array", "items": {"type": "string"},
                     "description": "List of tags for the YAML frontmatter"},
            "vault_path": {"type": "string", "description": "Path to an Obsidian vault or any folder"},
            "frontmatter": {"type": "object", "description": "Extra YAML frontmatter key/values"},
        },
        "required": ["title", "content"],
    },
}

SKILLS["generate_qr_code"] = {
    "function": _g_qr,
    "description": (
        "Generate a QR code PNG. ONLY call this when the user explicitly "
        "asks for a QR code, barcode, or a scannable image. Never call it "
        "for weather, search, document generation, or any unrelated task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "string", "description": "Text or URL to encode into the QR code"},
            "size": {"type": "integer", "description": "Box size in pixels (default 10)"},
            "filename": {"type": "string", "description": "Output filename (optional)"},
        },
        "required": ["data"],
    },
}

# ---- Local folder RAG ----
from mio.webui.skills_rag import (
    index_folder as _rag_index,
    search_local_folder as _rag_search,
    list_indexes as _rag_list,
    drop_index as _rag_drop,
)

SKILLS["index_folder"] = {
    "function": _rag_index,
    "description": (
        "Index a local folder of text files (.md/.py/.txt/etc.) into the "
        "RAG database so `search_local_folder` can find content. Call once "
        "when the user says 'index this folder' / 'add my notes folder'. "
        "Pass `replace=false` to accumulate rather than overwrite."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path or ~-expanded path"},
            "label": {"type": "string", "description": "Friendly name (default: folder name)"},
            "replace": {"type": "boolean", "description": "Wipe prior content for this folder (default true)"},
        },
        "required": ["path"],
    },
}

SKILLS["search_local_folder"] = {
    "function": _rag_search,
    "description": (
        "Full-text search the local RAG index built from user-indexed folders. "
        "Call this when the user asks about personal notes, their codebase, or "
        "content you'd expect to be on disk. Returns path, title, and snippet "
        "with hits highlighted in «guillemets»."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query":       {"type": "string", "description": "FTS5 query"},
            "limit":       {"type": "integer", "description": "Max results (default 8)"},
            "index_label": {"type": "string", "description": "Restrict to one indexed folder"},
        },
        "required": ["query"],
    },
}

SKILLS["list_indexes"] = {
    "function": _rag_list,
    "description": "List all folders currently indexed in the local RAG database.",
    "parameters": {"type": "object", "properties": {}},
}

SKILLS["drop_index"] = {
    "function": _rag_drop,
    "description": "Remove an indexed folder from the RAG database by its id.",
    "parameters": {
        "type": "object",
        "properties": {"index_id": {"type": "integer"}},
        "required": ["index_id"],
    },
}

# ---- Life & work skills (batch 2) ----
from mio.webui.skills_life import (
    scale_recipe as _g_scale_recipe,
    bookmark_save as _g_bookmark_save,
    bookmark_list as _g_bookmark_list,
    bookmark_search as _g_bookmark_search,
    color_palette as _g_color_palette,
    describe_image as _g_describe_image,
    review_code as _g_review_code,
    meeting_notes as _g_meeting_notes,
    explain_regex as _g_explain_regex,
    convert_currency as _g_convert_currency,
    url_preview as _g_url_preview,
    hn_top as _g_hn_top,
    reddit_top as _g_reddit_top,
    quote as _g_quote,
    http_request as _g_http_request,
    reading_briefing as _g_reading_briefing,
    blender_status as _g_blender_status,
    blender_exec as _g_blender_exec,
    blender_snapshot as _g_blender_snapshot,
    import_shadertoy as _g_import_shadertoy,
)

SKILLS["http_request"] = {
    "function": _g_http_request,
    "description": (
        "Issue a one-shot HTTP request for API debugging. Returns status, "
        "response headers, body (truncated to 20k chars), and timing. "
        "`body` may be a string or JSON-serializable value; dicts auto-set "
        "Content-Type to application/json."
    ),
    "parameters": {"type": "object", "properties": {
        "url":     {"type": "string"},
        "method":  {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]},
        "headers": {"type": "object"},
        "body":    {"type": "string"},
        "timeout": {"type": "integer"},
    }, "required": ["url"]},
}
SKILLS["reading_briefing"] = {
    "function": _g_reading_briefing,
    "description": "Summarize the top items in the local reading list.",
    "parameters": {"type": "object", "properties": {
        "limit": {"type": "integer"},
    }},
}

# ---- Fun / random / wikipedia ----
from mio.webui.skills_fun import (
    roll_dice as _g_roll_dice,
    flip_coin as _g_flip_coin,
    pick_random as _g_pick_random,
    generate_names as _g_generate_names,
    wordle_helper as _g_wordle_helper,
    wiki_summary as _g_wiki_summary,
)

SKILLS["roll_dice"] = {
    "function": _g_roll_dice,
    "description": "Roll dice in NdM notation (e.g. '2d6', '1d20+3', '4d6-1').",
    "parameters": {"type": "object", "properties": {
        "notation": {"type": "string"},
    }, "required": ["notation"]},
}
SKILLS["flip_coin"] = {
    "function": _g_flip_coin,
    "description": "Flip one or more coins.",
    "parameters": {"type": "object", "properties": {
        "count": {"type": "integer"},
    }},
}
SKILLS["pick_random"] = {
    "function": _g_pick_random,
    "description": "Pick random item(s) from a list.",
    "parameters": {"type": "object", "properties": {
        "items": {"type": "array", "items": {"type": "string"}},
        "count": {"type": "integer"},
    }, "required": ["items"]},
}
SKILLS["generate_names"] = {
    "function": _g_generate_names,
    "description": "Generate names. `kind`: person | company | product | pet | fantasy.",
    "parameters": {"type": "object", "properties": {
        "kind":  {"type": "string"},
        "count": {"type": "integer"},
        "theme": {"type": "string", "description": "Optional keyword bias"},
    }},
}
SKILLS["wordle_helper"] = {
    "function": _g_wordle_helper,
    "description": (
        "Suggest Wordle candidates. `green` is the 5-char mask (use '-' for "
        "unknown), `yellow` is letters that ARE in the word (position "
        "unknown), `grey` is letters confirmed NOT in the word."
    ),
    "parameters": {"type": "object", "properties": {
        "green":  {"type": "string"},
        "yellow": {"type": "string"},
        "grey":   {"type": "string"},
    }},
}
SKILLS["wiki_summary"] = {
    "function": _g_wiki_summary,
    "description": "Fetch a Wikipedia summary for a topic (REST API, no key).",
    "parameters": {"type": "object", "properties": {
        "topic": {"type": "string"},
        "lang":  {"type": "string", "description": "2-letter code (default 'en')"},
    }, "required": ["topic"]},
}


SKILLS["hn_top"] = {
    "function": _g_hn_top,
    "description": "Fetch current top stories from Hacker News (Firebase API, no key needed).",
    "parameters": {"type": "object", "properties": {
        "limit": {"type": "integer", "description": "Default 10, max 50"},
    }},
}
SKILLS["reddit_top"] = {
    "function": _g_reddit_top,
    "description": "Fetch top posts from a subreddit (public JSON). `period`: hour/day/week/month/year/all.",
    "parameters": {"type": "object", "properties": {
        "subreddit": {"type": "string", "description": "e.g. 'programming' or 'r/news'; defaults to r/all"},
        "limit":     {"type": "integer"},
        "period":    {"type": "string"},
    }},
}
SKILLS["quote"] = {
    "function": _g_quote,
    "description": "Random famous quote, optionally filtered by topic or author.",
    "parameters": {"type": "object", "properties": {
        "topic":  {"type": "string", "description": "e.g. 'stoic', 'tech', 'writing'"},
        "author": {"type": "string"},
    }},
}

# ---- Productivity skills (todo / habits / journal / analyzers) ----
from mio.webui.skills_productivity import (
    todo_add as _g_todo_add,
    todo_list as _g_todo_list,
    todo_done as _g_todo_done,
    todo_delete as _g_todo_delete,
    habit_add as _g_habit_add,
    habit_checkin as _g_habit_checkin,
    habit_list as _g_habit_list,
    journal_append as _g_journal_append,
    journal_read as _g_journal_read,
    journal_search as _g_journal_search,
    analyze_json as _g_analyze_json,
    analyze_csv as _g_analyze_csv,
)

SKILLS["todo_add"] = {
    "function": _g_todo_add,
    "description": "Add an item to the persistent todo list.",
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string"},
        "list_name": {"type": "string", "description": "Optional list/category (default 'inbox')"},
        "priority": {"type": "integer", "description": "1=low, 2=normal, 3=high"},
        "due": {"type": "string", "description": "ISO date (optional)"},
    }, "required": ["text"]},
}
SKILLS["todo_list"] = {
    "function": _g_todo_list,
    "description": "List todos. Defaults to open items only.",
    "parameters": {"type": "object", "properties": {
        "include_done": {"type": "boolean"},
        "list_name":    {"type": "string"},
        "limit":        {"type": "integer"},
    }},
}
SKILLS["todo_done"] = {
    "function": _g_todo_done,
    "description": "Mark a todo done by id.",
    "parameters": {"type": "object", "properties": {
        "todo_id": {"type": "integer"},
        "done":    {"type": "boolean", "description": "Set to false to un-complete"},
    }, "required": ["todo_id"]},
}
SKILLS["todo_delete"] = {
    "function": _g_todo_delete,
    "description": "Delete a todo by id.",
    "parameters": {"type": "object", "properties": {
        "todo_id": {"type": "integer"},
    }, "required": ["todo_id"]},
}

SKILLS["habit_add"] = {
    "function": _g_habit_add,
    "description": "Create a new habit to track (e.g. 'Read 20 min').",
    "parameters": {"type": "object", "properties": {
        "name":    {"type": "string"},
        "cadence": {"type": "string", "description": "daily | weekly | etc."},
    }, "required": ["name"]},
}
SKILLS["habit_checkin"] = {
    "function": _g_habit_checkin,
    "description": "Record a habit check-in for today.",
    "parameters": {"type": "object", "properties": {
        "habit_id": {"type": "integer"},
        "name":     {"type": "string", "description": "Habit name (alternative to habit_id)"},
        "note":     {"type": "string"},
    }},
}
SKILLS["habit_list"] = {
    "function": _g_habit_list,
    "description": "List habits with streak + checkin counts.",
    "parameters": {"type": "object", "properties": {}},
}

SKILLS["journal_append"] = {
    "function": _g_journal_append,
    "description": "Append an entry to today's journal file (~/.mio/journal/<day>.md).",
    "parameters": {"type": "object", "properties": {
        "entry": {"type": "string"},
        "mood":  {"type": "string"},
        "tags":  {"type": "array", "items": {"type": "string"}},
    }, "required": ["entry"]},
}
SKILLS["journal_read"] = {
    "function": _g_journal_read,
    "description": "Read a specific day's journal entry (default: today).",
    "parameters": {"type": "object", "properties": {
        "day": {"type": "string", "description": "ISO date (YYYY-MM-DD); empty = today"},
    }},
}
SKILLS["journal_search"] = {
    "function": _g_journal_search,
    "description": "Search journal entries for a substring.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
    }, "required": ["query"]},
}

SKILLS["analyze_json"] = {
    "function": _g_analyze_json,
    "description": "Return schema, size, depth, sample values for a JSON blob.",
    "parameters": {"type": "object", "properties": {
        "json_str": {"type": "string"},
    }, "required": ["json_str"]},
}
SKILLS["analyze_csv"] = {
    "function": _g_analyze_csv,
    "description": "Per-column type inference + stats (min/max/mean/stdev) for a CSV.",
    "parameters": {"type": "object", "properties": {
        "csv_text":  {"type": "string"},
        "delimiter": {"type": "string"},
    }, "required": ["csv_text"]},
}

SKILLS["explain_regex"] = {
    "function": _g_explain_regex,
    "description": "Walk through a regex pattern token-by-token and say what each part means.",
    "parameters": {
        "type": "object",
        "properties": {"pattern": {"type": "string"}},
        "required": ["pattern"],
    },
}

SKILLS["convert_currency"] = {
    "function": _g_convert_currency,
    "description": (
        "Convert an amount between ~40 currencies (including BTC/ETH) using "
        "an offline rate table. Good for quick 'how much is 50 EUR in JPY' "
        "intents without needing network access."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "amount":        {"type": "number"},
            "from_currency": {"type": "string", "description": "3-letter ISO code"},
            "to_currency":   {"type": "string", "description": "3-letter ISO code"},
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
}

SKILLS["url_preview"] = {
    "function": _g_url_preview,
    "description": (
        "Fetch a URL and extract OpenGraph / Twitter-card metadata: title, "
        "description, image, site name. Use for generating link preview "
        "cards in responses."
    ),
    "parameters": {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
}

SKILLS["scale_recipe"] = {
    "function": _g_scale_recipe,
    "description": (
        "Scale each ingredient in a recipe by a factor. Input is a list of "
        "lines like '1 1/2 cups flour' or '200 g butter'; quantities are "
        "multiplied and returned in the same units."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ingredients": {"type": "array", "items": {"type": "string"}},
            "scale":       {"type": "number", "description": "Multiplier (e.g. 0.5, 2, 3.5)"},
        },
        "required": ["ingredients", "scale"],
    },
}

SKILLS["bookmark_save"] = {
    "function": _g_bookmark_save,
    "description": "Save a URL to the local bookmarks/reading list.",
    "parameters": {
        "type": "object",
        "properties": {
            "url":     {"type": "string"},
            "title":   {"type": "string"},
            "snippet": {"type": "string", "description": "Short description or quote"},
            "tags":    {"type": "array", "items": {"type": "string"}},
        },
        "required": ["url"],
    },
}

SKILLS["bookmark_list"] = {
    "function": _g_bookmark_list,
    "description": "List saved bookmarks (most recent first). Filter by tag or unread.",
    "parameters": {
        "type": "object",
        "properties": {
            "tag":         {"type": "string"},
            "limit":       {"type": "integer"},
            "unread_only": {"type": "boolean"},
        },
    },
}

SKILLS["bookmark_search"] = {
    "function": _g_bookmark_search,
    "description": "Full-text search across saved bookmarks.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    },
}

SKILLS["color_palette"] = {
    "function": _g_color_palette,
    "description": (
        "Generate a 5-color palette from a seed hex color. `mode`: "
        "complementary | analogous | triadic | monochromatic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "seed": {"type": "string", "description": "Seed hex color (#3b82f6)"},
            "mode": {"type": "string", "enum": ["complementary", "analogous", "triadic", "monochromatic"]},
        },
    },
}

SKILLS["describe_image"] = {
    "function": _g_describe_image,
    "description": (
        "Describe a local image file: dimensions, format, dominant colors, "
        "and OCR'd text. Call this after the user attaches an image to get "
        "non-VL structured info about it."
    ),
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

SKILLS["review_code"] = {
    "function": _g_review_code,
    "description": (
        "Run a quick heuristic lint of a code snippet and return a structured "
        "scaffold (TODOs, debug statements, long functions, long lines, bare "
        "excepts). Use before doing semantic review to anchor feedback."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code":     {"type": "string"},
            "language": {"type": "string"},
            "focus":    {"type": "string", "description": "e.g. 'security', 'performance'"},
        },
        "required": ["code"],
    },
}

SKILLS["meeting_notes"] = {
    "function": _g_meeting_notes,
    "description": (
        "Extract scaffolding from a meeting transcript — candidate attendees, "
        "decision-like lines, action-item-like lines. The model should then "
        "produce clean notes."
    ),
    "parameters": {
        "type": "object",
        "properties": {"transcript": {"type": "string"}},
        "required": ["transcript"],
    },
}


SKILLS["blender_status"] = {
    "function": _g_blender_status,
    "description": (
        "Check whether a Blender MCP addon is listening on localhost:9876. "
        "Returns { connected, scene, objects, blender_version } or a hint "
        "on how to install the addon if it's not reachable."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SKILLS["blender_exec"] = {
    "function": _g_blender_exec,
    "description": (
        "Run arbitrary Python code inside the user's RUNNING Blender via "
        "the blender-mcp addon. Use for modelling (create / modify / "
        "delete objects), materials, lighting, camera, import/export, "
        "render. SAFETY: this runs with the user's full filesystem "
        "access — only invoke on explicit user request. Use `import bpy` "
        "at the top; Blender version is 4.2+ (no context-override dicts)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to exec inside Blender"},
        },
        "required": ["code"],
    },
}

SKILLS["import_shadertoy"] = {
    "function": _g_import_shadertoy,
    "description": (
        "Import a shader from shadertoy.com by ID (e.g. 'XsXXDn') or full URL. "
        "Returns a self-contained WebGL2 HTML artifact that runs the original "
        "mainImage() function with iTime / iResolution / iMouse uniforms. Use "
        "when the user wants 'that shader I saw at shadertoy.com/view/…'."
    ),
    "parameters": {
        "type": "object",
        "properties": {"id_or_url": {"type": "string", "description": "ShaderToy ID or URL"}},
        "required": ["id_or_url"],
    },
}

SKILLS["blender_snapshot"] = {
    "function": _g_blender_snapshot,
    "description": (
        "Render the current Blender viewport as a PNG and return the "
        "image URL. Use this after `blender_exec` to show the user "
        "what was built."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


SKILLS["generate_ical"] = {
    "function": _g_ical,
    "description": "Generate an .ics calendar file with one or more events. ISO 8601 datetimes for start/end.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Calendar name"},
            "events": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of {summary, start, end, location?, description?}",
            },
            "filename": {"type": "string", "description": "Output filename (optional)"},
        },
        "required": ["title", "events"],
    },
}

SKILLS["generate_csv"] = {
    "function": _g_csv,
    "description": "Generate a CSV file with headers and rows. Use for tabular data export.",
    "parameters": {
        "type": "object",
        "properties": {
            "headers": {"type": "array", "items": {"type": "string"}, "description": "Column headers"},
            "rows": {"type": "array", "items": {"type": "array"}, "description": "Data rows"},
            "filename": {"type": "string", "description": "Output filename (optional)"},
        },
        "required": ["headers", "rows"],
    },
}

SKILLS["generate_sqlite_db"] = {
    "function": _g_sqlite,
    "description": "Generate a SQLite database file with one table. Useful for structured data the user wants to query.",
    "parameters": {
        "type": "object",
        "properties": {
            "table": {"type": "string", "description": "Table name (alphanumeric + underscore)"},
            "headers": {"type": "array", "items": {"type": "string"}, "description": "Column names"},
            "rows": {"type": "array", "items": {"type": "array"}, "description": "Data rows"},
            "filename": {"type": "string", "description": "Output filename (optional)"},
        },
        "required": ["table", "headers", "rows"],
    },
}

SKILLS["generate_resume"] = {
    "function": _g_resume,
    "description": (
        "Generate a professionally-formatted one-page PDF resume/CV. Pass a "
        "`profile` object with name, title, contact {email, phone, location, "
        "website}, summary, experience [{role, company, dates, bullets}], "
        "education [{degree, school, dates}], skills []."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {"type": "object", "description": "Profile dict"},
            "filename": {"type": "string", "description": "Output filename (optional)"},
        },
        "required": ["profile"],
    },
}

SKILLS["generate_invoice"] = {
    "function": _g_invoice,
    "description": (
        "Generate a styled PDF invoice with line items, tax, and totals. "
        "`items` is a list of {description, qty, unit_price}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "issuer": {"type": "object", "description": "{name, address, city, country, tax_id, email}"},
            "recipient": {"type": "object", "description": "{name, address, city, country, tax_id, email}"},
            "items": {"type": "array", "items": {"type": "object"}, "description": "Line items"},
            "invoice_number": {"type": "string"},
            "invoice_date": {"type": "string"},
            "due_date": {"type": "string"},
            "currency": {"type": "string", "description": "EUR / USD / GBP etc."},
            "tax_rate": {"type": "number", "description": "Percentage, e.g. 22 for 22%"},
            "notes": {"type": "string"},
            "filename": {"type": "string"},
        },
        "required": ["issuer", "recipient", "items"],
    },
}

SKILLS["find_anime"] = {
    "function": _g_anime,
    "description": (
        "Search MyAnimeList (via Jikan) for an ANIME title. ONLY call this "
        "when (a) the user explicitly asks to find / recommend / suggest "
        "anime, or (b) the user names a title they know IS an anime and "
        "wants info about it. DO NOT call for ambiguous proper-noun "
        "queries (VTubers, streamers, idols, bands, people). DO NOT call "
        "when the user asks to 'search online' / 'search the web' — use "
        "web_search instead. For 'who is X' / 'what is X' when X might be "
        "a real person, call web_search FIRST; only fall back to "
        "find_anime if the web result confirms it's an anime."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search term (optional if genre or random)"},
            "genre": {"type": "string"},
            "count": {"type": "integer", "description": "Number of results (default 3)"},
            "random": {"type": "boolean", "description": "Return a single random anime"},
        }, "required": [],
    },
}
SKILLS["find_manga"] = {
    "function": _g_manga,
    "description": (
        "Search MyAnimeList for manga. Same gating as find_anime — ONLY "
        "use when the user explicitly asks about manga or names a known "
        "manga title. For ambiguous names or 'search online' requests, "
        "use web_search instead."
    ),
    "parameters": SKILLS["find_anime"]["parameters"],
}
SKILLS["find_movie_tv"] = {
    "function": _g_tv,
    "description": (
        "Search TVmaze for a TV show or movie. ONLY use when the user "
        "explicitly asks for movie/TV recommendations or names a known "
        "show. DO NOT call for actors, directors, studios, or ambiguous "
        "proper nouns — use web_search. DO NOT call when the user says "
        "'search online'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "type": {"type": "string", "description": "'tv' | 'movie' | 'any'"},
            "count": {"type": "integer"},
        }, "required": ["query"],
    },
}
SKILLS["find_game"] = {
    "function": _g_game,
    "description": (
        "Search Wikipedia for a video game (no API key). Returns title / "
        "extract / cover image / article URL. Emit a mediacard artifact."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer"},
        }, "required": ["query"],
    },
}
SKILLS["search_youtube"] = {
    "function": _g_yt,
    "description": (
        "Search YouTube (no API key — scrapes results page). Returns "
        "video_id, title, channel, duration, thumbnail, and embed URL. "
        "Use this whenever the user asks for a video, trailer, tutorial, "
        "music video, or any 'show me a video about X' request. "
        "The UI auto-renders a YouTube grid artifact with clickable "
        "thumbnails → embedded player."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer", "description": "Number of videos (default 5)"},
        }, "required": ["query"],
    },
}
SKILLS["search_images"] = {
    "function": _g_imgs,
    "description": (
        "Return an array of image thumbnails + source URLs from Wikimedia "
        "Commons matching the query (no API key; SFW). Good for quick "
        "visual references. After calling this, emit an "
        "application/vnd.pimio.imagegrid artifact with the results so the "
        "user sees them."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer", "description": "How many images (default 6)"},
        }, "required": ["query"],
    },
}

SKILLS["translate_text"] = {
    "function": _g_translate,
    "description": (
        "Translate text to another language via MyMemory (no API key). "
        "Supply ISO 639-1 target code (en, es, fr, de, ja, zh, it, pt, ar, ru)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to translate"},
            "target": {"type": "string", "description": "Target language code"},
            "source": {"type": "string", "description": "Source language code (default 'auto')"},
        },
        "required": ["text", "target"],
    },
}

from mio.webui.skills_python import (
    image_resize as _g_img_resize,
    image_convert as _g_img_convert,
    image_info as _g_img_info,
    hash_text as _g_hash,
    encode_decode as _g_enc,
    generate_uuid as _g_uuid,
    generate_password as _g_pw,
    generate_fake_data as _g_fake,
    decode_jwt as _g_jwt,
    json_to_yaml as _g_j2y,
    yaml_to_json as _g_y2j,
    timezone_convert as _g_tz,
    date_math as _g_dm,
    unit_convert as _g_unit,
    text_stats as _g_ts,
    fetch_rss as _g_rss,
    zip_files as _g_zip,
    unzip_file as _g_unzip,
    merge_pdfs as _g_mpdf,
    split_pdf as _g_spdf,
    symbolic_math as _g_sym,
    markdown_to_html as _g_m2h,
    html_to_markdown as _g_h2m,
    detect_language as _g_lang,
    json_query as _g_jq,
    generate_slug as _g_slug,
    format_json as _g_fmtj,
    extract_links as _g_links,
)


def _simple(name, desc, params, required):
    return {
        "function": globals()[f"_g_{name}"],
        "description": desc,
        "parameters": {"type": "object", "properties": params, "required": required},
    }


SKILLS["image_resize"] = {
    "function": _g_img_resize,
    "description": "Resize an image (PNG/JPG) in ~/Downloads. Specify either width or height to preserve aspect ratio.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Absolute path or filename in ~/Downloads"},
        "width": {"type": "integer"}, "height": {"type": "integer"},
        "filename": {"type": "string"},
    }, "required": ["path"]},
}
SKILLS["image_convert"] = {
    "function": _g_img_convert,
    "description": "Convert an image to a different format (png, jpg, webp, bmp, gif, tiff).",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "to_format": {"type": "string"},
        "filename": {"type": "string"},
    }, "required": ["path", "to_format"]},
}
SKILLS["image_info"] = {
    "function": _g_img_info,
    "description": "Get image dimensions, format, byte size, and EXIF metadata.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
}
SKILLS["hash_text"] = {
    "function": _g_hash,
    "description": "Hash text with md5/sha1/sha256/sha384/sha512.",
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string"}, "algorithm": {"type": "string"},
    }, "required": ["text"]},
}
SKILLS["encode_decode"] = {
    "function": _g_enc,
    "description": "Encode or decode text. Operation: base64-encode, base64-decode, url-encode, url-decode, hex-encode, hex-decode, rot13.",
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string"}, "operation": {"type": "string"},
    }, "required": ["text", "operation"]},
}
SKILLS["generate_uuid"] = {
    "function": _g_uuid,
    "description": "Generate one or more UUIDs (v1, v4, v5).",
    "parameters": {"type": "object", "properties": {
        "count": {"type": "integer"}, "version": {"type": "integer"},
    }, "required": []},
}
SKILLS["generate_password"] = {
    "function": _g_pw,
    "description": "Generate secure random password(s). Cryptographically strong via secrets module.",
    "parameters": {"type": "object", "properties": {
        "length": {"type": "integer"}, "symbols": {"type": "boolean"}, "count": {"type": "integer"},
    }, "required": []},
}
SKILLS["generate_fake_data"] = {
    "function": _g_fake,
    "description": "Generate fake/synthetic data via Faker. Kind: profile, company, address, credit_card, text, internet.",
    "parameters": {"type": "object", "properties": {
        "kind": {"type": "string"}, "count": {"type": "integer"}, "locale": {"type": "string"},
    }, "required": []},
}
SKILLS["decode_jwt"] = {
    "function": _g_jwt,
    "description": "Decode a JWT token; optionally verify with a secret.",
    "parameters": {"type": "object", "properties": {
        "token": {"type": "string"}, "secret": {"type": "string"},
    }, "required": ["token"]},
}
SKILLS["json_to_yaml"] = {
    "function": _g_j2y,
    "description": "Convert JSON string to YAML.",
    "parameters": {"type": "object", "properties": {"json_str": {"type": "string"}}, "required": ["json_str"]},
}
SKILLS["yaml_to_json"] = {
    "function": _g_y2j,
    "description": "Convert YAML string to JSON.",
    "parameters": {"type": "object", "properties": {"yaml_str": {"type": "string"}}, "required": ["yaml_str"]},
}
SKILLS["timezone_convert"] = {
    "function": _g_tz,
    "description": "Convert an ISO datetime between timezones (IANA names: 'America/New_York', 'Europe/Rome', etc.).",
    "parameters": {"type": "object", "properties": {
        "dt_iso": {"type": "string"}, "from_tz": {"type": "string"}, "to_tz": {"type": "string"},
    }, "required": ["dt_iso", "from_tz", "to_tz"]},
}
SKILLS["date_math"] = {
    "function": _g_dm,
    "description": "Add/subtract days/hours/minutes to an ISO datetime (defaults to now). Returns resulting datetime and weekday.",
    "parameters": {"type": "object", "properties": {
        "date_iso": {"type": "string"}, "days": {"type": "integer"},
        "hours": {"type": "integer"}, "minutes": {"type": "integer"},
    }, "required": []},
}
SKILLS["unit_convert"] = {
    "function": _g_unit,
    "description": "Convert physical units (length, mass, time, temperature, volume, energy, speed, etc.) via pint.",
    "parameters": {"type": "object", "properties": {
        "value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"},
    }, "required": ["value", "from_unit", "to_unit"]},
}
SKILLS["text_stats"] = {
    "function": _g_ts,
    "description": "Compute word/char counts, reading time, and Flesch readability for a text block.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
}
SKILLS["fetch_rss"] = {
    "function": _g_rss,
    "description": "Fetch an RSS or Atom feed, return parsed items (title, link, summary, author, published).",
    "parameters": {"type": "object", "properties": {
        "url": {"type": "string"}, "max_items": {"type": "integer"},
    }, "required": ["url"]},
}
SKILLS["zip_files"] = {
    "function": _g_zip,
    "description": "Create a ZIP archive from a list of files in ~/Downloads.",
    "parameters": {"type": "object", "properties": {
        "paths": {"type": "array", "items": {"type": "string"}},
        "filename": {"type": "string"},
    }, "required": ["paths"]},
}
SKILLS["unzip_file"] = {
    "function": _g_unzip,
    "description": "Extract a ZIP archive to a destination folder (default: ~/Downloads/<name>/).",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "dest_dir": {"type": "string"},
    }, "required": ["path"]},
}
SKILLS["merge_pdfs"] = {
    "function": _g_mpdf,
    "description": "Merge multiple PDFs from ~/Downloads into one file.",
    "parameters": {"type": "object", "properties": {
        "paths": {"type": "array", "items": {"type": "string"}},
        "filename": {"type": "string"},
    }, "required": ["paths"]},
}
SKILLS["split_pdf"] = {
    "function": _g_spdf,
    "description": "Split a PDF into pages. pages='all' → one PDF per page. pages='1-3,5,7-9' → extract those pages into a single PDF.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "pages": {"type": "string"},
    }, "required": ["path"]},
}
SKILLS["markdown_to_html"] = {
    "function": _g_m2h,
    "description": "Convert markdown to HTML (python-markdown with fenced code + tables + toc + nl2br).",
    "parameters": {"type": "object", "properties": {"markdown_text": {"type": "string"}}, "required": ["markdown_text"]},
}
SKILLS["html_to_markdown"] = {
    "function": _g_h2m,
    "description": "Convert HTML to markdown via markdownify. Useful for saving webpage content in a readable form.",
    "parameters": {"type": "object", "properties": {"html": {"type": "string"}}, "required": ["html"]},
}
SKILLS["detect_language"] = {
    "function": _g_lang,
    "description": "Detect natural-language of input text (langdetect). Returns top candidates with probabilities.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
}
SKILLS["json_query"] = {
    "function": _g_jq,
    "description": "Run a JSONPath query (e.g. '$.users[?(@.age>18)].name') against a JSON string.",
    "parameters": {"type": "object", "properties": {
        "json_str": {"type": "string"}, "path": {"type": "string"},
    }, "required": ["json_str", "path"]},
}
SKILLS["generate_slug"] = {
    "function": _g_slug,
    "description": "Slugify a string (lowercase, hyphens, strip punctuation) for filenames and URLs.",
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string"}, "max_length": {"type": "integer"},
    }, "required": ["text"]},
}
SKILLS["format_json"] = {
    "function": _g_fmtj,
    "description": "Pretty-print JSON with configurable indent and key sorting.",
    "parameters": {"type": "object", "properties": {
        "json_str": {"type": "string"}, "indent": {"type": "integer"}, "sort_keys": {"type": "boolean"},
    }, "required": ["json_str"]},
}
SKILLS["extract_links"] = {
    "function": _g_links,
    "description": "Extract all markdown [text](url) links and plain http(s) URLs from a block of text or HTML.",
    "parameters": {"type": "object", "properties": {"text_or_html": {"type": "string"}}, "required": ["text_or_html"]},
}

SKILLS["symbolic_math"] = {
    "function": _g_sym,
    "description": "Symbolic math via sympy. operation: simplify|expand|factor|solve|diff|integrate|latex. Example: simplify '(x+1)**2'.",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string"},
        "operation": {"type": "string"},
        "variable": {"type": "string"},
    }, "required": ["expression"]},
}


SKILLS["extract_pdf_text"] = {
    "function": _g_extract_pdf,
    "description": (
        "Extract plain text from a PDF in ~/Downloads. Use when the user "
        "asks about the contents of a previously generated or uploaded PDF."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path or filename in ~/Downloads"},
            "max_pages": {"type": "integer", "description": "Max pages to read (default 30)"},
        },
        "required": ["path"],
    },
}

SKILLS["get_weather"] = {
    "function": _g_weather,
    "description": (
        "Get current weather, 24-hour hourly forecast, and 7-day daily "
        "forecast for a location. Uses Open-Meteo (no API key). After "
        "calling this, ALWAYS emit an HTML artifact rendering the widget: "
        "in your reply write <antArtifact identifier=\"weather-<slug>\" "
        "type=\"application/vnd.pimio.weather\" title=\"Weather in <city>\">"
        "<JSON result verbatim></antArtifact>. The webui will render it as "
        "an animated widget with Meteocons icons."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City/location name, e.g. 'Imola' or 'Tokyo, Japan'",
            },
            "units": {
                "type": "string",
                "description": "'metric' (°C, km/h) or 'imperial' (°F, mph)",
            },
        },
        "required": ["location"],
    },
}


def get_tools_spec() -> list[dict]:
    """Return OpenAI-format tools spec for all skills."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": skill["description"],
                "parameters": skill["parameters"],
            },
        }
        for name, skill in SKILLS.items()
    ]


def execute_skill(name: str, arguments: dict) -> dict:
    """Execute a skill by name with given arguments.

    Silently drops any kwarg the target function doesn't accept — LLMs
    occasionally hallucinate extra parameters (e.g. `preset` on a skill
    that doesn't take one) and we'd rather produce a working result than
    crash the tool-use loop with a TypeError. Skills can opt out by
    declaring `**kwargs` in their signature to receive everything.
    """
    if name not in SKILLS:
        return {"error": f"Unknown skill: {name}"}
    fn = SKILLS[name]["function"]
    try:
        import inspect
        sig = inspect.signature(fn)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not accepts_var_kw:
            arguments = {k: v for k, v in arguments.items() if k in params}
    except (TypeError, ValueError):
        pass
    return fn(**arguments)


# ============================================================
# User-defined skills (filesystem convention)
# ============================================================
# Scan ~/.mio/skills/*/SKILL.md at import time. Each SKILL.md begins
# with YAML front-matter (name, description, parameters, script) and the
# named `script` (same dir) is invoked with JSON args on stdin and must
# write a JSON result to stdout.
def _load_user_skills() -> None:
    import re
    import subprocess
    from pathlib import Path as _P
    base = _P.home() / ".mio" / "skills"
    if not base.exists():
        return
    for skill_dir in base.iterdir():
        if not skill_dir.is_dir():
            continue
        md = skill_dir / "SKILL.md"
        if not md.exists():
            continue
        try:
            raw = md.read_text()
            m = re.match(r"^---\s*\n(.*?)\n---", raw, re.DOTALL)
            if not m:
                continue
            # Lightweight YAML parse (no dep): key: value or key: {json}
            meta = {}
            for line in m.group(1).splitlines():
                kv = line.split(":", 1)
                if len(kv) != 2:
                    continue
                k, v = kv[0].strip(), kv[1].strip()
                if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
                    try: v = json.loads(v)
                    except Exception: pass
                meta[k] = v
            name = meta.get("name")
            if not name:
                continue
            script = skill_dir / (meta.get("script") or "run.py")
            if not script.exists():
                continue

            def _make_fn(script_path):
                def _invoke(**kwargs):
                    try:
                        res = subprocess.run(
                            ["python3", str(script_path)],
                            input=json.dumps(kwargs), capture_output=True,
                            text=True, timeout=60, cwd=str(script_path.parent),
                        )
                        try:
                            return json.loads(res.stdout or "{}")
                        except Exception:
                            return {"stdout": res.stdout, "stderr": res.stderr, "returncode": res.returncode}
                    except Exception as e:
                        return {"error": str(e)}
                return _invoke

            params = meta.get("parameters") if isinstance(meta.get("parameters"), dict) else {
                "type": "object", "properties": {}, "required": []
            }
            SKILLS[name] = {
                "function": _make_fn(script),
                "description": (meta.get("description") or name) + " (user skill)",
                "parameters": params,
            }
        except Exception:
            continue


import json  # noqa: E402
_load_user_skills()
