"""Python utility skills modeled after Claude's skill set.

These are small, focused tools the model can invoke to do common
programming / data / file operations. Each returns a dict — file-based
outputs land in ~/Downloads.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import os
import re
import secrets
import stat
import string
import time
import unicodedata
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

from mio.webui.safe_files import (
    UnsafePathError,
    downloads_input_path,
    downloads_output_path,
    ensure_directory_chain,
    open_binary_no_follow,
    open_confined_binary_writer,
    relative_path_parts,
)


_MAX_ZIP_MEMBERS = 1024
_MAX_ZIP_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_ZIP_TOTAL_BYTES = 256 * 1024 * 1024
_MAX_ZIP_COMPRESSION_RATIO = 200.0
_ZIP_COPY_CHUNK_BYTES = 1024 * 1024


def _out(filename: str | None, ext: str) -> Path:
    fn = filename or f"mio-{int(time.time())}{ext}"
    return downloads_output_path(fn, ext)


# ============================================================
# Image processing (Pillow)
# ============================================================
def image_resize(path: str, width: int | None = None, height: int | None = None, filename: str | None = None) -> dict:
    try:
        from PIL import Image
    except ImportError:
        return {"skill": "image_resize", "error": "Pillow not installed"}
    try:
        src = downloads_input_path(path)
        with open_binary_no_follow(src) as source:
            with Image.open(source) as img:
                w0, h0 = img.size
                if width and not height:
                    height = round(h0 * width / w0)
                elif height and not width:
                    width = round(w0 * height / h0)
                elif not width and not height:
                    width, height = w0 // 2, h0 // 2
                img2 = img.resize((width, height), Image.LANCZOS)
                out = _out(filename, src.suffix or ".png")
                img2.save(out)
    except (OSError, UnsafePathError, ValueError) as exc:
        return {"skill": "image_resize", "error": str(exc)}
    return {"skill": "image_resize", "path": str(out), "filename": out.name,
            "from": [w0, h0], "to": [width, height]}


def image_convert(path: str, to_format: str, filename: str | None = None) -> dict:
    try:
        from PIL import Image
    except ImportError:
        return {"skill": "image_convert", "error": "Pillow not installed"}
    fmt = to_format.upper().lstrip(".")
    fmt = {"JPG": "JPEG"}.get(fmt, fmt)
    ext = "." + to_format.lower().lstrip(".")
    try:
        src = downloads_input_path(path)
        with open_binary_no_follow(src) as source:
            with Image.open(source) as img:
                if fmt == "JPEG" and img.mode != "RGB":
                    img = img.convert("RGB")
                out = _out(filename, ext)
                img.save(out, fmt)
    except (OSError, UnsafePathError, ValueError) as exc:
        return {"skill": "image_convert", "error": str(exc)}
    return {"skill": "image_convert", "path": str(out), "filename": out.name,
            "format": fmt}


def image_info(path: str) -> dict:
    try:
        from PIL import Image, ExifTags
    except ImportError:
        return {"skill": "image_info", "error": "Pillow not installed"}
    try:
        src = downloads_input_path(path)
        with open_binary_no_follow(src) as source:
            source_size = os.fstat(source.fileno()).st_size
            with Image.open(source) as img:
                image_size = img.size
                image_mode = img.mode
                image_format = img.format
                exif = {}
                try:
                    raw = img._getexif() or {}
                    for tag, val in raw.items():
                        name = ExifTags.TAGS.get(tag, str(tag))
                        if isinstance(val, (bytes, bytearray)):
                            val = val.hex()[:60]
                        if isinstance(val, (str, int, float)):
                            exif[name] = val
                except Exception:
                    pass
    except (OSError, UnsafePathError, ValueError) as exc:
        return {"skill": "image_info", "error": str(exc)}
    return {
        "skill": "image_info",
        "path": str(src), "size": image_size, "mode": image_mode,
        "format": image_format, "bytes": source_size,
        "exif": exif,
    }


# ============================================================
# Hash / encode / decode
# ============================================================
def hash_text(text: str, algorithm: str = "sha256") -> dict:
    algorithm = algorithm.lower()
    if algorithm not in ("md5", "sha1", "sha256", "sha512", "sha384", "sha224"):
        return {"skill": "hash_text", "error": f"unsupported algorithm: {algorithm}"}
    h = hashlib.new(algorithm, text.encode("utf-8")).hexdigest()
    return {"skill": "hash_text", "algorithm": algorithm, "hash": h, "length": len(h)}


def encode_decode(text: str, operation: str) -> dict:
    """operation ∈ base64-encode | base64-decode | url-encode | url-decode |
    hex-encode | hex-decode | rot13"""
    op = operation.lower().replace("_", "-")
    try:
        if op == "base64-encode":
            return {"skill": "encode_decode", "result": base64.b64encode(text.encode()).decode()}
        if op == "base64-decode":
            return {"skill": "encode_decode", "result": base64.b64decode(text).decode("utf-8", errors="replace")}
        if op == "url-encode":
            return {"skill": "encode_decode", "result": urllib.parse.quote(text, safe="")}
        if op == "url-decode":
            return {"skill": "encode_decode", "result": urllib.parse.unquote(text)}
        if op == "hex-encode":
            return {"skill": "encode_decode", "result": text.encode().hex()}
        if op == "hex-decode":
            return {"skill": "encode_decode", "result": bytes.fromhex(text).decode("utf-8", errors="replace")}
        if op == "rot13":
            return {"skill": "encode_decode", "result": text.translate(str.maketrans(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                "NOPQRSTUVWXYZABCDEFGHIJKLMabcdefghijklmnopqrstuvwxyz".lower() + "NOPQRSTUVWXYZABCDEFGHIJKLM"))}
    except Exception as e:
        return {"skill": "encode_decode", "error": str(e)}
    return {"skill": "encode_decode", "error": f"unknown operation: {operation}"}


# ============================================================
# UUID / password / fake data
# ============================================================
def generate_uuid(count: int = 1, version: int = 4) -> dict:
    count = max(1, min(100, int(count)))
    ids: list[str] = []
    for _ in range(count):
        if version == 1:
            ids.append(str(uuid.uuid1()))
        elif version == 5:
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, str(time.time_ns()))))
        else:
            ids.append(str(uuid.uuid4()))
    return {"skill": "generate_uuid", "uuids": ids, "count": len(ids)}


def generate_password(length: int = 20, symbols: bool = True, count: int = 1) -> dict:
    length = max(4, min(128, int(length)))
    count = max(1, min(50, int(count)))
    alpha = string.ascii_letters + string.digits
    if symbols:
        alpha += "!@#$%^&*()-_=+[]{}"
    out = [''.join(secrets.choice(alpha) for _ in range(length)) for _ in range(count)]
    return {"skill": "generate_password", "passwords": out, "length": length}


def generate_fake_data(kind: str = "profile", count: int = 1, locale: str = "en_US") -> dict:
    try:
        from faker import Faker
    except ImportError:
        return {"skill": "generate_fake_data", "error": "faker not installed"}
    f = Faker(locale)
    count = max(1, min(200, int(count)))
    out = []
    kind = kind.lower()
    for _ in range(count):
        if kind == "profile":
            out.append({
                "name": f.name(), "email": f.email(), "phone": f.phone_number(),
                "address": f.address().replace("\n", ", "), "company": f.company(),
                "job": f.job(), "dob": str(f.date_of_birth()),
            })
        elif kind == "company":
            out.append({"name": f.company(), "catch_phrase": f.catch_phrase(), "bs": f.bs(),
                        "url": f.url(), "email": f.company_email()})
        elif kind == "address":
            out.append({"street": f.street_address(), "city": f.city(), "state": f.state(),
                        "zip": f.zipcode(), "country": f.country()})
        elif kind == "credit_card":
            out.append({"number": f.credit_card_number(), "provider": f.credit_card_provider(),
                        "expire": f.credit_card_expire(), "security": f.credit_card_security_code()})
        elif kind == "text":
            out.append({"paragraph": f.paragraph(), "sentence": f.sentence(), "word": f.word()})
        elif kind == "internet":
            out.append({"ipv4": f.ipv4(), "ipv6": f.ipv6(), "user_agent": f.user_agent(),
                        "url": f.url(), "email": f.email()})
        else:
            return {"skill": "generate_fake_data", "error": f"unknown kind: {kind}"}
    return {"skill": "generate_fake_data", "kind": kind, "count": len(out), "data": out}


# ============================================================
# JWT / JSON / YAML
# ============================================================
def decode_jwt(token: str, secret: str = "") -> dict:
    try:
        import jwt
    except ImportError:
        return {"skill": "decode_jwt", "error": "PyJWT not installed"}
    # Decode without verify for inspection; if secret given, also verify.
    try:
        header = jwt.get_unverified_header(token)
        payload = jwt.decode(token, options={"verify_signature": False})
    except Exception as e:
        return {"skill": "decode_jwt", "error": str(e)}
    verified = None
    if secret:
        try:
            jwt.decode(token, secret, algorithms=[header.get("alg", "HS256")])
            verified = True
        except Exception as e:
            verified = f"failed: {e}"
    return {"skill": "decode_jwt", "header": header, "payload": payload, "verified": verified}


def json_to_yaml(json_str: str) -> dict:
    try:
        import yaml
    except ImportError:
        return {"skill": "json_to_yaml", "error": "PyYAML not installed"}
    try:
        data = json.loads(json_str)
        return {"skill": "json_to_yaml", "yaml": yaml.safe_dump(data, sort_keys=False, allow_unicode=True)}
    except Exception as e:
        return {"skill": "json_to_yaml", "error": str(e)}


def yaml_to_json(yaml_str: str) -> dict:
    try:
        import yaml
    except ImportError:
        return {"skill": "yaml_to_json", "error": "PyYAML not installed"}
    try:
        data = yaml.safe_load(yaml_str)
        return {"skill": "yaml_to_json", "json": json.dumps(data, indent=2, ensure_ascii=False)}
    except Exception as e:
        return {"skill": "yaml_to_json", "error": str(e)}


# ============================================================
# Date / time / timezone
# ============================================================
def timezone_convert(dt_iso: str, from_tz: str, to_tz: str) -> dict:
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        return {"skill": "timezone_convert", "error": "zoneinfo not available"}
    try:
        dt = _dt.datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(from_tz))
        target = dt.astimezone(ZoneInfo(to_tz))
        return {
            "skill": "timezone_convert",
            "input": {"datetime": dt_iso, "tz": from_tz},
            "output": {"datetime": target.isoformat(), "tz": to_tz},
            "utc_offset": str(target.utcoffset()),
        }
    except Exception as e:
        return {"skill": "timezone_convert", "error": str(e)}


def date_math(date_iso: str, days: int = 0, hours: int = 0, minutes: int = 0) -> dict:
    try:
        dt = _dt.datetime.fromisoformat(date_iso.replace("Z", "+00:00")) if date_iso else _dt.datetime.now()
        result = dt + _dt.timedelta(days=days, hours=hours, minutes=minutes)
        return {"skill": "date_math", "input": dt.isoformat(), "result": result.isoformat(),
                "weekday": result.strftime("%A"), "offset_days": days}
    except Exception as e:
        return {"skill": "date_math", "error": str(e)}


# ============================================================
# Unit conversion (pint)
# ============================================================
def unit_convert(value: float, from_unit: str, to_unit: str) -> dict:
    try:
        from pint import UnitRegistry
    except ImportError:
        return {"skill": "unit_convert", "error": "pint not installed"}
    try:
        ureg = UnitRegistry()
        q = (float(value) * ureg(from_unit)).to(to_unit)
        return {"skill": "unit_convert", "value": float(value), "from": from_unit,
                "to": to_unit, "result": q.magnitude, "result_str": f"{q:~P}"}
    except Exception as e:
        return {"skill": "unit_convert", "error": str(e)}


# ============================================================
# Text analysis
# ============================================================
def text_stats(text: str) -> dict:
    try:
        import textstat
    except ImportError:
        textstat = None
    words = re.findall(r"\w+", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    out = {
        "skill": "text_stats",
        "chars": len(text),
        "chars_no_spaces": len(text.replace(" ", "")),
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
        "avg_word_length": round(sum(len(w) for w in words) / max(1, len(words)), 2),
        "reading_time_min": round(len(words) / 200, 1),
    }
    if textstat:
        try:
            out["flesch_reading_ease"] = round(textstat.flesch_reading_ease(text), 1)
            out["flesch_kincaid_grade"] = round(textstat.flesch_kincaid_grade(text), 1)
            out["syllables"] = textstat.syllable_count(text)
        except Exception:
            pass
    return out


# ============================================================
# RSS / Atom fetch
# ============================================================
def fetch_rss(url: str, max_items: int = 20) -> dict:
    try:
        import feedparser
    except ImportError:
        return {"skill": "fetch_rss", "error": "feedparser not installed"}
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        items.append({
            "title": e.get("title", ""),
            "link": e.get("link", ""),
            "published": e.get("published", "") or e.get("updated", ""),
            "summary": (e.get("summary", "") or "")[:500],
            "author": e.get("author", ""),
        })
    return {
        "skill": "fetch_rss",
        "feed_title": feed.feed.get("title", ""),
        "feed_link": feed.feed.get("link", ""),
        "item_count": len(items),
        "items": items,
    }


# ============================================================
# ZIP / unzip
# ============================================================
def zip_files(paths: list, filename: str | None = None) -> dict:
    try:
        sources: list[Path] = []
        archive_names: set[str] = set()
        for raw_path in paths or []:
            src = downloads_input_path(raw_path)
            archive_name = unicodedata.normalize("NFC", src.name).casefold()
            if archive_name in archive_names:
                raise UnsafePathError("ZIP inputs contain duplicate filenames")
            archive_names.add(archive_name)
            sources.append(src)

        out = _out(filename, ".zip")
        files_added = []
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            for src in sources:
                with open_binary_no_follow(src) as source:
                    with archive.open(src.name, "w") as destination:
                        while chunk := source.read(_ZIP_COPY_CHUNK_BYTES):
                            destination.write(chunk)
                files_added.append(src.name)
    except (OSError, RuntimeError, UnsafePathError, zipfile.BadZipFile) as exc:
        return {"skill": "zip_files", "error": str(exc)}
    return {"skill": "zip_files", "path": str(out), "filename": out.name, "files": files_added}


def unzip_file(path: str, dest_dir: str | None = None) -> dict:
    try:
        downloads = ensure_directory_chain(Path.home(), "Downloads", create=True)

        src = downloads_input_path(path)

        raw_destination = Path(dest_dir).expanduser() if dest_dir else Path(src.stem)
        if raw_destination.is_absolute():
            try:
                destination_relative = raw_destination.relative_to(downloads).as_posix()
            except ValueError as exc:
                raise UnsafePathError("ZIP destination must stay inside Downloads") from exc
        else:
            destination_relative = raw_destination.as_posix()
        destination_parts = relative_path_parts(destination_relative)

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(src, flags)
        with os.fdopen(descriptor, "rb") as archive_handle:
            with zipfile.ZipFile(archive_handle, "r") as archive:
                members = archive.infolist()
                if len(members) > _MAX_ZIP_MEMBERS:
                    raise UnsafePathError(
                        f"ZIP contains more than {_MAX_ZIP_MEMBERS} members"
                    )

                validated: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
                seen: set[str] = set()
                file_paths: set[tuple[str, ...]] = set()
                total_declared = 0
                for member in members:
                    original_name = getattr(member, "orig_filename", member.filename)
                    if original_name != member.filename:
                        raise UnsafePathError("ZIP member contains a NUL byte")
                    parts = relative_path_parts(member.filename.rstrip("/"))
                    collision_key = unicodedata.normalize(
                        "NFC", "/".join(parts)
                    ).casefold()
                    if collision_key in seen:
                        raise UnsafePathError("ZIP contains duplicate output paths")
                    seen.add(collision_key)

                    unix_mode = (member.external_attr >> 16) & 0xFFFF
                    file_type = stat.S_IFMT(unix_mode)
                    if member.is_dir():
                        if file_type not in {0, stat.S_IFDIR}:
                            raise UnsafePathError("ZIP directory has an unsafe file type")
                    else:
                        if file_type not in {0, stat.S_IFREG}:
                            raise UnsafePathError(
                                "ZIP symlinks and special files are not allowed"
                            )
                        if member.flag_bits & 0x1:
                            raise UnsafePathError("encrypted ZIP members are not supported")
                        if member.file_size < 0 or member.compress_size < 0:
                            raise UnsafePathError("ZIP member has an invalid size")
                        if member.file_size > _MAX_ZIP_MEMBER_BYTES:
                            raise UnsafePathError(
                                f"ZIP member exceeds {_MAX_ZIP_MEMBER_BYTES} bytes"
                            )
                        ratio = member.file_size / max(member.compress_size, 1)
                        if ratio > _MAX_ZIP_COMPRESSION_RATIO:
                            raise UnsafePathError(
                                "ZIP member exceeds the compression-ratio limit"
                            )
                        total_declared += member.file_size
                        if total_declared > _MAX_ZIP_TOTAL_BYTES:
                            raise UnsafePathError(
                                f"ZIP expands beyond {_MAX_ZIP_TOTAL_BYTES} bytes"
                            )
                        file_paths.add(tuple(unicodedata.normalize("NFC", part).casefold() for part in parts))
                    validated.append((member, parts))

                # Refuse a file named ``a`` alongside ``a/child``. On extract,
                # that collision otherwise becomes filesystem-dependent.
                for _member, parts in validated:
                    folded = tuple(
                        unicodedata.normalize("NFC", part).casefold()
                        for part in parts
                    )
                    if any(folded[:index] in file_paths for index in range(1, len(folded))):
                        raise UnsafePathError("ZIP file/directory paths collide")

                dest = ensure_directory_chain(
                    downloads,
                    "/".join(destination_parts),
                    create=True,
                )
                extracted: list[str] = []
                actual_total = 0
                for member, parts in validated:
                    relative = "/".join(parts)
                    if member.is_dir():
                        ensure_directory_chain(dest, relative, create=True)
                        continue
                    member_total = 0
                    with archive.open(member, "r") as source:
                        with open_confined_binary_writer(
                            dest,
                            relative,
                            create_parents=True,
                        ) as (_output, output):
                            while True:
                                chunk = source.read(_ZIP_COPY_CHUNK_BYTES)
                                if not chunk:
                                    break
                                member_total += len(chunk)
                                actual_total += len(chunk)
                                if member_total > _MAX_ZIP_MEMBER_BYTES:
                                    raise UnsafePathError(
                                        "ZIP member exceeded its extraction budget"
                                    )
                                if actual_total > _MAX_ZIP_TOTAL_BYTES:
                                    raise UnsafePathError(
                                        "ZIP exceeded its total extraction budget"
                                    )
                                output.write(chunk)
                    if member_total != member.file_size:
                        raise UnsafePathError("ZIP member size changed during extraction")
                    extracted.append(relative)
    except (OSError, RuntimeError, UnsafePathError, zipfile.BadZipFile) as exc:
        return {"skill": "unzip_file", "error": str(exc)}

    return {
        "skill": "unzip_file",
        "dest": str(dest),
        "file_count": len(extracted),
        "files": extracted[:30],
    }


# ============================================================
# PDF merge / split
# ============================================================
def merge_pdfs(paths: list, filename: str | None = None) -> dict:
    try:
        from pypdf import PdfWriter
    except ImportError:
        return {"skill": "merge_pdfs", "error": "pypdf not installed"}
    try:
        sources = [downloads_input_path(path) for path in paths or []]
        w = PdfWriter()
        merged = []
        for src in sources:
            with open_binary_no_follow(src) as source:
                w.append(source)
            merged.append(src.name)
        out = _out(filename, ".pdf")
        with open(out, "wb") as f:
            w.write(f)
    except (OSError, UnsafePathError, ValueError) as exc:
        return {"skill": "merge_pdfs", "error": str(exc)}
    return {"skill": "merge_pdfs", "path": str(out), "filename": out.name, "merged": merged}


def split_pdf(path: str, pages: str = "all") -> dict:
    """pages: 'all' splits each page; '1-3,5,7-9' extracts those ranges into one PDF."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return {"skill": "split_pdf", "error": "pypdf not installed"}
    try:
        src = downloads_input_path(path)
        outputs = []
        with open_binary_no_follow(src) as source:
            r = PdfReader(source)
            if pages == "all":
                for i, page in enumerate(r.pages):
                    w = PdfWriter()
                    w.add_page(page)
                    out = _out(f"{src.stem}-page{i+1}.pdf", ".pdf")
                    with open(out, "wb") as f:
                        w.write(f)
                    outputs.append(str(out))
            else:
                idxs = []
                for part in pages.split(","):
                    part = part.strip()
                    if "-" in part:
                        a, b = part.split("-")
                        idxs.extend(range(int(a) - 1, int(b)))
                    elif part.isdigit():
                        idxs.append(int(part) - 1)
                w = PdfWriter()
                for i in idxs:
                    if 0 <= i < len(r.pages):
                        w.add_page(r.pages[i])
                out = _out(f"{src.stem}-extract.pdf", ".pdf")
                with open(out, "wb") as f:
                    w.write(f)
                outputs.append(str(out))
    except (OSError, UnsafePathError, ValueError) as exc:
        return {"skill": "split_pdf", "error": str(exc)}
    return {"skill": "split_pdf", "count": len(outputs), "files": outputs}


# ============================================================
# Symbolic math (sympy)
# ============================================================
def markdown_to_html(markdown_text: str) -> dict:
    try:
        import markdown as _md
    except ImportError:
        return {"skill": "markdown_to_html", "error": "markdown not installed"}
    html = _md.markdown(markdown_text or "", extensions=["fenced_code", "tables", "toc", "nl2br"])
    return {"skill": "markdown_to_html", "html": html}


def html_to_markdown(html: str) -> dict:
    try:
        from markdownify import markdownify as md
    except ImportError:
        return {"skill": "html_to_markdown", "error": "markdownify not installed"}
    return {"skill": "html_to_markdown", "markdown": md(html or "")}


def detect_language(text: str) -> dict:
    try:
        from langdetect import detect_langs
    except ImportError:
        return {"skill": "detect_language", "error": "langdetect not installed"}
    try:
        results = detect_langs(text)
        return {"skill": "detect_language",
                "detected": [{"lang": str(r.lang), "prob": round(r.prob, 3)} for r in results]}
    except Exception as e:
        return {"skill": "detect_language", "error": str(e)}


def json_query(json_str: str, path: str) -> dict:
    try:
        from jsonpath_ng.ext import parse
    except ImportError:
        return {"skill": "json_query", "error": "jsonpath-ng not installed"}
    try:
        data = json.loads(json_str)
        expr = parse(path)
        matches = [m.value for m in expr.find(data)]
        return {"skill": "json_query", "path": path, "count": len(matches), "matches": matches}
    except Exception as e:
        return {"skill": "json_query", "error": str(e)}


def generate_slug(text: str, max_length: int = 60) -> dict:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return {"skill": "generate_slug", "slug": s[:max_length]}


def format_json(json_str: str, indent: int = 2, sort_keys: bool = False) -> dict:
    try:
        data = json.loads(json_str)
        return {"skill": "format_json", "formatted": json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False)}
    except Exception as e:
        return {"skill": "format_json", "error": str(e)}


def extract_links(text_or_html: str) -> dict:
    text = text_or_html or ""
    # Markdown links [text](url) + plain http(s) urls
    md_links = re.findall(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", text)
    plain = re.findall(r"https?://[^\s<>\"'`]+", text)
    return {"skill": "extract_links",
            "markdown_links": [{"text": t, "url": u} for t, u in md_links],
            "urls": list(dict.fromkeys(plain)),  # dedupe preserving order
            "count": len(set(plain))}


def symbolic_math(expression: str, operation: str = "simplify", variable: str = "x") -> dict:
    """operation ∈ simplify | expand | factor | solve | diff | integrate | latex"""
    try:
        import sympy
    except ImportError:
        return {"skill": "symbolic_math", "error": "sympy not installed"}
    try:
        x = sympy.symbols(variable)
        expr = sympy.sympify(expression)
        op = operation.lower()
        if op == "simplify":
            r = sympy.simplify(expr)
        elif op == "expand":
            r = sympy.expand(expr)
        elif op == "factor":
            r = sympy.factor(expr)
        elif op == "solve":
            r = sympy.solve(expr, x)
        elif op == "diff":
            r = sympy.diff(expr, x)
        elif op == "integrate":
            r = sympy.integrate(expr, x)
        elif op == "latex":
            r = sympy.latex(expr)
        else:
            return {"skill": "symbolic_math", "error": f"unknown operation: {operation}"}
        return {"skill": "symbolic_math", "operation": op, "input": str(expr),
                "result": str(r), "latex": sympy.latex(r) if op != "latex" else r}
    except Exception as e:
        return {"skill": "symbolic_math", "error": str(e)}
