"""Personal productivity skills — todo list, habits, journal, analyzers.

All state lives under ~/.mio/ so it survives restarts and isn't tied
to any one chat session.
"""
from __future__ import annotations

import csv as _csv
import datetime as _dt
import io
import json
import re
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

PIMIO_DIR = Path.home() / ".mio"
PIMIO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Todo list
# ============================================================
_TODO_DB = PIMIO_DIR / "todos.sqlite"


def _todo_conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_TODO_DB))
    c.executescript("""
        CREATE TABLE IF NOT EXISTS todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            list_name TEXT DEFAULT 'inbox',
            priority INTEGER DEFAULT 2,
            due TEXT,
            created INTEGER,
            done INTEGER DEFAULT 0,
            done_at INTEGER
        );
    """)
    return c


def todo_add(text: str, list_name: str = "inbox", priority: int = 2, due: str = "") -> dict:
    if not text:
        return {"skill": "todo_add", "error": "text required"}
    c = _todo_conn()
    cur = c.cursor()
    cur.execute(
        "INSERT INTO todos (text, list_name, priority, due, created) VALUES (?, ?, ?, ?, ?)",
        (text, list_name or "inbox", int(priority) if priority else 2, due or "", int(time.time()))
    )
    new_id = cur.lastrowid
    c.commit()
    c.close()
    return {"skill": "todo_add", "ok": True, "id": new_id}


def todo_list(include_done: bool = False, list_name: str | None = None, limit: int = 50) -> dict:
    c = _todo_conn()
    cur = c.cursor()
    sql = "SELECT id, text, list_name, priority, due, created, done, done_at FROM todos WHERE 1=1"
    args: list[Any] = []
    if not include_done:
        sql += " AND done = 0"
    if list_name:
        sql += " AND list_name = ?"
        args.append(list_name)
    sql += " ORDER BY done, priority DESC, created DESC LIMIT ?"
    args.append(int(limit))
    rows = cur.execute(sql, args).fetchall()
    c.close()
    return {
        "skill": "todo_list",
        "todos": [
            {"id": r[0], "text": r[1], "list": r[2], "priority": r[3],
             "due": r[4], "created": r[5], "done": bool(r[6]), "done_at": r[7]}
            for r in rows
        ],
    }


def todo_done(todo_id: int, done: bool = True) -> dict:
    c = _todo_conn()
    cur = c.cursor()
    cur.execute("UPDATE todos SET done = ?, done_at = ? WHERE id = ?",
                (1 if done else 0, int(time.time()) if done else None, int(todo_id)))
    c.commit()
    changed = cur.rowcount
    c.close()
    return {"skill": "todo_done", "ok": bool(changed), "id": int(todo_id)}


def todo_delete(todo_id: int) -> dict:
    c = _todo_conn()
    cur = c.cursor()
    cur.execute("DELETE FROM todos WHERE id = ?", (int(todo_id),))
    c.commit()
    c.close()
    return {"skill": "todo_delete", "ok": True}


# ============================================================
# Habit tracker
# ============================================================
_HABIT_DB = PIMIO_DIR / "habits.sqlite"


def _habit_conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_HABIT_DB))
    c.executescript("""
        CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            cadence TEXT DEFAULT 'daily',
            created INTEGER
        );
        CREATE TABLE IF NOT EXISTS habit_checkins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            habit_id INTEGER,
            day TEXT,
            note TEXT,
            UNIQUE(habit_id, day)
        );
    """)
    return c


def habit_add(name: str, cadence: str = "daily") -> dict:
    c = _habit_conn()
    cur = c.cursor()
    cur.execute("INSERT OR IGNORE INTO habits (name, cadence, created) VALUES (?, ?, ?)",
                (name, cadence, int(time.time())))
    c.commit()
    cur.execute("SELECT id FROM habits WHERE name = ?", (name,))
    row = cur.fetchone()
    c.close()
    return {"skill": "habit_add", "ok": True, "id": row[0] if row else None}


def habit_checkin(habit_id: int | None = None, name: str | None = None, note: str = "") -> dict:
    c = _habit_conn()
    cur = c.cursor()
    if name and habit_id is None:
        cur.execute("SELECT id FROM habits WHERE name = ?", (name,))
        r = cur.fetchone()
        habit_id = r[0] if r else None
    if habit_id is None:
        c.close()
        return {"skill": "habit_checkin", "error": "unknown habit"}
    day = _dt.date.today().isoformat()
    cur.execute("INSERT OR IGNORE INTO habit_checkins (habit_id, day, note) VALUES (?, ?, ?)",
                (int(habit_id), day, note or ""))
    c.commit()
    c.close()
    return {"skill": "habit_checkin", "ok": True, "habit_id": habit_id, "day": day}


def habit_list() -> dict:
    c = _habit_conn()
    cur = c.cursor()
    rows = cur.execute(
        "SELECT h.id, h.name, h.cadence, COUNT(c.id) AS checkins, "
        "MAX(c.day) AS last_checkin "
        "FROM habits h LEFT JOIN habit_checkins c ON c.habit_id = h.id "
        "GROUP BY h.id ORDER BY h.created"
    ).fetchall()
    result = []
    today = _dt.date.today()
    for r in rows:
        hid, name, cadence, checkins, last = r
        # Compute streak — count consecutive days ending today with a check-in
        streak = 0
        cur.execute("SELECT day FROM habit_checkins WHERE habit_id = ? ORDER BY day DESC", (hid,))
        days = [x[0] for x in cur.fetchall()]
        expect = today
        for d in days:
            try:
                dparsed = _dt.date.fromisoformat(d)
            except Exception:
                continue
            if dparsed == expect:
                streak += 1
                expect = expect - _dt.timedelta(days=1)
            else:
                break
        result.append({
            "id": hid, "name": name, "cadence": cadence,
            "checkins": checkins or 0,
            "last_checkin": last,
            "streak_days": streak,
        })
    c.close()
    return {"skill": "habit_list", "habits": result}


# ============================================================
# Daily journal
# ============================================================
_JOURNAL_DIR = PIMIO_DIR / "journal"
_JOURNAL_DIR.mkdir(parents=True, exist_ok=True)


def journal_append(entry: str, mood: str = "", tags: list[str] | None = None) -> dict:
    if not entry:
        return {"skill": "journal_append", "error": "entry required"}
    day = _dt.date.today().isoformat()
    path = _JOURNAL_DIR / f"{day}.md"
    stamp = _dt.datetime.now().strftime("%H:%M:%S")
    header = f"\n\n## {stamp}"
    if mood: header += f"  _{mood}_"
    if tags: header += f"  `#{' #'.join(tags)}`"
    block = header + "\n\n" + entry + "\n"
    with path.open("a", encoding="utf-8") as f:
        if path.stat().st_size == 0:
            f.write(f"# {day}\n")
        f.write(block)
    return {"skill": "journal_append", "path": str(path), "day": day}


def journal_read(day: str = "") -> dict:
    target = day or _dt.date.today().isoformat()
    path = _JOURNAL_DIR / f"{target}.md"
    if not path.exists():
        return {"skill": "journal_read", "day": target, "content": "", "missing": True}
    return {"skill": "journal_read", "day": target, "content": path.read_text()}


def journal_search(query: str, limit: int = 20) -> dict:
    if not query:
        return {"skill": "journal_search", "error": "query required"}
    ql = query.lower()
    hits = []
    for f in sorted(_JOURNAL_DIR.glob("*.md"), reverse=True):
        try:
            text = f.read_text()
        except Exception:
            continue
        if ql in text.lower():
            # Grab a 200-char snippet around the match
            i = text.lower().find(ql)
            snippet = text[max(0, i - 100): i + 200]
            hits.append({"day": f.stem, "snippet": snippet.strip()})
        if len(hits) >= limit:
            break
    return {"skill": "journal_search", "query": query, "hits": hits}


# ============================================================
# JSON analyzer
# ============================================================
def analyze_json(json_str: str) -> dict:
    """Return schema skeleton, size, depth, and sample values for a JSON blob."""
    try:
        data = json.loads(json_str)
    except Exception as e:
        return {"skill": "analyze_json", "error": f"parse error: {e}"}

    def walk(v, depth=0):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        t = type(v).__name__
        if isinstance(v, dict):
            return {k: walk(val, depth + 1) for k, val in list(v.items())[:20]}
        if isinstance(v, list):
            if not v:
                return ["empty array"]
            sample = walk(v[0], depth + 1)
            return [f"list of {t} (len {len(v)}, sample):", sample]
        return f"{t} = {repr(v)[:80]}"

    max_depth = 0
    schema = walk(data)
    return {
        "skill": "analyze_json",
        "size_bytes": len(json_str),
        "root_type": type(data).__name__,
        "root_keys": list(data.keys())[:30] if isinstance(data, dict) else None,
        "root_length": len(data) if hasattr(data, "__len__") else None,
        "max_depth": max_depth,
        "schema": schema,
    }


# ============================================================
# CSV analyzer
# ============================================================
def analyze_csv(csv_text: str, delimiter: str = ",") -> dict:
    """Per-column type inference, stats, null counts."""
    if not csv_text:
        return {"skill": "analyze_csv", "error": "text required"}
    reader = _csv.reader(io.StringIO(csv_text), delimiter=delimiter or ",")
    rows = list(reader)
    if not rows:
        return {"skill": "analyze_csv", "error": "empty"}
    headers = rows[0]
    data = rows[1:]
    cols: list[dict] = []
    for i, name in enumerate(headers):
        values = [r[i] if i < len(r) else "" for r in data]
        non_null = [v for v in values if v not in ("", None)]
        # Type inference
        num_vals: list[float] = []
        is_int = is_float = is_date = True
        for v in non_null:
            try:
                f = float(v)
                num_vals.append(f)
                if f != int(f): is_int = False
            except Exception:
                is_int = is_float = False
            if is_date:
                try:
                    _dt.datetime.fromisoformat(v)
                except Exception:
                    is_date = False
        guessed = "integer" if (is_int and num_vals) else ("float" if (is_float and num_vals) else ("datetime" if is_date and non_null else "text"))
        entry = {
            "name": name,
            "type": guessed,
            "nulls": len(values) - len(non_null),
            "unique": len(set(non_null)),
            "sample": non_null[:3],
        }
        if num_vals:
            entry["min"] = min(num_vals)
            entry["max"] = max(num_vals)
            entry["mean"] = round(statistics.fmean(num_vals), 4)
            if len(num_vals) > 1:
                entry["stdev"] = round(statistics.stdev(num_vals), 4)
        cols.append(entry)
    return {
        "skill": "analyze_csv",
        "rows": len(data),
        "cols": len(headers),
        "headers": headers,
        "columns": cols,
    }
