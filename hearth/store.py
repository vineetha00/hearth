"""SQLite persistence for journal entries and commitments."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from difflib import SequenceMatcher

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL,
    mood TEXT,
    themes TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS commitments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    due_date TEXT,
    importance INTEGER NOT NULL DEFAULT 3,
    person TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    source_entry_id INTEGER,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_entries_user ON entries(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_commit_user ON commitments(user_id, status);
"""

_WORD = re.compile(r"[a-z0-9']+")
_STOP = {"the", "a", "an", "to", "my", "i", "and", "of", "for", "on", "in", "it", "that", "with", "up", "finish", "complete", "submit", "finalize", "turn", "hand", "do", "get", "work"}


_ALIAS = {
    "called": "call", "calling": "call", "phoned": "call", "phone": "call", "rang": "call", "talked": "call", "talk": "call",
    "spoke": "call", "speak": "call", "facetimed": "call", "texted": "text", "emailed": "email", "booked": "book",
    "submitted": "submit", "sent": "send",
}


def _tokens(s: str) -> set[str]:
    return {_ALIAS.get(w, w) for w in _WORD.findall(s.lower()) if w not in _STOP}


def similarity(a: str, b: str) -> float:
    """Blend of character-level ratio and token overlap, 0..1."""
    ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    ta, tb = _tokens(a), _tokens(b)
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb))) if ta and tb else 0.0
    return max(ratio, overlap)


class Store:
    def __init__(self, path: str | None = None):
        path = path or os.environ.get("HEARTH_DB", "data/hearth.db")
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    # ---- entries -------------------------------------------------------
    def add_entry(self, user_id: str, text: str, mood: str | None, themes: list[str], now_iso: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO entries(user_id, created_at, text, mood, themes) VALUES (?,?,?,?,?)",
                (user_id, now_iso, text, mood, json.dumps(themes)),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent_entries(self, user_id: str, since_iso: str, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM entries WHERE user_id=? AND created_at>=? ORDER BY created_at DESC LIMIT ?",
                (user_id, since_iso, limit),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["themes"] = json.loads(d["themes"])
            out.append(d)
        return out

    # ---- commitments ---------------------------------------------------
    def add_commitment(
        self,
        user_id: str,
        text: str,
        due_date: str | None,
        importance: int,
        person: str | None,
        source_entry_id: int | None,
        now_iso: str,
    ) -> tuple[dict, bool]:
        """Insert a commitment. Returns (row, created). Near-duplicates of an open item are merged."""
        with self._lock:
            for r in self._conn.execute(
                "SELECT * FROM commitments WHERE user_id=? AND status='open'", (user_id,)
            ).fetchall():
                if similarity(r["text"], text) >= 0.82:
                    # Keep the existing item but adopt a newly learned due date / higher importance.
                    new_due = due_date or r["due_date"]
                    new_imp = max(int(r["importance"]), int(importance))
                    self._conn.execute(
                        "UPDATE commitments SET due_date=?, importance=? WHERE id=?",
                        (new_due, new_imp, r["id"]),
                    )
                    self._conn.commit()
                    row = dict(self._conn.execute("SELECT * FROM commitments WHERE id=?", (r["id"],)).fetchone())
                    row["date_added"] = bool(due_date and not r["due_date"]) or bool(due_date and due_date != r["due_date"])
                    return row, False
            cur = self._conn.execute(
                "INSERT INTO commitments(user_id, text, due_date, importance, person, source_entry_id, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (user_id, text, due_date, int(importance), person, source_entry_id, now_iso),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM commitments WHERE id=?", (cur.lastrowid,)).fetchone()
            return dict(row), True

    def list_commitments(self, user_id: str, status: str | None = "open") -> list[dict]:
        q = "SELECT * FROM commitments WHERE user_id=?"
        args: list = [user_id]
        if status:
            q += " AND status=?"
            args.append(status)
        q += " ORDER BY (due_date IS NULL), due_date, importance DESC, id"
        with self._lock:
            return [dict(r) for r in self._conn.execute(q, args).fetchall()]

    def get_commitment(self, user_id: str, commitment_id: int) -> dict | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM commitments WHERE user_id=? AND id=?", (user_id, commitment_id)
            ).fetchone()
        return dict(r) if r else None

    def complete(self, user_id: str, commitment_id: int, now_iso: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE commitments SET status='done', completed_at=? WHERE user_id=? AND id=? AND status='open'",
                (now_iso, user_id, commitment_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def completed_since(self, user_id: str, since_iso: str) -> int:
        with self._lock:
            r = self._conn.execute(
                "SELECT COUNT(*) c FROM commitments WHERE user_id=? AND status='done' AND completed_at>=?",
                (user_id, since_iso),
            ).fetchone()
        return int(r["c"])

    def find_open_matches(self, user_id: str, query: str, threshold: float = 0.45) -> list[tuple[float, dict]]:
        scored = [(similarity(c["text"], query), c) for c in self.list_commitments(user_id, "open")]
        scored = [(s, c) for s, c in scored if s >= threshold]
        scored.sort(key=lambda x: -x[0])
        return scored
