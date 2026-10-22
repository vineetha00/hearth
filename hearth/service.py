"""Hearth's behaviour, independent of transport. The MCP tools are thin wrappers over this."""

from __future__ import annotations

from datetime import date, timedelta

from . import dates, llm
from .store import Store


def _listen(items: list[dict], today_: date, limit: int = 5, skip_when: str | None = None) -> str:
    """Join commitments into a sentence a voice assistant can read aloud.

    ``skip_when`` drops a due phrase the listener already knows (asking about "tomorrow" shouldn't repeat it).
    """
    parts = []
    for c in items[:limit]:
        when = dates.describe(date.fromisoformat(c["due_date"]), today_) if c["due_date"] else None
        if when == skip_when:
            when = None
        parts.append(f"{c['text']}" + (f", {when}" if when and when != "no date" else ""))
    extra = len(items) - limit
    sentence = "; ".join(parts)
    if extra > 0:
        sentence += f"; and {extra} more"
    return sentence


def _public(c: dict, today_: date) -> dict:
    due = date.fromisoformat(c["due_date"]) if c["due_date"] else None
    return {
        "id": c["id"],
        "text": c["text"],
        "due_date": c["due_date"],
        "due": dates.describe(due, today_) if due else None,
        "overdue": bool(due and due < today_),
        "importance": c["importance"],
        "person": c["person"],
        "status": c["status"],
    }


class Hearth:
    def __init__(self, store: Store | None = None, brain: llm.Brain | None = None):
        self.store = store or Store()
        self.brain = brain or llm.Brain()

    # ------------------------------------------------------------------ diary
    def log_journal_entry(self, text: str, user_id: str = "me") -> dict:
        text = (text or "").strip()
        if not text:
            return {"speech": "I'm listening. Tell me about your day whenever you're ready.", "saved": False}
        now = dates.now()
        today_ = now.date()
        analysis = self.brain.analyze_entry(text, today_)
        if len(text.split()) < 5:
            analysis.commitments = []  # a fragment like 'due by Friday' is a question, not a promise
        entry_id = self.store.add_entry(user_id, text, analysis.mood, analysis.themes, now.isoformat())

        new_items, existing_items = [], []
        for c in analysis.commitments:
            row, created = self.store.add_commitment(
                user_id, c.text, c.due_date, c.importance, c.person, entry_id, now.isoformat()
            )
            (new_items if created else existing_items).append(row)
        dated = [r for r in existing_items if r.pop("date_added", False)]

        speech = analysis.acknowledgement
        if new_items:
            speech += " I've added to your list: " + _listen(new_items, today_, 3) + "."
        if dated:
            speech += " I've set the date on what you already had: " + _listen(dated, today_, 3) + "."
        elif existing_items and not new_items:
            speech += " That's already on your list." if len(existing_items) == 1 else " Those are already on your list."
        return {
            "speech": speech,
            "saved": True,
            "entry_id": entry_id,
            "mood": analysis.mood,
            "themes": analysis.themes,
            "commitments_added": [_public(c, today_) for c in new_items],
            "commitments_already_tracked": [_public(c, today_) for c in existing_items],
            "engine": analysis.engine,
        }

    def get_reflection(self, days: int = 7, user_id: str = "me") -> dict:
        days = max(1, min(60, int(days)))
        now = dates.now()
        since = (now - timedelta(days=days)).isoformat()
        entries = self.store.recent_entries(user_id, since)
        done = self.store.completed_since(user_id, since)
        open_items = self.store.list_commitments(user_id, "open")
        speech, engine = self.brain.reflect(entries, done, open_items, days, now.date())
        return {"speech": speech, "entries_considered": len(entries), "completed": done, "open": len(open_items), "engine": engine}

    # ------------------------------------------------------------ commitments
    def get_commitments(self, timeframe: str = "all", user_id: str = "me") -> dict:
        today_ = dates.today()
        tf = (timeframe or "all").lower().replace(" ", "_").replace("-", "_")
        items = self.store.list_commitments(user_id, "open")

        def due_of(c):
            return date.fromisoformat(c["due_date"]) if c["due_date"] else None

        week_end = today_ + timedelta(days=(6 - today_.weekday()))  # through Sunday
        if tf in ("today", "tonight"):
            sel = [c for c in items if due_of(c) and due_of(c) <= today_]
            label = "today"
        elif tf == "tomorrow":
            sel = [c for c in items if due_of(c) == today_ + timedelta(days=1)]
            label = "tomorrow"
        elif tf in ("this_week", "week", "thisweek"):
            sel = [c for c in items if due_of(c) and due_of(c) <= week_end]
            label = "this week"
        elif tf == "overdue":
            sel = [c for c in items if due_of(c) and due_of(c) < today_]
            label = "overdue"
        elif tf in ("undated", "someday", "no_date"):
            sel = [c for c in items if not c["due_date"]]
            label = "with no date"
        else:
            sel, label = items, "open"
        sel.sort(key=lambda c: (due_of(c) is None, due_of(c) or date.max, -c["importance"]))

        if not sel:
            speech = {
                "today": "Nothing is due today. Nice.",
                "tomorrow": "Nothing is due tomorrow.",
                "this week": "You have nothing due this week that I know of.",
                "overdue": "Nothing is overdue. You're caught up.",
            }.get(label, "Your list is empty. Tell me about your day and I'll keep track of what you say you'll do.")
        else:
            n = len(sel)
            lead = {
                "today": f"You have {n} thing{'s' if n != 1 else ''} to handle today",
                "tomorrow": f"You have {n} thing{'s' if n != 1 else ''} due tomorrow",
                "this week": f"You said you'd do {n} thing{'s' if n != 1 else ''} this week",
                "overdue": f"{n} thing{'s are' if n != 1 else ' is'} overdue",
            }.get(label, f"You have {n} open thing{'s' if n != 1 else ''}")
            speech = f"{lead}: {_listen(sel, today_, skip_when=label if label in ('today', 'tomorrow') else None)}."
        return {"speech": speech, "timeframe": label, "count": len(sel), "commitments": [_public(c, today_) for c in sel]}

    def add_commitment(self, text: str, due: str | None = None, user_id: str = "me") -> dict:
        today_ = dates.today()
        text = (text or "").strip()
        if len(text.split()) < 2:
            return {"speech": "What should I add to your list?", "added": False}
        due_date = dates.parse_due(due or text, today_)
        clean = dates.strip_due_words(text) if dates.parse_due(text, today_) else text
        importance = 4 if llm._URGENT.search(text) else 3
        pm = llm._PEOPLE.search(text)
        row, created = self.store.add_commitment(
            user_id, clean[0].upper() + clean[1:], due_date.isoformat() if due_date else None, importance,
            pm.group(1) if pm else None, None, dates.now().isoformat(),
        )
        when = dates.describe(due_date, today_) if due_date else None
        speech = f"Added: {row['text']}" + (f", {when}" if when else "") + "." if created else f"That's already on your list: {row['text']}."
        return {"speech": speech, "added": created, "commitment": _public(row, today_)}

    def complete_commitment(self, description: str = "", commitment_id: int | None = None, user_id: str = "me") -> dict:
        today_ = dates.today()
        now_iso = dates.now().isoformat()
        if commitment_id is not None:
            c = self.store.get_commitment(user_id, int(commitment_id))
            if c and c["status"] == "open" and self.store.complete(user_id, c["id"], now_iso):
                return {"speech": f"Done. I crossed off: {c['text']}.", "completed": True, "commitment": _public({**c, "status": "done"}, today_)}
            return {"speech": "I couldn't find that item on your open list.", "completed": False}
        matches = self.store.find_open_matches(user_id, description or "")
        if not matches:
            return {"speech": "I couldn't find that on your list. Say 'what's on my list' to hear what I have.", "completed": False}
        best_score, best = matches[0]
        ambiguous = len(matches) > 1 and (best_score - matches[1][0]) < 0.12 and best_score < 0.9
        if ambiguous:
            options = " or ".join(c["text"] for _, c in matches[:3])
            return {
                "speech": f"Did you mean {options}?",
                "completed": False,
                "needs_clarification": True,
                "options": [_public(c, today_) for _, c in matches[:3]],
            }
        self.store.complete(user_id, best["id"], now_iso)
        remaining = len(self.store.list_commitments(user_id, "open"))
        tail = f" You have {remaining} left." if remaining else " Your list is clear."
        return {
            "speech": f"Nice. I crossed off: {best['text']}.{tail}",
            "completed": True,
            "commitment": _public({**best, "status": "done"}, today_),
            "remaining": remaining,
        }

    def prioritize_today(self, limit: int = 3, user_id: str = "me") -> dict:
        today_ = dates.today()
        items = self.store.list_commitments(user_id, "open")
        if not items:
            return {"speech": "Nothing on your list right now. A rare quiet moment.", "top": []}
        ranked = llm.prioritize(items, today_, max(1, min(5, int(limit))))
        top = ranked[0]
        speech = f"The most important thing right now is: {top['text']}, because {top['why']}."
        if len(ranked) > 1:
            nxt = "; then ".join(c["text"] for c in ranked[1:])
            speech += f" After that: {nxt}."
        return {
            "speech": speech,
            "top": [{**_public(c, today_), "why": c["why"], "score": c["score"]} for c in ranked],
        }
