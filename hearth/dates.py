"""Time helpers and a small natural-language due-date parser.

The LLM path asks Bedrock for ISO dates directly; this parser is the offline
fallback and is also used to sanity-check model output.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

DEFAULT_TZ = "America/Los_Angeles"

WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


def tz() -> ZoneInfo:
    return ZoneInfo(os.environ.get("HEARTH_TZ", DEFAULT_TZ))


def now() -> datetime:
    """Current time in the user's zone. HEARTH_NOW (ISO) pins the clock for demos/tests."""
    pinned = os.environ.get("HEARTH_NOW")
    if pinned:
        dt = datetime.fromisoformat(pinned)
        return dt if dt.tzinfo else dt.replace(tzinfo=tz())
    return datetime.now(tz())


def today() -> date:
    return now().date()


def _next_weekday(today_: date, target: int, *, allow_today: bool = False) -> date:
    delta = (target - today_.weekday()) % 7
    if delta == 0 and not allow_today:
        delta = 7
    return today_ + timedelta(days=delta)


def parse_due(text: str, today_: date | None = None) -> date | None:
    """Return the due date mentioned in ``text`` or None."""
    today_ = today_ or today()
    t = text.lower()

    if re.search(r"\b(today|tonight|this evening|this afternoon|this morning)\b", t):
        return today_
    if re.search(r"\b(tomorrow|tmrw|tmr)\b", t):
        return today_ + timedelta(days=1)

    m = re.search(r"\bin (\d+|one|two|three|four|five|six|seven) days?\b", t)
    if m:
        n = NUMBER_WORDS.get(m.group(1)) or int(m.group(1))
        return today_ + timedelta(days=n)

    if re.search(r"\bnext week\b", t):
        return _next_weekday(today_, 0)
    if re.search(r"\bthis weekend\b", t):
        return _next_weekday(today_, 5, allow_today=True)
    if re.search(r"\b(end of (the )?week|eow)\b", t):
        return _next_weekday(today_, 4, allow_today=True)

    m = re.search(
        r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", t
    )
    if m:
        month, day = MONTHS[m.group(1)], int(m.group(2))
        try:
            d = date(today_.year, month, day)
            if d < today_:
                d = date(today_.year + 1, month, day)
            return d
        except ValueError:
            pass

    m = re.search(r"\b(?:next\s+)?(" + "|".join(sorted(WEEKDAYS, key=len, reverse=True)) + r")\b", t)
    if m:
        target = WEEKDAYS[m.group(1)]
        d = _next_weekday(today_, target)
        if re.search(r"\bnext\s+" + m.group(1), t) and d - today_ < timedelta(days=7) and target > today_.weekday():
            # "next friday" said on a Monday means the one after this coming Friday
            d += timedelta(days=7)
        return d
    return None


DUE_PHRASES = re.compile(
    r"\b(by |on |before |due |for |until )?"
    r"(today|tonight|tomorrow|tmrw|this weekend|next week|end of (the )?week|"
    r"(next )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"in \w+ days?)\b",
    re.IGNORECASE,
)


def strip_due_words(text: str) -> str:
    cleaned = DUE_PHRASES.sub("", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,.;-")


def describe(due: date | None, today_: date | None = None) -> str:
    """Spoken-style description of a due date."""
    today_ = today_ or today()
    if due is None:
        return "no date"
    delta = (due - today_).days
    if delta < -1:
        return f"{-delta} days overdue"
    if delta == -1:
        return "overdue since yesterday"
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta < 7:
        return due.strftime("%A")
    return due.strftime("%A, %B %-d")
