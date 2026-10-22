"""Backfill a believable week of evenings so the demo can show real patterns.

Usage (server stopped, or pointing at the same DB):
    HEARTH_DB=data/hearth.db python scripts/seed_demo.py

It writes through the same service layer the MCP tools use, with the clock moved back one day at a time,
so entries, commitments and completions all carry realistic timestamps. Uses the real clock: "yesterday"
is yesterday. Safe to re-run on a fresh DB; refuses to run if the DB already has entries.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hearth import dates  # noqa: E402
from hearth.service import Hearth  # noqa: E402

# (days ago, what the person said, optional: description of something they finished that evening)
WEEK = [
    (6, "Started the week behind. Two meetings ran over and I'm already tired. I need to renew my lease by Thursday.", None),
    (5, "Rushed all day, no real lunch. I miss my mom's cooking. I should call my mom this weekend.", None),
    (4, "Better day, actually. I finished the first draft and felt good about it. I'll send it to my advisor tomorrow.", None),
    (3, "Deadline stress again. I'm exhausted and everything feels like it's piling up. I need to book the dentist soon.", "sent the draft to my advisor"),
    (2, "So drained tonight. Skipped dinner, stayed late at the lab. I promised my roommate I'd take out the trash and I keep forgetting.", None),
    (1, "Quiet evening, a bit lonely if I'm honest. Meetings all day and I'm behind on the lab report again.", "renewed my lease"),
]


def main() -> None:
    db = os.environ.get("HEARTH_DB", "data/hearth.db")
    os.environ.setdefault("HEARTH_LLM", "auto")
    app = Hearth()
    if app.store.recent_entries("me", "0001-01-01", 1):
        sys.exit(f"{db} already has entries. Delete it (or point HEARTH_DB elsewhere) to reseed.")
    real_now = dates.now()
    try:
        for days_ago, text, finished in WEEK:
            when = (real_now - timedelta(days=days_ago)).replace(hour=21, minute=15, second=0, microsecond=0)
            os.environ["HEARTH_NOW"] = when.isoformat()
            out = app.log_journal_entry(text)
            print(f"-{days_ago}d  {len(out['commitments_added'])} commitment(s)  [{out['engine']}]")
            if finished:
                os.environ["HEARTH_NOW"] = (when + timedelta(minutes=10)).isoformat()
                print("      ", app.complete_commitment(finished)["speech"])
    finally:
        os.environ.pop("HEARTH_NOW", None)
    open_items = app.get_commitments("all")
    print(f"\nSeeded. {open_items['count']} open commitments:")
    for c in open_items["commitments"]:
        print(f"  - {c['text']} ({c['due'] or 'no date'})")


if __name__ == "__main__":
    main()
