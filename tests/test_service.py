"""Behavioural tests. The clock is pinned to Tuesday 2026-10-20 evening (Pacific)."""

import json
from datetime import date

import pytest

from hearth import dates, llm
from hearth.service import Hearth
from hearth.store import Store


@pytest.fixture(autouse=True)
def pinned_clock(monkeypatch):
    monkeypatch.setenv("HEARTH_NOW", "2026-10-20T20:30:00")
    monkeypatch.setenv("HEARTH_TZ", "America/Los_Angeles")


@pytest.fixture
def h():
    return Hearth(Store(":memory:"), llm.Brain(mode="heuristic"))


TUE = date(2026, 10, 20)


def test_parse_due_variants():
    assert dates.parse_due("call mom tomorrow", TUE) == date(2026, 10, 21)
    assert dates.parse_due("submit by friday", TUE) == date(2026, 10, 23)
    assert dates.parse_due("on tuesday", TUE) == date(2026, 10, 27)  # same weekday means next week
    assert dates.parse_due("next week", TUE) == date(2026, 10, 26)
    assert dates.parse_due("in 3 days", TUE) == date(2026, 10, 23)
    assert dates.parse_due("by oct 25th", TUE) == date(2026, 10, 25)
    assert dates.parse_due("nothing here", TUE) is None


def test_extracts_commitments_from_rant(h):
    r = h.log_journal_entry(
        "Ugh, what a day. Back-to-back meetings and I'm exhausted. "
        "I told my sister I'd call her back tomorrow. I also need to submit the lab report by Friday."
    )
    assert r["saved"] and r["engine"] == "heuristic"
    items = {c["text"].lower(): c for c in r["commitments_added"]}
    assert len(items) == 2, items
    sister = next(c for c in r["commitments_added"] if "call" in c["text"].lower())
    lab = next(c for c in r["commitments_added"] if "lab report" in c["text"].lower())
    assert sister["due_date"] == "2026-10-21" and sister["person"] == "sister"
    assert lab["due_date"] == "2026-10-23" and lab["importance"] >= 4
    assert "I've added to your list" in r["speech"]
    assert r["mood"] in {"drained", "stressed", "mixed"}


def test_two_commitments_in_one_breath(h):
    r = h.log_journal_entry("I told my sister I'd call her back tomorrow and I need to submit the lab report by Friday")
    got = {c["text"]: c["due_date"] for c in r["commitments_added"]}
    assert got == {"Call my sister back": "2026-10-21", "Submit the lab report": "2026-10-23"}, got


def test_no_false_commitments(h):
    r = h.log_journal_entry("I'll be tired tomorrow probably. Work was fine and I think it rained.")
    assert r["commitments_added"] == []


def test_duplicate_mentions_merge_and_learn_due_date(h):
    h.log_journal_entry("I need to book the dentist.")
    r = h.log_journal_entry("I really need to book the dentist by friday.")
    assert r["commitments_added"] == [] and len(r["commitments_already_tracked"]) == 1
    cs = h.get_commitments("all")["commitments"]
    assert len(cs) == 1 and cs[0]["due_date"] == "2026-10-23"


def test_timeframes_and_overdue(h):
    h.add_commitment("pay the electric bill", "yesterday")  # no parse for yesterday -> undated
    h.log_journal_entry("I have to email my advisor tomorrow. I need to finish the slides by friday.")
    assert h.get_commitments("tomorrow")["count"] == 1
    assert h.get_commitments("this_week")["count"] == 2
    assert h.get_commitments("undated")["count"] == 1
    assert h.get_commitments("overdue")["count"] == 0
    assert "Nothing is overdue" in h.get_commitments("overdue")["speech"]


def test_overdue_surfaces_in_today(h, monkeypatch):
    monkeypatch.setenv("HEARTH_NOW", "2026-10-19T09:00:00")
    h.log_journal_entry("I need to renew my lease by today.")
    monkeypatch.setenv("HEARTH_NOW", "2026-10-21T09:00:00")
    out = h.get_commitments("today")
    assert out["count"] == 1 and out["commitments"][0]["overdue"]


def test_prioritize_prefers_promises_and_deadlines(h):
    h.log_journal_entry(
        "I told my sister I'd call her back tomorrow. I should organize my desk sometime. "
        "I need to submit the lab report by friday."
    )
    p = h.prioritize_today(3)
    assert "sister" in p["top"][0]["text"].lower() or "call" in p["top"][0]["text"].lower()
    assert p["top"][-1]["text"].lower().startswith("organize")
    assert p["top"][0]["why"]


def test_complete_by_meaning_and_ambiguity(h):
    h.log_journal_entry("I need to call my sister tomorrow. I need to call the landlord on friday.")
    amb = h.complete_commitment("called someone")
    assert not amb["completed"]
    done = h.complete_commitment("I called my sister")
    assert done["completed"] and "sister" in done["commitment"]["text"].lower()
    assert h.get_commitments("all")["count"] == 1
    assert not h.complete_commitment("learn to juggle")["completed"]


def test_reflection_notices_patterns(h, monkeypatch):
    for day, txt in [
        ("2026-10-17", "Deadline stress again, I'm exhausted and behind on homework."),
        ("2026-10-18", "Another rushed day. Tired, and I missed my mom's call."),
        ("2026-10-19", "Exam tomorrow, so drained. Skipped dinner."),
    ]:
        monkeypatch.setenv("HEARTH_NOW", f"{day}T21:00:00")
        h.log_journal_entry(txt)
    monkeypatch.setenv("HEARTH_NOW", "2026-10-20T21:00:00")
    r = h.get_reflection(7)
    assert r["entries_considered"] == 3
    assert "heavy" in r["speech"] or "came up" in r["speech"]
    assert "should" not in r["speech"].lower()  # observation, not advice


def test_empty_states(h):
    assert "empty" in h.get_commitments("all")["speech"].lower()
    assert "yet" in h.get_reflection()["speech"]
    assert h.prioritize_today()["top"] == []


class FakeBedrock:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def converse(self, **kw):
        self.calls.append(kw)
        return {"output": {"message": {"content": [{"text": "Sure! " + json.dumps(self.payload)}]}}}


def test_bedrock_path_parses_and_validates():
    fake = FakeBedrock({
        "acknowledgement": "That sounds like a lot.",
        "mood": "stressed",
        "themes": ["work", "family"],
        "commitments": [
            {"text": "Call Priya back", "due_date": "2026-10-22", "importance": 9, "person": "Priya"},
            {"text": "ok", "due_date": None, "importance": 3, "person": None},  # too short: dropped
            {"text": "Send the invoice", "due_date": "not-a-date", "importance": 2, "person": None},
        ],
    })
    hh = Hearth(Store(":memory:"), llm.Brain(client=fake, model_id="test-model"))
    r = hh.log_journal_entry("a long rambling entry about my whole day at work")
    assert r["engine"] == "bedrock"
    texts = [c["text"] for c in r["commitments_added"]]
    assert texts == ["Call Priya back", "Send the invoice"]
    assert r["commitments_added"][0]["importance"] == 5  # clamped
    assert r["commitments_added"][1]["due_date"] is None  # bad date discarded
    call = fake.calls[0]
    assert call["modelId"] == "test-model" and "Tuesday, 2026-10-20" in call["messages"][0]["content"][0]["text"]
    assert "Never give advice" in call["system"][0]["text"]


def test_bedrock_failure_falls_back():
    class Boom:
        def converse(self, **kw):
            raise RuntimeError("throttled")

    hh = Hearth(Store(":memory:"), llm.Brain(client=Boom()))
    r = hh.log_journal_entry("I need to call mom tomorrow.")
    assert r["engine"] == "heuristic" and len(r["commitments_added"]) == 1


def test_same_task_with_different_verb_is_one_item():
    from hearth.store import Store
    st = Store(":memory:")
    a, created_a = st.add_commitment("me", "Complete lab report", None, 3, None, None, "2026-10-20T21:00:00")
    b, created_b = st.add_commitment("me", "Submit the lab report", "2026-10-23", 4, None, None, "2026-10-20T21:05:00")
    assert created_a and not created_b and a["id"] == b["id"]
    assert b["due_date"] == "2026-10-23"   # the newly learned date is kept


def test_bedrock_weekday_is_resolved_in_code_not_by_the_model(monkeypatch):
    from datetime import date
    from hearth import llm
    monkeypatch.setenv("HEARTH_NOW", "2026-10-20T21:00:00-07:00")
    brain = llm.Brain()
    brain._bedrock = lambda: object()
    brain._converse = lambda *a, **k: (
        '{"acknowledgement":"Long day.","mood":"drained","themes":["work"],"commitments":'
        '[{"text":"Submit lab report","due_phrase":"by Friday","due_date":"2026-10-24","importance":4,"person":null}]}'
    )
    out = brain._analyze_bedrock("I need to submit the lab report by Friday.", date(2026, 10, 20))
    assert out.commitments[0].due_date == "2026-10-23"   # Friday, not the model's Saturday


def test_short_fragment_does_not_create_commitments():
    from hearth.service import Hearth
    from hearth.store import Store
    h = Hearth(Store(":memory:"))
    out = h.log_journal_entry("Due by Friday.")
    assert out["commitments_added"] == []


def test_repeat_task_with_new_date_says_it_set_the_date():
    from hearth.service import Hearth
    from hearth.store import Store
    h = Hearth(Store(":memory:"))
    h.add_commitment("Finish lab report")
    out = h.log_journal_entry("I need to submit the lab report by Friday.")
    assert out["commitments_added"] == []
    assert "set the date" in out["speech"]


def test_talked_to_my_sister_completes_call_my_sister():
    from hearth.service import Hearth
    from hearth.store import Store
    h = Hearth(Store(":memory:"))
    h.add_commitment("Call my sister back", "tomorrow")
    h.add_commitment("Call my mom")
    r = h.complete_commitment("I just talked to my sister")
    assert r["completed"] and "sister" in r["speech"].lower()
    assert [c["text"] for c in h.get_commitments("all")["commitments"]] == ["Call my mom"]
