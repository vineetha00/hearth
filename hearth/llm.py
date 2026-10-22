"""Language layer: Amazon Bedrock (Converse API) with an offline heuristic fallback.

Three jobs:
  * analyze_entry  - acknowledge a diary entry, tag mood/themes, pull out commitments
  * reflect        - notice patterns across recent entries (observations only, never advice)
  * (ranking is deterministic and lives in ``prioritize``)

HEARTH_LLM = auto | bedrock | heuristic
  auto      use Bedrock when AWS credentials resolve, otherwise the heuristic fallback
  bedrock   always Bedrock (errors surface in the response's ``engine`` field and fall back per call)
  heuristic never call AWS (tests, offline demos)
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from . import dates

log = logging.getLogger("hearth.llm")

DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

GUARDRAILS = (
    "You are the voice of Hearth, a private end-of-day diary companion for busy people who live alone. "
    "Acknowledge what the person said warmly and briefly, in plain spoken language (this is read aloud by a "
    "voice assistant). Never give advice, diagnoses, coping techniques, or therapy. Never offer crisis or "
    "mental-health services. Never tell the person how they should feel. Only reflect what they actually said."
)

THEME_WORDS: dict[str, list[str]] = {
    "work": ["work", "boss", "manager", "meeting", "meetings", "standup", "client", "shift", "office", "project", "deploy"],
    "school": ["class", "classes", "exam", "exams", "midterm", "final", "homework", "assignment", "lecture", "professor", "thesis", "lab", "study", "studying"],
    "deadlines": ["deadline", "deadlines", "due", "submit", "submission", "behind"],
    "family": ["mom", "mother", "dad", "father", "sister", "brother", "family", "parents", "grandma", "grandpa", "home"],
    "friends": ["friend", "friends", "roommate", "buddy"],
    "sleep & energy": ["tired", "exhausted", "sleep", "slept", "drained", "burnt", "burned", "insomnia", "awake"],
    "feeling rushed": ["rushed", "hurry", "hurried", "no time", "swamped", "overwhelmed", "packed", "slammed", "busy"],
    "feeling alone": ["lonely", "alone", "miss", "missing", "homesick", "quiet"],
    "food": ["ate", "lunch", "dinner", "breakfast", "skipped", "cooked", "cook", "groceries"],
    "money": ["rent", "money", "bills", "budget", "paycheck", "broke"],
}
NEG_WORDS = {"tired", "exhausted", "drained", "stressed", "anxious", "overwhelmed", "rough", "awful", "terrible",
             "bad", "hard", "sad", "lonely", "alone", "miss", "homesick", "behind", "rushed", "swamped", "worried",
             "frustrated", "annoyed", "slammed", "burnt", "burned"}
POS_WORDS = {"good", "great", "happy", "proud", "excited", "fun", "relaxed", "nice", "awesome", "finished",
             "done", "won", "love", "glad", "better", "calm"}

# --- commitment extraction (heuristic path) ---------------------------------
_ADV = r"(?:(?:also|really|still|just|definitely|totally|absolutely)\s+)*"
_TRIGGER = re.compile(
    r"(?:\bi\s+" + _ADV + r"(?:need to|have to|got to|gotta|must|should|ought to)\b"
    r"|\bi(?:'m| am)\s+" + _ADV + r"(?:going|supposed)\s+to\b"
    r"|\bi\s+(?:promised|told|said)\b[^.!?]*?\b(?:i(?:'d| would|'ll| will)|that i(?:'d| would))\b"
    r"|\bremind me to\b|\bdon'?t let me forget to\b|\bdon'?t forget to\b|\bi forgot to\b"
    r"|\bi(?:'ll| will)\b(?!\s+(?:be|probably not|never|try))"
    r"|\bgotta\b)",
    re.IGNORECASE,
)
_RELATIONS = {"mom", "mother", "dad", "father", "sister", "brother", "grandma", "grandpa", "friend",
              "roommate", "aunt", "uncle", "cousin"}
_HEDGE = re.compile(r"\b(maybe|eventually|sometime|someday|one day|probably should)\b", re.I)
_URGENT = re.compile(r"\b(deadline|exam|midterm|final|interview|submit|submission|rent|bill|appointment|doctor|dentist|flight|due|presentation|application)\b", re.I)
_REL = r"mom|mother|dad|father|sister|brother|grandma|grandpa|friend|roommate|aunt|uncle|cousin"
_PEOPLE = re.compile(
    r"\b(?i:call|text|email|message|ping|visit|see|meet|reply to|write to|facetime|phone|told|promised|tell)\s+"
    r"(?i:back\s+)?(?i:my\s+)?"
    r"((?i:" + _REL + r")|[A-Z][a-z]+)\b"
)


@dataclass
class Commitment:
    text: str
    due_date: str | None = None  # ISO yyyy-mm-dd
    importance: int = 3  # 1 (someday) .. 5 (can't miss)
    person: str | None = None


@dataclass
class Analysis:
    acknowledgement: str
    mood: str
    themes: list[str] = field(default_factory=list)
    commitments: list[Commitment] = field(default_factory=list)
    engine: str = "heuristic"


_SPLIT = re.compile(
    r"(?<=[.!?])\s+|\n+|;\s+"
    r"|,?\s+(?:and|then|plus|also)\s+(?=i(?:'|\s)|remind me|don'?t forget)"  # "...tomorrow and I need to..."
    r"|,\s+(?=i(?:'|\s)(?:need|have|should|must|gotta|promised|told|said|will|ll|forgot))",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    return [p.strip() for p in _SPLIT.split(text.strip()) if p and p.strip()]


def _clause_end(s: str) -> str:
    return re.split(r"\b(?:but|because|since|although|though)\b|,\s*(?:and|but|so)\b", s, maxsplit=1)[0]


def heuristic_commitments(text: str, today_: date) -> list[Commitment]:
    found: list[Commitment] = []
    for sent in _sentences(text):
        for m in _TRIGGER.finditer(sent):
            tail = sent[m.end():].strip(" ,")
            if not tail:
                continue
            # "I told my sister I'd call her back" -> keep the verb phrase after the trigger
            task = _clause_end(tail)
            if len(task.split()) < 2:
                continue
            due = dates.parse_due(sent, today_)
            task_clean = dates.strip_due_words(task)
            if len(task_clean.split()) < 2:
                continue
            importance = 3
            if _URGENT.search(sent):
                importance += 1
            person = None
            pm = _PEOPLE.search(sent) or _PEOPLE.search(task_clean)
            if pm:
                raw = pm.group(1)
                person = raw.lower() if raw.lower() in _RELATIONS else raw
                importance += 1 if re.search(r"\b(promised|told|said)\b", sent, re.I) else 0
                # "I told my sister I'd call her back" -> "Call my sister back" (don't lose who "her" is)
                who = f"my {person}" if person in _RELATIONS else person
                task_clean = re.sub(r"\b(her|him|them)\b", who, task_clean, count=1, flags=re.I)
            if _HEDGE.search(sent):
                importance -= 1
            importance = max(1, min(5, importance))
            task_clean = re.sub(r"^(?:to\s+)", "", task_clean)
            # Normalize "call her back" style trailing pronouns later; keep wording close to what was said.
            found.append(Commitment(text=task_clean[0].upper() + task_clean[1:], due_date=due.isoformat() if due else None,
                                    importance=importance, person=person))
            break  # one commitment per sentence
    return found


def heuristic_analysis(text: str, today_: date) -> Analysis:
    words = set(re.findall(r"[a-z']+", text.lower()))
    low = text.lower()
    themes = [name for name, kws in THEME_WORDS.items() if any((k in words) if " " not in k else (k in low) for k in kws)]
    neg, pos = len(words & NEG_WORDS), len(words & POS_WORDS)
    if neg >= 2 and pos >= 1:
        mood = "mixed"
    elif neg >= 2 or ("exhausted" in words or "drained" in words):
        mood = "drained" if ({"tired", "exhausted", "drained", "burnt", "burned"} & words) else "stressed"
    elif neg == 1 and pos == 0:
        mood = "a bit heavy"
    elif pos >= 1 and neg == 0:
        mood = "good"
    else:
        mood = "okay"
    commitments = heuristic_commitments(text, today_)

    opener = {
        "drained": "That sounds like a long, tiring day.",
        "stressed": "That's a lot to be carrying around today.",
        "mixed": "Sounds like a day with some good and some hard in it.",
        "a bit heavy": "Thanks for telling me about it. That sounded like a heavy one.",
        "good": "I'm glad you got a good day to tell me about.",
        "okay": "Got it. Thanks for talking it through with me.",
    }[mood]
    if "feeling alone" in themes and mood != "good":
        opener += " It's hard being far from the people you'd usually tell."
    return Analysis(opener, mood, themes, commitments, engine="heuristic")


def heuristic_reflection(entries: list[dict], done: int, open_items: list[dict], days: int, today_: date) -> str:
    if not entries:
        return "I don't have any entries from the last few days yet. Tell me about your day and I'll start noticing patterns."
    # Patterns are about days, not individual entries: several entries in one evening count once.
    by_day: dict[str, list[dict]] = {}
    for e in entries:
        by_day.setdefault(e["created_at"][:10], []).append(e)
    n = len(by_day)
    theme_days: Counter[str] = Counter()
    heavy_days = good_days = 0
    for day_entries in by_day.values():
        for t in {t for e in day_entries for t in e["themes"]}:
            theme_days[t] += 1
        moods = {e.get("mood") for e in day_entries}
        heavy_days += bool(moods & {"drained", "stressed", "a bit heavy"})
        good_days += bool("good" in moods and not (moods & {"drained", "stressed", "a bit heavy"}))
    bits: list[str] = []
    if n >= 2:
        for t, c in [(t, c) for t, c in theme_days.most_common() if c >= 2][:2]:
            bits.append(f"{t} came up on {c} of the last {n} days you talked to me")
        if heavy_days >= 2:
            bits.append(f"{heavy_days} of those {n} days sounded heavy")
        if good_days >= 2:
            bits.append(f"{good_days} of those days sounded good")
    overdue = [c for c in open_items if c["due_date"] and date.fromisoformat(c["due_date"]) < today_]
    if done:
        bits.append(f"you've checked off {done} thing{'s' if done != 1 else ''} in the last {days} days")
    if overdue:
        bits.append(f"{len(overdue)} thing{'s are' if len(overdue) != 1 else ' is'} still waiting past {'their dates' if len(overdue) != 1 else 'its date'}")
    if not bits:
        if n == 1:
            return "So far I've only heard from you on one day, so there's no pattern yet. Keep talking and I'll start to notice them."
        return f"I've heard from you on {n} days in the last {days}. Nothing is repeating yet, but I'm listening."
    return "Here's what I've noticed: " + "; ".join(bits) + "."


# --- ranking (deterministic, explainable) -----------------------------------
def score(c: dict, today_: date) -> tuple[float, str]:
    s = c["importance"] * 10.0
    reason = "it's marked important" if c["importance"] >= 4 else "it's on your list"
    due = date.fromisoformat(c["due_date"]) if c["due_date"] else None
    if due is not None:
        delta = (due - today_).days
        if delta < 0:
            s += 40 + min(10, -delta * 2)
            reason = f"it's {dates.describe(due, today_)}"
        elif delta == 0:
            s += 35
            reason = "it's due today"
        elif delta == 1:
            s += 25
            reason = "it's due tomorrow"
        elif delta <= 3:
            s += 15
            reason = f"it's due {dates.describe(due, today_)}"
        elif delta <= 7:
            s += 8
            reason = f"it's due {dates.describe(due, today_)}"
    if c.get("person"):
        s += 6
        if reason == "it's on your list" or reason == "it's marked important":
            reason = f"you made a promise to {c['person']}"
    return s, reason


def prioritize(open_items: list[dict], today_: date, limit: int = 3) -> list[dict]:
    ranked = []
    for c in open_items:
        s, why = score(c, today_)
        ranked.append({**c, "score": round(s, 1), "why": why})
    ranked.sort(key=lambda c: -c["score"])
    return ranked[:limit]


# --- Bedrock ------------------------------------------------------------------
def _strip_task_sentences(ack: str, commitments: list) -> str:
    """Drop acknowledgement sentences that merely restate a task; the app reads those back itself."""
    if not ack or not commitments:
        return ack
    stop = {"the", "a", "an", "to", "my", "your", "you", "i", "and", "of", "for", "on", "in", "it", "that", "with", "up", "have", "has", "got", "need", "call", "due"}
    task_words = [set(re.findall(r"[a-z']+", c.text.lower())) - stop for c in commitments]
    keep = []
    for sent in re.findall(r"[^.!?]+[.!?]*", ack):
        words = set(re.findall(r"[a-z']+", sent.lower())) - stop
        if any(len(words & tw) >= 2 or (tw and tw <= words) for tw in task_words):
            continue
        keep.append(sent.strip())
    return " ".join(keep).strip()


class Brain:
    def __init__(self, mode: str | None = None, client: Any | None = None, model_id: str | None = None):
        self.mode = (mode or os.environ.get("HEARTH_LLM", "auto")).lower()
        self.model_id = model_id or os.environ.get("HEARTH_BEDROCK_MODEL", DEFAULT_MODEL)
        self._client = client
        self._bedrock_ok: bool | None = None if client is None else True

    # lazily create the client so import never needs AWS
    def _bedrock(self):
        if self._client is not None:
            return self._client
        if self.mode == "heuristic":
            return None
        if self._bedrock_ok is False:
            return None
        try:
            import boto3

            session = boto3.Session()
            if session.get_credentials() is None:
                raise RuntimeError("no AWS credentials")
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
            self._client = session.client("bedrock-runtime", region_name=region)
            self._bedrock_ok = True
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            log.warning("Bedrock unavailable (%s); using heuristic engine", exc)
            self._bedrock_ok = False
            self._client = None
        return self._client

    @property
    def engine(self) -> str:
        return "bedrock" if self._bedrock() is not None else "heuristic"

    def _converse(self, system: str, user: str, max_tokens: int = 900) -> str:
        client = self._bedrock()
        resp = client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2},
        )
        return "".join(b.get("text", "") for b in resp["output"]["message"]["content"])

    @staticmethod
    def _json(raw: str) -> dict:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError("no JSON object in model output")
        return json.loads(m.group(0))

    # -- public API -------------------------------------------------------
    def analyze_entry(self, text: str, today_: date) -> Analysis:
        if self._bedrock() is not None:
            try:
                return self._analyze_bedrock(text, today_)
            except Exception as exc:  # noqa: BLE001
                log.warning("Bedrock analyze failed (%s); falling back", exc)
        return heuristic_analysis(text, today_)

    def _analyze_bedrock(self, text: str, today_: date) -> Analysis:
        system = GUARDRAILS + (
            " Reply with ONE JSON object only, no prose, with keys: "
            '"acknowledgement" (1-2 spoken sentences reflecting what they said; about their feelings and what happened; NEVER restate or list tasks, the app reads those back. If the entry is only tasks, just say \"Got it.\"), '
            '"mood" (one of: good, okay, mixed, drained, stressed, a bit heavy), '
            '"themes" (array of up to 4 short lowercase topics they actually mentioned), '
            '"commitments" (array of things the PERSON said they will do, need to do, or promised someone; each '
            'an object with "text" (short imperative, e.g. "Call my sister back"), "due_phrase" (the exact words they used for '
            'the deadline, like "by Friday" or "tomorrow", or null), "due_date" (ISO yyyy-mm-dd or null), "importance" (1-5; 5 = hard deadline or promise to a '
            'person that can\'t slip), "person" (who it involves, or null)). '
            "Do NOT invent commitments. Only include something if the person said they will, need to, have to, should, or promised to do it. Wishes, complaints and descriptions of a situation (\"I'm behind on X\", \"X is looming\", \"I miss home\") are NOT commitments. If the same task is mentioned with different verbs, list it once."
        )
        user = f"Today is {today_.strftime('%A, %Y-%m-%d')}.\n\nDiary entry:\n{text}"
        data = self._json(self._converse(system, user))
        commitments = []
        for c in data.get("commitments", []) or []:
            due = c.get("due_date")
            try:
                due = date.fromisoformat(due).isoformat() if due else None
            except ValueError:
                due = None
            # The model is unreliable at weekday arithmetic, so resolve the person's own words in code.
            phrase = (c.get("due_phrase") or "").strip()
            if phrase:
                parsed = dates.parse_due(phrase, today_)
                if parsed:
                    due = parsed.isoformat()
            t = (c.get("text") or "").strip()
            if len(t.split()) < 2:
                continue
            commitments.append(Commitment(t, due, max(1, min(5, int(c.get("importance") or 3))), c.get("person") or None))
        ack = _strip_task_sentences((data.get("acknowledgement") or "").strip(), commitments)
        return Analysis(
            ack or ("Got it." if commitments else "Thanks for telling me about your day."),
            data.get("mood") or "okay",
            [str(t).lower() for t in (data.get("themes") or [])][:4],
            commitments,
            engine="bedrock",
        )

    def reflect(self, entries: list[dict], done: int, open_items: list[dict], days: int, today_: date) -> tuple[str, str]:
        """Returns (speech, engine)."""
        fallback = heuristic_reflection(entries, done, open_items, days, today_)
        if not entries or self._bedrock() is None:
            return fallback, "heuristic"
        try:
            system = GUARDRAILS + (
                " Write 2-3 spoken sentences noticing PATTERNS across these diary entries (recurring topics, how many "
                "evenings sounded heavy or good, tasks piling up). Observations only: no advice, no suggestions, no "
                "'you should', no diagnosis. Use counts where you can. Count distinct days, not separate entries. At most 45 words in total. Do not describe how they must feel or comment on their load (no 'you're carrying a lot'); just state what recurs. Plain text only."
            )
            lines = [f"- {e['created_at'][:10]} [{e.get('mood')}] themes={','.join(e['themes'])}: {e['text'][:300]}" for e in reversed(entries)]
            user = (
                f"Today is {today_.isoformat()}. Entries from the last {days} days:\n" + "\n".join(lines)
                + f"\n\nCommitments completed in that period: {done}. The ONLY things still open are: "
                + ("; ".join(c["text"] for c in open_items) or "nothing")
                + ". Anything mentioned in an entry that is not in that list was finished or dropped, so never call it pending."
            )
            out = self._converse(system, user, max_tokens=300).strip()
            return (out or fallback), ("bedrock" if out else "heuristic")
        except Exception as exc:  # noqa: BLE001
            log.warning("Bedrock reflect failed (%s); falling back", exc)
            return fallback, "heuristic"
