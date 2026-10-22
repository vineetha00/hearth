"""Intent routing: which Hearth MCP tool should handle what the person just said?

Real Alexa+ does this itself (its orchestrator reads the tool schemas our server advertises and picks one).
This module stands in for that step in the simulator:

  * With AWS credentials, Amazon Bedrock (Converse API + tool use) chooses the tool from the live MCP schemas.
  * Without them, a small rule-based router does the same job so the demo still runs offline.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

SYSTEM = (
    "You simulate the Alexa+ orchestrator for a customer who has enabled the Hearth add-on, a private end-of-day "
    "voice diary that remembers what the person says they'll do. Pick exactly one tool for the utterance. "
    "If the person is simply talking about their day, venting or reflecting, use log_journal_entry with their words "
    "verbatim. Questions about what's due or on their list use get_commitments. 'What matters most' uses "
    "prioritize_today. Finishing something uses complete_commitment. 'Remind me to...' uses add_commitment. "
    "Questions about how they've been or patterns use get_reflection. "
    "Use add_commitment ONLY when the person explicitly asks to add or be reminded of one single item (\"remind me to...\", \"add ... to my list\"). "
    "If they are narrating, even if they mention things they promised or need to do, use log_journal_entry: it saves every commitment."
)


@dataclass
class Plan:
    tool: str
    arguments: dict[str, Any]
    router: str  # "bedrock" | "rules"
    fallback_to_journal: bool = False  # if a 'completed X' guess matches nothing, treat it as diary talk


# ----------------------------------------------------------------- rule router
_LIST_Q = re.compile(
    r"\b(what|which|anything)\b.*\b(due|on my list|supposed to|said i'?d|did i (say|tell|promise)|need to do|have to do|to-?do|"
    r"on my plate|do i have|am i forgetting|coming up)\b|\bwhat'?s due\b|\bmy (list|to-?do)\b|"
    r"^\W*(?:am i )?(?:anything )?(?:due|overdue|late)\b",
    re.I,
)
_PRIORITY_Q = re.compile(
    r"most important|what should i (do|work on|tackle) (first|now|next)|where (do|should) i start|priorit|top priority|"
    r"what matters most|what('?s| is) (the )?(first|next) thing",
    re.I,
)
_REFLECT_Q = re.compile(
    r"how (have|has) (i|my)|how (about|am i|was i).{0,20}\b(been|being|doing)\b|how.{0,12}\b(been|being)\b.{0,12}\bthis week|what (patterns|have you noticed)|patterns?\b.*\b(notice|see)|how('?s| was| has been) my week|reflect",
    re.I,
)
_DONE = re.compile(
    r"^\s*(?:ok(?:ay)?,?\s+|so,?\s+)?(?:i|i've|ive|i have)\s+(?:just\s+|finally\s+)?"
    r"(finished|did|called|texted|submitted|booked|sent|done|completed|paid|emailed|cleaned|replied|turned in|handed in|made|talked to|talked with|spoke to|spoke with|phoned|video called|facetimed|got|took|picked up|cancelled|canceled|renewed|scheduled|ordered|bought|washed|ran)\b",
    re.I,
)
_ADD = re.compile(r"^\s*(?:alexa,?\s+)?(remind me|add|put|note|remember|don'?t let me forget)\b", re.I)


def _timeframe(u: str) -> str:
    if re.search(r"\boverdue|late|behind\b", u):
        return "overdue"
    if re.search(r"\btomorrow\b", u):
        return "tomorrow"
    if re.search(r"\b(today|tonight|right now)\b", u):
        return "today"
    if re.search(r"\b(this week|the week|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", u):
        return "this_week"
    return "all"


def route_rules(utterance: str) -> Plan:
    u = utterance.strip()
    low = u.lower()
    if _PRIORITY_Q.search(low):
        return Plan("prioritize_today", {}, "rules")
    if _REFLECT_Q.search(low):
        return Plan("get_reflection", {"days": 7}, "rules")
    if "?" in u or low.startswith(("what", "which", "do i", "am i", "anything")) or re.match(r"\W*(due|overdue|late)\b", low):
        if _LIST_Q.search(low):
            return Plan("get_commitments", {"timeframe": _timeframe(low)}, "rules")
    if _ADD.match(u):
        text = re.sub(r"^\s*(?:alexa,?\s+)?(?:remind me to|remind me|don'?t let me forget to|don'?t let me forget|add|put|note|remember to|remember)\s+", "", u, flags=re.I)
        text = re.sub(r"\s+(?:to|on|in)\s+my\s+(?:list|to-?do(?:\s+list)?)\.?$", "", text, flags=re.I).strip(" .")
        return Plan("add_commitment", {"text": text, "due": None}, "rules")
    if _DONE.match(u) and len(u.split()) <= 14:
        return Plan("complete_commitment", {"description": u}, "rules", fallback_to_journal=True)
    return Plan("log_journal_entry", {"text": u}, "rules")


# -------------------------------------------------------------- bedrock router
def _tool_config(tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tools": [
            {"toolSpec": {"name": t["name"], "description": t["description"], "inputSchema": {"json": t["inputSchema"]}}}
            for t in tools
        ],
        "toolChoice": {"any": {}},
    }


def route_bedrock(utterance: str, tools: list[dict[str, Any]], client: Any, model_id: str) -> Plan:
    resp = client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [{"text": utterance}]}],
        toolConfig=_tool_config(tools),
        inferenceConfig={"maxTokens": 400, "temperature": 0},
    )
    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            use = block["toolUse"]
            args = dict(use.get("input") or {})
            if use["name"] == "log_journal_entry":
                args["text"] = utterance  # never let the model paraphrase a diary entry
            args.pop("user_id", None)
            if use["name"] == "log_journal_entry" and _DONE.match(utterance) and len(utterance.split()) <= 14:
                return Plan("complete_commitment", {"description": utterance}, "bedrock", fallback_to_journal=True)
            if use["name"] == "log_journal_entry" and len(utterance.split()) <= 8 and not re.match(r"\s*i\b", utterance, re.I):
                rule = route_rules(utterance)
                if rule.tool in ("get_commitments", "prioritize_today", "get_reflection"):
                    return Plan(rule.tool, rule.arguments, "bedrock")  # a short question fragment, not a diary entry
            if use["name"] == "add_commitment" and not _ADD.match(utterance):
                return Plan("log_journal_entry", {"text": utterance}, "bedrock")  # narration, not a list request
            return Plan(use["name"], args, "bedrock", fallback_to_journal=use["name"] == "complete_commitment")
    return Plan("log_journal_entry", {"text": utterance}, "bedrock")


def make_bedrock_client() -> Any | None:
    if os.environ.get("HEARTH_LLM", "auto").lower() == "heuristic":
        return None
    try:
        import boto3

        session = boto3.Session()
        if session.get_credentials() is None:
            return None
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
        return session.client("bedrock-runtime", region_name=region)
    except Exception:  # noqa: BLE001
        return None
