"""Retrieval + RAG eval for Hearth's semantic memory.

Two halves:

1. Retrieval: recall@1, recall@5 and MRR over eval/dataset.py, side by side for
   (a) the lexical matcher Hearth shipped with (token overlap + SequenceMatcher)
   (b) the FAISS/MiniLM semantic index in hearth/retrieval.py
   Reported overall and per slice (paraphrase / temporal / distractor / direct).

2. Answer faithfulness, RAGAS-style, on 25 of the queries: generate an answer
   from the top-k retrieved memories with a local LLM (ollama), decompose the
   answer into atomic claims with the same LLM, verify each claim against the
   retrieved context, and score faithfulness = supported claims / total claims.
   Direct ollama HTTP calls; no extra eval frameworks.

Run (from the repo root, with the venv that has faiss + sentence-transformers):

    python -m eval.run_eval                 # retrieval only
    python -m eval.run_eval --faithfulness  # retrieval + RAG faithfulness (needs ollama)

Writes eval/results.json and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.dataset import FAITHFULNESS_QUERY_IDS, load  # noqa: E402
from hearth.retrieval import MemoryIndex  # noqa: E402
from hearth.store import similarity  # noqa: E402

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("HEARTH_EVAL_MODEL", "llama3.1:latest")
TOP_K = 5


# --- retrieval ---------------------------------------------------------------
def rank_lexical(corpus: list[dict], query: str) -> list[str]:
    scored = sorted(((similarity(c["text"], query), c["id"]) for c in corpus), key=lambda x: (-x[0], x[1]))
    return [cid for _, cid in scored]


def build_semantic(corpus: list[dict]) -> MemoryIndex:
    index = MemoryIndex()
    ok = index.add_many([(c["id"], c["text"]) for c in corpus])
    if not ok:
        raise SystemExit("sentence-transformers is not available; install the eval extras first")
    return index


def rank_semantic(index: MemoryIndex, query: str) -> list[str]:
    return [key for _, key, _ in index.search(query, k=len(index))]


def metrics(rankings: dict[str, list[str]], queries: list[dict]) -> dict:
    """recall@1 / recall@5 (|gold in top-k| / |gold|, averaged) and MRR, overall + per slice."""
    per_slice: dict[str, list[dict]] = defaultdict(list)
    rows = []
    for q in queries:
        ranked, gold = rankings[q["id"]], set(q["gold"])
        r1 = len(gold & set(ranked[:1])) / len(gold)
        r5 = len(gold & set(ranked[:TOP_K])) / len(gold)
        rr = next((1.0 / (i + 1) for i, cid in enumerate(ranked) if cid in gold), 0.0)
        row = {"recall@1": r1, "recall@5": r5, "mrr": rr}
        rows.append(row)
        per_slice[q["slice"]].append(row)

    def avg(rs: list[dict]) -> dict:
        n = len(rs)
        return {k: round(sum(r[k] for r in rs) / n, 3) for k in ("recall@1", "recall@5", "mrr")} | {"n": n}

    return {"overall": avg(rows), "slices": {s: avg(rs) for s, rs in sorted(per_slice.items())}}


# --- faithfulness (RAGAS-style, judged by a local LLM over ollama) -----------
def ollama_chat(system: str, user: str, max_tokens: int = 400, json_mode: bool = False) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0, "num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"
    resp = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def generate_answer(question: str, context: list[str]) -> str:
    system = (
        "You are Hearth, a private voice diary that remembers what the person said. "
        "The memories below are your records: open to-do items and past diary entries. "
        "Answer the person in 1-3 short spoken sentences using ONLY those memories. "
        "An item on the to-do list is something they PLANNED, not something they did; never say something "
        "happened or was done unless a memory says so. If the memories don't answer it, say you don't have that. "
        "Never invent details."
    )
    mem = "\n".join(f"- {t}" for t in context)
    return ollama_chat(system, f"Memories:\n{mem}\n\nQuestion: {question}\n\nAnswer:")


def decompose_claims(question: str, answer: str) -> list[str]:
    system = (
        "Break the given answer into its atomic factual claims. Each claim must be ONE complete standalone "
        "declarative sentence (subject and verb) that makes sense on its own, never a fragment. "
        "Ignore pure filler like greetings or 'I don't have that'. "
        'Reply with a JSON array of strings only, e.g. ["claim one", "claim two"]. No other text.'
    )
    raw = ollama_chat(system, f"Question: {question}\nAnswer: {answer}")
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        claims = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(c).strip() for c in claims if str(c).strip()]


def verify_claim(claim: str, context: list[str]) -> bool:
    system = (
        "You check whether a claim is supported by a set of personal memories (to-do items and diary entries). "
        "The claim is supported if some memory states it outright or in a close paraphrase; wording may differ. "
        "A to-do item supports claims about what the person plans or needs to do. A claim that something was "
        "ALREADY DONE or happened is NOT supported by a to-do item, only by a memory saying it happened. "
        'Reply with JSON only: {"evidence": "the memory that supports the claim, or null", "supported": true or false}.'
    )
    mem = "\n".join(f"- {t}" for t in context)
    raw = ollama_chat(system, f"Memories:\n{mem}\n\nClaim: {claim}", max_tokens=150, json_mode=True)
    try:
        return bool(json.loads(raw).get("supported") is True)
    except json.JSONDecodeError:
        return False


def run_faithfulness(index: MemoryIndex, corpus: list[dict], queries: list[dict]) -> dict:
    by_id = {q["id"]: q for q in queries}
    picked = [by_id[qid] for qid in FAITHFULNESS_QUERY_IDS]
    transcripts, scores = [], []
    for i, q in enumerate(picked, 1):
        context = [text for _, _, text in index.search(q["query"], k=TOP_K)]
        answer = generate_answer(q["query"], context)
        claims = decompose_claims(q["query"], answer)
        verdicts = [{"claim": c, "supported": verify_claim(c, context)} for c in claims]
        supported = sum(v["supported"] for v in verdicts)
        score = supported / len(verdicts) if verdicts else None
        if score is not None:
            scores.append(score)
        transcripts.append({
            "query_id": q["id"], "slice": q["slice"], "question": q["query"],
            "retrieved": context, "answer": answer, "claims": verdicts,
            "faithfulness": None if score is None else round(score, 3),
        })
        print(f"  [{i:2}/{len(picked)}] {q['id']:<4} faithfulness={'n/a' if score is None else f'{score:.2f}'}  ({q['query'][:50]})")
    return {
        "judge_model": OLLAMA_MODEL,
        "n_queries": len(picked),
        "n_scored": len(scores),
        "faithfulness": round(sum(scores) / len(scores), 3) if scores else None,
        "transcripts": transcripts,
    }


# --- report ------------------------------------------------------------------
def markdown_table(lex: dict, sem: dict) -> str:
    lines = [
        "| Matcher | Slice | n | recall@1 | recall@5 | MRR |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in (("lexical (baseline)", lex), ("semantic (FAISS + MiniLM)", sem)):
        o = m["overall"]
        lines.append(f"| **{name}** | overall | {o['n']} | {o['recall@1']:.3f} | {o['recall@5']:.3f} | {o['mrr']:.3f} |")
        for s, v in m["slices"].items():
            lines.append(f"| {name} | {s} | {v['n']} | {v['recall@1']:.3f} | {v['recall@5']:.3f} | {v['mrr']:.3f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--faithfulness", action="store_true", help="also run the ollama RAG faithfulness eval")
    args = ap.parse_args()

    corpus, queries = load()
    print(f"Corpus: {len(corpus)} memories; queries: {len(queries)}")

    print("Ranking (lexical baseline)...")
    lex_rankings = {q["id"]: rank_lexical(corpus, q["query"]) for q in queries}
    print("Ranking (semantic index)...")
    index = build_semantic(corpus)
    sem_rankings = {q["id"]: rank_semantic(index, q["query"]) for q in queries}

    lex, sem = metrics(lex_rankings, queries), metrics(sem_rankings, queries)
    results = {"dataset": {"corpus": len(corpus), "queries": len(queries), "top_k": TOP_K},
               "retrieval": {"lexical": lex, "semantic": sem}}

    if args.faithfulness:
        print(f"Faithfulness eval over ollama ({OLLAMA_MODEL})...")
        results["rag_faithfulness"] = run_faithfulness(index, corpus, queries)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}\n")
    print(markdown_table(lex, sem))
    if args.faithfulness:
        ff = results["rag_faithfulness"]
        print(f"\nRAG answer faithfulness ({ff['judge_model']}, top-{TOP_K} semantic retrieval): "
              f"**{ff['faithfulness']}** over {ff['n_scored']}/{ff['n_queries']} answers with scorable claims.")


if __name__ == "__main__":
    main()
