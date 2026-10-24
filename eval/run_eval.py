"""Retrieval + RAG eval for Hearth's semantic memory.

Two halves:

1. Retrieval: recall@1, recall@5 and MRR over eval/dataset.py, side by side for
   (a) the lexical matcher Hearth shipped with (token overlap + SequenceMatcher)
   (b) the FAISS/MiniLM semantic index in hearth/retrieval.py
   Reported overall and per slice (paraphrase / temporal / distractor / direct).

2. Answer faithfulness, RAGAS-style, on 25 of the queries: generate an answer
   from the top-k retrieved memories with a local LLM (ollama), decompose the
   answer into atomic claims, verify each claim against the retrieved context,
   and score faithfulness = supported claims / total claims.

   The GENERATOR (the RAG system under test) and the JUDGE (decompose + verify)
   are separate. The judge defaults to Claude Sonnet through the `claude` CLI:
   on AgentAudit's 60-trace attribution benchmark the same llama3.1-8B that
   generates answers scored kappa = 0.00 as a judge (it never flags anything),
   while Sonnet scored kappa = 0.92 — a judge that always says "supported"
   reports high faithfulness no matter how good the answers are, so the
   faithfulness number is only as trustworthy as the judge behind it.

Run (from the repo root, with the venv that has faiss + sentence-transformers):

    python -m eval.run_eval                 # retrieval only
    python -m eval.run_eval --faithfulness                    # judge: claude-cli (Sonnet)
    python -m eval.run_eval --faithfulness --judge ollama     # judge: local llama3.1
    python -m eval.run_eval --faithfulness --reuse-answers eval/results.json
                                            # re-judge stored answers, same contexts

Writes eval/results.json and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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


# --- faithfulness (RAGAS-style; generator and judge are separate models) -----
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


JUDGE_BACKEND = os.environ.get("HEARTH_EVAL_JUDGE", "claude-cli")
JUDGE_MODEL = os.environ.get("HEARTH_EVAL_JUDGE_MODEL", "")


def claude_cli_chat(system: str, user: str, model: str) -> str:
    out = subprocess.run(
        ["claude", "-p", f"{system}\n\n{user}", "--model", model],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if out.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {out.stderr[:200]}")
    return out.stdout.strip()


def judge_chat(system: str, user: str, max_tokens: int = 400, json_mode: bool = False) -> str:
    """The judge's LLM call: Claude via the CLI by default, or the local ollama model."""
    if JUDGE_BACKEND == "claude-cli":
        return claude_cli_chat(system, user, JUDGE_MODEL or "claude-sonnet-5-5")
    global OLLAMA_MODEL
    prev, OLLAMA_MODEL = OLLAMA_MODEL, (JUDGE_MODEL or OLLAMA_MODEL)
    try:
        return ollama_chat(system, user, max_tokens=max_tokens, json_mode=json_mode)
    finally:
        OLLAMA_MODEL = prev


def judge_name() -> str:
    if JUDGE_BACKEND == "claude-cli":
        return JUDGE_MODEL or "claude-sonnet-5-5"
    return JUDGE_MODEL or OLLAMA_MODEL


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
    raw = judge_chat(system, f"Question: {question}\nAnswer: {answer}")
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
    raw = judge_chat(system, f"Memories:\n{mem}\n\nClaim: {claim}", max_tokens=150, json_mode=True)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return bool(json.loads(m.group(0) if m else raw).get("supported") is True)
    except (json.JSONDecodeError, AttributeError):
        return False


def run_faithfulness(index: MemoryIndex, corpus: list[dict], queries: list[dict],
                     reuse: dict[str, dict] | None = None) -> dict:
    """reuse: query_id -> stored transcript; re-judges those exact answers and
    retrieved contexts instead of regenerating, so two judges score the same answers."""
    by_id = {q["id"]: q for q in queries}
    picked = [by_id[qid] for qid in FAITHFULNESS_QUERY_IDS]

    def judge_one(q: dict) -> dict:
        if reuse and q["id"] in reuse:
            context = reuse[q["id"]]["retrieved"]
            answer = reuse[q["id"]]["answer"]
        else:
            context = [text for _, _, text in index.search(q["query"], k=TOP_K)]
            answer = generate_answer(q["query"], context)
        claims = decompose_claims(q["query"], answer)
        verdicts = [{"claim": c, "supported": verify_claim(c, context)} for c in claims]
        score = sum(v["supported"] for v in verdicts) / len(verdicts) if verdicts else None
        return {
            "query_id": q["id"], "slice": q["slice"], "question": q["query"],
            "retrieved": context, "answer": answer, "claims": verdicts,
            "faithfulness": None if score is None else round(score, 3),
        }

    workers = 5 if JUDGE_BACKEND == "claude-cli" else 1
    transcripts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, t in enumerate(pool.map(judge_one, picked), 1):
            transcripts.append(t)
            s = t["faithfulness"]
            print(f"  [{i:2}/{len(picked)}] {t['query_id']:<4} faithfulness={'n/a' if s is None else f'{s:.2f}'}  ({t['question'][:50]})")
    scores = [t["faithfulness"] for t in transcripts if t["faithfulness"] is not None]
    return {
        "generator_model": OLLAMA_MODEL,
        "judge_model": judge_name(),
        "answers_reused": bool(reuse),
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
    global JUDGE_BACKEND, JUDGE_MODEL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--faithfulness", action="store_true", help="also run the RAG faithfulness eval")
    ap.add_argument("--judge", choices=["claude-cli", "ollama"], default=JUDGE_BACKEND,
                    help="faithfulness judge backend (default claude-cli: Claude Sonnet)")
    ap.add_argument("--judge-model", default=JUDGE_MODEL,
                    help="judge model (default claude-sonnet-5-5 for claude-cli, the generator model for ollama)")
    ap.add_argument("--reuse-answers", default=None, metavar="RESULTS_JSON",
                    help="re-judge the answers and contexts stored in a previous eval/results.json")
    args = ap.parse_args()
    JUDGE_BACKEND, JUDGE_MODEL = args.judge, args.judge_model

    reuse = None
    if args.reuse_answers:
        prev = json.load(open(args.reuse_answers))
        reuse = {t["query_id"]: t for t in prev.get("rag_faithfulness", {}).get("transcripts", [])}
        if not reuse:
            raise SystemExit(f"no transcripts found in {args.reuse_answers}")

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
        print(f"Faithfulness eval: generator={OLLAMA_MODEL}, judge={judge_name()}"
              f"{' (re-judging stored answers)' if reuse else ''}...")
        results["rag_faithfulness"] = run_faithfulness(index, corpus, queries, reuse=reuse)

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
