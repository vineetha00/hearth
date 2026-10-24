# Hearth

**An end-of-day voice diary for Alexa+ that remembers what you said you'd do.**

Built for the *Build, Ship, Shape: Amazon Developer Hackathon 2026* (Alexa+ track, with the AWS Builder mini challenge).

## The problem

If you're in your twenties, living alone, far from family, and buried in work or classes, the day ends with nobody to
tell about it. The small promises you said out loud ("I'll call Mom back", "I need to book the dentist") evaporate by
morning, and the people you meant to call don't hear from you.

Hearth is the thing you talk to at the end of the day. It listens like a diary, and it quietly keeps track of what you said you'd do.

- **Talk freely.** Rant, reflect, ramble. Hearth acknowledges what you said, warmly and briefly.
- **It remembers your promises.** Anything you say you'll do, need to do, or promised someone becomes a to-do with a due date, pulled out of ordinary speech.
- **Ask it later.** "What's due tomorrow?" "What did I say I'd do this week?" "What's the most important thing right now?" "I just called my sister."
- **It notices patterns.** "Deadlines came up on 3 of the last 6 days you talked to me." Observations only.

**What Hearth deliberately is not:** therapy, coaching, or crisis support. It never gives advice and never offers
mental-health services. It acknowledges, remembers, and notices.

## How it works

```
 you ──voice──▶ Alexa+ (orchestrator) ──MCP, Streamable HTTP──▶ Hearth MCP server ──▶ SQLite
                                                                    │
                                                                    └──▶ Amazon Bedrock (Converse API)
                                                                         extraction · acknowledgement · reflection
```

| Piece | File | Role |
|---|---|---|
| MCP server | `hearth/server.py` | Self-hosted, **Streamable HTTP at `/mcp`**, six tools. Built on the official MCP Python SDK 2.3.0; negotiates protocol `2026-07-28` (the required floor is `2025-11-25`). |
| Behaviour | `hearth/service.py` | Transport-independent logic behind each tool. Every tool returns a `speech` field meant to be read aloud, plus structured data. |
| Language layer | `hearth/llm.py` | **Amazon Bedrock Converse API** extracts commitments (with ISO due dates resolved against today), tags mood and themes, and writes the acknowledgement and reflection. An offline heuristic engine takes over when AWS isn't reachable or a call fails. |
| Storage | `hearth/store.py` | SQLite. Near-duplicate commitments are merged ("I need to book the dentist" said twice is one item that learns its due date). |
| Prioritisation | `hearth/llm.py: prioritize` | Deterministic and explainable: overdue > due today > due soon, weighted by importance and by promises made to a person. Each answer says *why*. |
| Alexa+ simulator | `simulator/` | A web app that plays Alexa+: it takes what you say (typed or microphone), chooses a tool, calls the real MCP server as an MCP client, speaks the reply, and shows the live list and the exact tool calls. |

### MCP tools

| Tool | When Alexa+ should use it |
|---|---|
| `log_journal_entry(text)` | The person is talking about their day. Saves it, acknowledges it, auto-adds commitments. |
| `get_commitments(timeframe)` | "What's due tomorrow / this week / today?" `today` and `this_week` include overdue items. |
| `prioritize_today(limit)` | "What's the most important thing right now?" |
| `complete_commitment(description)` | "I just called my sister." Matches by meaning; asks which one if ambiguous. |
| `add_commitment(text, due)` | "Remind me to pay rent on Friday." |
| `get_reflection(days)` | "How have I been this week?" |

## Run it

Requires Python 3.12+.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Terminal 1: the MCP server
python -m hearth.server                  # http://127.0.0.1:8765/mcp   (health: /health)

# Terminal 2: the Alexa+ simulator
python -m simulator.app                  # http://127.0.0.1:8080
```

Or run both with one command: `./scripts/start.sh` (your own diary, kept in `~/.hearth/hearth.db`) or `./scripts/start.sh demo` (the seeded demo data).

Open http://127.0.0.1:8080. Use **Talk** (Chrome/Edge microphone), type, tap a suggested line, or press **Play a demo evening**.

**Bedrock:** export AWS credentials, `AWS_REGION` and `HEARTH_BEDROCK_MODEL` (default: Claude Haiku 4.5 via a `us.` inference profile; see `.env.example`). `python scripts/check_bedrock.py` verifies your setup first. Bedrock does the extraction, acknowledgements, reflections and the simulated Alexa+ tool routing; dates are resolved in code, not by the model. The server's `/health` shows
`"llm_engine": "bedrock"` when it's live. With no credentials everything still runs on the offline fallback, and the
trace panel labels which engine produced each reply.

**A week of history for the demo:** `python scripts/seed_demo.py` (fresh DB only) backfills six evenings so reflections
and overdue items have something real to show.

**Tests:** `python -m pytest -q`. They cover date parsing, extraction, de-duplication, ranking, fuzzy completion,
reflection, the Bedrock path (with a fake client), fallback on Bedrock failure, and an end-to-end run that boots the
server and drives it with an MCP client over Streamable HTTP, asserting protocol >= 2025-11-25.

## Semantic memory (FAISS) and RAG evaluation

Hearth's matching and recall were purely lexical (token overlap + `SequenceMatcher`). There is now an
embedding-backed semantic memory on top of it:

```
 journal entries + commitments ──▶ all-MiniLM-L6-v2 embeddings ──▶ FAISS IndexFlatIP (cosine)
                                                                       │  sidecar: <db>.index.npz
          complete_commitment / get_reflection(focus=…) ──search──────┘  lexical matcher = fallback
```

| Piece | File | Role |
|---|---|---|
| Vector store | `hearth/retrieval.py` | `MemoryIndex`: sentence-transformer embeddings (lazy model load, cached per process), FAISS `IndexFlatIP` over L2-normalised vectors, persisted next to the SQLite file. Degrades in order: no FAISS → numpy matmul; no sentence-transformers → the lexical matcher carries on alone, so nothing new is ever required. |
| Wiring | `hearth/service.py` | `complete_commitment` rescues paraphrases the lexical matcher misses ("I rang my mother" → "Call Mom back") and only overrides an ambiguous lexical result when the embedding match is confident; `get_reflection(focus="work")` picks the relevant evenings by meaning. |
| Eval | `eval/` | A fixed synthetic diary (60 memories, 50 annotated queries across paraphrase / temporal / distractor / direct slices) plus a runner that scores retrieval and RAGAS-style answer faithfulness. Results in `eval/results.json`. |

Measured on the eval set (recall@k averaged over gold items, k=5):

| Matcher | Slice | n | recall@1 | recall@5 | MRR |
|---|---|---|---|---|---|
| **lexical (baseline)** | overall | 50 | 0.600 | 0.800 | 0.704 |
| lexical | paraphrase | 18 | 0.333 | 0.556 | 0.438 |
| lexical | temporal | 8 | 0.375 | 0.875 | 0.667 |
| lexical | distractor | 14 | 0.857 | 1.000 | 0.917 |
| lexical | direct | 10 | 0.900 | 0.900 | 0.912 |
| **semantic (FAISS + MiniLM)** | overall | 50 | 0.910 | 0.970 | 0.958 |
| semantic | paraphrase | 18 | 0.944 | 1.000 | 0.963 |
| semantic | temporal | 8 | 0.562 | 0.812 | 0.818 |
| semantic | distractor | 14 | 1.000 | 1.000 | 1.000 |
| semantic | direct | 10 | 1.000 | 1.000 | 1.000 |

The paraphrase slice is the point: semantic recall@5 1.000 vs 0.556 lexical. The one slice where lexical
keeps an edge is temporal recall@5 (0.875 vs 0.812) — "what's due on friday" matches the literal token
"friday", which embeddings treat as near its weekday siblings; Hearth's real date questions go through
`get_commitments(timeframe)`, which resolves dates in code, so this costs nothing in the product.

Answer faithfulness is scored RAGAS-style: generate an answer from the top-5 retrieved memories with the
local generator (ollama `llama3.1:latest`, temperature 0), decompose it into atomic claims, verify each
claim against the retrieved context only, faithfulness = supported / total. The judge is deliberately a
*different, stronger* model than the generator — Claude Sonnet — because a judge that rubber-stamps
everything reports high faithfulness no matter how good the answers are. (On AgentAudit's 60-trace
attribution benchmark, llama3.1-8B as a judge scored Cohen's κ = 0.00 — it never flags anything — vs
κ = 0.92 for Sonnet; judged by llama3.1, this same eval read 0.924.)

**Measured: 0.84** over all 25 generated answers (judge `claude-sonnet-5-5`, same stored answers and
contexts re-judged). The misses are real hallucinations worth knowing about: statement-form queries
("booked the dentist") tempt the 8B generator to assert the thing was *done* when the memory only says it
was *planned*, and the judge marks that unsupported. Full transcripts (retrieved memories, answers,
per-claim verdicts) are in `eval/results.json`.

Reproduce:

```bash
pip install -r requirements-semantic.txt   # faiss-cpu + sentence-transformers (~90MB model on first use)
python -m eval.run_eval                    # retrieval metrics only
python -m eval.run_eval --faithfulness     # + RAG faithfulness (generator: ollama llama3.1;
                                           #   judge: claude CLI Sonnet — or --judge ollama)
```

## Putting it in front of real Alexa+

The server is a spec-compliant Streamable HTTP MCP endpoint, so it is what Alexa+ needs to connect to. Registering it
as an add-on goes through Amazon's tooling (the Alexa AI CLI or the Add-on Agent Skill; see the Alexa+ MCP QuickStart).
**That registration step has not been done or tested from this repo.** The simulator is the path this submission demonstrates.

To serve it publicly: `docker build -t hearth . && docker run -p 8765:8765 -v hearth-data:/data hearth` behind HTTPS.
Bind with `HEARTH_HOST=0.0.0.0` (the default localhost binding rejects other `Host` headers with HTTP 421).

## Honest limits

- **One user.** Tools take a `user_id` that defaults to `me`. A real deployment should map Alexa+ account linking to it; OAuth is not implemented.
- **Simulated orchestration.** In the simulator, tool choice is made by Bedrock tool-use (or simple rules offline). Real Alexa+ does its own routing from the tool descriptions the server advertises.
- **Offline fallback is simpler.** The regex/keyword engine handles common phrasings ("I need to…", "I told my sister I'd…", "remind me to…") but is much weaker than the model on nuance.
- **No crisis handling by design.** This is a product-scope decision (see above), not an oversight; anyone shipping this to real users should think carefully about what a diary product should do when someone writes something alarming.
- Data is stored unencrypted in a local SQLite file.

## Layout

```
hearth/       MCP server, service logic, Bedrock + fallback language layer, SQLite store, semantic memory index
simulator/    Alexa+ simulator (Starlette backend, single-file UI)
scripts/      seed_demo.py, check_bedrock.py (verify AWS access), start.sh (run everything)
tests/        unit + end-to-end tests
eval/         retrieval + RAG faithfulness eval (dataset, runner, results)
```

MIT licensed.
