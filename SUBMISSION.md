# Devpost submission: copy/paste kit

**Deadline: Friday, October 23, 2026, 12:00 PM PT.** Aim to submit Thursday night.

## Tracks and mini challenges

- **Primary track:** Alexa+ (self-hosted MCP server, Streamable HTTP, spec 2025-11-25 or later)
- **Mini challenge:** AWS Builder (Amazon Bedrock). Each project can win one track prize and one mini-challenge prize.
- Open Source mini challenge: optional. The repo is MIT licensed; it counts only if you also fill its extra fields (see the rules).

## Project description (text field)

**Hearth is an end-of-day voice diary for Alexa+ that remembers what you said you'd do.**

Many people in their twenties live alone, far from family, with work or classes consuming the day. By night there's no
one to tell about it, and the small promises said out loud ("I'll call Mom back", "I need to book the dentist")
vanish by morning. Hearth is the thing you talk to at the end of the day.

You talk freely, like a diary. Hearth acknowledges what you said in a sentence or two and quietly extracts anything you
said you'd do, with due dates resolved from phrases like "tomorrow" or "by Friday", into a to-do list. Later you can
ask "What's due tomorrow?", "What did I say I'd do this week?", "What's the most important thing right now?", or say "I
just called my sister" to cross it off. It also notices patterns across days ("deadlines came up on 3 of the last 6
days") using observations only. Hearth is deliberately not therapy, coaching, or crisis support: it never gives advice.

**How it works.** A self-hosted MCP server (official Python SDK 2.3.0, Streamable HTTP at `/mcp`, protocol 2026-07-28)
exposes six tools: `log_journal_entry`, `get_commitments`, `prioritize_today`, `complete_commitment`,
`add_commitment`, `get_reflection`. Each returns a `speech` string for Alexa+ to read aloud plus structured data.
Amazon Bedrock (Converse API) extracts commitments, tags mood and themes, and writes the acknowledgements and
reflections; an offline fallback keeps it working without AWS. Prioritisation is deterministic and explains itself
(overdue, due soon, a promise to a person). Storage is SQLite, with near-duplicate commitments merged.

**Simulated Alexa+ experience.** The repo includes a web app that plays Alexa+: it takes speech or text, chooses a tool
(Bedrock tool-use, or rules offline), calls the real MCP server as an MCP client, speaks the reply, and displays the
live list and each tool call. Registering the server as a live Alexa+ add-on was not completed for this submission.

**Built during the hackathon window:** everything. This is a new project.

## AWS Builder mini challenge (how AWS is used)

Amazon Bedrock via the Converse API, in two places: (1) `hearth/llm.py`, where a Bedrock model turns spoken entries
into structured commitments (text, ISO due date, importance, person), mood, themes and a spoken acknowledgement, and
writes pattern reflections under strict "observe, never advise" instructions; (2) `simulator/router.py`, where Bedrock
tool-use chooses which MCP tool handles an utterance, using the live tool schemas fetched from the server. Failures fall
back to local logic per call, and the UI labels which engine answered.

## Product feedback (required): starting points

Fill these in from your own experience; keep what's true.

- **MCP Python SDK 2.3.0:** see FRICTION_LOG.md entries 1 to 3 (rename, structured output, Host header).
- **Amazon Bedrock Converse API:** what you used it for, which model, latency/cost surprises, whether you'd use it again.
- **Alexa+ MCP Toolkit / Alexa AI CLI / Add-on Agent Skill:** write this after you've run the real onboarding (FRICTION_LOG entry 4).
- **Optional feature request:** e.g. an official local Alexa+ simulator that exercises an MCP server without a device (importance: important).

## Pre-submit checklist

- [ ] Repo pushed to GitHub, with the MIT `LICENSE` at the top level (public), **or** private and shared with `testing@devpost.com` plus `knmeiss`, `giolaq`, `anishamalde`, `mosesroth`, `emersonsklar`. Invitations must be accepted by each reviewer; add them when you submit.
- [ ] README runs from a clean clone (try it once in a fresh venv).
- [ ] Demo video under 3:00, public on YouTube or Vimeo, English, nothing copyrighted.
- [ ] Friction log entry 4 filled in (or removed), then pasted for the bonus.
- [ ] Track (Alexa+) and mini challenge (AWS Builder) selected.
- [ ] Read the official rules on the Devpost page once more; this kit was assembled from the rules text you shared.
