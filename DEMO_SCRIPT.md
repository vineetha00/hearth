# Demo video script (target 2:40, hard limit 3:00)

Judges may stop at 3:00, so the best material goes first. Record the simulator full-screen with the **What Alexa+ called**
panel open. Run `python scripts/seed_demo.py` first so the list already has history, then start both servers.
Use a voice you like: tap **Talk** and say the lines, or click the suggestion chips. Public YouTube or Vimeo, English, no music you don't own.

| Time | On screen | Say |
|---|---|---|
| 0:00 | Lamp dim, empty thread, list showing a few older items (one **Overdue**) | "It's 9pm. I live alone, I'm buried in work, and the only person I talk to tonight is my speaker. Hearth is an Alexa+ add-on for exactly that moment." |
| 0:12 | Tap **Talk**, say the rant: *"Long day. Back to back meetings, I skipped lunch, and I'm completely drained. I told my sister I'd call her back tomorrow, and I need to submit the lab report by Friday. Honestly I miss being home."* Stop. | (let it play; lamp glows while listening) |
| 0:40 | Reply appears; two new rows show up under **Tomorrow** and **Later this week** | "It didn't give me advice. It acknowledged me, and it caught the two things I promised, with dates, out of one ramble." |
| 0:55 | Say or click: *"What's due tomorrow?"* then *"What's the most important thing right now?"* | "Ask it later and it answers from my own words, and it tells me why: my mom's call is three days overdue, and I hadn't even noticed." (Point at the *because*.) |
| 1:20 | Say: *"I just called my sister."* Row disappears; "You have N left" | "I never said the exact wording. It matched by meaning and crossed it off." |
| 1:35 | Say: *"How have I been this week?"* | "And it notices patterns across days: deadlines and school keep coming up, most of the week sounded heavy. Observations, never advice. Hearth isn't therapy and says so." |
| 1:55 | Zoom on **What Alexa+ called**: scroll through `tools/call log_journal_entry`, `get_commitments`, `prioritize_today` | "Under the hood, every reply is a real tool call to my MCP server over Streamable HTTP, protocol 2026-07-28. Six tools. Alexa+ discovers them from their descriptions and picks one from what I say." |
| 2:15 | Cut to terminal showing `/health` → `"llm_engine": "bedrock"` and the `pytest` run passing | "Amazon Bedrock pulls the commitments and due dates out of natural speech and writes the acknowledgements. There's an offline fallback, and 18 tests including one that drives the live server with an MCP client." |
| 2:30 | Back to the lamp / app | "I built this because I'm this person: far from family, drowning in deadlines, and forgetting to call home. Hearth is the end-of-day check-in I wanted. Thanks for watching." |

**Retake tips:** if the language model writes a clunky acknowledgement, retake that line; the offline fallback is plainer than Bedrock.
Keep the browser at normal zoom so the trace text is readable on a laptop-sized video.
