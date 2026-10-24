"""Synthetic retrieval-eval corpus: a diary that sounds like a person.

60 memories (35 commitments + 25 journal entries) and 50 queries, each query
annotated with the gold item id(s) and a slice:

  paraphrase  synonym phrasing with little token overlap ("rang my mum" vs "Call Mom back")
  temporal    "what did I promise this week" style, answered by due words in the text
  distractor  several near-neighbours exist and only one is right
  direct      near-verbatim, the easy case both matchers should get

Everything is hand-written and fixed; SEED only governs the (reproducible)
shuffle of the corpus so that insertion order never flatters an index.
"""

from __future__ import annotations

import random

SEED = 2026

COMMITMENTS = [
    ("c1", "Call Mom back"),
    ("c2", "Book a dentist appointment"),
    ("c3", "Submit the lab report by Friday"),
    ("c4", "Pay the electricity bill"),
    ("c5", "Call the landlord about the leaking tap"),
    ("c6", "Email Professor Alvarez about the extension"),
    ("c7", "Buy groceries for the week"),
    ("c8", "Return the library books"),
    ("c9", "Pick up my prescription from the pharmacy"),
    ("c10", "Text Priya about the weekend cabin trip"),
    ("c11", "Finish the slide deck for Monday's standup"),
    ("c12", "Renew the apartment lease by the end of the month"),
    ("c13", "Call Grandma on Sunday"),
    ("c14", "Do the laundry"),
    ("c15", "Fix the flat tire on my bike"),
    ("c16", "Send the invoice to the design client"),
    ("c17", "Schedule a haircut"),
    ("c18", "Water the plants"),
    ("c19", "Sign up for the gym near work"),
    ("c20", "Mail the birthday card to Dad"),
    ("c21", "Back up my laptop"),
    ("c22", "Study for the statistics midterm this week"),
    ("c23", "RSVP to Ana's wedding"),
    ("c24", "Take the recycling out"),
    ("c25", "Refill my metro card"),
    ("c26", "Call the bank about the unrecognized charge"),
    ("c27", "Clean the bathroom this weekend"),
    ("c28", "Pack for the Denver flight on Thursday"),
    ("c29", "Write thank-you notes from the interview"),
    ("c30", "Register for spring classes before the end of the month"),
    ("c31", "Get a graduation gift for my roommate"),
    ("c32", "Drop off the dry cleaning"),
    ("c33", "Update my resume"),
    ("c34", "Cancel the streaming subscription I never use"),
    ("c35", "Make a vet appointment for Mochi"),
]

ENTRIES = [
    ("e1", "Work was brutal today. Three back-to-back meetings and my manager moved the standup deck deadline to Monday. I barely had time to eat."),
    ("e2", "I miss home. Mom called twice while I was in the lab and I couldn't pick up. I feel bad about it."),
    ("e3", "Quiet night. Made pasta, watched half a movie, fell asleep on the couch."),
    ("e4", "The tap in the kitchen is still dripping. The sound drove me crazy all night."),
    ("e5", "Statistics lecture went over my head today. The midterm is next week and I'm nervous."),
    ("e6", "Ana texted me her wedding invite today. It's in June, in Lisbon of all places. I'm happy for her."),
    ("e7", "I burned dinner so badly the smoke alarm went off. Ordered takeout and laughed about it with my roommate."),
    ("e8", "Landlord emailed about renewing the lease. Rent is going up again. This city is impossible."),
    ("e9", "Long call with Grandma. She told the story about the orchard again. I didn't mind."),
    ("e10", "The interview went okay, I think. They asked about my capstone project and I rambled."),
    ("e11", "Mochi threw up twice today. She seems fine now but I'm worried about her."),
    ("e12", "My flight to Denver on Thursday got moved to 6am. Who does that."),
    ("e13", "Skipped the gym again. Third week of saying I'll start."),
    ("e14", "Payday, finally. Paid rent, then stared at the electricity bill and closed the laptop."),
    ("e15", "Priya wants to do a cabin weekend next month. I said I'd check my schedule."),
    ("e16", "Lab ran late again. Professor Alvarez hinted extensions are possible if we ask early."),
    ("e17", "Dad's birthday is coming up. I saw a card at the bookstore and didn't buy it. Why am I like this."),
    ("e18", "My bike has a flat so I took the metro, and of course my card was empty."),
    ("e19", "There's a charge on my bank statement I don't recognize. Forty-two dollars at a store I've never heard of."),
    ("e20", "Roommate's graduation is in two weeks. The apartment is filling up with her boxes already."),
    ("e21", "Couldn't sleep. Kept thinking about the lab report and whether my data section is wrong."),
    ("e22", "Nice evening for once. Walked home the long way, called my sister, felt almost normal."),
    ("e23", "The laundry pile has become a chair. I keep pretending not to see it."),
    ("e24", "Library emailed: two books overdue. One of them I never even opened."),
    ("e25", "My laptop made a clicking noise today and my stomach dropped. Everything I have is on it."),
]

# (id, query, gold ids, slice)
QUERIES = [
    # --- paraphrase: synonym phrasing, little token overlap -------------------
    ("q1", "I rang my mum earlier", ["c1"], "paraphrase"),
    ("q2", "get my teeth looked at", ["c2"], "paraphrase"),
    ("q3", "turn in the lab writeup", ["c3"], "paraphrase"),
    ("q4", "settle the power bill", ["c4"], "paraphrase"),
    ("q5", "get hold of the landlord about the drip", ["c5"], "paraphrase"),
    ("q6", "shoot a note to my professor asking for more time", ["c6"], "paraphrase"),
    ("q7", "pick up food for the week", ["c7"], "paraphrase"),
    ("q8", "take back my library loans", ["c8"], "paraphrase"),
    ("q9", "grab my meds from the drugstore", ["c9"], "paraphrase"),
    ("q10", "let Priya know about our getaway", ["c10"], "paraphrase"),
    ("q11", "wrap up the presentation for the team meeting", ["c11"], "paraphrase"),
    ("q12", "sort out my housing contract renewal", ["c12"], "paraphrase"),
    ("q13", "swing by the barber", ["c17"], "paraphrase"),
    ("q14", "put money on my transit pass", ["c25"], "paraphrase"),
    ("q15", "sort out the strange transaction with my bank", ["c26"], "paraphrase"),
    ("q16", "save my files somewhere safe", ["c21"], "paraphrase"),
    ("q17", "ditch the video service I never watch", ["c34"], "paraphrase"),
    ("q18", "get the cat seen by the animal doctor", ["c35"], "paraphrase"),
    # --- temporal: the window is in the item's own words -----------------------
    ("q19", "what did I promise to do this week", ["c3", "c22"], "temporal"),
    ("q20", "what's happening this weekend", ["c27", "c10"], "temporal"),
    ("q21", "what do I need to do before the end of the month", ["c12", "c30"], "temporal"),
    ("q22", "what's due on friday", ["c3"], "temporal"),
    ("q23", "what did I say about sunday", ["c13"], "temporal"),
    ("q24", "anything happening on thursday", ["c28"], "temporal"),
    ("q25", "what's on for monday", ["c11"], "temporal"),
    ("q26", "which deadlines are coming up this week", ["c3", "c22"], "temporal"),
    # --- distractor: near-neighbours exist, only one is right ------------------
    ("q27", "I called my grandmother", ["c13"], "distractor"),
    ("q28", "phoned the bank just now", ["c26"], "distractor"),
    ("q29", "I spoke to the landlord", ["c5"], "distractor"),
    ("q30", "finally called my mother back", ["c1"], "distractor"),
    ("q31", "sent the bill to the client", ["c16"], "distractor"),
    ("q32", "paid the electric", ["c4"], "distractor"),
    ("q33", "booked the dentist", ["c2"], "distractor"),
    ("q34", "made the appointment for the cat", ["c35"], "distractor"),
    ("q35", "emailed my professor", ["c6"], "distractor"),
    ("q36", "bought the graduation present", ["c31"], "distractor"),
    ("q37", "mailed dad's card", ["c20"], "distractor"),
    ("q38", "studied for stats", ["c22"], "distractor"),
    ("q39", "packed my suitcase for the trip", ["c28"], "distractor"),
    ("q40", "did a load of laundry", ["c14"], "distractor"),
    # --- direct: the easy case --------------------------------------------------
    ("q41", "water the plants", ["c18"], "direct"),
    ("q42", "take the recycling out", ["c24"], "direct"),
    ("q43", "fix my bike tire", ["c15"], "direct"),
    ("q44", "sign up for the gym", ["c19"], "direct"),
    ("q45", "update my resume", ["c33"], "direct"),
    ("q46", "thank-you notes from the interview", ["c29"], "direct"),
    ("q47", "register for spring classes", ["c30"], "direct"),
    ("q48", "RSVP to the wedding", ["c23"], "direct"),
    ("q49", "the night I burned dinner", ["e7"], "direct"),
    ("q50", "when my flight time changed", ["e12"], "direct"),
]

# The 25 queries the RAG answer-faithfulness eval generates answers for:
# a deterministic mix across slices.
FAITHFULNESS_QUERY_IDS = [
    "q1", "q3", "q5", "q7", "q9", "q11", "q13", "q15", "q17",      # paraphrase
    "q19", "q20", "q21", "q22", "q24", "q26",                       # temporal
    "q27", "q29", "q31", "q33", "q35", "q39",                       # distractor
    "q41", "q45", "q49", "q50",                                     # direct
]


def load() -> tuple[list[dict], list[dict]]:
    """Returns (corpus, queries). The corpus order is a fixed-seed shuffle."""
    corpus = [{"id": i, "kind": "commitment", "text": t} for i, t in COMMITMENTS]
    corpus += [{"id": i, "kind": "entry", "text": t} for i, t in ENTRIES]
    random.Random(SEED).shuffle(corpus)
    queries = [{"id": i, "query": q, "gold": g, "slice": s} for i, q, g, s in QUERIES]
    assert len(corpus) == 60 and len(queries) == 50
    assert {g for q in queries for g in q["gold"]} <= {c["id"] for c in corpus}
    return corpus, queries
