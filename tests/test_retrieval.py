"""MemoryIndex tests. Offline by design: a deterministic fake encoder stands in for
sentence-transformers, so CI never downloads a model."""

import pytest

np = pytest.importorskip("numpy")  # numpy rides in with the retrieval extras; skip cleanly without it

from hearth import llm
from hearth.retrieval import MemoryIndex
from hearth.service import Hearth
from hearth.store import Store

# A few synonyms the lexical matcher does NOT know, so tests can exercise "match by meaning".
_SYN = {"settle": "pay", "power": "electricity", "invoice": "bill"}


def fake_encoder(texts: list[str]):
    """Deterministic bag-of-words embedding with a tiny synonym map: no model download."""
    dim = 64
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, hash(_SYN.get(w, w)) % dim] += 1.0
    return out


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch):
    monkeypatch.setenv("HEARTH_SEMANTIC", "off")  # belt and braces: never load a model here


def test_add_and_search_ranks_by_similarity():
    ix = MemoryIndex(encoder=fake_encoder)
    assert ix.add_many([("c:1", "call mom back"), ("c:2", "submit the lab report"), ("c:3", "water the plants")])
    hits = ix.search("call mom", k=2)
    assert hits[0][1] == "c:1" and hits[0][0] > hits[1][0]
    assert len(ix) == 3


def test_faiss_path_agrees_with_numpy():
    pytest.importorskip("faiss")
    from hearth.retrieval import _normalize

    texts = ["call mom back", "pay the electricity bill", "water the plants", "book the dentist", "do the laundry"]
    ix = MemoryIndex(encoder=fake_encoder)
    ix.add_many([(f"c:{i}", t) for i, t in enumerate(texts)])
    got = ix.search("pay the bill", k=3)  # a full-index search goes through FAISS
    assert ix._faiss_index is not None, "expected the FAISS searcher to be used"
    # Recompute by hand with numpy; a silently wrong libomp result would show up here.
    vecs = _normalize(fake_encoder(texts))
    scores = vecs @ _normalize(fake_encoder(["pay the bill"]))[0]
    want = sorted(((float(s), f"c:{i}") for i, s in enumerate(scores)), key=lambda x: -x[0])[:3]
    # ties may order differently between FAISS and numpy, so compare scores and the (sorted) key sets
    assert [round(s, 5) for s, _, _ in got] == [round(s, 5) for s, _ in want]
    assert sorted(k for _, k, _ in got) == sorted(k for _, k in want)


def test_search_filters_to_allowed_keys():
    ix = MemoryIndex(encoder=fake_encoder)
    ix.add_many([("c:1", "call mom back"), ("e:1", "mom called while I was out")])
    hits = ix.search("call mom", k=5, allowed_keys={"e:1"})
    assert [k for _, k, _ in hits] == ["e:1"]
    assert [k for _, k, _ in ix.search("call mom", k=5, prefix="c:")] == ["c:1"]


def test_duplicate_keys_are_not_reindexed():
    ix = MemoryIndex(encoder=fake_encoder)
    ix.add("c:1", "call mom back")
    ix.add("c:1", "call mom back")
    assert len(ix) == 1


def test_persists_and_reloads_npz(tmp_path):
    path = str(tmp_path / "mem.index.npz")
    ix = MemoryIndex(path=path, encoder=fake_encoder)
    ix.add_many([("c:1", "pay the electricity bill"), ("c:2", "book the dentist")])
    reloaded = MemoryIndex(path=path, encoder=fake_encoder)
    assert len(reloaded) == 2
    assert reloaded.search("dentist", k=1)[0][1] == "c:2"


def test_unavailable_encoder_degrades_to_empty_results():
    ix = MemoryIndex()  # HEARTH_SEMANTIC=off: the lazy model load is refused
    assert not ix.add("c:1", "call mom back")
    assert ix.search("call mom") == []
    assert not ix.available()


def test_service_still_completes_lexically_without_embeddings():
    h = Hearth(Store(":memory:"), llm.Brain(mode="heuristic"))  # index can't load a model here
    h.add_commitment("Call my sister back", "tomorrow")
    r = h.complete_commitment("I called my sister")
    assert r["completed"]


def test_service_semantic_rescue_matches_paraphrase():
    from hearth.store import similarity

    ix = MemoryIndex(encoder=fake_encoder)
    h = Hearth(Store(":memory:"), llm.Brain(mode="heuristic"), index=ix)
    h.add_commitment("Pay the electricity bill")
    h.add_commitment("Water the plants")
    query = "settle power invoice"
    assert similarity("Pay the electricity bill", query) < 0.45  # the lexical matcher really does miss this
    r = h.complete_commitment(query)
    assert r["completed"] and "electricity" in r["commitment"]["text"].lower()
