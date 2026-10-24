"""Semantic memory: an embedding index over journal entries and commitments.

``MemoryIndex`` embeds text with a sentence-transformer (all-MiniLM-L6-v2) and
searches with FAISS (inner product over L2-normalised vectors, i.e. cosine).
It degrades gracefully, in order:

  * no FAISS            -> plain numpy matmul (identical scores)
  * no sentence-transformers / no model -> the index reports ``available() is False``
                                           and callers keep using the lexical matcher

So nothing here is ever required: ``service.py`` treats semantic search as a
rescue layer on top of the existing token/SequenceMatcher matching, and the
test suite runs without downloading anything.

HEARTH_SEMANTIC = auto | off
  auto  use the embedding index when sentence-transformers is importable
  off   never load a model (CI, tests)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Iterable

log = logging.getLogger("hearth.retrieval")

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_MODELS: dict[str, object] = {}  # one SentenceTransformer per model name, shared process-wide


def _load_model(name: str):
    if name not in _MODELS:
        from sentence_transformers import SentenceTransformer

        try:
            # Prefer the local cache: the HuggingFace Hub freshness check can stall for minutes offline.
            _MODELS[name] = SentenceTransformer(name, local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached yet; fetch it once
            _MODELS[name] = SentenceTransformer(name)
    return _MODELS[name]


def _normalize(mat):
    import numpy as np

    mat = np.asarray(mat, dtype="float32")
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class MemoryIndex:
    """Cosine-similarity search over (key, text) memories, with .npz persistence.

    Keys are strings like ``commitment:12`` or ``entry:7`` so one index can hold
    both kinds of memory and callers can filter on whatever subset is live.
    """

    def __init__(
        self,
        path: str | None = None,
        model_name: str | None = None,
        encoder: Callable[[list[str]], "object"] | None = None,
    ):
        self.path = path  # None = in-memory only (e.g. Store(':memory:'))
        self.model_name = model_name or os.environ.get("HEARTH_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self._encoder = encoder  # injectable for tests; texts -> (n, d) array
        self._encoder_failed = False
        self._lock = threading.Lock()
        self._keys: list[str] = []
        self._texts: list[str] = []
        self._vectors = None  # (n, d) float32, L2-normalised
        self._faiss_index = None
        self._loaded_from_disk = False

    # ---- model ----------------------------------------------------------
    def _encode(self, texts: list[str]):
        """Return normalised (n, d) embeddings, or None when no encoder can be had."""
        if self._encoder is None:
            if self._encoder_failed or os.environ.get("HEARTH_SEMANTIC", "auto").lower() == "off":
                return None
            try:
                model = _load_model(self.model_name)
                self._encoder = lambda ts: model.encode(ts, show_progress_bar=False)
            except Exception as exc:  # noqa: BLE001 - any failure means lexical fallback
                log.warning("sentence-transformers unavailable (%s); semantic search off", exc)
                self._encoder_failed = True
                return None
        return _normalize(self._encoder(texts))

    def available(self) -> bool:
        """True once an encoder is loadable. May trigger the (lazy) model load."""
        return self._encode(["ping"]) is not None

    # ---- persistence ----------------------------------------------------
    def _load(self):
        if self._loaded_from_disk or not self.path or not os.path.exists(self.path):
            self._loaded_from_disk = True
            return
        try:
            import numpy as np

            data = np.load(self.path, allow_pickle=False)
            self._keys = [str(k) for k in data["keys"]]
            self._texts = [str(t) for t in data["texts"]]
            self._vectors = _normalize(data["vectors"])
            self._faiss_index = None
        except Exception as exc:  # noqa: BLE001 - a corrupt sidecar never breaks the app
            log.warning("could not load %s (%s); starting an empty index", self.path, exc)
        self._loaded_from_disk = True

    def _save(self):
        if not self.path or self._vectors is None:
            return
        import numpy as np

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        np.savez_compressed(
            self.path,
            keys=np.array(self._keys),
            texts=np.array(self._texts),
            vectors=self._vectors,
        )

    # ---- index ----------------------------------------------------------
    def _searcher(self):
        """A FAISS IndexFlatIP over the current vectors, or None to use numpy."""
        if self._faiss_index is not None:
            return self._faiss_index
        try:
            # faiss-cpu and torch each bundle their own libomp; on macOS loading both aborts the
            # process at the first faiss search (OMP error #15). The standard workaround is
            # KMP_DUPLICATE_LIB_OK plus a single faiss thread; tests cross-check faiss against
            # the numpy path so a silently wrong result would be caught.
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            import faiss

            faiss.omp_set_num_threads(1)
            index = faiss.IndexFlatIP(self._vectors.shape[1])
            index.add(self._vectors)
            self._faiss_index = index
        except Exception:  # noqa: BLE001 - numpy matmul gives the same scores
            self._faiss_index = None
        return self._faiss_index

    def add(self, key: str, text: str) -> bool:
        """Embed and remember one text. Returns False when semantic search is off."""
        return self.add_many([(key, text)])

    def add_many(self, items: Iterable[tuple[str, str]]) -> bool:
        with self._lock:
            self._load()
            known = set(self._keys)
            fresh = [(k, t) for k, t in items if k not in known and t.strip()]
            if not fresh:
                return self._vectors is not None
            vecs = self._encode([t for _, t in fresh])
            if vecs is None:
                return False
            import numpy as np

            self._keys += [k for k, _ in fresh]
            self._texts += [t for _, t in fresh]
            self._vectors = vecs if self._vectors is None else np.vstack([self._vectors, vecs])
            self._faiss_index = None  # rebuilt on next search
            self._save()
            return True

    def search(
        self, query: str, k: int = 5, allowed_keys: set[str] | None = None, prefix: str | None = None
    ) -> list[tuple[float, str, str]]:
        """Top-k ``(score, key, text)`` by cosine similarity; [] when unavailable.

        ``allowed_keys`` / ``prefix`` filter to live items (e.g. open commitments)
        without ever rebuilding the index.
        """
        with self._lock:
            self._load()
            if self._vectors is None or not query.strip():
                return []
            q = self._encode([query])
            if q is None:
                return []
            idx = [
                i
                for i, key in enumerate(self._keys)
                if (allowed_keys is None or key in allowed_keys) and (prefix is None or key.startswith(prefix))
            ]
            if not idx:
                return []
            if len(idx) == len(self._keys) and (searcher := self._searcher()) is not None:
                scores, ids = searcher.search(q, min(k, len(self._keys)))
                return [(float(s), self._keys[i], self._texts[i]) for s, i in zip(scores[0], ids[0]) if i >= 0]
            import numpy as np

            sub = self._vectors[idx]
            scores = (sub @ q[0]).tolist()
            order = np.argsort(scores)[::-1][:k]
            return [(float(scores[j]), self._keys[idx[j]], self._texts[idx[j]]) for j in order]

    def __len__(self) -> int:
        with self._lock:
            self._load()
            return len(self._keys)
