"""Saved passage embeddings, so repeated alignments skip the model (user, 2026-11-11).

Alignment (T3.1) embeds every paragraph that isn't identical across a comparison's two filings,
and that embedding is nearly all of its time on this CPU (about 10 minutes for the nine
comparisons). The same text comes up again and again: the Q2 2026 10-Q is in two comparisons,
and corpus tests, probes and full runs re-align the same filings. `CachedEmbedder` wraps any
`Embedder` and keeps each passage's vector in SQLite, keyed by the model and the exact text, so a
repeat reads the stored vector instead of running the model.

Vectors are stored as float32, the precision the bge model produces, so a stored vector is
returned exactly as the model gave it. A vector can differ in its last bits depending on which
other texts it was batched with, so a cached run is not always bit for bit the uncached one. That
matters only where alignment faces an exact tie. On the nine comparisons (2026-11-11), one tie
came out differently: Mattel's quarterly and year-to-date MD&A both say "Vehicles gross billings
increased 7%...", and those two identical paragraphs swapped their earlier partners; every other
change was identical. Once stored, vectors no longer depend on batching, so cached runs repeat
exactly.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sec_agent.config import Settings
from sec_agent.ingest.index import Embedder

# SQLite limits the number of parameters in one statement; lookups go in groups this size.
# Chosen well under the oldest default limit (999).
LOOKUP_GROUP = 500


def embedding_cache_path(settings: Settings) -> Path:
    """The shared cache file, in the data directory next to the LLM replay cache."""
    return settings.data_dir / "embedding_cache.sqlite"


def passage_key(model: str, text: str) -> str:
    """A fixed-length key for one passage. The model is part of it, so models never mix."""
    return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()


class CachedEmbedder:
    """An `Embedder` that stores passage vectors and reuses them. Queries pass straight through.

    Safe to share between threads (a lock serializes the database). `hits` and `misses` count
    distinct passages found and embedded, for reporting how much a run reused.
    """

    def __init__(self, inner: Embedder, model: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._inner = inner
        self._model = model
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS passages (key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )
        self._db.commit()
        self.hits = 0
        self.misses = 0

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [passage_key(self._model, text) for text in texts]
        with self._lock:
            found = self._lookup(set(keys))
        # Each missing text once, in order, in one call so the model batches them together.
        missing = dict.fromkeys(
            (key, text) for key, text in zip(keys, texts, strict=True) if key not in found
        )
        if missing:
            vectors = self._inner.embed_passages([text for _, text in missing])
            if len(vectors) != len(missing):
                raise ValueError(f"asked for {len(missing)} vectors, got {len(vectors)}")
            new = {
                key: np.asarray(vector, dtype=np.float32)
                for (key, _), vector in zip(missing, vectors, strict=True)
            }
            with self._lock:
                self._db.executemany(
                    "INSERT OR REPLACE INTO passages (key, vector) VALUES (?, ?)",
                    [(key, vector.tobytes()) for key, vector in new.items()],
                )
                self._db.commit()
            found.update(new)
        self.hits += len(set(keys)) - len(missing)
        self.misses += len(missing)
        return [found[key].tolist() for key in keys]

    def embed_query(self, text: str) -> list[float]:
        # Queries are short and searched once; caching them would only grow the file.
        return self._inner.embed_query(text)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _lookup(self, keys: set[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        ordered = sorted(keys)
        for start in range(0, len(ordered), LOOKUP_GROUP):
            group = ordered[start : start + LOOKUP_GROUP]
            marks = ",".join("?" * len(group))
            rows = self._db.execute(
                f"SELECT key, vector FROM passages WHERE key IN ({marks})", group
            ).fetchall()
            found.update((key, np.frombuffer(blob, dtype=np.float32)) for key, blob in rows)
        return found
