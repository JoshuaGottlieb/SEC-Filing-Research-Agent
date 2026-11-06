"""The vector index behind dense search: chunk embeddings in a persistent Chroma collection.

DuckDB stays the source of truth (see finance/fact_store.py). The index holds embeddings and the
metadata searches filter on, but not the chunk text, so there is one copy of the text to keep
right. A search returns chunk ids, and the caller reads the chunks from the store.

Embedding is the slow step, so a build only embeds what changed. Each entry records a fingerprint
of its chunk: a matching entry is skipped, a different one is embedded again, and an entry whose
chunk no longer exists is removed. Chunks are written in batches, so an interrupted build keeps
its progress and the next build finishes the rest.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sec_agent.finance.fact_store import FactStore
from sec_agent.ingest.chunker import EMBEDDING_MODEL
from sec_agent.schemas import Chunk, FormType, Section

# The name carries the model, because vectors from two models can't be compared. Changing models
# starts a new collection rather than mixing them in one.
COLLECTION_NAME = "chunks-bge-base-en-v1.5"
# The model card's instruction for short queries against longer passages. Only queries get it;
# passages are embedded as they are.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# How many chunks are embedded and written at a time. It bounds the work an interruption can lose
# and the memory used on CPU. Chosen by hand.
BATCH_SIZE = 32
METADATA_FINGERPRINT = "fingerprint"


class Embedder(Protocol):
    """Turns text into L2-normalized vectors. Tests pass a small fake, so they need no model."""

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BgeEmbedder:
    """The embedding model, on CPU. Creating one loads it, downloading it on first use."""

    def __init__(self) -> None:
        # Imported here because it is slow to import and tests inject a fake instead.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([QUERY_INSTRUCTION + text])[0]


class ChunkFilter(BaseModel):
    """Which chunks a search may return. Every field set must match."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str | None = None
    form: FormType | None = None
    # A `FiscalPeriod.label`, such as "Q2 2026" or "FY2025".
    period: str | None = None
    section: Section | None = None
    accession: str | None = None

    @model_validator(mode="after")
    def _limits_the_search(self) -> ChunkFilter:
        # Searching every company's text is never what a comparison wants, so forgetting the
        # filter is an error rather than a search that quietly returns other companies.
        if not self.model_dump(exclude_none=True):
            raise ValueError("a search must be limited by at least one field")
        return self

    def to_where(self) -> dict:
        """The filter in Chroma's form, which needs $and only for two or more conditions."""
        conditions = [{key: value} for key, value in self.model_dump(mode="json").items() if value]
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class IndexHit(BaseModel):
    """One dense search result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    # Cosine distance: 0 for the same direction, larger for less similar.
    distance: float


class IndexSummary(BaseModel):
    """What one build did, for reporting and tests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    added: int = Field(ge=0)
    updated: int = Field(ge=0)
    removed: int = Field(ge=0)
    unchanged: int = Field(ge=0)


def chunk_fingerprint(chunk: Chunk) -> str:
    """Changes whenever any part of the chunk does, since an id alone can't show that.

    A chunk's id comes from its position, so a re-parsed filing can reuse an id for new text.
    """
    return hashlib.sha256(chunk.model_dump_json().encode("utf-8")).hexdigest()


def _metadata(chunk: Chunk) -> dict[str, str | int]:
    return {
        "ticker": chunk.ticker,
        "form": chunk.form.value,
        "period": chunk.period.label,
        "section": chunk.section.value,
        "accession": chunk.accession,
        METADATA_FINGERPRINT: chunk_fingerprint(chunk),
    }


class ChunkIndex:
    """The persistent Chroma collection of chunk embeddings. Use as a context manager."""

    def __init__(self, path: Path, embedder: Embedder) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # embedding_function=None because vectors are always supplied. Chroma's default would
        # download a different model and could put its vectors in this collection.
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def __enter__(self) -> ChunkIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def count(self) -> int:
        return self._collection.count()

    def sync(self, chunks: Sequence[Chunk]) -> IndexSummary:
        """Make the index match these chunks, embedding only what changed."""
        stored = self._collection.get(include=["metadatas"])
        fingerprints = {
            chunk_id: metadata[METADATA_FINGERPRINT]
            for chunk_id, metadata in zip(stored["ids"], stored["metadatas"], strict=True)
        }
        stale = [
            chunk
            for chunk in chunks
            if fingerprints.get(chunk.chunk_id) != chunk_fingerprint(chunk)
        ]
        wanted = {chunk.chunk_id for chunk in chunks}
        gone = [chunk_id for chunk_id in fingerprints if chunk_id not in wanted]

        # Removals come first, so an interrupted build can't leave retired chunks searchable.
        for start in range(0, len(gone), BATCH_SIZE):
            self._collection.delete(ids=gone[start : start + BATCH_SIZE])
        for start in range(0, len(stale), BATCH_SIZE):
            batch = stale[start : start + BATCH_SIZE]
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=self._embedder.embed_passages([chunk.text for chunk in batch]),
                metadatas=[_metadata(chunk) for chunk in batch],
            )
        updated = sum(chunk.chunk_id in fingerprints for chunk in stale)
        return IndexSummary(
            added=len(stale) - updated,
            updated=updated,
            removed=len(gone),
            unchanged=len(chunks) - len(stale),
        )

    def search(self, query: str, where: ChunkFilter, top_k: int) -> list[IndexHit]:
        """The chunks nearest the query among those the filter allows, nearest first.

        Fewer than `top_k` come back when fewer match, and none when nothing does.
        """
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=top_k,
            where=where.to_where(),
            include=["distances"],
        )
        return [
            IndexHit(chunk_id=chunk_id, distance=distance)
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]


def build_index(store: FactStore, index: ChunkIndex) -> IndexSummary:
    """Bring the index in line with the chunks in the store.

    Run `chunk_corpus` first: a filing the store has no chunks for is removed from the index.

    A full build took about 15 minutes on the project's 4-core VM, and an open store blocks every
    other process from the DuckDB file. For a large build, read `store.chunks()`, close the store,
    and then call `index.sync` directly.
    """
    return index.sync(store.chunks())
