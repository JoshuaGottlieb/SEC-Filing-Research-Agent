"""A `Retriever` that keeps its answers on disk, so a rerun replays searches instead of redoing.

Each search takes about 3.5 seconds on this machine's CPU, mostly the cross-encoder, and the
evaluations rerun the same searches many times: a replayed T6.4 test run spent 811 of its 962
seconds searching while its model calls came back from the replay cache (2026-11-13). With no GPU,
remembering answers across runs is the way to speed that up (the user's decision, 2026-11-15).
`SearchRetriever` already remembers answers within one process; this keeps them between processes.

It wraps any `Retriever` and is opt-in: nothing in the pipeline uses it unless a caller builds it.

An answer is replayed only when nothing it depends on has changed. The key holds:

- the query (normalized as `SearchRetriever` normalizes it), both filings' accessions and the
  section;
- a fingerprint of each filing's chunks (every chunk's `chunk_fingerprint`), so rebuilding the
  chunks of a filing misses the cache;
- a hash of the search code's source (hybrid search, reranker, index) and the model names, so any
  change to the search, its constants included, misses the cache;
- `variant`, which a caller sets when it builds a search that differs from the defaults in some
  other way (a `Reranker` with its own `min_score` or `keep`, for example).

Not covered: a model's weights changing on the Hugging Face hub under the same name. The models are
pinned by name only, as everywhere else in the project.

Failures are not stored: they propagate, and the next call searches again. The store is read on the
calling thread only (computing a filing's fingerprint), as `SearchRetriever` requires.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from pathlib import Path

from sec_agent.contracts import Retriever
from sec_agent.ingest import index as index_module
from sec_agent.ingest.chunker import EMBEDDING_MODEL
from sec_agent.ingest.index import chunk_fingerprint
from sec_agent.llm.cache import ReplayCache, cache_key
from sec_agent.retrieval import hybrid as hybrid_module
from sec_agent.retrieval import rerank as rerank_module
from sec_agent.retrieval.rerank import CROSS_ENCODER_MODEL

# Shared on purpose, though private: a second copy could drift and split one query into two keys.
from sec_agent.retrieval.tools import _normalize
from sec_agent.schemas import Chunk, Comparison, RetrievalResult, Section

ChunkSource = Callable[[str], list[Chunk]]  # accession -> that filing's chunks


def search_code_version() -> str:
    """A hash of everything in code that decides a search's answer.

    The source of the three modules covers their constants (`MIN_SCORE`, `RRF_K`, the keyword
    tokenizer and so on) without listing them, so a constant added later is covered too. A comment
    edit also changes it, which only costs one rerun of searches.
    """
    sources = [inspect.getsource(m) for m in (hybrid_module, rerank_module, index_module)]
    parts = [*sources, EMBEDDING_MODEL, CROSS_ENCODER_MODEL]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


class PersistentRetriever:
    """Answers `Retriever.retrieve` from disk when it can, and from `inner` when it can't."""

    def __init__(
        self, inner: Retriever, path: Path, chunks: ChunkSource, *, variant: str = ""
    ) -> None:
        self._inner = inner
        self._cache = ReplayCache(path)
        self._chunks = chunks
        self._variant = variant
        self._code = search_code_version()
        # A filing's chunks don't change during a run (the searcher keeps what it read too), so
        # each fingerprint is computed once.
        self._filings: dict[str, str] = {}
        self._hits = 0
        self._misses = 0

    @property
    def cache_hits(self) -> int:
        """Calls answered from disk."""
        return self._hits

    @property
    def cache_misses(self) -> int:
        """Calls passed to the wrapped retriever."""
        return self._misses

    def _filing_fingerprint(self, accession: str) -> str:
        if accession not in self._filings:
            prints = sorted(chunk_fingerprint(chunk) for chunk in self._chunks(accession))
            self._filings[accession] = hashlib.sha256("".join(prints).encode()).hexdigest()
        return self._filings[accession]

    def _key(self, query: str, comparison: Comparison, section: Section | None) -> str:
        return cache_key(
            query=query,
            current=comparison.current.accession,
            prior=comparison.prior.accession,
            current_chunks=self._filing_fingerprint(comparison.current.accession),
            prior_chunks=self._filing_fingerprint(comparison.prior.accession),
            section=section.value if section else None,
            code=self._code,
            variant=self._variant,
        )

    async def retrieve(
        self, query: str, comparison: Comparison, section: Section | None = None
    ) -> RetrievalResult:
        normalized = _normalize(query)
        if not normalized:
            raise ValueError("a search needs a query")
        key = self._key(normalized, comparison, section)
        payload = self._cache.get(key)
        if payload is not None:
            self._hits += 1
            result = RetrievalResult.model_validate_json(payload)
        else:
            self._misses += 1
            result = await self._inner.retrieve(query, comparison, section)
            self._cache.put(key, result.model_dump_json())
        # The stored answer may have been asked for in other words, so give back this caller's.
        return result if result.query == query else result.model_copy(update={"query": query})

    def close(self) -> None:
        self._cache.close()
