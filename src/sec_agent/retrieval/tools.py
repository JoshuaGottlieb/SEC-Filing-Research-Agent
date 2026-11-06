"""The first `Retriever`: reranked search over a comparison's two filings, cached per query.

The search itself is synchronous and slow (about 3 to 4 seconds on this CPU, mostly the
cross-encoder), so it runs on a worker thread and the event loop stays free for the LLM calls that
run beside it. Repeating a query is common (the Tree-of-Thought branches ask similar questions, and
the Retrieval Agent retries), so each answer is kept and a repeat costs nothing.

Threads and the store. A DuckDB connection can't be used by two threads at once: tested
concurrently on the real store, it either raised "connection is busy" or silently returned no
rows, which here would read as "no match". The rest of the process shares that store, so the
worker must never touch it. `Reranker.prepare` reads what the search needs on the calling thread
first, and the worker then reads only the index and the models. Those are used under one lock:
Hugging Face's fast tokenizers refuse concurrent use, and the CPU is the limit anyway.

Use one retriever from one event loop, and create a new one after the chunks are rebuilt, because
the searcher keeps what it read.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict

from sec_agent.retrieval.rerank import Reranker
from sec_agent.schemas import Comparison, RetrievalResult, Section

# The most answers kept. A run asks a few hundred questions at most (nine comparisons, tens of
# queries each), so this is a bound on memory and not a limit that should ever matter. Chosen by
# hand.
CACHE_SIZE = 1000

# The normalized query, the two filings' accessions, and the section. The accessions identify the
# comparison's filings exactly, which a comparison id alone would not.
CacheKey = tuple[str, str, str, Section | None]


def _normalize(query: str) -> str:
    """Spacing and capitals don't change a search: both models lowercase and split on spaces."""
    return " ".join(query.split()).casefold()


class SearchRetriever:
    """Answers `Retriever.retrieve` from reranked hybrid search, remembering each answer.

    An empty side is a result and is cached like any other. A failure is not: it propagates to
    every caller waiting on that search and nothing is stored, so `IndexOutOfSyncError` stays a
    visible error and the next call tries again.
    """

    def __init__(self, reranker: Reranker, *, cache_size: int = CACHE_SIZE) -> None:
        if cache_size < 1:
            raise ValueError(f"cache_size must be at least 1, got {cache_size}")
        self._reranker = reranker
        self._cache_size = cache_size
        # Read and written only on the event loop's thread, so they need no lock.
        self._cache: OrderedDict[CacheKey, RetrievalResult] = OrderedDict()
        # Searches running now, so a second caller asking the same thing waits for the first
        # instead of starting a duplicate.
        self._pending: dict[CacheKey, asyncio.Task[RetrievalResult]] = {}
        self._worker_lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def cache_hits(self) -> int:
        """Calls answered without starting a search: from the cache, or by joining one running."""
        return self._hits

    @property
    def cache_misses(self) -> int:
        """Calls that started a search."""
        return self._misses

    async def retrieve(
        self, query: str, comparison: Comparison, section: Section | None = None
    ) -> RetrievalResult:
        normalized = _normalize(query)
        if not normalized:
            raise ValueError("a search needs a query")
        key: CacheKey = (
            normalized,
            comparison.current.accession,
            comparison.prior.accession,
            section,
        )

        result = self._cache.get(key)
        if result is not None:
            self._cache.move_to_end(key)
            self._hits += 1
        else:
            task = self._pending.get(key)
            if task is None:
                self._misses += 1
                task = asyncio.ensure_future(self._search(query, comparison, section))
                self._pending[key] = task
                task.add_done_callback(lambda done, key=key: self._finished(key, done))
            else:
                self._hits += 1
            # Shielded, so a caller that gives up doesn't cancel a search others are waiting on.
            result = await asyncio.shield(task)
        # The cached answer may have been asked for in other words, so give back this caller's.
        return result if result.query == query else result.model_copy(update={"query": query})

    async def _search(
        self, query: str, comparison: Comparison, section: Section | None
    ) -> RetrievalResult:
        # On this thread, which owns the store; see the module docstring.
        self._reranker.prepare(comparison, section)
        return await asyncio.to_thread(self._search_in_worker, query, comparison, section)

    def _search_in_worker(
        self, query: str, comparison: Comparison, section: Section | None
    ) -> RetrievalResult:
        with self._worker_lock:
            return self._reranker.search(query, comparison, section)

    def _finished(self, key: CacheKey, task: asyncio.Task[RetrievalResult]) -> None:
        self._pending.pop(key, None)
        # Asking for the exception also marks it as seen, which silences asyncio's warning when
        # every caller that was waiting has been cancelled.
        if task.cancelled() or task.exception() is not None:
            return
        self._cache[key] = task.result()
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
