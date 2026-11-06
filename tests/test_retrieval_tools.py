"""The retriever: repeats hit the cache, failures don't, and the worker never touches the store."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence
from datetime import date

import pytest

from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.contracts import Retriever
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore, SourcedChunk
from sec_agent.ingest.index import BgeEmbedder, ChunkFilter, ChunkIndex
from sec_agent.retrieval.hybrid import FusedCandidate, HybridSearcher, IndexOutOfSyncError
from sec_agent.retrieval.rerank import CrossEncoderScorer, Reranker
from sec_agent.retrieval.tools import SearchRetriever
from sec_agent.schemas import (
    Chunk,
    Comparison,
    ComparisonType,
    Filing,
    FiscalPeriod,
    FormType,
    Section,
)

CURRENT = "hsy-q2-26"
PRIOR = "hsy-q2-25"


def chunk(number: int, text: str, accession: str, section: Section = Section.MDNA) -> Chunk:
    year = 2026 if accession == CURRENT else 2025
    return Chunk(
        chunk_id=f"{accession}:{section.value}:{number:04d}:0",
        paragraph_id=f"{accession}:{section.value}:{number:04d}",
        ticker="HSY",
        cik="0000047111",
        form=FormType.FORM_10Q,
        period=FiscalPeriod(fiscal_year=year, fiscal_period="Q2", period_end=date(year, 6, 30)),
        section=section,
        accession=accession,
        text=text,
        token_count=len(text.split()),
    )


def filing(accession: str, year: int) -> Filing:
    return Filing(
        ticker="HSY",
        company="The Hershey Company",
        cik="0000047111",
        form=FormType.FORM_10Q,
        period=FiscalPeriod(fiscal_year=year, fiscal_period="Q2", period_end=date(year, 6, 30)),
        accession=accession,
        filed_date=date(year, 8, 1),
        url="https://www.sec.gov/Archives/example.htm",
    )


COMPARISON = Comparison(
    comparison_id="HSY:yoy-10q",
    ticker="HSY",
    comparison_type=ComparisonType.YOY_10Q,
    current=filing(CURRENT, 2026),
    prior=filing(PRIOR, 2025),
)
# The same company and comparison id, but other filings: a cache keyed on the id would mix them up.
OTHER_COMPARISON = Comparison(
    comparison_id="HSY:yoy-10q",
    ticker="HSY",
    comparison_type=ComparisonType.YOY_10Q,
    current=filing("hsy-q2-27", 2027),
    prior=filing(CURRENT, 2026),
)


class FakeSearcher:
    """A searcher with a fixed pool per accession that records where and when it ran."""

    def __init__(self, pool: dict[str, list[Chunk]] | None = None, delay: float = 0.0) -> None:
        self.pool = (
            pool
            if pool is not None
            else {
                CURRENT: [chunk(1, "current text", CURRENT)],
                PRIOR: [chunk(1, "prior text", PRIOR)],
            }
        )
        self.delay = delay
        self.fail_with: Exception | None = None
        self.gate: threading.Event | None = None
        self.searches: list[str] = []
        self.search_threads: list[int] = []
        self.prepare_threads: list[int] = []
        self._active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def prepare(self, where: ChunkFilter) -> None:
        self.prepare_threads.append(threading.get_ident())

    def search(self, query: str, where: ChunkFilter, top_k: int) -> list[FusedCandidate]:
        with self._guard:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            self.searches.append(f"{query}|{where.accession}|{where.section}")
            self.search_threads.append(threading.get_ident())
            if self.gate is not None:
                self.gate.wait(timeout=5)
            if self.delay:
                time.sleep(self.delay)
            if self.fail_with is not None:
                raise self.fail_with
            found = self.pool.get(where.accession, [])[:top_k]
            return [
                FusedCandidate(chunk=item, score=1 / (60 + rank), dense_rank=rank)
                for rank, item in enumerate(found, start=1)
            ]
        finally:
            with self._guard:
                self._active -= 1


class AlwaysRelevant:
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [1.0 for _ in passages]


class NeverRelevant:
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [-11.0 for _ in passages]


def make(searcher: FakeSearcher | None = None, scorer=None, **kwargs):
    searcher = searcher or FakeSearcher()
    return searcher, SearchRetriever(Reranker(searcher, scorer or AlwaysRelevant()), **kwargs)


def texts(passages) -> list[str]:
    return [passage.chunk.text for passage in passages]


# --- The interface ---


def test_it_satisfies_the_retriever_interface():
    _, retriever = make()
    assert isinstance(retriever, Retriever)


async def test_a_first_call_searches_both_filings_and_returns_the_result():
    searcher, retriever = make()

    result = await retriever.retrieve("gross margin", COMPARISON)

    assert texts(result.current) == ["current text"]
    assert texts(result.prior) == ["prior text"]
    assert result.query == "gross margin"
    assert not result.incomplete
    assert [item.split("|")[1] for item in searcher.searches] == [CURRENT, PRIOR]
    assert (retriever.cache_hits, retriever.cache_misses) == (0, 1)


async def test_the_section_reaches_the_search():
    searcher, retriever = make()

    await retriever.retrieve("q", COMPARISON, Section.RISK_FACTORS)

    assert [item.split("|")[2] for item in searcher.searches] == [Section.RISK_FACTORS.value] * 2


# --- The cache: the task's done-when ---


async def test_a_repeat_query_is_answered_from_the_cache_without_searching():
    searcher, retriever = make()

    first = await retriever.retrieve("gross margin", COMPARISON)
    searches_after_first = len(searcher.searches)
    second = await retriever.retrieve("gross margin", COMPARISON)

    assert second == first
    assert len(searcher.searches) == searches_after_first == 2
    assert (retriever.cache_hits, retriever.cache_misses) == (1, 1)


async def test_spacing_and_capitals_do_not_make_a_new_query_and_the_callers_own_words_come_back():
    searcher, retriever = make()

    first = await retriever.retrieve("Gross margin", COMPARISON)
    again = await retriever.retrieve("  gross   MARGIN ", COMPARISON)

    assert len(searcher.searches) == 2
    assert (first.query, again.query) == ("Gross margin", "  gross   MARGIN ")
    assert again.current == first.current
    # The stored answer is untouched, so the first spelling still comes back as it was asked.
    assert (await retriever.retrieve("Gross margin", COMPARISON)).query == "Gross margin"


@pytest.mark.parametrize(
    "other",
    [
        ("another question", COMPARISON, None),
        ("q", COMPARISON, Section.MDNA),
        ("q", OTHER_COMPARISON, None),
    ],
    ids=["query", "section", "filings"],
)
async def test_a_different_query_section_or_pair_of_filings_is_a_new_search(other):
    searcher, retriever = make()
    await retriever.retrieve("q", COMPARISON)
    before = len(searcher.searches)

    await retriever.retrieve(*other)

    assert len(searcher.searches) == before + 2
    assert retriever.cache_misses == 2


async def test_an_empty_side_is_a_result_not_an_error_and_is_cached():
    searcher, retriever = make(scorer=NeverRelevant())

    first = await retriever.retrieve("freight", COMPARISON)
    again = await retriever.retrieve("freight", COMPARISON)

    assert (first.current, first.prior) == ((), ())
    assert first.incomplete
    assert again == first
    assert len(searcher.searches) == 2


async def test_the_oldest_answer_is_dropped_when_the_cache_is_full():
    searcher, retriever = make(cache_size=2)
    for query in ("one", "two"):
        await retriever.retrieve(query, COMPARISON)
    await retriever.retrieve("one", COMPARISON)  # a hit, which makes "two" the oldest
    await retriever.retrieve("three", COMPARISON)  # evicts "two"
    before = len(searcher.searches)

    await retriever.retrieve("one", COMPARISON)
    await retriever.retrieve("three", COMPARISON)
    assert len(searcher.searches) == before  # both still cached
    await retriever.retrieve("two", COMPARISON)
    assert len(searcher.searches) == before + 2  # "two" was evicted


def test_the_cache_must_hold_something():
    with pytest.raises(ValueError, match="cache_size"):
        make(cache_size=0)


async def test_a_blank_query_is_refused_without_a_search():
    searcher, retriever = make()

    with pytest.raises(ValueError, match="query"):
        await retriever.retrieve("   ", COMPARISON)

    assert searcher.searches == []


# --- Failures are never cached ---


async def test_an_out_of_sync_index_is_raised_as_it_is_and_not_cached():
    searcher, retriever = make()
    searcher.fail_with = IndexOutOfSyncError("not in the store")

    with pytest.raises(IndexOutOfSyncError, match="not in the store"):
        await retriever.retrieve("q", COMPARISON)

    searcher.fail_with = None
    result = await retriever.retrieve("q", COMPARISON)

    assert not result.incomplete
    assert (retriever.cache_hits, retriever.cache_misses) == (0, 2)


async def test_a_failure_reaches_every_caller_waiting_on_that_search():
    searcher, retriever = make()
    searcher.fail_with = RuntimeError("boom")
    searcher.delay = 0.05

    outcomes = await asyncio.gather(
        *(retriever.retrieve("q", COMPARISON) for _ in range(3)), return_exceptions=True
    )

    assert [type(item) for item in outcomes] == [RuntimeError] * 3
    assert retriever.cache_misses == 1


# --- Concurrency ---


async def test_identical_calls_at_once_run_one_search():
    searcher, retriever = make(FakeSearcher(delay=0.05))

    results = await asyncio.gather(*(retriever.retrieve("q", COMPARISON) for _ in range(5)))

    assert all(item == results[0] for item in results)
    assert len(searcher.searches) == 2
    assert (retriever.cache_hits, retriever.cache_misses) == (4, 1)


async def test_a_caller_that_gives_up_does_not_cancel_the_search_for_the_others():
    searcher = FakeSearcher()
    searcher.gate = threading.Event()
    _, retriever = make(searcher)

    quitter = asyncio.ensure_future(retriever.retrieve("q", COMPARISON))
    await asyncio.sleep(0.1)  # the search is running on its worker, held at the gate
    waiter = asyncio.ensure_future(retriever.retrieve("q", COMPARISON))
    await asyncio.sleep(0)
    quitter.cancel()
    searcher.gate.set()

    result = await waiter
    with pytest.raises(asyncio.CancelledError):
        await quitter
    assert not result.incomplete
    assert (await retriever.retrieve("q", COMPARISON)) == result
    assert (retriever.cache_hits, retriever.cache_misses) == (2, 1)


async def test_searches_run_off_the_event_loop_one_at_a_time_and_the_store_is_read_on_it():
    loop_thread = threading.get_ident()
    searcher, retriever = make(FakeSearcher(delay=0.02))

    await asyncio.gather(*(retriever.retrieve(f"question {n}", COMPARISON) for n in range(6)))

    assert searcher.max_active == 1
    assert loop_thread not in searcher.search_threads
    assert set(searcher.prepare_threads) == {loop_thread}
    assert len(searcher.searches) == 12


async def test_the_event_loop_keeps_running_during_a_search():
    searcher, retriever = make(FakeSearcher(delay=0.3))
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker = asyncio.ensure_future(tick())
    await retriever.retrieve("q", COMPARISON)
    ticker.cancel()

    # A search that blocked the loop for its 0.6 s would leave almost no ticks.
    assert ticks > 15


# --- Through the real hybrid search, in a temporary store ---


class OneThreadStore:
    """Fails the way a shared DuckDB connection can: any read from another thread is an error."""

    def __init__(self, store: FactStore) -> None:
        self._store = store
        self._owner = threading.get_ident()
        self.reads = 0

    def chunks(self, accession: str | None = None) -> list[Chunk]:
        assert threading.get_ident() == self._owner, "the store was read from a worker thread"
        self.reads += 1
        return self._store.chunks(accession)


class SameVectorEmbedder:
    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class KeywordScorer:
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        words = set(query.lower().split())
        return [5.0 if words & set(p.lower().replace(".", "").split()) else -9.0 for p in passages]


async def test_with_the_real_searcher_the_worker_never_reads_the_store(settings):
    store = FactStore(settings.duckdb_path)
    index = ChunkIndex(settings.chroma_dir, SameVectorEmbedder())
    pool = [
        chunk(1, "Freight costs rose.", CURRENT),
        chunk(2, "Net sales increased.", CURRENT),
        chunk(1, "Freight rates eased.", PRIOR),
        chunk(2, "Net sales fell.", PRIOR),
    ]
    for accession in (CURRENT, PRIOR):
        store.replace_chunks(
            accession,
            [
                SourcedChunk(chunk=item, paragraph_ids=(item.paragraph_id,))
                for item in pool
                if item.accession == accession
            ],
        )
    index.sync(pool)
    guarded = OneThreadStore(store)
    try:
        retriever = SearchRetriever(Reranker(HybridSearcher(guarded, index), KeywordScorer()))

        results = await asyncio.gather(
            retriever.retrieve("freight", COMPARISON),
            retriever.retrieve("sales", COMPARISON),
            retriever.retrieve("tariff", COMPARISON),
            retriever.retrieve("freight", COMPARISON, Section.MDNA),
        )

        assert texts(results[0].current) == ["Freight costs rose."]
        assert texts(results[1].prior) == ["Net sales fell."]
        assert results[2].incomplete  # dense search returns every chunk; only the cutoff says no
        assert texts(results[3].prior) == ["Freight rates eased."]
        # Read once per filter, on this thread: (current, prior) x (no section, MD&A).
        assert guarded.reads == 4
    finally:
        store.close()
        index.close()


def test_searching_after_prepare_reads_nothing_from_the_store(settings):
    store = FactStore(settings.duckdb_path)
    index = ChunkIndex(settings.chroma_dir, SameVectorEmbedder())
    item = chunk(1, "Freight costs rose.", CURRENT)
    store.replace_chunks(CURRENT, [SourcedChunk(chunk=item, paragraph_ids=(item.paragraph_id,))])
    index.sync([item])
    counting = OneThreadStore(store)
    try:
        searcher = HybridSearcher(counting, index)
        where = ChunkFilter(accession=CURRENT)

        searcher.prepare(where)
        assert counting.reads == 1
        found = searcher.search("freight", where, 5)

        assert [candidate.chunk.chunk_id for candidate in found] == [item.chunk_id]
        assert counting.reads == 1
    finally:
        store.close()
        index.close()


# --- The real filings and models ---


class CountingScorer:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls += 1
        return self.inner.score(query, passages)


@pytest.fixture(scope="module")
def real():
    settings = get_settings()
    with (
        FactStore(settings.duckdb_path, read_only=True) as store,
        ChunkIndex(settings.chroma_dir, BgeEmbedder()) as index,
    ):
        if index.count() == 0:
            pytest.skip("The index has not been built.")
        scorer = CountingScorer(CrossEncoderScorer())
        reranker = Reranker(HybridSearcher(store, index), scorer)
        yield (
            SearchRetriever(reranker),
            scorer,
            reranker,
            Corpus.load(REPO_ROOT / "config" / "corpus.yaml"),
        )


@pytest.mark.corpus
async def test_real_repeat_and_concurrent_queries_on_the_real_store(real):
    retriever, scorer, reranker, corpus = real
    comparison = corpus.comparison("MAT", ComparisonType.YOY_10Q)
    queries = ["freight costs", "tariff impact on costs", "gross margin change", "liquidity"]

    results = await asyncio.gather(*(retriever.retrieve(q, comparison) for q in queries))

    assert scorer.calls == 2 * len(queries)
    # The same answers a plain sequential search gives: threads and cache change nothing.
    for query, result in zip(queries, results, strict=True):
        assert result == reranker.search(query, comparison)
    calls = scorer.calls
    again = await retriever.retrieve("Freight  costs", comparison)
    assert again.current == results[0].current and again.prior == results[0].prior
    assert scorer.calls == calls
