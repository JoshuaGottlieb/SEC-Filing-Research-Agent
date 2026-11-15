"""The persistent search cache: a rerun replays from disk, and any change to a search misses."""

from __future__ import annotations

from datetime import date

import pytest

from sec_agent.contracts import Retriever
from sec_agent.retrieval.search_cache import PersistentRetriever, search_code_version
from sec_agent.schemas import (
    Chunk,
    Comparison,
    ComparisonType,
    Filing,
    FiscalPeriod,
    FormType,
    RetrievalResult,
    RetrievedPassage,
    Section,
)

CURRENT = "hsy-q2-26"
PRIOR = "hsy-q2-25"


def chunk(accession: str, text: str) -> Chunk:
    year = 2026 if accession == CURRENT else 2025
    return Chunk(
        chunk_id=f"{accession}:mdna:0001:0",
        paragraph_id=f"{accession}:mdna:0001",
        ticker="HSY",
        cik="0000047111",
        form=FormType.FORM_10Q,
        period=FiscalPeriod(fiscal_year=year, fiscal_period="Q2", period_end=date(year, 6, 30)),
        section=Section.MDNA,
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


class FakeRetriever:
    """Counts searches and returns one passage per side, or raises when told to."""

    def __init__(self, chunks: dict[str, list[Chunk]]) -> None:
        self.calls = 0
        self.fail = False
        self._chunks = chunks

    async def retrieve(
        self, query: str, comparison: Comparison, section: Section | None = None
    ) -> RetrievalResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("search failed")
        return RetrievalResult(
            query=query,
            current=(RetrievedPassage(chunk=self._chunks[CURRENT][0], score=2.5),),
            prior=(RetrievedPassage(chunk=self._chunks[PRIOR][0], score=-1.0),),
        )


@pytest.fixture
def chunks() -> dict[str, list[Chunk]]:
    return {CURRENT: [chunk(CURRENT, "Cocoa costs rose.")], PRIOR: [chunk(PRIOR, "Prices rose.")]}


def make(tmp_path, chunks, inner=None, variant=""):
    inner = inner or FakeRetriever(chunks)
    cached = PersistentRetriever(
        inner, tmp_path / "search.sqlite", lambda accession: chunks[accession], variant=variant
    )
    return cached, inner


def test_it_is_a_retriever(tmp_path, chunks) -> None:
    cached, _ = make(tmp_path, chunks)
    assert isinstance(cached, Retriever)


async def test_a_new_process_replays_from_disk(tmp_path, chunks) -> None:
    first, inner = make(tmp_path, chunks)
    answer = await first.retrieve("cocoa costs", COMPARISON, Section.MDNA)
    first.close()

    second, _ = make(tmp_path, chunks, inner=inner)
    replayed = await second.retrieve("cocoa costs", COMPARISON, Section.MDNA)

    assert inner.calls == 1
    assert replayed == answer
    assert (second.cache_hits, second.cache_misses) == (1, 0)


async def test_spacing_and_capitals_hit_and_return_the_callers_query(tmp_path, chunks) -> None:
    cached, inner = make(tmp_path, chunks)
    await cached.retrieve("cocoa costs", COMPARISON, Section.MDNA)
    again = await cached.retrieve("  Cocoa   COSTS ", COMPARISON, Section.MDNA)
    assert inner.calls == 1
    assert again.query == "  Cocoa   COSTS "


async def test_another_section_misses(tmp_path, chunks) -> None:
    cached, inner = make(tmp_path, chunks)
    await cached.retrieve("cocoa costs", COMPARISON, Section.MDNA)
    await cached.retrieve("cocoa costs", COMPARISON, None)
    assert inner.calls == 2


async def test_rebuilt_chunks_miss(tmp_path, chunks) -> None:
    first, inner = make(tmp_path, chunks)
    await first.retrieve("cocoa costs", COMPARISON, Section.MDNA)
    first.close()

    # Same chunk id, new text: a re-parsed filing can reuse an id.
    chunks[CURRENT] = [chunk(CURRENT, "Cocoa costs rose sharply.")]
    second, _ = make(tmp_path, chunks, inner=inner)
    await second.retrieve("cocoa costs", COMPARISON, Section.MDNA)
    assert inner.calls == 2


async def test_another_variant_misses(tmp_path, chunks) -> None:
    first, inner = make(tmp_path, chunks)
    await first.retrieve("cocoa costs", COMPARISON, Section.MDNA)
    second, _ = make(tmp_path, chunks, inner=inner, variant="min_score=-8")
    await second.retrieve("cocoa costs", COMPARISON, Section.MDNA)
    assert inner.calls == 2


async def test_changed_search_code_misses(tmp_path, chunks, monkeypatch) -> None:
    first, inner = make(tmp_path, chunks)
    await first.retrieve("cocoa costs", COMPARISON, Section.MDNA)
    monkeypatch.setattr(
        "sec_agent.retrieval.search_cache.search_code_version", lambda: "edited search code"
    )
    second, _ = make(tmp_path, chunks, inner=inner)
    await second.retrieve("cocoa costs", COMPARISON, Section.MDNA)
    assert inner.calls == 2


async def test_a_failure_is_not_stored(tmp_path, chunks) -> None:
    cached, inner = make(tmp_path, chunks)
    inner.fail = True
    with pytest.raises(RuntimeError):
        await cached.retrieve("cocoa costs", COMPARISON, Section.MDNA)
    inner.fail = False
    await cached.retrieve("cocoa costs", COMPARISON, Section.MDNA)
    assert inner.calls == 2
    assert cached.cache_misses == 2


async def test_an_empty_query_is_refused(tmp_path, chunks) -> None:
    cached, inner = make(tmp_path, chunks)
    with pytest.raises(ValueError):
        await cached.retrieve("   ", COMPARISON)
    assert inner.calls == 0


def test_the_code_version_is_stable() -> None:
    assert search_code_version() == search_code_version()
