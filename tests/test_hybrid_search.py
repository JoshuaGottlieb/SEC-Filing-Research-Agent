"""Hybrid search: exact terms surface, filters hold, and the ranks fuse as the formula says."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest

from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore, SourcedChunk
from sec_agent.ingest.index import BgeEmbedder, ChunkFilter, ChunkIndex
from sec_agent.retrieval.hybrid import (
    RANKED_LIST_SIZE,
    RRF_K,
    HybridSearcher,
    IndexOutOfSyncError,
    reciprocal_rank_fusion,
    tokenize,
)
from sec_agent.schemas import Chunk, FiscalPeriod, FormType, Section


class SameVectorEmbedder:
    """Gives every text one vector, so dense search has no idea what is relevant.

    Whatever surfaces an exact term in these tests must then be keyword search.
    """

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class TableEmbedder:
    """Embeds by lookup, so a test can dictate exactly how dense search ranks."""

    def __init__(self, vectors: dict[str, list[float]], query: list[float]) -> None:
        self.vectors = vectors
        self.query = query

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.query


def chunk(
    number: int,
    text: str,
    *,
    ticker: str = "HSY",
    accession: str = "hsy-q2-26",
    fiscal_period: str = "Q2",
    year: int = 2026,
    form: FormType = FormType.FORM_10Q,
    section: Section = Section.MDNA,
) -> Chunk:
    return Chunk(
        chunk_id=f"{accession}:{section.value}:{number:04d}:0",
        paragraph_id=f"{accession}:{section.value}:{number:04d}",
        ticker=ticker,
        cik="0000000001",
        form=form,
        period=FiscalPeriod(
            fiscal_year=year, fiscal_period=fiscal_period, period_end=date(year, 6, 30)
        ),
        section=section,
        accession=accession,
        text=text,
        token_count=len(text.split()),
    )


class World:
    """A store and index in temporary folders, filled with the given chunks."""

    def __init__(self, settings, embedder) -> None:
        self.store = FactStore(settings.duckdb_path)
        self.index = ChunkIndex(settings.chroma_dir, embedder)

    def fill(self, chunks: Sequence[Chunk]) -> HybridSearcher:
        by_filing: dict[str, list[SourcedChunk]] = {}
        for item in chunks:
            by_filing.setdefault(item.accession, []).append(
                SourcedChunk(chunk=item, paragraph_ids=(item.paragraph_id,))
            )
        for accession, sourced in by_filing.items():
            self.store.replace_chunks(accession, sourced)
        self.index.sync(chunks)
        return HybridSearcher(self.store, self.index)

    def close(self) -> None:
        self.store.close()
        self.index.close()


@pytest.fixture
def world(settings):
    opened = World(settings, SameVectorEmbedder())
    yield opened
    opened.close()


FILLER = "Net sales increased during the quarter compared with the same period last year."
HERE = ChunkFilter(accession="hsy-q2-26")


def ids(candidates) -> list[str]:
    return [candidate.chunk.chunk_id for candidate in candidates]


# --- Exact terms: the task's done-when ---


def test_an_exact_term_surfaces_when_dense_search_cannot_tell(world):
    freight = chunk(30, "Higher freight expense reduced gross margin in the quarter.")
    others = [chunk(number, f"{FILLER} Item {number}.") for number in range(30)]
    searcher = world.fill([*others, freight])
    found = searcher.search("freight", HERE, 1)
    assert ids(found) == [freight.chunk_id]
    assert found[0].keyword_rank == 1


def test_an_exact_term_beats_a_crowd_of_dense_matches(settings):
    # Dense search puts the freight chunk dead last of 60, past the end of its list of 50.
    freight = chunk(99, "Higher freight expense reduced gross margin.")
    others = [chunk(number, f"Cocoa text number {number}.") for number in range(60)]
    vectors = {item.text: [1.0, number / 1000] for number, item in enumerate(others)}
    vectors[freight.text] = [0.0, 1.0]
    world = World(settings, TableEmbedder(vectors, query=[1.0, 0.0]))
    try:
        searcher = world.fill([*others, freight])
        found = searcher.search("freight", HERE, 20)
        match = next(item for item in found if item.chunk.chunk_id == freight.chunk_id)
        assert match.keyword_rank == 1
        assert match.dense_rank is None
    finally:
        world.close()


def test_a_plural_finds_a_singular_and_back(world):
    searcher = world.fill([chunk(1, "New tariffs raised costs."), chunk(2, FILLER)])
    assert ids(searcher.search("tariff", HERE, 5))[0].endswith("0001:0")
    assert searcher.search("tariff", HERE, 5)[0].keyword_rank == 1


def test_case_does_not_matter(world):
    searcher = world.fill([chunk(1, "FREIGHT costs rose."), chunk(2, FILLER)])
    assert searcher.search("freight", HERE, 5)[0].keyword_rank == 1


def test_a_hyphenated_term_is_found_by_its_words(world):
    searcher = world.fill([chunk(1, "Our supply-chain costs rose."), chunk(2, FILLER)])
    assert searcher.search("supply chain", HERE, 5)[0].chunk.chunk_id.endswith("0001:0")


def test_an_exact_figure_is_found(world):
    searcher = world.fill(
        [chunk(1, "Net sales | $ | 11,692.6"), chunk(2, "Net sales | $ | 10,000.0")]
    )
    found = searcher.search("11,692.6", HERE, 5)
    assert found[0].chunk.chunk_id.endswith("0001:0")
    assert found[0].keyword_rank == 1
    assert found[1].keyword_rank is None


def test_the_rarer_word_decides_the_keyword_ranking(world):
    searcher = world.fill(
        [
            chunk(1, "costs costs costs in the quarter"),
            chunk(2, "freight costs in the quarter"),
            *[chunk(number, "costs in the quarter") for number in range(3, 10)],
        ]
    )
    found = {item.chunk.chunk_id[-6:]: item for item in searcher.search("freight costs", HERE, 9)}
    assert found["0002:0"].keyword_rank == 1


def test_repeating_a_common_word_does_not_swamp_a_rare_one(world):
    # The defect that ruled out rank_bm25's BM25L: its score grows with the raw word count.
    searcher = world.fill(
        [
            chunk(1, " ".join(["sales"] * 25)),
            chunk(2, "freight sales in the quarter"),
            *[chunk(number, "sales in the quarter") for number in range(3, 10)],
        ]
    )
    found = {item.chunk.chunk_id[-6:]: item for item in searcher.search("freight sales", HERE, 9)}
    assert found["0002:0"].keyword_rank == 1


# --- Keyword search on very small filters, where the common BM25 variant fails ---


def test_a_single_chunk_filter_still_ranks_its_match(world):
    # As with Mattel's risk factors in a 10-Q, one short statement.
    only = chunk(
        1, "There have been no material changes to the risk factors.", section=Section.RISK_FACTORS
    )
    searcher = world.fill([only, chunk(2, FILLER)])
    where = ChunkFilter(accession="hsy-q2-26", section=Section.RISK_FACTORS)
    found = searcher.search("risk factors", where, 5)
    assert ids(found) == [only.chunk_id]
    assert found[0].keyword_rank == 1


def test_a_term_in_half_of_a_small_filter_still_counts(world):
    searcher = world.fill([chunk(1, "freight costs"), chunk(2, "cocoa prices")])
    found = searcher.search("freight", HERE, 5)
    assert found[0].keyword_rank == 1
    assert found[1].keyword_rank is None


def test_a_term_in_every_chunk_still_counts(world):
    searcher = world.fill(
        [chunk(1, "freight costs"), chunk(2, "freight rates"), chunk(3, "freight")]
    )
    found = searcher.search("freight", HERE, 5)
    assert {item.keyword_rank is not None for item in found} == {True}


def test_a_chunk_with_none_of_the_words_gets_no_keyword_rank(world):
    searcher = world.fill([chunk(1, "freight costs"), chunk(2, "cocoa prices")])
    found = {item.chunk.chunk_id[-6:]: item for item in searcher.search("freight", HERE, 5)}
    assert found["0002:0"].keyword_rank is None
    # Dense search still returns it. The reranker can drop it, and a paraphrase may need it.
    assert found["0002:0"].dense_rank is not None


# --- Filters hold for both kinds of search ---


@pytest.fixture
def mixed() -> dict[str, Chunk]:
    text = "Freight costs rose sharply."
    return {
        "q2": chunk(1, text),
        "q2_risk": chunk(2, text, section=Section.RISK_FACTORS),
        "q1": chunk(3, text, accession="hsy-q1-26", fiscal_period="Q1"),
        "fy": chunk(
            4, text, accession="hsy-fy-25", form=FormType.FORM_10K, year=2025, fiscal_period="FY"
        ),
        "other_company": chunk(5, text, accession="mat-q2-26", ticker="MAT"),
    }


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        (ChunkFilter(ticker="MAT"), {"other_company"}),
        (ChunkFilter(accession="hsy-q2-26"), {"q2", "q2_risk"}),
        (ChunkFilter(accession="hsy-q2-26", section=Section.RISK_FACTORS), {"q2_risk"}),
        (ChunkFilter(ticker="HSY", period="Q1 2026"), {"q1"}),
        (ChunkFilter(ticker="HSY", form=FormType.FORM_10K), {"fy"}),
        (ChunkFilter(ticker="HSY", section=Section.MDNA), {"q2", "q1", "fy"}),
        (ChunkFilter(ticker="SWK"), set()),
        (ChunkFilter(accession="hsy-q2-26", section=Section.BUSINESS), set()),
    ],
)
def test_nothing_outside_the_filter_comes_back(world, mixed, where, expected):
    searcher = world.fill(list(mixed.values()))
    found = searcher.search("freight", where, 20)
    assert set(ids(found)) == {mixed[name].chunk_id for name in expected}
    # Both searches found each one, so neither strayed outside the filter on its own.
    assert all(item.keyword_rank and item.dense_rank for item in found)


def test_a_filter_with_no_chunks_returns_nothing_rather_than_raising(world):
    searcher = world.fill([chunk(1, FILLER)])
    assert searcher.search("freight", ChunkFilter(ticker="NOPE"), 5) == []


def test_filters_match_what_the_vector_index_would_allow(mixed):
    assert ChunkFilter(ticker="HSY").matches(mixed["q1"])
    assert not ChunkFilter(ticker="MAT").matches(mixed["q1"])
    assert not ChunkFilter(ticker="HSY", period="Q2 2026").matches(mixed["q1"])
    assert ChunkFilter(section=Section.RISK_FACTORS).matches(mixed["q2_risk"])
    assert not ChunkFilter(form=FormType.FORM_10K).matches(mixed["q2"])


# --- Fusion ---


def test_fusion_matches_the_formula_by_hand():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_agreement_between_lists_beats_one_first_place():
    # "b" is second in both. "a" and "c" are each first in only one.
    assert reciprocal_rank_fusion([["a", "b"], ["c", "b"]])[0][0] == "b"


def test_equal_scores_break_the_same_way_every_time():
    fused = [name for name, _ in reciprocal_rank_fusion([["z"], ["m"], ["a"]])]
    assert fused == ["a", "m", "z"]


def test_empty_lists_fuse_to_nothing():
    assert reciprocal_rank_fusion([[], []]) == []


def test_candidates_record_the_score_and_both_ranks(settings):
    first, second = chunk(1, "freight costs"), chunk(2, "cocoa prices")
    vectors = {first.text: [0.0, 1.0], second.text: [1.0, 0.0]}
    world = World(settings, TableEmbedder(vectors, query=[1.0, 0.0]))
    try:
        found = world.fill([first, second]).search("freight", HERE, 5)
    finally:
        world.close()
    by_id = {item.chunk.chunk_id: item for item in found}
    # Keyword search ranks the first alone. Dense search ranks the second, then the first.
    assert (by_id[first.chunk_id].keyword_rank, by_id[first.chunk_id].dense_rank) == (1, 2)
    assert by_id[first.chunk_id].score == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 2))
    assert (by_id[second.chunk_id].keyword_rank, by_id[second.chunk_id].dense_rank) == (None, 1)
    assert by_id[second.chunk_id].score == pytest.approx(1 / (RRF_K + 1))


def test_top_k_limits_the_results_and_the_order_is_best_first(world):
    searcher = world.fill([chunk(number, f"freight {'word ' * number}") for number in range(1, 30)])
    found = searcher.search("freight", HERE, 20)
    assert len(found) == 20
    scores = [item.score for item in found]
    assert scores == sorted(scores, reverse=True)


def test_each_list_stops_at_its_size(world):
    searcher = world.fill([chunk(number, "freight") for number in range(RANKED_LIST_SIZE + 10)])
    found = searcher.search("freight", HERE, RANKED_LIST_SIZE + 10)
    assert max(item.keyword_rank for item in found if item.keyword_rank) == RANKED_LIST_SIZE
    assert max(item.dense_rank for item in found if item.dense_rank) == RANKED_LIST_SIZE


def test_the_same_search_gives_the_same_order(world):
    searcher = world.fill([chunk(number, f"freight costs item {number}") for number in range(12)])
    assert ids(searcher.search("freight", HERE, 12)) == ids(searcher.search("freight", HERE, 12))


def test_a_query_with_no_words_falls_back_on_dense_search(world):
    searcher = world.fill([chunk(1, FILLER)])
    found = searcher.search("???", HERE, 5)
    assert found[0].keyword_rank is None
    assert found[0].dense_rank == 1


# --- Keeping store and index honest ---


def test_a_stale_index_entry_fails_loudly(world):
    searcher = world.fill([chunk(1, FILLER), chunk(2, FILLER)])
    world.store.replace_chunks("hsy-q2-26", [])
    fresh = HybridSearcher(world.store, world.index)
    with pytest.raises(IndexOutOfSyncError, match="not in the store"):
        fresh.search("sales", HERE, 5)
    assert searcher is not fresh


def test_a_keyword_index_is_built_once_per_filter(world):
    searcher = world.fill([chunk(1, "freight")])
    searcher.search("freight", HERE, 5)
    built = searcher._scopes[HERE]
    searcher.search("cocoa", HERE, 5)
    assert searcher._scopes[HERE] is built


# --- Tokens ---


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("Freight COSTS", ["freight", "cost"]),
        ("tariffs", ["tariff"]),
        ("taxes", ["tax"]),
        ("subsidiaries", ["subsidiary"]),
        ("business analysis status", ["business", "analysis", "status"]),
        ("gas has", ["gas", "has"]),
        ("supply-chain", ["supply", "chain"]),
        ("11,692.6 and 3.2%", ["11,692.6", "and", "3.2"]),
        ("(151,372)", ["151,372"]),
        ("10-K", ["10", "k"]),
        ("", []),
    ],
)
def test_tokens(text, tokens):
    assert tokenize(text) == tokens


# --- The real filings ---


@pytest.fixture(scope="module")
def real():
    settings = get_settings()
    with (
        FactStore(settings.duckdb_path, read_only=True) as store,
        ChunkIndex(settings.chroma_dir, BgeEmbedder()) as index,
    ):
        if index.count() == 0:
            pytest.skip("The index has not been built.")
        yield HybridSearcher(store, index), store, Corpus.load(REPO_ROOT / "config" / "corpus.yaml")


@pytest.mark.corpus
@pytest.mark.parametrize("term", ["freight", "tariff", "cocoa"])
def test_real_exact_terms_surface_in_every_filing_that_uses_them(real, term):
    searcher, store, corpus = real
    checked = 0
    for company in corpus.companies:
        for pinned in company.filings:
            with_term = {
                item.chunk_id
                for item in store.chunks(pinned.accession)
                if term in tokenize(item.text)
            }
            if not with_term:
                continue
            checked += 1
            found = searcher.search(term, ChunkFilter(accession=pinned.accession), 20)
            assert with_term & set(ids(found)), (term, pinned.accession)
            assert all(item.chunk.accession == pinned.accession for item in found)
    assert checked


@pytest.mark.corpus
def test_real_search_stays_inside_section_and_filing(real):
    searcher, _, corpus = real
    accession = corpus.company("HSY").filings[0].accession
    found = searcher.search(
        "gross margin decline", ChunkFilter(accession=accession, section=Section.MDNA), 20
    )
    assert len(found) == 20
    assert {(item.chunk.accession, item.chunk.section) for item in found} == {
        (accession, Section.MDNA)
    }
