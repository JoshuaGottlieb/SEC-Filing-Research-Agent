"""Reranking: the cutoff makes a period empty, both periods are always searched, scores stay raw."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import pytest

from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore, SourcedChunk
from sec_agent.ingest.index import BgeEmbedder, ChunkFilter, ChunkIndex
from sec_agent.retrieval.hybrid import FusedCandidate, HybridSearcher, tokenize
from sec_agent.retrieval.rerank import (
    CANDIDATES_PER_PERIOD,
    KEEP_PER_PERIOD,
    MIN_SCORE,
    CrossEncoderScorer,
    Reranker,
)
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


def chunk(
    number: int,
    text: str,
    *,
    accession: str = CURRENT,
    section: Section = Section.MDNA,
) -> Chunk:
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


def candidates(chunks: Sequence[Chunk]) -> list[FusedCandidate]:
    return [
        FusedCandidate(chunk=item, score=1 / (60 + rank), dense_rank=rank)
        for rank, item in enumerate(chunks, start=1)
    ]


class FakeSearcher:
    """Returns the candidates it was given for a filter's accession, and remembers each call."""

    def __init__(self, by_accession: dict[str, list[Chunk]]) -> None:
        self.by_accession = by_accession
        self.calls: list[tuple[str, ChunkFilter, int]] = []

    def search(self, query: str, where: ChunkFilter, top_k: int) -> list[FusedCandidate]:
        self.calls.append((query, where, top_k))
        return candidates(self.by_accession.get(where.accession, [])[:top_k])


class TableScorer:
    """Scores a passage by lookup on its text, so a test dictates the cross-encoder's verdicts."""

    def __init__(self, scores: dict[str, float], default: float = -11.0) -> None:
        self.scores = scores
        self.default = default
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return [self.scores.get(passage, self.default) for passage in passages]


def texts(passages) -> list[str]:
    return [passage.chunk.text for passage in passages]


# --- Keeping, ordering and dropping ---


def test_keeps_the_top_five_best_first_with_raw_scores():
    pool = [chunk(n, f"passage {n}") for n in range(1, 21)]
    # Scores run from -4 down to -23: negative, and far outside any 0 to 1 range, so a sigmoid
    # sneaking in anywhere would change them.
    scores = {item.text: -4.0 - n for n, item in enumerate(pool)}
    shuffled = pool[::-1]
    reranker = Reranker(FakeSearcher({CURRENT: shuffled}), TableScorer(scores), min_score=-100)

    kept = reranker.rerank("q", ChunkFilter(accession=CURRENT))

    assert texts(kept) == [f"passage {n}" for n in range(1, 6)]
    assert [passage.score for passage in kept] == [-4.0, -5.0, -6.0, -7.0, -8.0]


def test_reads_twenty_candidates_and_keeps_five_by_default():
    searcher = FakeSearcher({CURRENT: [chunk(n, f"passage {n}") for n in range(1, 31)]})
    scorer = TableScorer({}, default=1.0)
    reranker = Reranker(searcher, scorer, min_score=0)

    kept = reranker.rerank("q", ChunkFilter(accession=CURRENT))

    assert (CANDIDATES_PER_PERIOD, KEEP_PER_PERIOD) == (20, 5)
    assert searcher.calls[0][2] == 20
    assert len(scorer.calls[0][1]) == 20
    assert len(kept) == 5


def test_a_passage_scoring_exactly_the_minimum_is_kept():
    pool = [chunk(1, "at the line"), chunk(2, "just under")]
    scorer = TableScorer({"at the line": -6.0, "just under": -6.0001})
    reranker = Reranker(FakeSearcher({CURRENT: pool}), scorer, min_score=-6.0)

    assert texts(reranker.rerank("q", ChunkFilter(accession=CURRENT))) == ["at the line"]


def test_the_default_minimum_is_minus_six():
    pool = [chunk(1, "kept"), chunk(2, "dropped")]
    scorer = TableScorer({"kept": -5.9, "dropped": -6.1})
    reranker = Reranker(FakeSearcher({CURRENT: pool}), scorer)

    assert MIN_SCORE == -6.0
    assert texts(reranker.rerank("q", ChunkFilter(accession=CURRENT))) == ["kept"]


def test_fewer_than_five_pass_so_fewer_are_returned():
    pool = [chunk(n, f"passage {n}") for n in range(1, 11)]
    scorer = TableScorer({"passage 3": 2.0, "passage 7": 1.0})
    reranker = Reranker(FakeSearcher({CURRENT: pool}), scorer, min_score=0)

    assert texts(reranker.rerank("q", ChunkFilter(accession=CURRENT))) == [
        "passage 3",
        "passage 7",
    ]


def test_equal_scores_keep_the_hybrid_order():
    pool = [chunk(n, f"passage {n}") for n in (4, 2, 9)]
    reranker = Reranker(FakeSearcher({CURRENT: pool}), TableScorer({}, default=1.0), min_score=0)

    assert texts(reranker.rerank("q", ChunkFilter(accession=CURRENT))) == [
        "passage 4",
        "passage 2",
        "passage 9",
    ]


# --- Nothing relevant: the point of the cutoff ---


def test_nothing_above_the_cutoff_gives_an_empty_period_and_not_an_error():
    pool = [chunk(n, f"passage {n}") for n in range(1, 21)]
    reranker = Reranker(FakeSearcher({CURRENT: pool}), TableScorer({}), min_score=-6.0)

    assert reranker.rerank("freight", ChunkFilter(accession=CURRENT)) == ()


def test_a_filter_with_no_chunks_is_an_empty_period_and_the_scorer_is_not_called():
    scorer = TableScorer({})
    reranker = Reranker(FakeSearcher({}), scorer, min_score=-6.0)

    assert reranker.rerank("q", ChunkFilter(accession=CURRENT)) == ()
    assert scorer.calls == []


# --- Both periods, always ---


def test_both_periods_are_searched_each_on_its_own_filing():
    searcher = FakeSearcher(
        {
            CURRENT: [chunk(1, "current text")],
            PRIOR: [chunk(1, "prior text", accession=PRIOR)],
        }
    )
    scorer = TableScorer({}, default=1.0)
    result = Reranker(searcher, scorer, min_score=0).search("gross margin", COMPARISON)

    assert texts(result.current) == ["current text"]
    assert texts(result.prior) == ["prior text"]
    assert not result.incomplete
    assert [call[1].accession for call in searcher.calls] == [CURRENT, PRIOR]
    assert result.query == "gross margin"


def test_an_empty_current_period_does_not_stop_the_prior_search():
    searcher = FakeSearcher({PRIOR: [chunk(1, "prior text", accession=PRIOR)]})
    scorer = TableScorer({"prior text": 1.0})
    result = Reranker(searcher, scorer, min_score=0).search("q", COMPARISON)

    assert result.current == ()
    assert texts(result.prior) == ["prior text"]
    assert result.incomplete
    assert [call[1].accession for call in searcher.calls] == [CURRENT, PRIOR]


def test_one_period_failing_the_cutoff_makes_the_result_incomplete():
    searcher = FakeSearcher(
        {
            CURRENT: [chunk(1, "relevant")],
            PRIOR: [chunk(1, "irrelevant", accession=PRIOR)],
        }
    )
    scorer = TableScorer({"relevant": 3.0, "irrelevant": -9.0})
    result = Reranker(searcher, scorer, min_score=-6.0).search("q", COMPARISON)

    assert texts(result.current) == ["relevant"]
    assert result.prior == ()
    assert result.incomplete


def test_neither_period_matching_is_an_empty_result_and_not_an_error():
    searcher = FakeSearcher({CURRENT: [chunk(1, "a")], PRIOR: [chunk(1, "b", accession=PRIOR)]})
    result = Reranker(searcher, TableScorer({}), min_score=-6.0).search("q", COMPARISON)

    assert (result.current, result.prior) == ((), ())
    assert result.incomplete


def test_the_section_limits_both_searches():
    searcher = FakeSearcher({})
    Reranker(searcher, TableScorer({}), min_score=0).search("q", COMPARISON, Section.RISK_FACTORS)

    assert [call[1].section for call in searcher.calls] == [Section.RISK_FACTORS] * 2


def test_no_section_searches_the_whole_filing():
    searcher = FakeSearcher({})
    Reranker(searcher, TableScorer({}), min_score=0).search("q", COMPARISON)

    assert [call[1].section for call in searcher.calls] == [None, None]


# --- Bad input and bad scorers ---


@pytest.mark.parametrize("query", ["", "   "])
def test_a_blank_query_is_refused_before_any_search(query):
    searcher = FakeSearcher({})
    with pytest.raises(ValueError, match="query"):
        Reranker(searcher, TableScorer({}), min_score=0).search(query, COMPARISON)
    assert searcher.calls == []


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_minimum_score_must_be_a_finite_number(bad):
    with pytest.raises(ValueError, match="finite"):
        Reranker(FakeSearcher({}), TableScorer({}), min_score=bad)


@pytest.mark.parametrize("keep, candidates", [(0, 20), (21, 20)])
def test_keep_must_fit_within_the_candidates(keep, candidates):
    with pytest.raises(ValueError, match="keep"):
        Reranker(FakeSearcher({}), TableScorer({}), min_score=0, keep=keep, candidates=candidates)


def test_a_scorer_returning_the_wrong_number_of_scores_is_an_error():
    class Short:
        def score(self, query, passages):
            return [1.0]

    searcher = FakeSearcher({CURRENT: [chunk(1, "a"), chunk(2, "b")]})
    with pytest.raises(ValueError, match="1 scores for 2"):
        Reranker(searcher, Short(), min_score=0).rerank("q", ChunkFilter(accession=CURRENT))


def test_a_nan_score_is_an_error_and_not_a_silent_pass_or_drop():
    class Broken:
        def score(self, query, passages):
            return [math.nan for _ in passages]

    searcher = FakeSearcher({CURRENT: [chunk(1, "a")]})
    with pytest.raises(ValueError, match="finite"):
        Reranker(searcher, Broken(), min_score=-6.0).rerank("q", ChunkFilter(accession=CURRENT))


# --- Through the real hybrid search, in a temporary store ---


class SameVectorEmbedder:
    """Every text gets one vector, so dense search fills its list with chunks it can't rank."""

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class KeywordScorer:
    """Stands in for the cross-encoder: relevant if the passage uses a word of the query."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        wanted = set(tokenize(query))
        return [5.0 if wanted & set(tokenize(passage)) else -9.0 for passage in passages]


@pytest.fixture
def hybrid(settings):
    store = FactStore(settings.duckdb_path)
    index = ChunkIndex(settings.chroma_dir, SameVectorEmbedder())
    pool = [
        chunk(1, "Freight costs rose on higher fuel prices."),
        chunk(2, "Net sales increased during the quarter."),
        chunk(3, "Selling expenses were flat."),
        chunk(1, "Freight rates eased in the prior year.", accession=PRIOR),
        chunk(2, "Net sales were lower in the prior year.", accession=PRIOR),
        chunk(1, "Freight in the risk factors.", section=Section.RISK_FACTORS),
    ]
    by_filing: dict[str, list[SourcedChunk]] = {}
    for item in pool:
        by_filing.setdefault(item.accession, []).append(
            SourcedChunk(chunk=item, paragraph_ids=(item.paragraph_id,))
        )
    for accession, sourced in by_filing.items():
        store.replace_chunks(accession, sourced)
    index.sync(pool)
    yield HybridSearcher(store, index)
    store.close()
    index.close()


def test_through_hybrid_search_each_period_gets_only_its_own_filing(hybrid):
    result = Reranker(hybrid, KeywordScorer(), min_score=0).search("freight", COMPARISON)

    assert {item.chunk.accession for item in result.current} == {CURRENT}
    assert {item.chunk.accession for item in result.prior} == {PRIOR}
    assert not result.incomplete


def test_through_hybrid_search_the_section_filter_holds(hybrid):
    result = Reranker(hybrid, KeywordScorer(), min_score=0).search(
        "freight", COMPARISON, Section.MDNA
    )

    assert texts(result.current) == ["Freight costs rose on higher fuel prices."]
    assert texts(result.prior) == ["Freight rates eased in the prior year."]


def test_through_hybrid_search_a_word_in_no_chunk_is_incomplete_though_dense_fills_the_list(
    hybrid,
):
    # Dense search returns every chunk of the filing here, so only the cutoff can say "no match".
    result = Reranker(hybrid, KeywordScorer(), min_score=0).search("tariff", COMPARISON)

    assert (result.current, result.prior) == ((), ())
    assert result.incomplete


# --- The real model and the real filings ---

MODEL_TOKEN_LIMIT = 512
# The query plus the two separator tokens and the leading one that the model adds around a pair.
PAIR_SPECIAL_TOKENS = 3


@pytest.fixture(scope="module")
def real():
    settings = get_settings()
    with (
        FactStore(settings.duckdb_path, read_only=True) as store,
        ChunkIndex(settings.chroma_dir, BgeEmbedder()) as index,
    ):
        if index.count() == 0:
            pytest.skip("The index has not been built.")
        searcher = HybridSearcher(store, index)
        reranker = Reranker(searcher, CrossEncoderScorer())
        yield reranker, store, Corpus.load(REPO_ROOT / "config" / "corpus.yaml")


@pytest.mark.corpus
def test_real_freight_is_absent_from_hershey_fy2025_mdna_so_both_periods_come_back_empty(real):
    reranker, store, corpus = real
    comparison = corpus.comparison("HSY", ComparisonType.YOY_10K)
    for side in (comparison.current, comparison.prior):
        mdna = [item for item in store.chunks(side.accession) if item.section is Section.MDNA]
        assert mdna
        # The premise: dense search has plenty to return, and none of it uses the word.
        assert not any("freight" in tokenize(item.text) for item in mdna)

    result = reranker.search("freight", comparison, Section.MDNA)

    assert (result.current, result.prior) == ((), ())
    assert result.incomplete


@pytest.mark.corpus
def test_real_freight_in_mattel_q2_mdna_is_found_in_both_periods(real):
    reranker, _, corpus = real
    comparison = corpus.comparison("MAT", ComparisonType.YOY_10Q)

    result = reranker.search("freight costs", comparison, Section.MDNA)

    assert not result.incomplete
    for passages in (result.current, result.prior):
        assert len(passages) <= KEEP_PER_PERIOD
        assert [item.score for item in passages] == sorted(
            (item.score for item in passages), reverse=True
        )
        assert all(item.score >= MIN_SCORE for item in passages)
        assert any("freight" in tokenize(item.chunk.text) for item in passages)


@pytest.mark.corpus
def test_real_scores_are_raw_logits_and_not_probabilities():
    scorer = CrossEncoderScorer()
    relevant, unrelated = scorer.score(
        "what drove the change in gross margin",
        [
            "Gross margin decreased due to higher commodity costs and tariffs.",
            "Our principal executive offices are located in Hershey, Pennsylvania.",
        ],
    )

    # A sigmoid would put both between 0 and 1. Raw, the unrelated one is far below zero.
    assert relevant > 1.0
    assert unrelated < -5.0


@pytest.mark.corpus
def test_real_chunks_leave_room_for_a_query_within_a_few_tokens(real):
    _, store, _ = real
    # The cross-encoder cuts the end of a passage past 512 tokens. This keeps the loss visible: if
    # the chunker's 500-token cap grows, this fails and the module docstring needs updating.
    longest = max(item.token_count for item in store.chunks())
    query_tokens = 30
    overflow = longest + query_tokens + PAIR_SPECIAL_TOKENS - MODEL_TOKEN_LIMIT
    assert overflow <= 21
