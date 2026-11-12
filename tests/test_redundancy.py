"""T2.4's redundancy measurement: near-copies in one filing's ranked list, counted per question."""

from __future__ import annotations

import math
from datetime import date

import pytest

from sec_agent.retrieval.redundancy import (
    MAX_REDUNDANT,
    NEAR_DUPLICATE,
    RedundancyQuestion,
    measure_redundancy,
    side_redundancy,
)
from sec_agent.schemas import (
    Chunk,
    Comparison,
    ComparisonType,
    FiscalPeriod,
    FormType,
    RetrievalResult,
    RetrievedPassage,
    Section,
)

CURRENT = "0000047111-26-000002"
PRIOR = "0000047111-25-000002"


@pytest.fixture
def comparison(make_filing) -> Comparison:
    current = make_filing(form=FormType.FORM_10Q, year=2026, fiscal_period="Q2")
    prior = make_filing(form=FormType.FORM_10Q, year=2025, fiscal_period="Q2")
    return Comparison(
        comparison_id="HSY:yoy-10q",
        ticker="HSY",
        comparison_type=ComparisonType.YOY_10Q,
        current=current.model_copy(update={"accession": CURRENT}),
        prior=prior.model_copy(update={"accession": PRIOR}),
    )


def passage(number: int, accession: str = CURRENT) -> RetrievedPassage:
    year = 2026 if accession == CURRENT else 2025
    return RetrievedPassage(
        chunk=Chunk(
            chunk_id=f"{accession}:mdna:{number:04d}:0",
            paragraph_id=f"{accession}:mdna:{number:04d}",
            ticker="HSY",
            cik="0000047111",
            form=FormType.FORM_10Q,
            period=FiscalPeriod(fiscal_year=year, fiscal_period="Q2", period_end=date(year, 6, 30)),
            section=Section.MDNA,
            accession=accession,
            text=f"passage {number}",
            token_count=2,
        ),
        score=1.0,
    )


def at(degrees: float, scale: float = 1.0) -> list[float]:
    # A 2-D vector at this angle: the cosine of two of them is the cosine of the angle between.
    radians = math.radians(degrees)
    return [scale * math.cos(radians), scale * math.sin(radians)]


def vectors_for(passages: list[RetrievedPassage], angles: list[float]) -> dict[str, list[float]]:
    return {p.chunk.chunk_id: at(a) for p, a in zip(passages, angles, strict=True)}


def test_each_near_copy_of_a_higher_passage_counts_once():
    # 0 and 1 are copies, 2 and 3 nearly so (5 degrees: cosine 0.996), 4 stands apart.
    passages = [passage(n) for n in range(5)]
    side = side_redundancy(passages, vectors_for(passages, [0, 0, 60, 65, 120]))
    assert (side.passages, side.redundant) == (5, 2)
    assert [(p.kept, p.redundant) for p in side.pairs] == [
        (passages[0].chunk.chunk_id, passages[1].chunk.chunk_id),
        (passages[2].chunk.chunk_id, passages[3].chunk.chunk_id),
    ]
    assert side.max_similarity == pytest.approx(1.0)
    assert not side.is_redundant  # 2 is not more than MAX_REDUNDANT


def test_more_than_two_near_copies_make_the_list_redundant():
    passages = [passage(n) for n in range(5)]
    side = side_redundancy(passages, vectors_for(passages, [0, 1, 2, 3, 90]))
    assert side.redundant == 3 > MAX_REDUNDANT
    assert side.is_redundant


def test_a_passage_copying_several_above_is_one_redundant_passage_paired_with_its_nearest():
    passages = [passage(n) for n in range(3)]
    side = side_redundancy(passages, vectors_for(passages, [0, 10, 9]))
    assert side.redundant == 2
    # Passage 2 is nearer passage 1 (1 degree) than passage 0 (9 degrees).
    assert side.pairs[1].kept == passages[1].chunk.chunk_id
    assert side.pairs[1].similarity == pytest.approx(math.cos(math.radians(1)))
    # The highest similarity between two different passages, never a passage with itself.
    assert side.max_similarity == pytest.approx(math.cos(math.radians(1)))


@pytest.mark.parametrize(("delta", "redundant"), [(0.001, 1), (-0.001, 0), (0.0, 0)])
def test_the_threshold_is_strictly_above_near_duplicate(delta, redundant):
    passages = [passage(0), passage(1)]
    angle = math.degrees(math.acos(NEAR_DUPLICATE + delta))
    vectors = {passages[0].chunk.chunk_id: at(0), passages[1].chunk.chunk_id: at(angle)}
    if delta == 0.0:
        # Exactly at the threshold, up to rounding: not above it.
        vectors[passages[1].chunk.chunk_id] = [NEAR_DUPLICATE, math.sqrt(1 - NEAR_DUPLICATE**2)]
    assert side_redundancy(passages, vectors).redundant == redundant


def test_vectors_are_normalized_before_comparing():
    passages = [passage(0), passage(1)]
    vectors = {passages[0].chunk.chunk_id: at(0, 10.0), passages[1].chunk.chunk_id: at(0, 0.5)}
    side = side_redundancy(passages, vectors)
    assert side.redundant == 1 and side.max_similarity == pytest.approx(1.0)


def test_lists_under_two_passages_have_nothing_to_compare():
    one = [passage(0)]
    assert side_redundancy(one, vectors_for(one, [0])).model_dump() == {
        "passages": 1,
        "redundant": 0,
        "max_similarity": None,
        "pairs": (),
    }
    assert side_redundancy([], {}).passages == 0


def test_a_passage_without_a_stored_vector_is_refused():
    passages = [passage(0), passage(1)]
    with pytest.raises(ValueError, match="no stored vector"):
        side_redundancy(passages, {passages[0].chunk.chunk_id: at(0)})


class FakeRetriever:
    def __init__(self, results: dict[str, tuple[list[int], list[int]]]) -> None:
        self.results = results
        self.searches: list[tuple[str, Section | None]] = []

    async def retrieve(self, query, comparison, section=None) -> RetrievalResult:
        self.searches.append((query, section))
        current, prior = self.results[query]
        return RetrievalResult(
            query=query,
            current=tuple(passage(n) for n in current),
            prior=tuple(passage(n, PRIOR) for n in prior),
        )


async def test_each_question_is_searched_as_asked_and_both_filings_measured(comparison):
    angles = {
        **{passage(n).chunk.chunk_id: at(a) for n, a in [(0, 0), (1, 0), (2, 1), (3, 2)]},
        # The earlier filing says the same as the later one; that is not redundancy.
        **{passage(n, PRIOR).chunk.chunk_id: at(a) for n, a in [(0, 0), (1, 90)]},
    }
    asked: list[list[str]] = []

    def vectors(ids):
        asked.append(list(ids))
        return {i: angles[i] for i in ids}

    retriever = FakeRetriever({"freight": ([0, 1, 2, 3], [0, 1]), "cocoa": ([], [0])})
    questions = [
        RedundancyQuestion(
            question_id="q1", query="freight", comparison=comparison, section=Section.MDNA
        ),
        RedundancyQuestion(question_id="q2", query="cocoa", comparison=comparison),
    ]
    report = await measure_redundancy(questions, retriever, vectors)
    assert retriever.searches == [("freight", Section.MDNA), ("cocoa", None)]
    assert len(asked) == 2 and len(asked[0]) == 6  # one lookup per question, both filings
    q1, q2 = report.questions
    assert (q1.current.redundant, q1.prior.redundant) == (3, 0)
    assert (q2.current.passages, q2.prior.passages) == (0, 1)
    # Lists that returned nothing are not measured lists.
    assert len(report.lists()) == 3
    assert report.redundant_lists() == [q1.current]
    assert report.redundant_counts() == {0: 2, 3: 1}
