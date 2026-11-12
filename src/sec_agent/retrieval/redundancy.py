"""Redundancy in search results (T2.4): are a search's top passages near-copies of each other?

The plan adds maximal marginal relevance (MMR) to the reranker only if redundancy is common: "more
than 2 of the top 5 exceed 0.95 similarity". This module measures that on labelled questions and
changes no search. One real source of near-copies: a 10-Q can repeat a sentence for the quarter
and for the year to date (Mattel's Q2 2026 MD&A does, word for word).

A passage is redundant when its cosine similarity to a passage ranked above it in the same
filing's list exceeds `NEAR_DUPLICATE`. Each filing's list is measured on its own: the two filings
are meant to say similar things, and a search always returns both. A list is redundant when more
than `MAX_REDUNDANT` of its passages are. How many redundant lists justify MMR is the user's call
on the numbers; this module only counts.

Similarity is measured on the vectors stored in the index (`ChunkIndex.vectors`), the ones dense
search compares, so the measurement sees what the search saw.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from pydantic import Field

from sec_agent.contracts import Retriever
from sec_agent.schemas import Comparison, Record, RetrievedPassage, Section

# Both from the plan's T2.4 row: "more than 2 of the top 5 exceed 0.95 similarity".
NEAR_DUPLICATE = 0.95
MAX_REDUNDANT = 2

VectorSource = Callable[[Sequence[str]], Mapping[str, Sequence[float]]]  # chunk ids -> vectors


class RedundantPair(Record):
    """A passage and the higher-ranked passage it nearly copies."""

    kept: str  # chunk id of the higher-ranked passage
    redundant: str
    similarity: float


class SideRedundancy(Record):
    """One filing's list for one question."""

    passages: int = Field(ge=0)
    # Passages whose similarity to a higher-ranked one exceeds NEAR_DUPLICATE.
    redundant: int = Field(ge=0)
    # The highest similarity between any two passages in the list; None under two passages.
    max_similarity: float | None = None
    pairs: tuple[RedundantPair, ...] = ()

    @property
    def is_redundant(self) -> bool:
        return self.redundant > MAX_REDUNDANT


class RedundancyQuestion(Record):
    """A query to measure, as a caller would search it."""

    question_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    comparison: Comparison
    section: Section | None = None


class QuestionRedundancy(Record):
    question_id: str
    query: str
    current: SideRedundancy
    prior: SideRedundancy


class RedundancyReport(Record):
    questions: tuple[QuestionRedundancy, ...]

    def lists(self) -> list[SideRedundancy]:
        """Every filing's list that returned at least one passage."""
        return [
            side
            for question in self.questions
            for side in (question.current, question.prior)
            if side.passages
        ]

    def redundant_lists(self) -> list[SideRedundancy]:
        return [side for side in self.lists() if side.is_redundant]

    def redundant_counts(self) -> dict[int, int]:
        """How many lists had 0, 1, 2, ... redundant passages."""
        counts: dict[int, int] = {}
        for side in self.lists():
            counts[side.redundant] = counts.get(side.redundant, 0) + 1
        return dict(sorted(counts.items()))


def side_redundancy(
    passages: Sequence[RetrievedPassage], vectors: Mapping[str, Sequence[float]]
) -> SideRedundancy:
    """Measure one ranked list. Every passage's chunk must have a vector."""
    ids = [passage.chunk.chunk_id for passage in passages]
    missing = [chunk_id for chunk_id in ids if chunk_id not in vectors]
    if missing:
        raise ValueError(f"no stored vector for {missing}")
    if len(ids) < 2:
        return SideRedundancy(passages=len(ids), redundant=0)
    matrix = np.asarray([vectors[chunk_id] for chunk_id in ids], dtype=float)
    # Normalized again so the dot product is the cosine whatever the index stored.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1.0, norms)
    similarity = matrix @ matrix.T
    pairs: list[RedundantPair] = []
    for lower in range(1, len(ids)):
        above = similarity[lower, :lower]
        best = int(np.argmax(above))
        if above[best] > NEAR_DUPLICATE:
            pairs.append(
                RedundantPair(kept=ids[best], redundant=ids[lower], similarity=float(above[best]))
            )
    upper = similarity[np.triu_indices(len(ids), k=1)]
    return SideRedundancy(
        passages=len(ids),
        redundant=len(pairs),
        max_similarity=float(upper.max()),
        pairs=tuple(pairs),
    )


async def measure_redundancy(
    questions: Sequence[RedundancyQuestion], retriever: Retriever, vectors: VectorSource
) -> RedundancyReport:
    """Search each question as a caller would and measure both filings' lists."""
    measured: list[QuestionRedundancy] = []
    for question in questions:
        result = await retriever.retrieve(question.query, question.comparison, question.section)
        found = vectors([p.chunk.chunk_id for p in (*result.current, *result.prior)])
        measured.append(
            QuestionRedundancy(
                question_id=question.question_id,
                query=question.query,
                current=side_redundancy(result.current, found),
                prior=side_redundancy(result.prior, found),
            )
        )
    return RedundancyReport(questions=tuple(measured))
