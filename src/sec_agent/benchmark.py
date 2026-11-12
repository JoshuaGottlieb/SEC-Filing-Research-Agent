"""Benchmark labels for evaluation (T6.1).

The labels are drafted from the filings' indexed text, separately from the agent: no retrieval,
alignment, materiality or Tree-of-Thought code produces them. The user then confirms every label.
`eval/benchmark/README.md` documents the method. Five kinds of item serve the evaluations:

- `RetrievalQuestion`: a query and every passage that answers it in each filing (T6.2 recall@5,
  the reranker's `MIN_SCORE`, T2.4 redundancy). A side with no passage means the topic is absent
  from that filing, which is what tunes the score cutoff.
- `LabeledChange`: a material change between two filings (T6.3 missed changes).
- `StatedCause`: a change whose cause the filing states, with plausible causes it does not state
  (T6.4 survival of the known cause in the beam, T6.5).
- `JudgeExample`: a claim, a quote and the verdict the entailment judge should give (T5.2).
- `SampledChange`: a text change drawn at random, with a fixed seed, from T3.1's alignment, labeled
  material or not (T6.3 false positives).

Every passage is a verbatim quote, not a chunk id, so labels survive re-chunking. The resolver
finds the chunks that contain each quote with the citation checker's own matching, so a label
quote that resolves is one the agent's checker would also accept. Hershey is the tuning set;
Mattel and Stanley are held out for the final numbers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

# The citation checker's normalization and boundary-aware match (T5.1). Shared on purpose: a
# private import, but a second copy could drift and let a label pass that the checker refuses.
from sec_agent.agents.verification import _appears_in, _normalize_quote
from sec_agent.schemas import (
    ChangeKind,
    Chunk,
    Comparison,
    Direction,
    EntailmentVerdict,
    MaterialityCriterion,
    MetricName,
    Record,
    Section,
)

# Thresholds may be tuned on these companies only; the rest give the reported results.
TUNING_TICKERS = frozenset({"HSY"})


class Split(StrEnum):
    TUNE = "tune"
    TEST = "test"


class Side(StrEnum):
    # Which of a comparison's two filings a quote comes from.
    CURRENT = "current"
    PRIOR = "prior"


class Quote(Record):
    """Verbatim text from one filing. `chunk_id` is the drafter's hint; the resolver decides."""

    side: Side
    text: str = Field(min_length=1)
    chunk_id: str | None = None


class Item(Record):
    """Fields every label shares. The ticker comes from the comparison id (`HSY:yoy-10k`)."""

    item_id: str = Field(min_length=1)
    comparison_id: str = Field(pattern=r"^[A-Z]+:(yoy-10k|yoy-10q|seq-10q)$")
    note: str | None = None

    @property
    def ticker(self) -> str:
        return self.comparison_id.split(":", 1)[0]

    @property
    def split(self) -> Split:
        return Split.TUNE if self.ticker in TUNING_TICKERS else Split.TEST


class RetrievalQuestion(Item):
    """A query and every passage that answers it. `section` is the filter a caller would pass;
    None searches the whole filing."""

    section: Section | None = None
    question: str = Field(min_length=1)
    quotes: tuple[Quote, ...] = Field(min_length=1)

    def absent_sides(self) -> tuple[Side, ...]:
        """The filings where the topic does not appear: the search there should come back empty."""
        present = {quote.side for quote in self.quotes}
        return tuple(side for side in Side if side not in present)


class LabeledChange(Item):
    """A material change. Added text has only current quotes, removed text only prior ones."""

    section: Section
    kind: Literal[ChangeKind.ADDED, ChangeKind.REMOVED, ChangeKind.MODIFIED]
    criterion: MaterialityCriterion
    summary: str = Field(min_length=1)
    quotes: tuple[Quote, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _sides_match_kind(self) -> LabeledChange:
        sides = {quote.side for quote in self.quotes}
        expected = {
            ChangeKind.ADDED: {Side.CURRENT},
            ChangeKind.REMOVED: {Side.PRIOR},
            ChangeKind.MODIFIED: {Side.CURRENT, Side.PRIOR},
        }[self.kind]
        if sides != expected:
            raise ValueError(f"a {self.kind.value} change needs quotes from {sorted(expected)}")
        return self


class StatedCause(Item):
    """A change whose cause the current filing states. `distractors` are plausible causes the
    filings do not state; the agent should not label them confirmed."""

    section: Section
    change: str = Field(min_length=1)
    metric: MetricName | None = None
    direction: Direction | None = None
    causes: tuple[str, ...] = Field(min_length=1)
    quote: Quote
    distractors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _quote_is_current(self) -> StatedCause:
        if self.quote.side is not Side.CURRENT:
            raise ValueError("a stated cause is quoted from the current filing")
        if (self.metric is None) != (self.direction is None):
            raise ValueError("metric and direction go together")
        return self


class JudgeExample(Item):
    """What the entailment judge (T5.2) should say about `claim` given only `quote`."""

    cause_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    quote: Quote
    verdict: EntailmentVerdict


class SampledChange(Item):
    """A T3.1 text change drawn at random. `material` is None until the user labels it."""

    change_id: str = Field(min_length=1)
    kind: Literal[ChangeKind.ADDED, ChangeKind.REMOVED, ChangeKind.MODIFIED]
    section: Section | None = None
    current_text: str | None = None
    prior_text: str | None = None
    material: bool | None = None


class Benchmark(Record):
    """One label file. Drafts and the frozen set share this shape."""

    retrieval: tuple[RetrievalQuestion, ...] = ()
    changes: tuple[LabeledChange, ...] = ()
    causes: tuple[StatedCause, ...] = ()
    judge: tuple[JudgeExample, ...] = ()
    sampled: tuple[SampledChange, ...] = ()

    def items(self) -> Iterable[Item]:
        yield from self.retrieval
        yield from self.changes
        yield from self.causes
        yield from self.judge
        yield from self.sampled

    @model_validator(mode="after")
    def _ids_unique_and_linked(self) -> Benchmark:
        seen: set[str] = set()
        for item in self.items():
            if item.item_id in seen:
                raise ValueError(f"duplicate item id {item.item_id}")
            seen.add(item.item_id)
        cause_ids = {cause.item_id for cause in self.causes}
        for example in self.judge:
            if example.cause_id not in cause_ids:
                raise ValueError(f"{example.item_id} names unknown cause {example.cause_id}")
        return self


def load_benchmark(path: Path) -> Benchmark:
    return Benchmark.model_validate_json(path.read_text(encoding="utf-8"))


def save_benchmark(benchmark: Benchmark, path: Path) -> None:
    data = benchmark.model_dump(mode="json", exclude_none=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def matching_chunks(text: str, chunks: Iterable[Chunk]) -> tuple[str, ...]:
    """Ids of the chunks containing `text` under the citation checker's matching. Chunks overlap
    by about 10%, so a quote near a boundary can sit in two; both count as the passage."""
    needle = _normalize_quote(text)
    return tuple(c.chunk_id for c in chunks if _appears_in(needle, _normalize_quote(c.text)))


ChunkSource = Callable[[str], list[Chunk]]  # accession -> that filing's chunks


def resolve_quote(
    quote: Quote, comparison: Comparison, chunks: ChunkSource, section: Section | None = None
) -> tuple[str, ...]:
    """The chunks of the quoted filing (within `section`, if given) that contain the quote."""
    filing = comparison.current if quote.side is Side.CURRENT else comparison.prior
    pool = [c for c in chunks(filing.accession) if section is None or c.section == section]
    return matching_chunks(quote.text, pool)


def check_benchmark(
    benchmark: Benchmark, comparisons: Mapping[str, Comparison], chunks: ChunkSource
) -> list[str]:
    """Every problem that would make a label unusable, as readable lines; empty when clean.

    A quote must appear in its filing; a retrieval quote must also lie inside the question's
    section filter, or the search could never return it.
    """
    problems: list[str] = []

    def check(item: Item, quote: Quote, section: Section | None) -> None:
        comparison = comparisons.get(item.comparison_id)
        if comparison is None:
            problems.append(f"{item.item_id}: unknown comparison {item.comparison_id}")
            return
        if not resolve_quote(quote, comparison, chunks, section):
            where = f" in {section.value}" if section else ""
            problems.append(
                f"{item.item_id}: {quote.side.value} quote not found{where}: {quote.text[:80]!r}"
            )

    for question in benchmark.retrieval:
        for quote in question.quotes:
            check(question, quote, question.section)
    for change in benchmark.changes:
        for quote in change.quotes:
            check(change, quote, None)
    for cause in benchmark.causes:
        check(cause, cause.quote, None)
    for example in benchmark.judge:
        check(example, example.quote, None)
    for sampled in benchmark.sampled:
        if sampled.comparison_id not in comparisons:
            problems.append(f"{sampled.item_id}: unknown comparison {sampled.comparison_id}")
    return problems
