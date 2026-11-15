"""Back-references: a filing explaining one line's change "by the same factors" as another's.

Hershey never states a cause for its operating margin directly. Every one of its five filings says
it moved "by the same factors noted above in gross margin" (or "the same factors that resulted in
lower gross margin"), and the factors are in the gross margin paragraph above. The critic never
quotes such a sentence, since it names no cause, and the judge rules on one quote at a time, so on
the third paid Hershey run no operating margin claim could be confirmed, though the 10-K states
the cause (2026-11-15). The user's decision that day (option A): when the filing explains the
change this way, the judge sees that sentence together with the paragraph it points to.

This module finds the pair, in code and from the filing alone, so it does not depend on what the
critic chose to quote. Both parts are verbatim passages of the current filing's MD&A, so the
citation checker can verify them like any other quote:

- the sentence names the changed line (or one that moves with it), gives its current and prior
  values as the filing rounds them ("12.3% in 2025 from 25.9% in 2024"), and says "same factors"
  before naming another line;
- the paragraph is the nearest one above it that names that other line, gives that line's current
  and prior values, and uses causal wording.

Both values are required so the pair is about the comparison's own two periods. A 10-Q's MD&A
compares the quarter with the same quarter a year before, so for a sequential comparison its
"23.1% in 2026 from 7.4% in 2025" explains a different change and must not match.

Only ratio metrics are matched: their values are written one way ("33.5%"), while dollar figures
are written many ("$1,441.5 million", "$1.4 billion"). Labeling decides what to do with the pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from sec_agent.agents.verification import find_causal_language
from sec_agent.contracts import MetricLookup
from sec_agent.finance.metrics import metric_id_for
from sec_agent.schemas import Change, Chunk, Citation, Comparison, MetricName, Section
from sec_agent.tot.critic import linked_metrics
from sec_agent.tot.lines import METRIC_WORDS, lines_named

# How a filing points back to another line's explanation. From Hershey's five filings: "by the same
# factors noted above in gross margin", "driven by the same factors noted above that resulted in
# lower gross margin", "driven by the same factors that resulted in higher gross margin". Chosen
# by hand; "as noted above" alone is left out, since Hershey uses it after a cause it has just
# stated ("due to lower gross profit and higher SM&A expenses, as noted above").
SAME_FACTORS = re.compile(r"\bsame factors\b", re.IGNORECASE)

# Where a sentence ends: after a full stop, question or exclamation mark followed by space, so
# "12.3%" and "$1,441.5" stay whole. An abbreviation such as "U.S. imports" splits a sentence early,
# which can only make a match fail, never invent one. Chosen by hand.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# The parser separates paragraphs with a blank line and writes table cells with " | " (as
# `labeling.TABLE_CELL`). A table row is not an explanation paragraph.
PARAGRAPH_BREAK = "\n\n"
TABLE_CELL = " | "

# Shown to the judge between the two passages, as a quotation marks a cut.
OMITTED = "[...]"

# The contract's unit for a ratio metric (`MetricLookup`).
RATIO = "ratio"


class FilingChunks(Protocol):
    """One filing's chunks in reading order. `FactStore` provides it."""

    def chunks(self, accession: str | None = None) -> list[Chunk]: ...


@dataclass(frozen=True)
class BackReference:
    """A sentence explaining the change by another line's factors, and the paragraph giving them."""

    # The other line's explanation, the paragraph the sentence points back to.
    referenced: Citation
    # The sentence that explains the change "by the same factors".
    sentence: Citation
    line: MetricName

    @property
    def citations(self) -> tuple[Citation, Citation]:
        """Both passages, in reading order: what a claim confirmed on them cites."""
        return (self.referenced, self.sentence)

    @property
    def judged_text(self) -> str:
        """What the judge reads: both passages in reading order, the cut between them marked."""
        return f"{self.referenced.quote}\n\n{OMITTED}\n\n{self.sentence.quote}"


def shown_percent(value: float) -> str:
    """A ratio as filings write it, to one decimal place: 0.1233 is "12.3%"."""
    return f"{value * 100:.1f}%"


def _paragraphs(chunk: Chunk) -> list[str]:
    return [part.strip() for part in chunk.text.split(PARAGRAPH_BREAK) if part.strip()]


def _referenced_line(tail: str, changed: MetricName) -> MetricName | None:
    """The first line `tail` names that does not move with `changed`, or None."""
    lowered = tail.casefold()
    linked = linked_metrics(changed)
    found = [
        (lowered.find(word), name)
        for name, words in METRIC_WORDS.items()
        if name not in linked
        for word in words
        if word in lowered
    ]
    return min(found)[1] if found else None


def _shown_values(
    comparison: Comparison, line: MetricName, metrics: MetricLookup
) -> tuple[str, str] | None:
    """`line`'s current and prior values as filings write them, or None if either is not a ratio."""
    shown = []
    for filing in (comparison.current, comparison.prior):
        value = metrics.get_metric(metric_id_for(filing, line))
        if value is None or value.unit != RATIO:
            return None
        shown.append(shown_percent(value.value))
    return shown[0], shown[1]


def _gives(text: str, values: tuple[str, str]) -> bool:
    return all(value in text for value in values)


def _explains(paragraph: str, line: MetricName, values: tuple[str, str]) -> bool:
    """Whether `paragraph` gives a cause for `line` moving between `values`."""
    return (
        line in lines_named(paragraph)
        and _gives(paragraph, values)
        and TABLE_CELL not in paragraph
        and not SAME_FACTORS.search(paragraph)
        and bool(find_causal_language(paragraph))
    )


def _citation(chunk: Chunk, quote: str, comparison: Comparison) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        accession=chunk.accession,
        ticker=chunk.ticker,
        form=chunk.form,
        period=chunk.period,
        section=chunk.section,
        quote=quote,
        url=comparison.current.url,
    )


def find_back_reference(
    change: Change,
    comparison: Comparison,
    changed: MetricName,
    metrics: MetricLookup,
    filing: FilingChunks,
) -> BackReference | None:
    """The first back-reference in the current filing's MD&A that explains `change`, or None.

    `changed` is the change's metric (`controller.changed_metric`). None for a text change, a
    dollar metric, or a filing that explains the change some other way.
    """
    if change.metric_id is None:
        return None
    values = _shown_values(comparison, changed, metrics)
    if values is None:
        return None
    mdna = [
        chunk
        for chunk in filing.chunks(comparison.current.accession)
        if chunk.section is Section.MDNA
    ]
    # Every paragraph in reading order, so "noted above" can be looked up among those before.
    passages = [(chunk, paragraph) for chunk in mdna for paragraph in _paragraphs(chunk)]
    linked = linked_metrics(changed)
    for at, (chunk, paragraph) in enumerate(passages):
        for sentence in SENTENCE_END.split(paragraph):
            match = SAME_FACTORS.search(sentence)
            if match is None:
                continue
            head = sentence[: match.start()]
            if not _gives(head, values) or not lines_named(head) & linked:
                continue
            line = _referenced_line(sentence[match.end() :], changed)
            if line is None:
                continue
            line_values = _shown_values(comparison, line, metrics)
            if line_values is None:
                continue
            above = next(
                (
                    (source, text)
                    for source, text in reversed(passages[:at])
                    if _explains(text, line, line_values)
                ),
                None,
            )
            if above is not None:
                return BackReference(
                    referenced=_citation(above[0], above[1], comparison),
                    sentence=_citation(chunk, sentence, comparison),
                    line=line,
                )
    return None
