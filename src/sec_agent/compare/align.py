"""Paragraph alignment between the two filings of a comparison (T3.1).

For each section the two filings share, the later filing's paragraphs are matched one to one with
the earlier filing's, and each pair is labeled added, removed, modified or unchanged. Only the
first three become `Change` records, so the materiality step (an Opus call per change, most of
the run's cost) never sees text that was carried forward.

Matching has two steps. Paragraphs whose text is identical in both filings pair off first, since
much of a filing repeats the previous one word for word. The rest are embedded and matched with
`linear_sum_assignment` on cosine similarity, which finds the pairing with the highest total
similarity rather than a greedy one. When two pairings tie, as they do when a filing repeats a
paragraph word for word, the one keeping passages at closer relative positions wins, so the
result never depends on float noise. A pair below a similarity floor is not a match: the earlier
paragraph was removed and the later one added.

A first-quarter 10-Q reports each figure once, since its quarter is also its year to date, while a
later 10-Q reports it twice, for the quarter and for the year to date. Compared quarter to
quarter, the first-quarter paragraph belongs with the later quarter's paragraph, but it reads
closer to the year-to-date one ("first quarter" against "first six months" or "first half"), by
up to 0.086 on the three pinned comparisons. So in such a comparison a pairing of a quarter
paragraph with a year-to-date one is matched at a penalty (`PERIOD_MISMATCH_PENALTY`); the floor
and the recorded similarity still use the unpenalized value.

A matched pair is unchanged ("boilerplate") when the only differences are the ones a filing makes
when its text is carried forward: punctuation and typography, years moved on by the gap between
the two filings, the month of the period end, and the day of a period-end date. Any other
difference, however small, is a modification. A text-similarity cutoff was tried and rejected: on
Hershey, "two" becoming "four", "could" becoming "can" and a new chief executive's name all score
98 or more out of 100.

The section's status decides whether there is anything to compare. A 10-Q that states "no
material changes" reports exactly that, not every earlier paragraph as removed, and a section that
only points elsewhere in the filing (a cross-reference) has no paragraphs to compare.

Hershey's notes carry a running page header, "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS—
(Continued)", as a paragraph of its own, and a page break there can split a sentence in two. The
breaks fall in different places in different years, so the header paragraphs are dropped and a
sentence split around one is joined again before matching. On Hershey's three comparisons this
removed about 17 false changes out of 235 in the notes. It is done here rather than in the parser
so the shared store need not be reloaded. Headings are not used at all: in the same filings the
running header "THE HERSHEY COMPANY" replaces the real heading of many paragraphs.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import Field, model_validator
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from sec_agent.ingest.index import Embedder
from sec_agent.ingest.pipeline import ParsedFiling, load_validated_filing, parsed_filing_path
from sec_agent.ingest.sections import Paragraph, ParsedSection, SectionStatus
from sec_agent.schemas import Change, ChangeKind, Comparison, Record, Section

# The running header of Hershey's notes, which the parser keeps as a paragraph of its own. The
# pattern takes the audited and unaudited forms and any dash before "(Continued)". No other
# pinned company has such a paragraph.
RUNNING_HEADER = re.compile(
    r"NOTES TO (?:UNAUDITED )?(?:CONDENSED )?CONSOLIDATED FINANCIAL STATEMENTS\W*\(CONTINUED\)",
    re.IGNORECASE,
)
# A paragraph ending in one of these finished its sentence, so it is never joined to the next.
SENTENCE_END = re.compile(r"[.:;!?)\]\"”’]\s*$")
# How the second half of a split sentence starts: in lowercase, or with a figure or date carried
# over ("31, 2022", "$79,136"). All 13 breaks found in the pinned Hershey notes start this way.
CONTINUATION_START = re.compile(r"^[a-z0-9$(]")
# Below this cosine similarity a pair the assignment made is not a match. On Hershey's three
# comparisons every forced pair under 0.72 joined unrelated paragraphs and every pair from 0.76
# up shared a topic, so the floor sits in that gap. Kept by the user on 2026-11-12 after the
# Hershey benchmark labels: the 7 labeled material changes' passages pair at 0.86 to 0.995, or sit
# inside a modified paragraph, and the one removed passage pairs at 0.64. Any floor from 0.65 to
# 0.85 gives all 7 the same result, so the labels do not move it.
MIN_MATCH_SIMILARITY = 0.75
# Similarities rounded to this step before matching, so pairings whose totals differ by less are
# a tie, which the passages' relative positions decide (see `_assign`). Chosen by hand: far above
# the rounding error in summing a section's similarities, and far below any gap that tells two
# texts apart (on the nine pinned comparisons it changed only an exact tie).
SIMILARITY_RESOLUTION = 1e-6
# Taken off a pairing of a quarter paragraph with a year-to-date one when the earlier filing is a
# first-quarter 10-Q (see the module docstring). On the three pinned comparisons it moved 39
# first-quarter paragraphs to the later quarter's paragraph on the same topic and paired nothing
# unrelated; 3 with a larger pull stay with the year to date. At 0.1 two unrelated Mattel
# paragraphs were paired, and 0.02 left about half the pulled paragraphs where they were.
# Found on Hershey's benchmark item HSY-F14.
PERIOD_MISMATCH_PENALTY = 0.05
# A paragraph's period is read from its opening, where the first sentence names it; later
# mentions often refer to another period. 40 words chosen by hand.
PERIOD_WORDS = 40
_QUARTER = re.compile(
    # "three and six months" names both periods, so it matches this as well as `_YEAR_TO_DATE`.
    r"\b(?:first|second|third|fourth) quarter\b|\bthree (?:and (?:six|nine) )?months\b"
    r"|\bquarter ended\b",
    re.IGNORECASE,
)
_YEAR_TO_DATE = re.compile(
    r"\b(?:six|nine) months\b|\bfirst half\b|\byear[- ]to[- ]date\b", re.IGNORECASE
)

# Typography that changes between years without changing what the text says.
_TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE = re.compile(rf"\b({_MONTHS}) (\d{{1,2}}), ((?:19|20)\d\d)\b")
_YEAR = re.compile(r"(?:19|20)\d\d")
# Words and figures. Punctuation is left out, so a comma added to a list is not a change.
_WORD = re.compile(r"\w+")


class PairLabel(StrEnum):
    """What happened to a paragraph between the two filings."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class SectionOutcome(StrEnum):
    """Whether a section could be compared paragraph by paragraph, and if not, why."""

    COMPARED = "compared"
    # The later filing states that nothing material changed (a 10-Q's risk factors).
    NO_MATERIAL_CHANGES = "no_material_changes"
    # A filing only points elsewhere for this section, so there is no text to compare.
    CROSS_REFERENCE = "cross_reference"
    # One filing has no such section. Missing text is not removed text, so nothing is reported.
    MISSING = "missing"


class Passage(Record):
    """Text to align: one paragraph, or a sentence that a page break split, joined again."""

    paragraph_ids: tuple[str, ...] = Field(min_length=1)
    text: str


class AlignedPair(Record):
    """A later passage, an earlier passage, or both, and what happened between them."""

    label: PairLabel
    current: Passage | None = None
    prior: Passage | None = None
    # The cosine similarity of the two passages when the assignment matched them. None when the
    # texts were identical (they are never embedded) and for a passage without a partner.
    similarity: float | None = None

    @model_validator(mode="after")
    def _sides_match_label(self) -> AlignedPair:
        has_current, has_prior = self.current is not None, self.prior is not None
        expected = {
            PairLabel.ADDED: (True, False),
            PairLabel.REMOVED: (False, True),
            PairLabel.MODIFIED: (True, True),
            PairLabel.UNCHANGED: (True, True),
        }[self.label]
        if (has_current, has_prior) != expected:
            raise ValueError(f"a {self.label.value} pair has the wrong passages")
        return self


class SectionAlignment(Record):
    """The aligned paragraphs of one section, or why it could not be aligned."""

    section: Section
    outcome: SectionOutcome
    # None when the filing has no such section at all.
    current_status: SectionStatus | None
    prior_status: SectionStatus | None
    pairs: tuple[AlignedPair, ...] = ()

    def count(self, label: PairLabel) -> int:
        return sum(pair.label is label for pair in self.pairs)


class ComparisonAlignment(Record):
    """Every section of a comparison, and the text changes found in them."""

    comparison_id: str
    sections: tuple[SectionAlignment, ...]
    changes: tuple[Change, ...]


class ParagraphChunks(Protocol):
    """Finds the chunks holding a paragraph's text. `FactStore` provides it."""

    def chunk_ids_for_paragraph(self, paragraph_id: str) -> tuple[str, ...]: ...


class PeriodRoll(Record):
    """How the later filing's dates follow from the earlier one's, to spot carried-forward text."""

    # The years later text may have moved on by: the gap in fiscal years (1 year over year, 0 for
    # consecutive quarters) and the gap in the calendar years the periods end in. They differ for
    # a 52/53-week year: Stanley's FY2025 ended January 3, 2026, its FY2024 December 28, 2024.
    year_shifts: frozenset[int]
    # The months the two periods end in, such as March and June for consecutive quarters.
    prior_month: str
    current_month: str
    # The earlier filing is a first-quarter 10-Q and the later one a later quarter's, so the
    # earlier has one paragraph where the later has a quarter and a year-to-date one (see
    # `period_kind`).
    prior_quarter_is_year_to_date: bool = False

    @classmethod
    def of(cls, comparison: Comparison) -> PeriodRoll:
        current, prior = comparison.current.period, comparison.prior.period
        return cls(
            # Only a 10-Q has a first quarter, and both filings of a comparison share a form.
            prior_quarter_is_year_to_date=(
                prior.fiscal_period == "Q1" and current.fiscal_period != "Q1"
            ),
            year_shifts=frozenset(
                {
                    current.fiscal_year - prior.fiscal_year,
                    current.period_end.year - prior.period_end.year,
                }
            ),
            prior_month=prior.period_end.strftime("%B"),
            current_month=current.period_end.strftime("%B"),
        )

    def rolls(self, old: str, new: str) -> bool:
        """True when replacing the earlier word with the later one is only the period moving."""
        if _YEAR.fullmatch(old) and _YEAR.fullmatch(new):
            return int(new) - int(old) in self.year_shifts - {0}
        return self.prior_month != self.current_month and (old, new) == (
            self.prior_month,
            self.current_month,
        )

    def drops_day(self, month: str) -> bool:
        # The day of a date in either period-end month is dropped, because period ends move by a
        # day or two between years (June 29, 2025 against June 28, 2026), and a later filing's
        # year-earlier date can fall in the other period's month (Stanley's July 4, 2026 quarter
        # compares with June 28, 2025). Dates in other months are events and keep their day.
        return month in (self.prior_month, self.current_month)


def normalize(text: str) -> str:
    """The text with typography and spacing made uniform, for comparing two years' wording."""
    return " ".join(unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY).split())


def _words(text: str, roll: PeriodRoll) -> list[str]:
    def drop_day(match: re.Match[str]) -> str:
        return f"{match[1]} {match[3]}" if roll.drops_day(match[1]) else match[0]

    return _WORD.findall(_DATE.sub(drop_day, normalize(text)))


def is_carried_forward(prior_text: str, current_text: str, roll: PeriodRoll) -> bool:
    """True when the later text is the earlier one with only its dates moved on.

    A year that moved by exactly the gap between the filings counts as moved on even when it
    names an event, so an event's year changing by that gap is missed. That is the price of
    filtering the many paragraphs that only roll their dates forward.
    """
    old = _words(prior_text, roll)
    new = _words(current_text, roll)
    for op in Levenshtein.opcodes(old, new):
        if op.tag == "equal":
            continue
        removed, added = old[op.src_start : op.src_end], new[op.dest_start : op.dest_end]
        if op.tag != "replace" or len(removed) != len(added):
            return False
        if not all(roll.rolls(a, b) for a, b in zip(removed, added, strict=True)):
            return False
    return True


def prepare_passages(paragraphs: Sequence[Paragraph]) -> tuple[Passage, ...]:
    """The paragraphs to align, without running headers and with split sentences joined.

    A sentence is joined only across a running header, because the header proves a page break
    lies between the two parts. Without one, a paragraph that ends without punctuation is as
    likely to introduce a list, and joining those made as many false changes as it removed.
    """
    passages: list[Passage] = []
    after_header = False
    for paragraph in paragraphs:
        text = paragraph.text.strip()
        if RUNNING_HEADER.fullmatch(text):
            after_header = True
            continue
        if not text:
            continue
        previous = passages[-1] if passages else None
        if (
            after_header
            and previous is not None
            and not SENTENCE_END.search(previous.text)
            and CONTINUATION_START.match(text)
        ):
            passages[-1] = Passage(
                paragraph_ids=(*previous.paragraph_ids, paragraph.paragraph_id),
                text=f"{previous.text} {text}",
            )
        else:
            passages.append(Passage(paragraph_ids=(paragraph.paragraph_id,), text=text))
        after_header = False
    return tuple(passages)


class PeriodKind(StrEnum):
    QUARTER = "quarter"
    YEAR_TO_DATE = "year_to_date"


def period_kind(text: str) -> PeriodKind | None:
    """The period a paragraph reports, from its opening; None when it names neither or both
    ("for the three and six months ended")."""
    opening = " ".join(text.split()[:PERIOD_WORDS])
    quarter, year_to_date = bool(_QUARTER.search(opening)), bool(_YEAR_TO_DATE.search(opening))
    if quarter == year_to_date:
        return None
    return PeriodKind.QUARTER if quarter else PeriodKind.YEAR_TO_DATE


def _period_penalty(current: Sequence[str], prior: Sequence[str]) -> np.ndarray:
    """`PERIOD_MISMATCH_PENALTY` where one text reports a quarter and the other a year to date."""
    current_kinds = [period_kind(text) for text in current]
    prior_kinds = [period_kind(text) for text in prior]
    return np.array(
        [
            [
                PERIOD_MISMATCH_PENALTY if a is not None and b is not None and a != b else 0.0
                for b in prior_kinds
            ]
            for a in current_kinds
        ]
    )


def _similarities(embedder: Embedder, current: Sequence[str], prior: Sequence[str]) -> np.ndarray:
    # One call embeds both sides, so the model batches them together. Each distinct text is
    # embedded once: two copies can fall in different batches inside the model and come back a
    # few bits apart, and those bits would then decide a tie between the copies. Vectors are
    # normalized again here so the dot product is the cosine whatever the embedder returns.
    texts = list(dict.fromkeys([*current, *prior]))
    where = {text: k for k, text in enumerate(texts)}
    vectors = np.asarray(embedder.embed_passages(texts), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1.0, norms)
    return vectors[[where[text] for text in current]] @ vectors[[where[text] for text in prior]].T


def _assign(
    similarity: np.ndarray, current_positions: np.ndarray, prior_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The pairing with the highest total similarity, and on a tie the one whose passages sit
    closest in relative position (0 at a section's start, 1 at its end).

    A tie is real when a filing repeats a paragraph, say once for the quarter and once for the
    year to date: swapping the copies' partners leaves the total unchanged, and without this the
    solver's internal order would decide which partner each copy gets.
    """
    units = np.round(similarity / SIMILARITY_RESOLUTION)
    distance = np.abs(current_positions[:, None] - prior_positions[None, :])
    # At most this many pairs are made, each distance is at most 1, so the distance term sums to
    # less than one unit and only decides between pairings whose rounded totals are equal.
    pairs = min(similarity.shape)
    return linear_sum_assignment(units - distance / (pairs + 1), maximize=True)


def align_passages(
    current: Sequence[Passage],
    prior: Sequence[Passage],
    embedder: Embedder,
    roll: PeriodRoll,
) -> tuple[AlignedPair, ...]:
    """Pair two sections' passages and label each pair.

    The pairs follow the later filing's order, then the earlier filing's passages that have no
    partner, in its order.
    """
    current_keys = [normalize(passage.text) for passage in current]
    prior_keys = [normalize(passage.text) for passage in prior]

    # Identical text pairs off first, in document order.
    waiting: dict[str, deque[int]] = defaultdict(deque)
    for j, key in enumerate(prior_keys):
        waiting[key].append(j)
    by_current: dict[int, AlignedPair] = {}
    by_prior: dict[int, AlignedPair] = {}
    for i, key in enumerate(current_keys):
        if waiting[key]:
            j = waiting[key].popleft()
            by_current[i] = AlignedPair(
                label=PairLabel.UNCHANGED, current=current[i], prior=prior[j]
            )
            by_prior[j] = by_current[i]

    # A surplus copy of a text the other filing also has, such as a footer repeated on more
    # pages, is unchanged too: the text is in both filings, only how often differs.
    first_current = {key: i for i, key in reversed(list(enumerate(current_keys)))}
    first_prior = {key: j for j, key in reversed(list(enumerate(prior_keys)))}
    for i, key in enumerate(current_keys):
        if i not in by_current and key in first_prior:
            by_current[i] = AlignedPair(
                label=PairLabel.UNCHANGED, current=current[i], prior=prior[first_prior[key]]
            )
    surplus_prior: dict[int, AlignedPair] = {}
    for j, key in enumerate(prior_keys):
        if j not in by_prior and key in first_current:
            surplus_prior[j] = AlignedPair(
                label=PairLabel.UNCHANGED, current=current[first_current[key]], prior=prior[j]
            )
    by_prior.update(surplus_prior)

    current_rest = [i for i in range(len(current)) if i not in by_current]
    prior_rest = [j for j in range(len(prior)) if j not in by_prior]
    if current_rest and prior_rest:
        current_texts = [current[i].text for i in current_rest]
        prior_texts = [prior[j].text for j in prior_rest]
        similarity = _similarities(embedder, current_texts, prior_texts)
        matching = similarity
        if roll.prior_quarter_is_year_to_date:
            matching = similarity - _period_penalty(current_texts, prior_texts)
        rows, columns = _assign(
            matching,
            (np.asarray(current_rest) + 0.5) / len(current),
            (np.asarray(prior_rest) + 0.5) / len(prior),
        )
        for row, column in zip(rows, columns, strict=True):
            score = float(similarity[row, column])
            if score < MIN_MATCH_SIMILARITY:
                continue
            i, j = current_rest[row], prior_rest[column]
            carried = is_carried_forward(prior[j].text, current[i].text, roll)
            pair = AlignedPair(
                label=PairLabel.UNCHANGED if carried else PairLabel.MODIFIED,
                current=current[i],
                prior=prior[j],
                similarity=score,
            )
            by_current[i] = by_prior[j] = pair

    pairs = [
        by_current[i] if i in by_current else AlignedPair(label=PairLabel.ADDED, current=passage)
        for i, passage in enumerate(current)
    ]
    for j, passage in enumerate(prior):
        if j in surplus_prior:
            pairs.append(surplus_prior[j])
        elif j not in by_prior:
            pairs.append(AlignedPair(label=PairLabel.REMOVED, prior=passage))
    return tuple(pairs)


def section_outcome(current: ParsedSection | None, prior: ParsedSection | None) -> SectionOutcome:
    """Whether a section can be compared, from what the parser found in each filing."""
    current_status = current.status if current else SectionStatus.MISSING
    prior_status = prior.status if prior else SectionStatus.MISSING
    # The later filing's own statement wins: it says nothing material changed since the 10-K,
    # whatever the earlier quarter said.
    if current_status is SectionStatus.NO_MATERIAL_CHANGES:
        return SectionOutcome.NO_MATERIAL_CHANGES
    if SectionStatus.MISSING in (current_status, prior_status):
        return SectionOutcome.MISSING
    if SectionStatus.CROSS_REFERENCE in (current_status, prior_status):
        return SectionOutcome.CROSS_REFERENCE
    # The later filing has text. If the earlier one said "no material changes", every later
    # paragraph is new disclosure, so it is compared against nothing and reported as added.
    return SectionOutcome.COMPARED


def align_section(
    section: Section,
    current: ParsedSection | None,
    prior: ParsedSection | None,
    embedder: Embedder,
    roll: PeriodRoll,
) -> SectionAlignment:
    """Align one section of the two filings, or record why it can't be aligned."""
    outcome = section_outcome(current, prior)
    pairs: tuple[AlignedPair, ...] = ()
    if outcome is SectionOutcome.COMPARED and current is not None and prior is not None:
        prior_passages = (
            prepare_passages(prior.paragraphs) if prior.status is SectionStatus.CONTENT else ()
        )
        pairs = align_passages(prepare_passages(current.paragraphs), prior_passages, embedder, roll)
    return SectionAlignment(
        section=section,
        outcome=outcome,
        current_status=current.status if current else None,
        prior_status=prior.status if prior else None,
        pairs=pairs,
    )


def align_filings(
    comparison: Comparison,
    current: ParsedFiling,
    prior: ParsedFiling,
    embedder: Embedder,
) -> tuple[SectionAlignment, ...]:
    """Align every section either filing has, in the order of `Section`."""
    if (current.accession, prior.accession) != (
        comparison.current.accession,
        comparison.prior.accession,
    ):
        raise ValueError(f"the parsed filings are not those of {comparison.comparison_id}")
    roll = PeriodRoll.of(comparison)
    return tuple(
        align_section(
            section, current.sections.get(section), prior.sections.get(section), embedder, roll
        )
        for section in Section
        # A section neither filing has, such as a 10-Q's business section, isn't reported.
        if section in current.sections or section in prior.sections
    )


def _chunk_ids(chunks: ParagraphChunks, passage: Passage | None) -> tuple[str, ...]:
    if passage is None:
        return ()
    found: dict[str, None] = {}
    for paragraph_id in passage.paragraph_ids:
        # A chunk usually spans several paragraphs, so each paragraph is looked up, not only the
        # chunk's first one.
        found.update(dict.fromkeys(chunks.chunk_ids_for_paragraph(paragraph_id)))
    return tuple(found)


def changes_for(
    comparison: Comparison, sections: Iterable[SectionAlignment], chunks: ParagraphChunks
) -> tuple[Change, ...]:
    """A `Change` for every added, removed or modified passage, citing the chunks that hold it.

    The id names the comparison, the kind and the passage's first paragraph (the later one, or the
    earlier one for a removal), so it is stable from run to run and leads back to the text.
    """
    changes: list[Change] = []
    for alignment in sections:
        for pair in alignment.pairs:
            if pair.label is PairLabel.UNCHANGED:
                continue
            kind = ChangeKind(pair.label.value)
            anchor = pair.current or pair.prior
            assert anchor is not None  # every pair has at least one side (see AlignedPair)
            changes.append(
                Change(
                    change_id=f"{comparison.comparison_id}:{kind.value}:{anchor.paragraph_ids[0]}",
                    comparison_id=comparison.comparison_id,
                    kind=kind,
                    section=alignment.section,
                    current_text=pair.current.text if pair.current else None,
                    prior_text=pair.prior.text if pair.prior else None,
                    current_chunk_ids=_chunk_ids(chunks, pair.current),
                    prior_chunk_ids=_chunk_ids(chunks, pair.prior),
                )
            )
    return tuple(changes)


def align_comparison(
    comparison: Comparison,
    parsed_dir: Path,
    embedder: Embedder,
    chunks: ParagraphChunks,
) -> ComparisonAlignment:
    """Align a comparison's two filings and list the text changes between them.

    Filings are read only through `load_validated_filing`, so a rejected parse never reaches
    change detection. `chunks` is read on the calling thread.
    """
    current = load_validated_filing(
        parsed_filing_path(parsed_dir, comparison.ticker, comparison.current.accession)
    )
    prior = load_validated_filing(
        parsed_filing_path(parsed_dir, comparison.ticker, comparison.prior.accession)
    )
    sections = align_filings(comparison, current, prior, embedder)
    return ComparisonAlignment(
        comparison_id=comparison.comparison_id,
        sections=sections,
        changes=changes_for(comparison, sections, chunks),
    )
