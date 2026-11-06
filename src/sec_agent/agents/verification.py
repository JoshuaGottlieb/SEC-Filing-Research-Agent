"""The checks a claim must pass before it can reach the brief.

Causal language (T5.3): only a confirmed finding may say why something happened. The label
rules decide what a claim has earned; this check makes sure its wording does not claim more.
Without it, a claim labeled plausible could still read "margin fell due to cocoa costs", and
the reader would take the sentence, not the label, as the finding. It is plain code with no
model involved, so it gives the same answer every time and cannot be talked out of it.

Citations (T5.1): a claim's evidence must really exist and really be from the period it is about.
Filings repeat near-identical language across years, so a real quote from the wrong year is the
failure this guards against most. This is also plain code. It proves a quote is in the filing,
not that the quote supports the claim; that is the entailment judge's job (T5.2).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator
from enum import StrEnum
from itertools import combinations
from typing import NamedTuple

from sec_agent.contracts import ChunkLookup, MetricLookup
from sec_agent.ingest.table_facts import SCALE_CAPTION, SCALE_FILLER, _scale_caption
from sec_agent.schemas import (
    Chunk,
    Citation,
    Claim,
    ExplanationLabel,
    FiscalPeriod,
    MetricName,
    MetricValue,
    Record,
)


class CausalPhrase(NamedTuple):
    pattern: str
    # Why this wording is on the list, and any exception the pattern carries.
    reason: str


# The list errs toward flagging. A false alarm costs one rewrite of a claim, while a miss states
# a cause nobody verified. So an exception is added only where the parsed corpus (140 filings,
# counted 2026-11-05) shows the wording is a fixed non-causal phrase, and a hedge or negation
# ("may be due to", "not caused by") is still flagged: it still offers a view on the cause.
#
# Left off on purpose, because each is mostly not a claim of cause, so flagging it would mean
# rewriting sound claims:
#   "offset by"  Arithmetic between two movements; 2,250 uses, the usual form in MD&A.
#   "since", "as"  Too often about time or comparison ("since 2023", "as of").
CAUSAL_PHRASES: tuple[CausalPhrase, ...] = (
    CausalPhrase(r"\bdue to\b", "The plainest statement of cause; 4,351 uses in the corpus."),
    CausalPhrase(r"\bbecause\b", "An explicit cause, and it covers 'because of'."),
    CausalPhrase(
        r"\bas a result(?: of)?\b|\bresult of\b",
        "An explicit cause. Singular only, so 'results of operations' is not flagged.",
    ),
    CausalPhrase(
        r"\bresulted (?:in|from)\b|\bresulting (?:in|from)\b"
        r"|\bresults from\b(?! (?:continuing|discontinued|operations|our|its)\b)",
        "An explicit cause. 'Results in' is left out and 'results from' has an exception because "
        "both are usually the noun: 'results in future periods', 'results from operations'.",
    ),
    CausalPhrase(
        r"\bcaus(?:e|es|ed|ing)\b",
        "The word itself. Bare 'cause' is included for 'could cause'; 'for cause' and 'without "
        "cause' are too rare to need an exception (19 uses).",
    ),
    CausalPhrase(
        r"\bdriven by\b|-driven\b|\bdrove\b|\bdrives\b|\bdriving\b",
        "The usual MD&A wording (1,878 uses of 'driven by'), and 'cost-driven' says the same. "
        "Bare 'drive' is left out because it is also a noun.",
    ),
    CausalPhrase(
        r"\b(?:lead|leads|led|leading) to\b",
        "States that one thing produced another.",
    ),
    CausalPhrase(
        r"(?<!income )(?<!earnings )(?<!loss )(?<!losses )(?<!equity )\battributable to\b"
        r"|\battributed to\b",
        "Assigns a cause. Excepted after income, earnings, loss or equity (193 uses), where it "
        "names an owner: 'net income attributable to common stockholders'.",
    ),
    CausalPhrase(
        r"\bcontribut(?:ed|es|ing) to\b",
        "A partial cause is still a cause, and it is an easy way to hedge past the other phrases.",
    ),
    CausalPhrase(
        r"\b(?:impacted|affected|influenced|pressured|hurt|boosted|fueled|fuelled|spurred"
        r"|triggered|prompted) by\b",
        "A movement credited to something; 904 uses of 'impacted by' and 'affected by' alone.",
    ),
    CausalPhrase(
        r"\bbenefit(?:s|ed|ted|ing)? from\b",
        "Credits a favorable movement to something.",
    ),
    CausalPhrase(
        r"\breflect(?:s|ed|ing) (?:the )?(?:higher|lower|increased?|increases|decreased?"
        r"|decreases|reduced|improved|greater|weaker|stronger)\b",
        "Implies a cause only when it names a movement (136 uses: 'reflecting higher costs'). "
        "Bare 'reflects' is left out: 'reflects the fair value' explains nothing.",
    ),
    CausalPhrase(r"\bin response to\b", "Gives the reason the company acted."),
    CausalPhrase(r"\bexplained by\b", "Names the explanation outright."),
    CausalPhrase(r"\bstem(?:s|med|ming)? from\b", "States an origin."),
    CausalPhrase(
        r"\bconsequently\b|\bconsequence of\b|\bas a consequence\b",
        "A synonym of 'as a result'.",
    ),
    CausalPhrase(
        r"\btherefore\b|\bthus\b(?! far\b)|\bhence\b",
        "Links a fact to its effect: 'costs rose, therefore margin fell'. In filings 59% to 79% of "
        "the 757 uses are not causal (hand count), mostly accounting notes, but a one-sentence "
        "claim rarely needs the word, so a false alarm is a cheap rewrite, and 83% of the causal "
        "uses had no other listed phrase to catch them. 'Thus far' is about time.",
    ),
    # These do not appear in the corpus, so they cost no false alarms there. They are on the list
    # because a model rewording a flagged claim could reach for them.
    CausalPhrase(r"\bowing to\b", "A synonym of 'due to'."),
    CausalPhrase(r"\bthanks to\b", "A synonym of 'due to' for a favorable movement."),
    CausalPhrase(r"\bon account of\b", "A synonym of 'due to'."),
    CausalPhrase(r"\bon the back of\b", "A financial idiom for 'due to'."),
)

_COMPILED = tuple(re.compile(phrase.pattern, re.IGNORECASE) for phrase in CAUSAL_PHRASES)


def find_causal_language(text: str) -> tuple[str, ...]:
    """The causal wording in `text`, lowercased, in order of appearance, without repeats.

    Run this on what the system wrote, not on quotes: a filing's own "due to" in a citation is
    evidence, and the entailment judge decides what it supports.
    """
    # The patterns expect single spaces, and text can wrap mid-phrase. split() also breaks on
    # non-breaking spaces, which filings use often.
    normalized = " ".join(text.split())
    found = [match for pattern in _COMPILED for match in pattern.finditer(normalized)]
    found.sort(key=lambda match: match.start())
    return tuple(dict.fromkeys(match.group().lower() for match in found))


class CausalLanguageResult(Record):
    """Whether a claim's wording is allowed for its label, and which phrases are not."""

    claim_id: str
    flagged_phrases: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        # Derived, never stored, so it cannot disagree with the phrases.
        return not self.flagged_phrases


def check_causal_language(claim: Claim) -> CausalLanguageResult:
    """Fail a claim that uses causal wording, unless it is confirmed (gap 13).

    A confirmed claim may say why, because `Claim` has already required a citation and a
    `states_cause` verdict for it. Only `claim.text` is read; see `find_causal_language`.
    """
    if claim.label is ExplanationLabel.CONFIRMED:
        return CausalLanguageResult(claim_id=claim.claim_id)
    return CausalLanguageResult(
        claim_id=claim.claim_id, flagged_phrases=find_causal_language(claim.text)
    )


# What a model copying a quote may reasonably change without changing the words. Counts are from
# the parsed filings: curly quotes (about 42,000), en and em dashes (about 5,900), non-breaking
# hyphens (424). The minus sign and the other hyphens are the same character for this purpose.
# "|" is how the chunker separates a table's cells: our formatting, not the filing's words.
_QUOTE_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "|": " ",
    }
)

# Tables put a figure's symbol in its own cell: "Net sales | $ | 11,692.6" and "4.4 | %" (6,338 and
# 491 times in the real chunks). A model writes "$11,692.6" and "4.4%". Applied to both sides, so
# this joins a symbol to its own number and cannot make a changed figure match. Never across a line
# break: a row ending "1,234" above a row starting "% of sales" would invent the figure "1,234%".
_SPLIT_SYMBOL = re.compile(r"(?<=[$€£])[^\S\n]+(?=\d)|(?<=[\d)])[^\S\n]+(?=%)")


def _normalize_quote(text: str) -> str:
    joined = _SPLIT_SYMBOL.sub("", text.translate(_QUOTE_TRANSLATION))
    # split() also breaks on non-breaking spaces and the line breaks between a table's rows.
    return " ".join(joined.casefold().split())


def _appears_in(needle: str, haystack: str) -> bool:
    """Whether normalized `needle` appears in normalized `haystack`, cut at no word or number.

    Plain substring matching would accept an altered figure: "5%" is inside "1.5%" and "$1" is
    inside "$1.5 billion". The lookarounds refuse a match that starts or ends partway through a
    word or a number, which also covers "rose" inside "arose".
    """
    if not needle:
        return False
    pattern = rf"(?<!\w)(?<!\d[.,]){re.escape(needle)}(?!\w)(?![.,]\d)"
    return re.search(pattern, haystack) is not None


class CitationFailureKind(StrEnum):
    # The cited chunk does not exist.
    UNKNOWN_CHUNK = "unknown_chunk"
    # The quote is not in the chunk's text.
    ALTERED_QUOTE = "altered_quote"
    # The citation describes the chunk wrongly: its company, filing, form, period or section.
    METADATA_MISMATCH = "metadata_mismatch"
    # The evidence is from a period the claim is not about. A hard gate in the controller (T4.4).
    WRONG_PERIOD = "wrong_period"
    # The claim cites a metric id that does not resolve.
    UNKNOWN_METRIC = "unknown_metric"
    # The claim states a figure, cites no metric, and no cited quote backs it.
    UNTRACED_FIGURE = "untraced_figure"
    # The claim cites metrics, and its figure matches neither them nor a cited quote.
    FIGURE_MISMATCH = "figure_mismatch"
    # The figure's digits are in a cited table, but the table states no scale to confirm it.
    UNKNOWN_SCALE = "unknown_scale"


class CitationFailure(Record):
    """One thing wrong with a claim's evidence.

    `detail` is for routing and for a person reading the log. It names ids, fields and figures
    but never repeats filing text, so passing it to a rework prompt adds no untrusted text.
    """

    kind: CitationFailureKind
    detail: str
    chunk_id: str | None = None
    metric_id: str | None = None


class CitationCheckResult(Record):
    """Every failure found in a claim's evidence, so one rework can fix them all at once."""

    claim_id: str
    failures: tuple[CitationFailure, ...] = ()

    @property
    def passed(self) -> bool:
        # Derived, never stored, so it cannot disagree with the failures.
        return not self.failures


def _check_citation(
    citation: Citation, claim_periods: Collection[FiscalPeriod], chunks: ChunkLookup
) -> list[CitationFailure]:
    chunk_id = citation.chunk_id

    def failure(kind: CitationFailureKind, detail: str) -> CitationFailure:
        return CitationFailure(kind=kind, detail=detail, chunk_id=chunk_id)

    chunk = chunks.get_chunk(chunk_id)
    # The chunk's own period is the truth, so a citation that mislabels a prior-year passage as
    # current is still wrong-period, not only a metadata mismatch.
    periods = [citation.period] if chunk is None else [citation.period, chunk.period]
    failures = [
        failure(
            CitationFailureKind.WRONG_PERIOD,
            f"evidence is from {period.label}, and the claim is about "
            f"{', '.join(sorted(about.label for about in claim_periods)) or 'no period'}",
        )
        for period in dict.fromkeys(periods)
        if period not in claim_periods
    ]
    if chunk is None:
        failures.append(failure(CitationFailureKind.UNKNOWN_CHUNK, f"no chunk {chunk_id!r}"))
        return failures

    if not _appears_in(_normalize_quote(citation.quote), _normalize_quote(chunk.text)):
        failures.append(
            failure(CitationFailureKind.ALTERED_QUOTE, f"the quote is not in chunk {chunk_id!r}")
        )
    for field in ("ticker", "accession", "form", "period", "section"):
        cited, actual = getattr(citation, field), getattr(chunk, field)
        if cited != actual:
            # FiscalPeriod prints every field; its label is what a reader needs.
            shown = (cited.label, actual.label) if field == "period" else (cited, actual)
            failures.append(
                failure(
                    CitationFailureKind.METADATA_MISMATCH,
                    f"the citation says {field} {shown[0]}, and the chunk has {shown[1]}",
                )
            )
    return failures


class _FigureKind(StrEnum):
    MONEY = "money"
    # Percent values are in percent (37.5), and point moves in percentage points (2.0 is 200 basis
    # points). They are kept apart because they differ: a margin going from 39.5% to 37.5% fell 2
    # points but 5.1%, and writing "fell 2%" for it is the common mistake.
    PERCENT = "percent"
    POINTS = "points"
    COUNT = "count"
    # A table cell with no symbol. It sits under a "$" column as often as not, so it can be money
    # or a count. Only read from table quotes: in prose a bare number is usually not a figure.
    BARE = "bare"


class _Figure(NamedTuple):
    kind: _FigureKind
    # Always a magnitude. The sign is left to `MetricChecker`, which judges a claim's direction,
    # and filings show a negative as "(60.2)", which reads as a move as often as a negative level.
    value: float
    # Half a unit in the last displayed place, so "37.5%" covers 37.45 to 37.55.
    tolerance: float
    text: str
    currency: str | None = None
    # What a word multiplied the number by: "million" is 1,000,000, and no word is 1.
    word_scale: float = 1.0


# A number with whatever marks it as a figure. Numbers cut from a word or a longer number are not
# matched, so the "2025" in "FY2025" is not read. A ")" can sit before the unit, as in "(60.2)%".
_NUMBER = re.compile(
    r"(?P<currency>[$€£])?(?<![\w.,])(?P<number>\d(?:[\d,]*\d)?(?:\.\d+)?)(?!\w)"
    r"(?:\)?[ -]?(?P<unit>%|percent\b|percentage[ -]points?\b|basis[ -]points?\b|bps\b"
    r"|thousand\b|million\b|billion\b|trillion\b))?"
)
_WORD_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
# How many percentage points one of each unit is.
_POINT_UNITS = {"percentage point": 1.0, "basis point": 0.01, "bp": 0.01}


def _parse_figures(text: str, *, bare: bool = False) -> Iterator[_Figure]:
    """The figures in normalized text, each as a kind, a value and a display tolerance.

    A number with no symbol or unit is not a figure unless `bare`: "Item 7", "Note 12" and
    "fiscal 2025" are not figures in a claim, so "200 stores" is not caught either.
    """
    for match in _NUMBER.finditer(text):
        digits = match["number"].replace(",", "")
        number = float(digits)
        step = 0.5 * 10 ** -len(digits.partition(".")[2])
        unit = (match["unit"] or "").replace("-", " ").removesuffix("s")
        currency = match["currency"]
        if unit in ("%", "percent"):
            yield _Figure(_FigureKind.PERCENT, number, step, match.group())
        elif unit in _POINT_UNITS:
            size = _POINT_UNITS[unit]
            yield _Figure(_FigureKind.POINTS, number * size, step * size, match.group())
        elif unit or currency:
            scale = _WORD_SCALES.get(unit, 1.0)
            kind = _FigureKind.MONEY if currency else _FigureKind.COUNT
            yield _Figure(
                kind, number * scale, step * scale, match.group(), currency, word_scale=scale
            )
        elif bare:
            yield _Figure(_FigureKind.BARE, number, step, match.group())


def _matches(claimed: _Figure, evidence: _Figure) -> bool:
    """Whether the evidence rounds to the claimed figure at the precision the claim displays.

    Only the claim's tolerance counts, so a claim may round its source ("$11.7 billion" from
    11,692.6 million) but not be more precise than it ("$11,692.6 million" from "$11.7 billion").
    """
    if evidence.kind is _FigureKind.BARE:
        if claimed.kind not in (_FigureKind.MONEY, _FigureKind.COUNT):
            return False
    elif (claimed.kind, claimed.currency) != (evidence.kind, evidence.currency):
        return False
    # The factor only absorbs floating point error in a value that rounds exactly to the edge.
    return abs(claimed.value - evidence.value) <= claimed.tolerance * (1 + 1e-9)


# A relative change means little from a prior ratio this small: a margin going from 0.5% to 1% is
# "up 100%". Lane 1 left the cutoff to the caller. 1% (0.01) is chosen by hand.
MIN_PRIOR_FOR_RELATIVE_CHANGE = 0.01


def _computed_figures(metrics: Iterable[MetricValue]) -> list[_Figure]:
    """What a claim may state from these metrics: levels, point changes and relative changes.

    The conventions are in `contracts.MetricLookup`. Only ratios are read: no `USD` metric exists
    yet, so one that appeared would match nothing and fail loudly rather than pass unread.
    """

    def figure(kind: _FigureKind, value: float, label: str) -> _Figure:
        return _Figure(kind, value, 0.0, label)

    figures: list[_Figure] = []
    pairable: defaultdict[tuple[str, str], list[MetricValue]] = defaultdict(list)
    for metric in metrics:
        if metric.unit != "ratio":
            continue
        level = abs(metric.value) * 100
        figures.append(figure(_FigureKind.PERCENT, level, f"{level:.2f}%"))
        # Revenue growth already describes a change, so it supports only its own value.
        if metric.name != MetricName.REVENUE_GROWTH.value:
            pairable[(metric.name, metric.duration)].append(metric)
    for values in pairable.values():
        for first, second in combinations(values, 2):
            if first.period == second.period:
                continue
            prior, current = sorted((first, second), key=lambda value: value.period.period_end)
            move = abs(current.value - prior.value)
            figures.append(figure(_FigureKind.POINTS, move * 100, f"{move * 100:.2f} points"))
            if prior.value >= MIN_PRIOR_FOR_RELATIVE_CHANGE:
                relative = move / prior.value * 100
                figures.append(figure(_FigureKind.PERCENT, relative, f"{relative:.2f}% change"))
    return figures


_EXCEPT = re.compile(r"\bexcept\b")
_ABOUT_SHARES = re.compile(r"\bshares?\b")
# A cell that names shares and no dollar quantity scales share counts only: "Denominator (shares in
# thousands):" says nothing about the table's dollars, and it is in 7 tables in the pinned filings.
# "Share-based" and "per share" are about dollars, so neither counts.
_NAMES_SHARES = re.compile(r"(?<!per )\bshares?\b(?!-)")
_NAMES_DOLLARS = re.compile(r"\$|\b(?:dollars?|value|amounts?|costs?|prices?)\b")
# The parser's caption rule also refuses text ending like a sentence: "We sold millions of units."
# only mentions a scale. `_scale_caption` accepts one that short, so it is refused here.
_SENTENCE_END = (".", "!", "?")


class _TableScales(NamedTuple):
    """The scales a table chunk states, by how much of the table each one reaches."""

    # A caption or a row's first cell: these govern the table's figures.
    table: set[float]
    # A later cell naming something else, such as "Approximate Dollar Value (In millions)". It
    # governs its column, and the chunk does not say which column a quoted number sits in.
    column: set[float]
    # A cell naming only shares. It governs share counts, never dollars.
    shares: set[float]
    # A whole-table caption makes an exception, as in "(In thousands, except share data)".
    has_exception: bool


def _table_scales(text: str) -> _TableScales:
    """Read a table chunk's scales, cell by cell, using the parser's rule for what states one.

    Each part of a caption before "except" is what it scales, so "(In thousands, except share data)"
    is a whole-table scale and not a share scale.
    """
    table: set[float] = set()
    column: set[float] = set()
    shares: set[float] = set()
    has_exception = False
    for line in text.split("\n"):
        for position, cell in enumerate(line.split(" | ")):
            scale, _ = _scale_caption([cell])
            if scale is None or cell.rstrip().endswith(_SENTENCE_END):
                continue
            covered, *exception = _EXCEPT.split(cell.casefold(), maxsplit=1)
            if _NAMES_SHARES.search(covered) and not _NAMES_DOLLARS.search(covered):
                shares.add(scale)
                continue
            match = SCALE_CAPTION.search(covered)
            around = covered[: match.start()] + " " + covered[match.end() :] if match else covered
            if position > 0 and set(re.findall(r"[a-z]+", around)) - SCALE_FILLER:
                column.add(scale)
            else:
                table.add(scale)
                has_exception = has_exception or bool(exception)
    return _TableScales(table, column, shares, has_exception)


def _quote_figures(quote: str, chunk: Chunk) -> tuple[list[_Figure], list[_Figure]]:
    """What a verified quote supports, and any figures from a table with no scale for them.

    Scale only applies to a table's own unscaled numbers. A prose chunk says "$5 million" in full,
    and its "(in millions)" would not govern its other sentences.

    A table-wide scale is required, so a bare "$11,692.6" fails against 11,692.6 in millions: that
    reads as about eleven thousand dollars. The exception is a caption such as "(In thousands,
    except share data)", which does not cover per-share rows. There, a quote about shares also
    supports the value as displayed. Two table-wide scales support either.

    A column's scale also supports the value as displayed when no table-wide scale exists, because
    the chunk does not say which column the number sits in: a repurchase table's "$98.50" price sits
    beside a dollar value in millions. A share scale only reads a symbol-less number as a count.

    A table with no scale for its figures supports only what it displays, so those numbers come
    back separately: they back a claim with no scale word, and a claim that adds one asserts what
    the table does not state, so it fails with `UNKNOWN_SCALE` instead of looking invented.
    """
    # The chunker joins a table's cells with this. Prose does not contain it.
    if " | " not in chunk.text:
        return list(_parse_figures(quote)), []
    scales = _table_scales(chunk.text)
    figures: list[_Figure] = []
    unscaled: list[_Figure] = []
    for figure in _parse_figures(quote, bare=True):
        if figure.word_scale != 1.0 or figure.kind in (_FigureKind.PERCENT, _FigureKind.POINTS):
            figures.append(figure)
            continue
        if figure.kind is _FigureKind.BARE:
            figures += [
                figure._replace(kind=_FigureKind.COUNT, value=figure.value * scale)
                for scale in scales.shares
            ]
        if not scales.table and not scales.column:
            unscaled.append(figure)
            continue
        if not scales.table or (scales.has_exception and _ABOUT_SHARES.search(quote)):
            figures.append(figure)
        figures += [
            figure._replace(value=figure.value * scale) for scale in scales.table | scales.column
        ]
    return figures, unscaled


def _check_figures(
    claim: Claim, chunks: ChunkLookup, metrics: Collection[MetricValue]
) -> list[CitationFailure]:
    """Require each figure in the claim's text to match a cited metric or a verified quote.

    Each figure is judged on its own, so one good metric does not excuse a wrong number beside it.
    Only a quote that really is in its chunk counts: an altered quote backs nothing. Details name
    the claim's figure and values computed here, never filing text.
    """
    figures = {
        figure.text: figure for figure in _parse_figures(_normalize_quote(claim.text))
    }.values()
    if not figures:
        return []
    computed = _computed_figures(metrics)
    evidence: list[_Figure] = list(computed)
    unscaled: list[_Figure] = []
    for citation in claim.citations:
        chunk = chunks.get_chunk(citation.chunk_id)
        quote = _normalize_quote(citation.quote)
        if chunk is not None and _appears_in(quote, _normalize_quote(chunk.text)):
            supported, from_unscaled_table = _quote_figures(quote, chunk)
            evidence += supported
            unscaled += from_unscaled_table

    failures: list[CitationFailure] = []
    for figure in figures:
        if any(_matches(figure, candidate) for candidate in evidence):
            continue
        if figure.word_scale == 1.0 and any(_matches(figure, number) for number in unscaled):
            continue
        if figure.word_scale != 1.0 and any(
            _matches(figure, number._replace(value=number.value * figure.word_scale))
            for number in unscaled
        ):
            kind = CitationFailureKind.UNKNOWN_SCALE
            detail = f"the cited table states no scale, so {figure.text!r} cannot be confirmed"
        elif metrics:
            kind = CitationFailureKind.FIGURE_MISMATCH
            values = ", ".join(candidate.text for candidate in computed) or "no ratio values"
            detail = (
                f"{figure.text!r} matches no cited quote and none of the values computed from "
                f"the cited metrics: {values}"
            )
        else:
            kind = CitationFailureKind.UNTRACED_FIGURE
            detail = f"no metric or cited quote backs the figure {figure.text!r}"
        failures.append(CitationFailure(kind=kind, detail=detail))
    return failures


def check_citations(
    claim: Claim,
    claim_periods: Collection[FiscalPeriod],
    chunks: ChunkLookup,
    metrics: MetricLookup,
) -> CitationCheckResult:
    """Check a claim's citations and figures, and report everything that is wrong.

    `claim_periods` are the periods the claim is about, which the caller must say because a
    `Claim` does not record them and the right answer depends on the claim: an explanation of a
    current-period move should pass only the current period, since a cause stated in last year's
    filing explains last year, while a removed risk factor is about the prior period.

    Each cited quote must appear in the chunk it cites, the citation must describe that chunk
    correctly, and the chunk must be from one of `claim_periods`. `Section.OTHER` needs no special
    case: section equality already stops an `other` table being cited as MD&A, and the reverse.

    Each metric id must resolve. Each figure in the claim's text must match, by value, something
    computed from a cited metric or a figure in a verified cited quote (see `_check_figures`). The
    plan's rule names only metrics, but a figure can be reported in text and have no computed
    metric, such as a table in `Section.OTHER`, and all current metrics are ratios, so a dollar
    figure can only trace to a quote. A metric's period is not checked because metric ids come
    from the metric checker, which is limited to the comparison, and not from the model.
    """
    failures = [
        failure
        for citation in claim.citations
        for failure in _check_citation(citation, claim_periods, chunks)
    ]
    resolved: list[MetricValue] = []
    for metric_id in claim.metric_ids:
        metric = metrics.get_metric(metric_id)
        if metric is None:
            failures.append(
                CitationFailure(
                    kind=CitationFailureKind.UNKNOWN_METRIC,
                    detail=f"no metric {metric_id!r}",
                    metric_id=metric_id,
                )
            )
        else:
            resolved.append(metric)
    failures += _check_figures(claim, chunks, resolved)
    return CitationCheckResult(claim_id=claim.claim_id, failures=tuple(failures))
