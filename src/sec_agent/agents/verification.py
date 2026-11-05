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
from collections.abc import Collection
from enum import StrEnum
from typing import NamedTuple

from sec_agent.contracts import ChunkLookup, MetricLookup
from sec_agent.schemas import (
    Citation,
    Claim,
    ExplanationLabel,
    FiscalPeriod,
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

# A stated figure: money, a percentage, basis points, or a count scaled in words. Bare numbers are
# left out because "Item 7", "Note 12" and "fiscal 2025" are not figures, so "200 stores" is not
# caught. A check that fails sound claims on those would teach nobody anything.
_FIGURE = re.compile(
    r"[$€£] ?\d[\d,]*(?:\.\d+)?(?: (?:thousand|million|billion|trillion)\b)?"
    r"|\d[\d,]*(?:\.\d+)? ?(?:%|percent\b|percentage points?\b|basis points?\b|bps\b)"
    r"|\d[\d,]*(?:\.\d+)? (?:thousand|million|billion|trillion)\b"
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
    # The claim states a figure that neither a metric nor a cited quote backs.
    UNTRACED_FIGURE = "untraced_figure"


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

    Each metric id must resolve. A figure in the claim's text must rest on a metric or appear in a
    cited quote: the plan's rule names only metrics, but a figure can be reported in text and have
    no computed metric, such as a table in `Section.OTHER`. A metric's value is not compared with
    the figure, since a claim may state a difference between two values. Its period is not checked
    because metric ids come from the metric checker, which is limited to the comparison, and not
    from the model.
    """
    failures = [
        failure
        for citation in claim.citations
        for failure in _check_citation(citation, claim_periods, chunks)
    ]
    failures += [
        CitationFailure(
            kind=CitationFailureKind.UNKNOWN_METRIC,
            detail=f"no metric {metric_id!r}",
            metric_id=metric_id,
        )
        for metric_id in claim.metric_ids
        if metrics.get_metric(metric_id) is None
    ]
    if not claim.metric_ids:
        text = _normalize_quote(claim.text)
        quotes = [_normalize_quote(citation.quote) for citation in claim.citations]
        figures = dict.fromkeys(match.group() for match in _FIGURE.finditer(text))
        failures += [
            CitationFailure(
                kind=CitationFailureKind.UNTRACED_FIGURE,
                detail=f"no metric or cited quote backs the figure {figure!r}",
            )
            for figure in figures
            if not any(_appears_in(figure, quote) for quote in quotes)
        ]
    return CitationCheckResult(claim_id=claim.claim_id, failures=tuple(failures))
