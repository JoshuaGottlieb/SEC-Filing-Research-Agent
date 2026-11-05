"""The checks a claim must pass before it can reach the brief.

Causal language (T5.3): only a confirmed finding may say why something happened. The label
rules decide what a claim has earned; this check makes sure its wording does not claim more.
Without it, a claim labeled plausible could still read "margin fell due to cocoa costs", and
the reader would take the sentence, not the label, as the finding. It is plain code with no
model involved, so it gives the same answer every time and cannot be talked out of it.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from sec_agent.schemas import Claim, ExplanationLabel, Record


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
