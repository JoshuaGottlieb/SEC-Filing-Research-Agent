"""The causal-language check (T5.3): causal wording fails unless the claim is confirmed."""

from __future__ import annotations

from datetime import date

import pytest

from sec_agent.agents.verification import (
    CAUSAL_PHRASES,
    check_causal_language,
    find_causal_language,
)
from sec_agent.schemas import (
    Citation,
    Claim,
    EntailmentVerdict,
    ExplanationLabel,
    FiscalPeriod,
    FormType,
    Section,
)

CITATION = Citation(
    chunk_id="chunk-1",
    accession="0000047111-2025-000001",
    ticker="HSY",
    form=FormType.FORM_10K,
    period=FiscalPeriod(fiscal_year=2025, fiscal_period="FY", period_end=date(2025, 12, 31)),
    section=Section.MDNA,
    quote="Gross margin decreased due to higher cocoa costs.",
    url="https://www.sec.gov/Archives/example.htm",
)


def plausible(text: str) -> Claim:
    return Claim(
        claim_id="c1", text=text, label=ExplanationLabel.PLAUSIBLE, metric_ids=("gross_margin",)
    )


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("Gross margin fell due to higher cocoa costs.", "due to"),
        ("Margin fell because input costs rose.", "because"),
        ("Margin fell because of input costs.", "because"),
        ("Margin fell as a result of input costs.", "as a result of"),
        ("Input costs rose. As a result, margin fell.", "as a result"),
        ("The decline is the result of input costs.", "result of"),
        ("Higher costs resulted in lower margins.", "resulted in"),
        ("The decline resulted from higher costs.", "resulted from"),
        ("The decline results from higher costs.", "results from"),
        ("Pricing, resulting in lower volume, changed.", "resulting in"),
        ("The decline was caused by input costs.", "caused"),
        ("Tariffs could cause margins to fall.", "cause"),
        ("Margin fell, driven by input costs.", "driven by"),
        ("It was a cost-driven decline.", "-driven"),
        ("Input costs drove the decline.", "drove"),
        ("Higher costs led to lower margins.", "led to"),
        ("The decline was attributable to input costs.", "attributable to"),
        ("Higher sales attributable to pricing were reported.", "attributable to"),
        ("Management attributed to pricing most of the gain.", "attributed to"),
        ("Cocoa costs contributed to the decline.", "contributed to"),
        ("Margin was negatively impacted by input costs.", "impacted by"),
        ("Sales were affected by weak demand.", "affected by"),
        ("Sales benefited from higher prices.", "benefited from"),
        ("The decline reflects higher input costs.", "reflects higher"),
        ("Margin fell, reflecting the lower volume.", "reflecting the lower"),
        ("Prices rose in response to input costs.", "in response to"),
        ("The decline is explained by input costs.", "explained by"),
        ("The decline stems from input costs.", "stems from"),
        ("Costs rose; consequently, margin fell.", "consequently"),
        ("The decline is a consequence of input costs.", "consequence of"),
        ("Costs rose, therefore margin fell.", "therefore"),
        ("Costs rose, thus reducing margin.", "thus"),
        ("Costs rose and hence margin fell.", "hence"),
        ("Margin fell owing to input costs.", "owing to"),
        ("Margin rose thanks to lower costs.", "thanks to"),
        ("Margin fell on account of input costs.", "on account of"),
        ("Sales rose on the back of pricing.", "on the back of"),
    ],
)
def test_causal_wording_is_flagged(text, phrase):
    assert find_causal_language(text) == (phrase,)
    result = check_causal_language(plausible(text))
    assert not result.passed
    assert result.flagged_phrases == (phrase,)


@pytest.mark.parametrize(
    "text",
    [
        "Gross margin fell 200 basis points, consistent with higher cocoa costs in the MD&A.",
        "Gross margin fell in the same period that input costs rose.",
        "A new risk factor on tariffs was added to the filing.",
        "Revenue rose 3%, partially offset by a 2% decline in volume.",
        # Standard filing wording that looks causal and is not.
        "Results of operations improved in the second half.",
        "Results from operations improved, and results from continuing operations did too.",
        "The company discussed results in future periods.",
        "Net income attributable to common stockholders fell 4%.",
        "Earnings attributable to Albemarle fell 4%.",
        "Net loss attributable to noncontrolling interests narrowed.",
        "The carrying amount reflects the fair value of the notes.",
        # Bare "drive" is also a noun.
        "Sales of the hard drive line fell.",
        # Word boundaries: these contain a phrase's letters without containing the phrase.
        "The residue to be removed and subdued total were reported.",
        "Causeway revenue fell; the leader to watch is unchanged.",
        "Henceforth the segment is reported separately.",
        # "Thus far" is about time.
        "Revenue has grown thus far in 2025.",
    ],
)
def test_clean_wording_passes(text):
    assert find_causal_language(text) == ()
    assert check_causal_language(plausible(text)).passed


@pytest.mark.parametrize(
    "text",
    [
        "Margin may be due to input costs.",
        "The decline is possibly driven by input costs.",
        "The decline was not caused by input costs.",
    ],
)
def test_a_hedge_or_negation_does_not_excuse_causal_wording(text):
    assert not check_causal_language(plausible(text)).passed


def test_a_confirmed_claim_may_use_causal_wording():
    claim = Claim(
        claim_id="c1",
        text="Gross margin fell due to higher cocoa costs.",
        label=ExplanationLabel.CONFIRMED,
        citations=(CITATION,),
        entailment=EntailmentVerdict.STATES_CAUSE,
    )
    result = check_causal_language(claim)
    assert result.passed
    assert result.flagged_phrases == ()


def test_an_unresolved_claim_may_not():
    claim = Claim(
        claim_id="c9", text="Margin fell due to input costs.", label=ExplanationLabel.UNRESOLVED
    )
    result = check_causal_language(claim)
    assert not result.passed
    assert result.claim_id == "c9"


def test_a_plausible_claim_is_judged_on_its_text_not_its_quotes():
    # The filing may say "due to" in its own words; that is the evidence, not the system's claim.
    claim = Claim(
        claim_id="c1",
        text="Gross margin fell 200 basis points, consistent with higher cocoa costs.",
        label=ExplanationLabel.PLAUSIBLE,
        citations=(CITATION,),
    )
    assert "due to" in CITATION.quote
    assert check_causal_language(claim).passed


def test_every_phrase_is_reported_in_order_without_repeats():
    text = "Sales fell because of pricing, driven by demand, and because of tariffs, due to policy."
    assert find_causal_language(text) == ("because", "driven by", "due to")


@pytest.mark.parametrize(
    "text",
    [
        "Margin fell DUE TO input costs.",
        "Margin fell due\nto input costs.",
        "Margin fell due to input costs.",
        "Margin fell due   to input costs.",
    ],
)
def test_case_and_line_wrapping_do_not_hide_a_phrase(text):
    assert find_causal_language(text) == ("due to",)


def test_every_phrase_records_why_it_is_on_the_list():
    assert all(phrase.reason.strip() for phrase in CAUSAL_PHRASES)
