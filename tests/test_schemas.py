from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from sec_agent.schemas import (
    Brief,
    Change,
    ChangeKind,
    Chunk,
    Citation,
    Claim,
    Comparison,
    ComparisonType,
    Duration,
    EntailmentVerdict,
    ExplanationLabel,
    Finding,
    FiscalPeriod,
    FormType,
    MetricFact,
    MetricValue,
    NodeStatus,
    NumericCheck,
    Section,
    TreeNode,
)

# The records are the contract between agents, so these tests check the rules they enforce:
# round trips through JSON, and combinations that must be refused.
PERIOD = FiscalPeriod(fiscal_year=2025, fiscal_period="Q2", period_end=date(2025, 6, 29))


def citation() -> Citation:
    """A valid citation, reused wherever a claim needs evidence."""
    return Citation(
        chunk_id="c1",
        accession="a1",
        ticker="HSY",
        form=FormType.FORM_10Q,
        period=PERIOD,
        section=Section.MDNA,
        quote="Gross margin decreased due to higher cocoa costs.",
        url="https://www.sec.gov/Archives/example.htm",
    )


def confirmed_claim() -> Claim:
    """The strongest label, so it needs both a citation and a states_cause verdict."""
    return Claim(
        claim_id="cl1",
        text="Gross margin fell because of higher cocoa costs.",
        label=ExplanationLabel.CONFIRMED,
        citations=(citation(),),
        entailment=EntailmentVerdict.STATES_CAUSE,
    )


def numeric_change() -> Change:
    """The simplest valid change: a numeric one only needs a metric id."""
    return Change(change_id="ch1", comparison_id="cmp1", kind=ChangeKind.NUMERIC, metric_id="m1")


def comparison(make_filing) -> Comparison:
    """A year-over-year 10-K comparison of 2025 against 2024."""
    return Comparison(
        comparison_id="cmp1",
        ticker="HSY",
        comparison_type=ComparisonType.YOY_10K,
        current=make_filing(year=2025),
        prior=make_filing(year=2024),
    )


def test_fiscal_period_labels():
    assert PERIOD.label == "Q2 2025"
    annual = FiscalPeriod(fiscal_year=2025, fiscal_period="FY", period_end=date(2025, 12, 31))
    assert annual.label == "FY2025"


def test_records_round_trip_through_json(make_filing):
    finding = Finding(finding_id="f1", change=numeric_change(), claims=(confirmed_claim(),))
    records = [
        make_filing(),
        comparison(make_filing),
        Chunk(
            chunk_id="c1",
            paragraph_id="p1",
            ticker="HSY",
            cik="1",
            form=FormType.FORM_10Q,
            period=PERIOD,
            section=Section.MDNA,
            accession="a1",
            text="Text.",
            token_count=2,
        ),
        MetricFact(
            fact_id="x1",
            ticker="HSY",
            concept="Revenues",
            period=PERIOD,
            start_date=date(2025, 3, 31),
            end_date=date(2025, 6, 29),
            duration=Duration.THREE_MONTHS,
            value=2.6e9,
            unit="USD",
            accession="a1",
        ),
        MetricValue(
            metric_id="m1",
            ticker="HSY",
            name="gross_margin",
            period=PERIOD,
            duration=Duration.THREE_MONTHS,
            value=0.387,
            unit="ratio",
            input_fact_ids=("x1", "x2"),
        ),
        TreeNode(
            node_id="n1",
            run_id="r1",
            change_id="ch1",
            parent_id="n0",
            depth=1,
            thought="Input costs rose.",
            numeric_check=NumericCheck.NOT_APPLICABLE,
            score=0.5,
            status=NodeStatus.KEPT,
        ),
        finding,
        Brief(
            run_id="r1",
            comparison=comparison(make_filing),
            generated_at=datetime(2026, 11, 3, tzinfo=UTC),
            summary="Summary.",
            findings=(finding,),
        ),
    ]
    # Saved runs and the MCP tree store pass these as JSON, so none may lose information.
    for record in records:
        assert type(record).model_validate_json(record.model_dump_json()) == record


def test_confirmed_requires_citation_and_states_cause():
    # The rule the brief depends on: no causal claim without a quote and a judge's agreement.
    # First case lacks a citation, second has a citation but only an "implies" verdict.
    with pytest.raises(ValidationError, match="confirmed"):
        Claim(
            claim_id="c",
            text="t",
            label=ExplanationLabel.CONFIRMED,
            entailment=EntailmentVerdict.STATES_CAUSE,
        )
    with pytest.raises(ValidationError, match="confirmed"):
        Claim(
            claim_id="c",
            text="t",
            label=ExplanationLabel.CONFIRMED,
            citations=(citation(),),
            entailment=EntailmentVerdict.IMPLIES,
        )


def test_plausible_requires_some_evidence():
    # A computed metric alone is enough for plausible (a margin that moved fits the story).
    with pytest.raises(ValidationError, match="plausible"):
        Claim(claim_id="c", text="t", label=ExplanationLabel.PLAUSIBLE)
    Claim(claim_id="c", text="t", label=ExplanationLabel.PLAUSIBLE, metric_ids=("m1",))


def test_unresolved_needs_no_evidence_and_flags_review():
    claim = Claim(claim_id="c", text="t", label=ExplanationLabel.UNRESOLVED)
    finding = Finding(finding_id="f", change=numeric_change(), claims=(claim,))
    assert finding.needs_review


def test_comparison_rejects_wrong_form_order_and_company(make_filing):
    # Four cases in order: a valid 10-Q pair, the wrong form for the type, reversed periods,
    # and a filing from another company.
    q_current = make_filing(FormType.FORM_10Q, 2025, "Q2", date(2025, 6, 29))
    q_prior = make_filing(FormType.FORM_10Q, 2024, "Q2", date(2024, 6, 30))
    Comparison(
        comparison_id="c",
        ticker="HSY",
        comparison_type=ComparisonType.YOY_10Q,
        current=q_current,
        prior=q_prior,
    )
    with pytest.raises(ValidationError, match="compares 10-K"):
        Comparison(
            comparison_id="c",
            ticker="HSY",
            comparison_type=ComparisonType.YOY_10K,
            current=q_current,
            prior=q_prior,
        )
    with pytest.raises(ValidationError, match="later period"):
        Comparison(
            comparison_id="c",
            ticker="HSY",
            comparison_type=ComparisonType.YOY_10Q,
            current=q_prior,
            prior=q_current,
        )
    with pytest.raises(ValidationError, match="same|belong"):
        Comparison(
            comparison_id="c",
            ticker="HSY",
            comparison_type=ComparisonType.YOY_10K,
            current=make_filing(year=2025, ticker="MAT"),
            prior=make_filing(year=2024),
        )


def test_change_evidence_must_match_kind():
    with pytest.raises(ValidationError, match="metric_id"):
        Change(change_id="c", comparison_id="x", kind=ChangeKind.NUMERIC)
    with pytest.raises(ValidationError, match="needs a section"):
        Change(change_id="c", comparison_id="x", kind=ChangeKind.ADDED, current_text="New.")
    with pytest.raises(ValidationError, match="prior_text"):
        Change(
            change_id="c",
            comparison_id="x",
            kind=ChangeKind.MODIFIED,
            section=Section.MDNA,
            current_text="New.",
        )


def test_tree_node_structure_rules():
    # Depth 1 with no parent, a pruned node with no reason, and a depth beyond 2 are all refused.
    with pytest.raises(ValidationError, match="root"):
        TreeNode(node_id="n", run_id="r", change_id="c", depth=1, thought="No parent.")
    with pytest.raises(ValidationError, match="pruned"):
        TreeNode(
            node_id="n",
            run_id="r",
            change_id="c",
            parent_id="p",
            depth=1,
            thought="t",
            status=NodeStatus.PRUNED,
        )
    with pytest.raises(ValidationError):
        TreeNode(node_id="n", run_id="r", change_id="c", parent_id="p", depth=3, thought="Deep.")
