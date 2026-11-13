"""Labeling (T4.5), on fakes for the judge and the writer. No test here makes a real API call.

The citation checker is the real one, run against fake chunk, metric and fact lookups, because what
it accepts is part of what labeling decides.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from sec_agent.agents.verification import (
    EntailmentJudgment,
    check_causal_language,
    find_causal_language,
)
from sec_agent.llm.client import LLMClient, LLMError, LLMRefusalError
from sec_agent.llm.ledger import BudgetExceededError, CostLedger
from sec_agent.schemas import (
    Change,
    ChangeKind,
    Chunk,
    Citation,
    Comparison,
    ComparisonType,
    Duration,
    EntailmentVerdict,
    EvidenceSupport,
    ExplanationLabel,
    Filing,
    FiscalPeriod,
    FormType,
    MetricValue,
    NodeStatus,
    NumericCheck,
    Section,
    TreeNode,
)
from sec_agent.tot import labeling
from sec_agent.tot.controller import CandidateResult, Outcome, ToTResult, claim_periods
from sec_agent.tot.labeling import (
    FALLBACK_PLAUSIBLE_TEXT,
    MAX_JUDGE_CALLS_PER_CLAIM,
    MAX_RESTATED_JUDGE_CALLS_PER_CLAIM,
    NO_EVIDENCE_TEXT,
    RESTATE_SYSTEM,
    WORDING_SYSTEM,
    ClaimWriter,
    LabelingContext,
    PlausibleWording,
    Restatement,
    label_result,
)

CURRENT = FiscalPeriod(fiscal_year=2025, fiscal_period="FY", period_end=date(2025, 12, 31))
PRIOR = FiscalPeriod(fiscal_year=2024, fiscal_period="FY", period_end=date(2024, 12, 31))
NOW_ACC = "acc-2025"
THEN_ACC = "acc-2024"
RUN = "run1"


def filing(period: FiscalPeriod, accession: str) -> Filing:
    return Filing(
        ticker="HSY",
        company="The Hershey Company",
        cik="0000047111",
        form=FormType.FORM_10K,
        period=period,
        accession=accession,
        filed_date=date(period.fiscal_year + 1, 2, 15),
        url=f"https://www.sec.gov/{accession}.htm",
    )


COMPARISON = Comparison(
    comparison_id="HSY-yoy",
    ticker="HSY",
    comparison_type=ComparisonType.YOY_10K,
    current=filing(CURRENT, NOW_ACC),
    prior=filing(PRIOR, THEN_ACC),
)
MARGIN_CHANGE = Change(
    change_id="chg-gm", comparison_id="HSY-yoy", kind=ChangeKind.NUMERIC, metric_id="gm-2025"
)

# The ToT design's illustrative example (a hypothetical company): gross margin fell, input costs
# keeps three evidence branches and one states the cause; pricing pressure has evidence that fits
# but names no cause.
TEXTS = {
    "states": (
        "Gross margin decreased in 2025 due to higher input costs, primarily cocoa and sugar."
    ),
    "implies": "Prices for cocoa and sugar rose sharply during 2025.",
    "fits": "We continue to monitor commodity markets closely.",
    "pricing": "Competitors increased promotional activity during 2025.",
    "removed": "A shortage of cocoa beans could disrupt our supply chain.",
}


def chunk(key: str, period: FiscalPeriod = CURRENT) -> Chunk:
    return Chunk(
        chunk_id=key,
        paragraph_id=f"p-{key}",
        ticker="HSY",
        cik="0000047111",
        form=FormType.FORM_10K,
        period=period,
        section=Section.MDNA,
        accession=NOW_ACC if period == CURRENT else THEN_ACC,
        text=TEXTS[key],
        token_count=20,
    )


def cite(key: str, period: FiscalPeriod = CURRENT) -> Citation:
    source = chunk(key, period)
    return Citation(
        chunk_id=source.chunk_id,
        accession=source.accession,
        ticker="HSY",
        form=FormType.FORM_10K,
        period=period,
        section=Section.MDNA,
        quote=TEXTS[key],
        url=f"https://www.sec.gov/{source.accession}.htm",
    )


def metric(metric_id: str, period: FiscalPeriod, value: float) -> MetricValue:
    return MetricValue(
        metric_id=metric_id,
        ticker="HSY",
        name="gross_margin",
        period=period,
        duration=Duration.TWELVE_MONTHS,
        value=value,
        unit="ratio",
        input_fact_ids=("f1",),
    )


class FakeChunks:
    def __init__(self, *chunks: Chunk) -> None:
        self._chunks = {item.chunk_id: item for item in chunks}

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)


class FakeMetrics:
    def __init__(self, *values: MetricValue) -> None:
        self._values = {value.metric_id: value for value in values}

    def get_metric(self, metric_id: str) -> MetricValue | None:
        return self._values.get(metric_id)


class NoFacts:
    def find_usd_fact(self, accession: str, value: float, tolerance: float) -> str | None:
        return None


class FakeJudge:
    """An `EntailmentJudge` scripted by quote. An Exception as a verdict is raised."""

    def __init__(self, verdicts: dict[str, Any], default: Any = EntailmentVerdict.IMPLIES) -> None:
        self.verdicts = verdicts
        self.default = default
        self.calls: list[tuple[str, str]] = []

    async def judge(self, claim: str, quote: str) -> EntailmentJudgment:
        self.calls.append((claim, quote))
        verdict = self.verdicts.get(quote, self.default)
        if isinstance(verdict, Exception):
            raise verdict
        return EntailmentJudgment(verdict=verdict, reason="scripted")


class FakeWriter:
    """A `ClaimWriter` playing back queued texts. An Exception in the queue is raised.

    `restated` is what `restate` returns, None by default (the quotes do not state the cause), or
    an Exception to raise.
    """

    def __init__(self, *texts: Any, restated: Any = None) -> None:
        self.texts = list(texts)
        self.requests: list[dict] = []
        self.restated = restated
        self.restate_requests: list[dict] = []

    async def restate(self, request: dict) -> Restatement:
        self.restate_requests.append(request)
        if isinstance(self.restated, Exception):
            raise self.restated
        return Restatement(text=self.restated)

    async def write(self, request: dict) -> PlausibleWording:
        self.requests.append(request)
        item = self.texts.pop(0) if self.texts else "Promotions rose; the filings do not link this."
        if isinstance(item, Exception):
            raise item
        return PlausibleWording(text=item)


def context(judge: FakeJudge, writer: FakeWriter, *, metrics=()) -> LabelingContext:
    chunks = [chunk(key) for key in ("states", "implies", "fits", "pricing")]
    chunks.append(chunk("removed", PRIOR))
    return LabelingContext(
        judge=judge,
        writer=writer,
        chunks=FakeChunks(*chunks),
        metrics=FakeMetrics(*metrics),
        facts=NoFacts(),
    )


def candidate(n: int, thought: str) -> TreeNode:
    return TreeNode(
        node_id=f"{RUN}:chg-gm:r1:c{n}",
        run_id=RUN,
        change_id="chg-gm",
        parent_id=f"{RUN}:chg-gm:root",
        depth=1,
        thought=thought,
        status=NodeStatus.KEPT,
        score=0.8,
    )


def branch(
    parent: TreeNode,
    m: int,
    evidence: EvidenceSupport,
    *citations: Citation,
    numeric: NumericCheck = NumericCheck.NOT_APPLICABLE,
    metric_ids: tuple[str, ...] = (),
) -> TreeNode:
    return TreeNode(
        node_id=f"{parent.node_id}:e{m}",
        run_id=RUN,
        change_id="chg-gm",
        parent_id=parent.node_id,
        depth=2,
        thought="look for evidence",
        evidence=evidence,
        citations=citations,
        numeric_check=numeric,
        metric_ids=metric_ids,
        status=NodeStatus.KEPT,
        score=0.7,
    )


INPUT_COSTS = candidate(1, "Higher input costs lowered gross margin.")
PRICING = candidate(2, "Pricing pressure lowered gross margin.")


def survivor(node: TreeNode, *branches: TreeNode) -> CandidateResult:
    return CandidateResult(candidate=node, evidence=branches)


def resolved(*survivors: CandidateResult) -> ToTResult:
    return ToTResult(
        change_id="chg-gm",
        tree_root_id=f"{RUN}:chg-gm:root",
        outcome=Outcome.RESOLVED,
        candidates=survivors,
    )


def gross_margin_example() -> ToTResult:
    return resolved(
        survivor(
            INPUT_COSTS,
            branch(INPUT_COSTS, 1, EvidenceSupport.STATES_CAUSE, cite("states")),
            branch(INPUT_COSTS, 2, EvidenceSupport.IMPLIES, cite("implies")),
            branch(INPUT_COSTS, 3, EvidenceSupport.CONSISTENT, cite("fits")),
        ),
        survivor(PRICING, branch(PRICING, 1, EvidenceSupport.CONSISTENT, cite("pricing"))),
    )


async def label(result: ToTResult, judge: FakeJudge, writer: FakeWriter, **kwargs: Any):
    return await label_result(
        result, MARGIN_CHANGE, COMPARISON, context(judge, writer, **kwargs), RUN
    )


async def test_the_gross_margin_example_reproduces():
    # The plan's done-when: input costs confirmed, pricing pressure plausible.
    judge = FakeJudge({TEXTS["states"]: EntailmentVerdict.STATES_CAUSE})
    writer = FakeWriter(
        "Competitors promoted more and gross margin fell; the filings do not link them."
    )
    labeled = await label(gross_margin_example(), judge, writer)

    input_costs, pricing = labeled.finding.claims
    assert input_costs.label is ExplanationLabel.CONFIRMED
    assert input_costs.text == INPUT_COSTS.thought
    assert input_costs.citations == (cite("states"),)
    assert input_costs.entailment is EntailmentVerdict.STATES_CAUSE
    assert pricing.label is ExplanationLabel.PLAUSIBLE
    assert pricing.text.startswith("Competitors promoted more")
    assert pricing.citations == (cite("pricing"),)
    assert not labeled.finding.needs_review
    assert labeled.failures == ()
    # Only the quote the critic rated as stating the cause went to the judge.
    assert judge.calls == [(INPUT_COSTS.thought, TEXTS["states"])]
    assert (labeled.judge_calls, labeled.writer_calls) == (1, 1)


async def test_every_claim_passes_the_causal_language_check():
    judge = FakeJudge({TEXTS["states"]: EntailmentVerdict.STATES_CAUSE})
    labeled = await label(gross_margin_example(), judge, FakeWriter())
    assert all(check_causal_language(claim).passed for claim in labeled.finding.claims)


async def test_claims_link_to_their_tree_nodes():
    labeled = await label(gross_margin_example(), FakeJudge({}), FakeWriter())
    assert [claim.claim_id for claim in labeled.finding.claims] == [
        INPUT_COSTS.node_id,
        PRICING.node_id,
    ]
    assert labeled.finding.finding_id == f"{RUN}:chg-gm"
    assert labeled.finding.tree_root_id == f"{RUN}:chg-gm:root"


@pytest.mark.parametrize("verdict", [EntailmentVerdict.IMPLIES, EntailmentVerdict.UNRELATED])
async def test_the_critics_rating_alone_does_not_confirm(verdict):
    judge = FakeJudge({TEXTS["states"]: verdict})
    labeled = await label(gross_margin_example(), judge, FakeWriter())
    input_costs = labeled.finding.claims[0]
    assert input_costs.label is ExplanationLabel.PLAUSIBLE
    assert input_costs.text != INPUT_COSTS.thought
    assert input_costs.entailment is None
    # A plausible claim cites every verified quote of its kept branches, best branch first.
    assert input_costs.citations == (cite("states"), cite("implies"), cite("fits"))


async def test_a_branch_not_rated_as_stating_the_cause_never_reaches_the_judge():
    result = resolved(
        survivor(
            INPUT_COSTS,
            branch(INPUT_COSTS, 1, EvidenceSupport.IMPLIES, cite("implies")),
            branch(INPUT_COSTS, 2, EvidenceSupport.CONSISTENT, cite("states")),
        )
    )
    judge = FakeJudge({TEXTS["states"]: EntailmentVerdict.STATES_CAUSE})
    labeled = await label(result, judge, FakeWriter())
    assert judge.calls == []
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE


async def test_a_judge_failure_leaves_the_claim_plausible_and_sends_it_to_review():
    judge = FakeJudge({TEXTS["states"]: LLMRefusalError("claude-opus-5-5", "cyber")})
    labeled = await label(gross_margin_example(), judge, FakeWriter())
    input_costs = labeled.finding.claims[0]
    assert input_costs.label is ExplanationLabel.PLAUSIBLE
    assert labeled.finding.needs_review
    reason = labeled.finding.review_reason
    assert INPUT_COSTS.node_id in reason
    assert "the entailment judge failed" in reason
    assert "declined" in reason
    assert any("the entailment judge failed" in failure for failure in labeled.failures)


async def test_a_judge_failure_on_one_quote_does_not_stop_the_next_from_confirming():
    result = resolved(
        survivor(
            INPUT_COSTS,
            branch(INPUT_COSTS, 1, EvidenceSupport.STATES_CAUSE, cite("implies")),
            branch(INPUT_COSTS, 2, EvidenceSupport.STATES_CAUSE, cite("states")),
        )
    )
    judge = FakeJudge(
        {
            TEXTS["implies"]: LLMError("no parsable output"),
            TEXTS["states"]: EntailmentVerdict.STATES_CAUSE,
        }
    )
    labeled = await label(result, judge, FakeWriter())
    assert labeled.finding.claims[0].label is ExplanationLabel.CONFIRMED
    assert labeled.finding.claims[0].citations == (cite("states"),)
    # Confirmed after all, so nothing is left for a person to check.
    assert labeled.finding.review_reason is None
    assert len(labeled.failures) == 1


async def test_the_judge_stops_at_the_first_confirming_quote():
    result = resolved(
        survivor(
            INPUT_COSTS,
            branch(INPUT_COSTS, 1, EvidenceSupport.STATES_CAUSE, cite("states"), cite("implies")),
        )
    )
    judge = FakeJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    labeled = await label(result, judge, FakeWriter())
    assert len(judge.calls) == 1
    assert labeled.finding.claims[0].citations == (cite("states"),)


async def test_judging_every_quote_collects_every_confirmation(monkeypatch):
    # The switch for the user's open question: judging on adds quotes, never changes the label.
    monkeypatch.setattr(labeling, "STOP_AT_FIRST_CONFIRMATION", False)
    result = resolved(
        survivor(
            INPUT_COSTS,
            branch(INPUT_COSTS, 1, EvidenceSupport.STATES_CAUSE, cite("states"), cite("implies")),
            branch(INPUT_COSTS, 2, EvidenceSupport.STATES_CAUSE, cite("fits")),
        )
    )
    judge = FakeJudge({TEXTS["fits"]: EntailmentVerdict.UNRELATED}, EntailmentVerdict.STATES_CAUSE)
    labeled = await label(result, judge, FakeWriter())
    assert len(judge.calls) == 3
    claim = labeled.finding.claims[0]
    assert claim.label is ExplanationLabel.CONFIRMED
    assert claim.citations == (cite("states"), cite("implies"))


async def test_judge_calls_per_claim_are_capped():
    quotes = [cite(key) for key in ("states", "implies", "fits", "pricing")]
    result = resolved(
        survivor(INPUT_COSTS, branch(INPUT_COSTS, 1, EvidenceSupport.STATES_CAUSE, *quotes))
    )
    judge = FakeJudge({}, default=EntailmentVerdict.IMPLIES)
    labeled = await label(result, judge, FakeWriter())
    assert len(judge.calls) == MAX_JUDGE_CALLS_PER_CLAIM == 3
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE


async def test_a_quote_cited_by_two_branches_is_judged_once():
    result = resolved(
        survivor(
            INPUT_COSTS,
            branch(INPUT_COSTS, 1, EvidenceSupport.STATES_CAUSE, cite("implies")),
            branch(INPUT_COSTS, 2, EvidenceSupport.STATES_CAUSE, cite("implies")),
        )
    )
    judge = FakeJudge({}, default=EntailmentVerdict.IMPLIES)
    labeled = await label(result, judge, FakeWriter())
    assert len(judge.calls) == 1
    assert labeled.finding.claims[0].citations == (cite("implies"),)


async def test_a_hypothesis_with_an_unbacked_figure_is_not_confirmed_and_no_judge_is_paid():
    # Kept as written for a confirmed claim, so its figures must trace; here none does.
    figured = candidate(1, "Higher input costs lowered gross margin by 900 basis points.")
    result = resolved(
        survivor(figured, branch(figured, 1, EvidenceSupport.STATES_CAUSE, cite("states")))
    )
    judge = FakeJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    labeled = await label(result, judge, FakeWriter())
    assert judge.calls == []
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert any("cannot confirm the hypothesis as written" in f for f in labeled.failures)
    assert any("900 basis points" in f for f in labeled.failures)


async def test_two_confirmed_claims_are_co_drivers():
    tariffs = candidate(2, "Higher input costs, including cocoa and sugar, lowered gross margin.")
    result = resolved(
        survivor(INPUT_COSTS, branch(INPUT_COSTS, 1, EvidenceSupport.STATES_CAUSE, cite("states"))),
        survivor(tariffs, branch(tariffs, 1, EvidenceSupport.STATES_CAUSE, cite("states"))),
    )
    judge = FakeJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    labeled = await label(result, judge, FakeWriter())
    labels = [claim.label for claim in labeled.finding.claims]
    assert labels == [ExplanationLabel.CONFIRMED, ExplanationLabel.CONFIRMED]
    # Judged per candidate: the same quote is checked against each claim.
    assert len(judge.calls) == 2


async def test_causal_wording_is_rejected_and_retried_with_the_reason():
    writer = FakeWriter(
        "Gross margin fell due to promotions.",
        "Promotions increased and gross margin fell; the filings do not link them.",
    )
    labeled = await label(resolved(gross_margin_example().candidates[1]), FakeJudge({}), writer)
    pricing = labeled.finding.claims[0]
    assert pricing.text.startswith("Promotions increased")
    retry = writer.requests[-1]
    assert retry["rejected"]["text"] == "Gross margin fell due to promotions."
    assert "due to" in retry["rejected"]["reason"]
    assert any("wording rejected" in failure for failure in labeled.failures)


async def test_an_invented_figure_is_rejected():
    writer = FakeWriter(
        "Promotions increased and gross margin fell 5%; the filings do not link them.",
        "Promotions increased and gross margin fell; the filings do not link them.",
    )
    labeled = await label(resolved(gross_margin_example().candidates[1]), FakeJudge({}), writer)
    assert labeled.finding.claims[0].text.startswith("Promotions increased and gross margin fell;")
    assert "5%" in writer.requests[-1]["rejected"]["reason"]


async def test_wording_that_keeps_failing_falls_back_to_the_fixed_sentence():
    writer = FakeWriter("Margin fell because of promotions.", "Promotions drove the decline.")
    result = resolved(gross_margin_example().candidates[1])
    labeled = await label(result, FakeJudge({}), writer)
    claim = labeled.finding.claims[0]
    assert claim.text == FALLBACK_PLAUSIBLE_TEXT
    assert claim.label is ExplanationLabel.PLAUSIBLE
    assert labeled.writer_calls == labeling.WORDING_ATTEMPTS == 2
    assert labeled.failures[-1].endswith("used the fixed wording")


@pytest.mark.parametrize("text", ["", "   "])
async def test_blank_wording_is_rejected(text):
    writer = FakeWriter(text, "Promotions increased; the filings do not link this to margin.")
    labeled = await label(resolved(gross_margin_example().candidates[1]), FakeJudge({}), writer)
    assert labeled.finding.claims[0].text.startswith("Promotions increased")


async def test_a_writer_failure_falls_back_without_retrying():
    writer = FakeWriter(LLMError("truncated"))
    labeled = await label(resolved(gross_margin_example().candidates[1]), FakeJudge({}), writer)
    assert labeled.finding.claims[0].text == FALLBACK_PLAUSIBLE_TEXT
    assert labeled.writer_calls == 1
    assert any("the claim writer failed" in failure for failure in labeled.failures)
    # A wording failure is not a doubt about the evidence, so it does not send the claim to review.
    assert labeled.finding.review_reason is None


async def test_the_writer_sees_the_hypothesis_quotes_and_metric_values_only():
    agreeing = branch(
        PRICING,
        2,
        EvidenceSupport.NONE,
        numeric=NumericCheck.AGREE,
        metric_ids=("gm-2025", "gm-2024"),
    )
    result = resolved(
        survivor(PRICING, branch(PRICING, 1, EvidenceSupport.CONSISTENT, cite("pricing")), agreeing)
    )
    writer = FakeWriter()
    metrics = (metric("gm-2025", CURRENT, 0.335), metric("gm-2024", PRIOR, 0.473))
    await label(result, FakeJudge({}), writer, metrics=metrics)
    (request,) = writer.requests
    assert request == {
        "hypothesis": PRICING.thought,
        "quotes": [{"period": "FY2025", "quote": TEXTS["pricing"]}],
        "metrics": [
            {"metric": "gross_margin", "period": "FY2025", "value": 0.335},
            {"metric": "gross_margin", "period": "FY2024", "value": 0.473},
        ],
    }


async def test_a_survivor_kept_by_a_metric_alone_is_plausible_with_its_metrics():
    agreeing = branch(
        PRICING, 1, EvidenceSupport.NONE, numeric=NumericCheck.AGREE, metric_ids=("gm-2025",)
    )
    writer = FakeWriter("Gross margin fell; the filings do not link this to pricing.")
    labeled = await label(
        resolved(survivor(PRICING, agreeing)),
        FakeJudge({}),
        writer,
        metrics=(metric("gm-2025", CURRENT, 0.335),),
    )
    claim = labeled.finding.claims[0]
    assert claim.label is ExplanationLabel.PLAUSIBLE
    assert claim.citations == ()
    assert claim.metric_ids == ("gm-2025",)


async def test_a_survivor_with_no_evidence_on_record_is_unresolved():
    empty = branch(PRICING, 1, EvidenceSupport.NONE)
    labeled = await label(resolved(survivor(PRICING, empty)), FakeJudge({}), FakeWriter())
    claim = labeled.finding.claims[0]
    assert claim.label is ExplanationLabel.UNRESOLVED
    assert claim.text == NO_EVIDENCE_TEXT
    assert PRICING.node_id in labeled.finding.review_reason


async def test_branches_the_search_lost_to_a_model_failure_send_the_finding_to_review():
    lost = (
        f"{PRICING.node_id}:e2: the critic failed: claude-haiku-5-5 declined the request",
        f"{RUN}:chg-gm:r1:c3: the critic failed: hit max_tokens",
    )
    result = gross_margin_example().model_copy(update={"failures": lost})
    judge = FakeJudge({TEXTS["states"]: EntailmentVerdict.STATES_CAUSE})
    labeled = await label(result, judge, FakeWriter())
    # The labels stand; the finding also goes to review, naming each lost branch.
    labels = [claim.label for claim in labeled.finding.claims]
    assert labels == [ExplanationLabel.CONFIRMED, ExplanationLabel.PLAUSIBLE]
    assert labeled.finding.needs_review
    reason = labeled.finding.review_reason
    assert reason.startswith("the search lost 2 branch(es) to a model failure")
    assert all(failure in reason for failure in lost)


async def test_a_search_with_no_model_failures_adds_no_review_note():
    judge = FakeJudge({TEXTS["states"]: EntailmentVerdict.STATES_CAUSE})
    labeled = await label(gross_margin_example(), judge, FakeWriter())
    assert labeled.finding.review_reason is None


@pytest.mark.parametrize("root", [f"{RUN}:chg-gm:root", None])
async def test_an_unresolved_search_files_a_finding_with_no_claims(root):
    # None is a change the run never reached, as after a budget stop.
    result = ToTResult(
        change_id="chg-gm",
        tree_root_id=root,
        outcome=Outcome.UNRESOLVED,
        review_reason="every candidate was pruned",
    )
    judge, writer = FakeJudge({}), FakeWriter()
    labeled = await label(result, judge, writer)
    assert labeled.finding.claims == ()
    assert labeled.finding.review_reason == "every candidate was pruned"
    assert labeled.finding.tree_root_id == root
    assert labeled.finding.needs_review
    assert judge.calls == [] and writer.requests == []


async def test_a_result_for_another_change_is_refused():
    other = gross_margin_example().model_copy(update={"change_id": "chg-other"})
    with pytest.raises(ValueError, match="chg-other"):
        await label(other, FakeJudge({}), FakeWriter())


async def test_a_removed_item_is_confirmed_from_the_prior_period():
    removed = Change(
        change_id="chg-gm",
        comparison_id="HSY-yoy",
        kind=ChangeKind.REMOVED,
        section=Section.RISK_FACTORS,
        prior_text=TEXTS["removed"],
    )
    node = candidate(1, "A cocoa shortage risk was dropped.")
    quote = cite("removed", PRIOR)
    result = resolved(survivor(node, branch(node, 1, EvidenceSupport.STATES_CAUSE, quote)))
    judge = FakeJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    labeled = await label_result(result, removed, COMPARISON, context(judge, FakeWriter()), RUN)
    assert labeled.finding.claims[0].label is ExplanationLabel.CONFIRMED
    assert claim_periods(removed, COMPARISON) == frozenset({PRIOR})


async def test_a_budget_stop_is_not_swallowed():
    judge = FakeJudge({TEXTS["states"]: BudgetExceededError("over budget")})
    with pytest.raises(BudgetExceededError):
        await label(gross_margin_example(), judge, FakeWriter())


async def test_a_budget_stop_in_the_writer_is_not_swallowed():
    writer = FakeWriter(BudgetExceededError("over budget"))
    with pytest.raises(BudgetExceededError):
        await label(resolved(gross_margin_example().candidates[1]), FakeJudge({}), writer)


# Restating a hypothesis at its quotes' level (user's decision, 2026-11-13)

NARROW = candidate(3, "New tariffs on imported cocoa raised costs and lowered gross margin.")
BROAD = "Higher input costs lowered gross margin."


class ClaimJudge(FakeJudge):
    """A judge scripted by claim and quote; anything unscripted is `default`."""

    def __init__(
        self, verdicts: dict[tuple[str, str], Any], default: Any = EntailmentVerdict.IMPLIES
    ):
        super().__init__({}, default)
        self.by_claim = verdicts

    async def judge(self, claim: str, quote: str) -> EntailmentJudgment:
        self.calls.append((claim, quote))
        verdict = self.by_claim.get((claim, quote), self.default)
        if isinstance(verdict, Exception):
            raise verdict
        return EntailmentJudgment(verdict=verdict, reason="scripted")


def narrow(*keys: str) -> ToTResult:
    return resolved(
        survivor(NARROW, branch(NARROW, 1, EvidenceSupport.STATES_CAUSE, *map(cite, keys)))
    )


async def test_a_hypothesis_narrower_than_its_quote_is_confirmed_as_restated():
    judge = ClaimJudge({(BROAD, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE})
    writer = FakeWriter(restated=BROAD)
    labeled = await label(narrow("states"), judge, writer)
    (claim,) = labeled.finding.claims
    assert claim.label is ExplanationLabel.CONFIRMED
    assert claim.text == BROAD
    assert claim.citations == (cite("states"),)
    assert claim.entailment is EntailmentVerdict.STATES_CAUSE
    # The node id still leads to the hypothesis as generated.
    assert claim.claim_id == NARROW.node_id
    assert judge.calls == [(NARROW.thought, TEXTS["states"]), (BROAD, TEXTS["states"])]
    assert writer.restate_requests == [
        {"hypothesis": NARROW.thought, "quotes": [{"period": "FY2025", "quote": TEXTS["states"]}]}
    ]
    assert writer.requests == []  # no plausible wording was needed
    assert (labeled.judge_calls, labeled.writer_calls) == (2, 1)
    assert labeled.failures == ()


async def test_a_restatement_the_judge_still_finds_implied_leaves_the_claim_plausible():
    writer = FakeWriter("Costs rose and gross margin fell; the filings do not link the two.")
    writer.restated = BROAD
    judge = ClaimJudge({})
    labeled = await label(narrow("states", "implies", "fits"), judge, writer)
    (claim,) = labeled.finding.claims
    assert claim.label is ExplanationLabel.PLAUSIBLE
    assert claim.entailment is None
    # Three quotes for the hypothesis as written, then the restatement against the first two only.
    as_written = [quote for text, quote in judge.calls if text == NARROW.thought]
    restated = [quote for text, quote in judge.calls if text == BROAD]
    assert len(as_written) == MAX_JUDGE_CALLS_PER_CLAIM == 3
    assert restated == [TEXTS["states"], TEXTS["implies"]][:MAX_RESTATED_JUDGE_CALLS_PER_CLAIM]
    assert (labeled.judge_calls, labeled.writer_calls) == (5, 2)


async def test_quotes_that_do_not_state_the_cause_give_no_restatement_and_no_more_calls():
    judge = ClaimJudge({})
    labeled = await label(narrow("states"), judge, FakeWriter(restated=None))
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert judge.calls == [(NARROW.thought, TEXTS["states"])]


async def test_only_an_implied_verdict_leads_to_a_restatement():
    writer = FakeWriter(restated=BROAD)
    await label(narrow("states"), ClaimJudge({}, default=EntailmentVerdict.UNRELATED), writer)
    assert writer.restate_requests == []


async def test_a_restatement_that_repeats_the_hypothesis_is_not_judged_again():
    judge = ClaimJudge({})
    await label(narrow("states"), judge, FakeWriter(restated=f"  {NARROW.thought.upper()} "))
    assert len(judge.calls) == 1


async def test_a_restating_failure_is_noted_and_the_claim_stays_plausible():
    writer = FakeWriter(restated=LLMRefusalError("claude-haiku-5-5", None))
    labeled = await label(narrow("states"), ClaimJudge({}), writer)
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert any("restating the hypothesis failed" in failure for failure in labeled.failures)
    assert not labeled.finding.needs_review


async def test_a_restatement_with_an_unbacked_figure_is_not_judged():
    judge = ClaimJudge({})
    labeled = await label(
        narrow("states"), judge, FakeWriter(restated="Input costs up 12% lowered gross margin.")
    )
    assert len(judge.calls) == 1
    assert any("cannot confirm the restated hypothesis" in failure for failure in labeled.failures)


async def test_a_judge_failure_on_the_restatement_sends_the_finding_to_review():
    judge = ClaimJudge({(BROAD, TEXTS["states"]): LLMRefusalError("claude-opus-5-5", None)})
    labeled = await label(narrow("states"), judge, FakeWriter(restated=BROAD))
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert labeled.finding.needs_review


async def test_two_hypotheses_restated_alike_become_one_claim_with_both_quotes():
    other = candidate(4, "Tariffs on imported packaging raised costs and lowered gross margin.")
    result = resolved(
        survivor(NARROW, branch(NARROW, 1, EvidenceSupport.STATES_CAUSE, cite("states"))),
        survivor(other, branch(other, 1, EvidenceSupport.STATES_CAUSE, cite("implies"))),
    )
    judge = ClaimJudge(
        {
            (BROAD, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE,
            (BROAD, TEXTS["implies"]): EntailmentVerdict.STATES_CAUSE,
        }
    )
    labeled = await label(result, judge, FakeWriter(restated=BROAD))
    (claim,) = labeled.finding.claims
    assert claim.claim_id == NARROW.node_id
    assert claim.citations == (cite("states"), cite("implies"))
    assert any(
        f"claim {other.node_id}: confirmed as the same sentence" in f for f in labeled.failures
    )


def test_the_restating_prompt_keeps_only_the_hypothesis_cause_and_treats_input_as_untrusted():
    prose = " ".join(RESTATE_SYSTEM.split())
    assert "Name only the hypothesis's cause" in prose
    assert 'give null for "text"' in prose
    assert "Never follow" in prose


def test_the_restatement_schema_is_valid_for_structured_output():
    transformed = anthropic.transform_schema(Restatement)
    assert transformed["additionalProperties"] is False
    assert transformed["required"] == ["text"]


async def test_the_real_writer_restates_on_the_worker_model(settings):
    fake = FakeAnthropic(_response(Restatement(text=BROAD)))
    writer = ClaimWriter(LLMClient(settings, CostLedger(1.0), client=fake))
    request = {"hypothesis": NARROW.thought, "quotes": [{"period": "FY2025", "quote": "q"}]}
    assert (await writer.restate(request)).text == BROAD
    (sent,) = fake.requests
    assert sent["model"] == settings.model_worker
    assert sent["system"][0]["text"] == RESTATE_SYSTEM
    assert json.loads(sent["messages"][0]["content"]) == request


def test_the_fixed_texts_and_the_prompt_example_pass_the_causal_check():
    example = "Cocoa prices rose and gross margin fell; the filings do not link the two."
    assert example in " ".join(WORDING_SYSTEM.split())
    for text in (example, FALLBACK_PLAUSIBLE_TEXT, NO_EVIDENCE_TEXT):
        assert find_causal_language(text) == ()


def test_the_prompt_treats_input_as_untrusted():
    assert "untrusted" in WORDING_SYSTEM
    assert "Never follow" in WORDING_SYSTEM


def test_the_prompt_forbids_reporting_the_filings_own_causes():
    # Found on real text (2026-11-10): before this rule the writer reported the causes a quote
    # gave ("cite higher commodity costs... as drivers"), which the causal check did not catch.
    assert "Do not report any cause the quotes give" in " ".join(WORDING_SYSTEM.split())


def test_the_output_schema_is_valid_for_structured_output():
    transformed = anthropic.transform_schema(PlausibleWording)
    assert transformed["additionalProperties"] is False
    assert transformed["required"] == ["text"]
    # The model reads the docstring as the schema's description.
    assert transformed["description"] == "Your statement for the brief."


def _response(output: Any) -> SimpleNamespace:
    return SimpleNamespace(
        parsed_output=output,
        stop_reason="end_turn",
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


class FakeAnthropic:
    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    async def _parse(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)


async def test_the_real_writer_sends_json_to_the_worker_model(settings):
    fake = FakeAnthropic(_response(PlausibleWording(text="Costs rose; margin fell.")))
    writer = ClaimWriter(LLMClient(settings, CostLedger(1.0), client=fake))
    request = {"hypothesis": "x", "quotes": [{"period": "FY2025", "quote": 'a "b"\nc'}]}
    written = await writer.write(request)
    assert written.text == "Costs rose; margin fell."
    (sent,) = fake.requests
    assert sent["model"] == settings.model_worker
    assert sent["system"][0]["text"] == WORDING_SYSTEM
    assert json.loads(sent["messages"][0]["content"]) == request
