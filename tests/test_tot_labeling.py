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
from sec_agent.finance.metrics import metric_id_for
from sec_agent.llm.client import LLMClient, LLMError, LLMRefusalError
from sec_agent.llm.ledger import BudgetExceededError, CostLedger
from sec_agent.report.summary import SUMMARY_SYSTEM
from sec_agent.schemas import (
    Change,
    ChangeKind,
    Chunk,
    Citation,
    Claim,
    Comparison,
    ComparisonType,
    Duration,
    EntailmentVerdict,
    EvidenceSupport,
    ExplanationLabel,
    Filing,
    Finding,
    FiscalPeriod,
    FormType,
    MetricName,
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
    # Hershey's Q2 2026 10-Q: a six-month paragraph, which cannot speak to the quarter alone.
    "six_months": (
        "Gross margin was 42.2% in the first six months of 2026, driven by higher input costs."
    ),
    # Hershey's segment table (FY2025 10-K): its columns are segments, not years.
    "segments": (
        "Our segment net sales and earnings were as follows:\n"
        "(amounts in thousands, except share data or if otherwise indicated)\n"
        "For the year ended December 31, 2025 | North America Confectionery | "
        "North America Salty Snacks | International | Total\n"
        "Net sales | $ | 9,479,709 | $ | 1,271,299 | $ | 941,568 | $ | 11,692,576\n"
        "Cost of sales | 5,796,018 | 804,635 | 740,764\n"
        "SM&A expense | 1,189,959 | 225,311 | 197,486"
    ),
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


# What the fake restater returns by default: the hypothesis itself, as a restater does when the
# hypothesis is already at its quote's level, so a claim the judge confirms as written stays so.
ECHO = object()


class FakeWriter:
    """A `ClaimWriter` playing back queued texts. An Exception in the queue is raised.

    `restated` is what `restate` returns: by default the hypothesis (`ECHO`), None when the quotes
    do not state the cause, a text, or an Exception to raise.
    """

    def __init__(self, *texts: Any, restated: Any = ECHO) -> None:
        self.texts = list(texts)
        self.requests: list[dict] = []
        self.restated = restated
        self.restate_requests: list[dict] = []

    async def restate(self, request: dict) -> Restatement:
        self.restate_requests.append(request)
        if isinstance(self.restated, Exception):
            raise self.restated
        if self.restated is ECHO:
            return Restatement(text=request["hypothesis"])
        return Restatement(text=self.restated)

    async def write(self, request: dict) -> PlausibleWording:
        self.requests.append(request)
        item = (
            self.texts.pop(0)
            if self.texts
            else "Promotions rose; this brief did not confirm a link to gross margin."
        )
        if isinstance(item, Exception):
            raise item
        return PlausibleWording(text=item)


def context(judge: FakeJudge, writer: FakeWriter, *, metrics=()) -> LabelingContext:
    keys = ("states", "implies", "fits", "pricing", "segments", "six_months")
    chunks = [chunk(key) for key in keys]
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
    # Only the quote the critic rated as stating the cause went to the judge. The writer restated
    # the confirmed claim (it was at the quote's level already) and worded the plausible one.
    assert judge.calls == [(INPUT_COSTS.thought, TEXTS["states"])]
    assert (labeled.judge_calls, labeled.writer_calls) == (1, 2)


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
    # A plausible claim cites every verified quote of its kept branches, best branch first, except
    # one the judge found unrelated (user's decision, 2026-11-15).
    kept = (cite("states"),) if verdict is EntailmentVerdict.IMPLIES else ()
    assert input_costs.citations == (*kept, cite("implies"), cite("fits"))


async def test_a_branch_rated_below_implies_never_reaches_the_judge_or_the_restater():
    result = resolved(
        survivor(
            INPUT_COSTS,
            branch(INPUT_COSTS, 1, EvidenceSupport.CONSISTENT, cite("states")),
            branch(INPUT_COSTS, 2, EvidenceSupport.NONE, cite("implies")),
        )
    )
    judge = FakeJudge({TEXTS["states"]: EntailmentVerdict.STATES_CAUSE})
    writer = FakeWriter()
    labeled = await label(result, judge, writer)
    assert judge.calls == []
    assert writer.restate_requests == []
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert labeled.rework == ()


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
    tariffs = candidate(2, "Tariffs on imports lowered gross margin.")
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
        "change_being_explained": "gross margin",
    }


def metric_only(node: TreeNode) -> TreeNode:
    return branch(
        node, 1, EvidenceSupport.NONE, numeric=NumericCheck.AGREE, metric_ids=("om-2025",)
    )


async def test_a_survivor_kept_by_a_metric_alone_is_left_out_and_the_finding_reviewed():
    # User's decision, 2026-11-13: with no quote, another margin falling is all it has.
    writer = FakeWriter()
    labeled = await label(resolved(survivor(PRICING, metric_only(PRICING))), FakeJudge({}), writer)
    assert labeled.finding.claims == ()
    assert labeled.finding.review_reason == labeling.METRIC_ONLY_REASON
    assert writer.requests == []
    assert any(f"claim {PRICING.node_id}: only computed metrics" in f for f in labeled.failures)


async def test_a_metric_only_survivor_beside_a_quoted_one_just_drops_out():
    result = resolved(
        survivor(INPUT_COSTS, branch(INPUT_COSTS, 1, EvidenceSupport.STATES_CAUSE, cite("states"))),
        survivor(PRICING, metric_only(PRICING)),
    )
    judge = FakeJudge({TEXTS["states"]: EntailmentVerdict.STATES_CAUSE})
    labeled = await label(result, judge, FakeWriter())
    assert [claim.claim_id for claim in labeled.finding.claims] == [INPUT_COSTS.node_id]
    assert not labeled.finding.needs_review


def test_the_metric_only_reason_names_no_cause():
    assert find_causal_language(labeling.METRIC_ONLY_REASON) == ()


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


# A quote the critic rated as only implying is restated from, never judged as written (user's
# decision, 2026-11-15)


def critic_implied(*keys: str, node: TreeNode = NARROW) -> ToTResult:
    return resolved(survivor(node, branch(node, 1, EvidenceSupport.IMPLIES, *map(cite, keys))))


async def test_a_quote_the_critic_found_only_implying_confirms_the_restatement():
    judge = ClaimJudge({(BROAD, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE})
    writer = FakeWriter(restated=BROAD)
    labeled = await label(critic_implied("states"), judge, writer)
    (claim,) = labeled.finding.claims
    assert (claim.label, claim.text, claim.citations) == (
        ExplanationLabel.CONFIRMED,
        BROAD,
        (cite("states"),),
    )
    # The hypothesis as written never went to the judge: the critic already found it says more.
    assert judge.calls == [(BROAD, TEXTS["states"])]
    assert [shown["quote"] for shown in writer.restate_requests[0]["quotes"]] == [TEXTS["states"]]
    assert labeled.rework == ()


async def test_a_hypothesis_restated_unchanged_is_judged_as_written_on_the_critics_quote():
    judge = ClaimJudge({(NARROW.thought, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE})
    labeled = await label(critic_implied("states"), judge, FakeWriter())
    (claim,) = labeled.finding.claims
    assert (claim.label, claim.text) == (ExplanationLabel.CONFIRMED, NARROW.thought)
    assert judge.calls == [(NARROW.thought, TEXTS["states"])]


async def test_an_unchanged_restatement_is_not_judged_again_on_a_quote_judged_as_written():
    # The same quote rated as stating the cause in one branch and implying it in another.
    result = resolved(
        survivor(
            NARROW,
            branch(NARROW, 1, EvidenceSupport.STATES_CAUSE, cite("states")),
            branch(NARROW, 2, EvidenceSupport.IMPLIES, cite("states")),
        )
    )
    judge = ClaimJudge({})
    await label(result, judge, FakeWriter())
    assert judge.calls == [(NARROW.thought, TEXTS["states"])]


async def test_a_critics_quote_cited_by_two_branches_is_restated_from_and_judged_once():
    result = resolved(
        survivor(
            NARROW,
            branch(NARROW, 1, EvidenceSupport.IMPLIES, cite("states")),
            branch(NARROW, 2, EvidenceSupport.IMPLIES, cite("states")),
        )
    )
    judge = ClaimJudge({})
    writer = FakeWriter(restated=BROAD)
    labeled = await label(result, judge, writer)
    assert len(writer.restate_requests[0]["quotes"]) == 1
    assert judge.calls == [(BROAD, TEXTS["states"])]
    assert labeled.rework[0].implied == (cite("states"),)


async def test_the_judges_implying_quotes_come_before_the_critics():
    result = resolved(
        survivor(
            NARROW,
            branch(NARROW, 1, EvidenceSupport.STATES_CAUSE, cite("implies")),
            branch(NARROW, 2, EvidenceSupport.IMPLIES, cite("states")),
        )
    )
    judge = ClaimJudge({})
    writer = FakeWriter(restated=BROAD)
    await label(result, judge, writer)
    quotes = [shown["quote"] for shown in writer.restate_requests[0]["quotes"]]
    assert quotes == [TEXTS["implies"], TEXTS["states"]]
    restated = [quote for text, quote in judge.calls if text == BROAD]
    assert restated == [TEXTS["implies"], TEXTS["states"]][:MAX_RESTATED_JUDGE_CALLS_PER_CLAIM]


async def test_a_critics_implying_quote_that_confirms_nothing_is_listed_for_rework():
    judge = ClaimJudge({})
    labeled = await label(critic_implied("states"), judge, FakeWriter(restated=None))
    (item,) = labeled.rework
    assert item.implied == (cite("states"),)
    assert judge.calls == []
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE


# A restatement must keep the change being explained (user's decision, 2026-11-13)

GROSS_MARGIN_NOW = metric("gm-2025", CURRENT, 0.335)


async def restate_as(text: str):
    judge = ClaimJudge({(text, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE})
    writer = FakeWriter(restated=text)
    labeled = await label(narrow("states"), judge, writer, metrics=(GROSS_MARGIN_NOW,))
    return labeled, judge, writer


async def test_the_restater_is_told_the_change_being_explained():
    _, _, writer = await restate_as(BROAD)
    assert writer.restate_requests[0]["change_being_explained"] == "gross margin"


@pytest.mark.parametrize(
    "text", ["Higher input costs lowered gross margin.", "Tariffs raised cost of sales in 2025."]
)
async def test_a_restatement_naming_the_change_or_a_metric_moving_with_it_is_judged(text):
    labeled, judge, _ = await restate_as(text)
    assert labeled.finding.claims[0].label is ExplanationLabel.CONFIRMED
    assert labeled.finding.claims[0].text == text


async def test_a_restatement_that_drops_the_change_is_not_judged():
    # The second paid run's case: the quote's own wording, no longer about the margin.
    drifted = "Tariffs had a material negative impact on results of operations."
    labeled, judge, _ = await restate_as(drifted)
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert all(text != drifted for text, _ in judge.calls)
    assert any("does not name gross margin" in failure for failure in labeled.failures)


@pytest.mark.parametrize(
    ("text", "changed", "named"),
    [
        ("Higher costs cut operating profit.", MetricName.OPERATING_MARGIN, True),
        ("Tariffs hurt results of operations.", MetricName.OPERATING_MARGIN, False),
        ("A tax charge lowered net earnings.", MetricName.NET_MARGIN, True),
        ("Price increases lifted Net Sales.", MetricName.REVENUE_GROWTH, True),
        ("Higher SM&A spending rose.", MetricName.SGA, True),
        ("Cocoa costs rose.", MetricName.GROSS_MARGIN, False),
    ],
)
def test_names_the_change_reads_how_filings_name_metrics(text, changed, named):
    assert labeling.names_the_change(text, changed) is named


def test_every_metric_has_words_for_the_check():
    assert set(labeling.METRIC_WORDS) == set(MetricName)


def test_the_restating_prompt_keeps_only_the_hypothesis_cause_and_treats_input_as_untrusted():
    prose = " ".join(RESTATE_SYSTEM.split())
    assert "Name only the hypothesis's cause" in prose
    assert '"change_being_explained"' in prose
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


# A quoted table row comes with its table's header (user's decision, 2026-11-13)

COST_ROW = "Cost of sales | 5,796,018 | 804,635 | 740,764"
SEGMENTS_HEADER = "\n".join(TEXTS["segments"].split("\n")[:4])


def row(text: str) -> Citation:
    return cite("segments").model_copy(update={"quote": text})


def test_a_table_row_gets_the_opening_lines_before_it():
    chunks = FakeChunks(chunk("segments"))
    assert labeling.table_header(row(COST_ROW), chunks) == SEGMENTS_HEADER
    # A row near the top brings only the lines above it, never the rows after it.
    sales = "Net sales | $ | 9,479,709 | $ | 1,271,299 | $ | 941,568 | $ | 11,692,576"
    assert labeling.table_header(row(sales), chunks) == "\n".join(TEXTS["segments"].split("\n")[:3])


def test_prose_and_unknown_chunks_get_no_table_header():
    chunks = FakeChunks(chunk("segments"), chunk("states"))
    assert labeling.table_header(cite("states"), chunks) is None
    # Prose quoted from a later line of its chunk is still not a table row.
    prose = chunk("states").model_copy(
        update={"chunk_id": "prose", "text": "Sales rose in 2025.\nCosts rose faster."}
    )
    later = cite("states").model_copy(update={"chunk_id": "prose", "quote": "Costs rose faster."})
    assert labeling.table_header(later, FakeChunks(prose)) is None
    unknown = row(COST_ROW).model_copy(update={"chunk_id": "missing"})
    assert labeling.table_header(unknown, chunks) is None


async def test_the_writer_sees_a_table_rows_header_and_prose_without_one():
    pricing_rows = branch(PRICING, 1, EvidenceSupport.CONSISTENT, cite("pricing"), row(COST_ROW))
    writer = FakeWriter("Competitors promoted more; the filings do not link this.")
    await label(resolved(survivor(PRICING, pricing_rows)), FakeJudge({}), writer)
    prose, table = writer.requests[0]["quotes"]
    assert "table_header" not in prose
    assert table == {"period": "FY2025", "quote": COST_ROW, "table_header": SEGMENTS_HEADER}


async def test_the_judge_reads_a_table_row_under_its_header_and_prose_as_it_is():
    result = resolved(
        survivor(
            INPUT_COSTS,
            branch(INPUT_COSTS, 1, EvidenceSupport.STATES_CAUSE, cite("states"), row(COST_ROW)),
        )
    )
    judge = FakeJudge({}, default=EntailmentVerdict.UNRELATED)
    await label(result, judge, FakeWriter())
    assert [quote for _, quote in judge.calls] == [
        TEXTS["states"],
        f"{SEGMENTS_HEADER}\n{COST_ROW}",
    ]


async def test_the_restater_sees_a_table_rows_header():
    result = resolved(
        survivor(NARROW, branch(NARROW, 1, EvidenceSupport.STATES_CAUSE, row(COST_ROW)))
    )
    writer = FakeWriter(restated=None)
    await label(result, ClaimJudge({}), writer)
    assert writer.restate_requests[0]["quotes"][0]["table_header"] == SEGMENTS_HEADER


def test_the_writer_is_told_a_rows_figures_mean_only_what_its_labels_say():
    prose = " ".join(WORDING_SYSTEM.split())
    assert '"table_header"' in prose
    assert "Never call them periods, years or anything else the labels do not name" in prose
    assert '"table_header"' in " ".join(RESTATE_SYSTEM.split())


def test_the_fixed_texts_and_the_prompt_example_pass_the_causal_check():
    example = (
        "Cocoa prices rose and gross margin fell; "
        "this brief did not confirm a link to gross margin."
    )
    assert example in " ".join(WORDING_SYSTEM.split())
    for text in (example, FALLBACK_PLAUSIBLE_TEXT, NO_EVIDENCE_TEXT):
        assert find_causal_language(text) == ()


def test_plausible_wording_never_claims_what_the_filings_do_not_say():
    # Third paid Hershey run (2026-11-15): "the filings do not link these to the margin decline"
    # beside a quote in which the 10-K did. Nothing checks a whole filing, so no text may say so.
    prose = " ".join(WORDING_SYSTEM.split())
    assert "this brief did not confirm a link" in prose
    assert "Never say that the filings do not link them" in prose
    assert "the filings do not link the two" not in prose
    assert "the filings do not state" not in FALLBACK_PLAUSIBLE_TEXT


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


# -- the rework loop (T5.5): which claims go back, and relabeling on a round's evidence -------


async def test_a_claim_its_quote_only_implied_is_listed_for_rework():
    labeled = await label(narrow("states"), ClaimJudge({}), FakeWriter(restated=None))
    (item,) = labeled.rework
    assert item.claim_id == NARROW.node_id
    assert item.implied == (cite("states"),)
    assert (item.unjudged, item.review_note) == ((), None)


async def test_a_judge_failure_is_listed_for_rework_with_its_review_note():
    judge = ClaimJudge({(NARROW.thought, TEXTS["states"]): LLMError("no parsable output")})
    labeled = await label(narrow("states"), judge, FakeWriter())
    (item,) = labeled.rework
    assert item.unjudged == (cite("states"),)
    assert item.review_note is not None and NARROW.node_id in item.review_note
    assert labeled.review_notes == (item.review_note,)
    assert labeled.finding.review_reason == item.review_note


@pytest.mark.parametrize("verdict", [EntailmentVerdict.UNRELATED, EntailmentVerdict.STATES_CAUSE])
async def test_an_unrelated_or_confirmed_claim_is_not_listed(verdict):
    labeled = await label(narrow("states"), ClaimJudge({}, default=verdict), FakeWriter())
    assert labeled.rework == ()


async def test_a_claim_the_judge_never_saw_is_not_listed():
    # Its critic rated no quote as stating the cause, so there was nothing close to confirm.
    result = resolved(
        survivor(PRICING, branch(PRICING, 1, EvidenceSupport.CONSISTENT, cite("pricing")))
    )
    labeled = await label(result, ClaimJudge({}), FakeWriter())
    assert labeled.rework == ()


def new_branch(parent: TreeNode, evidence: EvidenceSupport, *citations: Citation) -> TreeNode:
    found = branch(parent, 1, evidence, *citations)
    return found.model_copy(update={"node_id": f"{parent.node_id}:w1:e1"})


async def relabel(labeled, ctx, *evidence: TreeNode, node: TreeNode = NARROW):
    return await labeling.relabel_claim(labeled, node.node_id, node, evidence, COMPARISON, ctx)


async def test_a_new_quote_that_states_the_cause_confirms_the_claim():
    judge = ClaimJudge({(NARROW.thought, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE})
    writer = FakeWriter(restated=None)
    ctx = context(judge, writer)
    labeled = await label_result(narrow("implies"), MARGIN_CHANGE, COMPARISON, ctx, RUN)
    before = labeled.judge_calls
    writer.restated = ECHO

    after = await relabel(
        labeled, ctx, new_branch(NARROW, EvidenceSupport.STATES_CAUSE, cite("states"))
    )

    (claim,) = after.finding.claims
    assert claim.label is ExplanationLabel.CONFIRMED
    assert claim.text == NARROW.thought
    assert claim.citations == (cite("states"),)
    assert after.rework == ()
    # The quote judged in the first round is not judged as written again.
    as_written = [quote for text, quote in judge.calls if text == NARROW.thought]
    assert as_written == [TEXTS["implies"], TEXTS["states"]]
    assert after.judge_calls == before + 1


async def test_a_new_quote_rated_below_implies_is_not_judged():
    judge = ClaimJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    writer = FakeWriter(restated=BROAD)
    ctx = context(judge, writer)
    labeled = await label_result(
        narrow("implies"),
        MARGIN_CHANGE,
        COMPARISON,
        context(ClaimJudge({}), FakeWriter(restated=None)),
        RUN,
    )
    after = await relabel(
        labeled, ctx, new_branch(NARROW, EvidenceSupport.CONSISTENT, cite("states"))
    )
    assert judge.calls == []
    assert writer.restate_requests == []
    assert after.finding.claims == labeled.finding.claims
    assert after.rework == labeled.rework


async def test_a_new_quote_the_critic_found_only_implying_is_judged_as_restated():
    judge = ClaimJudge({(BROAD, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE})
    writer = FakeWriter(restated=None)
    ctx = context(judge, writer)
    labeled = await label_result(narrow("implies"), MARGIN_CHANGE, COMPARISON, ctx, RUN)
    calls_before = len(judge.calls)
    writer.restated = BROAD

    after = await relabel(labeled, ctx, new_branch(NARROW, EvidenceSupport.IMPLIES, cite("states")))

    quotes = [shown["quote"] for shown in writer.restate_requests[-1]["quotes"]]
    assert quotes == [TEXTS["states"], TEXTS["implies"]]
    assert judge.calls[calls_before:] == [(BROAD, TEXTS["states"])]
    (claim,) = after.finding.claims
    assert (claim.label, claim.text) == (ExplanationLabel.CONFIRMED, BROAD)


async def test_a_critics_implying_quote_from_an_earlier_round_brings_no_new_restatement():
    writer = FakeWriter(restated=None)
    ctx = context(ClaimJudge({}), writer)
    labeled = await label_result(critic_implied("states"), MARGIN_CHANGE, COMPARISON, ctx, RUN)
    requests_before = len(writer.restate_requests)
    after = await relabel(labeled, ctx, new_branch(NARROW, EvidenceSupport.IMPLIES, cite("states")))
    assert len(writer.restate_requests) == requests_before
    assert after.rework == labeled.rework


async def test_a_restatement_is_written_from_new_and_earlier_implied_quotes():
    judge = ClaimJudge({(BROAD, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE})
    writer = FakeWriter(restated=None)
    ctx = context(judge, writer)
    labeled = await label_result(narrow("implies"), MARGIN_CHANGE, COMPARISON, ctx, RUN)
    writer.restated = BROAD

    after = await relabel(
        labeled, ctx, new_branch(NARROW, EvidenceSupport.STATES_CAUSE, cite("states"))
    )

    quotes = [shown["quote"] for shown in writer.restate_requests[-1]["quotes"]]
    assert quotes == [TEXTS["states"], TEXTS["implies"]]
    (claim,) = after.finding.claims
    assert (claim.label, claim.text) == (ExplanationLabel.CONFIRMED, BROAD)


async def test_without_a_new_implying_quote_no_restatement_is_made():
    writer = FakeWriter(restated=BROAD)
    ctx = context(ClaimJudge({}, default=EntailmentVerdict.UNRELATED), writer)
    labeled = await label_result(
        narrow("implies"), MARGIN_CHANGE, COMPARISON, context(ClaimJudge({}), FakeWriter()), RUN
    )
    after = await relabel(
        labeled, ctx, new_branch(NARROW, EvidenceSupport.STATES_CAUSE, cite("states"))
    )
    assert writer.restate_requests == []
    # What the earlier round found still stands for the next one.
    assert after.rework[0].implied == (cite("implies"),)


async def test_a_quote_the_judge_failed_on_is_judged_again_first_and_its_note_cleared():
    judge = ClaimJudge({(NARROW.thought, TEXTS["states"]): LLMError("no parsable output")})
    ctx = context(judge, FakeWriter())
    labeled = await label_result(narrow("states"), MARGIN_CHANGE, COMPARISON, ctx, RUN)
    assert labeled.finding.needs_review
    judge.by_claim[(NARROW.thought, TEXTS["states"])] = EntailmentVerdict.STATES_CAUSE

    after = await relabel(labeled, ctx)

    assert after.finding.claims[0].label is ExplanationLabel.CONFIRMED
    assert after.finding.review_reason is None
    assert after.review_notes == ()
    assert after.rework == ()


async def test_a_judge_failure_that_repeats_keeps_the_finding_under_review():
    judge = ClaimJudge({(NARROW.thought, TEXTS["states"]): LLMError("no parsable output")})
    ctx = context(judge, FakeWriter())
    labeled = await label_result(narrow("states"), MARGIN_CHANGE, COMPARISON, ctx, RUN)
    after = await relabel(labeled, ctx)
    assert after.finding.review_reason == labeled.finding.review_reason
    assert after.rework[0].unjudged == (cite("states"),)


async def test_the_judge_ruling_on_a_failed_quote_clears_its_note_without_confirming():
    judge = ClaimJudge({(NARROW.thought, TEXTS["states"]): LLMError("no parsable output")})
    ctx = context(judge, FakeWriter())
    labeled = await label_result(narrow("states"), MARGIN_CHANGE, COMPARISON, ctx, RUN)
    judge.by_claim[(NARROW.thought, TEXTS["states"])] = EntailmentVerdict.UNRELATED
    after = await relabel(labeled, ctx)
    assert after.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert after.finding.review_reason is None
    assert after.rework == ()


async def test_a_note_stands_when_the_round_made_no_judge_call():
    # A restatement's judge failure leaves no quote to retry as written, so nothing rules again.
    judge = ClaimJudge({(BROAD, TEXTS["states"]): LLMRefusalError("claude-opus-5-5", None)})
    ctx = context(judge, FakeWriter(restated=BROAD))
    labeled = await label_result(narrow("states"), MARGIN_CHANGE, COMPARISON, ctx, RUN)
    after = await relabel(labeled, ctx)
    assert after.finding.review_reason == labeled.finding.review_reason
    assert after.rework[0].review_note == labeled.rework[0].review_note


async def test_the_review_note_keeps_its_place_among_the_others():
    result = narrow("states").model_copy(update={"failures": ("node x: refused",)})
    judge = ClaimJudge({(NARROW.thought, TEXTS["states"]): LLMError("no parsable output")})
    ctx = context(judge, FakeWriter())
    labeled = await label_result(result, MARGIN_CHANGE, COMPARISON, ctx, RUN)
    lost, failed = labeled.review_notes
    judge.by_claim[(NARROW.thought, TEXTS["states"])] = LLMError("still no parsable output")
    after = await relabel(labeled, ctx)
    assert after.review_notes[0] == lost
    assert "still no parsable output" in after.review_notes[1]
    assert after.finding.review_reason == "; ".join(after.review_notes)


async def test_relabeling_a_claim_not_waiting_for_rework_is_refused():
    labeled = await label(
        narrow("states"), ClaimJudge({}, default=EntailmentVerdict.UNRELATED), FakeWriter()
    )
    with pytest.raises(ValueError, match="not waiting for rework"):
        await relabel(labeled, context(ClaimJudge({}), FakeWriter()))


async def test_relabeling_with_another_candidate_is_refused():
    labeled = await label(narrow("states"), ClaimJudge({}), FakeWriter())
    with pytest.raises(ValueError, match="is not the candidate"):
        await labeling.relabel_claim(
            labeled, NARROW.node_id, PRICING, (), COMPARISON, context(ClaimJudge({}), FakeWriter())
        )


async def test_a_reworked_claim_confirmed_as_an_existing_sentence_is_merged():
    other = candidate(4, "Tariffs on imported packaging raised costs and lowered gross margin.")
    result = resolved(
        survivor(other, branch(other, 1, EvidenceSupport.STATES_CAUSE, cite("states"))),
        survivor(NARROW, branch(NARROW, 1, EvidenceSupport.STATES_CAUSE, cite("implies"))),
    )
    judge = ClaimJudge({(BROAD, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE})
    writer = FakeWriter("Costs rose; the filings do not link them.", restated=BROAD)
    ctx = context(judge, writer)
    labeled = await label_result(result, MARGIN_CHANGE, COMPARISON, ctx, RUN)
    assert [c.label for c in labeled.finding.claims] == [
        ExplanationLabel.CONFIRMED,
        ExplanationLabel.PLAUSIBLE,
    ]
    judge.by_claim[(BROAD, TEXTS["fits"])] = EntailmentVerdict.STATES_CAUSE
    after = await relabel(
        labeled, ctx, new_branch(NARROW, EvidenceSupport.STATES_CAUSE, cite("fits"))
    )
    (claim,) = after.finding.claims
    assert claim.claim_id == other.node_id
    assert claim.citations == (cite("states"), cite("fits"))


async def test_a_round_that_finds_an_earlier_implying_quote_again_does_not_judge_it_again():
    judge = ClaimJudge({})
    ctx = context(judge, FakeWriter(restated=None))
    labeled = await label_result(narrow("implies"), MARGIN_CHANGE, COMPARISON, ctx, RUN)
    calls = len(judge.calls)
    after = await relabel(
        labeled, ctx, new_branch(NARROW, EvidenceSupport.STATES_CAUSE, cite("implies"))
    )
    assert len(judge.calls) == calls
    assert after.rework == labeled.rework


# -- the guards from the T6.4 Hershey labeling run (user's decision, 2026-11-13) --------------

QUARTER_MARGIN = metric("gm-2025", CURRENT, 0.42).model_copy(
    update={"duration": Duration.THREE_MONTHS}
)


@pytest.mark.parametrize(
    ("quote", "span", "flagged"),
    [
        ("Cost of sales fell in the first six months of 2026.", Duration.THREE_MONTHS, True),
        ("Sales rose in the nine-month period, year-to-date.", Duration.THREE_MONTHS, True),
        ("For the year ended December 31, 2025, costs rose.", Duration.THREE_MONTHS, True),
        (
            "Margin rose in the second quarter and in the first six months of 2026.",
            Duration.THREE_MONTHS,
            False,
        ),
        ("Higher tariff costs lowered gross margin.", Duration.THREE_MONTHS, False),
        # A quarter is part of its year: a charge recorded then still explains the year.
        ("The charge was recorded in the fourth quarter of 2025.", Duration.TWELVE_MONTHS, False),
        ("For the full year, costs rose.", Duration.YEAR_TO_DATE, True),
        ("Costs rose in the second quarter.", Duration.YEAR_TO_DATE, False),
        ("Cost of sales fell in the first six months of 2026.", None, False),
    ],
)
def test_a_quote_covering_a_longer_span_than_the_change_is_flagged(quote, span, flagged):
    assert (labeling.span_problem(quote, span) is not None) is flagged


def test_the_changes_span_comes_from_its_metric():
    assert labeling.change_span(MARGIN_CHANGE, FakeMetrics(QUARTER_MARGIN)) is Duration.THREE_MONTHS
    assert labeling.change_span(MARGIN_CHANGE, FakeMetrics()) is None
    text_change = MARGIN_CHANGE.model_copy(update={"kind": ChangeKind.ADDED, "metric_id": None})
    assert labeling.change_span(text_change, FakeMetrics(QUARTER_MARGIN)) is None


async def test_a_six_month_quote_cannot_confirm_a_quarterly_change():
    judge = ClaimJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    labeled = await label(narrow("six_months"), judge, FakeWriter(), metrics=(QUARTER_MARGIN,))
    assert judge.calls == []
    # Nor can it back a plausible claim (user's decision, 2026-11-15), so with no other quote the
    # claim is left out and the finding goes to review.
    assert labeled.finding.claims == ()
    assert labeled.finding.review_reason == labeling.SPAN_REASON
    assert any("longer span" in failure for failure in labeled.failures)


async def test_the_same_quote_confirms_a_change_over_its_own_span():
    judge = ClaimJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    labeled = await label(narrow("six_months"), judge, FakeWriter(), metrics=())
    assert labeled.finding.claims[0].label is ExplanationLabel.CONFIRMED


async def test_a_table_rows_header_counts_toward_its_span():
    # The row names no span; its table opens "For the year ended December 31, 2025".
    row = cite("segments").model_copy(
        update={"quote": "Cost of sales | 5,796,018 | 804,635 | 740,764"}
    )
    assert labeling.span_problem(row.quote, Duration.THREE_MONTHS) is None
    result = resolved(survivor(NARROW, branch(NARROW, 1, EvidenceSupport.STATES_CAUSE, row)))
    judge = ClaimJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    labeled = await label(result, judge, FakeWriter(), metrics=(QUARTER_MARGIN,))
    assert judge.calls == []
    assert any("longer span" in failure for failure in labeled.failures)


async def test_a_rework_round_applies_the_span_guard_too():
    judge = ClaimJudge({})
    ctx = context(judge, FakeWriter(restated=None), metrics=(QUARTER_MARGIN,))
    labeled = await label_result(narrow("implies"), MARGIN_CHANGE, COMPARISON, ctx, RUN)
    judge.default = EntailmentVerdict.STATES_CAUSE
    calls = len(judge.calls)
    after = await relabel(
        labeled, ctx, new_branch(NARROW, EvidenceSupport.STATES_CAUSE, cite("six_months"))
    )
    assert len(judge.calls) == calls
    assert after.finding.claims[0].label is ExplanationLabel.PLAUSIBLE


DENYING = "Higher input costs raised net sales; the quotes do not tie this to gross margin."


@pytest.mark.parametrize(
    "text",
    [
        DENYING,
        "Higher input costs lowered gross margin, which the filings don't link.",
        "Higher input costs, not pricing, lowered gross margin.",
        "Gross margin fell with no stated cause.",
    ],
)
async def test_a_restatement_that_denies_or_hedges_is_never_judged(text):
    judge = ClaimJudge({(text, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE})
    labeled = await label(narrow("states"), judge, FakeWriter(restated=text))
    assert [claim for claim, _ in judge.calls] == [NARROW.thought]
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert any("denies or hedges a link" in failure for failure in labeled.failures)


@pytest.mark.parametrize(
    "text", ["Higher input costs lowered gross margin.", "Notably higher costs knocked margins."]
)
def test_the_denial_check_leaves_plain_causal_sentences_alone(text):
    assert labeling.DENIAL.search(text) is None


def test_the_restater_is_told_to_give_null_rather_than_deny():
    assert "Never deny or hedge a link" in RESTATE_SYSTEM


# -- a claim confirmed as written is confirmed at its quote's level (user's decision, 2026-11-15)

# From the T6.4 Hershey run: the quote states the compensation cause, not the added inference.
OVERREACH = candidate(
    5,
    "Excluding advertising, costs grew because compensation rose, meaning headcount, not "
    "marketing, is the main source of the increase.",
)
QUOTE_LEVEL = "Higher input costs lowered gross margin."


def overreach(*keys: str) -> ToTResult:
    return resolved(
        survivor(OVERREACH, branch(OVERREACH, 1, EvidenceSupport.STATES_CAUSE, *map(cite, keys)))
    )


async def test_a_confirmed_hypothesis_is_restated_and_the_restatement_confirmed():
    judge = ClaimJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    writer = FakeWriter(restated=QUOTE_LEVEL)
    labeled = await label(overreach("states"), judge, writer)
    (claim,) = labeled.finding.claims
    assert (claim.label, claim.text) == (ExplanationLabel.CONFIRMED, QUOTE_LEVEL)
    assert claim.citations == (cite("states"),)
    assert judge.calls == [(OVERREACH.thought, TEXTS["states"]), (QUOTE_LEVEL, TEXTS["states"])]
    # The restater saw the quote that confirmed the hypothesis as written.
    assert [q["quote"] for q in writer.restate_requests[0]["quotes"]] == [TEXTS["states"]]


async def test_a_hypothesis_already_at_its_quotes_level_is_confirmed_as_written():
    judge = ClaimJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    labeled = await label(narrow("states"), judge, FakeWriter())
    (claim,) = labeled.finding.claims
    assert (claim.label, claim.text) == (ExplanationLabel.CONFIRMED, NARROW.thought)
    # No second judge call: the judge already confirmed this very text.
    assert len(judge.calls) == 1


async def test_a_restatement_the_judge_does_not_confirm_leaves_the_claim_plausible():
    judge = ClaimJudge({(OVERREACH.thought, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE})
    labeled = await label(overreach("states"), judge, FakeWriter(restated=QUOTE_LEVEL))
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    # The quote still bears on the claim, so the rework loop may look further.
    (item,) = labeled.rework
    assert item.implied == (cite("states"),)


async def test_no_quote_level_wording_leaves_a_confirmed_hypothesis_unconfirmed():
    judge = ClaimJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    labeled = await label(overreach("states"), judge, FakeWriter(restated=None))
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert any("no wording at the quote's level" in failure for failure in labeled.failures)


async def test_a_hypothesis_kept_as_written_must_still_pass_the_checks():
    # The restater kept the hypothesis, whose "not marketing" denies a link.
    judge = ClaimJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    labeled = await label(overreach("states"), judge, FakeWriter())
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert any(
        "the hypothesis" in failure and "denies or hedges a link" in failure
        for failure in labeled.failures
    )


async def test_a_quote_level_restatement_of_a_numeric_change_must_name_it():
    judge = ClaimJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    writer = FakeWriter(restated="Higher input costs raised net sales.")
    labeled = await label(narrow("states"), judge, writer, metrics=(GROSS_MARGIN_NOW,))
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert any("does not name gross margin" in failure for failure in labeled.failures)


# -- a text change's claim must be about what the change is about (user's decision, 2026-11-15)

SGA_TEXT = Change(
    change_id="chg-gm",
    comparison_id="HSY-yoy",
    kind=ChangeKind.MODIFIED,
    section=Section.MDNA,
    prior_text="SM&A expenses were $2,373.6 million in 2024.",
    current_text="SM&A expenses were $2,460.6 million in 2025, driven by higher compensation.",
)
MARGIN_TEXT = SGA_TEXT.model_copy(
    update={
        "current_text": "Gross margin was 33.5% in 2025.",
        "prior_text": "Gross margin was 47.3% in 2024.",
    }
)
NEW_RISK = SGA_TEXT.model_copy(
    update={
        "kind": ChangeKind.ADDED,
        "current_text": "Conflict in the Middle East.",
        "prior_text": None,
    }
)


async def label_text(change: Change, writer: FakeWriter):
    judge = ClaimJudge({}, default=EntailmentVerdict.STATES_CAUSE)
    return await label_result(narrow("states"), change, COMPARISON, context(judge, writer), RUN)


async def test_a_text_changes_claim_about_another_line_is_not_confirmed():
    labeled = await label_text(SGA_TEXT, FakeWriter(restated=QUOTE_LEVEL))
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert any(
        "names gross margin, which the change does not mention" in failure
        for failure in labeled.failures
    )


async def test_a_text_changes_claim_about_its_own_line_is_confirmed():
    labeled = await label_text(MARGIN_TEXT, FakeWriter(restated=QUOTE_LEVEL))
    assert labeled.finding.claims[0].label is ExplanationLabel.CONFIRMED


async def test_a_line_that_moves_with_the_changes_line_fits():
    labeled = await label_text(MARGIN_TEXT, FakeWriter(restated="Tariffs raised cost of sales."))
    assert labeled.finding.claims[0].label is ExplanationLabel.CONFIRMED


async def test_a_text_change_that_names_no_line_checks_nothing():
    labeled = await label_text(NEW_RISK, FakeWriter(restated=QUOTE_LEVEL))
    assert labeled.finding.claims[0].label is ExplanationLabel.CONFIRMED


async def test_a_claim_that_names_no_line_fits_any_text_change():
    writer = FakeWriter(restated="Higher compensation costs rose in 2025.")
    labeled = await label_text(SGA_TEXT, writer)
    assert labeled.finding.claims[0].label is ExplanationLabel.CONFIRMED


async def test_the_restater_is_told_what_a_text_change_is_about():
    writer = FakeWriter(restated=QUOTE_LEVEL)
    await label_text(SGA_TEXT, writer)
    assert writer.restate_requests[0]["change_being_explained"] == "selling, marketing"
    writer = FakeWriter()
    await label_text(NEW_RISK, writer)
    assert "change_being_explained" not in writer.restate_requests[0]


def test_lines_are_read_as_filings_word_them():
    assert labeling.lines_named("SM&A expenses rose") == {MetricName.SGA, MetricName.SGA_RATIO}
    assert labeling.fits_lines("Costs rose.", frozenset({MetricName.SGA}))
    assert not labeling.fits_lines("Gross margin fell.", frozenset({MetricName.SGA}))
    assert labeling.fits_lines("Gross margin fell.", frozenset())


async def test_the_quote_level_wording_is_written_from_the_confirming_quote_only():
    judge = ClaimJudge(
        {
            (OVERREACH.thought, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE,
            (QUOTE_LEVEL, TEXTS["states"]): EntailmentVerdict.STATES_CAUSE,
        }
    )
    writer = FakeWriter(restated=QUOTE_LEVEL)
    labeled = await label(overreach("implies", "states"), judge, writer)
    assert [q["quote"] for q in writer.restate_requests[0]["quotes"]] == [TEXTS["states"]]
    (claim,) = labeled.finding.claims
    assert (claim.text, claim.citations) == (QUOTE_LEVEL, (cite("states"),))


# -- a back-reference: the filing explains the change by another line's factors -------------
# (user's decision, 2026-11-15, option A; Hershey's FY2025 10-K wording)

BACK_GROSS = (
    "Gross margin was 33.5% in 2025 compared with 47.3% in 2024. The decrease was driven by "
    "higher tariff costs."
)
BACK_SENTENCE = (
    "Operating profit margin decreased to 12.3% in 2025 from 25.9% in 2024 by the same factors "
    "noted above in gross margin."
)
BACK_JUDGED = f"{BACK_GROSS}\n\n[...]\n\n{BACK_SENTENCE}"
TARIFFS = candidate(4, "New tariffs on imported cocoa raised costs and lowered operating margin.")
CHAINED = "Higher tariff costs lowered operating margin."
OPERATING_CHANGE = Change(
    change_id="chg-gm",
    comparison_id="HSY-yoy",
    kind=ChangeKind.NUMERIC,
    metric_id=metric_id_for(COMPARISON.current, MetricName.OPERATING_MARGIN),
)


def back_chunk(n: int, text: str) -> Chunk:
    return chunk("fits").model_copy(
        update={"chunk_id": f"{NOW_ACC}:mdna:{n:04d}:0", "paragraph_id": f"bp{n}", "text": text}
    )


def back_cite(n: int, text: str) -> Citation:
    return cite("fits").model_copy(update={"chunk_id": f"{NOW_ACC}:mdna:{n:04d}:0", "quote": text})


class FilingStore(FakeChunks):
    """A `ChunkLookup` that also lists a filing's chunks, as `FactStore` does."""

    def chunks(self, accession: str | None = None) -> list[Chunk]:
        return [item for item in self._chunks.values() if accession in (None, item.accession)]


def ratio(name: MetricName, filing_: Filing, value: float) -> MetricValue:
    return metric(metric_id_for(filing_, name), filing_.period, value).model_copy(
        update={"name": name.value}
    )


class QueuedRestater(FakeWriter):
    """A writer whose restatements are played back in order, one per call."""

    def __init__(self, *restated: Any) -> None:
        super().__init__()
        self.queue = list(restated)

    async def restate(self, request: dict) -> Restatement:
        self.restate_requests.append(request)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return Restatement(text=request["hypothesis"] if item is ECHO else item)


BACK_METRICS = (
    ratio(MetricName.OPERATING_MARGIN, COMPARISON.current, 0.1233),
    ratio(MetricName.OPERATING_MARGIN, COMPARISON.prior, 0.2591),
    ratio(MetricName.GROSS_MARGIN, COMPARISON.current, 0.335),
    ratio(MetricName.GROSS_MARGIN, COMPARISON.prior, 0.473),
)


def back_context(
    judge: FakeJudge, writer: FakeWriter, *, filing: bool = True, metrics=BACK_METRICS
) -> LabelingContext:
    store = FilingStore(chunk("fits"), back_chunk(1, BACK_GROSS), back_chunk(2, BACK_SENTENCE))
    metrics = FakeMetrics(*metrics)
    return LabelingContext(
        judge=judge,
        writer=writer,
        chunks=store,
        metrics=metrics,
        facts=NoFacts(),
        filing=store if filing else None,
    )


async def label_operating(judge: FakeJudge, writer: FakeWriter, **kwargs: Any):
    result = resolved(
        survivor(TARIFFS, branch(TARIFFS, 1, EvidenceSupport.STATES_CAUSE, cite("fits")))
    )
    return await label_result(
        result, OPERATING_CHANGE, COMPARISON, back_context(judge, writer, **kwargs), RUN
    )


async def test_a_claim_its_quotes_leave_unconfirmed_is_confirmed_on_the_back_reference():
    judge = ClaimJudge({(CHAINED, BACK_JUDGED): EntailmentVerdict.STATES_CAUSE})
    writer = QueuedRestater(None, CHAINED)
    labeled = await label_operating(judge, writer)
    (claim,) = labeled.finding.claims
    assert (claim.label, claim.text) == (ExplanationLabel.CONFIRMED, CHAINED)
    assert claim.entailment is EntailmentVerdict.STATES_CAUSE
    assert claim.citations == (back_cite(1, BACK_GROSS), back_cite(2, BACK_SENTENCE))
    # Its own quote as written first, then the restatement against both passages read together.
    assert judge.calls == [(TARIFFS.thought, TEXTS["fits"]), (CHAINED, BACK_JUDGED)]
    request = writer.restate_requests[-1]
    assert [shown["quote"] for shown in request["quotes"]] == [BACK_GROSS, BACK_SENTENCE]
    assert request["change_being_explained"] == "operating margin"
    assert labeled.rework == ()
    assert labeled.finding.review_reason is None


async def test_a_claim_confirmed_on_its_own_quotes_never_reaches_the_back_reference():
    judge = ClaimJudge({(TARIFFS.thought, TEXTS["fits"]): EntailmentVerdict.STATES_CAUSE})
    writer = QueuedRestater(ECHO)
    labeled = await label_operating(judge, writer)
    assert labeled.finding.claims[0].citations == (cite("fits"),)
    assert len(writer.restate_requests) == 1
    assert all(quote != BACK_JUDGED for _, quote in judge.calls)


async def test_no_restatement_from_the_back_reference_means_no_more_judge_calls():
    judge = ClaimJudge({})
    labeled = await label_operating(judge, QueuedRestater(None, None))
    assert judge.calls == [(TARIFFS.thought, TEXTS["fits"])]
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    # What its own quote implied still goes to rework.
    assert labeled.rework[0].implied == (cite("fits"),)


async def test_a_restatement_naming_only_the_other_line_is_not_judged():
    judge = ClaimJudge({})
    drifted = "Higher tariff costs lowered gross margin."
    labeled = await label_operating(judge, QueuedRestater(None, drifted))
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert any(
        "from the back-reference does not name operating margin" in failure
        for failure in labeled.failures
    )
    assert judge.calls == [(TARIFFS.thought, TEXTS["fits"])]


async def test_a_restatement_with_an_unbacked_figure_is_not_judged_on_the_back_reference():
    judge = ClaimJudge({})
    unbacked = "Tariff costs up 9% lowered operating margin."
    labeled = await label_operating(judge, QueuedRestater(None, unbacked))
    assert all(quote != BACK_JUDGED for _, quote in judge.calls)
    assert any("the back-reference cannot confirm" in failure for failure in labeled.failures)


async def test_a_judge_failure_on_the_back_reference_sends_the_finding_to_review():
    judge = ClaimJudge({(CHAINED, BACK_JUDGED): LLMError("no parsable output")})
    labeled = await label_operating(judge, QueuedRestater(None, CHAINED))
    assert labeled.finding.needs_review
    assert TARIFFS.node_id in labeled.finding.review_reason
    # The claim's rework item carries the note, so a round that rules on the claim replaces it.
    assert labeled.rework[0].review_note == labeled.finding.review_reason


async def test_confirming_on_the_back_reference_clears_a_judge_failure_on_its_own_quotes():
    judge = ClaimJudge(
        {
            (TARIFFS.thought, TEXTS["fits"]): LLMError("no parsable output"),
            (CHAINED, BACK_JUDGED): EntailmentVerdict.STATES_CAUSE,
        }
    )
    labeled = await label_operating(judge, QueuedRestater(CHAINED))
    assert labeled.finding.claims[0].label is ExplanationLabel.CONFIRMED
    assert labeled.finding.review_reason is None
    assert labeled.review_notes == ()


async def test_without_the_filings_chunks_no_back_reference_is_looked_for():
    judge = ClaimJudge({(CHAINED, BACK_JUDGED): EntailmentVerdict.STATES_CAUSE})
    writer = QueuedRestater(None)
    labeled = await label_operating(judge, writer, filing=False)
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert len(writer.restate_requests) == 1


def test_the_restater_is_told_how_a_back_reference_states_a_cause():
    prose = " ".join(RESTATE_SYSTEM.split())
    assert '"by the same factors" as another line' in prose
    assert "name the change, not the other line" in prose


async def test_a_back_reference_the_judge_finds_only_implied_leaves_the_claim_plausible():
    judge = ClaimJudge({(CHAINED, BACK_JUDGED): EntailmentVerdict.IMPLIES})
    labeled = await label_operating(judge, QueuedRestater(None, CHAINED))
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert judge.calls[-1] == (CHAINED, BACK_JUDGED)


async def test_a_change_whose_metric_is_unknown_is_not_looked_up():
    # Without a known metric there is no line to match, and the lookup must not be attempted.
    judge = ClaimJudge({})
    labeled = await label_operating(judge, QueuedRestater(None), metrics=())
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE


# -- paid4's issues (user's decisions, 2026-11-15) ----------------------------------------------
# (a) a confirmed claim saying no more than another is folded into it


def confirmed(n: int, text: str, key: str) -> Claim:
    return Claim(
        claim_id=f"{RUN}:chg-gm:r1:c{n}",
        text=text,
        label=ExplanationLabel.CONFIRMED,
        citations=(cite(key),),
        metric_ids=(f"m{n}",),
        entailment=EntailmentVerdict.STATES_CAUSE,
    )


TARIFF_ONLY = "Higher tariff costs lowered operating margin."
COMMODITY_AND_TARIFF = "Higher commodity and tariff costs lowered operating margin."


def test_a_confirmed_claim_inside_a_broader_one_is_folded_into_it():
    tally = labeling._Tally()
    narrow_, broad_ = (
        confirmed(1, TARIFF_ONLY, "states"),
        confirmed(2, COMMODITY_AND_TARIFF, "fits"),
    )
    (claim,) = labeling._merge_same_claims([narrow_, broad_], tally)
    assert (claim.claim_id, claim.text) == (broad_.claim_id, COMMODITY_AND_TARIFF)
    assert claim.citations == (cite("fits"), cite("states"))
    assert claim.metric_ids == ("m2", "m1")
    assert tally.failures == [
        f"claim {narrow_.claim_id}: confirmed as saying no more than {broad_.claim_id}, so merged"
    ]


def test_a_quote_both_claims_cite_is_cited_once():
    narrow_, broad_ = (
        confirmed(1, TARIFF_ONLY, "states"),
        confirmed(2, COMMODITY_AND_TARIFF, "states"),
    )
    (claim,) = labeling._merge_same_claims([narrow_, broad_], labeling._Tally())
    assert claim.citations == (cite("states"),)


def test_claims_nested_three_deep_end_in_the_broadest():
    claims = [
        confirmed(1, "Higher costs lowered operating margin.", "states"),
        confirmed(2, TARIFF_ONLY, "implies"),
        confirmed(3, COMMODITY_AND_TARIFF, "fits"),
    ]
    (claim,) = labeling._merge_same_claims(claims, labeling._Tally())
    assert claim.text == COMMODITY_AND_TARIFF
    assert claim.citations == (cite("fits"), cite("states"), cite("implies"))


def test_claims_with_their_own_words_stay_co_drivers_and_keep_their_order():
    volume = confirmed(3, "Lower volume lowered operating margin.", "implies")
    claims = [
        confirmed(1, TARIFF_ONLY, "states"),
        volume,
        confirmed(2, COMMODITY_AND_TARIFF, "fits"),
    ]
    merged = labeling._merge_same_claims(claims, labeling._Tally())
    assert [claim.text for claim in merged] == [volume.text, COMMODITY_AND_TARIFF]


def test_claims_with_the_same_words_keep_the_first():
    first = confirmed(1, "Tariff costs, higher, lowered operating margin.", "states")
    second = confirmed(2, "Higher tariff costs lowered operating margin.", "fits")
    (claim,) = labeling._merge_same_claims([first, second], labeling._Tally())
    assert claim.claim_id == first.claim_id
    assert claim.citations == (cite("states"), cite("fits"))


def test_a_plausible_claim_is_never_folded():
    plausible = confirmed(1, TARIFF_ONLY, "states").model_copy(
        update={"label": ExplanationLabel.PLAUSIBLE, "entailment": None}
    )
    claims = [plausible, confirmed(2, COMMODITY_AND_TARIFF, "fits")]
    assert labeling._merge_same_claims(claims, labeling._Tally()) == claims


# (b) the plausible wording names the change it does not confirm a link to


async def test_plausible_wording_that_does_not_name_the_change_is_rejected_and_retried():
    vague = "Promotions rose and the margin fell; this brief did not confirm a link."
    named = "Promotions rose; this brief did not confirm a link to gross margin."
    writer = FakeWriter(vague, named)
    metrics = (metric("gm-2025", CURRENT, 0.335), metric("gm-2024", PRIOR, 0.473))
    result = resolved(
        survivor(PRICING, branch(PRICING, 1, EvidenceSupport.CONSISTENT, cite("pricing")))
    )
    labeled = await label(result, FakeJudge({}), writer, metrics=metrics)
    assert labeled.finding.claims[0].text == named
    reason = writer.requests[1]["rejected"]["reason"]
    assert reason == "it does not name gross margin, the change being explained"


def test_the_writer_is_told_to_name_the_change_being_explained():
    prose = " ".join(WORDING_SYSTEM.split())
    assert 'When "change_being_explained" is given, name it as written there' in prose
    assert "this brief did not confirm a link to gross margin" in prose
    summary = " ".join(SUMMARY_SYSTEM.split())
    assert (
        "this brief did not confirm a link to its finding's change, naming that change" in summary
    )


# (c) a quote the judge found unrelated is not the plausible claim's evidence


# The operating margin the refuted survivor's metric agrees with, so its quote reaches the judge.
OPERATING_2025 = (metric("om-2025", CURRENT, 0.123),)


def refuted_only(node: TreeNode) -> TreeNode:
    return branch(
        node,
        1,
        EvidenceSupport.STATES_CAUSE,
        cite("pricing"),
        numeric=NumericCheck.AGREE,
        metric_ids=("om-2025",),
    )


async def test_a_claim_whose_every_quote_the_judge_refuted_is_left_out_and_the_finding_reviewed():
    writer = FakeWriter()
    judge = FakeJudge({}, default=EntailmentVerdict.UNRELATED)
    labeled = await label(
        resolved(survivor(PRICING, refuted_only(PRICING))), judge, writer, metrics=OPERATING_2025
    )
    assert labeled.finding.claims == ()
    assert labeled.finding.review_reason == labeling.REFUTED_REASON
    assert writer.requests == []
    assert labeled.rework == ()
    assert any(
        f"claim {PRICING.node_id}: the judge found every quote unrelated" in failure
        for failure in labeled.failures
    )


async def test_a_finding_left_empty_for_both_reasons_gives_both():
    judge = FakeJudge({}, default=EntailmentVerdict.UNRELATED)
    result = resolved(
        survivor(PRICING, refuted_only(PRICING)), survivor(INPUT_COSTS, metric_only(INPUT_COSTS))
    )
    labeled = await label(result, judge, FakeWriter(), metrics=OPERATING_2025)
    assert labeled.finding.review_reason == (
        f"{labeling.REFUTED_REASON}; {labeling.METRIC_ONLY_REASON}"
    )


async def test_the_writer_never_sees_a_refuted_quote():
    judge = FakeJudge({TEXTS["states"]: EntailmentVerdict.UNRELATED})
    writer = FakeWriter()
    await label(gross_margin_example(), judge, writer)
    quotes = [shown["quote"] for shown in writer.requests[0]["quotes"]]
    assert TEXTS["states"] not in quotes
    assert quotes == [TEXTS["implies"], TEXTS["fits"]]


def test_the_refuted_reason_names_no_cause():
    assert find_causal_language(labeling.REFUTED_REASON) == ()


# -- paid5's issues (user's decisions, 2026-11-15) ----------------------------------------------
# (1) a plausible claim never mentions the hypothesis, which the reader does not see


@pytest.mark.parametrize(
    "leaked",
    [
        "Competitors promoted more and the hypothesis cites gross margin; this brief did not "
        "confirm a link to gross margin.",
        "Competitors promoted more, as hypothesized; this brief did not confirm a link to gross "
        "margin.",
        "Hypotheses aside, competitors promoted more; this brief did not confirm a link to gross "
        "margin.",
    ],
)
async def test_wording_that_mentions_the_hypothesis_is_rejected_and_retried(leaked):
    named = "Competitors promoted more; this brief did not confirm a link to gross margin."
    writer = FakeWriter(leaked, named)
    metrics = (metric("gm-2025", CURRENT, 0.335), metric("gm-2024", PRIOR, 0.473))
    labeled = await label(
        resolved(gross_margin_example().candidates[1]), FakeJudge({}), writer, metrics=metrics
    )
    assert labeled.finding.claims[0].text == named
    reason = writer.requests[1]["rejected"]["reason"]
    assert reason == "it mentions the hypothesis, which the reader does not see"


def test_the_writer_is_told_never_to_mention_the_hypothesis():
    assert "Never mention the hypothesis: the reader does not see it." in " ".join(
        WORDING_SYSTEM.split()
    )


# (2) a claim whose quotes are only table rows is left out when no description of them passes

CAUSAL = ("Promotions drove the decline.", "Margin fell because of promotions.")


async def test_a_table_only_claim_with_no_passing_wording_is_left_out_and_reviewed():
    rows = branch(PRICING, 1, EvidenceSupport.CONSISTENT, row(COST_ROW))
    labeled = await label(resolved(survivor(PRICING, rows)), FakeJudge({}), FakeWriter(*CAUSAL))
    assert labeled.finding.claims == ()
    assert labeled.finding.review_reason == labeling.TABLE_ONLY_REASON
    assert labeled.failures[-1] == (
        f"claim {PRICING.node_id}: no wording passed and its quotes are only table rows, so it is "
        "left out of the brief"
    )
    assert not any("used the fixed wording" in failure for failure in labeled.failures)


async def test_a_left_out_claim_gets_no_rework_item():
    # The judge finds the row only implies it, which would list a kept claim for rework.
    rows = branch(PRICING, 1, EvidenceSupport.STATES_CAUSE, row(COST_ROW))
    writer = FakeWriter(*CAUSAL, restated=None)
    labeled = await label(resolved(survivor(PRICING, rows)), FakeJudge({}), writer)
    assert labeled.finding.claims == ()
    assert labeled.rework == ()


async def test_a_claim_with_a_prose_quote_beside_its_rows_keeps_the_fixed_text():
    rows = branch(PRICING, 1, EvidenceSupport.CONSISTENT, row(COST_ROW), cite("pricing"))
    labeled = await label(resolved(survivor(PRICING, rows)), FakeJudge({}), FakeWriter(*CAUSAL))
    (claim,) = labeled.finding.claims
    assert claim.text == FALLBACK_PLAUSIBLE_TEXT
    assert labeled.finding.review_reason is None


async def test_a_table_only_claim_whose_wording_passes_is_kept():
    rows = branch(PRICING, 1, EvidenceSupport.CONSISTENT, row(COST_ROW))
    writer = FakeWriter(
        "Competitors promoted more; this brief did not confirm a link to gross margin."
    )
    labeled = await label(resolved(survivor(PRICING, rows)), FakeJudge({}), writer)
    (claim,) = labeled.finding.claims
    assert claim.citations == (row(COST_ROW),)


def test_the_table_only_reason_names_no_cause():
    assert find_causal_language(labeling.TABLE_ONLY_REASON) == ()


# -- paid5's brief (user's decision, 2026-11-15) -------------------------------------------------
# A plausible claim whose every quote a confirmed claim in the same finding already shows is left
# out once rework is over.


def plausible(n: int, *citations: Citation, metric_ids: tuple[str, ...] = ()) -> Claim:
    return Claim(
        claim_id=f"{RUN}:chg-gm:r1:c{n}",
        text="Freight costs rose; this brief did not confirm a link to gross margin.",
        label=ExplanationLabel.PLAUSIBLE,
        citations=citations,
        metric_ids=metric_ids,
    )


def finding_of(*claims: Claim, **fields: Any) -> labeling.LabeledFinding:
    return labeling.LabeledFinding(
        finding=Finding(finding_id=f"{RUN}:chg-gm", change=MARGIN_CHANGE, claims=claims),
        **fields,
    )


def claim_ids(labeled: labeling.LabeledFinding) -> list[str]:
    return [claim.claim_id for claim in labeled.finding.claims]


def test_a_plausible_claim_citing_only_a_confirmed_claims_quote_is_left_out():
    tariffs = confirmed(1, "Tariffs raised cost of sales.", "states")
    repeat, other = plausible(2, cite("states")), plausible(3, cite("pricing"))
    labeled = labeling.leave_out_repeated_quotes(
        finding_of(tariffs, repeat, other, failures=("earlier note",))
    )
    assert claim_ids(labeled) == [tariffs.claim_id, other.claim_id]
    assert labeled.failures == (
        "earlier note",
        f"claim {repeat.claim_id}: every quote it cites is shown by confirmed claim "
        f"{tariffs.claim_id}, so it is left out of the brief",
    )


def test_a_quote_inside_a_longer_confirmed_quote_from_the_same_chunk_counts_as_shown():
    part = cite("states").model_copy(update={"quote": "due to higher input  costs,\nprimarily"})
    labeled = labeling.leave_out_repeated_quotes(
        finding_of(confirmed(1, "Input costs lowered gross margin.", "states"), plausible(2, part))
    )
    assert claim_ids(labeled) == [f"{RUN}:chg-gm:r1:c1"]


def test_a_longer_plausible_quote_around_a_confirmed_one_is_not_shown():
    short = cite("states").model_copy(update={"quote": "due to higher input costs"})
    tariffs = confirmed(1, "Input costs lowered gross margin.", "states").model_copy(
        update={"citations": (short,)}
    )
    claims = (tariffs, plausible(2, cite("states")))
    assert claim_ids(labeling.leave_out_repeated_quotes(finding_of(*claims))) == [
        c.claim_id for c in claims
    ]


def test_the_same_words_from_another_chunk_are_not_shown():
    elsewhere = cite("states").model_copy(update={"chunk_id": "other-chunk"})
    claims = (confirmed(1, "Input costs lowered gross margin.", "states"), plausible(2, elsewhere))
    assert claim_ids(labeling.leave_out_repeated_quotes(finding_of(*claims))) == [
        c.claim_id for c in claims
    ]


def test_a_plausible_claim_with_one_quote_of_its_own_is_kept():
    claims = (
        confirmed(1, "Input costs lowered gross margin.", "states"),
        plausible(2, cite("states"), cite("pricing")),
    )
    labeled = finding_of(*claims)
    assert labeling.leave_out_repeated_quotes(labeled) is labeled


def test_quotes_shown_by_two_confirmed_claims_together_count_and_both_are_named():
    first = confirmed(1, "Input costs lowered gross margin.", "states")
    second = confirmed(2, "Pricing pressure lowered gross margin.", "pricing")
    repeat = plausible(3, cite("states"), cite("pricing"))
    labeled = labeling.leave_out_repeated_quotes(finding_of(first, second, repeat))
    assert claim_ids(labeled) == [first.claim_id, second.claim_id]
    assert labeled.failures == (
        f"claim {repeat.claim_id}: every quote it cites is shown by confirmed claim "
        f"{first.claim_id}, {second.claim_id}, so it is left out of the brief",
    )


def test_the_note_names_only_the_confirmed_claims_that_show_the_quotes():
    pricing = confirmed(1, "Pricing pressure lowered gross margin.", "pricing")
    tariffs = confirmed(2, "Input costs lowered gross margin.", "states")
    repeat = plausible(3, cite("states"))
    labeled = labeling.leave_out_repeated_quotes(finding_of(pricing, tariffs, repeat))
    assert labeled.failures == (
        f"claim {repeat.claim_id}: every quote it cites is shown by confirmed claim "
        f"{tariffs.claim_id}, so it is left out of the brief",
    )


def test_an_unresolved_claim_citing_a_shown_quote_is_kept():
    unresolved = plausible(2, cite("states")).model_copy(
        update={"label": ExplanationLabel.UNRESOLVED, "text": NO_EVIDENCE_TEXT}
    )
    labeled = finding_of(confirmed(1, "Input costs lowered gross margin.", "states"), unresolved)
    assert labeling.leave_out_repeated_quotes(labeled) is labeled


def test_confirmed_claims_sharing_a_quote_are_both_kept():
    claims = (
        confirmed(1, "Input costs lowered gross margin.", "states"),
        confirmed(2, "Cocoa costs lowered gross margin.", "states"),
    )
    labeled = finding_of(*claims)
    assert labeling.leave_out_repeated_quotes(labeled) is labeled


def test_a_plausible_claim_with_no_quote_is_not_left_out_for_this():
    metric_only_ = plausible(2, metric_ids=("m2",))
    claims = (confirmed(1, "Input costs lowered gross margin.", "states"), metric_only_)
    labeled = finding_of(*claims)
    assert labeling.leave_out_repeated_quotes(labeled) is labeled


def test_without_a_confirmed_claim_nothing_is_left_out():
    labeled = finding_of(plausible(1, cite("states")), plausible(2, cite("states")))
    assert labeling.leave_out_repeated_quotes(labeled) is labeled


def test_a_left_out_claims_rework_item_goes_and_review_notes_stay():
    repeat, other = plausible(2, cite("states")), plausible(3, cite("pricing"))
    labeled = labeling.leave_out_repeated_quotes(
        finding_of(
            confirmed(1, "Input costs lowered gross margin.", "states"),
            repeat,
            other,
            rework=(
                labeling.ReworkItem(claim_id=repeat.claim_id, review_note="judge failed"),
                labeling.ReworkItem(claim_id=other.claim_id),
            ),
            review_notes=("judge failed",),
        )
    )
    assert [item.claim_id for item in labeled.rework] == [other.claim_id]
    assert labeled.review_notes == ("judge failed",)


def test_leaving_out_repeated_quotes_twice_changes_nothing_more():
    once = labeling.leave_out_repeated_quotes(
        finding_of(
            confirmed(1, "Input costs lowered gross margin.", "states"),
            plausible(2, cite("states")),
        )
    )
    assert labeling.leave_out_repeated_quotes(once) is once


def test_the_left_out_note_names_no_cause():
    labeled = labeling.leave_out_repeated_quotes(
        finding_of(
            confirmed(1, "Input costs lowered gross margin.", "states"),
            plausible(2, cite("states")),
        )
    )
    assert find_causal_language(labeled.failures[-1]) == ()


# -- the filing's own explanation of the change (user's decision, 2026-11-15) -------------------
# A claim its own quotes leave unconfirmed is tried on the paragraph that explains the change
# itself, found in code, before the back-reference.

GROSS_CHANGE = OPERATING_CHANGE.model_copy(
    update={"metric_id": metric_id_for(COMPARISON.current, MetricName.GROSS_MARGIN)}
)
COCOA = candidate(5, "Cocoa and tariff costs rose and lowered gross margin.")
OWN_TEXT = "Higher tariff costs lowered gross margin."
OWN_OPERATING = (
    "Operating margin was 12.3% in 2025 compared with 25.9% in 2024, driven by lower gross profit."
)
OWN_OPERATING_TEXT = "Lower gross profit lowered operating margin."


def own_context(
    judge: FakeJudge, writer: FakeWriter, paragraphs, *, filing: bool = True
) -> LabelingContext:
    store = FilingStore(chunk("fits"), *(back_chunk(n, text) for n, text in paragraphs))
    return LabelingContext(
        judge=judge,
        writer=writer,
        chunks=store,
        metrics=FakeMetrics(*BACK_METRICS),
        facts=NoFacts(),
        filing=store if filing else None,
    )


async def label_own(  # noqa: PLR0913 -- the test's knobs
    judge: FakeJudge,
    writer: FakeWriter,
    *branches: TreeNode,
    node: TreeNode = COCOA,
    change: Change = GROSS_CHANGE,
    paragraphs=((1, BACK_GROSS),),
    filing: bool = True,
):
    branches = branches or (branch(node, 1, EvidenceSupport.STATES_CAUSE, cite("fits")),)
    return await label_result(
        resolved(survivor(node, *branches)),
        change,
        COMPARISON,
        own_context(judge, writer, paragraphs, filing=filing),
        RUN,
    )


async def test_a_claim_its_quotes_leave_unconfirmed_is_confirmed_on_the_filings_own_explanation():
    judge = ClaimJudge({(OWN_TEXT, BACK_GROSS): EntailmentVerdict.STATES_CAUSE})
    writer = QueuedRestater(None, OWN_TEXT)
    labeled = await label_own(judge, writer)
    (claim,) = labeled.finding.claims
    assert (claim.label, claim.text) == (ExplanationLabel.CONFIRMED, OWN_TEXT)
    assert claim.citations == (back_cite(1, BACK_GROSS),)
    assert judge.calls == [(COCOA.thought, TEXTS["fits"]), (OWN_TEXT, BACK_GROSS)]
    assert [shown["quote"] for shown in writer.restate_requests[-1]["quotes"]] == [BACK_GROSS]
    assert labeled.rework == ()
    assert labeled.finding.review_reason is None


async def test_a_claim_confirmed_on_its_own_quotes_never_reaches_its_own_explanation():
    judge = ClaimJudge({(COCOA.thought, TEXTS["fits"]): EntailmentVerdict.STATES_CAUSE})
    writer = QueuedRestater(ECHO)
    labeled = await label_own(judge, writer)
    assert labeled.finding.claims[0].citations == (cite("fits"),)
    assert len(writer.restate_requests) == 1


@pytest.mark.parametrize(
    ("rating", "quote"),
    [
        (EvidenceSupport.STATES_CAUSE, BACK_GROSS),
        (EvidenceSupport.STATES_CAUSE, BACK_GROSS.split(". ")[0] + "."),
        (EvidenceSupport.IMPLIES, BACK_GROSS),
        (EvidenceSupport.STATES_CAUSE, BACK_GROSS + "\n\nCocoa costs rose."),
    ],
)
async def test_a_paragraph_the_claims_rated_quotes_overlap_is_not_tried_again(rating, quote):
    # The chunk holds the paragraph and one more, so a quote can run past the paragraph.
    paragraphs = ((1, BACK_GROSS + "\n\nCocoa costs rose."),)
    judge = ClaimJudge({})
    writer = QueuedRestater(None)  # a second restatement would empty the queue and raise
    labeled = await label_own(
        judge, writer, branch(COCOA, 1, rating, back_cite(1, quote)), paragraphs=paragraphs
    )
    assert len(writer.restate_requests) == 1
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE


async def test_a_paragraph_quoted_only_as_consistent_or_from_another_chunk_is_still_tried():
    consistent = branch(COCOA, 1, EvidenceSupport.CONSISTENT, back_cite(1, BACK_GROSS))
    cause = branch(COCOA, 2, EvidenceSupport.STATES_CAUSE, cite("fits"))
    writer = QueuedRestater(None, None)
    await label_own(ClaimJudge({}), writer, consistent, cause)
    assert len(writer.restate_requests) == 2
    elsewhere = back_cite(9, BACK_GROSS)
    writer = QueuedRestater(None, None)
    await label_own(
        ClaimJudge({}),
        writer,
        branch(COCOA, 1, EvidenceSupport.STATES_CAUSE, elsewhere),
        paragraphs=((1, BACK_GROSS), (9, BACK_GROSS)),
    )
    assert len(writer.restate_requests) == 2


async def test_the_own_explanation_is_tried_before_the_back_reference():
    judge = ClaimJudge(
        {
            (OWN_OPERATING_TEXT, OWN_OPERATING): EntailmentVerdict.IMPLIES,
            (CHAINED, BACK_JUDGED): EntailmentVerdict.STATES_CAUSE,
        }
    )
    writer = QueuedRestater(None, OWN_OPERATING_TEXT, CHAINED)
    labeled = await label_own(
        judge,
        writer,
        node=TARIFFS,
        change=OPERATING_CHANGE,
        paragraphs=((1, BACK_GROSS), (2, BACK_SENTENCE), (3, OWN_OPERATING)),
    )
    assert labeled.finding.claims[0].text == CHAINED
    assert judge.calls == [
        (TARIFFS.thought, TEXTS["fits"]),
        (OWN_OPERATING_TEXT, OWN_OPERATING),
        (CHAINED, BACK_JUDGED),
    ]


async def test_confirming_on_the_back_reference_clears_a_judge_failure_on_the_own_explanation():
    judge = ClaimJudge(
        {
            (OWN_OPERATING_TEXT, OWN_OPERATING): LLMError("no parsable output"),
            (CHAINED, BACK_JUDGED): EntailmentVerdict.STATES_CAUSE,
        }
    )
    labeled = await label_own(
        judge,
        QueuedRestater(None, OWN_OPERATING_TEXT, CHAINED),
        node=TARIFFS,
        change=OPERATING_CHANGE,
        paragraphs=((1, BACK_GROSS), (2, BACK_SENTENCE), (3, OWN_OPERATING)),
    )
    assert labeled.finding.claims[0].label is ExplanationLabel.CONFIRMED
    assert labeled.finding.review_reason is None
    assert labeled.review_notes == ()


async def test_confirming_on_the_back_reference_clears_every_judge_failure_on_the_way():
    judge = ClaimJudge(
        {
            (TARIFFS.thought, TEXTS["fits"]): LLMError("no parsable output"),
            (OWN_OPERATING_TEXT, OWN_OPERATING): LLMError("no parsable output"),
            (CHAINED, BACK_JUDGED): EntailmentVerdict.STATES_CAUSE,
        }
    )
    # A judge failure on its own quote leaves nothing to restate from it.
    labeled = await label_own(
        judge,
        QueuedRestater(OWN_OPERATING_TEXT, CHAINED),
        node=TARIFFS,
        change=OPERATING_CHANGE,
        paragraphs=((1, BACK_GROSS), (2, BACK_SENTENCE), (3, OWN_OPERATING)),
    )
    assert labeled.finding.claims[0].label is ExplanationLabel.CONFIRMED
    assert len(judge.calls) == 3
    assert labeled.finding.review_reason is None
    assert labeled.review_notes == ()


async def test_a_judge_failure_on_the_own_explanation_sends_the_finding_to_review():
    judge = ClaimJudge({(OWN_TEXT, BACK_GROSS): LLMError("no parsable output")})
    labeled = await label_own(judge, QueuedRestater(None, OWN_TEXT))
    assert labeled.finding.needs_review
    assert COCOA.node_id in labeled.finding.review_reason
    assert labeled.rework[0].review_note == labeled.finding.review_reason


async def test_a_restatement_from_the_own_explanation_must_name_the_change():
    judge = ClaimJudge({})
    labeled = await label_own(judge, QueuedRestater(None, "Tariffs hurt results."))
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert any(
        "from the filing's own explanation does not name gross margin" in failure
        for failure in labeled.failures
    )
    assert judge.calls == [(COCOA.thought, TEXTS["fits"])]


async def test_an_unbacked_figure_is_not_judged_on_the_own_explanation():
    judge = ClaimJudge({})
    labeled = await label_own(
        judge, QueuedRestater(None, "Tariff costs up 9% lowered gross margin.")
    )
    assert judge.calls == [(COCOA.thought, TEXTS["fits"])]
    assert any(
        "the filing's own explanation cannot confirm" in failure for failure in labeled.failures
    )


async def test_without_the_filings_chunks_no_own_explanation_is_looked_for():
    judge = ClaimJudge({(OWN_TEXT, BACK_GROSS): EntailmentVerdict.STATES_CAUSE})
    writer = QueuedRestater(None)
    labeled = await label_own(judge, writer, filing=False)
    assert labeled.finding.claims[0].label is ExplanationLabel.PLAUSIBLE
    assert len(writer.restate_requests) == 1


# -- a quote about a longer span than the change is not a plausible claim's evidence -----------
# (user's decision, 2026-11-15)


async def test_a_plausible_claim_drops_a_six_month_quote_and_keeps_the_rest():
    rows = branch(PRICING, 1, EvidenceSupport.CONSISTENT, cite("pricing"), cite("six_months"))
    writer = FakeWriter()
    labeled = await label(
        resolved(survivor(PRICING, rows)), FakeJudge({}), writer, metrics=(QUARTER_MARGIN,)
    )
    (claim,) = labeled.finding.claims
    assert claim.citations == (cite("pricing"),)
    assert [shown["quote"] for shown in writer.requests[0]["quotes"]] == [TEXTS["pricing"]]
    assert (
        f"claim {PRICING.node_id}: the quote from six_months covers a longer span than the "
        "change, so the claim does not cite it"
    ) in labeled.failures
    assert labeled.finding.review_reason is None


async def test_a_change_over_the_quotes_own_span_keeps_it():
    rows = branch(PRICING, 1, EvidenceSupport.CONSISTENT, cite("pricing"), cite("six_months"))
    labeled = await label(resolved(survivor(PRICING, rows)), FakeJudge({}), FakeWriter())
    assert labeled.finding.claims[0].citations == (cite("pricing"), cite("six_months"))


async def test_a_claim_left_with_only_longer_span_quotes_is_left_out_with_no_rework():
    # The critic found the six-month quote implies it, which would list a kept claim for rework.
    rows = branch(PRICING, 1, EvidenceSupport.IMPLIES, cite("six_months"))
    labeled = await label(
        resolved(survivor(PRICING, rows)),
        FakeJudge({}),
        FakeWriter(restated=None),
        metrics=(QUARTER_MARGIN,),
    )
    assert labeled.finding.claims == ()
    assert labeled.rework == ()
    assert labeled.failures[-1] == (
        f"claim {PRICING.node_id}: every quote it has covers a longer span than the change, so it "
        "is left out of the brief"
    )


async def test_a_row_whose_table_header_names_a_longer_span_is_dropped():
    table = chunk("pricing").model_copy(
        update={
            "chunk_id": "half-year",
            "text": (
                "For the six months ended June 28, 2026 | 2026 | 2025\nCost of sales | 1,000 | 900"
            ),
        }
    )
    half_row = cite("pricing").model_copy(
        update={"chunk_id": "half-year", "quote": "Cost of sales | 1,000 | 900"}
    )
    rows = branch(PRICING, 1, EvidenceSupport.CONSISTENT, cite("pricing"), half_row)
    context_ = LabelingContext(
        judge=FakeJudge({}),
        writer=FakeWriter(),
        chunks=FakeChunks(chunk("pricing"), table),
        metrics=FakeMetrics(QUARTER_MARGIN),
        facts=NoFacts(),
    )
    labeled = await label_result(
        resolved(survivor(PRICING, rows)), MARGIN_CHANGE, COMPARISON, context_, RUN
    )
    assert labeled.finding.claims[0].citations == (cite("pricing"),)


def test_the_span_reason_names_no_cause():
    assert find_causal_language(labeling.SPAN_REASON) == ()
