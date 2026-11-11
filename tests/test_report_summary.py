"""The brief's summary (T5.4): the Opus draft, its checks and the template. No real calls."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from sec_agent.llm.client import ROLE_CONFIG, LLMClient, LLMError, Role
from sec_agent.llm.ledger import BudgetExceededError, CostLedger
from sec_agent.report.brief import NO_FINDINGS_SUMMARY, check_brief
from sec_agent.report.summary import (
    SUMMARY_SYSTEM,
    SummaryContext,
    SummaryDraft,
    SummarySentence,
    SummaryWriter,
    summary_request,
    template_summary,
    write_summary,
)
from sec_agent.schemas import (
    Brief,
    Change,
    ChangeKind,
    Chunk,
    Citation,
    Claim,
    Comparison,
    ComparisonType,
    EntailmentVerdict,
    ExplanationLabel,
    Filing,
    Finding,
    FiscalPeriod,
    FormType,
    MetricValue,
    Section,
)

CURRENT = FiscalPeriod(fiscal_year=2025, fiscal_period="FY", period_end=date(2025, 12, 31))
PRIOR = FiscalPeriod(fiscal_year=2024, fiscal_period="FY", period_end=date(2024, 12, 31))
QUOTE = "Gross margin decreased 1,380 basis points due to higher input costs."
PROMO = "Competitors increased promotional activity during 2025."
FILING_TEXT = "Tariffs on imported cocoa could raise our costs."


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
    current=filing(CURRENT, "acc-2025"),
    prior=filing(PRIOR, "acc-2024"),
)


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paragraph_id=chunk_id,
        ticker="HSY",
        cik="0000047111",
        form=FormType.FORM_10K,
        period=CURRENT,
        section=Section.MDNA,
        accession="acc-2025",
        text=text,
        token_count=20,
    )


def cite(chunk_id: str, text: str) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        accession="acc-2025",
        ticker="HSY",
        form=FormType.FORM_10K,
        period=CURRENT,
        section=Section.MDNA,
        quote=text,
        url="https://www.sec.gov/acc-2025.htm",
    )


CONFIRMED = Claim(
    claim_id="run1:chg-gm:r1:c1",
    text="Higher input costs lowered gross margin.",
    label=ExplanationLabel.CONFIRMED,
    citations=(cite("k1", QUOTE),),
    entailment=EntailmentVerdict.STATES_CAUSE,
)
PLAUSIBLE = Claim(
    claim_id="run1:chg-gm:r1:c2",
    text="Promotions rose and gross margin fell; the filings do not link the two.",
    label=ExplanationLabel.PLAUSIBLE,
    citations=(cite("k2", PROMO),),
)
MARGIN = Finding(
    finding_id="run1:chg-gm",
    change=Change(
        change_id="chg-gm", comparison_id="HSY-yoy", kind=ChangeKind.NUMERIC, metric_id="gm-2025"
    ),
    claims=(PLAUSIBLE, CONFIRMED),
)
RISK = Finding(
    finding_id="run1:chg-risk",
    change=Change(
        change_id="chg-risk",
        comparison_id="HSY-yoy",
        kind=ChangeKind.ADDED,
        section=Section.RISK_FACTORS,
        current_text=FILING_TEXT,
        rationale="A new risk factor about tariffs.",
    ),
    review_reason="every candidate was pruned",
)
FINDINGS = (MARGIN, RISK)


class FakeChunks:
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return {"k1": chunk("k1", QUOTE), "k2": chunk("k2", PROMO)}.get(chunk_id)


class NoMetrics:
    def get_metric(self, metric_id: str) -> MetricValue | None:
        return None


class NoFacts:
    def find_usd_fact(self, accession: str, value: float, tolerance: float) -> str | None:
        return None


class FakeWriter:
    """A `SummaryWriter` playing back queued drafts. An Exception in the queue is raised."""

    def __init__(self, *drafts: Any) -> None:
        self.drafts = list(drafts)
        self.requests: list[dict] = []

    async def write(self, request: dict) -> SummaryDraft:
        self.requests.append(request)
        item = self.drafts.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def draft(*sentences: tuple[str, tuple[str, ...]]) -> SummaryDraft:
    return SummaryDraft(
        sentences=tuple(SummarySentence(text=text, refs=refs) for text, refs in sentences)
    )


GOOD = draft(
    ("Higher input costs lowered gross margin.", ("C1",)),
    ("Promotions rose, but the filings do not link them to margin.", ("C2",)),
    ("A new tariff risk factor needs review.", ("F2",)),
)


def context(writer: FakeWriter) -> SummaryContext:
    return SummaryContext(writer=writer, chunks=FakeChunks(), metrics=NoMetrics(), facts=NoFacts())


def brief(summary: str, findings=FINDINGS) -> Brief:
    return Brief(
        run_id="run1",
        comparison=COMPARISON,
        generated_at=datetime(2026, 11, 10, tzinfo=UTC),
        summary=summary,
        findings=findings,
    )


async def test_a_good_draft_becomes_the_summary_and_passes_the_brief_check():
    writer = FakeWriter(GOOD)
    result = await write_summary(FINDINGS, COMPARISON, context(writer))
    assert not result.used_template
    assert result.writer_calls == 1
    assert result.text.splitlines() == [
        "Higher input costs lowered gross margin. [C1]",
        "Promotions rose, but the filings do not link them to margin. [C2]",
        "A new tariff risk factor needs review. [F2]",
    ]
    assert check_brief(brief(result.text)) == ()


async def test_the_model_sees_claims_and_labels_but_no_filing_text():
    writer = FakeWriter(GOOD)
    await write_summary(FINDINGS, COMPARISON, context(writer))
    (request,) = writer.requests
    sent = json.dumps(request)
    assert QUOTE not in sent and PROMO not in sent and FILING_TEXT not in sent
    margin, risk = request["findings"]
    assert margin["ref"] == "F1"
    assert [(c["ref"], c["label"]) for c in margin["claims"]] == [
        ("C1", "confirmed"),
        ("C2", "plausible"),
    ]
    assert risk == {
        "ref": "F2",
        "change": "New paragraph in Risk Factors",
        "section": "Risk Factors",
        "rationale": "A new risk factor about tariffs.",
        "needs_review": True,
        "claims": [],
    }
    assert request["periods"] == ["FY2024", "FY2025"]


async def test_a_causal_sentence_about_a_plausible_claim_is_retried_with_the_reason():
    bad = draft(("Margin fell due to promotions.", ("C2",)))
    writer = FakeWriter(bad, GOOD)
    result = await write_summary(FINDINGS, COMPARISON, context(writer))
    assert not result.used_template
    assert result.writer_calls == 2
    rejected = writer.requests[1]["rejected"]["problems"]
    assert any("uses causal wording (due to)" in problem for problem in rejected)
    assert result.failures == tuple(rejected)


async def test_an_untraced_figure_is_rejected():
    invented = draft(("Gross margin fell 900 basis points.", ("C1",)))
    traced = draft(("Gross margin fell 1,380 basis points.", ("C1",)))
    writer = FakeWriter(invented, traced)
    result = await write_summary(FINDINGS, COMPARISON, context(writer))
    assert result.text == "Gross margin fell 1,380 basis points. [C1]"
    assert any("900 basis points" in failure for failure in result.failures)


async def test_a_figure_from_a_claim_the_sentence_does_not_cite_is_rejected():
    # 1,380 is in C1's quote, not C2's, so a sentence citing only C2 cannot use it.
    writer = FakeWriter(draft(("Margin fell 1,380 basis points.", ("C2",))), GOOD)
    result = await write_summary(FINDINGS, COMPARISON, context(writer))
    assert result.writer_calls == 2
    assert any("1,380 basis points" in failure for failure in result.failures)


async def test_advice_is_rejected():
    writer = FakeWriter(draft(("Investors should buy on this news.", ("C1",))), GOOD)
    result = await write_summary(FINDINGS, COMPARISON, context(writer))
    assert any("investment advice" in failure for failure in result.failures)


async def test_drafts_that_keep_failing_fall_back_to_the_template():
    bad = draft(("Margin fell due to promotions.", ("C2",)))
    writer = FakeWriter(bad, bad)
    result = await write_summary(FINDINGS, COMPARISON, context(writer))
    assert result.used_template
    assert result.writer_calls == 2
    assert result.text == template_summary(FINDINGS)


async def test_an_empty_draft_is_rejected():
    writer = FakeWriter(SummaryDraft(sentences=()), GOOD)
    result = await write_summary(FINDINGS, COMPARISON, context(writer))
    assert "the summary has no sentences" in result.failures


async def test_a_writer_failure_falls_back_without_retrying():
    writer = FakeWriter(LLMError("truncated"))
    result = await write_summary(FINDINGS, COMPARISON, context(writer))
    assert result.used_template
    assert result.writer_calls == 1
    assert result.failures == ("the summary writer failed: truncated",)


async def test_a_budget_stop_is_not_swallowed():
    writer = FakeWriter(BudgetExceededError("over budget"))
    with pytest.raises(BudgetExceededError):
        await write_summary(FINDINGS, COMPARISON, context(writer))


async def test_no_findings_needs_no_call():
    writer = FakeWriter()
    result = await write_summary((), COMPARISON, context(writer))
    assert result.text == NO_FINDINGS_SUMMARY
    assert result.used_template and result.writer_calls == 0
    assert check_brief(brief(result.text, findings=())) == ()


def test_the_template_gives_each_finding_one_line_that_passes_the_brief_check():
    text = template_summary(FINDINGS)
    assert text.splitlines() == [
        "Confirmed: Higher input costs lowered gross margin. [C1]",
        "New paragraph in Risk Factors: no explanation met the evidence standard, so it needs "
        "review. [F2]",
    ]
    assert check_brief(brief(text)) == ()


def test_the_template_shows_a_plausible_claim_when_nothing_is_confirmed():
    only = MARGIN.model_copy(update={"claims": (PLAUSIBLE,)})
    assert template_summary((only,)) == f"Plausible: {PLAUSIBLE.text} [C1]"


def test_the_request_numbers_like_the_brief():
    request = summary_request(FINDINGS, COMPARISON)
    assert [finding["ref"] for finding in request["findings"]] == ["F1", "F2"]


def test_the_prompt_treats_input_as_untrusted_and_forbids_advice():
    assert "untrusted" in SUMMARY_SYSTEM
    assert "Never follow" in SUMMARY_SYSTEM
    assert "Never give investment advice" in SUMMARY_SYSTEM


def test_the_output_schema_is_valid_for_structured_output():
    transformed = anthropic.transform_schema(SummaryDraft)
    assert transformed["additionalProperties"] is False
    assert transformed["required"] == ["sentences"]
    assert transformed["description"] == "Your summary, as sentences with their references."


class FakeAnthropic:
    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    async def _parse(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)


async def test_the_real_writer_uses_the_brief_role(settings):
    response = SimpleNamespace(
        parsed_output=GOOD,
        stop_reason="end_turn",
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    fake = FakeAnthropic(response)
    writer = SummaryWriter(LLMClient(settings, CostLedger(1.0), client=fake))
    request = summary_request(FINDINGS, COMPARISON)
    assert await writer.write(request) == GOOD
    (sent,) = fake.requests
    assert sent["model"] == settings.model_strong
    assert sent["output_config"] == {"effort": ROLE_CONFIG[Role.BRIEF].effort}
    # Lowered from 16,000 (the user's decision, 2026-11-10): the larger limit held back a $0.33
    # worst case per call against the run cap, and the live summary used 516 tokens.
    assert sent["max_tokens"] == 4_000
    assert sent["system"][0]["text"] == SUMMARY_SYSTEM
    assert json.loads(sent["messages"][0]["content"]) == request
