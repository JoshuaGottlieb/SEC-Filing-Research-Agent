"""Records shared between agents, the MCP tree store, and the report.

Validators enforce rules that must hold no matter which agent produced the
record, so a bad state fails loudly instead of reaching the brief. The main one
is that a `confirmed` claim needs a citation and a `states_cause` verdict.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Printed on every brief. The system compares filings; it does not advise on investments.
DISCLAIMER = (
    "This brief is a first-pass research aid generated from SEC filings. It is not investment "
    "advice and makes no recommendation or price prediction. Verify findings against the cited "
    "filings before relying on them."
)


class FormType(StrEnum):
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"


class Section(StrEnum):
    """Canonical sections. 10-K and 10-Q item numbers map onto these."""

    BUSINESS = "business"
    RISK_FACTORS = "risk_factors"
    MDNA = "mdna"
    MARKET_RISK = "market_risk"
    FINANCIAL_NOTES = "financial_notes"


class ComparisonType(StrEnum):
    # Year over year on 10-Ks, year over year on the same quarter's 10-Qs, and sequential
    # quarters (the latest quarter against the one just before it).
    YOY_10K = "yoy-10k"
    YOY_10Q = "yoy-10q"
    SEQ_10Q = "seq-10q"


class ChangeKind(StrEnum):
    # Text changes are added, removed or modified paragraphs; NUMERIC is a metric moving.
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    NUMERIC = "numeric"


class MaterialityCriterion(StrEnum):
    # The explicit reasons a change can be called material. Each decision records one.
    NEW_OR_REMOVED_RISK = "new_or_removed_risk"
    CHANGED_QUANTITY = "changed_quantity"
    METRIC_LINKED_SHIFT = "metric_linked_shift"


class Duration(StrEnum):
    # How long a figure's period is: a point in time (balance sheet), a quarter, year to date
    # (six or nine months), or a full year.
    INSTANT = "instant"
    THREE_MONTHS = "3m"
    YEAR_TO_DATE = "ytd"
    TWELVE_MONTHS = "12m"


class NumericCheck(StrEnum):
    """Whether metrics agree with a claim. NOT_APPLICABLE drops the criterion from scoring."""

    AGREE = "agree"
    CONTRADICT = "contradict"
    NOT_APPLICABLE = "not_applicable"


class NodeStatus(StrEnum):
    # A Tree-of-Thought node is kept in the beam or pruned; pruned nodes stay on record.
    PENDING = "pending"
    KEPT = "kept"
    PRUNED = "pruned"


class ExplanationLabel(StrEnum):
    # Confirmed: a cited passage states the cause. Plausible: evidence is consistent with it.
    # Unresolved: nothing supports it well enough, so it goes to human review.
    CONFIRMED = "confirmed"
    PLAUSIBLE = "plausible"
    UNRESOLVED = "unresolved"


class EntailmentVerdict(StrEnum):
    # A separate judge's reading of a quote: it states the cause, only implies it, or doesn't
    # bear on it. Only STATES_CAUSE can support a confirmed claim.
    STATES_CAUSE = "states_cause"
    IMPLIES = "implies"
    UNRELATED = "unrelated"


class Record(BaseModel):
    """Base for all records: immutable, and unknown fields are errors, not ignored."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class FiscalPeriod(Record):
    fiscal_year: int
    # FY for a 10-K. There is no Q4 10-Q.
    fiscal_period: Literal["FY", "Q1", "Q2", "Q3"]
    period_end: date

    @property
    def label(self) -> str:
        if self.fiscal_period == "FY":
            return f"FY{self.fiscal_year}"
        return f"{self.fiscal_period} {self.fiscal_year}"


class Filing(Record):
    """One SEC filing and where it sits in the company's fiscal calendar."""

    ticker: str
    company: str
    cik: str
    form: FormType
    period: FiscalPeriod
    accession: str
    filed_date: date
    url: str


class Comparison(Record):
    """Two filings of the same kind to compare: `current` against the earlier `prior`."""

    comparison_id: str
    ticker: str
    comparison_type: ComparisonType
    current: Filing
    prior: Filing

    @model_validator(mode="after")
    def _filings_are_comparable(self) -> Comparison:
        # These checks stop a mismatched pair (wrong company, form or order) reaching analysis.
        if self.current.ticker != self.ticker or self.prior.ticker != self.ticker:
            raise ValueError("both filings must belong to the comparison's company")
        expected = (
            FormType.FORM_10K
            if self.comparison_type is ComparisonType.YOY_10K
            else FormType.FORM_10Q
        )
        if self.current.form is not expected or self.prior.form is not expected:
            raise ValueError(f"{self.comparison_type.value} compares {expected.value} filings")
        if self.current.period.period_end <= self.prior.period.period_end:
            raise ValueError("the current filing must cover a later period than the prior filing")
        return self


class Chunk(Record):
    """A passage of filing text, with the metadata retrieval filters on."""

    chunk_id: str
    # The parsed paragraph this chunk came from, so a citation can be traced back to it.
    paragraph_id: str
    ticker: str
    cik: str
    form: FormType
    period: FiscalPeriod
    section: Section
    accession: str
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)


class MetricFact(Record):
    """One reported XBRL value. `segment` is None for company-wide facts."""

    fact_id: str
    ticker: str
    concept: str
    period: FiscalPeriod
    start_date: date | None = None
    end_date: date
    duration: Duration
    value: float
    unit: str
    accession: str
    segment: str | None = None


class MetricValue(Record):
    """A computed metric, traceable to the facts it was computed from."""

    metric_id: str
    ticker: str
    name: str
    period: FiscalPeriod
    duration: Duration
    value: float
    unit: str
    input_fact_ids: tuple[str, ...] = Field(min_length=1)


class Change(Record):
    """Something that differs between the two filings in a comparison."""

    change_id: str
    comparison_id: str
    kind: ChangeKind
    section: Section | None = None
    current_text: str | None = None
    prior_text: str | None = None
    current_chunk_ids: tuple[str, ...] = ()
    prior_chunk_ids: tuple[str, ...] = ()
    metric_id: str | None = None
    # A sequential move that a year-earlier move of similar size would explain.
    possibly_seasonal: bool = False
    criterion: MaterialityCriterion | None = None
    materiality_rank: int | None = Field(default=None, ge=1)
    rationale: str | None = None

    @model_validator(mode="after")
    def _evidence_matches_kind(self) -> Change:
        # A numeric change points to a metric; a text change needs the text that changed.
        if self.kind is ChangeKind.NUMERIC:
            if self.metric_id is None:
                raise ValueError("a numeric change needs a metric_id")
            return self
        if self.section is None:
            raise ValueError("a text change needs a section")
        if self.kind in (ChangeKind.ADDED, ChangeKind.MODIFIED) and not self.current_text:
            raise ValueError(f"a {self.kind.value} change needs current_text")
        if self.kind in (ChangeKind.REMOVED, ChangeKind.MODIFIED) and not self.prior_text:
            raise ValueError(f"a {self.kind.value} change needs prior_text")
        return self


class TreeNode(Record):
    """One node of a ToT tree. Pruned nodes are kept for the audit trail.

    Depth 0 holds the flagged change, depth 1 a candidate explanation, and depth
    2 a proposed source of evidence.
    """

    node_id: str
    run_id: str
    change_id: str
    parent_id: str | None = None
    depth: int = Field(ge=0, le=2)
    thought: str
    expected_evidence: str | None = None
    query_plan: str | None = None
    retrieved_chunk_ids: tuple[str, ...] = ()
    metric_ids: tuple[str, ...] = ()
    numeric_check: NumericCheck | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    status: NodeStatus = NodeStatus.PENDING
    prune_reason: str | None = None

    @model_validator(mode="after")
    def _structure_is_valid(self) -> TreeNode:
        if (self.depth == 0) != (self.parent_id is None):
            raise ValueError("only the depth 0 root has no parent")
        if self.status is NodeStatus.PRUNED and not self.prune_reason:
            raise ValueError("a pruned node must record why it was pruned")
        return self


class Citation(Record):
    """A verbatim quote from a filing, with everything needed to check and link it."""

    chunk_id: str
    accession: str
    ticker: str
    form: FormType
    period: FiscalPeriod
    section: Section
    quote: str = Field(min_length=1)
    url: str


class Claim(Record):
    """One statement in the brief, with its label and the evidence behind it."""

    claim_id: str
    text: str = Field(min_length=1)
    label: ExplanationLabel
    citations: tuple[Citation, ...] = ()
    metric_ids: tuple[str, ...] = ()
    entailment: EntailmentVerdict | None = None

    @model_validator(mode="after")
    def _label_has_required_evidence(self) -> Claim:
        # The rule behind the brief's wording: causal language needs a confirmed claim.
        if self.label is ExplanationLabel.CONFIRMED and (
            not self.citations or self.entailment is not EntailmentVerdict.STATES_CAUSE
        ):
            raise ValueError("a confirmed claim needs a citation and a states_cause verdict")
        if self.label is ExplanationLabel.PLAUSIBLE and not (self.citations or self.metric_ids):
            raise ValueError("a plausible claim needs a citation or a computed metric")
        return self


class Finding(Record):
    """A material change with the claims that explain it."""

    finding_id: str
    change: Change
    claims: tuple[Claim, ...] = ()
    tree_root_id: str | None = None
    review_reason: str | None = None

    @property
    def needs_review(self) -> bool:
        return self.review_reason is not None or any(
            claim.label is ExplanationLabel.UNRESOLVED for claim in self.claims
        )


class Brief(Record):
    """The final report for one comparison."""

    run_id: str
    comparison: Comparison
    generated_at: datetime
    summary: str
    findings: tuple[Finding, ...] = ()
    cost_usd: float = Field(default=0.0, ge=0)
    disclaimer: str = DISCLAIMER

    @property
    def review_items(self) -> tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if finding.needs_review)
