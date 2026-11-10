"""The names a reader sees for the records' enums, each from one mapping that covers every member.

A test loops over each enum, so a member added later fails a test here instead of reaching the
brief as a raw value such as "financial_notes" or breaking the template.
"""

from __future__ import annotations

from sec_agent.schemas import (
    ChangeKind,
    ComparisonType,
    ExplanationLabel,
    MetricName,
    Section,
)

SECTION_NAMES: dict[Section, str] = {
    Section.BUSINESS: "Business",
    Section.RISK_FACTORS: "Risk Factors",
    Section.MDNA: "Management's Discussion and Analysis",
    Section.MARKET_RISK: "Quantitative and Qualitative Disclosures About Market Risk",
    Section.FINANCIAL_NOTES: "Notes to the Financial Statements",
    # Tables filed outside the five sections, such as share repurchases or stock performance.
    Section.OTHER: "Other tables (outside the main sections)",
}

LABEL_NAMES: dict[ExplanationLabel, str] = {
    ExplanationLabel.CONFIRMED: "Confirmed",
    ExplanationLabel.PLAUSIBLE: "Plausible",
    ExplanationLabel.UNRESOLVED: "Unresolved",
}

# How a text change reads in a finding's heading; a numeric change names its metric instead.
CHANGE_KIND_NAMES: dict[ChangeKind, str] = {
    ChangeKind.ADDED: "New paragraph",
    ChangeKind.REMOVED: "Removed paragraph",
    ChangeKind.MODIFIED: "Changed paragraph",
    ChangeKind.NUMERIC: "Metric change",
}

METRIC_NAMES: dict[MetricName, str] = {
    MetricName.GROSS_MARGIN: "Gross margin",
    MetricName.OPERATING_MARGIN: "Operating margin",
    MetricName.NET_MARGIN: "Net margin",
    MetricName.REVENUE_GROWTH: "Revenue growth",
    MetricName.COGS_RATIO: "Cost of sales to revenue",
    MetricName.SGA_RATIO: "SG&A to revenue",
    # Dollar metrics (lane 1's PR #32): signed whole dollars.
    MetricName.REVENUE: "Revenue",
    MetricName.COST_OF_GOODS_SOLD: "Cost of sales",
    MetricName.GROSS_PROFIT: "Gross profit",
    MetricName.SGA: "SG&A expense",
    MetricName.OPERATING_INCOME: "Operating income",
    MetricName.NET_INCOME: "Net income",
}

COMPARISON_NAMES: dict[ComparisonType, str] = {
    ComparisonType.YOY_10K: "year over year (10-K)",
    ComparisonType.YOY_10Q: "year over year (10-Q)",
    ComparisonType.SEQ_10Q: "sequential quarters (10-Q)",
}


def metric_name(name: str) -> str:
    """The display name of a `MetricValue.name`, which is a `MetricName` value stored as text."""
    try:
        return METRIC_NAMES[MetricName(name)]
    except ValueError:
        # A metric the catalog added before this mapping: show its raw name rather than fail.
        return name
