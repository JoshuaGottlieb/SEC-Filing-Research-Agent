"""How filings name the catalog's financial lines, so code can tell which lines a sentence names.

Used by labeling (a restatement must still name the change it explains; a text change's claim must
name only lines the change names) and by `back_reference` (which line a filing says the change
moved with). Kept apart from both so neither imports the other for it.
"""

from __future__ import annotations

from sec_agent.schemas import MetricName

# How filings name each metric. Chosen by hand from Hershey's MD&A and statements ("cost of sales",
# "net sales", "operating profit", "selling, marketing and administrative"). The first entry is
# shown to the restater as the change being explained.
_SALES = ("net sales", "revenue")
_COSTS = ("cost of sales", "cost of goods sold", "cost of revenue")
_SGA = ("selling, marketing", "selling, general", "sg&a", "sm&a")
METRIC_WORDS: dict[MetricName, tuple[str, ...]] = {
    MetricName.GROSS_MARGIN: ("gross margin",),
    MetricName.OPERATING_MARGIN: ("operating margin",),
    MetricName.NET_MARGIN: ("net margin",),
    MetricName.REVENUE_GROWTH: _SALES,
    MetricName.REVENUE: _SALES,
    MetricName.COGS_RATIO: _COSTS,
    MetricName.COST_OF_GOODS_SOLD: _COSTS,
    MetricName.GROSS_PROFIT: ("gross profit",),
    MetricName.SGA_RATIO: _SGA,
    MetricName.SGA: _SGA,
    MetricName.OPERATING_INCOME: ("operating income", "operating profit"),
    MetricName.NET_INCOME: ("net income", "net earnings"),
}


def lines_named(text: str) -> frozenset[MetricName]:
    """The financial lines `text` names, as filings word them (`METRIC_WORDS`)."""
    lowered = text.casefold()
    return frozenset(
        name for name, words in METRIC_WORDS.items() if any(word in lowered for word in words)
    )
