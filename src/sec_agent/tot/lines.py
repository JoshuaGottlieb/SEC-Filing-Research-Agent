"""How filings name the catalog's financial lines, so code can tell which lines a sentence names.

Used by labeling (a restatement must still name the change it explains; a text change's claim must
name only lines the change names) and by `back_reference` (which line a filing says the change
moved with). Kept apart from both so neither imports the other for it.
"""

from __future__ import annotations

import re

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


def _patterns() -> list[tuple[re.Pattern[str], frozenset[MetricName]]]:
    names_by_word: dict[str, set[MetricName]] = {}
    for name, words in METRIC_WORDS.items():
        for word in words:
            names_by_word.setdefault(word, set()).add(name)
    # Whole words only (a plural allowed), and not after a hyphen: "non-operating income" is not
    # operating income. Longest first, so "cost of revenue" is read, and blanked out, before
    # "revenue" is looked for inside it.
    return [
        (re.compile(rf"(?<![\w-]){re.escape(word)}s?(?!\w)"), frozenset(names))
        for word, names in sorted(names_by_word.items(), key=lambda item: -len(item[0]))
    ]


_PATTERNS = _patterns()


def lines_named(text: str) -> frozenset[MetricName]:
    """The financial lines `text` names, as filings word them (`METRIC_WORDS`)."""
    lowered = text.casefold()
    found: set[MetricName] = set()
    for pattern, names in _PATTERNS:
        lowered, count = pattern.subn(lambda match: " " * len(match.group()), lowered)
        if count:
            found |= names
    return frozenset(found)
