from __future__ import annotations

from datetime import date

import pytest

from sec_agent.edgar.xbrl import (
    XbrlContext,
    dimensions_json,
    duration_of,
    read_contexts,
    read_inline_fact,
)
from sec_agent.ingest.tables import parse_html_tree
from sec_agent.schemas import Duration


def nonfraction(text: str, **attributes: str):
    """An inline XBRL number element (tag name lowercased, as lxml's HTML parser sees it)."""
    pairs = " ".join(f'{key}="{value}"' for key, value in attributes.items())
    root = parse_html_tree(
        f"<html><body><ix:nonfraction {pairs}>{text}</ix:nonfraction></body></html>"
    )
    return next(root.iter("ix:nonfraction"))


def test_scale_and_sign_are_applied():
    # scale="6" is millions; sign="-" makes it negative: -1,234.5 million.
    fact = read_inline_fact(
        nonfraction("1,234.5", name="a", contextref="c", scale="6", sign="-", unitref="usd")
    )
    assert fact.value == pytest.approx(-1_234_500_000)
    assert fact.unit == "usd"


def test_dash_formats_are_zero():
    # A dash on the page means zero when the tag's format says so.
    assert (
        read_inline_fact(nonfraction("—", name="a", contextref="c", format="ixt:zerodash")).value
        == 0
    )


def test_text_that_is_not_a_number_is_skipped():
    assert read_inline_fact(nonfraction("none", name="a", contextref="c")) is None


def test_contexts_read_periods_and_dimensions():
    # Context "d" is a segment's six-month period; context "i" is a single date.
    root = parse_html_tree(
        '<html><body><xbrli:context id="d"><xbrli:entity><xbrli:segment>'
        '<xbrldi:explicitmember dimension="us-gaap:StatementBusinessSegmentsAxis">a:X'
        "</xbrldi:explicitmember></xbrli:segment></xbrli:entity><xbrli:period>"
        "<xbrli:startdate>2026-01-01</xbrli:startdate><xbrli:enddate>2026-06-30</xbrli:enddate>"
        '</xbrli:period></xbrli:context><xbrli:context id="i"><xbrli:period>'
        "<xbrli:instant>2026-06-30</xbrli:instant></xbrli:period></xbrli:context></body></html>"
    )
    contexts = read_contexts(root)
    assert contexts["d"].dimensions == {"us-gaap:StatementBusinessSegmentsAxis": "a:X"}
    assert contexts["i"].start_date is None
    assert duration_of(contexts["i"]) is Duration.INSTANT
    assert duration_of(contexts["d"]) is Duration.YEAR_TO_DATE
    assert dimensions_json(contexts["i"]) is None


@pytest.mark.parametrize(
    ("start", "end", "duration"),
    [
        (date(2026, 4, 1), date(2026, 6, 30), Duration.THREE_MONTHS),
        (date(2025, 1, 1), date(2025, 12, 31), Duration.TWELVE_MONTHS),
        (date(2024, 12, 29), date(2026, 1, 3), Duration.TWELVE_MONTHS),  # 53-week year (Stanley)
        (date(2026, 1, 1), date(2026, 9, 30), Duration.YEAR_TO_DATE),
    ],
)
def test_durations(start, end, duration):
    context = XbrlContext(context_id="c", start_date=start, end_date=end)
    assert duration_of(context) is duration
