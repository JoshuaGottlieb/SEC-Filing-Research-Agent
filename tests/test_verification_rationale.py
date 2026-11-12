"""The rationale figure check: a materiality rationale's figures must trace to the change."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from sec_agent.agents.verification import untraced_rationale_figures
from sec_agent.schemas import (
    Change,
    ChangeKind,
    Chunk,
    Duration,
    FiscalPeriod,
    FormType,
    MetricValue,
    Section,
)

CURRENT = FiscalPeriod(fiscal_year=2025, fiscal_period="FY", period_end=date(2025, 12, 31))
PRIOR = FiscalPeriod(fiscal_year=2024, fiscal_period="FY", period_end=date(2024, 12, 31))

LATER = "Cash paid for income taxes was $207.0 million in 2025, and 12 plants operated."
EARLIER = "Cash paid for income taxes was $65.5 million in 2024."


def make_chunk(chunk_id: str, text: str, period: FiscalPeriod = CURRENT) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paragraph_id="p1",
        ticker="HSY",
        cik="0000047111",
        form=FormType.FORM_10K,
        period=period,
        section=Section.MDNA,
        accession="0000047111-2026-000001",
        text=text,
        token_count=40,
    )


class FakeChunks:
    def __init__(self, *chunks: Chunk) -> None:
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)


class NoFacts:
    def find_usd_fact(self, accession: str, value: float, tolerance: float) -> str | None:
        return None


def text_change(later: str | None = LATER, earlier: str | None = EARLIER, **fields: Any) -> Change:
    return Change(
        **{
            "change_id": "HSY:yoy-10k:modified:p1",
            "comparison_id": "HSY:yoy-10k",
            "kind": ChangeKind.MODIFIED,
            "section": Section.MDNA,
            "current_text": later,
            "prior_text": earlier,
            "current_chunk_ids": ("later-1",) if later else (),
            "prior_chunk_ids": ("earlier-1",) if earlier else (),
            **fields,
        }
    )


# The chunks hold more than the change's own paragraph, as real chunks do.
CHUNKS = FakeChunks(
    make_chunk("later-1", f"{LATER}\n\nRestructuring costs were $9 million."),
    make_chunk("earlier-1", EARLIER, PRIOR),
)


def untraced(rationale: str, change: Change | None = None, *, chunks=CHUNKS, metrics=()):
    return untraced_rationale_figures(
        rationale,
        change or text_change(),
        metrics=metrics,
        chunks=chunks,
        facts=NoFacts(),
    )


def test_a_rationale_with_no_figures_passes():
    assert untraced("Cash taxes rose sharply, which matters for cash flow.") == ()


@pytest.mark.parametrize(
    "rationale",
    [
        "Cash taxes rose to $207.0 million from $65.5 million.",
        # Rounding is allowed, as for a claim.
        "Cash taxes rose to $207 million.",
        "Cash taxes rose to about $0.2 billion.",
        # An abbreviation is read at its scale.
        "Cash taxes rose to $207.0M from $65.5M.",
    ],
)
def test_figures_stated_in_the_changes_text_pass(rationale):
    assert untraced(rationale) == ()


@pytest.mark.parametrize(
    ("rationale", "figures"),
    [
        # Invented.
        ("Cash taxes rose to $270.0 million.", ("$270.0 million",)),
        # Derived: neither text states the difference or the percent change.
        ("Cash taxes rose $141.5 million.", ("$141.5 million",)),
        ("Cash taxes rose 216%.", ("216%",)),
        # More precise than the text.
        ("Cash taxes rose to $207.04 million.", ("$207.04 million",)),
        # An amount with unknown letters is never read as a shorter one.
        ("Cash taxes rose to $207x.", ("$207x",)),
    ],
)
def test_a_figure_nothing_backs_is_returned(rationale, figures):
    assert untraced(rationale) == figures


def test_only_the_changes_own_text_counts_not_the_rest_of_its_chunk():
    # The chunk also says "$9 million", but the change is about cash taxes.
    assert untraced("Restructuring costs were $9 million.") == ("$9 million",)


def test_each_figure_is_judged_on_its_own_and_reported_once():
    rationale = "Cash taxes rose to $207.0 million, up $141.5 million; $141.5 million is large."
    assert untraced(rationale) == ("$141.5 million",)


def test_an_added_paragraph_has_only_its_later_text():
    added = text_change(earlier=None, kind=ChangeKind.ADDED)
    assert untraced("Cash taxes were $207.0 million.", added) == ()
    assert untraced("Cash taxes were $65.5 million.", added) == ("$65.5 million",)


def test_text_no_chunk_holds_is_read_as_prose():
    # 116 of 5,847 real passages are not in their chunk word for word (page breaks joined).
    assert untraced("Cash taxes rose to $207.0 million.", chunks=FakeChunks()) == ()


IN_MILLIONS = "(In millions)\nNet sales | $ | 11,692.6 | $ | 11,164.9"


def test_a_tables_stated_scale_counts_when_the_text_is_a_table():
    table = text_change(later="Net sales | $ | 11,692.6", earlier=None, kind=ChangeKind.ADDED)
    chunks = FakeChunks(make_chunk("later-1", IN_MILLIONS))
    assert untraced("Net sales were $11.7 billion.", table, chunks=chunks) == ()


def test_prose_is_not_read_at_a_scale_it_does_not_state():
    prose = text_change(
        later="Net sales were 11,692.6 this year.", earlier=None, kind=ChangeKind.ADDED
    )
    chunks = FakeChunks(make_chunk("later-1", "Net sales were 11,692.6 this year."))
    assert untraced("Net sales were $11.7 billion.", prose, chunks=chunks) == ("$11.7 billion",)


def metric(metric_id: str, value: float, period: FiscalPeriod, **fields: Any) -> MetricValue:
    return MetricValue(
        **{
            "metric_id": metric_id,
            "ticker": "HSY",
            "name": "gross_margin",
            "period": period,
            "duration": Duration.TWELVE_MONTHS,
            "value": value,
            "unit": "ratio",
            "input_fact_ids": ("f1",),
            **fields,
        }
    )


NUMERIC = Change(
    change_id="HSY:yoy-10k:numeric:gross_margin",
    comparison_id="HSY:yoy-10k",
    kind=ChangeKind.NUMERIC,
    metric_id="gm-now",
)
MARGIN_PAIR = (metric("gm-now", 0.4593, CURRENT), metric("gm-before", 0.40, PRIOR))
REVENUE_PAIR = (
    metric("rev-now", 11_050_000_000.0, CURRENT, name="revenue", unit="USD"),
    metric("rev-before", 10_000_000_000.0, PRIOR, name="revenue", unit="USD"),
)


@pytest.mark.parametrize(
    ("rationale", "pair"),
    [
        # As the scan describes the move (materiality.describe_move).
        ("Gross margin rose +593 bps.", MARGIN_PAIR),
        ("Gross margin rose 5.93 percentage points to 45.93% from 40.0%.", MARGIN_PAIR),
        ("Revenue grew +10.5%.", REVENUE_PAIR),
        ("Revenue grew $1.05 billion to $11.05 billion.", REVENUE_PAIR),
    ],
)
def test_a_numeric_rationale_traces_to_the_metric_pair(rationale, pair):
    assert untraced(rationale, NUMERIC, metrics=pair) == ()


@pytest.mark.parametrize(
    ("rationale", "figures"),
    [
        ("Gross margin rose 600 bps.", ("600 bps",)),
        # "About" does not widen the precision written: the move is 593 bps.
        ("Gross margin rose about 590 bps.", ("590 bps",)),
        # Points are not percent: a 5.93-point rise is a 14.8% relative rise.
        ("Gross margin rose 5.93%.", ("5.93%",)),
    ],
)
def test_a_numeric_rationale_the_pair_does_not_back_is_returned(rationale, figures):
    assert untraced(rationale, NUMERIC, metrics=MARGIN_PAIR) == figures


def test_without_the_pair_only_the_level_given_is_backed():
    level_only = MARGIN_PAIR[:1]
    assert untraced("Gross margin was 45.93%.", NUMERIC, metrics=level_only) == ()
    assert untraced("Gross margin rose 593 bps.", NUMERIC, metrics=level_only) == ("593 bps",)
