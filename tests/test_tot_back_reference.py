"""Finding a back-reference: the filing explains a change "by the same factors" as another line's.

The texts are Hershey's FY2025 10-K MD&A, where the operating margin sentence points back to the
gross margin paragraph (user's decision, 2026-11-15, option A).
"""

from __future__ import annotations

from datetime import date

import pytest

from sec_agent.finance.metrics import metric_id_for
from sec_agent.schemas import (
    Change,
    ChangeKind,
    Chunk,
    Comparison,
    ComparisonType,
    Duration,
    Filing,
    FiscalPeriod,
    FormType,
    MetricName,
    MetricValue,
    Section,
)
from sec_agent.tot.back_reference import (
    OMITTED,
    find_back_reference,
    find_own_explanation,
    shown_percent,
)

CURRENT = FiscalPeriod(fiscal_year=2025, fiscal_period="FY", period_end=date(2025, 12, 31))
PRIOR = FiscalPeriod(fiscal_year=2024, fiscal_period="FY", period_end=date(2024, 12, 31))
NOW_ACC = "acc-2025"


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
    prior=filing(PRIOR, "acc-2024"),
)

GROSS = (
    "Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380 "
    "basis points. The decrease was driven by higher commodity and tariff costs and lower volume."
)
OPERATING = (
    "Operating profit was $1,441.5 million in 2025 compared to $2,898.2 million in 2024. "
    "Operating profit margin decreased to 12.3% in 2025 from 25.9% in 2024 by the same factors "
    "noted above in gross margin."
)
SENTENCE = (
    "Operating profit margin decreased to 12.3% in 2025 from 25.9% in 2024 by the same factors "
    "noted above in gross margin."
)
# The year before's sentence, in the same 10-K: its values are another comparison's.
EARLIER = (
    "Operating profit margin increased to 25.9% in 2024 from 22.9% in 2023 by the same factors "
    "noted above in gross margin."
)

VALUES = {
    MetricName.OPERATING_MARGIN: (0.1233, 0.2591),
    MetricName.GROSS_MARGIN: (0.335, 0.473),
}


def chunk(n: int, *paragraphs: str, section: Section = Section.MDNA) -> Chunk:
    return Chunk(
        chunk_id=f"{NOW_ACC}:{section.value}:{n:04d}:0",
        paragraph_id=f"p{n}",
        ticker="HSY",
        cik="0000047111",
        form=FormType.FORM_10K,
        period=CURRENT,
        section=section,
        accession=NOW_ACC,
        text="\n\n".join(paragraphs),
        token_count=50,
    )


class FakeFiling:
    def __init__(self, *chunks: Chunk) -> None:
        self._chunks = list(chunks)

    def chunks(self, accession: str | None = None) -> list[Chunk]:
        return [item for item in self._chunks if accession in (None, item.accession)]


def metric(name: MetricName, filing_: Filing, value: float, unit: str = "ratio") -> MetricValue:
    return MetricValue(
        metric_id=metric_id_for(filing_, name),
        ticker="HSY",
        name=name.value,
        period=filing_.period,
        duration=Duration.TWELVE_MONTHS,
        value=value,
        unit=unit,
        input_fact_ids=("f1",),
    )


class FakeMetrics:
    def __init__(self, values=VALUES, unit: str = "ratio") -> None:
        self._values = {
            item.metric_id: item
            for name, (now, then) in values.items()
            for item in (
                metric(name, COMPARISON.current, now, unit),
                metric(name, COMPARISON.prior, then, unit),
            )
        }

    def get_metric(self, metric_id: str) -> MetricValue | None:
        return self._values.get(metric_id)


def change(name: MetricName = MetricName.OPERATING_MARGIN) -> Change:
    return Change(
        change_id=f"chg-{name.value}",
        comparison_id="HSY-yoy",
        kind=ChangeKind.NUMERIC,
        metric_id=metric_id_for(COMPARISON.current, name),
    )


def find(*chunks: Chunk, name=MetricName.OPERATING_MARGIN, metrics=None):
    return find_back_reference(
        change(name), COMPARISON, name, metrics or FakeMetrics(), FakeFiling(*chunks)
    )


def test_hersheys_operating_margin_sentence_is_paired_with_the_gross_margin_paragraph():
    found = find(chunk(1, GROSS), chunk(2, "Some other paragraph.", OPERATING))
    assert found is not None
    assert found.line is MetricName.GROSS_MARGIN
    assert found.sentence.quote == SENTENCE
    assert found.referenced.quote == GROSS
    assert (found.referenced.chunk_id, found.sentence.chunk_id) == (
        f"{NOW_ACC}:mdna:0001:0",
        f"{NOW_ACC}:mdna:0002:0",
    )
    assert found.citations == (found.referenced, found.sentence)
    assert found.judged_text == f"{GROSS}\n\n{OMITTED}\n\n{SENTENCE}"
    # Verbatim from their chunks, from the current filing, linked to it.
    assert found.sentence.url == COMPARISON.current.url
    assert found.sentence.period == CURRENT


def test_a_paragraph_in_the_same_chunk_above_the_sentence_counts():
    found = find(chunk(1, GROSS, OPERATING))
    assert found is not None and found.referenced.quote == GROSS


def test_the_nearest_explanation_above_is_the_one_paired():
    overview = "Gross margin fell to 33.5% from 47.3%, driven by tariffs."
    found = find(chunk(1, overview), chunk(2, GROSS), chunk(3, OPERATING))
    assert found is not None and found.referenced.quote == GROSS


def test_an_explanation_only_below_the_sentence_is_not_noted_above():
    assert find(chunk(1, OPERATING), chunk(2, GROSS)) is None


def test_the_sentence_must_give_the_comparisons_own_two_values():
    # A sequential 10-Q comparison: the filing's sentence compares with a year before instead.
    assert find(chunk(1, GROSS), chunk(2, EARLIER)) is None
    other = {**VALUES, MetricName.OPERATING_MARGIN: (0.1233, 0.15)}
    assert find(chunk(1, GROSS), chunk(2, OPERATING), metrics=FakeMetrics(other)) is None


def test_the_paragraph_must_give_the_referenced_lines_two_values():
    other = {**VALUES, MetricName.GROSS_MARGIN: (0.335, 0.40)}
    assert find(chunk(1, GROSS), chunk(2, OPERATING), metrics=FakeMetrics(other)) is None


def test_a_paragraph_with_no_causal_wording_is_not_an_explanation():
    plain = "Gross margin was 33.5% in 2025 compared with 47.3% in 2024."
    assert find(chunk(1, plain), chunk(2, OPERATING)) is None


def test_a_table_row_is_not_an_explanation():
    row = "Gross margin | 33.5% | 47.3% | driven by tariffs"
    assert find(chunk(1, row), chunk(2, OPERATING)) is None


def test_only_the_mdna_is_read():
    assert find(chunk(1, GROSS, section=Section.RISK_FACTORS), chunk(2, OPERATING)) is None
    assert find(chunk(1, GROSS), chunk(2, OPERATING, section=Section.FINANCIAL_NOTES)) is None


def test_a_cause_stated_as_noted_above_is_not_a_back_reference():
    stated = (
        "Operating profit margin decreased to 12.3% in 2025 from 25.9% in 2024, due to lower "
        "gross profit, as noted above."
    )
    assert find(chunk(1, GROSS), chunk(2, stated)) is None


def test_the_sentence_must_be_about_the_change():
    # Gross margin's change: the operating margin sentence names neither its line nor its values.
    assert find(chunk(1, GROSS), chunk(2, OPERATING), name=MetricName.GROSS_MARGIN) is None


def test_a_line_moving_with_the_change_is_not_another_lines_factors():
    circular = (
        "Operating profit margin decreased to 12.3% in 2025 from 25.9% in 2024 by the same "
        "factors that lowered operating profit."
    )
    assert find(chunk(1, GROSS), chunk(2, circular)) is None


@pytest.mark.parametrize(
    "unit_change",
    [{"unit": "USD"}, {"values": {MetricName.OPERATING_MARGIN: (0.1233, 0.2591)}}],
)
def test_a_dollar_metric_or_an_unknown_referenced_value_gives_nothing(unit_change):
    assert find(chunk(1, GROSS), chunk(2, OPERATING), metrics=FakeMetrics(**unit_change)) is None


def test_a_text_change_gives_nothing():
    text_change = change().model_copy(update={"metric_id": None})
    found = find_back_reference(
        text_change,
        COMPARISON,
        MetricName.OPERATING_MARGIN,
        FakeMetrics(),
        FakeFiling(chunk(1, GROSS), chunk(2, OPERATING)),
    )
    assert found is None


@pytest.mark.parametrize(
    ("value", "shown"), [(0.1233, "12.3%"), (0.335, "33.5%"), (0.07449, "7.4%"), (0.2591, "25.9%")]
)
def test_values_are_shown_as_filings_round_them(value, shown):
    assert shown_percent(value) == shown


def test_a_sentence_with_the_values_but_another_line_is_not_about_the_change():
    other = (
        "Net margin decreased to 12.3% in 2025 from 25.9% in 2024 by the same factors noted "
        "above in gross margin."
    )
    assert find(chunk(1, GROSS), chunk(2, other)) is None


def test_a_paragraph_that_itself_points_elsewhere_is_not_the_explanation():
    pointing = (
        "Gross margin was 33.5% in 2025 compared with 47.3% in 2024, driven by the same factors "
        "noted above in net sales."
    )
    assert find(chunk(1, pointing), chunk(2, OPERATING)) is None


def test_a_paragraph_must_name_the_referenced_line():
    unnamed = "Results were 33.5% in 2025 compared with 47.3% in 2024, driven by tariffs."
    assert find(chunk(1, unnamed), chunk(2, OPERATING)) is None


def test_a_linked_ratio_is_not_another_lines_factors():
    # Cost of sales as a share of revenue is gross margin restated, so it explains nothing new.
    values = {**VALUES, MetricName.COGS_RATIO: (0.665, 0.527)}
    costs = (
        "Cost of sales was 66.5% of net sales in 2025 compared with 52.7% in 2024, due to tariffs."
    )
    sentence = (
        "Gross margin decreased to 33.5% in 2025 from 47.3% in 2024 by the same factors that "
        "raised cost of sales."
    )
    found = find(
        chunk(1, costs),
        chunk(2, sentence),
        name=MetricName.GROSS_MARGIN,
        metrics=FakeMetrics(values),
    )
    assert found is None


# -- the filing's own explanation of the change (user's decision, 2026-11-15) -------------------


def own(*chunks: Chunk, name=MetricName.GROSS_MARGIN, metrics=None):
    return find_own_explanation(
        change(name), COMPARISON, name, metrics or FakeMetrics(), FakeFiling(*chunks)
    )


def test_hersheys_gross_margin_paragraph_is_its_own_explanation():
    found = own(chunk(1, "Net sales rose."), chunk(2, "Some other paragraph.", GROSS))
    assert found is not None
    assert found.paragraph.chunk_id == f"{NOW_ACC}:mdna:0002:0"
    assert found.paragraph.quote == GROSS
    assert found.citations == (found.paragraph,)
    assert found.judged_text == GROSS
    assert found.paragraph.url == COMPARISON.current.url


def test_the_first_explaining_paragraph_in_reading_order_is_taken():
    later = GROSS.replace("tariff", "packaging")
    assert own(chunk(1, GROSS), chunk(2, later)).paragraph.quote == GROSS
    assert own(chunk(1, later, GROSS)).paragraph.quote == later


def test_the_paragraph_must_give_both_values_and_a_cause():
    one_value = GROSS.replace("47.3%", "the year before")
    assert own(chunk(1, one_value)) is None
    no_cause = "Gross margin was 33.5% in 2025 compared with 47.3% in 2024."
    assert own(chunk(1, no_cause)) is None


def test_the_paragraph_must_name_the_changed_line_itself():
    # Operating margin's change: the gross margin paragraph explains another line.
    assert own(chunk(1, GROSS), name=MetricName.OPERATING_MARGIN) is None


def test_a_same_factors_sentence_a_table_row_or_another_section_is_not_its_own_explanation():
    assert own(chunk(1, SENTENCE), name=MetricName.OPERATING_MARGIN) is None
    assert own(chunk(1, "Gross margin | 33.5% | 47.3% | driven by tariffs")) is None
    assert own(chunk(1, GROSS, section=Section.RISK_FACTORS)) is None


def test_another_filings_chunks_are_not_read():
    other = chunk(1, GROSS).model_copy(update={"accession": "acc-2024"})
    assert own(other) is None


def test_a_dollar_metric_or_a_text_change_gives_no_own_explanation():
    assert own(chunk(1, GROSS), metrics=FakeMetrics(unit="USD")) is None
    text_change = change(MetricName.GROSS_MARGIN).model_copy(update={"metric_id": None})
    found = find_own_explanation(
        text_change, COMPARISON, MetricName.GROSS_MARGIN, FakeMetrics(), FakeFiling(chunk(1, GROSS))
    )
    assert found is None
