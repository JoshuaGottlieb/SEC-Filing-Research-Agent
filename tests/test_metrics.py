"""The metric catalog, checked against values worked out by hand from small fixture facts."""

from __future__ import annotations

from datetime import date

import pytest

from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.contracts import MetricChecker, MetricLookup
from sec_agent.edgar.corpus import Corpus, CorpusCompany, CorpusFiling
from sec_agent.finance.fact_store import FactStore
from sec_agent.finance.metrics import (
    GROSS_PROFIT,
    OPERATING_INCOME,
    REVENUE,
    MetricCatalog,
    MetricsConfig,
    NoMoveTolerance,
)
from sec_agent.schemas import (
    ComparisonType,
    Direction,
    Duration,
    FormType,
    MetricExpectation,
    MetricName,
    NumericCheck,
)

CONFIG = MetricsConfig(
    line_items={
        "revenue": ("us-gaap:ContractRevenue", "us-gaap:Revenues"),
        "gross_profit": ("us-gaap:GrossProfit",),
        "cost_of_goods_sold": ("us-gaap:Cost",),
        "sga": ("us-gaap:Sga", "us-gaap:OtherSga"),
        "operating_income": ("us-gaap:OperatingIncomeLoss",),
        "net_income": ("us-gaap:NetIncomeLoss",),
    },
    no_move_tolerance=NoMoveTolerance(ratio=0.0005, revenue_growth=0.0),
)


def pinned(form: FormType, year: int, period: str, end: date, accession: str) -> CorpusFiling:
    return CorpusFiling(
        form=form,
        fiscal_year=year,
        fiscal_period=period,
        period_end=end,
        filed_date=date(2026, 8, 1),
        accession=accession,
        primary_document="doc.htm",
    )


Q = FormType.FORM_10Q
K = FormType.FORM_10K
FY_25 = pinned(K, 2025, "FY", date(2025, 12, 31), "fy-25")
FY_24 = pinned(K, 2024, "FY", date(2024, 12, 31), "fy-24")
Q2_26 = pinned(Q, 2026, "Q2", date(2026, 6, 30), "q2-26")
Q2_25 = pinned(Q, 2025, "Q2", date(2025, 6, 30), "q2-25")
Q1_26 = pinned(Q, 2026, "Q1", date(2026, 3, 31), "q1-26")
COMPANY = CorpusCompany(
    ticker="AAA", company="A", cik="0000000001", filings=(FY_25, FY_24, Q2_26, Q2_25, Q1_26)
)
CORPUS = Corpus(companies=(COMPANY,))
YOY_10Q = CORPUS.comparison("AAA", ComparisonType.YOY_10Q)
SEQ_10Q = CORPUS.comparison("AAA", ComparisonType.SEQ_10Q)
YOY_10K = CORPUS.comparison("AAA", ComparisonType.YOY_10K)


class Facts:
    """Writes fixture rows into xbrl_facts with the defaults the catalog wants."""

    def __init__(self, store: FactStore) -> None:
        self.store = store
        self.count = 0

    def add(
        self,
        filing: CorpusFiling,
        concept: str,
        value: float,
        *,
        duration: Duration | None = None,
        end: date | None = None,
        unit: str = "usd",
        segment: str | None = None,
        dimensions: str | None = None,
        accession: str | None = None,
    ) -> str:
        self.count += 1
        accession = accession or filing.accession
        fact_id = f"{accession}:{concept}:c{self.count}:{unit}"
        default = Duration.TWELVE_MONTHS if filing.form is K else Duration.THREE_MONTHS
        self.store._insert(
            "xbrl_facts",
            [
                {
                    "fact_id": fact_id,
                    "ticker": "AAA",
                    "accession": accession,
                    "form": filing.form.value,
                    "fiscal_year": filing.fiscal_year,
                    "fiscal_period": filing.fiscal_period,
                    "concept": concept,
                    "start_date": None,
                    "end_date": end or filing.period_end,
                    "duration": (duration or default).value,
                    "value": value,
                    "unit": unit,
                    "segment": segment,
                    "dimensions": dimensions,
                }
            ],
        )
        return fact_id

    def statement(
        self,
        filing: CorpusFiling,
        revenue: float = 1000.0,
        gross_profit: float | None = 400.0,
        cost: float = 600.0,
        sga: float = 200.0,
        operating: float | None = 150.0,
        net: float = 100.0,
    ) -> None:
        self.add(filing, "us-gaap:ContractRevenue", revenue)
        if gross_profit is not None:
            self.add(filing, "us-gaap:GrossProfit", gross_profit)
        self.add(filing, "us-gaap:Cost", cost)
        self.add(filing, "us-gaap:Sga", sga)
        if operating is not None:
            self.add(filing, "us-gaap:OperatingIncomeLoss", operating)
        self.add(filing, "us-gaap:NetIncomeLoss", net)


@pytest.fixture
def store(settings):
    with FactStore(settings.duckdb_path) as opened:
        yield opened


@pytest.fixture
def facts(store) -> Facts:
    return Facts(store)


@pytest.fixture
def catalog(store) -> MetricCatalog:
    return MetricCatalog(store, CORPUS, CONFIG)


def expect(metric: MetricName, direction: Direction = Direction.UP) -> MetricExpectation:
    return MetricExpectation(metric=metric, direction=direction)


def test_catalog_satisfies_both_interfaces(catalog):
    assert isinstance(catalog, MetricChecker)
    assert isinstance(catalog, MetricLookup)


# --- Values, against arithmetic done by hand ---


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (MetricName.GROSS_MARGIN, 0.40),  # 400 / 1000
        (MetricName.OPERATING_MARGIN, 0.15),  # 150 / 1000
        (MetricName.NET_MARGIN, 0.10),  # 100 / 1000
        (MetricName.COGS_RATIO, 0.60),  # 600 / 1000
        (MetricName.SGA_RATIO, 0.20),  # 200 / 1000
    ],
)
def test_each_ratio_matches_the_hand_value(facts, catalog, name, expected):
    facts.statement(Q2_26)
    value = catalog.ratio(YOY_10Q.current, name)
    assert value.value == pytest.approx(expected)
    assert value.unit == "ratio"
    assert value.name == name.value
    assert value.duration is Duration.THREE_MONTHS
    assert value.period == YOY_10Q.current.period


def test_a_ratio_is_an_unrounded_fraction(facts, catalog):
    # 375.12 / 1000 stays 0.37512, not 37.5 or 0.375.
    facts.statement(Q2_26, gross_profit=375.12)
    value = catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN)
    assert value.value == pytest.approx(0.37512, abs=1e-12)


def test_a_loss_gives_a_negative_margin(facts, catalog):
    facts.statement(Q2_26, net=-18.0)
    assert catalog.ratio(YOY_10Q.current, MetricName.NET_MARGIN).value == pytest.approx(-0.018)


def test_revenue_growth_matches_the_hand_value(facts, catalog):
    # 1044 / 1000 - 1 is 4.4% growth, stored as 0.044.
    facts.statement(Q2_26, revenue=1044.0)
    facts.statement(Q2_25, revenue=1000.0)
    value = catalog.growth(YOY_10Q.current, YOY_10Q.prior)
    assert value.value == pytest.approx(0.044)
    assert value.period == YOY_10Q.current.period
    assert value.unit == "ratio"
    assert value.name == "revenue_growth"


def test_falling_revenue_gives_negative_growth(facts, catalog):
    facts.statement(Q2_26, revenue=900.0)
    facts.statement(Q2_25, revenue=1000.0)
    assert catalog.growth(YOY_10Q.current, YOY_10Q.prior).value == pytest.approx(-0.10)


def test_a_10k_uses_twelve_month_facts_and_a_10q_three(facts, catalog):
    facts.statement(FY_25, revenue=4000.0, gross_profit=1000.0)
    # A three-month figure ending the same day must not be read as the year.
    facts.add(FY_25, "us-gaap:ContractRevenue", 1.0, duration=Duration.THREE_MONTHS)
    value = catalog.ratio(YOY_10K.current, MetricName.GROSS_MARGIN)
    assert value.value == pytest.approx(0.25)
    assert value.duration is Duration.TWELVE_MONTHS


def test_year_to_date_figures_are_ignored(facts, catalog):
    # A Q2 10-Q reports six months beside the quarter, ending the same day.
    facts.statement(Q2_26)
    facts.add(Q2_26, "us-gaap:ContractRevenue", 9999.0, duration=Duration.YEAR_TO_DATE)
    facts.add(Q2_26, "us-gaap:GrossProfit", 1.0, duration=Duration.YEAR_TO_DATE)
    value = catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN)
    assert value.value == pytest.approx(0.40)


def test_a_segment_figure_is_ignored(facts, catalog):
    facts.statement(Q2_26)
    facts.add(Q2_26, "us-gaap:ContractRevenue", 7.0, segment="a:Toys", dimensions='{"seg": "toys"}')
    assert catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN).value == pytest.approx(0.40)


def test_a_breakdown_on_another_axis_is_ignored(facts, catalog):
    # A product-line figure has no segment but does have a dimension, and it is not the total.
    facts.statement(Q2_26)
    facts.add(Q2_26, "us-gaap:ContractRevenue", 7.0, dimensions='{"product": "dolls"}')
    assert catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN).value == pytest.approx(0.40)


def test_a_per_share_figure_is_ignored(facts, catalog):
    facts.statement(Q2_26)
    facts.add(Q2_26, "us-gaap:NetIncomeLoss", 3.0, unit="usdPerShare")
    assert catalog.ratio(YOY_10Q.current, MetricName.NET_MARGIN).value == pytest.approx(0.10)


def test_each_period_uses_its_own_filing_not_a_later_restatement(facts, catalog):
    # The Q2 2026 10-Q also reports Q2 2025, restated. The value traces to the Q2 2025 10-Q.
    facts.statement(Q2_25, revenue=1000.0)
    facts.add(Q2_25, "us-gaap:ContractRevenue", 800.0, accession="q2-26")
    value = catalog.ratio(YOY_10Q.prior, MetricName.GROSS_MARGIN)
    assert value.value == pytest.approx(0.40)
    assert all(fact_id.startswith("q2-25:") for fact_id in value.input_fact_ids)


def test_another_period_ending_is_not_read(facts, catalog):
    facts.add(Q2_26, "us-gaap:ContractRevenue", 1000.0, end=date(2026, 3, 31))
    facts.add(Q2_26, "us-gaap:GrossProfit", 400.0, end=date(2026, 3, 31))
    assert catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN) is None


def test_the_first_alias_a_filing_reports_is_used(facts, catalog):
    facts.statement(Q2_26, revenue=1000.0)
    facts.add(Q2_26, "us-gaap:Revenues", 5000.0)
    assert catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN).value == pytest.approx(0.40)


def test_a_later_alias_is_used_when_the_first_is_absent(facts, catalog):
    # As with SWK's Revenues and MAT's other SG&A.
    facts.add(Q2_26, "us-gaap:Revenues", 2000.0)
    facts.add(Q2_26, "us-gaap:OtherSga", 500.0)
    assert catalog.ratio(YOY_10Q.current, MetricName.SGA_RATIO).value == pytest.approx(0.25)


def test_gross_profit_is_revenue_minus_cost_when_not_reported(facts, catalog):
    # As with SWK, which has no gross profit line. 1000 - 650 over 1000.
    facts.statement(Q2_26, gross_profit=None, cost=650.0)
    value = catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN)
    assert value.value == pytest.approx(0.35)
    assert len(value.input_fact_ids) == 2
    assert any("Cost" in fact_id for fact_id in value.input_fact_ids)
    assert any("ContractRevenue" in fact_id for fact_id in value.input_fact_ids)


def test_a_reported_gross_profit_is_preferred_over_computing_it(facts, catalog):
    facts.statement(Q2_26, gross_profit=400.0, cost=999.0)
    assert catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN).value == pytest.approx(0.40)


def test_no_other_line_item_is_computed(facts, catalog):
    # As with SWK's operating income. Nothing stands in for a line the filer does not report.
    facts.statement(Q2_26, operating=None)
    assert catalog.ratio(YOY_10Q.current, MetricName.OPERATING_MARGIN) is None


def test_a_line_item_is_the_dollar_figure_the_metrics_use(facts, catalog):
    facts.statement(Q2_26, revenue=1000.0, gross_profit=400.0)

    revenue = catalog.line_item(YOY_10Q.current, REVENUE)
    gross = catalog.line_item(YOY_10Q.current, GROSS_PROFIT)

    assert (revenue, gross) == (1000.0, 400.0)
    assert gross / revenue == pytest.approx(
        catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN).value
    )


def test_a_line_item_computes_gross_profit_only_where_the_ratio_does(facts, catalog):
    facts.statement(Q2_26, gross_profit=None, cost=650.0)

    assert catalog.line_item(YOY_10Q.current, GROSS_PROFIT) == pytest.approx(350.0)


def test_a_line_item_follows_the_filing_and_its_duration(facts, catalog):
    # A year-to-date figure and another period's filing are not the quarter's figure.
    facts.add(Q2_26, "us-gaap:ContractRevenue", 1000.0)
    facts.add(Q2_26, "us-gaap:ContractRevenue", 2100.0, duration=Duration.YEAR_TO_DATE)
    facts.add(Q2_25, "us-gaap:ContractRevenue", 900.0)

    assert catalog.line_item(YOY_10Q.current, REVENUE) == 1000.0
    assert catalog.line_item(YOY_10Q.prior, REVENUE) == 900.0


def test_a_line_item_a_filing_does_not_report_is_none(facts, catalog):
    facts.statement(Q2_26, operating=None)

    assert catalog.line_item(YOY_10Q.current, OPERATING_INCOME) is None


def test_a_line_item_reported_with_two_values_is_none_and_not_an_error(facts, catalog):
    facts.statement(Q2_26)
    facts.add(Q2_26, "us-gaap:GrossProfit", 410.0)

    assert catalog.line_item(YOY_10Q.current, GROSS_PROFIT) is None


def test_an_unknown_line_item_is_an_error(catalog):
    with pytest.raises(ValueError, match="unknown line item 'ebitda'"):
        catalog.line_item(YOY_10Q.current, "ebitda")


def test_input_facts_trace_to_the_rows_used(facts, catalog):
    revenue = facts.add(Q2_26, "us-gaap:ContractRevenue", 1000.0)
    gross = facts.add(Q2_26, "us-gaap:GrossProfit", 400.0)
    value = catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN)
    assert value.input_fact_ids == (gross, revenue)


def test_growth_cites_both_periods_revenue(facts, catalog):
    now = facts.add(Q2_26, "us-gaap:ContractRevenue", 1044.0)
    before = facts.add(Q2_25, "us-gaap:ContractRevenue", 1000.0)
    assert catalog.growth(YOY_10Q.current, YOY_10Q.prior).input_fact_ids == (now, before)


def test_a_figure_repeated_with_one_value_is_fine(facts, catalog):
    facts.statement(Q2_26)
    facts.add(Q2_26, "us-gaap:GrossProfit", 400.0)
    assert catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN).value == pytest.approx(0.40)


def test_a_figure_reported_with_two_values_gives_no_metric(facts, catalog):
    # Revenue minus cost could be computed here, but that would hide the conflict with a guess.
    facts.statement(Q2_26)
    facts.add(Q2_26, "us-gaap:GrossProfit", 410.0)
    assert catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN) is None


def test_revenue_reported_with_two_values_gives_no_margin_or_growth(facts, catalog):
    facts.statement(Q2_26)
    facts.add(Q2_26, "us-gaap:ContractRevenue", 1100.0)
    facts.statement(Q2_25)
    assert catalog.ratio(YOY_10Q.current, MetricName.NET_MARGIN) is None
    assert catalog.growth(YOY_10Q.current, YOY_10Q.prior) is None


def test_conflicting_figures_are_not_applicable_in_a_check(facts, catalog):
    facts.statement(Q2_26)
    facts.add(Q2_26, "us-gaap:GrossProfit", 410.0)
    facts.statement(Q2_25, gross_profit=300.0)
    outcome = catalog.check(YOY_10Q, [expect(MetricName.GROSS_MARGIN)]).outcomes[0]
    assert outcome.check is NumericCheck.NOT_APPLICABLE


def test_zero_revenue_gives_no_margin(facts, catalog):
    facts.statement(Q2_26, revenue=0.0)
    assert catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN) is None


def test_growth_from_a_zero_base_gives_no_value(facts, catalog):
    facts.statement(Q2_26)
    facts.statement(Q2_25, revenue=0.0)
    assert catalog.growth(YOY_10Q.current, YOY_10Q.prior) is None


# --- The check ---


def test_a_move_as_expected_agrees_and_cites_both_periods(facts, catalog):
    facts.statement(Q2_26, gross_profit=420.0)
    facts.statement(Q2_25, gross_profit=400.0)
    result = catalog.check(YOY_10Q, [expect(MetricName.GROSS_MARGIN, Direction.UP)])
    outcome = result.outcomes[0]
    assert outcome.check is NumericCheck.AGREE
    assert outcome.metric_ids == ("AAA:gross_margin:Q2 2026:3m", "AAA:gross_margin:Q2 2025:3m")
    assert result.check is NumericCheck.AGREE


def test_a_move_the_opposite_way_contradicts_and_cites_both_periods(facts, catalog):
    facts.statement(Q2_26, gross_profit=380.0)
    facts.statement(Q2_25, gross_profit=400.0)
    outcome = catalog.check(YOY_10Q, [expect(MetricName.GROSS_MARGIN, Direction.UP)]).outcomes[0]
    assert outcome.check is NumericCheck.CONTRADICT
    assert len(outcome.metric_ids) == 2


def test_each_cited_metric_resolves_and_gives_the_change(facts, catalog):
    # The caller's own check: 39.5% to 37.5% is a 200 basis point fall.
    facts.statement(Q2_26, gross_profit=375.0)
    facts.statement(Q2_25, gross_profit=395.0)
    outcome = catalog.check(YOY_10Q, [expect(MetricName.GROSS_MARGIN, Direction.DOWN)]).outcomes[0]
    now, before = (catalog.get_metric(metric_id) for metric_id in outcome.metric_ids)
    assert now.value - before.value == pytest.approx(-0.02)
    assert (now.name, now.duration) == (before.name, before.duration)


def test_a_sequential_comparison_uses_consecutive_quarters(facts, catalog):
    facts.statement(Q2_26, gross_profit=420.0)
    facts.statement(Q1_26, gross_profit=400.0)
    facts.statement(Q2_25, gross_profit=500.0)  # Would flip the verdict if it were read.
    outcome = catalog.check(SEQ_10Q, [expect(MetricName.GROSS_MARGIN)]).outcomes[0]
    assert outcome.check is NumericCheck.AGREE
    assert outcome.metric_ids[1] == "AAA:gross_margin:Q1 2026:3m"


def test_a_10k_comparison_cites_twelve_month_values(facts, catalog):
    facts.statement(FY_25, gross_profit=420.0)
    facts.statement(FY_24, gross_profit=400.0)
    outcome = catalog.check(YOY_10K, [expect(MetricName.GROSS_MARGIN)]).outcomes[0]
    assert outcome.metric_ids == ("AAA:gross_margin:FY2025:12m", "AAA:gross_margin:FY2024:12m")


@pytest.mark.parametrize("missing", [Q2_26, Q2_25])
def test_no_data_in_either_period_is_not_applicable(facts, catalog, missing):
    facts.statement(Q2_25 if missing is Q2_26 else Q2_26)
    outcome = catalog.check(YOY_10Q, [expect(MetricName.GROSS_MARGIN)]).outcomes[0]
    assert outcome.check is NumericCheck.NOT_APPLICABLE
    assert outcome.metric_ids == ()


def test_an_exact_tie_is_not_applicable(facts, catalog):
    facts.statement(Q2_26)
    facts.statement(Q2_25)
    outcome = catalog.check(YOY_10Q, [expect(MetricName.GROSS_MARGIN)]).outcomes[0]
    assert outcome.check is NumericCheck.NOT_APPLICABLE


@pytest.mark.parametrize(
    ("gross_profit", "direction", "expected"),
    [
        (400.49, Direction.UP, NumericCheck.NOT_APPLICABLE),  # Up 4.9 bps.
        (400.49, Direction.DOWN, NumericCheck.NOT_APPLICABLE),  # A tiny move contradicts nothing.
        (399.51, Direction.UP, NumericCheck.NOT_APPLICABLE),  # Down 4.9 bps.
        (400.51, Direction.UP, NumericCheck.AGREE),  # Up 5.1 bps.
        (400.51, Direction.DOWN, NumericCheck.CONTRADICT),
        (399.49, Direction.DOWN, NumericCheck.AGREE),  # Down 5.1 bps.
    ],
)
def test_a_ratio_move_under_five_basis_points_cant_be_called(
    facts, catalog, gross_profit, direction, expected
):
    facts.statement(Q2_26, gross_profit=gross_profit)
    facts.statement(Q2_25, gross_profit=400.0)
    outcome = catalog.check(YOY_10Q, [expect(MetricName.GROSS_MARGIN, direction)]).outcomes[0]
    assert outcome.check is expected


@pytest.mark.parametrize(
    ("revenue", "direction", "expected"),
    [
        (1044.0, Direction.UP, NumericCheck.AGREE),
        (1044.0, Direction.DOWN, NumericCheck.CONTRADICT),
        (900.0, Direction.DOWN, NumericCheck.AGREE),
        (900.0, Direction.UP, NumericCheck.CONTRADICT),
        (1000.0, Direction.UP, NumericCheck.NOT_APPLICABLE),
    ],
)
def test_revenue_growth_direction_is_about_revenue(facts, catalog, revenue, direction, expected):
    facts.statement(Q2_26, revenue=revenue)
    facts.statement(Q2_25, revenue=1000.0)
    outcome = catalog.check(YOY_10Q, [expect(MetricName.REVENUE_GROWTH, direction)]).outcomes[0]
    assert outcome.check is expected


def test_slowing_growth_is_still_up_when_revenue_rose(facts, catalog):
    # The contract's warning: 2% growth is not revenue_growth DOWN, however fast it used to be.
    facts.statement(Q2_26, revenue=1020.0)
    facts.statement(Q2_25, revenue=1000.0)
    outcome = catalog.check(YOY_10Q, [expect(MetricName.REVENUE_GROWTH, Direction.DOWN)])
    assert outcome.check is NumericCheck.CONTRADICT


def test_a_tiny_revenue_move_can_still_be_called(facts, catalog):
    # Its tolerance is zero, unlike a ratio's.
    facts.statement(Q2_26, revenue=1000.01)
    facts.statement(Q2_25, revenue=1000.0)
    outcome = catalog.check(YOY_10Q, [expect(MetricName.REVENUE_GROWTH)]).outcomes[0]
    assert outcome.check is NumericCheck.AGREE


def test_revenue_growth_cites_its_one_value(facts, catalog):
    facts.statement(Q2_26, revenue=1044.0)
    facts.statement(Q2_25, revenue=1000.0)
    outcome = catalog.check(YOY_10Q, [expect(MetricName.REVENUE_GROWTH)]).outcomes[0]
    assert outcome.metric_ids == ("AAA:revenue_growth:Q2 2026:3m:vs-Q2 2025",)
    assert catalog.get_metric(outcome.metric_ids[0]).value == pytest.approx(0.044)


def test_revenue_growth_with_a_missing_period_is_not_applicable(facts, catalog):
    facts.statement(Q2_26)
    outcome = catalog.check(YOY_10Q, [expect(MetricName.REVENUE_GROWTH)]).outcomes[0]
    assert outcome.check is NumericCheck.NOT_APPLICABLE


def test_outcomes_keep_the_order_of_the_expectations(facts, catalog):
    facts.statement(Q2_26, gross_profit=420.0, net=90.0)
    facts.statement(Q2_25, gross_profit=400.0, net=100.0)
    names = [MetricName.NET_MARGIN, MetricName.GROSS_MARGIN, MetricName.SGA_RATIO]
    result = catalog.check(YOY_10Q, [expect(name) for name in names])
    assert [outcome.metric for outcome in result.outcomes] == names
    assert [outcome.check for outcome in result.outcomes] == [
        NumericCheck.CONTRADICT,
        NumericCheck.AGREE,
        NumericCheck.NOT_APPLICABLE,
    ]


def test_one_contradiction_contradicts_the_claim(facts, catalog):
    facts.statement(Q2_26, gross_profit=420.0, net=90.0)
    facts.statement(Q2_25, gross_profit=400.0, net=100.0)
    result = catalog.check(
        YOY_10Q, [expect(MetricName.GROSS_MARGIN), expect(MetricName.NET_MARGIN)]
    )
    assert result.check is NumericCheck.CONTRADICT


def test_no_expectations_is_not_applicable(catalog):
    assert catalog.check(YOY_10Q, []).check is NumericCheck.NOT_APPLICABLE


# --- Looking a metric up by id ---


def test_a_ratio_id_rebuilds_the_same_metric(facts, catalog):
    facts.statement(Q2_26)
    value = catalog.ratio(YOY_10Q.current, MetricName.GROSS_MARGIN)
    assert value.metric_id == "AAA:gross_margin:Q2 2026:3m"
    assert catalog.get_metric(value.metric_id) == value


def test_a_growth_id_rebuilds_the_same_metric(facts, catalog):
    facts.statement(Q2_26, revenue=1044.0)
    facts.statement(Q2_25, revenue=1000.0)
    value = catalog.growth(YOY_10Q.current, YOY_10Q.prior)
    assert catalog.get_metric(value.metric_id) == value


def test_growth_differs_by_what_it_is_compared_with(facts, catalog):
    facts.statement(Q2_26, revenue=1100.0)
    facts.statement(Q2_25, revenue=1000.0)
    facts.statement(Q1_26, revenue=1250.0)
    assert catalog.get_metric("AAA:revenue_growth:Q2 2026:3m:vs-Q2 2025").value == pytest.approx(
        0.10
    )
    assert catalog.get_metric("AAA:revenue_growth:Q2 2026:3m:vs-Q1 2026").value == pytest.approx(
        -0.12
    )


@pytest.mark.parametrize(
    "metric_id",
    [
        "",
        "nonsense",
        "BBB:gross_margin:Q2 2026:3m",  # A company not in the corpus.
        "AAA:not_a_metric:Q2 2026:3m",
        "AAA:gross_margin:Q3 2026:3m",  # A period not in the corpus.
        "AAA:gross_margin:Q2 2026:12m",  # The wrong duration for a 10-Q.
        "AAA:gross_margin:FY2025:3m",  # The wrong duration for a 10-K.
        "AAA:gross_margin:Q2 2026:3m:vs-Q2 2025",  # A ratio isn't measured against a period.
        "AAA:revenue_growth:Q2 2026:3m",  # Growth must be.
        "AAA:revenue_growth:Q2 2026:3m:Q2 2025",  # A malformed comparison.
        "AAA:revenue_growth:Q2 2026:3m:vs-Q9 2099",
        "AAA:revenue_growth:Q2 2025:3m:vs-Q2 2026",  # Against a later period.
        "AAA:revenue_growth:Q2 2026:3m:vs-FY2025",  # A quarter against a year.
        "AAA:gross_margin:Q2 2026:3m:extra:parts",
    ],
)
def test_an_id_naming_nothing_returns_none(facts, catalog, metric_id):
    for filing in COMPANY.filings:
        facts.statement(filing)
    assert catalog.get_metric(metric_id) is None


def test_a_well_formed_id_with_no_data_returns_none(catalog):
    assert catalog.get_metric("AAA:gross_margin:Q2 2026:3m") is None


# --- The shipped configuration and the real filings ---


def test_the_shipped_config_names_every_line_item_the_catalog_reads():
    config = MetricsConfig.load()
    assert set(config.line_items) == {
        "revenue",
        "gross_profit",
        "cost_of_goods_sold",
        "sga",
        "operating_income",
        "net_income",
    }
    assert config.no_move_tolerance.ratio == pytest.approx(0.0005)
    assert config.no_move_tolerance.revenue_growth == 0.0


@pytest.fixture(scope="module")
def real():
    settings = get_settings()
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    with FactStore(settings.duckdb_path, read_only=True) as opened:
        yield MetricCatalog(opened, corpus, MetricsConfig.load()), corpus


# Hand arithmetic on figures in millions of dollars. HSY's and MAT's come from their XBRL facts, so
# these catch a wrong concept, period or duration, but T1.4 still has to check them against the
# filed statements. SWK's gross profit is the exception: its filed tables state it.
FILED_VALUES = [
    ("HSY", "Q2 2026", MetricName.GROSS_MARGIN, 1263.308 / 2787.306),
    ("HSY", "Q2 2025", MetricName.GROSS_MARGIN, 796.273 / 2614.718),
    ("HSY", "FY2025", MetricName.OPERATING_MARGIN, 1441.528 / 11692.576),
    ("MAT", "Q2 2026", MetricName.NET_MARGIN, -18.215 / 1125.326),
    ("MAT", "Q2 2026", MetricName.SGA_RATIO, 406.932 / 1125.326),
    ("MAT", "FY2024", MetricName.COGS_RATIO, 2645.478 / 5379.546),
    # SWK reports no gross profit line. 1,305.8 and 4,588.3 are in its filed non-GAAP tables.
    ("SWK", "Q2 2026", MetricName.GROSS_MARGIN, 1305.8 / 3960.7),
    ("SWK", "FY2025", MetricName.GROSS_MARGIN, 4588.3 / 15130.4),
    ("SWK", "FY2024", MetricName.GROSS_MARGIN, 4514.4 / 15365.7),
]


@pytest.mark.corpus
@pytest.mark.parametrize(("ticker", "label", "name", "expected"), FILED_VALUES)
def test_real_values_match_hand_arithmetic(real, ticker, label, name, expected):
    catalog, _ = real
    duration = "12m" if label.startswith("FY") else "3m"
    value = catalog.get_metric(f"{ticker}:{name.value}:{label}:{duration}")
    assert value is not None
    assert value.value == pytest.approx(expected, abs=1e-6)


@pytest.mark.corpus
def test_every_real_comparison_can_be_checked(real):
    catalog, corpus = real
    expectations = [MetricExpectation(metric=name, direction=Direction.UP) for name in MetricName]
    for company in corpus.companies:
        for comparison in company.comparisons():
            result = catalog.check(comparison, expectations)
            assert len(result.outcomes) == len(MetricName)
            for outcome in result.outcomes:
                # SWK reports no operating income. Every other metric has data for all nine.
                if outcome.metric is MetricName.OPERATING_MARGIN and company.ticker == "SWK":
                    assert outcome.check is NumericCheck.NOT_APPLICABLE
                    continue
                assert outcome.metric_ids, (comparison.comparison_id, outcome.metric)
                for metric_id in outcome.metric_ids:
                    value = catalog.get_metric(metric_id)
                    assert value is not None and value.metric_id == metric_id
                    assert value.input_fact_ids


@pytest.mark.corpus
def test_real_input_facts_exist_and_stay_on_one_concept_per_company(real):
    catalog, corpus = real
    for company in corpus.companies:
        concepts = set()
        for pinned_filing in company.filings:
            filing = company.to_filing(pinned_filing)
            value = catalog.ratio(filing, MetricName.SGA_RATIO)
            assert value is not None
            # An id is <accession>:<concept>:<context>:<unit>. A company changing concept between
            # filings would make its ratios incomparable without any error.
            assert all(fact_id.startswith(filing.accession) for fact_id in value.input_fact_ids)
            concepts |= {
                fact_id.split(":", 1)[1].rsplit(":", 2)[0] for fact_id in value.input_fact_ids
            }
        assert len(concepts) == 2, (company.ticker, concepts)
