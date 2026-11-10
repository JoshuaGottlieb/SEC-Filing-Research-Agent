"""The metric scan: thresholds, the numeric changes it makes, and the seasonal check."""

from __future__ import annotations

from datetime import date

import pytest

from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.corpus import Corpus, CorpusCompany, CorpusFiling
from sec_agent.finance.fact_store import FactStore
from sec_agent.finance.metrics import MetricCatalog, MetricsConfig, NoMoveTolerance
from sec_agent.finance.scan import (
    EXPENSES,
    MARGINS,
    MIN_MARGIN_MOVE,
    MIN_RELATIVE_MOVE,
    MoveKind,
    ScanThresholds,
    SeasonalStatus,
    is_similar,
    relative_change,
    scan_comparison,
)
from sec_agent.schemas import ChangeKind, ComparisonType, Duration, FormType, MetricName

CONFIG = MetricsConfig(
    line_items={
        "revenue": ("us-gaap:Revenue",),
        "gross_profit": ("us-gaap:GrossProfit",),
        "cost_of_goods_sold": ("us-gaap:Cost",),
        "sga": ("us-gaap:Sga",),
        "operating_income": ("us-gaap:OperatingIncomeLoss",),
        "net_income": ("us-gaap:NetIncomeLoss",),
    },
    no_move_tolerance=NoMoveTolerance(ratio=0.0005, revenue_growth=0.0, usd=0.0),
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


Q, K = FormType.FORM_10Q, FormType.FORM_10K
FY_25 = pinned(K, 2025, "FY", date(2025, 12, 31), "fy-25")
FY_24 = pinned(K, 2024, "FY", date(2024, 12, 31), "fy-24")
Q2_26 = pinned(Q, 2026, "Q2", date(2026, 6, 30), "q2-26")
Q2_25 = pinned(Q, 2025, "Q2", date(2025, 6, 30), "q2-25")
Q1_26 = pinned(Q, 2026, "Q1", date(2026, 3, 31), "q1-26")
Q1_25_END = date(2025, 3, 31)
CORPUS = Corpus(
    companies=(
        CorpusCompany(
            ticker="AAA", company="A", cik="0000000001", filings=(FY_25, FY_24, Q2_26, Q2_25, Q1_26)
        ),
    )
)
YOY_10Q = CORPUS.comparison("AAA", ComparisonType.YOY_10Q)
SEQ_10Q = CORPUS.comparison("AAA", ComparisonType.SEQ_10Q)


class Facts:
    """Writes fixture rows into xbrl_facts with the defaults the catalog wants."""

    def __init__(self, store: FactStore) -> None:
        self.store = store
        self.count = 0

    def add(
        self, filing: CorpusFiling, concept: str, value: float, end: date | None = None
    ) -> None:
        self.count += 1
        self.store._insert(
            "xbrl_facts",
            [
                {
                    "fact_id": f"{filing.accession}:{concept}:c{self.count}:usd",
                    "ticker": "AAA",
                    "accession": filing.accession,
                    "form": filing.form.value,
                    "fiscal_year": filing.fiscal_year,
                    "fiscal_period": filing.fiscal_period,
                    "concept": concept,
                    "start_date": None,
                    "end_date": end or filing.period_end,
                    "duration": (
                        Duration.TWELVE_MONTHS if filing.form is K else Duration.THREE_MONTHS
                    ).value,
                    "value": value,
                    "unit": "usd",
                    "segment": None,
                    "dimensions": None,
                }
            ],
        )

    def quarter(
        self,
        filing: CorpusFiling,
        *,
        end: date | None = None,
        revenue: float = 1000.0,
        gross_profit: float = 400.0,
        cost: float = 600.0,
        sga: float = 200.0,
        operating: float = 150.0,
        net: float = 100.0,
    ) -> None:
        """One period's figures, as `filing` reports them for the period ending `end`."""
        for concept, value in (
            ("us-gaap:Revenue", revenue),
            ("us-gaap:GrossProfit", gross_profit),
            ("us-gaap:Cost", cost),
            ("us-gaap:Sga", sga),
            ("us-gaap:OperatingIncomeLoss", operating),
            ("us-gaap:NetIncomeLoss", net),
        ):
            self.add(filing, concept, value, end)


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


def move_of(scan, metric: MetricName):
    return next(move for move in scan.moves if move.metric is metric)


# Thresholds


def test_the_scan_covers_margins_revenue_growth_and_major_expenses(facts, catalog):
    facts.quarter(Q2_26)
    facts.quarter(Q2_25)
    scan = scan_comparison(catalog, YOY_10Q)
    assert [move.metric for move in scan.moves] == [*MARGINS, MetricName.REVENUE_GROWTH, *EXPENSES]
    # The cost and SG&A ratios would repeat the gross margin and the dollar moves.
    assert MetricName.COGS_RATIO not in {move.metric for move in scan.moves}


@pytest.mark.parametrize(
    ("gross_profit", "flagged"), [(410.1, True), (409.9, False), (389.9, True)]
)
def test_a_margin_move_of_100_basis_points_is_flagged(facts, catalog, gross_profit, flagged):
    facts.quarter(Q2_26, gross_profit=gross_profit)
    facts.quarter(Q2_25, gross_profit=400.0)
    move = move_of(scan_comparison(catalog, YOY_10Q), MetricName.GROSS_MARGIN)
    assert move.kind is MoveKind.POINTS
    assert move.threshold == MIN_MARGIN_MOVE == 0.01
    assert move.move == pytest.approx((gross_profit - 400.0) / 1000.0)
    assert move.flagged is flagged


@pytest.mark.parametrize(("revenue", "flagged"), [(1101.0, True), (1099.0, False), (899.0, True)])
def test_revenue_growth_of_10_percent_is_flagged(facts, catalog, revenue, flagged):
    facts.quarter(Q2_26, revenue=revenue)
    facts.quarter(Q2_25, revenue=1000.0)
    move = move_of(scan_comparison(catalog, YOY_10Q), MetricName.REVENUE_GROWTH)
    assert (move.kind, move.threshold) == (MoveKind.RELATIVE, MIN_RELATIVE_MOVE)
    assert move.flagged is flagged
    assert move.metric_id == "AAA:revenue_growth:Q2 2026:3m:vs-Q2 2025"


@pytest.mark.parametrize(("sga", "flagged"), [(220.2, True), (219.8, False), (179.8, True)])
def test_an_expense_move_of_10_percent_in_dollars_is_flagged(facts, catalog, sga, flagged):
    facts.quarter(Q2_26, sga=sga)
    facts.quarter(Q2_25, sga=200.0)
    move = move_of(scan_comparison(catalog, YOY_10Q), MetricName.SGA)
    assert move.kind is MoveKind.RELATIVE
    assert move.move == pytest.approx(sga / 200.0 - 1)
    assert move.flagged is flagged
    assert move.metric_id == "AAA:sga:Q2 2026:3m"


def test_a_relative_move_needs_a_positive_start():
    assert relative_change(110.0, 100.0) == pytest.approx(0.10)
    assert relative_change(10.0, 0.0) is None
    assert relative_change(10.0, -5.0) is None


def test_missing_data_is_not_a_move(facts, catalog):
    facts.quarter(Q2_26)
    move = move_of(scan_comparison(catalog, YOY_10Q), MetricName.GROSS_MARGIN)
    assert (move.move, move.flagged, move.metric_id) == (None, False, None)


def test_thresholds_can_be_set(facts, catalog):
    facts.quarter(Q2_26, gross_profit=405.0)
    facts.quarter(Q2_25)
    tighter = ScanThresholds(margin_points=0.004)
    assert move_of(scan_comparison(catalog, YOY_10Q, tighter), MetricName.GROSS_MARGIN).flagged


# The changes


def test_each_flagged_move_becomes_a_numeric_change_that_resolves(facts, catalog):
    facts.quarter(Q2_26, gross_profit=450.0, sga=260.0)
    facts.quarter(Q2_25)
    scan = scan_comparison(catalog, YOY_10Q)
    assert [change.change_id for change in scan.changes] == [
        "AAA:yoy-10q:numeric:gross_margin",
        "AAA:yoy-10q:numeric:sga",
    ]
    for change in scan.changes:
        assert change.kind is ChangeKind.NUMERIC
        assert change.comparison_id == "AAA:yoy-10q"
        assert change.section is None
        value = catalog.get_metric(change.metric_id)
        assert value is not None and value.period == YOY_10Q.current.period


def test_nothing_flagged_gives_no_changes(facts, catalog):
    facts.quarter(Q2_26)
    facts.quarter(Q2_25)
    assert scan_comparison(catalog, YOY_10Q).changes == ()


def test_a_year_over_year_change_is_not_checked_for_seasonality(facts, catalog):
    facts.quarter(Q2_26, gross_profit=450.0)
    facts.quarter(Q2_25)
    scan = scan_comparison(catalog, YOY_10Q)
    assert move_of(scan, MetricName.GROSS_MARGIN).seasonal is SeasonalStatus.NOT_SEQUENTIAL
    (change,) = scan.changes
    assert (change.seasonality_checked, change.possibly_seasonal) == (False, False)


# The seasonal check (sequential quarters)


def sequential(facts, *, now: dict, before: dict, year_earlier_now=None, year_earlier_before=None):
    """Q2 2026 against Q1 2026, with Q2 2025 and Q1 2025 if given."""
    facts.quarter(Q2_26, **now)
    facts.quarter(Q1_26, **before)
    if year_earlier_now is not None:
        facts.quarter(Q2_25, **year_earlier_now)  # Q2 2025's own filing
    if year_earlier_before is not None:
        facts.quarter(Q1_26, end=Q1_25_END, **year_earlier_before)  # beside Q1 2026


def test_a_similar_move_a_year_earlier_marks_the_change_possibly_seasonal(facts, catalog):
    # Revenue up 30% now; up 20% from Q1 to Q2 a year earlier.
    sequential(
        facts,
        now={"revenue": 1300.0},
        before={"revenue": 1000.0},
        year_earlier_now={"revenue": 1200.0},
        year_earlier_before={"revenue": 1000.0},
    )
    scan = scan_comparison(catalog, SEQ_10Q)
    move = move_of(scan, MetricName.REVENUE_GROWTH)
    assert move.seasonal is SeasonalStatus.SIMILAR_MOVE
    assert move.year_earlier_move == pytest.approx(0.20)
    assert len(move.year_earlier_fact_ids) == 2
    change = next(c for c in scan.changes if c.metric_id == move.metric_id)
    assert (change.seasonality_checked, change.possibly_seasonal) == (True, True)


def test_an_opposite_move_a_year_earlier_is_checked_and_not_seasonal(facts, catalog):
    # Gross margin up 300 bps now; down 200 bps a year earlier.
    sequential(
        facts,
        now={"gross_profit": 430.0},
        before={"gross_profit": 400.0},
        year_earlier_now={"gross_profit": 380.0},
        year_earlier_before={"gross_profit": 400.0},
    )
    scan = scan_comparison(catalog, SEQ_10Q)
    move = move_of(scan, MetricName.GROSS_MARGIN)
    assert move.seasonal is SeasonalStatus.NO_SIMILAR_MOVE
    assert move.year_earlier_move == pytest.approx(-0.02)
    change = next(c for c in scan.changes if c.metric_id == move.metric_id)
    assert (change.seasonality_checked, change.possibly_seasonal) == (True, False)


def test_without_year_earlier_figures_the_seasonal_check_is_refused(facts, catalog):
    # Only Q2 2025 exists; Q1 2025 is missing, so the step a year earlier can't be measured.
    sequential(
        facts,
        now={"gross_profit": 450.0},
        before={"gross_profit": 400.0},
        year_earlier_now={"gross_profit": 450.0},
    )
    scan = scan_comparison(catalog, SEQ_10Q)
    move = move_of(scan, MetricName.GROSS_MARGIN)
    assert move.seasonal is SeasonalStatus.NO_YEAR_EARLIER_DATA
    assert move.year_earlier_move is None
    change = next(c for c in scan.changes if c.metric_id == move.metric_id)
    assert (change.seasonality_checked, change.possibly_seasonal) == (False, False)


def test_a_move_that_cant_be_measured_has_nothing_to_explain(facts, catalog):
    facts.quarter(Q2_26)
    facts.quarter(Q2_25)
    facts.quarter(Q1_26, end=Q1_25_END)
    scan = scan_comparison(catalog, SEQ_10Q)  # Q1 2026's own figures are missing
    move = move_of(scan, MetricName.GROSS_MARGIN)
    assert (move.seasonal, move.flagged) == (SeasonalStatus.NO_MOVE, False)


@pytest.mark.parametrize(
    ("move", "year_earlier", "similar"),
    [
        (0.10, 0.05, True),  # exactly half
        (0.10, 0.049, False),  # under half
        (0.10, 0.30, True),  # larger a year earlier still explains it
        (-0.10, -0.06, True),
        (0.10, -0.10, False),  # opposite direction
        (0.10, 0.0, False),
    ],
)
def test_similar_means_the_same_direction_and_at_least_half_as_large(move, year_earlier, similar):
    assert is_similar(move, year_earlier, 0.5) is similar


# The real filings


@pytest.fixture(scope="module")
def real():
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    with FactStore(get_settings().duckdb_path, read_only=True) as opened:
        yield MetricCatalog(opened, corpus, MetricsConfig.load()), corpus


@pytest.mark.corpus
def test_real_scan_of_all_nine_comparisons(real):
    catalog, corpus = real
    for company in corpus.companies:
        for comparison in company.comparisons():
            scan = scan_comparison(catalog, comparison)
            sequential = comparison.comparison_type is ComparisonType.SEQ_10Q
            for move in scan.moves:
                # SWK reports no operating income; every other metric has data for all nine.
                if company.ticker == "SWK" and move.metric is MetricName.OPERATING_MARGIN:
                    assert move.move is None and not move.flagged
                    continue
                assert move.move is not None, (comparison.comparison_id, move.metric)
                # The pinned 10-Qs hold every year-earlier quarter, so no check is refused.
                if sequential:
                    assert move.seasonal in (
                        SeasonalStatus.SIMILAR_MOVE,
                        SeasonalStatus.NO_SIMILAR_MOVE,
                    )
            for change in scan.changes:
                assert catalog.get_metric(change.metric_id) is not None
                assert change.seasonality_checked is sequential


@pytest.mark.corpus
def test_real_known_moves_are_flagged(real):
    catalog, corpus = real
    # Hershey's Q2 gross margin: 30.45% in 2025, 45.32% in 2026 (T1.4, filed statements).
    hsy = scan_comparison(catalog, corpus.comparison("HSY", ComparisonType.YOY_10Q))
    gross = move_of(hsy, MetricName.GROSS_MARGIN)
    assert gross.flagged and gross.move == pytest.approx(0.4532 - 0.3045, abs=0.0005)
    # Mattel's Q1 to Q2 revenue rise (+30.5%) follows a similar rise a year earlier (+23.2%).
    mat = scan_comparison(catalog, corpus.comparison("MAT", ComparisonType.SEQ_10Q))
    revenue = move_of(mat, MetricName.REVENUE_GROWTH)
    assert revenue.flagged and revenue.seasonal is SeasonalStatus.SIMILAR_MOVE
    assert revenue.year_earlier_move == pytest.approx(0.2322, abs=0.0005)
