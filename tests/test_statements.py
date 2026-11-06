"""Filed income statements: the reader is strict, and the catalog matches all 15 filings."""

from __future__ import annotations

import pytest

from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore
from sec_agent.finance.metrics import GROSS_PROFIT, REVENUE, MetricCatalog, MetricsConfig
from sec_agent.finance.statements import (
    CURRENT,
    YEAR_EARLIER,
    Figure,
    StatementError,
    expected_years,
    find_income_statement,
    read_income_statement,
)
from sec_agent.schemas import ComparisonType, FormType

THOUSANDS = "(in thousands, except per share amounts)"
MILLIONS = "(Millions of Dollars, Except Per Share Amounts)"

Q2_TEXT = "\n".join(
    [
        "CONSOLIDATED STATEMENTS OF INCOME",
        "Three Months Ended | Six Months Ended",
        "June 28, 2026 | June 29, 2025 | June 28, 2026 | June 29, 2025",
        "Net sales | $ | 2,787,306 | $ | 2,614,718 | $ | 5,891,473 | $ | 5,420,137",
        "Cost of sales | 1,523,998 | 1,818,445 | 3,405,434 | 3,679,597",
        "Gross profit | 1,263,308 | 796,273 | 2,486,039 | 1,740,540",
        "Other (income) expense, net | 4,398 | (2,336) | 2,578 | (1,391)",
        "Net income | $ | 545,000 | $ | 120,000 | $ | 1,000,000 | $ | 400,000",
    ]
)


def table(
    text: str = Q2_TEXT,
    *,
    caption: str | None = THOUSANDS,
    form: str = "10-Q",
    fiscal_period: str = "Q2",
    fiscal_year: int = 2026,
    section: str = "financial_notes",
    table_id: str = "acc:t9",
) -> dict:
    return {
        "table_id": table_id,
        "ticker": "HSY",
        "accession": "acc",
        "form": form,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "section": section,
        "table_index": 9,
        "title": None,
        "reasons": "no unit or scale for row: Common stock",
        "text": text,
        "scale_caption": caption,
    }


# --- Reading the figures ---


def test_reads_a_quarter_in_thousands_as_dollars_with_its_rounding():
    statement = read_income_statement(table())

    revenue = statement.revenue()
    assert revenue.dollars == 2_787_306_000
    # Rounded to the thousand, so a source may differ by up to 500 dollars.
    assert revenue.tolerance == 500
    assert not revenue.derived
    assert statement.gross_profit().dollars == 1_263_308_000
    assert statement.cost_of_sales().dollars == 1_523_998_000


def test_the_first_column_is_the_filings_period_the_second_a_year_earlier_then_year_to_date():
    statement = read_income_statement(table())

    assert statement.column_years == (2026, 2025, 2026, 2025)
    assert statement.revenue(CURRENT).dollars == 2_787_306_000
    assert statement.revenue(YEAR_EARLIER).dollars == 2_614_718_000
    assert statement.revenue(2).dollars == 5_891_473_000  # six months, which no check uses


def test_a_statement_in_millions_with_a_decimal_is_rounded_to_a_tenth_of_a_million():
    text = "\n".join(
        [
            "Fiscal years ended January 3, 2026, December 28, 2024, and December 30, 2023",
            "2025 | 2024 | 2023",
            "Net Sales | $ | 15,130.4 | $ | 15,365.7 | $ | 15,781.1",
            "Costs and Expenses",
            "Cost of sales | $ | 10,542.1 | $ | 10,851.3 | $ | 11,848.5",
            "Net earnings | $ | 100.0 | $ | 90.0 | $ | 80.0",
        ]
    )
    statement = read_income_statement(
        table(text, caption=MILLIONS, form="10-K", fiscal_period="FY", fiscal_year=2025)
    )

    assert statement.revenue().dollars == pytest.approx(15_130_400_000)
    assert statement.revenue().tolerance == pytest.approx(50_000)
    assert statement.revenue(2).dollars == pytest.approx(15_781_100_000)


def test_a_loss_in_parentheses_a_leading_minus_and_a_dash_are_read_correctly():
    text = Q2_TEXT.replace(
        "Gross profit | 1,263,308 | 796,273 | 2,486,039 | 1,740,540",
        "Gross profit | (1,263,308) | -796,273 | — | 1,740,540",
    )

    figure = [read_income_statement(table(text)).gross_profit(column) for column in range(4)]

    assert [item.dollars for item in figure] == [-1_263_308_000, -796_273_000, 0, 1_740_540_000]


def test_gross_profit_is_derived_when_the_statement_shows_none():
    text = "\n".join(
        [
            "2025 | 2024 | 2023",
            "Net Sales | $ | 15,130.4 | $ | 15,365.7 | $ | 15,781.1",
            "Cost of sales | $ | 10,542.1 | $ | 10,851.3 | $ | 11,848.5",
            "Net earnings | $ | 100.0 | $ | 90.0 | $ | 80.0",
        ]
    )
    statement = read_income_statement(
        table(text, caption=MILLIONS, form="10-K", fiscal_period="FY", fiscal_year=2025)
    )

    gross = statement.gross_profit()

    assert gross.derived
    assert gross.dollars == pytest.approx(4_588_300_000)
    # Revenue and cost are each rounded to 0.1 million, so the difference is good to 0.1 million.
    assert gross.tolerance == pytest.approx(100_000)


def test_a_reported_gross_profit_is_used_as_it_is_and_not_recomputed():
    text = Q2_TEXT.replace("1,263,308", "1,263,000")  # not revenue minus cost on purpose

    gross = read_income_statement(table(text)).gross_profit()

    assert gross.dollars == 1_263_000_000
    assert not gross.derived


def test_no_gross_profit_and_no_cost_of_sales_is_none_and_not_a_guess():
    text = "\n".join(
        [
            "Three Months Ended",
            "March 29, 2026 | March 30, 2025",
            "Net sales | $ | 3,104,167 | $ | 2,805,419",
            "Cost of sales | 1,881,436 | 1,861,152",
            "Net income | $ | 1 | $ | 2",
        ]
    ).replace("Cost of sales", "Cost of products")
    statement = read_income_statement(table(text, fiscal_period="Q1"))

    assert statement.cost_of_sales() is None
    assert statement.gross_profit() is None


def test_figures_agree_within_half_the_displayed_unit_and_not_beyond():
    figure = Figure(dollars=1_000_000, tolerance=500)

    assert figure.agrees_with(1_000_500)
    assert figure.agrees_with(999_500)
    assert not figure.agrees_with(1_000_501)


# --- The reader refuses what it doesn't understand ---


def test_column_years_that_disagree_with_the_filing_are_an_error():
    text = Q2_TEXT.replace("June 28, 2026 | June 29, 2025", "June 29, 2025 | June 30, 2024", 1)

    with pytest.raises(StatementError, match="column years"):
        read_income_statement(table(text))


def test_a_filing_with_a_different_period_than_the_header_is_an_error():
    with pytest.raises(StatementError, match="column years"):
        read_income_statement(table(fiscal_year=2025))


def test_a_row_with_a_missing_column_is_an_error():
    text = Q2_TEXT.replace("Cost of sales | 1,523,998 | 1,818,445 |", "Cost of sales | 1,523,998 |")

    with pytest.raises(StatementError, match="Cost of sales"):
        read_income_statement(table(text))


@pytest.mark.parametrize(
    "caption",
    [None, "(in dollars)", "(in thousands, except per share amounts; millions elsewhere)"],
    ids=["none", "neither", "both"],
)
def test_a_caption_without_exactly_one_scale_is_an_error(caption):
    with pytest.raises(StatementError, match="scale"):
        read_income_statement(table(caption=caption))


def test_no_revenue_row_is_an_error():
    with pytest.raises(StatementError, match="0 revenue rows"):
        read_income_statement(table(Q2_TEXT.replace("Net sales", "Receipts")))


def test_two_revenue_rows_are_an_error_and_not_a_pick():
    extra = Q2_TEXT.replace(
        "Cost of sales", "Net sales | $ | 1 | $ | 2 | $ | 3 | $ | 4\nCost of sales"
    )

    with pytest.raises(StatementError, match="2 revenue rows"):
        read_income_statement(table(extra))


def test_the_heading_nearest_the_data_labels_the_columns_and_not_the_title():
    # Stanley's title names three fiscal years, two of them the wrong ones for the columns below.
    text = "\n".join(
        [
            "Fiscal years ended January 3, 2026, December 28, 2024, and December 30, 2023",
            "2025 | 2024 | 2023",
            "Net Sales | $ | 15,130.4 | $ | 15,365.7 | $ | 15,781.1",
            "Cost of sales | $ | 1.0 | $ | 2.0 | $ | 3.0",
            "Net earnings | $ | 1.0 | $ | 2.0 | $ | 3.0",
        ]
    )

    statement = read_income_statement(
        table(text, caption=MILLIONS, form="10-K", fiscal_period="FY", fiscal_year=2025)
    )

    assert statement.column_years == (2025, 2024, 2023)


def test_expected_years_by_filing_kind():
    assert expected_years(FormType.FORM_10K, "FY", 2025) == (2025, 2024, 2023)
    assert expected_years(FormType.FORM_10Q, "Q1", 2026) == (2026, 2025)
    assert expected_years(FormType.FORM_10Q, "Q2", 2026) == (2026, 2025, 2026, 2025)


# --- Finding the statement among a filing's tables ---


def test_the_statement_in_the_financial_statements_is_found_and_the_mdna_copy_ignored():
    statement = table(table_id="acc:t9")
    mdna_copy = table(table_id="acc:t95", section="mdna")
    other = table(Q2_TEXT.replace("Net income", "Dividends"), table_id="acc:t12")

    assert find_income_statement([other, mdna_copy, statement]) == statement


def test_no_candidate_is_an_error():
    with pytest.raises(StatementError, match="0 income statement candidates"):
        find_income_statement([table(section="mdna")])


def test_two_candidates_are_an_error_and_not_a_pick():
    with pytest.raises(StatementError, match="2 income statement candidates"):
        find_income_statement([table(table_id="a"), table(table_id="b")])


# --- The 15 pinned filings against the metric catalog ---


def pinned_filings():
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    found = {}
    for company in corpus.companies:
        for comparison in company.comparisons():
            for filing in (comparison.current, comparison.prior):
                found[filing.accession] = filing
    return sorted(found.values(), key=lambda f: (f.ticker, f.period.period_end))


def yoy_comparisons():
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    return [
        comparison
        for company in corpus.companies
        for comparison in company.comparisons()
        if comparison.comparison_type is not ComparisonType.SEQ_10Q
    ]


def label(filing) -> str:
    return f"{filing.ticker}-{filing.period.label}"


class Real:
    """What the real store gives: each filing's statement and the catalog's figures, read once."""

    def __init__(self) -> None:
        corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
        with FactStore(get_settings().duckdb_path, read_only=True) as store:
            catalog = MetricCatalog(store, corpus, MetricsConfig.load())
            self.statements = {}
            self.catalog = {}
            for filing in pinned_filings():
                tables = store.unparsed_tables(filing.accession)
                self.statements[filing.accession] = read_income_statement(
                    find_income_statement(tables)
                )
                for item in (REVENUE, GROSS_PROFIT):
                    dollars = catalog.line_item(filing, item)
                    assert dollars is not None, (filing.accession, item)
                    self.catalog[(filing.accession, item)] = dollars


@pytest.fixture(scope="module")
def real():
    return Real()


@pytest.mark.corpus
@pytest.mark.parametrize("filing", pinned_filings(), ids=label)
def test_real_revenue_and_gross_profit_match_the_filed_statement(real, filing):
    statement = real.statements[filing.accession]
    for item, figure in (
        ("revenue", statement.revenue(CURRENT)),
        ("gross_profit", statement.gross_profit(CURRENT)),
    ):
        catalog = real.catalog[(filing.accession, item)]
        assert figure.agrees_with(catalog), (item, figure.dollars, catalog)


@pytest.mark.corpus
@pytest.mark.parametrize(
    "comparison", yoy_comparisons(), ids=lambda c: f"{c.ticker}-{c.comparison_type.value}"
)
def test_real_year_earlier_columns_match_the_earlier_filings_catalog_values(real, comparison):
    # A restatement between a filing and the one a year before would show here.
    statement = real.statements[comparison.current.accession]
    for item, figure in (
        ("revenue", statement.revenue(YEAR_EARLIER)),
        ("gross_profit", statement.gross_profit(YEAR_EARLIER)),
    ):
        catalog = real.catalog[(comparison.prior.accession, item)]
        assert figure.agrees_with(catalog), (item, figure.dollars, catalog)


@pytest.mark.corpus
def test_real_check_would_see_a_wrong_column(real):
    # Every figure matches, so show the comparison is able to notice when one doesn't.
    seen = 0
    for filing in pinned_filings():
        statement = real.statements[filing.accession]
        revenue = real.catalog[(filing.accession, "revenue")]
        if len(statement.column_years) == 4:  # a Q2 10-Q: the six-month column is not the quarter
            assert not statement.revenue(2).agrees_with(revenue), filing.accession
            seen += 1
        # The year-earlier column is a different amount in every one of these filings.
        assert not statement.revenue(YEAR_EARLIER).agrees_with(revenue), filing.accession
    assert seen == 6


# Figures read by eye from the filed HTML on 2026-11-06 and confirmed with a separate regex over
# the HTML's table rows (no code from this project), for one filing per company. They test the
# reader itself: a match with the catalog would also hold if both read the same wrong column.
# Each tuple is the statement's columns in dollars.
GOLDEN = {
    # Hershey Q2 2026 10-Q: three months 2026, 2025, then six months 2026, 2025 (in thousands).
    "0001628280-26-050900": {
        "revenue": (2_787_306_000, 2_614_718_000, 5_891_473_000, 5_420_137_000),
        "cost_of_sales": (1_523_998_000, 1_818_445_000, 3_405_434_000, 3_679_597_000),
        "gross_profit": (1_263_308_000, 796_273_000, 2_486_039_000, 1_740_540_000),
    },
    # Mattel Q2 2026 10-Q, same layout (in thousands).
    "0001628280-26-054347": {
        "revenue": (1_125_326_000, 1_018_562_000, 1_987_497_000, 1_845_191_000),
        "cost_of_sales": (583_205_000, 499_611_000, 1_058_596_000, 918_144_000),
        "gross_profit": (542_121_000, 518_951_000, 928_901_000, 927_047_000),
    },
    # Stanley FY2025 10-K: 2025, 2024, 2023 (in millions). It has no gross profit line.
    "0000093556-26-000009": {
        "revenue": (15_130_400_000, 15_365_700_000, 15_781_100_000),
        "cost_of_sales": (10_542_100_000, 10_851_300_000, 11_848_500_000),
        "gross_profit": (4_588_300_000, 4_514_400_000, 3_932_600_000),
    },
}


@pytest.mark.corpus
@pytest.mark.parametrize("accession", sorted(GOLDEN))
def test_real_reader_gives_the_hand_confirmed_figures(real, accession):
    statement = real.statements[accession]
    for name, expected in GOLDEN[accession].items():
        method = getattr(statement, name)
        got = tuple(method(column).dollars for column in range(len(expected)))
        assert got == pytest.approx(expected), name
