"""Table parsing, using the shapes that real filings and the hand audit showed were hard."""

from __future__ import annotations

from datetime import date

import pytest

from sec_agent.ingest.table_facts import (
    UNIT_PERCENT,
    UNIT_USD,
    FilingContext,
    ParsedTable,
    build_grid,
    disagrees_with_tag,
    parse_number,
    parse_tables,
)
from sec_agent.ingest.tables import parse_html_tree
from sec_agent.schemas import Duration, FormType

# The filings the tables belong to. The parser dates a bare-year column ("2025") from its
# filing's period end, so these set what a column's date should come out as.
K_2025 = FilingContext("acc", FormType.FORM_10K, 2025, "FY", date(2025, 12, 31))
Q2_2026 = FilingContext("acc", FormType.FORM_10Q, 2026, "Q2", date(2026, 6, 30))
Q1_2026 = FilingContext("acc", FormType.FORM_10Q, 2026, "Q1", date(2026, 3, 31))


# The helpers below build small HTML tables, so each test shows the layout it is about.
def cell(text: str, span: int = 1) -> str:
    """A table cell, optionally spanning several columns (filers split a $ from its value)."""
    attribute = f' colspan="{span}"' if span > 1 else ""
    return f"<td{attribute}>{text}</td>"


def row(*cells: str) -> str:
    """A table row of cells."""
    return f"<tr>{''.join(cells)}</tr>"


def filing(*tables_and_text: str) -> str:
    """A filing made of tables and paragraphs, in order. Text before a table is its context."""
    return f"<html><body>{''.join(tables_and_text)}</body></html>"


def table(*rows: str) -> str:
    """A table of rows."""
    return f"<table>{''.join(rows)}</table>"


def parse_one(html: str, context: FilingContext = K_2025, **kwargs: bool) -> ParsedTable:
    """Parse a filing that has exactly one data table, and return that table's result."""
    tables = parse_tables(html, context, **kwargs)
    assert len(tables) == 1
    return tables[0]


def fact(parsed: ParsedTable, label: str, column_part: str):
    """The one fact for a row label whose column heading contains `column_part`."""
    matches = [f for f in parsed.facts if f.row_label == label and column_part in f.column_label]
    assert len(matches) == 1, matches
    return matches[0]


def dollars_table(caption: str | None, header: str, *rows_: tuple[str, str]) -> str:
    """Caption, one period header spanning its $ and value, then rows plus a filler row.

    A table needs two figures to count as a data table at all.
    """
    body = [row(cell(label), cell("$"), cell(value)) for label, value in rows_]
    body.append(row(cell("Other items"), cell("$"), cell("5")))
    top = [row(cell(caption))] if caption else []
    return table(*top, row(cell(""), cell(header, 2)), *body)


# Hershey's segment table, as filed: two periods plus a percent-change column for each of three
# and six months, with "$" and "%" in cells of their own. Hard because the same row mixes
# dollars (Net sales) and percentages (Segment margin), and the change columns must be skipped.
HSY_SEGMENT = table(
    row(cell("", 3), cell("Three Months Ended", 9), cell("Six Months Ended", 9)),
    row(
        cell("", 3),
        cell("June 28, 2026", 3),
        cell("June 29, 2025", 3),
        cell("Percent Change", 3),
        cell("June 28, 2026", 3),
        cell("June 29, 2025", 3),
        cell("Percent Change", 3),
    ),
    row(cell("In millions of dollars", 3)),
    row(
        cell("Net sales", 3),
        cell("$"),
        cell("387.8"),
        cell(""),
        cell("$"),
        cell("315.5"),
        cell(""),
        cell("22.9", 2),
        cell("%"),
        cell("$"),
        cell("737.9"),
        cell(""),
        cell("$"),
        cell("600.0"),
        cell(""),
        cell("23.0", 2),
        cell("%"),
    ),
    row(
        cell("Segment margin", 3),
        cell("16.1", 2),
        cell("%"),
        cell("21.1", 2),
        cell("%"),
        cell("", 3),
        cell("13.1", 2),
        cell("%"),
        cell("18.3", 2),
        cell("%"),
    ),
)


def test_hershey_style_table_reads_values_periods_and_units():
    parsed = parse_one(filing(HSY_SEGMENT), Q2_2026)
    assert parsed.clean, parsed.reasons
    sales = fact(parsed, "Net sales", "Three Months Ended June 28")
    assert sales.value == pytest.approx(387_800_000)
    assert sales.unit == UNIT_USD
    assert sales.end_date == date(2026, 6, 28)
    assert sales.duration is Duration.THREE_MONTHS
    six_months = fact(parsed, "Net sales", "Six Months Ended June 29")
    assert six_months.value == pytest.approx(600_000_000)
    assert six_months.duration is Duration.YEAR_TO_DATE


def test_percent_in_the_same_column_makes_a_percentage():
    parsed = parse_one(filing(HSY_SEGMENT), Q2_2026)
    margin = fact(parsed, "Segment margin", "Three Months Ended June 28")
    assert margin.value == pytest.approx(16.1)
    assert margin.unit == UNIT_PERCENT


def test_percentage_column_without_a_percent_sign_is_a_percentage():
    """MKS's plan asset tables: an untagged table of this shape read as dollars."""
    parsed = parse_one(
        filing(
            table(
                row(cell("(Dollars in millions)")),
                row(cell(""), cell("December 31, 2025", 2)),
                row(cell(""), cell("$"), cell("Percentage")),
                row(cell("Equity securities"), cell("100"), cell("20")),
                row(cell("Debt securities"), cell("300"), cell("60")),
            )
        )
    )
    assert parsed.clean, parsed.reasons
    equity = fact(parsed, "Equity securities", "Percentage")
    assert equity.unit == UNIT_PERCENT
    assert equity.value == pytest.approx(20)


def test_percent_of_column_without_a_percent_sign_is_a_percentage():
    """JPMorgan: "Percent of retained loans" read as $12 million when this rule regressed."""
    parsed = parse_one(
        filing(
            table(
                row(cell("(Dollars in millions)")),
                row(cell(""), cell("2025 Allowance"), cell("2025 Percent of retained loans")),
                row(cell("Auto and other"), cell("$ 1,039"), cell("12")),
                row(cell("Credit card"), cell("$ 2,226"), cell("53")),
            )
        )
    )
    assert parsed.clean, parsed.reasons
    share = fact(parsed, "Auto and other", "Percent of retained loans")
    assert share.unit == UNIT_PERCENT
    assert share.value == pytest.approx(12)


def test_percentage_column_agrees_with_a_tag_stored_as_a_fraction():
    html = filing(
        XBRL_RESOURCES,
        table(
            row(cell("(Dollars in millions)")),
            row(cell(""), cell("December 31, 2025 Percentage")),
            row(cell("Equity securities"), cell(tagged("20", scale="-2"))),
            row(cell("Debt securities"), cell(tagged("60", scale="-2"))),
        ),
    )
    assert parse_one(html).clean


def test_percentage_change_column_is_still_skipped():
    parsed = parse_one(
        filing(
            table(
                row(cell("(Dollars in millions)")),
                row(cell(""), cell("2025", 2), cell("Percentage Change")),
                row(cell("Revenue"), cell("$"), cell("100"), cell("5")),
                row(cell("Costs"), cell("$"), cell("50"), cell("3")),
            )
        )
    )
    assert not [f for f in parsed.facts if "Change" in f.column_label]


def test_change_columns_are_not_facts():
    parsed = parse_one(filing(HSY_SEGMENT), Q2_2026)
    assert not [f for f in parsed.facts if "Percent Change" in f.column_label]


def test_percent_in_a_neighboring_column_does_not_change_the_unit():
    """Avnet: a rate column's % made the balance beside it a percentage."""
    parsed = parse_one(
        filing(
            table(
                row(cell(""), cell("Interest Rate", 2), cell("Carrying Balance", 2)),
                row(cell(""), cell("December 31, 2025", 2), cell("December 31, 2025", 2)),
                row(cell("(in thousands)")),
                row(cell("Notes due 2031"), cell("5.5"), cell("%"), cell("$"), cell("300,000")),
                row(cell("Notes due 2032"), cell("6.0"), cell("%"), cell("$"), cell("500,000")),
            )
        )
    )
    assert parsed.clean, parsed.reasons
    balance = fact(parsed, "Notes due 2031", "Carrying Balance")
    assert balance.unit == UNIT_USD
    assert balance.value == pytest.approx(300_000_000)


# --- Scale ---


def test_no_stated_scale_gives_no_facts_even_beside_a_dollar_sign():
    """Reading $13.8 billion as $13,789 is the worst error this module could make."""
    parsed = parse_one(filing(dollars_table(None, "2025", ("Expense", "13,789"))))
    assert not parsed.clean
    assert parsed.facts == ()
    assert parsed.needs_text_chunk


def test_scale_can_come_from_the_sentence_before_the_table():
    parsed = parse_one(
        filing(
            "<p>The following table shows expense for 2025 and 2024 (in millions):</p>",
            dollars_table(None, "2025", ("Expense", "13,789")),
        )
    )
    assert fact(parsed, "Expense", "2025").value == pytest.approx(13_789_000_000)


def test_first_scale_in_a_caption_governs():
    """Apple: "(In millions, except number of shares, which are reflected in thousands)"."""
    caption = "(In millions, except number of shares, which are reflected in thousands)"
    parsed = parse_one(filing(dollars_table(caption, "December 31, 2025", ("Term debt", "12,350"))))
    assert fact(parsed, "Term debt", "2025").value == pytest.approx(12_350_000_000)


# Every shorthand form a filer might use for a scale, and what it must multiply by. 000 means
# thousands, 000,000 and MM mean millions, and bn means billions.
@pytest.mark.parametrize(
    ("caption", "multiplier"),
    [
        ("($ in 000s)", 1_000),
        ("($000's)", 1_000),
        ("($000)", 1_000),
        ("(Dollars in 000s)", 1_000),
        ("(Dollars in 000's, except per share amounts)", 1_000),
        ("(Dollars in 000, except per share amounts)", 1_000),
        ("($ in 000,000s)", 1_000_000),
        ("($ in MM)", 1_000_000),
        ("($MM)", 1_000_000),
        ("($M)", 1_000_000),
        ("($ in bn)", 1_000_000_000),
    ],
)
def test_shorthand_dollar_scales_are_read(caption, multiplier):
    parsed = parse_one(filing(dollars_table(caption, "2025", ("Revenue", "100"))))
    assert parsed.clean, parsed.reasons
    assert fact(parsed, "Revenue", "2025").value == 100 * multiplier


@pytest.mark.parametrize("caption", ["(in 000s)", "(000s)", "(000's)", "(000)"])
def test_shorthand_in_the_table_outranks_the_page_banner(caption):
    """AT&T: 90,255 thousand subscribers became $90 billion under "Dollars in millions"."""
    subscribers = table(
        row(cell(caption), cell("2025")),
        row(cell("Postpaid"), cell("90,255")),
        row(cell("Prepaid"), cell("19,200")),
    )
    parsed = parse_one(filing("<p>Dollars in millions except per share amounts</p>", subscribers))
    assert not parsed.clean
    assert parsed.needs_text_chunk


def test_shorthand_on_one_column_does_not_scale_the_table():
    # "Shares (000s)" scales only the share column, so the table's dollars have no stated scale
    # and the banner above must not be trusted for them.
    parsed = parse_one(
        filing(
            "<p>Dollars in millions</p>",
            table(
                row(cell(""), cell("Shares (000s)"), cell("$"), cell("Amount")),
                row(cell("Repurchased"), cell("100"), cell("$"), cell("50")),
                row(cell("Issued"), cell("20"), cell("$"), cell("10")),
            ),
        )
    )
    assert not parsed.clean
    assert "a header scales one column" in parsed.reasons


@pytest.mark.parametrize(
    "text", ["Commission File No. 000-12345", "Sold 5 000 units.", "Revenue of $1,000.", "$2,000"]
)
def test_numbers_that_contain_000_are_not_a_scale(text):
    # "000" appears in file numbers and amounts. None of these states a scale, so the table
    # has no unit and goes to text.
    parsed = parse_one(filing(f"<p>{text}</p>", dollars_table(None, "2025", ("Revenue", "100"))))
    assert not parsed.clean
    assert any("no unit or scale" in reason for reason in parsed.reasons)


def test_shorthand_and_a_word_for_the_same_scale_are_one_scale():
    # "thousands" and "(in 000s)" agree, so this is not a contradiction.
    parsed = parse_one(
        filing(
            table(
                row(cell("(Dollars in thousands)")),
                row(cell("(in 000s)"), cell("2025", 2)),
                row(cell("Revenue"), cell("$"), cell("100")),
                row(cell("Other items"), cell("$"), cell("5")),
            )
        )
    )
    assert parsed.clean, parsed.reasons


def test_shorthand_and_a_different_word_are_two_scales():
    # "millions" against "(in 000s)": the table contradicts itself, so no figure is trusted.
    parsed = parse_one(
        filing(
            table(
                row(cell("(Dollars in millions)")),
                row(cell("(in 000s)"), cell("2025", 2)),
                row(cell("Revenue"), cell("$"), cell("100")),
                row(cell("Other items"), cell("$"), cell("5")),
            )
        )
    )
    assert "more than one scale in the table" in parsed.reasons


def test_two_scales_in_one_table_is_not_clean():
    """JPMorgan: a mid-table "(in billions)" made $23 billion of revenue $23 trillion."""
    parsed = parse_one(
        filing(
            table(
                row(cell("(in millions)")),
                row(cell(""), cell("2025", 2)),
                row(cell("Total net revenue"), cell("$"), cell("23,379")),
                row(cell("Client assets (in billions)")),
                row(cell("Liquidity"), cell("$"), cell("926")),
            )
        )
    )
    assert not parsed.clean
    assert "more than one scale in the table" in parsed.reasons


# --- Positive evidence for dollars ---


def test_caption_without_dollars_or_a_dollar_sign_is_not_clean():
    parsed = parse_one(
        filing(
            table(
                row(cell("(in millions)")),
                row(cell(""), cell("2025")),
                row(cell("A"), cell("7")),
                row(cell("B"), cell("8")),
            )
        )
    )
    assert not parsed.clean


@pytest.mark.parametrize(
    "label",
    [
        "Employees",
        "Headcount",
        "Stores open at year end",
        "Weighted-average remaining lease term",
        "Number of wells",
        "Refinery capacity",
    ],
)
def test_rows_that_name_a_non_dollar_quantity_are_not_clean(label):
    """JPMorgan's "Employees 94,563" became $94 billion in a table in millions."""
    parsed = parse_one(
        filing(
            dollars_table(
                "(Dollars in millions)", "2025", ("Net revenue", "100"), (label, "94,563")
            )
        )
    )
    assert not parsed.clean


@pytest.mark.parametrize(
    "label", ["Term debt", "Long-term debt", "Short-term investments", "Accounts receivable"]
)
def test_debt_terms_are_still_dollars(label):
    parsed = parse_one(filing(dollars_table("(Dollars in millions)", "2025", (label, "100"))))
    assert parsed.clean, parsed.reasons


def test_volumes_captioned_in_barrels_are_not_dollars():
    """Chevron: "Thousands of barrels per day" isn't money."""
    parsed = parse_one(
        filing(
            table(
                row(cell("Thousands of barrels per day")),
                row(cell(""), cell("2025")),
                row(cell("Africa"), cell("432")),
                row(cell("Asia"), cell("110")),
            )
        )
    )
    assert not parsed.clean


@pytest.mark.parametrize(
    "label",
    ["Net EPS", "Earnings per diluted share", "Weighted-average shares", "Common stock dividends"],
)
def test_share_rows_stay_as_text(label):
    caption = "(Dollars in millions, except per share)"
    parsed = parse_one(filing(dollars_table(caption, "2025", (label, "6.04"))))
    assert not parsed.clean


def test_text_beside_the_figures_is_not_clean():
    """Apple: "Term debt | 100 basis points | Increase in annual interest expense | $ 139"
    is a hypothetical cost. As a fact, it reads as $139 million of debt."""
    parsed = parse_one(
        filing(
            table(
                row(cell("(Dollars in millions)")),
                row(cell(""), cell("Impact"), cell("2025", 2)),
                row(
                    cell("Term debt"), cell("Increase in interest expense"), cell("$"), cell("139")
                ),
                row(cell("Investments"), cell("Decline in fair value"), cell("$"), cell("2,416")),
            )
        )
    )
    assert not parsed.clean
    assert "text beside the figures" in parsed.reasons[0]


def test_not_meaningful_marker_is_allowed_beside_figures():
    parsed = parse_one(
        filing(
            table(
                row(cell("(Dollars in millions)")),
                row(cell(""), cell("2025", 2), cell("% Change")),
                row(cell("Revenue"), cell("$"), cell("100"), cell("NM")),
                row(cell("Costs"), cell("$"), cell("50"), cell("NM")),
            )
        )
    )
    assert parsed.clean, parsed.reasons


def test_stray_precision_is_not_clean():
    """Chevron: EPS labeled only "- Basic" sits among whole millions."""
    parsed = parse_one(
        filing(
            dollars_table(
                "(Dollars in millions)",
                "2025",
                ("Net income", "100"),
                ("Interest", "300"),
                ("- Basic", "3.46"),
            )
        )
    )
    assert not parsed.clean


# --- Periods ---


def table_with_columns(*headers: str, label: str = "Revenue") -> str:
    """A one-row dollar table whose column headers are given, to test how headers are dated."""
    return filing(
        table(
            row(cell("(Dollars in millions)")),
            row(cell(""), *(cell(header) for header in headers)),
            row(cell(label), *(cell(f"{100 + n}") for n, _ in enumerate(headers))),
        )
    )


def test_bare_year_on_a_10k_has_an_end_date_but_no_duration():
    parsed = parse_one(table_with_columns("2025", "2024"))
    revenue = fact(parsed, "Revenue", "2024")
    assert revenue.end_date == date(2024, 12, 31)
    assert revenue.duration is None


def test_bare_year_on_a_10q_is_not_clean():
    assert not parse_one(table_with_columns("2026", "2025"), Q2_2026).clean


def test_date_without_a_flow_phrase_is_an_instant():
    parsed = parse_one(table_with_columns("December 31, 2025", "December 31, 2024"))
    assert fact(parsed, "Revenue", "2025").duration is Duration.INSTANT


def test_quarter_and_year_to_date_columns():
    """Stanley's headers read "Second Quarter" above "2026"."""
    parsed = parse_one(
        filing(
            table(
                row(cell("")),
                row(cell(""), cell("Second Quarter", 2), cell("Year-to-Date", 2)),
                row(
                    cell("(Millions of Dollars)"),
                    cell("2026"),
                    cell("2025"),
                    cell("2026"),
                    cell("2025"),
                ),
                row(cell("Net sales"), cell("1"), cell("2"), cell("3"), cell("4")),
            )
        ),
        Q2_2026,
    )
    assert parsed.clean, parsed.reasons
    assert fact(parsed, "Net sales", "Second Quarter 2025").duration is Duration.THREE_MONTHS
    assert fact(parsed, "Net sales", "Year-to-Date 2026").duration is Duration.YEAR_TO_DATE


def test_first_quarter_year_to_date_is_three_months():
    parsed = parse_one(
        filing(dollars_table("(Dollars in millions)", "Year-to-Date March 31, 2026", ("A", "1"))),
        Q1_2026,
    )
    assert parsed.facts[0].duration is Duration.THREE_MONTHS


def test_table_naming_two_durations_does_not_guess():
    """A caption saying "three and six months" can't date a column by itself."""
    parsed = parse_one(
        filing(
            "<p>Results for the three and six months ended June 30, 2026 (in millions of dollars):"
            "</p>",
            dollars_table(None, "June 30, 2026", ("Net sales", "1")),
        ),
        Q2_2026,
    )
    assert parsed.clean, parsed.reasons
    assert fact(parsed, "Net sales", "2026").duration is None


# Western Digital's fiscal year ends on the Friday nearest June 30, so its years end on varying
# dates (June 27, 2025 for fiscal 2025) instead of the 30th.
WDC_FILING = FilingContext("acc", FormType.FORM_10K, 2026, "FY", date(2026, 7, 3))


def annual_context(context_id: str, start: str, end: str) -> str:
    """An XBRL period definition, the filing's own record of a reporting period."""
    return (
        f'<xbrli:context id="{context_id}"><xbrli:period><xbrli:startdate>{start}'
        f"</xbrli:startdate><xbrli:enddate>{end}</xbrli:enddate></xbrli:period></xbrli:context>"
    )


def test_bare_year_uses_a_fiscal_year_end_over_a_closer_date():
    """Western Digital: a June 30 date in the filing beat its real June 27 year end."""
    contexts = (
        "<ix:header><ix:resources>"
        + annual_context("fy25", "2024-06-29", "2025-06-27")
        + '<xbrli:context id="other"><xbrli:period><xbrli:instant>2025-06-30</xbrli:instant>'
        "</xbrli:period></xbrli:context></ix:resources></ix:header>"
    )
    parsed = parse_one(
        filing(contexts, table_with_columns("2026", "2025").replace("<html><body>", "")),
        WDC_FILING,
    )
    assert fact(parsed, "Revenue", "2025").end_date == date(2025, 6, 27)


def test_year_end_is_not_snapped_to_a_segment_only_period():
    contexts = (
        '<ix:header><ix:resources><xbrli:context id="seg"><xbrli:entity><xbrli:segment>'
        '<xbrldi:explicitmember dimension="a:Axis">a:X</xbrldi:explicitmember></xbrli:segment>'
        "</xbrli:entity><xbrli:period><xbrli:startdate>2024-07-01</xbrli:startdate>"
        "<xbrli:enddate>2025-06-30</xbrli:enddate></xbrli:period></xbrli:context>"
        + annual_context("fy25", "2024-06-29", "2025-06-27")
        + "</ix:resources></ix:header>"
    )
    parsed = parse_one(
        filing(contexts, table_with_columns("2026", "2025").replace("<html><body>", "")),
        WDC_FILING,
    )
    assert fact(parsed, "Revenue", "2025").end_date == date(2025, 6, 27)


def test_quarter_column_snaps_to_a_quarter_end():
    filing_q = FilingContext("acc", FormType.FORM_10Q, 2026, "Q2", date(2026, 7, 4))
    contexts = (
        "<ix:header><ix:resources>"
        + annual_context("q", "2025-03-30", "2025-06-28")
        + "</ix:resources></ix:header>"
    )
    parsed = parse_one(
        filing(
            contexts,
            table(
                row(cell("(Dollars in millions)")),
                row(cell(""), cell("Second Quarter 2026"), cell("Second Quarter 2025")),
                row(cell("Net sales"), cell("1"), cell("2")),
            ),
        ),
        filing_q,
    )
    assert fact(parsed, "Net sales", "2025").end_date == date(2025, 6, 28)


def test_column_dated_after_the_filing_is_not_a_reported_period():
    """Hershey's contractual obligations run out to 2029. Those aren't results."""
    parsed = parse_one(table_with_columns("2025", "2029"))
    assert not parsed.clean


def test_second_header_inside_a_table_is_not_clean():
    """Apple and Mattel stack periods, which would misdate every later row."""
    parsed = parse_one(
        filing(
            table(
                row(cell("(Dollars in millions)")),
                row(cell(""), cell("December 31, 2025")),
                row(cell("Net sales"), cell("1")),
                row(cell("Other items"), cell("3")),
                row(cell("December 31, 2024")),
                row(cell("Net sales"), cell("2")),
            )
        )
    )
    assert not parsed.clean


def test_column_with_no_period_is_not_clean():
    parsed = parse_one(table_with_columns("Level 1", "Level 2"))
    assert not parsed.clean
    assert any("without a period" in reason for reason in parsed.reasons)


# --- Rows that report another date ---


@pytest.mark.parametrize(
    "label",
    ["Beginning balance", "Balance at January 1", "Balance at July 1", "Balance at June 30, 2024"],
)
def test_opening_balances_are_skipped_and_sent_to_text(label):
    """Flexsteel's "Balance at July 1" is the prior year-end, not the column's date."""
    parsed = parse_one(
        filing(
            table(
                row(cell("(Dollars in millions)")),
                row(cell(""), cell("December 31, 2025", 2)),
                row(cell(label), cell("$"), cell("777")),
                row(cell("Ending balance"), cell("$"), cell("800")),
            )
        )
    )
    assert parsed.clean, parsed.reasons
    assert [f.row_label for f in parsed.facts] == ["Ending balance"]
    assert parsed.skipped_rows == 1
    assert parsed.needs_text_chunk


# --- Reading cells ---


def table_element(html: str):
    """The first <table> element of a filing, for tests of the grid builder."""
    return next(parse_html_tree(filing(html)).iter("table"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,234", 1234.0),
        ("(1,234)", -1234.0),
        ("(18,795", -18795.0),
        ("-26", -26.0),
        ("$ 5.5", 5.5),
        ("—", 0.0),
        ("$ -", 0.0),
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_zero_width_padding_is_not_a_cell():
    grid = build_grid(table_element(table(row(cell("\u200b"), cell("A")))))
    assert [c.text for c in grid[0]] == ["A"]


def test_colspans_place_cells_by_position():
    grid = build_grid(table_element(table(row(cell("A", 3), cell("B")))))
    assert [(c.start, c.end) for c in grid[0]] == [(0, 3), (3, 4)]


def test_rowspan_holds_its_columns_in_the_next_row():
    html = "<table><tr><td rowspan='2'>A</td><td>B</td></tr><tr><td>C</td></tr></table>"
    grid = build_grid(table_element(html))
    assert grid[1][0].start == 1


def test_text_tables_are_not_parsed():
    assert parse_tables(filing(table(row(cell("•"), cell("A risk.")))), K_2025) == []


def test_unclean_table_keeps_its_text_for_the_chunker():
    parsed = parse_one(table_with_columns("Level 1", "Level 2"))
    assert "Revenue | 100 | 101" in parsed.as_text


# --- XBRL safeguard ---

# A hidden XBRL header giving a full-year 2025 period (c1), which tagged cells refer to.
XBRL_RESOURCES = (
    '<ix:header><ix:resources><xbrli:context id="c1"><xbrli:period>'
    "<xbrli:startdate>2025-01-01</xbrli:startdate><xbrli:enddate>2025-12-31</xbrli:enddate>"
    "</xbrli:period></xbrli:context></ix:resources></ix:header>"
)


def tagged(value: str, *, scale: str = "6", context: str = "c1") -> str:
    """A number with an inline XBRL tag. scale="6" means the tag is in millions."""
    return (
        f'<ix:nonfraction name="us-gaap:Revenues" contextref="{context}" scale="{scale}">'
        f"{value}</ix:nonfraction>"
    )


def test_parser_records_a_tag_without_using_it():
    parsed = parse_one(
        filing(
            XBRL_RESOURCES,
            table(
                row(cell("(Dollars in millions)")),
                row(cell(""), cell("December 31, 2025")),
                row(cell("Revenue"), cell(tagged("100"))),
                row(cell("Other figure"), cell("5")),
            ),
        )
    )
    revenue = fact(parsed, "Revenue", "2025")
    assert revenue.tagged
    assert revenue.xbrl_concept == "us-gaap:Revenues"
    assert revenue.xbrl_value == pytest.approx(100_000_000)
    assert disagrees_with_tag(revenue) is None


def test_table_contradicting_its_tags_is_not_clean():
    """Every figure in it is suspect, including the untagged ones."""
    html = filing(
        XBRL_RESOURCES,
        table(
            row(cell("(Dollars in millions)")),
            row(cell(""), cell("December 31, 2025")),
            row(cell("Revenue"), cell(tagged("100", scale="3"))),
            row(cell("Other figure"), cell("5")),
        ),
    )
    assert parse_one(html, verify=False).clean
    verified = parse_one(html)
    assert not verified.clean
    assert verified.facts == ()
    assert "disagrees with XBRL" in verified.reasons[0]


def test_sign_differences_are_not_contradictions():
    """A cost shown in parentheses can be a positive value in XBRL."""
    parsed = parse_one(
        filing(
            XBRL_RESOURCES,
            table(
                row(cell("(Dollars in millions)")),
                row(cell(""), cell("December 31, 2025")),
                row(cell("Cost"), cell(f"({tagged('100')})")),
                row(cell("Other figure"), cell("5")),
            ),
        )
    )
    assert parsed.clean, parsed.reasons
    cost = fact(parsed, "Cost", "2025")
    assert cost.value < 0 < cost.xbrl_value


def test_percentage_tagged_as_a_fraction_agrees():
    html = filing(
        XBRL_RESOURCES,
        table(
            row(cell("(Dollars in millions)")),
            row(cell(""), cell("December 31, 2025", 2)),
            row(cell("Share of sales"), cell(tagged("65", scale="-2")), cell("%")),
            row(cell("Other figure"), cell("5")),
        ),
    )
    assert parse_one(html).clean
