"""Table decisions, including the shapes that real filings showed were hard."""

from __future__ import annotations

import lxml.html
import pytest

from sec_agent.ingest.tables import (
    RowKind,
    classify_row,
    is_data_table,
    is_prose_cell,
    strip_data_tables,
)

# Long prose cells (over 200 characters) of the kind a layout table holds. A cell this long is
# prose however it is punctuated, which is the rule these tests lean on.
LONG_RISK = "Our supply chain could be disrupted by events outside our control. " * 5
LONG_FOOTNOTE = (
    "(a) We believe that adjusted revenue provides management and investors with useful "
    "measures to evaluate the performance of the business, and it should not be considered "
    "a substitute for the most comparable measure reported under accepted principles. "
) * 2


def table(html: str):
    """Parse a snippet of HTML into an element."""
    return lxml.html.fromstring(html)


def rows(*row_cells: tuple[str, ...]) -> str:
    """A table built from tuples of cell texts, one tuple per row."""
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>" for cells in row_cells
    )
    return f"<table>{body}</table>"


def page(*parts: str) -> str:
    """Wrap pieces of HTML in a minimal document."""
    return f"<html><body>{''.join(parts)}</body></html>"


# --- Tables of figures must be removed ---


def test_financial_table_is_data():
    assert is_data_table(
        table(
            rows(
                ("Net sales", "$", "2,787.3", "2,614.7", "6.6%"),
                ("Cost of sales", "$", "1,524.0", "1,818.4", "(16.2)%"),
            )
        )
    )


def test_long_footnote_does_not_save_a_data_table():
    """GE: an 858-character footnote used to make a 146-cell table count as prose."""
    assert is_data_table(
        table(
            rows(
                ("Total revenue (GAAP)", "$", "13,349", "$", "11,022"),
                ("Less: other", "(12)", "(9)"),
                (LONG_FOOTNOTE,),
            )
        )
    )


# National Vision's balance sheet row: over 270 characters of label beside two numbers. Long
# labels are not prose when everything beside them is a number.
LONG_EQUITY_LABEL = (
    "Common stock, $0.01 par value; 200,000,000 shares authorized; 86,278,538 and 85,444,263 "
    "shares issued as of January 3, 2026 and December 28, 2024, respectively; 79,416,050 and "
    "78,775,117 shares outstanding as of January 3, 2026 and December 28, 2024, respectively"
)


def test_long_label_beside_only_numbers_is_a_data_row():
    """National Vision: this row's figures reached the text because its label is so long."""
    assert classify_row([LONG_EQUITY_LABEL, "862", "854"]) is RowKind.DATA


def test_long_label_row_does_not_reach_the_text():
    result = strip_data_tables(
        page(rows(("Total assets", "$", "1,000", "$", "900"), (LONG_EQUITY_LABEL, "862", "854")))
    )
    assert result.removed[0].preserved_rows == ()
    assert "86,278,538" not in result.html


# A long cell beside text or dashes (not numbers) is a real paragraph, even inside a table.
def test_long_text_beside_a_word_is_still_prose():
    assert classify_row([LONG_RISK, "see Note 4"]) is RowKind.PROSE


def test_long_text_beside_only_dashes_is_still_prose():
    assert classify_row([LONG_RISK, "—"]) is RowKind.PROSE


def test_table_of_mostly_dashes_is_data():
    # A dash is a value (zero) in a figures column, so a table of them is still figures.
    """Mattel: `$ — $ 22,031` has few digits but is plainly a table of figures."""
    assert is_data_table(
        table(
            rows(
                ("December 31, 2024", "Level 1", "Level 2", "Level 3", "Total"),
                ("(In thousands)",),
                ("Assets:",),
                ("Foreign currency forward contracts (a)", "$", "—", "$", "22,031", "$", "—"),
                ("Liabilities:",),
                ("Foreign currency forward contracts (a)", "$", "—", "$", "4,120", "$", "—"),
            )
        )
    )


def test_one_row_of_figures_under_headers_is_data():
    """Mattel's small derivative tables have several header rows and one row of values."""
    assert is_data_table(
        table(
            rows(
                ("Derivatives Not Designated as Hedging Instruments",),
                ("For the Year Ended",),
                ("December 31, 2025", "December 31, 2024"),
                ("Statements of Operations Classification",),
                ("Other selling expense", "1,204", "(3,110)"),
            )
        )
    )


def test_zero_width_padding_does_not_hide_a_data_table():
    """Tootsie Roll: cells of zero-width spaces made tables look mostly like text."""
    pad = "​"
    assert is_data_table(
        table(
            rows(
                (pad, pad, "2024", pad, pad, "2023", pad),
                ("U.S. statutory rate", pad, "21.0", "%", pad, "21.0", "%"),
                ("State income taxes, net", pad, "2.6", pad, pad, "3.7", pad),
            )
        )
    )


def test_values_with_footnote_markers_count_as_numbers():
    assert is_data_table(table(rows(("Revenue", "1,200 (a)", "1,100"), ("Costs", "900*", "850"))))


# --- Tables of text must be kept ---


def test_bullet_layout_table_is_kept():
    assert not is_data_table(table(rows(("•", LONG_RISK))))


def test_short_text_table_is_kept():
    assert not is_data_table(table(rows(("•", "rising cocoa prices;"))))


def test_numbered_list_layout_is_kept():
    """A list marker in the first column isn't a value."""
    assert not is_data_table(
        table(rows(("(1)", "supply disruption;"), ("(2)", "cost inflation;"), ("(3)", "tariffs.")))
    )


def test_prose_with_a_few_numbers_is_kept():
    assert not is_data_table(table(rows(("2025", "2024", LONG_RISK))))


def test_table_mostly_of_labels_with_one_number_is_kept():
    assert not is_data_table(table(rows(("Exhibit", "10.1", "Credit agreement"))))


def test_text_table_with_page_references_is_kept():
    """JPMorgan's derivative usage table has words in every row and a page number."""
    assert not is_data_table(
        table(
            rows(
                ("Type of Derivative", "Use", "Designation", "Segment", "Page"),
                ("Interest rate", "Hedge fixed rate", "Fair value hedge", "Corporate", "112"),
                ("Foreign exchange", "Hedge earnings", "Cash flow hedge", "Corporate", "113"),
            )
        )
    )


def test_officer_table_with_ages_is_kept():
    """Chevron lists officers beside their ages, and those biographies are text."""
    assert not is_data_table(
        table(
            rows(
                ("Name", "Age", "Current and Prior Positions"),
                ("A. Person", "65", "Chairman of the Board and Chief Executive Officer"),
                ("B. Person", "58", "Vice Chairman and Executive Vice President"),
            )
        )
    )


def test_exhibit_list_is_kept():
    assert not is_data_table(
        table(rows(("4.5", "Indenture, dated March 19, 2021", "8-K", "001-05647", "4.1")))
    )


def test_table_of_contents_is_kept_for_section_detection():
    # Item headings in a contents table are how the parser finds where items start, so the
    # table stays. It sits before the first item, so it can't reach a section's text.
    assert not is_data_table(
        table(
            rows(
                ("ITEM 1A", "Risk Factors", "10"),
                ("ITEM 7", "MD&amp;A", "27"),
                ("ITEM 8", "Financial Statements", "55"),
            )
        )
    )


def test_empty_table_is_not_data():
    assert not is_data_table(table(rows((" ",))))


# --- Row and cell rules ---


@pytest.mark.parametrize(
    ("cells", "kind"),
    [
        (["Net sales", "100", "200"], RowKind.DATA),
        (["Net sales", "$", "—"], RowKind.DATA),
        (["Revenue growth", "n/m"], RowKind.DATA),
        (["(In millions)"], RowKind.LABEL),
        (["Year ended December 31,"], RowKind.LABEL),
        (["(1)", "supply disruption"], RowKind.LABEL),
        (
            ["(a) This sentence explains how the figures above were calculated in detail."],
            RowKind.PROSE,
        ),
        ([LONG_RISK], RowKind.PROSE),
        (
            ["Less: long row label naming a line item without any sentence at the end"],
            RowKind.LABEL,
        ),
    ],
)
def test_row_kinds(cells, kind):
    # Data rows are dropped, prose rows are kept as text, label rows go with their table.
    assert classify_row(cells) is kind


def test_row_label_is_not_prose():
    assert not is_prose_cell("Effect of exchange rate changes on cash and due from banks")


def test_footnote_is_prose():
    assert is_prose_cell("(a) Includes provision for credit losses and other related charges.")


# --- Stripping ---


def test_strip_removes_data_tables_and_keeps_surrounding_text():
    result = strip_data_tables(
        page(
            "<p>Before.</p>",
            rows(("Sales", "100", "200")),
            "<p>After.</p>",
            rows(("•", LONG_RISK)),
        )
    )
    assert len(result.removed) == 1
    assert len(result.kept) == 1
    for text in ("Before.", "After.", "supply chain"):
        assert text in result.html
    assert "200" not in result.html


def test_strip_keeps_prose_rows_of_a_removed_table_in_place():
    """Footnotes explain the figures, and are the only part that retrieval can use."""
    footnote = "(a) Canada shale assets were sold in December 2024 as part of the program."
    result = strip_data_tables(
        page(
            "<p>Before.</p>",
            rows(("Production", "100", "200"), ("Reserves", "300", "400"), (footnote,)),
            "<p>After.</p>",
        )
    )
    assert result.removed[0].preserved_rows == (footnote,)
    assert "Production" not in result.html
    assert result.html.index("Before.") < result.html.index(footnote) < result.html.index("After.")


def test_strip_records_what_it_removed_so_it_can_be_reviewed():
    result = strip_data_tables(page(rows(("Sales", "100", "200"))))
    assert result.removed[0].rows == (("Sales", "100", "200"),)
    assert "Sales" in result.removed[0].excerpt


def test_strip_counts_tagged_values():
    result = strip_data_tables(
        page(
            "<table><tr><td>Sales</td>"
            '<td><ix:nonfraction name="a">100</ix:nonfraction></td>'
            '<td><ix:nonfraction name="a">200</ix:nonfraction></td></tr></table>'
        )
    )
    assert result.tables[0].tagged_values == 2


def test_strip_accepts_xml_declaration_used_by_real_filings():
    result = strip_data_tables(
        '<?xml version="1.0" encoding="ASCII"?><html><body><p>Café text.</p>'
        + rows(("Sales", "100", "200"))
        + "</body></html>"
    )
    assert len(result.removed) == 1
    assert "Café text." in result.html


def test_strip_counts_nested_data_table_once():
    result = strip_data_tables(
        page(
            "<table><tr><td>Sales</td><td>100</td><td>200</td></tr><tr><td>"
            "<table><tr><td>Nested</td><td>1</td><td>2</td></tr></table>"
            "</td></tr></table>"
        )
    )
    assert len(result.removed) == 1
    assert "Nested" not in result.html
