"""The fallback parser on synthetic filings built in each layout real filers use."""

from __future__ import annotations

from sec_agent.ingest.fallback import (
    LAYOUT_HEADINGS,
    LAYOUT_INDEX,
    LAYOUT_POINTERS,
    extract_blocks,
    locate_items,
    parse_fallback,
    parse_page_ranges,
    split_pages,
)
from sec_agent.ingest.tables import parse_html_tree
from sec_agent.schemas import FormType

K = FormType.FORM_10K
Q = FormType.FORM_10Q


def page(*parts: str) -> str:
    """Wrap pieces of HTML in a minimal document."""
    return f"<html><body>{''.join(parts)}</body></html>"


def blocks_of(*parts: str) -> list[str]:
    """The visible text blocks the fallback sees in the given HTML."""
    return extract_blocks(parse_html_tree(page(*parts)))


def p(text: str) -> str:
    """A paragraph element."""
    return f"<p>{text}</p>"


def filler(label: str, count: int = 8) -> str:
    """Ordinary sentences starting with `label`, so tests can tell which part ended up where."""
    sentence = f"{label} text fills out this part of the filing with ordinary sentences."
    return "".join(p(sentence) for _ in range(count))


# --- Blocks ---


def test_hidden_xbrl_header_is_not_visible_text():
    assert blocks_of("<ix:header><p>Hidden</p></ix:header>", p("Seen.")) == ["Seen."]


def test_display_none_is_not_visible_text():
    assert blocks_of('<div style="display: none"><p>Hidden</p></div>', p("Seen.")) == ["Seen."]


def test_table_row_is_one_block():
    assert blocks_of("<table><tr><td>Item 7.</td><td>MD&amp;A</td></tr></table>") == [
        "Item 7. MD&A"
    ]


def test_row_wrapping_a_nested_table_is_split():
    html = "<table><tr><td><table><tr><td>Inner</td></tr></table></td></tr></table>"
    assert blocks_of(html) == ["Inner"]


def test_line_breaks_separate_blocks():
    assert blocks_of("<div>First line.<br>Second line.</div>") == ["First line.", "Second line."]


def test_text_beside_a_nested_block_keeps_its_order():
    assert blocks_of("<div>Before<div>Inside</div>After</div>") == ["Before", "Inside", "After"]


def test_zero_width_padding_is_removed():
    assert blocks_of(p("Sales​ rose."), p("​")) == ["Sales rose."]


# --- Pages ---


def test_page_ranges_keep_separate_runs():
    assert parse_page_ranges("4-7, 9, 71-73") == [[4, 5, 6, 7], [9], [71, 72, 73]]


def test_page_ranges_ignore_footnote_markers_and_words():
    assert parse_page_ranges("74, (a)") == [[74]]
    assert parse_page_ranges("Not applicable") == []


def test_pages_split_on_footers_in_either_position():
    # Footers read "2025 FORM 10-K 3" or "4 2025 FORM 10-K". Five or more matching footers make
    # a pattern, so the first and last of the eight here still count as footers.
    footer = "2025 FORM 10-K"
    pages = split_pages(
        [
            "First.",
            f"{footer} 3",
            "Second.",
            f"4 {footer}",
            "Third.",
            f"{footer} 5",
            f"6 {footer}",
            f"{footer} 7",
        ]
    )
    assert pages[3] == ["First."]
    assert pages[4] == ["Second."]
    assert pages[5] == ["Third."]


def test_no_repeating_footer_means_no_pages():
    assert split_pages(["One.", "Two."]) == {}


# --- Cross-reference index layout (GE) ---


def footer(number: int) -> str:
    """A page footer, which ends the page numbered `number`."""
    return p(f"2025 FORM 10-K {number}")


# GE-style cross-reference index: topics grouped by page, not by item. Page ranges like
# "1, 4" show one item can span separate page runs; "(b)" is a footnote marker instead of pages.
INDEX_10K = (
    "<table>"
    "<tr><td>FORM 10-K CROSS REFERENCE INDEX</td><td>Page(s)</td></tr>"
    "<tr><td>Part I</td></tr>"
    "<tr><td>Item 1.</td><td>Business</td><td>1, 4</td></tr>"
    "<tr><td>Item 1A.</td><td>Risk Factors</td><td>2-3</td></tr>"
    "<tr><td>Item 1B.</td><td>Unresolved Staff Comments</td><td>Not applicable</td></tr>"
    "<tr><td>Part II</td></tr>"
    "<tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>4</td></tr>"
    "<tr><td>Item 8.</td><td>Financial Statements</td><td>5</td></tr>"
    "<tr><td>Part III</td></tr>"
    "<tr><td>Item 11.</td><td>Executive Compensation</td><td>(b)</td></tr>"
    "</table>"
)


def index_filing() -> str:
    """A filing with no item headings; pages come from footers and topics from the index."""
    return page(
        p("About the company."),
        filler("Business"),
        footer(1),
        p("RISK FACTORS. The following risks apply."),
        footer(2),
        p("More risk text."),
        p("LEGAL PROCEEDINGS. See the notes."),
        footer(3),
        p("Segment detail for the business."),
        p("MANAGEMENT'S DISCUSSION AND ANALYSIS. Sales rose."),
        footer(4),
        p("NOTE 1. BASIS OF PRESENTATION."),
        footer(5),
        INDEX_10K,
        p("(b) Incorporated by reference to the proxy statement."),
        footer(6),
    )


def test_index_layout_reads_items_by_page():
    result = parse_fallback(index_filing(), K)
    assert result.layout == LAYOUT_INDEX
    assert "BASIS OF PRESENTATION" in result.rendered["part_ii_item_8"]
    assert "Sales rose" not in result.rendered["part_ii_item_8"]


def test_index_layout_joins_separate_page_ranges():
    business = parse_fallback(index_filing(), K).rendered["part_i_item_1"]
    assert "About the company" in business
    assert "Segment detail" in business


def test_index_layout_cuts_a_section_starting_midway_through_its_first_page():
    mdna = parse_fallback(index_filing(), K).rendered["part_ii_item_7"]
    assert "Sales rose" in mdna
    assert "Segment detail" not in mdna


def test_sentence_starting_with_a_title_word_is_not_a_heading():
    """Page 1 starts with "About the company", then says "Business text..."."""
    assert "About the company" in parse_fallback(index_filing(), K).rendered["part_i_item_1"]


def test_index_layout_cuts_a_section_ending_midway_through_its_last_page():
    """GE's Risk Factors share a page with Legal Proceedings."""
    index = INDEX_10K.replace(
        "<tr><td>Part II</td></tr>",
        "<tr><td>Item 3.</td><td>Legal Proceedings</td><td>3</td></tr><tr><td>Part II</td></tr>",
    )
    html = index_filing().replace(INDEX_10K, index)
    risk = parse_fallback(html, K).rendered["part_i_item_1a"]
    assert "More risk text" in risk
    assert "LEGAL PROCEEDINGS" not in risk


def test_index_layout_item_with_no_pages_keeps_what_the_index_says():
    assert "Not applicable" in parse_fallback(index_filing(), K).rendered["part_i_item_1b"]


def test_index_layout_resolves_footnoted_items():
    """GE's 10-Q lists risk factors with a footnote pointing to the 10-K."""
    assert "proxy statement" in parse_fallback(index_filing(), K).rendered["part_iii_item_11"]


def test_index_layout_tracks_parts_on_a_10q():
    html = page(
        filler("Notes"),
        p("2026 2Q FORM 10-Q 1"),
        filler("Risks"),
        p("2026 2Q FORM 10-Q 2"),
        p("2026 2Q FORM 10-Q 3"),
        p("2026 2Q FORM 10-Q 4"),
        p("2026 2Q FORM 10-Q 5"),
        "<table><tr><td>FORM 10-Q CROSS REFERENCE INDEX</td><td>Page(s)</td></tr>"
        "<tr><td>Part I</td></tr><tr><td>Item 1.</td><td>Financial Statements</td><td>1</td></tr>"
        "<tr><td>Item 2.</td><td>MD&amp;A</td><td>1</td></tr>"
        "<tr><td>Part II</td></tr><tr><td>Item 1.</td><td>Legal Proceedings</td><td>2</td></tr>"
        "<tr><td>Item 1A.</td><td>Risk Factors</td><td>2</td></tr></table>",
    )
    rendered = parse_fallback(html, Q).rendered
    assert "Notes text" in rendered["part_i_item_1"]
    assert "Risks text" in rendered["part_ii_item_1a"]


# --- Item headings ---


def items_10k(*items: tuple[str, str]) -> list[str]:
    """Blocks for each (heading, label): the heading, then 20 body lines starting with the label."""
    blocks = []
    for heading, label in items:
        blocks += [heading, *[f"{label} body text with enough length to count." for _ in range(20)]]
    return blocks


def test_items_are_found_in_order():
    blocks = items_10k(
        ("Item 1. Business", "B"), ("Item 1A. Risk Factors", "R"), ("Item 7. MD&A", "M")
    )
    spans = locate_items(blocks, K)
    assert list(spans) == ["part_i_item_1", "part_i_item_1a", "part_ii_item_7"]
    assert blocks[spans["part_i_item_1a"][0]] == "Item 1A. Risk Factors"


def test_table_of_contents_with_page_numbers_is_skipped():
    # The contents have 40 note lines under Item 1, enough text to look like a real section,
    # so only the trailing page numbers identify it.
    """Hershey's contents list its notes under Item 1, which looks like body text."""
    contents = ["Item 1. Financial Statements 2", *[f"Note {n} detail 5" for n in range(40)]]
    contents += ["Item 2. MD&A 36", "Item 3. Market Risk 47"]
    body = items_10k(("Item 1. Financial Statements", "F"), ("Item 2. MD&A", "M"))
    blocks = contents + body
    spans = locate_items(blocks, Q)
    assert spans["part_i_item_1"][0] == len(contents)


def test_contents_without_page_numbers_are_skipped_by_length():
    # No page numbers to go on: the contents' items have almost no text between them.
    contents = ["Item 1.", "Business", "Item 1A.", "Risk Factors"]
    blocks = contents + items_10k(("Item 1. Business", "B"), ("Item 1A. Risk Factors", "R"))
    assert locate_items(blocks, K)["part_i_item_1"][0] == len(contents)


def test_stray_item_reference_does_not_derail_later_items():
    # "Item 8 of this report." appears early and out of order. Taking the longest run of
    # headings in filing order ignores it.
    blocks = items_10k(("Item 1. Business", "B"))
    blocks += ["Item 8 of this report."]
    blocks += items_10k(
        ("Item 1A. Risk Factors", "R"), ("Item 7. MD&A", "M"), ("Item 8. Statements", "S")
    )
    spans = locate_items(blocks, K)
    assert "part_i_item_1a" in spans
    assert "part_ii_item_7" in spans
    assert blocks[spans["part_ii_item_8"][0]] == "Item 8. Statements"


def test_repeated_continued_heading_does_not_move_an_item():
    blocks = items_10k(
        ("Item 7. MD&A", "M"), ("Item 7. MD&A (continued)", "M"), ("Item 8. Statements", "S")
    )
    assert locate_items(blocks, K)["part_ii_item_7"][0] == 0


def test_10q_item_numbers_repeat_across_parts():
    blocks = items_10k(
        ("Item 1. Financial Statements", "F"),
        ("Item 2. MD&A", "M"),
        ("Item 1. Legal Proceedings", "L"),
        ("Item 1A. Risk Factors", "R"),
    )
    spans = locate_items(blocks, Q)
    assert blocks[spans["part_i_item_1"][0]] == "Item 1. Financial Statements"
    assert blocks[spans["part_ii_item_1"][0]] == "Item 1. Legal Proceedings"
    assert blocks[spans["part_ii_item_1a"][0]] == "Item 1A. Risk Factors"


def test_filing_with_no_items_gives_nothing():
    """Validation then rejects it: the fallback must not invent sections."""
    result = parse_fallback(page(p("A letter to shareholders.")), K)
    assert result.rendered == {}
    assert result.layout == LAYOUT_HEADINGS


# --- Pointer layout (JPMorgan, Chevron) ---


# Part of the MD&A heading, split out so a test can add a banner after it.
RESULTS = " and Results of Operations"


def pointer_filing(banner: str = "", report: str = "Report of Independent Registered") -> str:
    """Items near the front only point to a financial section later in the filing."""
    return page(
        p("Item 1. Business"),
        filler("Business"),
        p("Item 7. Management's Discussion and Analysis"),
        p("The index to this discussion is presented on page FS-1."),
        p("Item 8. Financial Statements and Supplementary Data"),
        p("The index to the statements is presented on page FS-1."),
        p("Item 14. Principal Accountant Fees and Services"),
        p("The information is in the proxy statement."),
        p(f"Management's Discussion and Analysis of Financial Condition{RESULTS}{banner}"),
        filler("Discussion"),
        p(report),
        p("We have audited the financial statements."),
        p(f"Notes to the Consolidated Financial Statements{banner}"),
        p("Note 1"),
        filler("Policy"),
        p("Supplemental Information on Oil and Gas"),
        p("Proved reserves are described here in this section."),
        p("Item 15. Exhibits"),
        filler("Exhibit"),
    )


def test_pointer_items_resolve_to_the_financial_section():
    result = parse_fallback(pointer_filing(), K)
    assert result.layout == LAYOUT_POINTERS
    assert "Discussion text" in result.rendered["part_ii_item_7"]
    assert "Policy text" in result.rendered["part_ii_item_8"]


def test_mdna_stops_where_the_statements_begin():
    mdna = parse_fallback(pointer_filing(), K).rendered["part_ii_item_7"]
    assert "We have audited" not in mdna
    assert "Policy text" not in mdna


def test_item_8_includes_reports_and_supplementary_data_but_not_exhibits():
    """Item 8 is "Financial Statements and Supplementary Data"."""
    notes = parse_fallback(pointer_filing(), K).rendered["part_ii_item_8"]
    assert "We have audited" in notes
    assert "Proved reserves" in notes
    assert "Exhibit text" not in notes


def test_item_before_the_financial_section_no_longer_contains_it():
    """Chevron: its statements sit inside the span of Item 14."""
    item_14 = parse_fallback(pointer_filing(), K).rendered["part_iii_item_14"]
    assert "proxy statement" in item_14
    assert "Discussion text" not in item_14


def test_page_banner_with_a_link_label_is_recognized():
    """Chevron's headers read "...Results of Operations Financial Table of Contents"."""
    html = pointer_filing(" Financial Table of Contents")
    assert "Discussion text" in parse_fallback(html, K).rendered["part_ii_item_7"]


def test_items_keep_their_own_text_when_no_pointers_exist():
    html = page(
        p("Item 7. Management's Discussion and Analysis"),
        filler("Discussion", 40),
        p("Item 8. Financial Statements"),
        filler("Statement", 40),
    )
    result = parse_fallback(html, K)
    assert result.layout == LAYOUT_HEADINGS
    assert "Discussion text" in result.rendered["part_ii_item_7"]


def test_running_headers_and_page_footers_are_dropped():
    """JPMorgan repeats "Management's discussion and analysis" and numbers its footers."""
    pieces = []
    for number in range(12):
        pieces += [
            p("Management's discussion and analysis"),
            filler("Discussion", 2),
            p(f"JPMorgan Chase/2025 Form 10-K {number + 1}"),
        ]
    html = page(
        p("Item 1. Business"),
        filler("Business"),
        p("Item 7. Management's Discussion and Analysis"),
        p("Refer to the section below."),
        p("Item 8. Financial Statements"),
        p("Refer to the section below."),
        *pieces,
        p("Report of Independent Registered"),
        p("Notes to consolidated financial statements"),
        filler("Policy", 2),
    )
    mdna = parse_fallback(html, K).rendered["part_ii_item_7"]
    assert "Discussion text" in mdna
    assert "Form 10-K 3" not in mdna
    assert "\n\nManagement's discussion and analysis\n\n" not in mdna
