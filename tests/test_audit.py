"""Each audit check is shown catching the failure it exists for, and staying quiet otherwise."""

from __future__ import annotations

import pytest

from sec_agent.ingest import tables as tables_module
from sec_agent.ingest.audit import (
    audit_tables,
    find_leftover_tables,
    find_lost_sentences,
    find_numeric_paragraphs,
    is_numeric_paragraph,
    note_numbering_problems,
    section_ending_problems,
    section_problems,
    sentence_coverage,
)
from sec_agent.ingest.sections import ParsedSection, SectionStatus, clean_paragraphs
from sec_agent.ingest.tables import strip_data_tables
from sec_agent.schemas import FormType, Section

# A prose sentence of 24 words ending in a period. It clears the audit's 15-word floor, so
# losing it is something the audit must notice.
SENTENCE = (
    "The company recorded a charge in the quarter because a major supplier closed two "
    "plants and moved its remaining production to another region."
)


def sentence(n: int) -> str:
    """A distinct 24-word sentence per number, so each one is found (or missed) on its own."""
    return (
        f"Paragraph number {n} says the company reviewed its operations during the period "
        "and concluded that several results were in line with its own expectations."
    )


def page(*parts: str) -> str:
    """Wrap pieces of HTML in a minimal document."""
    return f"<html><body>{''.join(parts)}</body></html>"


def data_table(*extra: str) -> str:
    """A two-row table of figures, which the classifier removes, plus optional one-cell rows."""
    extras = "".join(f"<tr><td>{text}</td></tr>" for text in extra)
    return (
        "<table><tr><td>Sales</td><td>100</td><td>200</td></tr>"
        f"<tr><td>Costs</td><td>50</td><td>60</td></tr>{extras}</table>"
    )


def section(
    kind: Section, text: str, *, note: str | None = None, item: str = "item"
) -> ParsedSection:
    """Build a CONTENT section from text, splitting paragraphs on blank lines."""
    paragraphs = clean_paragraphs(text, f"a:{kind.value}")
    return ParsedSection(
        section=kind,
        status=SectionStatus.CONTENT,
        source_item=item,
        paragraphs=paragraphs,
        note=note,
    )


# --- Wrongly removed tables ---


def test_lost_sentence_is_detected_when_a_table_is_dropped_whole():
    original = page(data_table(SENTENCE))
    lost = find_lost_sentences(original, page("<p>Nothing left.</p>"))
    assert len(lost) == 1
    assert lost[0].startswith("The company recorded")


def test_audit_catches_a_classifier_that_stops_preserving_prose(monkeypatch):
    """Break the classifier on purpose. The audit doesn't share its rules, so it notices."""
    original = page(data_table(SENTENCE))
    monkeypatch.setattr(tables_module, "is_prose_cell", lambda text: False)
    broken = strip_data_tables(original)
    assert broken.removed[0].preserved_rows == ()
    assert audit_tables(original, broken, {}).lost_sentences


def test_audit_catches_a_classifier_that_removes_a_prose_table(monkeypatch):
    original = page(f"<table><tr><td>•</td><td>{SENTENCE}</td></tr></table>")
    monkeypatch.setattr(tables_module, "is_data_table", lambda table: True)
    monkeypatch.setattr(tables_module, "is_prose_cell", lambda text: False)
    broken = strip_data_tables(original)
    assert audit_tables(original, broken, {}).lost_sentences


def test_preserved_footnote_is_not_reported_as_lost():
    original = page(data_table(SENTENCE))
    stripped = strip_data_tables(original)
    assert find_lost_sentences(original, stripped.html) == ()


def test_long_row_labels_are_not_sentences():
    """A label dropped with its row of figures isn't lost prose."""
    label = (
        "Tax effect on gains and losses on retained and sold ownership interests and other "
        "equity securities in the period (a)(c)"
    )
    original = page(f"<table><tr><td>{label}</td><td>100</td><td>200</td></tr></table>")
    assert find_lost_sentences(original, strip_data_tables(original).html) == ()


def test_lost_sentence_check_ignores_spacing_and_punctuation_changes():
    original = page(f"<table><tr><td>{SENTENCE}</td></tr></table>")
    assert find_lost_sentences(original, page(f"<p>{SENTENCE.replace(' ', '  ')}</p>")) == ()


# --- Leftover data tables ---


def tagged_row(label: str, *values: str) -> str:
    """A table row whose numbers carry inline XBRL tags, the audit's independent evidence."""
    cells = "".join(f'<td><ix:nonfraction name="a">{v}</ix:nonfraction></td>' for v in values)
    return f"<tr><td>{label}</td>{cells}</tr>"


def test_leftover_is_detected_when_a_tagged_data_table_is_kept(monkeypatch):
    html = page(f"<table>{tagged_row('Sales', '1', '2')}{tagged_row('Costs', '3', '4')}</table>")
    monkeypatch.setattr(tables_module, "is_data_table", lambda table: False)
    assert find_leftover_tables(strip_data_tables(html))


def test_text_table_with_a_few_tagged_values_is_not_a_leftover():
    """Trading plan tables name officers and tag share counts. They're text, and are kept."""
    row = (
        "<tr><td>A Person</td><td>Chief Risk Officer</td><td>Nov 12, 2025</td><td>through 2026</td>"
        '<td><ix:nonfraction name="a">5000</ix:nonfraction></td></tr>'
    )
    html = page(f"<table>{row * 4}</table>")
    assert strip_data_tables(html).kept
    assert find_leftover_tables(strip_data_tables(html)) == ()


def test_figures_rendered_into_a_paragraph_are_detected():
    leftover = (
        "2024 2023 2022 Current: Federal $ 24,804 $ 21,710 $ 13,070 Foreign 2,676 3,775 "
        "4,110 State 3,531 3,738 2,605"
    )
    parsed = {Section.MDNA: section(Section.MDNA, leftover)}
    assert find_numeric_paragraphs(parsed)


@pytest.mark.parametrize(
    "text",
    [
        "Gross margin increased 14.8 points to 45.3% in the quarter, from 30.5% a year ago.",
        "•iOS 26, macOS Tahoe 26, iPadOS 26, watchOS 26, visionOS 26 and tvOS 26",
        "Total as sold volumes were 2,872, 2,768 and 2,609 for 2025, 2024 and 2023, respectively.",
    ],
)
def test_ordinary_sentences_with_numbers_are_not_flagged(text):
    assert not is_numeric_paragraph(text)


# --- Sentence coverage ---


def body(count: int) -> list[str]:
    """`count` distinct sentences. With 20, coverage moves in steps of 5%, so results are exact."""
    return [sentence(n) for n in range(count)]


def test_coverage_is_full_when_every_sentence_reached_an_item():
    sentences = body(20)
    html = page(*(f"<p>{s}</p>" for s in sentences))
    assert sentence_coverage(html, {"item": " ".join(sentences)}) == 1.0


def test_cover_page_text_before_the_first_item_does_not_count():
    sentences = body(20)
    cover = (
        "Indicate by check mark whether the registrant has filed all reports required to be "
        "filed by the applicable sections of the act during the preceding twelve months."
    )
    html = page(f"<p>{cover}</p>", *(f"<p>{s}</p>" for s in sentences))
    assert sentence_coverage(html, {"item": " ".join(sentences)}) == 1.0


def test_gap_in_the_middle_lowers_coverage():
    sentences = body(20)
    html = page(*(f"<p>{s}</p>" for s in sentences))
    assert sentence_coverage(html, {"item": " ".join(sentences[:5] + sentences[15:])}) == 0.5


def test_lost_ending_lowers_coverage():
    sentences = body(20)
    html = page(*(f"<p>{s}</p>" for s in sentences))
    assert sentence_coverage(html, {"item": " ".join(sentences[:10])}) == 0.5


def test_no_output_gives_zero_coverage():
    assert sentence_coverage(page(f"<p>{sentence(1)}</p>"), {}) == 0.0


# --- Sections holding the wrong text ---

# A short MD&A-like text. It contains the terms MD&A must mention ("results of operations",
# "liquidity"), and stays under the 1,200-character limit that exempts short sections from
# the title check.
MDNA_TEXT = "Results of operations and liquidity were reviewed.\n\n" + "Sales rose.\n\n" * 30


def test_section_that_does_not_open_with_its_title_is_flagged():
    """GE: edgartools put text about trade controls at the start of MD&A."""
    text = MDNA_TEXT + "Sales rose again in this quarter.\n\n" * 50
    parsed = {Section.MDNA: section(Section.MDNA, text)}
    problems = section_problems(
        FormType.FORM_10Q, parsed, {"item": "International Trade Controls. We are subject."}
    )
    assert any("does not open with its title" in p for p in problems)


@pytest.mark.parametrize(
    "text",
    [
        "Not applicable. A smaller reporting company is not required to provide this.",
        "The Company is not required to provide the information called for by this item.",
    ],
)
def test_short_section_without_a_heading_is_accepted(text):
    """Cosmos Health: "Not applicable" with no heading rejected a correct parse."""
    parsed = {Section.MARKET_RISK: section(Section.MARKET_RISK, text)}
    assert section_problems(FormType.FORM_10Q, parsed, {"item": text}) == []


def test_long_section_without_a_heading_is_still_flagged():
    text = "An ordinary sentence about the business of the company in this period.\n\n" * 30
    parsed = {Section.MARKET_RISK: section(Section.MARKET_RISK, text)}
    problems = section_problems(FormType.FORM_10Q, parsed, {"item": text})
    assert any("does not open with its title" in problem for problem in problems)


def test_section_with_title_in_a_running_form_is_accepted():
    parsed = {Section.MDNA: section(Section.MDNA, MDNA_TEXT)}
    rendered = {"item": "MANAGEMENT'S DISCUSSION AND ANALYSIS (MD&A). The following."}
    assert section_problems(FormType.FORM_10Q, parsed, rendered) == []


def test_relocated_statements_are_exempt_from_the_title_check():
    """Stanley's statements begin under another item's heading, by design."""
    notes = "Note 1 describes significant accounting policies.\n\n" + "A sentence.\n\n" * 30
    parsed = {
        Section.FINANCIAL_NOTES: section(
            Section.FINANCIAL_NOTES, notes, note="Read from part_iv_item_16."
        )
    }
    assert section_problems(FormType.FORM_10K, parsed, {"item": "Item 16. Summary."}) == []


def test_mdna_that_ran_into_the_statements_is_flagged():
    """Chevron: every length looked fine, but MD&A continued through the statements."""
    text = MDNA_TEXT + "See accompanying Notes to the Consolidated Financial Statements."
    parsed = {Section.MDNA: section(Section.MDNA, text)}
    problems = section_problems(FormType.FORM_10Q, parsed, {"item": "Item 2. MD&A"})
    assert any("another part's heading" in p for p in problems)


def test_notes_that_ran_into_the_signatures_are_flagged():
    notes = "Note 1 policies.\n\n" + "A sentence.\n\n" * 30 + "SIGNATURES\n\nSigned by someone."
    parsed = {Section.FINANCIAL_NOTES: section(Section.FINANCIAL_NOTES, notes)}
    problems = section_problems(FormType.FORM_10K, parsed, {"item": "Item 8. Financial Statements"})
    assert any("another part's heading" in p for p in problems)


def test_signature_page_before_relocated_statements_is_not_flagged():
    notes = "SIGNATURES\n\nSigned by someone.\n\nNote 1 Policies\n\n" + "A sentence.\n\n" * 60
    parsed = {Section.FINANCIAL_NOTES: section(Section.FINANCIAL_NOTES, notes)}
    assert (
        section_problems(FormType.FORM_10K, parsed, {"item": "Item 8. Financial Statements"}) == []
    )


def test_required_section_without_expected_terms_is_flagged():
    parsed = {Section.MDNA: section(Section.MDNA, "An unrelated passage about nothing here.")}
    problems = section_problems(FormType.FORM_10Q, parsed, {"item": "Item 2. MD&A"})
    assert any("none of its expected terms" in p for p in problems)


def test_expected_terms_may_be_in_a_heading():
    """Chevron's "Note 1" stands alone as a heading above its text."""
    notes = "Note 1\n\nGeneral matters are described here."
    parsed = {Section.FINANCIAL_NOTES: section(Section.FINANCIAL_NOTES, notes)}
    problems = section_problems(FormType.FORM_10Q, parsed, {"item": "Item 1. Financial Statements"})
    assert not any("expected terms" in p for p in problems)


def test_pointer_sections_skip_content_checks():
    parsed = {
        Section.MARKET_RISK: ParsedSection(
            section=Section.MARKET_RISK,
            status=SectionStatus.CROSS_REFERENCE,
            source_item="item",
            paragraphs=clean_paragraphs("Refer to Item 7.", "a:b"),
        )
    }
    rendered = {"item": "Item 7A. Quantitative and Qualitative Disclosures About Market Risk."}
    assert section_problems(FormType.FORM_10K, parsed, rendered) == []


# --- Section endings ---


def test_market_risk_holding_the_audit_report_is_flagged():
    """AT&T: its reports sit between Item 7A and Item 8, so market risk ran through them."""
    text = "Item 7A. Market risk.\n\n" + "Rates could change our costs.\n\n" * 12
    text += "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM\n\nWe audited the company."
    parsed = {Section.MARKET_RISK: section(Section.MARKET_RISK, text)}
    problems = section_problems(FormType.FORM_10K, parsed, {"item": "Item 7A. Market Risk"})
    assert any("another part's heading" in p for p in problems)


def test_market_risk_that_only_mentions_the_report_is_not_flagged():
    text = "Rates could change our costs.\n\n" * 12 + "See the report of management for more."
    parsed = {Section.MARKET_RISK: section(Section.MARKET_RISK, text)}
    assert section_problems(FormType.FORM_10K, parsed, {"item": "Item 7A. Market Risk"}) == []


def numbered_notes(*numbers: int) -> str:
    """Note headings in the style "Note 3 – Some Title", each followed by a short sentence."""
    return "\n\n".join(f"Note {number} – Some Title\n\nA sentence." for number in numbers)


def notes_section(text: str) -> dict[Section, ParsedSection]:
    """A notes section whose source item is item_8, as the numbering check expects."""
    return {Section.FINANCIAL_NOTES: section(Section.FINANCIAL_NOTES, text, item="item_8")}


def test_complete_note_numbering_is_not_flagged():
    text = numbered_notes(1, 2, 3, 4, 5, 6)
    rendered = {"item_8": text, "item_7": "See Note 6.\n\nNote 3 – Debt is relevant."}
    assert note_numbering_problems(FormType.FORM_10K, notes_section(text), rendered) == []


def test_skipped_note_number_is_flagged():
    text = numbered_notes(1, 2, 3, 5, 6, 7)
    problems = note_numbering_problems(FormType.FORM_10K, notes_section(text), {"item_8": text})
    assert problems == ["financial_notes: note numbers skip [4]"]


def test_next_note_in_another_item_is_flagged():
    text = numbered_notes(1, 2, 3, 4, 5)
    rendered = {"item_8": text, "part_iv_signatures": "Note 6 – Segments\n\nThe segments."}
    problems = note_numbering_problems(FormType.FORM_10K, notes_section(text), rendered)
    assert problems == ["financial_notes: ends at note 5, but part_iv_signatures has note 6"]


def test_10k_notes_citing_a_later_note_are_flagged():
    """Everus's FY2024 notes end in a complete sentence, so only their citations show the cut."""
    text = numbered_notes(1, 2, 3, 4, 5) + "\n\nSee Note 9 for more."
    problems = note_numbering_problems(FormType.FORM_10K, notes_section(text), {"item_8": text})
    assert problems == ["financial_notes: ends at note 5, but cites note 9"]


def test_10q_citing_the_10ks_notes_is_not_flagged():
    """JPMorgan's 10-Q has 25 notes and cites Note 34 of its 10-K."""
    text = numbered_notes(1, 2, 3, 4, 5) + "\n\nSee Note 34 of the 2025 Form 10-K."
    assert note_numbering_problems(FormType.FORM_10Q, notes_section(text), {"item_8": text}) == []


def test_non_breaking_space_in_a_note_heading_is_read():
    """Cosmos Health and Ingram Micro pad their headings, which looked like skipped notes."""
    text = "\n\n".join(f"NOTE\xa0{number}\xa0– Title\n\nA sentence." for number in range(1, 7))
    assert note_numbering_problems(FormType.FORM_10K, notes_section(text), {"item_8": text}) == []


def test_a_short_numbered_list_is_not_taken_for_notes():
    text = "1. First Point\n\n2. Second Point\n\n9. Other Point"
    assert note_numbering_problems(FormType.FORM_10K, notes_section(text), {"item_8": text}) == []


# About 2,960 characters: over the ending check's 2,000-character limit, so leaving it out
# between a section and the next item must be flagged.
LONG_PROSE = "This long paragraph continues the discussion of the notes in some detail. " * 40


def ending_filing(*blocks: str) -> str:
    """A filing made of one paragraph per block, in order."""
    return page(*[f"<p>{block}</p>" for block in blocks])


def ending_problems(html: str, last_paragraph: str, rendered: dict[str, str]) -> list[str]:
    """Run the ending check on a notes section whose last paragraph is `last_paragraph`."""
    parsed = {
        Section.FINANCIAL_NOTES: section(Section.FINANCIAL_NOTES, last_paragraph, item="item_8")
    }
    return section_ending_problems(html, parsed, rendered)


# The last paragraph of the parsed section. The check finds it in the filing, then looks at
# what sits between it and the next item.
END = "The final sentence of what was parsed as this section appears here, and it ends now."


def test_text_left_between_a_section_and_the_next_item_is_flagged():
    """Everus: Item 8 stopped at Note 11, with Notes 12 to 15 still to come before Item 9."""
    html = ending_filing(END, "Note 12 – Segments", LONG_PROSE, "Item 9. Changes")
    problems = ending_problems(html, END, {"item_8": END, "item_9": "Changes"})
    assert len(problems) == 1
    assert "left out before the next item" in problems[0]


def test_text_filed_under_signatures_still_counts_as_left_out():
    html = ending_filing(END, LONG_PROSE, "Item 9. Changes")
    rendered = {"item_8": END, "item_9": "Changes", "part_iv_signatures": LONG_PROSE}
    assert ending_problems(html, END, rendered)


def test_section_running_up_to_the_next_item_is_not_flagged():
    html = ending_filing(END, "42", "Item 9. Changes")
    assert ending_problems(html, END, {"item_8": END, "item_9": "Changes"}) == []


def test_section_ending_with_a_colon_before_the_next_item_is_not_flagged():
    """Many sections really do end just before a removed table, such as Apple's 10-Q."""
    last = "The components of the balance for each period presented are as follows:"
    html = ending_filing(last, "Item 9. Changes")
    assert ending_problems(html, last, {"item_8": last, "item_9": "Changes"}) == []


def test_text_parsed_as_another_item_is_not_left_out():
    """JPMorgan: statements follow its MD&A with no heading, and they're parsed as Item 1."""
    html = ending_filing(END, LONG_PROSE, "Item 9. Changes")
    rendered = {"item_8": END, "item_9": "Changes", "item_1": LONG_PROSE}
    assert ending_problems(html, END, rendered) == []


def test_no_claim_when_another_item_comes_next():
    html = ending_filing(END, LONG_PROSE, "Item 12. Ownership")
    assert ending_problems(html, END, {"item_8": END, "item_9": "Changes"}) == []
