from __future__ import annotations

import pytest

from sec_agent.ingest.sections import (
    ParsedSection,
    SectionStatus,
    _notes_source,
    classify_status,
    clean_paragraphs,
    is_furniture,
    qa_problems,
)
from sec_agent.schemas import FormType, Section


# Page furniture: blocks that repeat on every page and carry no content, in the forms real
# filings use. Each is matched as a whole block.
@pytest.mark.parametrize(
    "text",
    [
        "22",
        "Page 22",
        "Table of Contents",
        "Table of Contents The Hershey Company | Q2 2026 Form 10-Q | Page 36",
        "PART I",
        "PART II—OTHER INFORMATION",
        "(Millions of Dollars)",
        "(in thousands, except per share data)",
        "(amounts in thousands, except share data or if otherwise indicated)",
    ],
)
def test_furniture_is_recognized(text):
    assert is_furniture(text)


@pytest.mark.parametrize(
    "text",
    ["Gross margin increased 14.8 percentage points.", "Part of our strategy is growth."],
)
def test_real_text_is_not_furniture(text):
    # Words like "Page" or "Part" at the start of a sentence must not make it furniture.
    assert not is_furniture(text)


def test_clean_drops_furniture_and_normalizes_whitespace():
    raw = "Table of Contents\n\n12\n\nNet\xa0sales   rose in the quarter.\n\n13"
    paragraphs = clean_paragraphs(raw, "a:mdna")
    assert [p.text for p in paragraphs] == ["Net sales rose in the quarter."]


def test_clean_removes_zero_width_padding_and_table_rules():
    """Tootsie Roll pads text with zero-width spaces; rendered tables add box rules."""
    paragraphs = clean_paragraphs(
        "\u200b\n\n\u200bNet sales rose.\u200b\n\n──────────\n\nPlan ──── Funded by assets.",
        "a:mdna",
    )
    assert [p.text for p in paragraphs] == ["Net sales rose.", "Plan Funded by assets."]


def test_clean_tracks_headings_without_making_them_paragraphs():
    # "Item 7" resets the heading; the two short title lines become headings of what follows.
    raw = (
        "Item 7. MANAGEMENT'S DISCUSSION\n\nCost of Sales and Gross Margin\n\n"
        "Gross margin rose.\n\nThe increase was large.\n\nNet Sales\n\nSales rose."
    )
    paragraphs = clean_paragraphs(raw, "a:mdna")
    assert [(p.heading, p.text) for p in paragraphs] == [
        ("Cost of Sales and Gross Margin", "Gross margin rose."),
        ("Cost of Sales and Gross Margin", "The increase was large."),
        ("Net Sales", "Sales rose."),
    ]


def test_clean_rejoins_sentence_split_by_page_break():
    # The sentence stops mid-clause, the page number "36" is dropped, and it resumes in lower case.
    raw = "Costs increased because of higher\n\n36\n\nfreight rates in the quarter."
    paragraphs = clean_paragraphs(raw, "a:mdna")
    assert [p.text for p in paragraphs] == [
        "Costs increased because of higher freight rates in the quarter."
    ]


def test_clean_keeps_short_bullets_as_paragraphs():
    # A short line would look like a heading, but a leading bullet marks it as list text.
    paragraphs = clean_paragraphs("• rising cocoa prices", "a:risk")
    assert [p.text for p in paragraphs] == ["• rising cocoa prices"]


def test_paragraph_ids_are_sequential_and_scoped():
    paragraphs = clean_paragraphs("First sentence.\n\nSecond sentence.", "acc:mdna")
    assert [p.paragraph_id for p in paragraphs] == ["acc:mdna:0000", "acc:mdna:0001"]


def status_of(text: str) -> SectionStatus:
    """The status classify_status gives a section made of this text."""
    return classify_status(clean_paragraphs(text, "a:s"))


def test_no_material_changes_is_a_result_not_a_failure():
    text = "There have been no material changes to the risk factors in our 2025 Form 10-K."
    assert status_of(text) is SectionStatus.NO_MATERIAL_CHANGES


def test_short_pointer_is_a_cross_reference():
    assert status_of("See Item 7 of this report for this information.") is (
        SectionStatus.CROSS_REFERENCE
    )


def test_incorporation_by_reference_is_a_cross_reference():
    text = "The Company incorporates by reference the material captioned Market Risk in Item 7."
    assert status_of(text) is SectionStatus.CROSS_REFERENCE


def test_short_real_content_stays_content():
    assert status_of("Interest rates are our largest exposure.") is SectionStatus.CONTENT


def test_long_section_is_content_even_with_trigger_words():
    # About 1,800 characters, over the 1,200 limit: the phrase is just text, not the answer.
    assert status_of("See no material changes here. " * 60) is SectionStatus.CONTENT


def test_notes_location_prefers_item_8():
    # 5,000 characters is well over the 1,000 that would make Item 8 only a pointer.
    assert _notes_source({"part_ii_item_8": "x" * 5_000}) == ("part_ii_item_8", None)


def test_notes_location_follows_pointer_to_relocated_statements():
    # Item 15 is under the 20,000-character minimum (probably just an exhibit list); Item 16
    # clears it, so the statements are read from there.
    source, note = _notes_source(
        {
            "part_ii_item_8": "See Item 15.",
            "part_iv_item_15": "x" * 2_000,
            "part_iv_item_16": "x" * 200_000,
        }
    )
    assert source == "part_iv_item_16"
    assert "Item 8" in note


def test_notes_location_keeps_item_8_when_nothing_better_exists():
    assert _notes_source({"part_ii_item_8": "See Item 15."}) == ("part_ii_item_8", None)


def parsed(section: Section, status: SectionStatus, text: str = "") -> ParsedSection:
    """A section with the given status and text."""
    paragraphs = clean_paragraphs(text, "a:s") if text else ()
    return ParsedSection(section=section, status=status, paragraphs=paragraphs)


def test_qa_flags_fragment_and_missing_sections():
    sections = {
        Section.MDNA: parsed(Section.MDNA, SectionStatus.CONTENT, "Too short to be real."),
        Section.FINANCIAL_NOTES: parsed(Section.FINANCIAL_NOTES, SectionStatus.MISSING),
        Section.MARKET_RISK: parsed(Section.MARKET_RISK, SectionStatus.MISSING),
        Section.RISK_FACTORS: parsed(
            Section.RISK_FACTORS, SectionStatus.NO_MATERIAL_CHANGES, "No material changes."
        ),
    }
    problems = qa_problems(FormType.FORM_10Q, sections)
    # MD&A is a fragment; the notes are missing; market risk is missing (and is not required,
    # but a missing section is still reported); risk factors "no material changes" is fine.
    assert any(p.startswith("mdna:") and "expected 10000+" in p for p in problems)
    assert any(p.startswith("financial_notes:") for p in problems)
    assert "market_risk: missing" in problems
    assert not any(p.startswith("risk_factors") for p in problems)
