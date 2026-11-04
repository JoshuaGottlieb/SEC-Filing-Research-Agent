"""Splits a filing's HTML into canonical sections of clean narrative paragraphs.

Figures come from XBRL, so `tables.py` removes data rows first and retrieval only
handles language. This module turns the remaining text into canonical sections.
"""

from __future__ import annotations

import re
from enum import StrEnum

from edgar.documents import ParserConfig, parse_html
from pydantic import BaseModel, ConfigDict

from sec_agent.ingest.tables import ITEM_HEADING
from sec_agent.schemas import FormType, Section

# Which edgartools item name holds each canonical section. The same canonical section sits
# in different places by form: MD&A is Item 7 in a 10-K but Part I Item 2 in a 10-Q, and a
# 10-Q's risk factors are Part II Item 1A (usually only updates since the last 10-K). A
# 10-Q's financial statements are Part I Item 1, so FINANCIAL_NOTES maps there.
SECTION_MAP: dict[FormType, dict[str, Section]] = {
    FormType.FORM_10K: {
        "part_i_item_1": Section.BUSINESS,
        "part_i_item_1a": Section.RISK_FACTORS,
        "part_ii_item_7": Section.MDNA,
        "part_ii_item_7a": Section.MARKET_RISK,
        "part_ii_item_8": Section.FINANCIAL_NOTES,
    },
    FormType.FORM_10Q: {
        "part_i_item_1": Section.FINANCIAL_NOTES,
        "part_i_item_2": Section.MDNA,
        "part_i_item_3": Section.MARKET_RISK,
        "part_ii_item_1a": Section.RISK_FACTORS,
    },
}

# Some filers (Stanley Black & Decker) make Item 8 a pointer and put the financial
# statements after Item 15 or 16.
RELOCATED_NOTES_ITEMS = ("part_iv_item_16", "part_iv_item_15")
# The relocated item must hold at least this much text to be taken as the statements. Item 15
# also holds the exhibit list, which is far shorter than a set of financial statements.
RELOCATED_NOTES_MIN_CHARS = 20_000
# An Item 8 under this length is a pointer ("see page F-1"), not the statements themselves.
ITEM_8_POINTER_MAX_CHARS = 1_000

# At or under this length a section may be a real answer rather than a parsing failure:
# "no material changes", a cross-reference, or "Not applicable". Above it, the section is
# treated as content. Also the audit's exemption from the title check (Cosmos Health's
# "Not applicable" sections have no heading above them).
SHORT_SECTION_MAX_CHARS = 1_200
# A 10-Q's risk factors often say only that nothing changed since the 10-K.
NO_CHANGE_PATTERN = re.compile(
    r"no material changes|not materially changed|have not been any material changes", re.I
)
# A short section that points elsewhere ("see Note 12", "incorporated by reference").
CROSS_REFERENCE_PATTERN = re.compile(
    r"\b(see|refer to|incorporat\w*|included in|set forth|contained in|appears? in|"
    r"discussed in|described in|presented in|under the (heading|caption))\b",
    re.I,
)

# Page furniture: whole blocks that repeat on every page and carry no content. Each pattern
# is anchored to the full block, so a sentence that happens to contain "page" is kept.
FURNITURE_PATTERNS = (
    # A bare page number, or "Page 4 of 90".
    re.compile(r"^\d{1,3}$"),
    re.compile(r"^(page\s+)?\d{1,3}(\s+of\s+\d{1,3})?$", re.I),
    # Running headers such as "Table of Contents" and "Company | 2025 Form 10-K | Page 12".
    re.compile(r"^table of contents\b.*$", re.I),
    re.compile(r"^.*\|\s*page\s+\d{1,3}$", re.I),
    # A repeated part banner such as "Part II".
    re.compile(r"^part\s+i{1,3}v?\s*[—–-]?\s*[a-z ]*$", re.I),
    # A unit caption left behind when its table was removed: "(In millions)".
    re.compile(r"^\((amounts\s+)?(in\s+)?(millions|thousands|billions)\b[^)]*\)$", re.I),
)
# Zero-width characters, non-breaking spaces and box-drawing rules that rendering adds.
RENDERING_NOISE = re.compile(r"[\xa0\u200b\u200c\u200d\ufeff\u2500-\u257f]+")
# A block is a heading, not a paragraph, if it is short and has no sentence punctuation:
# at most 120 characters and 15 words. Prose paragraphs are almost always longer; a short
# line that ends in a period is still a sentence. Real headings run to roughly a line.
HEADING_MAX_CHARS = 120
HEADING_MAX_WORDS = 15
# A block ending in one of these is a finished sentence. A block that does not end in one
# and is followed by a lowercase block was split by a page break.
TERMINAL_PUNCTUATION = (".", ":", ";", "?", "!", '."', ".”", ")")


class SectionStatus(StrEnum):
    """What a section turned out to be. Only CONTENT counts as readable filing text."""

    CONTENT = "content"
    # A real finding in a 10-Q (for example, no change in risk factors), not a failure.
    NO_MATERIAL_CHANGES = "no_material_changes"
    # The section only points to another place in the filing.
    CROSS_REFERENCE = "cross_reference"
    # The parser found no such item at all.
    MISSING = "missing"


class Paragraph(BaseModel):
    """One cleaned paragraph, with the nearest heading above it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # "<accession>:<section>:<number>", stable so citations can point at it later.
    paragraph_id: str
    text: str
    heading: str | None = None


class ParsedSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    section: Section
    status: SectionStatus
    # The edgartools item the text was read from (usually the section's own item).
    source_item: str | None = None
    paragraphs: tuple[Paragraph, ...] = ()
    # Set when the text came from somewhere unexpected, such as relocated statements.
    note: str | None = None

    @property
    def char_count(self) -> int:
        return sum(len(paragraph.text) for paragraph in self.paragraphs)


def _normalize(text: str) -> str:
    # Some filers pad with zero-width spaces, and table rendering adds box-drawing rules.
    return " ".join(RENDERING_NOISE.sub(" ", text).split())


def is_furniture(text: str) -> bool:
    """True for page numbers, running headers and similar blocks that carry no content."""
    return any(pattern.match(text) for pattern in FURNITURE_PATTERNS)


def _is_heading(text: str) -> bool:
    if ITEM_HEADING.match(text):
        return True
    # Bullets, list items and anything ending like a sentence are paragraphs.
    if text[0] in "•●·-–—*(" or text.endswith(TERMINAL_PUNCTUATION):
        return False
    return len(text) <= HEADING_MAX_CHARS and len(text.split()) <= HEADING_MAX_WORDS


def _continues_previous(previous: str, text: str) -> bool:
    """True when a page break split a sentence across two blocks."""
    return not previous.endswith(TERMINAL_PUNCTUATION) and text[0].islower()


def clean_paragraphs(raw_text: str, id_prefix: str) -> tuple[Paragraph, ...]:
    """Split rendered text into paragraphs, dropping page furniture and tracking headings."""
    # Blocks are separated by blank lines in the rendered text.
    blocks = (_normalize(block) for block in re.split(r"\n\s*\n", raw_text))
    texts: list[str] = []
    for block in blocks:
        if not block or is_furniture(block):
            continue
        # Join a sentence that a page break cut in two.
        if texts and _continues_previous(texts[-1], block):
            texts[-1] = f"{texts[-1]} {block}"
        else:
            texts.append(block)

    paragraphs: list[Paragraph] = []
    heading: str | None = None
    for text in texts:
        if _is_heading(text):
            # An "Item N" heading resets the context; any other heading labels what follows.
            heading = None if ITEM_HEADING.match(text) else text
            continue
        paragraphs.append(
            Paragraph(
                paragraph_id=f"{id_prefix}:{len(paragraphs):04d}",
                text=text,
                heading=heading,
            )
        )
    return tuple(paragraphs)


def classify_status(paragraphs: tuple[Paragraph, ...]) -> SectionStatus:
    """A short section may be a real result, not a parsing failure.

    A 10-Q saying risk factors have "no material changes" is itself a finding.
    """
    text = " ".join(paragraph.text for paragraph in paragraphs)
    # A long section is content whatever it says; the phrases below only mean something
    # in a short one.
    if len(text) > SHORT_SECTION_MAX_CHARS:
        return SectionStatus.CONTENT
    if NO_CHANGE_PATTERN.search(text):
        return SectionStatus.NO_MATERIAL_CHANGES
    if CROSS_REFERENCE_PATTERN.search(text):
        return SectionStatus.CROSS_REFERENCE
    return SectionStatus.CONTENT


def _rendered_sections(document: object) -> dict[str, str]:
    """Item name to text, as edgartools split the document."""
    return {name: section.text() for name, section in document.sections.items()}


def _notes_source(rendered: dict[str, str]) -> tuple[str, str | None]:
    """Find where a 10-K's financial statements actually are."""
    item_8 = rendered.get("part_ii_item_8", "")
    if len(item_8) > ITEM_8_POINTER_MAX_CHARS:
        return "part_ii_item_8", None
    # Item 8 is only a pointer, so look for the statements after Item 15 or 16 and take the
    # longest. Stanley Black & Decker files this way.
    candidates = [
        (len(rendered.get(item, "")), item)
        for item in RELOCATED_NOTES_ITEMS
        if len(rendered.get(item, "")) >= RELOCATED_NOTES_MIN_CHARS
    ]
    if not candidates:
        return "part_ii_item_8", None
    _, item = max(candidates)
    return item, f"Item 8 points elsewhere; financial statements read from {item}."


def render_items(stripped_html: str, form: FormType) -> dict[str, str]:
    """Render each item's text with edgartools. HTML must already have its tables stripped."""
    document = parse_html(stripped_html, ParserConfig(form=form.value))
    return _rendered_sections(document)


def sections_from_items(
    rendered: dict[str, str], form: FormType, accession: str
) -> dict[Section, ParsedSection]:
    """Map rendered items to canonical sections and clean their text."""
    results: dict[Section, ParsedSection] = {}
    for item, section in SECTION_MAP[form].items():
        source, note = item, None
        # Only a 10-K's notes can be relocated; every other section reads from its own item.
        if form is FormType.FORM_10K and section is Section.FINANCIAL_NOTES:
            source, note = _notes_source(rendered)
        if source not in rendered:
            results[section] = ParsedSection(section=section, status=SectionStatus.MISSING)
            continue
        paragraphs = clean_paragraphs(rendered[source], f"{accession}:{section.value}")
        results[section] = ParsedSection(
            section=section,
            status=classify_status(paragraphs),
            source_item=source,
            paragraphs=paragraphs,
            note=note,
        )
    return results


# Minimum narrative characters for sections every filing must have. These catch a
# parser that silently returns a fragment, and are set below the shortest real section
# seen across 8 companies, including a small filer (Tootsie Roll) and one with brief
# notes (Apple). Length can't show that a section is right: see `audit.section_problems`.
# The floors are set for companies of that size, so they can reject a very small filer
# whose sections are genuinely short (First America Resources: 5 filings wrongly rejected
# in the fourth held-out test). Market risk is not listed because smaller companies may
# omit it ("Not required for Smaller Reporting Companies").
REQUIRED_CONTENT_CHARS: dict[FormType, dict[Section, int]] = {
    FormType.FORM_10K: {
        Section.BUSINESS: 5_000,
        Section.RISK_FACTORS: 15_000,
        Section.MDNA: 15_000,
        Section.FINANCIAL_NOTES: 40_000,
    },
    FormType.FORM_10Q: {
        Section.MDNA: 10_000,
        Section.FINANCIAL_NOTES: 8_000,
    },
}


def qa_problems(form: FormType, sections: dict[Section, ParsedSection]) -> list[str]:
    problems = []
    for section, minimum in REQUIRED_CONTENT_CHARS[form].items():
        parsed = sections[section]
        if parsed.status is not SectionStatus.CONTENT:
            problems.append(f"{section.value}: expected content, got {parsed.status.value}")
        elif parsed.char_count < minimum:
            problems.append(f"{section.value}: {parsed.char_count} chars, expected {minimum}+")
    for section in SECTION_MAP[form].values():
        if sections[section].status is SectionStatus.MISSING and (
            section not in REQUIRED_CONTENT_CHARS[form]
        ):
            problems.append(f"{section.value}: missing")
    return problems
