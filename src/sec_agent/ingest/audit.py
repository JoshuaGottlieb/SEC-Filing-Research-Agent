"""Checks parser output against evidence the parser didn't use.

The table checks don't reuse the classifier's rules. Lost text is measured against the
original HTML, leftover figures against inline XBRL tags and the final paragraphs.
The section checks look for what a section must and must not contain, because a filing
can have every section at a plausible length with the wrong text in each.
"""

from __future__ import annotations

import re
from bisect import bisect_right

from pydantic import BaseModel, ConfigDict

from sec_agent.ingest.fallback import extract_blocks
from sec_agent.ingest.sections import (
    REQUIRED_CONTENT_CHARS,
    SHORT_SECTION_MAX_CHARS,
    ParsedSection,
    SectionStatus,
    is_furniture,
)
from sec_agent.ingest.tables import (
    ITEM_HEADING,
    SENTENCE_ENDINGS,
    StripResult,
    parse_html_tree,
    table_rows,
)
from sec_agent.schemas import FormType, Section

# --- Table checks: did table removal take out prose, or leave figures behind? ---

# A sentence long enough that a coincidental match is unlikely. This is also the check's
# limit: a lost sentence shorter than 15 words is not noticed. Row labels and captions
# are mostly short, so a short floor would flag them as lost prose.
AUDIT_SENTENCE_MIN_WORDS = 15
# Coverage fingerprints every Nth sentence rather than all of them, so about 400 are
# tested however long the filing is. This keeps the check fast on 300-page 10-Ks.
COVERAGE_SAMPLE_SIZE = 400
# At least 90% of sampled sentences must appear in a parsed item. A fragment from
# `edgartools` reached only 66% to 71% on AT&T's 10-Qs; a good parse is near 100%, and
# the gap leaves room for sentences that really belong to no item (cover page, etc.).
MIN_SENTENCE_COVERAGE = 0.90

# Kept tables with this many XBRL-tagged numbers, mostly in numeric cells, are data.
# A single tagged number can sit in prose that happens to be inside a table, so one or
# two tags prove nothing; four tags in a mostly numeric table is a financial table that
# should have been removed. These values were set on the 115 development filings, so
# there is no fair error rate for them (see eval/parser_holdout.md).
LEFTOVER_MIN_TAGGED_VALUES = 4
LEFTOVER_MIN_NUMERIC_SHARE = 0.3

# A table cell or word that is a number: "$1,234", "(5.2)", "12%", "-3".
NUMERIC_TOKEN = re.compile(r"^[$(]*-?\d[\d,]*(\.\d+)?[%)]*$")
# A paragraph of at least 8 numbers making up at least 40% of its words is a table row
# that got flattened into prose. Found on National Vision's long balance-sheet row; real
# prose with many figures ("revenue grew 5% to $1.2 billion") stays under 40%.
NUMERIC_PARAGRAPH_MIN_TOKENS = 8
NUMERIC_PARAGRAPH_MIN_SHARE = 0.4

# Problem messages quote at most this much of the offending text.
EXCERPT_CHARS = 200
# Split after sentence-ending punctuation. Semicolons count, because filings join
# long list items with them.
SENTENCE_SPLIT = re.compile(r"(?<=[.;?!])\s+")

# --- Section checks: is each section the right text? ---

# How much of a section's start is searched for its title. Titles sit right at the top,
# so a longer window would let an unrelated mention of "business" pass.
TITLE_WINDOW_CHARS = 400
# What a section's opening text should say. Deliberately loose: it catches a section that
# starts in the wrong place, not one with an unusual heading.
SECTION_TITLES: dict[Section, re.Pattern[str]] = {
    Section.BUSINESS: re.compile(r"\bbusiness\b", re.I),
    Section.RISK_FACTORS: re.compile(r"\brisk factors\b", re.I),
    Section.MDNA: re.compile(r"management.s discussion|\bMD&A\b", re.I),
    Section.MARKET_RISK: re.compile(r"market risk", re.I),
    Section.FINANCIAL_NOTES: re.compile(
        r"financial statements|statements? of (operations|income|earnings|financial)"
        r"|notes? to|\bnote 1\b",
        re.I,
    ),
}
# A section with real content must mention these somewhere.
REQUIRED_TERMS: dict[Section, re.Pattern[str]] = {
    Section.BUSINESS: re.compile(r"\b(compet\w+|employees|customers|products|services)\b", re.I),
    Section.RISK_FACTORS: re.compile(r"\b(risks?|advers\w+)\b", re.I),
    Section.MDNA: re.compile(r"results of operations|liquidity|capital resources", re.I),
    Section.FINANCIAL_NOTES: re.compile(
        r"\bnote\s*1\b|notes to|significant accounting policies|basis of presentation", re.I
    ),
}
# Headings that show a section has absorbed another part of the filing.
FOREIGN_HEADINGS: dict[Section, re.Pattern[str]] = {
    # Each pattern matches a heading that belongs to a different part of the filing. It is
    # anchored with ^ and often $ so a sentence that merely mentions "signatures" or "the
    # notes" doesn't count. The list only knows these headings, so other misfiled text
    # (Chevron's MD&A running into the statements was found this way) gets through.
    Section.MDNA: re.compile(
        r"^(notes? to (the )?(condensed )?consolidated financial statements\W*$"
        r"|report of independent registered public accounting firm\W*$"
        r"|see accompanying notes\b)",
        re.I,
    ),
    Section.FINANCIAL_NOTES: re.compile(
        r"^(management.s discussion and analysis( of financial condition"
        r" and results of operations)?\W*$"
        r"|signatures?\W*$"
        r"|pursuant to the requirements of section 13)",
        re.I,
    ),
    Section.BUSINESS: re.compile(r"^item\s*1a\b", re.I),
    Section.RISK_FACTORS: re.compile(r"^(item\s*7\b|management.s discussion and analysis)", re.I),
    # AT&T's reports sit between Item 7A and Item 8, under neither heading.
    Section.MARKET_RISK: re.compile(
        r"^(report of (independent registered public accounting firm|management)\W*$"
        r"|management.s (annual )?report on internal control"
        r"|notes? to (the )?(condensed )?consolidated financial statements\W*$"
        r"|see accompanying notes\b)",
        re.I,
    ),
}


# The first 10% of a section's paragraphs are not searched for foreign headings. Some
# filers put the statements after the signature page (Stanley Black & Decker), so a
# foreign heading right at the start is legitimate; absorbed text shows up further along.
FOREIGN_HEADING_OPENING_SHARE = 0.1

# --- Note numbering and section ending checks (added after Everus's notes lost their end) ---

# The item name the parser gives the signature page. Text filed here does not count as
# "parsed elsewhere" in the ending check, because Everus's missing notes were filed there.
SIGNATURES_ITEM = "part_iv_signatures"
# A heading's title starts with a capital, which separates it from "Note 7 of the 10-K".
NOTE_HEADING = re.compile(
    r"(?m)^[^\S\n]*(?:[Nn]otes?|NOTES?)[^\S\n]*(\d{1,2})[^\S\n]*[-–—:.]?[^\S\n]*[A-Z(]"
)
# Inside the notes, a looser match only adds numbers, so it can't invent a skipped note.
NOTE_LINE = re.compile(r"(?im)^[^\S\n]*notes?[^\S\n]*(\d{1,2})\b")
# Filers that number notes as "7. Income Taxes" instead of "Note 7".
NUMBERED_HEADING = re.compile(r"(?m)^[^\S\n]*\(?(\d{1,2})[.)][^\S\n]+[A-Z]")
# An in-text citation such as "see Note 15".
NOTE_REFERENCE = re.compile(r"\bNotes?\s+(\d{1,2})\b")
# Fewer than this could be a numbered list, not a filer's numbering of its notes. Filers
# with lettered notes (Stanley, Autoscope) or none are skipped; the check cannot read them.
MIN_NUMBERED_NOTES = 5

# To find where a parsed paragraph sits in the filing, its last 80 characters are tried
# if the whole paragraph isn't found (a page-break banner can split a paragraph). Keys
# under 40 characters are too common to place a paragraph reliably, so they are skipped.
ENDING_KEY_CHARS = 80
ENDING_MIN_KEY_CHARS = 40
# More than 2,000 characters of prose between a section's end and the next item means
# text was left out. Everus's 10-Ks lost about 21,700; the most lost anywhere else on the
# 115 development filings was 296. The limit therefore misses any loss under 2,000.
ENDING_MAX_TRAILING_CHARS = 2_000


class TableAudit(BaseModel):
    """The result of the three table checks for one filing. Each tuple holds excerpts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tables_total: int
    tables_removed: int
    # Rows of removed tables that were really prose and were kept in the text.
    prose_rows_preserved: int
    lost_sentences: tuple[str, ...] = ()
    leftover_tables: tuple[str, ...] = ()
    numeric_paragraphs: tuple[str, ...] = ()

    @property
    def problems(self) -> list[str]:
        problems = [f"wrongly removed table text: {text}" for text in self.lost_sentences]
        problems += [f"leftover data table: {text}" for text in self.leftover_tables]
        problems += [f"numeric paragraph: {text}" for text in self.numeric_paragraphs]
        return problems


def _fingerprint(text: str) -> str:
    """Letters and digits only, so spacing and punctuation changes don't cause misses."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _sentences(text: str) -> list[str]:
    """Complete sentences only. A long row label isn't one, and it isn't prose to keep."""
    return [
        sentence
        for sentence in SENTENCE_SPLIT.split(text)
        if len(re.findall(r"[A-Za-z]{2,}", sentence)) >= AUDIT_SENTENCE_MIN_WORDS
        and sentence.endswith(SENTENCE_ENDINGS)
    ]


def find_lost_sentences(original_html: str, stripped_html: str) -> tuple[str, ...]:
    """Sentences inside the original's tables that are gone from the stripped HTML."""
    original = parse_html_tree(original_html)
    remaining = _fingerprint(parse_html_tree(stripped_html).text_content())
    lost = []
    for table in original.iter("table"):
        # Nested tables are covered by the outer table's cells.
        if any(ancestor.tag == "table" for ancestor in table.iterancestors()):
            continue
        for row in table_rows(table):
            for cell in row:
                lost += [s for s in _sentences(cell) if _fingerprint(s) not in remaining]
    return tuple(sentence[:EXCERPT_CHARS] for sentence in lost)


def find_leftover_tables(strip: StripResult) -> tuple[str, ...]:
    """Kept tables that XBRL tagging shows are figures.

    The tags come from the filing itself, not from the classifier's rules, so this is an
    independent check. Untagged tables can slip by (Comstock's share issuance tables).
    """
    return tuple(
        table.excerpt
        for table in strip.kept
        if table.tagged_values >= LEFTOVER_MIN_TAGGED_VALUES
        and table.numeric_cells / table.cell_count >= LEFTOVER_MIN_NUMERIC_SHARE
    )


def is_numeric_paragraph(text: str) -> bool:
    """True for a table row that was flattened into one line of mostly numbers."""
    tokens = text.split()
    numeric = sum(1 for token in tokens if NUMERIC_TOKEN.match(token))
    return (
        numeric >= NUMERIC_PARAGRAPH_MIN_TOKENS
        and numeric / len(tokens) >= NUMERIC_PARAGRAPH_MIN_SHARE
    )


def find_numeric_paragraphs(sections: dict[Section, ParsedSection]) -> tuple[str, ...]:
    return tuple(
        paragraph.text[:EXCERPT_CHARS]
        for parsed in sections.values()
        for paragraph in parsed.paragraphs
        if is_numeric_paragraph(paragraph.text)
    )


def audit_tables(
    original_html: str, strip: StripResult, sections: dict[Section, ParsedSection]
) -> TableAudit:
    return TableAudit(
        tables_total=len(strip.tables),
        tables_removed=len(strip.removed),
        prose_rows_preserved=sum(len(table.preserved_rows) for table in strip.removed),
        lost_sentences=find_lost_sentences(original_html, strip.html),
        leftover_tables=find_leftover_tables(strip),
        numeric_paragraphs=find_numeric_paragraphs(sections),
    )


def sentence_coverage(stripped_html: str, rendered: dict[str, str]) -> float:
    """Share of the filing's sentences that reached a parsed item.

    Counting starts at the first sentence found, because cover pages and forward-looking
    statements come before Item 1 and belong to no item. Gaps and lost endings still count.
    """
    blocks = extract_blocks(parse_html_tree(stripped_html))
    sentences = [sentence for block in blocks for sentence in _sentences(block)]
    if not sentences:
        return 0.0
    output = _fingerprint(" ".join(rendered.values()))
    # Take every step-th sentence so about COVERAGE_SAMPLE_SIZE are tested.
    step = max(1, len(sentences) // COVERAGE_SAMPLE_SIZE)
    found = [_fingerprint(sentence) in output for sentence in sentences[::step]]
    if True not in found:
        return 0.0
    # Start counting at the first sentence that was found (see the docstring).
    body = found[found.index(True) :]
    return sum(body) / len(body)


def section_problems(
    form: FormType, sections: dict[Section, ParsedSection], rendered: dict[str, str]
) -> list[str]:
    """Problems that show a section holds the wrong text."""
    problems = []
    for section, parsed in sections.items():
        if parsed.status is SectionStatus.MISSING:
            continue
        title = SECTION_TITLES[section]
        opening = " ".join(rendered.get(parsed.source_item or "", "").split())[:TITLE_WINDOW_CHARS]
        # Relocated statements begin under another item's heading, by definition. A short
        # section has no room for misfiled text, and filers such as Cosmos Health write
        # "Not applicable." with no heading above it.
        # SHORT_SECTION_MAX_CHARS (1,200) is the exemption: it was set after Cosmos Health's
        # "Not applicable" sections, which have no heading above them, failed wrongly.
        if (
            parsed.note is None
            and parsed.char_count > SHORT_SECTION_MAX_CHARS
            and not title.search(opening)
        ):
            problems.append(f"{section.value}: does not open with its title: {opening[:80]!r}")
        if parsed.status is not SectionStatus.CONTENT:
            continue
        # Headings count: a filer's "Note 1" is often a heading of its own.
        text = " ".join(f"{p.heading or ''} {p.text}" for p in parsed.paragraphs)
        required = REQUIRED_TERMS.get(section)
        # Only sections that must have real content are checked, and only when they have
        # some (status CONTENT, above), so "not applicable" stubs don't fail for lacking words.
        if section in REQUIRED_CONTENT_CHARS[form] and required and not required.search(text):
            problems.append(f"{section.value}: has none of its expected terms")
        hit = _foreign_heading(section, parsed)
        if hit:
            problems.append(f"{section.value}: contains another part's heading: {hit[:80]!r}")
    return problems


def _foreign_heading(section: Section, parsed: ParsedSection) -> str | None:
    """A heading from another part of the filing, past the section's opening.

    The opening is skipped because statements can follow the signature page (Stanley
    Black & Decker). Text absorbed from later in the filing shows up further along.
    """
    foreign = FOREIGN_HEADINGS.get(section)
    if foreign is None:
        return None
    opening = int(len(parsed.paragraphs) * FOREIGN_HEADING_OPENING_SHARE)
    for paragraph in parsed.paragraphs[opening:]:
        for text in (paragraph.heading, paragraph.text):
            if text and foreign.match(text):
                return text
    return None


def _note_numbers(text: str) -> set[int]:
    """The numbers of a filer's note headings, in whichever style it numbers them."""
    named = {int(number) for number in NOTE_LINE.findall(text)}
    numbered = {int(number) for number in NUMBERED_HEADING.findall(text)}
    # A filer uses one style, so the style that finds more notes is the right reading.
    return max(named, numbered, key=len)


def note_numbering_problems(
    form: FormType, sections: dict[Section, ParsedSection], rendered: dict[str, str]
) -> list[str]:
    """Notes are numbered, so a section that lost some can show it. Lettered notes can't."""
    source = sections[Section.FINANCIAL_NOTES].source_item
    text = rendered.get(source or "", "")
    numbers = _note_numbers(text)
    if len(numbers) < MIN_NUMBERED_NOTES:
        return []
    last = max(numbers)
    problems = []
    # Three ways to show notes were lost: a gap in the numbers...
    skipped = [number for number in range(1, last) if number not in numbers]
    if skipped:
        problems.append(f"financial_notes: note numbers skip {skipped[:5]}")
    # ...the next note turning up in a different item...
    for item, other in rendered.items():
        if item != source and str(last + 1) in NOTE_HEADING.findall(other):
            problems.append(f"financial_notes: ends at note {last}, but {item} has note {last + 1}")
    # ...or the text citing a note higher than the last one found. A 10-Q can cite notes in
    # the 10-K, which run higher than its own, so this applies to 10-Ks only.
    if form is FormType.FORM_10K:
        cited = max((int(number) for number in NOTE_REFERENCE.findall(text)), default=0)
        if cited > last:
            problems.append(f"financial_notes: ends at note {last}, but cites note {cited}")
    return problems


def _next_item_heading(source_item: str | None, rendered: dict[str, str]) -> re.Pattern[str] | None:
    """The heading that should come right after an item, by the order the items were parsed."""
    items = list(rendered)
    # No next item to measure against: the section wasn't found, or it is the last one.
    if source_item not in items or source_item == items[-1]:
        return None
    number = re.search(r"item_(\d+[a-c]?)$", items[items.index(source_item) + 1])
    return re.compile(rf"^item\s*{number.group(1)}\b", re.I) if number else None


def _trailing_blocks(
    joined: str,
    starts: list[int],
    blocks: list[str],
    parsed: ParsedSection,
    next_heading: re.Pattern[str],
) -> list[str]:
    """The blocks between a section's last paragraph and the item that should follow it.

    Paragraphs are placed in order, so text repeated later in the filing can't mislead. Empty
    means there's nothing to measure against: the section wasn't found, or another item comes
    next.
    """
    position = 0
    for paragraph in parsed.paragraphs:
        whole = _fingerprint(paragraph.text)
        if len(whole) < ENDING_MIN_KEY_CHARS:
            continue
        # The ending alone finds a paragraph that a page break split with a banner, but many
        # paragraphs can share one, so it's only tried when the whole paragraph isn't there.
        for key in (whole, whole[-ENDING_KEY_CHARS:]):
            found = joined.find(key, position)
            if found >= 0:
                position = found + len(key)
                break
    if position == 0:
        return []
    trailing: list[str] = []
    # Walk forward from the last placed paragraph until the next item heading.
    for block in blocks[bisect_right(starts, position - 1) :]:
        if ITEM_HEADING.match(block):
            # Only count the text if the heading is the expected next item. Another item
            # means the layout is unusual (GE, JPMorgan's 10-Q), so nothing is measured.
            return trailing if next_heading.match(block) else []
        # Page numbers and running headers are not prose that was left out.
        if not is_furniture(block):
            trailing.append(block)
    # Reached the end of the filing without finding any item heading.
    return []


def section_ending_problems(
    stripped_html: str, sections: dict[Section, ParsedSection], rendered: dict[str, str]
) -> list[str]:
    """A section should run up to the next item, so long text between them was left out.

    This reads the filing's own order. A section ending in a colon isn't evidence, because
    many sections really do end just before a removed table.
    """
    blocks = [" ".join(block.split()) for block in extract_blocks(parse_html_tree(stripped_html))]
    prints = [_fingerprint(block) for block in blocks]
    starts = []
    length = 0
    for print_ in prints:
        starts.append(length)
        length += len(print_)
    joined = "".join(prints)
    problems = []
    for section, parsed in sections.items():
        next_heading = _next_item_heading(parsed.source_item, rendered)
        if next_heading is None:
            continue
        trailing = _trailing_blocks(joined, starts, blocks, parsed, next_heading)
        if sum(len(block) for block in trailing) <= ENDING_MAX_TRAILING_CHARS:
            continue
        # JPMorgan's statements follow its MD&A with no heading, but they're parsed as Item 1.
        # Signatures don't count: Everus's missing notes were filed there.
        elsewhere = _fingerprint(
            " ".join(
                text
                for item, text in rendered.items()
                if item not in (parsed.source_item, SIGNATURES_ITEM)
            )
        )
        left_out = sum(len(b) for b in trailing if _fingerprint(b) not in elsewhere)
        if left_out > ENDING_MAX_TRAILING_CHARS:
            problems.append(
                f"{section.value}: {left_out} chars left out before the next item, "
                f"after {parsed.paragraphs[-1].text[-60:]!r}"
            )
    return problems
