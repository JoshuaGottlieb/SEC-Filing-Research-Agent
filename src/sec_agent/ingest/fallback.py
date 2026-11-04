"""Finds a filing's items directly in its HTML, for layouts edgartools misreads.

Three layouts are handled, tried in this order:
  - A cross-reference index maps items to pages, and text is grouped by topic instead
    of by item (GE). One item can span separate page ranges.
  - Item headings in the usual order (the common case, if edgartools fails on it).
  - Item headings that only point to a financial section later in the filing, which is
    found by its named headings (JPMorgan, Chevron).

Results have the same shape as edgartools' output, so they go through the same cleaning
and the same validation. A layout this can't read gives missing or wrong sections, which
validation rejects.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import lxml.html

from sec_agent.ingest.tables import clean_text, parse_html_tree, table_rows
from sec_agent.schemas import FormType

# Names recorded on a result to say which of the three layouts (module docstring) was read.
LAYOUT_INDEX = "cross_reference_index"
LAYOUT_HEADINGS = "item_headings"
LAYOUT_POINTERS = "item_headings_with_pointers"

# HTML elements that start a new line of text. Text inside one of these is its own block;
# inline elements (span, b, a) are joined into the surrounding text.
BLOCK_TAGS = frozenset(
    {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "center", "tr", "table", "br"}
    | {"ul", "ol", "tbody", "thead", "tfoot", "td", "th", "section", "article", "blockquote"}
)

# Items in filing order, as (key, item number). The key combines the part, because a 10-Q's
# item numbers repeat across parts ("Item 1" is the statements in Part I and legal
# proceedings in Part II). A 10-K's numbers are unique, so the part is only decoration.
ITEMS: dict[FormType, tuple[tuple[str, str], ...]] = {
    FormType.FORM_10K: tuple(
        (f"part_{part}_item_{item}", item)
        for part, items in (
            ("i", "1 1a 1b 1c 2 3 4"),
            ("ii", "5 6 7 7a 8 9 9a 9b 9c"),
            ("iii", "10 11 12 13 14"),
            ("iv", "15 16"),
        )
        for item in items.split()
    ),
    FormType.FORM_10Q: tuple(
        (f"part_{part}_item_{item}", item)
        for part, items in (("i", "1 2 3 4"), ("ii", "1 1a 2 3 4 5 6"))
        for item in items.split()
    ),
}

# A block that looks like an item heading ("Item 7A. Quantitative..."). Over 200 characters
# it is a sentence that happens to start with "Item", e.g. a cross-reference in the text.
ITEM_CANDIDATE = re.compile(r"^item\s*(\d{1,2}[a-c]?)\b", re.I)
ITEM_CANDIDATE_MAX_CHARS = 200
# A heading ending in a page number ("Item 7 ... 32", "... FS-4") is a table of contents entry.
TOC_ENTRY = re.compile(r"\s(fs-)?\d{1,3}$", re.I)
# A table of contents has items close together. The first item with this much text after
# it is the first in the body. A longer first item can't be missed: its own text counts.
FIRST_BODY_ITEM_MIN_CHARS = 500

# A short block that repeats this many times without sentence punctuation is a running
# header or footer (a filing has dozens of pages, so ten repeats is rarely real text).
# Chosen by hand; a section heading would repeat once or twice. Known gap: headers that
# repeat fewer times than this, or vary a little, stay in the text (Hershey, Western
# Digital, Regis in the held-out tests).
RUNNING_HEADER_MIN_REPEATS = 10
RUNNING_HEADER_MAX_CHARS = 120

# Footers that carry a page number plus a fixed label, like "2025 FORM 10-K 36" or
# "36 | Company". Needs five matching labels before it is treated as a footer, to avoid
# mistaking a few coincidental lines for one.
PAGE_FOOTER_BEFORE = re.compile(r"^(\d{1,3})\s+(.{3,40})$")
PAGE_FOOTER_AFTER = re.compile(r"^(.{3,40}?)\s+(\d{1,3})$")
PAGE_FOOTER_MIN_REPEATS = 5
# GE-style layout: a table titled "Cross-reference index" lists each item with its pages.
INDEX_TITLE = re.compile(r"cross.?reference index", re.I)
# An index table needs at least four items to count; fewer is probably some other table.
INDEX_MIN_ITEMS = 4
PART_ROW = re.compile(r"^part\s+(i{1,3}v?)\b", re.I)
# "36", "36-40": one page or a range, not part of a longer number or a bracket.
PAGE_RANGE = re.compile(r"(?<![\w(])(\d{1,3})(?:\s*[-–—]\s*(\d{1,3}))?(?![\w)])")
# "(a)" after a page spec points to a footnote under the index.
FOOTNOTE_MARKER = re.compile(r"\(([a-z])\)")
# A block counts as just a heading if it is at most this many characters longer than the title.
TITLE_BLOCK_SLACK_CHARS = 10

# An item that renders to no more than this many characters is a pointer ("see page 36",
# "incorporated by reference"), not the item's real text. A real MD&A or set of statements is
# many times longer. Chosen by hand.
POINTER_MAX_CHARS = 2_000
MDNA_START = re.compile(
    r"^management.s discussion and analysis"
    r"( of financial condition and results of operations)?( \(md&a\))?"
    r"( financial table of contents)?\W*(fs-)?\d*$",
    re.I,
)
# JPMorgan and Chevron layout: MD&A and the statements sit later in the filing under their own
# headings, not under Item 7/8. MD&A_START finds the former, STATEMENTS_START and NOTES_START
# the latter. A heading match must be the whole block, so a sentence that mentions them is ignored.
# Where the statements begin. MD&A ends here and Item 8 starts here.
STATEMENTS_START = re.compile(
    r"^(report of (independent|management)|reports? of independent"
    r"|management.s (report on|responsibility))",
    re.I,
)
NOTES_START = re.compile(
    r"^(note\s*1\b|notes? to (the )?(condensed )?consolidated financial statements"
    r"( financial table of contents)?\W*$)",
    re.I,
)
# Item 8 includes supplementary data, so that isn't an ending.
NOTES_END = re.compile(r"^(glossary|exhibit index|signatures?\b|item\s*15\b)", re.I)


@dataclass(frozen=True)
class FallbackResult:
    """Item name to text, in the same shape edgartools returns, plus the layout that was read."""

    rendered: dict[str, str]
    layout: str


def _flush(buffer: list[str], blocks: list[str]) -> None:
    """Turn the text gathered so far into one block, if it has any, and start a new one."""
    text = clean_text("".join(buffer))
    if text:
        blocks.append(text)
    buffer.clear()


def _is_hidden(element: lxml.html.HtmlElement) -> bool:
    """Inline XBRL header data and display:none elements are not visible text."""
    style = (element.get("style") or "").replace(" ", "").lower()
    return element.tag == "ix:header" or "display:none" in style


def _has_block(element: lxml.html.HtmlElement) -> bool:
    """True if anything inside the element starts a new line (so it can't be one block)."""
    return next(element.iter(*BLOCK_TAGS), None) is not None


def _collect(element: lxml.html.HtmlElement, blocks: list[str]) -> None:
    buffer = [element.text or ""]
    for child in element:
        if not isinstance(child.tag, str) or _is_hidden(child):
            buffer.append(child.tail or "")
            continue
        # A table row is one block, unless it wraps a nested table.
        if child.tag == "tr" and next(child.iter("table"), None) is None:
            _flush(buffer, blocks)
            cells = (clean_text(cell.text_content()) for cell in child)
            _flush([" ".join(cell for cell in cells if cell)], blocks)
            buffer = [child.tail or ""]
        elif _has_block(child):
            _flush(buffer, blocks)
            _collect(child, blocks)
            buffer = [child.tail or ""]
        else:
            buffer += [child.text_content(), child.tail or ""]
    _flush(buffer, blocks)


def extract_blocks(root: lxml.html.HtmlElement) -> list[str]:
    """The filing's visible text as a list of blocks, in document order."""
    blocks: list[str] = []
    _collect(root, blocks)
    return blocks


def _running_headers(blocks: list[str]) -> set[str]:
    """Blocks repeated on page after page, including numbered page footers."""
    counts = Counter(block for block in blocks if len(block) <= RUNNING_HEADER_MAX_CHARS)
    # Blocks ending like a sentence are excluded: a short sentence that repeats is real text.
    repeated = {
        block
        for block, count in counts.items()
        if count >= RUNNING_HEADER_MIN_REPEATS and not block.endswith((".", ";", ":"))
    }
    footer = _page_footer_pattern(blocks)
    return repeated | ({block for block in blocks if footer.match(block)} if footer else set())


def _join(blocks: list[str], skip: set[str], title: str = "") -> str:
    """Blocks as blank-line separated paragraphs, minus running headers, under an optional title."""
    kept = [block for block in blocks if block not in skip]
    return "\n\n".join([title, *kept] if title else kept)


def _page_footer_label(blocks: list[str]) -> str | None:
    """The text that repeats beside page numbers, if there is a clear winner."""
    labels: Counter[str] = Counter()
    for block in blocks:
        before = PAGE_FOOTER_BEFORE.match(block)
        after = PAGE_FOOTER_AFTER.match(block)
        if before:
            labels[before.group(2)] += 1
        elif after:
            labels[after.group(1)] += 1
    if not labels:
        return None
    label, count = labels.most_common(1)[0]
    return label if count >= PAGE_FOOTER_MIN_REPEATS else None


def _page_footer_pattern(blocks: list[str]) -> re.Pattern[str] | None:
    """A pattern matching this filing's page footer in either order (number first or last)."""
    label = _page_footer_label(blocks)
    if label is None:
        return None
    escaped = re.escape(label)
    return re.compile(rf"^(?:(\d{{1,3}})\s+{escaped}|{escaped}\s+(\d{{1,3}}))$")


def split_pages(blocks: list[str]) -> dict[int, list[str]]:
    """Group blocks by page, using a repeating footer such as `2025 FORM 10-K 36`."""
    footer = _page_footer_pattern(blocks)
    if footer is None:
        return {}
    pages: dict[int, list[str]] = {}
    current: list[str] = []
    for block in blocks:
        match = footer.match(block)
        # The footer closes a page: everything since the last footer belongs to that page.
        if match:
            pages.setdefault(int(match.group(1) or match.group(2)), []).extend(current)
            current = []
        else:
            current.append(block)
    return pages


def parse_page_ranges(spec: str) -> list[list[int]]:
    """Split `4-7, 9, 71-73` into runs of pages: [[4, 5, 6, 7], [9], [71, 72, 73]]."""
    return [
        list(range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1))
        for m in PAGE_RANGE.finditer(spec)
    ]


@dataclass(frozen=True)
class _IndexRow:
    key: str
    number: str
    title: str
    spec: str


def _index_rows(root: lxml.html.HtmlElement, form: FormType) -> list[_IndexRow]:
    """Read the cross-reference index table: one row per item, with its title and pages."""
    item_keys = {key for key, _ in ITEMS[form]}
    for table in root.iter("table"):
        rows = table_rows(table)
        if not rows or not any(INDEX_TITLE.search(cell) for cell in rows[0]):
            continue
        parsed: list[_IndexRow] = []
        part = "i"
        for row in rows[1:]:
            part_match = PART_ROW.match(row[0])
            if part_match:
                part = part_match.group(1).lower()
                continue
            item_match = ITEM_CANDIDATE.match(row[0])
            if item_match and len(row) >= 2:
                key = _key_for_index_row(item_match.group(1).lower(), part, form)
                if key in item_keys:
                    title = " ".join(row[1:-1])
                    parsed.append(_IndexRow(key, row[0], title, row[-1]))
        if len(parsed) >= INDEX_MIN_ITEMS:
            return parsed
    return []


def _key_for_index_row(item: str, part: str, form: FormType) -> str:
    """The item key for an index row. A 10-K's numbers are unique; a 10-Q's need the part."""
    if form is FormType.FORM_10K:
        return next((key for key, number in ITEMS[form] if number == item), "")
    return f"part_{part}_item_{item}"


def _footnote_text(blocks: list[str], marker: str) -> str | None:
    """Index footnotes follow the index, so the last matching block is the right one."""
    prefix = f"({marker})"
    return next((block for block in reversed(blocks) if block.startswith(prefix)), None)


def _letters(text: str) -> str:
    """Letters only, lowercase, so a title matches whatever spacing and punctuation surrounds it."""
    return re.sub(r"[^a-z]", "", text.lower())


def _opens_with_title(block: str, title: str) -> bool:
    """True when a block is a heading for this title, not a sentence starting with its words.

    GE runs its capitalized headings into the text, as in "RISK FACTORS. The following...".
    """
    if not _letters(block).startswith(_letters(title)):
        return False
    return len(block) <= len(title) + TITLE_BLOCK_SLACK_CHARS or block[: len(title)].isupper()


def _trim_range(
    pages: dict[int, list[str]], page_run: list[int], own_title: str, other_titles: list[str]
) -> list[str]:
    """Take a run of whole pages, then cut text that shares its first or last page.

    An index gives pages, not positions, so a page can hold the end of one topic and the
    start of the next. Titles mark the real boundaries, when they appear.
    """
    blocks = [block for page in page_run for block in pages.get(page, ())]
    # Cut the start: skip the previous topic's text on the first page, up to this topic's title.
    for index in range(len(pages.get(page_run[0], ()))):
        if own_title and _opens_with_title(blocks[index], own_title):
            blocks = blocks[index:]
            break
    # Cut the end: stop where another topic's title appears on the last page.
    others = [title for title in other_titles if title and title != own_title]
    last_page_start = max(1, len(blocks) - len(pages.get(page_run[-1], ())))
    for index in range(last_page_start, len(blocks)):
        if any(_opens_with_title(blocks[index], title) for title in others):
            return blocks[:index]
    return blocks


def _render_from_index(root: lxml.html.HtmlElement, form: FormType) -> dict[str, str] | None:
    rows = _index_rows(root, form)
    if not rows:
        return None
    blocks = extract_blocks(root)
    pages = split_pages(blocks)
    if not pages:
        return None
    titles = [row.title for row in rows]
    rendered = {}
    # Each item may span several page runs ("4-7, 71-73"), so each run is trimmed separately.
    for row in rows:
        selected = [
            block
            for page_run in parse_page_ranges(row.spec)
            for block in _trim_range(pages, page_run, row.title, titles)
        ]
        footnotes = [_footnote_text(blocks, m) for m in FOOTNOTE_MARKER.findall(row.spec)]
        selected += [note for note in footnotes if note]
        if not selected:
            selected = [row.spec]
        rendered[row.key] = "\n\n".join([f"{row.number} {row.title}", *selected])
    return rendered


def _item_ranks(item: str, form: FormType) -> list[int]:
    """Positions of an item number in the filing's item order. A 10-Q can have two."""
    return [rank for rank, (_, number) in enumerate(ITEMS[form]) if number == item]


def _item_candidates(blocks: list[str], form: FormType) -> list[tuple[int, str]]:
    """Every block that looks like a real item heading, as (block index, item number)."""
    candidates = []
    for index, block in enumerate(blocks):
        match = ITEM_CANDIDATE.match(block)
        if match and len(block) <= ITEM_CANDIDATE_MAX_CHARS and not TOC_ENTRY.search(block):
            item = match.group(1).lower()
            if _item_ranks(item, form):
                candidates.append((index, item))
    return candidates


def _skip_table_of_contents(
    candidates: list[tuple[int, str]], blocks: list[str]
) -> list[tuple[int, str]]:
    """Drop the table of contents: start at the first item with real text after it."""
    for position, (index, _) in enumerate(candidates):
        end = candidates[position + 1][0] if position + 1 < len(candidates) else len(blocks)
        if sum(len(block) for block in blocks[index + 1 : end]) >= FIRST_BODY_ITEM_MIN_CHARS:
            return candidates[position:]
    return []


def locate_items(blocks: list[str], form: FormType) -> dict[str, tuple[int, int]]:
    """Find each item's block range, as the longest run of headings in filing order.

    Taking the longest run means a stray "Item 8" in the text can't derail the items
    after it. Ties go to the earliest heading, so a repeated "Item 7 (continued)"
    can't move an item's start.
    """
    candidates = _skip_table_of_contents(_item_candidates(blocks, form), blocks)
    # Each state is (candidate position, rank). 10-Q item numbers repeat across parts,
    # so one candidate can hold several ranks.
    states = [
        (position, rank)
        for position, (_, item) in enumerate(candidates)
        for rank in _item_ranks(item, form)
    ]
    # Dynamic programming for the longest run of headings in filing order: for each state,
    # the best run length ending there and the state before it.
    best: dict[tuple[int, int], tuple[int, tuple[int, int] | None]] = {}
    for state in states:
        position, rank = state
        best[state] = (1, None)
        for earlier in states:
            if earlier[0] < position and earlier[1] < rank:
                length = best[earlier][0] + 1
                if length > best[state][0]:
                    best[state] = (length, earlier)
    if not best:
        return {}
    # Start from the end of the longest run (the earliest position wins ties), and walk back.
    state: tuple[int, int] | None = max(best, key=lambda s: (best[s][0], -s[0]))
    chain = []
    while state is not None:
        chain.append(state)
        state = best[state][1]
    chain.reverse()

    # Each item runs from its heading to the next heading in the chain.
    spans = {}
    for order, (position, rank) in enumerate(chain):
        start = candidates[position][0]
        end = candidates[chain[order + 1][0]][0] if order + 1 < len(chain) else len(blocks)
        spans[ITEMS[form][rank][0]] = (start, end)
    return spans


def _first_after(blocks: list[str], start: int, pattern: re.Pattern[str]) -> int | None:
    """Index of the first block at or after `start` that matches, or None."""
    return next((i for i in range(start, len(blocks)) if pattern.match(blocks[i])), None)


def _pointer_items(form: FormType) -> tuple[str, str]:
    """The items holding MD&A and the financial statements."""
    if form is FormType.FORM_10K:
        return "part_ii_item_7", "part_ii_item_8"
    return "part_i_item_2", "part_i_item_1"


def _resolve_pointers(
    blocks: list[str],
    spans: dict[str, tuple[int, int]],
    rendered: dict[str, str],
    skip: set[str],
    form: FormType,
) -> bool:
    """Replace MD&A and notes items that only point elsewhere with the text they point to.

    Returns whether any were replaced.
    """
    mdna_item, notes_item = _pointer_items(form)
    # An item only needs resolving if it is short enough to be a pointer (POINTER_MAX_CHARS).
    resolved = False
    section_start: int | None = None

    def is_pointer(item: str) -> bool:
        return item in spans and len(rendered[item]) <= POINTER_MAX_CHARS

    statements_start = 0
    # MD&A: runs from its own heading later in the filing to where the statements begin.
    if is_pointer(mdna_item):
        start = _first_after(blocks, spans[mdna_item][1], MDNA_START)
        end = _first_after(blocks, start + 1, STATEMENTS_START) if start is not None else None
        if start is not None and end is not None:
            rendered[mdna_item] = _join(blocks[start:end], skip, blocks[spans[mdna_item][0]])
            resolved = True
            statements_start = end
            section_start = start
    # Notes: start at the statements (or at "Note 1" if earlier), end at the glossary,
    # exhibit list or signatures.
    if is_pointer(notes_item):
        begin = max(statements_start, spans[notes_item][1])
        candidates = [
            found
            for found in (
                statements_start or _first_after(blocks, begin, STATEMENTS_START),
                _first_after(blocks, begin, NOTES_START),
            )
            if found
        ]
        if candidates:
            start = min(candidates)
            notes_begin = _first_after(blocks, start, NOTES_START) or start
            end = _first_after(blocks, notes_begin + 1, NOTES_END) or len(blocks)
            rendered[notes_item] = _join(blocks[start:end], skip, blocks[spans[notes_item][0]])
            resolved = True
            section_start = start if section_start is None else section_start
    # The text just taken came out of whichever item sat before it; cut it back out of that item.
    if section_start is not None:
        _end_item_at(section_start, blocks, spans, rendered, skip, keep={mdna_item, notes_item})
    return resolved


def _end_item_at(
    position: int,
    blocks: list[str],
    spans: dict[str, tuple[int, int]],
    rendered: dict[str, str],
    skip: set[str],
    keep: set[str],
) -> None:
    """A financial section sits inside whichever item comes before it. Take it back out."""
    for item, (start, end) in spans.items():
        if start < position < end and item not in keep:
            rendered[item] = _join(blocks[start:position], skip)


def parse_fallback(stripped_html: str, form: FormType) -> FallbackResult:
    """Render a filing's items from HTML that has already had its data tables stripped."""
    root = parse_html_tree(stripped_html)
    # Try the layouts in the order of the module docstring. The index is the most specific
    # (it needs a titled table), so it goes first; plain headings are the common case.
    from_index = _render_from_index(root, form)
    if from_index is not None:
        return FallbackResult(rendered=from_index, layout=LAYOUT_INDEX)

    blocks = extract_blocks(root)
    spans = locate_items(blocks, form)
    skip = _running_headers(blocks)
    rendered = {key: _join(blocks[start:end], skip) for key, (start, end) in spans.items()}
    resolved = _resolve_pointers(blocks, spans, rendered, skip, form)
    return FallbackResult(
        rendered=rendered, layout=LAYOUT_POINTERS if resolved else LAYOUT_HEADINGS
    )
