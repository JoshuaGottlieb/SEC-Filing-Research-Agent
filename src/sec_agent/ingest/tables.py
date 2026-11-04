"""Removes rows of figures from a filing's HTML tables and keeps everything else.

Figures come from XBRL, so data rows are dropped here. Filers also lay out plain text
as tables, and put explanatory footnotes in the same table as the figures, so decisions
are made row by row: a removed table's prose rows move out of it and stay in the text.
Every decision is recorded so that it can be audited later.
"""

from __future__ import annotations

import re
from enum import StrEnum

import lxml.html
from pydantic import BaseModel, ConfigDict

# What can follow a number and still leave the cell a value: basis points, points, a multiple
# ("2.1x"), then a footnote marker "(a)" or "*".
_VALUE_SUFFIX = r"(bps|pts?|x)?\s*(\([a-z]\)|\*+)?"
# A cell that is a number: digits with currency signs, brackets (negatives), dashes, commas
# and percent signs around them. It must contain at least one digit.
NUMERIC_CELL = re.compile(rf"^[\s$€£%()\-–—.,\d]*\d[\s$€£%()\-–—.,\d]*{_VALUE_SUFFIX}\s*$", re.I)
# A cell that is a value slot with no number: a lone "$", a dash, "n/a", "nil", "none".
# Filings split "$ 1,234" across cells, so the bare "$" cells must count as values.
EMPTY_VALUE_CELL = re.compile(r"^([\s$€£%()\-–—]+|n/?[am]\.?|nil|none)$", re.I)
# "Item 7", "ITEM 1A", "Item 8." at the start of a block. Shared with the section checks.
ITEM_HEADING = re.compile(r"^item\s*\d+[a-c]?\b", re.I)
ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")

# The values below were chosen by hand while developing. The held-out tests measure the whole
# gate, not each value on its own, so none has its own measured error rate.

# A row is data when more than half of the cells after the label are values. Half, not all,
# so a row with one stray text cell (a footnote marker, a unit) is still data.
DATA_ROW_MIN_VALUE_SHARE = 0.5
# A table needs at least 2 numeric cells and 15% data rows to be removed. These are low on
# purpose: a table that is mostly prose with a few figures still keeps its prose, because
# prose rows are moved out before the table is dropped. Raising them would leave more figure
# tables in the text, which the leftover-table audit would then flag.
DATA_TABLE_MIN_NUMERIC_CELLS = 2
DATA_TABLE_MIN_ROW_SHARE = 0.15
# A cell of 200 characters or more is prose however it is punctuated. Shorter cells are prose
# only if they read as a sentence: 8 or more words and sentence-ending punctuation. Row labels
# are often long but rarely end with a period, which is the main way they are kept apart.
# (A label that does read as a sentence is the known cause of some false rejections: CapForce.)
NARRATIVE_CELL_MIN_CHARS = 200
SENTENCE_MIN_WORDS = 8
SENTENCE_ENDINGS = (".", ";", "?", "!", '."', ".”", ".)")


class RowKind(StrEnum):
    """What a table row is. Only DATA rows are dropped; PROSE rows are kept as text."""

    DATA = "data"
    PROSE = "prose"
    # A heading or caption row with no figures and no sentence. Dropped with its table.
    LABEL = "label"


class TableRecord(BaseModel):
    """What a table held and what was done with it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    removed: bool
    rows: tuple[tuple[str, ...], ...]
    # Prose rows moved out of a removed table into the text.
    preserved_rows: tuple[str, ...] = ()
    numeric_cells: int = 0
    cell_count: int = 0
    # Inline XBRL tagged numbers in the table. The audit uses them as independent evidence
    # that a kept table is really figures.
    tagged_values: int = 0

    @property
    def excerpt(self) -> str:
        # The first 200 characters, for problem messages.
        return " ".join(cell for row in self.rows for cell in row)[:200]


class StripResult(BaseModel):
    """The filing's HTML without data tables, and a record of every table seen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    html: str
    tables: tuple[TableRecord, ...]

    @property
    def removed(self) -> tuple[TableRecord, ...]:
        return tuple(table for table in self.tables if table.removed)

    @property
    def kept(self) -> tuple[TableRecord, ...]:
        return tuple(table for table in self.tables if not table.removed)


def is_numeric_cell(text: str) -> bool:
    """True for a cell holding a figure. A bare dash or "$" is not numeric."""
    return bool(NUMERIC_CELL.match(text))


def _is_value_cell(text: str) -> bool:
    """A figure or an empty value slot ("$", "—", "n/a")."""
    return bool(NUMERIC_CELL.match(text) or EMPTY_VALUE_CELL.match(text))


def is_prose_cell(text: str) -> bool:
    """A cell written as a sentence, as opposed to a label such as a line item name."""
    if len(text) >= NARRATIVE_CELL_MIN_CHARS:
        return True
    words = re.findall(r"[A-Za-z]{2,}", text)
    return len(words) >= SENTENCE_MIN_WORDS and text.endswith(SENTENCE_ENDINGS)


def classify_row(cells: list[str]) -> RowKind:
    """The first cell is a label or list marker, so it doesn't count as a value.

    A label's length doesn't make a row prose when everything beside it is a number. National
    Vision's "Common stock, $0.01 par value; 200,000,000 shares authorized; ..." runs past
    270 characters, and it's still a balance-sheet row.
    """
    values = cells[1:]
    # Every cell after the label is a value and at least one is a real number: data.
    if (
        values
        and all(_is_value_cell(cell) for cell in values)
        and any(is_numeric_cell(cell) for cell in values)
    ):
        return RowKind.DATA
    # Otherwise a short row is still data if most of its cells are values. A row with a
    # 200-character cell is not tried here, so a paragraph beside a figure stays prose.
    if values and max(len(cell) for cell in cells) < NARRATIVE_CELL_MIN_CHARS:
        value_share = sum(1 for cell in values if _is_value_cell(cell)) / len(values)
        if value_share > DATA_ROW_MIN_VALUE_SHARE:
            return RowKind.DATA
    return RowKind.PROSE if any(is_prose_cell(cell) for cell in cells) else RowKind.LABEL


def clean_text(text: str) -> str:
    """Collapse whitespace. Some filers pad with zero-width spaces, which aren't whitespace."""
    return " ".join(ZERO_WIDTH.sub(" ", text).split())


def _row_cells(row: lxml.html.HtmlElement) -> list[str]:
    cells = (clean_text(cell.text_content()) for cell in row if cell.tag in ("td", "th"))
    # Empty cells are layout spacers, so they are dropped.
    return [cell for cell in cells if cell]


def table_rows(table: lxml.html.HtmlElement) -> list[list[str]]:
    """Non-empty rows, including those of nested tables but not the rows that wrap them."""
    rows = []
    for row in table.iter("tr"):
        # A row that contains a table is just a wrapper; its inner rows are visited anyway.
        if next(row.iter("table"), None) is not None:
            continue
        cells = _row_cells(row)
        if cells:
            rows.append(cells)
    return rows


def is_data_table(table: lxml.html.HtmlElement) -> bool:
    """True when a table is mainly rows of figures. False when it's used to lay out text."""
    rows = table_rows(table)
    if not rows:
        return False
    # A table of contents has item names beside page numbers, but section detection
    # relies on it. It sits before the first item, so it can't reach a section's text.
    if any(ITEM_HEADING.match(cell) for row in rows for cell in row):
        return False
    # The first cell of each row is its label, so only later cells count as figures.
    numeric = sum(1 for row in rows for cell in row[1:] if is_numeric_cell(cell))
    data_rows = sum(1 for row in rows if classify_row(row) is RowKind.DATA)
    return (
        numeric >= DATA_TABLE_MIN_NUMERIC_CELLS
        and data_rows / len(rows) >= DATA_TABLE_MIN_ROW_SHARE
    )


def _record(index: int, table: lxml.html.HtmlElement, removed: bool) -> TableRecord:
    rows = table_rows(table)
    # Prose is only preserved from tables that are being removed; a kept table stays whole.
    preserved = (
        [" ".join(row) for row in rows if classify_row(row) is RowKind.PROSE] if removed else []
    )
    return TableRecord(
        index=index,
        removed=removed,
        rows=tuple(tuple(row) for row in rows),
        preserved_rows=tuple(preserved),
        numeric_cells=sum(1 for row in rows for cell in row if is_numeric_cell(cell)),
        cell_count=sum(len(row) for row in rows),
        # ix:nonfraction is the inline XBRL element that tags a number.
        tagged_values=sum(1 for _ in table.iter("ix:nonfraction")),
    )


def parse_html_tree(html: str) -> lxml.html.HtmlElement:
    # Bytes, because filings open with an XML encoding declaration that lxml rejects on str.
    # huge_tree lifts lxml's default limits, which large 10-Ks exceed.
    parser = lxml.html.HTMLParser(encoding="utf-8", huge_tree=True)
    return lxml.html.fromstring(html.encode("utf-8"), parser=parser)


def strip_data_tables(html: str) -> StripResult:
    """Remove data tables from HTML, moving any prose rows out of them first."""
    root = parse_html_tree(html)
    records = []
    # Outermost first: a removed table takes its nested tables with it.
    for table in list(root.iter("table")):
        if table.getparent() is None or any(a.tag == "table" for a in table.iterancestors()):
            continue
        record = _record(len(records), table, removed=is_data_table(table))
        records.append(record)
        if record.removed:
            # Put each prose row in a paragraph just before the table, then drop the table.
            for text in record.preserved_rows:
                paragraph = lxml.html.Element("p")
                paragraph.text = text
                table.addprevious(paragraph)
            table.drop_tree()
    return StripResult(html=lxml.html.tostring(root, encoding="unicode"), tables=tuple(records))
