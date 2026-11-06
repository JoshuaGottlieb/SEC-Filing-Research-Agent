"""Turns a filing's data tables into facts: a labeled number for a known period.

Parsing never looks at XBRL tags, so tagged cells can serve as an answer key for it, and
it behaves the same on the untagged tables that are the point of this module. A table
only yields facts if every part of it can be explained. Anything else is marked not clean
with its reasons, and goes to the chunker as text instead of being guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import lxml.html
from pydantic import BaseModel, ConfigDict

from sec_agent.edgar.xbrl import (
    InlineFact,
    XbrlContext,
    duration_of,
    read_contexts,
    read_inline_fact,
)
from sec_agent.ingest.fallback import extract_blocks
from sec_agent.ingest.tables import clean_text, is_data_table, parse_html_tree
from sec_agent.schemas import Duration, FormType

UNIT_USD = "USD"
UNIT_PERCENT = "percent"

# What each stated scale multiplies a displayed number by: "(in thousands)" turns 1,234 into
# 1,234,000. Words and the shorthand filers use ("000", "MM", "bn") map to the same sizes.
# A bare "m", "b" or "k" is only accepted after a "$" (see SCALE_CAPTION).
SCALES = {
    "thousands": 1_000.0,
    "000": 1_000.0,
    "k": 1_000.0,
    "millions": 1_000_000.0,
    "000,000": 1_000_000.0,
    "mm": 1_000_000.0,
    "mil": 1_000_000.0,
    "mils": 1_000_000.0,
    "m": 1_000_000.0,
    "billions": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "b": 1_000_000_000.0,
}
# The word, or shorthand such as "(in 000s)", "$000's", "(000,000)", "($ in MM)" and "$M". A bare
# 000 needs a marker, because "Commission File No. 000-12345" isn't a scale.
SCALE_CAPTION = re.compile(
    r"\b(thousands|millions|billions)\b"
    r"|(?:[($‘’']\s*|\bin\s+)(000,000|000)(?:[‘’']?s)?(?![\d‘’']|[,.\-–]\d)"
    r"|(?<![\d,.\-–])(000)[‘’']?s\b"
    r"|(?:\bin\s+|\$\s*)(mm|mils?|bn)\b"
    r"|\$\s*(m|b|k)\b",
    re.I,
)
# A caption longer than this is a sentence, not a unit label, so it only counts if it ends
# with the scale (SCALE_AT_END). This stops "revenue rose to millions of subscribers" from
# setting a table's scale. 160 is roughly one line of a caption; chosen by hand.
SCALE_CAPTION_MAX_CHARS = 160
# A sentence can state a scale, as in "...expense for 2026 and 2025 (in millions):".
SCALE_AT_END = re.compile(
    r"\bin\s+(thousands|millions|billions|[‘’']?000(,000)?([‘’']?s)?|mm|mils?|bn)\b"
    r"[^.]{0,40}\)?\s*:?\s*$",
    re.I,
)
SENTENCE_END = (".", "!", "?")
# What a header can say around shorthand and still mean the whole table. "Shares (000s)" scales
# one column, so it says nothing about the table's dollars.
SCALE_FILLER = frozenset({"amount", "amounts", "dollar", "dollars", "in", "of", "omitted", "usd"})
# How many text blocks before a table are kept as its context. The caption, title and
# "in millions" line usually sit within the last three blocks; further back is unrelated text.
CONTEXT_BLOCKS = 3
# A placeholder paragraph inserted where a table was, so the text before it can be found.
TABLE_MARKER = "@@TABLE:{}@@"
TABLE_MARKER_PATTERN = re.compile(r"^@@TABLE:(\d+)@@$")

# A displayed figure: "1,234", "(5.2)", "$ 12", "45%", with an optional footnote marker.
VALUE = re.compile(
    r"^\(?-?\$?\s*\d[\d,]*(\.\d+)?\s*%?\s*\)?\s*%?\s*(\([a-z]\)|\*+)?$",
    re.I,
)
# A dash in a figures column means zero.
DASH = re.compile(r"^[\s$()]*[-–—]+[\s$()]*$")
# Cells holding only "$", "%" or brackets: pieces of a figure that a filer split across cells.
SYMBOL_ONLY = re.compile(r"^[\s$€£%()]+$")
# Cells that may sit among a row's figures without changing what the row means.
HARMLESS_CELL = re.compile(r"^(n/?[am]\.?|nil|none|\([a-z0-9]\)|\*+)$", re.I)
# A header cell that is just a year ("2025", "Fiscal 2025"). Years look numeric, so header
# detection has to tell them apart from data.
YEAR_ONLY = re.compile(r"^(fiscal\s+|fy\s*)?(19|20)\d{2}\W*$", re.I)

# Month names and abbreviations to month numbers, for reading dates in column headers.
MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
# "June 30, 2025" (with or without the comma).
FULL_DATE = re.compile(r"\b([A-Za-z]{3,9})\.?\s*(\d{1,2})\s*,?\s*((?:19|20)\d{2})\b")
YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
# "July 1" with no year, as in a row label "Balance at July 1".
MONTH_DAY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})\b(?!\s*,?\s*\d{4})")
# Columns that compare periods instead of reporting one ("% change", "increase", "bps").
# They carry no period of their own, so they are skipped rather than stored as facts.
CHANGE_COLUMN = re.compile(
    r"chang|varianc|increas|decreas|\bvs\.?\b|versus|basis point|\bbps\b|favorab|\bbetter\b"
    r"|\bworse\b|\bpercent\b|^\W*%\W*$|impact|growth",
    re.I,
)
# "% of revenue" is a share, not a change, so it is kept (as a percent fact).
SHARE_OF_COLUMN = re.compile(r"%\s*of\b|percent(age)?\s+of\b|as a %", re.I)
# MKS's plan asset tables have a "Percentage" column with no % sign beside the values.
PERCENT_COLUMN = re.compile(r"\bpercentages?\b|%", re.I)
# The three phrases below name a column's duration. Each is matched on its own; a header that
# matches more than one ("Three and six months ended") is ambiguous and is not guessed at.
# 13 weeks, 26 weeks and 39 weeks are the 52/53-week equivalents of 3, 6 and 9 months.
THREE_MONTHS = re.compile(
    r"\b(three|3)[- ]months?\b|\b(three|3)\s+and\s+(six|nine|6|9)\b|\bquarter\b"
    r"|\b(thirteen|13)[- ]weeks?\b|\bq[1-4]\b",
    re.I,
)
YEAR_TO_DATE = re.compile(
    r"\b(six|nine|6|9)[- ]months?\b|year[- ]to[- ]date|\bytd\b|\b(26|39|twenty-six|thirty-nine)"
    r"[- ]weeks?\b",
    re.I,
)
TWELVE_MONTHS = re.compile(
    r"\byears?\s+ended\b|\b(twelve|12)[- ]months?\b|\bfiscal\s+years?\b|\b5[23][- ]weeks?\b",
    re.I,
)
# How far a computed period end may be from a date the filing really reports, and still be
# snapped to it. Filers with 52/53-week years end on a different day each year (Western
# Digital's fiscal 2025 ended June 27, not June 30). Also the margin by which a column may
# be dated after the filing's own period end, and how far a tag's date may differ from the
# parsed date. Chosen by hand; ten days covers a week-based year-end without merging periods.
PERIOD_END_TOLERANCE_DAYS = 10

# Quantities that are never dollars, whatever the table's caption says. Without this, a
# table "in millions" turns 94,563 employees into $94 billion.
NON_DOLLAR_ROW = re.compile(
    r"\bemployees\b|\bheadcount\b|\bstores\b|\blocations\b|\bbarrels?\b|\bwells\b|\bacres\b"
    r"|\bcustomers\b|\bclients\b|\baccounts\b(?! (receivable|payable))|\bdays\b|\byears\b"
    r"|\b(lease|remaining|contract\w*) term\b|\bnumber of\b|\bvolumes?\b|\btons\b|\bsquare f"
    r"|\bweighted[- ]average\b"
    r"|\bbasis points\b|\bcapacity\b|\btimes\b|\bcoverage\b",
    re.I,
)
# A table is in dollars only on positive evidence. A caption or a "$" in the table gives it;
# a caption naming barrels, tons, units or employees takes it away.
DOLLAR_CAPTION = re.compile(r"\bdollars?\b|\$|\busd\b", re.I)
NON_DOLLAR_CAPTION = re.compile(
    r"\bbarrels?\b|\bcubic\b|\bmbd\b|\bmmcfd\b|\bboe\b|\btons\b|\bunits\b|\bemployees\b", re.I
)
# Row labels that are rates, not amounts: "margin", "rate", "yield", "return on".
PERCENT_LABEL = re.compile(
    r"%|\bpercent|\bmargins?\b|\brates?\b|\bratios?\b|\byields?\b|\breturn on\b", re.I
)
# "per share", "per diluted share", "per unit", "EPS".
PER_SHARE_LABEL = re.compile(r"\bper\s+\w*\s*\w*\s*(share|unit|ads)\b|\beps\b", re.I)
# Share counts and per-share amounts have unreliable captions ("in millions, except per
# share") and are nearly always in XBRL, so tables holding them stay as text.
SHARE_ROW = re.compile(r"\bshares?\b|\bdilut\w+\b|\bstock\b|\bunits\b", re.I)
# Column headings that mean share counts or prices, whose scale the table caption can't be
# trusted for.
PRICE_COLUMN = re.compile(r"\bshares?\b|\bprice\b|\bgrant[- ]date\b|\bfair value per\b", re.I)
# A dollar table's values share a precision. EPS of 3.46 in a table of whole millions is
# a different kind of number, even when its row label doesn't say so. Two decimals is
# the usual precision of a per-share amount, so a cell with two or more is "fine".
FINE_DECIMALS = 2
# These rows report the start of a period under a column naming its end.
OPENING_BALANCE_LABEL = re.compile(r"\b(beginning|opening|start)\b|\bat january 1\b", re.I)
# Text that is not a table title: "(Table continued)", a form name from a running header, or
# a bare page number.
NOT_A_TITLE = re.compile(r"table continued|form 10-[kq]|^\W*\d+\W*$", re.I)
# If more than a quarter of the data rows have no label, the labels were probably lost in
# the layout, so the facts can't be trusted and the whole table goes to text. Chosen by hand.
UNLABELED_ROWS_MAX_SHARE = 0.25


class TableFact(BaseModel):
    """One number from a clean table, with the row, column and period that explain it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # "<accession>:t<table>:r<row>:c<column position>", unique within the filing.
    fact_id: str
    table_index: int
    row_index: int
    table_title: str | None
    # A row of one cell above the data rows ("Segment revenue:") that groups what follows.
    row_group: str | None
    row_label: str
    column_label: str
    start_date: date | None
    end_date: date
    duration: Duration | None
    # The number as displayed, before scaling.
    raw_text: str
    # In dollars (after applying the table's scale) or in percent, with the displayed sign.
    value: float
    unit: str
    # True for "% of revenue" style columns.
    is_share_of: bool = False
    # What XBRL says about this cell, when it's tagged. Parsing never reads these.
    xbrl_concept: str | None = None
    xbrl_value: float | None = None
    xbrl_end_date: date | None = None
    xbrl_context_id: str | None = None

    @property
    def tagged(self) -> bool:
        return self.xbrl_concept is not None


class ParsedTable(BaseModel):
    """A data table and the outcome of parsing it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table_index: int
    title: str | None
    # True only when every part of the table was explained. A clean table's facts are stored;
    # a table that isn't clean has no facts and waits as text, with `reasons` saying why.
    clean: bool
    reasons: tuple[str, ...]
    facts: tuple[TableFact, ...]
    rows: tuple[tuple[str, ...], ...]
    # The text blocks just before the table (captions, units), kept for the chunker.
    preceding: tuple[str, ...] = ()
    # Rows with figures that yielded no fact, such as opening balances.
    skipped_rows: int = 0
    # The filing's own words stating the table's scale, for the chunker: a figure read as text
    # means nothing without them. Empty when none was found, because a scale is never guessed.
    # A table stating two scales keeps every statement, since neither one governs the table.
    scale_captions: tuple[str, ...] = ()

    @property
    def needs_text_chunk(self) -> bool:
        """Every figure must be reachable: as a fact, or else as text for retrieval."""
        return not self.clean or self.skipped_rows > 0

    @property
    def as_text(self) -> str:
        """The whole table as text, for the chunker when it isn't clean."""
        lines = [self.title] if self.title else []
        lines += [" | ".join(row) for row in self.rows]
        return "\n".join(lines)


@dataclass(frozen=True)
class FilingContext:
    """What the parser needs to know about the filing around a table."""

    accession: str
    form: FormType
    fiscal_year: int
    fiscal_period: str
    period_end: date


@dataclass(frozen=True)
class _Cell:
    """A table cell placed by column position (start to end, so colspans line up)."""

    start: int
    end: int
    text: str
    # The inline XBRL tag on the cell's number, if any. Used only to check the parse.
    tag: InlineFact | None = None

    def overlaps(self, other: _Cell) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class _Column:
    """What a column holds: its period, and whether to skip it (change columns)."""

    label: str
    start_date: date | None
    end_date: date | None
    duration: Duration | None
    skip: bool
    share_of: bool = False
    percent: bool = False


def build_grid(table: lxml.html.HtmlElement) -> list[list[_Cell]]:
    """Each row's non-empty cells, placed by column position so that colspans line up."""
    grid = []
    pending: dict[int, int] = {}  # column -> rows still covered by a rowspan
    for row in table.iter("tr"):
        if next(row.iter("table"), None) is not None:
            continue
        position = 0
        cells = []
        next_pending: dict[int, int] = {}
        for cell in row:
            if cell.tag not in ("td", "th"):
                continue
            # Skip columns still covered by a cell from an earlier row (rowspan).
            while pending.get(position, 0) > 0:
                next_pending[position] = pending[position] - 1
                position += 1
            span = int(re.sub(r"\D", "", cell.get("colspan") or "1") or 1)
            rowspan = int(re.sub(r"\D", "", cell.get("rowspan") or "1") or 1)
            text = clean_text(cell.text_content())
            if text:
                element = next(cell.iter("ix:nonfraction"), None)
                tag = read_inline_fact(element) if element is not None else None
                cells.append(_Cell(position, position + span, text, tag))
            if rowspan > 1:
                for column in range(position, position + span):
                    next_pending[column] = rowspan - 1
            position += span
        pending = next_pending
        if cells:
            grid.append(cells)
    return grid


def _is_value(text: str) -> bool:
    """A figure or a dash (zero)."""
    return bool(VALUE.match(text) or DASH.match(text))


def _is_header_row(cells: list[_Cell]) -> bool:
    """Period headers contain years, which look numeric, so they're excluded here."""
    values = [cell for cell in cells[1:] if _is_value(cell.text)]
    return not values or all(YEAR_ONLY.match(cell.text) for cell in values)


def _is_data_row(cells: list[_Cell]) -> bool:
    """A row with a label and figures. Headers and single-cell rows are not data."""
    return bool(cells[1:]) and not _is_header_row(cells)


def parse_number(text: str) -> float:
    """Displayed parentheses mean negative. A dash means zero."""
    if DASH.match(text):
        return 0.0
    digits = re.search(r"\d[\d,]*(\.\d+)?", text)
    assert digits is not None
    number = float(digits.group().replace(",", ""))
    return (
        -number if "(" in text.split(digits.group())[0] or text.lstrip().startswith("-") else number
    )


def reporting_period_ends(contexts: dict[str, XbrlContext], form: FormType) -> set[date]:
    """Dates that close a period this filing reports on, for dating a bare-year column.

    Other dates don't count. Western Digital's fiscal 2025 ended on June 27, and a June 30
    date elsewhere in its 10-K was closer to the estimate, so that column was misdated.
    """
    wanted = (
        {Duration.TWELVE_MONTHS}
        if form is FormType.FORM_10K
        else {Duration.THREE_MONTHS, Duration.YEAR_TO_DATE}
    )
    ends = {
        context.end_date
        for context in contexts.values()
        if not context.dimensions and duration_of(context) in wanted
    }
    return ends or {context.end_date for context in contexts.values()}


def _nearest_known_end(target: date, known_ends: set[date]) -> date:
    """Snap an estimated period end to a date the filing reports, if one is within reach."""
    candidates = [
        known for known in known_ends if abs((known - target).days) <= PERIOD_END_TOLERANCE_DAYS
    ]
    return min(candidates, key=lambda known: abs((known - target).days)) if candidates else target


def _shift_years(day: date, years: int) -> date:
    """The same calendar day some years away. February 29 falls back to the 28th."""
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # February 29
        return day.replace(year=day.year + years, day=28)


def _label_names_another_date(label: str, end: date) -> bool:
    """A row such as "Balance at July 1" reports a different date from its column."""
    if any(day != end for day in _dates_in(label)):
        return True
    for month, day in MONTH_DAY.findall(label):
        number = MONTHS.get(month.lower())
        if number is not None and (number, int(day)) != (end.month, end.day):
            return True
    return False


def _shift_days(day: date, days: int) -> date:
    """Add days to a date (negative to subtract)."""
    return date.fromordinal(day.toordinal() + days)


def _dates_in(text: str) -> list[date]:
    """Every full date ("June 30, 2025") in a piece of text. Impossible dates are ignored."""
    found = []
    for month, day, year in FULL_DATE.findall(text):
        number = MONTHS.get(month.lower())
        if number is None:
            continue
        try:
            found.append(date(int(year), number, int(day)))
        except ValueError:
            continue
    return found


def _one_duration(text: str) -> Duration | None:
    """A duration stated without ambiguity. "Three and six months ended" names two."""
    found = {
        duration
        for duration, pattern in (
            (Duration.THREE_MONTHS, THREE_MONTHS),
            (Duration.YEAR_TO_DATE, YEAR_TO_DATE),
            (Duration.TWELVE_MONTHS, TWELVE_MONTHS),
        )
        if pattern.search(text)
    }
    return found.pop() if len(found) == 1 else None


def parse_column(
    label: str, context: str, filing: FilingContext, known_ends: set[date]
) -> _Column | None:
    """Work out what a column holds. None means it can't be explained.

    Duration comes from the column's own header when it has one, and otherwise from a
    single unambiguous phrase around the table. It describes the column, so a balance
    reported at period end can sit under a flow's duration. `end_date` is the firm part.
    """
    if CHANGE_COLUMN.search(label) and not SHARE_OF_COLUMN.search(label):
        return _Column(label, None, None, None, skip=True)
    share_of = bool(SHARE_OF_COLUMN.search(label))
    percent = bool(PERCENT_COLUMN.search(label))
    dates = _dates_in(label)
    duration = _one_duration(label)
    # The surrounding text only fills in a duration the column itself leaves unstated, and
    # never overrides a column that names an ambiguous one.
    if duration is None and not YEAR_TO_DATE.search(label) and not THREE_MONTHS.search(label):
        duration = _one_duration(context)
    # In a Q1 filing, "year to date" and "three months" are the same period.
    if duration is Duration.YEAR_TO_DATE and filing.fiscal_period == "Q1":
        duration = Duration.THREE_MONTHS
    # Two dates are a period's start and end ("July 1 2024 - June 30 2025").
    if len(dates) == 2:
        first, second = sorted(dates)
        duration_between = _duration_between(first, second)
        return _Column(label, first, second, duration_between, False, share_of, percent)
    if len(dates) > 2:
        return None
    # One date is the end of the period. With no "ended" nearby it is a point in time
    # (a balance sheet date).
    if len(dates) == 1:
        if duration is None and "ended" not in context.lower() and "ended" not in label.lower():
            duration = Duration.INSTANT
        return _Column(label, None, dates[0], duration, False, share_of, percent)
    # A bare year ("2024"): move the filing's own period end by the year difference, then
    # snap it to a date the filing really reports. Several years, or none, can't be dated.
    years = YEAR.findall(label)
    if len(years) != 1:
        return None
    target = _shift_years(filing.period_end, int(years[0]) - filing.fiscal_year)
    end = _nearest_known_end(target, known_ends)
    # A 10-Q's bare-year column could be three months or year to date, so it needs a duration.
    if filing.form is FormType.FORM_10Q and duration is None:
        return None
    return _Column(label, None, end, duration, False, share_of, percent)


def _duration_between(first: date, second: date) -> Duration:
    days = (second - first).days
    # A quarter is about 91 days; a 52 or 53-week year is 364 or 371. The ranges leave room for
    # month-length differences. Anything else (six or nine months) is treated as year to date.
    if 80 <= days <= 100:
        return Duration.THREE_MONTHS
    if 350 <= days <= 380:
        return Duration.TWELVE_MONTHS
    return Duration.YEAR_TO_DATE


def _scale_of(match: re.Match[str]) -> float:
    """The multiplier for whichever alternative of SCALE_CAPTION matched."""
    return SCALES[next(group for group in match.groups() if group).lower()]


def _scale_caption(texts: list[str]) -> tuple[float | None, str]:
    """Find a stated scale. A sentence that merely mentions millions doesn't count.

    In a short caption the first scale governs: "(In millions, except number of shares,
    which are reflected in thousands)" is a table in millions.
    """
    for text in texts:
        if len(text) > SCALE_CAPTION_MAX_CHARS:
            at_end = SCALE_AT_END.search(text)
            match = SCALE_CAPTION.search(at_end.group(0)) if at_end else None
            if match:
                return _scale_of(match), text
            continue
        match = SCALE_CAPTION.search(text)
        # Shorthand only ever states a scale. The words need "in", as in "(in millions)".
        if match and (not match.group(1) or re.search(r"\b(in|of|dollars|\$)\b", text, re.I)):
            return _scale_of(match), text
    return None, ""


def _stated_scale_cells(grid: list[list[_Cell]]) -> list[tuple[float, str]]:
    """Every cell stating a scale, with that scale. Only the first counts in any one cell,
    because "(In millions, except shares, which are in thousands)" is still a table in millions."""
    found = []
    for row in grid:
        for cell in row:
            match = SCALE_CAPTION.search(cell.text)
            if match and not _is_value(cell.text):
                found.append((_scale_of(match), cell.text))
    return found


def _scales_one_column(text: str) -> bool:
    """Shorthand attached to a quantity, as in "Shares (000s)" or "Purchased (000's)"."""
    match = SCALE_CAPTION.search(text)
    if match is None or match.group(1):
        return False
    around = (text[: match.start()] + " " + text[match.end() :]).lower().split("except")[0]
    return bool(set(re.findall(r"[a-z]+", around)) - SCALE_FILLER)


def _column_label(value: _Cell, headers: list[list[_Cell]]) -> str | None:
    """A value's column heading: all header text that sits above its column position."""
    parts = [cell.text for row in headers for cell in row if cell.overlaps(value)]
    return " ".join(parts) or None


def parse_table(
    table_index: int,
    table: lxml.html.HtmlElement,
    filing: FilingContext,
    contexts: dict[str, XbrlContext],
    preceding: list[str],
) -> ParsedTable:
    # The route below returns early with a reason whenever the table can't be fully explained.
    # Nothing is guessed: each early return sends the whole table to text.
    known_ends = reporting_period_ends(contexts, filing.form)
    grid = build_grid(table)
    rows = tuple(tuple(cell.text for cell in row) for row in grid)
    reasons: list[str] = []
    first_data = next((i for i, row in enumerate(grid) if _is_data_row(row)), None)
    title = _table_title(grid, first_data, preceding)

    skipped: set[int] = set()

    # Builds the answer. Facts are only returned if no reason was recorded.
    def result(facts: tuple[TableFact, ...] = ()) -> ParsedTable:
        return ParsedTable(
            table_index=table_index,
            title=title,
            clean=not reasons,
            reasons=tuple(reasons),
            facts=facts if not reasons else (),
            rows=rows,
            preceding=tuple(preceding),
            skipped_rows=len(skipped),
            scale_captions=captions,
        )

    # With no rows of figures, every row counts as header.
    header_rows = grid[:first_data]
    # Where a scale can be stated: a one-cell row anywhere, the header, or the text before the
    # table (nearest first). This runs before the early returns below, because the tables they
    # send to text are the ones whose caption the chunker needs.
    caption_texts = [cell.text for row in grid for cell in row if len(row) == 1]
    caption_texts += [cell.text for row in header_rows for cell in row]
    scale, scale_text = _scale_caption(caption_texts + list(reversed(preceding)))
    stated_cells = _stated_scale_cells(grid)
    stated = {cell_scale for cell_scale, _ in stated_cells}
    # Text that ends like a sentence may only mention a scale ("we sold millions of units."), and
    # the chunker's caption is what a reader trusts for the scale, so it isn't recorded as one.
    captions = (scale_text,) if scale_text and not scale_text.endswith(SENTENCE_END) else ()
    if len(stated) > 1:
        captions = tuple(dict.fromkeys((*captions, *(text for _, text in stated_cells))))

    if first_data is None:
        reasons.append("no rows of figures")
        return result()
    if not header_rows:
        reasons.append("no header rows")
        return result()
    if len(stated) > 1:
        reasons.append("more than one scale in the table")
        return result()
    if any(_scales_one_column(cell.text) for row in header_rows for cell in row):
        reasons.append("a header scales one column")
        return result()
    # Dollars need positive evidence: a "$" in the table or a dollar caption.
    table_has_dollar = any("$" in cell.text for row in grid for cell in row)
    dollars_stated = bool(DOLLAR_CAPTION.search(scale_text)) or table_has_dollar
    if NON_DOLLAR_CAPTION.search(" ".join(preceding + caption_texts)):
        dollars_stated = False
    context = " ".join(cell.text for row in header_rows for cell in row) + " " + " ".join(preceding)

    # Labels sit in the leftmost columns; a first cell starting beyond that is not a label.
    label_end = min((row[0].end for row in grid[first_data:] if len(row) > 1), default=1)
    # Header rows without the lone scale caption, which isn't part of any column name.
    headers = [
        [cell for cell in row if not (SCALE_CAPTION.search(cell.text) and len(row) == 1)]
        for row in header_rows
    ]
    columns: dict[str, _Column | None] = {}
    facts: list[TableFact] = []
    group: str | None = None
    unlabeled = data_rows = 0
    skipped_rows = skipped
    for row_index, cells in enumerate(grid[first_data:], start=first_data):
        if not _is_data_row(cells):
            # A table can restart with new periods, which would misdate every row after it.
            if any(YEAR_ONLY.match(cell.text) or _dates_in(cell.text) for cell in cells):
                reasons.append("a second header inside the table")
                return result()
            if len(cells) == 1:
                group = cells[0].text.rstrip(":")
            continue
        data_rows += 1
        label = cells[0]
        # A row whose first cell is a figure has lost its label.
        if label.start >= label_end or _is_value(label.text):
            unlabeled += 1
            skipped_rows.add(row_index)
            continue
        # An opening balance sits under the column for the period's end, so its date would be
        # wrong. Skipped rows stay reachable as text (see ParsedTable.needs_text_chunk).
        if OPENING_BALANCE_LABEL.search(label.text):
            skipped_rows.add(row_index)
            continue
        # "Term debt | 100 basis points | Increase in annual interest expense | $ 139" isn't
        # $139 million of term debt. Text beside figures holds meaning a fact can't carry.
        beside = next(
            (
                cell.text
                for cell in cells[1:]
                if not _is_value(cell.text)
                and not SYMBOL_ONLY.match(cell.text)
                and not HARMLESS_CELL.match(cell.text)
            ),
            None,
        )
        if beside is not None:
            reasons.append(f"text beside the figures: {beside[:60]}")
            return result()
        row_columns: set[str] = set()
        for position, cell in enumerate(cells[1:], start=1):
            if not _is_value(cell.text):
                continue
            column_label = _column_label(cell, headers)
            if column_label is None:
                reasons.append(f"value outside every column: {cell.text}")
                return result()
            if column_label in row_columns:
                reasons.append(f"two values in one column: {column_label}")
                return result()
            row_columns.add(column_label)
            if column_label not in columns:
                columns[column_label] = parse_column(column_label, context, filing, known_ends)
            column = columns[column_label]
            if column is None:
                reasons.append(f"column without a period: {column_label[:60]}")
                return result()
            if column.skip:
                continue
            assert column.end_date is not None
            # A column dated after the filing's period can't be right; usually a misread year.
            if column.end_date > _shift_days(filing.period_end, PERIOD_END_TOLERANCE_DAYS):
                reasons.append(f"column dated after the filing's period: {column_label[:60]}")
                return result()
            # A row about a different date than its column ("Balance at July 1") is skipped.
            if _label_names_another_date(label.text, column.end_date):
                skipped_rows.add(row_index)
                continue
            # A symbol belongs to this value only if it sits in the same column. In a row
            # with a rate column beside a balance column, the rate's % isn't the balance's.
            neighbors = " ".join(
                c.text
                for c in cells[position - 1 : position + 2]
                if SYMBOL_ONLY.match(c.text) and _column_label(c, headers) == column_label
            )
            has_percent = "%" in neighbors or "%" in cell.text or column.percent or column.share_of
            unit = _row_unit(group, label.text, column_label, has_percent, scale, dollars_stated)
            if unit is None:
                reasons.append(f"no unit or scale for row: {label.text[:60]}")
                return result()
            unit_name, multiplier = unit
            facts.append(
                TableFact(
                    fact_id=(f"{filing.accession}:t{table_index}:r{row_index}:c{cell.start}"),
                    table_index=table_index,
                    row_index=row_index,
                    table_title=title,
                    row_group=group,
                    row_label=label.text,
                    column_label=column_label,
                    start_date=column.start_date,
                    end_date=column.end_date,
                    duration=column.duration,
                    raw_text=cell.text,
                    value=parse_number(cell.text) * multiplier,
                    unit=unit_name,
                    is_share_of=column.share_of,
                    xbrl_concept=cell.tag.concept if cell.tag else None,
                    xbrl_value=cell.tag.value if cell.tag else None,
                    xbrl_end_date=(
                        contexts[cell.tag.context_id].end_date
                        if cell.tag and cell.tag.context_id in contexts
                        else None
                    ),
                    xbrl_context_id=cell.tag.context_id if cell.tag else None,
                )
            )
    # Whole-table checks, after every row is read.
    if data_rows and unlabeled / data_rows > UNLABELED_ROWS_MAX_SHARE:
        reasons.append(f"{unlabeled} of {data_rows} rows have no label")
    if _has_stray_precision(facts):
        reasons.append("a few values are more precise than the rest of the table")
    # A table whose columns were all comparisons produced nothing to store.
    if not facts and not reasons:
        reasons.append("only comparison columns")
    return result(tuple(facts))


def _row_unit(
    group: str | None,
    label: str,
    column_label: str,
    has_percent: bool,
    scale: float | None,
    dollars_stated: bool,
) -> tuple[str, float] | None:
    """A cell's unit and multiplier. None when they can't be known.

    Dollars need positive evidence: a stated scale, and a caption or a $ saying they're
    dollars. Reading $13.8 billion as $13,789, or oil volumes as money, is the worst error
    this module could make, so a table that doesn't say goes to text.
    """
    # A percent needs no scale, so it is accepted first.
    if has_percent:
        return UNIT_PERCENT, 1.0
    row_name = f"{group or ''} {label}"
    # Per-share amounts, share counts and rates are left to XBRL or to text: their captions
    # are unreliable (e.g. "in millions, except per share").
    if PRICE_COLUMN.search(column_label) or PER_SHARE_LABEL.search(column_label):
        return None
    if PER_SHARE_LABEL.search(row_name) or SHARE_ROW.search(row_name):
        return None
    if PERCENT_LABEL.search(row_name) or NON_DOLLAR_ROW.search(f"{row_name} {column_label}"):
        return None
    return (UNIT_USD, scale) if scale is not None and dollars_stated else None


def _decimals(text: str) -> int:
    """Digits after the decimal point in a displayed number."""
    match = re.search(r"\.(\d+)", text)
    return len(match.group(1)) if match else 0


def _has_stray_precision(facts: list[TableFact]) -> bool:
    """True when a few dollar cells are far more precise than the rest of the table."""
    decimals = [_decimals(fact.raw_text) for fact in facts if fact.unit == UNIT_USD and fact.value]
    fine = sum(1 for places in decimals if places >= FINE_DECIMALS)
    # Some cells are fine but they are the minority: likely per-share figures inside a dollars
    # table, where the table caption's scale would be wrong for them.
    return 0 < fine < len(decimals) - fine


def _table_title(
    grid: list[list[_Cell]], first_data: int | None, preceding: list[str]
) -> str | None:
    # First choice: a one-cell row above the data that reads like a title (not a unit caption,
    # a date or a bare year). Second choice: the nearest text before the table.
    # 120 characters is about a line; longer text is a sentence, not a title.
    limit = first_data if first_data is not None else len(grid)
    for row in grid[:limit]:
        text = row[0].text
        if (
            len(row) == 1
            and not SCALE_CAPTION.search(text)
            and len(text) <= 120
            and not _dates_in(text)
            and not YEAR_ONLY.match(text)
        ):
            return text
    for text in reversed(preceding):
        if len(text) <= 120 and not SCALE_CAPTION.search(text) and not NOT_A_TITLE.search(text):
            return text
    return None


def disagrees_with_tag(fact: TableFact) -> str | None:
    """Why a tagged cell contradicts what was parsed, if it does.

    Signs aren't compared. Facts keep the sign as displayed, and a tag records the sign of
    its concept, so a cost in parentheses can properly be a positive value in XBRL.
    """
    if fact.xbrl_value is None or fact.xbrl_end_date is None:
        return None
    parsed, tagged = abs(fact.value), abs(fact.xbrl_value)
    # Allow a millionth of relative rounding error; anything larger means the wrong scale
    # or the wrong cell.
    tolerance = max(tagged * 1e-6, 1e-9)
    # Percentages can be tagged as a fraction: 65% as 0.65. The looser 0.1% tolerance allows
    # for the rounding in a displayed percentage.
    same_value = abs(parsed - tagged) <= tolerance or (
        fact.unit == UNIT_PERCENT and abs(parsed / 100 - tagged) <= max(tagged * 1e-3, 1e-6)
    )
    if not same_value:
        return f"value {fact.raw_text} vs tagged {fact.xbrl_value}"
    # The tag's date is the filing's own record of the period, so a larger difference means
    # the column was misdated (this caught Regis's "Fiscal Year 2026" rate cells).
    if abs((fact.end_date - fact.xbrl_end_date).days) > PERIOD_END_TOLERANCE_DAYS:
        return f"period {fact.end_date} vs tagged {fact.xbrl_end_date}"
    return None


def verify_against_tags(table: ParsedTable) -> ParsedTable:
    """A table whose tagged cells contradict the parse can't be trusted for its other cells."""
    for fact in table.facts:
        problem = disagrees_with_tag(fact)
        if problem:
            # One bad cell makes the whole table untrusted, because the others rely on the
            # same column and scale reading.
            return table.model_copy(
                update={
                    "clean": False,
                    "reasons": (*table.reasons, f"disagrees with XBRL: {problem}"),
                    "facts": (),
                }
            )
    return table


def parse_tables(html: str, filing: FilingContext, *, verify: bool = True) -> list[ParsedTable]:
    """Parse every data table, numbered as `tables.strip_data_tables` numbers them.

    `verify=False` skips the check against XBRL tags. It exists so that evaluation can
    measure the parser alone.
    """
    root = parse_html_tree(html)
    contexts = read_contexts(root)
    # Number the outermost tables in document order, as strip_data_tables does, so an index
    # here is the same table there.
    numbered = []
    index = 0
    for table in list(root.iter("table")):
        if table.getparent() is None or any(a.tag == "table" for a in table.iterancestors()):
            continue
        numbered.append((index, table, is_data_table(table)))
        index += 1
    # Mark each table's place, so the text just before it can be found afterwards.
    preceding: dict[int, list[str]] = {}
    marked = parse_html_tree(html)
    position = 0
    for table in list(marked.iter("table")):
        if table.getparent() is None or any(a.tag == "table" for a in table.iterancestors()):
            continue
        marker = lxml.html.Element("p")
        marker.text = TABLE_MARKER.format(position)
        table.addprevious(marker)
        table.drop_tree()
        position += 1
    recent: list[str] = []
    for block in extract_blocks(marked):
        match = TABLE_MARKER_PATTERN.match(block)
        if match:
            preceding[int(match.group(1))] = list(recent)
        else:
            recent = [*recent[-(CONTEXT_BLOCKS - 1) :], block]
    parsed = [
        parse_table(number, table, filing, contexts, preceding.get(number, []))
        for number, table, is_data in numbered
        if is_data
    ]
    return [verify_against_tags(table) for table in parsed] if verify else parsed
