"""DuckDB store of the numbers the Financial Agent can search.

  xbrl_facts       Every figure a filer tagged in inline XBRL, including segment values.
  table_facts      Untagged figures read from tables that parsed cleanly.
  facts            A view of both, leaving out table figures XBRL already reports, and
                   any whose row can't be told apart from another in the same filing.
  unparsed_tables  Tables that didn't parse cleanly, waiting to be chunked as text.
  chunks           The passages retrieval searches. This is their source of truth; the vector
                   index is built from it and can always be rebuilt.

Only filings that passed validation can be loaded. A table fact always keeps the text it
was read from. Its sign is as displayed, and its duration describes its column, so a
balance can sit under a flow's duration; its end date is the part to rely on. `row_group`
is a hint: it's the last label-only row above, which isn't always a real grouping.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from types import TracebackType

import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from sec_agent.edgar.corpus import CorpusCompany, CorpusFiling
from sec_agent.edgar.xbrl import dimensions_json, duration_of, read_contexts, read_inline_fact
from sec_agent.ingest.audit import _fingerprint
from sec_agent.ingest.pipeline import FilingRejectedError, ParsedFiling
from sec_agent.ingest.table_facts import (
    PERIOD_END_TOLERANCE_DAYS,
    UNIT_USD,
    FilingContext,
    ParsedTable,
    TableFact,
    parse_number,
    parse_tables,
)
from sec_agent.ingest.tables import parse_html_tree
from sec_agent.schemas import Chunk, FiscalPeriod, FormType, Section

# Bump when table parsing or fact loading changes, so that stored facts are rebuilt.
FACTS_VERSION = 7
# The XBRL breakdown that separates one business segment's figures from the company total.
SEGMENT_AXIS = "us-gaap:StatementBusinessSegmentsAxis"
# Below this many significant digits, two equal numbers are likely to be a coincidence.
# "5" or "120" appears all over a filing; "2,787.3" does not. Chosen by hand.
DUPLICATE_MIN_SIGNIFICANT_DIGITS = 4
# The text before a table must be at least this long (letters and digits only) to be used to
# find which section the table sits in; a shorter caption would match unrelated text.
MIN_LOCATOR_CHARS = 15

# Five tables and one view. Each filing's rows are replaced together in one transaction, and
# `loaded_filings` records the FACTS_VERSION used, so out-of-date rows can be rebuilt.
# In table_facts, xbrl_fact_id is set when the same figure is already tagged, and `ambiguous`
# marks facts that share every label but differ in value; the `facts` view hides both kinds.
# The view is what the Financial Agent searches: both sources in one shape, without duplicates.
SCHEMA = """
CREATE TABLE IF NOT EXISTS loaded_filings (
    accession VARCHAR PRIMARY KEY, facts_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS xbrl_facts (
    fact_id VARCHAR PRIMARY KEY, ticker VARCHAR NOT NULL, accession VARCHAR NOT NULL,
    form VARCHAR NOT NULL, fiscal_year INTEGER NOT NULL, fiscal_period VARCHAR NOT NULL,
    concept VARCHAR NOT NULL, start_date DATE, end_date DATE NOT NULL, duration VARCHAR NOT NULL,
    value DOUBLE NOT NULL, unit VARCHAR, segment VARCHAR, dimensions VARCHAR
);
CREATE TABLE IF NOT EXISTS table_facts (
    fact_id VARCHAR PRIMARY KEY, ticker VARCHAR NOT NULL, accession VARCHAR NOT NULL,
    form VARCHAR NOT NULL, fiscal_year INTEGER NOT NULL, fiscal_period VARCHAR NOT NULL,
    section VARCHAR, heading VARCHAR, context VARCHAR, table_index INTEGER NOT NULL,
    table_title VARCHAR, row_group VARCHAR, row_label VARCHAR NOT NULL,
    column_label VARCHAR NOT NULL, start_date DATE, end_date DATE NOT NULL, duration VARCHAR,
    raw_text VARCHAR NOT NULL, value DOUBLE NOT NULL, unit VARCHAR NOT NULL,
    is_share_of BOOLEAN NOT NULL, xbrl_fact_id VARCHAR, ambiguous BOOLEAN NOT NULL
);
CREATE TABLE IF NOT EXISTS unparsed_tables (
    table_id VARCHAR PRIMARY KEY, ticker VARCHAR NOT NULL, accession VARCHAR NOT NULL,
    form VARCHAR NOT NULL, fiscal_year INTEGER NOT NULL, fiscal_period VARCHAR NOT NULL,
    section VARCHAR, table_index INTEGER NOT NULL, title VARCHAR, reasons VARCHAR NOT NULL,
    text VARCHAR NOT NULL, scale_caption VARCHAR
);
ALTER TABLE unparsed_tables ADD COLUMN IF NOT EXISTS scale_caption VARCHAR;
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR PRIMARY KEY, paragraph_id VARCHAR NOT NULL, paragraph_ids VARCHAR[] NOT NULL,
    ticker VARCHAR NOT NULL, cik VARCHAR NOT NULL, accession VARCHAR NOT NULL,
    form VARCHAR NOT NULL, fiscal_year INTEGER NOT NULL, fiscal_period VARCHAR NOT NULL,
    period_end DATE NOT NULL, section VARCHAR NOT NULL, position INTEGER NOT NULL,
    text VARCHAR NOT NULL, token_count INTEGER NOT NULL
);
CREATE OR REPLACE VIEW facts AS
    SELECT 'xbrl' AS source, fact_id, ticker, accession, form, fiscal_year, fiscal_period,
           concept AS name, CAST(NULL AS VARCHAR) AS table_title,
           CAST(NULL AS VARCHAR) AS row_group, CAST(NULL AS VARCHAR) AS column_label,
           segment, dimensions, start_date, end_date, duration, value, unit,
           CAST(NULL AS VARCHAR) AS raw_text, CAST(NULL AS VARCHAR) AS section,
           CAST(NULL AS VARCHAR) AS heading, CAST(NULL AS VARCHAR) AS context
    FROM xbrl_facts
    UNION ALL
    SELECT 'table', fact_id, ticker, accession, form, fiscal_year, fiscal_period, row_label,
           table_title, row_group, column_label, NULL, NULL, start_date, end_date, duration,
           value, unit, raw_text, section, heading, context
    FROM table_facts WHERE xbrl_fact_id IS NULL AND NOT ambiguous;
"""

# What marks a stored XBRL unit as plain US dollars. The stored unit is the label a filer chose,
# not a standard name: the store holds usd, USD, U_USD and Unit_Standard_USD_<random>. Anchoring
# keeps out usdPerShare, USDPShares and Unit_Divide_USD_shares_<random>, which also contain "USD".
USD_UNIT_PATTERN = r"^(u_)?usd$|^unit_standard_usd_"

# The columns of the `facts` view, in the order FactRow reads them.
SEARCH_COLUMNS = (
    "source, fact_id, ticker, accession, form, fiscal_year, fiscal_period, name, table_title, "
    "row_group, column_label, segment, dimensions, start_date, end_date, duration, value, unit, "
    "raw_text, section, heading, context"
)
# The columns a Chunk is rebuilt from, in the order _chunk_from_row reads them.
CHUNK_COLUMNS = (
    "chunk_id, paragraph_id, ticker, cik, form, fiscal_year, fiscal_period, period_end, section, "
    "accession, text, token_count"
)
# Everything keyed to a filing that a reload makes out of date. Chunks are included so a reload
# can't leave passages behind from text that has since changed.
FILING_TABLES = ("xbrl_facts", "table_facts", "unparsed_tables", "chunks", "loaded_filings")


class FactRow(BaseModel):
    """One search result, carrying what's needed to cite it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    fact_id: str
    ticker: str
    accession: str
    form: str
    fiscal_year: int
    fiscal_period: str
    name: str
    table_title: str | None = None
    row_group: str | None = None
    column_label: str | None = None
    segment: str | None = None
    dimensions: str | None = None
    start_date: date | None = None
    end_date: date
    duration: str | None = None
    value: float
    unit: str | None = None
    raw_text: str | None = None
    section: str | None = None
    heading: str | None = None
    context: str | None = None


class SourcedChunk(BaseModel):
    """A chunk and every paragraph its text came from.

    `Chunk.paragraph_id` holds one id, but a chunk usually joins several short paragraphs. Change
    detection works on paragraphs and must find every chunk that holds one, so the full list is
    stored beside the chunk.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: Chunk
    paragraph_ids: tuple[str, ...] = Field(min_length=1)


class LoadSummary(BaseModel):
    """Counts from loading one filing, for reporting and tests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    xbrl_facts: int
    table_facts: int
    # Table figures that XBRL already reports; stored but hidden from search.
    duplicates_of_xbrl: int
    ambiguous_facts: int
    tables_clean: int
    # Tables queued as text for the chunker.
    tables_unparsed: int


def xbrl_rows(root, company: CorpusCompany, filing: CorpusFiling) -> list[dict]:
    """Each distinct tagged figure once. A filing can repeat a fact in several places."""
    contexts = read_contexts(root)
    rows: dict[str, dict] = {}
    for element in root.iter("ix:nonfraction"):
        fact = read_inline_fact(element)
        # A tag with no readable number or no matching period can't be stored.
        if fact is None or fact.context_id not in contexts:
            continue
        # Concept + period + unit identifies a figure, so a repeat in another table is skipped.
        fact_id = f"{filing.accession}:{fact.concept}:{fact.context_id}:{fact.unit}"
        if fact_id in rows:
            continue
        context = contexts[fact.context_id]
        rows[fact_id] = {
            "fact_id": fact_id,
            "ticker": company.ticker,
            "accession": filing.accession,
            "form": filing.form.value,
            "fiscal_year": filing.fiscal_year,
            "fiscal_period": filing.fiscal_period,
            "concept": fact.concept,
            "start_date": context.start_date,
            "end_date": context.end_date,
            "duration": duration_of(context).value,
            "value": fact.value,
            "unit": fact.unit,
            "segment": context.dimensions.get(SEGMENT_AXIS),
            "dimensions": dimensions_json(context),
        }
    return list(rows.values())


class _Locator:
    """Finds where a table sits in a filing's sections, from the text just before it."""

    def __init__(self, parsed: ParsedFiling) -> None:
        # Two indexes over the parsed text: whole paragraphs and headings (exact), and each
        # section's text joined together (for a looser "contained in" match).
        self._exact: dict[str, tuple[Section, str | None]] = {}
        self._blobs: dict[Section, str] = {}
        for section, content in parsed.sections.items():
            parts = []
            for paragraph in content.paragraphs:
                text = _fingerprint(paragraph.text)
                parts.append(text)
                self._exact.setdefault(text, (section, paragraph.heading))
                if paragraph.heading:
                    self._exact.setdefault(
                        _fingerprint(paragraph.heading), (section, paragraph.heading)
                    )
            self._blobs[section] = " ".join(parts)

    def locate(self, table: ParsedTable) -> tuple[Section | None, str | None]:
        """The table's section and nearest heading, when the text before it can be found."""
        # Nearest text first: the closest block before a table says where it sits.
        for text in reversed(table.preceding):
            print_ = _fingerprint(text)
            if len(print_) < MIN_LOCATOR_CHARS:
                continue
            if print_ in self._exact:
                return self._exact[print_]
            for section, blob in self._blobs.items():
                if print_ in blob:
                    return section, None
        return None, None


def _significant_digits(raw_text: str) -> int:
    """Digits that carry information: all digits, minus trailing zeros ("1,200" has 2)."""
    return len(re.sub(r"\D", "", raw_text).strip("0"))


def _display_tolerance(fact: TableFact) -> float:
    """Half the last shown digit. MD&A's 2,787.3 (millions) is XBRL's 2,787,298 (thousands)."""
    match = re.search(r"\.(\d+)", fact.raw_text)
    places = len(match.group(1)) if match else 0
    number = abs(parse_number(fact.raw_text)) or 1.0
    return 0.5 * 10.0**-places * abs(fact.value) / number


def find_xbrl_duplicate(fact: TableFact, by_end: dict[date, list[tuple[float, str]]]) -> str | None:
    """A tagged fact with the same period and the same amount, allowing for rounding.

    Only a distinctive amount counts. Two small numbers being equal is often a coincidence.
    """
    if (
        fact.unit != UNIT_USD
        or _significant_digits(fact.raw_text) < DUPLICATE_MIN_SIGNIFICANT_DIGITS
    ):
        return None
    tolerance = _display_tolerance(fact)
    amount = abs(fact.value)
    # Look at nearby period ends too, since the table's date and the tag's can differ by days
    # in 52/53-week years.
    for offset in range(-PERIOD_END_TOLERANCE_DAYS, PERIOD_END_TOLERANCE_DAYS + 1):
        end = date.fromordinal(fact.end_date.toordinal() + offset)
        for value, fact_id in by_end.get(end, ()):
            if abs(value - amount) <= tolerance:
                return fact_id
    return None


def _chunk_from_row(row: tuple) -> Chunk:
    """Rebuild a Chunk from a row read with CHUNK_COLUMNS."""
    chunk_id, paragraph_id, ticker, cik, form, year, fiscal_period, period_end, *rest = row
    section, accession, text, token_count = rest
    return Chunk(
        chunk_id=chunk_id,
        paragraph_id=paragraph_id,
        ticker=ticker,
        cik=cik,
        form=FormType(form),
        period=FiscalPeriod(fiscal_year=year, fiscal_period=fiscal_period, period_end=period_end),
        section=Section(section),
        accession=accession,
        text=text,
        token_count=token_count,
    )


def mark_ambiguous(rows: list[dict]) -> set[int]:
    """Flag facts that share every name but differ in value. Returns their table numbers.

    Mattel reports "Fisher-Price" for each segment. With the heading, those can be told
    apart. Where nothing separates two different numbers, the agent mustn't have to guess.
    """
    # Rows with the same key look identical to a searcher, so they must agree on value.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["heading"],
            row["table_title"],
            row["row_group"],
            row["row_label"],
            row["column_label"],
            row["end_date"],
            row["duration"],
            row["unit"],
        )
        groups[key].append(row)
    tables = set()
    for group in groups.values():
        if len({row["value"] for row in group}) > 1:
            for row in group:
                row["ambiguous"] = True
                tables.add(row["table_index"])
    return tables


class FactStore:
    """The DuckDB file holding facts. Use as a context manager so the file is closed."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        """Open the store, read-only if asked.

        Several processes can hold the file read-only at once, but while any does, nothing can
        write to it, and while one holds it read-write, nothing else can open it. One process can't
        hold it both ways, so everything in a process must share one store. A read-only store
        applies no schema, so it reads the file as the last writer left it.
        """
        if read_only:
            self._db = duckdb.connect(path, read_only=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = duckdb.connect(path)
            self._db.execute(SCHEMA)
        # Each filing's dollar facts for find_usd_fact, as parallel lists sorted by magnitude.
        self._usd_facts: dict[str, tuple[list[float], list[str]]] = {}

    def __enter__(self) -> FactStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def is_loaded(self, accession: str) -> bool:
        """True only if the filing was loaded with the current FACTS_VERSION."""
        row = self._db.execute(
            "SELECT facts_version FROM loaded_filings WHERE accession = ?", [accession]
        ).fetchone()
        return row is not None and row[0] == FACTS_VERSION

    def remove_filing(self, accession: str) -> None:
        """Take out a filing that stopped passing, because rules can tighten after it loaded."""
        for name in FILING_TABLES:
            self._db.execute(f"DELETE FROM {name} WHERE accession = ?", [accession])
        self._usd_facts.pop(accession, None)

    def _insert(self, table: str, rows: list[dict]) -> None:
        if not rows:
            return
        # DuckDB looks up the local variable `frame` by name inside the SQL below.
        frame = pd.DataFrame(rows)  # noqa: F841 - DuckDB reads it by name
        columns = ", ".join(rows[0])
        self._db.execute(f"INSERT INTO {table} ({columns}) SELECT {columns} FROM frame")

    def load_filing(
        self, company: CorpusCompany, filing: CorpusFiling, html: str, parsed: ParsedFiling
    ) -> LoadSummary:
        """Replace a filing's rows. Raises if the filing didn't pass validation."""
        # The rule that keeps unvalidated filings out of the store.
        if not parsed.passed:
            raise FilingRejectedError(f"{filing.accession} failed validation; loading no facts.")
        root = parse_html_tree(html)
        xbrl = xbrl_rows(root, company, filing)
        # Tagged amounts by period end, so table figures can be checked against them quickly.
        by_end: dict[date, list[tuple[float, str]]] = defaultdict(list)
        for row in xbrl:
            by_end[row["end_date"]].append((abs(row["value"]), row["fact_id"]))

        context = FilingContext(
            filing.accession,
            filing.form,
            filing.fiscal_year,
            filing.fiscal_period,
            filing.period_end,
        )
        tables = parse_tables(html, context)
        identity = {
            "ticker": company.ticker,
            "accession": filing.accession,
            "form": filing.form.value,
            "fiscal_year": filing.fiscal_year,
            "fiscal_period": filing.fiscal_period,
        }
        locator = _Locator(parsed)
        fact_rows: list[dict] = []
        duplicates = 0
        placed = {table.table_index: locator.locate(table) for table in tables}
        for table in tables:
            section, heading = placed[table.table_index]
            for fact in table.facts:
                # A tagged cell is already in xbrl_facts, so only untagged cells are added here.
                if fact.tagged:
                    continue
                duplicate = find_xbrl_duplicate(fact, by_end)
                duplicates += duplicate is not None
                fact_rows.append(
                    {
                        **identity,
                        "fact_id": fact.fact_id,
                        "section": section.value if section else None,
                        "heading": heading,
                        "context": " / ".join(table.preceding),
                        "table_index": fact.table_index,
                        "table_title": fact.table_title,
                        "row_group": fact.row_group,
                        "row_label": fact.row_label,
                        "column_label": fact.column_label,
                        "start_date": fact.start_date,
                        "end_date": fact.end_date,
                        "duration": fact.duration.value if fact.duration else None,
                        "raw_text": fact.raw_text,
                        "value": fact.value,
                        "unit": fact.unit,
                        "is_share_of": fact.is_share_of,
                        "xbrl_fact_id": duplicate,
                        "ambiguous": False,
                    }
                )
        ambiguous_tables = mark_ambiguous(fact_rows)

        # Tables that need text: not clean, with rows that gave no fact, or with ambiguous facts.
        # This is how no figure ends up unreachable.
        text_rows = []
        for table in tables:
            is_ambiguous = table.table_index in ambiguous_tables
            if not table.needs_text_chunk and not is_ambiguous:
                continue
            section, _ = placed[table.table_index]
            reasons = list(table.reasons)
            if table.clean and table.skipped_rows:
                reasons.append(f"{table.skipped_rows} rows gave no fact")
            if is_ambiguous:
                reasons.append("some rows can't be told apart from another table's")
            text_rows.append(
                {
                    **identity,
                    "table_id": f"{filing.accession}:t{table.table_index}",
                    "section": section.value if section else None,
                    "table_index": table.table_index,
                    "title": table.title,
                    "reasons": "; ".join(reasons),
                    "text": table.as_text,
                    # One statement per line. NULL when the table states no scale.
                    "scale_caption": "\n".join(table.scale_captions) or None,
                }
            )

        # One transaction, so a failure never leaves a filing half replaced.
        self._db.execute("BEGIN TRANSACTION")
        try:
            for name in FILING_TABLES:
                self._db.execute(f"DELETE FROM {name} WHERE accession = ?", [filing.accession])
            self._insert("xbrl_facts", xbrl)
            self._insert("table_facts", fact_rows)
            self._insert("unparsed_tables", text_rows)
            self._db.execute(
                "INSERT INTO loaded_filings VALUES (?, ?)", [filing.accession, FACTS_VERSION]
            )
            self._db.execute("COMMIT")
        except Exception:
            self._db.execute("ROLLBACK")
            raise
        self._usd_facts.pop(filing.accession, None)
        return LoadSummary(
            xbrl_facts=len(xbrl),
            table_facts=len(fact_rows),
            duplicates_of_xbrl=duplicates,
            ambiguous_facts=sum(1 for row in fact_rows if row["ambiguous"]),
            tables_clean=sum(1 for table in tables if table.clean),
            tables_unparsed=len(text_rows),
        )

    def search(
        self,
        ticker: str,
        query: str | None = None,
        *,
        end_date: date | None = None,
        fiscal_year: int | None = None,
        source: str | None = None,
        limit: int = 50,
    ) -> list[FactRow]:
        """Find facts for one company. Every word of `query` must appear in or near each fact."""
        clauses = ["ticker = ?"]
        parameters: list[object] = [ticker]
        # Each query word must appear somewhere in the fact's labels or nearby text.
        for word in (query or "").split():
            clauses.append(
                "concat_ws(' ', name, table_title, row_group, column_label, segment, heading, "
                "context) ILIKE ?"
            )
            parameters.append(f"%{word}%")
        if end_date is not None:
            clauses.append("end_date = ?")
            parameters.append(end_date)
        if fiscal_year is not None:
            clauses.append("fiscal_year = ?")
            parameters.append(fiscal_year)
        if source is not None:
            clauses.append("source = ?")
            parameters.append(source)
        # Facts named by the query come before those merely mentioned near it.
        words = [f"%{word}%" for word in (query or "").split()]
        named = " AND ".join(["name ILIKE ?"] * len(words)) or "TRUE"
        cursor = self._db.execute(
            f"SELECT {SEARCH_COLUMNS} FROM facts WHERE {' AND '.join(clauses)} "
            f"ORDER BY CASE WHEN {named} THEN 0 ELSE 1 END, end_date DESC, source, name LIMIT ?",
            [*parameters, *words, limit],
        )
        names = [column[0] for column in cursor.description]
        return [FactRow(**dict(zip(names, row, strict=True))) for row in cursor.fetchall()]

    def unparsed_tables(self, accession: str | None = None) -> list[dict]:
        # (The chunker must call this, or table figures that did not parse cleanly are lost.)
        """Tables for the chunker, which must index them as text."""
        where = "WHERE accession = ?" if accession else ""
        parameters = [accession] if accession else []
        cursor = self._db.execute(
            "SELECT table_id, ticker, accession, form, fiscal_year, fiscal_period, section, "
            f"table_index, title, reasons, text, scale_caption FROM unparsed_tables {where} "
            "ORDER BY accession, table_index",
            parameters,
        )
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    def find_usd_fact(self, accession: str, value: float, tolerance: float) -> str | None:
        """A tagged dollar fact of this magnitude in one filing, or None. See `FactLookup`.

        Of several matches it returns the nearest, then the lowest id, so a figure always traces
        to the same fact. Segment values count. Table facts don't, as they would let a table
        confirm itself. A caller makes many of these calls per citation, so a filing's facts are
        read once and then searched in memory.
        """
        if accession not in self._usd_facts:
            rows = self._db.execute(
                "SELECT abs(value) AS magnitude, fact_id FROM xbrl_facts "
                "WHERE accession = ? AND regexp_matches(unit, ?, 'i') ORDER BY magnitude, fact_id",
                [accession, USD_UNIT_PATTERN],
            ).fetchall()
            self._usd_facts[accession] = ([row[0] for row in rows], [row[1] for row in rows])
        magnitudes, fact_ids = self._usd_facts[accession]
        target = abs(value)
        start = bisect_left(magnitudes, target - tolerance)
        stop = bisect_right(magnitudes, target + tolerance)
        if start >= stop:
            return None
        nearest = min(range(start, stop), key=lambda i: (abs(magnitudes[i] - target), fact_ids[i]))
        return fact_ids[nearest]

    def replace_chunks(self, accession: str, chunks: Sequence[SourcedChunk]) -> None:
        """Swap in a filing's chunks, so re-chunking can't duplicate or orphan any."""
        if any(item.chunk.accession != accession for item in chunks):
            raise ValueError(f"every chunk must belong to {accession}")
        rows = [
            {
                "chunk_id": item.chunk.chunk_id,
                "paragraph_id": item.chunk.paragraph_id,
                "paragraph_ids": list(item.paragraph_ids),
                "ticker": item.chunk.ticker,
                "cik": item.chunk.cik,
                "accession": accession,
                "form": item.chunk.form.value,
                "fiscal_year": item.chunk.period.fiscal_year,
                "fiscal_period": item.chunk.period.fiscal_period,
                "period_end": item.chunk.period.period_end,
                "section": item.chunk.section.value,
                "position": position,
                "text": item.chunk.text,
                "token_count": item.chunk.token_count,
            }
            for position, item in enumerate(chunks)
        ]
        # One transaction, so a failure never leaves a filing with half its chunks.
        self._db.execute("BEGIN TRANSACTION")
        try:
            self._db.execute("DELETE FROM chunks WHERE accession = ?", [accession])
            self._insert("chunks", rows)
            self._db.execute("COMMIT")
        except Exception:
            self._db.execute("ROLLBACK")
            raise

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """One chunk by id, or None. This makes the store a `ChunkLookup`."""
        row = self._db.execute(
            f"SELECT {CHUNK_COLUMNS} FROM chunks WHERE chunk_id = ?", [chunk_id]
        ).fetchone()
        return _chunk_from_row(row) if row else None

    def chunks(self, accession: str | None = None) -> list[Chunk]:
        """One filing's chunks, or every filing's, each filing in reading order."""
        where = "WHERE accession = ?" if accession else ""
        rows = self._db.execute(
            f"SELECT {CHUNK_COLUMNS} FROM chunks {where} ORDER BY accession, position",
            [accession] if accession else [],
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def chunk_ids_for_paragraph(self, paragraph_id: str) -> tuple[str, ...]:
        """Every chunk holding text from a paragraph, including chunks that only overlap it."""
        rows = self._db.execute(
            "SELECT chunk_id FROM chunks WHERE list_contains(paragraph_ids, ?) ORDER BY position",
            [paragraph_id],
        ).fetchall()
        return tuple(row[0] for row in rows)

    def count(self, table: str, ticker: str | None = None) -> int:
        # The table name is placed into SQL text, so it must be one of these.
        allowed = {"xbrl_facts", "table_facts", "unparsed_tables", "facts", "chunks"}
        if table not in allowed:
            raise ValueError(f"Unknown table {table}")
        where = "WHERE ticker = ?" if ticker else ""
        row = self._db.execute(
            f"SELECT count(*) FROM {table} {where}", [ticker] if ticker else []
        ).fetchone()
        return row[0] if row else 0
