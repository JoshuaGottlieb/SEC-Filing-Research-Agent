"""A table's scale caption is found for tables sent to text, and is stored for the chunker."""

from __future__ import annotations

from datetime import date

import duckdb

from sec_agent.edgar.corpus import CorpusCompany, CorpusFiling
from sec_agent.finance.fact_store import FactStore
from sec_agent.ingest.audit import TableAudit
from sec_agent.ingest.pipeline import ParsedFiling, ParserName
from sec_agent.ingest.table_facts import FilingContext, ParsedTable, parse_tables
from sec_agent.schemas import FormType

CONTEXT = FilingContext("acc-1", FormType.FORM_10K, 2025, "FY", date(2025, 12, 31))
PINNED = CorpusFiling(
    form=FormType.FORM_10K,
    fiscal_year=2025,
    fiscal_period="FY",
    period_end=date(2025, 12, 31),
    filed_date=date(2026, 2, 1),
    accession="acc-1",
    primary_document="doc.htm",
)
COMPANY = CorpusCompany(ticker="AAA", company="A", cik="0000000001", filings=(PINNED,))


def row(*cells: str) -> str:
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def page(*parts: str) -> str:
    return f"<html><body>{''.join(parts)}</body></html>"


def table(*rows: str) -> str:
    return f"<table>{''.join(rows)}</table>"


def parse_one(html: str) -> ParsedTable:
    tables = parse_tables(html, CONTEXT)
    assert len(tables) == 1
    return tables[0]


HEADER = row("", "2025", "2024")
FIGURES = (row("Net sales", "$ 100", "$ 90"), row("Cost of sales", "60", "50"))


def test_a_clean_table_records_the_caption_it_was_read_with():
    parsed = parse_one(page("<p>(in millions)</p>", table(HEADER, *FIGURES)))
    assert parsed.clean
    assert parsed.scale_captions == ("(in millions)",)


def test_the_caption_is_the_filings_own_words():
    caption = "(Unaudited, Millions of Dollars, Except Share and Per Share Amounts)"
    assert parse_one(page(f"<p>{caption}</p>", table(HEADER, *FIGURES))).scale_captions == (
        caption,
    )


def test_a_table_with_no_caption_records_none_rather_than_guessing():
    parsed = parse_one(page("<p>Sales were strong this year.</p>", table(HEADER, *FIGURES)))
    assert parsed.scale_captions == ()


def test_a_sentence_that_mentions_millions_is_not_a_caption():
    sentence = "We sold millions of units to customers around the world during the year."
    assert parse_one(page(f"<p>{sentence}</p>", table(HEADER, *FIGURES))).scale_captions == ()


def test_a_table_with_no_header_rows_still_finds_its_caption():
    # This table returns early, before the search for a scale used to run.
    parsed = parse_one(page("<p>(in thousands)</p>", table(*FIGURES)))
    assert parsed.reasons == ("no header rows",)
    assert parsed.scale_captions == ("(in thousands)",)


def test_a_caption_inside_the_table_is_found():
    parsed = parse_one(page(table(row("(in thousands)"), HEADER, *FIGURES)))
    assert parsed.scale_captions == ("(in thousands)",)


def test_a_table_stating_two_scales_keeps_both_as_written():
    parsed = parse_one(
        page(
            table(
                row("Sales (in millions)"),
                HEADER,
                row("Net sales", "$ 100", "$ 90"),
                row("Headcount (in thousands)"),
                row("Employees", "60", "50"),
            )
        )
    )
    assert parsed.reasons == ("more than one scale in the table",)
    assert parsed.scale_captions == ("Sales (in millions)", "Headcount (in thousands)")


def test_a_table_failing_later_checks_keeps_its_caption():
    # One column disagreeing with an XBRL tag empties the facts, but not the caption.
    parsed = parse_one(
        page("<p>(in millions)</p>", table(row("", "Change", "Other"), *FIGURES))
    ).model_copy(update={"clean": False, "facts": ()})
    assert parsed.scale_captions == ("(in millions)",)


def parsed_filing() -> ParsedFiling:
    return ParsedFiling(
        accession="acc-1",
        form=FormType.FORM_10K,
        parser=ParserName.EDGARTOOLS,
        layout="edgartools",
        sections={},
        coverage=1.0,
        table_audit=TableAudit(tables_total=1, tables_removed=1, prose_rows_preserved=0),
        problems=(),
    )


def test_the_store_keeps_the_caption_of_a_table_sent_to_text(tmp_path):
    html = page("<p>(in thousands)</p>", table(*FIGURES))
    with FactStore(tmp_path / "facts.duckdb") as store:
        store.load_filing(COMPANY, PINNED, html, parsed_filing())
        tables = store.unparsed_tables("acc-1")
    assert [item["scale_caption"] for item in tables] == ["(in thousands)"]
    # The caption stays out of the table's own text. The chunker places it.
    assert "(in thousands)" not in tables[0]["text"]


def test_the_store_holds_no_caption_when_none_was_found(tmp_path):
    with FactStore(tmp_path / "facts.duckdb") as store:
        store.load_filing(COMPANY, PINNED, page(table(*FIGURES)), parsed_filing())
        assert [item["scale_caption"] for item in store.unparsed_tables("acc-1")] == [None]


def test_the_store_keeps_both_scales_one_per_line(tmp_path):
    html = page(
        table(
            row("Sales (in millions)"),
            HEADER,
            row("Net sales", "$ 100", "$ 90"),
            row("Headcount (in thousands)"),
            row("Employees", "60", "50"),
        )
    )
    with FactStore(tmp_path / "facts.duckdb") as store:
        store.load_filing(COMPANY, PINNED, html, parsed_filing())
        captions = store.unparsed_tables("acc-1")[0]["scale_caption"]
    assert captions == "Sales (in millions)\nHeadcount (in thousands)"


def test_a_database_from_before_the_column_gains_it(tmp_path):
    # The shared DuckDB file already has this table, and CREATE IF NOT EXISTS won't change it.
    path = tmp_path / "facts.duckdb"
    with duckdb.connect(path) as db:
        db.execute(
            """CREATE TABLE unparsed_tables (
                table_id VARCHAR PRIMARY KEY, ticker VARCHAR NOT NULL, accession VARCHAR NOT NULL,
                form VARCHAR NOT NULL, fiscal_year INTEGER NOT NULL,
                fiscal_period VARCHAR NOT NULL, section VARCHAR, table_index INTEGER NOT NULL,
                title VARCHAR, reasons VARCHAR NOT NULL, text VARCHAR NOT NULL
            )"""
        )
        db.execute(
            "INSERT INTO unparsed_tables VALUES "
            "('acc-1:t1', 'AAA', 'acc-1', '10-K', 2025, 'FY', NULL, 1, NULL, 'why', 'text')"
        )
    with FactStore(path) as store:
        assert [item["scale_caption"] for item in store.unparsed_tables("acc-1")] == [None]


def test_opening_the_store_twice_keeps_the_captions(tmp_path):
    path = tmp_path / "facts.duckdb"
    with FactStore(path) as store:
        store.load_filing(
            COMPANY, PINNED, page("<p>(in thousands)</p>", table(*FIGURES)), parsed_filing()
        )
    with FactStore(path) as store:
        assert store.unparsed_tables("acc-1")[0]["scale_caption"] == "(in thousands)"
