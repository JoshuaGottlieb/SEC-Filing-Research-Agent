"""Every downloaded filing must pass the gate, and known text must land in the right place.

Run with `pytest -m corpus` after `sec-agent ingest` and
`sec-agent ingest --corpus config/parser_validation.yaml`. Filings not yet downloaded are
skipped. The passages are quoted from the filings and were checked by reading them, so
they test where text ended up, not just whether a section looks plausible.
"""

from __future__ import annotations

import json
import re
from datetime import date
from functools import cache
from pathlib import Path

import duckdb
import pytest

from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.corpus import Corpus, assign_fiscal_periods
from sec_agent.edgar.xbrl import read_contexts
from sec_agent.finance.fact_store import xbrl_rows
from sec_agent.ingest.audit import _note_numbers
from sec_agent.ingest.fallback import parse_fallback
from sec_agent.ingest.pipeline import ParsedFiling, ParserName, parse_filing
from sec_agent.ingest.sections import render_items
from sec_agent.ingest.table_facts import (
    UNIT_PERCENT,
    FilingContext,
    disagrees_with_tag,
    parse_tables,
    reporting_period_ends,
)
from sec_agent.ingest.tables import parse_html_tree, strip_data_tables
from sec_agent.schemas import FormType, Section

# These tests read real downloaded filings, so they run only with `pytest -m corpus`.
pytestmark = pytest.mark.corpus

# Every pinned corpus: the three analysis companies, the five layout-stress companies, and
# one file per held-out draw. A filing in any of them must pass the gate unless listed below.
CORPUS_FILES = (
    "corpus.yaml",
    "parser_validation.yaml",
    "parser_holdout.yaml",
    "parser_holdout2.yaml",
    "parser_holdout3.yaml",
    "parser_holdout4.yaml",
)
# Rejections that are right, or not yet fixed. If one starts passing, read its sections.
# Tickers: UBOH United Bancshares, COSM Cosmos Health, LODE Comstock, T AT&T, CFOR CapForce,
# MEGR First America Resources. Keys are (ticker, fiscal year, fiscal period).
KNOWN_REJECTIONS = {
    ("UBOH", 2022, "FY"),  # MD&A and statements are in Exhibit 13, which isn't read.
    ("UBOH", 2021, "FY"),
    ("UBOH", 2022, "Q3"),  # A row label holds a sentence, dropped with its figures.
    ("UBOH", 2021, "Q3"),
    ("UBOH", 2022, "Q2"),
    ("COSM", 2025, "Q2"),  # Both parsers lose the notes.
    ("LODE", 2024, "FY"),  # Share issuance tables with one number column stay in the text.
    ("T", 2025, "FY"),  # Both parsers run market risk through the audit reports.
    ("T", 2024, "FY"),
    ("CFOR", 2025, "FY"),  # A row label reads as a sentence, dropped with its figures.
    ("CFOR", 2024, "FY"),
    ("CFOR", 2025, "Q2"),
    ("MEGR", 2025, "FY"),  # A very small company, below length floors set for larger ones.
    ("MEGR", 2024, "FY"),
    ("MEGR", 2026, "Q2"),
    ("MEGR", 2025, "Q2"),
    ("MEGR", 2026, "Q1"),
}
# Short names used in the test tables below.
K, Q = FormType.FORM_10K, FormType.FORM_10Q
MDNA, NOTES, RISK = Section.MDNA, Section.FINANCIAL_NOTES, Section.RISK_FACTORS


@pytest.mark.parametrize("name", CORPUS_FILES)
def test_pinned_fiscal_labels_match_the_derived_calendar(name):
    """These labels become retrieval metadata, so a filer's bad tag mustn't reach them."""
    for company in Corpus.load(REPO_ROOT / "config" / name).companies:
        derived = assign_fiscal_periods(list(company.filings))
        assert [(f.accession, f.fiscal_year, f.fiscal_period) for f in derived] == [
            (f.accession, f.fiscal_year, f.fiscal_period) for f in company.filings
        ]


def _pinned() -> list[tuple[str, str, int, str, FormType]]:
    """Every pinned filing as a test case, so each gets its own pass or fail."""
    pinned = []
    for name in CORPUS_FILES:
        corpus = Corpus.load(REPO_ROOT / "config" / name)
        pinned += [
            (name, c.ticker, f.fiscal_year, f.fiscal_period, f.form)
            for c in corpus.companies
            for f in c.filings
        ]
    return pinned


def filing_path(ticker: str, year: int, period: str, form: FormType) -> Path:
    """Where a pinned filing was downloaded. Skips the test if it hasn't been."""
    settings = get_settings()
    for name in CORPUS_FILES:
        for company in Corpus.load(REPO_ROOT / "config" / name).companies:
            if company.ticker != ticker:
                continue
            for filing in company.filings:
                if (filing.fiscal_year, filing.fiscal_period, filing.form) == (year, period, form):
                    path = (
                        settings.raw_filings_dir
                        / ticker
                        / filing.accession
                        / filing.primary_document
                    )
                    if not path.exists():
                        pytest.skip(f"{path} is not downloaded")
                    return path
    raise LookupError(f"{ticker} {form} {period} {year} is not pinned")


@cache  # Parsing a 10-K takes seconds, and many tests read the same filing.
def parsed(ticker: str, year: int, period: str, form: FormType) -> ParsedFiling:
    path = filing_path(ticker, year, period, form)
    html = path.read_text(encoding="utf-8", errors="replace")
    return parse_filing(html, form, path.parent.name)


def text_of(filing: ParsedFiling, section: Section) -> str:
    """A section's paragraphs joined into one string, for substring checks."""
    return " ".join(paragraph.text for paragraph in filing.sections[section].paragraphs)


@pytest.mark.parametrize(
    ("corpus", "ticker", "year", "period", "form"),
    _pinned(),
    ids=lambda value: str(getattr(value, "value", value)),
)
def test_filing_passes_the_gate(corpus, ticker, year, period, form):
    filing = parsed(ticker, year, period, form)
    # A known rejection must keep being rejected; if it starts passing, someone should look.
    if (ticker, year, period) in KNOWN_REJECTIONS:
        assert not filing.passed
        return
    assert filing.passed, filing.problems
    # Redundant with the gate on purpose: these three are the table checks, asserted by name.
    audit = filing.table_audit
    assert audit.lost_sentences == ()
    assert audit.leftover_tables == ()
    assert audit.numeric_paragraphs == ()


@pytest.mark.parametrize(
    ("ticker", "year", "period", "form", "parser"),
    [
        # edgartools handles these; the others need the fallback's layouts.
        ("HSY", 2025, "FY", K, ParserName.EDGARTOOLS),
        ("MAT", 2025, "FY", K, ParserName.EDGARTOOLS),
        ("AAPL", 2026, "FY", K, ParserName.EDGARTOOLS),
        ("TR", 2025, "FY", K, ParserName.EDGARTOOLS),
        ("SWK", 2025, "FY", K, ParserName.FALLBACK),
        ("GE", 2025, "FY", K, ParserName.FALLBACK),
        ("GE", 2026, "Q2", Q, ParserName.FALLBACK),
        ("JPM", 2025, "FY", K, ParserName.FALLBACK),
        ("CVX", 2025, "FY", K, ParserName.FALLBACK),
    ],
)
def test_expected_parser_is_used(ticker, year, period, form, parser):
    """A change here means a filing moved between parsers, which deserves a look."""
    assert parsed(ticker, year, period, form).parser is parser


@pytest.mark.parametrize(
    ("ticker", "year", "period", "form", "section", "passage", "wrong_section"),
    [
        ("HSY", 2026, "Q2", Q, MDNA, "favorable net price realization", NOTES),
        ("MAT", 2026, "Q2", Q, MDNA, "gross incremental tariff costs of 170 basis points", NOTES),
        # GE: no item headings; pages come from the cross-reference index.
        ("GE", 2025, "FY", K, RISK, "The following discussion of the material factors", MDNA),
        ("GE", 2025, "FY", K, NOTES, "Our financial statements are prepared in conformity", RISK),
        # JPMorgan: Item 7 points to MD&A later in the filing.
        (
            "JPM",
            2025,
            "FY",
            K,
            MDNA,
            "The following is Management’s discussion and analysis",
            NOTES,
        ),
        (
            "JPM",
            2025,
            "FY",
            K,
            NOTES,
            "accounting and financial reporting policies of JPMorganChase",
            MDNA,
        ),
        # Chevron: the financial section sits after Item 14.
        ("CVX", 2025, "FY", K, MDNA, "Chevron Corporation is a global energy company", NOTES),
        ("CVX", 2025, "FY", K, NOTES, "Consolidated Financial Statements are prepared", MDNA),
    ],
)
def test_known_passage_is_in_the_right_section(
    ticker, year, period, form, section, passage, wrong_section
):
    # Each passage was read from the filing by hand, so this tests where text landed.
    filing = parsed(ticker, year, period, form)
    assert passage in text_of(filing, section)
    assert passage not in text_of(filing, wrong_section)


def test_att_subscriber_counts_are_not_stored_as_dollars():
    """They're "(in 000s)" under a "Dollars in millions" banner: 90,255 became $90 billion."""
    path = filing_path("T", 2025, "Q3", Q)
    # The held-out file knows this filing's period end, which the parser needs.
    filing = next(
        f
        for company in Corpus.load(REPO_ROOT / "config" / "parser_holdout3.yaml").companies
        for f in company.filings
        if f.accession == path.parent.name
    )
    context = FilingContext(filing.accession, Q, 2025, "Q3", filing.period_end)
    html = path.read_text(encoding="utf-8", errors="replace")
    # The AT&T tables that label themselves "(in 000s)".
    shorthand = [
        table
        for table in parse_tables(html, context)
        if any(cell == "(in 000s)" for row in table.rows for cell in row)
    ]
    assert shorthand
    for table in shorthand:
        assert not table.facts
        assert table.needs_text_chunk
    assert any("90,255" in table.as_text for table in shorthand)


def test_national_visions_equity_row_no_longer_reaches_the_text():
    # A long balance-sheet row used to be read as prose and left in the notes.
    filing = parsed("EYE", 2025, "FY", K)
    assert "86,278,538 and 85,444,263 shares issued" not in text_of(filing, NOTES)


@pytest.mark.parametrize(("year", "period"), [(2026, "Q2"), (2026, "Q1")])
def test_cosmos_health_keeps_its_correct_edgartools_parse(year, period):
    # These 10-Qs were wrongly rejected once; the fallback should not replace a correct parse.
    filing = parsed("COSM", year, period, Q)
    assert filing.passed, filing.problems
    assert filing.parser is ParserName.EDGARTOOLS
    assert filing.sections[NOTES].char_count > 8_000


def test_chevron_mdna_does_not_contain_its_statements():
    """edgartools passed every length check here while MD&A ran through the statements."""
    filing = parsed("CVX", 2025, "FY", K)
    assert "See accompanying Notes to the Consolidated" not in text_of(filing, MDNA)
    assert filing.parser is ParserName.FALLBACK


def test_stanley_notes_exclude_exhibits_and_signatures():
    # Stanley's statements are filed after Item 15, so the end of the notes must be found.
    filing = parsed("SWK", 2025, "FY", K)
    notes = text_of(filing, NOTES)
    assert "SIGNIFICANT ACCOUNTING POLICIES" in " ".join(
        p.heading or "" for p in filing.sections[NOTES].paragraphs
    )
    assert "Pursuant to the requirements of the Securities Exchange Act" not in notes


@pytest.mark.parametrize(
    ("ticker", "year", "period", "form", "footnote"),
    [
        ("CVX", 2025, "FY", K, "acquired Guyana assets as part of the acquisition of Hess"),
        ("GE", 2025, "FY", K, "are a subset of Commercial Engines"),
    ],
)
def test_footnote_inside_a_removed_table_survives(ticker, year, period, form, footnote):
    # Footnotes share a table with the figures. They must be moved out before it is dropped.
    filing = parsed(ticker, year, period, form)
    preserved = [row for table in filing.removed_tables for row in table.preserved_rows]
    assert any(footnote in row for row in preserved)
    assert any(footnote in text_of(filing, section) for section in filing.sections)


@pytest.mark.parametrize(
    ("ticker", "year", "period", "form", "row_label"),
    [
        ("GE", 2025, "FY", K, "gross additions to property"),
        ("TR", 2025, "FY", K, "Net product sales $"),
        ("MAT", 2025, "FY", K, "Foreign currency forward exchange and other contracts (a) $"),
    ],
)
def test_rows_of_figures_do_not_reach_the_text(ticker, year, period, form, row_label):
    # The opposite check: rows of figures must be dropped, wherever the layout hides them.
    filing = parsed(ticker, year, period, form)
    for section in filing.sections:
        assert row_label not in text_of(filing, section)


# --- Table figures against XBRL's answer key, and XBRL against the SEC's own API ---

# The share of tagged cells the parser may get wrong. The measured rate is near 100% (value
# 100%, period 99.8% on the fourth held-out test); 99% leaves a little room for new layouts
# while still catching a real regression.
MIN_TAGGED_CELL_ACCURACY = 0.99


def _downloaded_filings():
    """Every pinned filing that is on disk, as (company, filing, html)."""
    settings = get_settings()
    for name in CORPUS_FILES:
        for company in Corpus.load(REPO_ROOT / "config" / name).companies:
            for filing in company.filings:
                path = (
                    settings.raw_filings_dir
                    / company.ticker
                    / filing.accession
                    / filing.primary_document
                )
                if path.exists():
                    yield company, filing, path.read_text(encoding="utf-8", errors="replace")


def test_table_parser_agrees_with_tagged_cells():
    """Parsing doesn't read tags, so tagged cells are an answer key for it."""
    # verify=False below turns off the tag check inside the parser, so the parse is judged alone.
    checked = agreed = 0
    for _, filing, html in _downloaded_filings():
        context = FilingContext(
            filing.accession,
            filing.form,
            filing.fiscal_year,
            filing.fiscal_period,
            filing.period_end,
        )
        for table in parse_tables(html, context, verify=False):
            for fact in table.facts:
                if fact.tagged and fact.xbrl_end_date is not None:
                    checked += 1
                    agreed += disagrees_with_tag(fact) is None
    if not checked:
        pytest.skip("no filings downloaded")
    assert agreed / checked >= MIN_TAGGED_CELL_ACCURACY


def test_xbrl_loader_matches_the_secs_company_facts_api():
    """The API is a separate source for company-wide values, by filing and concept."""
    settings = get_settings()
    checked = agreed = 0
    for company, filing, html in _downloaded_filings():
        path = settings.raw_filings_dir / company.ticker / "companyfacts.json"
        # Build the SEC API's view of this filing: (concept, period end) -> values reported.
        api: dict[tuple[str, str], set[float]] = {}
        for taxonomy, concepts in json.loads(path.read_text(encoding="utf-8"))["facts"].items():
            for concept, content in concepts.items():
                for entries in content["units"].values():
                    for entry in entries:
                        if entry["accn"] == filing.accession:
                            key = (f"{taxonomy}:{concept}", entry["end"])
                            api.setdefault(key, set()).add(float(entry["val"]))
        for row in xbrl_rows(parse_html_tree(html), company, filing):
            key = (row["concept"], row["end_date"].isoformat())
            # The API has only company-wide values, so segment facts (dimensions) are skipped.
            if row["dimensions"] is None and key in api:
                checked += 1
                agreed += any(abs(row["value"] - v) <= abs(v) * 1e-6 + 1e-6 for v in api[key])
    if not checked:
        pytest.skip("no filings downloaded")
    assert agreed / checked >= MIN_TAGGED_CELL_ACCURACY


def test_bare_year_columns_land_on_fiscal_year_ends():
    """Western Digital's June 27 year end read as June 30 before this was fixed."""
    for company, filing, html in _downloaded_filings():
        # This test covers 10-Ks only.
        if filing.form is not K:
            continue
        contexts = read_contexts(parse_html_tree(html))
        year_ends = reporting_period_ends(contexts, K)
        context = FilingContext(
            filing.accession,
            filing.form,
            filing.fiscal_year,
            filing.fiscal_period,
            filing.period_end,
        )
        for table in parse_tables(html, context):
            for fact in table.facts:
                if re.fullmatch(r"(Fiscal |FY ?)?(19|20)\d\d", fact.column_label):
                    # Allowed: a real year end, or an older year before any the filing reports.
                    assert fact.end_date in year_ends or fact.end_date < min(year_ends), (
                        company.ticker,
                        fact.column_label,
                        fact.end_date,
                    )


def test_western_digital_fiscal_2025_ends_on_june_27():
    # Its 52/53-week year ends on a Friday, and a "2025" column once read as June 30.
    company = Corpus.load(REPO_ROOT / "config" / "parser_holdout2.yaml").company("WDC")
    filing = next(f for f in company.filings if f.form is K and f.fiscal_year == 2026)
    path = get_settings().raw_filings_dir / "WDC" / filing.accession / filing.primary_document
    if not path.exists():
        pytest.skip(f"{path} is not downloaded")
    context = FilingContext(
        filing.accession, filing.form, filing.fiscal_year, filing.fiscal_period, filing.period_end
    )
    ends = {
        fact.end_date
        for table in parse_tables(path.read_text(encoding="utf-8", errors="replace"), context)
        for fact in table.facts
        if fact.column_label == "2025"
    }
    assert date(2025, 6, 27) in ends
    assert date(2025, 6, 30) not in ends


def test_no_percent_column_in_the_database_is_stored_as_dollars():
    # Reads the real database that `sec-agent ingest` built, so it is skipped before that.
    settings = get_settings()
    if not settings.duckdb_path.exists():
        pytest.skip("run `sec-agent ingest` first")
    with duckdb.connect(settings.duckdb_path, read_only=True) as db:
        rows = db.execute(
            "SELECT ticker, column_label, raw_text FROM facts WHERE source = 'table' "
            "AND unit = 'USD' AND regexp_matches(column_label, '(?i)percent|% of') LIMIT 5"
        ).fetchall()
    assert rows == []


def test_mks_plan_asset_percentages_are_not_dollars():
    # MKS's plan-asset table has a "Percentage" column with no % sign beside the values.
    company = Corpus.load(REPO_ROOT / "config" / "parser_holdout2.yaml").company("MKSI")
    filing = next(f for f in company.filings if f.form is K and f.fiscal_year == 2025)
    path = get_settings().raw_filings_dir / "MKSI" / filing.accession / filing.primary_document
    if not path.exists():
        pytest.skip(f"{path} is not downloaded")
    context = FilingContext(
        filing.accession, filing.form, filing.fiscal_year, filing.fiscal_period, filing.period_end
    )
    tables = parse_tables(path.read_text(encoding="utf-8", errors="replace"), context, verify=False)
    percent_columns = [
        fact for table in tables for fact in table.facts if "Percentage" in fact.column_label
    ]
    assert percent_columns
    assert {fact.unit for fact in percent_columns} == {UNIT_PERCENT}


@pytest.mark.parametrize(
    ("ticker", "words", "expected_heading"),
    [
        ("MAT", "Fisher-Price", "Segment"),
        ("HSY", "Segment margin", None),
    ],
)
def test_non_xbrl_figures_are_searchable(ticker, words, expected_heading, tmp_path):
    # The Financial Agent finds untagged table figures by label; the heading says which table.
    settings = get_settings()
    if not settings.duckdb_path.exists():
        pytest.skip("run `sec-agent ingest` first")
    with duckdb.connect(settings.duckdb_path, read_only=True) as db:
        rows = db.execute(
            "SELECT heading FROM facts WHERE ticker = ? AND source = 'table' AND name = ?",
            [ticker, words],
        ).fetchall()
    assert rows
    if expected_heading:
        assert any(expected_heading in (row[0] or "") for row in rows)


@pytest.mark.parametrize("year", [2025, 2024])
def test_everus_notes_keep_the_notes_edgartools_filed_under_signatures(year):
    # Everus (ECG): edgartools filed Notes 12 to 15 under signatures; the fallback keeps them.
    filing = parsed("ECG", year, "FY", K)
    assert filing.passed
    assert filing.parser is ParserName.FALLBACK
    assert any("part_iv_signatures has note 12" in p for p in filing.primary_problems)
    notes = text_of(filing, NOTES)
    assert "reportable segments are those" in notes
    assert "repaid $27.0 million" in notes


@pytest.mark.parametrize("year", [2025, 2024])
def test_att_market_risk_holding_audit_reports_is_rejected(year):
    # A known, unfixed error (see KNOWN_REJECTIONS). This test keeps the gate catching it.
    filing = parsed("T", year, "FY", K)
    assert any("market_risk: contains another part's heading" in p for p in filing.problems)


def rendered_notes(ticker: str, year: int, period: str, form: FormType) -> str:
    """The notes as the gate reads them. A paragraph's `heading` keeps only the nearest one."""
    filing = parsed(ticker, year, period, form)
    assert filing.passed, filing.problems
    html = filing_path(ticker, year, period, form).read_text(encoding="utf-8", errors="replace")
    stripped = strip_data_tables(html).html
    rendered = (
        parse_fallback(stripped, form).rendered
        if filing.parser is ParserName.FALLBACK
        else render_items(stripped, form)
    )
    return rendered[filing.sections[NOTES].source_item]


@pytest.mark.parametrize(
    ("ticker", "year", "period", "form", "last_note"),
    [
        ("HSY", 2025, "FY", K, 19),
        ("HSY", 2024, "FY", K, 19),
        ("HSY", 2026, "Q2", Q, 18),
        ("HSY", 2025, "Q2", Q, 18),
        ("HSY", 2026, "Q1", Q, 18),
        ("MAT", 2025, "FY", K, 18),
        ("MAT", 2024, "FY", K, 17),
        ("MAT", 2026, "Q2", Q, 23),
        ("MAT", 2025, "Q2", Q, 22),
        ("MAT", 2026, "Q1", Q, 23),
    ],
)
def test_analysis_company_notes_run_through_their_last_note(ticker, year, period, form, last_note):
    # The last note numbers come from each filing's own note headings.
    numbers = _note_numbers(rendered_notes(ticker, year, period, form))
    assert numbers == set(range(1, last_note + 1))


@pytest.mark.parametrize(
    ("year", "period", "form", "last_letter"),
    [
        (2025, "FY", K, "S"),
        (2024, "FY", K, "S"),
        (2026, "Q2", Q, "Q"),
        (2025, "Q2", Q, "Q"),
        (2026, "Q1", Q, "Q"),
    ],
)
def test_stanley_notes_run_through_their_last_letter(year, period, form, last_letter):
    """Stanley letters its notes, so the gate's numbering check can't see a cut here."""
    text = rendered_notes("SWK", year, period, form)
    letters = set(re.findall(r"(?m)^[^\S\n]*([A-Z])[.)][^\S\n]+[A-Z][A-Z ,&’'\-]{3,}", text))
    assert letters == {chr(code) for code in range(ord("A"), ord(last_letter) + 1)}
