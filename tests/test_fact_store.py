"""The store: what goes in, what's searchable, and what's refused."""

from __future__ import annotations

from datetime import date

import pytest

from sec_agent.edgar.corpus import CorpusCompany, CorpusFiling
from sec_agent.finance.fact_store import FactStore, mark_ambiguous
from sec_agent.ingest.audit import TableAudit
from sec_agent.ingest.pipeline import FilingRejectedError, ParsedFiling, ParserName
from sec_agent.ingest.sections import ParsedSection, SectionStatus, clean_paragraphs
from sec_agent.schemas import FormType, Section

# One fake company with one 10-K, shared by every test below.
FILING = CorpusFiling(
    form=FormType.FORM_10K,
    fiscal_year=2025,
    fiscal_period="FY",
    period_end=date(2025, 12, 31),
    filed_date=date(2026, 2, 1),
    accession="acc-1",
    primary_document="doc.htm",
)
COMPANY = CorpusCompany(ticker="AAA", company="A", cik="0000000001", filings=(FILING,))

# The hidden XBRL header every inline filing has: a full-year period (c1) and the same period
# broken out for one business segment (seg).
RESOURCES = (
    '<ix:header><ix:resources><xbrli:context id="c1"><xbrli:period>'
    "<xbrli:startdate>2025-01-01</xbrli:startdate><xbrli:enddate>2025-12-31</xbrli:enddate>"
    "</xbrli:period></xbrli:context>"
    '<xbrli:context id="seg"><xbrli:entity><xbrli:segment><xbrldi:explicitmember '
    'dimension="us-gaap:StatementBusinessSegmentsAxis">a:Toys</xbrldi:explicitmember>'
    "</xbrli:segment></xbrli:entity><xbrli:period>"
    "<xbrli:startdate>2025-01-01</xbrli:startdate><xbrli:enddate>2025-12-31</xbrli:enddate>"
    "</xbrli:period></xbrli:context></ix:resources></ix:header>"
)


def tag(value: str, context: str = "c1", concept: str = "us-gaap:Revenues") -> str:
    """An inline XBRL number. scale="6" means the displayed value is in millions."""
    return (
        f'<ix:nonfraction name="{concept}" contextref="{context}" scale="6" unitref="usd">'
        f"{value}</ix:nonfraction>"
    )


def data_table(*rows: tuple[str, str]) -> str:
    """A table needs two figures to count as a data table, so each gets a filler row."""
    rows = (*rows, ("Other items", "7"))
    body = "".join(f"<tr><td>{label}</td><td>$</td><td>{value}</td></tr>" for label, value in rows)
    return (
        "<table><tr><td>(Dollars in millions)</td></tr>"
        f"<tr><td></td><td colspan='2'>December 31, 2025</td></tr>{body}</table>"
    )


# The heading and paragraph a table sits under, so the store can attach a section and heading.
HEADING = "Toys Segment"
INTRO = "The following table shows results for this segment of the business."


def parsed_filing(*, problems: tuple[str, ...] = ()) -> ParsedFiling:
    """A parse record. Pass problems to make it a rejected filing, which the store must refuse."""
    paragraphs = clean_paragraphs(f"{HEADING}\n\n{INTRO}", "acc-1:mdna")
    return ParsedFiling(
        accession="acc-1",
        form=FormType.FORM_10K,
        parser=ParserName.EDGARTOOLS,
        layout="edgartools",
        sections={
            Section.MDNA: ParsedSection(
                section=Section.MDNA,
                status=SectionStatus.CONTENT,
                source_item="part_ii_item_7",
                paragraphs=paragraphs,
            )
        },
        coverage=1.0,
        table_audit=TableAudit(tables_total=1, tables_removed=1, prose_rows_preserved=0),
        problems=problems,
    )


def html(*parts: str) -> str:
    """A filing: the XBRL header followed by whatever parts the test needs."""
    return f"<html><body>{RESOURCES}{''.join(parts)}</body></html>"


@pytest.fixture
def store(tmp_path):
    # A fresh DuckDB file per test, closed afterwards.
    with FactStore(tmp_path / "facts.duckdb") as opened:
        yield opened


def load(store: FactStore, *parts: str):
    """Load a filing built from the parts into the store."""
    return store.load_filing(COMPANY, FILING, html(*parts), parsed_filing())


def test_tagged_figures_are_xbrl_facts_and_untagged_are_table_facts(store):
    load(store, data_table(("Revenue", tag("100")), ("Gross billings", "250.7")))
    assert store.count("xbrl_facts") == 1
    assert store.count("table_facts") == 2
    found = store.search("AAA", "billings")
    assert [(f.source, f.value) for f in found] == [("table", pytest.approx(250_700_000))]
    assert found[0].raw_text == "250.7"


def test_xbrl_segment_facts_are_labeled(store):
    load(store, f"<p>{tag('5', 'seg')}</p>")
    fact = store.search("AAA", "Revenues")[0]
    assert fact.segment == "a:Toys"
    assert "StatementBusinessSegmentsAxis" in fact.dimensions


def test_repeated_tag_is_stored_once(store):
    load(store, f"<p>{tag('100')}</p><p>{tag('100')}</p>")
    assert store.count("xbrl_facts") == 1


def test_untagged_copy_of_an_xbrl_figure_is_not_searchable_twice(store):
    """MD&A repeats net sales without a tag. That figure is already in XBRL."""
    load(
        store,
        f"<p>{tag('1,234.5')}</p>",
        data_table(("Net sales in MD&amp;A", "1,234.5"), ("Gross billings", "250.7")),
    )
    assert store.count("table_facts") == 3
    assert {f.name for f in store.search("AAA", source="table")} == {
        "Gross billings",
        "Other items",
    }


def test_rounded_copy_of_an_xbrl_figure_is_not_searchable_twice(store):
    """Hershey: MD&A's 2,787.3 in millions is XBRL's 2,787,298 in thousands."""
    tagged = (
        '<ix:nonfraction name="us-gaap:Revenues" contextref="c1" scale="3" unitref="usd">'
        "2,787,298</ix:nonfraction>"
    )
    load(store, f"<p>{tagged}</p>", data_table(("Net sales", "2,787.3")))
    assert store.search("AAA", "Net sales", source="table") == []


def test_nearby_but_different_amount_stays_searchable(store):
    # 2,787.5 is outside the half-a-last-digit rounding of 2,787.3, so it is a different figure.
    load(store, f"<p>{tag('2,787.5')}</p>", data_table(("Gross billings", "2,787.3")))
    assert store.search("AAA", "billings", source="table")


def test_facts_named_by_the_query_come_first(store):
    # "Zinc costs" matches only through nearby text; "Gross billings" is named by the query.
    load(
        store,
        "<p>Discussion of gross billings across the business in this year's results.</p>",
        data_table(("Zinc costs", "250.7"), ("Gross billings", "11.2")),
    )
    assert store.search("AAA", "billings", source="table")[0].name == "Gross billings"


def test_short_number_matching_xbrl_is_kept_as_its_own_fact(store):
    """Equal two-digit amounts are often a coincidence, so they aren't treated as copies."""
    load(store, f"<p>{tag('12')}</p>", data_table(("Gross billings", "12")))
    assert store.search("AAA", "billings", source="table")


def test_facts_carry_their_section_and_heading(store):
    load(store, f"<p>{INTRO}</p>", data_table(("Gross billings", "250.7")))
    fact = store.search("AAA", "billings")[0]
    assert fact.section == "mdna"
    assert fact.heading == HEADING


def test_search_covers_the_heading_above_the_table(store):
    """Mattel's "Fisher-Price" rows differ only by the segment named above each table."""
    load(store, f"<p>{INTRO}</p>", data_table(("Fisher-Price", "250.7")))
    assert store.search("AAA", "Fisher-Price Toys")
    assert not store.search("AAA", "Fisher-Price International")


def test_rows_nothing_can_separate_are_not_searchable_and_go_to_text(store):
    # Same label, same heading, different numbers: an agent would have to guess, so neither is
    # offered as a fact and both tables wait as text.
    load(
        store,
        f"<p>{INTRO}</p>",
        data_table(("Fisher-Price", "250.7")),
        f"<p>{INTRO}</p>",
        data_table(("Fisher-Price", "99.9")),
    )
    assert store.search("AAA", "Fisher-Price") == []
    queued = store.unparsed_tables("acc-1")
    assert len(queued) == 2
    assert "told apart" in queued[0]["reasons"]


def test_equal_values_with_the_same_name_are_not_ambiguous():
    rows = [
        {
            "heading": None,
            "table_title": None,
            "row_group": None,
            "row_label": "A",
            "column_label": "2025",
            "end_date": date(2025, 12, 31),
            "duration": None,
            "unit": "USD",
            "value": 1.0,
            "table_index": n,
            "ambiguous": False,
        }
        for n in range(2)
    ]
    assert mark_ambiguous(rows) == set()


def test_unclean_tables_wait_for_the_chunker(store):
    # No "in millions" caption, so the figures can't safely be turned into dollars.
    no_scale = (
        "<table><tr><td></td><td colspan='2'>2025</td></tr>"
        "<tr><td>Expense</td><td>$</td><td>9</td></tr>"
        "<tr><td>Other</td><td>$</td><td>8</td></tr></table>"
    )
    load(store, no_scale)
    assert store.count("table_facts") == 0
    queued = store.unparsed_tables("acc-1")
    assert len(queued) == 1
    assert "Expense | $ | 9" in queued[0]["text"]
    assert queued[0]["reasons"]


def test_table_with_skipped_rows_gives_facts_and_a_text_chunk(store):
    # An opening balance is skipped (its date doesn't match its column), so the table is also
    # queued as text and the skipped figure stays reachable.
    load(store, data_table(("Beginning balance", "1"), ("Ending balance", "2")))
    assert {f.name for f in store.search("AAA", source="table")} == {
        "Ending balance",
        "Other items",
    }
    assert store.count("unparsed_tables") == 1


def test_rejected_filing_cannot_be_loaded(store):
    with pytest.raises(FilingRejectedError):
        store.load_filing(COMPANY, FILING, html(), parsed_filing(problems=("bad section",)))
    assert store.count("xbrl_facts") == 0
    assert not store.is_loaded("acc-1")


def test_reloading_replaces_a_filing_without_duplicates(store):
    # The second load has a different value, so a leftover first copy would show up.
    load(store, data_table(("Gross billings", "250.7")))
    load(store, data_table(("Gross billings", "260.0")))
    assert [f.raw_text for f in store.search("AAA", "billings")] == ["260.0"]


def test_is_loaded_tracks_the_version(store):
    assert not store.is_loaded("acc-1")
    load(store)
    assert store.is_loaded("acc-1")


def test_search_filters_by_company_period_and_source(store):
    # 3 facts: the tagged figure, the table's row, and its "Other items" filler row.
    load(store, f"<p>{tag('100')}</p>", data_table(("Gross billings", "250.7")))
    assert store.search("ZZZ") == []
    assert len(store.search("AAA", end_date=date(2025, 12, 31))) == 3
    assert store.search("AAA", end_date=date(2024, 12, 31)) == []
    assert [f.source for f in store.search("AAA", source="xbrl")] == ["xbrl"]


def test_search_uses_parameters_not_string_building(store):
    """Filing text and queries are untrusted, so neither may reach the SQL as code."""
    load(store, data_table(("Gross billings", "250.7")))
    assert store.search("AAA", "' OR 1=1 --") == []
    assert store.count("table_facts") == 2


def test_filing_rejected_after_loading_can_be_removed(store):
    """AT&T's 10-Ks loaded under older rules, and rejecting them must also clear their facts."""
    load(store, data_table(("Revenue", tag("100")), ("Untagged costs", "40")))
    assert store.is_loaded("acc-1")
    store.remove_filing("acc-1")
    assert not store.is_loaded("acc-1")
    assert store.search("AAA") == []
    assert store.count("unparsed_tables", "AAA") == 0
