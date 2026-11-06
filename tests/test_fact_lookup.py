"""Finding a tagged dollar fact by value (`FactLookup`), and opening the store read-only."""

from __future__ import annotations

import time
from datetime import date

import duckdb
import pytest

from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.contracts import ChunkLookup, FactLookup
from sec_agent.edgar.corpus import Corpus, CorpusCompany, CorpusFiling
from sec_agent.finance.fact_store import FactStore
from sec_agent.finance.metrics import MetricCatalog, MetricsConfig
from sec_agent.ingest.audit import TableAudit
from sec_agent.ingest.pipeline import ParsedFiling, ParserName
from sec_agent.schemas import ComparisonType, FormType

ACCESSION = "acc-1"


def add_fact(
    store: FactStore,
    fact_id: str,
    value: float,
    *,
    accession: str = ACCESSION,
    unit: str = "usd",
    segment: str | None = None,
) -> str:
    store._insert(
        "xbrl_facts",
        [
            {
                "fact_id": fact_id,
                "ticker": "AAA",
                "accession": accession,
                "form": "10-K",
                "fiscal_year": 2025,
                "fiscal_period": "FY",
                "concept": "us-gaap:Revenues",
                "start_date": None,
                "end_date": date(2025, 12, 31),
                "duration": "12m",
                "value": value,
                "unit": unit,
                "segment": segment,
                "dimensions": None,
            }
        ],
    )
    return fact_id


@pytest.fixture
def store(settings):
    with FactStore(settings.duckdb_path) as opened:
        yield opened


def test_the_store_satisfies_fact_lookup(store):
    assert isinstance(store, FactLookup)


def test_an_exact_value_finds_its_fact(store):
    add_fact(store, "f1", 661_949_000.0)
    assert store.find_usd_fact(ACCESSION, 661_949_000.0, 0.0) == "f1"


def test_a_value_with_no_fact_returns_none(store):
    add_fact(store, "f1", 661_949_000.0)
    assert store.find_usd_fact(ACCESSION, 661_949.0, 500.0) is None


def test_a_filing_with_no_facts_returns_none(store):
    assert store.find_usd_fact("nothing", 1.0, 1.0) is None


def test_the_tolerance_is_inclusive(store):
    add_fact(store, "f1", 1000.0)
    assert store.find_usd_fact(ACCESSION, 1500.0, 500.0) == "f1"
    assert store.find_usd_fact(ACCESSION, 500.0, 500.0) == "f1"
    assert store.find_usd_fact(ACCESSION, 1500.5, 500.0) is None
    assert store.find_usd_fact(ACCESSION, 499.5, 500.0) is None


@pytest.mark.parametrize(("stored", "asked"), [(-100.0, 100.0), (100.0, -100.0), (-100.0, -100.0)])
def test_values_compare_as_magnitudes(store, stored, asked):
    # A table shows "(151,372)", and a tagged fact's sign depends on its concept.
    add_fact(store, "f1", stored)
    assert store.find_usd_fact(ACCESSION, asked, 0.0) == "f1"


def test_only_that_filing_is_searched(store):
    add_fact(store, "other", 100.0, accession="acc-2")
    assert store.find_usd_fact(ACCESSION, 100.0, 1.0) is None
    assert store.find_usd_fact("acc-2", 100.0, 1.0) == "other"


def test_segment_values_count(store):
    add_fact(store, "seg", 100.0, segment="a:Toys")
    assert store.find_usd_fact(ACCESSION, 100.0, 0.0) == "seg"


def test_parsed_table_facts_do_not_count(store):
    # They come from the same tables, so a table would confirm itself.
    store._insert(
        "table_facts",
        [
            {
                "fact_id": "table-fact",
                "ticker": "AAA",
                "accession": ACCESSION,
                "form": "10-K",
                "fiscal_year": 2025,
                "fiscal_period": "FY",
                "table_index": 1,
                "row_label": "Net sales",
                "column_label": "2025",
                "end_date": date(2025, 12, 31),
                "raw_text": "100",
                "value": 100.0,
                "unit": "USD",
                "is_share_of": False,
                "ambiguous": False,
            }
        ],
    )
    assert store.find_usd_fact(ACCESSION, 100.0, 1.0) is None


@pytest.mark.parametrize(
    "unit",
    [
        "usd",
        "USD",
        "U_USD",
        "u_usd",
        "Unit_Standard_USD_TBPlctbz2kak3ReC6sW3DQ",
        "Unit_Standard_USD_Fx2jV8ifRkKihxPc1LY-aA",
    ],
)
def test_every_spelling_of_dollars_counts(store, unit):
    add_fact(store, "f1", 100.0, unit=unit)
    assert store.find_usd_fact(ACCESSION, 100.0, 0.0) == "f1"


@pytest.mark.parametrize(
    "unit",
    [
        "usdPerShare",
        "USDPerShare",
        "USDPShares",
        "U_UnitedStatesOfAmericaDollarsShare",
        "Unit_Divide_USD_shares_Oi9Vq2KZ7kuNgMR7wa45Qg",
        "usdPerMMBTU",
        "usdPerBbl",
        "usdPerUnit",
        "shares",
        "U_shares",
        "pure",
        "eur",
        "Unit_Standard_pure_abc",
    ],
)
def test_other_units_do_not_count(store, unit):
    add_fact(store, "f1", 100.0, unit=unit)
    assert store.find_usd_fact(ACCESSION, 100.0, 0.0) is None


def test_a_fact_with_no_unit_does_not_count(store):
    add_fact(store, "f1", 100.0, unit=None)
    assert store.find_usd_fact(ACCESSION, 100.0, 0.0) is None


def test_several_matches_always_return_the_same_fact(store):
    # Inserted out of order, so the answer can't depend on the order rows come back in.
    add_fact(store, "z-fact", 100.0)
    add_fact(store, "a-fact", 100.0)
    add_fact(store, "m-fact", 100.0)
    answers = {store.find_usd_fact(ACCESSION, 100.0, 0.0) for _ in range(5)}
    assert answers == {"a-fact"}


def test_the_nearest_of_several_matches_is_returned(store):
    add_fact(store, "a-far", 104.0)
    add_fact(store, "z-near", 101.0)
    assert store.find_usd_fact(ACCESSION, 100.0, 10.0) == "z-near"


def test_a_tie_on_distance_goes_to_the_lowest_id(store):
    add_fact(store, "b-above", 105.0)
    add_fact(store, "a-below", 95.0)
    assert store.find_usd_fact(ACCESSION, 100.0, 10.0) == "a-below"


def test_a_new_store_gives_the_same_answer(settings):
    with FactStore(settings.duckdb_path) as first:
        add_fact(first, "b", 100.0)
        add_fact(first, "a", 100.0)
        answer = first.find_usd_fact(ACCESSION, 100.0, 0.0)
    with FactStore(settings.duckdb_path) as second:
        assert second.find_usd_fact(ACCESSION, 100.0, 0.0) == answer == "a"


def test_repeat_calls_do_not_query_the_database_again(store):
    add_fact(store, "f1", 100.0)
    store.find_usd_fact(ACCESSION, 100.0, 0.0)

    class NoQueries:
        def execute(self, *_: object) -> None:
            raise AssertionError("a repeat call should be answered from memory")

    real = store._db
    store._db = NoQueries()
    try:
        assert store.find_usd_fact(ACCESSION, 100.0, 0.0) == "f1"
        assert store.find_usd_fact(ACCESSION, 7.0, 0.0) is None
    finally:
        store._db = real


def filing_record() -> tuple[CorpusCompany, CorpusFiling, ParsedFiling]:
    pinned = CorpusFiling(
        form=FormType.FORM_10K,
        fiscal_year=2025,
        fiscal_period="FY",
        period_end=date(2025, 12, 31),
        filed_date=date(2026, 2, 1),
        accession=ACCESSION,
        primary_document="doc.htm",
    )
    company = CorpusCompany(ticker="AAA", company="A", cik="0000000001", filings=(pinned,))
    parsed = ParsedFiling(
        accession=ACCESSION,
        form=FormType.FORM_10K,
        parser=ParserName.EDGARTOOLS,
        layout="edgartools",
        sections={},
        coverage=1.0,
        table_audit=TableAudit(tables_total=0, tables_removed=0, prose_rows_preserved=0),
        problems=(),
    )
    return company, pinned, parsed


def test_reloading_a_filing_forgets_its_old_facts(store):
    add_fact(store, "old", 100.0)
    assert store.find_usd_fact(ACCESSION, 100.0, 0.0) == "old"
    company, pinned, parsed = filing_record()
    store.load_filing(company, pinned, "<html><body></body></html>", parsed)
    assert store.find_usd_fact(ACCESSION, 100.0, 0.0) is None


def test_removing_a_filing_forgets_its_facts(store):
    add_fact(store, "old", 100.0)
    assert store.find_usd_fact(ACCESSION, 100.0, 0.0) == "old"
    store.remove_filing(ACCESSION)
    assert store.find_usd_fact(ACCESSION, 100.0, 0.0) is None


# --- Opening the store read-only ---


@pytest.fixture
def filled(settings):
    with FactStore(settings.duckdb_path) as store:
        add_fact(store, "f1", 100.0)
    return settings.duckdb_path


def test_a_read_only_store_answers_every_lookup(filled):
    with FactStore(filled, read_only=True) as store:
        assert isinstance(store, FactLookup)
        assert isinstance(store, ChunkLookup)
        assert store.find_usd_fact(ACCESSION, 100.0, 0.0) == "f1"
        assert store.get_chunk("nothing") is None
        assert store.count("xbrl_facts") == 1
        assert store.search("AAA", "Revenues")[0].fact_id == "f1"


def test_a_read_only_store_refuses_writes(filled):
    with FactStore(filled, read_only=True) as store, pytest.raises(duckdb.Error):
        store.replace_chunks(ACCESSION, [])


def test_two_read_only_stores_can_share_the_file(filled):
    with FactStore(filled, read_only=True) as first, FactStore(filled, read_only=True) as second:
        assert first.count("xbrl_facts") == second.count("xbrl_facts") == 1


def test_read_only_needs_the_file_to_exist(tmp_path):
    with pytest.raises(duckdb.Error):
        FactStore(tmp_path / "missing.duckdb", read_only=True)
    # It must not have made the file, or the directories on the way to it.
    assert not (tmp_path / "missing.duckdb").exists()


# --- The real filings ---


@pytest.mark.corpus
def test_real_figures_find_the_right_fact_and_stay_cheap():
    settings = get_settings()
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    accession = corpus.comparison("HSY", ComparisonType.YOY_10K).current.accession
    with FactStore(settings.duckdb_path, read_only=True) as store:
        # Lane 2's case: "$ | 661,949" in a table with no dollar scale is in thousands.
        assert store.find_usd_fact(accession, 661_949_000.0, 500.0) is not None
        assert store.find_usd_fact(accession, 661_949.0, 0.5) is None
        # Net sales, as the filing tags it.
        assert "us-gaap" in store.find_usd_fact(accession, 11_692_576_000.0, 500.0)

        # About as many calls as one citation makes.
        started = time.perf_counter()
        for number in range(90):
            store.find_usd_fact(accession, float(number) * 1000.0, 0.5)
        assert time.perf_counter() - started < 0.5


@pytest.mark.corpus
def test_the_metric_catalog_works_on_a_read_only_store():
    settings = get_settings()
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    with FactStore(settings.duckdb_path, read_only=True) as store:
        catalog = MetricCatalog(store, corpus, MetricsConfig.load())
        assert catalog.get_metric("HSY:gross_margin:Q2 2026:3m") is not None
