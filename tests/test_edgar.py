from __future__ import annotations

import time
from datetime import date

import httpx
import pytest
import respx

from sec_agent.edgar.client import (
    COMPANY_FACTS_URL,
    SUBMISSIONS_PAGE_URL,
    SUBMISSIONS_URL,
    EdgarClient,
    RateLimiter,
)
from sec_agent.edgar.corpus import (
    Corpus,
    CorpusError,
    CorpusFiling,
    assign_fiscal_periods,
    download_corpus,
    fetch_submissions,
    fiscal_tags_by_accession,
    latest_common_quarter,
    periodic_filings,
    select_corpus,
)
from sec_agent.schemas import ComparisonType, FormType

URL = "https://data.sec.gov/test.json"
# A fake company's filings, newest first: three 10-Ks and two years of 10-Qs. Fiscal year ends
# December 31 and quarters end about 91 days apart. Tuples are filing form, fiscal year,
# fiscal period, period end and accession (the accession names the filing in assertions).
ROWS = [
    # form, fy, fp, period end, accession
    ("10-Q", 2026, "Q3", "2026-09-27", "q3-26"),
    ("10-Q", 2026, "Q2", "2026-06-28", "q2-26"),
    ("10-Q", 2026, "Q1", "2026-03-29", "q1-26"),
    ("10-K", 2025, "FY", "2025-12-31", "k-25"),
    ("10-Q", 2025, "Q3", "2025-09-28", "q3-25"),
    ("10-Q", 2025, "Q2", "2025-06-29", "q2-25"),
    ("10-Q", 2025, "Q1", "2025-03-30", "q1-25"),
    ("10-K", 2024, "FY", "2024-12-31", "k-24"),
    ("10-K", 2023, "FY", "2023-12-31", "k-23"),
]


def company_data(rows=ROWS, extra_forms=()):
    """The inputs discover_corpus would fetch for one company, built from the ROWS above."""
    # extra_forms appear in the filing list but have no XBRL facts, like amendments.
    all_rows = [*rows, *extra_forms]
    facts = {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"accn": accession, "fy": fy, "fp": fp}
                        for form, fy, fp, _, accession in rows
                    ]
                }
            }
        }
    }
    return {
        "company": "Example Co",
        "cik": "0000000001",
        "submissions": {
            "filings": {
                "recent": {
                    "form": [r[0] for r in all_rows],
                    "accessionNumber": [r[4] for r in all_rows],
                    "reportDate": [r[3] for r in all_rows],
                    "filingDate": [r[3] for r in all_rows],
                    "primaryDocument": [f"{r[4]}.htm" for r in all_rows],
                }
            }
        },
        "company_facts": {"facts": facts},
    }


def test_client_requires_a_rate_within_sec_limits():
    with pytest.raises(ValueError, match="at most 10"):
        EdgarClient("Name a@b.c", requests_per_second=11)


def test_rate_limiter_spaces_requests():
    # Five waits at 20 per second must take at least four gaps of 0.05 s.
    limiter = RateLimiter(requests_per_second=20)
    start = time.monotonic()
    for _ in range(5):
        limiter.wait()
    assert time.monotonic() - start >= 0.19


@respx.mock
def test_client_sends_declared_user_agent():
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"ok": True}))
    client = EdgarClient("Name a@b.c", requests_per_second=10)
    assert client.get_json(URL) == {"ok": True}
    assert route.calls[0].request.headers["User-Agent"] == "Name a@b.c"


@respx.mock
def test_client_retries_rate_limit_responses():
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json={"ok": True})]
    )
    assert EdgarClient("Name a@b.c").get_json(URL) == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_client_does_not_retry_not_found():
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        EdgarClient("Name a@b.c").get_json(URL)
    assert route.call_count == 1


@respx.mock
def test_download_skips_existing_file(tmp_path):
    route = respx.get(URL).mock(return_value=httpx.Response(200, content=b"data"))
    client = EdgarClient("Name a@b.c", requests_per_second=10)
    destination = tmp_path / "nested" / "file.htm"
    assert client.download(URL, destination) is True
    assert destination.read_bytes() == b"data"
    assert client.download(URL, destination) is False
    assert route.call_count == 1


@respx.mock
def test_download_rejects_empty_response(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, content=b""))
    with pytest.raises(ValueError, match="empty"):
        EdgarClient("Name a@b.c").download(URL, tmp_path / "file.htm")
    assert not (tmp_path / "file.htm").exists()


def test_fiscal_tags_use_most_common_pair():
    facts = {
        "facts": {
            "t": {
                "c": {
                    "units": {
                        "USD": [
                            {"accn": "a", "fy": 2025, "fp": "FY"},
                            {"accn": "a", "fy": 2025, "fp": "FY"},
                            {"accn": "a", "fy": 2024, "fp": "Q3"},
                            {"accn": "b", "fy": None, "fp": None},
                        ]
                    }
                }
            }
        }
    }
    assert fiscal_tags_by_accession(facts) == {"a": (2025, "FY")}


def test_amendments_are_excluded():
    amended = [("10-Q/A", 2025, "Q1", "2025-03-30", "amended")]
    data = company_data(extra_forms=amended)
    accessions = {f.accession for f in periodic_filings(data["submissions"], data["company_facts"])}
    assert "amended" not in accessions
    assert "q1-25" in accessions


def test_filings_on_older_submission_pages_are_found():
    """Heavy filers push their 10-Ks out of the `recent` window (found on JPMorgan)."""
    data = company_data()
    recent = data["submissions"]["filings"]["recent"]
    empty = {key: [] for key in recent}
    submissions = {"filings": {"recent": empty, "older_pages": [recent]}}
    assert periodic_filings(submissions, data["company_facts"]) == periodic_filings(
        data["submissions"], data["company_facts"]
    )


@respx.mock
def test_fetch_submissions_only_follows_pages_in_the_history_window():
    recent = {"form": [], "accessionNumber": []}
    files = [
        {"name": "CIK1-submissions-001.json", "filingTo": "2026-01-15"},
        {"name": "CIK1-submissions-002.json", "filingTo": "2015-01-15"},
    ]
    respx.get(SUBMISSIONS_URL.format(cik="0000000001")).respond(
        json={"filings": {"recent": recent, "files": files}}
    )
    wanted = respx.get(SUBMISSIONS_PAGE_URL.format(name=files[0]["name"])).respond(
        json={"form": ["10-K"]}
    )
    unwanted = respx.get(SUBMISSIONS_PAGE_URL.format(name=files[1]["name"]))

    client = EdgarClient("Test test@example.com", requests_per_second=10)
    result = fetch_submissions(client, "0000000001", today=date(2026, 11, 3))

    assert wanted.called and not unwanted.called
    assert result["filings"]["older_pages"] == [{"form": ["10-K"]}]


def corpus_filing(
    form: str, year: int, period: str, end: str, accession: str = "a"
) -> CorpusFiling:
    """A pinned filing with only the fields the fiscal-calendar logic reads."""
    return CorpusFiling(
        form=FormType(form),
        fiscal_year=year,
        fiscal_period=period,
        period_end=date.fromisoformat(end),
        filed_date=date.fromisoformat(end),
        accession=accession,
        primary_document="doc.htm",
    )


def periods(filings: list[CorpusFiling]) -> dict[str, tuple[int, str]]:
    """Period end date to (fiscal year, fiscal period), for easy assertions."""
    return {f.period_end.isoformat(): (f.fiscal_year, f.fiscal_period) for f in filings}


# Flexsteel's real mistake: its fiscal year ends June 30, yet its March 2026 10-Q is tagged
# Q3 of fiscal 2025, the same label as March 2025. Two quarters would collide on one label.
FLEXSTEEL_AS_TAGGED = [
    corpus_filing("10-K", 2024, "FY", "2024-06-30"),
    corpus_filing("10-K", 2025, "FY", "2025-06-30"),
    corpus_filing("10-K", 2026, "FY", "2026-06-30"),
    corpus_filing("10-Q", 2024, "Q1", "2024-09-30"),
    corpus_filing("10-Q", 2025, "Q2", "2024-12-31"),
    corpus_filing("10-Q", 2025, "Q3", "2025-03-31"),
    corpus_filing("10-Q", 2025, "Q1", "2025-09-30"),
    corpus_filing("10-Q", 2025, "Q2", "2025-12-31"),
    corpus_filing("10-Q", 2025, "Q3", "2026-03-31"),
    corpus_filing("10-Q", 2025, "Q1", "2026-09-30"),
]


def test_wrong_filer_tags_are_corrected_from_the_fiscal_calendar():
    """Flexsteel tags March 2026 as Q3 of fiscal 2025, the same as March 2025."""
    result = periods(assign_fiscal_periods(FLEXSTEEL_AS_TAGGED))
    assert result["2026-03-31"] == (2026, "Q3")
    assert result["2025-03-31"] == (2025, "Q3")
    assert result["2025-12-31"] == (2026, "Q2")
    assert result["2024-09-30"] == (2025, "Q1")


def test_quarter_after_the_latest_10k_is_projected():
    assert periods(assign_fiscal_periods(FLEXSTEEL_AS_TAGGED))["2026-09-30"] == (2027, "Q1")


def test_filer_tag_is_kept_only_when_corrected():
    by_end = {f.period_end.isoformat(): f for f in assign_fiscal_periods(FLEXSTEEL_AS_TAGGED)}
    corrected = by_end["2026-03-31"]
    assert (corrected.filer_fiscal_year, corrected.filer_fiscal_period) == (2025, "Q3")
    assert by_end["2025-03-31"].filer_fiscal_year is None


def test_corrected_quarters_no_longer_collide_when_selecting():
    # Rebuild the SEC inputs from the mistagged filings, then run the real selection.
    data = {
        "form": [f.form.value for f in FLEXSTEEL_AS_TAGGED],
        "accessionNumber": [f"a{n}" for n, _ in enumerate(FLEXSTEEL_AS_TAGGED)],
        "reportDate": [f.period_end.isoformat() for f in FLEXSTEEL_AS_TAGGED],
        "filingDate": [f.period_end.isoformat() for f in FLEXSTEEL_AS_TAGGED],
        "primaryDocument": ["doc.htm"] * len(FLEXSTEEL_AS_TAGGED),
    }
    entries = [
        {"accn": f"a{n}", "fy": f.fiscal_year, "fp": f.fiscal_period}
        for n, f in enumerate(FLEXSTEEL_AS_TAGGED)
    ]
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": entries}}}}}
    corpus = select_corpus(
        {
            "FLXS": {
                "company": "Flexsteel",
                "cik": "0000037472",
                "submissions": {"filings": {"recent": data}},
                "company_facts": facts,
            }
        }
    )
    ends = {f.period_end.isoformat() for f in corpus.companies[0].filings}
    assert {"2026-03-31", "2025-03-31", "2025-12-31"} <= ends


@pytest.mark.parametrize(
    ("label", "year_end", "quarter_end", "expected"),
    [
        ("Walmart labels by the year it ends in", 2026, "2026-07-31", (2027, "Q2")),
        ("Target labels by the year it starts in", 2025, "2026-08-01", (2026, "Q2")),
    ],
)
def test_january_year_ends_keep_each_filers_own_label(label, year_end, quarter_end, expected):
    # The 10-K labels are the filers' own; the 10-Q's tag (1900) is deliberately nonsense, so
    # the result can only have come from the calendar.
    filings = [
        corpus_filing("10-K", year_end - 1, "FY", "2025-01-31"),
        corpus_filing("10-K", year_end, "FY", "2026-01-31"),
        corpus_filing("10-Q", 1900, "Q1", quarter_end),
    ]
    assert periods(assign_fiscal_periods(filings))[quarter_end] == expected


def test_53_week_year_ending_in_january_keeps_its_label():
    """Stanley's FY2025 ended January 3, 2026, and its FY2024 on December 28, 2024."""
    filings = [
        corpus_filing("10-K", 2024, "FY", "2024-12-28"),
        corpus_filing("10-K", 2025, "FY", "2026-01-03"),
        corpus_filing("10-Q", 2026, "Q2", "2026-07-04"),
        corpus_filing("10-Q", 2025, "Q2", "2025-06-28"),
    ]
    result = periods(assign_fiscal_periods(filings))
    assert result["2026-01-03"] == (2025, "FY")
    assert result["2026-07-04"] == (2026, "Q2")
    assert result["2025-06-28"] == (2025, "Q2")


def test_long_final_quarter_still_fits_the_calendar():
    """Costco's 53-week 2023 ended with 17 weeks, so Q3 fell 119 days before year end."""
    filings = [
        corpus_filing("10-K", 2023, "FY", "2023-09-03"),
        corpus_filing("10-Q", 2023, "Q3", "2023-05-07"),
    ]
    assert periods(assign_fiscal_periods(filings))["2023-05-07"] == (2023, "Q3")


def test_one_wrong_10k_tag_does_not_shift_the_labels():
    """Albemarle tags its 2022 10-K as FY2021."""
    filings = [
        corpus_filing("10-K", 2021, "FY", "2022-12-31"),
        corpus_filing("10-K", 2023, "FY", "2023-12-31"),
        corpus_filing("10-K", 2024, "FY", "2024-12-31"),
    ]
    assert periods(assign_fiscal_periods(filings))["2022-12-31"] == (2022, "FY")


def test_10k_tagged_as_a_quarter_is_a_full_year():
    """Costco tagged a 10-K as Q4."""
    filings = [corpus_filing("10-K", 2019, "Q4", "2019-09-01")]
    assert periods(assign_fiscal_periods(filings))["2019-09-01"] == (2019, "FY")


def test_10q_that_does_not_fit_the_calendar_is_left_out():
    """After a change of fiscal year, a date can't be labeled, so it isn't guessed at."""
    filings = [
        corpus_filing("10-K", 2025, "FY", "2025-12-31"),
        corpus_filing("10-Q", 2026, "Q1", "2026-02-15"),
    ]
    assert "2026-02-15" not in periods(assign_fiscal_periods(filings))


def test_without_a_10k_filer_tags_are_kept():
    filings = [corpus_filing("10-Q", 2026, "Q2", "2026-06-30")]
    assert assign_fiscal_periods(filings) == filings


def test_latest_common_quarter_uses_what_all_companies_filed():
    ahead = company_data()
    behind = company_data(rows=[row for row in ROWS if row[4] != "q3-26"])
    by_ticker = {
        "A": periodic_filings(ahead["submissions"], ahead["company_facts"]),
        "B": periodic_filings(behind["submissions"], behind["company_facts"]),
    }
    assert latest_common_quarter(by_ticker) == (2026, "Q2")


def test_latest_common_quarter_never_picks_q1():
    rows = [row for row in ROWS if row[4] in ("q1-26", "k-25", "q1-25", "k-24")]
    data = company_data(rows=rows)
    by_ticker = {"A": periodic_filings(data["submissions"], data["company_facts"])}
    with pytest.raises(CorpusError, match="Q2 or Q3"):
        latest_common_quarter(by_ticker)


def test_select_corpus_picks_five_filings_per_company():
    behind = company_data(rows=[row for row in ROWS if row[4] != "q3-26"])
    corpus = select_corpus({"AAA": company_data(), "BBB": behind})
    for company in corpus.companies:
        assert [f.accession for f in company.filings] == [
            "k-25",
            "k-24",
            "q2-26",
            "q2-25",
            "q1-26",
        ]


def test_comparisons_pair_the_right_filings():
    corpus = select_corpus({"AAA": company_data()})
    yoy_k = corpus.comparison("AAA", ComparisonType.YOY_10K)
    yoy_q = corpus.comparison("AAA", ComparisonType.YOY_10Q)
    seq_q = corpus.comparison("AAA", ComparisonType.SEQ_10Q)
    assert (yoy_k.current.accession, yoy_k.prior.accession) == ("k-25", "k-24")
    assert (yoy_q.current.accession, yoy_q.prior.accession) == ("q3-26", "q3-25")
    assert (seq_q.current.accession, seq_q.prior.accession) == ("q3-26", "q2-26")
    assert yoy_k.comparison_id == "AAA:yoy-10k"


def test_missing_prior_quarter_is_an_error():
    data = company_data(rows=[row for row in ROWS if row[4] not in ("q3-25", "q2-25")])
    with pytest.raises(CorpusError, match="no 10-Q"):
        select_corpus({"AAA": data})


def test_fiscal_year_comes_from_tags_not_calendar():
    rows = [("10-K", 2025, "FY", "2026-01-03", "k-53-week"), *ROWS[4:8]]
    rows = [
        *rows,
        ("10-Q", 2026, "Q2", "2026-07-04", "q2-26"),
        ("10-Q", 2026, "Q1", "2026-04-04", "q1-26"),
    ]
    corpus = select_corpus({"SWK": company_data(rows=rows)})
    annual = next(f for f in corpus.company("SWK").filings if f.accession == "k-53-week")
    assert annual.fiscal_year == 2025
    assert annual.period_end == date(2026, 1, 3)


def test_filing_url_uses_unpadded_cik_and_undashed_accession():
    corpus = select_corpus({"AAA": company_data()})
    company = corpus.company("AAA")
    filing = company.filings[0].model_copy(update={"accession": "0001628280-26-008586"})
    assert company.filing_url(filing) == (
        "https://www.sec.gov/Archives/edgar/data/1/000162828026008586/k-25.htm"
    )


def test_corpus_round_trips_through_yaml(tmp_path):
    corpus = select_corpus({"AAA": company_data()})
    path = tmp_path / "config" / "corpus.yaml"
    corpus.save(path)
    assert Corpus.load(path) == corpus
    filing = corpus.company("AAA").filings[0]
    assert corpus.company("AAA").to_filing(filing).form is FormType.FORM_10K


def test_missing_corpus_file_explains_next_step(tmp_path):
    with pytest.raises(CorpusError, match="discover-corpus"):
        Corpus.load(tmp_path / "corpus.yaml")


@respx.mock
def test_second_download_makes_no_requests(tmp_path):
    corpus = select_corpus({"AAA": company_data()})
    company = corpus.company("AAA")
    routes = [
        respx.get(COMPANY_FACTS_URL.format(cik=company.cik)).mock(
            return_value=httpx.Response(200, content=b"{}")
        )
    ]
    for filing in company.filings:
        routes.append(
            respx.get(company.filing_url(filing)).mock(
                return_value=httpx.Response(200, content=b"<html></html>")
            )
        )
    client = EdgarClient("Name a@b.c", requests_per_second=10)
    assert download_corpus(client, corpus, tmp_path) == (6, 0)
    assert download_corpus(client, corpus, tmp_path) == (0, 6)
    assert all(route.call_count == 1 for route in routes)
