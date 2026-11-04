"""Chooses the filings to analyze and pins them in config/corpus.yaml.

Each company gets 5 filings and 3 comparisons:
  - yoy-10k: latest 10-K vs the previous 10-K
  - yoy-10q: latest common quarter vs the same quarter a year earlier
  - seq-10q: latest common quarter vs the preceding quarter of that year

The quarter must be Q2 or later so a preceding 10-Q exists; there is no Q4 10-Q.
It must also be filed by every company, so all companies cover the same period.

Fiscal periods are derived, because they become retrieval metadata and filers' own tags
can be wrong: Flexsteel tags its March 2026 10-Q as Q3 of fiscal 2025, the same as March
2025. Each company's 10-Ks anchor its fiscal calendar and supply the year's label, which
calendar dates can't: Stanley's FY2025 ended on January 3, 2026, and Walmart and Target
label the same January year end differently. A quarter follows from how far its period end
falls before the fiscal year end. Where that differs from the filer's tag, the tag is kept
on the record so the correction can be seen.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from sec_agent.edgar.client import (
    COMPANY_FACTS_URL,
    SUBMISSIONS_PAGE_URL,
    SUBMISSIONS_URL,
    TICKERS_URL,
    EdgarClient,
)
from sec_agent.schemas import Comparison, ComparisonType, Filing, FiscalPeriod, FormType

# Where a filing's main document lives on EDGAR (cik without leading zeros, accession
# without dashes).
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
# The three analysis companies: Hershey, Mattel, Stanley Black & Decker.
CORPUS_TICKERS = ("HSY", "MAT", "SWK")
# Q1 can't be the "latest" quarter: a sequential comparison needs an earlier 10-Q, and there
# is no 10-Q for Q4 (the 10-K covers it). So only Q2 and Q3 qualify.
PRECEDING_QUARTER = {"Q2": "Q1", "Q3": "Q2"}
# A quarter is about 13 weeks. A quarter is assigned by counting how many of these fit
# between its period end and the fiscal year end.
QUARTER_DAYS = 91
# Quarters aren't equal. Costco's 53-week years end with a 17-week quarter, 28 days long.
QUARTER_TOLERANCE_DAYS = 30
# A 10-Q ending more than this many days before the year end can't belong to that year, so
# the year end is stepped back (320 is a little under the 3 x 91 days a Q1 can be before it).
MAX_DAYS_BEFORE_YEAR_END = 320
# A 52/53-week year can end days into the next calendar year. Stepping back first means
# January 3, 2026 and December 28, 2024 read as the years 2025 and 2024.
WEEK_YEAR_SHIFT_DAYS = 15
# Long enough to reach two 10-Ks and a year-old 10-Q.
HISTORY_DAYS = 3 * 365


class CorpusError(RuntimeError):
    """The required filings could not be found."""


class CorpusFiling(BaseModel):
    """One pinned filing: enough to download it and label it, nothing more."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    form: FormType
    fiscal_year: int
    # "FY" for a 10-K, "Q1" to "Q3" for a 10-Q. Derived from the fiscal calendar, not the tag.
    fiscal_period: str
    period_end: date
    filed_date: date
    accession: str
    # The filing's main HTML file name inside its EDGAR folder.
    primary_document: str
    # What the filer tagged, recorded only when it differs from the derived period.
    filer_fiscal_year: int | None = None
    filer_fiscal_period: str | None = None


class CorpusCompany(BaseModel):
    """A company and its five pinned filings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str
    company: str
    cik: str
    filings: tuple[CorpusFiling, ...]

    def filing_url(self, filing: CorpusFiling) -> str:
        """The EDGAR address of a filing's main document."""
        return ARCHIVE_URL.format(
            cik=int(self.cik),
            accession=filing.accession.replace("-", ""),
            document=filing.primary_document,
        )

    def to_filing(self, filing: CorpusFiling) -> Filing:
        """Convert to the full Filing record that the rest of the system passes around."""
        return Filing(
            ticker=self.ticker,
            company=self.company,
            cik=self.cik,
            form=filing.form,
            period=FiscalPeriod(
                fiscal_year=filing.fiscal_year,
                fiscal_period=filing.fiscal_period,
                period_end=filing.period_end,
            ),
            accession=filing.accession,
            filed_date=filing.filed_date,
            url=self.filing_url(filing),
        )

    def find(self, form: FormType, fiscal_year: int, fiscal_period: str) -> Filing:
        for filing in self.filings:
            if (filing.form, filing.fiscal_year, filing.fiscal_period) == (
                form,
                fiscal_year,
                fiscal_period,
            ):
                return self.to_filing(filing)
        raise CorpusError(f"{self.ticker} has no {form.value} for {fiscal_period} {fiscal_year}.")

    def comparisons(self) -> tuple[Comparison, Comparison, Comparison]:
        """The three comparisons: year over year on 10-Ks and on 10-Qs, and sequential quarters."""
        annuals = sorted(
            (f for f in self.filings if f.form is FormType.FORM_10K),
            key=lambda f: f.period_end,
            reverse=True,
        )
        quarter = max(
            (f for f in self.filings if f.form is FormType.FORM_10Q),
            key=lambda f: f.period_end,
        )
        fy, fp = quarter.fiscal_year, quarter.fiscal_period
        current_q = self.to_filing(quarter)

        def build(kind: ComparisonType, current: Filing, prior: Filing) -> Comparison:
            return Comparison(
                comparison_id=f"{self.ticker}:{kind.value}",
                ticker=self.ticker,
                comparison_type=kind,
                current=current,
                prior=prior,
            )

        return (
            build(
                ComparisonType.YOY_10K,
                self.to_filing(annuals[0]),
                self.to_filing(annuals[1]),
            ),
            build(ComparisonType.YOY_10Q, current_q, self.find(FormType.FORM_10Q, fy - 1, fp)),
            build(
                ComparisonType.SEQ_10Q,
                current_q,
                self.find(FormType.FORM_10Q, fy, PRECEDING_QUARTER[fp]),
            ),
        )


class Corpus(BaseModel):
    """All companies under analysis. Saved to and loaded from config/corpus.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    companies: tuple[CorpusCompany, ...]

    def company(self, ticker: str) -> CorpusCompany:
        for company in self.companies:
            if company.ticker == ticker:
                return company
        raise CorpusError(f"{ticker} is not in the corpus.")

    def comparison(self, ticker: str, kind: ComparisonType) -> Comparison:
        for comparison in self.company(ticker).comparisons():
            if comparison.comparison_type is kind:
                return comparison
        raise CorpusError(f"{ticker} has no {kind.value} comparison.")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # exclude_none keeps the filer's own tags out of the file when they matched.
        data = self.model_dump(mode="json", exclude_none=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Corpus:
        if not path.exists():
            raise CorpusError(f"{path} not found. Run `sec-agent discover-corpus` first.")
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def fiscal_tags_by_accession(company_facts: dict[str, Any]) -> dict[str, tuple[int, str]]:
    """Map each accession number to its filing's fiscal year and period.

    Every fact in the company facts API carries the `fy` and `fp` of the filing
    that reported it. The most common pair per accession guards against a stray
    tag. Amendments often have no facts, so they're absent from the result.
    """
    counts: dict[str, Counter[tuple[int, str]]] = {}
    for taxonomy in company_facts.get("facts", {}).values():
        for concept in taxonomy.values():
            for entries in concept.get("units", {}).values():
                for entry in entries:
                    fy, fp = entry.get("fy"), entry.get("fp")
                    if fy is not None and fp is not None:
                        counts.setdefault(entry["accn"], Counter())[(fy, fp)] += 1
    return {accession: pairs.most_common(1)[0][0] for accession, pairs in counts.items()}


def _year_label_base(year_end: date) -> int:
    """The calendar year a fiscal year-end belongs to, after stepping back WEEK_YEAR_SHIFT_DAYS."""
    return (year_end - timedelta(days=WEEK_YEAR_SHIFT_DAYS)).year


def _shift_years(day: date, years: int) -> date:
    """The same calendar day some years away. February 29 falls back to the 28th."""
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # February 29
        return day.replace(year=day.year + years, day=28)


def _fiscal_year_end_after(period_end: date, annual_ends: list[date]) -> date:
    """The fiscal year end a 10-Q falls in, projecting from 10-Ks not yet filed or fetched."""
    later = [end for end in annual_ends if end >= period_end]
    # Use the first known year end after the quarter; if there is none yet (the quarter is in
    # the current year, whose 10-K isn't filed), project forward from the last known one.
    year_end = later[0] if later else annual_ends[-1]
    while year_end < period_end:
        year_end = _shift_years(year_end, 1)
    # If a known year end is too many years ahead, step back to the nearest plausible one.
    while (year_end - period_end).days > MAX_DAYS_BEFORE_YEAR_END:
        year_end = _shift_years(year_end, -1)
    return year_end


def assign_fiscal_periods(filings: list[CorpusFiling]) -> list[CorpusFiling]:
    """Replace filers' tags with periods derived from the company's fiscal calendar.

    A filing whose date doesn't fit that calendar, as after a change of fiscal year, is
    left out, so a later step fails loudly instead of labeling it wrongly. With no 10-K
    to anchor on, tags are all there is, and selection will reject the company anyway.
    """
    annuals = [filing for filing in filings if filing.form is FormType.FORM_10K]
    if not annuals:
        return filings
    annual_ends = sorted(filing.period_end for filing in annuals)
    # Companies name fiscal years differently (Walmart and Target label the same January year
    # end differently), so learn the company's own convention from its 10-Ks: the usual gap
    # between the calendar year of the year end and the fiscal year it is labelled.
    offsets = Counter(f.fiscal_year - _year_label_base(f.period_end) for f in annuals)
    offset = offsets.most_common(1)[0][0]

    assigned = []
    for filing in filings:
        if filing.form is FormType.FORM_10K:
            year_end, period = filing.period_end, "FY"
        else:
            year_end = _fiscal_year_end_after(filing.period_end, annual_ends)
            days = (year_end - filing.period_end).days
            # Quarters left in the year: 3 left means this is Q1, 2 means Q2, 1 means Q3.
            quarters_left = round(days / QUARTER_DAYS)
            # Drop a filing that doesn't land near a quarter boundary (a fiscal year change).
            if (
                quarters_left not in (1, 2, 3)
                or abs(days - quarters_left * QUARTER_DAYS) > QUARTER_TOLERANCE_DAYS
            ):
                continue
            period = f"Q{4 - quarters_left}"
        year = _year_label_base(year_end) + offset
        # Record the filer's own tags only when they disagree, so corrections stay visible.
        same = (year, period) == (filing.fiscal_year, filing.fiscal_period)
        assigned.append(
            filing.model_copy(
                update={
                    "fiscal_year": year,
                    "fiscal_period": period,
                    "filer_fiscal_year": None if same else filing.fiscal_year,
                    "filer_fiscal_period": None if same else filing.fiscal_period,
                }
            )
        )
    return assigned


def periodic_filings(
    submissions: dict[str, Any], company_facts: dict[str, Any]
) -> list[CorpusFiling]:
    """Original 10-K and 10-Q filings with derived fiscal periods. Amendments are excluded."""
    tags = fiscal_tags_by_accession(company_facts)
    # "recent" plus any older pages that fetch_submissions added.
    pages = [submissions["filings"]["recent"], *submissions["filings"].get("older_pages", ())]
    filings = []
    for page in pages:
        for index, form in enumerate(page["form"]):
            accession = page["accessionNumber"][index]
            # Amendments (10-K/A) and filings with no XBRL facts (so no tags) are skipped.
            if form not in ("10-K", "10-Q") or accession not in tags:
                continue
            fiscal_year, fiscal_period = tags[accession]
            filings.append(
                CorpusFiling(
                    form=FormType(form),
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period,
                    period_end=date.fromisoformat(page["reportDate"][index]),
                    filed_date=date.fromisoformat(page["filingDate"][index]),
                    accession=accession,
                    primary_document=page["primaryDocument"][index],
                )
            )
    return assign_fiscal_periods(filings)


def fetch_submissions(
    client: EdgarClient, cik: str, *, today: date | None = None
) -> dict[str, Any]:
    """Fetch a company's filing history, following pages when `recent` isn't enough.

    `recent` holds only the last 1,000 filings. A bank issuing structured notes
    passes that in months, which pushes its 10-Ks onto older pages.
    """
    submissions = client.get_json(SUBMISSIONS_URL.format(cik=cik))
    # Only fetch older pages that reach back into the history window we need.
    cutoff = (today or date.today()) - timedelta(days=HISTORY_DAYS)
    older = [
        client.get_json(SUBMISSIONS_PAGE_URL.format(name=page["name"]))
        for page in submissions["filings"].get("files", ())
        if date.fromisoformat(page["filingTo"]) >= cutoff
    ]
    filings = {**submissions["filings"], "older_pages": older}
    return {**submissions, "filings": filings}


def latest_common_quarter(by_ticker: dict[str, list[CorpusFiling]]) -> tuple[int, str]:
    """The latest Q2 or Q3 that every company has filed."""
    # Intersect across companies so every company covers the same period.
    common: set[tuple[int, str]] | None = None
    for filings in by_ticker.values():
        quarters = {
            (f.fiscal_year, f.fiscal_period)
            for f in filings
            if f.form is FormType.FORM_10Q and f.fiscal_period in PRECEDING_QUARTER
        }
        common = quarters if common is None else common & quarters
    if not common:
        raise CorpusError("The companies have no Q2 or Q3 10-Q in common.")
    return max(common)


def _select_for_company(
    ticker: str, filings: list[CorpusFiling], fiscal_year: int, fiscal_period: str
) -> tuple[CorpusFiling, ...]:
    annuals = sorted(
        (f for f in filings if f.form is FormType.FORM_10K and f.fiscal_period == "FY"),
        key=lambda f: f.period_end,
        reverse=True,
    )
    # Two 10-Ks are needed for the year-over-year annual comparison.
    if len(annuals) < 2:
        raise CorpusError(f"{ticker} has fewer than two 10-K filings.")

    # Oldest first, so that a re-filed quarter resolves to the latest filing.
    quarters = {
        (f.fiscal_year, f.fiscal_period): f
        for f in sorted(filings, key=lambda f: f.filed_date)
        if f.form is FormType.FORM_10Q
    }
    selected = []
    # The latest quarter, the same quarter a year earlier, and the quarter just before it.
    for key in (
        (fiscal_year, fiscal_period),
        (fiscal_year - 1, fiscal_period),
        (fiscal_year, PRECEDING_QUARTER[fiscal_period]),
    ):
        if key not in quarters:
            raise CorpusError(f"{ticker} has no 10-Q for {key[1]} {key[0]}.")
        selected.append(quarters[key])
    return (annuals[0], annuals[1], *selected)


def select_corpus(companies: dict[str, dict[str, Any]]) -> Corpus:
    """Build the corpus from `{ticker: {company, cik, submissions, company_facts}}`."""
    by_ticker = {
        ticker: periodic_filings(data["submissions"], data["company_facts"])
        for ticker, data in companies.items()
    }
    fiscal_year, fiscal_period = latest_common_quarter(by_ticker)
    return Corpus(
        companies=tuple(
            CorpusCompany(
                ticker=ticker,
                company=companies[ticker]["company"],
                cik=companies[ticker]["cik"],
                filings=_select_for_company(ticker, filings, fiscal_year, fiscal_period),
            )
            for ticker, filings in by_ticker.items()
        )
    )


def discover_corpus(client: EdgarClient, tickers: tuple[str, ...] = CORPUS_TICKERS) -> Corpus:
    """Look up each ticker on EDGAR and choose its filings. Network calls happen here only."""
    rows = {row["ticker"]: row for row in client.get_json(TICKERS_URL).values()}
    companies: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        if ticker not in rows:
            raise CorpusError(f"EDGAR has no ticker {ticker}.")
        # EDGAR's endpoints want the CIK padded to 10 digits.
        cik = f"{rows[ticker]['cik_str']:010d}"
        companies[ticker] = {
            "company": rows[ticker]["title"],
            "cik": cik,
            "submissions": fetch_submissions(client, cik),
            "company_facts": client.get_json(COMPANY_FACTS_URL.format(cik=cik)),
        }
    return select_corpus(companies)


def download_corpus(client: EdgarClient, corpus: Corpus, raw_dir: Path) -> tuple[int, int]:
    """Download filings and XBRL facts. Returns (downloaded, already present)."""
    downloaded = skipped = 0

    def fetch(url: str, destination: Path) -> None:
        nonlocal downloaded, skipped
        if client.download(url, destination):
            downloaded += 1
        else:
            skipped += 1

    for company in corpus.companies:
        fetch(
            COMPANY_FACTS_URL.format(cik=company.cik),
            raw_dir / company.ticker / "companyfacts.json",
        )
        for filing in company.filings:
            fetch(
                company.filing_url(filing),
                raw_dir / company.ticker / filing.accession / filing.primary_document,
            )
    return downloaded, skipped
