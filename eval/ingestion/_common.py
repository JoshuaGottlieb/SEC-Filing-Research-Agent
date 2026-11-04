"""Helpers shared by the ingestion measurements."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from scipy.stats import beta

from sec_agent.config import REPO_ROOT, Settings
from sec_agent.edgar.corpus import Corpus, CorpusCompany, CorpusFiling
from sec_agent.ingest.table_facts import FilingContext


# Every script takes corpus names the same way, and walks the same company/filing pairs.
def resolve_corpus(name: str) -> Path:
    """Accept a path or a bare name from config/, such as `parser_holdout3`."""
    path = Path(name)
    return path if path.exists() else REPO_ROOT / "config" / f"{name.removesuffix('.yaml')}.yaml"


def corpus_filings(names: list[str]) -> Iterator[tuple[CorpusCompany, CorpusFiling]]:
    """Every (company, filing) pair in the named corpus files, in file order."""
    for name in names:
        for company in Corpus.load(resolve_corpus(name)).companies:
            for filing in company.filings:
                yield company, filing


def read_html(settings: Settings, company: CorpusCompany, filing: CorpusFiling) -> str:
    """A downloaded filing's HTML. Run `sec-agent ingest --corpus ...` first to download it."""
    path = settings.raw_filings_dir / company.ticker / filing.accession / filing.primary_document
    return path.read_text(encoding="utf-8", errors="replace")


def filing_context(filing: CorpusFiling) -> FilingContext:
    """What the table parser needs to know about a filing."""
    return FilingContext(
        filing.accession, filing.form, filing.fiscal_year, filing.fiscal_period, filing.period_end
    )


def exact_interval(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson interval, which stays valid for small samples and for 0 or n."""
    # Each tail holds half of the leftover probability. At 0 or n successes one side is open.
    tail = (1 - level) / 2
    low = 0.0 if successes == 0 else beta.ppf(tail, successes, trials - successes + 1)
    high = 1.0 if successes == trials else beta.ppf(1 - tail, successes + 1, trials - successes)
    return float(low), float(high)
