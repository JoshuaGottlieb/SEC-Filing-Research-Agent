"""Draw unseen companies at random from EDGAR's ticker list and pin their filings.

    python eval/ingestion/draw_companies.py --seed 20261105 --out config/parser_holdout3.yaml

Excludes every company in a config file, plus those looked at without being pinned. Prints each
draw it skips and why, so the record shows that nothing was chosen by hand.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Annotated

import httpx
import typer

from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.client import COMPANY_FACTS_URL, TICKERS_URL, EdgarClient
from sec_agent.edgar.corpus import Corpus, CorpusError, fetch_submissions, select_corpus

# Looked at during development or skipped by an earlier draw, but not in any config file.
# Excluding these keeps every later draw fresh: a company already read can't be "unseen".
SEEN_WITHOUT_PINNING = frozenset(["XOM", "FINV", "ATLQ", "WMT", "TGT", "COST", "MSFT", "DE"])


def used_tickers() -> set[str]:
    """Every company already used: the analysis corpus, all parser test files, and the above."""
    used = set(SEEN_WITHOUT_PINNING)
    for path in (REPO_ROOT / "config").glob("*.yaml"):
        if path.name == "corpus.yaml" or path.name.startswith("parser_"):
            used.update(company.ticker for company in Corpus.load(path).companies)
    return used


def main(
    # The seed is chosen before the draw and recorded, so nobody can steer the result.
    seed: Annotated[int, typer.Option(help="Pick a new seed for each test, before the draw.")],
    out: Annotated[Path, typer.Option(help="Corpus file to write.")],
    count: int = 5,
) -> None:
    client = EdgarClient(get_settings().require_sec_user_agent())
    rows = list(client.get_json(TICKERS_URL).values())
    used = used_tickers()
    used_ciks = {row["cik_str"] for row in rows if row["ticker"] in used}
    # A company can list several tickers, and only its first counts.
    unique = {row["cik_str"]: row for row in reversed(rows) if row["cik_str"] not in used_ciks}
    # Sorted first so the same seed always shuffles to the same order.
    candidates = sorted(unique.values(), key=lambda row: row["cik_str"])
    typer.echo(f"{len(rows)} tickers, {len(candidates)} unused companies, seed {seed}")
    random.Random(seed).shuffle(candidates)

    # Walk the shuffled list and keep the first companies with usable filings. Each skip is
    # printed with its reason.
    picked = []
    for row in candidates:
        ticker, title, cik = row["ticker"], row["title"], f"{row['cik_str']:010d}"
        try:
            submissions = fetch_submissions(client, cik)
            # Funds and shells file neither form.
            forms = submissions["filings"]["recent"]["form"]
            if "10-K" not in forms and "10-Q" not in forms:
                typer.echo(f"skip {ticker} ({title}): files no 10-K or 10-Q")
                continue
            facts = client.get_json(COMPANY_FACTS_URL.format(cik=cik))
            selected = select_corpus(
                {
                    ticker: {
                        "company": title,
                        "cik": cik,
                        "submissions": submissions,
                        "company_facts": facts,
                    }
                }
            )
        # Too few filings in the right shape (see select_corpus), for example a new company.
        except CorpusError as error:
            typer.echo(f"skip {ticker} ({title}): not eligible: {error}")
            continue
        except httpx.HTTPStatusError as error:
            typer.echo(f"skip {ticker} ({title}): HTTP {error.response.status_code}")
            continue
        picked.append(selected.companies[0])
        typer.echo(f"SELECTED {ticker} ({title})")
        if len(picked) == count:
            break

    Corpus(companies=tuple(picked)).save(out)
    typer.echo(f"Wrote {out}")
    for company in picked:
        typer.echo(f"{company.ticker} {company.company} CIK {company.cik}")
        for filing in company.filings:
            typer.echo(
                f"  {filing.form.value} {filing.fiscal_period} {filing.fiscal_year} "
                f"ended {filing.period_end} {filing.accession} {filing.primary_document}"
            )


if __name__ == "__main__":
    typer.run(main)
