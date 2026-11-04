"""Compare derived fiscal periods with filers' tags over each company's full filing history.

    python eval/ingestion/fiscal_check.py
    python eval/ingestion/fiscal_check.py --ticker FLXS --ticker WMT

Saves nothing. Read every difference: the check can't tell which side is right by itself.
The defaults include Walmart, Target, Costco, Microsoft and Deere for unusual fiscal years.
"""

from __future__ import annotations

from typing import Annotated

import typer

from sec_agent.config import get_settings
from sec_agent.edgar.client import COMPANY_FACTS_URL, TICKERS_URL, EdgarClient
from sec_agent.edgar.corpus import fetch_submissions, periodic_filings

# The three analysis companies, the parser test companies, and filers with unusual fiscal
# years (the docstring names the January and June year ends).
DEFAULT_TICKERS = [
    "HSY",
    "MAT",
    "SWK",
    "JPM",
    "GE",
    "CVX",
    "TR",
    "AAPL",
    "AVT",
    "ALB",
    "EYE",
    "CVV",
    "FLXS",
    "WMT",
    "TGT",
    "COST",
    "MSFT",
    "DE",
]


def main(tickers: Annotated[list[str] | None, typer.Option("--ticker")] = None) -> None:
    client = EdgarClient(get_settings().require_sec_user_agent())
    rows = {row["ticker"]: row for row in client.get_json(TICKERS_URL).values()}
    # Count filings whose derived label differs from the filer's own tag.
    total = corrected = 0
    for ticker in tickers or DEFAULT_TICKERS:
        cik = f"{rows[ticker]['cik_str']:010d}"
        filings = periodic_filings(
            fetch_submissions(client, cik), client.get_json(COMPANY_FACTS_URL.format(cik=cik))
        )
        differences = [filing for filing in filings if filing.filer_fiscal_year is not None]
        total += len(filings)
        corrected += len(differences)
        typer.echo(f"{ticker}: {len(filings)} filings, {len(differences)} differ from filer's tag")
        for filing in differences:
            typer.echo(
                f"  {filing.form.value} ended {filing.period_end}: filer said "
                f"{filing.filer_fiscal_period} {filing.filer_fiscal_year}, derived "
                f"{filing.fiscal_period} {filing.fiscal_year}"
            )
    typer.echo(f"Total {total} filings, {corrected} differ.")


if __name__ == "__main__":
    typer.run(main)
