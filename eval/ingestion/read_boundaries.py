"""Print each section's first and last paragraph, so its boundaries can be read by hand.

    python eval/ingestion/read_boundaries.py parser_holdout3

Reads saved records, so run `sec-agent ingest` first. This is the check on whether a filing the
gate accepted is actually right. It doesn't look inside sections: `compare_parsers.py` does that.
"""

from __future__ import annotations

from typing import Annotated

import typer
from _common import corpus_filings

from sec_agent.config import get_settings
from sec_agent.ingest.pipeline import parsed_filing_path, read_parsed_filing


def main(corpus: Annotated[list[str], typer.Argument(help="Corpus names or paths.")]) -> None:
    settings = get_settings()
    for company, filing in corpus_filings(corpus):
        record = parsed_filing_path(settings.parsed_filings_dir, company.ticker, filing.accession)
        parsed = read_parsed_filing(record)
        if parsed is None:
            typer.echo(f"== {company.ticker} {filing.accession}: no current record")
            continue
        # A rejected filing is still printed: reading it shows whether the rejection was right.
        verdict = "PASS" if parsed.passed else "REJECT"
        typer.echo(
            f"== {company.ticker} {filing.form.value} {filing.fiscal_period} "
            f"{filing.fiscal_year} {verdict} {parsed.parser.value}/{parsed.layout}"
        )
        for problem in parsed.problems:
            typer.echo(f"   PROBLEM: {problem[:150]}")
        for problem in parsed.primary_problems:
            typer.echo(f"   edgartools: {problem[:150]}")
        # Start and end of each section, where the typical errors are (wrong start, early cut-off).
        for name, section in parsed.sections.items():
            if not section.paragraphs:
                typer.echo(f"  {name.value:16} {section.status.value}")
                continue
            typer.echo(
                f"  {name.value:16} {section.status.value[:7]} {section.char_count:7} "
                f"first: {section.paragraphs[0].text[:90]!r}"
            )
            typer.echo(f"  {'':33}last:  {section.paragraphs[-1].text[:90]!r}")


if __name__ == "__main__":
    typer.run(main)
