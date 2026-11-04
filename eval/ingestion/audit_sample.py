"""Sample searchable untagged facts, with the raw table rows beside each, for a hand audit.

    python eval/ingestion/audit_sample.py parser_holdout3 --seed 20261105 --per-company 12

Only hand audits test the actual target directly: figures no tag can confirm. Read each fact
against its rows, and check the value, the unit, the period, and that the row and column match.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Annotated

import duckdb
import typer
from _common import corpus_filings, filing_context, read_html

from sec_agent.config import get_settings
from sec_agent.ingest.table_facts import parse_tables

# The facts an agent could actually retrieve: untagged, not hidden as a duplicate or ambiguous,
# and with at least one digit in the displayed text.
SEARCHABLE_WITH_DIGITS = """
    SELECT ticker, accession, form, fiscal_period, table_index, fact_id, heading, table_title,
           row_label, column_label, raw_text, value, unit, end_date, duration
    FROM table_facts
    WHERE xbrl_fact_id IS NULL AND NOT ambiguous AND regexp_matches(raw_text, '[0-9]')
"""


def main(
    corpus: Annotated[list[str], typer.Argument(help="Corpus names or paths.")],
    seed: Annotated[int, typer.Option(help="Sampling seed.")],
    per_company: int = 12,
) -> None:
    settings = get_settings()
    # Only facts from the named corpus are sampled, since the database holds every corpus.
    filings = {filing.accession: (company, filing) for company, filing in corpus_filings(corpus)}
    with duckdb.connect(settings.duckdb_path, read_only=True) as db:
        rows = db.execute(SEARCHABLE_WITH_DIGITS).fetchall()
    by_company: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        if row[1] in filings:
            by_company[row[0]].append(row)
    typer.echo({ticker: len(found) for ticker, found in sorted(by_company.items())})

    # The same seed gives the same sample, so the audit can be repeated.
    sampler = random.Random(seed)
    tables_by_filing: dict[str, dict] = {}
    number = 0
    for ticker in sorted(by_company):
        found = by_company[ticker]
        for row in sampler.sample(found, min(per_company, len(found))):
            _, accession, form, period, index, fact_id, heading, title = row[:8]
            label, column, raw, value, unit, end_date, duration = row[8:]
            if accession not in tables_by_filing:
                company, filing = filings[accession]
                parsed = parse_tables(read_html(settings, company, filing), filing_context(filing))
                tables_by_filing[accession] = {table.table_index: table for table in parsed}
            # Re-parse the table to show its raw rows beside the fact it produced.
            table = tables_by_filing[accession][index]
            row_index = int(fact_id.split(":r")[1].split(":")[0])
            number += 1
            typer.echo(
                f"#{number} {ticker} {form} {period} table {index} "
                f"heading={str(heading)[:40]!r} title={str(title)[:40]!r}"
            )
            typer.echo(
                f"   FACT {label[:48]} | {column[:48]} | {raw} => {value:,.2f} {unit} "
                f"end {end_date} {duration}"
            )
            # Up to three rows above the fact: usually the captions and column headers.
            for header in table.rows[: min(row_index, 3)]:
                typer.echo(f"     head: {' | '.join(header)[:120]}")
            typer.echo(f"     row:  {' | '.join(table.rows[row_index])[:120]}")


if __name__ == "__main__":
    typer.run(main)
