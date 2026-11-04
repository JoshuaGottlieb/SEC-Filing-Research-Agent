"""Query the fact store for known failure shapes, and print what's searchable per company.

    python eval/ingestion/store_checks.py

- Bare-year columns ("2025") whose end date isn't a fiscal year end the filing reports.
- Percent columns stored as dollars.
Both counts should be 0.
"""

from __future__ import annotations

import duckdb
import typer

from sec_agent.config import get_settings

# What the Financial Agent can search for each company, and what is waiting as text.
PER_COMPANY = """
    SELECT t.ticker,
           (SELECT count(*) FROM xbrl_facts x WHERE x.ticker = t.ticker) AS xbrl,
           count(*) AS table_cells,
           count(xbrl_fact_id) AS copies_of_xbrl,
           sum(ambiguous::int) AS ambiguous,
           sum((xbrl_fact_id IS NULL AND NOT ambiguous)::int) AS searchable,
           (SELECT count(*) FROM unparsed_tables u WHERE u.ticker = t.ticker) AS queued_tables
    FROM table_facts t GROUP BY 1 ORDER BY 1
"""
# A bare-year column ("2025") is dated from the filing's year end. This counts those facts
# whose date matches none of the full-year periods the filing's own XBRL reports. Should be 0.
MISDATED_BARE_YEARS = """
    WITH year_ends AS (
        SELECT DISTINCT accession, end_date FROM xbrl_facts
        WHERE duration = '12m' AND dimensions IS NULL
    ), bare AS (
        SELECT * FROM table_facts
        WHERE form = '10-K' AND xbrl_fact_id IS NULL AND NOT ambiguous
          AND regexp_matches(column_label, '^(Fiscal |FY ?)?(19|20)[0-9][0-9]$')
    )
    SELECT count(*), coalesce(sum((y.end_date IS NULL)::int), 0)
    FROM bare b LEFT JOIN year_ends y ON b.accession = y.accession AND b.end_date = y.end_date
"""
# A column named "percent" or "% of" must never hold a dollar amount. Should be none.
PERCENT_AS_DOLLARS = """
    SELECT ticker, count(*) FROM facts
    WHERE source = 'table' AND unit = 'USD'
      AND regexp_matches(column_label, '(?i)percent|\\bpct\\b|% of')
    GROUP BY 1 ORDER BY 2 DESC
"""
# Why tables wait as text. The part after the colon is trimmed so similar reasons group together.
QUEUE_REASONS = """
    SELECT regexp_replace(reasons, ':.*', '') AS reason, count(*) AS n
    FROM unparsed_tables GROUP BY 1 ORDER BY 2 DESC LIMIT 10
"""


def main() -> None:
    # Read-only: this script reports and never changes the store.
    with duckdb.connect(get_settings().duckdb_path, read_only=True) as db:
        typer.echo("ticker | XBRL facts | table cells | copies | ambiguous | searchable | queued")
        rows = db.execute(PER_COMPANY).fetchall()
        for row in rows:
            typer.echo(" | ".join(str(value) for value in row))
        typer.echo(
            f"Total searchable {sum(row[5] for row in rows)}, "
            f"queued as text {db.execute('SELECT count(*) FROM unparsed_tables').fetchone()[0]}"
        )
        total, misdated = db.execute(MISDATED_BARE_YEARS).fetchone()
        typer.echo(f"Bare-year facts {total}, misdated {misdated}")
        typer.echo(
            f"Percent columns stored as dollars: {db.execute(PERCENT_AS_DOLLARS).fetchall()}"
        )
        typer.echo(f"Top reasons for queuing: {db.execute(QUEUE_REASONS).fetchall()}")


if __name__ == "__main__":
    typer.run(main)
