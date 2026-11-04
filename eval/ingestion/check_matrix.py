"""Run every gate check on both parsers' output, and show what each one contributes.

    python eval/ingestion/check_matrix.py corpus parser_validation parser_holdout

Both parsers run on every filing, even when `edgartools` passes, so a check is seen on output
the pipeline would normally skip. A check that's never the only failure overlaps with others, but
it may still catch what they can't, so read these results before removing anything.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Annotated

import typer
from _common import corpus_filings, exact_interval, read_html

from sec_agent.config import get_settings
from sec_agent.ingest.fallback import parse_fallback
from sec_agent.ingest.pipeline import ParserName, _build
from sec_agent.ingest.sections import render_items
from sec_agent.ingest.tables import strip_data_tables

# Each gate check, as a pattern that matches the problem message it writes. The messages come
# from audit.py, sections.py and pipeline.py, so changing a message there means updating it here.
CHECKS = {
    "required section length": r"chars, expected|expected content|: missing",
    "opens with its title": r"does not open with its title",
    "expected terms": r"has none of its expected terms",
    "another part's heading": r"contains another part",
    "sentence coverage": r"sampled sentences",
    "lost table sentence": r"wrongly removed table text",
    "leftover table": r"leftover data table",
    "numeric paragraph": r"numeric paragraph",
    "note numbering": r"note numbers skip|ends at note",
    "section ending": r"left out before",
}


def failed_checks(problems: tuple[str, ...]) -> set[str]:
    """Which named checks a list of problem messages shows. Unmatched messages are kept as
    "other" so a new kind of problem can't hide."""
    found = {
        name for name, pattern in CHECKS.items() if any(re.search(pattern, p) for p in problems)
    }
    unknown = [p for p in problems if not any(re.search(rx, p) for rx in CHECKS.values())]
    return found | {f"other: {problem[:50]}" for problem in unknown}


def route(primary: set[str], fallback: set[str], without: str | None = None) -> str:
    """Where a filing ends up, as the pipeline decides it. `without` pretends one check
    doesn't exist, to see which filings would change route if it were removed."""
    if not primary - {without}:
        return "edgartools"
    return "fallback" if not fallback - {without} else "rejected"


def share(count: int, total: int) -> str:
    """A count as 'x of n (pct, interval)' with an exact 95% interval."""
    low, high = exact_interval(count, total)
    return f"{count} of {total} ({count / total:.0%}, interval {low:.0%} to {high:.0%})"


def main(corpus: Annotated[list[str], typer.Argument(help="Corpus names or paths.")]) -> None:
    settings = get_settings()
    rows = []
    for company, filing in corpus_filings(corpus):
        html = read_html(settings, company, filing)
        strip = strip_data_tables(html)
        # Both parsers always run (unlike the real pipeline), so each check sees both outputs.
        label = f"{company.ticker} {filing.form.value} {filing.fiscal_period} {filing.fiscal_year}"
        try:
            primary = failed_checks(
                _build(
                    html,
                    strip,
                    filing.form,
                    filing.accession,
                    ParserName.EDGARTOOLS,
                    "edgartools",
                    render_items(strip.html, filing.form),
                ).problems
            )
        # edgartools is third-party code, so a crash counts as a failure, as in the pipeline.
        except Exception as error:  # noqa: BLE001
            primary = {f"edgartools raised {type(error).__name__}"}
        result = parse_fallback(strip.html, filing.form)
        fallback = failed_checks(
            _build(
                html,
                strip,
                filing.form,
                filing.accession,
                ParserName.FALLBACK,
                result.layout,
                result.rendered,
            ).problems
        )
        rows.append((label, filing.form.value, primary, fallback))

    # Per check: how often it fails on each parser, how often it is the only failure, and how many
    # filings would change route if it were removed. That last column decides whether a check
    # is redundant.
    typer.echo(
        "check | fails on edgartools | on fallback | only failure | filings rerouted if removed"
    )
    for name in CHECKS:
        rerouted = [r[0] for r in rows if route(r[2], r[3]) != route(r[2], r[3], name)]
        only = sum(r[2] == {name} for r in rows) + sum(r[3] == {name} for r in rows)
        typer.echo(
            f"{name:30} {sum(name in r[2] for r in rows):3} {sum(name in r[3] for r in rows):3} "
            f"{only:3} {len(rerouted):3}  {rerouted[:4]}"
        )

    total = len(rows)
    routes = Counter(route(r[2], r[3]) for r in rows)
    called = routes["fallback"] + routes["rejected"]
    typer.echo(f"Fallback called: {share(called, total)}")
    typer.echo(f"Fallback succeeded: {share(routes['fallback'], total)}")
    typer.echo(f"Both parsers failed: {share(routes['rejected'], total)}")
    for form in ("10-K", "10-Q"):
        of_form = [r for r in rows if r[1] == form]
        if of_form:
            count = sum(route(r[2], r[3]) != "edgartools" for r in of_form)
            typer.echo(f"Fallback called, {form}: {share(count, len(of_form))}")


if __name__ == "__main__":
    typer.run(main)
