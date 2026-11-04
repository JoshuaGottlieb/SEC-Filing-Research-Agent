"""Run both parsers on each filing and compare their sections sentence by sentence.

    python eval/ingestion/compare_parsers.py parser_holdout3

The parsers find boundaries differently, so where they agree, a shared error is unlikely. Where
they disagree, the printed sentences show which one to read. Unlike `read_boundaries.py`, this
can notice a problem in the middle of a section.
"""

from __future__ import annotations

from typing import Annotated

import typer
from _common import corpus_filings, read_html

from sec_agent.config import get_settings
from sec_agent.ingest.audit import _fingerprint, _sentences
from sec_agent.ingest.fallback import parse_fallback
from sec_agent.ingest.pipeline import parsed_filing_path, read_parsed_filing
from sec_agent.ingest.sections import render_items, sections_from_items
from sec_agent.ingest.tables import strip_data_tables


def share_found(sentences: list[str], other_fingerprint: str) -> float:
    """The share of these sentences that also appear in the other parser's text."""
    found = sum(_fingerprint(sentence) in other_fingerprint for sentence in sentences)
    return found / max(len(sentences), 1)


def main(
    corpus: Annotated[list[str], typer.Argument(help="Corpus names or paths.")],
    # Sections where both parsers agree on at least this share of sentences are not printed. A
    # little under 100% allows for small boundary differences; chosen by hand.
    agreement: Annotated[float, typer.Option(help="Report sections below this share.")] = 0.97,
) -> None:
    settings = get_settings()
    compared = different = 0
    for company, filing in corpus_filings(corpus):
        stripped = strip_data_tables(read_html(settings, company, filing)).html
        # Run both parsers on the same stripped HTML, whatever the pipeline chose.
        try:
            primary = sections_from_items(
                render_items(stripped, filing.form), filing.form, filing.accession
            )
        except Exception as error:  # noqa: BLE001
            typer.echo(f"{company.ticker} {filing.accession}: edgartools raised {error!r}")
            continue
        fallback = sections_from_items(
            parse_fallback(stripped, filing.form).rendered, filing.form, filing.accession
        )
        record = parsed_filing_path(settings.parsed_filings_dir, company.ticker, filing.accession)
        saved = read_parsed_filing(record)
        # Which parser the pipeline actually kept, shown beside each disagreement.
        used = saved.parser.value if saved else "no record"
        for section in primary:
            first = " ".join(p.text for p in primary[section].paragraphs)
            second = " ".join(p.text for p in fallback[section].paragraphs)
            # Complete sentences only (15+ words), the same unit the gate's coverage check uses.
            first_sentences, second_sentences = _sentences(first), _sentences(second)
            if not first_sentences and not second_sentences:
                continue
            compared += 1
            first_print, second_print = _fingerprint(first), _fingerprint(second)
            first_share = share_found(first_sentences, second_print)
            second_share = share_found(second_sentences, first_print)
            if first_share >= agreement and second_share >= agreement:
                continue
            different += 1
            # One example sentence each side, to show which parser to read.
            only_first = [s for s in first_sentences if _fingerprint(s) not in second_print][:1]
            only_second = [s for s in second_sentences if _fingerprint(s) not in first_print][:1]
            typer.echo(
                f"{company.ticker} {filing.form.value} {filing.fiscal_period} "
                f"{filing.fiscal_year} {section.value} used={used} "
                f"edgartools-in-fallback={first_share:.2f} "
                f"fallback-in-edgartools={second_share:.2f} "
                f"sentences={len(first_sentences)}/{len(second_sentences)}"
            )
            typer.echo(f"    only edgartools: {[s[:100] for s in only_first]}")
            typer.echo(f"    only fallback:   {[s[:100] for s in only_second]}")
    typer.echo(f"Sections compared: {compared}. Below {agreement:.0%} agreement: {different}.")


if __name__ == "__main__":
    typer.run(main)
