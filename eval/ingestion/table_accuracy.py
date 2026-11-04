"""Score table parsing against XBRL tags, which the parser doesn't read while parsing.

    python eval/ingestion/table_accuracy.py parser_holdout3 --facts-out facts.json

Reports three things:
- Tagged cells: an answer key for value and period end. It overstates accuracy on untagged tables.
- Untagged cells whose distinctive value is in XBRL for another period only: likely misdated.
- The share of tables that parse cleanly.

`--facts-out` saves the untagged facts for `cross_filing_agreement.py`.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from _common import corpus_filings, filing_context, read_html

from sec_agent.config import get_settings
from sec_agent.edgar.xbrl import read_contexts, read_inline_fact
from sec_agent.ingest.table_facts import disagrees_with_tag, parse_tables
from sec_agent.ingest.tables import parse_html_tree

# A value needs at least 4 significant digits to count as distinctive: small numbers repeat by
# coincidence across periods. This matches the store's duplicate rule.
DISTINCTIVE_DIGITS = 4
# Two period ends within 10 days are the same period (52/53-week years). This matches the
# parser's own tolerance.
SAME_PERIOD_DAYS = 10
# Enough to read; the errors are shuffled with a fixed seed below, so the same ones print each run.
ERRORS_TO_PRINT = 30


def xbrl_ends_by_value(html: str) -> dict[float, set[str]]:
    """Each tagged amount in the filing, mapped to the period ends it is reported for."""
    root = parse_html_tree(html)
    contexts = read_contexts(root)
    ends: dict[float, set[str]] = defaultdict(set)
    for element in root.iter("ix:nonfraction"):
        fact = read_inline_fact(element)
        if fact and fact.context_id in contexts:
            ends[round(abs(fact.value), 4)].add(str(contexts[fact.context_id].end_date))
    return ends


def main(
    corpus: Annotated[list[str], typer.Argument(help="Corpus names or paths.")],
    facts_out: Annotated[Path | None, typer.Option(help="Save untagged facts as JSON.")] = None,
) -> None:
    settings = get_settings()
    totals: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    errors: list[tuple] = []
    untagged: list[dict] = []
    for company, filing in corpus_filings(corpus):
        html = read_html(settings, company, filing)
        # verify=False keeps tables that disagree with their tags, so they can be counted.
        parsed = parse_tables(html, filing_context(filing), verify=False)
        ends_by_value = xbrl_ends_by_value(html)
        for table in parsed:
            totals["tables"] += 1
            totals["clean"] += table.clean
            if not table.clean:
                reasons.update(re.sub(r":.*", "", reason) for reason in table.reasons)
            for fact in table.facts:
                where = (company.ticker, filing.form.value, filing.fiscal_period)
                label = (fact.row_label[:30], fact.column_label[:30], fact.raw_text)
                # Tagged cell: XBRL is the answer key. A "period" problem means the value matched
                # but the date didn't, and a "value" problem the reverse, so each is counted
                # separately.
                if fact.tagged and fact.xbrl_end_date is not None:
                    problem = disagrees_with_tag(fact)
                    totals["checked"] += 1
                    totals["value_ok"] += problem is None or problem.startswith("period")
                    totals["end_ok"] += problem is None or problem.startswith("value")
                    if problem:
                        errors.append((*where, table.table_index, *label, problem[:70]))
                    continue
                if fact.tagged:
                    continue
                totals["untagged"] += 1
                untagged.append(
                    {
                        "t": company.ticker,
                        "acc": filing.accession,
                        "fp": filing.fiscal_period,
                        "group": fact.row_group,
                        "label": fact.row_label,
                        "col": fact.column_label,
                        "end": str(fact.end_date),
                        "dur": fact.duration.value if fact.duration else None,
                        "raw": fact.raw_text,
                        "val": fact.value,
                        "unit": fact.unit,
                    }
                )
                # Untagged cell: if its distinctive amount exists in XBRL, but only for another
                # period, the cell was probably misdated.
                digits = re.sub(r"\D", "", fact.raw_text).strip("0")
                if fact.unit != "USD" or len(digits) < DISTINCTIVE_DIGITS:
                    continue
                ends = ends_by_value.get(round(abs(fact.value), 4))
                if not ends:
                    continue
                totals["untagged_in_xbrl"] += 1
                near = any(
                    abs((date.fromisoformat(end) - fact.end_date).days) <= SAME_PERIOD_DAYS
                    for end in ends
                )
                if not near:
                    totals["untagged_other_period"] += 1
                    errors.append(
                        ("PERIOD CONFLICT", *where, *label, str(fact.end_date), sorted(ends)[:3])
                    )

    checked = totals["checked"] or 1
    typer.echo(
        f"Tagged cells {totals['checked']}: value {totals['value_ok'] / checked:.3%}, "
        f"period end {totals['end_ok'] / checked:.3%}"
    )
    tables = totals["tables"] or 1
    typer.echo(
        f"Clean tables {totals['clean']} of {totals['tables']}: {totals['clean'] / tables:.1%}"
    )
    typer.echo(f"Why tables weren't clean: {reasons.most_common(10)}")
    typer.echo(
        f"Untagged facts {totals['untagged']}. Distinctive values also in XBRL "
        f"{totals['untagged_in_xbrl']}, in another period only {totals['untagged_other_period']}"
    )
    typer.echo(f"Errors {len(errors)}, showing up to {ERRORS_TO_PRINT}:")
    random.Random(9).shuffle(errors)
    for error in errors[:ERRORS_TO_PRINT]:
        typer.echo(f"  {error}")
    if facts_out:
        facts_out.write_text(json.dumps(untagged))
        typer.echo(f"Wrote {len(untagged)} untagged facts to {facts_out}")


if __name__ == "__main__":
    typer.run(main)
