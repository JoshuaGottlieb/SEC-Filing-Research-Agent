"""Check untagged figures against each other: one row and period in two filings should match.

    python eval/ingestion/cross_filing_agreement.py facts.json

Takes the output of `table_accuracy.py --facts-out`. No answer key exists for untagged cells, so
this is an independent signal. A disagreement isn't always an error: companies restate figures,
and two tables can share a label. Each one needs tracing.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import typer

# The three patterns below strip a column label down to what it says besides its period (a
# segment name, "Total"), so that "Three Months Ended June 28, 2026" in one filing and
# "Three Months Ended June 29, 2025" in another reduce to the same text.
PERIOD_WORDS = re.compile(
    r"(?i)\b(three|six|nine|twelve|months?|ended|year|years|fiscal|for|the|quarter|second|first|"
    r"third|fourth|to|date|at|as|of|in|millions|thousands|dollars)\b"
)
MONTH_DAY = re.compile(
    r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{0,2},?"
)
YEAR = re.compile(r"\b(19|20)\d\d\b")
# Enough to read in one sitting. The disagreements are shuffled with a fixed seed below, so the
# same ones print each time.
DISAGREEMENTS_TO_PRINT = 14


def normalize(text: str | None) -> str:
    """Lowercase letters and digits only, with single spaces, so labels compare equal."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def column_dimension(column: str) -> str:
    """What a column says beyond its period, so that segment columns don't collide."""
    return normalize(YEAR.sub("", MONTH_DAY.sub("", PERIOD_WORDS.sub("", column))))


def main(facts_path: Path) -> None:
    rows = json.loads(facts_path.read_text())
    # Group dollar facts by what identifies one figure across filings: table, group, row label,
    # column, period end and duration. Each group maps filing -> the facts found there.
    by_key: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["unit"] == "USD":
            key = (
                row["t"],
                normalize(row["group"]),
                normalize(row["label"]),
                column_dimension(row["col"]),
                row["end"],
                row["dur"],
            )
            by_key[key][row["acc"]].append(row)

    agree = 0
    disagreements = []
    for key, by_filing in by_key.items():
        # Repeats inside one filing can't be matched across filings with any confidence.
        # A key found in only one filing has nothing to compare against.
        if len(by_filing) < 2 or any(len(found) != 1 for found in by_filing.values()):
            continue
        # Equal to the cent in every filing counts as agreement.
        if len({round(found[0]["val"], 2) for found in by_filing.values()}) == 1:
            agree += 1
        else:
            seen = [(found[0]["fp"], found[0]["raw"]) for found in by_filing.values()]
            disagreements.append((key[0], key[2][:44], key[3][:16], key[4], seen))

    total = agree + len(disagreements)
    share = agree / total if total else 0.0
    typer.echo(f"Facts {len(rows)}. Comparable keys {total}: agree {agree} ({share:.1%}).")
    random.Random(2).shuffle(disagreements)
    for disagreement in disagreements[:DISAGREEMENTS_TO_PRINT]:
        typer.echo(f"  {disagreement}")


if __name__ == "__main__":
    typer.run(main)
