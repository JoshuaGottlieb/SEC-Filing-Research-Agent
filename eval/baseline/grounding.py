"""T6.5: are the baseline's reported changes real differences between the two filings? (free)

Run from the repository root, on one or more `baseline.py` output folders:

    uv run --frozen --no-sync python eval/baseline/grounding.py BASELINE_DIR [...] --out out.json

The agent's changes come from the paragraph diff and the metric scan, so they are real by
construction. The baseline's come from one model call, so this checks them against the same code:

- **A text change** is grounded when one of its claims' quotes sits in a passage the paragraph
  diff marks as changed (`align_comparison`, matched by T6.3's `holds`: the quote inside the
  changed text, or the changed text inside the quote).
- **A numeric change** is grounded when its metric has values in both periods
  (`MetricCatalog.compare`) and moved the way the description's first direction word says.

Anything else is listed for a person to read: a metric with no catalog values (a segment, or a
move from a loss to a profit, which has no relative change), a description whose direction words
the pattern misses or misreads, or a text change with no quote in a changed passage. No model
calls; the alignment's embeddings come from the local cache.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from sec_agent.benchmark import Side
from sec_agent.compare.align import align_comparison
from sec_agent.compare.embedding_cache import CachedEmbedder, embedding_cache_path
from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore
from sec_agent.finance.metrics import MetricCatalog, MetricsConfig
from sec_agent.ingest.chunker import EMBEDDING_MODEL
from sec_agent.ingest.index import BgeEmbedder
from sec_agent.schemas import Change, ComparisonType, MetricName

sys.path.insert(0, str(REPO_ROOT / "eval" / "materiality"))
from materiality import holds  # noqa: E402

CORPUS_PATH = REPO_ROOT / "config" / "corpus.yaml"

GROUNDED = "grounded"
NOT_IN_CHANGE = "no quote in a changed passage"
NO_VALUES = "no catalog values"
NO_DIRECTION = "direction not stated"
DISAGREES = "direction disagrees"

# Chosen by hand from the baseline's descriptions. The first match decides, so a description that
# leads with the metric's move ("Gross margin fell ... as costs rose") reads correctly.
_DOWN = (
    r"fell|fall\w*|decreas\w*|declin\w*|dropp?\w*|lower|contract\w*|compress\w*|shr[ai]nk\w*"
    r"|collaps\w*|slipp?\w*|plung\w*|sank|tumbl\w*|down"
)
_UP = r"rose|rise|increas\w*|gr[eo]w\w*|expand\w*|higher|improv\w*|jump\w*|surg\w*|climb\w*|up"
DIRECTION = re.compile(rf"\b(?:(?P<down>{_DOWN})|(?P<up>{_UP}))\b", re.IGNORECASE)


def direction(text: str) -> int:
    """-1 or 1 for the first direction word in `text`, 0 if there is none."""
    match = DIRECTION.search(text)
    if match is None:
        return 0
    return -1 if match.group("down") else 1


def numeric_verdict(description: str, move: float | None) -> str:
    if move is None:
        return NO_VALUES
    said = direction(description)
    if said == 0:
        return NO_DIRECTION
    return GROUNDED if move != 0 and (move > 0) == (said > 0) else DISAGREES


def text_verdict(quotes: Iterable[dict], detected: Sequence[Change]) -> tuple[str, list[str]]:
    """The verdict, and the kinds of the detected changes that hold a quote."""
    kinds = sorted(
        {
            change.kind.value
            for quote in quotes
            for change in detected
            if holds(change, Side(quote["filing"]), quote["text"])
        }
    )
    return (GROUNDED if kinds else NOT_IN_CHANGE), kinds


def check_record(record: dict, detected: Sequence[Change], catalog, comparison) -> list[dict]:
    rows = []
    for n, change in enumerate(record["brief"]["changes"], 1):
        row = {
            "id": f"{record['comparison_id']}:b{n}",
            "kind": change["kind"],
            "description": change["description"],
        }
        if change["kind"] == "numeric":
            compared = None
            if change.get("metric") in set(MetricName):
                compared = catalog.compare(comparison, MetricName(change["metric"]))
            move = compared.move if compared is not None else None
            row |= {"metric": change.get("metric"), "move": move}
            row["verdict"] = numeric_verdict(change["description"], move)
        else:
            quotes = [quote for claim in change["claims"] for quote in claim["quotes"]]
            row["verdict"], row["matched_kinds"] = text_verdict(quotes, detected)
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("baselines", type=Path, nargs="+", help="baseline.py output folders")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    corpus = Corpus.load(CORPUS_PATH)
    embedder = CachedEmbedder(BgeEmbedder(), EMBEDDING_MODEL, embedding_cache_path(settings))
    rows: list[dict] = []
    with FactStore(settings.duckdb_path, read_only=True) as store:
        catalog = MetricCatalog(store, corpus, MetricsConfig.load())
        for folder in args.baselines:
            for path in sorted(folder.glob("*-*.json")):
                record = json.loads(path.read_text())
                if "brief" not in record:
                    continue
                ticker, kind = record["comparison_id"].split(":", 1)
                comparison = corpus.comparison(ticker, ComparisonType(kind))
                detected = align_comparison(
                    comparison, settings.parsed_filings_dir, embedder, store
                ).changes
                rows += check_record(record, detected, catalog, comparison)
    summary = Counter(
        f"{'numeric' if r['kind'] == 'numeric' else 'text'}: {r['verdict']}" for r in rows
    )
    print(f"{len(rows)} changes")
    for key, count in sorted(summary.items()):
        print(f"  {key}: {count}")
    for row in rows:
        if row["verdict"] != GROUNDED:
            print(f"\nto read: {row['id']} ({row['verdict']}): {row['description']}")
    if args.out is not None:
        args.out.write_text(
            json.dumps({"summary": dict(summary), "changes": rows}, indent=2, ensure_ascii=False)
            + "\n"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
