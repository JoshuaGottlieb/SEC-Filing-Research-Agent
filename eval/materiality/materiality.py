"""T6.3: the materiality step against the labels: missed changes and false positives.

Run from the repository root (paid: Opus materiality batches, as in `sec-agent analyze`):

    uv run --frozen --no-sync python eval/materiality/materiality.py \
        eval/benchmark/test.json out_dir --cap 8.00
    uv run --frozen --no-sync python eval/materiality/materiality.py \
        eval/benchmark/test.json out_dir --dry-run

For each comparison of the label file's companies, it runs the pipeline's own detection and
materiality code, wired as `graph.run_analysis` wires it: `align_comparison` for text changes,
`scan_comparison` for numeric ones, then `rank_changes`. Nothing after materiality runs. It then
scores the decisions against the labels, with the method used to tune Hershey on 2026-11-12:

- **Missed changes.** A labeled material change is found when an aligned change that holds one
  of its quotes, from the same filing, was called material. A label whose quotes no aligned
  change holds is reported apart (`not_aligned`): that is a detection miss, not a materiality
  one. A quote longer than the changed paragraph is matched the other way round (the paragraph
  inside the quote), and the match records which way it went.
- **False positives.** Sampled changes are matched to the run by `change_id`. A sampled change
  labeled not material that the step called material is a false positive. The sample's text is
  compared with the run's, so drift in alignment since labeling shows up (`text_changed`).

Material means what the pipeline decides: a text change from `MIN_MATERIAL_IMPORTANCE` (3) up.
The scores at importance 1 and 2 are reported beside it, as the Hershey tuning did. Numeric
changes are always material and have no labels here.

Model answers go to the pipeline's own replay cache (`Settings.llm_cache_path`), not a separate
one: the requests are exactly the ones `sec-agent analyze` sends, so a comparison already ranked
by a pipeline run replays at no cost, and a later pipeline run on these filings replays this
script's answers. A rerun of this script replays too, so scoring again costs nothing. A dry run
aligns and scans every comparison (local, a few minutes) and prints each one's batches, how many
the cache answers, and the expected and worst-case cost; it sends nothing and writes nothing.

Thresholds were tuned on Hershey only. The test file (Mattel, Stanley) is run once, after the
materiality code is final, and nothing is tuned on what it shows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from sec_agent.agents.verification import _appears_in, _normalize_quote
from sec_agent.benchmark import Benchmark, LabeledChange, SampledChange, Side, load_benchmark
from sec_agent.compare.align import align_comparison
from sec_agent.compare.embedding_cache import CachedEmbedder, embedding_cache_path
from sec_agent.compare.materiality import (
    MIN_MATERIAL_IMPORTANCE,
    MaterialityDecision,
    MaterialityResult,
    materiality_headroom,
    rank_changes,
)
from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore
from sec_agent.finance.metrics import MetricCatalog, MetricsConfig
from sec_agent.finance.scan import scan_comparison
from sec_agent.graph import MATERIALITY_BATCH_USD
from sec_agent.ingest.chunker import EMBEDDING_MODEL
from sec_agent.ingest.index import BgeEmbedder
from sec_agent.llm.cache import ReplayCache
from sec_agent.llm.client import LLMClient
from sec_agent.llm.ledger import CostLedger
from sec_agent.schemas import Change, ChangeKind, ComparisonType

CORPUS_PATH = REPO_ROOT / "config" / "corpus.yaml"
# Importance cutoffs reported beside the pipeline's own, as in the Hershey tuning.
CUTOFFS = (1, 2, MIN_MATERIAL_IMPORTANCE)


# --- Matching --------------------------------------------------------------------------------


def holds(change: Change, side: Side, quote: str) -> str | None:
    """How a change's text on `side` holds a label quote: "quote_in_change", "change_in_quote",
    or None. Matched with the citation checker's normalization and boundary rule."""
    text = change.current_text if side is Side.CURRENT else change.prior_text
    if not text:
        return None
    needle, haystack = _normalize_quote(quote), _normalize_quote(text)
    if _appears_in(needle, haystack):
        return "quote_in_change"
    if _appears_in(haystack, needle):
        return "change_in_quote"
    return None


def matching_changes(label: LabeledChange, changes: Iterable[Change]) -> list[tuple[Change, str]]:
    """The comparison's text changes that hold one of the label's quotes, with how."""
    found = []
    for change in changes:
        if change.comparison_id != label.comparison_id or change.kind is ChangeKind.NUMERIC:
            continue
        for quote in label.quotes:
            how = holds(change, quote.side, quote.text)
            if how:
                found.append((change, how))
                break
    return found


# --- Scoring ---------------------------------------------------------------------------------


def score_label(
    label: LabeledChange, changes: Sequence[Change], decisions: Mapping[str, MaterialityDecision]
) -> dict:
    """One labeled material change: found, missed, not aligned, or never decided."""
    matched = matching_changes(label, changes)
    decided = [(change, how) for change, how in matched if change.change_id in decisions]
    importances = [decisions[change.change_id].importance for change, _ in decided]
    if not matched:
        status = "not_aligned"
    elif not decided:
        status = "undecided"
    elif any(decisions[change.change_id].material for change, _ in decided):
        status = "found"
    else:
        status = "missed"
    return {
        "item_id": label.item_id,
        "comparison_id": label.comparison_id,
        "kind": label.kind.value,
        "criterion": label.criterion.value,
        "summary": label.summary,
        "status": status,
        "best_importance": max(importances, default=None),
        "matches": [
            {
                "change_id": change.change_id,
                "how": how,
                "importance": decisions[change.change_id].importance
                if change.change_id in decisions
                else None,
                "material": decisions[change.change_id].material
                if change.change_id in decisions
                else None,
            }
            for change, how in matched
        ],
    }


def score_sample(
    sample: SampledChange, by_id: Mapping[str, Change], decisions: Mapping[str, MaterialityDecision]
) -> dict:
    """One sampled change: what the step decided, and whether its text drifted since labeling."""
    change = by_id.get(sample.change_id)
    decision = decisions.get(sample.change_id)
    if change is None:
        status = "not_in_run"
    elif decision is None:
        status = "undecided"
    elif (change.current_text or None, change.prior_text or None) != (
        sample.current_text or None,
        sample.prior_text or None,
    ):
        status = "text_changed"
    else:
        status = "decided"
    return {
        "item_id": sample.item_id,
        "comparison_id": sample.comparison_id,
        "change_id": sample.change_id,
        "labeled_material": sample.material,
        "status": status,
        "importance": decision.importance if decision else None,
        "called_material": decision.material if decision else None,
        "rationale": decision.rationale if decision else None,
    }


def _ratio(hits: int, total: int) -> str:
    return f"{hits}/{total}"


def summarize(labels: list[dict], samples: list[dict]) -> dict:
    """Headline counts at the pipeline's cutoff, and at each importance cutoff."""
    usable = [s for s in samples if s["status"] == "decided" and s["labeled_material"] is not None]
    negatives = [s for s in usable if not s["labeled_material"]]
    positives = [s for s in usable if s["labeled_material"]]
    statuses: dict[str, int] = {}
    for label in labels:
        statuses[label["status"]] = statuses.get(label["status"], 0) + 1
    by_cutoff = {}
    for cutoff in CUTOFFS:
        found = sum(1 for label in labels if (label["best_importance"] or 0) >= cutoff)
        false_pos = sum(1 for s in negatives if (s["importance"] or 0) >= cutoff)
        sample_hits = sum(1 for s in positives if (s["importance"] or 0) >= cutoff)
        by_cutoff[str(cutoff)] = {
            "labeled_found": _ratio(found, len(labels)),
            "sampled_false_positives": _ratio(false_pos, len(negatives)),
            "sampled_material_found": _ratio(sample_hits, len(positives)),
        }
    return {
        "labeled_found": _ratio(statuses.get("found", 0), len(labels)),
        "labeled_by_status": statuses,
        "sampled_false_positives": _ratio(
            sum(1 for s in negatives if s["called_material"]), len(negatives)
        ),
        "sampled_material_found": _ratio(
            sum(1 for s in positives if s["called_material"]), len(positives)
        ),
        "samples_not_scored": {s["item_id"]: s["status"] for s in samples if s not in usable},
        "by_importance_cutoff": by_cutoff,
        "pipeline_cutoff": MIN_MATERIAL_IMPORTANCE,
    }


def score(benchmark: Benchmark, records: Sequence[dict]) -> dict:
    """Score saved comparison records (see `record_for`) against the label file."""
    changes = [Change.model_validate(c) for record in records for c in record["text_changes"]]
    decisions = {
        d["change_id"]: MaterialityDecision.model_validate(d)
        for record in records
        for d in record["result"]["decisions"]
    }
    run_comparisons = {record["comparison_id"] for record in records}
    labels = [
        score_label(label, changes, decisions)
        for label in benchmark.changes
        if label.comparison_id in run_comparisons
    ]
    by_id = {change.change_id: change for change in changes}
    samples = [
        score_sample(sample, by_id, decisions)
        for sample in benchmark.sampled
        if sample.comparison_id in run_comparisons
    ]
    return {"summary": summarize(labels, samples), "labeled": labels, "sampled": samples}


# --- Running ---------------------------------------------------------------------------------


def tickers_of(benchmark: Benchmark) -> list[str]:
    return sorted({item.ticker for item in (*benchmark.changes, *benchmark.sampled)})


def record_for(result: MaterialityResult, text_changes: Sequence[Change], numeric: int) -> dict:
    return {
        "comparison_id": result.comparison_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "numeric_changes": numeric,
        "text_changes": [change.model_dump(mode="json") for change in text_changes],
        "result": result.model_dump(mode="json"),
    }


def print_summary(summary: dict) -> None:
    print(f"Labeled material changes found: {summary['labeled_found']}")
    print(f"  by status: {summary['labeled_by_status']}")
    print(
        f"Sampled changes labeled not material, called material: "
        f"{summary['sampled_false_positives']}"
    )
    print(f"Sampled changes labeled material, called material: {summary['sampled_material_found']}")
    if summary["samples_not_scored"]:
        print(f"  not scored: {summary['samples_not_scored']}")
    for cutoff, row in summary["by_importance_cutoff"].items():
        print(
            f"  importance >= {cutoff}: labeled {row['labeled_found']}, "
            f"false positives {row['sampled_false_positives']}, "
            f"sampled material {row['sampled_material_found']}"
        )


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    corpus = Corpus.load(CORPUS_PATH)
    benchmark = load_benchmark(args.labels)
    # A dry run writes nothing, so it leaves no empty output folder behind.
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
    ledger = CostLedger(args.cap, path=None if args.dry_run else args.out / "ledger.jsonl")
    cache = ReplayCache(settings.llm_cache_path) if settings.llm_cache else None
    llm = LLMClient(settings, ledger, cache=cache)
    embedder = CachedEmbedder(BgeEmbedder(), EMBEDDING_MODEL, embedding_cache_path(settings))
    records = []
    expected_total = worst_total = 0.0
    try:
        with FactStore(settings.duckdb_path, read_only=True) as store:
            catalog = MetricCatalog(store, corpus, MetricsConfig.load())
            for ticker in tickers_of(benchmark):
                for kind in ComparisonType:
                    comparison = corpus.comparison(ticker, kind)
                    alignment = align_comparison(
                        comparison, settings.parsed_filings_dir, embedder, store
                    )
                    text_changes = alignment.changes
                    scan = scan_comparison(catalog, comparison)
                    numeric = len(scan.changes)
                    # With no limit on calls in flight, the headroom is every uncached batch's
                    # worst case added up.
                    need = materiality_headroom(
                        comparison, text_changes, scan, llm, max_in_flight=10**6
                    )
                    expected = need.uncached * MATERIALITY_BATCH_USD
                    expected_total += expected
                    worst_total += need.headroom_usd
                    print(
                        f"{comparison.comparison_id}: {len(text_changes)} text + {numeric} "
                        f"numeric changes, {need.batches} batches, {need.uncached} uncached: "
                        f"about ${expected:.2f}, worst case ${need.headroom_usd:.2f}"
                    )
                    if args.dry_run:
                        continue
                    result = await rank_changes(
                        comparison, text_changes, scan, llm,
                        levels=catalog, chunks=store, facts=store,
                    )  # fmt: skip
                    record = record_for(result, text_changes, numeric)
                    path = args.out / f"{ticker}-{kind.value}.json"
                    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                    records.append(record)
                    print(
                        f"  ${result.cost_usd:.4f}, {result.calls} calls, "
                        f"{len(result.ranked.changes)} material, "
                        f"{len(result.unresolved)} unresolved -> {path}"
                    )
    finally:
        if cache is not None:
            cache.close()
    if args.dry_run:
        print(
            f"Uncached batches: about ${expected_total:.2f} at ${MATERIALITY_BATCH_USD:.3f} a "
            f"batch; worst case ${worst_total:.2f} (failed batches are split and asked again)."
        )
        return 0
    scores = score(benchmark, records)
    scores["spent_usd"] = ledger.spent_usd
    scores["labels"] = str(args.labels)
    (args.out / "scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False) + "\n")
    print_summary(scores["summary"])
    print(f"spent ${ledger.spent_usd:.4f}; scores -> {args.out / 'scores.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("labels", type=Path, help="eval/benchmark/HSY.json or test.json")
    parser.add_argument("out", type=Path, help="directory for one JSON file per comparison")
    parser.add_argument("--cap", type=float, default=2.00, help="dollar cap for this run")
    parser.add_argument(
        "--dry-run", action="store_true", help="print batches and costs; send nothing"
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
