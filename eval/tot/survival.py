"""T6.4: does the known cause survive the Tree-of-Thought search?

Run from the repository root (paid: the generator and critic calls on the worker model):

    uv run --frozen python eval/tot/survival.py eval/benchmark/HSY.json numeric out.json --cap 1.00

Each labeled stated cause (`StatedCause`) goes straight to `explore_change`, skipping the scan's
flag and the materiality cutoff (the user's decision, 2026-11-12), so survival is measured apart
from materiality, which T6.3 measures. Two input forms, also the user's decision:

- `numeric`, the reported rate. A numeric `Change` for the label's metric, built as the scan
  builds one (`scan_metric`, then the same fields as `finance/scan.py`'s `_change`), whether or not
  the scan would flag it. It shows the search only the move, never the filing's explanation.
- `text`, a secondary measure. The alignment text change whose current paragraph holds the label's
  cause quote. That paragraph states the cause, so this form shows how far reading the answer lifts
  survival and isolates whether the critic prunes a cause it was handed.

Nothing here decides whether a candidate *is* the known cause: the output lists every depth-1
candidate with where it ended (pruned at the pre-screen, pruned at depth 2, or surviving), next to
the label's causes and distractors, for matching as a separate step.

Isolation from other sessions: its own tree database and replay cache under
`<data_dir>/eval/t64/` (never `settings.tree_db_path`), and a run id per split, form and start time,
so it cannot collide with integration runs. The fact store opens read-only. Tune on HSY only; the
test file (MAT, SWK) is run once, after any beam change is settled on HSY.

`--candidate-width` overrides the beam's kept-candidate count for this run only, leaving
`config/scoring.yaml` as it is. The test run used 5 (the user's decision, 2026-11-13): HSY's
rerun showed two labeled causes passing the pre-screen but cut by the width of 4 behind candidates
scoring 1.0, and 5 still fits the 40-node first round (6 + 5 x 6 = 36). The config's width became
6, with budgets of 55 and 45, on 2026-11-15 (the user's decision), so the reported runs (HSY 4,
test 5, both with 40 and 30) predate it.

`--label` also runs labeling (`label_result`, with the same context the workflow builds) on each
search's survivors, and records every candidate's final label: `confirmed`, `plausible`, left out
for having only an agreeing metric, or merged into an identical confirmed claim. It is paid (the
entailment judge on the strong model, the claim writer on the worker model) and answers what the
search alone cannot: whether a surviving distractor is confirmed, or only reaches the brief as a
plausible claim worded without a cause (the user's question, 2026-11-13). Labeling raises on the
dollar cap instead of stopping, so a capped run ends there and writes what it has.

Searches are kept on disk (`PersistentRetriever`, `<data_dir>/eval/t64/search_cache.sqlite`), so a
rerun whose model calls replay also replays its searches, which took most of a replayed run's time
(the user's decision, 2026-11-15). Any change to the chunks or the search code misses the cache.
`--no-search-cache` searches afresh, for timing a run.

Timing (T6.6): beside the ledger, `<out>.timings.jsonl` holds each change's search and labeling
wall time and every search actually run (a replayed one is not), and each ledger line names the
change it served and how long the call took. `eval/cost/cost_latency.py` reads them.
`--worker-model` runs the generator, critic and claim writer on another model for this run only
(T6.6's Haiku against Sonnet); the output records both models. The replay cache keys on the model,
so a new worker model's calls are all live.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from sec_agent.agents.verification import EntailmentJudge
from sec_agent.benchmark import StatedCause, load_benchmark
from sec_agent.compare.align import align_comparison
from sec_agent.compare.embedding_cache import CachedEmbedder, embedding_cache_path
from sec_agent.config import get_settings
from sec_agent.contracts import Retriever
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore
from sec_agent.finance.metrics import MetricCatalog, MetricsConfig
from sec_agent.finance.scan import ScanThresholds, SeasonalStatus, scan_metric
from sec_agent.ingest.chunker import EMBEDDING_MODEL
from sec_agent.ingest.index import BgeEmbedder, ChunkIndex
from sec_agent.llm.cache import ReplayCache
from sec_agent.llm.client import LLMClient
from sec_agent.llm.ledger import BudgetExceededError, CostLedger
from sec_agent.retrieval.hybrid import HybridSearcher
from sec_agent.retrieval.rerank import CrossEncoderScorer, Reranker
from sec_agent.retrieval.search_cache import PersistentRetriever
from sec_agent.retrieval.tools import SearchRetriever
from sec_agent.schemas import Change, ChangeKind, Comparison, NodeStatus, TreeNode
from sec_agent.timing import STAGE_LABEL, STAGE_SEARCH, TimedRetriever, TimingLog
from sec_agent.tot.controller import StopReason, ToTContext, ToTResult, explore_change
from sec_agent.tot.critic import Critic
from sec_agent.tot.generator import ThoughtGenerator
from sec_agent.tot.labeling import ClaimWriter, LabeledFinding, LabelingContext, label_result
from sec_agent.tot.scoring import ScoringConfig
from sec_agent.tree_mcp.client import InProcessTreeClient
from sec_agent.tree_mcp.store import TreeStore

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMS = ("numeric", "text")


def numeric_change(catalog: MetricCatalog, comparison: Comparison, cause: StatedCause) -> Change:
    """The scan's numeric change for the label's metric, built even when the scan would not flag
    it."""
    if cause.metric is None:
        raise SystemExit(f"{cause.item_id}: no metric, so no numeric form")
    move = scan_metric(catalog, comparison, cause.metric, ScanThresholds())
    if move.metric_id is None or move.move is None:
        raise SystemExit(f"{cause.item_id}: the catalog has no {cause.metric.value} move")
    checked = move.seasonal in (SeasonalStatus.SIMILAR_MOVE, SeasonalStatus.NO_SIMILAR_MOVE)
    return Change(
        change_id=f"{comparison.comparison_id}:numeric:{cause.metric.value}",
        comparison_id=comparison.comparison_id,
        kind=ChangeKind.NUMERIC,
        metric_id=move.metric_id,
        seasonality_checked=checked,
        possibly_seasonal=move.seasonal is SeasonalStatus.SIMILAR_MOVE,
    )


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_change(changes: tuple[Change, ...], cause: StatedCause) -> Change:
    """The one alignment change whose current text holds the label's cause quote."""
    quote = _squash(cause.quote.text)
    found = [c for c in changes if c.current_text and quote in _squash(c.current_text)]
    if len(found) != 1:
        ids = [c.change_id for c in found]
        raise SystemExit(f"{cause.item_id}: {len(found)} text changes hold the quote {ids}")
    return found[0]


def _candidate_rows(tree: tuple[TreeNode, ...], result: ToTResult) -> list[dict]:
    survivors = {c.candidate.node_id: c for c in result.candidates}
    children: dict[str, list[TreeNode]] = {}
    for node in tree:
        if node.parent_id is not None:
            children.setdefault(node.parent_id, []).append(node)
    rows = []
    for node in (n for n in tree if n.depth == 1):
        evidence = children.get(node.node_id, [])
        if node.node_id in survivors:
            stage = "survived"
        elif not evidence:
            stage = "pruned_at_prescreen"
        else:
            stage = "pruned_at_depth2"
        kept = survivors[node.node_id].evidence if node.node_id in survivors else ()
        rows.append(
            {
                "node_id": node.node_id,
                "thought": node.thought,
                "stage": stage,
                "status": node.status.value,
                "score": node.score,
                "prune_reason": node.prune_reason,
                "evidence_tried": len(evidence),
                "evidence_pruned": sum(e.status is NodeStatus.PRUNED for e in evidence),
                "kept_evidence": [
                    {
                        "thought": e.thought,
                        "evidence": e.evidence.value if e.evidence else None,
                        "score": e.score,
                        "quotes": [c.quote for c in e.citations],
                    }
                    for e in kept
                ],
            }
        )
    return rows


def _add_labels(rows: list[dict], labeled: LabeledFinding) -> None:
    """Each candidate's final label, by node id (labeling keeps the candidate's node id as the
    claim id). A survivor with no claim was left out or merged, and the failure notes say which."""
    claims = {claim.claim_id: claim for claim in labeled.finding.claims}
    for row in rows:
        node_id = row["node_id"]
        claim = claims.get(node_id)
        if claim is not None:
            row["label"] = claim.label.value
            row["claim_text"] = claim.text
            row["entailment"] = claim.entailment.value if claim.entailment else None
            row["claim_quotes"] = [citation.quote for citation in claim.citations]
        elif row["stage"] != "survived":
            row["label"] = None
        else:
            notes = [f for f in labeled.failures if f.startswith(f"claim {node_id}:")]
            if any("only computed metrics" in note for note in notes):
                row["label"] = "left_out_metric_only"
            elif any("merged" in note for note in notes):
                row["label"] = "merged"
            else:
                row["label"] = "no_claim"


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    if args.worker_model is not None:
        settings = settings.model_copy(update={"model_worker": args.worker_model})
    causes = load_benchmark(args.labels).causes
    if args.only:
        causes = tuple(c for c in causes if c.item_id in args.only)
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    comparisons = {c.comparison_id: c for co in corpus.companies for c in co.comparisons()}
    work = settings.data_dir / "eval" / "t64"
    started = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_id = f"t64-{args.labels.stem.lower()}-{args.form}-{started}"
    ledger = CostLedger(args.cap, path=args.out.with_suffix(".ledger.jsonl"))
    timings = TimingLog(args.out.with_suffix(".timings.jsonl"))
    config = ScoringConfig.load()
    if args.candidate_width is not None:
        beam = config.beam.model_copy(update={"candidate_width": args.candidate_width})
        config = config.model_copy(update={"beam": beam})
    llm = LLMClient(settings, ledger, cache=ReplayCache(work / "replay_cache.sqlite"))
    with FactStore(settings.duckdb_path, read_only=True) as store:
        catalog = MetricCatalog(store, corpus, MetricsConfig.load())
        bge = BgeEmbedder()  # loaded once and shared, as `graph.run_analysis` does
        changes: dict[str, Change] = {}
        alignments: dict[str, tuple[Change, ...]] = {}
        for cause in causes:
            comparison = comparisons[cause.comparison_id]
            if args.form == "numeric":
                changes[cause.item_id] = numeric_change(catalog, comparison, cause)
                continue
            if cause.comparison_id not in alignments:
                embedder = CachedEmbedder(bge, EMBEDDING_MODEL, embedding_cache_path(settings))
                alignments[cause.comparison_id] = align_comparison(
                    comparison, settings.parsed_filings_dir, embedder, store
                ).changes
            changes[cause.item_id] = text_change(alignments[cause.comparison_id], cause)
        # Timed under the disk cache, so only searches actually run are logged.
        retriever: Retriever = TimedRetriever(
            SearchRetriever(
                Reranker(
                    HybridSearcher(store, ChunkIndex(settings.chroma_dir, bge)),
                    CrossEncoderScorer(),
                )
            ),
            timings,
        )
        if not args.no_search_cache:
            retriever = PersistentRetriever(retriever, work / "search_cache.sqlite", store.chunks)
        tree = InProcessTreeClient(TreeStore(work / "trees.sqlite"))
        context = ToTContext(
            retriever=retriever,
            checker=catalog,
            metrics=catalog,
            generator=ThoughtGenerator(llm),
            critic=Critic(llm),
            tree=tree,
            config=config,
        )
        labeling = LabelingContext(
            judge=EntailmentJudge(llm),
            writer=ClaimWriter(llm),
            chunks=store,
            metrics=catalog,
            facts=store,
        )
        print(f"run {run_id}: {len(causes)} causes, form {args.form}, cap ${args.cap:.2f}")
        items = []
        for cause in causes:
            change = changes[cause.item_id]
            before = ledger.spent_usd
            with timings.measure(STAGE_SEARCH, change.change_id):
                result = await explore_change(
                    change, comparisons[cause.comparison_id], context, run_id
                )
            nodes = await tree.read_tree(result.tree_root_id) if result.tree_root_id else ()
            rows = _candidate_rows(nodes, result)
            capped = False
            labeled = None
            if args.label:
                try:
                    with timings.measure(STAGE_LABEL, change.change_id):
                        labeled = await label_result(
                            result, change, comparisons[cause.comparison_id], labeling, run_id
                        )
                    _add_labels(rows, labeled)
                except BudgetExceededError:
                    capped = True
            items.append(
                {
                    "item_id": cause.item_id,
                    "comparison_id": cause.comparison_id,
                    "change_label": cause.change,
                    "metric": cause.metric.value if cause.metric else None,
                    "direction": cause.direction.value if cause.direction else None,
                    "causes": list(cause.causes),
                    "distractors": list(cause.distractors),
                    "change_id": change.change_id,
                    "outcome": result.outcome.value,
                    "review_reason": result.review_reason,
                    "stopped": result.stopped.value if result.stopped else None,
                    "fresh_round": result.fresh_round,
                    "failures": list(result.failures),
                    "node_evaluations": result.node_evaluations,
                    "model_calls": result.model_calls,
                    "cost_usd": round(ledger.spent_usd - before, 4),
                    "labeling": None
                    if labeled is None
                    else {
                        "review_reason": labeled.finding.review_reason,
                        "failures": list(labeled.failures),
                        "judge_calls": labeled.judge_calls,
                        "writer_calls": labeled.writer_calls,
                    },
                    "candidates": rows,
                }
            )
            stages = [r["stage"] for r in rows]
            print(
                f"{cause.item_id}: {result.outcome.value}, {stages.count('survived')} of "
                f"{len(rows)} candidates survived, ${ledger.spent_usd - before:.4f}"
            )
            # A spent node budget ends only this change's search; the dollar cap ends the run,
            # as in `explore_changes`.
            if result.stopped is StopReason.NODE_BUDGET:
                print("stopped: node_budget for this change; the next one still runs")
            elif result.stopped is StopReason.RUN_BUDGET or capped:
                print("stopped: run_budget; later causes not run")
                break
    args.out.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "labels": str(args.labels),
                "form": args.form,
                "models": {"worker": settings.model_worker, "strong": settings.model_strong},
                "beam": config.beam.model_dump(),
                "spent_usd": round(ledger.spent_usd, 4),
                "items": items,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"spent ${ledger.spent_usd:.4f} -> {args.out}")
    if isinstance(retriever, PersistentRetriever):
        print(f"searches: {retriever.cache_hits} replayed, {retriever.cache_misses} run")
        retriever.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path)
    parser.add_argument("form", choices=FORMS)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cap", type=float, required=True, help="hard stop in dollars")
    parser.add_argument("--only", nargs="*", default=(), help="item ids to run")
    parser.add_argument(
        "--candidate-width", type=int, help="kept candidates for this run (default: the config's)"
    )
    parser.add_argument("--label", action="store_true", help="also label survivors (paid)")
    parser.add_argument(
        "--no-search-cache", action="store_true", help="search afresh instead of replaying"
    )
    parser.add_argument(
        "--worker-model", help="worker model for this run (default: SEC_AGENT_MODEL_WORKER)"
    )
    asyncio.run(run(parser.parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
