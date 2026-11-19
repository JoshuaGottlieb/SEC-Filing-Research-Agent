"""T6.2: recall@5 on the benchmark's retrieval questions, with and without the reranker.

Run from the repository root (local models only: no network, no paid calls):

    uv run --frozen python eval/retrieval/recall.py eval/benchmark/HSY.json out.json

For every retrieval question, each filing is searched the way the agent searches it, and every
labeled passage is looked for in four lists:

- `hybrid@5`: the hybrid search's top 5 (BM25 plus dense, fused), with no reranker.
- `reranked@5`: what the agent gets, the `Reranker` as built (`MIN_SCORE`, `KEEP_PER_PERIOD`).
- `reranked@5, no cutoff`: the reranker's top 5 with no minimum score, so the cutoff's own cost
  shows apart from the reordering.
- `hybrid@pool`: the reranker's candidate pool (`CANDIDATES_PER_PERIOD`), the most it can reach.
- `hybrid@20`: the hybrid top 20, the pool before it became 10 (2026-11-15), so the table shows
  what the smaller pool gives up.

The scored unit is the labeled quote (the method used to tune Hershey). A quote
is found when any chunk holding it within the question's section is in the list; chunks overlap,
so a quote near a boundary can sit in two, and either counts. A side with no quote is a filing
where the topic is absent; there the agent's list should come back empty, and the report says
whether it did.

Hershey is the tuning set (measured on 2026-11-12: 17 of 21 found at `MIN_SCORE` -6, all
4 misses in HSY-R01, and the absent side not emptied). The test file (MAT, SWK) is run once, for
the reported table. The fact store and the index open read-only.

Results: `recall_2026-11-13.json` (HSY and test, pool of 20), `recall_hsy_2026-11-15.json` (HSY,
pool of 10) and `recall_test_2026-11-16.json` (test, pool of 10). The reported table uses the two
pool-of-10 files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sec_agent.benchmark import RetrievalQuestion, Side, load_benchmark, resolve_quote
from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore
from sec_agent.ingest.index import BgeEmbedder, ChunkFilter, ChunkIndex
from sec_agent.retrieval.hybrid import HybridSearcher
from sec_agent.retrieval.rerank import (
    CANDIDATES_PER_PERIOD,
    KEEP_PER_PERIOD,
    MIN_SCORE,
    CrossEncoderScorer,
    Reranker,
)
from sec_agent.schemas import Comparison

NO_CUTOFF = -1e9
# The candidate pool before it became 10, reported for comparison.
EARLIER_POOL = 20
LISTS = ("hybrid@5", "reranked@5", "reranked@5_no_cutoff", "hybrid@pool", "hybrid@20")


def _side_filter(comparison: Comparison, side: Side, question: RetrievalQuestion) -> ChunkFilter:
    filing = comparison.current if side is Side.CURRENT else comparison.prior
    return ChunkFilter(accession=filing.accession, section=question.section)


def _rank(chunk_ids: tuple[str, ...], ranked: list[str]) -> int | None:
    """The best rank from 1 at which any of the quote's chunks appears, or None."""
    ranks = [ranked.index(c) + 1 for c in chunk_ids if c in ranked]
    return min(ranks) if ranks else None


def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    questions = load_benchmark(args.labels).retrieval
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    comparisons = {c.comparison_id: c for co in corpus.companies for c in co.comparisons()}
    rows = []
    with (
        FactStore(settings.duckdb_path, read_only=True) as store,
        ChunkIndex(settings.chroma_dir, BgeEmbedder()) as index,
    ):
        hybrid = HybridSearcher(store, index)
        scorer = CrossEncoderScorer()
        agent = Reranker(hybrid, scorer)
        # Every candidate, reordered, with no cutoff: the diagnostic view of the same scores.
        # The reranker refuses an infinite cutoff; raw scores run about -11.4 to +6.3, so
        # -1e9 keeps every candidate.
        wide = Reranker(hybrid, scorer, min_score=NO_CUTOFF, keep=CANDIDATES_PER_PERIOD)
        for question in questions:
            comparison = comparisons[question.comparison_id]
            for side in Side:
                where = _side_filter(comparison, side, question)
                hybrid_ids = [
                    c.chunk.chunk_id
                    for c in hybrid.search(
                        question.question, where, max(CANDIDATES_PER_PERIOD, EARLIER_POOL)
                    )
                ]
                agent_ids = [p.chunk.chunk_id for p in agent.rerank(question.question, where)]
                wide_passages = wide.rerank(question.question, where)
                wide_ids = [p.chunk.chunk_id for p in wide_passages]
                scores = {p.chunk.chunk_id: p.score for p in wide_passages}
                lists = {
                    "hybrid@5": hybrid_ids[:KEEP_PER_PERIOD],
                    "reranked@5": agent_ids,
                    "reranked@5_no_cutoff": wide_ids[:KEEP_PER_PERIOD],
                    "hybrid@pool": hybrid_ids[:CANDIDATES_PER_PERIOD],
                    "hybrid@20": hybrid_ids[:EARLIER_POOL],
                }
                quotes = [q for q in question.quotes if q.side is side]
                gold = []
                for quote in quotes:
                    chunk_ids = resolve_quote(quote, comparison, store.chunks, question.section)
                    if not chunk_ids:
                        raise SystemExit(
                            f"{question.item_id}: a {side.value} quote resolves to no chunk"
                        )
                    found = {name: _rank(chunk_ids, ids) is not None for name, ids in lists.items()}
                    best = [scores[c] for c in chunk_ids if c in scores]
                    gold.append(
                        {
                            "quote": quote.text[:120],
                            "chunk_ids": list(chunk_ids),
                            "found": found,
                            "hybrid_rank": _rank(chunk_ids, hybrid_ids),
                            "reranked_rank": _rank(chunk_ids, wide_ids),
                            "rerank_score": round(max(best), 3) if best else None,
                        }
                    )
                rows.append(
                    {
                        "item_id": question.item_id,
                        "comparison_id": question.comparison_id,
                        "section": question.section.value if question.section else None,
                        "question": question.question,
                        "side": side.value,
                        "absent": not quotes,
                        "agent_returned": len(agent_ids),
                        "top_score": round(wide_passages[0].score, 3) if wide_passages else None,
                        "gold": gold,
                    }
                )

    gold = [g for r in rows for g in r["gold"]]
    totals = {name: sum(g["found"][name] for g in gold) for name in LISTS}
    absent = [r for r in rows if r["absent"]]
    emptied = sum(r["agent_returned"] == 0 for r in absent)
    answered_empty = sum(r["agent_returned"] == 0 for r in rows if not r["absent"])
    summary = {
        "labels": str(args.labels),
        "questions": len(questions),
        "gold_quotes": len(gold),
        "min_score": MIN_SCORE,
        "candidates_per_period": CANDIDATES_PER_PERIOD,
        "found": totals,
        "absent_sides": len(absent),
        "absent_sides_emptied": emptied,
        "answered_sides_emptied": answered_empty,
    }
    print(
        f"{len(questions)} questions, {len(gold)} labeled passages, MIN_SCORE {MIN_SCORE}, "
        f"pool {CANDIDATES_PER_PERIOD}"
    )
    for name in LISTS:
        share = totals[name] / len(gold) if gold else 0.0
        print(f"  {name:22} {totals[name]:3}/{len(gold)}  {share:.0%}")
    print(f"absent sides emptied by the agent's list: {emptied}/{len(absent)}")
    print(f"answered sides the cutoff emptied: {answered_empty}")
    for r in rows:
        for g in r["gold"]:
            if not g["found"]["reranked@5"]:
                print(
                    f"  missed {r['item_id']} {r['side']}: hybrid rank {g['hybrid_rank']}, "
                    f"reranked rank {g['reranked_rank']}, score {g['rerank_score']}"
                )
    args.out.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"-> {args.out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path)
    parser.add_argument("out", type=Path)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
