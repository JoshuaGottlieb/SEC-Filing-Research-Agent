"""T2.4: measure redundancy in the search's top 5 on the benchmark's retrieval questions.

    python eval/retrieval/redundancy.py LABELS.json [...] [--tickers HSY,MAT] [--out report.json]
    python eval/retrieval/redundancy.py --stand-in [--out report.json]

The plan adds MMR only if "more than 2 of the top 5 exceed 0.95 similarity" (see
`sec_agent.retrieval.redundancy`). This runs every retrieval question through the real search
(`SearchRetriever`: hybrid search, cross-encoder rerank, top 5 per filing) and prints how many of
each filing's passages nearly copy a higher-ranked one, then the totals the decision rests on.

Label files are read for `retrieval[].item_id`, `comparison_id`, `question` and `section` only, so
this runs before the benchmark module is merged; once it is, it can load files through
`sec_agent.benchmark.load_benchmark` instead. `--stand-in` searches hand-picked queries over every
comparison's MD&A instead: a preview of the measurement, not the benchmark.

Local models only: no network, no paid calls. Reads the chunk store and the index read-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore
from sec_agent.ingest.index import BgeEmbedder, ChunkIndex
from sec_agent.retrieval.hybrid import HybridSearcher
from sec_agent.retrieval.redundancy import (
    MAX_REDUNDANT,
    NEAR_DUPLICATE,
    RedundancyQuestion,
    measure_redundancy,
)
from sec_agent.retrieval.rerank import CrossEncoderScorer, Reranker
from sec_agent.retrieval.tools import SearchRetriever
from sec_agent.schemas import Comparison, Section

# Topics an analyst asks of consumer and industrial MD&A, chosen by hand for the preview. They
# are not labels: nothing says which passages should come back.
STAND_IN_QUERIES = (
    "freight and shipping costs",
    "cocoa prices",
    "tariffs",
    "restructuring charges",
    "foreign currency exchange rates",
    "inventory",
    "pricing actions",
    "volume declines",
    "advertising expense",
    "supply chain costs",
    "share repurchases",
    "goodwill impairment",
    "consumer demand",
    "new product launches",
    "gross margin",
)


def read_questions(
    paths: list[Path], comparisons: dict[str, Comparison], tickers: set[str] | None
) -> list[RedundancyQuestion]:
    questions = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("retrieval", []):
            comparison = comparisons[item["comparison_id"]]
            if tickers and comparison.ticker not in tickers:
                continue
            section = item.get("section")
            questions.append(
                RedundancyQuestion(
                    question_id=item["item_id"],
                    query=item["question"],
                    comparison=comparison,
                    section=Section(section) if section else None,
                )
            )
    return questions


def stand_in_questions(comparisons: dict[str, Comparison]) -> list[RedundancyQuestion]:
    return [
        RedundancyQuestion(
            question_id=f"{comparison_id}#{n}",
            query=query,
            comparison=comparison,
            section=Section.MDNA,
        )
        for comparison_id, comparison in comparisons.items()
        for n, query in enumerate(STAND_IN_QUERIES)
    ]


async def run(questions: list[RedundancyQuestion], out: Path | None) -> None:
    settings = get_settings()
    with (
        FactStore(settings.duckdb_path, read_only=True) as store,
        ChunkIndex(settings.chroma_dir, BgeEmbedder()) as index,
    ):
        searcher = SearchRetriever(Reranker(HybridSearcher(store, index), CrossEncoderScorer()))
        report = await measure_redundancy(questions, searcher, index.vectors)
        accessions = {
            filing.accession
            for question in questions
            for filing in (question.comparison.current, question.comparison.prior)
        }
        texts = {
            chunk.chunk_id: chunk.text
            for accession in accessions
            for chunk in store.chunks(accession)
        }

    print(
        f"\nnear-duplicate: similarity above {NEAR_DUPLICATE}; a list is redundant with more "
        f"than {MAX_REDUNDANT} near-duplicates\n"
    )
    for question in report.questions:
        sides = []
        for name, side in (("later", question.current), ("earlier", question.prior)):
            top = "-" if side.max_similarity is None else f"{side.max_similarity:.3f}"
            flag = " REDUNDANT" if side.is_redundant else ""
            sides.append(f"{name} {side.redundant}/{side.passages} (max {top}){flag}")
        print(f"{question.question_id} {question.query!r}: {'; '.join(sides)}")
        for side in (question.current, question.prior):
            for pair in side.pairs:
                print(
                    f"    {pair.similarity:.3f} {pair.redundant} copies {pair.kept}: "
                    f"{' '.join(texts.get(pair.redundant, '')[:120].split())!r}"
                )

    lists = report.lists()
    redundant = report.redundant_lists()
    full = [side for side in lists if side.passages == 5]
    print(
        f"\nquestions: {len(report.questions)}; lists with passages: {len(lists)} "
        f"({len(full)} with all 5)"
    )
    print(f"near-duplicates per list: {report.redundant_counts()}")
    share = f"{len(redundant) / len(lists):.0%}" if lists else "n/a"
    print(
        f"redundant lists (more than {MAX_REDUNDANT}): {len(redundant)} of {len(lists)} ({share})"
    )
    if out:
        out.write_text(report.model_dump_json(indent=1), encoding="utf-8")
        print(f"report written to {out}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", nargs="*", type=Path, help="benchmark label files")
    parser.add_argument("--stand-in", action="store_true", help="preview on stand-in queries")
    parser.add_argument("--tickers", help="comma-separated tickers to keep (default: all)")
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)
    if bool(args.labels) == args.stand_in:
        parser.error("give label files or --stand-in, not both")
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    comparisons = {c.comparison_id: c for co in corpus.companies for c in co.comparisons()}
    if args.stand_in:
        questions = stand_in_questions(comparisons)
    else:
        tickers = set(args.tickers.split(",")) if args.tickers else None
        questions = read_questions(args.labels, comparisons, tickers)
    if not questions:
        print("no retrieval questions found")
        return 1
    asyncio.run(run(questions, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
