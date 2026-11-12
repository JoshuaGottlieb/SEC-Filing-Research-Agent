"""Check label files: every quote must appear in its filing (and inside a retrieval question's
section filter), ids must be unique, and judge examples must name a known cause.

    uv run --frozen python eval/benchmark/check.py eval/benchmark/drafts/HSY.json [...]

Exits 1 and lists each problem if any file fails. Reads the chunk store read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

from sec_agent.benchmark import check_benchmark, load_benchmark
from sec_agent.config import get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(paths: list[Path]) -> int:
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    comparisons = {c.comparison_id: c for co in corpus.companies for c in co.comparisons()}
    failed = False
    with FactStore(get_settings().duckdb_path, read_only=True) as store:
        cache: dict[str, list] = {}

        def chunks(accession: str) -> list:
            if accession not in cache:
                cache[accession] = store.chunks(accession)
            return cache[accession]

        for path in paths:
            try:
                benchmark = load_benchmark(path)
            except ValidationError as error:
                print(f"{path}: invalid\n{error}")
                failed = True
                continue
            problems = check_benchmark(benchmark, comparisons, chunks)
            counts = ", ".join(
                f"{len(getattr(benchmark, kind))} {kind}"
                for kind in ("retrieval", "changes", "causes", "judge", "sampled")
            )
            print(f"{path}: {counts}; {len(problems)} problem(s)")
            for problem in problems:
                print(f"  {problem}")
            failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main([Path(p) for p in sys.argv[1:]]))
