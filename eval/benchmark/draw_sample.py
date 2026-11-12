"""Draw the false-positive sample for T6.3: text changes picked at random from T3.1's alignment.

Run from the repository root:

    uv run --frozen python eval/benchmark/draw_sample.py eval/benchmark/drafts/sampled.json

The seed and the per-company counts were fixed on 2026-11-11, before any change was labeled, and
must not change: a different draw after seeing labels would no longer be a fair sample. Alignment
is deterministic for fixed filings and model, so a rerun draws the same changes. Each company's
changes from its three comparisons are pooled and drawn uniformly without replacement; the
user then labels each one material or not.

A change to alignment can change the draw. The sample was redrawn once, on 2026-11-12, after
PR #45 corrected seq-10q pairing (quarter paragraphs had paired with six-month ones). The seed and
counts stayed the same, and only HSY-F14 and MAT-F07 changed. Both were relabeled, and
`review_log.json` keeps the superseded decisions.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from sec_agent.benchmark import Benchmark, SampledChange, save_benchmark
from sec_agent.compare.align import align_comparison
from sec_agent.compare.embedding_cache import CachedEmbedder, embedding_cache_path
from sec_agent.config import get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore
from sec_agent.ingest.chunker import EMBEDDING_MODEL
from sec_agent.ingest.index import BgeEmbedder
from sec_agent.schemas import Change

REPO_ROOT = Path(__file__).resolve().parents[2]
# Fixed before labeling (the date it was chosen). Never change it.
SEED = 20261111
# 40 in all: Hershey (tuning) slightly larger because thresholds are tuned on it alone.
PER_COMPANY = {"HSY": 14, "MAT": 13, "SWK": 13}


def all_changes() -> list[Change]:
    settings = get_settings()
    embedder = CachedEmbedder(BgeEmbedder(), EMBEDDING_MODEL, embedding_cache_path(settings))
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    changes: list[Change] = []
    with FactStore(settings.duckdb_path, read_only=True) as store:
        for company in corpus.companies:
            for comparison in company.comparisons():
                result = align_comparison(comparison, settings.parsed_filings_dir, embedder, store)
                changes.extend(result.changes)
    return changes


def draw(changes: list[Change]) -> list[SampledChange]:
    rng = random.Random(SEED)
    sampled: list[SampledChange] = []
    for ticker, count in PER_COMPANY.items():
        # Sorted by id first so the draw depends only on the set of changes, not their order.
        pool = sorted(
            (c for c in changes if c.comparison_id.startswith(f"{ticker}:")),
            key=lambda c: c.change_id,
        )
        for n, change in enumerate(rng.sample(pool, count), start=1):
            sampled.append(
                SampledChange(
                    item_id=f"{ticker}-F{n:02d}",
                    comparison_id=change.comparison_id,
                    change_id=change.change_id,
                    kind=change.kind,
                    section=change.section,
                    current_text=change.current_text,
                    prior_text=change.prior_text,
                )
            )
    return sampled


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    changes = all_changes()
    sampled = draw(changes)
    save_benchmark(Benchmark(sampled=tuple(sampled)), out)
    print(f"{len(changes)} changes, {len(sampled)} drawn with seed {SEED} -> {out}")
