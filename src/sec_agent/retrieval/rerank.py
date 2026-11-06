"""Reranking: a cross-encoder reads the hybrid candidates and decides which are worth keeping.

Hybrid search never returns nothing for a filter that has chunks, because dense search fills its
list even when no chunk is relevant. This module is where "nothing relevant" becomes possible: a
passage must score at least `min_score` to be kept, and a period where none does comes back empty,
which is what makes `RetrievalResult.incomplete` true.

Both of a comparison's filings are always searched, each on its own. A change is a difference, so
evidence from one period alone cannot show one, and the caller needs to know which side is missing.

The score is the model's raw output, not a probability. `sentence-transformers` 6.2 already returns
it raw for this model, but older versions applied a sigmoid for single-label models, so the scorer
sets the activation explicitly. A sigmoid would squash every score into 0 to 1 and the contract's
cutoff would then mean something else.

The model reads at most 512 tokens for the query and passage together, and cuts the end of a
passage that doesn't fit. A chunk can be 500 tokens, so with a 20-token query about one pair in ten
loses up to 11 tokens at its end (measured on the 2,176 pinned chunks on 2026-11-06). Chunks overlap
by about 10%, so that text is usually the start of the next chunk, and nothing is done about it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from sec_agent.ingest.index import ChunkFilter
from sec_agent.retrieval.hybrid import FusedCandidate
from sec_agent.schemas import Comparison, RetrievalResult, RetrievedPassage, Section

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# How many fused candidates per period the cross-encoder reads, and how many passages are kept.
# Both are the plan's figures.
CANDIDATES_PER_PERIOD = 20
KEEP_PER_PERIOD = 5
# The lowest raw score a passage may have and still be kept. A provisional value, set by the user
# on 2026-11-06 from a study with no labels (proxy labels, three companies): of 120 searches for
# topics a filing lacks, 2 still looked complete at -6, against 11 at -7 and 1 at -4; of 60 MD&A
# searches any filing answers, 2 came back empty at -6, against 0 at -7 and 14 at -4. Genuine
# answers do score between -8 and -6, so some are lost. Revisit with the benchmark labels (T6.1).
MIN_SCORE = -6.0
# Pairs scored per forward pass. Speed was flat from 8 to 64 on this 4-core CPU (10 to 14 pairs a
# second, one run each), so this is the fastest of those runs and not a tuned value.
SCORING_BATCH_SIZE = 16


class CandidateSearcher(Protocol):
    """What the reranker needs from a search. `HybridSearcher` provides it."""

    def prepare(self, where: ChunkFilter) -> None: ...

    def search(self, query: str, where: ChunkFilter, top_k: int) -> list[FusedCandidate]: ...


class PairScorer(Protocol):
    """Scores how well each passage answers the query. Higher is more relevant."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderScorer:
    """The cross-encoder, on CPU. Creating one loads it, downloading it on first use."""

    def __init__(self, batch_size: int = SCORING_BATCH_SIZE) -> None:
        # Imported here because they are slow to import and tests inject a fake scorer instead.
        import torch
        from sentence_transformers import CrossEncoder

        self._batch_size = batch_size
        # Identity keeps the raw score whatever the library's default is (see the module docstring).
        self._model = CrossEncoder(
            CROSS_ENCODER_MODEL, device="cpu", activation_fn=torch.nn.Identity()
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


class Reranker:
    """Hybrid search followed by the cross-encoder, one filing at a time.

    `min_score` decides when a period counts as having no match, so it sets how often a result
    is `incomplete`. It defaults to `MIN_SCORE`, which is provisional. The caller owns the
    searcher and the scorer.
    """

    def __init__(
        self,
        searcher: CandidateSearcher,
        scorer: PairScorer,
        *,
        min_score: float = MIN_SCORE,
        candidates: int = CANDIDATES_PER_PERIOD,
        keep: int = KEEP_PER_PERIOD,
    ) -> None:
        if not math.isfinite(min_score):
            raise ValueError(f"min_score must be a finite number, not {min_score}")
        if not 1 <= keep <= candidates:
            raise ValueError(f"need 1 <= keep <= candidates, got keep={keep} of {candidates}")
        self._searcher = searcher
        self._scorer = scorer
        self._min_score = min_score
        self._candidates = candidates
        self._keep = keep

    def rerank(self, query: str, where: ChunkFilter) -> tuple[RetrievedPassage, ...]:
        """The best passages the filter allows, best first. Empty if none reaches `min_score`."""
        candidates = self._searcher.search(query, where, self._candidates)
        if not candidates:
            return ()
        scores = self._scorer.score(query, [candidate.chunk.text for candidate in candidates])
        if len(scores) != len(candidates):
            raise ValueError(f"scorer returned {len(scores)} scores for {len(candidates)} passages")
        # A NaN compares false with everything, so it would pass or fail the cutoff by accident.
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("scorer returned a score that is not a finite number")
        scored = [
            RetrievedPassage(chunk=candidate.chunk, score=score)
            for candidate, score in zip(candidates, scores, strict=True)
            if score >= self._min_score
        ]
        # The sort is stable, so equal scores keep their hybrid order.
        scored.sort(key=lambda passage: -passage.score)
        return tuple(scored[: self._keep])

    @staticmethod
    def _filters(
        comparison: Comparison, section: Section | None
    ) -> tuple[ChunkFilter, ChunkFilter]:
        return (
            ChunkFilter(accession=comparison.current.accession, section=section),
            ChunkFilter(accession=comparison.prior.accession, section=section),
        )

    def prepare(self, comparison: Comparison, section: Section | None = None) -> None:
        """Read what `search` will need from the store. See `HybridSearcher.prepare`."""
        for where in self._filters(comparison, section):
            self._searcher.prepare(where)

    def search(
        self, query: str, comparison: Comparison, section: Section | None = None
    ) -> RetrievalResult:
        """Search both filings of the comparison. An empty side means nothing passed the cutoff."""
        if not query.strip():
            raise ValueError("a search needs a query")
        current_filter, prior_filter = self._filters(comparison, section)
        # Each side is searched even when the other came back empty, so the caller learns
        # about both and not just the first gap.
        current = self.rerank(query, current_filter)
        prior = self.rerank(query, prior_filter)
        return RetrievalResult(query=query, current=current, prior=prior)
