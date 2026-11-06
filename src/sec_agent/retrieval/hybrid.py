"""Hybrid search: keyword and dense results merged by reciprocal rank fusion.

Embeddings blur exact terms such as "freight" or "tariff", which matter in filings, while keyword
search misses a passage that says the same thing in other words. Each finds what the other misses.
Both run inside the same filter, so a question about one company's Q2 MD&A never competes with
other text. The cross-encoder (T2.3) then reads the fused candidates.

Fusion uses only ranks, because BM25 scores and cosine distances are not on comparable scales.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field
from rank_bm25 import BM25Plus

from sec_agent.finance.fact_store import FactStore
from sec_agent.ingest.index import ChunkFilter, ChunkIndex
from sec_agent.schemas import Chunk

# The constant from the paper introducing reciprocal rank fusion (Cormack et al., 2009). It keeps a
# single first place from outweighing broad agreement between the two lists.
RRF_K = 60
# How far down each list fusion looks. It is more than the 20 candidates the reranker reads, so a
# chunk ranked modestly by both lists can still beat one ranked highly by only one. Chosen by hand.
RANKED_LIST_SIZE = 50
# Whole numbers keep their separators, so "11,692.6" stays one token. Letters split at everything
# else, so "supply-chain" also matches a search for "supply chain".
TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[a-z]+")
# Words ending in these keep their final "s": business, analysis, status.
KEEPS_FINAL_S = ("ss", "is", "us")
# Plurals that add "es", so that "taxes" matches "tax".
ES_PLURAL = ("ches", "shes", "sses", "xes", "zes")
# Shorter words are left alone, so that "gas" and "has" stay as they are.
MIN_PLURAL_LENGTH = 4


class IndexOutOfSyncError(RuntimeError):
    """Dense search returned a chunk the store doesn't have. Rebuild the index."""


class FusedCandidate(BaseModel):
    """A chunk a hybrid search found, with where it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: Chunk
    # The sum of 1 / (RRF_K + rank) over the lists that found the chunk.
    score: float = Field(gt=0)
    # Each list's rank from 1, or None if that list did not find the chunk.
    keyword_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)


def _singular(word: str) -> str:
    """A light plural fold, so that "tariff" finds "tariffs". Other word forms stay distinct."""
    if len(word) < MIN_PLURAL_LENGTH or not word.endswith("s") or word.endswith(KEEPS_FINAL_S):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(ES_PLURAL):
        return word[:-2]
    return word[:-1]


def tokenize(text: str) -> list[str]:
    """Words and numbers for keyword search. Queries and chunks must both go through this."""
    return [_singular(token) for token in TOKEN.findall(text.lower())]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Merge ranked ids into (id, score), best first. Equal scores go to the lower id."""
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class _KeywordIndex:
    """BM25 over the chunks one filter allows, returning only chunks with a word of the query.

    It uses BM25Plus, for two reasons found by testing rank_bm25. A filter can leave very few
    chunks, and in a set that small its Okapi variant scores a match zero when the term is in half
    the chunks or more, and negative in a single chunk. Its BM25L variant multiplies by the raw
    word count, so a word repeated 25 times scores about 47 times one use, and a long table
    repeating "sales" would outrank everything. BM25Plus has neither problem, but it gives a chunk
    with none of the query's words a small baseline score, so those are removed by their words.
    """

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        tokens = [tokenize(chunk.text) for chunk in chunks]
        self._words = [set(words) for words in tokens]
        # BM25Plus can't be built on no documents, which an empty filter leaves.
        self._bm25 = BM25Plus(tokens) if chunks else None

    def search(self, query: str, top_k: int) -> list[str]:
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        wanted = set(tokens)
        scored = zip(self._chunk_ids, self._bm25.get_scores(tokens), self._words, strict=True)
        matches = [(chunk_id, score) for chunk_id, score, words in scored if wanted & words]
        matches.sort(key=lambda item: (-item[1], item[0]))
        return [chunk_id for chunk_id, _ in matches[:top_k]]


class HybridSearcher:
    """Keyword and dense search over the same filtered chunks, fused into one ranking.

    Each filter's keyword index is built once and kept. Chunks read at that point don't change
    during a run, so create a new searcher after re-chunking. The caller owns the store and index.
    """

    def __init__(self, store: FactStore, index: ChunkIndex) -> None:
        self._store = store
        self._index = index
        self._scopes: dict[ChunkFilter, tuple[dict[str, Chunk], _KeywordIndex]] = {}

    def _scope(self, where: ChunkFilter) -> tuple[dict[str, Chunk], _KeywordIndex]:
        if where not in self._scopes:
            chunks = [
                chunk for chunk in self._store.chunks(where.accession) if where.matches(chunk)
            ]
            by_id = {chunk.chunk_id: chunk for chunk in chunks}
            self._scopes[where] = (by_id, _KeywordIndex(chunks))
        return self._scopes[where]

    def search(self, query: str, where: ChunkFilter, top_k: int) -> list[FusedCandidate]:
        """The best `top_k` chunks the filter allows, best first. Fewer if fewer match."""
        by_id, keywords = self._scope(where)
        keyword_ranked = keywords.search(query, RANKED_LIST_SIZE)
        dense_ranked = [hit.chunk_id for hit in self._index.search(query, where, RANKED_LIST_SIZE)]
        stale = [chunk_id for chunk_id in dense_ranked if chunk_id not in by_id]
        # Passing over these would hide the problem, and a stale chunk can't be cited anyway.
        if stale:
            raise IndexOutOfSyncError(f"not in the store for {where}: {stale[:3]}")

        keyword_ranks = {chunk_id: rank for rank, chunk_id in enumerate(keyword_ranked, start=1)}
        dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(dense_ranked, start=1)}
        fused = reciprocal_rank_fusion([keyword_ranked, dense_ranked])
        return [
            FusedCandidate(
                chunk=by_id[chunk_id],
                score=score,
                keyword_rank=keyword_ranks.get(chunk_id),
                dense_rank=dense_ranks.get(chunk_id),
            )
            for chunk_id, score in fused[:top_k]
        ]
