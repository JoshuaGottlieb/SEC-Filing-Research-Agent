"""The Retrieval agent's tools: hybrid search over a comparison's two filings (design gap 14).

- `hybrid.py`: BM25 plus dense search, combined by reciprocal rank fusion (T2.2).
- `rerank.py`: a cross-encoder keeps the best passages per filing above a minimum score, and flags
  a filing with no match as incomplete (T2.3).
- `tools.py`, `search_cache.py`: the `Retriever` the agents call, cached per query in memory and on
  disk (T2.5).
- `redundancy.py`: measures near-duplicate passages in the top results, the test for adding MMR
  (T2.4; not added).

The agent's rewording loop is in `agents/retrieval.py`.
"""
