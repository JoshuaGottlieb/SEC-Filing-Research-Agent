"""Change Analysis, first half: what changed between two filings, and which changes matter.

- `align.py`: pairs paragraphs across the two filings by embedding similarity and labels each pair
  added, removed, modified or unchanged (T3.1, design gap 3).
- `embedding_cache.py`: saved paragraph embeddings, so a repeated alignment skips the model.
- `materiality.py`: Opus judges which text and numeric changes are material and why, and code
  ranks them into `RankedChanges` (T3.3).

The second half, explaining each material change, is the Tree-of-Thought search in `tot/`.
"""
