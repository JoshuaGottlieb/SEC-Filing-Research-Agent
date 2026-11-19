"""Turns raw filings into searchable text and structured facts.

- `pipeline.py`: the one route from a downloaded filing to text later stages may use; a filing
  that fails validation is never read.
- `sections.py`, `fallback.py`, `tables.py`: split a filing into canonical sections of narrative
  paragraphs, with tables kept apart (T1.2, design gap 8).
- `table_facts.py`: facts read from a filing's data tables, for figures XBRL doesn't tag.
- `audit.py`: checks parser output against evidence the parser didn't use.
- `chunker.py`: passages of 300 to 500 tokens for retrieval (T1.3).
- `index.py`: the Chroma index of chunk embeddings behind dense search (T2.1).

How ingestion is tested is described in `docs/How_SEC_Filing_Ingestion_Is_Tested.md`.
"""
