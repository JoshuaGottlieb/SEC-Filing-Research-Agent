"""Verification and Reporting, the output side: the brief, and saved runs read back.

- `brief.py`, `summary.py`, `templates/`: the Markdown brief and its JSON audit trail (T5.4). The
  summary is written by Opus and checked in code; the whole brief is refused if it breaks the
  causal-language rule (design gap 13).
- `names.py`: the reader-facing names of the records' enums.
- `run_view.py`, `replay.py`, `run_cost.py`: a saved run read back for the app's views: evidence
  trees, review queue, step-by-step replay, and cost and time (T7.1 to T7.3, T6.6). They hold no
  Streamlit code, so tests can check them directly.
"""
