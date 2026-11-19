"""The Financial Analysis agent: fixed calculations in code, with no model and no free code.

- `fact_store.py`: the DuckDB store of XBRL and table facts, the source of truth for numbers (T1.4).
- `statements.py`: reads a filed income statement from its table, to check the loaded facts.
- `metrics.py`: the metric catalog (T1.6, design gap 10): margins, growth, cost ratios, segment and
  dollar metrics, and the direction check the critic uses (agree, contradict, not applicable).
- `scan.py`: the metric scan that proposes numeric changes, with the seasonality comparison for
  sequential quarters (T3.2, design gaps 6 and 7).

The agent runs twice (design gap 2): first as the scan, then as a checking tool inside the
Tree-of-Thought search.
"""
