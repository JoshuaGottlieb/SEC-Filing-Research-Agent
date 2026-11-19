"""The single path from the agents to Claude (T0.4).

- `client.py`: one model per role (Opus for materiality, the judge and the brief; Haiku for the
  generator and critic), structured output and prompt caching.
- `ledger.py`: records every call's tokens and cost and stops a run at its budget (design gap 11).
- `pricing.py`: the per-token prices the budget is enforced with.
- `cache.py`: replays an identical request from SQLite at no cost, so reruns and tests are cheap.
"""
