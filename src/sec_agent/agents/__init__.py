"""The two agents that are more than a package of tools: Retrieval and Verification.

- `retrieval.py`: the Retrieval Agent's bounded loop (T3.4). When a search comes back empty for one
  filing, it rewords the query, at most 3 rounds. Rewording is off by default (DESIGN_CHANGES.md,
  departure 2), so a run's searches use no model.
- `verification.py`: the checks a claim must pass before it reaches the brief: the citation
  checker (T5.1), the fresh-context entailment judge (T5.2) and the causal-language check (T5.3).
  Together they fill design gaps 4 and 13.

The other two agents of the design live in their own packages: Change Analysis in `compare/` and
`tot/`, Financial Analysis in `finance/`. The deterministic controller that routes between them
is `graph.py`.
"""
