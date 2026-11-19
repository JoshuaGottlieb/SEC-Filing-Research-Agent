"""Change Analysis, second half: the Tree-of-Thought search that explains each material change.

- `generator.py`: proposes candidate explanations (depth 1) and the evidence to check them
  (depth 2) (T4.2).
- `critic.py`, `scoring.py`: rate each branch on the rubric; the numeric criterion has a third
  state, not applicable (T4.3, design gap 1). The critic cannot award `confirmed`.
- `controller.py`: the beam, gates, ties, retry and budget rules, all in plain code (T4.4).
- `labeling.py`: turns the survivors into labeled claims: confirmed, plausible or unresolved (T4.5,
  design gap 4).
- `rework.py`: at most 2 rounds of more evidence for a claim the judge found only implied (T5.5,
  design gap 11).
- `lines.py`, `back_reference.py`: which financial lines a sentence names, and filings that explain
  one line "by the same factors" as another.

Every node, score and pruned branch is kept in the tree store (`tree_mcp/`).
"""
