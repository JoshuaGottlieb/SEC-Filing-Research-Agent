# Evaluation

This note covers what has been measured so far, and what is still to run. Each evaluation can be rerun with `sec-agent eval` (see the [README](README.md#evaluation)). Design changes made in response to these results are in [DESIGN_CHANGES.md](DESIGN_CHANGES.md).

**Status (2026-11-15):** retrieval, the judge, Tree-of-Thought survival and the baseline's Hershey run are measured. The agent-against-baseline comparison and the cost and latency measurements are waiting on the final version of the agent, so that the paid runs are done once.

## Benchmark

Labels are in [eval/benchmark/](eval/benchmark/README.md):
- Claude drafted them from the indexed filing text, without running any of the agent's code, so a label cannot inherit the agent's mistakes.
- A person reviewed every one of the 130 items. None was edited, and three were rejected.

| Split | Companies | Retrieval questions | Material changes | Stated causes | Judge examples | Sampled changes |
|---|---|---|---|---|---|---|
| Tune | Hershey (HSY) | 9 | 7 | 3 | 9 | 14 |
| Test | Mattel (MAT), Stanley Black & Decker (SWK) | 19 | 16 | 6 | 18 | 26 |

Each company has three comparisons: FY2025 10-K against FY2024 (`yoy-10k`), Q2 2026 against Q2 2025 (`yoy-10q`), and Q2 2026 against Q1 2026 (`seq-10q`).

**Fair-test rule.** Thresholds and prompts were tuned on Hershey only. The test split is run once, after settings are fixed, and nothing is tuned on it. With a set this small, results are directional, not precise rates.

**One finding from labeling.** The 10-Qs state no causes for sequential (quarter-to-quarter) moves. Their explanations compare with the same quarter a year earlier. So sequential comparisons can rarely produce a `confirmed` cause for the move itself.

## Results

### Retrieval: recall@5 (T6.2, free)

The scored unit is a labeled quote. It counts as found if any chunk holding it is in the top 5 for its filing.

| Search | HSY (tune) | Test |
|---|---|---|
| Hybrid search only (BM25 plus dense) | 18/21 | 46/48 |
| **Agent: hybrid, reranked, with the score cutoff** | **17/21** | **44/48** |
| Reranked, no cutoff | 18/21 | 47/48 |
| Hybrid top 20 (the reranker's input) | 21/21 | 48/48 |

The score cutoff costs some recall. In exchange, it empties the filing on most questions whose topic is absent from one filing (test: 2 of 3), instead of returning unrelated passages. The hardest questions in every company ask what drove gross margin.

The test figures were measured with a rerank pool of 20, which has since been cut to 10. On HSY the cut changed nothing (17/21). The test table has not been rerun at 10.

**Redundancy (T2.4).** The plan was to add MMR diversification only if more than 2 of the top 5 passages exceeded 0.95 similarity. The highest similarity measured was 0.947, so MMR was not added.

### Materiality (T6.3)

Tuned on HSY: at the materiality cutoff, the agent found 6 of the 7 labeled material changes. None of the 11 sampled changes labeled not material was ranked material.

**Pending:** the test-split table.

### Entailment judge agreement (T5.2, $0.13)

The Opus judge's verdict (states cause, implies, unrelated) is compared with the labeled verdict for each claim and quote.

| Split | Agreement |
|---|---|
| HSY (tune) | 8/9 |
| Test | 15/18 |

Every miss in both splits leaned toward a weaker verdict, never a stronger one. The judge errs on the strict side, which is the safe direction for `confirmed`.

On HSY, two quotes that began "The increase was driven by..." without naming what increased were read as "implies". After the quotes were extended to include the sentence before, the judge agreed. The search's critic now asks for quotes that name what changed.

### Tree-of-Thought survival (T6.4)

Does each labeled cause survive the search? Each labeled change goes straight to the search, skipping materiality, so survival is measured on its own. An Opus matcher decides whether a surviving candidate names a labeled cause. On HSY it agreed with a hand check on every cause.

Two forms are measured:
- **Numeric:** the change is given as a metric move.
- **Text:** the change is given as the changed passage.

| Run | Numeric | Text |
|---|---|---|
| HSY, beam width 4 | 5/8 | 7/8 |
| HSY, beam width 6 (current setting) | 6/8 | 8/8 |
| Test, reported run (width 5, before the generator fix) | 3/17 | 15/17 |
| Test, comparison rerun after the generator fix (width 5) | 11/17 | 16/17 |

What the rows mean:
- **The reported test numeric rate** was low because the generator never saw the earlier value or the direction of the move. In one Mattel change, all 12 candidates said gross margin rose when it fell, and the direction check pruned them all. The generator now sees both. The rerun is labeled a comparison, because the fix came after the test run.
- **The widening to 6** came from HSY: the narrower beam cut labeled causes that had passed the threshold. The cost was more surviving distractors (HSY numeric 1 to 2, text 0 to 2) and about 40% more searches.
- **Test at width 6:** not run.

**After labeling, no distractor was confirmed** in either split. A distractor is a plausible cause the filings do not state.

The evaluation did find claims confirmed without explaining their change:
- a six-month quote used for a quarterly change;
- a claim that named the change only inside a denial;
- a claim about one metric filed under a text change about another.

Each now has a code guard (see DESIGN_CHANGES.md, departure 7). One questionable confirmation is open: a Stanley claim that pairs a stated cause with an unstated one, which the judge accepted on the stated part.

### Agent against a prompt-only baseline (T6.5)

This is the evaluation D1 commits to: unsupported claims and wrong-period citations, compared with a prompt-only baseline.

The baseline is one Opus call per comparison. It gets the filing text (MD&A, risk factors, market risk) and the same metric table the agent uses, and returns up to 10 changes with cited explanations. Its prompt carries the agent's causal-wording rule and its quote rule, but not its period rule, since periods are what is measured. The prompt was fixed after the Hershey check (version `e1b5caf7b755`).

Both systems are scored by the same code (`eval/baseline/score.py`):
- **Citation checker:** each quote is checked for verbatim text, filing, period and section.
- **Judge:** the T5.2 judge rates each asserted claim, meaning each claim that states a cause.
- **Wrong-period support:** a claim counts only if it rests on a quote from the wrong period, with no right-period quote stating the cause.
- **Cause matcher:** the T6.4 matcher compares claims with the labeled causes.

**Hershey, baseline only:**

| Measure | Baseline |
|---|---|
| Changes / claims / asserted claims | 30 / 51 / 35 |
| Quotes found verbatim | 71/71 |
| Asserted claims unsupported by their quotes | 7/35 (20%) |
| Asserted claims resting on a wrong-period quote | 2/35 |
| Claims with figures not traceable to a metric | 6/51 |
| Labeled causes named / distractors asserted | 6/8 / 0 |
| Labeled material changes covered | 5/7 |

Most unsupported claims were compound: the quote states one part, and the claim adds more.

One of the two wrong-period cases is a changed tariff risk that the claim contrasts with the prior filing. The agent's own rule would also count it.

In the sequential comparison the baseline asserted 6 causes. Only 2 of them say the comparison is with a year earlier, which is all the 10-Q states.

**Pending:**
- the agent's runs on Hershey;
- both systems on the test split, run once;
- a blinded hand check of 30 asserted claims, 15 per system, by a person.

### Cost and latency (T6.6)

The instrumentation is built:
- every model call records its cost, latency, step and change;
- every step and every local search is timed;
- `sec-agent eval cost RUN_DIR` reports cost and wall time per step, per finding and per model.

Measurements at the current settings are pending.

Early observations, from runs before the beam was widened:
- Three Hershey changes cost $0.15 to $0.19 and took about 7 minutes.
- Most of the time is local search on the CPU: about 2 to 6 seconds per search, depending on machine load.
- Model latency was a small share.
- A run killed mid-search resumed without repeating its finished work, and its replayed calls cost nothing.

## Still to run

| Item | Cost | Waiting on |
|---|---|---|
| Agent runs on Hershey, then both systems on the test split | Paid | Final version of the agent |
| Blinded hand check of 30 claims | Free (a person's time) | The runs above |
| Cost and latency at current settings | Paid | Final version of the agent |
| Materiality table on the test split | Paid | A runner, then a cost estimate |
| Survival on the test split at width 6 | Paid | A decision to rerun |
| Retrieval test table at rerank pool 10 | Free | Not scheduled |
| Full-corpus run (9 comparisons) | Paid | Approval of its spend |
