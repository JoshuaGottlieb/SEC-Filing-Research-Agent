# Evaluation

This note covers what was measured, on which data, and the results. Each evaluation can be rerun with `sec-agent eval` (see [eval/README.md](../eval/README.md)). Design changes made in response to these results are in [DESIGN_CHANGES.md](DESIGN_CHANGES.md). Task IDs in the headings, such as T6.2, are the build tasks listed in [TASKS.md](TASKS.md).

**Status (2026-11-16):** every planned evaluation has been run on the final version of the agent, including the test split, once. The three optional reruns also ran on 2026-11-16 (retrieval at rerank pool 10, survival on the test split at width 6, and Haiku against Sonnet).

**Rerun after the review fixes (2026-11-19):** two bugs found in the code review changed what the search produced: the beam lost tied branches to float rounding, and metric names were matched as substrings ([DESIGN_CHANGES.md](DESIGN_CHANGES.md) has both). Everything that depends on the search or on labeling was rerun on the fixed code: the 9 agent runs (T8.1), with every model call live except materiality; T6.5 scoring of the agent's side; T6.4 survival at width 6; and T6.6. The figures below are from those reruns. Nothing was tuned on the test split's new results. Retrieval (T6.2), materiality (T6.3), the judge (T5.2), the baseline and the hand check do not depend on the search, so they were not rerun.

## Benchmark

Labels are in [eval/benchmark/](../eval/benchmark/README.md):
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
| Reranked, no cutoff | 19/21 | 47/48 |
| Hybrid top 10 (the reranker's input) | 21/21 | 47/48 |
| Hybrid top 20 (the input before it was cut to 10) | 21/21 | 48/48 |

The score cutoff costs some recall. In exchange, it empties the filing on most questions whose topic is absent from one filing (test: 2 of 3), instead of returning unrelated passages. The hardest questions in every company ask what drove gross margin.

Both columns use the current rerank pool of 10 (HSY measured 2026-11-15, test 2026-11-16). Cutting the pool from 20 changed none of the agent's figures, and the test misses are the same four passages. One test passage (MAT-R06) now ranks 15th in the hybrid search, outside the reranker's input. At a pool of 20 it was reranked to 7th, still outside the top 5.

**Redundancy (T2.4).** The plan was to add MMR diversification only if more than 2 of the top 5 passages exceeded 0.95 similarity. The highest similarity measured was 0.947, so MMR was not added.

### Materiality (T6.3, test $5.22)

The pipeline's own detection and materiality code runs on each comparison, and its decisions are scored against the labels. A labeled change counts as found when a changed paragraph holding one of its quotes is ranked material. The pipeline's cutoff is importance 3 of 5.

| Measure, at the pipeline's cutoff | HSY (tune) | Test |
|---|---|---|
| Labeled material changes found | 6/7 | **16/16** |
| Sampled changes labeled not material but ranked material | 0/11 | 1/21 |
| Sampled changes labeled material and ranked material | 3/3 | 2/5 |

What the test run shows:
- **Nothing labeled was missed.** Half of the 16 (8) were found at importance exactly 3, so the cutoff sits close to the labeled changes. That was also true on HSY, where the one miss (the removed Russia-Ukraine risk) was scored 2.
- **Lowering the cutoff would cost precision.** At 2, the test split gives 4/21 false positives (4/5 sampled material found); at 1, 10/21 (5/5).
- **Cost:** 76 batches of 20 changes, $0.069 a batch. The answers are cached, so the agent runs below did not pay for materiality again.

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
| HSY, beam width 6 (current setting, rechecked after the fixes) | 6/8 | 8/8 |
| Test, reported run (width 5, before the generator fix) | 3/17 | 15/17 |
| Test, comparison rerun after the generator fix (width 5) | 11/17 | 16/17 |
| Test, beam width 6 (current setting, rechecked after the fixes) | 12/17 | 16/17 |

What the rows mean:
- **The reported test numeric rate** was low because the generator never saw the earlier value or the direction of the move. In one Mattel change, all 12 candidates said gross margin rose when it fell, and the direction check pruned them all. The generator now sees both. The rerun is labeled a comparison, because the fix came after the test run.
- **The widening to 6** came from HSY: the narrower beam cut labeled causes that had passed the threshold. The cost was more surviving distractors (HSY numeric 1 to 2, text 0 to 2) and about 40% more searches.
- **Test at width 6** (run 2026-11-16, $0.32): one more numeric cause survived, 12/17, and text stayed at 16/17. Surviving distractors rose, as on HSY: numeric 6 to 7, text 2 to 4. Most calls replayed from the width-5 rerun's cache: the generator's prompt does not depend on the width, which only sets how many candidates go on to evidence. This run is at the final settings, and nothing was changed because of it. The rerun on the fixed code (2026-11-19) gave the same survival on both splits, with the same surviving distractors.

**After labeling, no distractor reached a claim as fact** in either split. A distractor is a plausible cause the filings do not state. In the rerun on the fixed code, one HSY numeric candidate that named a distractor ("lower freight and warehousing costs", for a quarterly gross-margin rise) was confirmed, but the claim written from its quote says only what the quote states: "Transformation program net savings lowered cost of sales." On the test split, no distractor was confirmed.

The evaluation did find claims confirmed without explaining their change:
- a six-month quote used for a quarterly change;
- a claim that named the change only inside a denial;
- a claim about one metric filed under a text change about another.

Each now has a code guard (see [DESIGN_CHANGES.md](DESIGN_CHANGES.md), departure 7). One questionable confirmation, found in the runs before the fixes, is open: a Stanley claim that pairs a stated cause with an unstated one, which the judge accepted on the stated part.

### Agent against a prompt-only baseline (T6.5)

This is the evaluation the [Module 5 plan](design/Module5_Multi-Agent_Architecture_Plan.pdf) commits to: unsupported claims and wrong-period citations, compared with a prompt-only baseline.

The baseline is one Opus call per comparison. It gets the filing text (MD&A, risk factors, market risk) and the same metric table the agent uses, and returns up to 10 changes with cited explanations. Its prompt carries the agent's causal-wording rule and its quote rule, but not its period rule, since periods are what is measured. The prompt was fixed after the Hershey check (version `e1b5caf7b755`).

Both systems are scored by the same code (`eval/baseline/score.py`):
- **Citation checker:** each quote is checked for verbatim text, filing, period and section.
- **Judge:** the T5.2 judge rates each asserted claim, meaning each claim that states a cause.
- **Wrong-period support:** a claim counts only if it rests on a quote from the wrong period, with no right-period quote stating the cause.
- **Cause matcher:** the T6.4 matcher compares claims with the labeled causes.
- **Coverage:** an Opus matcher decides which labeled material changes each brief covers. The agent's own headings name only a text change's kind and section, so its changes are described to the matcher by its materiality rationale, the counterpart of the baseline's one-line description.

The agent's side is the full-corpus run on the fixed code (2026-11-19): 10 changes per comparison, as for the baseline. Scoring the agent's side cost $1.22 for both splits; the baseline's scores are unchanged.

| Measure | HSY agent | HSY baseline | Test agent | Test baseline |
|---|---|---|---|---|
| Changes / claims / asserted claims | 29 / 98 / 52 | 30 / 51 / 35 | 57 / 177 / 58 | 59 / 91 / 56 |
| Quotes found verbatim | 128/128 | 71/71 | 261/261 | 134/134 |
| **Asserted claims unsupported by their quotes** | **5/52 (10%)** | 7/35 (20%) | **0/58** | 14/56 (25%) |
| **Asserted claims resting on a wrong-period quote** | **0/52** | 2/35 | **0/58** | 2/56 |
| Claims with figures not traceable to a metric | 0/98 | 6/51 | 0/177 | 4/91 |
| Labeled causes named | 7/8 | 6/8 | 13/17 | **17/17** |
| Distractors asserted as fact | 0 | 0 | 0 | 0 |
| Labeled material changes covered | 2/7 | 5/7 | 7/16 | **13/16** |
| Sequential: causes asserted / saying "year over year" | 16 / 0 | 6 / 2 | 10 / 1 | 11 / 4 |

What the table shows:
- **By the judge, the agent's assertions hold up better.** On the test split, none of the agent's 58 asserted causes was unsupported by its quotes, against 14 of the baseline's 56 (25%). On Hershey, the tune split, 5 of the agent's 52 were (10%), against 7 of 35 (20%). These are small samples, so the rates are directional: none during testing does not mean none in general. Part of this is by construction, because the agent's `confirmed` claims already passed the same judge. The hand check below does not rest on the judge, and it narrows this gap.
- **The citation and figure checks favor the agent.** The agent has no wrong-period support and no untraceable figures; the baseline has a few of each. Its untraceable figures are real numbers, though (below), so the difference is that the agent's figures are checked by code, not that the baseline's are made up.
- **The agent asserts less.** 33% of its test claims state a cause, against 62% of the baseline's. The rest are labeled `plausible` or `unresolved`, so a reader is told they are not established.
- **The baseline covers more labeled changes.** The agent explores only its 10 most material changes. In these comparisons, those are mostly margin moves and the MD&A paragraphs that explain them. Most labeled changes (risk factors, litigation, acquisitions, buybacks) rank lower. On the test split, every labeled change the agent missed was outside its 10, and every one it explored was covered. Materiality had found all 16 (above), so the gap comes from the 10-change cap and the ranking, not from detection. A larger `--max-changes` would close it, at the cost and time measured below.
- **The baseline also names more labeled causes on the test split** (17 against 13). Of the 4 the agent missed, 3 belong to a change it did not explore, because of the cap (Mattel's quarterly SG&A rise). The other 1 belongs to a change it did explore (Mattel's FY2025 gross margin), where its search did not surface it. That is consistent with T6.4, where the search kept 11 to 16 of the 17 test causes depending on the form of the change.
- **Sequential comparisons stay hard for both.** The 10-Qs explain moves only against the year-earlier quarter. Both systems still offer those causes for quarter-to-quarter moves, mostly without saying the cause is year over year: the baseline says so for 6 of its 17 such causes, the agent for 1 of 26. The agent's quotes cite the right filing and period, so its problem is the wording.

Notes on the baseline:
- **Compound claims:** most of its unsupported claims were compound: the quote states one part, and the claim adds more.
- **Wrong-period cases:** on HSY, one of the two is a changed tariff risk that the claim contrasts with the prior filing. The agent's own rule would also count it. One test case (`SWK:yoy-10q:b2:c2`, no valid citation) describes the prior-year quarter itself and is counted the same way.
- **Untraceable figures:** every figure in its 10 flagged claims appears in that company's filings, in the claim's own quotes, elsewhere in the filing text, or in the metric table it was given. The flag means code cannot check the figure against the metric catalog, not that the figure is wrong.

**Are the baseline's changes real?** (`sec-agent eval grounding`, free.) The agent's changes come from the paragraph diff and the metric scan, so they are real differences between the filings by construction. The baseline's come from one model call, so they were checked against the same code:
- **Text changes:** all 50 are grounded. Each has a quote in a passage the diff marks as changed.
- **Numeric changes:** 34 of 39 moved the way their description says, by the metric catalog. The other 5 name no catalog metric (a segment's result, corporate expense, cash, segment margins), or move from a loss to a profit, which has no relative change. All 5 were checked by hand against the filings and the metric table, and all are right.

So neither system was found inventing changes. The difference between them is in the explanations, not the changes.

**What each system costs.** Per comparison, on the same 9 comparisons:

| | Agent (10 changes) | Baseline |
|---|---|---|
| Model cost | $0.44 to $1.41, plus materiality on a comparison's first run: about $1.00 to $2.70 | $0.22 to $0.58 |
| Wall time | 23 to 44 minutes | 29 to 62 seconds |
| Model calls sent | 410 to 770 | 1 |
| All 9 comparisons | $8.59, plus materiality | $3.18 |

The baseline is one Opus call that reads the filing text at once. The agent makes hundreds of smaller calls, searches locally between them, and keeps the record behind every claim. That buys traceability and checking, not coverage: at 10 changes, it covers fewer labeled changes than the baseline does.

**Blinded hand check.** It rated claims from the runs before the fixes, and was not repeated, since it measures the judge against a person on the claims as they were. A person rated 30 asserted claims, 15 per system, drawn with a seed fixed in advance from test-split claims that have a valid quote. The two systems' claims were mixed together and their source hidden. The question for each claim is the judge's: does a quote state that the claim's cause produced this change?

| | Agent | Baseline |
|---|---|---|
| Supported, by the person | 13/15 | 14/15 |
| Supported, by the judge | 14/15 | 12/15 |
| Person and judge agree | 14/15 | 13/15 |

The person and the judge agree on 27 of 30. The person applied two rules the judge's prompt does not spell out:
- **Offsets:** a claim that says "X reduced Y" is not supported when the quote says Y moved the other way and X only partly offset it. The claim implies Y moved the wrong way overall.
- **Direction:** a quote that names only "the year-over-year change" in a metric does not confirm that it rose or fell. The judge applies this one too, reading such quotes as "implies".

The three disagreements:
- **Agent, 1:** the offset rule. The judge accepted the claim.
- **Baseline, 2:** the judge was stricter than the person. One claim has two parts, each stated in a different quote, and the judge reads one quote at a time. In the other, the quote says "the net proceeds" without naming their source.

Apart from the offset case, the judge never accepted a claim the person rejected, so it errs on the strict side, as in T5.2. That strictness fell only on baseline claims (2 of 15). On this sample, the person found the two systems about equally well supported (13/15 and 14/15). So part of the gap in unsupported claims above (0% and 10% against 20 to 25%) comes from the judge's strictness, not from the baseline's claims. With 15 claims per system, the sample cannot say how much.

One answer was revised on a reread, under the direction rule. The first pass agreed on 26 of 30. The person also noted that some claims are weak or unspecific, though supported. This is recorded as a known limitation in [DESIGN_CHANGES.md](DESIGN_CHANGES.md).

### Full-corpus run (T8.1, $8.59)

The agent ran on all 9 comparisons, one after another, on the fixed code (2026-11-19; first run 2026-11-15 to 16): 3 companies × 3 comparisons, 10 changes each, a $2.00 cap per run. All 9 finished, wrote their brief, and searched all 10 changes; none hit its cap (the most was $1.41). Every model call was live except materiality, which was cached from T6.3. Each run's `brief.md`, `audit.json`, `ledger.jsonl` and `timings.jsonl` are in `<data dir>/runs/`.

### Cost and latency (T6.6, free from the runs above)

Every model call records its cost, latency, step and change, and every step and local search is timed. `sec-agent eval cost RUN_DIR` reports them per step, per finding and per model.

The runs used the final settings (beam width 6), on a 4-core CPU machine with no GPU and no other work running (load average 0.6 to 1.3). All 90 changes ran fresh.

| Per explored change | Mean | Median | 90th percentile | Max |
|---|---|---|---|---|
| Model cost | $0.091 | $0.078 | $0.166 | $0.221 |
| Wall time | 3.5 min | 3.4 min | 5.5 min | 7.6 min |

Per run of 10 changes: $0.44 to $1.41 and 23 to 44 minutes. A comparison seen for the first time also pays for materiality: 6 to 22 batches here, at about $0.07 each. That would bring these runs to about $1.00 to $2.70.

Where the money goes (all 9 runs, $8.59):
- **By step:** labeling $3.31, rework $3.51, search $1.36, summaries $0.42 (about $0.05 a run).
- **By model:** Opus (the judge and the summary) made 14% of the calls and cost 67%. Haiku (the generator, the critic and the claim writer) made the rest.

Where the time goes:
- **Model calls are now most of it:** 71% of a change's wall time. A typical call takes 1.0 to 1.9 seconds on Haiku and 3.1 to 4.1 on Opus (per-run medians), and about 5 to 6 at the 90th percentile.
- **Local search is the rest:** 27%. A change runs about 41 searches at 1.4 seconds each. Earlier runs, with a rerank pool of 20, spent most of their time searching, at 2 to 6 seconds a search.

Other observations:
- **Repeated calls replay free within a run.** 15 to 35% of each change's calls repeat an identical earlier request (mostly in rework) and replay from the local cache. A fresh run gets the same saving.
- **Cut-off replies were rare.** 6 of 5,575 live calls reached their output limit (5 critic, 1 generator), and each ended only its own branch. The five critic replies could not be parsed and were charged their reserved worst case, $0.012 in all. `sec-agent eval cost` reports these per step, finding and model, since their cost is an upper bound.
- **A killed run resumes.** A run killed mid-search resumed without repeating its finished work, and its replayed calls cost nothing.
- **The pre-run estimate was too low.** It assumed $0.06 a change and printed $0.65 for every run. It now uses the measured mean from the rerun, $0.091 (it was $0.085 before the fixes).

### Haiku against Sonnet as the search model (T6.6, HSY, $2.52)

Should the generator and critic run on Sonnet instead of Haiku? Both ran the HSY survival evaluation at beam width 6, scored by the same Opus matcher (Sonnet run 2026-11-16).

| Search model | Numeric | Text | Surviving distractors (numeric, text) | Cost per live call |
|---|---|---|---|---|
| Haiku (current) | 6/8 | 8/8 | 2, 2 | $0.0008 |
| Sonnet | 7/8 | 8/8 | 4, 1 | $0.010 |

- **Sonnet kept one more cause:** "unfavorable mark-to-market activity on commodity derivatives", in Hershey's FY2025 gross margin change. Both models missed the same other numeric cause.
- **Sonnet cost about 13 times as much:** $2.36 for its two searches (about $0.49 per numeric change and $0.30 per text change), against about $0.19 for Haiku at the same number of calls.
- **Time was about the same:** a median call of 1.5 to 1.6 seconds on Sonnet against 1.7 to 3.3 on Haiku, and similar wall times.

Haiku stays the search model. One cause in 8, on a single run, is not a measurable gain at 13 times the search cost, and Sonnet let more distractors survive on the numeric form.

## Still to run

Nothing. The three optional reruns ran on 2026-11-16 and are reported above: the retrieval test table at rerank pool 10, survival on the test split at width 6, and Haiku against Sonnet.
