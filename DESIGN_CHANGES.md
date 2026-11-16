# Design changes

This note compares what was built with the design: the Module 3 and Module 4 designs, and the Multi-Agent Architecture Plan (D1). It covers:
1. how the build fills each of the 14 gaps found in the design before building;
2. where the build departs from D1, and why;
3. what was built that the design did not ask for;
4. what is still open.

Evaluation results are in [EVALUATION.md](EVALUATION.md). "HSY" is Hershey, the company settings were tuned on. Mattel (MAT) and Stanley Black & Decker (SWK) are the test companies.

## The system as built

D1's structure stands: four agents, no planner, and a deterministic LangGraph controller that does the routing, retry limits and budget stops in code (`src/sec_agent/graph.py`).

| Agent | Where | Models |
|---|---|---|
| Retrieval | `agents/retrieval.py`, `retrieval/` | None by default (see departure 2) |
| Change Analysis | `compare/` (alignment, materiality), `tot/` (Tree-of-Thought) | Opus for materiality; Haiku for the generator and critic |
| Financial Analysis | `finance/` (fact store, metric catalog, scan) | None: fixed calculations in code |
| Verification and Reporting | `agents/verification.py`, `tot/labeling.py`, `report/` | Code checks; Opus for the judge and the brief |

The workflow runs these steps: detect, estimate, rank, select, explore (with rework), summarize, write. Agents pass typed Pydantic records through the graph's state. Every Tree-of-Thought node, score and pruned branch is kept in a local MCP tree store (`tree_mcp/`).

## How the 14 design gaps were filled

**1. Scoring bug: text-only explanations were always pruned.** Planned: numerical consistency gets a third state, "not applicable", which drops the criterion and rescales the other weights. Built as planned.

The HSY evaluation then found a second form of the bug. An explanation was pruned when one extra metric it expected to move contradicted the filings, even though the metric being explained agreed. Two changes followed:
- Numeric changes are now gated only on the changed metric, or a metric linked to it by arithmetic. Text changes are not gated.
- Numerical consistency is now the share of checked metrics that agree.

**2. Agent order conflict.** Planned: Financial Analysis runs twice, first as a metric scan, then as a checking tool inside the search. Built as planned (`finance/scan.py`, then `MetricCatalog` inside `tot/`).

**3. How changes are detected.** Planned: align paragraphs by embedding similarity with an assignment solver, then diff. Built as planned (`compare/align.py`):
- Identical paragraphs pair off first.
- The rest are matched with `linear_sum_assignment` on cosine similarity, with a floor of 0.75.
- A matched pair is "unchanged" only when its differences are the ones a filing makes when text is carried forward, such as years moving on. A text-similarity cutoff was tried and rejected, because real edits ("two" becoming "four", a new chief executive's name) scored 98 out of 100.
- A first-quarter 10-Q paragraph is kept from pairing with a later quarter's year-to-date paragraph.

**4. Deterministic verification against paraphrase.** Planned: every claim carries a verbatim quote, code checks the quote and its metadata, and a separate fresh-context judge classifies it. Built as planned, with D1's stricter version:
- The judge sees only the claim and its quote, never the reasoning behind them.
- The code checks (`agents/verification.py`) reject altered quotes, quotes from the wrong period, company or section, and figures that cannot be traced to a metric.
- `confirmed` requires both the code checks and a "states cause" verdict.

**5. Search budget had no slack.** Planned: 6 candidates and 6 sources each, 40 nodes per finding. Changed after measurement; see departure 1.

**6. Numeric materiality thresholds.** Planned: margins moving 100 basis points or more; revenue or a major expense line moving 10% or more. Kept after checking them against the HSY labels.

**7. Seasonality of sequential quarters.** Planned: compare each sequential move with the same move a year earlier, using XBRL history, and label it possibly seasonal. Built as planned in the metric scan.

The benchmark also found that 10-Qs state no causes for sequential moves; their explanations compare with the year before. The evaluation scores sequential causes with this in mind.

**8. 10-K and 10-Q sections differ.** Planned: map to canonical section names, and store a 10-Q's "no material changes" statement as a result. Built as planned (`ingest/sections.py`).

**9. Segment data.** Planned: a time-boxed check, using inline XBRL if it works. It works: segment values load from inline XBRL with the segment axis.

**10. Unavailable metrics.** Planned: a fixed metric catalog, with no arbitrary code execution. Built as planned (`finance/metrics.py`, `config/metrics.yaml`): margins, growth, cost ratios, segment values and dollar metrics. Unit-level evidence comes from MD&A text.

**11. Retry and spending limits.** Planned: 2 verification retries per claim, a $10 cap per run, and 40 nodes per finding.
- The $10 cap is the default (`SEC_AGENT_MAX_USD_PER_RUN`).
- Rework allows at most 2 rounds per finding (see departure 3).
- Node budgets changed (see departure 1).

**12. Human escalation.** Planned: non-blocking; unresolved items go to the brief and a Streamlit review queue. The run never pauses:
- The review queue is the brief's "Needs review" section, with the details in `audit.json`.
- The Streamlit app is not part of this submission.

**13. Causal-language rule.** Planned: a deterministic check that fails phrases such as "due to" unless the finding is `confirmed`. Built as planned (`find_causal_language`). It runs when claims are labeled, and again on the whole brief, which is refused if it breaks the rule. Each phrase in the list carries the reason it is there.

**14. Reranker and stores.** Planned: `cross-encoder/ms-marco-MiniLM-L-6-v2`, Chroma, and DuckDB as the source of truth. Built as planned, with these changes:
- Search is hybrid: BM25 plus dense search, combined by reciprocal rank fusion, then reranked.
- The rerank pool was cut from 20 to 10 to save CPU time. On HSY, recall did not change.
- A faster ONNX reranker was tried and rejected.
- MMR was measured and not added: no top-5 list exceeded the trigger.

## Departures from D1

**1. A wider search, with a larger node budget.** D1 set 40 nodes per finding, which allows a beam of 4 explanations.

On HSY, the depth-1 pre-screen gave a perfect score to more candidates than the beam held, in all six changes. Ties fall back to the order the generator wrote them in, so the beam of 4 cut labeled causes that had passed the threshold.

Now:
- the beam is 6, so every candidate past the 0.30 threshold is kept (`config/scoring.yaml`);
- the first round may use 55 nodes, 42 of its own plus 13 for retries;
- a fresh round, if needed, may use 45.

On HSY, causes kept rose from 5 to 6 of 8 (numeric) and from 7 to 8 of 8 (text). The cost is more surviving distractors and about 40% more searches per change. The dollar cap still bounds each run.

**2. The Retrieval Agent's rewording is off by default.** D1 has a bounded ReAct loop on Haiku that rewords a query when one filing comes back empty. It is built and tested (`agents/retrieval.py`), but switched off, so retrieval is one plain search.

In a live check, rewording filled two of five empty filings with passages about another topic. The citation checker confirms a quote's text, period and section, but not its relevance. An off-topic quote could therefore support a `plausible` claim, which is worse than an empty filing. The search controller has its own retry, across every section.

**3. Rework targets what the judge rejected.** D1 says failed verification returns claims for rework, at most 2 retries per claim.

As built:
- A claim the judge finds only implied gets another round of searching and is relabeled.
- There are at most 2 rounds per finding.
- A round stops if it would eat into the money reserved for labeling.
- A claim whose quote fails the code checks is not retried. It stays out of the brief.

**4. At most 10 changes are explored per run.** D1 explored findings in materiality order until the budget ran out. Each explored change takes minutes of local search and some model spend, so the run explores the 10 most material by default (`--max-changes`). The brief says how many were left, and `audit.json` lists them.

**5. Two budget checks before spending.** D1 had a dollar cap that drops the least material findings. Two checks were added, so money is not spent on work the cap cannot finish:
- Before materiality, the run needs enough free budget for the whole step, or it ends having spent nothing.
- Before each search, it needs enough left to label the result and write the summary.

**6. A stopped run can resume.** Not in D1. The graph saves its state after every step. `--resume` continues a killed run from its last finished change, and paid calls already made replay from a local cache at no cost. This was tested by killing a paid run mid-search.

**7. Further guards on `confirmed`.** The evaluation found ways a claim could be confirmed without explaining the change. Each now has a code guard (`tot/labeling.py`):
- A quote covering six months cannot confirm a claim about a quarter.
- A claim that names the change only inside a denial ("not because gross margin...") is rejected.
- A claim about a text change must fit the changed passage.
- The generator is shown the metric's earlier value and its direction. Without them, every candidate for one Mattel change said gross margin rose when it fell.
- When the judge says "implies", the claim is restated at the level its quote supports, then judged again.

**8. Claude through the official SDK.** The design used `langchain-anthropic`. Claude is called through the `anthropic` SDK instead, wrapped in a LangChain `Runnable`, because current models reject forced tool choice and sampling parameters. The wrapper (`llm/client.py`) adds:
- structured output, through `messages.parse`;
- prompt caching;
- refusal handling;
- a cost ledger with a hard budget stop;
- a replay cache.

A reply cut off mid-JSON is charged its worst case, and only that branch ends.

**9. A pinned, offline corpus.** EDGAR is read once, for 15 pinned filings (`config/corpus.yaml`). The agents query local data at run time. Module 3 allows this for a prototype.

## Built beyond the design

- **A validation gate for parsing.** Every filing is parsed, audited against evidence the parser did not use, and retried with a fallback parser if it fails. Rejected filings can never reach the index. The gate was measured on four sets of randomly drawn companies ([eval/parser_holdout.md](eval/parser_holdout.md)).
- **Table facts.** Figures in untagged tables are loaded as searchable facts when the whole table can be explained. Other tables are indexed as text.
- **Cost and latency logs.** Every model call records its step, change and latency, and every step and local search is timed (`timing.py`, `report/run_cost.py`).
- **An evaluation CLI.** `sec-agent eval` runs each evaluation. It requires a dollar cap on paid commands, and asks before any test-set run.
- **A persistent search cache, for evaluation only.** Repeated evaluation runs replay searches instead of recomputing them on the CPU. The pipeline does not use it.

## Known limitations

- **An explanation can be pruned for one wrong extra expectation.** Agreeing with the metric being explained earns an explanation nothing, so that it cannot survive on the change alone. If the only other metric it expects contradicts the filings, its numerical consistency is 0. The depth-1 pre-screen then prunes it, at a score of 0.25 against a threshold of 0.30.
  - On HSY's 10-K comparison, "lower sales volume compressed gross margin" was pruned this way. It also expected revenue to fall, but revenue rose on price. Lower volume still reached that brief, as a confirmed cause of the operating margin change.
  - In the cached HSY runs this happened to 5 of 90 numeric explanations, all in the 10-K comparison. The other four expected SG&A's share of revenue to rise, which it did not.
  - Counting the agreement whenever another metric is checked would score such an explanation 0.625, as the text form of a change already does. Because the problem is rare, this was left unchanged.

## Still open

- **Rewording.** It stays off until it is measured against the labeled findings.
- **Cost settings.** The per-change cost estimate (`PER_CHANGE_USD`) predates the wider search, and will be updated from the cost measurements.
- **Parser rejections.** The gate still rejects some sound filings from very small companies (see the README's known limits).
