# Design changes

This note compares what was built with what was designed. The design is the six module documents in [design/](design/), written before and during the build:

| Module | Document | What it set |
|---|---|---|
| 1 | [Project Scoping and Initial Agent Design](design/Module1_Project_Scoping_and_Initial_Agent_Design.pdf) | The problem, the user, four responsibilities, the scope |
| 2 | [Agent Design Refinement](design/Module2_Agent_Design_Refinement.pdf) | The four agents and how each reasons, memory, tools, the three labels and the causal-wording rule |
| 3 | [RAG Design Integration](design/Module3_RAG_Design_Integration.pdf) | Ingestion, chunking, hybrid search and reranking |
| 4 | [Tree-of-Thought Design](design/Module4_Tree-of-Thought_Design.pdf) | The search tree, its beam, scoring rubric, pruning and budgets |
| 5 | [Multi-Agent Architecture Plan](design/Module5_Multi-Agent_Architecture_Plan.pdf) | The agents' models, the controller and its loops, communication, trade-offs |
| 6 | [Safety and Intervention Plan](design/Module6_Safety_and_Intervention_Plan.pdf) | Guardrails, evaluation metrics, when a human steps in |

This note covers:
1. the 14 gaps found in Modules 1 to 4 before building: what the design said, how each was resolved, and what was built;
2. where the build departs from the design, and why;
3. what was built that the design did not ask for;
4. known limitations and future enhancements.

Evaluation results are in [EVALUATION.md](EVALUATION.md). Code comments cite the gaps as "design gap N", and build tasks by ID, such as "T4.4", which [TASKS.md](TASKS.md) lists. "HSY" is Hershey, the company settings were tuned on. Mattel (MAT) and Stanley Black & Decker (SWK) are the test companies.

## The system as built

The Module 5 plan's structure stands: four agents, no planner, and a deterministic LangGraph controller that does the routing, retry limits and budget stops in code (`src/sec_agent/graph.py`).

| Agent | Where | Models |
|---|---|---|
| Retrieval | `agents/retrieval.py`, `retrieval/` | None by default (see departure 2) |
| Change Analysis | `compare/` (alignment, materiality), `tot/` (Tree-of-Thought) | Opus for materiality; Haiku for the generator and critic |
| Financial Analysis | `finance/` (fact store, metric catalog, scan) | None: fixed calculations in code |
| Verification and Reporting | `agents/verification.py`, `tot/labeling.py`, `report/` | Code checks; Opus for the judge and the brief |

The workflow runs these steps: detect, estimate, rank, select, explore (with rework), summarize, write. Agents pass typed Pydantic records through the graph's state. Every Tree-of-Thought node, score and pruned branch is kept in a local MCP tree store (`tree_mcp/`). [src/sec_agent/README.md](../src/sec_agent/README.md) has a diagram and maps each step to its code.

## How the 14 design gaps were filled

Before any code was written, Modules 1 to 4 were read for questions the build would have to answer. Fourteen places were silent, contradicted each other, or would not work as written. Each was given a resolution before building, and Modules 5 and 6 adopted several of them. For each gap below:
- **Design:** what the module documents said, or left out;
- **Resolution:** what was decided before building;
- **Built:** what the code does, and anything measurement changed.

**1. A scoring bug pruned every text-only explanation** (Module 4).
- **Design:** the depth-1 pre-screen scores a candidate on numerical consistency (weight 0.75) and specificity (0.25), and prunes below 0.30. A candidate with no metric to check, such as one explaining a new risk factor, scores at most 0.25, so every text-only explanation would be pruned.
- **Resolution:** numerical consistency gets a third state, "not applicable", which drops the criterion and rescales the other weights.
- **Built:** as resolved. The HSY evaluation then found a second form of the bug. An explanation was pruned when one extra metric it expected to move contradicted the filings, even though the metric being explained agreed. Two changes followed:
  - Numeric changes are now gated only on the changed metric, or a metric linked to it by arithmetic. Text changes are not gated.
  - Numerical consistency is now the share of checked metrics that agree.

**2. The agent order conflicted** (Modules 2 and 3).
- **Design:** the workflow diagrams put Financial Analysis after Change Analysis, but Module 3's worked example starts with Financial Analysis computing a margin drop.
- **Resolution:** Financial Analysis runs twice: first as a metric scan that flags numeric changes, then as a checking tool inside the Tree-of-Thought search. Module 5 adopted this.
- **Built:** as resolved (`finance/scan.py`, then `MetricCatalog` inside `tot/`).

**3. How changes are detected was undefined** (Modules 2 and 3).
- **Design:** the Change Analysis Agent "compares the aligned section text from each period directly", with no method for aligning the text. Top-5 retrieval finds passages on a topic, not what changed.
- **Resolution:** align paragraphs section by section by embedding similarity with an assignment solver, then diff the pairs. Only added, removed and modified paragraphs go to a model.
- **Built:** as resolved (`compare/align.py`):
  - Identical paragraphs pair off first.
  - The rest are matched with `linear_sum_assignment` on cosine similarity, with a floor of 0.75.
  - A matched pair is "unchanged" only when its differences are the ones a filing makes when text is carried forward, such as years moving on. A text-similarity cutoff was tried and rejected, because real edits ("two" becoming "four", a new chief executive's name) scored 98 out of 100.
  - A first-quarter 10-Q paragraph is kept from pairing with a later quarter's year-to-date paragraph.

**4. "Deterministic" verification could not judge a paraphrase** (Modules 2 and 4).
- **Design:** verification stays deterministic, and a citation checker confirms that "a quoted or paraphrased claim actually appears in the cited passage". A claim is `confirmed` when a passage "directly states the cause". Code cannot decide either of these for a paraphrase.
- **Resolution:** every claim carries a verbatim quote. Code checks the quote and its period, company and section. A separate fresh-context judge then classifies the quote as stating the cause, implying it, or unrelated. `confirmed` requires both. Module 5 adopted the judge.
- **Built:** as resolved, with Module 5's stricter version:
  - The judge sees only the claim and its quote, never the reasoning behind them.
  - The code checks (`agents/verification.py`) reject altered quotes, quotes from the wrong period, company or section, and figures that cannot be traced to a metric.
  - `confirmed` requires both the code checks and a "states cause" verdict.

**5. The search budget had no slack** (Module 4).
- **Design:** up to 8 candidates, then up to 8 evidence sources for each of the 4 kept: 8 + 4 × 8 = 40 nodes, the whole budget of about 40, leaving nothing for retries or the fresh round.
- **Resolution:** the low end of the design's range: 6 candidates and 6 sources each (30 nodes), leaving 10 of the 40 for retries.
- **Built:** changed after measurement; see departure 1.

**6. Numeric materiality had no thresholds** (Module 2).
- **Design:** a change is material if it meets one of three criteria, one being "a changed quantity", with no size given.
- **Resolution:** margins moving 100 basis points or more; revenue or a major expense line moving 10% or more.
- **Built:** kept after checking them against the HSY labels (`finance/scan.py`).

**7. Sequential quarters are seasonal** (Module 1).
- **Design:** filings are compared across reporting periods, with nothing on seasonality, which matters for Hershey in particular.
- **Resolution:** compare each sequential move with the same move a year earlier, using XBRL history, and label it possibly seasonal.
- **Built:** as resolved, in the metric scan. The benchmark also found that 10-Qs state no causes for sequential moves; their explanations compare with the year before. The evaluation scores sequential causes with this in mind.

**8. 10-K and 10-Q sections differ** (Module 3).
- **Design:** filings are parsed by 10-K Item (1A Risk Factors, 7 MD&A, 7A). A 10-Q numbers them differently (MD&A is Part I Item 2), and often reports only updates to risk factors.
- **Resolution:** map both forms to canonical section names, and store a 10-Q's "no material changes" statement as a result, not a parsing failure.
- **Built:** as resolved (`ingest/sections.py`).

**9. Segment data looked unavailable** (Module 4).
- **Design:** the Tree-of-Thought example has a "Segment revenue (XBRL)" evidence branch, but the SEC's aggregate XBRL APIs appeared to cover company-wide facts only.
- **Resolution:** a time-boxed check: use inline XBRL if it works, otherwise read the segment notes as text.
- **Built:** inline XBRL works: segment values load with the segment axis.

**10. Some metrics are rarely tagged** (Modules 2 and 4).
- **Design:** the example's "Revenue per unit (XBRL)" is rarely a standard tag, and Module 2 allowed "code execution or structured XBRL data" for calculations.
- **Resolution:** a fixed metric catalog limited to reliably available data, with no arbitrary code execution. Unit-level evidence comes from MD&A text.
- **Built:** as resolved (`finance/metrics.py`, `config/metrics.yaml`): margins, growth, cost ratios, segment values and dollar metrics.

**11. Retry and spending limits were unnamed** (Modules 1, 2 and 4).
- **Design:** a "retry limit", "about 40 node evaluations" per finding and "a token budget" per run, without values for the first and last.
- **Resolution:** 2 verification retries per claim, a $10 cap per run, and 40 nodes per finding. Module 5 adopted these.
- **Built:**
  - The $10 cap is the default (`SEC_AGENT_MAX_USD_PER_RUN`).
  - Rework allows at most 2 rounds per finding (see departure 3).
  - Node budgets changed (see departure 1).

**12. Human escalation was unspecified** (Modules 1 and 2).
- **Design:** unresolved items are "flagged for human review", with no say on how, or whether the run waits.
- **Resolution:** non-blocking: unresolved items go to the brief and a Streamlit review queue, and the run never pauses. Modules 5 and 6 adopted this; Module 6 adds what sends an item to review.
- **Built:**
  - The review queue is the brief's "Needs review" section, with the details in `audit.json`.
  - The Streamlit app (`sec-agent app`) has a fuller review queue, read from `audit.json`. It also lists explanations that labeling left out or annotated, with their evidence, searches that never finished, dropped rationales, and the changes past the cap.
  - The app shows saved runs only. Live runs from the app (planned as T7.4: start `analyze` from the app, show its progress, and confirm the cost estimate there) are a future enhancement. A paid run is started and confirmed in the terminal.

**13. The causal-language rule had no enforcement** (Module 2).
- **Design:** "Causal language is allowed only for confirmed findings", with nothing to check it.
- **Resolution:** a deterministic check that fails phrases such as "due to" or "driven by" unless the finding is `confirmed`.
- **Built:** as resolved (`find_causal_language`). It runs when claims are labeled, and again on the whole brief, which is refused if it breaks the rule. Each phrase in the list carries the reason it is there.

**14. The reranker and vector store were unnamed** (Module 3).
- **Design:** "a cross-encoder reranker (run locally)" and "an approximate nearest-neighbor index", without naming either.
- **Resolution:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, fast on a CPU, and Chroma, with DuckDB as the source of truth for chunks.
- **Built:** as resolved, with these changes:
  - Search is hybrid: BM25 plus dense search, combined by reciprocal rank fusion, then reranked, as Module 3 planned.
  - The rerank pool was cut from 20 to 10 to save CPU time. On HSY, recall did not change.
  - An ONNX version of the reranker was tried, to speed it up, and rejected: on the build machine (a 4-core CPU with no GPU) every variant was slower than the standard PyTorch model, taking 1.2 to 1.6 times as long. The quantized variant also shifted scores enough to change the top 5 in 17 of 62 searches.
  - MMR was measured and not added: no top-5 list exceeded the trigger Module 3 set.

## Departures from the design

**1. A wider search, with a larger node budget.** Module 4 set about 40 nodes per finding, which allows a beam of 4 explanations.

On HSY, the depth-1 pre-screen gave a perfect score to more candidates than the beam held, in all six changes. Ties fall back to the order the generator wrote them in, so the beam of 4 cut labeled causes that had passed the threshold. Module 4 had planned for this: widen the beam if too few known causes survive.

Now:
- the beam is 6, so every candidate past the 0.30 threshold is kept (`config/scoring.yaml`);
- the first round may use 55 nodes, 42 of its own plus 13 for retries;
- a fresh round, if needed, may use 45.

On HSY, causes kept rose from 5 to 6 of 8 (numeric) and from 7 to 8 of 8 (text); a recheck after the review fixes gave the same 6 and 8. The cost is more surviving distractors and about 40% more searches per change. The dollar cap still bounds each run.

**2. The Retrieval Agent's rewording is off by default.** Modules 2 and 5 have a bounded ReAct loop on Haiku (at most 3 rounds) that rewords a query when one filing comes back empty. It is built and tested (`agents/retrieval.py`), but switched off, so retrieval is one plain search.

In a live check, rewording filled two of five empty filings with passages about another topic. The citation checker confirms a quote's text, period and section, but not its relevance. An off-topic quote could therefore support a `plausible` claim, which is worse than an empty filing. The search controller has its own retry, across every section.

**3. Rework targets what the judge rejected.** Module 5 says failed verification returns claims for rework, at most 2 retries per claim.

As built:
- A claim the judge finds only implied gets another round of searching and is relabeled.
- There are at most 2 rounds per finding.
- A round stops if it would eat into the money reserved for labeling.
- A claim whose quote fails the code checks is not retried. It stays out of the brief.

**4. At most 10 changes are explored per run.** Modules 4 and 5 explore findings in materiality order until the budget runs out. The cap is set by wall time, not money: each explored change takes about 3.5 minutes on average and up to 8, so the run explores the 10 most material by default (`--max-changes`). The brief says how many were left, and `audit.json` lists them.

Most of a change's time goes to two things:
- **Model calls** take 71% of it, more than planned. A run of 10 changes sends 410 to 770 of them, and a typical call takes 1.0 to 4.1 seconds.
- **Local search** takes most of the rest. Embedding and cross-encoder reranking run on a 4-core CPU with no GPU, and would be far faster on a GPU.

The evaluation measured the trade-off. A run of 10 changes costs $0.44 to $1.41 and takes 23 to 44 minutes on a 4-core CPU. In exchange, the agent covers fewer of the labeled material changes than the prompt-only baseline: 7 of 16 on the test split, against 13. The ones it missed were all found material but ranked below its 10 (see [EVALUATION.md](EVALUATION.md), T6.5).

**5. Two budget checks before spending.** Modules 4 and 5 have a dollar cap that drops the least material findings. Two checks were added, so money is not spent on work the cap cannot finish:
- Before materiality, the run needs enough free budget for the whole step, or it ends having spent nothing.
- Before each search, it needs enough left to label the result and write the summary.

**6. A stopped run can resume.** Not in the design. The graph saves its state after every step. `--resume` continues a killed run from its last finished change, and paid calls already made replay from a local cache at no cost. This was tested by killing a paid run mid-search.

**7. Further guards on `confirmed`.** The evaluation found ways a claim could be confirmed without explaining the change. Each now has a code guard (`tot/labeling.py`):
- A quote covering six months cannot confirm a claim about a quarter.
- A claim that names the change only inside a denial ("not because gross margin...") is rejected.
- A claim about a text change must fit the changed passage.
- The generator is shown the metric's earlier value and its direction. Without them, every candidate for one Mattel change said gross margin rose when it fell.
- When the judge says "implies", the claim is restated at the level its quote supports, then judged again.
- One addition works the other way, recovering a confirmation the searches missed. A ratio claim its own quotes leave unconfirmed is tried once more, on the current filing's MD&A paragraph that names the ratio, gives both of its values and states a cause. In one Hershey run, the searches had retrieved only the six-month paragraphs, so gross margin had no confirmed cause although the 10-Q states one.
- A `plausible` claim cannot cite a quote that covers a longer span than the change. A claim left with no quotes stays out of the brief.

`confirmed` does not also require a second, independent branch to find the same cause. Each confirmed claim already rests on a verbatim quote from the right filing, period and section, a fresh-context judge's verdict that the quote states the cause, and the direction check against the metric. Requiring a corroborating branch would mostly drop causes a filing states once. In the evaluation on the fixed code, no distractor was asserted as fact on either split, and none of the agent's asserted causes on the test split was unsupported ([EVALUATION.md](EVALUATION.md), T6.5).

**8. Claude through the official SDK.** Module 4 runs the generator and critic as LangChain chains, which would call Claude through `langchain-anthropic`. Claude is called through the `anthropic` SDK instead, wrapped in a LangChain `Runnable`, because current models reject forced tool choice and sampling parameters. The wrapper (`llm/client.py`) adds:
- structured output, through `messages.parse`;
- prompt caching;
- refusal handling;
- a cost ledger with a hard budget stop;
- a replay cache.

A reply cut off mid-JSON is charged its worst case, and only that branch ends.

**9. A pinned, offline corpus.** EDGAR is read once, for 15 pinned filings (`config/corpus.yaml`). The agents query local data at run time. Module 3 allows this for a prototype.

## Built beyond the design

- **A validation gate for parsing.** Every filing is parsed, audited against evidence the parser did not use, and retried with a fallback parser if it fails. Rejected filings can never reach the index. The gate was measured on four sets of randomly drawn companies ([eval/ingestion/parser_holdout.md](../eval/ingestion/parser_holdout.md)).
- **Table facts.** Figures in untagged tables are loaded as searchable facts when the whole table can be explained. Other tables are indexed as text.
- **Cost and latency logs.** Every model call records its step, change and latency, and every step and local search is timed (`timing.py`, `report/run_cost.py`).
- **An evaluation CLI.** `sec-agent eval` runs each evaluation. It requires a dollar cap on paid commands, and asks before any test-set run.
- **A persistent search cache, for evaluation only.** Repeated evaluation runs replay searches instead of recomputing them on the CPU. The pipeline does not use it.
- **A replay view of each run.** The Streamlit app's Replay audit tab rebuilds a saved run step by step from its timings, ledger and audit trail.

## Known limitations

- **An explanation can be pruned for one wrong extra expectation.** Agreeing with the metric being explained earns an explanation nothing, so that it cannot survive on the change alone. If the only other metric it expects contradicts the filings, its numerical consistency is 0. The depth-1 pre-screen then prunes it, at a score of 0.25 against a threshold of 0.30.
  - On HSY's 10-K comparison, "lower sales volume compressed gross margin" was pruned this way. It also expected revenue to fall, but revenue rose on price. Lower volume still reached that brief, as a confirmed cause of the operating margin change.
  - In the HSY runs on the fixed code this happened to 4 of 54 numeric explanations, 3 of them in the 10-K comparison. Two of the others expected SG&A's share of revenue to rise, which it did not; the fourth, in the sequential comparison, expected revenue growth to rise.
  - Counting the agreement whenever another metric is checked would score such an explanation 0.625, as the text form of a change already does. Because the problem is rare, this was left unchanged.

- **Sequential causes are not flagged as year over year.** The 10-Qs explain each move only against the same quarter a year earlier. In the sequential comparisons, the agent offers those causes for the quarter-to-quarter move and almost never says they are year over year (1 of 26 asserted causes across both splits). Its quotes cite the right filing and period, so the problem is the wording, which a reader can misread.
- **Some claims are weak or unspecific.** In the blinded hand check, the reviewer found some supported claims weak or vague. The prompts were not changed for this; better claim prompting could improve it.
- **Parser rejections.** The gate still rejects some sound filings from very small companies (see the known limits in [src/sec_agent/ingest/](../src/sec_agent/ingest/README.md#known-limits)).
- **A missed search and missing evidence end the same way.** With rewording off (departure 2), a search that finds nothing is not retried in new words. An explanation whose evidence the search missed goes to review as unsupported, just like one the filing does not support.
- **Some disclosures are untested.** The benchmark labels changes in the notes to the financial statements (an acquisition, a divestiture, litigation, credit facilities, a share repurchase), and materiality ranked them material. It labels no change in an accounting assumption, such as an estimate or a discount rate, and none that appears only in a table.
- **Amended filings are not handled.** If a company files a 10-K/A or 10-Q/A that corrects figures, nothing marks the original figures as superseded or flags briefs built on them. The pinned corpus holds no amendments, so the evaluation never meets one.

## Future enhancements

- **Latency.** Cutting it would let a run explore more changes in the same time. Fewer and faster model calls come first, since they are about 70% of a change's time; then GPU support for local search. See departure 4.
- **A wider search.** Test a wider beam, and more candidates and evidence sources per candidate. The current limits (6 candidates, all 6 kept, 6 sources each and 3 kept, node budgets of 55 and 45) are set to fit the time budget, not because a wider search was measured and found no better.
- **Other models for each role.** Test other models to trade off latency, cost and quality, role by role. Only one such test was run: Sonnet against Haiku as the search model, on HSY (see [EVALUATION.md](EVALUATION.md)). The Opus roles (materiality, the judge, the brief) were not tested with cheaper or faster models.
- **Models outside Claude.** Each role's model is a setting, but the client (`llm/client.py`) calls the Anthropic SDK only. A provider-neutral client would let any model serve each role, including other vendors' models and models run locally.
- **Tests on tables and accounting assumptions.** Label changes in critical accounting estimates and in complex tables (debt maturities, segment results), and measure detection and support on them.
- **Amended filings.** Detect 10-K/A and 10-Q/A filings when filings are fetched, mark the figures they supersede, and flag saved briefs that relied on them.
- **Rewording.** It stays off until it is measured against the labeled findings.
- **Live runs from the app.** The Streamlit app shows saved runs only, and runs start from `sec-agent analyze` (see gap 12).
