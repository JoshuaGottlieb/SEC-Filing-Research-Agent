# Build tasks

The code, tests and evaluation files cite build tasks by ID, for example "T4.4" or "T6.5". This index says what each task built and where its code lives. Paths under `src/sec_agent/` are written without that prefix.

The tasks implement the design in [design/](design/), chiefly the Module 3 to 5 documents ([README.md](README.md) summarizes each). [DESIGN_CHANGES.md](DESIGN_CHANGES.md) says how the build fills the design's 14 gaps; code comments cite those as "design gap N". [EVALUATION.md](EVALUATION.md) has the results of Phase 6.

## Phase 0: Foundation

| Task | What it built | Where |
|---|---|---|
| T0.2 | Project layout, settings and setup checks | `pyproject.toml`, `config.py`, `cli.py` (`doctor`) |
| T0.3 | The shared Pydantic records passed between agents | `schemas.py`, `contracts.py` |
| T0.4 | LLM client: a model per role, structured output, prompt caching, a cost ledger, the budget stop and a replay cache | `llm/` |

## Phase 1: Data

| Task | What it built | Where |
|---|---|---|
| T1.1 | EDGAR client (rate limit, retries, declared User-Agent) and the corpus pinned by accession number | `edgar/client.py`, `edgar/corpus.py`, `config/corpus.yaml` |
| T1.2 | Section parser mapping 10-K and 10-Q items to canonical names, with a fallback parser; tables kept apart from narrative text | `ingest/sections.py`, `ingest/fallback.py`, `ingest/tables.py`, `ingest/audit.py`, `eval/ingestion/` |
| T1.3 | Chunker: 300 to 500 tokens, 10% overlap, no chunk crossing an Item, full metadata | `ingest/chunker.py`, `ingest/pipeline.py` |
| T1.4 | XBRL facts loaded into DuckDB, with tag aliases, quarter and year-to-date durations and each company's fiscal calendar; revenue and gross profit checked against the filed statements | `edgar/xbrl.py`, `finance/fact_store.py`, `finance/statements.py` |
| T1.5 | Segment values from inline XBRL (design gap 9) | `edgar/xbrl.py`, `finance/fact_store.py` |
| T1.6 | Metric catalog: margins, growth, cost ratios, segment and dollar metrics, and the direction check (agree, contradict, not applicable) | `finance/metrics.py`, `config/metrics.yaml` |

## Phase 2: Retrieval

| Task | What it built | Where |
|---|---|---|
| T2.1 | Chunk embeddings in a persistent Chroma collection, filtered by company, form, period and section | `ingest/index.py` |
| T2.2 | Hybrid search: BM25 plus dense search, combined by reciprocal rank fusion | `retrieval/hybrid.py` |
| T2.3 | Cross-encoder reranking with a minimum score, and an `incomplete` flag when one period has no match | `retrieval/rerank.py` |
| T2.4 | Redundancy measurement on the benchmark, to decide on MMR (not added) | `retrieval/redundancy.py`, `eval/retrieval/redundancy.py` |
| T2.5 | Retrieval tools with a per-query cache | `retrieval/tools.py`, `retrieval/search_cache.py` |

## Phase 3: Change detection

| Task | What it built | Where |
|---|---|---|
| T3.1 | Paragraph alignment and diff per section; each pair labeled added, removed, modified or unchanged (design gap 3) | `compare/align.py`, `compare/embedding_cache.py` |
| T3.2 | Metric scan for numeric changes, with the seasonality comparison for sequential quarters (design gaps 6 and 7) | `finance/scan.py` |
| T3.3 | Materiality ranking, producing the ranked change list (`RankedChanges`) | `compare/materiality.py` |
| T3.4 | Retrieval Agent: a bounded loop (at most 3 rounds) that reformulates a query when a period comes back empty | `agents/retrieval.py` |

## Phase 4: Tree-of-Thought

| Task | What it built | Where |
|---|---|---|
| T4.1 | Tree store: a local MCP server over SQLite that keeps every node, score and pruned branch | `tree_mcp/` |
| T4.2 | Thought generator: candidate explanations at depth 1, evidence sources at depth 2 | `tot/generator.py` |
| T4.3 | Critic and scoring: the rubric, the depth-1 pre-screen and the three-state numeric criterion (design gap 1) | `tot/critic.py`, `tot/scoring.py`, `config/scoring.yaml` |
| T4.4 | Deterministic controller: beam widths, hard gates, one reformulation retry, the score threshold, tie rules, co-drivers, one fresh round and the node budgets | `tot/controller.py` |
| T4.5 | Labeling of each surviving explanation as confirmed, plausible or unresolved (design gap 4) | `tot/labeling.py` |

## Phase 5: Verification, workflow, CLI

| Task | What it built | Where |
|---|---|---|
| T5.1 | Citation checker: normalized quote match, metadata match, and figures traced to a metric | `agents/verification.py` |
| T5.2 | Entailment judge: a fresh-context model call that decides whether a quote states the cause | `agents/verification.py`, `eval/judge/` |
| T5.3 | Causal-language check: only a confirmed claim may say why something happened (design gap 13) | `agents/verification.py` |
| T5.4 | Brief writer: the Markdown brief and its JSON audit trail, findings grouped by label, the review section | `report/brief.py`, `report/summary.py`, `report/templates/` |
| T5.5 | LangGraph workflow: state, retry edges, materiality-ordered processing, the run budget, checkpoint and resume | `graph.py`, `tot/rework.py` |
| T5.6 | Command line: `ingest`, `index`, `analyze`, `app` and `eval` | `cli.py`, `eval_cli.py` |

## Phase 6: Evaluation

| Task | What it measured | Where |
|---|---|---|
| T6.1 | The benchmark: retrieval questions, material changes and stated causes, drafted apart from the agent and confirmed by hand | `eval/benchmark/`, `benchmark.py` |
| T6.2 | Retrieval recall@5, with and without the reranker | `eval/retrieval/recall.py` |
| T6.3 | Materiality: missed changes and false positives | `eval/materiality/` |
| T6.4 | Tree-of-Thought survival: how often the labeled cause stays in the beam | `eval/tot/` |
| T6.5 | A prompt-only baseline against the agent: unsupported claims, wrong-period citations, citation validity | `eval/baseline/` |
| T6.6 | Cost and latency per finding, and Haiku against Sonnet as the worker model | `eval/cost/`, `timing.py`, `report/run_cost.py` |

## Phase 7: Streamlit app

| Task | What it built | Where |
|---|---|---|
| T7.1 | Brief view, with each claim's quote, passage and SEC link | `ui/streamlit_app.py`, `report/run_view.py` |
| T7.2 | Evidence tree view, with scores, status and pruned branches | `ui/streamlit_app.py`, `report/run_view.py` |
| T7.3 | Review queue: unresolved items and their evidence | `ui/streamlit_app.py`, `report/run_view.py` |
| T7.4 | Live runs started from the app. Not built: a future enhancement (design gap 12) | |

The step-by-step replay and the cost and time tab were added beyond the plan (`ui/replay_page.py`, `report/replay.py`).

## Phase 8: Delivery

| Task | What it produced | Where |
|---|---|---|
| T8.1 | The full-corpus run: 9 comparisons, three companies by three comparison types | `samples/` |
| T8.2 | README, design changes note and evaluation summary | `README.md`, `docs/DESIGN_CHANGES.md`, `docs/EVALUATION.md` |
| T8.8 | Repository ready for outside review: sample runs, link check, license | `samples/`, `LICENSE` |
| T8.10 | Cleanup, documentation and code review | |
