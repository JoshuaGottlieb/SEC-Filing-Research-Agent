# Documents

This folder holds the project's design documents and the notes written during and after the build. The [main README](../README.md) is the overview; start there to run or review the project.

## The design (written before the build)

Six design documents, one per course module, written as the design developed. The build follows Module 5's architecture. [DESIGN_CHANGES.md](DESIGN_CHANGES.md) says where the build filled gaps in these documents and where it departs from them.

| Module | Document | What it contains |
|---|---|---|
| 1 | [Project Scoping and Initial Agent Design](design/Module1_Project_Scoping_and_Initial_Agent_Design.pdf) | The problem and the user (a fundamental analyst), why the task needs an agent, the four responsibilities (retrieval, change analysis, financial analysis, verification and reporting), success criteria, and the scope: three companies, 10-K and 10-Q filings, no investment advice |
| 2 | [Agent Design Refinement](design/Module2_Agent_Design_Refinement.pdf) | The four agents and how each reasons (a ReAct loop for retrieval, fixed tool calls for financial analysis and verification), working and long-term memory, the three tools (EDGAR access with vector search, a calculator, a citation checker), the labels `confirmed`, `plausible` and `unresolved`, and the rule that only confirmed findings use causal wording |
| 3 | [RAG Design Integration](design/Module3_RAG_Design_Integration.pdf) | Why retrieval is required; ingestion by Item section; 300 to 500 token chunks with full metadata; figures from XBRL in DuckDB rather than embedded tables; hybrid BM25 and dense search fused by reciprocal rank, reranked by a cross-encoder, top 5 per period with a score floor; the wrong-period failure mode; the recall@5 evaluation and the condition for adding MMR |
| 4 | [Tree-of-Thought Design](design/Module4_Tree-of-Thought_Design.pdf) | Tree-of-Thought inside the Change Analysis Agent only: a depth-2 tree of candidate explanations and evidence sources, beam search (4 candidates, then 3 evidence branches each), the weighted scoring rubric and depth-1 pre-screen, the 0.30 threshold and hard gates, tie rules, the 40-node budget, and the tools behind each role (LangChain chains, a LangGraph controller, an MCP tree store) |
| 5 | [Multi-Agent Architecture Plan](design/Module5_Multi-Agent_Architecture_Plan.pdf) | Why four agents and no planner; each agent's method and model (Haiku or Opus); a LangGraph controller with a sequential backbone and two bounded loops; typed records in shared state; the fresh-context judge; the trade-offs; and the commitment to measure the agent against a prompt-only baseline |
| 6 | [Safety and Intervention Plan](design/Module6_Safety_and_Intervention_Plan.pdf) | The risks (wrong-period evidence, causal overreach, prompt injection, missed changes, runaway cost); the guardrails for input, tools, verification, output and monitoring; the evaluation metrics; when a finding goes to a person; and the trade-offs between autonomy, coverage and cost |

## After the build

| Document | What it contains |
|---|---|
| [DESIGN_CHANGES.md](DESIGN_CHANGES.md) | How the build filled the 14 gaps found in the design, where it departs from the design and why, what was built beyond it, known limitations and future enhancements |
| [EVALUATION.md](EVALUATION.md) | The benchmark, the fair-test rule, and every result: retrieval, materiality, the judge, Tree-of-Thought survival, the agent against a prompt-only baseline, the blinded hand check, and cost and latency |
| [TASKS.md](TASKS.md) | The index of build tasks that code comments and documents cite by ID, such as "T4.4": what each built and where its code is |
| [How SEC Filing Ingestion Is Tested](How_SEC_Filing_Ingestion_Is_Tested.md) | A plain-language guide to the four layers of tests that keep parsing errors out of the index |
| [images/architecture.svg](images/architecture.svg) | The architecture diagram: the controller's steps, the four agents and the local stores |

## Suggested reading order

1. The [main README](../README.md), for what the agent does and how to review it.
2. Modules 1 and 5, for the problem and the architecture as designed.
3. [DESIGN_CHANGES.md](DESIGN_CHANGES.md), for what changed in the build.
4. [EVALUATION.md](EVALUATION.md), for how well it works.
5. [TASKS.md](TASKS.md) and [src/sec_agent/README.md](../src/sec_agent/README.md), to find the code behind any part.
