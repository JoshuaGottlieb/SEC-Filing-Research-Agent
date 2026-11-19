# The agent's code

This folder is the `sec_agent` package: the agents, the workflow that runs them, the command line and the Streamlit app. The [main README](../../README.md) gives the overview. This page shows how a run moves through the code and where each part lives.

Each package's `__init__.py` describes its modules. Code comments cite build tasks by ID, such as "T4.4", which [TASKS.md](../../TASKS.md) explains. They also cite the design's gaps as "design gap N", which [DESIGN_CHANGES.md](../../DESIGN_CHANGES.md) explains.

## The workflow

`graph.py` is a fixed LangGraph workflow. It is the deterministic controller of the design: code, not a model, decides the route, the retries, the budget checks and when to stop. There is no planner agent.

```
detect → estimate → rank → select ─┬─→ explore ⇄ rework ─┬─→ summarize → write
                                   └──────────────────────┘
```

| Step | What it does | Code |
|---|---|---|
| `detect` | Aligns the two filings' paragraphs section by section and scans the XBRL metrics for moves, with the seasonality check for sequential quarters. No model is used. | `compare/align.py`, `finance/scan.py` |
| `estimate` | Prints a cost estimate, asks before spending, and stops before sending anything if the budget can't cover the materiality step. | `graph.py` |
| `rank` | Opus judges which changes are material and why, and code ranks them. | `compare/materiality.py` |
| `select` | Takes the most material changes, up to `--max-changes`; the rest are recorded as not examined. Before each search, the run checks it has enough left to label the result and write the summary. | `graph.py` |
| `explore` | Runs the Tree-of-Thought search on one change, then labels its claims. | `tot/`, `retrieval/`, `agents/verification.py` |
| `rework` | Gives claims the judge found only implied another round of searching, at most 2 rounds per change. | `tot/rework.py` |
| `summarize` | Opus writes the brief's summary, which code checks; a template is the fallback. | `report/summary.py` |
| `write` | Writes the brief and its audit trail, refusing a brief that breaks the causal-language rule. | `report/brief.py` |

The graph saves its state after every step, so `sec-agent analyze --resume RUN_ID` continues a stopped run from its last finished change.

## The four agents

| Agent | Where | Models |
|---|---|---|
| Retrieval | `agents/retrieval.py`, `retrieval/` | None by default: the rewording loop is built but off (DESIGN_CHANGES.md, departure 2) |
| Change Analysis | `compare/` (alignment, materiality), `tot/` (Tree-of-Thought) | Opus for materiality; Haiku for the generator and critic |
| Financial Analysis | `finance/` (fact store, metric catalog, scan) | None: fixed calculations in code |
| Verification and Reporting | `agents/verification.py`, `tot/labeling.py`, `report/` | Code checks; Opus for the judge and the brief |

Agents pass typed Pydantic records through the graph's state (`schemas.py`). The interfaces between them are protocols in `contracts.py`, so tests can swap in fakes.

## Where the code is

| Step or part | Agent | Code |
|---|---|---|
| Filings, sections, chunks and facts (run once by `ingest` and `index`) | | `edgar/`, [`ingest/`](ingest/README.md), `finance/fact_store.py` |
| Detect: paragraph alignment and diff | Change Analysis | `compare/align.py` |
| Detect: metric scan and seasonality | Financial Analysis | `finance/scan.py`, `finance/metrics.py` |
| Rank: materiality | Change Analysis | `compare/materiality.py` |
| Explore: Tree-of-Thought search | Change Analysis | `tot/generator.py`, `tot/critic.py`, `tot/scoring.py`, `tot/controller.py` |
| Explore: evidence search | Retrieval | `retrieval/`, `agents/retrieval.py` |
| Explore: metric checks inside the search | Financial Analysis | `finance/metrics.py` |
| Explore: labeling and rework | Verification and Reporting | `tot/labeling.py`, `tot/rework.py`, `agents/verification.py` |
| Write: summary and brief | Verification and Reporting | `report/brief.py`, `report/summary.py` |
| Tree store (every node, score and pruned branch) | | `tree_mcp/` |
| Model calls, cost ledger, budget stop, replay cache | | `llm/` |
| Shared records passed between agents | | `schemas.py`, `contracts.py` |
| Settings, read from `.env` | | `config.py` |
| Command line | | `cli.py`, `eval_cli.py` |
| Streamlit app over saved runs | | [`ui/`](ui/README.md), `report/run_view.py` |
| Evaluation scripts | | [`eval/`](../../eval/README.md), outside the package |

## What a run leaves on disk

Everything goes under the data directory (`SEC_AGENT_DATA_DIR`, by default `~/.local/share/sec-agent`):

| Path | Written by | Contents |
|---|---|---|
| `raw/`, `parsed/` | `ingest` | Downloaded filings, and each filing's parsed and validated record |
| `sec_agent.duckdb` | `ingest`, `index` | Chunks and figures: the source of truth for text and numbers |
| `chroma/` | `index` | The chunk embeddings behind dense search |
| `runs/<run id>/` | `analyze` | `brief.md`, `audit.json`, `ledger.jsonl` and `timings.jsonl` (see the [main README](../../README.md#running-an-analysis)) |
| `tot_tree.sqlite` | `analyze` | Every Tree-of-Thought node, pruned branches and their scores included. The workflow reaches it through a local MCP server. |
| `checkpoints.sqlite` | `analyze` | The graph's saved state, for `--resume` |
| `llm_cache.sqlite` | every paid command | Model replies by request, so an identical call replays at no cost |
