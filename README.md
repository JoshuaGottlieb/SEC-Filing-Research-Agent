# SEC Filing Research Agent

A multi-agent system that compares a company's 10-K and 10-Q filings across reporting periods, finds material changes, tests competing explanations with Tree-of-Thought search, and writes a cited research brief for a fundamental analyst.

It is a first-pass research aid. It does not give investment advice, recommendations, or price predictions. Every explanation in a brief carries a verbatim quote from a filing. Only explanations that pass every check are labeled `confirmed`.

## Problem and user

The user is a fundamental equity analyst who follows a company through its SEC filings. Each quarter they need to know what changed since the last filing, which changes matter, and what the company says caused them. Today that means reading two long filings side by side, section by section. A careful reader still misses a reworded risk factor or a cause stated three sections away from the number it explains.

A single prompt to a language model is fast but unreliable here. It can quote the wrong period, attribute a cause the filing never states, or skip changes it doesn't find interesting. This agent splits the job into steps that code can check:
- code finds every changed paragraph and every metric move;
- a model ranks them by materiality;
- Tree-of-Thought search tests competing explanations against retrieved passages;
- only an explanation whose quote states the cause, in the right filing and period, is labeled `confirmed`.

Weaker explanations are labeled `plausible`, and anything left open is queued for the analyst to check.

## Documents

- **[DESIGN_CHANGES.md](DESIGN_CHANGES.md):** how the build fills the gaps in the original design, and where it departs from it.
- **[EVALUATION.md](EVALUATION.md):** what was measured, on which data, and the results.
- **[TASKS.md](TASKS.md):** the build tasks that code comments cite by ID, such as "T4.4", and where each one's code is.
- **[samples/](samples/README.md):** nine real runs (three companies by three comparisons). Read the briefs on GitHub, or browse them in the Streamlit app with a light install: no ingestion, no models, no API key. Instructions are in [samples/README.md](samples/README.md).

## How it works

`sec-agent analyze HSY --comparison yoy-10k` runs one comparison through a fixed LangGraph workflow. Code, not a model, decides the route:

1. **Detect.** Paragraphs are aligned section by section to find added, removed and modified text. A scan of XBRL metrics finds numeric moves. Sequential-quarter moves are compared with the same move a year earlier, to flag seasonality. No model is used.
2. **Estimate.** A cost estimate is printed, and the run asks before spending anything.
3. **Rank.** Claude Opus ranks the changes by materiality. The most material ones (10 by default) are explored, in order.
4. **Explore.** For each change:
   - Claude Haiku proposes explanations.
   - Local search finds evidence in both filings.
   - A Haiku critic scores each explanation, and code prunes the tree.
   - A fresh-context Opus judge sees only each claim and its quote.
   - Claims the judge finds only implied get up to two more rounds of searching.
5. **Write.** Opus writes the summary. The brief is checked against its rules before it is saved.

Each claim is labeled:
- **`confirmed`:** the quote is verbatim, from the right filing, period and section, and states the cause.
- **`plausible`:** supported, but not stated outright.
- **`unresolved`:** queued for human review in the brief, without pausing the run.

Causal wording, such as "due to" or "driven by", is only allowed in `confirmed` claims.

### Where the code is

The design has four agents and no planner. `src/sec_agent/graph.py` is the deterministic controller: it runs the steps in order, and makes the retry, budget and stop decisions in code. Paths below are under `src/sec_agent/`, and each package's `__init__.py` describes its modules.

| Step or part | Agent | Code |
|---|---|---|
| Filings, sections, chunks and facts (run once by `ingest` and `index`) | | `edgar/`, `ingest/`, `finance/fact_store.py` |
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
| Command line | | `cli.py`, `eval_cli.py` |
| Streamlit app over saved runs | | `ui/`, `report/run_view.py` |
| Evaluation scripts | | `eval/` (see [Evaluation](#evaluation)) |

## Reviewing without running

Four ways to review the project, none of which needs an API key or ingestion:

1. **Read a sample brief.** [samples/](samples/README.md) holds nine real runs: three companies by three comparisons, each with its brief, full audit trail, model-call ledger and step timings. Start with [the Hershey 10-K brief](samples/runs/HSY-yoy-10k-t81-20261115/brief.md), which compares Hershey's FY2025 10-K with its FY2024 10-K. GitHub renders it directly.
2. **Browse the runs in the app.** The Streamlit app shows each run's brief, evidence trees, a step-by-step replay, the review queue and the cost and time. A light install of about 730 MB runs it on the samples in about 5 minutes; see [samples/README.md](samples/README.md#browse-the-runs-in-the-app-about-5-minutes-no-api-key).
3. **Check the evaluation.** [EVALUATION.md](EVALUATION.md) reports every result. The hand-checked benchmark is in [eval/benchmark/](eval/benchmark/README.md), the scripts are under [eval/](eval/), and the retrieval and judge result files are committed beside their scripts.
4. **Run the tests.** After the [setup](#setup) below (`uv sync --frozen` only), `uv run --frozen pytest` runs over 2,100 unit tests in about a minute. They need no network, no API key, no downloaded models and no ingested filings.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). No GPU or Docker is needed.

The first `index` and `analyze` download two local models from Hugging Face, about 0.5 GB in all:
- `BAAI/bge-base-en-v1.5` for embeddings;
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking.

```bash
pip install uv

uv sync --frozen
cp .env.example .env    # then fill in the two required values
uv run --frozen sec-agent doctor
```

uv creates the virtual environment as `.venv` inside the repo, which works on a normal local disk. The one exception is a repo on a VirtualBox shared folder, where symlinks and renames fail. There, keep the environment on local disk: run this before `uv sync` and add it to your shell profile.

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/sec-agent"
```

### Configuration

Two values in `.env` are required:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key from the [Claude Console](https://platform.claude.com/). Billed separately from Claude subscriptions. |
| `SEC_USER_AGENT` | Your name and contact email. [EDGAR requires this](https://www.sec.gov/about/developer-resources) from automated clients. |

The optional settings are in [.env.example](.env.example):

| Variable | Default | Purpose |
|---|---|---|
| `SEC_AGENT_DATA_DIR` | `~/.local/share/sec-agent` | Where filings, the database, the index and runs are stored |
| `SEC_AGENT_MODEL_STRONG` | `claude-opus-5-5` | Materiality, the judge, and the brief |
| `SEC_AGENT_MODEL_WORKER` | `claude-haiku-5-5` | The Tree-of-Thought generator and critic |
| `SEC_AGENT_MAX_USD_PER_RUN` | `10` | Hard stop on model spend per run |
| `SEC_AGENT_LLM_CACHE` | `on` | Replays an identical model call from a local cache, at no cost |

## Quick start

These three commands take a fresh setup to a first brief. Only the last one costs money.

```bash
uv run --frozen sec-agent ingest     # download and validate the 15 pinned filings; safe to re-run
uv run --frozen sec-agent index      # chunk them and build the search index; safe to re-run
uv run --frozen sec-agent analyze HSY --comparison yoy-10k --max-changes 3
```

`ingest` reads the pinned corpus in [config/corpus.yaml](config/corpus.yaml). It covers Hershey (HSY), Mattel (MAT) and Stanley Black & Decker (SWK), with five filings each:
- the latest two 10-Ks;
- the latest 10-Q;
- the same quarter a year earlier;
- the quarter before.

`ingest` exits 1 if any filing fails validation (see [How filings are validated](#how-filings-are-validated)).

`index` splits the validated filings into chunks of 300 to 500 tokens and embeds them into a local Chroma index. On the pinned corpus that is about 2,200 chunks, roughly 15 minutes on a 4-core CPU the first time. A re-run embeds only chunks that changed.

### Running an analysis

```bash
uv run --frozen sec-agent analyze TICKER --comparison yoy-10k|yoy-10q|seq-10q [options]
```

| Comparison | Filings compared |
|---|---|
| `yoy-10k` | The latest 10-K against the one before it |
| `yoy-10q` | The latest 10-Q against the same quarter a year earlier |
| `seq-10q` | The latest 10-Q against the quarter before it |

| Option | Effect |
|---|---|
| `--max-changes N` | Explore the N most material changes (default 10). The rest are listed in `audit.json`, not examined. |
| `--max-usd X` | Dollar cap for this run (default `SEC_AGENT_MAX_USD_PER_RUN`). A budget stop drops the least material changes first. |
| `--yes` | Print the estimate and go ahead without asking. |
| `--run-id ID` | Name the run (a fresh id by default). |
| `--resume ID` | Continue a stopped or killed run from its last finished change. Paid calls already made replay at no cost. |

**Cost and time.** The estimate printed before each run is the guide. Measured on 9 runs of 10 changes each, on a 4-core CPU:
- **Each change:** about $0.085 and 3.5 minutes, but up to $0.25 and 8 minutes. Three quarters of the time is waiting on model calls; the rest is local search.
- **A run with the default 10 changes:** $0.55 to $1.29 and 22 to 47 minutes.
- **The first run on a comparison** also pays for materiality: about $0.07 per batch of 20 changes, 6 to 22 batches per comparison. Later runs replay it from the cache at no cost.

The full measurements are in [EVALUATION.md](EVALUATION.md#cost-and-latency-t66-free-from-the-runs-above).

**Outputs** go to `<data dir>/runs/<run id>/`:

| File | Contents |
|---|---|
| `brief.md` | The research brief: a summary, then each finding with its labeled claims, quotes and figures, and the review queue |
| `audit.json` | Everything behind the brief: each claim's citations and check results, the judge's verdicts, and the changes not examined |
| `ledger.jsonl` | Every model call, with its tokens, cost, latency and the step it served |
| `timings.jsonl` | Wall time of every step and every local search |

Every Tree-of-Thought node is kept in `<data dir>/tot_tree.sqlite`, including pruned branches and their scores. The workflow reads and writes trees through a local MCP server.

### Browsing saved runs

```bash
uv run --frozen sec-agent app             # opens http://localhost:8501; --port to change it
```

A Streamlit app over the saved runs in `<data dir>/runs/`. It reads each run's `audit.json`, `ledger.jsonl` and `timings.jsonl` and calls no model, so browsing is free. Pick a company, a comparison and a run (the newest is the default). The app has five tabs:
- **Brief:** the summary and each change with its labeled claims. Filter by strongest label and by section. The changed text shows both versions side by side, with the words that differ marked. Expand a quote to see the full passage it came from, with the quote highlighted, and a link to the filing on SEC.gov.
- **Evidence trees:** one change's Tree-of-Thought tree, drawn top-down: the change, then one card per candidate explanation with its score and label. A card's evidence searches show as a row of coloured chips, one per search: the colour is what the search found, solid if kept, hollow if pruned. Open a candidate's branch to see each search as its own box, stacked under it. Pruned branches are dashed, with the reason on hover and in the node table, and can be hidden. Candidates that reached the brief are outlined in their label's colour. Below the tree, pick a candidate to see its searches and the quotes they found.
- **Replay audit:** an animated replay of the run, step by step: detecting the changes (with paragraphs from this run aligned and their changed sentences marked), ranking them (the changes explored, then some of those past the cap), then for each change the candidates proposed and pre-screened, each candidate's evidence searches, labeling and rework rounds, and the summary. The tree grows as the run built it, with the run's clock, running cost and model calls, and a note on what each step does. Play, pause, step, change the speed, scrub, or jump to a change. It is built from the saved timings, ledger and audit trail; runs saved before timings were recorded replay without a clock.
- **Review queue:** what to check by hand. Findings that need review, explanations left out of the brief or annotated by labeling (with the reason and the evidence found), searches that did not finish, rationales left out, and the changes past the `--max-changes` cap. Nothing in it stopped the run.
- **Cost and time:** the run's cost, wall time and calls by step and model, as `sec-agent eval cost` reports them.

To browse the committed sample runs instead, point the data directory at them: `SEC_AGENT_DATA_DIR=samples uv run --frozen sec-agent app`. Without a full setup, use the light install in [samples/README.md](samples/README.md).

The app shows saved runs only. Starting a run from the app is a future enhancement and is not currently included; start runs with `sec-agent analyze`.

The full passages come from the DuckDB store, opened read-only. While `ingest` or `index` holds it for writing, the app shows the quotes alone.

## Evaluation

`sec-agent eval` runs the evaluations against the hand-checked benchmark in [eval/benchmark/](eval/benchmark/README.md):
- **Tune split (default):** Hershey. Settings were tuned on it.
- **Test split:** Mattel and Stanley Black & Decker, run once after settings were fixed. A test-split run asks for confirmation first.

```bash
uv run --frozen sec-agent eval tasks                  # list the commands, their task and cost
uv run --frozen sec-agent eval retrieval              # recall@5 (free, local models)
uv run --frozen sec-agent eval judge --cap 0.30       # paid commands require --cap and ask first
uv run --frozen sec-agent eval cost RUN_DIR           # cost and latency of a saved run (free)
```

Results go to `<data dir>/eval/results/` unless `--out` is given. The results so far are in [EVALUATION.md](EVALUATION.md).

## Other commands

```bash
uv run --frozen sec-agent search-facts HSY "segment margin"   # search figures as the Financial Agent does
uv run --frozen sec-agent parse-report                         # re-parse every filing and print what happened
uv run --frozen sec-agent discover-corpus                      # re-pin the latest filings (changes the corpus)
```

## How filings are validated

`sec-agent ingest` sends every filing through one gate (`ingest/pipeline.py`). Later stages read filings with `load_validated_filing`, which refuses anything that failed, so a bad parse can't reach the index.

1. **Tables:** rows of figures are removed, because figures come from XBRL. Prose rows, such as footnotes, stay in the text. Each removed table is kept in the filing's record.
2. **Sections:** `edgartools` splits the filing into items. They are mapped to canonical names (MD&A, risk factors, market risk), so that a 10-K and a 10-Q can be compared. If validation fails, the fallback parser (`ingest/fallback.py`) tries instead, under the same checks.
3. **Validation:** `ingest/audit.py` checks results against evidence the parser didn't use. It looks for:
   - sentences wrongly removed with tables;
   - leftover tables;
   - lost text;
   - wrong section boundaries.
4. **Facts:** filings that pass are loaded into DuckDB.

A rejected filing makes `ingest` exit 1 and print its problems. Records are in `<data dir>/parsed/<ticker>/<accession>.json`. Bump `PARSER_VERSION` after changing parsing or validation, so that existing records are redone.

### Figures

`ingest` loads two kinds of figures. `search-facts` and the Financial Agent search both through the `facts` view:

- **XBRL facts:** everything a filer tagged, read from the filing's inline XBRL. This includes segment values, which the SEC's aggregate API leaves out.
- **Table facts:** untagged figures from tables that parsed cleanly. Figures XBRL already reports are left out, allowing for rounding.

A table only yields facts if all of it can be explained: stated scale, evidence of dollars, a period for each column, and rows that can be told apart. Anything else is indexed as text, so that no figure is unreachable.

Read table facts with care:
- **Sign:** as displayed, so a cost in parentheses is negative.
- **Duration:** describes the column, so `end_date` is the reliable part.
- **`row_group` and `heading`:** hints only. Check the source table before relying on them.

### Trying a new company

The parser was developed against the three analysis companies, plus five chosen for unusual layouts (`config/parser_validation.yaml`). It was measured on four sets of five randomly drawn companies (`config/parser_holdout*.yaml`). To try others:

```bash
uv run --frozen sec-agent discover-corpus --corpus /tmp/new.yaml --ticker ABC --ticker XYZ
uv run --frozen sec-agent ingest --corpus /tmp/new.yaml
uv run --frozen sec-agent index --corpus /tmp/new.yaml
uv run --frozen sec-agent analyze ABC --comparison yoy-10k --corpus /tmp/new.yaml
```

Fiscal years and quarters are derived from each company's 10-K dates, because filers' own tags are sometimes wrong. `discover-corpus` prints the filer's tag beside any filing it corrected.

Passing the gate is good evidence, but not proof:
- The latest fair test accepted 17 of 25 filings.
- None of the accepted filings had a materially wrong section.
- Its 8 rejections look like false alarms, such as long row labels or length floors that don't fit very small companies.

Results are in [eval/parser_holdout.md](eval/parser_holdout.md), and the scripts are in [eval/ingestion/](eval/ingestion/README.md). [docs/How_SEC_Filing_Ingestion_Is_Tested.md](docs/How_SEC_Filing_Ingestion_Is_Tested.md) explains the tests in plain language.

Known limits:
- A company whose ticker moved to a new registrant has its older filings under the old CIK, so `discover-corpus` stops with an error (ExxonMobil).
- Filings whose MD&A or statements are in Exhibit 13 can't be read.
- AT&T's 10-Ks are rejected until their market-risk section is repaired.
- A 10-Q whose date doesn't fit the company's fiscal calendar, as after a change of fiscal year, is left out. Selection may then stop with an error for that company.

## Development

```bash
uv run --frozen pytest              # unit tests, with no network or paid calls
uv run --frozen pytest -m live      # paid smoke tests against the Claude API (under $0.01)
uv run --frozen pytest -m corpus    # validate all downloaded filings (run ingest first)
uv run --frozen ruff check src tests
uv run --frozen ruff format src tests
```

After changing dependencies, run `scripts/lock.sh` instead of `uv lock`, then `uv sync --frozen`. `uv lock` fails on VirtualBox shared folders with "Text file busy".

## License

[MIT](LICENSE).
