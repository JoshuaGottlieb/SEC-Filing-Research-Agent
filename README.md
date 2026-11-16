# SEC Filing Research Agent

A multi-agent system that compares a company's 10-K and 10-Q filings across reporting periods, finds material changes, tests competing explanations with Tree-of-Thought search, and writes a cited research brief for a fundamental analyst.

It is a first-pass research aid. It does not give investment advice, recommendations, or price predictions. Every explanation in a brief carries a verbatim quote from a filing. Only explanations that pass every check are labeled `confirmed`.

- **[DESIGN_CHANGES.md](DESIGN_CHANGES.md):** how the build fills the gaps in the original design, and where it departs from it.
- **[EVALUATION.md](EVALUATION.md):** what has been measured so far, and what is still to run.

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

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). No GPU or Docker is needed.

The first `index` and `analyze` download two local models from Hugging Face, about 0.5 GB in all:
- `BAAI/bge-base-en-v1.5` for embeddings;
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking.

```bash
pip install uv

# Keep the virtual environment on local disk (add this to your shell profile).
# Symlinks and renames fail on VirtualBox shared folders.
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/sec-agent"

uv sync --frozen
cp .env.example .env    # then fill in the two required values
uv run --frozen sec-agent doctor
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

**Cost and time.** Use the estimate printed before each run as the guide. In the paid runs so far, three Hershey changes cost $0.15 to $0.19 and took about 7 minutes, mostly local search on the CPU. Those runs used the earlier, narrower search. Measurements at the current settings are pending (see [EVALUATION.md](EVALUATION.md)).

**Outputs** go to `<data dir>/runs/<run id>/`:

| File | Contents |
|---|---|
| `brief.md` | The research brief: a summary, then each finding with its labeled claims, quotes and figures, and the review queue |
| `audit.json` | Everything behind the brief: each claim's citations and check results, the judge's verdicts, and the changes not examined |
| `ledger.jsonl` | Every model call, with its tokens, cost, latency and the step it served |
| `timings.jsonl` | Wall time of every step and every local search |

Every Tree-of-Thought node is kept in `<data dir>/tot_tree.sqlite`, including pruned branches and their scores. The workflow reads and writes trees through a local MCP server.

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
