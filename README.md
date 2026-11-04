# SEC Filing Research Agent

A multi-agent system that compares a company's 10-K and 10-Q filings across reporting periods, finds material changes, tests competing explanations, and writes a cited research brief for a fundamental analyst.

It is a first-pass research aid. It does not give investment advice, recommendations, or price predictions.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). No GPU or Docker is needed.

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

See [.env.example](.env.example) for optional settings, including models and the per-run budget. Data is stored in `~/.local/share/sec-agent` unless `SEC_AGENT_DATA_DIR` says otherwise.

## Usage

```bash
uv run --frozen sec-agent discover-corpus   # re-pin the latest filings to config/corpus.yaml (optional)
uv run --frozen sec-agent ingest            # download and validate the pinned filings; safe to re-run
uv run --frozen sec-agent parse-report      # re-parse every filing and print what happened to each
```

### How filings are validated

`sec-agent ingest` sends every filing through one gate (`ingest/pipeline.py`). Later stages read filings with `load_validated_filing`, which refuses anything that failed, so a bad parse can't reach the index.

1. **Tables:** rows of figures are removed, because figures come from XBRL. Prose rows, such as footnotes, stay in the text. Each removed table is kept in the filing's record.
2. **Sections:** `edgartools` splits the filing into items. If validation fails, the fallback parser (`ingest/fallback.py`) tries instead, under the same checks. It reads cross-reference indexes, item headings, and items that point to a later financial section.
3. **Validation:** `ingest/audit.py` checks results against evidence the parser didn't use:
   - **Wrongly removed tables:** sentences in the original tables that are gone from the output.
   - **Leftover tables:** kept tables full of XBRL-tagged numbers, and paragraphs made mostly of numbers.
   - **Lost text:** the share of the filing's sentences that reached a parsed item.
   - **Wrong sections:** a section that doesn't open with its title, lacks its expected terms, contains another part's heading, or an item that's far too long.

4. **Facts:** filings that pass are loaded into DuckDB, as described below.

A rejected filing makes `ingest` exit 1 and print its problems. Records are in `<data dir>/parsed/<ticker>/<accession>.json`. Bump `PARSER_VERSION` after changing parsing or validation, so that existing records are redone.

### Searching figures

`ingest` loads two kinds of figures, which `sec-agent search-facts HSY "segment margin"` and the Financial Agent search through the `facts` view:

- **XBRL facts:** everything a filer tagged, read from the filing's inline XBRL. This includes segment values, which the SEC's aggregate API leaves out.
- **Table facts:** untagged figures from tables that parsed cleanly, such as Mattel's gross billings by brand. Figures XBRL already reports are left out, allowing for rounding.

A table only yields facts if all of it can be explained: stated scale, evidence of dollars, a period for each column, and rows that can be told apart. Anything else goes to `unparsed_tables`, and the chunker must index those tables as text so that no figure is unreachable.

Read table facts with care:
- **Sign:** it's as displayed, so a cost in parentheses is negative even though XBRL reports it as positive.
- **Duration:** it describes the column, so a balance can appear under "three months". `end_date` is the reliable part, and `duration` is empty when the header doesn't say.
- **`row_group` and `heading`:** these are hints, and can be stale. Check the source table before relying on them.

Results and limits are in [eval/parser_holdout.md](eval/parser_holdout.md). [docs/How_SEC_Filing_Ingestion_Is_Tested.md](docs/How_SEC_Filing_Ingestion_Is_Tested.md) explains the tests in plain language.

### Trying a new company

The parser was developed against the three analysis companies plus five chosen for unusual layouts (`config/parser_validation.yaml`: JPMorgan, GE, Chevron, Tootsie Roll, Apple). `config/parser_holdout.yaml`, `parser_holdout2.yaml`, `parser_holdout3.yaml` and `parser_holdout4.yaml` each hold five randomly drawn companies, used to measure it. These files only test parsing, and their companies aren't analyzed. To test others:

```bash
uv run --frozen sec-agent discover-corpus --corpus /tmp/new.yaml --ticker ABC --ticker XYZ
uv run --frozen sec-agent ingest --corpus /tmp/new.yaml
```

Fiscal years and quarters are derived from each company's 10-K dates, because they become retrieval metadata and filers' own tags are sometimes wrong. `discover-corpus` prints the filer's tag beside any filing it corrected, and the corpus file keeps it.

If `ingest` rejects a filing, the printed problems say what failed. Passing is good evidence but not proof, so read the first and last paragraphs of each section for a company with an unfamiliar layout.

Known limits:
- A company whose ticker moved to a new registrant has its older filings under the old CIK, so `discover-corpus` stops with an error (ExxonMobil, CIK 0002115436).
- A 10-Q with no Item 1 or Item 2 headings in its body (JPMorgan) can't be read by the fallback. It works today because `edgartools` reads it, and it would be rejected if that stopped working.
- Cross-reference indexes give pages, not positions, so text sharing a page with a section's boundary may land on the wrong side when no heading marks the change.
- A 10-Q whose date doesn't fit the company's fiscal calendar, as after a change of fiscal year, is left out rather than labeled, so selection may stop with an error for that company.
- The gate rejects some usable filings, and it can't read filings whose MD&A or statements are in Exhibit 13.
- The gate can accept wrong sections. New checks catch the two cases found so far, Everus's cut-off notes and AT&T's misfiled reports, so AT&T's 10-Ks are now rejected until that section is repaired. Others may exist.
- The gate rejects many sound filings. The latest fair test accepted 17 of 25, and its 8 rejections look like false alarms: long row labels, and length floors that don't fit very small companies. Measurements are in [eval/parser_holdout.md](eval/parser_holdout.md), and the scripts behind them are in [eval/ingestion/](eval/ingestion/README.md).

## Development

```bash
uv run --frozen pytest              # unit tests, with no network or paid calls
uv run --frozen pytest -m live      # paid smoke tests against the Claude API
uv run --frozen pytest -m corpus    # validate all downloaded filings (run both ingests first)
uv run --frozen ruff check src tests
uv run --frozen ruff format src tests
```

After changing dependencies, run `scripts/lock.sh` instead of `uv lock`, then `uv sync --frozen`. `uv lock` fails on VirtualBox shared folders with "Text file busy".
