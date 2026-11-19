# Ingestion: filings to searchable text and figures

`sec-agent ingest` and `sec-agent index` turn the pinned filings into what the agents search: chunks of narrative text and a store of figures. This page covers how filings are validated, what figures are loaded, and how to try a company outside the corpus. The [main README](../../../README.md#quick-start) has the commands.

[How SEC Filing Ingestion Is Tested](../../../docs/How_SEC_Filing_Ingestion_Is_Tested.md) explains the tests in plain language. The scripts behind the measurements are in [eval/ingestion/](../../../eval/ingestion/README.md).

## The corpus

`ingest` reads the pinned corpus in [config/corpus.yaml](../../../config/corpus.yaml). It covers Hershey (HSY), Mattel (MAT) and Stanley Black & Decker (SWK), with five filings each:
- the latest two 10-Ks;
- the latest 10-Q;
- the same quarter a year earlier;
- the quarter before.

`index` splits the validated filings into chunks of 300 to 500 tokens and embeds them into a local Chroma index. On the pinned corpus that is about 2,200 chunks, roughly 15 minutes on a 4-core CPU the first time. A re-run embeds only chunks that changed.

## How filings are validated

`sec-agent ingest` sends every filing through one gate (`pipeline.py`). Later stages read filings with `load_validated_filing`, which refuses anything that failed, so a bad parse can't reach the index.

1. **Tables:** rows of figures are removed, because figures come from XBRL. Prose rows, such as footnotes, stay in the text. Each removed table is kept in the filing's record (`tables.py`).
2. **Sections:** `edgartools` splits the filing into items. They are mapped to canonical names (MD&A, risk factors, market risk), so that a 10-K and a 10-Q can be compared (`sections.py`). If validation fails, the fallback parser (`fallback.py`) tries instead, under the same checks.
3. **Validation:** `audit.py` checks results against evidence the parser didn't use. It looks for:
   - sentences wrongly removed with tables;
   - leftover tables;
   - lost text;
   - wrong section boundaries.
4. **Facts:** filings that pass are loaded into DuckDB.

A rejected filing makes `ingest` exit 1 and print its problems. Records are in `<data dir>/parsed/<ticker>/<accession>.json`. Bump `PARSER_VERSION` in `pipeline.py` after changing parsing or validation, so that existing records are redone.

## Figures

`ingest` loads two kinds of figures. `search-facts` and the Financial Analysis Agent search both through the `facts` view:

- **XBRL facts:** everything a filer tagged, read from the filing's inline XBRL (`edgar/xbrl.py`). This includes segment values, which the SEC's aggregate API leaves out.
- **Table facts:** untagged figures from tables that parsed cleanly (`table_facts.py`). Figures XBRL already reports are left out, allowing for rounding.

A table only yields facts if all of it can be explained: stated scale, evidence of dollars, a period for each column, and rows that can be told apart. Anything else is indexed as text, so that no figure is unreachable.

Read table facts with care:
- **Sign:** as displayed, so a cost in parentheses is negative.
- **Duration:** describes the column, so `end_date` is the reliable part.
- **`row_group` and `heading`:** hints only. Check the source table before relying on them.

## Commands for checking ingestion

```bash
uv run --frozen sec-agent search-facts HSY "segment margin"   # search figures as the Financial Analysis Agent does
uv run --frozen sec-agent parse-report                         # re-parse every filing and print what happened
uv run --frozen sec-agent discover-corpus                      # re-pin the latest filings (changes the corpus)
uv run --frozen pytest -m corpus                               # validate all downloaded filings
```

## Trying a new company

The parser was developed against the three analysis companies, plus five chosen for unusual layouts (`config/parser_validation.yaml`). It was measured on four sets of five randomly drawn companies (`config/parser_holdout*.yaml`). To try others:

```bash
uv run --frozen sec-agent discover-corpus --corpus /tmp/new.yaml --ticker ABC --ticker XYZ
uv run --frozen sec-agent ingest --corpus /tmp/new.yaml
uv run --frozen sec-agent index --corpus /tmp/new.yaml
uv run --frozen sec-agent analyze ABC --comparison yoy-10k --corpus /tmp/new.yaml
```

Fiscal years and quarters are derived from each company's 10-K dates, because filers' own tags are sometimes wrong. `discover-corpus` prints the filer's tag beside any filing it corrected.

Passing the gate is good evidence, but not proof:
- The latest fair test (the fourth) accepted 17 of 25 filings.
- None of the accepted filings had a materially wrong section.
- Its 8 rejections look like false alarms, such as long row labels or length floors that don't fit very small companies.

Results are in [eval/ingestion/parser_holdout.md](../../../eval/ingestion/parser_holdout.md).

## Known limits

- A company whose ticker moved to a new registrant has its older filings under the old CIK, so `discover-corpus` stops with an error (ExxonMobil).
- Filings whose MD&A or statements are in Exhibit 13 can't be read.
- AT&T's 10-Ks are rejected until their market-risk section is repaired.
- A 10-Q whose date doesn't fit the company's fiscal calendar, as after a change of fiscal year, is left out. Selection may then stop with an error for that company.
