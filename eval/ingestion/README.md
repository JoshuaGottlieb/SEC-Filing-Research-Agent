# Ingestion measurements

These scripts produce the numbers in [../parser_holdout.md](../parser_holdout.md). They aren't
in the test suite because a person has to read their output. Run them from the repo root with
`uv run --frozen python eval/ingestion/<script>`. A corpus is a path or a bare name from `config/`.

| Script | What it answers | Reads |
|---|---|---|
| `fingerprint.py` | Did the rules change during a fair test? | Source files |
| `draw_companies.py` | Which unseen companies does a seeded draw pick? | EDGAR |
| `read_boundaries.py` | Does each section start and end where it should? | Saved records |
| `compare_parsers.py` | Do the two parsers agree, sentence by sentence? | Raw filings |
| `table_accuracy.py` | Do parsed tables match XBRL tags, and how many are clean? | Raw filings |
| `cross_filing_agreement.py` | Do two filings agree on one untagged figure? | `table_accuracy.py` output |
| `audit_sample.py` | Which searchable facts should be checked by hand? | Fact store, raw filings |
| `store_checks.py` | Are known failure shapes in the store? | Fact store |
| `fiscal_check.py` | Do derived fiscal periods match filers' tags? | EDGAR |
| `check_matrix.py` | Which checks fail on each parser, and what if one is removed? How often does the fallback run? | Raw filings |
| `intervals.py` | What's the 95% interval for a small-sample result? | Arguments |

## Running a fair test

Choose a new seed first, and change no rules until the test is written up.

```bash
python eval/ingestion/fingerprint.py save eval/ingestion/holdout5.sha256
python eval/ingestion/draw_companies.py --seed <new seed> --out config/parser_holdout5.yaml
sec-agent ingest --corpus config/parser_holdout5.yaml    # exits 1 if it rejects a filing
python eval/ingestion/fingerprint.py check eval/ingestion/holdout5.sha256

python eval/ingestion/read_boundaries.py parser_holdout5
python eval/ingestion/compare_parsers.py parser_holdout5
python eval/ingestion/table_accuracy.py parser_holdout5 --facts-out facts.json
python eval/ingestion/cross_filing_agreement.py facts.json
python eval/ingestion/audit_sample.py parser_holdout5 --seed <new seed> --per-company 9
python eval/ingestion/store_checks.py
```

Score each accepted filing by hand from `read_boundaries.py` and `compare_parsers.py`. The gate's
pass rate alone says nothing about correctness: in the third test, 4 of 24 accepted filings had a
materially wrong section. Trace every parser disagreement to the filing's text, because a shared
error makes the parsers agree, and read every section whose last paragraph looks out of place.

`draw_companies.py` excludes every company in `config/`, so new companies stay new. Add the
new corpus to `CORPUS_FILES` in `tests/test_corpus_parsing.py` after the write-up, along with any
rejections, so the gate's result on those filings is pinned.
