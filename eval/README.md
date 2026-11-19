# Evaluation

This folder holds the benchmark, the evaluation scripts and the committed result files. [EVALUATION.md](../EVALUATION.md) reports every result. This page says how to rerun each evaluation and where its pieces are.

## The benchmark and the fair-test rule

The hand-checked labels are in [benchmark/](benchmark/README.md), which also explains how they were drafted and reviewed.
- **Tune split (default):** Hershey. Settings were tuned on it.
- **Test split:** Mattel and Stanley Black & Decker, run once after settings were fixed. Nothing is tuned on it, and a test-split run asks for confirmation first.

## Running an evaluation

`sec-agent eval` runs each evaluation's script with the project's settings:

```bash
uv run --frozen sec-agent eval tasks                  # list the commands, their task and cost
uv run --frozen sec-agent eval retrieval              # recall@5 (free, local models)
uv run --frozen sec-agent eval judge --cap 0.30       # paid commands require --cap and ask first
uv run --frozen sec-agent eval cost RUN_DIR           # cost and latency of a saved run (free)
```

Add `--split test` for the test split. Results go to `<data dir>/eval/results/` unless `--out` is given. Paid calls go through the replay cache, so rerunning an evaluation with the same inputs costs nothing.

| Command | Task | Cost | What it measures | Script |
|---|---|---|---|---|
| `retrieval` | T6.2 | free | recall@5 with and without the reranker (local models) | [retrieval/recall.py](retrieval/recall.py) |
| `redundancy` | T2.4 | free | near-duplicate passages in the search's top 5 | [retrieval/redundancy.py](retrieval/redundancy.py) |
| `materiality` | T6.3 | paid | missed changes and false positives of the materiality step | [materiality/materiality.py](materiality/materiality.py) |
| `judge` | T5.2 | paid | the entailment judge against the labeled judge examples (Opus) | [judge/agreement.py](judge/agreement.py) |
| `survival` | T6.4 | paid | known causes kept by the Tree-of-Thought search (worker model) | [tot/survival.py](tot/survival.py) |
| `match-causes` | T6.4 | paid | match a survival run's candidates to labeled causes (Opus) | [tot/match_causes.py](tot/match_causes.py) |
| `baseline` | T6.5 | paid | the prompt-only baseline, one Opus call per comparison | [baseline/baseline.py](baseline/baseline.py) |
| `score` | T6.5 | paid | score the baseline and agent runs alike (free with `--free`) | [baseline/score.py](baseline/score.py) |
| `grounding` | T6.5 | free | check the baseline's changes against the diff and metrics | [baseline/grounding.py](baseline/grounding.py) |
| `hand-check` | T6.5 | free | draw or tally the blinded hand check | [baseline/hand_check.py](baseline/hand_check.py) |
| `cost` | T6.6 | free | cost and latency per finding from saved runs | [cost/cost_latency.py](cost/cost_latency.py) |

[TASKS.md](../TASKS.md) explains the task IDs.

## Committed results

| File | What it holds |
|---|---|
| [retrieval/recall_2026-11-13.json](retrieval/recall_2026-11-13.json) | Retrieval recall, both splits, at the earlier rerank pool of 20 |
| [retrieval/recall_hsy_2026-11-15.json](retrieval/recall_hsy_2026-11-15.json) | Retrieval recall on HSY at the final rerank pool of 10 |
| [retrieval/recall_test_2026-11-16.json](retrieval/recall_test_2026-11-16.json) | Retrieval recall on the test split at the final rerank pool of 10 |
| [judge/agreement_2026-11-12.json](judge/agreement_2026-11-12.json) | The judge's agreement with the labeled examples |
| [parser_holdout.md](parser_holdout.md) | The parser's validation gate on four sets of randomly drawn companies |

The nine full agent runs that the agent's results come from are in [samples/](../samples/README.md).

## Ingestion

[ingestion/](ingestion/README.md) holds the scripts that measured the parser and its validation gate. [src/sec_agent/ingest/](../src/sec_agent/ingest/README.md) describes the gate itself.
