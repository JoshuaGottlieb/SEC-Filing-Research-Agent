"""T5.2 agreement check: the entailment judge against the benchmark's labeled judge examples.

Run from the repository root (paid: about $0.004 per example on Opus 5.5):

    uv run --frozen python eval/judge/agreement.py eval/benchmark/HSY.json out.json [cap] [ids]

One call per example, sequential, so the cached system prompt is reused after the first call. The
ledger stops at `cap` dollars (default $0.30). `ids` is an optional comma-separated list of item
ids to re-judge. Tune on HSY only; the test set (MAT, SWK) is judged once and not used for tuning.
Results of the 2026-11-12 run are in `agreement_2026-11-12.json`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from sec_agent.agents.verification import EntailmentJudge, judge_entailment
from sec_agent.benchmark import load_benchmark
from sec_agent.config import get_settings
from sec_agent.llm.client import LLMClient, Role
from sec_agent.llm.ledger import CostLedger

HARD_STOP_USD = 0.30


async def main(label_file: Path, out: Path, cap: float, only: set[str]) -> None:
    examples = [e for e in load_benchmark(label_file).judge if not only or e.item_id in only]
    ledger = CostLedger(cap, path=out.with_suffix(".ledger.jsonl"))
    llm = LLMClient(get_settings(), ledger, max_concurrency=1)
    print("model:", llm.model_for(Role.ENTAILMENT), "| examples:", len(examples))
    judge = EntailmentJudge(llm)
    rows = []
    for ex in examples:
        result = await judge_entailment(ex.claim, ex.quote.text, judge)
        got = result.verdict.value if result.verdict else None
        rows.append(
            {
                "item_id": ex.item_id,
                "cause_id": ex.cause_id,
                "expected": ex.verdict.value,
                "got": got,
                "agree": got == ex.verdict.value,
                "reason": result.reason,
                "failure": result.failure,
            }
        )
        print(f"{ex.item_id}: expected {ex.verdict.value}, got {got}, ${ledger.spent_usd:.4f}")
    out.write_text(json.dumps({"spent_usd": ledger.spent_usd, "rows": rows}, indent=2))
    agree = sum(r["agree"] for r in rows)
    print(f"agreement {agree}/{len(rows)}; spent ${ledger.spent_usd:.4f}")


if __name__ == "__main__":
    cap = float(sys.argv[3]) if len(sys.argv) > 3 else HARD_STOP_USD
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else set()
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2]), cap, only))
