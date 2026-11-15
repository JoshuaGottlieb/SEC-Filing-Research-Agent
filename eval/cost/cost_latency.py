"""T6.6: cost and latency per finding, from a run's ledger and timing log.

Run from the repository root (reads files only: no model calls, no searches):

    uv run --frozen python eval/cost/cost_latency.py RUN [RUN ...] --out report.json

Each RUN is either a pipeline run folder (`<data_dir>/runs/<run id>/`, holding `ledger.jsonl` and
`timings.jsonl`) or a `survival.py` output file (`out.json`, beside `out.ledger.jsonl` and
`out.timings.jsonl`). Every ledger line names the step and change its call served and how long
it took; the timing log holds each step's wall time and each local evidence search
(`sec_agent.timing`).

What it reports per run (steps, findings, models) is described in `sec_agent.report.run_cost`.

Costs are what the ledger recorded, so a replayed call counts $0: a run measures real cost only
for the calls it sent. The report shows how many were replayed, so a mostly replayed run is not
read as cheap. Wall times are this machine's (CPU only, shared with other sessions), so they are
an upper bound for a quieter machine and far above a GPU's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sec_agent.report.run_cost import locate, report_run


def _money(value: float | None) -> str:
    return "-" if value is None else f"${value:.4f}"


def _secs(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}s"


def print_run(report: dict) -> None:
    print(f"\n## {report['run']}")
    note = "" if report["timed"] else " (no timing log: a run from before T6.6)"
    print(
        f"cost {_money(report['cost_usd'])}, {report['live_calls']} live / "
        f"{report['replayed_calls']} replayed calls, wall {_secs(report['wall_s'])}{note}"
    )
    print("| step | runs | wall | live | replayed | cost | in model calls | searches | in search |")
    print("|---|---|---|---|---|---|---|---|---|")
    for step, row in report["steps"].items():
        print(
            f"| {step} | {row['runs']} | {_secs(row['wall_s'])} | {row['live_calls']} | "
            f"{row['replayed_calls']} | {_money(row['cost_usd'])} | {_secs(row['model_s'])} | "
            f"{row['searches']} | {_secs(row['search_s'])} |"
        )
    per = report["per_finding"]
    if report["findings"]:
        cost, wall = per["cost_usd"], per["wall_s"]
        print(
            f"per finding ({report['findings']}): cost mean {_money(cost['mean'])}, median "
            f"{_money(cost['median'])}, max {_money(cost['max'])}; wall mean "
            f"{_secs(wall['mean'])}, median {_secs(wall['median'])}, max {_secs(wall['max'])}; "
            f"all-in {_money(per['all_in_cost_usd'])}"
        )
    print("| model | live | cost | median call | p90 call | output tokens |")
    print("|---|---|---|---|---|---|")
    for model, row in report["models"].items():
        print(
            f"| {model} | {row['live_calls']} | {_money(row['cost_usd'])} | "
            f"{_secs(row['latency_s_median'])} | {_secs(row['latency_s_p90'])} | "
            f"{row['output_tokens']} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args = parser.parse_args()
    reports = [report_run(locate(path)) for path in args.runs]
    for report in reports:
        print_run(report)
    if args.out is not None:
        args.out.write_text(json.dumps({"runs": reports}, indent=2) + "\n")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
