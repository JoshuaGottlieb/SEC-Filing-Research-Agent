"""Cost and latency per finding, read back from a run's ledger and timing log (T6.6).

Every ledger line names the step and change its call served and how long it took, and the timing
log holds each step's wall time and each local evidence search (`sec_agent.timing`). From those
two files this builds, for one run:

- each step: wall time, model calls (live and replayed), cost, and the time inside model calls and
  inside local searches;
- each finding (a change that was searched): wall time and cost across its search, labeling and
  rework rounds, with the cost by step and role;
- per finding over the run: mean, median and largest cost and wall time, and the all-in cost per
  finding, which spreads the run's own steps (materiality, summary) over its findings;
- each model: live calls, cost, and the call latency's median and 90th percentile.

Time inside model calls and searches is summed over calls, so where calls overlap (parallel
candidates) it can exceed the step's wall time. A run's wall time is the sum of its steps; process
start-up (loading the search models, a few seconds) is outside every step.

Costs are what the ledger recorded, so a replayed call counts $0. Ledgers written before T6.6 have
no step or change on their lines; their calls are reported under `unscoped`.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sec_agent.llm.ledger import CostLedger, LedgerEntry
from sec_agent.timing import (
    RETRIEVE,
    STAGE_DETECT,
    STAGE_LABEL,
    STAGE_MATERIALITY,
    STAGE_REWORK,
    STAGE_SEARCH,
    STAGE_SUMMARY,
    Timing,
    read_timings,
)

UNSCOPED = "unscoped"
# A finding's own steps, in the order they run.
FINDING_STEPS = (STAGE_SEARCH, STAGE_LABEL, STAGE_REWORK)


@dataclass(frozen=True)
class RunFiles:
    name: str
    ledger: Path
    timings: Path


def locate(path: Path) -> RunFiles:
    """A run's ledger and timing log, from a run folder or a `survival.py` output file."""
    if path.is_dir():
        return RunFiles(path.name, path / "ledger.jsonl", path / "timings.jsonl")
    return RunFiles(
        path.stem, path.with_suffix(".ledger.jsonl"), path.with_suffix(".timings.jsonl")
    )


def read_ledger(path: Path) -> list[LedgerEntry]:
    if not path.exists():
        raise FileNotFoundError(f"no ledger at {path}")
    # The cap does not matter here: resuming only reads the lines back.
    return list(CostLedger.resume(float("inf"), path).entries)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _spread(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _calls(entries: list[LedgerEntry]) -> dict[str, float | int]:
    live = [e for e in entries if not e.replayed]
    return {
        "calls": len(entries),
        "live_calls": len(live),
        "replayed_calls": len(entries) - len(live),
        "cost_usd": sum(e.cost_usd for e in entries),
        "model_s": sum(e.elapsed_s for e in live),
    }


def step_table(entries: list[LedgerEntry], timings: list[Timing]) -> dict[str, dict]:
    """Per step: wall time, calls, cost, and time inside model calls and local searches."""
    by_step: dict[str, list[LedgerEntry]] = defaultdict(list)
    for entry in entries:
        by_step[entry.stage or UNSCOPED].append(entry)
    walls: dict[str, float] = defaultdict(float)
    runs: dict[str, int] = defaultdict(int)
    searches: dict[str, list[float]] = defaultdict(list)
    for timing in timings:
        if timing.stage == RETRIEVE:
            searches[timing.within or UNSCOPED].append(timing.elapsed_s)
        else:
            walls[timing.stage] += timing.elapsed_s
            runs[timing.stage] += 1
    table = {}
    for step in sorted({*by_step, *walls}, key=_step_order):
        table[step] = {
            "runs": runs.get(step, 0),
            "wall_s": walls.get(step),
            **_calls(by_step.get(step, [])),
            "searches": len(searches.get(step, [])),
            "search_s": sum(searches.get(step, [])),
        }
    return table


def _step_order(step: str) -> int:
    order = [STAGE_DETECT, STAGE_MATERIALITY, *FINDING_STEPS, STAGE_SUMMARY, UNSCOPED]
    return order.index(step) if step in order else len(order)


def finding_table(entries: list[LedgerEntry], timings: list[Timing]) -> dict[str, dict]:
    """Per searched change: wall time and cost over its own steps, and cost by role."""
    findings: dict[str, dict] = {}
    for timing in timings:
        if timing.change_id is None or timing.stage not in FINDING_STEPS:
            continue
        row = findings.setdefault(
            timing.change_id,
            {"wall_s": 0.0, "steps": defaultdict(float), "searches": 0, "search_s": 0.0},
        )
        row["wall_s"] += timing.elapsed_s
        row["steps"][timing.stage] += timing.elapsed_s
    for timing in timings:
        if timing.stage == RETRIEVE and timing.change_id in findings:
            findings[timing.change_id]["searches"] += 1
            findings[timing.change_id]["search_s"] += timing.elapsed_s
    by_change: dict[str, list[LedgerEntry]] = defaultdict(list)
    for entry in entries:
        if entry.change_id is not None:
            by_change[entry.change_id].append(entry)
    for change_id, row in findings.items():
        own = by_change.get(change_id, [])
        roles: dict[str, float] = defaultdict(float)
        for entry in own:
            roles[f"{entry.stage}:{entry.role}"] += entry.cost_usd
        row.update(_calls(own))
        row["steps"] = dict(row["steps"])
        row["cost_by_step_role"] = dict(sorted(roles.items()))
    return findings


def model_table(entries: list[LedgerEntry]) -> dict[str, dict]:
    """Per model: live calls, cost, and call latency (sent calls only)."""
    table = {}
    for model in sorted({e.model for e in entries}):
        mine = [e for e in entries if e.model == model]
        live = [e for e in mine if not e.replayed]
        latencies = [e.elapsed_s for e in live if e.elapsed_s > 0]
        table[model] = {
            **_calls(mine),
            "output_tokens": sum(e.output_tokens for e in live),
            "latency_s_median": _quantile(latencies, 0.5),
            "latency_s_p90": _quantile(latencies, 0.9),
            "timed_calls": len(latencies),
        }
    return table


def report_run(files: RunFiles) -> dict:
    entries = read_ledger(files.ledger)
    timings = read_timings(files.timings)
    steps = step_table(entries, timings)
    findings = finding_table(entries, timings)
    totals = _calls(entries)
    run_level = sum(row["cost_usd"] for step, row in steps.items() if step not in FINDING_STEPS)
    top = [t for t in timings if t.within is None and t.stage != RETRIEVE]
    count = len(findings)
    return {
        "run": files.name,
        **totals,
        "wall_s": sum(t.elapsed_s for t in top) if top else None,
        "timed": bool(timings),
        "unscoped_calls": sum(1 for e in entries if e.stage is None),
        "findings": count,
        "per_finding": {
            "cost_usd": _spread([row["cost_usd"] for row in findings.values()]),
            "wall_s": _spread([row["wall_s"] for row in findings.values()]),
            "live_calls": _spread([float(row["live_calls"]) for row in findings.values()]),
            # The run's own steps (materiality, summary) spread over its findings.
            "all_in_cost_usd": (totals["cost_usd"] / count) if count else None,
            "run_level_cost_usd": run_level,
        },
        "steps": steps,
        "models": model_table(entries),
        "finding_rows": findings,
    }
