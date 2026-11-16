"""`sec-agent eval`: the Phase 6 evaluations, run from the command line.

Each command runs one of the evaluation scripts under `eval/` (the user's decision, 2026-11-13:
the main session wires `eval` to the T6.x scripts). The scripts stay the single implementation:
a command picks the label file for a split, chooses an output path, asks before anything is spent,
and runs the script in a fresh interpreter from the repository root, which is how each script is
documented to run (a script imports its neighbours, such as `score.py` importing `baseline.py`).
This needs a source checkout, as the editable install is.

Two guards apply before a script starts:

- **Paid commands** need `--cap`, the script's own dollar limit, and ask before running unless
  `--yes` is given. Free ones (local search, reading saved runs) just run.
- **The test split** (Mattel and Stanley, `eval/benchmark/test.json`) also asks first, paid or not.
  The project's fair-test rule is that the test set runs once, after settings are fixed on
  Hershey; this cannot enforce "once", but it stops an accidental run.

Outputs default to `<data_dir>/eval/results/<task>-<split>-<UTC time>.json` (a directory for the
baseline), so runs never overwrite each other; `--out` overrides.

T6.3 (`materiality`) runs the pipeline's own materiality step, so it spends what `analyze` spends
on that step; its dry run prints the batches and their cost first.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from sec_agent.config import REPO_ROOT, get_settings

EVAL_DIR = REPO_ROOT / "eval"
SCRIPTS = {
    "retrieval": EVAL_DIR / "retrieval" / "recall.py",
    "redundancy": EVAL_DIR / "retrieval" / "redundancy.py",
    "judge": EVAL_DIR / "judge" / "agreement.py",
    "materiality": EVAL_DIR / "materiality" / "materiality.py",
    "survival": EVAL_DIR / "tot" / "survival.py",
    "match-causes": EVAL_DIR / "tot" / "match_causes.py",
    "baseline": EVAL_DIR / "baseline" / "baseline.py",
    "score": EVAL_DIR / "baseline" / "score.py",
    "hand-check": EVAL_DIR / "baseline" / "hand_check.py",
    "cost": EVAL_DIR / "cost" / "cost_latency.py",
}

# What `sec-agent eval tasks` lists: the plan task, whether it spends money, and what it does.
TASKS = (
    ("retrieval", "T6.2", "free", "recall@5 with and without the reranker (local models)"),
    ("redundancy", "T2.4", "free", "near-duplicate passages in the search's top 5"),
    ("materiality", "T6.3", "paid", "missed changes and false positives of the materiality step"),
    ("judge", "T5.2", "paid", "entailment judge against the labeled judge examples (Opus)"),
    ("survival", "T6.4", "paid", "known causes kept by the Tree-of-Thought search (worker model)"),
    ("match-causes", "T6.4", "paid", "match a survival run's candidates to labeled causes (Opus)"),
    ("baseline", "T6.5", "paid", "the prompt-only baseline, one Opus call per comparison"),
    ("score", "T6.5", "paid", "score the baseline and agent runs alike (free with --free)"),
    ("hand-check", "T6.5", "free", "draw or tally the user's blinded hand check"),
    ("cost", "T6.6", "free", "cost and latency per finding from saved runs"),
)


class Split(StrEnum):
    TUNE = "tune"
    TEST = "test"


LABELS = {
    Split.TUNE: EVAL_DIR / "benchmark" / "HSY.json",
    Split.TEST: EVAL_DIR / "benchmark" / "test.json",
}
TICKERS = {Split.TUNE: ("HSY",), Split.TEST: ("MAT", "SWK")}

SplitOption = Annotated[
    Split, typer.Option("--split", help="tune (Hershey) or test (Mattel and Stanley, run once).")
]
OutOption = Annotated[
    Path | None, typer.Option("--out", help="Output path (default: under <data_dir>/eval/results).")
]
CapOption = Annotated[
    float, typer.Option("--cap", min=0.0, help="Dollar limit for this run's model calls.")
]
YesOption = Annotated[bool, typer.Option("--yes", "-y", help="Go ahead without asking.")]

app = typer.Typer(no_args_is_help=True, help="Run the evaluations (Phase 6).")


def run_script(script: Path, args: list[str]) -> int:
    """Run an evaluation script from the repository root and return its exit code."""
    return subprocess.run(
        [sys.executable, str(script), *args], cwd=REPO_ROOT, check=False
    ).returncode


def _finish(task: str, args: list[str]) -> None:
    code = run_script(SCRIPTS[task], args)
    if code:
        raise typer.Exit(code=code)


def _default_out(task: str, split: Split | None, suffix: str = ".json") -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"{task}-{split.value}-{stamp}" if split else f"{task}-{stamp}"
    folder = get_settings().data_dir / "eval" / "results"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}{suffix}"


def _confirm(*, split: Split | None, cap: float | None, yes: bool) -> None:
    """Ask before a paid run or any test-set run; raise `typer.Exit` if the answer is no."""
    lines = []
    if cap is not None:
        lines.append(f"This makes live model calls, stopped at ${cap:.2f}.")
    if split is Split.TEST:
        lines.append(
            "This runs on the test set (MAT, SWK). The fair-test rule: run it once, after "
            "settings are fixed on Hershey, and change nothing because of what it shows."
        )
    if not lines:
        return
    for line in lines:
        typer.echo(line)
    if not (yes or typer.confirm("Go ahead?", default=False)):
        typer.echo("Not run.")
        raise typer.Exit(code=1)


@app.command()
def tasks() -> None:
    """List the evaluation commands, the plan task each serves, and whether it spends money."""
    for name, task, cost, what in TASKS:
        typer.echo(f"{name:13} {task:5} {cost:5} {what}")


@app.command()
def retrieval(split: SplitOption = Split.TUNE, out: OutOption = None, yes: YesOption = False):
    """T6.2: recall@5 on the labeled retrieval questions, with and without the reranker."""
    _confirm(split=split, cap=None, yes=yes)
    out = out or _default_out("retrieval", split)
    _finish("retrieval", [str(LABELS[split]), str(out)])


@app.command()
def redundancy(split: SplitOption = Split.TUNE, out: OutOption = None, yes: YesOption = False):
    """T2.4: how often the search's top 5 holds near-duplicate passages."""
    _confirm(split=split, cap=None, yes=yes)
    out = out or _default_out("redundancy", split)
    _finish("redundancy", [str(LABELS[split]), "--out", str(out)])


@app.command()
def judge(
    cap: CapOption,
    split: SplitOption = Split.TUNE,
    out: OutOption = None,
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Item id to re-judge; repeat for more.")
    ] = None,
    yes: YesOption = False,
):
    """T5.2: the entailment judge against the labeled judge examples (paid, Opus)."""
    _confirm(split=split, cap=cap, yes=yes)
    out = out or _default_out("judge", split)
    args = [str(LABELS[split]), str(out), str(cap)]
    if only:
        args.append(",".join(only))
    _finish("judge", args)


class Form(StrEnum):
    NUMERIC = "numeric"
    TEXT = "text"


@app.command()
def survival(  # noqa: PLR0913 -- one argument per option of the script
    cap: CapOption,
    form: Annotated[
        Form, typer.Option("--form", help="numeric (the reported rate) or text.")
    ] = Form.NUMERIC,
    split: SplitOption = Split.TUNE,
    out: OutOption = None,
    label: Annotated[bool, typer.Option("--label", help="Also label survivors (paid).")] = False,
    candidate_width: Annotated[
        int | None, typer.Option("--candidate-width", min=1, help="Beam width for this run.")
    ] = None,
    worker_model: Annotated[
        str | None, typer.Option("--worker-model", help="Worker model for this run.")
    ] = None,
    no_search_cache: Annotated[
        bool, typer.Option("--no-search-cache", help="Search afresh, for timing.")
    ] = False,
    yes: YesOption = False,
):
    """T6.4: whether each labeled cause survives the Tree-of-Thought search (paid)."""
    _confirm(split=split, cap=cap, yes=yes)
    out = out or _default_out(f"survival-{form.value}", split)
    args = [str(LABELS[split]), form.value, str(out), "--cap", str(cap)]
    if label:
        args.append("--label")
    if candidate_width is not None:
        args += ["--candidate-width", str(candidate_width)]
    if worker_model:
        args += ["--worker-model", worker_model]
    if no_search_cache:
        args.append("--no-search-cache")
    _finish("survival", args)


@app.command("match-causes")
def match_causes(
    run: Annotated[Path, typer.Argument(help="A survival output file.", exists=True)],
    cap: CapOption,
    out: OutOption = None,
    yes: YesOption = False,
):
    """T6.4: match a survival run's candidates to the labeled causes (paid, Opus)."""
    _confirm(split=None, cap=cap, yes=yes)
    out = out or run.with_name(f"{run.stem}_matcher.json")
    _finish("match-causes", [str(run), str(out), "--cap", str(cap)])


@app.command()
def materiality(
    cap: CapOption,
    split: SplitOption = Split.TUNE,
    out: OutOption = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print batches and expected cost; send nothing."),
    ] = False,
    yes: YesOption = False,
):
    """T6.3: missed changes and false positives of the materiality step (paid, Opus)."""
    if not dry_run:
        _confirm(split=split, cap=cap, yes=yes)
    out = out or _default_out("materiality", split, suffix="")
    args = [str(LABELS[split]), str(out), "--cap", str(cap)]
    if dry_run:
        args.append("--dry-run")
    _finish("materiality", args)


@app.command()
def baseline(
    cap: CapOption,
    split: SplitOption = Split.TUNE,
    out: OutOption = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print sizes and worst-case costs; send nothing.")
    ] = False,
    yes: YesOption = False,
):
    """T6.5: the prompt-only baseline, one Opus call per comparison (paid)."""
    if not dry_run:
        _confirm(split=split, cap=cap, yes=yes)
    out = out or _default_out("baseline", split, suffix="")
    args = [*TICKERS[split], str(out), "--cap", str(cap)]
    if dry_run:
        args.append("--dry-run")
    _finish("baseline", args)


@app.command()
def score(  # noqa: PLR0913 -- one argument per option of the script
    split: SplitOption = Split.TUNE,
    baseline_dir: Annotated[
        Path | None,
        typer.Option("--baseline", help="A baseline output directory.", exists=True),
    ] = None,
    agent: Annotated[
        list[Path] | None,
        typer.Option("--agent", help="A pipeline run directory; repeat for more.", exists=True),
    ] = None,
    cap: Annotated[
        float | None, typer.Option("--cap", min=0.0, help="Dollar limit (not needed with --free).")
    ] = None,
    free: Annotated[bool, typer.Option("--free", help="Code checks only; no model calls.")] = False,
    out: OutOption = None,
    yes: YesOption = False,
):
    """T6.5: score the baseline and the agent's runs with the same checks."""
    if baseline_dir is None and not agent:
        raise typer.BadParameter("give --baseline, --agent, or both")
    if not free and cap is None:
        raise typer.BadParameter("give --cap, or --free for the code checks only")
    _confirm(split=split, cap=None if free else cap, yes=yes)
    out = out or _default_out("score", split)
    args = [str(LABELS[split]), str(out)]
    if baseline_dir is not None:
        args += ["--baseline", str(baseline_dir)]
    if agent:
        args += ["--agent", *map(str, agent)]
    args += ["--free"] if free else ["--cap", str(cap)]
    _finish("score", args)


hand_check = typer.Typer(no_args_is_help=True, help="T6.5: the user's blinded hand check.")
app.add_typer(hand_check, name="hand-check")


@hand_check.command("sample")
def hand_check_sample(
    scores: Annotated[Path, typer.Argument(help="A score output, judged.", exists=True)],
    sheet: Annotated[Path, typer.Argument(help="Where to write the sheet.")],
):
    """Draw the blinded sheet; its key is written beside it."""
    _finish("hand-check", ["sample", str(scores), str(sheet)])


@hand_check.command("tally")
def hand_check_tally(
    sheet: Annotated[Path, typer.Argument(help="The sheet that was answered.", exists=True)],
    answers: Annotated[Path, typer.Argument(help="Answers: {id: yes|no}.", exists=True)],
):
    """Compare the answers with the judge, per system."""
    _finish("hand-check", ["tally", str(sheet), str(answers)])


@app.command()
def cost(
    runs: Annotated[
        list[Path], typer.Argument(help="Run folders or survival outputs.", exists=True)
    ],
    out: OutOption = None,
):
    """T6.6: cost and latency per step, finding and model, from saved runs (free)."""
    args = [*map(str, runs)]
    if out is not None:
        args += ["--out", str(out)]
    _finish("cost", args)
