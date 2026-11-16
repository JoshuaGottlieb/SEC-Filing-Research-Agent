"""`sec-agent eval`: each command runs the right script with the right arguments, and asks first
before a paid run or a test-set run. The scripts themselves are faked."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import sec_agent.eval_cli as eval_cli
from sec_agent.cli import app

runner = CliRunner()


class FakeScripts:
    def __init__(self, code: int = 0) -> None:
        self.calls: list[tuple[Path, list[str]]] = []
        self.code = code

    def __call__(self, script: Path, args: list[str]) -> int:
        self.calls.append((script, args))
        return self.code


@pytest.fixture
def scripts(monkeypatch, settings) -> FakeScripts:
    fake = FakeScripts()
    monkeypatch.setattr(eval_cli, "run_script", fake)
    monkeypatch.setattr(eval_cli, "get_settings", lambda: settings)
    return fake


def invoke(*args: str, input: str | None = None):
    return runner.invoke(app, ["eval", *args], input=input)


def test_every_script_exists():
    for script in eval_cli.SCRIPTS.values():
        assert script.is_file(), script


def test_every_task_listed_has_a_command():
    commands = {
        command.name or command.callback.__name__.replace("_", "-")
        for command in eval_cli.app.registered_commands
    } | {group.name for group in eval_cli.app.registered_groups}
    assert {name for name, *_ in eval_cli.TASKS} <= commands


def test_free_tuning_run_does_not_ask(scripts, tmp_path):
    result = invoke("retrieval", "--out", str(tmp_path / "r.json"))
    assert result.exit_code == 0, result.output
    ((script, args),) = scripts.calls
    assert script.name == "recall.py"
    assert args == [str(eval_cli.LABELS[eval_cli.Split.TUNE]), str(tmp_path / "r.json")]


def test_default_output_goes_under_the_data_dir(scripts, settings):
    assert invoke("retrieval").exit_code == 0
    ((_, args),) = scripts.calls
    out = Path(args[1])
    assert out.parent == settings.data_dir / "eval" / "results"
    assert out.name.startswith("retrieval-tune-") and out.suffix == ".json"


def test_test_split_asks_even_when_free(scripts):
    result = invoke("retrieval", "--split", "test", input="n\n")
    assert result.exit_code == 1
    assert "run it once" in result.output and not scripts.calls
    assert invoke("retrieval", "--split", "test", input="y\n").exit_code == 0
    assert scripts.calls[0][1][0] == str(eval_cli.LABELS[eval_cli.Split.TEST])


def test_paid_run_needs_a_cap(scripts):
    result = invoke("judge")
    assert result.exit_code != 0 and not scripts.calls


def test_paid_run_asks_unless_yes(scripts, tmp_path):
    out = str(tmp_path / "j.json")
    declined = invoke("judge", "--cap", "0.30", "--out", out, input="n\n")
    assert declined.exit_code == 1 and "$0.30" in declined.output and not scripts.calls
    assert (
        invoke("judge", "--cap", "0.30", "--out", out, "--only", "HSY-J01", "--yes").exit_code == 0
    )
    ((script, args),) = scripts.calls
    assert script.name == "agreement.py"
    assert args[1:] == [out, "0.3", "HSY-J01"]


def test_survival_passes_its_options(scripts, tmp_path):
    out = str(tmp_path / "s.json")
    result = invoke(
        "survival", "--cap", "1", "--form", "text", "--out", out, "--label",
        "--candidate-width", "6", "--worker-model", "claude-sonnet-5-5", "--no-search-cache", "-y",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    ((script, args),) = scripts.calls
    assert script.name == "survival.py"
    assert args[1:] == [
        "text", out, "--cap", "1.0", "--label", "--candidate-width", "6",
        "--worker-model", "claude-sonnet-5-5", "--no-search-cache",
    ]  # fmt: skip


def test_baseline_uses_the_splits_companies(scripts, tmp_path):
    out = str(tmp_path / "b")
    assert invoke("baseline", "--cap", "2", "--split", "test", "--out", out, "-y").exit_code == 0
    assert scripts.calls[0][1] == ["MAT", "SWK", out, "--cap", "2.0"]


def test_materiality_passes_labels_and_cap(scripts, tmp_path):
    out = str(tmp_path / "m")
    result = invoke("materiality", "--cap", "8", "--split", "test", "--out", out, input="n\n")
    assert result.exit_code == 1 and not scripts.calls
    assert invoke("materiality", "--cap", "8", "--split", "test", "--out", out, "-y").exit_code == 0
    ((script, args),) = scripts.calls
    assert script.name == "materiality.py"
    assert args == [str(eval_cli.LABELS[eval_cli.Split.TEST]), out, "--cap", "8.0"]


def test_materiality_dry_run_does_not_ask(scripts):
    result = invoke("materiality", "--cap", "8", "--split", "test", "--dry-run", "--out", "x")
    assert result.exit_code == 0 and "Go ahead" not in result.output
    assert scripts.calls[0][1][-1] == "--dry-run"


def test_baseline_dry_run_does_not_ask(scripts, tmp_path):
    result = invoke("baseline", "--cap", "2", "--split", "test", "--dry-run", "--out", "x")
    assert result.exit_code == 0 and "Go ahead" not in result.output
    assert scripts.calls[0][1][-1] == "--dry-run"


def test_score_free_needs_no_cap_and_does_not_ask(scripts, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    out = str(tmp_path / "s.json")
    result = invoke("score", "--baseline", str(base), "--free", "--out", out)
    assert result.exit_code == 0, result.output
    assert scripts.calls[0][1][1:] == [out, "--baseline", str(base), "--free"]


def test_score_needs_something_to_score_and_a_cap(scripts, tmp_path):
    assert invoke("score", "--free").exit_code != 0
    base = tmp_path / "base"
    base.mkdir()
    assert invoke("score", "--baseline", str(base)).exit_code != 0
    assert not scripts.calls


def test_score_with_agent_runs(scripts, tmp_path):
    runs = [tmp_path / "r1", tmp_path / "r2"]
    for run in runs:
        run.mkdir()
    out = str(tmp_path / "s.json")
    args = ["score", "--agent", str(runs[0]), "--agent", str(runs[1]), "--cap", "1", "-y"]
    assert invoke(*args, "--out", out).exit_code == 0
    assert scripts.calls[0][1][1:] == [out, "--agent", str(runs[0]), str(runs[1]), "--cap", "1.0"]


def test_hand_check_and_cost_run_free(scripts, tmp_path):
    scores, sheet, answers = (tmp_path / name for name in ("s.json", "h.json", "a.json"))
    for path in (scores, sheet, answers):
        path.write_text("{}")
    assert invoke("hand-check", "sample", str(scores), str(sheet)).exit_code == 0
    assert invoke("hand-check", "tally", str(sheet), str(answers)).exit_code == 0
    assert invoke("cost", str(tmp_path)).exit_code == 0
    assert [args[0] for _, args in scripts.calls[:2]] == ["sample", "tally"]
    assert scripts.calls[2][1] == [str(tmp_path)]


def test_a_failing_script_fails_the_command(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(eval_cli, "run_script", FakeScripts(code=3))
    assert invoke("cost", str(tmp_path)).exit_code == 3


def test_grounding_passes_every_baseline_folder_and_does_not_ask(scripts, tmp_path):
    first, second = tmp_path / "hsy", tmp_path / "test"
    first.mkdir()
    second.mkdir()
    result = invoke("grounding", str(first), str(second), "--out", str(tmp_path / "g.json"))
    assert result.exit_code == 0, result.output
    ((script, args),) = scripts.calls
    assert script.name == "grounding.py"
    assert args == [str(first), str(second), "--out", str(tmp_path / "g.json")]
