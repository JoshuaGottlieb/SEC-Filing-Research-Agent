"""`sec-agent analyze`: the estimate, its confirmation and resuming, with the workflow faked."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import sec_agent.cli
import sec_agent.graph
from sec_agent.cli import app
from sec_agent.compare.materiality import MaterialityHeadroom
from sec_agent.graph import RunDeclined, RunReport, estimate_cost
from sec_agent.llm.ledger import BudgetExceededError


class FakeAnalysis:
    """Stands in for `run_analysis`: shows the estimate and stops if it is declined."""

    def __init__(self, raises: BaseException | None = None) -> None:
        self.ran = False
        self.raises = raises
        self.options: dict = {}

    async def __call__(self, settings, ticker, kind, **options) -> RunReport:
        self.options = options
        options["announce"](options["run_id"] or "HSY-yoy-10k-20261113T120000Z")
        if self.raises is not None:
            raise self.raises
        estimate = estimate_cost(
            MaterialityHeadroom(batches=11, uncached=0, headroom_usd=0.0),
            options["max_changes"],
            options["max_usd"] or settings.max_usd_per_run,
        )
        if not options["confirm"](estimate):
            raise RunDeclined("the cost estimate was declined. Nothing was sent.")
        self.ran = True
        return RunReport(
            run_id="HSY-run",
            brief=Path("runs/HSY-run/brief.md"),
            audit=Path("runs/HSY-run/audit.json"),
            cost_usd=0.31,
            estimated_usd=estimate.total_usd,
            selected=3,
            searched=3,
            not_examined=49,
            node_evaluations=40,
        )


@pytest.fixture
def analysis(monkeypatch, settings) -> FakeAnalysis:
    fake = FakeAnalysis()
    # The command imports `run_analysis` from the module when it runs, so this replaces it, and
    # the settings point at the test's temporary folder.
    monkeypatch.setattr(sec_agent.graph, "run_analysis", fake)
    monkeypatch.setattr(sec_agent.cli, "get_settings", lambda: settings)
    return fake


COMMAND = ["analyze", "hsy", "--comparison", "yoy-10k", "--max-changes", "3", "--max-usd", "1"]


def test_with_yes_the_estimate_is_printed_and_the_run_goes_ahead(analysis):
    result = CliRunner().invoke(app, [*COMMAND, "--yes"])
    assert result.exit_code == 0, result.output
    assert "Estimated model cost: about $0.23, with a hard stop at $1.00." in result.output
    assert "Go ahead?" not in result.output
    assert analysis.ran
    assert "model cost $0.3100 (estimated $0.23)" in result.output


def test_without_yes_it_asks_and_goes_ahead_on_yes(analysis):
    result = CliRunner().invoke(app, COMMAND, input="y\n")
    assert result.exit_code == 0, result.output
    assert "Go ahead? [y/N]" in result.output
    assert analysis.ran


@pytest.mark.parametrize("reply", ["n\n", "\n"])
def test_a_no_or_an_empty_reply_writes_no_brief(analysis, reply):
    result = CliRunner().invoke(app, COMMAND, input=reply)
    assert result.exit_code == 1
    assert "No brief written: the cost estimate was declined. Nothing was sent." in result.output
    assert not analysis.ran


def test_resume_continues_the_named_run(analysis):
    result = CliRunner().invoke(app, [*COMMAND, "--yes", "--resume", "HSY-old"])
    assert result.exit_code == 0, result.output
    assert (analysis.options["run_id"], analysis.options["resume"]) == ("HSY-old", True)
    assert result.output.startswith("Run HSY-old\n")


def test_a_fresh_run_is_not_a_resume_and_names_its_run_first(analysis):
    result = CliRunner().invoke(app, [*COMMAND, "--yes"])
    assert analysis.options["resume"] is False
    assert result.output.startswith("Run HSY-yoy-10k-20261113T120000Z\n")


def test_resume_and_run_id_together_are_refused(analysis):
    result = CliRunner().invoke(app, [*COMMAND, "--resume", "a", "--run-id", "b"])
    assert result.exit_code == 2
    assert not analysis.options


@pytest.mark.parametrize("raises", [KeyboardInterrupt()])
def test_a_stopped_run_says_how_to_continue_it(monkeypatch, settings, raises):
    fake = FakeAnalysis(raises=raises)
    monkeypatch.setattr(sec_agent.graph, "run_analysis", fake)
    monkeypatch.setattr(sec_agent.cli, "get_settings", lambda: settings)
    result = CliRunner().invoke(app, COMMAND)
    assert result.exit_code == 130
    assert (
        "sec-agent analyze HSY --comparison yoy-10k --resume HSY-yoy-10k-20261113T120000Z"
        in result.output
    )


def test_a_budget_stop_says_how_to_continue_with_a_higher_cap(monkeypatch, settings):
    fake = FakeAnalysis(raises=BudgetExceededError("Budget of $0.01 is too small."))
    monkeypatch.setattr(sec_agent.graph, "run_analysis", fake)
    monkeypatch.setattr(sec_agent.cli, "get_settings", lambda: settings)
    result = CliRunner().invoke(app, COMMAND)
    assert result.exit_code == 1
    assert "No brief written: Budget of $0.01 is too small." in result.output
    assert "--resume HSY-yoy-10k-20261113T120000Z --max-usd AMOUNT" in result.output


def test_a_refused_resume_says_nothing_ran(monkeypatch, settings):
    from sec_agent.graph import ResumeError

    fake = FakeAnalysis(raises=ResumeError("run HSY-old already finished; its brief is b.md"))
    monkeypatch.setattr(sec_agent.graph, "run_analysis", fake)
    monkeypatch.setattr(sec_agent.cli, "get_settings", lambda: settings)
    result = CliRunner().invoke(app, [*COMMAND, "--resume", "HSY-old"])
    assert result.exit_code == 1
    assert "Not run: run HSY-old already finished; its brief is b.md" in result.output
    assert "No brief written" not in result.output
