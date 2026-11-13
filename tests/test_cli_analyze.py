"""`sec-agent analyze`: the cost estimate and its confirmation, with the workflow faked."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import sec_agent.cli
import sec_agent.graph
from sec_agent.cli import app
from sec_agent.compare.materiality import MaterialityHeadroom
from sec_agent.graph import RunDeclined, RunReport, estimate_cost


class FakeAnalysis:
    """Stands in for `run_analysis`: shows the estimate and stops if it is declined."""

    def __init__(self) -> None:
        self.ran = False

    async def __call__(self, settings, ticker, kind, **options) -> RunReport:
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
    assert "Estimated model cost: about $0.41, with a hard stop at $1.00." in result.output
    assert "Go ahead?" not in result.output
    assert analysis.ran
    assert "model cost $0.3100 (estimated $0.41)" in result.output


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
