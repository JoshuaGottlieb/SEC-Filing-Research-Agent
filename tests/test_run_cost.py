"""T6.6's per-finding cost and latency, read back from a run's ledger and timing log."""

from __future__ import annotations

import pytest

from sec_agent.llm.ledger import CostLedger, LedgerEntry
from sec_agent.report.run_cost import UNSCOPED, locate, report_run
from sec_agent.timing import Timing, TimingLog

HAIKU, OPUS = "claude-haiku-5-5", "claude-opus-5-5"


def call(role, model, cost, stage=None, change=None, elapsed=1.0, replayed=False) -> LedgerEntry:
    return LedgerEntry(
        role, model, 10, 5, 0, 0, cost, replayed, elapsed_s=elapsed, stage=stage, change_id=change
    )


@pytest.fixture
def run_dir(tmp_path):
    """A small run: materiality, two searched changes (one reworked), and the summary."""
    folder = tmp_path / "runs" / "HSY-run"
    ledger = CostLedger(10.0, path=folder / "ledger.jsonl")
    for entry in (
        call("materiality", OPUS, 0.30, "materiality", elapsed=20.0),
        call("generator", HAIKU, 0.01, "search", "A", elapsed=2.0),
        call("critic", HAIKU, 0.02, "search", "A", elapsed=3.0),
        call("critic", HAIKU, 0.0, "search", "A", elapsed=0.0, replayed=True),
        call("entailment", OPUS, 0.05, "label", "A", elapsed=8.0),
        call("generator", HAIKU, 0.01, "rework", "A", elapsed=2.0),
        call("generator", HAIKU, 0.02, "search", "B", elapsed=4.0),
        call("brief", OPUS, 0.04, "summary", elapsed=10.0),
    ):
        ledger.record(entry)
    log = TimingLog(folder / "timings.jsonl")
    for timing in (
        Timing("detect", None, 30.0),
        Timing("materiality", None, 25.0),
        Timing("retrieve", "A", 3.5, within="search"),
        Timing("retrieve", "A", 3.5, within="search"),
        Timing("search", "A", 60.0),
        Timing("label", "A", 10.0),
        Timing("retrieve", "A", 3.5, within="rework"),
        Timing("rework", "A", 15.0),
        Timing("search", "B", 40.0),
        Timing("label", "B", 0.5),
        Timing("summary", None, 12.0),
    ):
        log.add(timing)
    return folder


def test_each_step_has_its_wall_time_cost_calls_and_searches(run_dir) -> None:
    steps = report_run(locate(run_dir))["steps"]
    assert list(steps) == ["detect", "materiality", "search", "label", "rework", "summary"]
    search = steps["search"]
    assert (search["runs"], search["wall_s"]) == (2, 100.0)
    assert (search["live_calls"], search["replayed_calls"]) == (3, 1)
    assert search["cost_usd"] == pytest.approx(0.05)
    assert search["model_s"] == pytest.approx(9.0)
    assert (search["searches"], search["search_s"]) == (2, 7.0)
    assert steps["detect"]["calls"] == 0
    assert steps["rework"]["searches"] == 1


def test_each_finding_adds_up_its_own_steps(run_dir) -> None:
    rows = report_run(locate(run_dir))["finding_rows"]
    assert set(rows) == {"A", "B"}
    a = rows["A"]
    assert a["wall_s"] == pytest.approx(85.0)
    assert a["steps"] == {"search": 60.0, "label": 10.0, "rework": 15.0}
    assert a["cost_usd"] == pytest.approx(0.09)
    assert (a["searches"], a["search_s"]) == (3, 10.5)
    assert a["cost_by_step_role"] == {
        "label:entailment": pytest.approx(0.05),
        "rework:generator": pytest.approx(0.01),
        "search:critic": pytest.approx(0.02),
        "search:generator": pytest.approx(0.01),
    }
    assert rows["B"]["cost_usd"] == pytest.approx(0.02)


def test_per_finding_figures_and_the_all_in_cost(run_dir) -> None:
    report = report_run(locate(run_dir))
    per = report["per_finding"]
    assert report["findings"] == 2
    assert per["cost_usd"]["mean"] == pytest.approx(0.055)
    assert per["wall_s"]["max"] == pytest.approx(85.0)
    # Materiality and the summary are the run's own; spread over two findings.
    assert per["run_level_cost_usd"] == pytest.approx(0.34)
    assert per["all_in_cost_usd"] == pytest.approx(0.45 / 2)
    # Top-level steps only: searches inside steps are not counted twice.
    assert report["wall_s"] == pytest.approx(30 + 25 + 60 + 10 + 15 + 40 + 0.5 + 12)


def test_models_report_latency_from_sent_calls_only(run_dir) -> None:
    models = report_run(locate(run_dir))["models"]
    haiku = models[HAIKU]
    assert (haiku["live_calls"], haiku["replayed_calls"], haiku["timed_calls"]) == (4, 1, 4)
    assert haiku["latency_s_median"] in {2.0, 3.0}
    assert models[OPUS]["latency_s_p90"] == 20.0


def test_worst_case_charges_are_counted_wherever_calls_are(tmp_path) -> None:
    folder = tmp_path / "runs" / "cut-off"
    ledger = CostLedger(10.0, path=folder / "ledger.jsonl")
    ledger.record(call("critic", HAIKU, 0.02, "search", "A"))
    ledger.record(
        LedgerEntry(
            "critic",
            HAIKU,
            10,
            2000,
            0,
            0,
            0.011,
            False,
            elapsed_s=5.0,
            stage="search",
            change_id="A",
            estimated=True,
        )
    )
    TimingLog(folder / "timings.jsonl").add(Timing("search", "A", 10.0))
    report = report_run(locate(folder))
    assert (report["estimated_calls"], report["estimated_usd"]) == (1, pytest.approx(0.011))
    assert report["steps"]["search"]["estimated_calls"] == 1
    assert report["finding_rows"]["A"]["estimated_calls"] == 1
    assert report["models"][HAIKU]["estimated_usd"] == pytest.approx(0.011)
    assert report["cost_usd"] == pytest.approx(0.031)


def test_a_run_with_no_worst_case_charges_counts_none(run_dir) -> None:
    report = report_run(locate(run_dir))
    assert (report["estimated_calls"], report["estimated_usd"]) == (0, 0)


def test_a_ledger_from_before_timing_is_reported_unscoped(tmp_path) -> None:
    folder = tmp_path / "old-run"
    CostLedger(1.0, path=folder / "ledger.jsonl").record(
        LedgerEntry("critic", HAIKU, 1, 1, 0, 0, 0.2, False)
    )
    report = report_run(locate(folder))
    assert not report["timed"]
    assert report["findings"] == 0
    assert report["unscoped_calls"] == 1
    assert list(report["steps"]) == [UNSCOPED]


def test_a_survival_output_finds_its_ledger_and_timings_beside_it(tmp_path) -> None:
    out = tmp_path / "hsy_numeric.json"
    files = locate(out)
    assert files.ledger == tmp_path / "hsy_numeric.ledger.jsonl"
    assert files.timings == tmp_path / "hsy_numeric.timings.jsonl"
    with pytest.raises(FileNotFoundError):
        report_run(files)
