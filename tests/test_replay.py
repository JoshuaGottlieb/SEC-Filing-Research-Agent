"""The replay audit (T7): a saved run's steps in order, and the page they are shown on.

The runs are built in `conftest._saved_runs`; `timed_run` adds one with known times, generator
calls in the controller's order, and two rework rounds.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime, timedelta

import pytest

from conftest import MARGIN_ID, RUN_ID
from sec_agent.report import replay
from sec_agent.report import run_view as rv
from sec_agent.ui import replay_page

START = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


def kinds(data: dict) -> list[str]:
    return [step["kind"] for step in data["steps"]]


def step(data: dict, kind: str, nth: int = 0) -> dict:
    return [s for s in data["steps"] if s["kind"] == kind][nth]


def after(step: dict) -> dict[str, dict]:
    return {update["id"]: update for update in step["after"]}


def test_a_saved_run_replays_in_pipeline_order(saved_runs) -> None:
    data = replay.build_replay(rv.read_run(saved_runs / RUN_ID))
    # The change the run's timings show first, then the brief's other findings (no trees).
    assert [c["ref"] for c in data["changes"]] == ["F1", "F2", "F3"]
    assert kinds(data) == [
        "detect",
        "materiality",
        "change",
        "propose",
        "prescreen",
        "evidence",
        "evidence",
        "evidence",
        "label",
        "change",
        "change",
        "summary",
    ]
    margin = data["changes"][0]
    assert margin["tone"] == "confirmed"
    assert [n["depth"] for n in margin["nodes"]] == [0, 1, 1, 1, 2, 2, 2, 2]
    assert "This change's search tree was not saved." in data["steps"][9]["facts"]
    # One search timing and no generator calls: the search's time is shared out evenly.
    search = data["steps"][3:8]
    assert search[0]["start"] == 0.0
    assert [s["end"] for s in search[1:]] == [7.5, 15.0, 22.5, 30.0]
    assert data["steps"][-1]["totals"]["cost"] == 0.02
    # Writing the summary and saving the brief are one closing step.
    closing = data["steps"][-1]["facts"]
    assert any("in the review queue" in fact for fact in closing)
    assert any(fact.startswith("Model cost $") for fact in closing)
    assert data["overview"]["material"] == 4 and data["overview"]["explored"] == 3


def test_labels_appear_at_labeling_and_left_out_candidates_say_so(saved_runs) -> None:
    data = replay.build_replay(rv.read_run(saved_runs / RUN_ID))
    ids = {n["num"]: n["id"] for n in data["changes"][0]["nodes"] if n["depth"] == 1}
    labeled = after(step(data, "label"))
    assert labeled[ids[1]] == {"id": ids[1], "badge": "Confirmed [C1]", "tone": "confirmed"}
    assert labeled[ids[2]]["badge"] == "Plausible [C2]"
    assert labeled[ids[3]]["tone"] == "left_out"
    evidence = step(data, "evidence")
    assert evidence["focus"] == ids[1]
    assert len(evidence["reveal"]) == 2
    assert any(fact.startswith("2 searches: MD&A") for fact in evidence["facts"])


def test_the_detect_and_rank_steps_show_the_runs_own_changes(saved_runs) -> None:
    data = replay.build_replay(rv.read_run(saved_runs / RUN_ID))
    overview = data["overview"]
    # Text changes first, each side as runs of words flagged as changed or not.
    added, removed, metric = overview["alignments"]
    assert (added["ref"], added["prior"]) == ("F2", [])
    assert added["current"] == [["Tariffs on imported cocoa could raise our costs.", False]]
    assert (removed["ref"], removed["current"]) == ("F3", [])
    assert metric["ref"] == "F1" and "metric" in metric
    # The explored changes in the order the run took them, then those past the cap.
    assert [row["ref"] for row in overview["ranked"]] == ["F1", "F2", "F3"]
    assert [row["order"] for row in overview["ranked"]] == [1, 2, 3]
    (skipped,) = overview["skipped"]
    assert (skipped["rank"], skipped["importance"]) == (4, 3)
    assert skipped["heading"].startswith("Changed paragraph in Management's Discussion")
    # Each change carries a description; titles give the brief's number.
    assert data["changes"][2]["desc"] == "We operate a retail store in Hershey, Pennsylvania."
    titles = [s["title"] for s in data["steps"] if s["kind"] == "change"]
    assert titles[0] == "Change 1 of 3: Gross margin change: 33.5% in FY2025 (F1 in the brief)"
    # When the brief's numbers match the run's order, there is nothing to explain.
    facts = [fact for s in data["steps"] if s["kind"] == "change" for fact in s["facts"]]
    assert not any(fact.startswith("Listed in the brief") for fact in facts)


def test_a_run_without_timings_replays_without_a_clock(saved_runs) -> None:
    data = replay.build_replay(rv.read_run(saved_runs / "run-mat"))
    assert data["run"]["timed"] is False
    assert kinds(data) == ["detect", "materiality", "summary"]
    assert all(s["start"] is None and "totals" not in s for s in data["steps"])


def _line(at: datetime, **fields) -> str:
    return json.dumps({"at": at.isoformat(), **fields})


@pytest.fixture
def timed_run(saved_runs):
    """The HSY run with exact timings, generator calls in the controller's order, and two rework
    rounds for candidate 1."""
    folder = saved_runs / "timed"
    shutil.copytree(saved_runs / RUN_ID, folder)
    trail = json.loads((folder / rv.AUDIT_FILE).read_text())
    nodes = trail["trees"][MARGIN_ID]
    first = next(n for n in nodes if n["node_id"].endswith(":r1:c1:e1"))
    for number in (1, 2):
        nodes.append({**first, "node_id": f"{MARGIN_ID}:r1:c1:w{number}:e1"})
    (folder / rv.AUDIT_FILE).write_text(json.dumps(trail))
    at = lambda seconds: START + timedelta(seconds=seconds)  # noqa: E731
    change = "chg-gm"
    timings = [
        _line(at(1), stage="detect", change_id=None, elapsed_s=1.0, within=None),
        _line(at(5), stage="materiality", change_id=None, elapsed_s=4.0, within=None),
        _line(at(45), stage="search", change_id=change, elapsed_s=40.0, within=None),
        _line(at(55), stage="label", change_id=change, elapsed_s=10.0, within=None),
        _line(at(65), stage="rework", change_id=change, elapsed_s=10.0, within=None),
        _line(at(80), stage="rework", change_id=change, elapsed_s=15.0, within=None),
        _line(at(90), stage="summary", change_id=None, elapsed_s=10.0, within=None),
    ]
    call = dict(model="claude-haiku-5-5", replayed=False, elapsed_s=2.0, change_id=change)
    calls = [_line(at(4), role="materiality", stage="materiality", cost_usd=0.5, **call)]
    # Propose, then candidates 1, 2 and 3 (each has a search), one generator call each.
    for seconds in (8, 20, 30, 38):
        calls.append(_line(at(seconds), role="generator", stage="search", cost_usd=0.01, **call))
    calls.append(_line(at(9), role="critic", stage="search", cost_usd=0.002, **call))
    calls.append(_line(at(89), role="brief", stage="summary", cost_usd=0.1, **call))
    (folder / "timings.jsonl").write_text("\n".join(timings) + "\n")
    (folder / "ledger.jsonl").write_text("\n".join(calls) + "\n{not json\n")
    return folder


def test_search_steps_take_their_times_from_the_generator_calls(timed_run) -> None:
    data = replay.build_replay(rv.read_run(timed_run))
    assert data["run"]["timed"] is True
    assert (step(data, "detect")["start"], step(data, "detect")["end"]) == (0.0, 1.0)
    propose, prescreen = step(data, "propose"), step(data, "prescreen")
    # The search starts at 5 s; the first generator call ends at 8 s, the next starts at 18 s.
    assert (propose["start"], propose["end"]) == (5.0, 8.0)
    assert (prescreen["start"], prescreen["end"]) == (8.0, 18.0)
    assert [(s["start"], s["end"]) for s in data["steps"] if s["kind"] == "evidence"] == [
        (18.0, 28.0),
        (28.0, 36.0),
        (36.0, 45.0),
    ]
    assert prescreen["totals"]["roles"] == {"materiality": 1, "generator": 1, "critic": 1}
    assert data["steps"][-1]["totals"]["cost"] == pytest.approx(0.642)
    assert any("1 model call (claude-haiku-5-5)" in fact for fact in prescreen["facts"])


def test_a_reworked_claim_is_implied_at_labeling_and_labeled_after_its_last_round(
    timed_run,
) -> None:
    data = replay.build_replay(rv.read_run(timed_run))
    ids = {n["num"]: n["id"] for n in data["changes"][0]["nodes"] if n["depth"] == 1}
    assert after(step(data, "label"))[ids[1]]["tone"] == "implied"
    first, last = step(data, "rework", 0), step(data, "rework", 1)
    assert (first["title"], last["title"]) == (
        "Rework round 1: candidate #1",
        "Rework round 2: candidate #1",
    )
    assert (first["start"], first["end"], last["start"], last["end"]) == (55.0, 65.0, 65.0, 80.0)
    assert ids[1] not in after(first)
    assert "Still only implied; another round follows." in first["facts"]
    assert after(last)[ids[1]]["badge"] == "Confirmed [C1]"
    assert len(last["reveal"]) == 1


def test_the_notes_use_the_settings_the_code_runs_with() -> None:
    notes = replay.workflow_notes()
    assert set(notes) >= {"detect", "materiality", "propose", "prescreen", "evidence", "label"}
    assert "below 0.30 is pruned" in notes["prescreen"]
    assert "at most 2 rounds" in notes["rework"]


def test_the_page_carries_the_data_inertly(saved_runs) -> None:
    data = replay.build_replay(rv.read_run(saved_runs / RUN_ID))
    data["steps"][0]["title"] = "</script><script>alert(1)</script> & more"
    page = replay_page.render(data, dark=True)
    assert replay_page.SLOT not in page
    assert "alert(1)</script>" not in page
    assert page.count("</script>") == 2  # the data block's and the page script's own
    block = re.search(
        r'<script id="replay-data" type="application/json">(.*?)</script>', page, re.S
    )
    parsed = json.loads(block.group(1))
    assert parsed["steps"][0]["title"] == "</script><script>alert(1)</script> & more"
    assert parsed["theme"] == "dark"
    # The page sets text, never markup.
    assert "innerHTML" not in replay_page.PAGE.read_text()
