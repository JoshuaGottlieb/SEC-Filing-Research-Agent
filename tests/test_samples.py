"""The saved runs committed under `samples/` still open in the app.

`samples/README.md` tells readers to browse these runs with the light install in
`samples/requirements-viewer.txt`, so every run must keep loading as the code changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from sec_agent import cli
from sec_agent.config import get_settings
from sec_agent.report import replay
from sec_agent.report import run_view as rv

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
RUNS = 9
# Each run renders all five tabs; give a slow shared machine room.
TIMEOUT = 120


def test_every_sample_run_reads() -> None:
    folders = rv.find_runs(SAMPLES / "runs")
    assert len(folders) == RUNS
    for folder in folders:
        run = rv.read_run(folder)
        assert rv.summarize(run).findings > 0, folder.name
        assert rv.review_items(run), folder.name
        assert replay.build_replay(run)["run"]["timed"], folder.name


@pytest.fixture
def samples_dir(monkeypatch):
    monkeypatch.setenv("SEC_AGENT_DATA_DIR", str(SAMPLES))
    get_settings.cache_clear()
    yield SAMPLES
    get_settings.cache_clear()


def test_the_app_opens_every_sample_run(samples_dir) -> None:
    before = sorted(p.relative_to(SAMPLES) for p in SAMPLES.rglob("*"))
    at = AppTest.from_file(str(cli.APP_SCRIPT), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    opened = set()
    for company in at.selectbox(key="company").options:
        at.selectbox(key="company").set_value(company).run()
        for comparison in at.selectbox(key="comparison").options:
            at.selectbox(key="comparison").set_value(comparison).run()
            assert not at.exception, (company, comparison, [e.value for e in at.exception])
            opened.add(at.selectbox(key="run").value)
    assert len(opened) == RUNS
    # Viewing writes nothing into the committed folder.
    assert sorted(p.relative_to(SAMPLES) for p in SAMPLES.rglob("*")) == before
