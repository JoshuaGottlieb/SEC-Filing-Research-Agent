"""The Streamlit app (T7) on saved runs, driven headless with Streamlit's `AppTest`.

The runs are built in `conftest._saved_runs`; the data directory points at them, and there is no
chunk store, so a citation shows its quote alone.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from sec_agent import cli
from sec_agent.config import get_settings
from sec_agent.report import run_view as rv

# The first run imports the package (about 3 s); give a slow shared machine room.
TIMEOUT = 60


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SEC_AGENT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def app(data_dir, saved_runs):
    at = AppTest.from_file(str(cli.APP_SCRIPT), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def texts(at) -> str:
    return "\n".join(element.value for kind in (at.markdown, at.caption) for element in kind)


def first_metrics(at) -> dict[str, str]:
    """Each metric's value by label; the header's "Model cost" comes before the cost tab's."""
    found: dict[str, str] = {}
    for metric in at.metric:
        found.setdefault(metric.label, metric.value)
    return found


def test_it_opens_on_a_saved_runs_brief(app) -> None:
    # Companies are listed by name, so Mattel comes first.
    assert app.selectbox(key="company").value == "Mattel, Inc. (MAT)"
    app.selectbox(key="company").set_value("The Hershey Company (HSY)").run()
    assert not app.exception
    assert app.title[0].value == "The Hershey Company (HSY)"
    assert [tab.label for tab in app.tabs] == [
        "Brief",
        "Evidence trees",
        "Review queue",
        "Cost and time",
    ]
    metrics = first_metrics(app)
    assert metrics["Changes examined"] == "3"
    assert metrics["Confirmed explanations"] == "1"
    assert metrics["Model cost"] == "$0.42"
    body = texts(app)
    assert "Higher input costs lowered gross margin." in body
    assert "Showing 3 of 3 changes." in body
    # Dollar amounts are escaped, so Streamlit does not read them as LaTeX.
    assert rv.md(rv.shorten("Net sales rose to $2.8 billion", 30)) in body
    assert "The chunk store could not be opened" in body


def test_the_brief_filters_by_strongest_label(app) -> None:
    app.selectbox(key="company").set_value("The Hershey Company (HSY)").run()
    app.multiselect(key="labels:run1").set_value(["Plausible"]).run()
    assert "Showing 1 of 3 changes." in texts(app)
    app.multiselect(key="sections:run1").set_value([]).run()
    assert "Showing 0 of 3 changes." in texts(app)


def test_the_company_and_comparison_choose_the_run(app) -> None:
    app.selectbox(key="company").set_value("Mattel, Inc. (MAT)").run()
    assert not app.exception
    assert app.title[0].value == "Mattel, Inc. (MAT)"
    assert app.selectbox(key="run").value.startswith("run-mat")
    assert "This run saved no evidence trees." in [info.value for info in app.info]
    assert "Nothing in this run needs review." in [s.value for s in app.success]


def test_the_tree_tab_can_hide_pruned_branches(app) -> None:
    app.selectbox(key="company").set_value("The Hershey Company (HSY)").run()
    rows = lambda: len(app.dataframe[0].value)  # noqa: E731 - the tree's node table
    assert rows() == 8
    app.toggle(key="show_pruned").set_value(False).run()
    assert not app.exception
    assert rows() == 5


def test_the_review_queue_hides_unexamined_changes_until_asked(app) -> None:
    app.selectbox(key="company").set_value("The Hershey Company (HSY)").run()
    shown = app.multiselect(key="review_kinds:run1")
    assert rv.NOT_EXAMINED in shown.options
    assert rv.NOT_EXAMINED not in shown.value
    headings = [s.value for s in app.subheader]
    assert f"{rv.LEFT_OUT} (1)" in headings
    assert not any(h.startswith(rv.NOT_EXAMINED) for h in headings)
    shown.set_value(list(shown.options)).run()
    assert f"{rv.NOT_EXAMINED} (1)" in [s.value for s in app.subheader]


def test_the_cost_tab_reads_the_ledger(app) -> None:
    app.selectbox(key="company").set_value("The Hershey Company (HSY)").run()
    metrics = first_metrics(app)
    assert metrics["Model calls sent"] == "1"
    assert metrics["Wall time"] == "0 min"


def test_with_no_saved_runs_it_says_how_to_make_one(data_dir) -> None:
    at = AppTest.from_file(str(cli.APP_SCRIPT), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert "sec-agent analyze" in at.info[0].value


def test_the_app_command_starts_streamlit_on_the_app(monkeypatch) -> None:
    calls = []

    class Done:
        returncode = 3

    monkeypatch.setattr(cli.subprocess, "run", lambda command, check: calls.append(command) or Done)
    result = CliRunner().invoke(cli.app, ["app", "--port", "8600"])
    assert result.exit_code == 3
    (command,) = calls
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert command[4] == str(cli.APP_SCRIPT)
    assert command[5:7] == ["--server.port", "8600"]
    assert cli.APP_SCRIPT.exists()
