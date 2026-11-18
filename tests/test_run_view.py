"""Saved runs read back for the Streamlit app (T7): runs, briefs, trees and the review queue.

The runs are built in `conftest._saved_runs`.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from conftest import DOLLAR_QUOTE, GONE_ID, MARGIN_ID, RISK_ID, RUN_ID
from sec_agent.report import run_view as rv
from sec_agent.report.brief import assign_refs
from sec_agent.schemas import ExplanationLabel, NodeStatus, NumericCheck


def test_runs_with_an_audit_trail_are_listed_newest_first(saved_runs) -> None:
    os.utime(saved_runs / "run-mat" / rv.AUDIT_FILE, (1, 1))
    assert [folder.name for folder in rv.find_runs(saved_runs)] == [RUN_ID, "run-mat"]
    assert rv.find_runs(saved_runs / "missing") == []


def test_a_saved_run_reads_back_with_the_briefs_references(saved_runs) -> None:
    run = rv.read_run(saved_runs / RUN_ID)
    refs = assign_refs(run.brief.findings)
    assert run.finding_refs == refs.findings
    assert run.claim_refs == refs.claims
    assert len(run.trees[MARGIN_ID]) == 8
    assert run.metrics.get_metric("gm-25").value == 0.335
    assert run.not_examined.max_changes == 3
    assert run.summary_used_template is False


def test_the_run_picker_line(saved_runs) -> None:
    summary = rv.summarize(rv.read_run(saved_runs / RUN_ID))
    assert (summary.ticker, summary.company) == ("HSY", "The Hershey Company")
    assert summary.comparison == "year over year (10-K)"
    assert summary.periods == "FY2025 against FY2024"
    assert summary.labels == {"Plausible": 2, "Confirmed": 1}
    assert summary.findings == 3


def test_summary_lines_keep_their_references(saved_runs) -> None:
    assert rv.summary_lines(rv.read_run(saved_runs / RUN_ID)) == [
        ("Higher input costs lowered gross margin.", ("C1",)),
        ("A tariff risk was added.", ("C3",)),
    ]


def test_findings_filter_by_strongest_label_and_section(saved_runs) -> None:
    run = rv.read_run(saved_runs / RUN_ID)
    ids = lambda findings: [f.finding_id for f in findings]  # noqa: E731
    # Brief order: strongest label first.
    assert ids(rv.filter_findings(run)) == [MARGIN_ID, RISK_ID, GONE_ID]
    assert ids(rv.filter_findings(run, labels=[ExplanationLabel.PLAUSIBLE])) == [RISK_ID]
    assert ids(rv.filter_findings(run, labels=[ExplanationLabel.UNRESOLVED])) == [GONE_ID]
    assert ids(rv.filter_findings(run, sections=[rv.METRIC_SECTION])) == [MARGIN_ID]
    assert ids(rv.filter_findings(run, sections=["Risk Factors"])) == [RISK_ID]
    assert rv.filter_findings(run, labels=[]) == []


def test_text_for_streamlit_markdown_is_inert_including_dollar_signs() -> None:
    assert rv.md(DOLLAR_QUOTE).count("\\$") == 3
    assert rv.md("[link](javascript:x) *bold*") == "\\[link\\](javascript:x) \\*bold\\*"
    quoted = rv.md_quote("First $1 line\n\n# not a heading")
    assert quoted == "> First \\$1 line\n> \\# not a heading"


def test_a_passage_highlights_its_quote() -> None:
    passage = "Intro.  Gross margin fell\n by $5 million. After."
    assert rv.highlight(passage, "Gross margin fell by $5 million.") == (
        "Intro. :orange-background[**Gross margin fell by \\$5 million.**] After."
    )
    # A quote not found word for word is not guessed at.
    assert (
        rv.highlight(passage, "something else")
        == "Intro. Gross margin fell by \\$5 million. After."
    )


def test_a_quote_can_be_shown_in_bold() -> None:
    assert (
        rv.md_quote("First *line*\n\nSecond", bold=True) == "> **First \\*line\\***\n> **Second**"
    )


def test_only_the_words_that_differ_are_marked() -> None:
    prior = "Net sales were $3,104.2 million in the first quarter, an increase of 10.6%."
    current = "Net sales were $2,787.3 million in the second quarter, an increase of 6.6%."
    removed, added = rv.changed_words(prior, current)
    assert removed == [
        ("Net sales were", False),
        ("$3,104.2", True),
        ("million in the", False),
        ("first", True),
        ("quarter, an increase of", False),
        ("10.6%.", True),
    ]
    assert [text for text, changed in added if changed] == ["$2,787.3", "second", "6.6%."]


def test_a_short_unchanged_stretch_between_changes_is_marked_with_them() -> None:
    _, added = rv.changed_words("Costs rose on cocoa prices.", "Costs fell by a lot on tariffs.")
    # "on" alone between two changes would be confetti; it joins them.
    assert added == [("Costs", False), ("fell by a lot on tariffs.", True)]


def test_paragraph_breaks_survive_and_one_sided_text_is_not_marked() -> None:
    removed, added = rv.changed_words("First line.\n\nSecond   line.", "")
    assert removed == [("First line.", False), ("\n", False), ("Second line.", False)]
    assert added == []
    assert rv.md_quote_runs(removed, rv.REMOVED_MARK) == "> First line.\n> Second line."
    _, added = rv.changed_words("A rose $1.", "A fell $2.")
    assert rv.md_quote_runs(added, rv.ADDED_MARK) == "> A :green-background[fell \\$2.]"


def test_a_change_is_described_beyond_its_heading(saved_runs) -> None:
    run = rv.read_run(saved_runs / RUN_ID)
    assert rv.change_description(run.finding(GONE_ID)) == (
        "We operate a retail store in Hershey, Pennsylvania."
    )
    assert len(rv.change_description(run.finding(RISK_ID), limit=20)) <= 20


def tree(saved_runs, **options):
    run = rv.read_run(saved_runs / RUN_ID)
    claims = {claim.claim_id: claim for claim in run.finding(MARGIN_ID).claims}
    return rv.tree_view(run.trees[MARGIN_ID], claims, run.claim_refs, **options)


def candidate(number: int) -> str:
    return f"{MARGIN_ID}:r1:c{number}"


def drawn(dot: str) -> int:
    """How many tree nodes the DOT draws (notes and rework headers are plain text)."""
    return sum(1 for line in dot.splitlines() if "[label=" in line and "plaintext" not in line)


def test_the_tree_draws_the_change_and_one_card_per_candidate(saved_runs) -> None:
    view = tree(saved_runs)
    assert (view.nodes, view.pruned, view.pending, view.hidden) == (8, 3, 1, 0)
    assert 'rankdir="TB"' in view.dot
    # The root and three candidate cards; the searches are chips on the cards, not boxes.
    assert drawn(view.dot) == 4
    assert "cluster_" not in view.dot
    assert "Confirmed [C1]" in view.dot and "Plausible [C2]" in view.dot
    assert rv.LABEL_COLOURS[ExplanationLabel.CONFIRMED] in view.dot
    # Candidate 1's two searches: a kept one that states the cause, a pruned one with nothing.
    assert f'BGCOLOR="{rv.CHIP_STYLES["states_cause"].colour}"' in view.dot
    assert f'BORDER="2" COLOR="{rv.CHIP_STYLES["none"].colour}"' in view.dot
    assert "2 searches · 1 kept · 1 quotes" in view.dot
    assert "dashed" in view.dot
    # A quote inside a thought cannot end the label or the tooltip early.
    assert "&quot;sharply&quot;" in view.dot
    assert '\\"sharply\\"' in view.dot
    # A pruned card's tooltip gives the claim and why it was pruned, on one line.
    assert 'tooltip="A plant closure lowered margin. — Pruned: below the pre-screen threshold"' in (
        view.dot
    )
    # Every node is still in the table.
    assert len(view.rows) == 8
    reasons = {row["prune reason"] for row in view.rows}
    assert "below the pre-screen threshold" in reasons
    assert {row["evidence"] for row in view.rows} >= {"states cause", "none"}
    assert {row["candidate"] for row in view.rows if row["depth"] == 2} == {1, 2, 3}


def test_an_opened_candidate_shows_its_searches_in_a_column(saved_runs) -> None:
    view = tree(saved_runs, expanded={candidate(1)})
    assert drawn(view.dot) == 6
    assert view.dot.count("subgraph cluster_") == 1
    assert "Candidate 1: evidence searches" in view.dot
    assert "MD&amp;A" in view.dot and "Risk factors" in view.dot
    # The second search hangs under the first, so the branch grows down, not across.
    assert "[style=invis]" in view.dot
    assert "opened below" in view.dot
    every = tree(saved_runs, expanded={candidate(1), candidate(2), candidate(3)})
    assert every.dot.count("subgraph cluster_") == 3


def test_hiding_pruned_branches_drops_their_subtrees_and_says_how_many(saved_runs) -> None:
    view = tree(saved_runs, show_pruned=False, expanded={candidate(1)})
    assert view.hidden == 3  # the pruned candidate, its child, and a pruned search
    assert len(view.rows) == 5
    assert all(row["status"] != NodeStatus.PRUNED.value for row in view.rows)
    assert "1 pruned candidate hidden" in view.dot
    assert "1 pruned hidden" in view.dot  # candidate 1's pruned search
    assert "plant closure" not in view.dot
    assert "Risk factors" not in view.dot


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz's dot is not installed")
def test_every_tree_drawing_is_valid_graphviz(saved_runs) -> None:
    every = {candidate(1), candidate(2), candidate(3)}
    for options in ({}, {"expanded": every}, {"expanded": every, "show_pruned": False}):
        result = subprocess.run(
            ["dot", "-Tsvg"], input=tree(saved_runs, **options).dot, capture_output=True, text=True
        )
        assert result.returncode == 0 and not result.stderr, result.stderr


def test_the_branch_picker_numbers_candidates_and_marks_their_fate(saved_runs) -> None:
    run = rv.read_run(saved_runs / RUN_ID)
    claims = {claim.claim_id: claim for claim in run.finding(MARGIN_ID).claims}
    choices = rv.candidate_choices(run.trees[MARGIN_ID], claims)
    assert list(choices.values()) == [candidate(1), candidate(2), candidate(3)]
    labels = list(choices)
    assert labels[0].startswith("1 ✓ · Cocoa costs rose")
    assert labels[1].startswith("2 ✓ · ")
    assert labels[2].startswith("3 ✗ · ")


def test_searches_are_grouped_by_rework_round_and_coloured_by_finding(saved_runs) -> None:
    nodes = rv.read_run(saved_runs / RUN_ID).trees[MARGIN_ID]
    by_suffix = {node.node_id.removeprefix(MARGIN_ID + ":"): node for node in nodes}
    first = by_suffix["r1:c1:e1"]
    rework = first.model_copy(update={"node_id": f"{candidate(1)}:w2:e1"})
    assert rv.rework_round(first) == 0
    assert rv.rework_round(rework) == 2
    rounds = rv.searches_of([*nodes, rework], candidate(1))
    assert list(rounds) == [0, 2]
    assert [rv.chip_kind(node) for node in rounds[0]] == ["states_cause", "none"]
    assert rv.chip_kind(by_suffix["r1:c2:e1"]) == "pending"
    agrees = first.model_copy(update={"evidence": None, "numeric_check": NumericCheck.AGREE})
    assert rv.chip_kind(agrees) == "metric_agrees"
    assert rv.short_source_name(first) == "MD&A"
    assert rv.short_source_name(by_suffix["r1:c2:e1"]) == "Everywhere"


def test_a_candidates_evidence_lists_kept_searches_first(saved_runs) -> None:
    run = rv.read_run(saved_runs / RUN_ID)
    children = rv.candidate_evidence(run.trees[MARGIN_ID], f"{MARGIN_ID}:r1:c1")
    assert [node.status for node in children] == [NodeStatus.KEPT, NodeStatus.PRUNED]
    assert children[0].citations


def test_the_review_queue_lists_everything_to_check_most_pressing_first(saved_runs) -> None:
    items = rv.review_items(rv.read_run(saved_runs / RUN_ID))
    assert [item.kind for item in items] == [
        rv.NEEDS_REVIEW,
        rv.LEFT_OUT,
        rv.SHOWN_WITH_NOTE,
        rv.SEARCH_INCOMPLETE,
        rv.SEARCH_INCOMPLETE,
        rv.RATIONALE_DROPPED,
        rv.NOT_EXAMINED,
    ]
    review, left_out, noted, rework, pending, rationale, skipped = items
    assert (review.subject, review.reasons) == (
        "Removed paragraph in Business",
        ("every candidate was pruned",),
    )
    # Left out: the candidate as proposed, with its evidence branch to show.
    assert left_out.subject == "A plant closure lowered margin."
    assert left_out.candidate_id == f"{MARGIN_ID}:r1:c3"
    assert left_out.reasons[0].startswith("the judge found every quote unrelated")
    # Shown: the claim as the brief words it, under its reference.
    assert noted.subject.startswith("[C2] Promotions rose")
    assert rework.reasons == ("the critic failed: the reply was cut off",)
    assert rework.finding_id == RISK_ID
    assert "1 evidence search never finished" in pending.reasons[0]
    assert rationale.reasons == ("the reason it was flagged cites a figure nothing backs: 12.4%",)
    assert skipped.subject == (
        "Changed paragraph in Management's Discussion and Analysis (paragraph 0044)"
    )
    assert skipped.reasons == (
        "materiality rank 4, importance 3; the run searches at most 3 changes",
    )


def test_a_run_with_nothing_to_check_has_an_empty_queue(saved_runs) -> None:
    assert rv.review_items(rv.read_run(saved_runs / "run-mat")) == []


def test_change_ids_read_as_words() -> None:
    assert (
        rv.describe_change_id("MAT:yoy-10k:numeric:gross_margin") == "Metric change: Gross margin"
    )
    assert rv.describe_change_id("MAT:yoy-10k:added:acc:risk_factors:0059") == (
        "New paragraph in Risk Factors (paragraph 0059)"
    )
    assert rv.describe_change_id("odd-id") == "odd-id"
