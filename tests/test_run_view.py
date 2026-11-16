"""Saved runs read back for the Streamlit app (T7): runs, briefs, trees and the review queue.

The runs are built in `conftest._saved_runs`.
"""

from __future__ import annotations

import os

from conftest import DOLLAR_QUOTE, GONE_ID, MARGIN_ID, RISK_ID, RUN_ID
from sec_agent.report import run_view as rv
from sec_agent.report.brief import assign_refs
from sec_agent.schemas import ExplanationLabel, NodeStatus


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


def test_a_passage_shows_its_quote_in_bold() -> None:
    passage = "Intro.  Gross margin fell\n by $5 million. After."
    assert rv.highlight(passage, "Gross margin fell by $5 million.") == (
        "Intro. **Gross margin fell by \\$5 million.** After."
    )
    # A quote not found word for word is not guessed at.
    assert (
        rv.highlight(passage, "something else")
        == "Intro. Gross margin fell by \\$5 million. After."
    )


def tree(saved_runs, **options):
    run = rv.read_run(saved_runs / RUN_ID)
    claims = {claim.claim_id: claim for claim in run.finding(MARGIN_ID).claims}
    return rv.tree_view(run.trees[MARGIN_ID], claims, run.claim_refs, **options)


def test_the_tree_draws_every_node_and_marks_claims_and_pruned_branches(saved_runs) -> None:
    view = tree(saved_runs)
    assert (view.nodes, view.pruned, view.pending, view.hidden) == (8, 3, 1, 0)
    assert view.dot.count("[label=") == 8
    assert "Confirmed [C1]" in view.dot and "Plausible [C2]" in view.dot
    assert rv.LABEL_COLOURS[ExplanationLabel.CONFIRMED] in view.dot
    assert "dashed" in view.dot
    # A quote inside a thought cannot end the DOT string early.
    assert '\\"sharply\\"' in view.dot
    reasons = {row["prune reason"] for row in view.rows}
    assert "below the pre-screen threshold" in reasons
    assert {row["evidence"] for row in view.rows} >= {"states cause", "none"}


def test_hiding_pruned_branches_drops_their_subtrees_and_says_how_many(saved_runs) -> None:
    view = tree(saved_runs, show_pruned=False)
    assert view.hidden == 3  # the pruned candidate, its child, and a pruned search
    assert len(view.rows) == 5
    assert all(row["status"] != NodeStatus.PRUNED.value for row in view.rows)
    assert view.dot.count("1 pruned branch hidden") == 2
    assert "plant closure" not in view.dot


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
