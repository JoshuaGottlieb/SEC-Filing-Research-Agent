"""The Streamlit app (T7): saved runs' briefs, evidence trees, a replay, review queue, and cost.

Start it with `sec-agent app` (or `streamlit run src/sec_agent/ui/streamlit_app.py`). It reads
the run folders under the data directory (`SEC_AGENT_DATA_DIR`, default
`~/.local/share/sec-agent`) and never calls a model, so browsing costs nothing. A citation's full
passage comes from the DuckDB chunk store, opened read-only for one lookup per run; if a pipeline
run holds the store open for writing, the app shows the quote alone.

The shaping is in `sec_agent.report.run_view`; this file only lays it out. Filing text and model
text are escaped before they reach Markdown, as in the brief.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sec_agent.config import get_settings
from sec_agent.report import replay
from sec_agent.report import run_view as rv
from sec_agent.report.brief import metric_line, ordered_claims, strongest_label
from sec_agent.report.names import LABEL_NAMES, SECTION_NAMES
from sec_agent.report.run_cost import locate, report_run
from sec_agent.schemas import Citation, ExplanationLabel, Finding, NodeStatus, TreeNode
from sec_agent.ui import replay_page

BADGE_COLOURS = {
    ExplanationLabel.CONFIRMED: "green",
    ExplanationLabel.PLAUSIBLE: "orange",
    ExplanationLabel.UNRESOLVED: "violet",
}
REVIEW_COLOURS = {
    rv.NEEDS_REVIEW: "red",
    rv.LEFT_OUT: "orange",
    rv.SHOWN_WITH_NOTE: "blue",
    rv.SEARCH_INCOMPLETE: "red",
    rv.RATIONALE_DROPPED: "gray",
    rv.NOT_EXAMINED: "gray",
}
# The review queue lists at most this many items of one kind; the rest are counted.
REVIEW_PAGE = 60
# The replay page's height in the app, in pixels; the tree scrolls inside it.
REPLAY_HEIGHT = 860


# --- Loading (cached: a run folder's files only change when the run is rewritten) ---------------


@st.cache_resource(show_spinner=False, max_entries=32)
def load_run(folder: str, mtime: float) -> rv.SavedRun:
    del mtime  # part of the cache key, so a rewritten run is read again
    return rv.read_run(Path(folder))


def _audit_mtime(folder: Path) -> float:
    return (folder / rv.AUDIT_FILE).stat().st_mtime


@st.cache_data(show_spinner=False, max_entries=64)
def _read_passages(chunk_ids: tuple[str, ...]) -> dict[str, str]:
    """The full text of these chunks. Raises if the store cannot be opened, so a failure is not
    cached and the next rerun tries again."""
    from sec_agent.finance.fact_store import FactStore

    with FactStore(get_settings().duckdb_path, read_only=True) as store:
        found = {chunk_id: store.get_chunk(chunk_id) for chunk_id in chunk_ids}
    return {chunk_id: chunk.text for chunk_id, chunk in found.items() if chunk is not None}


def load_passages(chunk_ids: tuple[str, ...]) -> dict[str, str] | None:
    """The chunks' full text, or None if the chunk store is missing or held by a writer."""
    if not get_settings().duckdb_path.exists():
        return None
    try:
        return _read_passages(chunk_ids)
    except Exception:  # noqa: BLE001 - locked by a running pipeline, or an older schema
        return None


@st.cache_data(show_spinner=False, max_entries=32)
def load_cost(folder: str, mtime: float) -> dict | None:
    del mtime
    files = locate(Path(folder))
    if not files.ledger.exists():
        return None
    return report_run(files)


@st.cache_data(show_spinner=False, max_entries=16)
def load_replay(folder: str, mtime: float) -> dict:
    return replay.build_replay(load_run(folder, mtime))


def runs_dir() -> Path:
    return get_settings().runs_dir


# --- Small pieces --------------------------------------------------------------------------------


# Filters are keyed by run (and the candidate picker by change), so each run opens with its own
# defaults: a key shared across runs would carry a filter emptied on one run over to the next.


def pick(place, label: str, choices: dict[str, str], key: str) -> str:
    """A selectbox over display labels that returns the chosen label's value.

    The options are the labels themselves, not values with a `format_func`, so a choice that
    disappears when the options change (another company) falls back to the first option cleanly,
    in the browser and in `AppTest` alike.
    """
    shown = place.selectbox(label, list(choices), key=key)
    return choices[shown]


def sentence(text: str) -> str:
    """Text with its first letter capitalized and the rest as written ("10-K" stays upper case)."""
    return text[:1].upper() + text[1:]


def ref_tags(refs: tuple[str, ...]) -> str:
    return " ".join(f"`{ref}`" for ref in refs)


def label_badge(label: ExplanationLabel) -> None:
    st.badge(LABEL_NAMES[label], color=BADGE_COLOURS[label])


def citation_title(citation: Citation) -> str:
    return (
        f"Quote · {citation.form.value} for {citation.period.label} · "
        f"{SECTION_NAMES[citation.section]}"
    )


def show_citation(citation: Citation, passages: dict[str, str] | None) -> None:
    with st.expander(citation_title(citation)):
        st.markdown(rv.md_quote(citation.quote, bold=True))
        passage = (passages or {}).get(citation.chunk_id)
        if passage is not None:
            st.caption("Full passage (the quote highlighted):")
            st.markdown(rv.highlight(passage, citation.quote))
        elif passages is None:
            st.caption("The chunk store could not be opened, so only the quote is shown.")
        st.markdown(f"[Open the filing on SEC.gov]({citation.url}) · chunk `{citation.chunk_id}`")


def run_chunk_ids(run: rv.SavedRun) -> tuple[str, ...]:
    ids = {
        citation.chunk_id
        for finding in run.brief.findings
        for claim in finding.claims
        for citation in claim.citations
    }
    ids |= {
        citation.chunk_id
        for tree in run.trees.values()
        for node in tree
        for citation in node.citations
    }
    return tuple(sorted(ids))


# --- Sidebar: choose a run -----------------------------------------------------------------------


def choose_run() -> rv.SavedRun | None:
    folders = rv.find_runs(runs_dir())
    summaries: list[rv.RunSummary] = []
    unreadable: list[str] = []
    for folder in folders:
        try:
            summaries.append(rv.summarize(load_run(str(folder), _audit_mtime(folder))))
        except Exception:  # noqa: BLE001 - a damaged or very old audit trail: list it, move on
            unreadable.append(folder.name)
    if not summaries:
        st.info(
            f"No saved runs in `{runs_dir()}`. Run `sec-agent analyze TICKER COMPARISON` to make "
            "one; its brief and audit trail are written there."
        )
        return None
    companies = {f"{s.company} ({s.ticker})": s.ticker for s in summaries}
    ticker = pick(st.sidebar, "Company", dict(sorted(companies.items())), "company")
    mine = [s for s in summaries if s.ticker == ticker]
    kinds = {sentence(s.comparison): s.comparison_type for s in mine}
    comparison = pick(st.sidebar, "Comparison", dict(sorted(kinds.items())), "comparison")
    candidates = sorted(
        (s for s in mine if s.comparison_type == comparison),
        key=lambda s: s.generated_at,
        reverse=True,
    )
    by_id = {s.run_id: s for s in candidates}
    run_id = pick(
        st.sidebar,
        "Run (newest first)",
        {f"{s.run_id} · ${s.cost_usd:.2f}": s.run_id for s in candidates},
        "run",
    )
    chosen = by_id[run_id]
    st.sidebar.caption(
        f"{len(summaries)} saved runs in `{runs_dir()}`. Nothing here calls a model."
    )
    if unreadable:
        st.sidebar.caption("Could not read: " + ", ".join(unreadable))
    return load_run(str(chosen.folder), _audit_mtime(chosen.folder))


# --- Tab 1: the brief (T7.1) ---------------------------------------------------------------------


def show_header(run: rv.SavedRun) -> None:
    summary = rv.summarize(run)
    comparison = run.brief.comparison
    st.title(f"{summary.company} ({summary.ticker})")
    st.markdown(
        f"**{sentence(summary.comparison)}**: {summary.periods}. Run `{run.run_id}`, generated "
        f"{run.brief.generated_at:%Y-%m-%d %H:%M} UTC."
    )
    cols = st.columns(4)
    cols[0].metric("Changes examined", summary.findings)
    cols[1].metric("Confirmed explanations", summary.labels.get("Confirmed", 0))
    cols[2].metric("Plausible explanations", summary.labels.get("Plausible", 0))
    cols[3].metric("Model cost", f"${summary.cost_usd:.2f}")
    st.markdown(
        f"Filings: [{comparison.current.form.value} for {comparison.current.period.label}]"
        f"({comparison.current.url}) against [{comparison.prior.form.value} for "
        f"{comparison.prior.period.label}]({comparison.prior.url})"
    )


def show_changed_text(run: rv.SavedRun, finding: Finding) -> None:
    """A text change's two versions side by side, the words that differ highlighted."""
    change = finding.change
    if not (change.prior_text or change.current_text):
        return
    comparison = run.brief.comparison
    removed, added = rv.changed_words(change.prior_text or "", change.current_text or "")
    with st.expander("The changed text"):
        if change.prior_text and change.current_text:
            st.caption(
                "Highlighted: the words that differ between the two versions (red in the "
                "earlier filing, green in the later one)."
            )
        cols = st.columns(2)
        cols[0].markdown(f"**Before ({comparison.prior.period.label})**")
        cols[0].markdown(rv.md_quote_runs(removed, rv.REMOVED_MARK) if removed else "(none)")
        cols[1].markdown(f"**After ({comparison.current.period.label})**")
        cols[1].markdown(rv.md_quote_runs(added, rv.ADDED_MARK) if added else "(none)")


def show_finding(run: rv.SavedRun, finding: Finding, passages: dict[str, str] | None) -> None:
    with st.container(border=True):
        top = st.columns([0.8, 0.2])
        top[0].markdown(
            f"#### {run.finding_refs.get(finding.finding_id, '')} · {rv.md(run.heading(finding))}"
        )
        with top[1]:
            label_badge(strongest_label(finding))
        rationale = rv.shown_rationale(finding)
        if rationale:
            st.markdown(f"**Why it was flagged:** {rv.md(rationale)}")
        show_changed_text(run, finding)
        if not finding.claims:
            st.markdown("_No explanation was supported._")
        for claim in ordered_claims(finding):
            cols = st.columns([0.14, 0.86])
            with cols[0]:
                label_badge(claim.label)
            cols[1].markdown(
                f"`{run.claim_refs.get(claim.claim_id, '?')}` {rv.md(claim.text)}"
                + (f" _(judge: {rv.words(claim.entailment.value)})_" if claim.entailment else "")
            )
            for metric_id in claim.metric_ids:
                cols[1].caption(rv.md(metric_line(metric_id, run.metrics)))
            with cols[1]:
                for citation in claim.citations:
                    show_citation(citation, passages)
        if finding.needs_review:
            st.warning(
                "Needs review: " + rv.md(finding.review_reason or "an explanation is unresolved")
            )


def brief_tab(run: rv.SavedRun, passages: dict[str, str] | None) -> None:
    st.info(run.brief.disclaimer, icon=":material/info:")
    st.subheader("Summary")
    lines = rv.summary_lines(run)
    st.markdown(
        "\n".join(f"- {rv.md(text)} {ref_tags(refs)}" for text, refs in lines) or "_No summary._"
    )
    if run.summary_used_template:
        st.caption("This summary was filled in from the fixed template; the writer's was not used.")
    st.subheader("Changes")
    present_labels = {
        LABEL_NAMES[label]: label
        for label in ExplanationLabel
        if any(strongest_label(f) is label for f in run.brief.findings)
    }
    present_sections = sorted({rv.section_of(f) for f in run.brief.findings})
    cols = st.columns(2)
    shown_labels = cols[0].multiselect(
        "Strongest explanation",
        list(present_labels),
        default=list(present_labels),
        key=f"labels:{run.run_id}",
    )
    labels = [present_labels[name] for name in shown_labels]
    sections = cols[1].multiselect(
        "Where", present_sections, default=present_sections, key=f"sections:{run.run_id}"
    )
    findings = rv.filter_findings(run, labels, sections)
    st.caption(f"Showing {len(findings)} of {len(run.brief.findings)} changes.")
    for finding in findings:
        show_finding(run, finding, passages)
    if run.not_examined is not None and run.not_examined.changes:
        st.caption(
            f"{len(run.not_examined.changes)} further material changes were past the run's cap of "
            f"{run.not_examined.max_changes} and were not examined; the Review queue lists them."
        )


# --- Tab 2: evidence trees (T7.2) ----------------------------------------------------------------


def show_branch(node: TreeNode, passages: dict[str, str] | None) -> None:
    status = {NodeStatus.KEPT: "green", NodeStatus.PRUNED: "gray", NodeStatus.PENDING: "red"}
    with st.container(border=True):
        cols = st.columns([0.15, 0.85])
        with cols[0]:
            st.badge(node.status.value, color=status[node.status])
        where = rv.source_name(node) or "a search of every section"
        evidence = rv.words(node.evidence.value) if node.evidence else "not checked"
        score = "-" if node.score is None else f"{node.score:.2f}"
        cols[1].markdown(
            f"**{rv.md(where)}** · score {score} · evidence: {evidence}  \n{rv.md(node.thought)}"
        )
        if node.prune_reason:
            cols[1].caption("Pruned: " + rv.md(node.prune_reason))
        with cols[1]:
            for citation in node.citations:
                show_citation(citation, passages)


def change_label(run: rv.SavedRun, finding: Finding) -> str:
    """A change as the pickers list it: reference, heading, and a few words on what it is."""
    label = f"{run.finding_refs.get(finding.finding_id, '')} · {run.heading(finding)}"
    description = rv.change_description(finding, limit=90)
    return f"{label} — {description}" if description else label


def trees_tab(run: rv.SavedRun, passages: dict[str, str] | None) -> None:
    with_trees = [f for f in rv.findings_in_brief_order(run) if f.finding_id in run.trees]
    if not with_trees:
        st.info("This run saved no evidence trees.")
        return
    finding_id = pick(
        st,
        "Change",
        {change_label(run, f): f.finding_id for f in with_trees},
        f"tree_finding:{run.run_id}",
    )
    finding = run.finding(finding_id)
    description = rv.change_description(finding, limit=400)
    if description:
        st.markdown(f"**What changed:** {rv.md(description)}")
    show_changed_text(run, finding)
    nodes = run.trees[finding_id]
    claims = {claim.claim_id: claim for claim in finding.claims}
    candidates = [n for n in nodes if n.depth == 1]
    cols = st.columns(4)
    cols[0].metric("Nodes", len(nodes))
    cols[1].metric("Candidate explanations", len(candidates))
    cols[2].metric("Pruned", sum(1 for n in nodes if n.status is NodeStatus.PRUNED))
    cols[3].metric("Still pending", sum(1 for n in nodes if n.status is NodeStatus.PENDING))

    choices = rv.candidate_choices(nodes, claims)
    open_key = f"tree_open:{finding_id}"

    def open_all() -> None:
        st.session_state[open_key] = list(choices)

    def close_all() -> None:
        st.session_state[open_key] = []

    opened = st.pills(
        "Open a candidate's branch to see its searches",
        list(choices),
        selection_mode="multi",
        key=open_key,
    )
    with st.container(horizontal=True, vertical_alignment="center"):
        st.button("Open all", on_click=open_all, key=f"open_all:{finding_id}")
        st.button("Close all", on_click=close_all, key=f"close_all:{finding_id}")
        show_pruned = st.toggle("Show pruned branches", value=True, key="show_pruned")
    view = rv.tree_view(
        nodes,
        claims,
        run.claim_refs,
        show_pruned=show_pruned,
        expanded={choices[label] for label in opened or ()},
    )
    st.caption(
        "Top to bottom: the change, then one card per candidate explanation the generator "
        "proposed. A card's chips are its evidence searches, one per search, coloured by what "
        "the search found; solid chips were kept, hollow ones pruned. Open a candidate to see "
        "each search as its own box. A card outlined in green or orange reached the brief as a "
        "confirmed or plausible claim (✓ in the list above); dashed cards were pruned (✗). Hover "
        "a box for its full text; the chart's fullscreen button zooms in."
    )
    # The legend's text and colours are the app's own constants, not run text.
    st.markdown(
        " &nbsp; ".join(
            f"<span style='color:{style.colour}'>■</span> {style.meaning}"
            for style in rv.CHIP_STYLES.values()
        )
        + " &nbsp; · &nbsp; □ hollow: pruned",
        unsafe_allow_html=True,
    )
    st.graphviz_chart(view.dot, width="stretch")
    with st.expander(f"Every node as a table ({len(view.rows)} rows)"):
        st.dataframe(pd.DataFrame(view.rows), hide_index=True, width="stretch")
    st.subheader("A candidate's evidence")
    if candidates:
        candidate_id = pick(
            st,
            "Candidate explanation",
            {f"{i}. {rv.shorten(n.thought, 140)}": n.node_id for i, n in enumerate(candidates, 1)},
            f"tree_candidate:{finding_id}",
        )
        for branch in rv.candidate_evidence(nodes, candidate_id):
            show_branch(branch, passages)


# --- Tab 3: the replay ---------------------------------------------------------------------------


def replay_tab(run: rv.SavedRun) -> None:
    st.markdown(
        "The run step by step, as it happened: detecting the changes, ranking them, then each "
        "change's Tree-of-Thought search, labeling and rework, and the summary. Built from the "
        "run's saved timings, cost ledger and audit trail; nothing is run again. Press play, "
        "step with the arrows, or jump to a change."
    )
    data = load_replay(str(run.folder), _audit_mtime(run.folder))
    if not data["run"]["timed"]:
        st.caption(
            "This run was saved before step timings were recorded, so the replay has no clock or "
            "running cost, and each step's length is fixed."
        )
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    # The page is the app's own template; run text is only in its escaped JSON, set as text.
    st.iframe(replay_page.render(data, dark=theme == "dark"), height=REPLAY_HEIGHT)


# --- Tab 4: the review queue (T7.3) --------------------------------------------------------------


def review_tab(run: rv.SavedRun, passages: dict[str, str] | None) -> None:
    items = rv.review_items(run)
    counts = {kind: sum(1 for item in items if item.kind == kind) for kind in rv.REVIEW_KINDS}
    st.markdown(
        "Items to check by hand. None of them stopped the run: unresolved items are listed, not "
        "waited on (the design's non-blocking escalation)."
    )
    present = [kind for kind in rv.REVIEW_KINDS if counts[kind]]
    if not present:
        st.success("Nothing in this run needs review.")
        return
    st.dataframe(
        pd.DataFrame({"kind": present, "items": [counts[kind] for kind in present]}),
        hide_index=True,
    )
    default = [kind for kind in present if kind != rv.NOT_EXAMINED] or present
    chosen = st.multiselect("Show", present, default=default, key=f"review_kinds:{run.run_id}")
    trees = {node.node_id: tree for tree in run.trees.values() for node in tree}
    for kind in chosen:
        mine = [item for item in items if item.kind == kind]
        st.subheader(f"{kind} ({len(mine)})")
        for item in mine[:REVIEW_PAGE]:
            with st.container(border=True):
                cols = st.columns([0.75, 0.25])
                cols[0].markdown(f"**{rv.md(item.subject)}**")
                if item.finding_id:
                    cols[1].caption(
                        f"Change {run.finding_refs.get(item.finding_id, '')} · "
                        f"{rv.md(run.heading(run.finding(item.finding_id)))}"
                    )
                cols[0].markdown("\n".join(f"- {rv.md(reason)}" for reason in item.reasons))
                if item.candidate_id and item.candidate_id in trees:
                    branches = rv.candidate_evidence(trees[item.candidate_id], item.candidate_id)
                    quotes = [c for branch in branches for c in branch.citations]
                    if quotes:
                        st.caption(f"Evidence this explanation found ({len(quotes)} quotes):")
                        for citation in quotes:
                            show_citation(citation, passages)
                    else:
                        st.caption("Its evidence searches found no quote that passed the checks.")
        if len(mine) > REVIEW_PAGE:
            st.caption(f"{len(mine) - REVIEW_PAGE} more not shown; `audit.json` lists them all.")


# --- Tab 5: cost and time ------------------------------------------------------------------------


def cost_tab(run: rv.SavedRun) -> None:
    folder = run.folder
    report = load_cost(
        str(folder),
        (folder / "ledger.jsonl").stat().st_mtime if (folder / "ledger.jsonl").exists() else 0.0,
    )
    if report is None:
        st.info("This run has no cost ledger.")
        return
    cols = st.columns(4)
    cols[0].metric("Model cost", f"${report['cost_usd']:.2f}")
    wall = report["wall_s"]
    cols[1].metric("Wall time", "-" if wall is None else f"{wall / 60:.0f} min")
    cols[2].metric("Model calls sent", report["live_calls"])
    cols[3].metric("Replayed from cache", report["replayed_calls"])
    if report["estimated_calls"]:
        st.caption(
            f"{report['estimated_calls']} unparsable replies were charged their worst case, so "
            f"\\${report['estimated_usd']:.4f} of the cost is an upper bound."
        )
    st.subheader("By step")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "step": step,
                    "runs": row["runs"],
                    "wall (s)": None if row["wall_s"] is None else round(row["wall_s"], 1),
                    "calls sent": row["live_calls"],
                    "replayed": row["replayed_calls"],
                    "cost ($)": round(row["cost_usd"], 4),
                    "searches": row["searches"],
                }
                for step, row in report["steps"].items()
            ]
        ),
        hide_index=True,
    )
    per = report["per_finding"]
    if report["findings"]:
        st.subheader("Per change searched")
        cost, wall_s = per["cost_usd"], per["wall_s"]
        st.markdown(
            f"{report['findings']} changes: cost mean \\${cost['mean']:.3f}, median "
            f"\\${cost['median']:.3f}, max \\${cost['max']:.3f}; wall time mean "
            f"{wall_s['mean'] / 60:.1f} min, max {wall_s['max'] / 60:.1f} min. All-in (with "
            f"materiality and the summary spread over them): \\${per['all_in_cost_usd']:.3f}."
        )
    st.subheader("By model")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "model": model,
                    "calls sent": row["live_calls"],
                    "cost ($)": round(row["cost_usd"], 4),
                    "median call (s)": row["latency_s_median"],
                    "p90 call (s)": row["latency_s_p90"],
                }
                for model, row in report["models"].items()
            ]
        ),
        hide_index=True,
    )
    st.caption(
        "Median call: half of that model's calls took less time than this. p90 call: the 90th "
        "percentile, so 9 in 10 calls finished within this time and the slowest tenth took "
        "longer. Replayed calls are left out of both."
    )


# --- Page ----------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="SEC filing research briefs", page_icon="📄", layout="wide")
    run = choose_run()
    if run is None:
        return
    show_header(run)
    passages = load_passages(run_chunk_ids(run))
    brief, trees, replay_view, review, cost = st.tabs(
        ["Brief", "Evidence trees", "Replay audit", "Review queue", "Cost and time"]
    )
    with brief:
        brief_tab(run, passages)
    with trees:
        trees_tab(run, passages)
    with replay_view:
        replay_tab(run)
    with review:
        review_tab(run, passages)
    with cost:
        cost_tab(run)


main()
