"""Saved runs read back for the Streamlit app (T7): briefs, evidence trees and the review queue.

Everything here works from a run folder's `audit.json`, which `brief.write_run` writes beside
`brief.md` and which holds every record behind the brief. Nothing is recomputed and no model is
called, so the app costs nothing to use and shows exactly what the run produced.

This module has no Streamlit code, so the shaping can be tested on its own; `sec_agent.ui` draws
it. Text is returned raw unless a function says it is Markdown, which is escaped with
`brief.escape` the same way the brief escapes it: filing text and model text are untrusted.
"""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sec_agent.report.brief import (
    NotExamined,
    assign_refs,
    change_heading,
    escape,
    grouped,
    parse_summary_line,
    quote_lines,
    rationale_shown,
    strongest_label,
)
from sec_agent.report.names import COMPARISON_NAMES, LABEL_NAMES, SECTION_NAMES, metric_name
from sec_agent.schemas import (
    Brief,
    ChangeKind,
    Claim,
    ExplanationLabel,
    Finding,
    MetricValue,
    NodeStatus,
    TreeNode,
)

AUDIT_FILE = "audit.json"
# What a metric change is filed under where the app filters findings by section.
METRIC_SECTION = "Metric changes"


class AuditMetrics:
    """The metrics a saved run stored, as a `MetricLookup`, so the brief's helpers can name them."""

    def __init__(self, metrics: Mapping[str, MetricValue]) -> None:
        self._metrics = dict(metrics)

    def get_metric(self, metric_id: str) -> MetricValue | None:
        return self._metrics.get(metric_id)


@dataclass(frozen=True)
class SavedRun:
    """One run's `audit.json`, as records."""

    folder: Path
    brief: Brief
    # Full id -> short reference ("F1", "C3"), as `brief.md` prints them.
    finding_refs: dict[str, str]
    claim_refs: dict[str, str]
    trees: dict[str, tuple[TreeNode, ...]]
    metrics: AuditMetrics
    labeling_failures: dict[str, tuple[str, ...]]
    not_examined: NotExamined | None
    rationales_omitted: tuple[str, ...]
    # Materiality decisions whose rationale was dropped for a figure nothing backs.
    rationales_untraced: tuple[dict, ...]
    # None for a run from before the audit trail recorded how the summary was made.
    summary_used_template: bool | None
    summary_failures: tuple[str, ...] = ()

    @property
    def run_id(self) -> str:
        return self.brief.run_id

    def finding(self, finding_id: str) -> Finding:
        return next(f for f in self.brief.findings if f.finding_id == finding_id)

    def heading(self, finding: Finding) -> str:
        """The finding's heading as the brief shows it. Raw text."""
        return change_heading(finding.change, self.metrics)


def find_runs(runs_dir: Path) -> list[Path]:
    """The run folders that hold an audit trail, newest first. Folders without one are skipped:
    a run stopped before its brief, or an estimate-only run, has nothing to show."""
    if not runs_dir.is_dir():
        return []
    folders = [path.parent for path in runs_dir.glob(f"*/{AUDIT_FILE}")]
    return sorted(folders, key=lambda folder: (folder / AUDIT_FILE).stat().st_mtime, reverse=True)


def read_run(folder: Path) -> SavedRun:
    """A run folder's audit trail as records. Raises if the file is missing or does not parse."""
    trail = json.loads((folder / AUDIT_FILE).read_text(encoding="utf-8"))
    brief = Brief.model_validate(trail["brief"])
    stored = trail.get("refs") or {}
    if stored:
        finding_refs = {full: ref for ref, full in stored.items() if ref.startswith("F")}
        claim_refs = {full: ref for ref, full in stored.items() if ref.startswith("C")}
    else:
        refs = assign_refs(brief.findings)
        finding_refs, claim_refs = refs.findings, refs.claims
    summary = trail.get("summary")
    not_examined = trail.get("not_examined")
    return SavedRun(
        folder=folder,
        brief=brief,
        finding_refs=finding_refs,
        claim_refs=claim_refs,
        trees={
            finding_id: tuple(TreeNode.model_validate(node) for node in nodes)
            for finding_id, nodes in (trail.get("trees") or {}).items()
        },
        metrics=AuditMetrics(
            {
                metric_id: MetricValue.model_validate(metric)
                for metric_id, metric in (trail.get("metrics") or {}).items()
            }
        ),
        labeling_failures={
            finding_id: tuple(notes)
            for finding_id, notes in (trail.get("labeling_failures") or {}).items()
        },
        not_examined=None if not not_examined else NotExamined.model_validate(not_examined),
        rationales_omitted=tuple(trail.get("rationales_omitted") or ()),
        rationales_untraced=tuple(trail.get("rationales_untraced") or ()),
        summary_used_template=None if summary is None else summary["used_template"],
        summary_failures=tuple(summary["failures"]) if summary else (),
    )


@dataclass(frozen=True)
class RunSummary:
    """One line of the run picker."""

    folder: Path
    run_id: str
    ticker: str
    company: str
    comparison_type: str
    comparison: str  # "year over year (10-K)"
    periods: str  # "FY2025 against FY2024"
    generated_at: datetime
    cost_usd: float
    findings: int
    labels: dict[str, int]  # display label -> claims with it


def summarize(run: SavedRun) -> RunSummary:
    comparison = run.brief.comparison
    labels: dict[str, int] = {}
    for finding in run.brief.findings:
        for claim in finding.claims:
            name = LABEL_NAMES[claim.label]
            labels[name] = labels.get(name, 0) + 1
    return RunSummary(
        folder=run.folder,
        run_id=run.run_id,
        ticker=comparison.ticker,
        company=comparison.current.company,
        comparison_type=comparison.comparison_type.value,
        comparison=COMPARISON_NAMES[comparison.comparison_type],
        periods=f"{comparison.current.period.label} against {comparison.prior.period.label}",
        generated_at=run.brief.generated_at,
        cost_usd=run.brief.cost_usd,
        findings=len(run.brief.findings),
        labels=labels,
    )


def summary_lines(run: SavedRun) -> list[tuple[str, tuple[str, ...]]]:
    """The summary's lines, each with the references it cites. Raw text."""
    lines = []
    for line in run.brief.summary.splitlines():
        line = line.strip().removeprefix("- ").strip()
        if line:
            lines.append(parse_summary_line(line))
    return lines


def section_of(finding: Finding) -> str:
    """The section a finding is filtered under: its filing section, or metric changes."""
    change = finding.change
    if change.kind is ChangeKind.NUMERIC or change.section is None:
        return METRIC_SECTION
    return SECTION_NAMES[change.section]


def findings_in_brief_order(run: SavedRun) -> list[Finding]:
    """Findings as the brief lists them: strongest label first, materiality order within."""
    return [finding for _, members in grouped(run.brief.findings) for finding in members]


def filter_findings(
    run: SavedRun,
    labels: Sequence[ExplanationLabel] | None = None,
    sections: Sequence[str] | None = None,
) -> list[Finding]:
    """The findings whose strongest label and section are among those chosen (None: any)."""
    return [
        finding
        for finding in findings_in_brief_order(run)
        if (labels is None or strongest_label(finding) in labels)
        and (sections is None or section_of(finding) in sections)
    ]


def shown_rationale(finding: Finding) -> str | None:
    """Why the change was flagged, if the brief may show it (a causal rationale is left out)."""
    return rationale_shown(finding.change)


def _squash(text: str) -> str:
    return " ".join(text.split())


def md(text: str) -> str:
    """Untrusted text made inert in Streamlit's Markdown, on one line.

    `brief.escape` covers Markdown itself; Streamlit also reads a pair of dollar signs as LaTeX,
    and filing text is full of dollar amounts ("$2.8 billion ... $118 million"), so `$` is escaped
    too.
    """
    return escape(text).replace("$", "\\$")


def md_quote(text: str) -> str:
    """Filing text as a Markdown block quote for Streamlit, one quoted line per paragraph line."""
    lines = [line.replace("$", "\\$") for line in quote_lines(text)]
    return "\n".join(f"> {line}" for line in lines) or "> (empty)"


def words(value: str) -> str:
    """An enum value for a reader: "states_cause" -> "states cause"."""
    return value.replace("_", " ")


def highlight(passage: str, quote: str) -> str:
    """The passage as escaped Markdown, with the quote in bold where it appears word for word.

    Whitespace is compared loosely (a quote is one line, a passage may wrap); when the quote is
    not found as written, the passage is shown without emphasis rather than guessed at.
    """
    text, wanted = _squash(passage), _squash(quote)
    start = text.find(wanted) if wanted else -1
    if start < 0:
        return md(text)
    end = start + len(wanted)
    return f"{md(text[:start])} **{md(wanted)}** {md(text[end:])}".strip()


# --- The evidence tree (T7.2) -------------------------------------------------------------------

LABEL_COLOURS = {
    ExplanationLabel.CONFIRMED: "#2e7d32",
    ExplanationLabel.PLAUSIBLE: "#b26a00",
    ExplanationLabel.UNRESOLVED: "#8e24aa",
}
STATUS_STYLE = {
    NodeStatus.KEPT: 'style="rounded,filled", fillcolor="#eef3fb", color="#3b6db3"',
    NodeStatus.PRUNED: 'style="rounded,dashed", color="#9e9e9e", fontcolor="#757575"',
    NodeStatus.PENDING: 'style="rounded,dotted", color="#c62828", fontcolor="#c62828"',
}


def shorten(text: str, limit: int) -> str:
    """Text on one line, cut at `limit` characters with an ellipsis."""
    text = _squash(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _dot_text(text: str, width: int, limit: int) -> str:
    """Text for a DOT label: shortened, wrapped, and quoted characters escaped."""
    lines = textwrap.wrap(shorten(text, limit), width) or [""]
    return "\\n".join(line.replace("\\", "\\\\").replace('"', '\\"') for line in lines)


def _node_kind(node: TreeNode) -> str:
    return ("Change", "Candidate explanation", "Evidence search")[node.depth]


def source_name(node: TreeNode) -> str | None:
    """Where a depth 2 node looked: a section, a computed metric, or None for everywhere."""
    if node.source is None:
        return None
    if node.source.metric is not None:
        expected = node.source.metric
        return f"{metric_name(expected.metric)} ({expected.direction.value}), computed"
    if node.source.section:
        return SECTION_NAMES[node.source.section]
    return None


def _score(node: TreeNode) -> str:
    return "-" if node.score is None else f"{node.score:.2f}"


@dataclass(frozen=True)
class TreeView:
    dot: str
    nodes: int
    pruned: int
    pending: int
    hidden: int = 0
    rows: list[dict] = field(default_factory=list)


def tree_view(
    nodes: Sequence[TreeNode],
    claims: Mapping[str, Claim],
    claim_refs: Mapping[str, str],
    *,
    show_pruned: bool = True,
) -> TreeView:
    """A finding's Tree-of-Thought tree as Graphviz DOT, and one table row per node.

    `claims` maps a claim id to the claim the brief shows; a depth 1 candidate has the same id as
    the claim it became, so a candidate that reached the brief is drawn in its label's colour. With
    `show_pruned` off, pruned branches are left out and each parent notes how many it lost.
    """
    by_id = {node.node_id: node for node in nodes}
    hidden_ids: set[str] = set()
    if not show_pruned:
        for node in nodes:
            parent, cut = node, False
            while parent is not None:
                if parent.status is NodeStatus.PRUNED:
                    cut = True
                    break
                parent = by_id.get(parent.parent_id or "")
            if cut:
                hidden_ids.add(node.node_id)
    lines = [
        "digraph tree {",
        '  rankdir="LR"; nodesep=0.15; ranksep=0.5;',
        '  node [shape=box, fontname="Helvetica", fontsize=10, margin="0.12,0.06"];',
        '  edge [color="#9e9e9e", arrowsize=0.6];',
    ]
    hidden_per_parent: dict[str, int] = {}
    rows = []
    for index, node in enumerate(nodes):
        if node.node_id in hidden_ids:
            if node.parent_id not in hidden_ids and node.parent_id is not None:
                hidden_per_parent[node.parent_id] = hidden_per_parent.get(node.parent_id, 0) + 1
            continue
        claim = claims.get(node.node_id)
        if node.depth == 0:
            label = "Change\\n" + _dot_text(node.thought, 42, 180)
            style = 'style="filled", fillcolor="#263238", fontcolor="white", color="#263238"'
        elif node.depth == 1:
            head = f"score {_score(node)}"
            if claim is not None:
                head += f" · {LABEL_NAMES[claim.label]} [{claim_refs.get(claim.claim_id, '?')}]"
            label = head + "\\n" + _dot_text(node.thought, 40, 160)
            style = STATUS_STYLE[node.status]
            if claim is not None:
                colour = LABEL_COLOURS[claim.label]
                style = f'style="rounded,filled,bold", fillcolor="white", color="{colour}"'
        else:
            where = source_name(node) or "search"
            evidence = words(node.evidence.value) if node.evidence else "not checked"
            label = (
                f"{_dot_text(where, 40, 60)} · score {_score(node)}\\nevidence: {evidence}"
                f"\\n{_dot_text(node.thought, 40, 100)}"
            )
            style = STATUS_STYLE[node.status]
        tooltip = _dot_text(node.prune_reason or node.thought, 80, 400)
        lines.append(f'  n{index} [label="{label}", tooltip="{tooltip}", {style}];')
        rows.append(
            {
                "depth": node.depth,
                "kind": _node_kind(node),
                "status": node.status.value,
                "score": node.score,
                "where": source_name(node),
                "evidence": words(node.evidence.value) if node.evidence else None,
                "numeric check": words(node.numeric_check.value) if node.numeric_check else None,
                "quotes": len(node.citations),
                "in brief": (
                    f"{LABEL_NAMES[claim.label]} [{claim_refs.get(claim.claim_id, '?')}]"
                    if claim is not None
                    else None
                ),
                "prune reason": node.prune_reason,
                "thought": node.thought,
            }
        )
    index_of = {node.node_id: i for i, node in enumerate(nodes)}
    for node in nodes:
        if node.node_id in hidden_ids or node.parent_id not in index_of:
            continue
        lines.append(f"  n{index_of[node.parent_id]} -> n{index_of[node.node_id]};")
    for parent_id, count in hidden_per_parent.items():
        note = f"h{index_of[parent_id]}"
        branches = "branch" if count == 1 else "branches"
        lines.append(
            f'  {note} [label="{count} pruned {branches} hidden", shape=plaintext, '
            'fontcolor="#757575", fontsize=9];'
        )
        lines.append(f"  n{index_of[parent_id]} -> {note} [style=dotted];")
    lines.append("}")
    return TreeView(
        dot="\n".join(lines),
        nodes=len(nodes),
        pruned=sum(1 for node in nodes if node.status is NodeStatus.PRUNED),
        pending=sum(1 for node in nodes if node.status is NodeStatus.PENDING),
        hidden=len(hidden_ids),
        rows=rows,
    )


def candidate_evidence(nodes: Sequence[TreeNode], candidate_id: str) -> list[TreeNode]:
    """A depth 1 candidate's evidence searches (its depth 2 children), kept ones first."""
    children = [node for node in nodes if node.parent_id == candidate_id]
    return sorted(children, key=lambda node: node.status is not NodeStatus.KEPT)


# --- The review queue (T7.3) --------------------------------------------------------------------

NEEDS_REVIEW = "Needs review"
LEFT_OUT = "Explanation left out of the brief"
SHOWN_WITH_NOTE = "Explanation shown, with a labeling note"
SEARCH_INCOMPLETE = "Search incomplete"
RATIONALE_DROPPED = "Rationale left out"
NOT_EXAMINED = "Not examined (past the cap)"
REVIEW_KINDS = (
    NEEDS_REVIEW,
    LEFT_OUT,
    SHOWN_WITH_NOTE,
    SEARCH_INCOMPLETE,
    RATIONALE_DROPPED,
    NOT_EXAMINED,
)

_CLAIM_NOTE = re.compile(r"^claim (?P<claim_id>\S+?):? (?P<reason>.+)$", re.DOTALL)
_STAGE_NOTE = re.compile(r"^(?P<stage>[a-z]+): (?P<node_id>\S+?):? (?P<reason>.+)$", re.DOTALL)


@dataclass(frozen=True)
class ReviewItem:
    kind: str
    subject: str  # raw text: the claim, the change heading, or the change id
    reasons: tuple[str, ...]
    finding_id: str | None = None
    # The depth 1 candidate whose evidence the item can show, if any.
    candidate_id: str | None = None


def _note_reason(note: str, ids: Mapping[str, str]) -> str:
    """A labeling note with full ids replaced by short references where the brief has them."""
    for full, short in ids.items():
        note = note.replace(full, short)
    return note


def review_items(run: SavedRun) -> list[ReviewItem]:
    """What an analyst should look at beyond the brief's own findings, most pressing first.

    - findings the brief lists under "Needs review" (a review note, or an unresolved claim);
    - candidate explanations labeling left out or annotated (`labeling_failures`), with the
      candidate's own wording from the tree;
    - search branches still pending, and labeling notes about a whole stage (a rework round);
    - rationales the brief left out (causal wording, or a figure nothing backs);
    - material changes past the run's cap, which were never searched.
    """
    items: list[ReviewItem] = []
    shown = {claim.claim_id: claim for finding in run.brief.findings for claim in finding.claims}
    nodes = {node.node_id: node for tree in run.trees.values() for node in tree}
    refs = {**run.finding_refs, **run.claim_refs}
    for finding in findings_in_brief_order(run):
        if finding.needs_review:
            reasons = []
            if finding.review_reason:
                reasons.append(finding.review_reason)
            if any(claim.label is ExplanationLabel.UNRESOLVED for claim in finding.claims):
                reasons.append("an explanation is unresolved")
            items.append(
                ReviewItem(NEEDS_REVIEW, run.heading(finding), tuple(reasons), finding.finding_id)
            )
    per_claim: dict[tuple[str, str], list[str]] = {}
    for finding_id, notes in run.labeling_failures.items():
        for note in notes:
            match = _CLAIM_NOTE.match(note)
            if match:
                key = (finding_id, match["claim_id"])
                per_claim.setdefault(key, []).append(_note_reason(match["reason"], refs))
                continue
            stage = _STAGE_NOTE.match(note)
            reason = stage["reason"] if stage else note
            finding = run.finding(finding_id) if finding_id in run.finding_refs else None
            subject = run.heading(finding) if finding else finding_id
            items.append(
                ReviewItem(
                    SEARCH_INCOMPLETE,
                    subject,
                    (_note_reason(reason, refs),),
                    finding_id if finding else None,
                )
            )
    for (finding_id, claim_id), reasons in per_claim.items():
        node = nodes.get(claim_id)
        if claim_id in shown:
            # As the brief words it, under its reference.
            kind = SHOWN_WITH_NOTE
            subject = f"[{run.claim_refs.get(claim_id, '?')}] {shown[claim_id].text}"
        else:
            # Never worded for the brief: the candidate as the generator proposed it.
            kind = LEFT_OUT
            subject = node.thought if node else claim_id
        items.append(
            ReviewItem(
                kind,
                subject,
                tuple(dict.fromkeys(reasons)),
                finding_id if finding_id in run.finding_refs else None,
                claim_id if node else None,
            )
        )
    for finding_id, tree in run.trees.items():
        pending = [node for node in tree if node.status is NodeStatus.PENDING]
        if pending:
            finding = run.finding(finding_id) if finding_id in run.finding_refs else None
            count = len(pending)
            items.append(
                ReviewItem(
                    SEARCH_INCOMPLETE,
                    run.heading(finding) if finding else finding_id,
                    (
                        f"{count} evidence search{'es' if count > 1 else ''} never finished "
                        "(still pending when the run ended), so "
                        f"{'their' if count > 1 else 'its'} evidence was not weighed",
                    ),
                    finding_id if finding else None,
                )
            )
    for finding_id in run.rationales_omitted:
        if finding_id in run.finding_refs:
            finding = run.finding(finding_id)
            items.append(
                ReviewItem(
                    RATIONALE_DROPPED,
                    run.heading(finding),
                    ("the reason it was flagged was worded causally, so the brief leaves it out",),
                    finding_id,
                )
            )
    by_change = {finding.change.change_id: finding for finding in run.brief.findings}
    for decision in run.rationales_untraced:
        change_id = decision.get("change_id", "?")
        figures = decision.get("untraced_figures") or ()
        reason = "the reason it was flagged cites a figure nothing backs"
        if figures:
            reason += ": " + ", ".join(str(figure) for figure in figures)
        finding = by_change.get(change_id)
        items.append(
            ReviewItem(
                RATIONALE_DROPPED,
                run.heading(finding) if finding else describe_change_id(change_id),
                (reason,),
                finding.finding_id if finding else None,
            )
        )
    if run.not_examined is not None:
        cap = run.not_examined.max_changes
        for skipped in run.not_examined.changes:
            items.append(
                ReviewItem(
                    NOT_EXAMINED,
                    describe_change_id(skipped.change_id),
                    (
                        f"materiality rank {skipped.materiality_rank}, importance "
                        f"{skipped.importance}; the run searches at most {cap} changes",
                    ),
                )
            )
    order = {kind: index for index, kind in enumerate(REVIEW_KINDS)}
    return sorted(items, key=lambda item: order[item.kind])


def describe_change_id(change_id: str) -> str:
    """A change id ("MAT:yoy-10k:modified:<accession>:mdna:0044") as words, for a change that was
    never searched and so has no record beyond its id."""
    parts = change_id.split(":")
    if len(parts) >= 4 and parts[2] == ChangeKind.NUMERIC.value:
        return f"Metric change: {metric_name(parts[3])}"
    if len(parts) >= 6:
        kind, section, number = parts[2], parts[4], parts[5]
        try:
            kind_name = {
                "added": "New paragraph",
                "removed": "Removed paragraph",
                "modified": "Changed paragraph",
            }[kind]
            section_name = next(name for key, name in SECTION_NAMES.items() if key.value == section)
        except (KeyError, StopIteration):
            return change_id
        return f"{kind_name} in {section_name} (paragraph {number})"
    return change_id
