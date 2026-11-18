"""Saved runs read back for the Streamlit app (T7): briefs, evidence trees and the review queue.

Everything here works from a run folder's `audit.json`, which `brief.write_run` writes beside
`brief.md` and which holds every record behind the brief. Nothing is recomputed and no model is
called, so the app costs nothing to use and shows exactly what the run produced.

This module has no Streamlit code, so the shaping can be tested on its own; `sec_agent.ui` draws
it. Text is returned raw unless a function says it is Markdown, which is escaped with
`brief.escape` the same way the brief escapes it: filing text and model text are untrusted.
"""

from __future__ import annotations

import difflib
import html
import itertools
import json
import re
import textwrap
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sec_agent.report.brief import (
    NotExamined,
    _guard_line_start,
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
    EvidenceSupport,
    ExplanationLabel,
    Finding,
    MetricValue,
    NodeStatus,
    NumericCheck,
    Section,
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


def md_quote(text: str, *, bold: bool = False) -> str:
    """Filing text as a Markdown block quote for Streamlit, one quoted line per paragraph line;
    with `bold`, each line in bold (a citation's quote, so it stands out from its passage)."""
    lines = [line.replace("$", "\\$") for line in quote_lines(text)]
    if bold:
        lines = [f"**{line}**" for line in lines]
    return "\n".join(f"> {line}" for line in lines) or "> (empty)"


# Streamlit Markdown's background colours: a quote within its passage, and the words of a
# changed paragraph that differ between its two versions.
QUOTE_MARK = "orange"
REMOVED_MARK = "red"
ADDED_MARK = "green"
# A sentence ends at ".", "!" or "?" (and any closing quotes or brackets), before a space and a
# capital, digit, "$" or opening bracket. The space is where the text is split.
# A marked stretch of text: (words, changed). A paragraph break is its own run, ("\n", False).
TextRun = tuple[str, bool]
PARAGRAPH_BREAK = "\n"
# An unchanged stretch this short between two changed ones is marked with them, so a reworded
# phrase reads as one change rather than as confetti around "the" and "of".
BRIDGE_WORDS = 2


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for line in text.splitlines():
        found = line.split()
        if not found:
            continue
        if tokens:
            tokens.append(PARAGRAPH_BREAK)
        tokens.extend(found)
    return tokens


def _bridge(tokens: Sequence[str], flags: list[bool]) -> list[bool]:
    """Mark short unchanged stretches that sit between two changes (not across paragraphs)."""
    out = list(flags)
    i = 0
    while i < len(out):
        if out[i]:
            i += 1
            continue
        j = i
        while j < len(out) and not out[j]:
            j += 1
        between = i > 0 and j < len(out)
        if between and j - i <= BRIDGE_WORDS and PARAGRAPH_BREAK not in tokens[i:j]:
            out[i:j] = [True] * (j - i)
        i = j
    return out


def _runs(tokens: Sequence[str], flags: Sequence[bool]) -> list[TextRun]:
    runs: list[TextRun] = []
    for token, changed in zip(tokens, flags, strict=True):
        if token == PARAGRAPH_BREAK:
            runs.append((PARAGRAPH_BREAK, False))
        elif runs and runs[-1][0] != PARAGRAPH_BREAK and runs[-1][1] == changed:
            runs[-1] = (f"{runs[-1][0]} {token}", changed)
        else:
            runs.append((token, changed))
    return runs


def changed_words(prior: str, current: str) -> tuple[list[TextRun], list[TextRun]]:
    """Both versions of a paragraph as runs of words, the words that differ marked: removed ones
    in the first, added ones in the second. Whitespace is squashed; paragraph breaks are kept.

    The comparison is word by word, so a changed figure marks only the figure. When one side is
    empty (an added or removed paragraph), nothing is marked: all of it is new or gone.
    """
    before, after = _tokens(prior), _tokens(current)
    gone, new = [False] * len(before), [False] * len(after)
    if before and after:
        matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                gone[i1:i2] = [True] * (i2 - i1)
                new[j1:j2] = [True] * (j2 - j1)
        gone, new = _bridge(before, gone), _bridge(after, new)
    return _runs(before, gone), _runs(after, new)


def md_mark(text: str, colour: str) -> str:
    """Escaped text with a Streamlit background colour."""
    return f":{colour}-background[{md(text)}]"


def md_quote_runs(runs: Sequence[TextRun], colour: str) -> str:
    """Runs of filing text as a Markdown block quote, the changed runs highlighted in `colour`."""
    lines: list[list[str]] = [[]]
    for text, changed in runs:
        if text == PARAGRAPH_BREAK:
            lines.append([])
        elif changed:
            lines[-1].append(md_mark(text, colour))
        else:
            plain = md(text)
            lines[-1].append(plain if lines[-1] else _guard_line_start(plain))
    return "\n".join(f"> {' '.join(line)}" for line in lines if line) or "> (empty)"


def words(value: str) -> str:
    """An enum value for a reader: "states_cause" -> "states cause"."""
    return value.replace("_", " ")


def highlight(passage: str, quote: str) -> str:
    """The passage as escaped Markdown, with the quote highlighted and in bold where it appears
    word for word.

    Whitespace is compared loosely (a quote is one line, a passage may wrap); when the quote is
    not found as written, the passage is shown without emphasis rather than guessed at.
    """
    text, wanted = _squash(passage), _squash(quote)
    start = text.find(wanted) if wanted else -1
    if start < 0:
        return md(text)
    end = start + len(wanted)
    marked = f":{QUOTE_MARK}-background[**{md(wanted)}**]"
    return f"{md(text[:start])} {marked} {md(text[end:])}".strip()


def change_description(finding: Finding, limit: int = 110) -> str | None:
    """A few words on what a change is about, beyond its heading ("Changed paragraph in MD&A"):
    why it was flagged, as the brief may show it, or else the start of its text. Raw text."""
    change = finding.change
    text = rationale_shown(change) or change.current_text or change.prior_text
    return shorten(text, limit) if text else None


# --- The evidence tree (T7.2) -------------------------------------------------------------------
#
# Drawn top-down: the change, then one card per candidate explanation. A card sums up its evidence
# searches as a row of coloured chips (one per search, solid if kept, hollow if pruned), so a tree
# of 50 to 80 nodes draws as about a dozen. A candidate that is opened (`expanded`) also shows its
# searches as their own boxes, stacked in a column under it, so opening a branch makes the tree
# longer rather than wider. Labels are Graphviz HTML-like labels, with all text escaped.

LABEL_COLOURS = {
    ExplanationLabel.CONFIRMED: "#2e7d32",
    ExplanationLabel.PLAUSIBLE: "#b26a00",
    ExplanationLabel.UNRESOLVED: "#8e24aa",
}


@dataclass(frozen=True)
class ChipStyle:
    colour: str
    meaning: str


# What a search found, strongest first: the critic's verdict on its passages, or for a computed
# metric, whether the metric moved the way the candidate expects.
CHIP_STYLES = {
    "states_cause": ChipStyle("#2e7d32", "states the cause"),
    "implies": ChipStyle("#3b6db3", "implies it"),
    "consistent": ChipStyle("#c49000", "consistent with it"),
    "metric_agrees": ChipStyle("#7e57c2", "metric agrees"),
    "metric_contradicts": ChipStyle("#c62828", "metric disagrees"),
    "none": ChipStyle("#9e9e9e", "no evidence"),
    "not_checked": ChipStyle("#cfd8dc", "not checked"),
    "pending": ChipStyle("#e57373", "never finished"),
}
_PRUNED_GREY = "#9e9e9e"
_REWORK = re.compile(r":w(?P<round>\d+)(?::|$)")
_ROUND = re.compile(r":r(?P<round>\d+):c\d+$")


def chip_kind(node: TreeNode) -> str:
    """The `CHIP_STYLES` key for a depth 2 search."""
    if node.status is NodeStatus.PENDING:
        return "pending"
    if node.evidence is not None and node.evidence is not EvidenceSupport.NONE:
        return node.evidence.value
    if node.numeric_check is NumericCheck.AGREE:
        return "metric_agrees"
    if node.numeric_check is NumericCheck.CONTRADICT:
        return "metric_contradicts"
    return "none" if node.evidence is EvidenceSupport.NONE else "not_checked"


def rework_round(node: TreeNode) -> int:
    """0 for a search made during the search itself, n for one made in rework round n."""
    if node.parent_id is None:
        return 0
    match = _REWORK.search(node.node_id.removeprefix(node.parent_id))
    return int(match["round"]) if match else 0


def candidate_round(node: TreeNode) -> int:
    """The round of candidates a depth 1 node came from: 1, or 2 for a fresh round."""
    match = _ROUND.search(node.node_id)
    return int(match["round"]) if match else 1


def candidate_numbers(nodes: Sequence[TreeNode]) -> dict[str, int]:
    """Each candidate explanation's number in the tree (1, 2, ...), in the order proposed."""
    return {node.node_id: i for i, node in enumerate((n for n in nodes if n.depth == 1), 1)}


def searches_of(nodes: Sequence[TreeNode], candidate_id: str) -> dict[int, list[TreeNode]]:
    """A candidate's evidence searches by round (0 for the search, n for rework round n)."""
    rounds: dict[int, list[TreeNode]] = {}
    for node in nodes:
        if node.parent_id == candidate_id:
            rounds.setdefault(rework_round(node), []).append(node)
    return dict(sorted(rounds.items()))


def shorten(text: str, limit: int) -> str:
    """Text on one line, cut at `limit` characters with an ellipsis."""
    text = _squash(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _html(text: str) -> str:
    """Text for a Graphviz HTML-like label."""
    return html.escape(text, quote=True)


def _html_lines(text: str, width: int, limit: int) -> str:
    """Text shortened, wrapped and escaped, as left-aligned HTML-like label lines."""
    lines = textwrap.wrap(shorten(text, limit), width) or [""]
    return "".join(f'{_html(line)}<BR ALIGN="LEFT"/>' for line in lines)


def _dot_text(text: str, width: int, limit: int) -> str:
    """Text for a quoted DOT attribute: shortened, wrapped, and quote characters escaped."""
    lines = textwrap.wrap(shorten(text, limit), width) or [""]
    return "\\n".join(line.replace("\\", "\\\\").replace('"', '\\"') for line in lines)


def _tooltip(node: TreeNode) -> str:
    """A node's hover text for a DOT attribute: its full thought, then why it was pruned, on one
    line (the browser wraps it; a line break here would land mid-sentence)."""
    text = _squash(node.thought)
    if node.depth == 2:
        text = f"{short_source_name(node)}: {text}"
    if node.prune_reason:
        text += f" — Pruned: {_squash(node.prune_reason)}"
    return shorten(text, 1200).replace("\\", "\\\\").replace('"', '\\"')


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


# Section names short enough for a search's box in the tree.
SHORT_SECTION_NAMES = {
    Section.BUSINESS: "Business",
    Section.RISK_FACTORS: "Risk factors",
    Section.MDNA: "MD&A",
    Section.MARKET_RISK: "Market risk",
    Section.FINANCIAL_NOTES: "Financial notes",
    Section.OTHER: "Other tables",
}


def short_source_name(node: TreeNode) -> str:
    """Where a depth 2 node looked, in a few words: "MD&A", "Gross margin (down)", "Everywhere"."""
    if node.source is not None and node.source.metric is not None:
        expected = node.source.metric
        return f"{metric_name(expected.metric)} ({expected.direction.value})"
    if node.source is not None and node.source.section:
        return SHORT_SECTION_NAMES.get(node.source.section, SECTION_NAMES[node.source.section])
    return "Everywhere"


def _score(node: TreeNode) -> str:
    return "-" if node.score is None else f"{node.score:.2f}"


def _chip(node: TreeNode) -> str:
    style = CHIP_STYLES[chip_kind(node)]
    if node.status is NodeStatus.KEPT:
        return f'<TD WIDTH="12" HEIGHT="12" FIXEDSIZE="TRUE" BGCOLOR="{style.colour}"></TD>'
    return (
        f'<TD WIDTH="12" HEIGHT="12" FIXEDSIZE="TRUE" BORDER="2" COLOR="{style.colour}" '
        'BGCOLOR="#ffffff"></TD>'
    )


def _chip_rows(rounds: Mapping[int, Sequence[TreeNode]]) -> str:
    rows = []
    for number, searches in rounds.items():
        if not searches:
            continue
        name = "searches" if number == 0 else f"rework {number}"
        chips = "".join(_chip(node) for node in searches)
        rows.append(
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#616161">{name}'
            f"</FONT></TD>{chips}</TR>"
        )
    if not rows:
        return ""
    return f'<TABLE BORDER="0" CELLSPACING="3" CELLPADDING="0">{"".join(rows)}</TABLE>'


@dataclass(frozen=True)
class TreeView:
    dot: str
    nodes: int
    pruned: int
    pending: int
    hidden: int = 0
    rows: list[dict] = field(default_factory=list)


def _hidden_ids(nodes: Sequence[TreeNode]) -> set[str]:
    """Every node that is pruned or sits under a pruned node."""
    by_id = {node.node_id: node for node in nodes}
    hidden: set[str] = set()
    for node in nodes:
        parent: TreeNode | None = node
        while parent is not None:
            if parent.status is NodeStatus.PRUNED:
                hidden.add(node.node_id)
                break
            parent = by_id.get(parent.parent_id or "")
    return hidden


def _table(rows: Sequence[str], padding: int) -> str:
    """Table rows as a Graphviz HTML-like label."""
    return (
        f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="{padding}">'
        f"{''.join(rows)}</TABLE>>"
    )


def _candidate_label(
    node: TreeNode,
    number: int,
    claim: Claim | None,
    claim_refs: Mapping[str, str],
    rounds: Mapping[int, Sequence[TreeNode]],
    hidden_searches: int,
    expanded: bool,
) -> str:
    head = f"<B>Candidate {number}</B>"
    if candidate_round(node) > 1:
        head += ' <FONT COLOR="#616161">(fresh round)</FONT>'
    rows = [f'<TR><TD ALIGN="LEFT">{head}</TD><TD ALIGN="RIGHT">score {_score(node)}</TD></TR>']
    if claim is not None:
        colour = LABEL_COLOURS[claim.label]
        verdict = f"{LABEL_NAMES[claim.label]} [{claim_refs.get(claim.claim_id, '?')}]"
        rows.append(
            f'<TR><TD COLSPAN="2" ALIGN="LEFT"><FONT COLOR="{colour}"><B>{_html(verdict)}</B>'
            "</FONT></TD></TR>"
        )
    elif node.status is NodeStatus.PRUNED:
        reason = shorten(node.prune_reason or "pruned", 60)
        rows.append(
            f'<TR><TD COLSPAN="2" ALIGN="LEFT"><FONT COLOR="#757575">Pruned: {_html(reason)}'
            "</FONT></TD></TR>"
        )
    rows.append(f'<TR><TD COLSPAN="2" ALIGN="LEFT">{_html_lines(node.thought, 34, 130)}</TD></TR>')
    chips = _chip_rows(rounds)
    if chips:
        rows.append(f'<TR><TD COLSPAN="2" ALIGN="LEFT">{chips}</TD></TR>')
    searches = [search for group in rounds.values() for search in group]
    if searches or hidden_searches:
        kept = sum(1 for search in searches if search.status is NodeStatus.KEPT)
        quotes = sum(len(search.citations) for search in searches)
        line = f"{len(searches) + hidden_searches} searches · {kept} kept · {quotes} quotes"
        if hidden_searches:
            line += f" · {hidden_searches} pruned hidden"
        if expanded:
            line += " · opened below"
        rows.append(
            f'<TR><TD COLSPAN="2" ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#616161">'
            f"{_html(line)}</FONT></TD></TR>"
        )
    return _table(rows, padding=2)


def _search_label(node: TreeNode) -> str:
    style = CHIP_STYLES[chip_kind(node)]
    rows = [
        f'<TR><TD ALIGN="LEFT"><B>{_html(shorten(short_source_name(node), 34))}</B></TD>'
        f'<TD ALIGN="RIGHT">{_score(node)}</TD></TR>',
    ]
    found = style.meaning
    if node.citations:
        found += f" · {len(node.citations)} quote{'s' if len(node.citations) > 1 else ''}"
    rows.append(
        f'<TR><TD COLSPAN="2" ALIGN="LEFT"><FONT COLOR="{style.colour}">{_html(found)}</FONT>'
        "</TD></TR>"
    )
    if node.prune_reason:
        rows.append(
            f'<TR><TD COLSPAN="2" ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#757575">'
            f"{_html_lines('Pruned: ' + node.prune_reason, 40, 80)}</FONT></TD></TR>"
        )
    rows.append(
        f'<TR><TD COLSPAN="2" ALIGN="LEFT"><FONT POINT-SIZE="9">'
        f"{_html_lines(node.thought, 40, 90)}</FONT></TD></TR>"
    )
    return _table(rows, padding=1)


_NODE_STYLE = {
    NodeStatus.KEPT: 'style="rounded,filled", fillcolor="#ffffff", color="#3b6db3"',
    NodeStatus.PRUNED: (
        f'style="rounded,filled,dashed", fillcolor="#f5f5f5", color="{_PRUNED_GREY}", '
        'fontcolor="#757575"'
    ),
    NodeStatus.PENDING: 'style="rounded,filled,dotted", fillcolor="#ffffff", color="#c62828"',
}


def tree_view(
    nodes: Sequence[TreeNode],
    claims: Mapping[str, Claim],
    claim_refs: Mapping[str, str],
    *,
    show_pruned: bool = True,
    expanded: Collection[str] = (),
) -> TreeView:
    """A finding's Tree-of-Thought tree as top-down Graphviz DOT, and one table row per node.

    `claims` maps a claim id to the claim the brief shows; a candidate has the same id as the claim
    it became, so a candidate that reached the brief is outlined in its label's colour.
    `expanded` holds the candidates whose searches are drawn as their own boxes. With
    `show_pruned` off, pruned branches are left out and the counts say how many.
    """
    hidden = set() if show_pruned else _hidden_ids(nodes)
    index_of = {node.node_id: i for i, node in enumerate(nodes)}
    numbers = candidate_numbers(nodes)
    lines = [
        "digraph tree {",
        '  rankdir="TB"; nodesep=0.25; ranksep=0.22; compound=true; bgcolor="transparent";',
        '  graph [fontname="Helvetica"];',
        '  node [shape=box, fontname="Helvetica", fontsize=10, margin="0.10,0.06"];',
        '  edge [color="#9e9e9e", arrowsize=0.6];',
    ]
    rows = []
    for node in nodes:
        if node.node_id in hidden:
            continue
        claim = claims.get(node.node_id)
        rows.append(
            {
                "depth": node.depth,
                "kind": _node_kind(node),
                "candidate": numbers.get(node.node_id) or numbers.get(node.parent_id or ""),
                "rework round": (rework_round(node) or None) if node.depth == 2 else None,
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
    root = next((node for node in nodes if node.depth == 0), None)
    hidden_candidates = 0
    for node in nodes:
        if node.depth == 0:
            label = "Change\\n" + _dot_text(node.thought, 50, 200)
            lines.append(
                f'  n{index_of[node.node_id]} [label="{label}", '
                f'tooltip="{_tooltip(node)}", style="rounded,filled", '
                'fillcolor="#263238", fontcolor="white", color="#263238", fontsize=11];'
            )
            continue
        if node.depth != 1:
            continue
        if node.node_id in hidden:
            hidden_candidates += 1
            continue
        rounds_all = searches_of(nodes, node.node_id)
        rounds = {
            number: [search for search in group if search.node_id not in hidden]
            for number, group in rounds_all.items()
        }
        hidden_searches = sum(len(group) for group in rounds_all.values()) - sum(
            len(group) for group in rounds.values()
        )
        is_open = node.node_id in expanded
        claim = claims.get(node.node_id)
        style = _NODE_STYLE[node.status]
        if claim is not None:
            colour = LABEL_COLOURS[claim.label]
            style = (
                f'style="rounded,filled,bold", fillcolor="#ffffff", color="{colour}", penwidth=2'
            )
        label = _candidate_label(
            node,
            numbers[node.node_id],
            claim,
            claim_refs,
            rounds,
            hidden_searches,
            is_open and any(rounds.values()),
        )
        tooltip = _tooltip(node)
        me = f"n{index_of[node.node_id]}"
        lines.append(f'  {me} [label={label}, tooltip="{tooltip}", {style}];')
        if root is not None and node.parent_id == root.node_id:
            lines.append(f"  n{index_of[root.node_id]} -> {me} [minlen=2];")
        if not is_open or not any(rounds.values()):
            continue
        # The opened branch: its searches stacked in a column, one cluster per candidate.
        lines.append(f"  subgraph cluster_{me} {{")
        lines.append(
            f'    label="Candidate {numbers[node.node_id]}: evidence searches"; fontsize=9; '
            'fontcolor="#616161"; style="rounded,filled"; fillcolor="#f7f9fc"; color="#cfd8dc";'
        )
        column: list[str] = []
        for number, group in rounds.items():
            if not group:
                continue
            if number > 0:
                header = f"{me}_w{number}"
                lines.append(
                    f'    {header} [label="Rework round {number}", shape=plaintext, '
                    'fontsize=9, fontcolor="#616161", height=0.2];'
                )
                column.append(header)
            for search in group:
                child = f"n{index_of[search.node_id]}"
                tip = _tooltip(search)
                lines.append(
                    f'    {child} [label={_search_label(search)}, tooltip="{tip}", '
                    f"width=2.7, {_NODE_STYLE[search.status]}];"
                )
                column.append(child)
        lines.append("  }")
        lines.append(f"  {me} -> {column[0]} [lhead=cluster_{me}];")
        for upper, lower in itertools.pairwise(column):
            lines.append(f"  {upper} -> {lower} [style=invis];")
    if hidden_candidates and root is not None:
        note = f"{hidden_candidates} pruned candidate{'s' if hidden_candidates > 1 else ''} hidden"
        lines.append(
            f'  hidden_note [label="{note}", shape=plaintext, fontcolor="#757575", fontsize=9];'
        )
        lines.append(f"  n{index_of[root.node_id]} -> hidden_note [style=dotted];")
    lines.append("}")
    return TreeView(
        dot="\n".join(lines),
        nodes=len(nodes),
        pruned=sum(1 for node in nodes if node.status is NodeStatus.PRUNED),
        pending=sum(1 for node in nodes if node.status is NodeStatus.PENDING),
        hidden=len(hidden),
        rows=rows,
    )


def candidate_choices(nodes: Sequence[TreeNode], claims: Mapping[str, Claim]) -> dict[str, str]:
    """A short label per candidate ("3 · Cocoa costs rose…", with a mark if it reached the brief),
    for the control that opens branches, mapped to the candidate's node id."""
    numbers = candidate_numbers(nodes)
    choices = {}
    for node in nodes:
        if node.depth != 1:
            continue
        mark = (
            " ✓" if node.node_id in claims else (" ✗" if node.status is NodeStatus.PRUNED else "")
        )
        choices[f"{numbers[node.node_id]}{mark} · {shorten(node.thought, 38)}"] = node.node_id
    return choices


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
