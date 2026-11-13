"""The brief writer (T5.4): a Markdown brief for the analyst and a JSON audit trail beside it.

Everything in the brief comes from the records through a template, except the summary (see
`summary.py`). Every statement carries a short reference, `[C1]` for a claim or `[F1]` for a
finding, that the audit trail maps to the full id, so a reader can trace any line to its evidence
and its Tree-of-Thought tree.

Findings are grouped by their strongest label, and each shows all its claims with their labels, so
one change's competing explanations stay together (the ToT design: each candidate keeps its own
evidence chain, and the report combines them). Findings that need review are listed again at the
end, with the reason.

Two kinds of text reach the brief, and they are treated differently:

- Text the system wrote (claims, the summary, change rationales, review reasons) goes through the
  causal-language check: only a confirmed claim may say why something happened (gap 13).
- Filing text (quotes and changed paragraphs) is shown as a block quote and never checked: a
  filing's own "due to" is evidence, not a claim of ours.

All of it is escaped before it reaches the Markdown. Filing text is untrusted and model text may
repeat it, so a quote must not be able to add a link, a heading or a fake reference.

`write_run` refuses to write a brief that breaks a rule, such as a plausible claim worded causally:
labeling should have prevented it, so it is a bug upstream, and a brief that quietly contained it
would state a cause nobody verified.
"""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import Field

from sec_agent.agents.verification import check_causal_language, find_causal_language
from sec_agent.compare.materiality import MaterialityDecision
from sec_agent.contracts import MetricLookup
from sec_agent.report.names import (
    CHANGE_KIND_NAMES,
    COMPARISON_NAMES,
    LABEL_NAMES,
    SECTION_NAMES,
    metric_name,
)
from sec_agent.schemas import (
    Brief,
    Change,
    ChangeKind,
    Citation,
    Claim,
    Comparison,
    ExplanationLabel,
    Finding,
    MetricValue,
    Record,
    TreeNode,
)

TEMPLATES = Path(__file__).parent / "templates"

# Strongest first. A finding is grouped by its strongest claim; one with no claims is unresolved.
_LABEL_ORDER = (ExplanationLabel.CONFIRMED, ExplanationLabel.PLAUSIBLE, ExplanationLabel.UNRESOLVED)
GROUP_TITLES: dict[ExplanationLabel, str] = {
    ExplanationLabel.CONFIRMED: "Changes with a confirmed explanation",
    ExplanationLabel.PLAUSIBLE: "Changes with plausible explanations only",
    ExplanationLabel.UNRESOLVED: "Changes without a supported explanation",
}

# A reference as it appears in the brief and in the summary lines.
REF = re.compile(r"\[([CF]\d+)\]")
# A summary line: its text, then one or more references at the end.
_SUMMARY_LINE = re.compile(r"^(?P<text>.*?)\s*(?P<refs>(?:\[[CF]\d+\]\s*)+)$")

# Wording that gives investment advice, which the brief must never do (the disclaimer says so).
# Checked on the summary, the one free-written part. Chosen by hand: the usual words of a
# recommendation or a price call. Plain "buy" and "sell" are left out because a filing's own
# business can be about buying and selling.
ADVICE = re.compile(
    r"\brecommend(?:s|ed|ation|ations)?\b|\bprice targets?\b|\b(?:out|under)perform\b"
    r"|\b(?:over|under)weight\b|\b(?:over|under)valued\b|\binvestors should\b"
    r"|\bshould (?:buy|sell|hold)\b|\b(?:buy|sell|hold) (?:rating|recommendation)\b",
    re.IGNORECASE,
)


# The whole summary of a brief with no findings. It is the only summary line allowed to cite
# nothing, because there is nothing to cite.
NO_FINDINGS_SUMMARY = "No material changes were found between these two filings."


class BriefError(ValueError):
    """The brief breaks a rule and was not written. `problems` says exactly what."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("the brief was not written: " + "; ".join(problems))
        self.problems = tuple(problems)


class SkippedChange(Record):
    """A material change the run did not examine, as `audit.json` lists it."""

    change_id: str
    materiality_rank: int = Field(ge=1)
    # From the materiality decision; None if the change has none on record.
    importance: int | None = Field(default=None, ge=0, le=5)


class NotExamined(Record):
    """The material changes past the run's cap on how many it explores (T5.5).

    They were never searched, so they are not findings and do not go under review: the brief says
    how many there are in one line and `audit.json` lists them (user's decision, 2026-11-12). A
    finding list of dozens of unexamined changes would bury the ones that were examined.
    """

    max_changes: int = Field(ge=1)
    changes: tuple[SkippedChange, ...] = ()


@dataclass(frozen=True)
class Refs:
    """The short references of one brief, assigned in reading order."""

    findings: dict[str, str]  # finding id -> "F1"
    claims: dict[str, str]  # claim id -> "C1"

    def resolve(self) -> dict[str, str]:
        """Reference -> full id, as the audit trail records it."""
        return {
            **{ref: finding_id for finding_id, ref in self.findings.items()},
            **{ref: claim_id for claim_id, ref in self.claims.items()},
        }


def strongest_label(finding: Finding) -> ExplanationLabel:
    labels = {claim.label for claim in finding.claims}
    return next((label for label in _LABEL_ORDER if label in labels), ExplanationLabel.UNRESOLVED)


def ordered_claims(finding: Finding) -> list[Claim]:
    """A finding's claims, confirmed first, keeping the search's order within a label."""
    return sorted(finding.claims, key=lambda claim: _LABEL_ORDER.index(claim.label))


def grouped(findings: Sequence[Finding]) -> list[tuple[ExplanationLabel, list[Finding]]]:
    """Findings by strongest label, strongest group first, keeping materiality order within."""
    groups = {label: [] for label in _LABEL_ORDER}
    for finding in findings:
        groups[strongest_label(finding)].append(finding)
    return [(label, members) for label, members in groups.items() if members]


def assign_refs(findings: Sequence[Finding]) -> Refs:
    """Numbers findings and claims in the order the brief shows them.

    The summary is written before the brief is rendered and cites these, so both must call this
    on the same findings.
    """
    finding_refs: dict[str, str] = {}
    claim_refs: dict[str, str] = {}
    for _, members in grouped(findings):
        for finding in members:
            finding_refs[finding.finding_id] = f"F{len(finding_refs) + 1}"
            for claim in ordered_claims(finding):
                claim_refs[claim.claim_id] = f"C{len(claim_refs) + 1}"
    return Refs(findings=finding_refs, claims=claim_refs)


def parse_summary_line(line: str) -> tuple[str, tuple[str, ...]]:
    """A summary line's text and its references; no references if the line has none at the end."""
    match = _SUMMARY_LINE.match(line.strip())
    if match is None:
        return line.strip(), ()
    return match["text"], tuple(REF.findall(match["refs"]))


def format_summary_line(text: str, refs: Iterable[str]) -> str:
    return f"{text.strip()} " + " ".join(f"[{ref}]" for ref in refs)


def summary_line_problems(
    text: str, refs: Sequence[str], claims_by_ref: Mapping[str, Claim], finding_refs: set[str]
) -> list[str]:
    """What is wrong with one summary sentence, without looking anything up outside the brief."""
    problems: list[str] = []
    if not text.strip():
        problems.append("a summary sentence is blank")
    if not refs:
        problems.append(f"summary sentence {text!r} cites no claim or finding")
    if REF.search(text):
        problems.append(f"summary sentence {text!r} has a reference inside its text")
    unknown = [ref for ref in refs if ref not in claims_by_ref and ref not in finding_refs]
    if unknown:
        problems.append(f"summary sentence {text!r} cites unknown {', '.join(unknown)}")
    flagged = find_causal_language(text)
    if flagged:
        # Causal wording only where everything cited is a confirmed claim: a finding reference or
        # a plausible claim cannot carry a cause (gap 13).
        all_confirmed = bool(refs) and all(
            ref in claims_by_ref and claims_by_ref[ref].label is ExplanationLabel.CONFIRMED
            for ref in refs
        )
        if not all_confirmed:
            problems.append(
                f"summary sentence {text!r} uses causal wording ({', '.join(flagged)}) but does "
                "not cite only confirmed claims"
            )
    advice = sorted({match.group().lower() for match in ADVICE.finditer(text)})
    if advice:
        problems.append(
            f"summary sentence {text!r} reads as investment advice ({', '.join(advice)})"
        )
    return problems


def check_brief(brief: Brief) -> tuple[str, ...]:
    """Every rule the brief breaks. Empty means it may be written.

    Change rationales are not here: a flagged one is left out of the brief rather than stopping it
    (see `rationale_shown`), because it says why a change was flagged, not what caused it.
    """
    problems: list[str] = []
    finding_ids = [finding.finding_id for finding in brief.findings]
    claim_ids = [claim.claim_id for finding in brief.findings for claim in finding.claims]
    for ids, kind in ((finding_ids, "finding"), (claim_ids, "claim")):
        repeated = sorted({item for item in ids if ids.count(item) > 1})
        if repeated:
            # References and the audit trail are keyed by id, so a repeat would merge two records.
            problems.append(f"{kind} ids repeat: {', '.join(repeated)}")
    if problems:
        return tuple(problems)

    for finding in brief.findings:
        for claim in finding.claims:
            result = check_causal_language(claim)
            if not result.passed:
                problems.append(
                    f"claim {claim.claim_id} is {claim.label.value} but worded causally "
                    f"({', '.join(result.flagged_phrases)})"
                )
        if finding.review_reason and find_causal_language(finding.review_reason):
            problems.append(f"the review reason of {finding.finding_id} uses causal wording")

    if not brief.findings:
        if brief.summary.strip() != NO_FINDINGS_SUMMARY:
            problems.append("a brief with no findings must use the fixed no-findings summary")
        return tuple(problems)
    refs = assign_refs(brief.findings)
    claims_by_ref = {
        refs.claims[claim.claim_id]: claim for finding in brief.findings for claim in finding.claims
    }
    lines = [line for line in brief.summary.splitlines() if line.strip()]
    if not lines:
        problems.append("the summary is empty")
    for line in lines:
        text, line_refs = parse_summary_line(line)
        problems += summary_line_problems(
            text, line_refs, claims_by_ref, set(refs.findings.values())
        )
    return tuple(problems)


def rationale_shown(change: Change) -> str | None:
    """The change's rationale if the brief may show it; None if it has none or it is causal."""
    if change.rationale and not find_causal_language(change.rationale):
        return change.rationale
    return None


# Characters that would let text change the Markdown around it: emphasis, code, headings, tables,
# HTML, block quotes and strikethrough. Square brackets are enough to stop links and images, which
# both need "[...](...)", so parentheses stay readable, as in "(FY2025)". A backslash before each
# makes it literal.
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|~])")


def escape(text: str) -> str:
    """Untrusted text made inert in Markdown, on one line."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", " ".join(text.split()))


def quote_lines(text: str) -> list[str]:
    """Filing text as escaped block-quote lines, one per paragraph line, none blank."""
    return [_guard_line_start(escape(line)) for line in text.splitlines() if line.strip()]


# The brief is hard-wrapped so the raw file reads well in a terminal or editor (the user's decision,
# 2026-11-10): an unwrapped quote ran to 451 characters, and a fixed-width view cut words in half.
# 100 matches the code's line length. Markdown joins wrapped lines back into one paragraph.
WRAP_WIDTH = 100

# What can start a Markdown block when it begins a line: a list item ("-", "+", "1." or "1)") or a
# setext heading underline ("---", "==="). "*", "#" and ">" are already escaped by `escape`.
_LINE_START_MARKER = re.compile(r"^(?:[-+=]|\d+(?=[.)]))")


def _guard_line_start(content: str) -> str:
    """`content` made safe to begin a line: a leading block marker is escaped."""
    match = _LINE_START_MARKER.match(content)
    if match is None:
        return content
    if match.group()[0].isdigit():
        # "2025. Sales" would start an ordered list: escape the period or parenthesis instead.
        end = match.end()
        return f"{content[:end]}\\{content[end:]}"
    return f"\\{content}"


# A line's prefix and the prefix its continuation lines need: a block quote keeps "> " (with any
# list indent before it), a list item's continuation is indented past "- ", and plain text keeps
# its own indent.
_QUOTE = re.compile(r"^( *> )")
_LIST_ITEM = re.compile(r"^( *)- ")


def _wrap_line(line: str) -> list[str]:
    if len(line) <= WRAP_WIDTH or line.startswith("#"):
        # A Markdown heading cannot continue onto a second line, so it stays whole.
        return [line]
    if quote := _QUOTE.match(line):
        first = later = quote[1]
    elif item := _LIST_ITEM.match(line):
        first, later = f"{item[1]}- ", f"{item[1]}  "
    else:
        first = later = line[: len(line) - len(line.lstrip(" "))]
    wrapped = textwrap.wrap(
        line[len(first) :],
        # One less, because guarding a continuation line can add a backslash.
        width=WRAP_WIDTH - 1,
        initial_indent=first,
        subsequent_indent=later,
        # A URL or other long token stays whole even past the width, so a link never breaks.
        break_long_words=False,
        break_on_hyphens=False,
    )
    return [wrapped[0]] + [later + _guard_line_start(rest[len(later) :]) for rest in wrapped[1:]]


def wrap_markdown(markdown: str) -> str:
    """Hard-wrap every line past `WRAP_WIDTH` at spaces, keeping its Markdown structure."""
    return "\n".join(wrapped for line in markdown.split("\n") for wrapped in _wrap_line(line))


def _value(metric: MetricValue) -> str:
    if metric.unit == "ratio":
        return f"{metric.value * 100:.1f}%"
    if metric.unit == "USD":
        # Signed whole dollars (a net loss is negative), shown as "-$1,234" rather than "$-1,234".
        sign = "-" if metric.value < 0 else ""
        return f"{sign}${abs(metric.value):,.0f}"
    return f"{metric.value:g} {metric.unit}"


def metric_line(metric_id: str, metrics: MetricLookup | None) -> str:
    """A cited metric for the reader, or its id when it cannot be looked up. Not escaped."""
    metric = metrics.get_metric(metric_id) if metrics is not None else None
    if metric is None:
        return f"metric {metric_id}"
    return f"{metric_name(metric.name)}: {_value(metric)} ({metric.period.label})"


def change_heading(change: Change, metrics: MetricLookup | None) -> str:
    """A finding's heading: the metric and its current value, or the kind and section of a text
    change. A `Change` names only the current metric, so the earlier value is not shown here."""
    if change.kind is ChangeKind.NUMERIC:
        metric = metrics.get_metric(change.metric_id or "") if metrics is not None else None
        if metric is None:
            return f"{CHANGE_KIND_NAMES[change.kind]}: metric {change.metric_id}"
        return f"{metric_name(metric.name)} change: {_value(metric)} in {metric.period.label}"
    assert change.section is not None  # `Change` requires a section for a text change
    return f"{CHANGE_KIND_NAMES[change.kind]} in {SECTION_NAMES[change.section]}"


def _source(citation: Citation) -> str:
    return (
        f"{citation.form.value} for {citation.period.label}, {SECTION_NAMES[citation.section]}, "
        f"<{citation.url}>"
    )


def _claim_view(claim: Claim, refs: Refs, metrics: MetricLookup | None) -> dict:
    return {
        "ref": refs.claims[claim.claim_id],
        "label": LABEL_NAMES[claim.label],
        "text": escape(claim.text),
        "citations": [
            {"quote": quote_lines(citation.quote), "source": _source(citation)}
            for citation in claim.citations
        ],
        "metrics": [escape(metric_line(metric_id, metrics)) for metric_id in claim.metric_ids],
    }


def _finding_view(
    finding: Finding, comparison: Comparison, refs: Refs, metrics: MetricLookup | None
) -> dict:
    change = finding.change
    texts = []
    if change.prior_text:
        texts.append(
            {
                "title": f"Before ({comparison.prior.period.label})",
                "lines": quote_lines(change.prior_text),
            }
        )
    if change.current_text:
        texts.append(
            {
                "title": f"After ({comparison.current.period.label})",
                "lines": quote_lines(change.current_text),
            }
        )
    rationale = rationale_shown(change)
    return {
        "ref": refs.findings[finding.finding_id],
        "heading": escape(change_heading(change, metrics)),
        "rationale": escape(rationale) if rationale else None,
        "texts": texts,
        "claims": [_claim_view(claim, refs, metrics) for claim in ordered_claims(finding)],
    }


def _review_reason(finding: Finding) -> str:
    if finding.review_reason:
        return finding.review_reason
    return "an explanation is unresolved"


def not_examined_line(not_examined: NotExamined | None) -> str | None:
    """The brief's one line on changes past the cap, or None when there are none."""
    if not_examined is None or not not_examined.changes:
        return None
    count = len(not_examined.changes)
    changes = "change was" if count == 1 else "changes were"
    return (
        f"This run explores at most {not_examined.max_changes} material changes, in order of "
        f"materiality, so {count} further material {changes} not examined. audit.json lists "
        "them under not_examined, with their rank and importance."
    )


def render_markdown(
    brief: Brief,
    metrics: MetricLookup | None = None,
    not_examined: NotExamined | None = None,
) -> str:
    """The brief as Markdown. Call `check_brief` first; `write_run` does."""
    refs = assign_refs(brief.findings)
    comparison = brief.comparison
    summary = []
    for line in brief.summary.splitlines():
        if line.strip():
            text, line_refs = parse_summary_line(line)
            summary.append({"text": escape(text), "refs": line_refs})
    view = {
        "title": (
            f"{escape(comparison.ticker)}: {comparison.current.form.value} for "
            f"{comparison.current.period.label} against {comparison.prior.period.label}"
        ),
        "comparison": COMPARISON_NAMES[comparison.comparison_type],
        "run_id": escape(brief.run_id),
        "generated_at": brief.generated_at.isoformat(timespec="seconds"),
        "cost": f"${brief.cost_usd:.2f}",
        "disclaimer": escape(brief.disclaimer),
        "summary": summary,
        "groups": [
            {
                "title": GROUP_TITLES[label],
                "findings": [
                    _finding_view(finding, comparison, refs, metrics) for finding in members
                ],
            }
            for label, members in grouped(brief.findings)
        ],
        "review": [
            {
                "ref": refs.findings[finding.finding_id],
                "heading": escape(change_heading(finding.change, metrics)),
                "reason": escape(_review_reason(finding)),
            }
            for _, members in grouped(brief.findings)
            for finding in members
            if finding.needs_review
        ],
        "not_examined": not_examined_line(not_examined),
    }
    environment = Environment(
        loader=FileSystemLoader(TEMPLATES),
        # Markdown, not HTML: escaping is done above by `escape`, which knows Markdown's syntax.
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return wrap_markdown(environment.get_template("brief.md.j2").render(**view))


def audit_trail(
    brief: Brief,
    *,
    trees: Mapping[str, Sequence[TreeNode]] | None = None,
    labeling_failures: Mapping[str, Sequence[str]] | None = None,
    metrics: MetricLookup | None = None,
    not_examined: NotExamined | None = None,
    untraced_rationales: Sequence[MaterialityDecision] = (),
    summary_used_template: bool | None = None,
    summary_failures: Sequence[str] = (),
) -> dict:
    """Everything behind the brief, as JSON-ready data, so a saved run stands on its own.

    `trees` maps a finding id to its whole tree, pruned branches included, as
    `TreeClient.read_tree` returns it. `labeling_failures` maps a finding id to what labeling
    could not do (`LabeledFinding.failures`). Metrics are looked up and stored when a lookup is
    given, so the reader of a saved run can see the values a claim cites. `not_examined` lists
    the material changes past the run's cap. `untraced_rationales` are the materiality decisions
    whose rationale was dropped from the change for a figure nothing backs: the change reaches
    the brief with no rationale, so this is the only record of what was dropped and why.
    `summary_used_template` and `summary_failures` say how the summary was made
    (`report.summary.SummaryResult`): a template summary looks like a written one in the brief, so
    only the audit trail shows that the writer was rejected, failed or ran out of budget. They are
    plain fields because `report.summary` imports this module. None means the caller did not say.
    """
    refs = assign_refs(brief.findings)
    metric_ids = {
        metric_id
        for finding in brief.findings
        for metric_id in (
            *(claim_metric for claim in finding.claims for claim_metric in claim.metric_ids),
            *((finding.change.metric_id,) if finding.change.metric_id else ()),
        )
    }
    stored = {}
    if metrics is not None:
        for metric_id in sorted(metric_ids):
            metric = metrics.get_metric(metric_id)
            if metric is not None:
                stored[metric_id] = metric.model_dump(mode="json")
    omitted = [
        finding.finding_id
        for finding in brief.findings
        if finding.change.rationale and rationale_shown(finding.change) is None
    ]
    return {
        "brief": brief.model_dump(mode="json"),
        "refs": refs.resolve(),
        "trees": {
            finding_id: [node.model_dump(mode="json") for node in nodes]
            for finding_id, nodes in (trees or {}).items()
        },
        "labeling_failures": {
            finding_id: list(failures) for finding_id, failures in (labeling_failures or {}).items()
        },
        "metrics": stored,
        # Findings whose rationale was left out of the brief because it was worded causally.
        "rationales_omitted": omitted,
        # Rationales dropped at materiality for a figure nothing backs, with those figures.
        "rationales_untraced": [
            decision.model_dump(mode="json") for decision in untraced_rationales
        ],
        "not_examined": None if not_examined is None else not_examined.model_dump(mode="json"),
        "summary": None
        if summary_used_template is None
        else {"used_template": summary_used_template, "failures": list(summary_failures)},
    }


@dataclass(frozen=True)
class RunFiles:
    brief: Path
    audit: Path


def write_run(
    brief: Brief,
    out_dir: Path,
    *,
    trees: Mapping[str, Sequence[TreeNode]] | None = None,
    labeling_failures: Mapping[str, Sequence[str]] | None = None,
    metrics: MetricLookup | None = None,
    not_examined: NotExamined | None = None,
    untraced_rationales: Sequence[MaterialityDecision] = (),
    summary_used_template: bool | None = None,
    summary_failures: Sequence[str] = (),
) -> RunFiles:
    """Check the brief, then write `brief.md` and `audit.json` into `out_dir`.

    Raises `BriefError` and writes nothing if the brief breaks a rule. `out_dir` is normally
    `settings.runs_dir / brief.run_id`; files of the same run are replaced, so a resumed run
    rewrites its brief.
    """
    problems = check_brief(brief)
    if problems:
        raise BriefError(problems)
    markdown = render_markdown(brief, metrics, not_examined)
    trail = audit_trail(
        brief,
        trees=trees,
        labeling_failures=labeling_failures,
        metrics=metrics,
        not_examined=not_examined,
        untraced_rationales=untraced_rationales,
        summary_used_template=summary_used_template,
        summary_failures=summary_failures,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    files = RunFiles(brief=out_dir / "brief.md", audit=out_dir / "audit.json")
    files.brief.write_text(markdown, encoding="utf-8")
    files.audit.write_text(json.dumps(trail, indent=2, ensure_ascii=False), encoding="utf-8")
    return files
