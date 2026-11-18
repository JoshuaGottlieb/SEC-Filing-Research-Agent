"""The replay audit (T7): a saved run's steps in order, for the app's animated replay.

A run folder already holds everything a replay needs, so nothing is recomputed and no model is
called:

- `timings.jsonl` times every step (detect, materiality, each change's search, labeling and rework
  rounds, the summary), with the time it ended;
- `ledger.jsonl` has every model call, with its role, model, cost and the step it ran in;
- `audit.json` has each change's Tree-of-Thought tree and the brief's labeled claims.

`build_replay` lines them up into one list of steps. A change's search is split the way the
controller runs it: the generator proposes candidates, the critic pre-screens them, then each kept
candidate gets its evidence searches, one generator call each. When the run's generator calls match
that order, each step takes its times from them; otherwise the search's time is shared out evenly.
A rework round is matched to its timing the same way.

Tree nodes carry no times, so a replay shows each node in the step that made it, not at the
second it was made. Labels are only saved as they ended: a claim with rework searches was found
only implied at first labeling (rework is only for those), so the replay shows it that way, then
gives it its final label after its last round. A run saved before the timings were written replays
without a clock.

The result is plain JSON for the page in `sec_agent/ui/replay.html`. Text in it is raw; the page
sets it as text, never as HTML.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sec_agent.report import run_view as rv
from sec_agent.report.brief import strongest_label
from sec_agent.report.names import LABEL_NAMES
from sec_agent.schemas import ExplanationLabel, Finding, NodeStatus, TreeNode

# The pipeline's stages, in order, as the replay's progress strip shows them.
STAGES = ("detect", "materiality", "search", "label", "rework", "summary")
STAGE_NAMES = {
    "detect": "Detect",
    "materiality": "Rank",
    "search": "Explore",
    "label": "Label",
    "rework": "Rework",
    "summary": "Write",
}
TONES = {
    ExplanationLabel.CONFIRMED: "confirmed",
    ExplanationLabel.PLAUSIBLE: "plausible",
    ExplanationLabel.UNRESOLVED: "unresolved",
}


def workflow_notes() -> dict[str, str]:
    """What each kind of step does, in general, for the replay's notes panel. The numbers come
    from the settings the code runs with (`config/scoring.yaml`, `tot/rework.py`)."""
    from sec_agent.tot.rework import MAX_REWORK_ROUNDS
    from sec_agent.tot.scoring import ScoringConfig

    config = ScoringConfig.load()
    beam = config.beam
    return {
        "detect": (
            "The two filings are lined up section by section (business, risk factors, MD&A, "
            "market risk, financial notes) to find every paragraph added, removed or reworded. "
            "A scan of the XBRL figures computes metrics such as gross and operating margin and "
            "flags the ones that moved. This is plain code: no model is called."
        ),
        "materiality": (
            "Claude Opus reads the changes in batches of about 20 and rates each one's importance "
            "from 0 to 5 against a written rubric. Text changes at 3 or above are material, and "
            "flagged metric moves always are. The most important are explored in order, up to "
            "the run's cap; the rest go to the review queue."
        ),
        "change": (
            "Each material change gets its own Tree-of-Thought search. The change is the root; "
            "the tree's branches are competing explanations for it and the evidence tried for "
            "each. Code, not a model, decides what is kept and what is pruned."
        ),
        "propose": (
            f"Depth 1: the generator (Claude Haiku) proposes up to {beam.candidates} candidate "
            "explanations for the change, each with the evidence it expects to find and a plan "
            "for searching the filings."
        ),
        "prescreen": (
            "The critic (Claude Haiku) rates each candidate before any search: does it fit how "
            "the metrics moved, and is it specific? Code turns the ratings into a score from 0 "
            f"to 1. A candidate below {config.threshold:.2f} is pruned; the best "
            f"{beam.candidate_width} go on."
        ),
        "evidence": (
            "Depth 2: for one candidate, the generator proposes evidence searches, each a filing "
            "section or a computed metric. Each search runs local hybrid retrieval (keywords and "
            "embeddings, reranked) over both filings, and the critic says whether the passages "
            "state the cause, imply it, are only consistent with it, or say nothing. A search "
            "that finds nothing gets one reformulated search of everywhere. Code keeps the best "
            f"{beam.evidence_width} searches per candidate and prunes the rest."
        ),
        "label": (
            "Each surviving candidate becomes a claim. A fresh-context judge (Claude Opus) sees "
            "only the claim and one quote. Confirmed means the quote is verbatim, from the right "
            "filing, period and section, and states the cause. Plausible means supported but "
            "not stated outright. Causal wording is allowed only in confirmed claims."
        ),
        "rework": (
            "A claim the judge found only implied gets another round of searches, told what was "
            f"already tried, and is judged again: at most {MAX_REWORK_ROUNDS} rounds. A claim "
            "still not confirmed keeps its plausible label. The run never waits for a person."
        ),
        "summary": (
            "Claude Opus writes the brief's summary. Code checks every sentence: each reference "
            "exists, every figure traces to a quote or metric, causal wording only rests on "
            "confirmed claims, and nothing reads as advice. A rejected draft gets one retry, "
            "then a fixed template is used. The brief and its audit trail are then saved: every "
            "node, score, prune reason and quote, and every model call with its cost. The other "
            "tabs read the same files."
        ),
    }


# --- Reading the run's logs -------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """One timed step or one model call. `at` is when it ended (seconds since the epoch)."""

    at: float
    elapsed: float
    stage: str | None
    change_id: str | None
    within: str | None = None
    role: str | None = None
    model: str | None = None
    cost: float = 0.0
    replayed: bool = False

    @property
    def start(self) -> float:
        return self.at - self.elapsed


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut off by a crash
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _when(text: object) -> float | None:
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except ValueError:
        return None


def read_events(folder: Path) -> tuple[list[Event], list[Event]]:
    """The run's step timings and model calls, each in the order they ended."""
    timings, calls = [], []
    for row in _rows(folder / "timings.jsonl"):
        at = _when(row.get("at"))
        if at is not None:
            timings.append(
                Event(
                    at,
                    float(row.get("elapsed_s") or 0.0),
                    row.get("stage"),
                    row.get("change_id"),
                    within=row.get("within"),
                )
            )
    for row in _rows(folder / "ledger.jsonl"):
        at = _when(row.get("at"))
        if at is not None:
            calls.append(
                Event(
                    at,
                    float(row.get("elapsed_s") or 0.0),
                    row.get("stage"),
                    row.get("change_id"),
                    role=row.get("role"),
                    model=row.get("model"),
                    cost=float(row.get("cost_usd") or 0.0),
                    replayed=bool(row.get("replayed")),
                )
            )
    return sorted(timings, key=lambda e: e.at), sorted(calls, key=lambda e: e.at)


# --- Building the steps -----------------------------------------------------------------------


def _plural(count: int, word: str, many: str | None = None) -> str:
    return f"{count} {word if count == 1 else (many or word + 's')}"


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest:02d} s"


def _spread(start: float, end: float, parts: int) -> list[tuple[float, float]]:
    """`parts` equal spans of [start, end]."""
    width = (end - start) / max(parts, 1)
    return [(start + i * width, start + (i + 1) * width) for i in range(parts)]


def _counted(counter: Counter, describe=lambda key: key) -> str:
    return ", ".join(f"{count} {describe(key)}" for key, count in counter.most_common())


class _Builder:
    """Collects the steps, with times relative to the run's start and running totals."""

    def __init__(self, calls: Sequence[Event], t0: float | None) -> None:
        self.calls = list(calls)
        self.t0 = t0
        self.steps: list[dict] = []

    def _rel(self, moment: float | None) -> float | None:
        if moment is None or self.t0 is None:
            return None
        return round(moment - self.t0, 2)

    def calls_between(self, start: float | None, end: float | None, **match) -> list[Event]:
        if start is None or end is None:
            return []
        return [
            call
            for call in self.calls
            if start - 0.01 <= call.at <= end + 0.01
            and all(getattr(call, key) == value for key, value in match.items())
        ]

    def add(
        self,
        kind: str,
        stage: str,
        title: str,
        facts: Sequence[str],
        start: float | None = None,
        end: float | None = None,
        change: int | None = None,
        reveal: Sequence[str] = (),
        after: Sequence[dict] = (),
        focus: str | None = None,
    ) -> None:
        step = {
            "kind": kind,
            "stage": stage,
            "title": title,
            "facts": [fact for fact in facts if fact],
            "start": self._rel(start),
            "end": self._rel(end),
            "change": change,
            "reveal": list(reveal),
            "after": list(after),
            "focus": focus,
        }
        if end is not None and self.t0 is not None:
            done = [call for call in self.calls if call.at <= end + 0.01]
            step["totals"] = {
                "cost": round(sum(call.cost for call in done), 4),
                "live": sum(1 for call in done if not call.replayed),
                "replayed": sum(1 for call in done if call.replayed),
                "roles": dict(Counter(call.role for call in done if call.role)),
            }
        self.steps.append(step)


def _models(calls: Sequence[Event]) -> str:
    return ", ".join(sorted({call.model for call in calls if call.model}))


def _call_fact(calls: Sequence[Event]) -> str | None:
    if not calls:
        return None
    replayed = sum(1 for call in calls if call.replayed)
    cost = sum(call.cost for call in calls)
    fact = f"{_plural(len(calls), 'model call')} ({_models(calls)})"
    if replayed == len(calls):
        return fact + ", all replayed from the cache at no cost."
    fact += f", ${cost:.4f}"
    if replayed:
        fact += f"; {replayed} replayed from the cache at no cost"
    return fact + "."


@dataclass
class _Tree:
    """One change's tree, shaped for the page, with the ids the page uses ("n0", "n1", ...)."""

    nodes: Sequence[TreeNode]
    finding: Finding
    run: rv.SavedRun

    def __post_init__(self) -> None:
        self.js = {node.node_id: f"n{i}" for i, node in enumerate(self.nodes)}
        self.numbers = rv.candidate_numbers(self.nodes)
        self.claims = {claim.claim_id: claim for claim in self.finding.claims}
        self.candidates = [node for node in self.nodes if node.depth == 1]
        self.rounds = {c.node_id: rv.searches_of(self.nodes, c.node_id) for c in self.candidates}
        left_out = set()
        for note in self.run.labeling_failures.get(self.finding.finding_id, ()):
            match = rv._CLAIM_NOTE.match(note)
            if match and match["claim_id"] not in self.claims:
                left_out.add(match["claim_id"])
        self.left_out = left_out

    def reworked(self, candidate: TreeNode) -> bool:
        return any(number > 0 for number in self.rounds[candidate.node_id])

    def final_badge(self, candidate: TreeNode) -> dict | None:
        claim = self.claims.get(candidate.node_id)
        if claim is not None:
            ref = self.run.claim_refs.get(claim.claim_id, "?")
            return {"badge": f"{LABEL_NAMES[claim.label]} [{ref}]", "tone": TONES[claim.label]}
        if candidate.node_id in self.left_out:
            return {"badge": "Left out of the brief", "tone": "left_out"}
        if candidate.status is NodeStatus.KEPT:
            return {"badge": "Not in the brief", "tone": "unused"}
        return None

    def page_nodes(self) -> list[dict]:
        shown = []
        for node in self.nodes:
            item = {
                "id": self.js[node.node_id],
                "depth": node.depth,
                "status": node.status.value,
                "score": None if node.score is None else round(node.score, 2),
                "reason": rv.shorten(node.prune_reason, 110) if node.prune_reason else None,
            }
            if node.depth == 0:
                item["text"] = rv.shorten(node.thought, 500)
            elif node.depth == 1:
                item |= {
                    "num": self.numbers[node.node_id],
                    "text": rv.shorten(node.thought, 700),
                    "fresh": rv.candidate_round(node) > 1,
                }
            else:
                chip = rv.chip_kind(node)
                item |= {
                    "parent": self.js.get(node.parent_id or ""),
                    "round": rv.rework_round(node),
                    "where": rv.short_source_name(node),
                    "chip": chip,
                    "colour": rv.CHIP_STYLES[chip].colour,
                    "meaning": rv.CHIP_STYLES[chip].meaning,
                    "quotes": len(node.citations),
                    "text": rv.shorten(node.thought, 100),
                }
            shown.append(item)
        return shown


def _search_facts(searches: Sequence[TreeNode]) -> list[str]:
    kept = [s for s in searches if s.status is NodeStatus.KEPT]
    pruned = [s for s in searches if s.status is NodeStatus.PRUNED]
    pending = [s for s in searches if s.status is NodeStatus.PENDING]
    places = Counter(rv.short_source_name(s) for s in searches)
    verdicts = Counter(rv.CHIP_STYLES[rv.chip_kind(s)].meaning for s in searches)
    quotes = sum(len(s.citations) for s in searches)
    facts = [
        f"{_plural(len(searches), 'search', 'searches')}: "
        + ", ".join(f"{place} ×{n}" if n > 1 else place for place, n in places.most_common())
        + ".",
        f"What they found: {_counted(verdicts, str)}.",
        f"{len(kept)} kept, {len(pruned)} pruned"
        + (f", {len(pending)} never finished" if pending else "")
        + f"; {_plural(quotes, 'quote')} passed the checks.",
    ]
    reasons = Counter(_reason_kind(s.prune_reason) for s in pruned if s.prune_reason)
    if reasons:
        facts.append(f"Pruned because: {_counted(reasons, str)}.")
    return facts


def _reason_kind(reason: str) -> str:
    """A prune reason without its figures, so like reasons count together."""
    if reason.startswith("outside the beam"):
        return "outside the beam (lower score than the kept searches)"
    if reason.startswith("no evidence after one reformulated"):
        return "no evidence, even after a reformulated search"
    if reason.startswith("no evidence: the metric is the change"):
        return "the metric is the change itself"
    if reason.startswith("no evidence: the metric moved the other way"):
        return "the metric moved the other way"
    if "below the" in reason and "threshold" in reason:
        return "score below the threshold"
    return rv.shorten(reason, 60)


# The Rank step lists the explored changes and this many of those past the run's cap.
SKIPPED_SHOWN = 6
# The Detect step shows this many text changes as aligned before-and-after pairs, each side cut
# to this many words, starting a few words before the first change.
ALIGNMENTS_SHOWN = 3
WORDS_SHOWN = 90
LEAD_IN_WORDS = 12


def _excerpt(runs: Sequence[rv.TextRun]) -> list[list]:
    """Runs as [text, changed] pairs, cut to a window around the first change when the paragraph
    is long; a cut is marked with an ellipsis. Paragraph breaks read as spaces here."""
    marked = [
        (word, changed)
        for text, changed in runs
        if text != rv.PARAGRAPH_BREAK
        for word in text.split()
    ]
    first = next((i for i, (_, changed) in enumerate(marked) if changed), 0)
    start = max(0, min(first - LEAD_IN_WORDS, len(marked) - WORDS_SHOWN))
    end = start + WORDS_SHOWN
    shown: list[list] = [["…", False]] if start > 0 else []
    for word, changed in marked[start:end]:
        if shown and shown[-1][0] != "…" and shown[-1][1] == changed:
            shown[-1][0] += " " + word
        else:
            shown.append([word, changed])
    if end < len(marked):
        shown.append(["…", False])
    return shown


def _alignments(findings: Sequence[Finding], run: rv.SavedRun) -> list[dict]:
    """A few of the run's text changes as the alignment found them: the paragraph in each filing,
    the words that differ marked. Changed paragraphs first, then an added and a removed one."""
    modified = [f for f in findings if f.change.prior_text and f.change.current_text]
    added = [f for f in findings if f.change.current_text and not f.change.prior_text]
    removed = [f for f in findings if f.change.prior_text and not f.change.current_text]
    chosen = (modified[:2] + added[:1] + removed[:1] + modified[2:])[:ALIGNMENTS_SHOWN]
    shown = []
    for finding in chosen:
        change = finding.change
        prior, current = change.prior_text or "", change.current_text or ""
        gone, new = rv.changed_words(prior, current)
        shown.append(
            {
                "ref": run.finding_refs.get(finding.finding_id, ""),
                "heading": run.heading(finding),
                "prior": _excerpt(gone),
                "current": _excerpt(new),
            }
        )
    metrics = [f for f in findings if not (f.change.prior_text or f.change.current_text)]
    for finding in metrics[:3]:
        shown.append(
            {
                "ref": run.finding_refs.get(finding.finding_id, ""),
                "heading": run.heading(finding),
                "metric": rv.change_description(finding, limit=160),
            }
        )
    return shown


def build_replay(run: rv.SavedRun) -> dict:
    """The run's steps for the replay page: the pipeline in order, each change's search split
    into its Tree-of-Thought steps, with times, running cost and what each step showed."""
    timings, calls = read_events(run.folder)
    starts = [event.start for event in timings] + [call.start for call in calls if timings]
    t0 = min(starts) if timings and starts else None
    build = _Builder(calls, t0)
    findings = run.brief.findings
    comparison = run.brief.comparison
    by_change = {finding.change.change_id: finding for finding in findings}

    def timing(stage: str, change_id: str | None = None) -> list[Event]:
        return [
            event
            for event in timings
            if event.stage == stage and event.within is None and event.change_id == change_id
        ]

    # 1. Detect and rank.
    detect = (timing("detect") or [None])[0]
    build.add(
        "detect",
        "detect",
        "Detect: compare the two filings",
        [
            f"{run.brief.comparison.current.form.value} for "
            f"{run.brief.comparison.current.period.label} against "
            f"{run.brief.comparison.prior.form.value} for "
            f"{run.brief.comparison.prior.period.label}.",
            f"Took {_duration(detect.elapsed)}." if detect else None,
            "No model calls.",
        ],
        detect.start if detect else None,
        detect.at if detect else None,
    )
    rank = (timing("materiality") or [None])[0]
    rank_calls = [call for call in calls if call.role == "materiality"]
    skipped = run.not_examined.changes if run.not_examined else ()
    material = len(findings) + len(skipped)
    rank_facts = [
        _call_fact(rank_calls),
        f"{_plural(material, 'change')} rated material.",
        (
            f"The {len(findings)} most important are explored; {len(skipped)} more are past the "
            f"run's cap of {run.not_examined.max_changes} and go to the review queue."
            if skipped and run.not_examined
            else f"All {len(findings)} are explored."
        ),
    ]
    build.add(
        "materiality",
        "materiality",
        "Rank: which changes matter",
        rank_facts,
        rank.start if rank else (rank_calls[0].start if rank_calls and t0 else None),
        rank.at if rank else (rank_calls[-1].at if rank_calls and t0 else None),
    )

    # 2. Each change, in the order the run explored them.
    order: list[str] = []
    for event in timings:
        explored = event.stage == "search" and event.within is None
        if explored and event.change_id in by_change and event.change_id not in order:
            order.append(event.change_id)
    order += [change_id for change_id in by_change if change_id not in order]
    changes = []
    for index, change_id in enumerate(order):
        finding = by_change[change_id]
        nodes = run.trees.get(finding.finding_id, ())
        tree = _Tree(nodes, finding, run) if nodes else None
        root = next((node for node in nodes if node.depth == 0), None)
        changes.append(
            {
                "ref": run.finding_refs.get(finding.finding_id, ""),
                "heading": run.heading(finding),
                "desc": rv.change_description(finding, limit=160),
                "rank": finding.change.materiality_rank,
                "tone": TONES[strongest_label(finding)] if finding.claims else None,
                "nodes": tree.page_nodes() if tree else [],
                "root": tree.js[root.node_id] if tree and root else None,
            }
        )
        search = (timing("search", change_id) or [None])[0]
        label = (timing("label", change_id) or [None])[0]
        reworks = timing("rework", change_id)
        rationale = rv.shown_rationale(finding)
        ref = changes[-1]["ref"]
        intro = [
            (
                f"Listed in the brief as {ref}. The run explores changes in materiality order"
                + (
                    f" (this is rank {finding.change.materiality_rank})"
                    if finding.change.materiality_rank
                    else ""
                )
                + "; the brief lists them by their strongest label, confirmed first, so its "
                "numbers can differ from this order."
            )
            if ref != f"F{index + 1}"
            else None,
            f"Why it was flagged: {rv.shorten(rationale, 260)}" if rationale else None,
            f"Before: {rv.shorten(finding.change.prior_text, 160)}"
            if finding.change.prior_text
            else None,
            f"After: {rv.shorten(finding.change.current_text, 160)}"
            if finding.change.current_text
            else None,
        ]
        if tree is None:
            intro.append("This change's search tree was not saved.")
        build.add(
            "change",
            "search",
            f"Change {index + 1} of {len(order)}: {changes[-1]['heading']} ({ref} in the brief)",
            intro,
            search.start if search else None,
            search.start if search else None,
            change=index,
            reveal=[tree.js[root.node_id]] if tree and root else [],
        )
        if tree is None:
            continue
        _search_steps(build, tree, index, search, calls)
        _label_step(build, tree, index, label, calls)
        _rework_steps(build, tree, index, reworks, calls)

    # 3. Write, and save: one closing step.
    summary = (timing("summary") or [None])[0]
    summary_calls = [call for call in calls if call.role == "brief"]
    labels = Counter(claim.label for finding in findings for claim in finding.claims)
    review = rv.review_items(run)
    end = max([event.at for event in timings] + [call.at for call in calls]) if t0 else None
    build.add(
        "summary",
        "summary",
        "Write: the brief's summary, then save the brief",
        [
            _call_fact(summary_calls),
            "The writer's draft failed the checks, so the fixed template was used."
            if run.summary_used_template
            else ("The draft passed the checks." if run.summary_used_template is False else None),
            f"{_plural(len(findings), 'change')} explored: "
            + (
                _counted(
                    Counter({LABEL_NAMES[label]: count for label, count in labels.items()}),
                    lambda name: name.lower(),
                )
                or "no claims"
            )
            + " explanations.",
            f"{_plural(len(review), 'item')} in the review queue.",
            f"Model cost ${run.brief.cost_usd:.2f}"
            + (f" over {_duration(end - t0)}." if end is not None and t0 is not None else "."),
        ],
        summary.start if summary else end,
        end,
    )
    return {
        "run": {
            "id": run.run_id,
            "timed": t0 is not None,
            "cost": run.brief.cost_usd,
        },
        "overview": {
            "current": f"{comparison.current.form.value} for {comparison.current.period.label}",
            "prior": f"{comparison.prior.form.value} for {comparison.prior.period.label}",
            "material": material,
            "alignments": _alignments([by_change[change_id] for change_id in order], run),
            "ranked": [
                {
                    "order": order.index(f.change.change_id) + 1,
                    "ref": run.finding_refs.get(f.finding_id, ""),
                    "rank": f.change.materiality_rank,
                    "heading": run.heading(f),
                    "desc": rv.change_description(f, limit=120),
                    "criterion": rv.words(f.change.criterion.value) if f.change.criterion else None,
                }
                for f in sorted(
                    findings,
                    key=lambda f: (
                        f.change.materiality_rank or 10**6,
                        order.index(f.change.change_id),
                    ),
                )
            ],
            "skipped": [
                {
                    "rank": skip.materiality_rank,
                    "heading": rv.describe_change_id(skip.change_id),
                    "importance": skip.importance,
                }
                for skip in list(skipped)[:SKIPPED_SHOWN]
            ],
            "explored": len(findings),
            "cap": run.not_examined.max_changes if run.not_examined else None,
            "labels": {TONES[label]: labels.get(label, 0) for label in ExplanationLabel},
            "review": len(review),
        },
        "stages": [{"key": key, "name": STAGE_NAMES[key]} for key in STAGES],
        "notes": workflow_notes(),
        "chips": [
            {"key": key, "colour": style.colour, "meaning": style.meaning}
            for key, style in rv.CHIP_STYLES.items()
        ],
        "changes": changes,
        "steps": build.steps,
    }


def _search_steps(
    build: _Builder,
    tree: _Tree,
    index: int,
    search: Event | None,
    calls: Sequence[Event],
) -> None:
    """Propose and pre-screen each round of candidates, then one step per candidate's searches."""
    units: list[tuple[str, TreeNode | int]] = []
    for number in sorted({rv.candidate_round(c) for c in tree.candidates}):
        units.append(("propose", number))
        for candidate in tree.candidates:
            if rv.candidate_round(candidate) == number and tree.rounds[candidate.node_id].get(0):
                units.append(("evidence", candidate))
    change_id = tree.finding.change.change_id
    generator = [
        call
        for call in calls
        if call.change_id == change_id and call.stage == "search" and call.role == "generator"
    ]
    spans: list[tuple[float | None, float | None]]
    mids: list[float | None]
    if search is None or build.t0 is None:
        spans = [(None, None)] * len(units)
        mids = [None] * len(units)
    elif len(generator) == len(units):
        starts = [search.start] + [call.start for call in generator[1:]]
        ends = starts[1:] + [search.at]
        spans = list(zip(starts, ends, strict=True))
        mids = [call.at for call in generator]
    else:
        spans = _spread(search.start, search.at, len(units))
        mids = [(start + end) / 2 for start, end in spans]
    for (kind, subject), (start, end), mid in zip(units, spans, mids, strict=True):
        if kind == "propose":
            assert isinstance(subject, int)
            batch = [c for c in tree.candidates if rv.candidate_round(c) == subject]
            gen_calls = build.calls_between(start, mid, role="generator")
            build.add(
                "propose",
                "search",
                "Propose candidate explanations" + (" (a fresh round)" if subject > 1 else ""),
                [
                    f"The generator proposed {_plural(len(batch), 'candidate')}.",
                    "Every candidate in the first round was pruned, so a fresh round was tried."
                    if subject > 1
                    else None,
                    _call_fact(gen_calls),
                ],
                start,
                mid,
                change=index,
                reveal=[tree.js[c.node_id] for c in batch],
            )
            screened = [c for c in batch if not tree.rounds[c.node_id]]
            pruned = [c for c in screened if c.status is NodeStatus.PRUNED]
            going = [c for c in batch if c not in pruned]
            reasons = Counter(_reason_kind(c.prune_reason or "pruned") for c in pruned)
            build.add(
                "prescreen",
                "search",
                "Pre-screen the candidates",
                [
                    f"{len(going)} go on to the evidence search; {len(pruned)} pruned"
                    + (f" ({_counted(reasons, str)})." if pruned else "."),
                    "Scores: "
                    + ", ".join(
                        f"#{tree.numbers[c.node_id]} "
                        + ("-" if c.score is None else f"{c.score:.2f}")
                        for c in batch
                    )
                    + ".",
                    _call_fact(build.calls_between(mid, end, role="critic")),
                ],
                mid,
                end,
                change=index,
                after=[
                    {"id": tree.js[c.node_id], "status": "pruned" if c in pruned else "screened"}
                    for c in batch
                ],
            )
            continue
        assert isinstance(subject, TreeNode)
        searches = tree.rounds[subject.node_id].get(0, [])
        after = [{"id": tree.js[s.node_id], "status": s.status.value} for s in searches]
        facts = _search_facts(searches)
        if subject.status is NodeStatus.PRUNED:
            after.append({"id": tree.js[subject.node_id], "status": "pruned"})
            facts.append(
                "The candidate is pruned: "
                + rv.shorten(subject.prune_reason or "no search kept it", 120)
            )
        else:
            after.append({"id": tree.js[subject.node_id], "status": "kept"})
        step_calls = build.calls_between(start, end)
        facts.append(_call_fact([call for call in step_calls if call.change_id == change_id]))
        build.add(
            "evidence",
            "search",
            f"Candidate #{tree.numbers[subject.node_id]}: search for evidence",
            facts,
            start,
            end,
            change=index,
            reveal=[tree.js[s.node_id] for s in searches],
            after=after,
            focus=tree.js[subject.node_id],
        )


def _label_step(
    build: _Builder,
    tree: _Tree,
    index: int,
    label: Event | None,
    calls: Sequence[Event],
) -> None:
    after = []
    counts: Counter = Counter()
    sent = 0
    for candidate in tree.candidates:
        badge = tree.final_badge(candidate)
        if tree.reworked(candidate):
            # Rework is only for claims the judge found only implied; the final label comes
            # after the last round.
            badge = {"badge": "Only implied: back for rework", "tone": "implied"}
            sent += 1
        elif badge is None:
            continue
        else:
            counts[badge["badge"].split(" [")[0].lower()] += 1
        after.append({"id": tree.js[candidate.node_id], **badge})
    change_id = tree.finding.change.change_id
    step_calls = [call for call in calls if call.change_id == change_id and call.stage == "label"]
    roles = Counter(call.role for call in step_calls)
    facts = [
        ("Labeled: " + _counted(counts, str) + ".") if counts else "No claim was labeled yet.",
        f"{_plural(sent, 'claim')} only implied, sent back for rework." if sent else None,
        _call_fact(step_calls),
        (f"The judge ran {roles['entailment']} times; the claim writer {roles['generator']} times.")
        if step_calls
        else None,
    ]
    if tree.finding.needs_review:
        facts.append(
            "Needs review: " + (tree.finding.review_reason or "an explanation is unresolved")
        )
    build.add(
        "label",
        "label",
        "Label: judge each claim against its quote",
        facts,
        label.start if label else None,
        label.at if label else None,
        change=index,
        after=after,
    )


def _rework_steps(
    build: _Builder,
    tree: _Tree,
    index: int,
    reworks: Sequence[Event],
    calls: Sequence[Event],
) -> None:
    groups = sorted(
        (
            (number, candidate)
            for candidate in tree.candidates
            for number in tree.rounds[candidate.node_id]
            if number > 0
        ),
        key=lambda group: (group[0], tree.numbers[group[1].node_id]),
    )
    if not groups:
        return
    last_round = {
        candidate.node_id: max(n for n in tree.rounds[candidate.node_id]) for _, candidate in groups
    }
    spans: list[tuple[float | None, float | None]]
    if not reworks or build.t0 is None:
        spans = [(None, None)] * len(groups)
    elif len(reworks) == len(groups):
        spans = [(event.start, event.at) for event in reworks]
    else:
        spans = _spread(reworks[0].start, reworks[-1].at, len(groups))
    change_id = tree.finding.change.change_id
    for (number, candidate), (start, end) in zip(groups, spans, strict=True):
        searches = tree.rounds[candidate.node_id][number]
        after = [{"id": tree.js[s.node_id], "status": s.status.value} for s in searches]
        facts = _search_facts(searches)
        if last_round[candidate.node_id] == number:
            badge = tree.final_badge(candidate)
            if badge is not None:
                after.append({"id": tree.js[candidate.node_id], **badge})
                facts.append(f"Judged again: {badge['badge']}.")
        else:
            facts.append("Still only implied; another round follows.")
        step_calls = [
            call for call in build.calls_between(start, end) if call.change_id == change_id
        ]
        facts.append(_call_fact(step_calls))
        build.add(
            "rework",
            "rework",
            f"Rework round {number}: candidate #{tree.numbers[candidate.node_id]}",
            facts,
            start,
            end,
            change=index,
            reveal=[tree.js[s.node_id] for s in searches],
            after=after,
            focus=tree.js[candidate.node_id],
        )
