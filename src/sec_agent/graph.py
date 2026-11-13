"""The workflow (T5.5): one comparison, from its two filings to a written, cited brief.

A deterministic LangGraph controller runs the steps in order; no model decides the route
(architecture plan: "four agents, no planner"). The nodes:

1. `detect`: paragraph alignment (text changes, T3.1) and the metric scan (numeric changes, T3.2).
   No model, no cost.
2. `estimate`: the materiality headroom check, then a cost estimate shown to the user, who may
   decline it (safety plan: an estimate before spending; `Workflow.confirm`). Nothing is spent
   until it passes.
3. `rank`: the materiality ranking (T3.3), the only paid step before the search.
4. `select`: keeps the first `max_changes` ranked changes. The rest are not examined: the brief
   says how many in one line and `audit.json` lists them (user's decision, 2026-11-12). Changes
   materiality could not decide go to review as findings with no claims.
5. `explore`: one change per pass, most material first. Tree-of-Thought search (T4.4), then
   labeling (T4.5) of what survived. Each pass ends at a finished finding, so a checkpoint after
   it holds every finding so far (the unit resume is built on).
6. `summarize`: the brief's summary (T5.4).
7. `write`: the `Brief` record, each finding's tree, and `write_run`, which refuses to write a
   brief that breaks a rule.

`explore` does what `explore_changes` does (budget snapshot after each change; once the dollar
cap stops a search, the remaining changes are filed unexamined), one change at a time, so the
graph can checkpoint between changes and label each change as soon as its search ends.

Budget stops. The ledger's dollar cap is one per run. `explore_change` catches
`BudgetExceededError` itself. `label_result` and `write_summary` do not, so the graph decides:
- a stop while labeling files that change as a finding with no claims, under review, and the
  remaining changes are not searched, as after a stop inside the search;
- a stop while summarizing uses the template summary, which costs nothing.
A stop in `rank` ends the run with no brief: with no ranking there is nothing to examine in order.
Two checks keep money from being spent on work the cap cannot finish (user's decision, T5.5):
- before `rank` sends anything (in `estimate`), the cap must have free what the materiality step
  can hold at once (`materiality_headroom`); if not, the run ends with no brief, having spent
  nothing;
- before each new search, the cap must have `labeling_reserve_usd` free, so the change's
  explanations can still be checked and the summary written. A change that starts with less is
  filed for review unexamined, and so is every change after it.

The steps are passed in as `Stages`, so tests drive the graph with fakes. `run_analysis` builds
the real ones.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import Field

from sec_agent.agents.retrieval import RetrievalAgent
from sec_agent.agents.verification import EntailmentJudge
from sec_agent.compare.align import align_comparison
from sec_agent.compare.embedding_cache import CachedEmbedder, embedding_cache_path
from sec_agent.compare.materiality import (
    MaterialityHeadroom,
    MaterialityResult,
    materiality_headroom,
    rank_changes,
)
from sec_agent.config import Settings
from sec_agent.contracts import MetricLookup
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore
from sec_agent.finance.metrics import MetricCatalog, MetricsConfig
from sec_agent.finance.scan import MetricScan, scan_comparison
from sec_agent.ingest.chunker import EMBEDDING_MODEL
from sec_agent.ingest.index import BgeEmbedder, ChunkIndex
from sec_agent.llm.cache import ReplayCache
from sec_agent.llm.client import LLMClient
from sec_agent.llm.ledger import BudgetExceededError, CostLedger
from sec_agent.report.brief import NotExamined, RunFiles, SkippedChange, write_run
from sec_agent.report.summary import (
    SummaryContext,
    SummaryResult,
    SummaryWriter,
    template_summary,
    write_summary,
)
from sec_agent.retrieval.hybrid import HybridSearcher
from sec_agent.retrieval.rerank import CrossEncoderScorer, Reranker
from sec_agent.retrieval.tools import SearchRetriever
from sec_agent.schemas import (
    Brief,
    Change,
    Comparison,
    ComparisonType,
    Finding,
    RankedChanges,
    Record,
)
from sec_agent.tot.controller import StopReason, ToTContext, ToTResult, explore_change
from sec_agent.tot.critic import Critic
from sec_agent.tot.generator import ThoughtGenerator
from sec_agent.tot.labeling import ClaimWriter, LabeledFinding, LabelingContext, label_result
from sec_agent.tot.scoring import ScoringConfig
from sec_agent.tree_mcp.client import TreeClient, connect_tree_store
from sec_agent.tree_mcp.store import BudgetUse

# How many ranked changes a run explores unless told otherwise (user's decision, 2026-11-12, "for
# now"). Chosen by hand: in Hershey's yoy-10k ranking it covers every importance-5 change and the
# labeled tariff change (rank 10). At 1 to 4 minutes of search per change, 10 changes take about
# 10 to 40 minutes; all 52 material changes would take hours. Revisit with measured time and cost.
DEFAULT_MAX_CHANGES = 10

# What a search must leave free for after it: the entailment judge's and the summary's worst-case
# reservations on Opus (about $0.09 each at their 4,000-token caps) plus what they actually cost.
# Chosen by hand, provisional: the first paid run's measured labeling and summary costs replace it.
LABELING_RESERVE_USD = 0.25

# The cost estimate's figures (`estimate_cost`). Rough on purpose: the estimate is for deciding
# whether to go ahead, and the hard stop is the dollar cap, not these.
# One 20-change materiality batch cost $0.074 to $0.075 live (lane 1, 2026-11-11, two calls).
MATERIALITY_BATCH_USD = 0.075
# One change's search (generator, critic and retrieval on Haiku) and labeling (the judge and the
# claim writer). Chosen by hand, provisional: the guess before any measurement was $0.04 to $0.13,
# so it is set at the top and the estimate errs high. The first paid run measures it.
PER_CHANGE_USD = 0.13
# The one live summary (2026-11-10) used 516 output tokens on Opus, about $0.02.
SUMMARY_USD = 0.02

# Review reasons the graph writes. Each must pass the causal-language check (`check_brief`).
UNREACHED_REASON = "the run's dollar budget was reached before this change was examined"
RESERVE_REASON = (
    "too little of the run's dollar budget was left to examine this change and check its "
    "explanations"
)
LABELING_BUDGET_REASON = (
    "the run's dollar budget was reached while this change's explanations were being checked"
)
MATERIALITY_UNRESOLVED_REASON = (
    "the materiality step got no usable answer for this change, so it was not examined"
)
SUMMARY_BUDGET_NOTE = "the run's dollar budget was reached before the summary was written"


class CostEstimate(Record):
    """What a run is expected to cost, shown before anything is spent (`Workflow.confirm`).

    Made before the ranking, so how many changes will be material is not known yet: the changes'
    share assumes all `max_changes` are searched, which is the most the run will do.
    """

    materiality_batches: int = Field(ge=0)
    materiality_uncached: int = Field(ge=0)
    materiality_usd: float = Field(ge=0)
    max_changes: int = Field(ge=1)
    changes_usd: float = Field(ge=0)
    summary_usd: float = Field(ge=0)
    total_usd: float = Field(ge=0)
    cap_usd: float = Field(ge=0)


def estimate_cost(headroom: MaterialityHeadroom, max_changes: int, cap_usd: float) -> CostEstimate:
    """The run's estimate: uncached materiality batches, `max_changes` searches, one summary."""
    materiality = headroom.uncached * MATERIALITY_BATCH_USD
    changes = max_changes * PER_CHANGE_USD
    return CostEstimate(
        materiality_batches=headroom.batches,
        materiality_uncached=headroom.uncached,
        materiality_usd=materiality,
        max_changes=max_changes,
        changes_usd=changes,
        summary_usd=SUMMARY_USD,
        total_usd=materiality + changes + SUMMARY_USD,
        cap_usd=cap_usd,
    )


def describe_estimate(estimate: CostEstimate) -> list[str]:
    """The estimate as the CLI prints it before asking whether to go ahead."""
    lines = [
        f"Estimated model cost: about ${estimate.total_usd:.2f}, "
        f"with a hard stop at ${estimate.cap_usd:.2f}.",
        f"  Materiality: {estimate.materiality_uncached} of {estimate.materiality_batches} "
        f"batches need a call, the rest are cached: ${estimate.materiality_usd:.2f}.",
        f"  Search and checking: up to {estimate.max_changes} change(s) at about "
        f"${PER_CHANGE_USD:.2f} each: ${estimate.changes_usd:.2f}.",
        f"  Summary: ${estimate.summary_usd:.2f}.",
        "The per-change figure is a guess until a paid run measures it.",
    ]
    if estimate.total_usd > estimate.cap_usd:
        lines.append(
            "The estimate is above the hard stop, so some changes may be filed for review "
            "unexamined."
        )
    return lines


class RunDeclined(RuntimeError):
    """The user declined the cost estimate. Nothing was spent and no brief is written."""


@dataclass(frozen=True)
class Stages:
    """The steps the graph calls. `run_analysis` builds the real ones; tests pass fakes."""

    # Text changes and the metric scan for a comparison.
    detect: Callable[[Comparison], tuple[Sequence[Change], MetricScan]]
    # `materiality_headroom` with its client bound: what `rank` could hold at once, sending nothing.
    headroom: Callable[[Comparison, Sequence[Change], MetricScan], MaterialityHeadroom]
    # `rank_changes` with its dependencies bound.
    rank: Callable[[Comparison, Sequence[Change], MetricScan], Awaitable[MaterialityResult]]
    # `explore_change` with its `ToTContext` bound: (change, comparison, run_id).
    explore: Callable[[Change, Comparison, str], Awaitable[ToTResult]]
    # `label_result` with its `LabelingContext` bound: (result, change, comparison, run_id).
    label: Callable[[ToTResult, Change, Comparison, str], Awaitable[LabeledFinding]]
    # `write_summary` with its `SummaryContext` bound.
    summarize: Callable[[Sequence[Finding], Comparison], Awaitable[SummaryResult]]


@dataclass(frozen=True)
class Workflow:
    """Everything the graph needs besides its state. Built once per run, never checkpointed."""

    stages: Stages
    tree: TreeClient
    ledger: CostLedger
    runs_dir: Path
    # Stored in the audit trail beside the brief; None in tests that do not check metrics.
    metrics: MetricLookup | None = None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    labeling_reserve_usd: float = LABELING_RESERVE_USD
    # Shown the estimate before anything is spent; False declines the run. None asks nothing
    # (tests, and the CLI's `--yes` prints the estimate and goes ahead). Run in a thread, since
    # the CLI's prompt blocks on the terminal.
    confirm: Callable[[CostEstimate], bool] | None = None

    def free_usd(self) -> float:
        """What the cap has left, not counting reservations: none are held between steps."""
        return self.ledger.max_usd - self.ledger.spent_usd


class RunState(TypedDict, total=False):
    """The graph's state. Records only, so a checkpoint can hold it."""

    run_id: str
    comparison: Comparison
    max_changes: int
    text_changes: tuple[Change, ...]
    scan: MetricScan
    estimate: CostEstimate
    materiality: MaterialityResult
    # The changes this run explores, most material first, and where the next one is.
    to_explore: tuple[Change, ...]
    position: int
    not_examined: NotExamined
    # Findings materiality could not decide, filed for review before any search.
    unresolved: tuple[LabeledFinding, ...]
    # One per explored change, in order.
    findings: tuple[LabeledFinding, ...]
    node_evaluations: int
    out_of_budget: bool
    summary: SummaryResult
    brief_path: str
    audit_path: str


def finding_id(run_id: str, change: Change) -> str:
    # The same id `label_result` gives, so every finding of a run is keyed alike.
    return f"{run_id}:{change.change_id}"


def claimless(run_id: str, change: Change, reason: str, *, failures=()) -> LabeledFinding:
    """A finding with no claims, under review for `reason`."""
    return LabeledFinding(
        finding=Finding(finding_id=finding_id(run_id, change), change=change, review_reason=reason),
        failures=tuple(failures),
    )


def select_changes(
    ranked: RankedChanges, materiality: MaterialityResult, max_changes: int
) -> tuple[tuple[Change, ...], NotExamined]:
    """The first `max_changes` ranked changes, and the rest as not examined."""
    if max_changes < 1:
        raise ValueError("max_changes must be at least 1")
    importance = {decision.change_id: decision.importance for decision in materiality.decisions}
    chosen = ranked.changes[:max_changes]
    rest = tuple(
        SkippedChange(
            change_id=change.change_id,
            materiality_rank=change.materiality_rank,
            importance=importance.get(change.change_id),
        )
        for change in ranked.changes[max_changes:]
    )
    return chosen, NotExamined(max_changes=max_changes, changes=rest)


def build_graph(workflow: Workflow, checkpointer=None):
    """The compiled graph. Invoke it with `initial_state(...)`."""
    stages = workflow.stages

    async def detect(state: RunState) -> RunState:
        # Synchronous and CPU-bound, run on the event loop's thread: nothing else runs yet, and
        # alignment reads DuckDB, which must never be read from two threads at once.
        text_changes, scan = stages.detect(state["comparison"])
        return {"text_changes": tuple(text_changes), "scan": scan}

    async def estimate(state: RunState) -> RunState:
        comparison, text_changes, scan = state["comparison"], state["text_changes"], state["scan"]
        needed = stages.headroom(comparison, text_changes, scan)
        if needed.headroom_usd > workflow.free_usd():
            raise BudgetExceededError(
                f"Budget of ${workflow.ledger.max_usd:.2f} is too small for the materiality step: "
                f"{needed.uncached} of its {needed.batches} batches are not cached, and the ones "
                f"run at once can hold up to ${needed.headroom_usd:.2f}, with "
                f"${workflow.free_usd():.2f} free. Nothing was sent."
            )
        cost = estimate_cost(needed, state["max_changes"], workflow.ledger.max_usd)
        if workflow.confirm is not None and not await asyncio.to_thread(workflow.confirm, cost):
            raise RunDeclined("the cost estimate was declined. Nothing was sent.")
        return {"estimate": cost}

    async def rank(state: RunState) -> RunState:
        result = await stages.rank(state["comparison"], state["text_changes"], state["scan"])
        return {"materiality": result}

    async def select(state: RunState) -> RunState:
        materiality = state["materiality"]
        chosen, rest = select_changes(materiality.ranked, materiality, state["max_changes"])
        by_id = {
            change.change_id: change for change in (*state["scan"].changes, *state["text_changes"])
        }
        unresolved = tuple(
            claimless(
                state["run_id"],
                by_id[item.change_id],
                MATERIALITY_UNRESOLVED_REASON,
                failures=(f"materiality: {item.reason}",),
            )
            for item in materiality.unresolved
        )
        return {
            "to_explore": chosen,
            "position": 0,
            "not_examined": rest,
            "unresolved": unresolved,
            "findings": (),
            "node_evaluations": state.get("node_evaluations", 0),
            "out_of_budget": False,
        }

    async def explore(state: RunState) -> RunState:
        run_id, comparison = state["run_id"], state["comparison"]
        change = state["to_explore"][state["position"]]
        evaluations = state["node_evaluations"]
        out_of_budget = state["out_of_budget"]
        if out_of_budget:
            labeled = claimless(run_id, change, UNREACHED_REASON)
        elif workflow.free_usd() < workflow.labeling_reserve_usd:
            # Not `out_of_budget`: nothing more is spent, so each later change gets this reason too.
            labeled = claimless(run_id, change, RESERVE_REASON)
        else:
            result = await stages.explore(change, comparison, run_id)
            evaluations += result.node_evaluations
            await workflow.tree.record_budget(
                BudgetUse(
                    run_id=run_id,
                    spent_usd=workflow.ledger.spent_usd,
                    max_usd=workflow.ledger.max_usd,
                    node_evaluations=evaluations,
                )
            )
            out_of_budget = result.stopped is StopReason.RUN_BUDGET
            try:
                labeled = await stages.label(result, change, comparison, run_id)
            except BudgetExceededError:
                labeled = LabeledFinding(
                    finding=Finding(
                        finding_id=finding_id(run_id, change),
                        change=change,
                        tree_root_id=result.tree_root_id,
                        review_reason=LABELING_BUDGET_REASON,
                    ),
                )
                out_of_budget = True
        return {
            "findings": (*state["findings"], labeled),
            "position": state["position"] + 1,
            "node_evaluations": evaluations,
            "out_of_budget": out_of_budget,
        }

    def after_select(state: RunState) -> str:
        return "explore" if state["to_explore"] else "summarize"

    def after_explore(state: RunState) -> str:
        return "explore" if state["position"] < len(state["to_explore"]) else "summarize"

    async def summarize(state: RunState) -> RunState:
        findings = _brief_findings(state)
        try:
            summary = await stages.summarize(findings, state["comparison"])
        except BudgetExceededError:
            summary = SummaryResult(
                text=template_summary(findings, workflow.metrics),
                used_template=True,
                failures=(SUMMARY_BUDGET_NOTE,),
            )
        return {"summary": summary}

    async def write(state: RunState) -> RunState:
        labeled = (*state["findings"], *state["unresolved"])
        findings = _brief_findings(state)
        brief = Brief(
            run_id=state["run_id"],
            comparison=state["comparison"],
            generated_at=workflow.clock(),
            summary=state["summary"].text,
            findings=findings,
            cost_usd=workflow.ledger.spent_usd,
        )
        trees = {
            finding.finding_id: await workflow.tree.read_tree(finding.tree_root_id)
            for finding in findings
            if finding.tree_root_id is not None
        }
        files: RunFiles = write_run(
            brief,
            workflow.runs_dir / state["run_id"],
            trees=trees,
            labeling_failures={
                item.finding.finding_id: item.failures for item in labeled if item.failures
            },
            metrics=workflow.metrics,
            not_examined=state["not_examined"],
            untraced_rationales=tuple(
                decision for decision in state["materiality"].decisions if decision.untraced_figures
            ),
            summary_used_template=state["summary"].used_template,
            summary_failures=state["summary"].failures,
        )
        return {"brief_path": str(files.brief), "audit_path": str(files.audit)}

    builder = StateGraph(RunState)
    builder.add_node("detect", detect)
    builder.add_node("estimate", estimate)
    builder.add_node("rank", rank)
    builder.add_node("select", select)
    builder.add_node("explore", explore)
    builder.add_node("summarize", summarize)
    builder.add_node("write", write)
    builder.add_edge(START, "detect")
    builder.add_edge("detect", "estimate")
    builder.add_edge("estimate", "rank")
    builder.add_edge("rank", "select")
    builder.add_conditional_edges("select", after_select, ["explore", "summarize"])
    builder.add_conditional_edges("explore", after_explore, ["explore", "summarize"])
    builder.add_edge("summarize", "write")
    builder.add_edge("write", END)
    return builder.compile(checkpointer=checkpointer)


def _brief_findings(state: RunState) -> tuple[Finding, ...]:
    # Explored findings in materiality order, then the changes materiality could not decide.
    return tuple(item.finding for item in (*state["findings"], *state["unresolved"]))


def initial_state(run_id: str, comparison: Comparison, max_changes: int) -> RunState:
    if max_changes < 1:
        raise ValueError("max_changes must be at least 1")
    return {"run_id": run_id, "comparison": comparison, "max_changes": max_changes}


def recursion_limit(max_changes: int) -> int:
    # LangGraph counts every node run as a step and stops at its limit (25 by default). A run takes
    # five fixed steps plus one per explored change; the margin of 5 is chosen by hand.
    return max_changes + 10


def new_run_id(comparison: Comparison, now: datetime | None = None) -> str:
    """A fresh run id. Tree node ids are built from it, and the tree store refuses a repeat."""
    # No colons: node ids join their parts with ":", and the id names the run's folder.
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{comparison.ticker}-{comparison.comparison_type.value}-{stamp}"


@dataclass(frozen=True)
class RunReport:
    """What `run_analysis` produced, for the CLI to print."""

    run_id: str
    brief: Path
    audit: Path
    cost_usd: float
    estimated_usd: float
    # The changes the cap let in, and how many of them a search actually ran for: a budget stop or
    # the labeling reserve files the rest unexamined, with no tree.
    selected: int
    searched: int
    not_examined: int
    node_evaluations: int


async def run_analysis(  # noqa: PLR0913 -- the CLI's options, passed through
    settings: Settings,
    ticker: str,
    kind: ComparisonType,
    *,
    max_changes: int = DEFAULT_MAX_CHANGES,
    max_usd: float | None = None,
    run_id: str | None = None,
    corpus_path: Path,
    confirm: Callable[[CostEstimate], bool] | None = None,
) -> RunReport:
    """Analyze one comparison with the real pieces and write its brief.

    Wiring from both lanes' hand-offs: one read-only fact store per process (it is also the chunk
    and fact lookup); one bge model shared by the embedding cache and the chunk index; one
    retriever on this event loop; the tree store opened once around the whole run, in this task.
    `confirm` sees the cost estimate before anything is spent (`Workflow.confirm`); if it returns
    False, `RunDeclined` is raised and nothing is written.
    """
    corpus = Corpus.load(corpus_path)
    comparison = corpus.comparison(ticker, kind)
    run_id = run_id or new_run_id(comparison)
    run_dir = settings.runs_dir / run_id
    ledger = CostLedger(
        max_usd if max_usd is not None else settings.max_usd_per_run,
        path=run_dir / "ledger.jsonl",
    )
    cache = ReplayCache(settings.llm_cache_path) if settings.llm_cache else None
    llm = LLMClient(settings, ledger, cache=cache)
    store = FactStore(settings.duckdb_path, read_only=True)
    bge = BgeEmbedder()
    embedder = CachedEmbedder(bge, EMBEDDING_MODEL, embedding_cache_path(settings))
    index = ChunkIndex(settings.chroma_dir, bge)
    try:
        catalog = MetricCatalog(store, corpus, MetricsConfig.load())
        # Rewording is off (the agent's default), so this is one plain search per call; turning
        # it on is a flag here and needs the user (decision 2, 2026-11-12).
        retriever = RetrievalAgent(
            SearchRetriever(Reranker(HybridSearcher(store, index), CrossEncoderScorer())), llm
        )

        def detect(target: Comparison) -> tuple[Sequence[Change], MetricScan]:
            alignment = align_comparison(target, settings.parsed_filings_dir, embedder, store)
            return alignment.changes, scan_comparison(catalog, target)

        async def rank(target, text_changes, scan) -> MaterialityResult:
            return await rank_changes(
                target, text_changes, scan, llm, levels=catalog, chunks=store, facts=store
            )

        async with connect_tree_store(settings.tree_db_path) as tree:
            tot = ToTContext(
                retriever=retriever,
                checker=catalog,
                metrics=catalog,
                generator=ThoughtGenerator(llm),
                critic=Critic(llm),
                tree=tree,
                config=ScoringConfig.load(),
            )
            labeling = LabelingContext(
                judge=EntailmentJudge(llm),
                writer=ClaimWriter(llm),
                chunks=store,
                metrics=catalog,
                facts=store,
            )
            summary = SummaryContext(
                writer=SummaryWriter(llm), chunks=store, metrics=catalog, facts=store
            )
            stages = Stages(
                detect=detect,
                headroom=lambda target, changes, scan: materiality_headroom(
                    target, changes, scan, llm
                ),
                rank=rank,
                explore=lambda change, target, run: explore_change(change, target, tot, run),
                label=lambda result, change, target, run: label_result(
                    result, change, target, labeling, run
                ),
                summarize=lambda findings, target: write_summary(findings, target, summary),
            )
            workflow = Workflow(
                stages=stages,
                tree=tree,
                ledger=ledger,
                runs_dir=settings.runs_dir,
                metrics=catalog,
                confirm=confirm,
            )
            graph = build_graph(workflow)
            try:
                state = await graph.ainvoke(
                    initial_state(run_id, comparison, max_changes),
                    config={"recursion_limit": recursion_limit(max_changes)},
                )
            except Exception as error:  # noqa: BLE001 -- re-raised unchanged just below
                # The MCP client's task group wraps anything raised inside it in an
                # ExceptionGroup, so the CLI could not catch `RunDeclined`, `BriefError` or a
                # budget stop. Held here and raised once the tree store has closed cleanly.
                failure = error
            else:
                failure = None
        if failure is not None:
            raise failure
    finally:
        index.close()
        embedder.close()
        store.close()
        if cache is not None:
            cache.close()
    return RunReport(
        run_id=run_id,
        brief=Path(state["brief_path"]),
        audit=Path(state["audit_path"]),
        cost_usd=ledger.spent_usd,
        estimated_usd=state["estimate"].total_usd,
        selected=len(state["findings"]),
        searched=sum(1 for item in state["findings"] if item.finding.tree_root_id is not None),
        not_examined=len(state["not_examined"].changes),
        node_evaluations=state["node_evaluations"],
    )
