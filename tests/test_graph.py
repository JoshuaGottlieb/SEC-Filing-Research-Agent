"""The workflow graph (T5.5): order, the cap, budget stops, review filing, and the written brief."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from sec_agent.agents.verification import find_causal_language
from sec_agent.compare.materiality import (
    MaterialityDecision,
    MaterialityHeadroom,
    MaterialityResult,
    Unresolved,
)
from sec_agent.finance.scan import MetricScan
from sec_agent.graph import (
    LABELING_BUDGET_REASON,
    LABELING_RESERVE_USD,
    MATERIALITY_BATCH_USD,
    MATERIALITY_UNRESOLVED_REASON,
    PER_CHANGE_USD,
    RESERVE_REASON,
    REWORK_BUDGET_NOTE,
    REWORK_RESERVE_NOTE,
    SUMMARY_BUDGET_NOTE,
    SUMMARY_USD,
    UNREACHED_REASON,
    CostEstimate,
    ResumeError,
    RunDeclined,
    Stages,
    Workflow,
    attempt_id,
    build_graph,
    check_resumable,
    checkpoint_serializer,
    describe_estimate,
    estimate_cost,
    initial_state,
    new_run_id,
    recursion_limit,
    run_graph,
    select_changes,
)
from sec_agent.llm.ledger import BudgetExceededError, CostLedger, LedgerEntry
from sec_agent.report.brief import BriefError
from sec_agent.report.summary import SummaryResult, template_summary
from sec_agent.schemas import (
    Change,
    ChangeKind,
    Claim,
    Comparison,
    ComparisonType,
    Duration,
    ExplanationLabel,
    Filing,
    Finding,
    FiscalPeriod,
    FormType,
    MaterialityCriterion,
    MetricName,
    MetricValue,
    RankedChanges,
    Section,
    TreeNode,
)
from sec_agent.tot.controller import Outcome, StopReason, ToTResult
from sec_agent.tot.labeling import LabeledFinding, ReworkItem
from sec_agent.tot.rework import MAX_REWORK_ROUNDS, ReworkResult
from sec_agent.tree_mcp.client import InProcessTreeClient
from sec_agent.tree_mcp.store import TreeStore

CURRENT = FiscalPeriod(fiscal_year=2025, fiscal_period="FY", period_end=date(2025, 12, 31))
PRIOR = FiscalPeriod(fiscal_year=2024, fiscal_period="FY", period_end=date(2024, 12, 31))
RUN = "HSY-yoy-10k-test"


def filing(period: FiscalPeriod, accession: str) -> Filing:
    return Filing(
        ticker="HSY",
        company="The Hershey Company",
        cik="0000047111",
        form=FormType.FORM_10K,
        period=period,
        accession=accession,
        filed_date=date(period.fiscal_year + 1, 2, 15),
        url=f"https://www.sec.gov/{accession}.htm",
    )


COMPARISON = Comparison(
    comparison_id="HSY:yoy-10k",
    ticker="HSY",
    comparison_type=ComparisonType.YOY_10K,
    current=filing(CURRENT, "acc-2025"),
    prior=filing(PRIOR, "acc-2024"),
)


def numeric(name: str) -> Change:
    return Change(
        change_id=f"HSY:yoy-10k:numeric:{name}",
        comparison_id=COMPARISON.comparison_id,
        kind=ChangeKind.NUMERIC,
        metric_id=f"HSY:{name}:FY 2025:12m",
    )


def text(n: int) -> Change:
    return Change(
        change_id=f"HSY:yoy-10k:added:acc-2025:mdna:{n:04d}",
        comparison_id=COMPARISON.comparison_id,
        kind=ChangeKind.ADDED,
        section=Section.MDNA,
        current_text=f"New paragraph {n}.",
    )


def ranked(changes: list[Change]) -> RankedChanges:
    return RankedChanges(
        comparison_id=COMPARISON.comparison_id,
        changes=tuple(
            change.model_copy(update={"materiality_rank": rank})
            for rank, change in enumerate(changes, start=1)
        ),
    )


def decision(change: Change, importance: int, untraced: tuple[str, ...] = ()):
    return MaterialityDecision(
        change_id=change.change_id,
        material=importance >= 3 or change.kind is ChangeKind.NUMERIC,
        importance=importance,
        criterion=MaterialityCriterion.CHANGED_QUANTITY if importance else None,
        rationale=f"Rated {importance}.",
        untraced_figures=untraced,
    )


class Killed(Exception):  # noqa: N818 -- a stand-in for the process being killed
    """Raised mid-search to stand in for a killed process."""


class FakePipeline:
    """Every stage, scripted, with a log of what each was called with."""

    def __init__(  # noqa: PLR0913 -- one knob per scenario
        self,
        tree: InProcessTreeClient,
        *,
        numeric_changes: list[Change],
        text_changes: list[Change],
        order: list[Change],
        unresolved: tuple[Unresolved, ...] = (),
        untraced: dict[str, tuple[str, ...]] | None = None,
        run_budget_on: str | None = None,
        kill_on: str | None = None,
        label_budget_on: str | None = None,
        summary_budget: bool = False,
        claim_text: str = "Gross margin moved.",
        headroom_usd: float = 0.0,
        ledger: CostLedger | None = None,
        explore_cost: float = 0.0,
        rework_on: dict[str, int | None] | None = None,
        rework_budget_on: str | None = None,
        rework_stop_on: str | None = None,
        rework_kill_on: str | None = None,
    ) -> None:
        self.tree = tree
        self.numeric_changes = numeric_changes
        self.text_changes = text_changes
        self.order = order
        self.unresolved = unresolved
        self.untraced = untraced or {}
        self.run_budget_on = run_budget_on
        self.label_budget_on = label_budget_on
        self.summary_budget = summary_budget
        self.claim_text = claim_text
        # The change whose first search is cut off, as if the process were killed there.
        self.kill_on = kill_on
        # What the materiality step would hold at once, and how often the graph asked.
        self.headroom_usd = headroom_usd
        self.headroom_checks = 0
        # Each search records `explore_cost` on `ledger`, as real calls would.
        self.ledger = ledger
        self.explore_cost = explore_cost
        self.calls: list[tuple[str, str]] = []
        self.tree_run_ids: list[str] = []
        self.summarized: list[tuple[Finding, ...]] = []
        # Changes whose claim is listed for rework, and the round that clears it (None: never).
        self.rework_on = rework_on or {}
        self.rework_budget_on = rework_budget_on
        self.rework_stop_on = rework_stop_on
        self.rework_kill_on = rework_kill_on
        self.rework_rounds: dict[str, int] = {}
        self.prefixes: list[str] = []

    def stages(self) -> Stages:
        return Stages(
            detect=self.detect,
            headroom=self.headroom,
            rank=self.rank,
            explore=self.explore,
            label=self.label,
            summarize=self.summarize,
            rework=self.rework,
        )

    def detect(self, comparison):
        self.calls.append(("detect", comparison.comparison_id))
        scan = MetricScan(
            comparison_id=comparison.comparison_id, moves=(), changes=tuple(self.numeric_changes)
        )
        return self.text_changes, scan

    def headroom(self, comparison, text_changes, scan):
        self.headroom_checks += 1
        return MaterialityHeadroom(batches=3, uncached=2, headroom_usd=self.headroom_usd)

    async def rank(self, comparison, text_changes, scan):
        self.calls.append(("rank", f"{len(text_changes)}+{len(scan.changes)}"))
        everything = [*scan.changes, *text_changes]
        return MaterialityResult(
            comparison_id=comparison.comparison_id,
            ranked=ranked(self.order),
            decisions=tuple(
                decision(change, 5 - min(n, 2), self.untraced.get(change.change_id, ()))
                for n, change in enumerate(everything)
            ),
            unresolved=self.unresolved,
            calls=1,
            cost_usd=0.0,
        )

    async def explore(self, change, comparison, run_id):
        self.calls.append(("explore", change.change_id))
        self.tree_run_ids.append(run_id)
        root = TreeNode(
            node_id=f"{run_id}:{change.change_id}:root",
            run_id=run_id,
            change_id=change.change_id,
            depth=0,
            thought="what changed",
        )
        await self.tree.create_tree(root)
        if change.change_id == self.kill_on:
            self.kill_on = None
            raise Killed(change.change_id)
        if self.explore_cost:
            self.ledger.record(LedgerEntry("generator", "m", 1, 1, 0, 0, self.explore_cost, False))
        stopped = StopReason.RUN_BUDGET if change.change_id == self.run_budget_on else None
        return ToTResult(
            change_id=change.change_id,
            tree_root_id=root.node_id,
            outcome=Outcome.UNRESOLVED,
            stopped=stopped,
            review_reason="no explanation survived",
            node_evaluations=3,
        )

    async def label(self, result, change, comparison, run_id):
        self.calls.append(("label", change.change_id))
        if change.change_id == self.label_budget_on:
            raise BudgetExceededError("Budget of $1.00 reached")
        claim = Claim(
            claim_id=f"{run_id}:{change.change_id}:r1:c1",
            text=self.claim_text,
            label=ExplanationLabel.PLAUSIBLE,
            metric_ids=("gm-2025",),
        )
        finding = Finding(
            finding_id=f"{run_id}:{change.change_id}",
            change=change,
            claims=(claim,),
            tree_root_id=result.tree_root_id,
        )
        rework = (
            (ReworkItem(claim_id=claim.claim_id),) if change.change_id in self.rework_on else ()
        )
        return LabeledFinding(
            finding=finding, failures=("claim c1: used the fixed wording",), rework=rework
        )

    async def rework(self, labeled, claim_id, comparison, prefix):
        change_id = labeled.finding.change.change_id
        self.calls.append(("rework", change_id))
        self.prefixes.append(prefix)
        if change_id == self.rework_kill_on:
            self.rework_kill_on = None
            raise Killed(change_id)
        if change_id == self.rework_budget_on:
            raise BudgetExceededError("Budget of $1.00 reached")
        if self.explore_cost:
            self.ledger.record(LedgerEntry("judge", "m", 1, 1, 0, 0, self.explore_cost, False))
        rounds = self.rework_rounds[change_id] = self.rework_rounds.get(change_id, 0) + 1
        if self.rework_on[change_id] == rounds:
            (claim,) = labeled.finding.claims
            reworked = claim.model_copy(update={"text": "Gross margin moved, on more evidence."})
            finding = labeled.finding.model_copy(update={"claims": (reworked,)})
            labeled = labeled.model_copy(update={"finding": finding, "rework": ()})
        stopped = StopReason.RUN_BUDGET if change_id == self.rework_stop_on else None
        return ReworkResult(labeled=labeled, node_evaluations=2, stopped=stopped)

    async def summarize(self, findings, comparison):
        self.calls.append(("summarize", str(len(findings))))
        self.summarized.append(tuple(findings))
        if self.summary_budget:
            raise BudgetExceededError("Budget of $1.00 reached")
        return SummaryResult(text=template_summary(findings), used_template=False, writer_calls=1)


@pytest.fixture
def tree(tmp_path) -> InProcessTreeClient:
    return InProcessTreeClient(TreeStore(tmp_path / "tree.db"))


GM, OM = numeric("gross_margin"), numeric("operating_margin")
TEXTS = [text(n) for n in range(1, 5)]


def pipeline(tree, **changes) -> FakePipeline:
    defaults = {"numeric_changes": [GM, OM], "text_changes": TEXTS, "order": [GM, OM, *TEXTS]}
    return FakePipeline(tree, **{**defaults, **changes})


class MarginLookup:
    """A `MetricLookup` holding only the gross margin change's current value."""

    def get_metric(self, metric_id: str) -> MetricValue | None:
        if metric_id != GM.metric_id:
            return None
        return MetricValue(
            metric_id=GM.metric_id,
            ticker="HSY",
            name=MetricName.GROSS_MARGIN,
            period=CURRENT,
            duration=Duration.TWELVE_MONTHS,
            value=0.335,
            unit="ratio",
            input_fact_ids=("f1",),
        )


async def run(  # noqa: PLR0913 -- the test's knobs, all optional
    tree,
    fake: FakePipeline,
    tmp_path,
    max_changes: int = 3,
    ledger=None,
    metrics=None,
    confirm=None,
):
    workflow = Workflow(
        stages=fake.stages(),
        tree=tree,
        ledger=ledger or fake.ledger or CostLedger(5.0),
        runs_dir=tmp_path / "runs",
        metrics=metrics,
        confirm=confirm,
        clock=lambda: datetime(2026, 11, 12, 12, 0, tzinfo=UTC),
    )
    graph = build_graph(workflow)
    return await graph.ainvoke(
        initial_state(RUN, COMPARISON, max_changes),
        config={"recursion_limit": recursion_limit(max_changes)},
    )


def audit(state) -> dict:
    with open(state["audit_path"], encoding="utf-8") as handle:
        return json.load(handle)


def explored(fake: FakePipeline) -> list[str]:
    return [change_id for step, change_id in fake.calls if step == "explore"]


async def test_the_run_goes_through_every_step_in_order(tree, tmp_path):
    fake = pipeline(tree)
    state = await run(tree, fake, tmp_path, max_changes=3)
    steps = [step for step, _ in fake.calls]
    assert steps == ["detect", "rank", *["explore", "label"] * 3, "summarize"]
    # Search and labeling alternate, most material first.
    assert explored(fake) == [GM.change_id, OM.change_id, TEXTS[0].change_id]
    assert fake.calls[1] == ("rank", "4+2")
    assert state["brief_path"].endswith(f"runs/{RUN}/brief.md")


async def test_only_the_first_max_changes_are_explored_and_the_rest_are_listed(tree, tmp_path):
    fake = pipeline(tree)
    state = await run(tree, fake, tmp_path, max_changes=2)
    assert explored(fake) == [GM.change_id, OM.change_id]
    trail = audit(state)
    skipped = trail["not_examined"]
    assert skipped["max_changes"] == 2
    assert [item["change_id"] for item in skipped["changes"]] == [c.change_id for c in TEXTS]
    assert [item["materiality_rank"] for item in skipped["changes"]] == [3, 4, 5, 6]
    # Importance comes from the materiality decisions (the fake rates 5, 4, then 3).
    assert [item["importance"] for item in skipped["changes"]] == [3, 3, 3, 3]
    # They are not findings, so they are not under review.
    finding_ids = [finding["finding_id"] for finding in trail["brief"]["findings"]]
    assert finding_ids == [f"{RUN}:{GM.change_id}", f"{RUN}:{OM.change_id}"]
    markdown = Path(state["brief_path"]).read_text(encoding="utf-8")
    assert "4 further material changes were not examined" in " ".join(markdown.split())


async def test_a_cap_larger_than_the_list_explores_everything(tree, tmp_path):
    fake = pipeline(tree)
    state = await run(tree, fake, tmp_path, max_changes=30)
    assert len(explored(fake)) == 6
    assert audit(state)["not_examined"]["changes"] == []
    assert "Not examined" not in Path(state["brief_path"]).read_text(encoding="utf-8")


async def test_with_no_material_changes_nothing_is_searched_and_the_brief_says_so(tree, tmp_path):
    fake = pipeline(tree, order=[])
    state = await run(tree, fake, tmp_path)
    assert explored(fake) == []
    trail = audit(state)
    assert trail["brief"]["findings"] == []
    assert trail["brief"]["summary"] == template_summary(())


async def test_the_brief_holds_each_findings_tree_labeling_notes_and_the_cost(tree, tmp_path):
    fake = pipeline(tree)
    ledger = CostLedger(5.0)
    state = await run(tree, fake, tmp_path, max_changes=2, ledger=ledger)
    trail = audit(state)
    gm = f"{RUN}:{GM.change_id}"
    assert trail["trees"][gm][0]["node_id"] == f"{gm}:root"
    assert trail["labeling_failures"][gm] == ["claim c1: used the fixed wording"]
    assert trail["brief"]["cost_usd"] == ledger.spent_usd
    assert trail["brief"]["run_id"] == RUN


async def test_the_node_count_and_budget_snapshot_carry_across_changes(tree, tmp_path):
    fake = pipeline(tree)
    state = await run(tree, fake, tmp_path, max_changes=3)
    assert state["node_evaluations"] == 9
    budget = await tree.read_budget(RUN)
    assert budget.node_evaluations == 9
    assert budget.max_usd == 5.0


async def test_a_run_budget_stop_in_the_search_files_the_rest_unexamined(tree, tmp_path):
    fake = pipeline(tree, run_budget_on=OM.change_id)
    state = await run(tree, fake, tmp_path, max_changes=4)
    # The stopped search is still labeled; nothing after it is searched.
    assert explored(fake) == [GM.change_id, OM.change_id]
    assert ("label", OM.change_id) in fake.calls
    findings = audit(state)["brief"]["findings"]
    assert len(findings) == 4
    assert [finding["review_reason"] for finding in findings[2:]] == [UNREACHED_REASON] * 2
    assert all(finding["claims"] == [] for finding in findings[2:])


async def test_a_budget_stop_while_labeling_keeps_the_tree_and_stops_the_search(tree, tmp_path):
    fake = pipeline(tree, label_budget_on=GM.change_id)
    state = await run(tree, fake, tmp_path, max_changes=3)
    assert explored(fake) == [GM.change_id]
    findings = audit(state)["brief"]["findings"]
    assert findings[0]["review_reason"] == LABELING_BUDGET_REASON
    assert findings[0]["tree_root_id"] == f"{RUN}:{GM.change_id}:root"
    assert findings[0]["claims"] == []
    assert [finding["review_reason"] for finding in findings[1:]] == [UNREACHED_REASON] * 2


async def test_a_budget_stop_while_summarizing_uses_the_template(tree, tmp_path):
    # The margin's labeling stops too, so its finding has no claim and the template names it.
    fake = pipeline(tree, summary_budget=True, label_budget_on=GM.change_id)
    state = await run(tree, fake, tmp_path, max_changes=2, metrics=MarginLookup())
    assert state["summary"].used_template
    assert state["summary"].failures == (SUMMARY_BUDGET_NOTE,)
    trail = audit(state)
    assert trail["brief"]["summary"] == template_summary(fake.summarized[0], MarginLookup())
    # The template names the metric, not its id.
    assert trail["brief"]["summary"].startswith("Gross margin change: 33.5% in FY2025")
    # A template summary reads like a written one, so only the audit trail says what happened.
    assert trail["summary"] == {"used_template": True, "failures": [SUMMARY_BUDGET_NOTE]}


async def test_the_audit_says_how_a_written_summary_was_made(tree, tmp_path):
    state = await run(tree, pipeline(tree), tmp_path)
    assert audit(state)["summary"] == {"used_template": False, "failures": []}


async def test_changes_materiality_could_not_decide_go_to_review(tree, tmp_path):
    stuck = Unresolved(change_id=TEXTS[3].change_id, reason="refused: due to policy")
    fake = pipeline(tree, order=[GM, OM, *TEXTS[:3]], unresolved=(stuck,))
    state = await run(tree, fake, tmp_path, max_changes=2)
    trail = audit(state)
    last = trail["brief"]["findings"][-1]
    assert last["finding_id"] == f"{RUN}:{TEXTS[3].change_id}"
    assert last["review_reason"] == MATERIALITY_UNRESOLVED_REASON
    # The raw error can say anything, so it goes to the audit trail, never to the brief's text.
    assert trail["labeling_failures"][last["finding_id"]] == ["materiality: refused: due to policy"]
    assert TEXTS[3].change_id not in explored(fake)
    # It reached the summary too.
    assert len(fake.summarized[0]) == 3


async def test_rationales_dropped_for_an_unbacked_figure_are_in_the_audit(tree, tmp_path):
    fake = pipeline(tree, untraced={TEXTS[0].change_id: ("12.4%",)})
    state = await run(tree, fake, tmp_path, max_changes=2)
    untraced = audit(state)["rationales_untraced"]
    assert [item["change_id"] for item in untraced] == [TEXTS[0].change_id]
    assert untraced[0]["untraced_figures"] == ["12.4%"]


async def test_a_brief_that_breaks_a_rule_is_not_written(tree, tmp_path):
    # A plausible claim worded causally is a bug upstream; write_run refuses it.
    fake = pipeline(tree, claim_text="Gross margin fell due to cocoa costs.")
    with pytest.raises(BriefError):
        await run(tree, fake, tmp_path, max_changes=1)
    assert not (tmp_path / "runs" / RUN).exists()


async def test_a_cap_too_small_for_materiality_ends_the_run_before_anything_is_sent(tree, tmp_path):
    fake = pipeline(tree, headroom_usd=0.60)
    with pytest.raises(BudgetExceededError, match="too small for the materiality step"):
        await run(tree, fake, tmp_path, ledger=CostLedger(0.50))
    assert fake.headroom_checks == 1
    assert [step for step, _ in fake.calls] == ["detect"]  # never ranked
    assert not (tmp_path / "runs" / RUN).exists()


async def test_a_cap_that_covers_materiality_goes_on(tree, tmp_path):
    fake = pipeline(tree, headroom_usd=0.60)
    state = await run(tree, fake, tmp_path, ledger=CostLedger(0.60))
    assert fake.headroom_checks == 1
    assert ("rank", "4+2") in fake.calls
    assert Path(state["brief_path"]).exists()


async def test_a_change_without_the_labeling_reserve_left_is_not_searched(tree, tmp_path):
    # $1.00 cap, $0.40 per search, $0.25 reserve: the first two searches leave $0.60 then $0.20,
    # so the third change and every one after it are filed unexamined.
    ledger = CostLedger(1.0)
    fake = pipeline(tree, ledger=ledger, explore_cost=0.40)
    state = await run(tree, fake, tmp_path, max_changes=4)
    assert explored(fake) == [GM.change_id, OM.change_id]
    findings = audit(state)["brief"]["findings"]
    assert [finding["review_reason"] for finding in findings[2:]] == [RESERVE_REASON] * 2
    assert all(finding["claims"] == [] for finding in findings[2:])
    # The searched changes were labeled, and the run still wrote its summary and brief.
    assert [change for step, change in fake.calls if step == "label"] == explored(fake)
    assert ("summarize", "4") in fake.calls
    assert ledger.spent_usd == pytest.approx(0.80)


async def test_the_reserve_is_provisional_and_set_on_the_workflow(tree, tmp_path):
    assert LABELING_RESERVE_USD == 0.25
    workflow = Workflow(
        stages=pipeline(tree).stages(), tree=tree, ledger=CostLedger(1.0), runs_dir=tmp_path
    )
    assert workflow.labeling_reserve_usd == LABELING_RESERVE_USD
    assert workflow.free_usd() == 1.0


def test_the_reserve_reason_is_not_worded_causally():
    assert find_causal_language(RESERVE_REASON) == ()


class Answer:
    """A `Workflow.confirm` that records each estimate it is shown and gives a fixed answer."""

    def __init__(self, go_ahead: bool) -> None:
        self.go_ahead = go_ahead
        self.shown: list[CostEstimate] = []

    def __call__(self, estimate: CostEstimate) -> bool:
        self.shown.append(estimate)
        return self.go_ahead


def test_the_estimate_adds_uncached_batches_every_change_and_the_summary():
    headroom = MaterialityHeadroom(batches=11, uncached=2, headroom_usd=0.46)
    estimate = estimate_cost(headroom, max_changes=3, cap_usd=1.0)
    assert estimate.materiality_usd == pytest.approx(2 * MATERIALITY_BATCH_USD)
    assert estimate.changes_usd == pytest.approx(3 * PER_CHANGE_USD)
    assert estimate.summary_usd == SUMMARY_USD
    assert estimate.total_usd == pytest.approx(
        2 * MATERIALITY_BATCH_USD + 3 * PER_CHANGE_USD + SUMMARY_USD
    )
    assert (estimate.materiality_batches, estimate.cap_usd) == (11, 1.0)


def test_the_printed_estimate_warns_only_when_it_is_above_the_hard_stop():
    headroom = MaterialityHeadroom(batches=11, uncached=0, headroom_usd=0.0)
    under = describe_estimate(estimate_cost(headroom, max_changes=3, cap_usd=1.0))
    assert under[0] == "Estimated model cost: about $0.23, with a hard stop at $1.00."
    assert "0 of 11 batches need a call" in under[1]
    assert not any("above the hard stop" in line for line in under)
    over = describe_estimate(estimate_cost(headroom, max_changes=20, cap_usd=1.0))
    assert "above the hard stop" in over[-1]


async def test_a_declined_estimate_ends_the_run_before_anything_is_spent(tree, tmp_path):
    fake = pipeline(tree, headroom_usd=0.30)
    answer = Answer(go_ahead=False)
    with pytest.raises(RunDeclined):
        await run(tree, fake, tmp_path, max_changes=2, confirm=answer)
    assert len(answer.shown) == 1
    assert answer.shown[0].max_changes == 2
    assert answer.shown[0].materiality_uncached == 2
    assert [step for step, _ in fake.calls] == ["detect"]  # never ranked
    assert not (tmp_path / "runs" / RUN).exists()


async def test_an_accepted_estimate_runs_and_is_kept_in_the_state(tree, tmp_path):
    fake = pipeline(tree)
    answer = Answer(go_ahead=True)
    state = await run(tree, fake, tmp_path, confirm=answer)
    assert answer.shown == [state["estimate"]]
    assert ("rank", "4+2") in fake.calls
    assert Path(state["brief_path"]).exists()


async def test_a_cap_too_small_for_materiality_is_refused_before_the_estimate_is_shown(
    tree, tmp_path
):
    fake = pipeline(tree, headroom_usd=0.60)
    answer = Answer(go_ahead=True)
    with pytest.raises(BudgetExceededError):
        await run(tree, fake, tmp_path, ledger=CostLedger(0.50), confirm=answer)
    assert answer.shown == []


# Resume (T5.5 decision 3)


def workflow_for(fake: FakePipeline, tree, tmp_path, *, tree_run_id: str | None = None) -> Workflow:
    return Workflow(
        stages=fake.stages(),
        tree=tree,
        ledger=CostLedger(5.0),
        runs_dir=tmp_path / "runs",
        clock=lambda: datetime(2026, 11, 13, 12, 0, tzinfo=UTC),
        tree_run_id=tree_run_id,
    )


async def test_a_killed_run_resumes_from_its_last_finished_change(tree, tmp_path, settings):
    fake = pipeline(tree, kill_on=OM.change_id)
    with pytest.raises(Killed):
        await run_graph(workflow_for(fake, tree, tmp_path), settings, RUN, COMPARISON, 3, False)
    assert not (tmp_path / "runs" / RUN).exists()
    # A new process: a new workflow and a new connection to the same checkpoint file.
    attempt = f"{RUN}.r2"
    resumed = workflow_for(fake, tree, tmp_path, tree_run_id=attempt)
    state = await run_graph(resumed, settings, RUN, COMPARISON, 3, True)
    steps = [step for step, _ in fake.calls]
    # Detection and ranking are not repeated; the finished change is not searched again.
    assert steps.count("detect") == 1 and steps.count("rank") == 1
    assert explored(fake) == [GM.change_id, OM.change_id, OM.change_id, TEXTS[0].change_id]
    assert [change for step, change in fake.calls if step == "label"] == [
        GM.change_id,
        OM.change_id,
        TEXTS[0].change_id,
    ]
    # The cut-off change runs again under the attempt id; its finding keeps the run's id.
    assert fake.tree_run_ids == [RUN, RUN, attempt, attempt]
    findings = audit(state)["brief"]["findings"]
    assert [finding["finding_id"] for finding in findings] == [
        f"{RUN}:{GM.change_id}",
        f"{RUN}:{OM.change_id}",
        f"{RUN}:{TEXTS[0].change_id}",
    ]
    assert findings[1]["tree_root_id"] == f"{attempt}:{OM.change_id}:root"
    # The cut-off attempt's node count was never saved, so only finished searches count.
    assert state["node_evaluations"] == 9
    assert Path(state["brief_path"]).exists()


async def test_a_run_id_with_a_checkpoint_is_not_started_again(tree, tmp_path, settings):
    fake = pipeline(tree)
    await run_graph(workflow_for(fake, tree, tmp_path), settings, RUN, COMPARISON, 3, False)
    with pytest.raises(ResumeError, match="resume it with --resume"):
        await run_graph(workflow_for(fake, tree, tmp_path), settings, RUN, COMPARISON, 3, False)
    with pytest.raises(ResumeError, match="already finished"):
        await run_graph(workflow_for(fake, tree, tmp_path), settings, RUN, COMPARISON, 3, True)


async def test_only_a_saved_run_can_be_resumed(tree, tmp_path, settings):
    fake = pipeline(tree)
    with pytest.raises(ResumeError, match="no checkpoint"):
        await run_graph(
            workflow_for(fake, tree, tmp_path), settings, "never-ran", COMPARISON, 3, True
        )
    assert fake.calls == []


def test_a_resume_must_be_of_the_same_comparison():
    other = COMPARISON.model_copy(update={"comparison_id": "HSY:yoy-10q"})
    saved = SimpleNamespace(values={"comparison": COMPARISON}, next=("explore",))
    check_resumable(saved, RUN, COMPARISON)
    with pytest.raises(ResumeError, match="analyzes"):
        check_resumable(saved, RUN, other)


def test_an_attempt_id_has_no_colon_and_names_its_run():
    made = attempt_id(RUN, datetime(2026, 11, 13, 18, 5, 9, tzinfo=UTC))
    assert made == f"{RUN}.r20261113T180509Z"
    assert ":" not in made


def test_the_checkpoint_restores_records_as_records(monkeypatch):
    # LangGraph turns a class it is not allowed to restore into plain data, which the graph's
    # nodes could not read on a resume. Its default still restores unlisted classes, with a
    # warning that a future version will block them; strict mode is that future default.
    from langgraph.checkpoint.serde import _msgpack

    monkeypatch.setattr(_msgpack, "STRICT_MSGPACK_ENABLED", True)
    serde = checkpoint_serializer()
    finding = LabeledFinding(
        finding=Finding(finding_id=f"{RUN}:x", change=GM, review_reason="r"),
        failures=("note",),
    )
    state = {"findings": (finding,), "comparison": COMPARISON, "estimate": None}
    restored = serde.loads_typed(serde.dumps_typed(state))
    assert restored["findings"][0] == finding
    assert isinstance(restored["findings"][0].finding.change.kind, ChangeKind)
    assert restored["comparison"] == COMPARISON


def test_selecting_needs_a_positive_cap():
    materiality = MaterialityResult(
        comparison_id=COMPARISON.comparison_id,
        ranked=ranked([GM]),
        decisions=(),
        unresolved=(),
        calls=0,
        cost_usd=0.0,
    )
    with pytest.raises(ValueError, match="at least 1"):
        select_changes(materiality.ranked, materiality, 0)
    chosen, rest = select_changes(materiality.ranked, materiality, 1)
    assert [change.change_id for change in chosen] == [GM.change_id]
    assert rest.changes == ()
    with pytest.raises(ValueError, match="at least 1"):
        initial_state(RUN, COMPARISON, 0)


def test_a_run_id_is_fresh_and_has_no_colons():
    run_id = new_run_id(COMPARISON, datetime(2026, 11, 12, 20, 15, 3, tzinfo=UTC))
    assert run_id == "HSY-yoy-10k-20261112T201503Z"
    assert ":" not in run_id


# -- the rework loop's retry edge (T5.5 decision 4) -------------------------------------------


def reworks(fake: FakePipeline) -> list[str]:
    return [change_id for step, change_id in fake.calls if step == "rework"]


async def test_a_listed_claim_gets_at_most_two_rounds_then_the_run_goes_on(tree, tmp_path):
    # Every change is reworked twice, which also checks the graph's step limit.
    fake = pipeline(
        tree, rework_on={GM.change_id: None, OM.change_id: None, TEXTS[0].change_id: None}
    )
    state = await run(tree, fake, tmp_path, max_changes=3)
    assert MAX_REWORK_ROUNDS == 2
    steps = [(step, change) for step, change in fake.calls if step in {"explore", "rework"}]
    assert steps == [
        ("explore", GM.change_id),
        ("rework", GM.change_id),
        ("rework", GM.change_id),
        ("explore", OM.change_id),
        ("rework", OM.change_id),
        ("rework", OM.change_id),
        ("explore", TEXTS[0].change_id),
        ("rework", TEXTS[0].change_id),
        ("rework", TEXTS[0].change_id),
    ]
    claim = f"{RUN}:{GM.change_id}:r1:c1"
    assert fake.prefixes[:2] == [f"{claim}:w1", f"{claim}:w2"]
    # Still listed after the last round: the claim keeps its label and the brief is written.
    assert Path(state["brief_path"]).exists()


async def test_a_claim_cleared_in_the_first_round_gets_no_second(tree, tmp_path):
    fake = pipeline(tree, rework_on={GM.change_id: 1})
    state = await run(tree, fake, tmp_path, max_changes=2)
    assert reworks(fake) == [GM.change_id]
    findings = audit(state)["brief"]["findings"]
    assert findings[0]["claims"][0]["text"] == "Gross margin moved, on more evidence."
    assert explored(fake) == [GM.change_id, OM.change_id]


async def test_a_finding_with_nothing_listed_is_not_reworked(tree, tmp_path):
    fake = pipeline(tree, rework_on={OM.change_id: 1})
    await run(tree, fake, tmp_path, max_changes=2)
    assert reworks(fake) == [OM.change_id]


async def test_with_no_rework_stage_the_loop_is_off(tree, tmp_path):
    fake = pipeline(tree, rework_on={GM.change_id: None})
    workflow = Workflow(
        stages=Stages(**{**vars(fake.stages()), "rework": None}),
        tree=tree,
        ledger=CostLedger(5.0),
        runs_dir=tmp_path / "runs",
    )
    await build_graph(workflow).ainvoke(
        initial_state(RUN, COMPARISON, 2), config={"recursion_limit": recursion_limit(2)}
    )
    assert reworks(fake) == []


async def test_rework_node_counts_join_the_run_total_and_the_budget_snapshot(tree, tmp_path):
    fake = pipeline(tree, rework_on={GM.change_id: None})
    state = await run(tree, fake, tmp_path, max_changes=2)
    # Two searches of 3 and two rounds of 2.
    assert state["node_evaluations"] == 10
    assert (await tree.read_budget(RUN)).node_evaluations == 10


async def test_a_round_is_skipped_without_the_labeling_reserve(tree, tmp_path):
    # $0.50 cap, $0.30 per search: $0.20 is left after the first, under the $0.25 reserve.
    ledger = CostLedger(0.5)
    fake = pipeline(tree, ledger=ledger, explore_cost=0.30, rework_on={GM.change_id: None})
    state = await run(tree, fake, tmp_path, max_changes=2)
    assert reworks(fake) == []
    gm = f"{RUN}:{GM.change_id}"
    assert audit(state)["labeling_failures"][gm][-1] == f"claim {gm}:r1:c1: {REWORK_RESERVE_NOTE}"
    assert Path(state["brief_path"]).exists()


async def test_a_budget_stop_in_a_round_keeps_the_finding_and_stops_the_search(tree, tmp_path):
    fake = pipeline(tree, rework_on={GM.change_id: None}, rework_budget_on=GM.change_id)
    state = await run(tree, fake, tmp_path, max_changes=3)
    assert reworks(fake) == [GM.change_id]
    assert explored(fake) == [GM.change_id]
    trail = audit(state)
    first, *rest = trail["brief"]["findings"]
    gm = f"{RUN}:{GM.change_id}"
    assert first["claims"][0]["text"] == "Gross margin moved."
    assert first["review_reason"] is None
    assert trail["labeling_failures"][gm][-1] == f"claim {gm}:r1:c1: {REWORK_BUDGET_NOTE}"
    assert [finding["review_reason"] for finding in rest] == [UNREACHED_REASON] * 2


async def test_a_round_the_dollar_cap_stopped_ends_the_loop_and_the_search(tree, tmp_path):
    fake = pipeline(tree, rework_on={GM.change_id: None}, rework_stop_on=GM.change_id)
    state = await run(tree, fake, tmp_path, max_changes=3)
    assert reworks(fake) == [GM.change_id]
    assert explored(fake) == [GM.change_id]
    assert state["out_of_budget"]


def test_the_step_limit_allows_every_change_its_rounds():
    assert recursion_limit(3) == 3 * (1 + MAX_REWORK_ROUNDS) + 10


def test_the_rework_notes_are_not_worded_causally():
    assert not find_causal_language(REWORK_RESERVE_NOTE)
    assert not find_causal_language(REWORK_BUDGET_NOTE)


async def test_a_run_killed_in_a_round_resumes_it_under_new_node_ids(tree, tmp_path, settings):
    fake = pipeline(tree, rework_on={GM.change_id: None}, rework_kill_on=GM.change_id)
    with pytest.raises(Killed):
        await run_graph(workflow_for(fake, tree, tmp_path), settings, RUN, COMPARISON, 2, False)
    resumed = workflow_for(fake, tree, tmp_path, tree_run_id=f"{RUN}.r2")
    state = await run_graph(resumed, settings, RUN, COMPARISON, 2, True)
    claim = f"{RUN}:{GM.change_id}:r1:c1"
    # The search is not repeated: the checkpoint after it holds the labeled finding.
    assert explored(fake) == [GM.change_id, OM.change_id]
    assert fake.prefixes == [f"{claim}:w1", f"{claim}:w1.r2", f"{claim}:w2.r2"]
    assert state["node_evaluations"] == 3 + 2 + 2 + 3
