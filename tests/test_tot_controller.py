"""The ToT controller (T4.4): beam, pruning, ties, retry and stop paths, on fakes.

The generator, critic, retriever and metric checker are fakes with the real classes' signatures,
and the tree store is real (in process, on `tmp_path`). The scoring and evidence checks the
controller leans on are the real code, so a number asserted here is the number a run would get.
No test makes a network or paid call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pytest
from pydantic import ValidationError

from sec_agent.agents.verification import EntailmentJudgment
from sec_agent.llm.client import LLMError
from sec_agent.llm.ledger import BudgetExceededError, CostLedger
from sec_agent.retrieval.hybrid import IndexOutOfSyncError
from sec_agent.schemas import (
    Change,
    ChangeKind,
    Chunk,
    Comparison,
    ComparisonType,
    Direction,
    Duration,
    EntailmentVerdict,
    EvidenceSourceRef,
    EvidenceSupport,
    ExplanationLabel,
    Filing,
    FiscalPeriod,
    FormType,
    MetricCheckResult,
    MetricExpectation,
    MetricName,
    MetricOutcome,
    MetricValue,
    NodeStatus,
    NumericCheck,
    RankedChanges,
    RetrievalResult,
    RetrievedPassage,
    Section,
    TreeNode,
)
from sec_agent.tot.controller import (
    CandidateResult,
    Outcome,
    StopReason,
    ToTContext,
    ToTResult,
    changed_metric,
    claim_node,
    claim_of,
    explore_change,
    explore_changes,
    gather_evidence,
    select_beam,
    tried_searches,
)
from sec_agent.tot.critic import (
    EvidenceCritique,
    EvidenceReassessment,
    ProposedCitation,
    SpecificityCritique,
)
from sec_agent.tot.generator import (
    CandidateExplanation,
    CandidateGeneration,
    EvidenceGeneration,
    EvidenceSource,
)
from sec_agent.tot.labeling import LabelingContext, PlausibleWording, Restatement, label_result
from sec_agent.tot.rework import ReworkContext, rework_claim, rework_prefix
from sec_agent.tot.scoring import Contradiction, ScoringConfig, Specificity
from sec_agent.tree_mcp.client import InProcessTreeClient
from sec_agent.tree_mcp.store import TreeStore

CURRENT = FiscalPeriod(fiscal_year=2025, fiscal_period="FY", period_end=date(2025, 12, 31))
PRIOR = FiscalPeriod(fiscal_year=2024, fiscal_period="FY", period_end=date(2024, 12, 31))


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
    comparison_id="HSY-yoy",
    ticker="HSY",
    comparison_type=ComparisonType.YOY_10K,
    current=filing(CURRENT, "acc-2025"),
    prior=filing(PRIOR, "acc-2024"),
)
CHANGE = Change(
    change_id="chg-1",
    comparison_id="HSY-yoy",
    kind=ChangeKind.MODIFIED,
    section=Section.MDNA,
    current_text="Gross margin decreased.",
    prior_text="Gross margin increased.",
    rationale="Gross margin fell 200 basis points",
)
GROSS_MARGIN_DOWN = MetricExpectation(metric=MetricName.GROSS_MARGIN, direction=Direction.DOWN)
REVENUE_DOWN = MetricExpectation(metric=MetricName.REVENUE, direction=Direction.DOWN)
RUN = "run-1"


# -- fakes ------------------------------------------------------------------------------------


def passage(
    chunk_id: str,
    *,
    section: Section = Section.MDNA,
    period: FiscalPeriod = CURRENT,
    score: float = 4.0,
) -> RetrievedPassage:
    chunk = Chunk(
        chunk_id=chunk_id,
        paragraph_id="p1",
        ticker="HSY",
        cik="0000047111",
        form=FormType.FORM_10K,
        period=period,
        section=section,
        accession="acc-2025" if period == CURRENT else "acc-2024",
        text=f"Gross margin decreased due to higher cocoa costs ({chunk_id}).",
        token_count=10,
    )
    return RetrievedPassage(chunk=chunk, score=score)


class FakeRetriever:
    """Returns what was scripted for a (query, section), and nothing otherwise.

    A named section never returns a chunk filed as `other`: only a search of everywhere can, as in
    the real retriever. The script is what decides that here, so a test that wants an `other`
    chunk scripts it under `section=None` only.
    """

    def __init__(self, script: dict[tuple[str, Section | None], RetrievalResult] | None = None):
        self.script = script or {}
        self.calls: list[tuple[str, Section | None]] = []

    async def retrieve(
        self, query: str, comparison: Comparison, section: Section | None = None
    ) -> RetrievalResult:
        self.calls.append((query, section))
        return self.script.get((query, section), RetrievalResult(query=query))


def found(query: str, *current: RetrievedPassage, prior: Sequence[RetrievedPassage] = ()):
    return RetrievalResult(query=query, current=tuple(current), prior=tuple(prior))


class FakeChecker:
    """A `MetricChecker` with one verdict per metric. A metric with none is not applicable."""

    def __init__(self, verdicts: dict[MetricName, NumericCheck] | None = None) -> None:
        self.verdicts = verdicts or {}
        self.calls: list[tuple[MetricExpectation, ...]] = []

    def check(
        self, comparison: Comparison, expectations: Sequence[MetricExpectation]
    ) -> MetricCheckResult:
        self.calls.append(tuple(expectations))
        outcomes = []
        for item in expectations:
            verdict = self.verdicts.get(item.metric, NumericCheck.NOT_APPLICABLE)
            cited = () if verdict is NumericCheck.NOT_APPLICABLE else ("m-now", "m-before")
            outcomes.append(MetricOutcome(metric=item.metric, check=verdict, metric_ids=cited))
        return MetricCheckResult(outcomes=tuple(outcomes))


class NoMetrics:
    """A `MetricMoveLookup`. The fake generator never reads it."""

    def get_metric(self, metric_id: str):
        return None

    def get_move(self, comparison, metric_id: str):
        return None


@dataclass(frozen=True)
class Verdict:
    """What the fake critic says about the passages behind one chunk."""

    evidence: EvidenceSupport = EvidenceSupport.CONSISTENT
    contradiction: Contradiction = Contradiction.NONE
    specificity: Specificity = Specificity.SPECIFIC
    # Overrides the one quote the critic would give: the whole text of the chunk.
    citations: tuple[ProposedCitation, ...] | None = None


class FakeGenerator:
    """Plays back one generation per round, and the sources scripted for each claim."""

    def __init__(
        self,
        rounds: Sequence[CandidateGeneration],
        sources: dict[str, EvidenceGeneration] | None = None,
        rework: dict[str, EvidenceGeneration] | None = None,
    ) -> None:
        self.rounds = list(rounds)
        self.sources = sources or {}
        self.rework = rework or {}
        self.candidate_calls: list[dict] = []
        self.source_calls: list[str] = []
        self.rework_calls: list[tuple[str, tuple]] = []

    async def propose_candidates(
        self, change, comparison, metrics, *, count: int = 6, already_tried: Sequence[str] = ()
    ) -> CandidateGeneration:
        self.candidate_calls.append({"count": count, "already_tried": tuple(already_tried)})
        return self.rounds.pop(0) if self.rounds else CandidateGeneration()

    async def propose_sources(
        self, node: TreeNode, change, comparison, metrics, *, count: int = 6
    ) -> EvidenceGeneration:
        self.source_calls.append(node.thought)
        return self.sources.get(node.thought, EvidenceGeneration())

    async def propose_rework_sources(
        self, node: TreeNode, change, comparison, metrics, *, count: int = 6, already_tried=()
    ) -> EvidenceGeneration:
        self.rework_calls.append((node.thought, tuple(already_tried)))
        return self.rework.get(node.thought, EvidenceGeneration())


class FakeCritic:
    """A `Critic` that rates by what was scripted. It picks one passage to rely on, as a model
    would: the first from a period the claim is about, or else the first it was given."""

    def __init__(
        self,
        *,
        specificity: dict[str, Specificity] | None = None,
        by_chunk: dict[str, Verdict] | None = None,
        fail_on_chunks: dict[str, Exception] | None = None,
        raise_after: tuple[int, Exception] | None = None,
    ) -> None:
        self.specificity = specificity or {}
        self.by_chunk = by_chunk or {}
        self.fail_on_chunks = fail_on_chunks or {}
        self.raise_after = raise_after
        self.calls: list[tuple] = []

    def _count(self) -> None:
        if self.raise_after and len(self.calls) > self.raise_after[0]:
            raise self.raise_after[1]

    async def rate_specificity(self, node: TreeNode) -> SpecificityCritique:
        self.calls.append(("specificity", node.thought, node.expected_metrics))
        self._count()
        level = self.specificity.get(node.thought, Specificity.SPECIFIC)
        return SpecificityCritique(specificity=level)

    async def judge_evidence(self, node: TreeNode, passages, claim_periods) -> EvidenceCritique:
        ids = tuple(p.chunk.chunk_id for p in passages)
        self.calls.append(("evidence", node.thought, ids, frozenset(claim_periods)))
        self._count()
        chosen = next((p for p in passages if p.chunk.period in claim_periods), passages[0])
        chunk_id = chosen.chunk.chunk_id
        if chunk_id in self.fail_on_chunks:
            raise self.fail_on_chunks[chunk_id]
        verdict = self.by_chunk.get(chunk_id, Verdict())
        if verdict.evidence is EvidenceSupport.NONE:
            citations: tuple[ProposedCitation, ...] = ()
        else:
            citations = verdict.citations or (
                ProposedCitation(chunk_id=chunk_id, quote=chosen.chunk.text),
            )
        return EvidenceCritique(
            evidence=verdict.evidence,
            contradiction=verdict.contradiction,
            specificity=verdict.specificity,
            citations=citations,
        )

    async def reassess_evidence(self, node: TreeNode, citations) -> EvidenceReassessment:
        self.calls.append(("reassess", node.thought, len(citations)))
        return EvidenceReassessment(evidence=EvidenceSupport.STATES_CAUSE)


# -- scenario helpers -------------------------------------------------------------------------


def candidate(claim: str, metrics: tuple[MetricExpectation, ...] = ()) -> CandidateExplanation:
    return CandidateExplanation(
        claim=claim,
        expected_evidence=f"Filing says {claim}",
        query_plan=f"search for {claim}",
        expected_metrics=metrics,
    )


def search(query: str, section: Section | None = Section.MDNA) -> EvidenceSource:
    return EvidenceSource(
        description=f"look for {query}", query=query, section=section, metric=None
    )


def compute(metric: MetricExpectation = GROSS_MARGIN_DOWN) -> EvidenceSource:
    return EvidenceSource(description="compute it", query=None, section=None, metric=metric)


def candidates(*claims: str) -> CandidateGeneration:
    return CandidateGeneration(candidates=tuple(candidate(claim) for claim in claims))


def sources(*items: EvidenceSource) -> EvidenceGeneration:
    return EvidenceGeneration(sources=tuple(items))


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    return ScoringConfig.load()


def with_beam(config: ScoringConfig, **changes) -> ScoringConfig:
    return config.model_copy(update={"beam": config.beam.model_copy(update=changes)})


def with_budget(config: ScoringConfig, **changes) -> ScoringConfig:
    return config.model_copy(update={"budget": config.budget.model_copy(update=changes)})


@pytest.fixture
def tree(tmp_path) -> InProcessTreeClient:
    return InProcessTreeClient(TreeStore(tmp_path / "tree.db"))


def context(tree, config, generator, critic, retriever=None, checker=None) -> ToTContext:  # noqa: PLR0913
    return ToTContext(
        retriever=retriever or FakeRetriever(),
        checker=checker or FakeChecker(),
        metrics=NoMetrics(),
        generator=generator,
        critic=critic,
        tree=tree,
        config=config,
    )


async def explore(tree, config, generator, critic, retriever=None, checker=None, change=CHANGE):
    ctx = context(tree, config, generator, critic, retriever, checker)
    return await explore_change(change, COMPARISON, ctx, RUN)


async def nodes_by_id(tree, result: ToTResult) -> dict[str, TreeNode]:
    assert result.tree_root_id is not None
    return {node.node_id: node for node in await tree.read_tree(result.tree_root_id)}


def only_evidence(result: ToTResult) -> TreeNode:
    (survivor,) = result.candidates
    (node,) = survivor.evidence
    return node


# -- the beam and its tie rule ----------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    name: str
    score: float
    level: int = 0  # lower is more direct evidence
    passage: float = 0.0


def pick(items, width, margin=0.05):
    return select_beam(items, width, margin, lambda i: i.score, lambda i: (i.level, -i.passage))


def names(items) -> list[str]:
    return [item.name for item in items]


def test_the_beam_keeps_the_top_scores_when_nothing_is_close():
    kept, dropped, tied = pick([Item("a", 0.9), Item("b", 0.8), Item("c", 0.6)], 2)
    assert (names(kept), names(dropped), tied) == (["a", "b"], ["c"], False)


def test_a_beam_wider_than_the_field_keeps_everything_best_first():
    kept, dropped, tied = pick([Item("a", 0.5), Item("b", 0.9)], 4)
    assert (names(kept), dropped, tied) == (["b", "a"], [], False)


def test_within_the_margin_at_the_cutoff_more_direct_evidence_wins():
    # b is 0.04 below a, inside the 0.05 margin, and states the cause where a only implies it.
    kept, dropped, tied = pick([Item("a", 0.80, level=1), Item("b", 0.76, level=0)], 1)
    assert (names(kept), names(dropped), tied) == (["b"], ["a"], True)


def test_outside_the_margin_the_higher_score_wins_whatever_the_evidence():
    kept, _, tied = pick([Item("a", 0.80, level=1), Item("b", 0.70, level=0)], 1)
    assert (names(kept), tied) == (["a"], False)


def test_the_margin_is_inclusive_of_its_edge():
    kept, _, tied = pick([Item("a", 0.80, level=1), Item("b", 0.75, level=0)], 1, margin=0.05)
    # 0.80 - 0.75 is not exactly 0.05 in floating point, so test the intent with a clear gap.
    assert tied is (abs(0.80 - 0.75) <= 0.05)
    assert names(kept) == (["b"] if tied else ["a"])


def test_the_better_supporting_passage_breaks_a_tie_in_evidence_level():
    kept, dropped, _ = pick(
        [Item("a", 0.80, level=0, passage=2.0), Item("b", 0.79, level=0, passage=5.0)], 1
    )
    assert (names(kept), names(dropped)) == (["b"], ["a"])


def test_a_full_tie_keeps_the_generators_order():
    kept, dropped, _ = pick([Item("a", 0.8), Item("b", 0.8), Item("c", 0.8)], 2)
    assert (names(kept), names(dropped)) == (["a", "b"], ["c"])


def test_clear_winners_keep_their_slots_and_only_the_cutoff_is_contested():
    items = [
        Item("top", 0.95, level=3),  # clearly above the cutoff, so its weak evidence is irrelevant
        Item("x", 0.70, level=2),
        Item("y", 0.68, level=0),
        Item("low", 0.30, level=0),
    ]
    kept, dropped, tied = pick(items, 2)
    assert (names(kept), names(dropped), tied) == (["top", "y"], ["x", "low"], True)


def test_the_evidence_order_the_tie_rule_reads_is_the_enum_order():
    assert list(EvidenceSupport)[0] is EvidenceSupport.STATES_CAUSE
    assert list(EvidenceSupport)[-1] is EvidenceSupport.NONE


# -- the config -------------------------------------------------------------------------------


def test_the_beam_and_budgets_come_from_the_config(config):
    beam, budget = config.beam, config.budget
    assert (beam.candidates, beam.candidate_width, beam.sources, beam.evidence_width) == (
        6,
        4,
        6,
        3,
    )
    assert beam.tie_margin == 0.05
    # The user's decision, 2026-11-06: a fresh round of its own beside the design's about 40.
    assert (budget.first_round_nodes, budget.fresh_round_nodes) == (40, 30)


# -- a search that works ----------------------------------------------------------------------


async def test_a_stated_cause_survives_with_everything_recorded(tree, config):
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(search("cocoa"))}
    )
    critic = FakeCritic(by_chunk={"now-1": Verdict(EvidenceSupport.STATES_CAUSE)})
    retriever = FakeRetriever(
        {
            ("cocoa", Section.MDNA): found(
                "cocoa",
                passage("now-1", score=4.5),
                prior=[passage("old-1", period=PRIOR, score=2.0)],
            )
        }
    )
    result = await explore(tree, config, generator, critic, retriever)

    assert result.outcome is Outcome.RESOLVED
    assert (result.stopped, result.review_reason, result.fresh_round) == (None, None, False)
    node = only_evidence(result)
    assert node.status is NodeStatus.KEPT
    assert node.evidence is EvidenceSupport.STATES_CAUSE
    assert [c.chunk_id for c in node.citations] == ["now-1"]
    # The score is the best among the passages behind a verified quote, not among all retrieved.
    assert node.best_passage_score == 4.5
    assert node.retrieved_chunk_ids == ("now-1", "old-1")
    assert node.score == pytest.approx(1.0)
    assert node.source is not None and node.source.section is Section.MDNA
    # One prescreen and one evaluation. Model calls: both generators, and the critic twice.
    assert (result.node_evaluations, result.model_calls) == (2, 4)
    nodes = await nodes_by_id(tree, result)
    assert nodes[result.tree_root_id].status is NodeStatus.KEPT
    assert result.candidates[0].candidate.status is NodeStatus.KEPT


async def test_the_critic_judges_the_claim_not_the_sources_description(tree, config):
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(search("cocoa"))}
    )
    critic = FakeCritic(by_chunk={"now-1": Verdict(EvidenceSupport.IMPLIES)})
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    await explore(tree, config, generator, critic, retriever)
    judged = [call for call in critic.calls if call[0] == "evidence"]
    # The depth 2 node's own thought is "look for cocoa", which is not what is being tested.
    assert [call[1] for call in judged] == ["Cocoa costs rose"]


async def test_passages_from_both_periods_reach_the_critic_current_first(tree, config):
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(search("cocoa"))}
    )
    critic = FakeCritic()
    retriever = FakeRetriever(
        {
            ("cocoa", Section.MDNA): found(
                "cocoa", passage("now-1"), prior=[passage("old-1", period=PRIOR)]
            )
        }
    )
    await explore(tree, config, generator, critic, retriever)
    (judged,) = [call for call in critic.calls if call[0] == "evidence"]
    assert judged[2] == ("now-1", "old-1")
    assert judged[3] == frozenset({CURRENT})


async def test_a_removed_item_is_judged_against_the_prior_period(tree, config):
    removed = CHANGE.model_copy(update={"kind": ChangeKind.REMOVED, "current_text": None})
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(search("cocoa"))}
    )
    critic = FakeCritic()
    retriever = FakeRetriever(
        {("cocoa", Section.MDNA): found("cocoa", prior=[passage("old-1", period=PRIOR)])}
    )
    await explore(tree, config, generator, critic, retriever, change=removed)
    (judged,) = [call for call in critic.calls if call[0] == "evidence"]
    assert judged[3] == frozenset({PRIOR})


async def test_survivors_come_back_best_first(tree, config):
    generator = FakeGenerator(
        [candidates("weak", "strong")],
        {"weak": sources(search("q-weak")), "strong": sources(search("q-strong"))},
    )
    critic = FakeCritic(
        by_chunk={
            "weak-1": Verdict(EvidenceSupport.IMPLIES),
            "strong-1": Verdict(EvidenceSupport.STATES_CAUSE),
        }
    )
    retriever = FakeRetriever(
        {
            ("q-weak", Section.MDNA): found("q-weak", passage("weak-1")),
            ("q-strong", Section.MDNA): found("q-strong", passage("strong-1")),
        }
    )
    result = await explore(tree, config, generator, critic, retriever)
    assert [c.candidate.thought for c in result.candidates] == ["strong", "weak"]


# -- co-drivers -------------------------------------------------------------------------------


def two_candidates(first: EvidenceSupport, second: EvidenceSupport):
    generator = FakeGenerator(
        [candidates("one", "two")],
        {"one": sources(search("q1")), "two": sources(search("q2"))},
    )
    critic = FakeCritic(by_chunk={"c1": Verdict(first), "c2": Verdict(second)})
    retriever = FakeRetriever(
        {
            ("q1", Section.MDNA): found("q1", passage("c1")),
            ("q2", Section.MDNA): found("q2", passage("c2")),
        }
    )
    return generator, critic, retriever


async def test_two_candidates_that_each_state_a_cause_are_reported_as_co_drivers(tree, config):
    result = await explore(
        tree, config, *two_candidates(EvidenceSupport.STATES_CAUSE, EvidenceSupport.STATES_CAUSE)
    )
    assert len(result.candidates) == 2
    assert set(result.co_drivers) == {c.candidate.node_id for c in result.candidates}


async def test_one_stated_cause_is_the_answer_and_not_a_co_driver(tree, config):
    result = await explore(
        tree, config, *two_candidates(EvidenceSupport.STATES_CAUSE, EvidenceSupport.IMPLIES)
    )
    assert len(result.candidates) == 2
    assert result.co_drivers == ()


# -- the depth 1 pre-screen -------------------------------------------------------------------


async def test_the_beam_keeps_four_candidates_and_prunes_the_rest_with_a_reason(tree, config):
    claims = [f"claim {i}" for i in range(1, 7)]
    generator = FakeGenerator(
        [candidates(*claims)], {c: sources(search(f"q{i}")) for i, c in enumerate(claims, 1)}
    )
    retriever = FakeRetriever(
        {(f"q{i}", Section.MDNA): found(f"q{i}", passage(f"now-{i}")) for i in range(1, 7)}
    )
    result = await explore(tree, config, generator, FakeCritic(), retriever)
    nodes = await nodes_by_id(tree, result)
    first_round = {n.thought: n for n in nodes.values() if n.depth == 1}
    # All equally specific, so the generator's order decides: depth 1 has no evidence to compare.
    assert [first_round[c].status for c in claims] == [NodeStatus.KEPT] * 4 + [
        NodeStatus.PRUNED
    ] * 2
    assert "outside the beam" in first_round["claim 5"].prune_reason
    assert generator.source_calls == claims[:4]
    assert len(result.candidates) == 4


async def test_a_vague_candidate_is_pruned_below_the_threshold(tree, config):
    generator = FakeGenerator(
        [candidates("Conditions changed", "Cocoa costs rose")],
        {"Cocoa costs rose": sources(search("cocoa"))},
    )
    critic = FakeCritic(specificity={"Conditions changed": Specificity.VAGUE})
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    result = await explore(tree, config, generator, critic, retriever)
    nodes = await nodes_by_id(tree, result)
    vague = next(n for n in nodes.values() if n.thought == "Conditions changed")
    assert vague.status is NodeStatus.PRUNED
    assert "below the 0.30 threshold" in vague.prune_reason
    assert "Conditions changed" not in generator.source_calls


async def test_on_a_text_change_a_lone_wrong_metric_prunes_on_score_not_the_gate(tree, config):
    # A text change has no changed metric, so nothing gates. One metric, contradicted, scores 0
    # on numerical consistency, which leaves at most 0.25 at depth 1: under the threshold.
    generator = FakeGenerator(
        [CandidateGeneration(candidates=(candidate("Margin rose", (GROSS_MARGIN_DOWN,)),))]
    )
    critic = FakeCritic()
    checker = FakeChecker({MetricName.GROSS_MARGIN: NumericCheck.CONTRADICT})
    result = await explore(tree, config, generator, critic, checker=checker)
    nodes = await nodes_by_id(tree, result)
    pruned = next(n for n in nodes.values() if n.thought == "Margin rose")
    assert pruned.status is NodeStatus.PRUNED
    assert "below the 0.30 threshold" in pruned.prune_reason
    assert pruned.numeric_check is NumericCheck.CONTRADICT
    assert [call[0] for call in critic.calls] == ["specificity"]


async def test_on_a_text_change_a_true_cause_with_one_wrong_metric_goes_on(tree, config):
    # HSY-S01 in T6.4 (2026-11-12): "lower sales volume" came with revenue down (wrong: revenue
    # rose on price) and gross margin down (right). It used to be gated at depth 1.
    claim = "Lower volume deleveraged fixed costs"
    generator = FakeGenerator(
        [CandidateGeneration(candidates=(candidate(claim, (REVENUE_DOWN, GROSS_MARGIN_DOWN)),))],
        {claim: sources(search("volume"))},
    )
    checker = FakeChecker(
        {MetricName.REVENUE: NumericCheck.CONTRADICT, MetricName.GROSS_MARGIN: NumericCheck.AGREE}
    )
    retriever = FakeRetriever({("volume", Section.MDNA): found("volume", passage("now-1"))})
    result = await explore(tree, config, generator, FakeCritic(), retriever, checker)
    nodes = await nodes_by_id(tree, result)
    first = next(n for n in nodes.values() if n.thought == claim)
    assert first.status is not NodeStatus.PRUNED
    assert first.score == pytest.approx(0.75 * 0.5 + 0.25 * 1.0)
    assert retriever.calls  # it reached depth 2 and was searched
    assert result.outcome is Outcome.RESOLVED


async def test_a_text_only_candidate_is_not_pruned_for_lacking_numbers(tree, config):
    generator = FakeGenerator(
        [candidates("A new risk factor")], {"A new risk factor": sources(search("risk"))}
    )
    retriever = FakeRetriever({("risk", Section.MDNA): found("risk", passage("now-1"))})
    result = await explore(tree, config, generator, FakeCritic(), retriever)
    assert result.outcome is Outcome.RESOLVED


# -- depth 2 pruning --------------------------------------------------------------------------


async def test_an_agreeing_metric_source_keeps_a_branch_with_no_citation(tree, config):
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(compute())}
    )
    critic = FakeCritic()
    retriever = FakeRetriever()
    checker = FakeChecker({MetricName.GROSS_MARGIN: NumericCheck.AGREE})
    result = await explore(tree, config, generator, critic, retriever, checker)
    node = only_evidence(result)
    # A metric is computed, not searched, and its node tests that one metric.
    assert retriever.calls == []
    assert node.expected_metrics == (GROSS_MARGIN_DOWN,)
    assert node.source.metric == GROSS_MARGIN_DOWN and node.source.section is None
    assert (node.numeric_check, node.citations, node.best_passage_score) == (
        NumericCheck.AGREE,
        (),
        None,
    )
    assert node.status is NodeStatus.KEPT
    # With no passages the critic is asked only about specificity, and about the claim.
    assert [c[:2] for c in critic.calls[1:]] == [("specificity", "Cocoa costs rose")]
    assert critic.calls[1][2] == (GROSS_MARGIN_DOWN,)


async def test_a_metric_with_no_data_is_pruned_and_never_retried(tree, config):
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(compute())}
    )
    retriever = FakeRetriever()
    result = await explore(tree, config, generator, FakeCritic(), retriever)
    nodes = await nodes_by_id(tree, result)
    metric_node = next(n for n in nodes.values() if n.depth == 2)
    assert metric_node.status is NodeStatus.PRUNED
    assert metric_node.prune_reason.startswith("no evidence: the metric has no data")
    # A search cannot supply a number the catalog does not have.
    assert retriever.calls == []


async def test_a_contradicting_metric_source_is_pruned_and_never_retried(tree, config):
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(compute())}
    )
    retriever = FakeRetriever()
    checker = FakeChecker({MetricName.GROSS_MARGIN: NumericCheck.CONTRADICT})
    result = await explore(tree, config, generator, FakeCritic(), retriever, checker)
    nodes = await nodes_by_id(tree, result)
    metric_node = next(n for n in nodes.values() if n.depth == 2)
    # A text change: not the gate, but a metric that moved the other way supports nothing.
    assert metric_node.prune_reason == (
        "no evidence: the metric moved the other way from what the claim expects"
    )
    assert retriever.calls == []
    assert result.outcome is Outcome.UNRESOLVED


LINKED_REASON = (
    "no evidence: the metric is the change being explained, or moves with it, so "
    "it cannot support the claim"
)


class ChangeMetrics:
    """A `MetricLookup` that knows the numeric change's own metric: gross margin."""

    def get_metric(self, metric_id: str):
        if metric_id != "HSY:gross_margin:FY2025:12m":
            return None
        return MetricValue(
            metric_id=metric_id,
            ticker="HSY",
            name="gross_margin",
            period=CURRENT,
            duration=Duration.TWELVE_MONTHS,
            value=0.335,
            unit="ratio",
            input_fact_ids=("f1",),
        )

    def get_move(self, comparison, metric_id: str):
        return None


MARGIN_CHANGE = Change(
    change_id="chg-1",
    comparison_id="HSY-yoy",
    kind=ChangeKind.NUMERIC,
    metric_id="HSY:gross_margin:FY2025:12m",
)


async def explore_margin_change(tree, config, generator, checker, critic=None):
    ctx = ToTContext(
        retriever=FakeRetriever(),
        checker=checker,
        metrics=ChangeMetrics(),
        generator=generator,
        critic=critic or FakeCritic(),
        tree=tree,
        config=config,
    )
    return await explore_change(MARGIN_CHANGE, COMPARISON, ctx, RUN)


def test_the_changed_metric_comes_from_the_lookup_not_the_id():
    assert changed_metric(MARGIN_CHANGE, ChangeMetrics()) is MetricName.GROSS_MARGIN
    # A text change has no metric, and an id the lookup does not know gives none either.
    assert changed_metric(CHANGE, ChangeMetrics()) is None
    unknown = MARGIN_CHANGE.model_copy(update={"metric_id": "HSY:gross_margin:FY1999:12m"})
    assert changed_metric(unknown, ChangeMetrics()) is None


async def test_agreeing_with_the_changed_metric_is_not_evidence(tree, config):
    # The user's decision (option A): gross margin fell by definition, so a branch whose only
    # support is "gross margin down" has nothing behind it.
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(compute())}
    )
    checker = FakeChecker({MetricName.GROSS_MARGIN: NumericCheck.AGREE})
    result = await explore_margin_change(tree, config, generator, checker)
    nodes = await nodes_by_id(tree, result)
    metric_node = next(n for n in nodes.values() if n.depth == 2)
    assert metric_node.status is NodeStatus.PRUNED
    assert metric_node.prune_reason == LINKED_REASON
    assert metric_node.numeric_check is NumericCheck.NOT_APPLICABLE
    assert result.outcome is Outcome.UNRESOLVED


async def test_an_unlinked_metric_still_counts_as_evidence_for_a_numeric_change(tree, config):
    # Cost of sales in dollars can rise while gross margin holds, so agreeing with it says
    # something about a fall in gross margin.
    cogs_dollars = MetricExpectation(metric=MetricName.COST_OF_GOODS_SOLD, direction=Direction.UP)
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(compute(cogs_dollars))}
    )
    checker = FakeChecker({MetricName.COST_OF_GOODS_SOLD: NumericCheck.AGREE})
    result = await explore_margin_change(tree, config, generator, checker)
    assert only_evidence(result).numeric_check is NumericCheck.AGREE
    assert result.outcome is Outcome.RESOLVED


async def test_a_metric_linked_to_the_change_by_arithmetic_is_not_evidence(tree, config):
    # Gross margin plus the cost of sales ratio is 1, so "cost of sales ratio up" is the same
    # fact as "gross margin down", the change itself.
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(compute(COGS_UP))}
    )
    checker = FakeChecker({MetricName.COGS_RATIO: NumericCheck.AGREE})
    result = await explore_margin_change(tree, config, generator, checker)
    nodes = await nodes_by_id(tree, result)
    metric_node = next(n for n in nodes.values() if n.depth == 2)
    assert metric_node.prune_reason == LINKED_REASON
    assert result.outcome is Outcome.UNRESOLVED


async def test_contradicting_the_changed_metric_still_prunes(tree, config):
    # A candidate that says gross margin rose contradicts the change itself.
    generator = FakeGenerator(
        [
            CandidateGeneration(
                candidates=(candidate("Pricing lifted margin", (GROSS_MARGIN_DOWN,)),)
            )
        ],
        {},
    )
    checker = FakeChecker({MetricName.GROSS_MARGIN: NumericCheck.CONTRADICT})
    critic = FakeCritic()
    result = await explore_margin_change(tree, config, generator, checker, critic)
    nodes = await nodes_by_id(tree, result)
    first = next(n for n in nodes.values() if n.depth == 1)
    assert first.status is NodeStatus.PRUNED
    assert first.prune_reason == "hard gate: computed metrics contradict the change being explained"
    # Pruned before the critic is paid; one generator call per round and nothing else.
    assert critic.calls == []
    assert result.model_calls == 2


async def test_a_text_change_keeps_metric_agreement_as_evidence(tree, config):
    # Nothing is set aside when the change has no metric: the existing rule stands.
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(compute())}
    )
    checker = FakeChecker({MetricName.GROSS_MARGIN: NumericCheck.AGREE})
    result = await explore(tree, config, generator, FakeCritic(), checker=checker)
    assert only_evidence(result).numeric_check is NumericCheck.AGREE


async def test_a_search_inherits_the_candidates_expected_metrics(tree, config):
    generator = FakeGenerator(
        [CandidateGeneration(candidates=(candidate("Cocoa costs rose", (GROSS_MARGIN_DOWN,)),))],
        {"Cocoa costs rose": sources(search("cocoa"))},
    )
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    checker = FakeChecker({MetricName.GROSS_MARGIN: NumericCheck.AGREE})
    result = await explore(tree, config, generator, FakeCritic(), retriever, checker)
    assert only_evidence(result).expected_metrics == (GROSS_MARGIN_DOWN,)
    assert checker.calls[-1] == (GROSS_MARGIN_DOWN,)


async def test_a_weak_branch_below_the_threshold_is_pruned(tree, config):
    # Vague with no support: 0.4 * 0 + 0.2 * 1 + 0.1 * 0 over 0.7 is 0.29, under 0.30.
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(search("cocoa"))}
    )
    critic = FakeCritic(
        by_chunk={"now-1": Verdict(EvidenceSupport.NONE, specificity=Specificity.VAGUE)}
    )
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    result = await explore(tree, config, generator, critic, retriever)
    nodes = await nodes_by_id(tree, result)
    branch = next(n for n in nodes.values() if n.depth == 2)
    assert branch.status is NodeStatus.PRUNED
    assert result.outcome is Outcome.UNRESOLVED


async def test_the_evidence_beam_keeps_three_branches_per_candidate(tree, config):
    names_ = ["a", "b", "c", "d"]
    generator = FakeGenerator(
        [candidates("claim")],
        {"claim": sources(*[search(f"q-{n}", Section.MDNA if n == "a" else None) for n in names_])},
    )
    # Sections must differ for the generator's own dedupe, which is faked away here, so keep the
    # queries distinct instead.
    levels = {
        "a-1": EvidenceSupport.STATES_CAUSE,
        "b-1": EvidenceSupport.IMPLIES,
        "c-1": EvidenceSupport.CONSISTENT,
        "d-1": EvidenceSupport.CONSISTENT,
    }
    critic = FakeCritic(by_chunk={k: Verdict(v) for k, v in levels.items()})
    retriever = FakeRetriever(
        {
            (f"q-{n}", Section.MDNA if n == "a" else None): found(f"q-{n}", passage(f"{n}-1"))
            for n in names_
        }
    )
    result = await explore(tree, config, generator, critic, retriever)
    nodes = await nodes_by_id(tree, result)
    depth2 = sorted((n for n in nodes.values() if n.depth == 2), key=lambda n: n.node_id)
    statuses = [n.status for n in depth2]
    assert statuses.count(NodeStatus.KEPT) == 3 and statuses.count(NodeStatus.PRUNED) == 1
    dropped = next(n for n in depth2 if n.status is NodeStatus.PRUNED)
    assert "outside the beam" in dropped.prune_reason
    assert len(result.candidates[0].evidence) == 3


async def test_a_tie_at_the_evidence_cutoff_goes_to_the_more_direct_evidence(tree, config):
    # a: consistent and uncontradicted, 0.657. b: states the cause but possibly contradicted and
    # only partly specific, 0.643. Within 0.05, so the statement of cause takes the one slot.
    generator = FakeGenerator(
        [candidates("claim")],
        {"claim": sources(search("qa", Section.MDNA), search("qb", Section.FINANCIAL_NOTES))},
    )
    critic = FakeCritic(
        by_chunk={
            "a-1": Verdict(EvidenceSupport.CONSISTENT),
            "b-1": Verdict(EvidenceSupport.STATES_CAUSE, Contradiction.DIRECT, Specificity.PARTIAL),
        }
    )
    retriever = FakeRetriever(
        {
            ("qa", Section.MDNA): found("qa", passage("a-1")),
            ("qb", Section.FINANCIAL_NOTES): found(
                "qb", passage("b-1", section=Section.FINANCIAL_NOTES)
            ),
        }
    )
    narrow = with_beam(config, evidence_width=1)
    result = await explore(tree, narrow, generator, critic, retriever)
    assert only_evidence(result).evidence is EvidenceSupport.STATES_CAUSE


async def test_beyond_the_margin_the_higher_score_wins_the_slot(tree, config):
    generator = FakeGenerator(
        [candidates("claim")],
        {"claim": sources(search("qa", Section.MDNA), search("qb", Section.FINANCIAL_NOTES))},
    )
    critic = FakeCritic(
        by_chunk={
            "a-1": Verdict(EvidenceSupport.CONSISTENT),
            # 0.571: well below a's 0.657, so stating the cause no longer matters.
            "b-1": Verdict(EvidenceSupport.STATES_CAUSE, Contradiction.DIRECT, Specificity.VAGUE),
        }
    )
    retriever = FakeRetriever(
        {
            ("qa", Section.MDNA): found("qa", passage("a-1")),
            ("qb", Section.FINANCIAL_NOTES): found(
                "qb", passage("b-1", section=Section.FINANCIAL_NOTES)
            ),
        }
    )
    narrow = with_beam(config, evidence_width=1)
    result = await explore(tree, narrow, generator, critic, retriever)
    assert only_evidence(result).evidence is EvidenceSupport.CONSISTENT


async def test_the_better_supporting_passage_breaks_a_tie_between_equal_branches(tree, config):
    generator = FakeGenerator(
        [candidates("claim")],
        {"claim": sources(search("qa", Section.MDNA), search("qb", Section.FINANCIAL_NOTES))},
    )
    critic = FakeCritic(
        by_chunk={"a-1": Verdict(EvidenceSupport.IMPLIES), "b-1": Verdict(EvidenceSupport.IMPLIES)}
    )
    retriever = FakeRetriever(
        {
            ("qa", Section.MDNA): found("qa", passage("a-1", score=2.0)),
            ("qb", Section.FINANCIAL_NOTES): found(
                "qb", passage("b-1", section=Section.FINANCIAL_NOTES, score=5.0)
            ),
        }
    )
    narrow = with_beam(config, evidence_width=1)
    result = await explore(tree, narrow, generator, critic, retriever)
    # The generator listed a first, but b's best supporting passage scored higher.
    assert only_evidence(result).node_id.endswith(":e2")
    assert only_evidence(result).best_passage_score == 5.0


# -- the one reformulated search --------------------------------------------------------------


def one_search(query: str = "cocoa", section: Section | None = Section.MDNA):
    return FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(search(query, section))}
    )


def sections_searched(retriever: FakeRetriever) -> list[Section | None]:
    return [section for _, section in retriever.calls]


async def test_a_branch_with_no_evidence_is_searched_again_everywhere_first(tree, config):
    retriever = FakeRetriever({("cocoa", None): found("cocoa", passage("any-1"))})
    critic = FakeCritic(by_chunk={"any-1": Verdict(EvidenceSupport.STATES_CAUSE)})
    result = await explore(tree, config, one_search(), critic, retriever)
    # MD&A was the branch's own search and found nothing, so the retry is one search of everywhere.
    assert sections_searched(retriever) == [Section.MDNA, None]
    node = only_evidence(result)
    assert node.status is NodeStatus.KEPT and node.evidence is EvidenceSupport.STATES_CAUSE
    assert node.retrieved_chunk_ids == ("any-1",)
    # The retry's search is free; only its evaluation counts: prescreen, first try, retry.
    assert result.node_evaluations == 3


async def test_a_branch_with_a_wrong_extra_metric_still_gets_its_retry(tree, config):
    # Only a contradiction of the change itself rules out the retry; a wrong extra metric on a
    # text change lowers the score but is still missing evidence worth one more search.
    claim = "Lower volume deleveraged fixed costs"
    generator = FakeGenerator(
        [CandidateGeneration(candidates=(candidate(claim, (REVENUE_DOWN, GROSS_MARGIN_DOWN)),))],
        {claim: sources(search("volume"))},
    )
    checker = FakeChecker(
        {MetricName.REVENUE: NumericCheck.CONTRADICT, MetricName.GROSS_MARGIN: NumericCheck.AGREE}
    )
    retriever = FakeRetriever({("volume", None): found("volume", passage("any-1"))})
    critic = FakeCritic(by_chunk={"any-1": Verdict(EvidenceSupport.STATES_CAUSE)})
    result = await explore(tree, config, generator, critic, retriever, checker)
    assert sections_searched(retriever) == [Section.MDNA, None]
    assert only_evidence(result).status is NodeStatus.KEPT


@pytest.mark.parametrize("own", [Section.MDNA, Section.FINANCIAL_NOTES, Section.MARKET_RISK])
async def test_the_retry_is_a_single_search_of_everywhere_whatever_the_own_section(
    tree, config, own
):
    # Named sections are not searched, even when one holds the evidence. It would be found only by a
    # fallback that real filings never needed (see the note at the top of controller.py).
    retriever = FakeRetriever({("cocoa", Section.RISK_FACTORS): found("cocoa", passage("risk-1"))})
    result = await explore(tree, config, one_search(section=own), FakeCritic(), retriever)
    assert sections_searched(retriever) == [own, None]
    assert result.outcome is Outcome.UNRESOLVED


async def test_a_claim_with_evidence_only_in_an_other_table_is_found_by_the_retry(tree, config):
    other = passage("other-1", section=Section.OTHER)
    retriever = FakeRetriever({("cocoa", None): found("cocoa", other)})
    critic = FakeCritic(by_chunk={"other-1": Verdict(EvidenceSupport.STATES_CAUSE)})
    result = await explore(tree, config, one_search(), critic, retriever)
    # A named section never returns an `other` table; only everywhere does.
    assert sections_searched(retriever) == [Section.MDNA, None]
    assert Section.OTHER not in sections_searched(retriever)
    assert only_evidence(result).citations[0].section is Section.OTHER


async def test_a_branch_that_searched_everywhere_is_not_retried(tree, config):
    retriever = FakeRetriever()
    result = await explore(tree, config, one_search(section=None), FakeCritic(), retriever)
    assert result.outcome is Outcome.UNRESOLVED
    # The same query over the same scope can only return the same passages.
    assert sections_searched(retriever) == [None]
    nodes = await nodes_by_id(tree, result)
    branch = next(n for n in nodes.values() if n.depth == 2)
    assert branch.status is NodeStatus.PRUNED
    assert "already covered every section" in branch.prune_reason
    # Prescreen and the one evaluation.
    assert result.node_evaluations == 2


async def test_no_evidence_anywhere_prunes_the_branch_and_then_the_candidate(tree, config):
    retriever = FakeRetriever()
    result = await explore(tree, config, one_search(), FakeCritic(), retriever)
    nodes = await nodes_by_id(tree, result)
    branch = next(n for n in nodes.values() if n.depth == 2)
    assert branch.status is NodeStatus.PRUNED
    assert branch.prune_reason == "no evidence after one reformulated search of everywhere"
    parent = next(n for n in nodes.values() if n.depth == 1)
    assert (parent.status, parent.prune_reason) == (
        NodeStatus.PRUNED,
        "no evidence branch survived",
    )
    assert result.outcome is Outcome.UNRESOLVED
    assert sections_searched(retriever) == [Section.MDNA, None]
    # The prescreen and the first try. The retry found nothing to evaluate, and the fresh round
    # proposed nothing, so neither added an evaluation.
    assert result.node_evaluations == 2


async def test_the_retry_is_one_search_even_if_the_critic_then_finds_nothing(tree, config):
    retriever = FakeRetriever({("cocoa", None): found("cocoa", passage("any-1"))})
    critic = FakeCritic(by_chunk={"any-1": Verdict(EvidenceSupport.NONE)})
    result = await explore(tree, config, one_search(), critic, retriever)
    # Everywhere returned a passage, which the critic found useless: that ends the branch, and no
    # named section is tried as a second chance.
    assert sections_searched(retriever) == [Section.MDNA, None]
    nodes = await nodes_by_id(tree, result)
    branch = next(n for n in nodes.values() if n.depth == 2)
    assert branch.prune_reason == "no evidence after one reformulated search of everywhere"
    assert result.node_evaluations == 3


async def test_a_prior_period_hit_is_not_evidence_for_a_claim_about_this_period(tree, config):
    # Only the prior period has anything. The citation is dropped as the wrong period, so the
    # branch has no evidence; the retry returns only the prior period again, so it judges nothing.
    old = passage("old-1", period=PRIOR)
    retriever = FakeRetriever(
        {("cocoa", section): found("cocoa", prior=[old]) for section in [Section.MDNA, None]}
    )
    critic = FakeCritic(by_chunk={"old-1": Verdict(EvidenceSupport.STATES_CAUSE)})
    result = await explore(tree, config, one_search(), critic, retriever)
    assert sections_searched(retriever) == [Section.MDNA, None]
    nodes = await nodes_by_id(tree, result)
    branch = next(n for n in nodes.values() if n.depth == 2)
    assert branch.status is NodeStatus.PRUNED
    assert branch.citations == ()


async def test_a_failed_critic_is_not_missing_evidence_and_is_not_retried(tree, config):
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    critic = FakeCritic(fail_on_chunks={"now-1": LLMError("the model refused")})
    result = await explore(tree, config, one_search(), critic, retriever)
    assert len(retriever.calls) == 1
    nodes = await nodes_by_id(tree, result)
    branch = next(n for n in nodes.values() if n.depth == 2)
    assert branch.status is NodeStatus.PRUNED
    assert branch.prune_reason.startswith("unresolved: the critic failed")
    assert any("the critic failed" in failure for failure in result.failures)


# -- model failures ---------------------------------------------------------------------------


async def test_a_refused_prescreen_ends_that_candidate_for_review_and_not_the_run(tree, config):
    class RefusesOne(FakeCritic):
        async def rate_specificity(self, node):
            if node.thought == "Refused":
                raise LLMError("the model refused")
            return await super().rate_specificity(node)

    generator = FakeGenerator(
        [candidates("Refused", "Cocoa costs rose")], {"Cocoa costs rose": sources(search("cocoa"))}
    )
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    result = await explore(tree, config, generator, RefusesOne(), retriever)
    assert result.outcome is Outcome.RESOLVED
    nodes = await nodes_by_id(tree, result)
    refused = next(n for n in nodes.values() if n.thought == "Refused")
    assert refused.status is NodeStatus.PRUNED
    assert refused.prune_reason.startswith("unresolved: the critic failed")
    assert refused.score is None
    assert len(result.failures) == 1


async def test_a_generator_refusal_ends_the_item_for_review_without_a_fresh_round(tree, config):
    generator = FakeGenerator([CandidateGeneration(failure="the generator failed: refused")])
    result = await explore(tree, config, generator, FakeCritic())
    assert result.outcome is Outcome.UNRESOLVED
    assert "the generator failed" in result.review_reason
    assert len(generator.candidate_calls) == 1 and result.fresh_round is False
    assert result.failures == ("the generator failed: refused",)


async def test_a_source_generation_failure_ends_that_candidate_only(tree, config):
    generator = FakeGenerator(
        [candidates("Broken", "Cocoa costs rose")],
        {
            "Broken": EvidenceGeneration(failure="the generator failed: refused"),
            "Cocoa costs rose": sources(search("cocoa")),
        },
    )
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    result = await explore(tree, config, generator, FakeCritic(), retriever)
    assert result.outcome is Outcome.RESOLVED
    nodes = await nodes_by_id(tree, result)
    broken = next(n for n in nodes.values() if n.thought == "Broken")
    assert broken.status is NodeStatus.PRUNED
    assert broken.prune_reason.startswith("unresolved: the generator failed")


async def test_a_reassessment_counts_as_one_more_model_call(tree, config):
    text = passage("now-1").chunk.text
    critic = FakeCritic(
        by_chunk={
            "now-1": Verdict(
                EvidenceSupport.STATES_CAUSE,
                citations=(
                    ProposedCitation(chunk_id="now-1", quote=text),
                    ProposedCitation(chunk_id="now-1", quote="words the chunk never says"),
                ),
            )
        }
    )
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    result = await explore(tree, config, one_search(), critic, retriever)
    assert any(call[0] == "reassess" for call in critic.calls)
    # Two generators, the prescreen, the judgment, and the reassessment.
    assert result.model_calls == 5


class FailsOnCall(FakeRetriever):
    """A retriever that works until its `fail_on`th call, then raises as the real one does."""

    def __init__(self, fail_on: int, script=None) -> None:
        super().__init__(script)
        self.fail_on = fail_on

    async def retrieve(self, query, comparison, section=None):
        if len(self.calls) + 1 == self.fail_on:
            raise IndexOutOfSyncError("the index and the fact store disagree")
        return await super().retrieve(query, comparison, section)


@pytest.mark.parametrize("fail_on", [1, 2], ids=["first_search", "during_the_retry"])
async def test_a_retriever_failure_ends_the_run_and_is_not_read_as_missing_evidence(
    tree, config, fail_on
):
    # `IndexOutOfSyncError` means the search is broken, not that the filings lack an answer. Reading
    # it as "no evidence" would prune sound branches and label them unsupported, so it propagates.
    generator = one_search()
    ctx = context(tree, config, generator, FakeCritic(), FailsOnCall(fail_on))
    with pytest.raises(IndexOutOfSyncError):
        await explore_changes(ranked("chg-1", "chg-2"), COMPARISON, ctx, CostLedger(1.0), RUN)
    stored = await tree.read_tree(f"{RUN}:chg-1:root")
    branch = next(node for node in stored if node.depth == 2)
    # Never judged, so it is pending with no reason and no score, and the audit trail shows that.
    assert (branch.status, branch.prune_reason, branch.score) == (NodeStatus.PENDING, None, None)
    # Known gap: a change that ends this way has no budget snapshot, and the later change was never
    # started, so whoever resumes the run must treat the tree as unfinished.
    assert await tree.read_budget(RUN) is None
    assert len(generator.candidate_calls) == 1


# -- the fresh round and the budgets ----------------------------------------------------------


async def test_every_candidate_pruned_makes_one_fresh_round_that_remembers_the_first(tree, config):
    generator = FakeGenerator(
        [candidates("Conditions changed"), candidates("Cocoa costs rose")],
        {"Cocoa costs rose": sources(search("cocoa"))},
    )
    critic = FakeCritic(specificity={"Conditions changed": Specificity.VAGUE})
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    result = await explore(tree, config, generator, critic, retriever)
    assert result.outcome is Outcome.RESOLVED and result.fresh_round is True
    assert generator.candidate_calls[1]["already_tried"] == ("Conditions changed",)
    nodes = await nodes_by_id(tree, result)
    assert {n.node_id.split(":")[2] for n in nodes.values() if n.depth == 1} == {"r1", "r2"}
    # The first round's pruned branch is still in the tree, with its reason.
    assert nodes[f"{RUN}:chg-1:r1:c1"].status is NodeStatus.PRUNED


async def test_a_fresh_round_that_also_fails_leaves_the_item_unresolved(tree, config):
    generator = FakeGenerator([candidates("Vague one"), candidates("Vague two")])
    critic = FakeCritic(
        specificity={"Vague one": Specificity.VAGUE, "Vague two": Specificity.VAGUE}
    )
    result = await explore(tree, config, generator, critic)
    assert result.outcome is Outcome.UNRESOLVED and result.fresh_round is True
    assert "first round and in the fresh one" in result.review_reason
    assert len(generator.candidate_calls) == 2  # one fresh round, never a third


async def test_the_fresh_round_has_its_own_node_budget(tree, config):
    # A first round capped at 3: its prescreen and two evaluations. It ends with nothing, and the
    # fresh round still has all of its own 30 to test a new candidate.
    generator = FakeGenerator(
        [candidates("First try"), candidates("Second try")],
        {
            "First try": sources(*[search(f"q{i}", None if i else Section.MDNA) for i in range(6)]),
            "Second try": sources(search("cocoa")),
        },
    )
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    small = with_budget(config, first_round_nodes=3)
    result = await explore(tree, small, generator, FakeCritic(), retriever)
    assert result.outcome is Outcome.RESOLVED
    assert result.fresh_round is True and result.stopped is StopReason.NODE_BUDGET
    assert result.node_evaluations == 3 + 2


async def test_a_spent_node_budget_ends_the_search_with_the_best_branches_so_far(tree, config):
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")],
        {
            "Cocoa costs rose": sources(
                search("qa"), search("qb", None), search("qc", Section.RISK_FACTORS)
            )
        },
    )
    retriever = FakeRetriever({("qa", Section.MDNA): found("qa", passage("a-1"))})
    small = with_budget(config, first_round_nodes=2)
    result = await explore(tree, small, generator, FakeCritic(), retriever)
    assert result.outcome is Outcome.RESOLVED and result.stopped is StopReason.NODE_BUDGET
    assert result.fresh_round is False  # it has survivors
    nodes = await nodes_by_id(tree, result)
    statuses = {n.node_id.rsplit(":", 1)[1]: n.status for n in nodes.values() if n.depth == 2}
    # The branches never evaluated stay pending: they were not judged weak, they were not judged.
    assert statuses == {"e1": NodeStatus.KEPT, "e2": NodeStatus.PENDING, "e3": NodeStatus.PENDING}
    assert result.node_evaluations == 2


async def test_a_branch_waiting_for_its_retry_when_the_budget_ends_is_pruned_for_no_evidence(
    tree, config
):
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(search("cocoa"))}
    )
    # Prescreen and the first try use both nodes, so the retry has no room.
    small = with_budget(config, first_round_nodes=2)
    result = await explore(tree, small, generator, FakeCritic(), FakeRetriever())
    nodes = await nodes_by_id(tree, result)
    branch = next(n for n in nodes.values() if n.depth == 2)
    assert branch.status is NodeStatus.PRUNED
    assert "ended before its reformulated attempt" in branch.prune_reason


async def test_a_spent_run_budget_ends_the_search_and_skips_the_fresh_round(tree, config):
    generator = FakeGenerator(
        [candidates("Cocoa costs rose"), candidates("Never asked")],
        {"Cocoa costs rose": sources(search("qa"), search("qb", None))},
    )
    retriever = FakeRetriever(
        {
            ("qa", Section.MDNA): found("qa", passage("a-1")),
            ("qb", None): found("qb", passage("b-1")),
        }
    )
    # The prescreen and the first judgment succeed, and the third critic call is refused.
    critic = FakeCritic(raise_after=(2, BudgetExceededError("cap reached")))
    result = await explore(tree, config, generator, critic, retriever)
    assert result.outcome is Outcome.RESOLVED and result.stopped is StopReason.RUN_BUDGET
    assert len(generator.candidate_calls) == 1
    assert only_evidence(result).node_id.endswith(":e1")


async def test_a_spent_run_budget_before_any_evidence_is_unresolved_for_review(tree, config):
    generator = FakeGenerator([candidates("Cocoa costs rose"), candidates("Never asked")])
    critic = FakeCritic(raise_after=(0, BudgetExceededError("cap reached")))
    result = await explore(tree, config, generator, critic)
    assert result.outcome is Outcome.UNRESOLVED and result.stopped is StopReason.RUN_BUDGET
    assert "dollar budget" in result.review_reason
    assert len(generator.candidate_calls) == 1


# -- the whole tree ---------------------------------------------------------------------------


async def test_every_node_is_stored_with_a_unique_id_and_pruned_ones_say_why(tree, config):
    generator = FakeGenerator(
        [candidates("Conditions changed", "Cocoa costs rose")],
        {"Cocoa costs rose": sources(search("cocoa"), compute())},
    )
    critic = FakeCritic(specificity={"Conditions changed": Specificity.VAGUE})
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    result = await explore(tree, config, generator, critic, retriever)
    stored = await tree.read_tree(result.tree_root_id)
    ids = [node.node_id for node in stored]
    assert len(ids) == len(set(ids)) == 5  # root, two candidates, two sources
    assert all(node_id.startswith(f"{RUN}:chg-1:") for node_id in ids)
    pruned = [node for node in stored if node.status is NodeStatus.PRUNED]
    assert len(pruned) == 2 and all(node.prune_reason for node in pruned)


async def test_the_result_matches_what_the_store_holds(tree, config):
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(search("cocoa"))}
    )
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    result = await explore(tree, config, generator, FakeCritic(), retriever)
    nodes = await nodes_by_id(tree, result)
    (survivor,) = result.candidates
    assert survivor.candidate == nodes[survivor.candidate.node_id]
    assert survivor.evidence == tuple(nodes[n.node_id] for n in survivor.evidence)


# -- finding the claim again ------------------------------------------------------------------

COGS_UP = MetricExpectation(metric=MetricName.COGS_RATIO, direction=Direction.UP)


def stored(node_id: str, depth: int, parent_id: str | None, thought: str, **fields) -> TreeNode:
    return TreeNode(
        node_id=node_id,
        run_id="r",
        change_id="c",
        parent_id=parent_id,
        depth=depth,
        thought=thought,
        **fields,
    )


CLAIM_NODE = stored(
    "c1",
    1,
    "root",
    "Cocoa costs rose",
    expected_evidence="MD&A names cocoa",
    expected_metrics=(GROSS_MARGIN_DOWN,),
)
SEARCH_NODE = stored(
    "c1:e1",
    2,
    "c1",
    "look for cocoa",
    expected_metrics=(GROSS_MARGIN_DOWN,),
    source=EvidenceSourceRef(section=Section.MDNA),
)
METRIC_NODE = stored(
    "c1:e2",
    2,
    "c1",
    "compute it",
    expected_metrics=(COGS_UP,),
    source=EvidenceSourceRef(metric=COGS_UP),
)


def test_the_claim_node_is_the_candidate_with_the_branchs_own_metrics():
    searched = claim_node(CLAIM_NODE, SEARCH_NODE)
    computed = claim_node(CLAIM_NODE, METRIC_NODE)
    # What the critic reads is the claim, never the source's description of where to look.
    for rebuilt in (searched, computed):
        assert rebuilt.thought == "Cocoa costs rose"
        assert rebuilt.expected_evidence == "MD&A names cocoa"
    # A search tests the candidate's metrics, and a metric source tests only its own.
    assert searched.expected_metrics == (GROSS_MARGIN_DOWN,)
    assert computed.expected_metrics == (COGS_UP,)
    # The candidate itself is a frozen record and is left as it was.
    assert CLAIM_NODE.expected_metrics == (GROSS_MARGIN_DOWN,)


@pytest.mark.parametrize(
    ("candidate_node", "evidence_node"),
    [
        (SEARCH_NODE, SEARCH_NODE),  # not a candidate
        (CLAIM_NODE, CLAIM_NODE),  # not an evidence branch
        (CLAIM_NODE, stored("other:e1", 2, "other", "elsewhere")),  # another candidate's branch
    ],
)
def test_the_claim_node_refuses_nodes_that_do_not_belong_together(candidate_node, evidence_node):
    with pytest.raises(ValueError):
        claim_node(candidate_node, evidence_node)


def test_a_claim_is_found_from_a_candidate_or_from_its_evidence():
    tree_nodes = [stored("root", 0, None, "Gross margin fell"), CLAIM_NODE, SEARCH_NODE]
    assert claim_of(CLAIM_NODE, tree_nodes) == "Cocoa costs rose"
    # The evidence node's own thought is "look for cocoa"; the claim is its parent's.
    assert claim_of(SEARCH_NODE, tree_nodes) == "Cocoa costs rose"


def test_the_root_and_an_orphaned_branch_have_no_claim_to_show():
    root = stored("root", 0, None, "Gross margin fell")
    with pytest.raises(ValueError, match="not a claim"):
        claim_of(root, [root, CLAIM_NODE])
    with pytest.raises(ValueError, match="no candidate"):
        claim_of(SEARCH_NODE, [root])


async def test_a_branch_rebuilt_from_the_stored_tree_is_what_the_critic_was_shown(tree, config):
    generator = FakeGenerator(
        [candidates("Cocoa costs rose")], {"Cocoa costs rose": sources(compute())}
    )
    critic = FakeCritic()
    checker = FakeChecker({MetricName.GROSS_MARGIN: NumericCheck.AGREE})
    result = await explore(tree, config, generator, critic, checker=checker)
    stored_tree = await tree.read_tree(result.tree_root_id)
    (survivor,) = result.candidates
    evidence = survivor.evidence[0]
    rebuilt = claim_node(survivor.candidate, evidence)
    # calls[0] is the candidate's prescreen; calls[1] is this branch, a metric with no passages.
    kind, claim_seen, metrics_seen = critic.calls[1]
    assert (rebuilt.thought, rebuilt.expected_metrics) == (claim_seen, metrics_seen)
    assert claim_of(evidence, stored_tree) == "Cocoa costs rose" == rebuilt.thought
    assert evidence.thought == "compute it"


# -- a run of changes -------------------------------------------------------------------------


def ranked(*ids: str) -> RankedChanges:
    return RankedChanges(
        comparison_id="HSY-yoy",
        changes=tuple(
            CHANGE.model_copy(update={"change_id": cid, "materiality_rank": rank})
            for rank, cid in enumerate(ids, start=1)
        ),
    )


async def test_changes_are_searched_in_order_and_the_budget_is_recorded(tree, config):
    generator = FakeGenerator(
        [candidates("Cocoa costs rose"), candidates("Cocoa costs rose")],
        {"Cocoa costs rose": sources(search("cocoa"))},
    )
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    ledger = CostLedger(1.0)
    ctx = context(tree, config, generator, FakeCritic(), retriever)
    results = await explore_changes(
        ranked("chg-1", "chg-2"), COMPARISON, ctx, ledger, RUN, starting_evaluations=10
    )
    assert [r.change_id for r in results] == ["chg-1", "chg-2"]
    assert all(r.outcome is Outcome.RESOLVED for r in results)
    budget = await tree.read_budget(RUN)
    # The resumed run's earlier count is carried forward, because a new ledger starts at zero.
    assert budget.node_evaluations == 10 + sum(r.node_evaluations for r in results)
    assert (budget.spent_usd, budget.max_usd) == (0.0, 1.0)


async def test_when_the_run_budget_is_reached_the_least_material_changes_go_to_review(tree, config):
    generator = FakeGenerator([candidates("Cocoa costs rose")])
    critic = FakeCritic(raise_after=(0, BudgetExceededError("cap reached")))
    ctx = context(tree, config, generator, critic)
    results = await explore_changes(
        ranked("chg-1", "chg-2", "chg-3"), COMPARISON, ctx, CostLedger(1.0), RUN
    )
    assert [r.outcome for r in results] == [Outcome.UNRESOLVED] * 3
    assert results[0].tree_root_id is not None
    for skipped in results[1:]:
        assert skipped.tree_root_id is None and skipped.stopped is StopReason.RUN_BUDGET
        assert "before this change was examined" in skipped.review_reason
    # Only the first change was searched.
    assert len(generator.candidate_calls) == 1


# -- the result record ------------------------------------------------------------------------


def a_node() -> TreeNode:
    return TreeNode(node_id="n", run_id="r", change_id="c", parent_id="root", depth=1, thought="t")


def test_a_resolved_result_needs_a_survivor_and_no_review_reason():
    survivor = CandidateResult(candidate=a_node(), evidence=())
    with pytest.raises(ValidationError, match="surviving candidate"):
        ToTResult(change_id="c", tree_root_id="root", outcome=Outcome.RESOLVED)
    with pytest.raises(ValidationError, match="does not go to review"):
        ToTResult(
            change_id="c",
            tree_root_id="root",
            outcome=Outcome.RESOLVED,
            candidates=(survivor,),
            review_reason="why",
        )


def test_an_unresolved_result_has_no_survivors_and_says_why():
    with pytest.raises(ValidationError, match="says why"):
        ToTResult(change_id="c", tree_root_id="root", outcome=Outcome.UNRESOLVED)
    with pytest.raises(ValidationError, match="says why"):
        ToTResult(
            change_id="c",
            tree_root_id="root",
            outcome=Outcome.UNRESOLVED,
            candidates=(CandidateResult(candidate=a_node(), evidence=()),),
            review_reason="why",
        )


# -- a rework round (T5.5) --------------------------------------------------------------------

COCOA = "Cocoa costs rose"


async def implied_survivor(
    tree, config, rework: EvidenceGeneration, *, by_chunk=None, script=None, raise_after=None
):  # noqa: PLR0913
    """A search whose one survivor's quote only implies it, ready for a rework round."""
    generator = FakeGenerator(
        [candidates(COCOA)], {COCOA: sources(search("cocoa"))}, rework={COCOA: rework}
    )
    critic = FakeCritic(
        by_chunk={"now-1": Verdict(EvidenceSupport.IMPLIES), **(by_chunk or {})},
        raise_after=raise_after,
    )
    retriever = FakeRetriever(
        {("cocoa", Section.MDNA): found("cocoa", passage("now-1")), **(script or {})}
    )
    ctx = context(tree, config, generator, critic, retriever)
    result = await explore_change(CHANGE, COMPARISON, ctx, RUN)
    (survivor,) = result.candidates
    nodes = await nodes_by_id(tree, result)
    tried = [node for node in nodes.values() if node.parent_id == survivor.candidate.node_id]
    return ctx, generator, critic, survivor.candidate, tried


async def rework_round(ctx, node, tried):
    return await gather_evidence(node, tried, CHANGE, COMPARISON, ctx, prefix=f"{node.node_id}:w1")


async def test_a_rework_round_keeps_a_new_quote_that_states_the_cause(tree, config):
    ctx, generator, _, node, tried = await implied_survivor(
        tree,
        config,
        sources(search("tariff costs")),
        by_chunk={"now-2": Verdict(EvidenceSupport.STATES_CAUSE)},
        script={("tariff costs", Section.MDNA): found("tariff costs", passage("now-2"))},
    )
    found_round = await rework_round(ctx, node, tried)

    (kept,) = found_round.evidence
    assert kept.node_id == f"{node.node_id}:w1:e1"
    assert (kept.parent_id, kept.run_id, kept.depth) == (node.node_id, node.run_id, 2)
    assert kept.status is NodeStatus.KEPT
    assert kept.evidence is EvidenceSupport.STATES_CAUSE
    assert [c.chunk_id for c in kept.citations] == ["now-2"]
    # The critic judged the claim, and the generator was told the search already made.
    assert generator.rework_calls == [(COCOA, ((Section.MDNA, "cocoa"),))]
    assert (found_round.node_evaluations, found_round.model_calls) == (1, 2)
    assert found_round.stopped is None
    assert (await tree.read_node(node.node_id)).status is NodeStatus.KEPT


async def test_a_round_that_finds_nothing_never_prunes_the_kept_candidate(tree, config):
    ctx, _, _, node, tried = await implied_survivor(tree, config, sources(search("tariff costs")))
    found_round = await rework_round(ctx, node, tried)
    assert found_round.evidence == ()
    stored = await tree.read_node(node.node_id)
    assert (stored.status, stored.prune_reason) == (NodeStatus.KEPT, None)
    new = await tree.read_node(f"{node.node_id}:w1:e1")
    assert new.status is NodeStatus.PRUNED
    assert new.prune_reason == "no evidence after one reformulated search of everywhere"
    # The earlier branch is untouched.
    assert [(await tree.read_node(b.node_id)).status for b in tried] == [NodeStatus.KEPT]


async def test_every_new_search_has_room_for_its_reformulated_attempt(tree, config):
    queries = [f"query {n}" for n in range(config.beam.sources)]
    script = {(query, None): found(query, passage(f"wide-{n}")) for n, query in enumerate(queries)}
    by_chunk = {f"wide-{n}": Verdict(EvidenceSupport.NONE) for n in range(len(queries))}
    ctx, _, _, node, tried = await implied_survivor(
        tree, config, sources(*map(search, queries)), by_chunk=by_chunk, script=script
    )
    found_round = await rework_round(ctx, node, tried)
    assert found_round.stopped is None
    assert found_round.node_evaluations == 2 * config.beam.sources
    reasons = {
        (await tree.read_node(f"{node.node_id}:w1:e{n}")).prune_reason
        for n in range(1, len(queries) + 1)
    }
    assert reasons == {"no evidence after one reformulated search of everywhere"}


async def test_the_new_branches_have_their_own_evidence_beam(tree, config):
    narrow = with_beam(config, evidence_width=1)
    ctx, _, _, node, tried = await implied_survivor(
        tree,
        narrow,
        sources(search("tariff costs"), search("freight costs", Section.FINANCIAL_NOTES)),
        by_chunk={
            "now-2": Verdict(EvidenceSupport.STATES_CAUSE),
            "now-3": Verdict(EvidenceSupport.IMPLIES),
        },
        script={
            ("tariff costs", Section.MDNA): found("tariff costs", passage("now-2")),
            ("freight costs", Section.FINANCIAL_NOTES): found("freight costs", passage("now-3")),
        },
    )
    found_round = await rework_round(ctx, node, tried)
    assert [n.node_id for n in found_round.evidence] == [f"{node.node_id}:w1:e1"]
    dropped = await tree.read_node(f"{node.node_id}:w1:e2")
    assert dropped.prune_reason.startswith("outside the beam")
    # The candidate's earlier kept branch still counts as kept beside the new one.
    assert [(await tree.read_node(b.node_id)).status for b in tried] == [NodeStatus.KEPT]


async def test_a_budget_stop_in_a_round_is_reported_not_raised(tree, config):
    # The search makes two critic calls; the round's first call is the third.
    ctx, _, _, node, tried = await implied_survivor(
        tree,
        config,
        sources(search("tariff costs")),
        script={("tariff costs", Section.MDNA): found("tariff costs", passage("now-2"))},
        raise_after=(2, BudgetExceededError("cap reached")),
    )
    found_round = await rework_round(ctx, node, tried)
    assert found_round.stopped is StopReason.RUN_BUDGET
    assert found_round.evidence == ()


async def test_a_generator_failure_ends_the_round_with_a_note(tree, config):
    failed = EvidenceGeneration(failure="the generator failed: refused")
    ctx, _, _, node, tried = await implied_survivor(tree, config, failed)
    found_round = await rework_round(ctx, node, tried)
    assert found_round.evidence == ()
    assert found_round.failures == (f"{node.node_id}: the generator failed: refused",)


async def test_a_round_is_only_for_a_candidate_of_the_change(tree, config):
    ctx, _, _, node, tried = await implied_survivor(tree, config, sources())
    with pytest.raises(ValueError, match="is not a candidate"):
        await rework_round(ctx, tried[0], tried)


def test_tried_searches_leave_out_metrics_and_empty_queries():
    def node(n: int, **fields) -> TreeNode:
        return TreeNode(
            node_id=f"n{n}",
            run_id=RUN,
            change_id="c",
            parent_id="p",
            depth=2,
            thought="t",
            **fields,
        )

    branches = [
        node(1, query_plan="cocoa", source=EvidenceSourceRef(section=Section.MDNA)),
        node(2, query_plan="wide", source=EvidenceSourceRef(section=None)),
        node(
            3,
            source=EvidenceSourceRef(metric=GROSS_MARGIN_DOWN),
            expected_metrics=(GROSS_MARGIN_DOWN,),
        ),
        node(4, query_plan="  ", source=EvidenceSourceRef(section=Section.MDNA)),
    ]
    assert tried_searches(branches) == ((Section.MDNA, "cocoa"), (None, "wide"))


# -- a whole rework round for one claim: search, then relabel (tot/rework.py) ------------------


class QuoteJudge:
    """An entailment judge that finds a quote states the cause only if it names one of `states`."""

    def __init__(self, *states: str) -> None:
        self.states = states
        self.calls: list[tuple[str, str]] = []

    async def judge(self, claim: str, quote: str) -> EntailmentJudgment:
        self.calls.append((claim, quote))
        stated = any(key in quote for key in self.states)
        verdict = EntailmentVerdict.STATES_CAUSE if stated else EntailmentVerdict.IMPLIES
        return EntailmentJudgment(verdict=verdict, reason="scripted")


class PlainWriter:
    async def write(self, request: dict) -> PlausibleWording:
        return PlausibleWording(text="Cocoa prices rose; the filings do not link them.")

    async def restate(self, request: dict) -> Restatement:
        return Restatement(text=None)


class Chunks:
    def __init__(self, *chunks: Chunk) -> None:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)


class NoFacts:
    def find_usd_fact(self, accession: str, value: float, tolerance: float) -> str | None:
        return None


async def test_a_rework_round_confirms_a_claim_its_first_quote_only_implied(tree, config):
    generator = FakeGenerator(
        [candidates(COCOA)],
        {COCOA: sources(search("cocoa"))},
        rework={COCOA: sources(search("tariff costs"))},
    )
    critic = FakeCritic(
        by_chunk={
            "now-1": Verdict(EvidenceSupport.STATES_CAUSE),
            "now-2": Verdict(EvidenceSupport.STATES_CAUSE),
        }
    )
    retriever = FakeRetriever(
        {
            ("cocoa", Section.MDNA): found("cocoa", passage("now-1")),
            ("tariff costs", Section.MDNA): found("tariff costs", passage("now-2")),
        }
    )
    tot = context(tree, config, generator, critic, retriever)
    judge = QuoteJudge("(now-2)")
    labeling_context = LabelingContext(
        judge=judge,
        writer=PlainWriter(),
        chunks=Chunks(passage("now-1").chunk, passage("now-2").chunk),
        metrics=NoMetrics(),
        facts=NoFacts(),
    )
    result = await explore_change(CHANGE, COMPARISON, tot, RUN)
    labeled = await label_result(result, CHANGE, COMPARISON, labeling_context, RUN)
    (claim,) = labeled.finding.claims
    assert claim.label is ExplanationLabel.PLAUSIBLE
    assert [item.claim_id for item in labeled.rework] == [claim.claim_id]

    reworked = await rework_claim(
        labeled,
        claim.claim_id,
        COMPARISON,
        ReworkContext(tot=tot, labeling=labeling_context),
        prefix=rework_prefix(claim.claim_id, 1),
    )

    (confirmed,) = reworked.labeled.finding.claims
    assert confirmed.label is ExplanationLabel.CONFIRMED
    assert [c.chunk_id for c in confirmed.citations] == ["now-2"]
    assert reworked.labeled.rework == ()
    assert (reworked.node_evaluations, reworked.stopped) == (1, None)
    # The generator was told the first search, read back from the tree.
    assert generator.rework_calls == [(COCOA, ((Section.MDNA, "cocoa"),))]
    assert [quote for _, quote in judge.calls] == [
        passage("now-1").chunk.text,
        passage("now-2").chunk.text,
    ]


async def test_a_rework_round_with_nothing_new_keeps_the_claim_and_notes_the_failure(tree, config):
    generator = FakeGenerator(
        [candidates(COCOA)],
        {COCOA: sources(search("cocoa"))},
        rework={COCOA: EvidenceGeneration(failure="the generator failed: refused")},
    )
    critic = FakeCritic(by_chunk={"now-1": Verdict(EvidenceSupport.STATES_CAUSE)})
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    tot = context(tree, config, generator, critic, retriever)
    labeling_context = LabelingContext(
        judge=QuoteJudge(),
        writer=PlainWriter(),
        chunks=Chunks(passage("now-1").chunk),
        metrics=NoMetrics(),
        facts=NoFacts(),
    )
    result = await explore_change(CHANGE, COMPARISON, tot, RUN)
    labeled = await label_result(result, CHANGE, COMPARISON, labeling_context, RUN)
    claim_id = labeled.finding.claims[0].claim_id
    reworked = await rework_claim(
        labeled,
        claim_id,
        COMPARISON,
        ReworkContext(tot=tot, labeling=labeling_context),
        prefix=rework_prefix(claim_id, 1),
    )
    assert reworked.labeled.finding.claims == labeled.finding.claims
    assert reworked.labeled.rework == labeled.rework
    assert reworked.labeled.failures[-1] == f"rework: {claim_id}: the generator failed: refused"


def test_a_resumed_attempt_gets_its_own_rework_node_ids():
    assert rework_prefix("run:c:r1:c2", 1) == "run:c:r1:c2:w1"
    assert (
        rework_prefix("run:c:r1:c2", 2, "r20261113T120000Z") == "run:c:r1:c2:w2.r20261113T120000Z"
    )


class FailingOnceJudge(QuoteJudge):
    """Fails on its first call, then rules as `QuoteJudge` does."""

    async def judge(self, claim: str, quote: str) -> EntailmentJudgment:
        if not self.calls:
            self.calls.append((claim, quote))
            raise LLMError("no parsable output")
        return await super().judge(claim, quote)


async def test_a_round_that_finds_nothing_new_still_retries_a_failed_judge_call(tree, config):
    generator = FakeGenerator(
        [candidates(COCOA)], {COCOA: sources(search("cocoa"))}, rework={COCOA: sources()}
    )
    critic = FakeCritic(by_chunk={"now-1": Verdict(EvidenceSupport.STATES_CAUSE)})
    retriever = FakeRetriever({("cocoa", Section.MDNA): found("cocoa", passage("now-1"))})
    tot = context(tree, config, generator, critic, retriever)
    labeling_context = LabelingContext(
        judge=FailingOnceJudge("(now-1)"),
        writer=PlainWriter(),
        chunks=Chunks(passage("now-1").chunk),
        metrics=NoMetrics(),
        facts=NoFacts(),
    )
    result = await explore_change(CHANGE, COMPARISON, tot, RUN)
    labeled = await label_result(result, CHANGE, COMPARISON, labeling_context, RUN)
    assert labeled.finding.needs_review
    claim_id = labeled.finding.claims[0].claim_id
    reworked = await rework_claim(
        labeled,
        claim_id,
        COMPARISON,
        ReworkContext(tot=tot, labeling=labeling_context),
        prefix=rework_prefix(claim_id, 1),
    )
    assert reworked.labeled.finding.claims[0].label is ExplanationLabel.CONFIRMED
    assert not reworked.labeled.finding.needs_review
