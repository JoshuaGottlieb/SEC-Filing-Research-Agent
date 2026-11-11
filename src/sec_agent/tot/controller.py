"""The Tree-of-Thought controller: beam, pruning, ties, retry and stop rules, all in plain code.

The submitted plan commits to this being code and never a model's decision. The generator proposes,
the critic rates, `scoring.py` turns ratings into scores, and this module decides what to keep, what
to prune and when to stop. It follows the ToT design:

- Depth 1 proposes candidate explanations and pre-screens them; the best few go on.
- Depth 2 proposes evidence sources for each kept candidate, tests each with retrieved passages and
  the metric checker, and keeps the best few per candidate.
- A branch with no evidence gets one reformulated search before it is pruned, because missing
  evidence is not contradicting evidence.
- If every candidate is pruned, one fresh round of candidates is tried. If that also fails the item
  is unresolved, for human review.

It returns what survived and how, not a `Finding`. A finding's claims need a label, and a confirmed
label needs the entailment judge (T4.5, T5.2), so the labeling step turns this result into claims.
Every node, score, status and prune reason is written to the tree store as it happens, pruned
branches included, so the audit trail does not depend on the result.

Search and evaluation run one node at a time. That keeps the budget count and every tie
deterministic, and the cost is latency, to be measured on a real finding at integration.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import model_validator

from sec_agent.contracts import MetricChecker, MetricLookup, Retriever
from sec_agent.llm.ledger import BudgetExceededError, CostLedger
from sec_agent.schemas import (
    Change,
    ChangeKind,
    Comparison,
    EvidenceSourceRef,
    EvidenceSupport,
    FiscalPeriod,
    MetricName,
    NodeStatus,
    NumericCheck,
    RankedChanges,
    Record,
    RetrievedPassage,
    Section,
    TreeNode,
)
from sec_agent.tot.critic import (
    CandidateEvaluation,
    Critic,
    EvidenceEvaluation,
    evaluate_candidate,
    evaluate_evidence,
    linked_metrics,
)
from sec_agent.tot.generator import ThoughtGenerator
from sec_agent.tot.scoring import ScoringConfig
from sec_agent.tree_mcp.client import TreeClient
from sec_agent.tree_mcp.store import BudgetUse, NodeUpdate

# The reformulated search is one search of everywhere (`section=None`), and nothing else. Measured
# on the real retriever (HSY FY2025 against FY2024, 2026-11-06, one filing pair and a handful of
# queries written by hand):
#
# - Everywhere is the only search that reaches a table filed as `other`. It ranked the one such
#   chunk first, and no named-section search returned it.
# - A retry that stopped at the first named section with any passage almost never got past its first
#   section: only 8 of 30 named-section searches came back empty on six realistic queries.
# - A fallback to the named sections after an empty everywhere never fired: everywhere returned 8 to
#   10 passages on every realistic query, and on the one off-topic query where it was empty, every
#   named search was empty too. So it was removed rather than kept as dead weight.
#
# POSSIBLY BRING BACK named-section searching if coverage needs it. Everywhere keeps only the best
# five passages per filing across all sections, and on eight queries it missed 98 of the 176
# passages the five named searches kept. Most scored below the weakest passage it did keep (only 8
# above), but a weaker passage can still state a cause, and that was not measured. If real
# findings show evidence lost to crowding, the fix is not a fallback for an empty search, which
# never triggers. It is to search the named sections when the critic finds no evidence among
# everywhere's passages, at the price of more critic calls and node evaluations per branch, so the
# retry budget would need to change. A search took 2 to 8 s (two filings on CPU), and five more per
# branch is 20 to 40 s. The order tried before, by hand and unmeasured, was MD&A (where management
# says why results moved), financial notes (detail behind the numbers), risk factors, business,
# then market risk. `OTHER` would not be in it: only everywhere reaches those tables.

# Strongest first, the order of the enum (a test pins it). A branch with no level sorts last.
_EVIDENCE_RANK = {level: rank for rank, level in enumerate(EvidenceSupport)}


class Outcome(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class StopReason(StrEnum):
    # The finding used its node evaluations, or the run reached its dollar cap. Either way the
    # search ended with the best branches so far.
    NODE_BUDGET = "node_budget"
    RUN_BUDGET = "run_budget"


class CandidateResult(Record):
    """A surviving explanation and the evidence branches that kept it alive, best first."""

    candidate: TreeNode
    evidence: tuple[TreeNode, ...]


class ToTResult(Record):
    """What the search found for one change."""

    change_id: str
    # None only for a change the run never reached.
    tree_root_id: str | None
    outcome: Outcome
    # Survivors, best first. Each has at least one kept evidence branch.
    candidates: tuple[CandidateResult, ...] = ()
    # Node ids of survivors that each have a kept branch whose evidence states the cause. This is
    # only the search's half of the confirmed standard: labeling still needs the judge, and may
    # narrow it. Empty unless there are at least two, because one is just the answer.
    co_drivers: tuple[str, ...] = ()
    stopped: StopReason | None = None
    # Why the change goes to human review. Set exactly when the outcome is unresolved.
    review_reason: str | None = None
    # Branches that ended because a model failed, such as a refusal. They are not judgments.
    failures: tuple[str, ...] = ()
    node_evaluations: int = 0
    # Critic and generator calls, counting a reassessment as one more.
    model_calls: int = 0
    fresh_round: bool = False

    @model_validator(mode="after")
    def _outcome_matches_what_it_holds(self) -> ToTResult:
        if self.outcome is Outcome.RESOLVED:
            if not self.candidates:
                raise ValueError("a resolved result needs at least one surviving candidate")
            if self.review_reason is not None:
                raise ValueError("a resolved result does not go to review")
        elif self.candidates or not self.review_reason:
            raise ValueError("an unresolved result has no survivors and says why")
        return self


@dataclass(frozen=True)
class ToTContext:
    """Everything the search depends on, passed in so the graph wiring (T5.5) is mechanical."""

    retriever: Retriever
    checker: MetricChecker
    metrics: MetricLookup
    generator: ThoughtGenerator
    critic: Critic
    tree: TreeClient
    config: ScoringConfig


def select_beam[T](
    items: Sequence[T],
    width: int,
    margin: float,
    score: Callable[[T], float],
    tie_key: Callable[[T], tuple],
) -> tuple[list[T], list[T], bool]:
    """Choose the top `width` by score, with the design's tie rule at the cutoff.

    Returns the kept items, the dropped ones and whether the tie rule decided anything. Branches
    within `margin` of the cutoff score compete for the slots left by those clearly above it, and
    `tie_key` orders them (lower is better). Anything still equal keeps its original order, so a
    tie never depends on chance. Items are returned best score first.
    """
    ranked = sorted(items, key=lambda item: -score(item))
    if len(ranked) <= width:
        return ranked, [], False
    cutoff = score(ranked[width - 1])
    if cutoff - score(ranked[width]) > margin:
        return ranked[:width], ranked[width:], False
    clear = [item for item in ranked if score(item) > cutoff + margin]
    contenders = [item for item in ranked if abs(score(item) - cutoff) <= margin]
    below = [item for item in ranked if score(item) < cutoff - margin]
    # sorted() is stable, so a full tie keeps the generator's order.
    ordered = sorted(contenders, key=tie_key)
    slots = width - len(clear)
    kept = sorted(clear + ordered[:slots], key=lambda item: -score(item))
    return kept, ordered[slots:] + below, True


class _NodeCap(Exception):
    """Raised inside a round when the node budget is used up. Never leaves this module."""


@dataclass(eq=False)
class _Branch:
    """A depth 2 node and what the search has learned about it so far."""

    node: TreeNode
    # The candidate with this source's metrics: what the critic reads. The depth 2 node's own
    # thought is the source's description, not the claim being tested.
    claim: TreeNode
    evaluation: EvidenceEvaluation | None = None
    passages: tuple[RetrievedPassage, ...] = ()
    waiting_for_retry: bool = False
    retried: bool = False
    decided: bool = False
    eligible: bool = False

    @property
    def score(self) -> float:
        assert self.evaluation is not None and self.evaluation.score is not None
        return self.evaluation.score

    @property
    def best_passage_score(self) -> float | None:
        if self.evaluation is None:
            return None
        cited = {citation.chunk_id for citation in self.evaluation.citations}
        scores = [p.score for p in self.passages if p.chunk.chunk_id in cited]
        return max(scores) if scores else None

    def tie_key(self) -> tuple[int, float]:
        assert self.evaluation is not None
        passage_score = self.best_passage_score
        # Direct evidence first, then the better supporting passage. No passage sorts last.
        return (
            _EVIDENCE_RANK[self.evaluation.evidence],
            -passage_score if passage_score is not None else math.inf,
        )


@dataclass(eq=False)
class _Candidate:
    """A depth 1 node and its branches."""

    node: TreeNode
    evaluation: CandidateEvaluation | None = None
    eligible: bool = False
    in_beam: bool = False
    # Set when the generator failed to propose sources for it.
    failure: str | None = None
    branches: list[_Branch] = field(default_factory=list)
    kept: list[_Branch] = field(default_factory=list)

    @property
    def score(self) -> float:
        assert self.evaluation is not None and self.evaluation.score is not None
        return self.evaluation.score


def _describe(change: Change) -> str:
    """The root node's thought: what was flagged, in a line."""
    where = f" in {change.section.value}" if change.section else ""
    what = change.rationale or change.current_text or change.prior_text or change.metric_id or ""
    return f"{change.kind.value} change{where}: {what}".strip(" :")[:500]


def claim_periods(change: Change, comparison: Comparison) -> frozenset[FiscalPeriod]:
    """The period a claim about `change` is about: the earlier one only for something removed.

    Public because labeling (T4.5) must pass `check_citations` the same periods the search used.
    """
    period = (
        comparison.prior.period if change.kind is ChangeKind.REMOVED else comparison.current.period
    )
    return frozenset({period})


def changed_metric(change: Change, metrics: MetricLookup) -> MetricName | None:
    """The metric a numeric change is about, or None for a text change or an unknown metric.

    `metric_id` is opaque (the contract says not to parse it), so the name comes from the lookup.
    Agreeing with this metric earns a candidate nothing: see `critic.without_free_agreement`.
    """
    if change.metric_id is None:
        return None
    metric = metrics.get_metric(change.metric_id)
    if metric is None:
        return None
    try:
        return MetricName(metric.name)
    except ValueError:
        return None


def claim_node(candidate: TreeNode, evidence: TreeNode) -> TreeNode:
    """The node the critic reads for an evidence branch: the claim, with the branch's metrics.

    `evaluate_evidence` and `check_citations` take the claim from `node.thought`. A depth 2 node's
    thought is the description of its source, not the claim, so passing the stored evidence node
    would have the critic judge "look for cocoa costs". This copies the depth 1 `candidate`, whose
    thought is the claim, and swaps in the branch's `expected_metrics` (one metric for a metric
    source, the candidate's own for a search). The controller builds every branch's node with it,
    and anything that rebuilds a branch from the stored tree, such as a resumed run or a retry, must
    too. Nothing is stored for it: the tree already holds both nodes.
    """
    if candidate.depth != 1:
        raise ValueError(f"the claim is on a depth 1 node, not depth {candidate.depth}")
    if evidence.depth != 2 or evidence.parent_id != candidate.node_id:
        raise ValueError(
            f"node {evidence.node_id!r} is not an evidence branch of {candidate.node_id!r}"
        )
    return candidate.model_copy(update={"expected_metrics": evidence.expected_metrics})


def claim_of(node: TreeNode, tree: Sequence[TreeNode]) -> str:
    """The claim a node is or tests, for showing it: a candidate's own, an evidence node's parent's.

    `tree` is what `TreeClient.read_tree` returns. The root is the change that was flagged and no
    claim, so it raises, and so does a branch whose candidate is missing from `tree`: showing a
    source description in a claim's place is the mistake this exists to prevent.
    """
    if node.depth == 1:
        return node.thought
    if node.depth == 2:
        parent = next((other for other in tree if other.node_id == node.parent_id), None)
        if parent is None or parent.depth != 1:
            raise ValueError(f"the tree has no candidate {node.parent_id!r} for {node.node_id!r}")
        return parent.thought
    raise ValueError("the root is the change that was flagged, not a claim")


class _Exploration:
    """One change's search. Its state lives only as long as the search does."""

    def __init__(
        self, change: Change, comparison: Comparison, context: ToTContext, run_id: str
    ) -> None:
        self._change = change
        self._comparison = comparison
        self._context = context
        self._config = context.config
        self._prefix = f"{run_id}:{change.change_id}"
        self._root_id = f"{self._prefix}:root"
        self._run_id = run_id
        self._periods = claim_periods(change, comparison)
        self._changed_metric = changed_metric(change, context.metrics)
        # Every claim proposed so far, so the fresh round does not repeat them.
        self._claims: list[str] = []
        self._failures: list[str] = []
        self._stopped: StopReason | None = None
        self._generation_failed = False
        self._node_evaluations = 0
        self._model_calls = 0
        self._used = 0
        self._limit = 0

    # -- the search ---------------------------------------------------------------------------

    async def run(self) -> ToTResult:
        await self._context.tree.create_tree(
            TreeNode(
                node_id=self._root_id,
                run_id=self._run_id,
                change_id=self._change.change_id,
                depth=0,
                thought=_describe(self._change),
            )
        )
        survivors = await self._round(1, self._config.budget.first_round_nodes)
        fresh_round = False
        # A refusal or a spent dollar budget ends the search: another round would meet the same
        # wall.
        if (
            not survivors
            and not self._generation_failed
            and self._stopped is not StopReason.RUN_BUDGET
        ):
            fresh_round = True
            survivors = await self._round(2, self._config.budget.fresh_round_nodes)
        return await self._result(survivors, fresh_round)

    async def _round(self, number: int, limit: int) -> list[_Candidate]:
        self._used, self._limit = 0, limit
        self._generation_failed = False
        candidates: list[_Candidate] = []
        try:
            await self._propose(f"{self._prefix}:r{number}", candidates)
            await self._prescreen(candidates)
            beam = await self._choose_candidates(candidates)
            for candidate in beam:
                await self._expand(candidate)
            for candidate in beam:
                for branch in candidate.branches:
                    if branch.waiting_for_retry:
                        await self._retry(branch)
        except _NodeCap:
            self._stop(StopReason.NODE_BUDGET)
        except BudgetExceededError:
            self._stop(StopReason.RUN_BUDGET)
        return await self._finalize(candidates)

    def _stop(self, reason: StopReason) -> None:
        # A spent dollar budget outranks a spent node budget: it also stops the other findings.
        if self._stopped is not StopReason.RUN_BUDGET:
            self._stopped = reason

    def _has_room(self) -> bool:
        return self._used < self._limit

    def _take(self) -> None:
        if not self._has_room():
            raise _NodeCap
        self._used += 1
        self._node_evaluations += 1

    # -- depth 1 ------------------------------------------------------------------------------

    async def _propose(self, prefix: str, candidates: list[_Candidate]) -> None:
        generation = await self._context.generator.propose_candidates(
            self._change,
            self._comparison,
            self._context.metrics,
            count=self._config.beam.candidates,
            already_tried=tuple(self._claims),
        )
        self._model_calls += 1
        if generation.failure:
            self._generation_failed = True
            self._failures.append(generation.failure)
            return
        for number, proposal in enumerate(generation.candidates, start=1):
            node = TreeNode(
                node_id=f"{prefix}:c{number}",
                run_id=self._run_id,
                change_id=self._change.change_id,
                parent_id=self._root_id,
                depth=1,
                thought=proposal.claim,
                expected_evidence=proposal.expected_evidence,
                query_plan=proposal.query_plan,
                expected_metrics=proposal.expected_metrics,
            )
            await self._context.tree.create_node(node)
            self._claims.append(proposal.claim)
            candidates.append(_Candidate(node))

    async def _prescreen(self, candidates: list[_Candidate]) -> None:
        for candidate in candidates:
            self._take()
            result = await evaluate_candidate(
                candidate.node,
                self._comparison,
                self._context.critic,
                self._context.checker,
                self._config,
                changed_metric=self._changed_metric,
            )
            candidate.evaluation = result
            # A contradicted candidate is pruned before the critic is paid.
            if result.specificity is not None or result.failure is not None:
                self._model_calls += 1
            fields = {
                "score": result.score,
                "metric_ids": result.metric_ids,
                "numeric_check": result.numeric_check,
            }
            if result.failure:
                await self._close(candidate.node, f"unresolved: {result.failure}", **fields)
                self._failures.append(f"{candidate.node.node_id}: {result.failure}")
            elif result.prune_reason:
                await self._close(candidate.node, result.prune_reason, **fields)
            else:
                candidate.eligible = True
                await self._update(candidate.node, **fields)

    async def _choose_candidates(self, candidates: list[_Candidate]) -> list[_Candidate]:
        eligible = [candidate for candidate in candidates if candidate.eligible]
        beam = self._config.beam
        # Nothing has been retrieved at depth 1, so the tie rule has no evidence level or passage
        # score to compare. A tie there falls back to the generator's order.
        kept, dropped, _ = select_beam(
            eligible, beam.candidate_width, beam.tie_margin, lambda c: c.score, lambda c: ()
        )
        for candidate in kept:
            candidate.in_beam = True
            await self._update(candidate.node, status=NodeStatus.KEPT)
        for candidate in dropped:
            await self._close(
                candidate.node,
                f"outside the beam: scored {candidate.score:.2f}, and only the top "
                f"{beam.candidate_width} candidates go on to evidence",
            )
        return kept

    # -- depth 2 ------------------------------------------------------------------------------

    async def _expand(self, candidate: _Candidate) -> None:
        proposal = await self._context.generator.propose_sources(
            candidate.node,
            self._change,
            self._comparison,
            self._context.metrics,
            count=self._config.beam.sources,
        )
        self._model_calls += 1
        if proposal.failure:
            candidate.failure = proposal.failure
            self._failures.append(f"{candidate.node.node_id}: {proposal.failure}")
            return
        for number, source in enumerate(proposal.sources, start=1):
            metric = source.metric
            expected = (metric,) if metric is not None else candidate.node.expected_metrics
            node = TreeNode(
                node_id=f"{candidate.node.node_id}:e{number}",
                run_id=self._run_id,
                change_id=self._change.change_id,
                parent_id=candidate.node.node_id,
                depth=2,
                thought=source.description,
                query_plan=source.query,
                expected_metrics=expected,
                source=EvidenceSourceRef(
                    section=None if metric is not None else source.section, metric=metric
                ),
            )
            await self._context.tree.create_node(node)
            candidate.branches.append(_Branch(node=node, claim=claim_node(candidate.node, node)))
        for branch in candidate.branches:
            await self._evaluate(branch)

    async def _passages(
        self, branch: _Branch, section: Section | None
    ) -> tuple[RetrievedPassage, ...]:
        """What a search of `section` returns for the branch's query. A metric has no search."""
        query = branch.node.query_plan
        if branch.node.source.metric is not None or not query or not query.strip():
            return ()
        result = await self._context.retriever.retrieve(query, self._comparison, section)
        return (*result.current, *result.prior)

    async def _judge(self, branch: _Branch, passages: tuple[RetrievedPassage, ...]) -> None:
        evaluation = await evaluate_evidence(
            branch.claim,
            passages,
            self._comparison,
            self._periods,
            self._context.critic,
            self._context.checker,
            self._config,
            changed_metric=self._changed_metric,
        )
        # Set together and only after the call returns, so a stop in the middle of a retry leaves
        # the first evaluation and the passages it was made from, not a mixture.
        branch.passages, branch.evaluation = passages, evaluation
        self._model_calls += 1 + evaluation.reassessed

    async def _evaluate(self, branch: _Branch) -> None:
        self._take()
        await self._judge(branch, await self._passages(branch, branch.node.source.section))
        evaluation = branch.evaluation
        assert evaluation is not None
        # A metric has nothing to search again, a refusal is not missing evidence, a contradiction
        # prunes whatever else is found, and a search that already covered everywhere can only
        # return the same passages again. The rest get one reformulated search, of everywhere.
        branch.waiting_for_retry = (
            not evaluation.has_evidence
            and evaluation.failure is None
            and evaluation.numeric_check is not NumericCheck.CONTRADICT
            and branch.node.source.metric is None
            and branch.node.source.section is not None
            and bool((branch.node.query_plan or "").strip())
        )
        if not branch.waiting_for_retry:
            await self._decide(branch)

    async def _retry(self, branch: _Branch) -> None:
        """The one reformulated attempt: the branch's query, searched everywhere.

        Everywhere is the only search that reaches a table outside the named sections, and on real
        filings it returned passages for every realistic query (see the note at the top of this
        module, which also says when to bring named sections back). It is judged only if it returns
        a passage from the claim's period, since anything else cannot support the claim. The search
        is not counted against the budget, only the evaluation that follows.
        """
        if not self._has_room():
            raise _NodeCap
        found = await self._passages(branch, None)
        if any(passage.chunk.period in self._periods for passage in found):
            self._take()
            await self._judge(branch, found)
        # Set only now, so a stop part way through leaves it waiting and `_finalize` prunes it.
        branch.waiting_for_retry = False
        branch.retried = True
        await self._decide(branch)

    async def _decide(self, branch: _Branch) -> None:
        """Prune the branch or mark it eligible for the beam, and record what was learned."""
        evaluation = branch.evaluation
        assert evaluation is not None
        branch.decided = True
        if evaluation.failure:
            reason = f"unresolved: {evaluation.failure}"
            self._failures.append(f"{branch.node.node_id}: {evaluation.failure}")
        elif branch.retried and not evaluation.has_evidence:
            reason = "no evidence after one reformulated search of everywhere"
        elif evaluation.prune_reason:
            reason = evaluation.prune_reason
        elif not evaluation.has_evidence:
            reason = self._no_evidence_reason(branch)
        else:
            branch.eligible = True
            reason = None
        await self._record(branch, reason)

    def _no_evidence_reason(self, branch: _Branch) -> str:
        metric = branch.node.source.metric
        if (
            metric is not None
            and self._changed_metric is not None
            and metric.metric in linked_metrics(self._changed_metric)
        ):
            # Agreement with it is set aside as free (`critic.without_free_agreement`), so whether
            # or not it had data, this source could never support the claim.
            return (
                "no evidence: the metric is the change being explained, or moves with it by "
                "arithmetic, so it cannot support the claim"
            )
        if metric is not None:
            return "no evidence: the metric has no data to compare in either period"
        if not (branch.node.query_plan or "").strip():
            return "no evidence: the source has no query to search"
        return "no evidence: its search already covered every section, so a retry would repeat it"

    async def _record(self, branch: _Branch, reason: str | None) -> None:
        evaluation = branch.evaluation
        assert evaluation is not None
        fields = {
            "score": evaluation.score,
            "retrieved_chunk_ids": tuple(dict.fromkeys(p.chunk.chunk_id for p in branch.passages)),
            "best_passage_score": branch.best_passage_score,
            "evidence": evaluation.evidence,
            "citations": evaluation.citations,
            "metric_ids": evaluation.metric_ids,
            "numeric_check": evaluation.numeric_check,
        }
        if reason is None:
            await self._update(branch.node, **fields)
        else:
            await self._close(branch.node, reason, **fields)

    # -- finishing a round --------------------------------------------------------------------

    async def _finalize(self, candidates: list[_Candidate]) -> list[_Candidate]:
        """Apply the evidence beam, prune what lost, and return the candidates that survived."""
        beam = self._config.beam
        survivors: list[_Candidate] = []
        for candidate in candidates:
            if not candidate.in_beam:
                continue
            for branch in candidate.branches:
                if branch.waiting_for_retry:
                    # The budget ended before its reformulated search, so it is pruned for lack of
                    # evidence, which says so honestly in the audit trail.
                    branch.waiting_for_retry = False
                    branch.decided = True
                    await self._record(
                        branch, "no evidence, and the search ended before its reformulated attempt"
                    )
            eligible = [branch for branch in candidate.branches if branch.eligible]
            kept, dropped, _ = select_beam(
                eligible,
                beam.evidence_width,
                beam.tie_margin,
                lambda b: b.score,
                lambda b: b.tie_key(),
            )
            for branch in kept:
                await self._update(branch.node, status=NodeStatus.KEPT)
            for branch in dropped:
                await self._close(
                    branch.node,
                    f"outside the beam: scored {branch.score:.2f}, and only the top "
                    f"{beam.evidence_width} evidence branches per candidate stay",
                )
            candidate.kept = kept
            if kept:
                survivors.append(candidate)
            elif all(branch.decided for branch in candidate.branches):
                if candidate.failure:
                    reason = f"unresolved: {candidate.failure}"
                elif candidate.branches:
                    reason = "no evidence branch survived"
                else:
                    reason = "no evidence source could be proposed for it"
                await self._close(candidate.node, reason)
        survivors.sort(key=lambda c: -c.kept[0].score)
        return survivors

    async def _result(self, survivors: list[_Candidate], fresh_round: bool) -> ToTResult:
        common = {
            "change_id": self._change.change_id,
            "tree_root_id": self._root_id,
            "stopped": self._stopped,
            "failures": tuple(self._failures),
            "node_evaluations": self._node_evaluations,
            "model_calls": self._model_calls,
            "fresh_round": fresh_round,
        }
        if not survivors:
            return ToTResult(
                outcome=Outcome.UNRESOLVED, review_reason=self._unresolved_reason(), **common
            )
        await self._update_root(NodeStatus.KEPT)
        tree = self._context.tree
        results = []
        for candidate in survivors:
            results.append(
                CandidateResult(
                    candidate=await tree.read_node(candidate.node.node_id),
                    evidence=tuple([await tree.read_node(b.node.node_id) for b in candidate.kept]),
                )
            )
        states_cause = tuple(
            result.candidate.node_id
            for result in results
            if any(node.evidence is EvidenceSupport.STATES_CAUSE for node in result.evidence)
        )
        return ToTResult(
            outcome=Outcome.RESOLVED,
            candidates=tuple(results),
            co_drivers=states_cause if len(states_cause) >= 2 else (),
            **common,
        )

    def _unresolved_reason(self) -> str:
        if self._generation_failed:
            return f"a model failed before any explanation could be tested: {self._failures[-1]}"
        if self._stopped is StopReason.RUN_BUDGET:
            return "the run's dollar budget was reached before any explanation had evidence"
        if self._stopped is StopReason.NODE_BUDGET:
            return "the finding's node budget was used up before any explanation had evidence"
        return "every candidate explanation was pruned, in the first round and in the fresh one"

    # -- the tree store -----------------------------------------------------------------------

    async def _update(self, node: TreeNode, **fields) -> None:
        await self._context.tree.update_node(node.node_id, NodeUpdate(**fields))

    async def _close(self, node: TreeNode, reason: str, **fields) -> None:
        await self._update(node, status=NodeStatus.PRUNED, prune_reason=reason, **fields)

    async def _update_root(self, status: NodeStatus) -> None:
        await self._context.tree.update_node(self._root_id, NodeUpdate(status=status))


async def explore_change(
    change: Change, comparison: Comparison, context: ToTContext, run_id: str
) -> ToTResult:
    """Search for explanations of one change and return the survivors.

    Node ids include `run_id` and the change, because the tree store keeps them unique across the
    whole file. A model failure ends only the branch it hit. `BudgetExceededError` is caught here:
    the search ends with what survived so far, and `stopped` says the run's dollar cap was the
    reason, so the caller can stop and flag the changes it did not reach.
    """
    return await _Exploration(change, comparison, context, run_id).run()


async def explore_changes(
    ranked: RankedChanges,
    comparison: Comparison,
    context: ToTContext,
    ledger: CostLedger,
    run_id: str,
    *,
    starting_evaluations: int = 0,
) -> tuple[ToTResult, ...]:
    """Search each change in order of materiality, until the run's dollar budget is reached.

    The changes not reached come back unresolved with no tree, for human review, and are the least
    material ones. The budget snapshot in the tree store is rewritten after each change.
    `starting_evaluations` carries a resumed run's count forward, as a new ledger starts at 0.
    """
    results: list[ToTResult] = []
    total = starting_evaluations
    out_of_budget = False
    for change in ranked.changes:
        if out_of_budget:
            results.append(
                ToTResult(
                    change_id=change.change_id,
                    tree_root_id=None,
                    outcome=Outcome.UNRESOLVED,
                    stopped=StopReason.RUN_BUDGET,
                    review_reason="the run's dollar budget was reached before this change was "
                    "examined",
                )
            )
            continue
        result = await explore_change(change, comparison, context, run_id)
        results.append(result)
        total += result.node_evaluations
        await context.tree.record_budget(
            BudgetUse(
                run_id=run_id,
                spent_usd=ledger.spent_usd,
                max_usd=ledger.max_usd,
                node_evaluations=total,
            )
        )
        out_of_budget = result.stopped is StopReason.RUN_BUDGET
    return tuple(results)
