"""The rework loop (T5.5): more evidence for a claim the judge found only implied, then relabel it.

The submitted plans say failed verification returns a claim for rework, at most two retries per
claim, and after that it goes on without pausing the run (master plan gap 11; architecture plan
"verification retry loop"; design refinement "loops back for more evidence until the retry
limit"). Here a retry is one round (user's decision, 2026-11-13, option 1):

1. `controller.gather_evidence` asks the generator for new searches, told every search the
   candidate's branches already made, and runs them through the critic and the evidence beam as
   new branches under the same candidate.
2. `labeling.relabel_claim` judges the round's new quotes, and again any the judge failed on.

The claims sent back are those `label_result` lists in `LabeledFinding.rework`. A claim still
unconfirmed after `MAX_REWORK_ROUNDS` keeps its plausible label: it already has evidence and
wording that passed the checks, and the brief shows it without a cause. Only a judge failure still
standing sends the finding to review, as it did before rework. The graph runs one round per pass
and decides the budget stops (`graph.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

from sec_agent.schemas import Comparison, Record
from sec_agent.tot.controller import StopReason, ToTContext, gather_evidence
from sec_agent.tot.labeling import LabeledFinding, LabelingContext, relabel_claim

# Rounds per finding, each giving every claim still listed one more search. The plans' "at most 2
# retries per claim".
MAX_REWORK_ROUNDS = 2


class ReworkResult(Record):
    """A finding after one claim's round, and what the round used."""

    labeled: LabeledFinding
    node_evaluations: int = 0
    stopped: StopReason | None = None


@dataclass(frozen=True)
class ReworkContext:
    """What a round depends on: the search's context and labeling's."""

    tot: ToTContext
    labeling: LabelingContext


def rework_prefix(claim_id: str, round_number: int, attempt: str | None = None) -> str:
    """The node id prefix for a round's new branches under the candidate `claim_id`.

    `attempt` names a resumed attempt (`graph.attempt_id`), so a round cut off by a crash can run
    again: the tree store refuses a node id twice.
    """
    prefix = f"{claim_id}:w{round_number}"
    return f"{prefix}.{attempt}" if attempt else prefix


async def rework_claim(
    labeled: LabeledFinding,
    claim_id: str,
    comparison: Comparison,
    context: ReworkContext,
    *,
    prefix: str,
) -> ReworkResult:
    """One round for one claim of `labeled`: search again, then relabel on what was kept.

    The candidate and its earlier branches are read from the finding's tree. `BudgetExceededError`
    from labeling propagates, so the graph can keep the finding as it was before the round.
    """
    finding = labeled.finding
    if finding.tree_root_id is None:
        raise ValueError(f"finding {finding.finding_id!r} has no tree to rework")
    nodes = await context.tot.tree.read_tree(finding.tree_root_id)
    candidate = next((node for node in nodes if node.node_id == claim_id), None)
    if candidate is None:
        raise ValueError(f"the tree has no candidate {claim_id!r}")
    tried = [node for node in nodes if node.depth == 2 and node.parent_id == claim_id]
    found = await gather_evidence(
        candidate, tried, finding.change, comparison, context.tot, prefix=prefix
    )
    failures = tuple(f"rework: {failure}" for failure in found.failures)
    relabeled = labeled.model_copy(update={"failures": (*labeled.failures, *failures)})
    # Relabeled even when nothing new was kept, so a quote the judge failed on still gets its
    # second try; with neither, nothing is judged.
    relabeled = await relabel_claim(
        relabeled, claim_id, candidate, found.evidence, comparison, context.labeling
    )
    return ReworkResult(
        labeled=relabeled, node_evaluations=found.node_evaluations, stopped=found.stopped
    )
