"""Turns a branch's criteria into its score. Plain code: the same inputs always give the same score.

The critic and the metric checker supply the criteria, and nothing here calls a model. The rubric
is in the ToT design and its numbers are in config/scoring.yaml.

Numerical consistency has three states (gap 1). Not applicable removes the criterion and rescales
the rest, because the design as first written had a bug: a text-only explanation, such as a new
risk factor, scored at most 0.25 at depth 1, below the 0.30 threshold, so all of them were pruned.

When it applies, numerical consistency is the share of a branch's checked metrics that agree
(`numeric_consistency`), not all or nothing. The generator sometimes adds a metric that moved the
other way to an otherwise true cause, and with all or nothing one wrong metric set the criterion
to 0, which caps a depth 1 score at 0.25, under the threshold. On Hershey (T6.4, 2026-11-12) that
pruned two stated causes: "lower sales volume" with [revenue down, gross margin down] while revenue
rose, and "higher compensation" with [SG&A up, SG&A ratio up] while the ratio fell. As a share,
each is 0.5. The hard gate on metrics now covers only a contradiction of the change being
explained (see `critic.contradicts_change`); any other contradiction lowers the score (user's
decision, T5.5).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sec_agent.config import REPO_ROOT
from sec_agent.schemas import EvidenceSupport, MetricCheckResult, NumericCheck

SCORING_CONFIG_PATH = REPO_ROOT / "config" / "scoring.yaml"


class Contradiction(StrEnum):
    """Whether a retrieved passage conflicts with a claim."""

    NONE = "none"
    POSSIBLE = "possible"
    DIRECT = "direct"


class Specificity(StrEnum):
    """Whether a claim is testable and names the evidence that would support it."""

    SPECIFIC = "specific"
    PARTIAL = "partial"
    VAGUE = "vague"


class Weights(BaseModel):
    """One set of criterion weights. They sum to 1 before any rescaling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_support: float = Field(ge=0)
    numerical_consistency: float = Field(ge=0)
    no_contradiction: float = Field(ge=0)
    specificity: float = Field(ge=0)

    @model_validator(mode="after")
    def _sums_to_one(self) -> Self:
        total = sum(self.model_dump().values())
        # The tolerance only absorbs floating point error in the sum.
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1, not {total}")
        return self


class Beam(BaseModel):
    """How many branches the controller proposes and keeps at each depth (T4.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: int = Field(ge=1)
    candidate_width: int = Field(ge=1)
    sources: int = Field(ge=1)
    evidence_width: int = Field(ge=1)
    tie_margin: float = Field(ge=0, le=1)


class Budget(BaseModel):
    """Node evaluations one finding may use, per round of candidates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    first_round_nodes: int = Field(ge=1)
    fresh_round_nodes: int = Field(ge=1)


class ScoringConfig(BaseModel):
    """config/scoring.yaml. A level left out is an error, so no level silently scores nothing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float = Field(ge=0, le=1)
    rubric: Weights
    prescreen: Weights
    evidence_support: dict[EvidenceSupport, float]
    no_contradiction: dict[Contradiction, float]
    specificity: dict[Specificity, float]
    beam: Beam
    budget: Budget

    @model_validator(mode="after")
    def _every_level_has_a_value(self) -> Self:
        for levels, kind in (
            (self.evidence_support, EvidenceSupport),
            (self.no_contradiction, Contradiction),
            (self.specificity, Specificity),
        ):
            if set(levels) != set(kind):
                raise ValueError(f"{kind.__name__} needs a value for every level")
            if not all(0 <= value <= 1 for value in levels.values()):
                raise ValueError(f"{kind.__name__} values must be between 0 and 1")
        return self

    @classmethod
    def load(cls, path: Path = SCORING_CONFIG_PATH) -> ScoringConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def numeric_consistency(check: MetricCheckResult) -> float | None:
    """The share of checked metrics that agree, or None if none could be checked.

    Not applicable outcomes (no data, or agreement set aside as free) are left out of the share, so
    one agreeing metric and one with no data score 1.0.
    """
    decided = [
        outcome.check
        for outcome in check.outcomes
        if outcome.check in (NumericCheck.AGREE, NumericCheck.CONTRADICT)
    ]
    if not decided:
        return None
    return decided.count(NumericCheck.AGREE) / len(decided)


def _numeric_value(numeric: NumericCheck | float | None) -> float | None:
    """Numerical consistency on the rubric. Not applicable is None, which removes the criterion.

    A share from `numeric_consistency` is used as it is. A bare verdict counts as all metrics
    agreeing (1.0) or all contradicting (0.0).
    """
    if numeric is None or isinstance(numeric, float):
        return numeric
    return {NumericCheck.AGREE: 1.0, NumericCheck.CONTRADICT: 0.0}.get(numeric)


def _weighted(values: dict[str, float | None], weights: Weights) -> float:
    """The weighted mean of the criteria that apply, so their weights rescale to sum to 1."""
    weight_of = weights.model_dump()
    applicable = {name: value for name, value in values.items() if value is not None}
    total = sum(weight_of[name] for name in applicable)
    score = sum(weight_of[name] * value for name, value in applicable.items()) / total
    # Floating point error can land just above 1, and `TreeNode.score` refuses that.
    return min(score, 1.0)


def prescreen_score(
    check: NumericCheck | float | None, specificity: Specificity, config: ScoringConfig
) -> float:
    """A depth 1 candidate's score: numerical consistency and specificity only.

    With numerical consistency not applicable, specificity is all that remains, so its weight
    rescales from 0.25 to 1. This is the text-only case.
    """
    return _weighted(
        {
            "numerical_consistency": _numeric_value(check),
            "specificity": config.specificity[specificity],
        },
        config.prescreen,
    )


def branch_score(
    check: NumericCheck | float | None,
    evidence: EvidenceSupport,
    contradiction: Contradiction,
    specificity: Specificity,
    config: ScoringConfig,
) -> float:
    """A depth 2 evidence branch's score on the full rubric."""
    return _weighted(
        {
            "evidence_support": config.evidence_support[evidence],
            "numerical_consistency": _numeric_value(check),
            "no_contradiction": config.no_contradiction[contradiction],
            "specificity": config.specificity[specificity],
        },
        config.rubric,
    )


def prune_reason(
    score: float,
    config: ScoringConfig,
    *,
    contradicts_change: bool = False,
    only_wrong_period: bool = False,
    rating_withdrawn: bool = False,
) -> str | None:
    """Why a scored branch must be pruned, or None if it may stay.

    The hard gates come first and apply regardless of score. `only_wrong_period` means dropping
    the branch's wrong-period citations left it with no evidence. This softens the ToT design,
    which prunes on any wrong-period citation, so one careless pick cannot cost a good branch (the
    user's decision, 2026-11-06). `rating_withdrawn` means quotes were dropped and the critic found
    the verified ones left do not support the claim. `contradicts_change` means the branch expects
    the changed metric, or one that moves with it, to move the other way from how it moved.
    What is kept in the beam, ties and the retry for missing evidence are the controller's (T4.4).
    """
    if only_wrong_period:
        return "hard gate: its only cited evidence is from the wrong period"
    if rating_withdrawn:
        return "no evidence: the quotes left after dropping failed ones do not support the claim"
    if contradicts_change:
        return "hard gate: computed metrics contradict the change being explained"
    if score < config.threshold:
        return f"score {score:.2f} is below the {config.threshold:.2f} threshold"
    return None
