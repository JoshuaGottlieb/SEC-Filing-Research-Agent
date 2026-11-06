"""Scoring (T4.3): plain code, so these need no model. Hand arithmetic is in the comments."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sec_agent.schemas import NumericCheck, TreeNode
from sec_agent.tot.scoring import (
    Contradiction,
    EvidenceSupport,
    ScoringConfig,
    Specificity,
    Weights,
    branch_score,
    prescreen_score,
    prune_reason,
)

AGREE = NumericCheck.AGREE
CONTRADICT = NumericCheck.CONTRADICT
NOT_APPLICABLE = NumericCheck.NOT_APPLICABLE


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    return ScoringConfig.load()


def test_the_config_holds_the_designs_numbers(config):
    assert config.threshold == 0.30
    assert config.rubric == Weights(
        evidence_support=0.40, numerical_consistency=0.30, no_contradiction=0.20, specificity=0.10
    )
    assert config.prescreen.numerical_consistency == 0.75
    assert config.prescreen.specificity == 0.25


def test_a_text_only_candidate_is_not_pruned_at_depth_1(config):
    # Gap 1, the bug this design fixes. With numerical consistency scored as 0, a text-only claim
    # could reach at most 0.25, below the threshold, so every one of them would be pruned.
    score = prescreen_score(NOT_APPLICABLE, Specificity.SPECIFIC, config)
    assert score == pytest.approx(1.0)
    assert prune_reason(score, NOT_APPLICABLE, config) is None


def test_a_text_only_candidate_is_judged_on_specificity_alone(config):
    # Its weight rescales from 0.25 to 1, so the score is the level's value.
    assert prescreen_score(NOT_APPLICABLE, Specificity.PARTIAL, config) == pytest.approx(0.5)
    vague = prescreen_score(NOT_APPLICABLE, Specificity.VAGUE, config)
    assert vague == 0.0
    assert "below" in prune_reason(vague, NOT_APPLICABLE, config)


@pytest.mark.parametrize(
    ("check", "specificity", "expected"),
    [
        (AGREE, Specificity.SPECIFIC, 1.0),  # 0.75 + 0.25
        (AGREE, Specificity.PARTIAL, 0.875),  # 0.75 + 0.25 * 0.5
        (AGREE, Specificity.VAGUE, 0.75),  # 0.75 + 0
        (CONTRADICT, Specificity.SPECIFIC, 0.25),  # 0 + 0.25
        (CONTRADICT, Specificity.VAGUE, 0.0),
    ],
)
def test_prescreen_scores(config, check, specificity, expected):
    assert prescreen_score(check, specificity, config) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("check", "evidence", "contradiction", "specificity", "expected"),
    [
        (
            AGREE,
            EvidenceSupport.STATES_CAUSE,
            Contradiction.NONE,
            Specificity.SPECIFIC,
            1.0,  # 0.40 + 0.30 + 0.20 + 0.10
        ),
        (
            AGREE,
            EvidenceSupport.IMPLIES,
            Contradiction.NONE,
            Specificity.SPECIFIC,
            0.88,  # 0.40 * 0.7 + 0.30 + 0.20 + 0.10
        ),
        (
            AGREE,
            EvidenceSupport.CONSISTENT,
            Contradiction.POSSIBLE,
            Specificity.PARTIAL,
            0.61,  # 0.40 * 0.4 + 0.30 + 0.20 * 0.5 + 0.10 * 0.5
        ),
        (
            CONTRADICT,
            EvidenceSupport.STATES_CAUSE,
            Contradiction.NONE,
            Specificity.SPECIFIC,
            0.70,  # 0.40 + 0 + 0.20 + 0.10
        ),
        (
            AGREE,
            EvidenceSupport.NONE,
            Contradiction.DIRECT,
            Specificity.VAGUE,
            0.30,  # 0 + 0.30 + 0 + 0
        ),
    ],
)
def test_full_rubric_scores(config, check, evidence, contradiction, specificity, expected):
    score = branch_score(check, evidence, contradiction, specificity, config)
    assert score == pytest.approx(expected)


@pytest.mark.parametrize(
    ("evidence", "contradiction", "specificity", "expected"),
    [
        # The weights rescale over the 0.70 that remains.
        (EvidenceSupport.STATES_CAUSE, Contradiction.NONE, Specificity.SPECIFIC, 1.0),
        # (0.40 * 0.7 + 0.20 + 0.10) / 0.70
        (EvidenceSupport.IMPLIES, Contradiction.NONE, Specificity.SPECIFIC, 0.58 / 0.70),
        # (0.40 * 0.4 + 0.20 * 0.5 + 0.10 * 0.5) / 0.70
        (EvidenceSupport.CONSISTENT, Contradiction.POSSIBLE, Specificity.PARTIAL, 0.31 / 0.70),
        # (0.20 + 0.10) / 0.70: no evidence still clears the threshold, see the next test.
        (EvidenceSupport.NONE, Contradiction.NONE, Specificity.SPECIFIC, 0.30 / 0.70),
        (EvidenceSupport.NONE, Contradiction.DIRECT, Specificity.VAGUE, 0.0),
    ],
)
def test_a_text_only_branch_rescales_the_full_rubric(
    config, evidence, contradiction, specificity, expected
):
    score = branch_score(NOT_APPLICABLE, evidence, contradiction, specificity, config)
    assert score == pytest.approx(expected)


def test_a_text_only_branch_can_reach_the_top_of_the_scale(config):
    # Without rescaling, its best would be 0.70, so it would always lose to a numeric claim.
    score = branch_score(
        NOT_APPLICABLE,
        EvidenceSupport.STATES_CAUSE,
        Contradiction.NONE,
        Specificity.SPECIFIC,
        config,
    )
    assert score == pytest.approx(1.0)


def test_the_score_alone_does_not_catch_a_branch_with_no_evidence(config):
    # Recorded so nobody relies on the threshold for this: 0.43 clears 0.30. The controller uses
    # `EvidenceEvaluation.has_evidence` to give it one retry and then prune it.
    score = branch_score(
        NOT_APPLICABLE, EvidenceSupport.NONE, Contradiction.NONE, Specificity.SPECIFIC, config
    )
    assert prune_reason(score, NOT_APPLICABLE, config) is None


def test_every_score_fits_a_tree_node(config):
    # Floating point error can land just above 1, and a TreeNode refuses a score above 1.
    for check in NumericCheck:
        for evidence in EvidenceSupport:
            for contradiction in Contradiction:
                for specificity in Specificity:
                    score = branch_score(check, evidence, contradiction, specificity, config)
                    TreeNode(
                        node_id="n",
                        run_id="r",
                        change_id="c",
                        parent_id="p",
                        depth=2,
                        thought="t",
                        score=score,
                    )


def test_a_numeric_contradiction_is_pruned_regardless_of_score(config):
    score = branch_score(
        CONTRADICT, EvidenceSupport.STATES_CAUSE, Contradiction.NONE, Specificity.SPECIFIC, config
    )
    assert score > config.threshold
    assert "metrics contradict" in prune_reason(score, CONTRADICT, config)


def test_a_branch_left_with_only_wrong_period_evidence_is_pruned_regardless_of_score(config):
    assert "wrong period" in prune_reason(1.0, NOT_APPLICABLE, config, only_wrong_period=True)


def test_a_branch_at_the_threshold_stays(config):
    # The design prunes branches "below" 0.30.
    assert prune_reason(0.30, AGREE, config) is None
    assert "below" in prune_reason(0.29, AGREE, config)


def test_prune_reasons_are_what_a_tree_node_requires(config):
    # A pruned TreeNode must record why, so a reason must never be empty.
    assert prune_reason(0.0, AGREE, config)


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1"):
        Weights(
            evidence_support=0.5, numerical_consistency=0.5, no_contradiction=0.5, specificity=0.5
        )


def _raw(config: ScoringConfig) -> dict:
    return config.model_dump(mode="json")


def test_a_level_left_out_of_the_config_is_an_error(config):
    raw = _raw(config)
    del raw["evidence_support"]["implies"]
    with pytest.raises(ValidationError, match="every level"):
        ScoringConfig.model_validate(raw)


def test_a_level_value_outside_0_to_1_is_an_error(config):
    raw = _raw(config)
    raw["specificity"]["partial"] = 1.5
    with pytest.raises(ValidationError, match="between 0 and 1"):
        ScoringConfig.model_validate(raw)


def test_an_unknown_config_field_is_an_error(config):
    with pytest.raises(ValidationError):
        ScoringConfig.model_validate({**_raw(config), "beam": 4})
