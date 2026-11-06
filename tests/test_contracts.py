from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest
from pydantic import ValidationError

from sec_agent.contracts import ChunkLookup, FactLookup, MetricChecker, MetricLookup, Retriever
from sec_agent.schemas import (
    Change,
    ChangeKind,
    Chunk,
    Citation,
    Comparison,
    Direction,
    EvidenceSourceRef,
    EvidenceSupport,
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

# The two build lanes meet at these records and interfaces, so these tests pin down the rules
# each side can rely on: what counts as incomplete, what a verdict must cite, and that the
# new tree node field is optional so earlier saved trees still load.
PERIOD = FiscalPeriod(fiscal_year=2025, fiscal_period="Q2", period_end=date(2025, 6, 29))


def passage(chunk_id: str = "c1", score: float = 4.2) -> RetrievedPassage:
    return RetrievedPassage(
        chunk=Chunk(
            chunk_id=chunk_id,
            paragraph_id="p1",
            ticker="HSY",
            cik="1",
            form=FormType.FORM_10Q,
            period=PERIOD,
            section=Section.MDNA,
            accession="a1",
            text="Freight costs rose.",
            token_count=4,
        ),
        score=score,
    )


def test_retrieval_result_is_incomplete_unless_both_periods_have_passages():
    both = RetrievalResult(query="freight", current=(passage("c1"),), prior=(passage("c2"),))
    assert not both.incomplete
    assert RetrievalResult(query="freight", current=(passage(),)).incomplete
    assert RetrievalResult(query="freight", prior=(passage(),)).incomplete
    assert RetrievalResult(query="freight").incomplete


def test_retrieval_result_needs_a_query():
    with pytest.raises(ValidationError):
        RetrievalResult(query="")


def test_metric_expectations_must_name_catalog_metrics():
    # Strings from saved JSON are accepted, but a misspelled metric is refused.
    expectation = MetricExpectation(metric="gross_margin", direction="down")
    assert expectation.metric is MetricName.GROSS_MARGIN
    assert expectation.direction is Direction.DOWN
    with pytest.raises(ValidationError):
        MetricExpectation(metric="gross margin", direction=Direction.DOWN)
    with pytest.raises(ValidationError):
        MetricExpectation(metric=MetricName.GROSS_MARGIN, direction="sideways")


def outcome(metric: MetricName, check: NumericCheck, *metric_ids: str) -> MetricOutcome:
    return MetricOutcome(metric=metric, check=check, metric_ids=metric_ids)


GROSS = MetricName.GROSS_MARGIN
COGS = MetricName.COGS_RATIO


def test_agree_and_contradict_outcomes_must_cite_metrics():
    for check in (NumericCheck.AGREE, NumericCheck.CONTRADICT):
        with pytest.raises(ValidationError, match="cite"):
            MetricOutcome(metric=GROSS, check=check)
        assert outcome(GROSS, check, "m1").metric_ids == ("m1",)
    # Not applicable rests on no metric, so it needs none.
    assert outcome(GROSS, NumericCheck.NOT_APPLICABLE).metric_ids == ()


def test_a_contradicting_metric_outweighs_agreeing_ones_in_any_order():
    # One metric moved as expected and the other the opposite way: the claim is contradicted.
    agree = outcome(GROSS, NumericCheck.AGREE, "m1")
    contradict = outcome(COGS, NumericCheck.CONTRADICT, "m2")
    assert MetricCheckResult(outcomes=(agree, contradict)).check is NumericCheck.CONTRADICT
    assert MetricCheckResult(outcomes=(contradict, agree)).check is NumericCheck.CONTRADICT
    # Unchecked metrics change nothing: missing data is not contradicting evidence.
    unchecked = outcome(MetricName.NET_MARGIN, NumericCheck.NOT_APPLICABLE)
    assert MetricCheckResult(outcomes=(agree, unchecked)).check is NumericCheck.AGREE
    assert MetricCheckResult(outcomes=(unchecked, contradict)).check is NumericCheck.CONTRADICT


def test_overall_check_without_any_checked_metric_is_not_applicable():
    unchecked = outcome(GROSS, NumericCheck.NOT_APPLICABLE)
    assert MetricCheckResult().check is NumericCheck.NOT_APPLICABLE
    assert MetricCheckResult(outcomes=(unchecked,)).check is NumericCheck.NOT_APPLICABLE
    agree = outcome(GROSS, NumericCheck.AGREE, "m1")
    assert MetricCheckResult(outcomes=(agree, agree)).check is NumericCheck.AGREE


def test_result_lists_each_cited_metric_once_in_order():
    result = MetricCheckResult(
        outcomes=(
            outcome(GROSS, NumericCheck.AGREE, "m1", "m2"),
            outcome(COGS, NumericCheck.AGREE, "m2", "m3"),
        )
    )
    assert result.metric_ids == ("m1", "m2", "m3")


def test_new_records_round_trip_through_json():
    records = [
        passage(),
        RetrievalResult(query="freight", current=(passage("c1"),), prior=(passage("c2"),)),
        MetricExpectation(metric=MetricName.COGS_RATIO, direction=Direction.UP),
        outcome(GROSS, NumericCheck.AGREE, "m1", "m2"),
        MetricCheckResult(
            outcomes=(
                outcome(GROSS, NumericCheck.AGREE, "m1"),
                outcome(COGS, NumericCheck.CONTRADICT, "m2"),
            )
        ),
    ]
    for record in records:
        assert type(record).model_validate_json(record.model_dump_json()) == record


def change(change_id: str, rank: int | None, comparison_id: str = "cmp1") -> Change:
    return Change(
        change_id=change_id,
        comparison_id=comparison_id,
        kind=ChangeKind.NUMERIC,
        metric_id="m1",
        materiality_rank=rank,
    )


def test_ranked_changes_accept_a_complete_ranking_and_an_empty_list():
    ranked = RankedChanges(comparison_id="cmp1", changes=(change("a", 1), change("b", 2)))
    assert [c.change_id for c in ranked.changes] == ["a", "b"]
    assert RankedChanges(comparison_id="cmp1").changes == ()
    assert RankedChanges.model_validate_json(ranked.model_dump_json()) == ranked


@pytest.mark.parametrize(
    "ranks",
    [
        pytest.param((1, None), id="missing rank"),
        pytest.param((1, 1), id="repeated rank"),
        pytest.param((1, 3), id="skipped rank"),
        pytest.param((2, 3), id="does not start at 1"),
        pytest.param((2, 1), id="out of order"),
    ],
)
def test_ranked_changes_refuse_a_missing_repeated_skipped_or_misordered_rank(ranks):
    changes = tuple(change(f"c{i}", rank) for i, rank in enumerate(ranks))
    with pytest.raises(ValidationError, match="ranked 1 to n"):
        RankedChanges(comparison_id="cmp1", changes=changes)


def test_ranked_changes_refuse_a_change_from_another_comparison():
    with pytest.raises(ValidationError, match="comparison"):
        RankedChanges(
            comparison_id="cmp1", changes=(change("a", 1), change("b", 2, comparison_id="cmp2"))
        )


def test_tree_node_carries_expected_metrics_and_best_passage_score():
    node = TreeNode(
        node_id="n1",
        run_id="r1",
        change_id="ch1",
        parent_id="n0",
        depth=1,
        thought="Input costs rose, so gross margin fell.",
        expected_metrics=(
            MetricExpectation(metric=MetricName.GROSS_MARGIN, direction=Direction.DOWN),
            MetricExpectation(metric=MetricName.COGS_RATIO, direction=Direction.UP),
        ),
        best_passage_score=3.7,
        status=NodeStatus.KEPT,
    )
    assert TreeNode.model_validate_json(node.model_dump_json()) == node
    # A tree saved before these fields existed has no such keys and must still load: the claim
    # is treated as text-only and no passage score is known.
    saved = node.model_dump(mode="json")
    del saved["expected_metrics"], saved["best_passage_score"]
    loaded = TreeNode.model_validate(saved)
    assert loaded.expected_metrics == ()
    assert loaded.best_passage_score is None


GROSS_MARGIN_DOWN = MetricExpectation(metric=MetricName.GROSS_MARGIN, direction=Direction.DOWN)


def evidence_node(**fields) -> TreeNode:
    return TreeNode(
        **{
            "node_id": "e1",
            "run_id": "r1",
            "change_id": "ch1",
            "parent_id": "n1",
            "depth": 2,
            "thought": "Freight costs, per MD&A",
            **fields,
        }
    )


def test_tree_node_holds_what_a_resumed_run_needs():
    citation = Citation(
        chunk_id="c1",
        accession="a1",
        ticker="HSY",
        form=FormType.FORM_10Q,
        period=PERIOD,
        section=Section.MDNA,
        quote="Freight costs rose.",
        url="https://example.com/a1",
    )
    node = evidence_node(
        source=EvidenceSourceRef(section=Section.MDNA),
        evidence=EvidenceSupport.STATES_CAUSE,
        citations=(citation,),
        best_passage_score=3.7,
    )
    assert TreeNode.model_validate_json(node.model_dump_json()) == node
    # A tree saved before these fields existed must still load: no source, no evaluation, no
    # quotes. `None` is "everywhere", so a search with no section is a source, not a missing one.
    saved = node.model_dump(mode="json")
    del saved["source"], saved["evidence"], saved["citations"]
    loaded = TreeNode.model_validate(saved)
    assert (loaded.source, loaded.evidence, loaded.citations) == (None, None, ())
    everywhere = evidence_node(source=EvidenceSourceRef())
    assert everywhere.source is not None and everywhere.source.section is None


def test_a_metric_source_must_agree_with_the_nodes_expected_metrics():
    source = EvidenceSourceRef(metric=GROSS_MARGIN_DOWN)
    assert evidence_node(source=source, expected_metrics=(GROSS_MARGIN_DOWN,)).source == source
    # Without the check a metric source and a search under a one-metric parent would be
    # indistinguishable, which is why `source` exists.
    with pytest.raises(ValidationError, match="that one metric"):
        evidence_node(source=source)
    other = MetricExpectation(metric=MetricName.COGS_RATIO, direction=Direction.UP)
    with pytest.raises(ValidationError, match="that one metric"):
        evidence_node(source=source, expected_metrics=(other,))


def test_a_source_is_for_depth_two_nodes_and_is_a_search_or_a_metric_not_both():
    with pytest.raises(ValidationError, match="only a depth 2 node"):
        TreeNode(
            node_id="n1",
            run_id="r1",
            change_id="ch1",
            parent_id="n0",
            depth=1,
            thought="Input costs rose",
            source=EvidenceSourceRef(section=Section.MDNA),
        )
    with pytest.raises(ValidationError, match="names no section"):
        EvidenceSourceRef(section=Section.MDNA, metric=GROSS_MARGIN_DOWN)


def test_evidence_support_is_ordered_strongest_first_as_the_tie_rule_reads_it():
    assert list(EvidenceSupport) == [
        EvidenceSupport.STATES_CAUSE,
        EvidenceSupport.IMPLIES,
        EvidenceSupport.CONSISTENT,
        EvidenceSupport.NONE,
    ]


# Minimal stand-ins for the real implementations. They show the shape each lane builds and
# fakes; a lane that needs richer behavior writes its own fake beside its own tests.
class FakeRetriever:
    async def retrieve(
        self, query: str, comparison: Comparison, section: Section | None = None
    ) -> RetrievalResult:
        return RetrievalResult(query=query)


class FakeMetricChecker:
    def check(
        self, comparison: Comparison, expectations: Sequence[MetricExpectation]
    ) -> MetricCheckResult:
        return MetricCheckResult()


class FakeChunks:
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return None


class FakeMetrics:
    def get_metric(self, metric_id: str) -> MetricValue | None:
        return None


class FakeFacts:
    def find_usd_fact(self, accession: str, value: float, tolerance: float) -> str | None:
        return None


def test_stand_ins_satisfy_the_interfaces_and_other_objects_do_not():
    assert isinstance(FakeRetriever(), Retriever)
    assert isinstance(FakeMetricChecker(), MetricChecker)
    assert isinstance(FakeChunks(), ChunkLookup)
    assert isinstance(FakeMetrics(), MetricLookup)
    assert isinstance(FakeFacts(), FactLookup)
    # The check is by method name, so an object with none of them is refused.
    for interface in (Retriever, MetricChecker, ChunkLookup, MetricLookup, FactLookup):
        assert not isinstance(object(), interface)
