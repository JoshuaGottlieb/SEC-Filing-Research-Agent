"""Interfaces between the parts that are built in parallel.

The retrieval and change-detection code (lane 1) and the Tree-of-Thought and verification
code (lane 2) meet at these four points. Each lane tests against fakes of the other side's
interface, so the fakes must match what the real code will return. The records they pass
are in schemas.py.

Ranked changes are not a call, so they have no interface here: lane 1's materiality step
(T3.3) returns a `RankedChanges` record and the controller receives it through the graph
state (T5.5). Its validator enforces the ranking. One rule it cannot enforce: a numeric
change's `metric_id` should resolve through `MetricLookup`.

These are structural interfaces: a class satisfies one by having the method, with no base
class to import. A runtime isinstance check only confirms the method name exists, not its
signature, so the type checker and the tests carry the rest.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from sec_agent.schemas import (
    Chunk,
    Comparison,
    MetricCheckResult,
    MetricExpectation,
    MetricValue,
    RetrievalResult,
    Section,
)


@runtime_checkable
class Retriever(Protocol):
    """Finds passages for a query in both of a comparison's filings.

    Built by lane 1 (the search tools, T2.5, and the Retrieval Agent, T3.4); called by the
    Tree-of-Thought controller. It is async because the Retrieval Agent calls the LLM to
    reformulate a weak query; a plain search tool can be wrapped.

    The search is limited to the comparison's two filings, so the caller cannot forget the
    company and period filters. It returns the best passages per period, and an empty side
    means that period had no match above the score cutoff. It does not raise for that:
    the caller reads `RetrievalResult.incomplete`.
    """

    async def retrieve(
        self, query: str, comparison: Comparison, section: Section | None = None
    ) -> RetrievalResult: ...


@runtime_checkable
class MetricChecker(Protocol):
    """Tests whether computed metrics agree with the movements a claim implies.

    Built by lane 1 (the metric catalog, T1.6); called by the critic's scoring (T4.3).

    It returns one outcome per expectation, in order. An expectation that moved the opposite
    way is CONTRADICT, one that moved as expected is AGREE, and one with no data in either
    period is NOT_APPLICABLE: missing evidence is not contradicting evidence. AGREE and
    CONTRADICT outcomes cite the metrics they rest on. The overall verdict is
    `MetricCheckResult.check`: any CONTRADICT outcome makes it CONTRADICT, so a claim with one
    metric moving as expected and another the opposite way is contradicted. No expectations
    gives NOT_APPLICABLE.

    An AGREE or CONTRADICT outcome cites the values from both periods, and each resolves through
    `MetricLookup`, so a caller can compute the change. A ratio move under about 5 basis points
    (chosen by hand) is too small to call, so it is NOT_APPLICABLE. A NOT_APPLICABLE outcome may
    cite nothing, and callers must not rely on what it cites.

    `revenue_growth` differs on both counts. Its outcome cites its one value, and its direction
    is about revenue, not about growth: UP means revenue rose (the value is above 0) and DOWN
    means revenue fell. "Growth slowed to 2%" must not be written as `revenue_growth` DOWN:
    revenue still rose, so this returns CONTRADICT. Such a claim carries no expectation for
    `revenue_growth`.
    """

    def check(
        self, comparison: Comparison, expectations: Sequence[MetricExpectation]
    ) -> MetricCheckResult: ...


@runtime_checkable
class ChunkLookup(Protocol):
    """Fetches a stored passage by id, or None if there is no such chunk.

    Built by lane 1 (the chunk store, T1.3 and T2.1); used by the citation checker (T5.1) to
    confirm a quote appears in the chunk it cites and that the chunk is from the right period.
    """

    def get_chunk(self, chunk_id: str) -> Chunk | None: ...


@runtime_checkable
class MetricLookup(Protocol):
    """Fetches a computed metric by id, or None if there is no such metric.

    Built by lane 1 (the metric catalog, T1.6); used by the citation checker (T5.1) to trace
    a numeric claim to the metric it cites and compare the claim's figure with its value.

    What a `MetricValue` holds:

    - `unit` is `ratio` or `USD`. A `ratio` is an unrounded, signed fraction: 0.37512 is 37.512%,
      so rounding for comparison is the caller's job. `USD` is whole dollars with any table scale
      already applied. All six current metrics are ratios, so `USD` does not appear yet, and a
      dollar figure cannot trace through this interface.
    - `name` is a `MetricName` value.
    - `metric_id` is opaque: do not parse it. Pair two values by `name`, `period` and `duration`.
    - `duration` is 3m for a 10-Q and 12m for a 10-K. Compare only values of the same duration.

    A change between two values is the caller's to compute. A relative change needs a positive
    prior value: a net margin can be negative or near zero, where "a 5.1% fall" means nothing.

    `revenue_growth` is never paired. It is one signed value (0.044 is 4.4%) dated to the current
    period, with both periods' revenue in `input_fact_ids`. Its value already describes a change,
    so the only figure it supports is that value, with no basis-point or relative change from it.
    """

    def get_metric(self, metric_id: str) -> MetricValue | None: ...
