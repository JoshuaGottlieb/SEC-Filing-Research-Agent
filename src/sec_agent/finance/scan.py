"""The metric scan (T3.2): numeric changes between the two filings of a comparison.

The catalog's metrics are computed for both periods, and a move large enough to matter becomes a
`Change(kind=NUMERIC)` citing the current period's metric, which `MetricLookup` resolves (the
checker pairs it with the prior period by name). Three kinds of move are flagged:

- a margin (gross, operating, net) that moved by 100 basis points or more;
- revenue that grew or fell by 10% or more (`revenue_growth`);
- SG&A or cost of goods sold, in dollars, that rose or fell by 10% or more.

The cost-of-goods-sold and SG&A ratios are not scanned on their own: the cost ratio is one minus
the gross margin, and the dollar moves cover the expenses, so scanning them too would raise the
same finding twice, and every change costs a materiality call and a Tree-of-Thought run.

Consecutive quarters are seasonal for some companies (Mattel sells most toys late in the year),
so for a sequential comparison each move is set beside the same move a year earlier: Q1 to Q2 of
the year before, read through `MetricCatalog.year_earlier`. A similar move then marks the change
as possibly seasonal. Without both year-earlier figures the check is refused, and the change says
so (`seasonality_checked` False) rather than calling the move "not seasonal".
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from sec_agent.finance.metrics import MetricCatalog, PastValue
from sec_agent.schemas import (
    Change,
    ChangeKind,
    Comparison,
    ComparisonType,
    MetricName,
    Record,
)

# The plan's initial thresholds (gap 6 of the master plan), kept by the user on 2026-11-12 after
# the Hershey benchmark labels. Every Hershey margin move is far from 100 bps (the smallest is
# 240 bps), so the labels cannot move that one. At 10% the stated-cause gross-margin moves are
# flagged and no labeled material change depends on a smaller relative move: net sales +4.4%
# (labeled material) arrives as a text change. Lower cutoffs would add revenue +6.6% and SG&A
# +7.7% at 5%, and revenue +4.4% and SG&A +3.7% at 3%, none of them a labeled material change, as
# numeric changes, which are always material.
MIN_MARGIN_MOVE = 0.01  # 100 basis points, as a fraction
MIN_RELATIVE_MOVE = 0.10  # 10%
# How much of a move the year-earlier move must match, in the same direction, to explain it as
# seasonal: at least half. Chosen by hand and kept by the user on 2026-11-10 and again on
# 2026-11-12: Hershey's shares are 0.67, 1.04 or opposite in sign, so its labels cannot move it.
# On the pinned filings it leaves Mattel's Q1-to-Q2 gross margin (+331 bps, against +158 bps a
# year earlier: a share of 0.48) just short of seasonal; Mattel is held out for testing.
MIN_SEASONAL_SHARE = 0.5

MARGINS = (MetricName.GROSS_MARGIN, MetricName.OPERATING_MARGIN, MetricName.NET_MARGIN)
EXPENSES = (MetricName.SGA, MetricName.COST_OF_GOODS_SOLD)


class MoveKind(StrEnum):
    """How a move is measured."""

    # The difference of two ratios: 0.01 is 100 basis points.
    POINTS = "points"
    # The relative change of a positive figure: 0.10 is 10%.
    RELATIVE = "relative"


class SeasonalStatus(StrEnum):
    """What the seasonal check found for one move."""

    # Not a sequential comparison: both periods are the same quarter or year, so it doesn't apply.
    NOT_SEQUENTIAL = "not_sequential"
    # The move itself couldn't be measured, so there is nothing to explain. Never flagged.
    NO_MOVE = "no_move"
    # The year-earlier figures aren't there, so the check was refused.
    NO_YEAR_EARLIER_DATA = "no_year_earlier_data"
    SIMILAR_MOVE = "similar_move"
    NO_SIMILAR_MOVE = "no_similar_move"


class MetricMove(Record):
    """One scanned metric: how far it moved, whether that is flagged, and the seasonal check."""

    metric: MetricName
    kind: MoveKind
    threshold: float = Field(ge=0)
    # None when the move can't be measured: a period has no data, or a relative change would
    # start from zero or a negative figure.
    move: float | None
    flagged: bool
    # The current period's metric, which a flagged move's change cites.
    metric_id: str | None = None
    seasonal: SeasonalStatus
    year_earlier_move: float | None = None
    # The facts behind the year-earlier move, so the seasonal label can be traced.
    year_earlier_fact_ids: tuple[str, ...] = ()


class MetricScan(Record):
    """Every scanned move of a comparison, and the numeric changes made from the flagged ones."""

    comparison_id: str
    moves: tuple[MetricMove, ...]
    changes: tuple[Change, ...]


class ScanThresholds(Record):
    """The cutoffs, in one place so a tuning run can try others."""

    margin_points: float = Field(default=MIN_MARGIN_MOVE, ge=0)
    relative: float = Field(default=MIN_RELATIVE_MOVE, ge=0)
    seasonal_share: float = Field(default=MIN_SEASONAL_SHARE, ge=0, le=1)


def relative_change(current: float, prior: float) -> float | None:
    """The change as a fraction of a positive starting figure, or None when it has none."""
    return current / prior - 1 if prior > 0 else None


def is_similar(move: float, year_earlier: float, share: float) -> bool:
    """True when the year-earlier move went the same way and was at least `share` as large."""
    return move * year_earlier > 0 and abs(year_earlier) >= share * abs(move)


def _year_earlier_move(
    catalog: MetricCatalog, comparison: Comparison, metric: MetricName, kind: MoveKind
) -> tuple[float | None, tuple[str, ...]]:
    # The same step a year earlier: the year-earlier figures for the prior and current quarters.
    level = MetricName.REVENUE if metric is MetricName.REVENUE_GROWTH else metric
    before: PastValue | None = catalog.year_earlier(comparison.prior, level)
    after: PastValue | None = catalog.year_earlier(comparison.current, level)
    if before is None or after is None:
        return None, ()
    facts = tuple(dict.fromkeys((*before.input_fact_ids, *after.input_fact_ids)))
    if kind is MoveKind.POINTS:
        return after.value - before.value, facts
    return relative_change(after.value, before.value), facts


def _move(
    catalog: MetricCatalog, comparison: Comparison, metric: MetricName
) -> tuple[MoveKind, float | None, str | None]:
    """How far a metric moved between the comparison's periods, and the current metric's id."""
    if metric is MetricName.REVENUE_GROWTH:
        growth = catalog.growth(comparison.current, comparison.prior)
        if growth is None:
            return MoveKind.RELATIVE, None, None
        return MoveKind.RELATIVE, growth.value, growth.metric_id
    current = catalog.level(comparison.current, metric)
    prior = catalog.level(comparison.prior, metric)
    kind = MoveKind.POINTS if metric in MARGINS else MoveKind.RELATIVE
    if current is None or prior is None:
        return kind, None, None
    if kind is MoveKind.POINTS:
        return kind, current.value - prior.value, current.metric_id
    return kind, relative_change(current.value, prior.value), current.metric_id


def scan_metric(
    catalog: MetricCatalog,
    comparison: Comparison,
    metric: MetricName,
    thresholds: ScanThresholds,
) -> MetricMove:
    """Measure one metric's move, flag it if large enough, and run the seasonal check."""
    kind, move, metric_id = _move(catalog, comparison, metric)
    threshold = thresholds.margin_points if kind is MoveKind.POINTS else thresholds.relative
    flagged = move is not None and abs(move) >= threshold
    if comparison.comparison_type is not ComparisonType.SEQ_10Q:
        return MetricMove(
            metric=metric,
            kind=kind,
            threshold=threshold,
            move=move,
            flagged=flagged,
            metric_id=metric_id,
            seasonal=SeasonalStatus.NOT_SEQUENTIAL,
        )
    year_earlier, facts = _year_earlier_move(catalog, comparison, metric, kind)
    if move is None:
        seasonal = SeasonalStatus.NO_MOVE
    elif year_earlier is None:
        seasonal = SeasonalStatus.NO_YEAR_EARLIER_DATA
    elif is_similar(move, year_earlier, thresholds.seasonal_share):
        seasonal = SeasonalStatus.SIMILAR_MOVE
    else:
        seasonal = SeasonalStatus.NO_SIMILAR_MOVE
    return MetricMove(
        metric=metric,
        kind=kind,
        threshold=threshold,
        move=move,
        flagged=flagged,
        metric_id=metric_id,
        seasonal=seasonal,
        year_earlier_move=year_earlier,
        year_earlier_fact_ids=facts,
    )


def _change(comparison: Comparison, move: MetricMove) -> Change:
    checked = move.seasonal in (SeasonalStatus.SIMILAR_MOVE, SeasonalStatus.NO_SIMILAR_MOVE)
    return Change(
        change_id=f"{comparison.comparison_id}:numeric:{move.metric.value}",
        comparison_id=comparison.comparison_id,
        kind=ChangeKind.NUMERIC,
        metric_id=move.metric_id,
        seasonality_checked=checked,
        possibly_seasonal=move.seasonal is SeasonalStatus.SIMILAR_MOVE,
    )


def scan_comparison(
    catalog: MetricCatalog,
    comparison: Comparison,
    thresholds: ScanThresholds | None = None,
) -> MetricScan:
    """Scan a comparison's margins, revenue growth and major expenses for large moves."""
    thresholds = thresholds or ScanThresholds()
    metrics = (*MARGINS, MetricName.REVENUE_GROWTH, *EXPENSES)
    moves = tuple(scan_metric(catalog, comparison, metric, thresholds) for metric in metrics)
    return MetricScan(
        comparison_id=comparison.comparison_id,
        moves=moves,
        changes=tuple(_change(comparison, move) for move in moves if move.flagged),
    )
