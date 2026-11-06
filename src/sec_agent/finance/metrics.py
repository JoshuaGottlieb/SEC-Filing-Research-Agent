"""The metric catalog: six figures computed from company-wide XBRL facts, and the check on them.

`MetricCatalog` is the `MetricChecker` and `MetricLookup` of contracts.py, whose docstrings define
what a value means. Every value is computed when asked for rather than stored, so it can't drift
from the facts, and a metric id holds all that is needed to compute it again.

Each period's figures come from that period's own filing. A later filing can restate an earlier
period, so a comparison shows what each filing reported, and a value traces to one filing.

Missing data is never filled in. A line item a filing doesn't report, or reports twice with
different values, gives no metric, and a check on it is NOT_APPLICABLE.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sec_agent.config import REPO_ROOT
from sec_agent.edgar.corpus import Corpus, CorpusError
from sec_agent.finance.fact_store import FactStore
from sec_agent.schemas import (
    Comparison,
    Direction,
    Duration,
    Filing,
    FormType,
    MetricCheckResult,
    MetricExpectation,
    MetricName,
    MetricOutcome,
    MetricValue,
    NumericCheck,
)

METRICS_CONFIG_PATH = REPO_ROOT / "config" / "metrics.yaml"
# The unit names the contract fixes. Every current metric is a ratio.
UNIT_RATIO = "ratio"
# How xbrl_facts writes US dollars. A per-share figure has a different unit and is never used.
XBRL_UNIT_USD = "usd"
# More rows than one company reports for one period end, so that FactStore.search's limit can't
# silently drop the row a metric needs and make it look missing.
FACT_SEARCH_LIMIT = 100_000

REVENUE = "revenue"
GROSS_PROFIT = "gross_profit"
COST_OF_GOODS_SOLD = "cost_of_goods_sold"
SGA = "sga"
OPERATING_INCOME = "operating_income"
NET_INCOME = "net_income"
# The line item each ratio divides by revenue. Revenue growth is computed on its own.
RATIO_NUMERATORS = {
    MetricName.GROSS_MARGIN: GROSS_PROFIT,
    MetricName.OPERATING_MARGIN: OPERATING_INCOME,
    MetricName.NET_MARGIN: NET_INCOME,
    MetricName.COGS_RATIO: COST_OF_GOODS_SOLD,
    MetricName.SGA_RATIO: SGA,
}


class NoMoveTolerance(BaseModel):
    """How small a move is too small to call, as a fraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ratio: float = Field(ge=0)
    revenue_growth: float = Field(ge=0)


class MetricsConfig(BaseModel):
    """config/metrics.yaml: which concepts mean each line item, and the no-move tolerances."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Line item to concepts, most specific first.
    line_items: dict[str, tuple[str, ...]]
    no_move_tolerance: NoMoveTolerance

    @classmethod
    def load(cls, path: Path = METRICS_CONFIG_PATH) -> MetricsConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class _ConflictingFigures(Exception):
    """A filing reports one line item with two different values, so neither can be used.

    An exception rather than None, because None means "not reported", and gross profit that isn't
    reported gets computed. A conflict must never be replaced by a computed figure.
    """


@dataclass(frozen=True)
class _Amount:
    """A line item's dollar value and the facts it rests on."""

    value: float
    fact_ids: tuple[str, ...]


def duration_for(form: FormType) -> Duration:
    """A 10-Q's metrics use its quarter and a 10-K's its year, never a year-to-date figure."""
    return Duration.TWELVE_MONTHS if form is FormType.FORM_10K else Duration.THREE_MONTHS


def metric_id_for(filing: Filing, name: MetricName, compared_to: Filing | None = None) -> str:
    """A deterministic id, such as HSY:gross_margin:Q2 2026:3m.

    Growth also names the period it is measured against, because Q2 2026 growth differs between
    a year-over-year and a sequential comparison.
    """
    base = f"{filing.ticker}:{name.value}:{filing.period.label}:{duration_for(filing.form).value}"
    return f"{base}:vs-{compared_to.period.label}" if compared_to else base


class MetricCatalog:
    """Computes metrics from a store's facts. It satisfies MetricChecker and MetricLookup.

    The caller owns the store and closes it. The corpus is what lets an id be resolved back to
    the filing it names, so only the pinned filings' metrics can be looked up.
    """

    def __init__(self, store: FactStore, corpus: Corpus, config: MetricsConfig) -> None:
        self._store = store
        self._corpus = corpus
        self._config = config

    def _reported(self, filing: Filing, item: str) -> _Amount | None:
        """A line item as the filing itself reports it, company-wide, for its own period."""
        duration = duration_for(filing.form).value
        for concept in self._config.line_items[item]:
            rows = [
                row
                for row in self._store.search(
                    filing.ticker,
                    concept,
                    end_date=filing.period.period_end,
                    source="xbrl",
                    limit=FACT_SEARCH_LIMIT,
                )
                if row.name == concept
                and row.accession == filing.accession
                and row.duration == duration
                and row.unit == XBRL_UNIT_USD
                # `dimensions` covers every breakdown, while `segment` only covers one axis. A
                # revenue figure for one product line has no segment but does have a dimension.
                and row.dimensions is None
                and row.segment is None
            ]
            if not rows:
                continue
            # A figure repeated in the filing is fine. Two different ones can't be chosen between.
            if len({row.value for row in rows}) > 1:
                raise _ConflictingFigures(f"{filing.accession} {concept}")
            rows.sort(key=lambda row: row.fact_id)
            return _Amount(rows[0].value, (rows[0].fact_id,))
        return None

    def _amount(self, filing: Filing, item: str) -> _Amount | None:
        """A line item, computing gross profit when the filer reports no such line."""
        reported = self._reported(filing, item)
        if reported is not None or item != GROSS_PROFIT:
            return reported
        revenue = self._reported(filing, REVENUE)
        cost = self._reported(filing, COST_OF_GOODS_SOLD)
        if revenue is None or cost is None:
            return None
        return _Amount(revenue.value - cost.value, (*revenue.fact_ids, *cost.fact_ids))

    def ratio(self, filing: Filing, name: MetricName) -> MetricValue | None:
        """One of the ratios over revenue, for one filing's period, or None if it can't be."""
        try:
            numerator = self._amount(filing, RATIO_NUMERATORS[name])
            revenue = self._reported(filing, REVENUE)
        except _ConflictingFigures:
            return None
        # A company with no revenue in a period has no margin, rather than an infinite one.
        if numerator is None or revenue is None or revenue.value == 0:
            return None
        return MetricValue(
            metric_id=metric_id_for(filing, name),
            ticker=filing.ticker,
            name=name.value,
            period=filing.period,
            duration=duration_for(filing.form),
            value=numerator.value / revenue.value,
            unit=UNIT_RATIO,
            input_fact_ids=tuple(dict.fromkeys((*numerator.fact_ids, *revenue.fact_ids))),
        )

    def growth(self, current: Filing, prior: Filing) -> MetricValue | None:
        """Revenue growth, dated to the current period, or None if either revenue is missing."""
        try:
            now = self._reported(current, REVENUE)
            before = self._reported(prior, REVENUE)
        except _ConflictingFigures:
            return None
        # Growth from zero or from a negative base has no meaningful percentage.
        if now is None or before is None or before.value <= 0:
            return None
        return MetricValue(
            metric_id=metric_id_for(current, MetricName.REVENUE_GROWTH, prior),
            ticker=current.ticker,
            name=MetricName.REVENUE_GROWTH.value,
            period=current.period,
            duration=duration_for(current.form),
            value=now.value / before.value - 1,
            unit=UNIT_RATIO,
            input_fact_ids=(*now.fact_ids, *before.fact_ids),
        )

    def _outcome(self, comparison: Comparison, expected: MetricExpectation) -> MetricOutcome:
        tolerance = self._config.no_move_tolerance
        if expected.metric is MetricName.REVENUE_GROWTH:
            value = self.growth(comparison.current, comparison.prior)
            if value is None:
                return _not_applicable(expected)
            cited = (value.metric_id,)
            move, limit = value.value, tolerance.revenue_growth
        else:
            current = self.ratio(comparison.current, expected.metric)
            prior = self.ratio(comparison.prior, expected.metric)
            if current is None or prior is None:
                return _not_applicable(expected)
            cited = (current.metric_id, prior.metric_id)
            move, limit = current.value - prior.value, tolerance.ratio
        # An exact tie has no direction, whatever the tolerance, and a smaller move can't be called.
        if move == 0 or abs(move) < limit:
            return _not_applicable(expected)
        moved = Direction.UP if move > 0 else Direction.DOWN
        verdict = NumericCheck.AGREE if moved is expected.direction else NumericCheck.CONTRADICT
        return MetricOutcome(metric=expected.metric, check=verdict, metric_ids=cited)

    def check(
        self, comparison: Comparison, expectations: Sequence[MetricExpectation]
    ) -> MetricCheckResult:
        """One outcome per expectation, in order. See `MetricChecker` for what each one means."""
        return MetricCheckResult(
            outcomes=tuple(self._outcome(comparison, expected) for expected in expectations)
        )

    def _filing_for(self, ticker: str, label: str) -> Filing | None:
        try:
            company = self._corpus.company(ticker)
        except CorpusError:
            return None
        for pinned in company.filings:
            filing = company.to_filing(pinned)
            if filing.period.label == label:
                return filing
        return None

    def get_metric(self, metric_id: str) -> MetricValue | None:
        """Rebuild a metric from its id, or None if the id names nothing the catalog can compute."""
        parts = metric_id.split(":")
        if len(parts) not in (4, 5):
            return None
        ticker, name_text, label, duration, *compared = parts
        try:
            name = MetricName(name_text)
        except ValueError:
            return None
        filing = self._filing_for(ticker, label)
        if filing is None or duration != duration_for(filing.form).value:
            return None
        # Only growth is measured against another period, and growth always is.
        if (name is MetricName.REVENUE_GROWTH) != bool(compared):
            return None
        if name is not MetricName.REVENUE_GROWTH:
            return self.ratio(filing, name)
        prior = None
        if compared[0].startswith("vs-"):
            prior = self._filing_for(ticker, compared[0].removeprefix("vs-"))
        # The same rule a Comparison enforces: same kind of filing, and an earlier period.
        if (
            prior is None
            or prior.form is not filing.form
            or prior.period.period_end >= filing.period.period_end
        ):
            return None
        return self.growth(filing, prior)


def _not_applicable(expected: MetricExpectation) -> MetricOutcome:
    return MetricOutcome(metric=expected.metric, check=NumericCheck.NOT_APPLICABLE)
