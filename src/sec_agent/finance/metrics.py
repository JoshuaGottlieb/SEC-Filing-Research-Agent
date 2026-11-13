"""The metric catalog: ratios and dollar figures from company-wide XBRL facts, and their check.

`MetricCatalog` is the `MetricChecker` and `MetricLookup` of contracts.py, whose docstrings define
what a value means. Every value is computed when asked for rather than stored, so it can't drift
from the facts, and a metric id holds all that is needed to compute it again.

Each period's figures come from that period's own filing. A later filing can restate an earlier
period, so a comparison shows what each filing reported, and a value traces to one filing.

Missing data is never filled in. A line item a filing doesn't report, or reports twice with
different values, gives no metric, and a check on it is NOT_APPLICABLE.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sec_agent.config import REPO_ROOT
from sec_agent.contracts import ComparedMetric
from sec_agent.edgar.corpus import Corpus, CorpusError
from sec_agent.finance.fact_store import FactRow, FactStore
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
# The unit names the contract fixes (see contracts.MetricLookup).
UNIT_RATIO = "ratio"
UNIT_USD = "USD"
# How xbrl_facts writes US dollars. A per-share figure has a different unit and is never used.
XBRL_UNIT_USD = "usd"
# More rows than one company reports for one period end, so that FactStore.search's limit can't
# silently drop the row a metric needs and make it look missing.
FACT_SEARCH_LIMIT = 100_000
# How many days before a filing's period end the same quarter a year earlier ends. A calendar
# quarter ends on the same date (365 or 366 days), and a 52/53-week year moves it by up to a week
# either way (Stanley's Q2 ended June 28, 2025 and July 4, 2026: 371 days).
YEAR_EARLIER_DAYS = range(357, 372)

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
# The line item each dollar metric reports as it is. Its name and the config key are the same.
DOLLAR_ITEMS = {
    MetricName.REVENUE: REVENUE,
    MetricName.COST_OF_GOODS_SOLD: COST_OF_GOODS_SOLD,
    MetricName.GROSS_PROFIT: GROSS_PROFIT,
    MetricName.SGA: SGA,
    MetricName.OPERATING_INCOME: OPERATING_INCOME,
    MetricName.NET_INCOME: NET_INCOME,
}


def measured_in_points(name: MetricName) -> bool:
    """Whether a metric's move is the difference of two ratios, not a relative change.

    A ratio moves in points (a margin "fell 1,377 bps"); a dollar figure moves relative to its
    earlier value ("+10.5%"), and revenue growth is itself a relative change.
    """
    return name is not MetricName.REVENUE_GROWTH and name not in DOLLAR_ITEMS


def relative_change(current: float, prior: float) -> float | None:
    """The change as a fraction of a positive starting figure, or None when it has none."""
    return current / prior - 1 if prior > 0 else None


class NoMoveTolerance(BaseModel):
    """How small a move is too small to call, as a fraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ratio: float = Field(ge=0)
    revenue_growth: float = Field(ge=0)
    # In dollars, for the dollar metrics.
    usd: float = Field(ge=0)


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


# Finds one reported line item for a filing: its own period, or the period a year earlier.
_Reader = Callable[[Filing, str], _Amount | None]


class PastValue(BaseModel):
    """A metric for the period a year before a filing's own, from `MetricCatalog.year_earlier`.

    It has no metric id: that period need not be a pinned filing, so `get_metric` couldn't
    rebuild it. It is evidence for the seasonal check, not a figure a claim can cite.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: MetricName
    value: float
    unit: str
    input_fact_ids: tuple[str, ...] = Field(min_length=1)


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

    def _company_rows(self, filing: Filing, concept: str, end_date: date | None) -> list[FactRow]:
        """A concept's company-wide dollar facts over the filing's kind of period (3m or 12m)."""
        duration = duration_for(filing.form).value
        return [
            row
            for row in self._store.search(
                filing.ticker, concept, end_date=end_date, source="xbrl", limit=FACT_SEARCH_LIMIT
            )
            if row.name == concept
            and row.duration == duration
            and row.unit == XBRL_UNIT_USD
            # `dimensions` covers every breakdown, while `segment` only covers one axis. A
            # revenue figure for one product line has no segment but does have a dimension.
            and row.dimensions is None
            and row.segment is None
        ]

    @staticmethod
    def _one_figure(rows: Sequence[FactRow], where: str) -> _Amount:
        # A figure repeated in a filing is fine. Two different ones can't be chosen between.
        if len({row.value for row in rows}) > 1:
            raise _ConflictingFigures(where)
        first = min(rows, key=lambda row: row.fact_id)
        return _Amount(first.value, (first.fact_id,))

    def _reported(self, filing: Filing, item: str) -> _Amount | None:
        """A line item as the filing itself reports it, company-wide, for its own period."""
        for concept in self._config.line_items[item]:
            rows = [
                row
                for row in self._company_rows(filing, concept, filing.period.period_end)
                if row.accession == filing.accession
            ]
            if rows:
                return self._one_figure(rows, f"{filing.accession} {concept}")
        return None

    def _reported_year_earlier(self, filing: Filing, item: str) -> _Amount | None:
        """A line item for the same quarter (or year) one year before the filing's own period.

        That period's own filing is read first, as for every other figure, when the store has
        it. Otherwise the year-earlier figure the later filing itself reports is used: every
        10-Q and 10-K repeats the year-earlier period beside its own. With neither, it is None.
        """
        own = (filing.form.value, filing.period.fiscal_year - 1, filing.period.fiscal_period)
        for concept in self._config.line_items[item]:
            rows = [
                row
                for row in self._company_rows(filing, concept, None)
                if (filing.period.period_end - row.end_date).days in YEAR_EARLIER_DAYS
            ]
            from_own = [
                row for row in rows if (row.form, row.fiscal_year, row.fiscal_period) == own
            ]
            from_later = [row for row in rows if row.accession == filing.accession]
            for found in (from_own, from_later):
                if found:
                    return self._one_figure(found, f"{filing.accession} {concept} a year earlier")
        return None

    def _amount(self, filing: Filing, item: str, read: _Reader | None = None) -> _Amount | None:
        """A line item, computing gross profit when the filer reports no such line.

        `read` finds a reported figure: the filing's own period unless told otherwise.
        """
        read = read or self._reported
        reported = read(filing, item)
        if reported is not None or item != GROSS_PROFIT:
            return reported
        revenue = read(filing, REVENUE)
        cost = read(filing, COST_OF_GOODS_SOLD)
        if revenue is None or cost is None:
            return None
        return _Amount(revenue.value - cost.value, (*revenue.fact_ids, *cost.fact_ids))

    def line_item(self, filing: Filing, item: str) -> float | None:
        """A line item's dollars for the filing's own period, exactly as the metrics use it.

        `item` is a key of `line_items` in the config, such as REVENUE or GROSS_PROFIT. Gross
        profit is computed when the filer reports none, as for the ratios. None means the figure
        can't be given: the filing doesn't report it, or reports it twice with different values.
        This is what T1.4 checks against the filed statements.
        """
        if item not in self._config.line_items:
            known = sorted(self._config.line_items)
            raise ValueError(f"unknown line item {item!r}, expected one of {known}")
        try:
            amount = self._amount(filing, item)
        except _ConflictingFigures:
            return None
        return None if amount is None else amount.value

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

    def dollars(self, filing: Filing, name: MetricName) -> MetricValue | None:
        """One of the dollar metrics, for one filing's period, or None if it can't be given.

        The same figure `line_item` gives, as a traceable metric: gross profit is computed where
        the filer reports none, and a figure reported twice with different values gives nothing.
        """
        try:
            amount = self._amount(filing, DOLLAR_ITEMS[name])
        except _ConflictingFigures:
            return None
        if amount is None:
            return None
        return MetricValue(
            metric_id=metric_id_for(filing, name),
            ticker=filing.ticker,
            name=name.value,
            period=filing.period,
            duration=duration_for(filing.form),
            value=amount.value,
            unit=UNIT_USD,
            input_fact_ids=amount.fact_ids,
        )

    def level(self, filing: Filing, name: MetricName) -> MetricValue | None:
        """A ratio or dollar metric for one period. Revenue growth spans two, so it isn't one."""
        if name in DOLLAR_ITEMS:
            return self.dollars(filing, name)
        if name in RATIO_NUMERATORS:
            return self.ratio(filing, name)
        raise ValueError(f"{name.value} is measured between two periods; use growth()")

    def year_earlier(self, filing: Filing, name: MetricName) -> PastValue | None:
        """A ratio or dollar metric for the period a year before the filing's own, or None.

        Read as `_reported_year_earlier` describes. None means there is no year-earlier figure
        to read, or it conflicts, so a check that needs it must be refused, not guessed.
        """
        read = self._reported_year_earlier
        try:
            if name in DOLLAR_ITEMS:
                amount = self._amount(filing, DOLLAR_ITEMS[name], read)
                if amount is None:
                    return None
                return PastValue(
                    name=name, value=amount.value, unit=UNIT_USD, input_fact_ids=amount.fact_ids
                )
            if name not in RATIO_NUMERATORS:
                raise ValueError(f"{name.value} is measured between two periods")
            numerator = self._amount(filing, RATIO_NUMERATORS[name], read)
            revenue = read(filing, REVENUE)
        except _ConflictingFigures:
            return None
        if numerator is None or revenue is None or revenue.value == 0:
            return None
        return PastValue(
            name=name,
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

    def compare(self, comparison: Comparison, name: MetricName) -> ComparedMetric | None:
        """`name` in both of the comparison's periods and how far it moved, or None without data.

        The one place a move is computed: the metric scan flags changes from it and `get_move`
        tells the thought generator the same move (user's decision, 2026-11-13).
        """
        if name is MetricName.REVENUE_GROWTH:
            growth = self.growth(comparison.current, comparison.prior)
            if growth is None:
                return None
            return ComparedMetric(later=growth, earlier=None, kind="relative", move=growth.value)
        later = self.level(comparison.current, name)
        earlier = self.level(comparison.prior, name)
        if later is None or earlier is None:
            return None
        if measured_in_points(name):
            return ComparedMetric(
                later=later, earlier=earlier, kind="points", move=later.value - earlier.value
            )
        return ComparedMetric(
            later=later,
            earlier=earlier,
            kind="relative",
            move=relative_change(later.value, earlier.value),
        )

    def get_move(self, comparison: Comparison, metric_id: str) -> ComparedMetric | None:
        """How `metric_id`'s metric moved in `comparison` (`MetricMoveLookup`)."""
        later = self.get_metric(metric_id)
        if later is None:
            return None
        compared = self.compare(comparison, MetricName(later.name))
        # A change built for other periods must not be told this comparison's move.
        if compared is None or compared.later.metric_id != metric_id:
            return None
        return compared

    def _outcome(self, comparison: Comparison, expected: MetricExpectation) -> MetricOutcome:
        tolerance = self._config.no_move_tolerance
        if expected.metric is MetricName.REVENUE_GROWTH:
            value = self.growth(comparison.current, comparison.prior)
            if value is None:
                return _not_applicable(expected)
            cited = (value.metric_id,)
            move, limit = value.value, tolerance.revenue_growth
        else:
            current = self.level(comparison.current, expected.metric)
            prior = self.level(comparison.prior, expected.metric)
            if current is None or prior is None:
                return _not_applicable(expected)
            cited = (current.metric_id, prior.metric_id)
            limit = tolerance.usd if current.unit == UNIT_USD else tolerance.ratio
            move = current.value - prior.value
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
            return self.level(filing, name)
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
