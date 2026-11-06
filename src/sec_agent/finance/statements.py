"""Reads a filed income statement from the text of its table.

T1.4 asks whether the revenue and gross profit the metric catalog uses match the filed statements.
The catalog reads XBRL, so the check needs a second source that doesn't. The statements are one:
the table parser rejects every one of the 15 pinned filings' income statements, so each is kept
as text in `unparsed_tables`, copied from the filed HTML and not from the XBRL tags. This module
reads figures out of that text.

It is deliberately strict. A table whose layout it doesn't understand raises `StatementError`
instead of being read on a guess, because a reader that quietly picked the wrong column would
make the check agree with the wrong number or disagree with the right one.

The layout it understands: the first value column is the filing's own period (the quarter for a
10-Q, the year for a 10-K) and the second is the same period a year earlier. A Q2 or Q3 10-Q then
adds a year-to-date pair, and a 10-K adds a third year. The header must say so in years, and a
header that disagrees is an error.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from sec_agent.schemas import FormType, Section

# The columns the filing's own period and the same period a year earlier occupy.
CURRENT = 0
YEAR_EARLIER = 1

# A statement's figures are rounded to its displayed unit, so two sources agree when they differ
# by no more than half of it. The tiny extra absorbs floating-point noise, not a real difference.
ROUNDING_SLACK = 1e-6

# Row labels, matched whole and ignoring case. Filers differ: Hershey and Mattel say "Net sales",
# Stanley "Net Sales"; Mattel capitalizes "Gross Profit". Add a label only after reading a
# statement that uses it.
REVENUE_LABEL = re.compile(r"net sales|net revenues?|total revenues?|revenues?", re.IGNORECASE)
COST_LABEL = re.compile(r"cost of (sales|goods sold|revenues?)", re.IGNORECASE)
GROSS_PROFIT_LABEL = re.compile(r"gross profit", re.IGNORECASE)
# The last line of an income statement, which tells it apart from a table that only has sales.
NET_INCOME_LABEL = re.compile(r"net (income|earnings|loss|\(loss\))", re.IGNORECASE)
# The rows the check reads, each of which must have one value per column.
READ_LABELS = (REVENUE_LABEL, COST_LABEL, GROSS_PROFIT_LABEL)

NUMBER = re.compile(r"^\(?-?\$?\s?[\d,]*\d(\.\d+)?\)?$")
# A dash stands for zero in a filed table.
DASHES = {"—", "–", "-"}
# A column heading's year, as in "June 28, 2026", "March 31,2026" or a bare "2025".
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


class StatementError(ValueError):
    """A statement is missing, ambiguous, or laid out in a way this reader doesn't understand."""


class Figure(BaseModel):
    """A dollar amount read from a statement, with how closely it is known."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dollars: float
    # Half of the statement's displayed unit, in dollars: the rounding a filed figure carries.
    tolerance: float
    # True for revenue minus cost of sales, used where a statement has no gross profit line.
    derived: bool = False

    def agrees_with(self, dollars: float) -> bool:
        return abs(self.dollars - dollars) <= self.tolerance + ROUNDING_SLACK


class StatementRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    # The displayed numbers in column order, in the statement's own unit and without "$" or "%".
    values: tuple[float, ...]
    # The most decimals any of the row's numbers displays, which sets how finely it is rounded.
    decimals: int


class IncomeStatement(BaseModel):
    """The rows of one filed income statement and what unit they are in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table_id: str
    # Dollars in one displayed unit: 1,000 for a statement in thousands, 1,000,000 for millions.
    dollars_per_unit: float
    column_years: tuple[int, ...]
    rows: tuple[StatementRow, ...]

    def _row(self, pattern: re.Pattern[str]) -> StatementRow | None:
        found = [row for row in self.rows if pattern.fullmatch(row.label)]
        if len(found) > 1:
            raise StatementError(f"{self.table_id}: {len(found)} rows match {pattern.pattern!r}")
        return found[0] if found else None

    def _figure(self, row: StatementRow, column: int) -> Figure:
        if column >= len(row.values):
            raise StatementError(f"{self.table_id}: {row.label!r} has no column {column}")
        unit = self.dollars_per_unit / 10**row.decimals
        return Figure(dollars=row.values[column] * self.dollars_per_unit, tolerance=unit / 2)

    def revenue(self, column: int = CURRENT) -> Figure:
        row = self._row(REVENUE_LABEL)
        if row is None:  # read_income_statement refuses such a statement, so this is a bad build
            raise StatementError(f"{self.table_id}: no revenue row")
        return self._figure(row, column)

    def cost_of_sales(self, column: int = CURRENT) -> Figure | None:
        row = self._row(COST_LABEL)
        return self._figure(row, column) if row else None

    def gross_profit(self, column: int = CURRENT) -> Figure | None:
        """The filed gross profit, or revenue minus cost of sales where the filer shows none.

        Stanley Black & Decker reports no gross profit line, and the metric catalog computes it
        the same way, so the derived figure is what is compared. The two roundings can add up, so
        its tolerance is both of theirs.
        """
        row = self._row(GROSS_PROFIT_LABEL)
        if row is not None:
            return self._figure(row, column)
        cost = self.cost_of_sales(column)
        if cost is None:
            return None
        revenue = self.revenue(column)
        return Figure(
            dollars=revenue.dollars - cost.dollars,
            tolerance=revenue.tolerance + cost.tolerance,
            derived=True,
        )


def _number(cell: str) -> tuple[float, int] | None:
    """A table cell as (value, decimals), or None if it isn't a number."""
    if cell in DASHES:
        return 0.0, 0
    if not NUMBER.match(cell):
        return None
    # Filed tables show a loss as (1,234); a leading minus is accepted too.
    negative = cell.startswith(("(", "-"))
    text = re.sub(r"[()$,\s-]", "", cell)
    decimals = len(text.split(".")[1]) if "." in text else 0
    value = float(text)
    return (-value if negative else value), decimals


def _parse_row(line: str) -> StatementRow | None:
    cells = [cell.strip() for cell in line.split(" | ")]
    label, rest = cells[0], cells[1:]
    if not label or _number(label) is not None:
        return None
    numbers = [parsed for cell in rest if cell not in ("$", "%") and (parsed := _number(cell))]
    if not numbers:
        return None
    return StatementRow(
        label=label,
        values=tuple(value for value, _ in numbers),
        decimals=max(decimals for _, decimals in numbers),
    )


def _scale(table: dict) -> float:
    """Dollars per displayed unit, from the table's recorded scale caption."""
    caption = table.get("scale_caption") or ""
    thousands, millions = "thousands" in caption.lower(), "millions" in caption.lower()
    if thousands == millions:  # neither, or both
        raise StatementError(f"{table['table_id']}: no single scale in caption {caption!r}")
    return 1_000.0 if thousands else 1_000_000.0


def expected_years(form: FormType, fiscal_period: str, fiscal_year: int) -> tuple[int, ...]:
    """The column years of a filing's income statement, as the layout rule says."""
    if form is FormType.FORM_10K:
        return (fiscal_year, fiscal_year - 1, fiscal_year - 2)
    if fiscal_period == "Q1":
        return (fiscal_year, fiscal_year - 1)
    return (fiscal_year, fiscal_year - 1, fiscal_year, fiscal_year - 1)


def _header_years(lines: Sequence[str]) -> tuple[int, ...]:
    """The years of the column heading: the last line above the first row that names some.

    A title line can name years too (Stanley's reads "Fiscal years ended January 3, 2026, ..."),
    so the line nearest the data is the one that labels the columns. A caption in parentheses,
    such as "(In thousands...)", sits between the two and names none.
    """
    for line in reversed(lines):
        years = YEAR.findall(line)
        if years and not line.startswith("("):
            return tuple(int(year) for year in years)
    return ()


def read_income_statement(table: dict) -> IncomeStatement:
    """One `unparsed_tables` record as an income statement, or `StatementError`."""
    table_id = table["table_id"]
    lines = table["text"].split("\n")
    parsed = [(index, _parse_row(line)) for index, line in enumerate(lines)]
    rows = [(index, row) for index, row in parsed if row is not None]
    revenue = [(index, row) for index, row in rows if REVENUE_LABEL.fullmatch(row.label)]
    if len(revenue) != 1:
        raise StatementError(f"{table_id}: {len(revenue)} revenue rows, expected one")

    form = FormType(table["form"])
    years = _header_years(lines[: revenue[0][0]])
    expected = expected_years(form, table["fiscal_period"], table["fiscal_year"])
    if years != expected:
        raise StatementError(f"{table_id}: column years {years}, expected {expected}")
    statement = IncomeStatement(
        table_id=table_id,
        dollars_per_unit=_scale(table),
        column_years=years,
        rows=tuple(row for _, row in rows),
    )
    # Every figure the check reads must have one value per column, or a column is missing.
    for _, row in rows:
        if any(pattern.fullmatch(row.label) for pattern in READ_LABELS) and len(row.values) != len(
            years
        ):
            raise StatementError(f"{table_id}: {row.label!r} has {len(row.values)} values")
    return statement


def find_income_statement(tables: Sequence[dict]) -> dict:
    """The one table of a filing that is its income statement, or `StatementError`.

    It must sit in the financial statements and have a revenue row, a cost of sales row and a
    net income row. Hershey's MD&A repeats its statement in millions with the same rows, so the
    section matters. More or fewer than one candidate is an error and not a pick.
    """
    candidates = []
    for table in tables:
        if table["section"] != Section.FINANCIAL_NOTES.value:
            continue
        labels = [row.label for line in table["text"].split("\n") if (row := _parse_row(line))]
        if (
            any(REVENUE_LABEL.fullmatch(label) for label in labels)
            and any(COST_LABEL.fullmatch(label) for label in labels)
            and any(NET_INCOME_LABEL.match(label) for label in labels)
        ):
            candidates.append(table)
    if len(candidates) != 1:
        raise StatementError(f"{len(candidates)} income statement candidates, expected one")
    return candidates[0]
