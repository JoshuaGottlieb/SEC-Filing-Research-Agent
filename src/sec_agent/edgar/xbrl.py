"""Reads the XBRL facts embedded in a filing's inline HTML.

Inline XBRL carries what the SEC's aggregate APIs drop, including segment values.
"""

from __future__ import annotations

import json
import re
from datetime import date

import lxml.html
from pydantic import BaseModel, ConfigDict

from sec_agent.schemas import Duration

# A context's length in days decides its duration. A quarter is about 91 days and a 52 or
# 53-week year is 364 or 371, so these ranges leave room for calendar differences. Anything
# else (a six or nine month span) is treated as year to date.
THREE_MONTH_DAYS = (80, 100)
TWELVE_MONTH_DAYS = (350, 380)
# Inline XBRL number formats that display a dash but mean zero.
ZERO_FORMATS = re.compile(r"zerodash|fixed-zero|fixedzero|numdash", re.I)
NUMBER = re.compile(r"-?\d+(\.\d+)?")


class XbrlContext(BaseModel):
    """The period (and any breakdown) a tagged number applies to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    context_id: str
    # None for a point in time such as a balance sheet date.
    start_date: date | None = None
    end_date: date
    # Breakdowns such as a segment. An empty dict means the company-wide figure.
    dimensions: dict[str, str] = {}


class InlineFact(BaseModel):
    """One tagged number from the filing's HTML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    concept: str
    context_id: str
    # Fully scaled and signed, in the tag's unit.
    value: float
    unit: str | None = None
    decimals: str | None = None


def _text(element: lxml.html.HtmlElement, tag: str) -> str | None:
    """The text of the first matching child element, or None."""
    found = next(element.iter(tag), None)
    return found.text_content().strip() if found is not None else None


def read_contexts(root: lxml.html.HtmlElement) -> dict[str, XbrlContext]:
    """Every period definition in the filing, by id. Tags refer to these by `contextref`."""
    contexts = {}
    # lxml's HTML parser lowercases tag names, hence "xbrli:startdate" and not "startDate".
    for element in root.iter("xbrli:context"):
        instant = _text(element, "xbrli:instant")
        start = _text(element, "xbrli:startdate")
        end = instant or _text(element, "xbrli:enddate")
        # A context with no date can't place a number in time.
        if not end:
            continue
        dimensions = {
            member.get("dimension", ""): member.text_content().strip()
            for member in element.iter("xbrldi:explicitmember", "xbrldi:typedmember")
        }
        context_id = element.get("id", "")
        contexts[context_id] = XbrlContext(
            context_id=context_id,
            start_date=date.fromisoformat(start[:10]) if start and not instant else None,
            end_date=date.fromisoformat(end[:10]),
            dimensions=dimensions,
        )
    return contexts


def read_inline_fact(element: lxml.html.HtmlElement) -> InlineFact | None:
    """One `ix:nonFraction` tag. None when its text isn't a plain number."""
    text = " ".join(element.text_content().split()).replace(",", "")
    format_name = element.get("format") or ""
    # A dash means zero, whether the tag's format says so or the text is just a dash.
    if ZERO_FORMATS.search(format_name) or text in ("-", "—", "–"):
        number = 0.0
    else:
        match = NUMBER.search(text)
        if match is None:
            return None
        number = float(match.group())
    # `scale` is a power of ten (6 means millions); `sign="-"` marks a negative value that
    # the page may display without a minus sign.
    scale = int(element.get("scale") or 0)
    sign = -1.0 if element.get("sign") == "-" else 1.0
    return InlineFact(
        concept=element.get("name", ""),
        context_id=element.get("contextref", ""),
        value=sign * number * 10.0**scale,
        unit=element.get("unitref"),
        decimals=element.get("decimals"),
    )


def duration_of(context: XbrlContext) -> Duration:
    """Classify a context as a point in time, a quarter, a year, or year to date."""
    if context.start_date is None:
        return Duration.INSTANT
    days = (context.end_date - context.start_date).days
    if THREE_MONTH_DAYS[0] <= days <= THREE_MONTH_DAYS[1]:
        return Duration.THREE_MONTHS
    if TWELVE_MONTH_DAYS[0] <= days <= TWELVE_MONTH_DAYS[1]:
        return Duration.TWELVE_MONTHS
    return Duration.YEAR_TO_DATE


def dimensions_json(context: XbrlContext) -> str | None:
    """A context's breakdowns as stable JSON text (sorted keys), or None if company-wide."""
    return json.dumps(context.dimensions, sort_keys=True) if context.dimensions else None
