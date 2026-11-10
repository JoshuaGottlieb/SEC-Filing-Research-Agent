"""A change's seasonal label needs the seasonal check behind it; a refused check says so."""

from __future__ import annotations

import pytest

from sec_agent.schemas import Change, ChangeKind


def numeric(**fields: bool) -> Change:
    return Change(
        change_id="HSY:seq-10q:numeric:gross_margin",
        comparison_id="HSY:seq-10q",
        kind=ChangeKind.NUMERIC,
        metric_id="HSY:gross_margin:Q2 2026:3m",
        **fields,
    )


def test_the_check_is_refused_unless_it_ran():
    change = numeric()
    assert (change.seasonality_checked, change.possibly_seasonal) == (False, False)


def test_checked_and_not_seasonal():
    change = numeric(seasonality_checked=True)
    assert (change.seasonality_checked, change.possibly_seasonal) == (True, False)


def test_checked_and_possibly_seasonal():
    change = numeric(seasonality_checked=True, possibly_seasonal=True)
    assert change.possibly_seasonal


def test_a_seasonal_label_without_the_check_is_refused():
    with pytest.raises(ValueError, match="seasonality_checked"):
        numeric(possibly_seasonal=True)
