"""Step timing and scope (T6.6): what each step took, and which change work was done for."""

from __future__ import annotations

import asyncio

import pytest

from sec_agent.contracts import Retriever
from sec_agent.schemas import RetrievalResult
from sec_agent.timing import (
    RETRIEVE,
    Scope,
    TimedRetriever,
    Timing,
    TimingLog,
    current_scope,
    read_timings,
)


def test_a_step_is_timed_and_scoped_only_while_it_runs() -> None:
    log = TimingLog()
    assert current_scope() is None
    with log.measure("search", "HSY:c1"):
        assert current_scope() == Scope("search", "HSY:c1")
    assert current_scope() is None
    (timing,) = log.entries
    assert (timing.stage, timing.change_id, timing.failed, timing.within) == (
        "search",
        "HSY:c1",
        False,
        None,
    )
    assert timing.elapsed_s >= 0


def test_a_nested_step_carries_the_change_over_and_names_its_parent() -> None:
    log = TimingLog()
    with log.measure("search", "HSY:c1"), log.measure(RETRIEVE):
        assert current_scope() == Scope(RETRIEVE, "HSY:c1")
    inner, outer = log.entries
    assert (inner.stage, inner.change_id, inner.within) == (RETRIEVE, "HSY:c1", "search")
    assert outer.stage == "search"


def test_a_failed_step_is_recorded_and_the_error_goes_on() -> None:
    log = TimingLog()
    with pytest.raises(RuntimeError), log.measure("label", "HSY:c1"):
        raise RuntimeError("judge failed")
    assert log.entries[0].failed
    assert current_scope() is None


async def test_tasks_started_inside_a_step_inherit_its_scope() -> None:
    # The search evaluates candidates in parallel tasks; their model calls must keep the change.
    seen: list[Scope | None] = []

    async def candidate() -> None:
        await asyncio.sleep(0)
        seen.append(current_scope())

    with TimingLog().measure("search", "HSY:c1"):
        await asyncio.gather(candidate(), candidate())
    assert seen == [Scope("search", "HSY:c1")] * 2


async def test_parallel_steps_keep_their_own_scopes() -> None:
    log = TimingLog()

    async def step(change_id: str) -> Scope | None:
        with log.measure("search", change_id):
            await asyncio.sleep(0)
            return current_scope()

    first, second = await asyncio.gather(step("A"), step("B"))
    assert (first, second) == (Scope("search", "A"), Scope("search", "B"))


def test_the_log_round_trips_through_its_file(tmp_path) -> None:
    path = tmp_path / "run" / "timings.jsonl"
    log = TimingLog(path)
    with log.measure("detect"):
        pass
    with log.measure("search", "HSY:c1"):
        pass
    assert read_timings(path) == list(log.entries)
    # Appending, as a resumed run does, keeps what was there.
    TimingLog(path).add(Timing("summary", None, 1.5))
    assert [t.stage for t in read_timings(path)] == ["detect", "search", "summary"]


def test_a_last_line_cut_off_by_a_crash_is_skipped_and_others_are_errors(tmp_path) -> None:
    path = tmp_path / "timings.jsonl"
    TimingLog(path).add(Timing("detect", None, 2.0))
    with path.open("a") as handle:
        handle.write('{"stage": "sea')
    assert [t.stage for t in read_timings(path)] == ["detect"]
    path.write_text('{"stage": "sea\n' + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError):
        read_timings(path)


def test_a_missing_log_reads_as_empty(tmp_path) -> None:
    assert read_timings(tmp_path / "missing.jsonl") == []


class FakeRetriever:
    def __init__(self) -> None:
        self.scopes: list[Scope | None] = []

    async def retrieve(self, query, comparison, section=None) -> RetrievalResult:
        self.scopes.append(current_scope())
        return RetrievalResult(query=query, current=(), prior=())


async def test_each_search_is_timed_under_the_change_that_asked() -> None:
    log = TimingLog()
    inner = FakeRetriever()
    timed = TimedRetriever(inner, log)
    assert isinstance(timed, Retriever)
    with log.measure("search", "HSY:c1"):
        result = await timed.retrieve("cocoa costs", None)  # type: ignore[arg-type]
    assert result.query == "cocoa costs"
    assert inner.scopes == [Scope(RETRIEVE, "HSY:c1")]
    search, step = log.entries
    assert (search.stage, search.change_id, search.within) == (RETRIEVE, "HSY:c1", "search")
    assert step.stage == "search"
