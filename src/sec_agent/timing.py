"""Where a run's time goes, and which step and change each model call served (T6.6).

Cost and latency per finding need two things the ledger alone does not give: how long each step
took, and which change a model call was made for. A step runs inside `TimingLog.measure`, which
times it and sets the current `Scope` (the step and, for a change's own steps, the change id). The
LLM client stamps every ledger line with that scope and the call's own duration, so a ledger reads
back by change. `TimedRetriever` times each local evidence search (hybrid search and the CPU
cross-encoder), which takes no model call and so never reaches the ledger.

The scope is a context variable: tasks started inside a step (the search's parallel candidates)
inherit it, and a step's scope ends with the step, so nothing leaks into the next one.

A run's timings go to `timings.jsonl` beside its ledger, one JSON line per step or search,
appended, so a stopped run keeps what it measured and a resumed one adds to it.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from sec_agent.contracts import Retriever
from sec_agent.schemas import Comparison, RetrievalResult, Section

# Step names in `timings.jsonl` and on ledger lines. A change's own steps also carry its id.
STAGE_DETECT = "detect"  # alignment and the metric scan; no model
STAGE_MATERIALITY = "materiality"
STAGE_SEARCH = "search"  # the Tree-of-Thought search for one change
STAGE_LABEL = "label"  # labeling what survived: the judge and the claim writer
STAGE_REWORK = "rework"  # one rework round for one claim
STAGE_SUMMARY = "summary"
RETRIEVE = "retrieve"  # one local evidence search, inside a change's step


@dataclass(frozen=True)
class Scope:
    """The step a piece of work belongs to, and the change it was for (None: the whole run)."""

    stage: str
    change_id: str | None = None


_SCOPE: ContextVar[Scope | None] = ContextVar("sec_agent_scope", default=None)


def current_scope() -> Scope | None:
    """The innermost step running here, or None outside any."""
    return _SCOPE.get()


@dataclass(frozen=True)
class Timing:
    """One step's or one search's wall time. `within` is the step it ran inside, if any."""

    stage: str
    change_id: str | None
    elapsed_s: float
    failed: bool = False
    within: str | None = None


class TimingLog:
    """Times steps, sets their scope, and keeps the timings (in memory, and on disk if given)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._entries: list[Timing] = []

    @property
    def entries(self) -> tuple[Timing, ...]:
        with self._lock:
            return tuple(self._entries)

    @contextmanager
    def measure(self, stage: str, change_id: str | None = None) -> Iterator[None]:
        """Time the block as `stage`. With no `change_id`, the enclosing step's carries over.

        A block that raises is still recorded, marked failed, and the error goes on unchanged.
        """
        parent = _SCOPE.get()
        if change_id is None and parent is not None:
            change_id = parent.change_id
        token = _SCOPE.set(Scope(stage, change_id))
        started = time.perf_counter()
        failed = True
        try:
            yield
            failed = False
        finally:
            elapsed = time.perf_counter() - started
            _SCOPE.reset(token)
            within = parent.stage if parent is not None else None
            self.add(Timing(stage, change_id, elapsed, failed, within))

    def add(self, timing: Timing) -> None:
        with self._lock:
            self._entries.append(timing)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                row = {"at": datetime.now(UTC).isoformat(), **asdict(timing)}
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")


def read_timings(path: Path) -> list[Timing]:
    """A run's timing log. A last line cut off by a crash is skipped, as in the ledger."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    timings = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                break
            raise
        row.pop("at", None)
        timings.append(Timing(**row))
    return timings


class TimedRetriever:
    """A `Retriever` that logs each search's wall time under the step that asked for it."""

    def __init__(self, inner: Retriever, log: TimingLog) -> None:
        self._inner = inner
        self._log = log

    async def retrieve(
        self, query: str, comparison: Comparison, section: Section | None = None
    ) -> RetrievalResult:
        with self._log.measure(RETRIEVE):
            return await self._inner.retrieve(query, comparison, section)
