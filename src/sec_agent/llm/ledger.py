"""Tracks LLM spend for a run and stops it at the budget."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


class BudgetExceededError(RuntimeError):
    """A call could push the run past its dollar cap, so it was not sent."""


@dataclass(frozen=True)
class LedgerEntry:
    """One call's tokens and cost. A role is the job the call did (for example "critic")."""

    role: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    # True when the answer came from the replay cache and cost nothing.
    replayed: bool
    fallback_used: bool = False


class CostLedger:
    """Reserve a call's worst-case cost first, then settle to the actual cost.

    Reserving makes the cap a hard stop even with calls running in parallel,
    because in-flight calls already count against it. The run stops a little
    below the cap rather than slightly above it.
    """

    def __init__(self, max_usd: float, path: Path | None = None) -> None:
        self.max_usd = max_usd
        self.path = path
        self._lock = threading.Lock()
        # Money already paid, and money held for calls still in flight.
        self._spent = 0.0
        self._reserved = 0.0
        self._entries: list[LedgerEntry] = []

    @classmethod
    def resume(cls, max_usd: float, path: Path) -> CostLedger:
        """A ledger that continues a run's log: every call it already holds counts as spent.

        A restarted process would otherwise start at $0, so the cap and the brief's cost would
        miss what the run paid before it stopped (T5.5 resume). New calls append to the same file.
        A last line cut off by a crash is skipped; any other unreadable line is an error.
        """
        ledger = cls(max_usd, path)
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
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
            entry = LedgerEntry(**row)
            ledger._spent += entry.cost_usd
            ledger._entries.append(entry)
        return ledger

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def reserve(self, amount: float) -> None:
        with self._lock:
            # Refuse a call that could take the total over the cap, before it is sent.
            committed = self._spent + self._reserved
            if committed + amount > self.max_usd:
                raise BudgetExceededError(
                    f"Budget of ${self.max_usd:.2f} reached: ${self._spent:.4f} spent, "
                    f"${self._reserved:.4f} in flight, next call could cost ${amount:.4f}."
                )
            self._reserved += amount

    def release(self, amount: float) -> None:
        """Give back a reservation when the call failed or never ran."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - amount)

    def record(self, entry: LedgerEntry, reserved: float = 0.0) -> None:
        """Settle a call: replace its reservation with the actual cost, and log it."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - reserved)
            self._spent += entry.cost_usd
            self._entries.append(entry)
            # One JSON line per call, appended, so the log survives a crash.
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                row = {"at": datetime.now(UTC).isoformat(), **asdict(entry)}
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")

    def cost_by_role(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for entry in self.entries:
            totals[entry.role] += entry.cost_usd
        return dict(totals)
