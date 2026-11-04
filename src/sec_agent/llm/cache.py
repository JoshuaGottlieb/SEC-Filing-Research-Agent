"""Replays identical LLM calls from SQLite.

Current Claude models reject temperature settings, so a live call can't be made
repeatable. Replaying saved results is what makes reruns free and repeatable.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def cache_key(**request: object) -> str:
    """A fixed-length key for a request. The same request always gives the same key."""
    # sort_keys makes the JSON independent of argument order.
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ReplayCache:
    """A key-value store of past responses. Safe to share between threads."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # check_same_thread=False because calls may run in parallel; the lock serializes use.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._db.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT payload FROM responses WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, payload: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses (key, payload) VALUES (?, ?)", (key, payload)
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
