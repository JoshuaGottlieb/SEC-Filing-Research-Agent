"""SQLite storage for Tree-of-Thought trees and each run's budget use.

This is the logic behind the MCP tree store. The server (server.py) exposes it as tools, and
the in-process client (client.py) calls it directly so controller tests need no subprocess.
Keeping the rules here means both routes enforce the same ones.

Each node is stored as its `TreeNode` JSON and validated on the way in and on the way out, so
a file that no longer matches the record fails loudly instead of reaching the audit trail.
Nothing is ever deleted: pruned nodes stay on record with their reason, because the audit
trail and the tree view depend on them.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import Field, ValidationError

from sec_agent.schemas import (
    Citation,
    EvidenceSupport,
    NodeStatus,
    NumericCheck,
    Record,
    TreeNode,
)


class TreeStoreError(RuntimeError):
    """The store refused a request: an unknown or duplicate id, or a change that breaks a rule."""


class NodeUpdate(Record):
    """The fields of a node that can change once it exists.

    A field left out stays as it is. What a node claims and where it sits in the tree (its
    thought, expected evidence, parent, depth) cannot change, or the audit trail would no
    longer show what was actually tested. Ranges and rules live on `TreeNode`, which checks the
    merged result, so they are not repeated here.
    """

    score: float | None = None
    status: NodeStatus | None = None
    prune_reason: str | None = None
    retrieved_chunk_ids: tuple[str, ...] | None = None
    best_passage_score: float | None = None
    evidence: EvidenceSupport | None = None
    citations: tuple[Citation, ...] | None = None
    metric_ids: tuple[str, ...] | None = None
    numeric_check: NumericCheck | None = None


class BudgetUse(Record):
    """A snapshot of what a run has used so far. Recording again replaces the last snapshot."""

    run_id: str
    spent_usd: float = Field(ge=0)
    max_usd: float = Field(gt=0)
    node_evaluations: int = Field(ge=0)


class TreeStore:
    """Trees of `TreeNode` records in one SQLite file. Safe to share between threads.

    The file belongs on local disk: SQLite is unreliable on the shared folder (see
    `Settings.data_dir`). Node ids are unique across the whole file, not per tree, so the
    controller must build them to include the run.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # check_same_thread=False because the MCP server runs each tool call on a worker thread;
        # the lock serializes use.
        self._db = sqlite3.connect(path, check_same_thread=False)
        # WAL so the progress view (T7.4) can read from another process during a run.
        self._db.execute("PRAGMA journal_mode=WAL")
        # seq keeps creation order, so a reloaded tree lists every parent before its children.
        # root_id is copied out of the JSON so that a tree can be read without parsing every row.
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL UNIQUE,
                root_id TEXT NOT NULL,
                record TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS nodes_by_root ON nodes (root_id, seq);
            CREATE TABLE IF NOT EXISTS budgets (
                run_id TEXT PRIMARY KEY,
                record TEXT NOT NULL
            );
            """
        )
        self._db.commit()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        # Every write checks existing rows first. BEGIN IMMEDIATE takes the write lock before
        # that check, so another process cannot change the row between the check and the write.
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._db.rollback()
                raise
            self._db.commit()

    def _fetch(self, node_id: str) -> tuple[str, TreeNode]:
        """Return a node and the id of its tree's root. The caller holds the lock."""
        row = self._db.execute(
            "SELECT root_id, record FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            raise TreeStoreError(f"no node with id {node_id!r}")
        return row[0], TreeNode.model_validate_json(row[1])

    def _insert(self, node: TreeNode, root_id: str) -> None:
        try:
            self._db.execute(
                "INSERT INTO nodes (node_id, root_id, record) VALUES (?, ?, ?)",
                (node.node_id, root_id, node.model_dump_json()),
            )
        except sqlite3.IntegrityError as error:
            raise TreeStoreError(f"a node with id {node.node_id!r} already exists") from error

    def create_tree(self, root: TreeNode) -> TreeNode:
        """Store a depth 0 node as the root of a new tree. Its node id is the tree's id."""
        if root.depth != 0:
            raise TreeStoreError(f"a tree's root must be at depth 0, not {root.depth}")
        with self._transaction():
            self._insert(root, root_id=root.node_id)
        return root

    def create_node(self, node: TreeNode) -> TreeNode:
        """Add a node under an existing parent, in the same run and change, one level down."""
        if node.parent_id is None:
            raise TreeStoreError("a node without a parent must be created with create_tree")
        with self._transaction():
            root_id, parent = self._fetch(node.parent_id)
            if (node.run_id, node.change_id) != (parent.run_id, parent.change_id):
                raise TreeStoreError(
                    f"node {node.node_id!r} must have the same run and change as its parent"
                )
            if node.depth != parent.depth + 1:
                raise TreeStoreError(
                    f"node {node.node_id!r} is at depth {node.depth} under a parent at depth "
                    f"{parent.depth}"
                )
            self._insert(node, root_id=root_id)
        return node

    def update_node(self, node_id: str, update: NodeUpdate) -> TreeNode:
        """Apply the fields set on `update` and return the node as stored."""
        with self._transaction():
            _, node = self._fetch(node_id)
            # model_copy would skip validation, so rebuild the record. That is what stops, for
            # example, a pruned node with no reason or a score above 1 from being saved.
            try:
                updated = TreeNode.model_validate(
                    {**node.model_dump(), **update.model_dump(exclude_unset=True)}
                )
            except ValidationError as error:
                raise TreeStoreError(f"invalid update to node {node_id!r}: {error}") from error
            # A pruned node stays pruned with its original reason, so a later write cannot erase
            # why a branch was dropped. A retry adds a new branch instead of reviving this one.
            if node.status is NodeStatus.PRUNED and (
                updated.status is not NodeStatus.PRUNED or updated.prune_reason != node.prune_reason
            ):
                raise TreeStoreError(f"node {node_id!r} is pruned: its status and reason are final")
            self._db.execute(
                "UPDATE nodes SET record = ? WHERE node_id = ?",
                (updated.model_dump_json(), node_id),
            )
        return updated

    def read_node(self, node_id: str) -> TreeNode:
        with self._lock:
            return self._fetch(node_id)[1]

    def read_tree(self, root_id: str) -> tuple[TreeNode, ...]:
        """Every node of a tree, pruned ones included, each parent before its children."""
        with self._lock:
            rows = self._db.execute(
                "SELECT record FROM nodes WHERE root_id = ? ORDER BY seq", (root_id,)
            ).fetchall()
        # An empty result would look like a tree with no nodes, which cannot exist.
        if not rows:
            raise TreeStoreError(f"no tree with root id {root_id!r}")
        return tuple(TreeNode.model_validate_json(row[0]) for row in rows)

    def record_budget(self, budget: BudgetUse) -> BudgetUse:
        with self._transaction():
            self._db.execute(
                "INSERT OR REPLACE INTO budgets (run_id, record) VALUES (?, ?)",
                (budget.run_id, budget.model_dump_json()),
            )
        return budget

    def read_budget(self, run_id: str) -> BudgetUse | None:
        """The run's last recorded budget use, or None if it has recorded none yet."""
        with self._lock:
            row = self._db.execute(
                "SELECT record FROM budgets WHERE run_id = ?", (run_id,)
            ).fetchone()
        return BudgetUse.model_validate_json(row[0]) if row else None

    def close(self) -> None:
        with self._lock:
            self._db.close()
