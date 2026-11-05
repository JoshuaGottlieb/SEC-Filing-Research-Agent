"""The tree store: its rules, the MCP tools over it, and the real server subprocess.

Most tests run twice, once through the in-process client the controller's tests will use and
once through the real MCP tools over an in-memory connection, so the two cannot drift apart.
One test starts the actual stdio server. It is local and free.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp.client import Client
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

import sec_agent
from sec_agent.schemas import (
    Direction,
    MetricExpectation,
    MetricName,
    NodeStatus,
    NumericCheck,
    TreeNode,
)
from sec_agent.tree_mcp.client import (
    InProcessTreeClient,
    McpTreeClient,
    TreeClient,
    connect_tree_store,
    server_parameters,
)
from sec_agent.tree_mcp.server import build_server
from sec_agent.tree_mcp.store import BudgetUse, NodeUpdate, TreeStore, TreeStoreError


def root(node_id: str = "root", run_id: str = "run-1", change_id: str = "chg-1") -> TreeNode:
    return TreeNode(
        node_id=node_id,
        run_id=run_id,
        change_id=change_id,
        depth=0,
        thought="Gross margin fell 200 basis points",
    )


def child(node_id: str, parent_id: str = "root", depth: int = 1, **fields: Any) -> TreeNode:
    return TreeNode(
        **{
            "node_id": node_id,
            "run_id": "run-1",
            "change_id": "chg-1",
            "parent_id": parent_id,
            "depth": depth,
            "thought": f"Explanation {node_id}",
            **fields,
        }
    )


@pytest.fixture
def store(tmp_path: Path) -> TreeStore:
    return TreeStore(tmp_path / "tree.sqlite")


@pytest.fixture(params=["in_process", "mcp_tools"])
def connect(request: pytest.FixtureRequest, store: TreeStore, tmp_path: Path) -> Callable:
    # A context manager, not a yielding async fixture: the MCP client must be closed in the task
    # that opened it, and pytest-asyncio does not promise that for fixture teardown.
    @asynccontextmanager
    async def open_client() -> AsyncIterator[TreeClient]:
        if request.param == "in_process":
            yield InProcessTreeClient(store)
            return
        async with Client(build_server(store, tmp_path / "tree.sqlite")) as client:
            yield McpTreeClient(client)

    return open_client


async def test_both_clients_satisfy_the_interface(connect):
    async with connect() as client:
        assert isinstance(client, TreeClient)


async def test_a_node_reads_back_exactly_as_written(connect):
    # Covers what JSON could change: tuples, nested records and enums.
    node = child(
        "c1",
        expected_evidence="MD&A names input costs",
        query_plan="Search MD&A for cocoa",
        expected_metrics=(
            MetricExpectation(metric=MetricName.GROSS_MARGIN, direction=Direction.DOWN),
        ),
    )
    async with connect() as client:
        assert await client.create_tree(root()) == root()
        assert await client.create_node(node) == node
        assert await client.read_node("c1") == node
        assert await client.read_node("root") == root()


async def test_a_tree_must_start_at_a_root(connect):
    async with connect() as client:
        await client.create_tree(root())
        with pytest.raises(TreeStoreError, match="depth 0"):
            await client.create_tree(child("c1"))


async def test_create_node_refuses_a_root(connect):
    async with connect() as client:
        with pytest.raises(TreeStoreError, match="create_tree"):
            await client.create_node(root())


async def test_a_duplicate_id_is_refused_and_the_store_keeps_working(connect):
    async with connect() as client:
        await client.create_tree(root())
        with pytest.raises(TreeStoreError, match="already exists"):
            await client.create_tree(root())
        await client.create_node(child("c1"))
        with pytest.raises(TreeStoreError, match="already exists"):
            await client.create_node(child("c1", thought="A different thought"))
        # The first write stands, and the failed ones left no transaction open.
        assert (await client.read_node("c1")).thought == "Explanation c1"
        await client.create_node(child("c2"))


async def test_a_node_must_fit_under_its_parent(connect):
    async with connect() as client:
        await client.create_tree(root())
        with pytest.raises(TreeStoreError, match="no node"):
            await client.create_node(child("c1", parent_id="missing"))
        with pytest.raises(TreeStoreError, match="same run and change"):
            await client.create_node(child("c1", run_id="run-2"))
        with pytest.raises(TreeStoreError, match="same run and change"):
            await client.create_node(child("c1", change_id="chg-2"))
        # A depth 2 node directly under the root skips a level.
        with pytest.raises(TreeStoreError, match="depth 2 under a parent at depth 0"):
            await client.create_node(child("c1", depth=2))
        # None of the refused nodes were stored.
        assert await client.read_tree("root") == (root(),)


async def test_update_changes_only_the_fields_it_sets(connect):
    async with connect() as client:
        await client.create_tree(root())
        await client.create_node(child("c1", query_plan="Search MD&A"))
        await client.create_node(child("e1", parent_id="c1", depth=2))
        await client.update_node("e1", NodeUpdate(retrieved_chunk_ids=("a", "b")))
        updated = await client.update_node(
            "e1",
            NodeUpdate(
                score=0.8,
                status=NodeStatus.KEPT,
                best_passage_score=4.5,
                metric_ids=("m1",),
                numeric_check=NumericCheck.AGREE,
            ),
        )
        assert updated == child(
            "e1",
            parent_id="c1",
            depth=2,
            retrieved_chunk_ids=("a", "b"),
            score=0.8,
            status=NodeStatus.KEPT,
            best_passage_score=4.5,
            metric_ids=("m1",),
            numeric_check=NumericCheck.AGREE,
        )
        assert await client.read_node("e1") == updated
        # The parent was not touched.
        assert (await client.read_node("c1")).query_plan == "Search MD&A"


async def test_update_can_clear_a_field(connect):
    # An explicit None is a value, not "leave it alone".
    async with connect() as client:
        await client.create_tree(root())
        await client.update_node("root", NodeUpdate(score=0.5))
        cleared = await client.update_node("root", NodeUpdate(score=None))
        assert cleared.score is None


@pytest.mark.parametrize(
    "update",
    [
        NodeUpdate(status=NodeStatus.PRUNED),  # a pruned node must record why
        NodeUpdate(score=1.5),  # scores are 0 to 1
        NodeUpdate(status=None),  # a node always has a status
    ],
)
async def test_an_update_that_breaks_the_record_is_refused(connect, update):
    async with connect() as client:
        await client.create_tree(root())
        with pytest.raises(TreeStoreError, match="invalid update"):
            await client.update_node("root", update)
        assert await client.read_node("root") == root()


async def test_update_of_an_unknown_node_is_refused(connect):
    async with connect() as client:
        with pytest.raises(TreeStoreError, match="no node"):
            await client.update_node("missing", NodeUpdate(score=0.5))


@pytest.mark.parametrize(
    "update",
    [NodeUpdate(status=NodeStatus.KEPT), NodeUpdate(prune_reason="A different reason")],
)
async def test_a_pruned_node_keeps_its_status_and_reason(connect, update):
    async with connect() as client:
        await client.create_tree(root())
        await client.create_node(child("c1"))
        await client.update_node(
            "c1", NodeUpdate(status=NodeStatus.PRUNED, prune_reason="numbers contradict")
        )
        with pytest.raises(TreeStoreError, match="final"):
            await client.update_node("c1", update)
        stored = await client.read_node("c1")
        assert stored.status is NodeStatus.PRUNED
        assert stored.prune_reason == "numbers contradict"


async def test_a_pruned_node_can_still_record_evidence(connect):
    # A score or metric worked out for the audit trail is not a change to the decision.
    async with connect() as client:
        await client.create_tree(root())
        await client.create_node(child("c1"))
        await client.update_node("c1", NodeUpdate(status=NodeStatus.PRUNED, prune_reason="low"))
        updated = await client.update_node("c1", NodeUpdate(score=0.1))
        assert updated.score == 0.1
        assert updated.status is NodeStatus.PRUNED


async def test_read_tree_returns_the_whole_tree_with_pruned_branches(connect):
    async with connect() as client:
        await client.create_tree(root())
        await client.create_tree(root("other-root", change_id="chg-2"))
        await client.create_node(child("c1"))
        await client.create_node(child("c2"))
        await client.create_node(child("e1", parent_id="c1", depth=2))
        await client.create_node(
            child("other", parent_id="other-root", change_id="chg-2", thought="Another tree")
        )
        await client.update_node(
            "c2", NodeUpdate(status=NodeStatus.PRUNED, prune_reason="below threshold")
        )

        tree = await client.read_tree("root")

        assert [node.node_id for node in tree] == ["root", "c1", "c2", "e1"]
        assert tree[2].status is NodeStatus.PRUNED
        assert tree[2].prune_reason == "below threshold"


@pytest.mark.parametrize("root_id", ["missing", "c1"])
async def test_read_tree_needs_a_root_id(connect, root_id):
    async with connect() as client:
        await client.create_tree(root())
        await client.create_node(child("c1"))
        with pytest.raises(TreeStoreError, match="no tree"):
            await client.read_tree(root_id)


async def test_read_node_of_an_unknown_id_is_refused(connect):
    async with connect() as client:
        with pytest.raises(TreeStoreError, match="no node"):
            await client.read_node("missing")


async def test_budget_snapshots_replace_each_other_per_run(connect):
    first = BudgetUse(run_id="run-1", spent_usd=0.5, max_usd=10.0, node_evaluations=6)
    later = BudgetUse(run_id="run-1", spent_usd=1.25, max_usd=10.0, node_evaluations=30)
    other = BudgetUse(run_id="run-2", spent_usd=0.0, max_usd=5.0, node_evaluations=0)
    async with connect() as client:
        assert await client.read_budget("run-1") is None
        assert await client.record_budget(first) == first
        assert await client.record_budget(later) == later
        await client.record_budget(other)
        assert await client.read_budget("run-1") == later
        assert await client.read_budget("run-2") == other


def test_a_negative_budget_is_not_a_record():
    with pytest.raises(ValidationError):
        BudgetUse(run_id="run-1", spent_usd=-1.0, max_usd=10.0, node_evaluations=0)


def test_the_store_survives_being_closed_and_reopened(tmp_path):
    path = tmp_path / "nested" / "tree.sqlite"
    store = TreeStore(path)
    store.create_tree(root())
    store.create_node(child("c1"))
    budget = BudgetUse(run_id="run-1", spent_usd=0.5, max_usd=10.0, node_evaluations=1)
    store.record_budget(budget)
    store.close()

    reopened = TreeStore(path)
    assert reopened.read_tree("root") == (root(), child("c1"))
    assert reopened.read_budget("run-1") == budget
    reopened.close()


def test_a_tampered_row_fails_on_the_way_out(tmp_path):
    path = tmp_path / "tree.sqlite"
    store = TreeStore(path)
    store.create_tree(root())
    record = json.loads(root().model_dump_json()) | {"unexpected": "field"}
    with sqlite3.connect(path) as db:
        db.execute("UPDATE nodes SET record = ?", (json.dumps(record),))
    db.close()

    with pytest.raises(ValidationError):
        store.read_node("root")
    with pytest.raises(ValidationError):
        store.read_tree("root")


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        # What a node claims cannot be rewritten after the fact.
        ("update_node", {"node_id": "root", "update": {"thought": "Rewritten"}}),
        ("update_node", {"node_id": "root", "update": {"depth": 1}}),
        # The records forbid unknown fields, and that holds at the tool boundary too.
        ("create_tree", {"root": root().model_dump(mode="json") | {"unexpected": 1}}),
        # A pruned node with no reason cannot arrive through create_node either.
        (
            "create_node",
            {
                "node": child("c1").model_dump(mode="json")
                | {"status": "pruned", "prune_reason": None}
            },
        ),
    ],
)
async def test_the_tools_validate_what_comes_in(store, tmp_path, tool, arguments):
    # Calls the tools directly, past the typed client, as any other MCP client could.
    store.create_tree(root())
    async with Client(build_server(store, tmp_path / "tree.sqlite")) as client:
        result = await client.call_tool(tool, arguments)
    assert result.is_error
    assert store.read_tree("root") == (root(),)


async def test_a_connection_failure_becomes_a_store_error():
    class BrokenClient:
        async def call_tool(self, tool: str, arguments: dict[str, Any]) -> None:
            raise MCPError(-32000, "Connection closed")

    with pytest.raises(TreeStoreError, match="Connection closed"):
        await McpTreeClient(BrokenClient()).read_node("root")


def test_the_server_is_started_with_this_checkouts_code(tmp_path):
    package_root = str(Path(sec_agent.__file__).resolve().parents[1])
    parameters = server_parameters(tmp_path / "tree.sqlite")
    assert parameters.env == {"PYTHONPATH": package_root}
    assert parameters.cwd == package_root
    # Absolute, because the server does not run in the caller's working directory.
    assert parameters.args[-1] == str((tmp_path / "tree.sqlite").resolve())


async def test_real_server_round_trip_and_reload(tmp_path):
    """Everything through the actual stdio subprocess, then a new process reloads the tree."""
    path = tmp_path / "tree.sqlite"
    budget = BudgetUse(run_id="run-1", spent_usd=0.25, max_usd=10.0, node_evaluations=3)

    async with connect_tree_store(path) as client:
        # The subprocess gets a bare environment. It must still run this checkout's code, not
        # the main checkout's through the shared virtualenv's editable install.
        info = await client.server_info()
        assert Path(info["package_path"]).resolve() == Path(sec_agent.__file__).resolve()
        assert Path(info["db_path"]) == path.resolve()

        await client.create_tree(root())
        await client.create_node(child("kept"))
        await client.create_node(child("cut"))
        await client.create_node(child("evidence", parent_id="kept", depth=2))
        await client.update_node(
            "kept",
            NodeUpdate(
                score=0.7,
                status=NodeStatus.KEPT,
                numeric_check=NumericCheck.AGREE,
                metric_ids=("m1",),
            ),
        )
        await client.update_node(
            "cut",
            NodeUpdate(score=0.1, status=NodeStatus.PRUNED, prune_reason="below threshold"),
        )
        await client.record_budget(budget)
        assert await client.read_budget("run-2") is None
        # A refusal crosses the process boundary as the same exception, with the store's message.
        with pytest.raises(TreeStoreError, match="already exists"):
            await client.create_tree(root())
        written = await client.read_tree("root")

    async with connect_tree_store(path) as client:
        reloaded = await client.read_tree("root")
        assert reloaded == written
        assert [node.node_id for node in reloaded] == ["root", "kept", "cut", "evidence"]
        assert reloaded[2].status is NodeStatus.PRUNED
        assert reloaded[2].prune_reason == "below threshold"
        assert await client.read_budget("run-1") == budget
