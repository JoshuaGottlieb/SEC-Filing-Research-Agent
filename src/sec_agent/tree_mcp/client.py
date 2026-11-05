"""What the controller uses to reach the tree store.

`TreeClient` is the interface. `McpTreeClient` is the real one, talking to the server process
over MCP. `InProcessTreeClient` calls the store directly, so controller tests can run the same
rules without a subprocess.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.shared.exceptions import MCPError
from mcp_types import TextContent

import sec_agent
from sec_agent.schemas import TreeNode
from sec_agent.tree_mcp.store import BudgetUse, NodeUpdate, TreeStore, TreeStoreError

# Every call is one local SQLite operation, so anything this slow means the server has hung.
# Failing lets the run stop with an error instead of waiting forever. Chosen by hand.
CALL_TIMEOUT_SECONDS = 30.0


@runtime_checkable
class TreeClient(Protocol):
    """The tree store's operations. All of them raise `TreeStoreError` on a refused request."""

    async def create_tree(self, root: TreeNode) -> TreeNode: ...

    async def create_node(self, node: TreeNode) -> TreeNode: ...

    async def update_node(self, node_id: str, update: NodeUpdate) -> TreeNode: ...

    async def read_node(self, node_id: str) -> TreeNode: ...

    async def read_tree(self, root_id: str) -> tuple[TreeNode, ...]: ...

    async def record_budget(self, budget: BudgetUse) -> BudgetUse: ...

    async def read_budget(self, run_id: str) -> BudgetUse | None: ...


class InProcessTreeClient:
    """Runs the store in this process. For tests: production goes through the MCP server."""

    def __init__(self, store: TreeStore) -> None:
        self._store = store

    async def create_tree(self, root: TreeNode) -> TreeNode:
        return self._store.create_tree(root)

    async def create_node(self, node: TreeNode) -> TreeNode:
        return self._store.create_node(node)

    async def update_node(self, node_id: str, update: NodeUpdate) -> TreeNode:
        return self._store.update_node(node_id, update)

    async def read_node(self, node_id: str) -> TreeNode:
        return self._store.read_node(node_id)

    async def read_tree(self, root_id: str) -> tuple[TreeNode, ...]:
        return self._store.read_tree(root_id)

    async def record_budget(self, budget: BudgetUse) -> BudgetUse:
        return self._store.record_budget(budget)

    async def read_budget(self, run_id: str) -> BudgetUse | None:
        return self._store.read_budget(run_id)


class McpTreeClient:
    """The tree store's tools as typed methods. Get one from `connect_tree_store`."""

    def __init__(self, client: Client) -> None:
        self._client = client

    async def _call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        # Both failure routes become TreeStoreError, so the caller handles one exception type
        # whether the store refused the request or the server itself failed.
        try:
            result = await self._client.call_tool(tool, arguments)
        except MCPError as error:
            raise TreeStoreError(f"tree store call {tool} failed: {error}") from error
        if result.is_error:
            text = " ".join(part.text for part in result.content if isinstance(part, TextContent))
            raise TreeStoreError(text or f"tree store call {tool} failed")
        if result.structured_content is None:
            raise TreeStoreError(f"tree store call {tool} returned no structured result")
        return result.structured_content

    async def _node(self, tool: str, **arguments: Any) -> TreeNode:
        return TreeNode.model_validate(await self._call(tool, **arguments))

    async def create_tree(self, root: TreeNode) -> TreeNode:
        return await self._node("create_tree", root=root.model_dump(mode="json"))

    async def create_node(self, node: TreeNode) -> TreeNode:
        return await self._node("create_node", node=node.model_dump(mode="json"))

    async def update_node(self, node_id: str, update: NodeUpdate) -> TreeNode:
        # exclude_unset keeps "leave this field alone" apart from "set this field to None".
        fields = update.model_dump(mode="json", exclude_unset=True)
        return await self._node("update_node", node_id=node_id, update=fields)

    async def read_node(self, node_id: str) -> TreeNode:
        return await self._node("read_node", node_id=node_id)

    async def read_tree(self, root_id: str) -> tuple[TreeNode, ...]:
        # The MCP SDK wraps a result that is not an object in {"result": ...}.
        nodes = (await self._call("read_tree", root_id=root_id))["result"]
        return tuple(TreeNode.model_validate(node) for node in nodes)

    async def record_budget(self, budget: BudgetUse) -> BudgetUse:
        result = await self._call("record_budget", budget=budget.model_dump(mode="json"))
        return BudgetUse.model_validate(result)

    async def read_budget(self, run_id: str) -> BudgetUse | None:
        budget = (await self._call("read_budget", run_id=run_id))["result"]
        return None if budget is None else BudgetUse.model_validate(budget)

    async def server_info(self) -> dict[str, Any]:
        """Which code and database the server is using. Not part of `TreeClient`: it is a
        diagnostic for the real server, and an in-process store has no separate code to check."""
        return await self._call("server_info")


def server_parameters(db_path: Path) -> StdioServerParameters:
    """How to start the server so that it runs the same code as this process.

    The stdio client passes only a short list of environment variables to the subprocess, and
    PYTHONPATH is not on it. Without this, the server would import `sec_agent` from the shared
    virtualenv's editable install, which points at the main checkout rather than this one.
    """
    # sec_agent/__init__.py -> sec_agent -> the directory that contains the package.
    package_root = str(Path(sec_agent.__file__).resolve().parents[1])
    return StdioServerParameters(
        command=sys.executable,
        # Absolute because the working directory below is not the caller's.
        args=["-m", "sec_agent.tree_mcp.server", "--db", str(db_path.resolve())],
        env={"PYTHONPATH": package_root},
        # `python -m` puts the working directory first on sys.path, ahead of PYTHONPATH, so a
        # caller running from another checkout's src would otherwise load that checkout's code.
        cwd=package_root,
    )


@asynccontextmanager
async def connect_tree_store(db_path: Path) -> AsyncIterator[McpTreeClient]:
    """Start the server on `db_path`, yield a client, and stop the server on exit.

    The connection must be opened and closed in the same asyncio task (the MCP SDK's cancel
    scopes require it), so open it around the whole run, not per graph node.
    """
    parameters = server_parameters(db_path)
    async with Client(parameters, read_timeout_seconds=CALL_TIMEOUT_SECONDS) as client:
        yield McpTreeClient(client)
