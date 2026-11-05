"""The tree store as a local stdio MCP server.

Tree-of-Thought state lives behind MCP, in its own process, so that every node, score and
pruned branch stays auditable whichever agent wrote it (the submitted architecture commits to
this). The tools are thin: the rules are in store.py.

Started by client.py as `python -m sec_agent.tree_mcp.server --db <path>`. The path is required
rather than read from settings: the client's environment is deliberately not passed down, so
this process never sees the API key and has no settings of its own to disagree with.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import sec_agent
from sec_agent.schemas import Record, TreeNode
from sec_agent.tree_mcp.store import BudgetUse, NodeUpdate, TreeStore, TreeStoreError


class ServerInfo(Record):
    """Which code and file this server is using."""

    package_path: str
    db_path: str


@contextmanager
def _refusals_reach_the_client() -> Iterator[None]:
    # The MCP SDK treats any other exception as a crash: it logs a traceback and sends the client
    # a generic message. A refusal is an expected answer, and its message says what was wrong.
    try:
        yield
    except TreeStoreError as error:
        raise ToolError(str(error)) from error


def build_server(store: TreeStore, db_path: Path) -> MCPServer:
    # WARNING because INFO logs a line per request to stderr, which is the run's console.
    server = MCPServer("sec-agent-tree-store", log_level="WARNING")

    @server.tool()
    @_refusals_reach_the_client()
    def create_tree(root: TreeNode) -> TreeNode:
        """Create a tree from its depth 0 root node. The root's node_id is the tree's id."""
        return store.create_tree(root)

    @server.tool()
    @_refusals_reach_the_client()
    def create_node(node: TreeNode) -> TreeNode:
        """Add a node under an existing parent in the same run and change, one level down."""
        return store.create_node(node)

    @server.tool()
    @_refusals_reach_the_client()
    def update_node(node_id: str, update: NodeUpdate) -> TreeNode:
        """Change a node's score, status, prune reason, evidence or metrics; returns the node."""
        return store.update_node(node_id, update)

    @server.tool()
    @_refusals_reach_the_client()
    def read_node(node_id: str) -> TreeNode:
        """Read one node."""
        return store.read_node(node_id)

    @server.tool()
    @_refusals_reach_the_client()
    def read_tree(root_id: str) -> list[TreeNode]:
        """Read every node of a tree, pruned branches included, each parent before its children."""
        return list(store.read_tree(root_id))

    @server.tool()
    def record_budget(budget: BudgetUse) -> BudgetUse:
        """Save a run's budget use, replacing its previous snapshot."""
        return store.record_budget(budget)

    @server.tool()
    def read_budget(run_id: str) -> BudgetUse | None:
        """Read a run's last budget snapshot, or null if it has recorded none."""
        return store.read_budget(run_id)

    @server.tool()
    def server_info() -> ServerInfo:
        """Report the code and database in use. The checkouts share one virtualenv, so a server
        started without the right path would quietly run another checkout's code."""
        return ServerInfo(package_path=sec_agent.__file__, db_path=str(db_path))

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree-of-Thought tree store (stdio MCP server)")
    parser.add_argument("--db", type=Path, required=True, help="SQLite file, on local disk")
    db_path = parser.parse_args().db
    store = TreeStore(db_path)
    try:
        build_server(store, db_path).run("stdio")
    finally:
        store.close()


if __name__ == "__main__":
    main()
