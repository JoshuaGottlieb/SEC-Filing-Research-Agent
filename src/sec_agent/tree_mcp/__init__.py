"""The tree store: a local MCP server over SQLite that keeps every Tree-of-Thought tree (T4.1).

- `store.py`: the SQLite tables for nodes, trees and each run's budget use. Pruned branches are kept
  with their reasons.
- `server.py`: the store as a stdio MCP server.
- `client.py`: what the controller uses to reach it.
"""
