"""Prove the rules didn't change during a fair test.

    python eval/ingestion/fingerprint.py save eval/ingestion/holdout3.sha256
    python eval/ingestion/fingerprint.py check eval/ingestion/holdout3.sha256

Exits 1 if any rule file differs from the saved fingerprint.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import typer

from sec_agent.config import REPO_ROOT

# The source files whose behaviour decides how a filing is parsed, validated and stored. A
# change to any of them could change results, so it counts as changing the rules. Tests,
# docs and eval scripts are left out: they don't affect the outcome being measured.
RULE_FILES = ("ingest/*.py", "edgar/*.py", "finance/*.py", "cli.py")


def fingerprints() -> dict[str, str]:
    """A SHA-256 hash for each rule file, keyed by its path from the repository root."""
    source = REPO_ROOT / "src" / "sec_agent"
    paths = sorted(path for pattern in RULE_FILES for path in source.glob(pattern))
    return {
        str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def main(action: Literal["save", "check"], path: Path) -> None:
    current = fingerprints()
    if action == "save":
        path.write_text("".join(f"{digest}  {name}\n" for name, digest in current.items()))
        typer.echo(f"Saved {len(current)} fingerprints to {path}")
        return
    # Lines are "<hash>  <path>", the same layout `sha256sum` writes.
    saved = dict(reversed(line.split("  ", 1)) for line in path.read_text().splitlines())
    # A file that was added or removed counts as changed too.
    changed = sorted(
        name for name in saved.keys() | current.keys() if saved.get(name) != current.get(name)
    )
    for name in changed:
        typer.echo(f"CHANGED {name}")
    typer.echo(f"{len(current)} files checked, {len(changed)} changed.")
    if changed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
