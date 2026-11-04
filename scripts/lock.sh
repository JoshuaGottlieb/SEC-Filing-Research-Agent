#!/usr/bin/env bash
# Regenerate uv.lock on local disk, then copy it back.
#
# `uv lock` writes a temp file and renames it, which fails on VirtualBox shared
# folders with "Text file busy". Use this instead of `uv lock` after changing
# dependencies, then install with `uv sync --frozen`.
# Stop on the first error, on unset variables, and on a failure inside a pipe.
set -euo pipefail

# The repository root: the folder above this script, wherever the script is run from.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# A scratch folder on local disk, where renaming files works.
work_dir="$(mktemp -d)"
# Delete the scratch folder when the script ends, whether it succeeded or failed.
trap 'rm -rf "$work_dir"' EXIT

# Copy only what `uv lock` reads. pyproject.toml names README.md as the project readme and
# requires Python 3.12, which .python-version pins, so uv needs all three.
cp "$repo_root/pyproject.toml" "$repo_root/.python-version" "$repo_root/README.md" "$work_dir/"
# Start from the existing lock, so unchanged dependencies keep their locked versions.
if [[ -f "$repo_root/uv.lock" ]]; then
    cp "$repo_root/uv.lock" "$work_dir/"
fi

# Extra arguments go straight to `uv lock` (for example --upgrade).
(cd "$work_dir" && uv lock "$@")
# A plain copy works on the shared folder, where the rename `uv lock` does would fail.
cp "$work_dir/uv.lock" "$repo_root/uv.lock"
echo "Wrote $repo_root/uv.lock"
