#!/usr/bin/env bash
# Pre-commit hook entry: regenerate doc tag pages and re-sync changed
# ipynb from their .py sources.
#
# Wiring lives in .pre-commit-config.yaml. The hook fires whenever a .py
# file under docs/{en,ja}/{tutorial,algorithm,optimization,collaboration}/
# is staged. pre-commit hands us those changed files as positional args.
#
# Steps:
#   1. Run build_doc_tags.py to regenerate tag indexes + per-tag pages,
#      refresh inline chip blocks in every tagged .py, and update the
#      auto-managed Tags region inside myst.yml.
#   2. Re-sync .ipynb from .py for every changed file via
#      `jupytext --to ipynb --update` (preserves outputs).
#
# If the hook ends up modifying tracked files, pre-commit will detect
# that and instruct the user to re-stage and commit again.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

uv run python docs/scripts/build_doc_tags.py

if [ "$#" -gt 0 ]; then
    uv run jupytext --to ipynb --update "$@"
fi
