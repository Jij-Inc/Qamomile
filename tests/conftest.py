"""Configure repository-wide pytest collection and command-line options.

The collection hook isolates the CUDA-Q runtime, which drags in torch and a
third copy of the OpenMP runtime alongside the copies bundled by torch and
qiskit-aer. See ``tests/_cudaq_isolation.py`` for the full background and
``tests/test_cudaq_import_isolation.py`` for the guard that keeps the module
table in sync. The command-line options let documentation CI select only the
runnable pages changed by a docs-only pull request.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._cudaq_isolation import (
    CUDAQ_MODULE_LEVEL_IMPORTERS,
    markexpr_can_select_cudaq,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register options for selecting changed documentation pages.

    Args:
        parser (pytest.Parser): Parser used to register pytest command-line
            options.
    """
    group = parser.getgroup("documentation")
    group.addoption(
        "--changed-docs",
        action="store_true",
        help="Run executable documentation tests only for --docs-file paths.",
    )
    group.addoption(
        "--docs-file",
        action="append",
        default=[],
        help="Repository-relative documentation path changed by the pull request.",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skip collecting module-level cudaq importers in cudaq-less runs.

    Args:
        collection_path (Path): Candidate path pytest is about to collect.
        config (pytest.Config): The active pytest configuration, used for
            the rootdir and the ``-m`` marker expression.

    Returns:
        bool | None: True to skip the path (a registered cudaq module
            while every cudaq-marked test is deselected); None to leave
            the decision to other plugins.
    """
    try:
        relative = collection_path.relative_to(config.rootpath).as_posix()
    except ValueError:
        return None
    if relative not in CUDAQ_MODULE_LEVEL_IMPORTERS:
        return None
    if markexpr_can_select_cudaq(config.getoption("markexpr") or ""):
        return None
    return True
