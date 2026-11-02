"""Configure repository-wide pytest collection and command-line options.

The collection hook isolates the CUDA-Q runtime, which drags in torch and an
OpenMP runtime. The configure hook bounds CUDA-Q native thread pools and gives
Matplotlib and Zig persistent pytest caches. Qamomile backend workload bounds live in
``tests/qamomile/conftest.py`` beside the single active implementation. The
command-line options let documentation CI select only the runnable pages
changed by a docs-only pull request.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests._cudaq_isolation import (
    CUDAQ_MODULE_LEVEL_IMPORTERS,
    configure_cudaq_thread_limits,
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
        help="Repository-relative path changed by the pull request.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Install repository-wide caches and simulator resource bounds.

    CUDA-Q loads an OpenMP runtime and its simulator may otherwise consume
    every available CPU core. Dedicated ``-m cudaq`` runs contain many
    simulation cases, so unconstrained native pools can saturate several
    cores for minutes. Explicit environment settings remain authoritative.
    Matplotlib uses pytest's persistent cache so isolated drawer tests do not
    rebuild the font index on every process. Selene configures Zig's local cache
    itself; its child process inherits this writable global cache instead of
    requiring access to the user's home cache during sandboxed tests.

    Args:
        config (pytest.Config): Active pytest configuration containing the
            marker expression.
    """
    os.environ.setdefault("MPLCONFIGDIR", str(config.cache.mkdir("matplotlib")))
    if "ZIG_GLOBAL_CACHE_DIR" not in os.environ:
        os.environ["ZIG_GLOBAL_CACHE_DIR"] = str(config.cache.mkdir("zig-global"))
    configure_cudaq_thread_limits(config.getoption("markexpr") or "")


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
