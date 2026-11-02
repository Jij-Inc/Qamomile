"""Regression tests for Qamomile's published Python type contracts."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def test_function_ref_rejects_invalid_calls_with_zuban() -> None:
    """The source decorator must retain its original callable signature."""

    sibling_checker = Path(sys.executable).with_name("zuban")
    checker = sibling_checker if sibling_checker.is_file() else shutil.which("zuban")
    assert checker is not None, "zuban is required by the development environment"
    repository = Path(__file__).parents[2]
    fixture = repository / "tests/typing_cases/qamomile_qkernel_invalid.py"
    completed = subprocess.run(
        [checker, "check", "--show-error-codes", str(fixture)],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    diagnostics = completed.stdout + completed.stderr
    assert completed.returncode == 1, diagnostics
    assert diagnostics.count("[arg-type]") == 2, diagnostics
    assert diagnostics.count("[call-arg]") == 1, diagnostics
    assert diagnostics.count("[assignment]") == 1, diagnostics
