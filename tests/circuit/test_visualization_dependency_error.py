"""Diagnostics for optional circuit-visualization dependencies."""

import builtins
import sys
from typing import Any

import pytest

import qamomile.circuit as qmc


def test_draw_reports_visualization_extra_when_matplotlib_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernel drawing replaces a raw matplotlib import failure with guidance."""

    @qmc.qkernel
    def kernel() -> qmc.Qubit:
        return qmc.qubit("q")

    for name in tuple(sys.modules):
        if name == "qamomile.circuit.visualization" or name.startswith(
            "qamomile.circuit.visualization."
        ):
            monkeypatch.delitem(sys.modules, name)

    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """Raise as if matplotlib were absent while delegating other imports."""
        if name.startswith("matplotlib"):
            raise ModuleNotFoundError(
                "No module named 'matplotlib'",
                name="matplotlib",
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match=r"qamomile\[visualization\]"):
        kernel.draw()
