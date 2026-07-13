"""Quration target integration through its PyQret Python frontend.

Quration is the user-facing FTQC resource-estimation toolchain; PyQret is
its Python construction API. The public transpiler is therefore named
``QurationTranspiler`` while the target-native materializer is explicitly
named ``PyQretMaterializer``.

This backend depends only on ``qamomile.circuit`` and optional ``pyqret``.
It consumes backend-neutral ``CircuitProgram`` artifacts and never reaches
back into Qamomile's semantic IR.
"""

from qamomile.quration.materializer import PyQretMaterializer
from qamomile.quration.transpiler import (
    QurationExecutor,
    QurationResourceResult,
    QurationTranspiler,
)

__all__ = [
    "PyQretMaterializer",
    "QurationExecutor",
    "QurationResourceResult",
    "QurationTranspiler",
]
