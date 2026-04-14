"""CUDA-Q backend for Qamomile.

This module provides transpiler, executor, and observable conversion
for the CUDA-Q quantum computing platform.

All circuits are emitted through a unified ``CudaqKernelEmitter`` codegen
path, producing ``CudaqKernelArtifact`` instances.  The artifact's
``ExecutionMode`` (STATIC or RUNNABLE) determines which CUDA-Q runtime
API is used for execution.

Public symbols are exported lazily so that a mere ``import qamomile.cudaq``
does not fail in environments where the ``cudaq`` package is absent.  Accessing
any public symbol (e.g. ``qamomile.cudaq.CudaqTranspiler``) will raise an
:class:`ImportError` with actionable install guidance when ``cudaq`` is not
available.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qamomile.cudaq.emitter import (
        CudaqKernelArtifact,
        CudaqKernelEmitter,
        ExecutionMode,
    )
    from qamomile.cudaq.observable import hamiltonian_to_cudaq_spin_op
    from qamomile.cudaq.transpiler import (
        BoundCudaqKernelArtifact,
        CudaqEmitPass,
        CudaqExecutor,
        CudaqTranspiler,
    )

_CUDAQ_MISSING_MSG = """\
CUDA-Q backend requires the `cudaq` package.
Install with `pip install qamomile[cudaq-cu12]` for CUDA 12 or `pip install qamomile[cudaq-cu13]` for CUDA 13.
CUDA-Q currently supports Linux, macOS ARM64 (Apple silicon), and Windows via WSL2.
Native Windows is not supported.
See: https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html\
"""

# Mapping from public symbol name to the submodule that defines it.
_SYMBOL_TO_MODULE: dict[str, str] = {
    # New unified types
    "ExecutionMode": "qamomile.cudaq.emitter",
    "CudaqKernelArtifact": "qamomile.cudaq.emitter",
    "CudaqKernelEmitter": "qamomile.cudaq.emitter",
    "BoundCudaqKernelArtifact": "qamomile.cudaq.transpiler",
    # Transpiler and executor
    "CudaqTranspiler": "qamomile.cudaq.transpiler",
    "CudaqExecutor": "qamomile.cudaq.transpiler",
    "CudaqEmitPass": "qamomile.cudaq.transpiler",
    # Observable
    "hamiltonian_to_cudaq_spin_op": "qamomile.cudaq.observable",
}

__all__ = [
    # New unified types
    "ExecutionMode",
    "CudaqKernelArtifact",
    "CudaqKernelEmitter",
    "BoundCudaqKernelArtifact",
    # Transpiler and executor
    "CudaqTranspiler",
    "CudaqExecutor",
    "CudaqEmitPass",
    # Observable
    "hamiltonian_to_cudaq_spin_op",
]


def __getattr__(name: str) -> object:
    """Lazily resolve public CUDA-Q symbols.

    Raises :class:`AttributeError` for unknown names and :class:`ImportError`
    with install guidance when the ``cudaq`` package is not present.

    Args:
        name: Attribute name being accessed on this module.

    Returns:
        The requested symbol, cached into module globals for subsequent access.

    Raises:
        AttributeError: If *name* is not a known public symbol of this module.
        ImportError: If the ``cudaq`` package is not installed.
    """
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module 'qamomile.cudaq' has no attribute {name!r}")

    if importlib.util.find_spec("cudaq") is None:
        raise ImportError(_CUDAQ_MISSING_MSG)

    mod = importlib.import_module(_SYMBOL_TO_MODULE[name])
    obj = getattr(mod, name)
    # Cache into module globals to avoid repeated __getattr__ calls.
    globals()[name] = obj
    return obj
