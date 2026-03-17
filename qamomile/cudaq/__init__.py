"""CUDA-Q backend for Qamomile.

This module provides transpiler, executor, and observable conversion
for the CUDA-Q quantum computing platform.

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
    from qamomile.cudaq.transpiler import (
        CudaqTranspiler,
        CudaqExecutor,
        CudaqEmitPass,
        BoundCudaqCircuit,
    )
    from qamomile.cudaq.emitter import CudaqGateEmitter, CudaqCircuit
    from qamomile.cudaq.observable import hamiltonian_to_cudaq_spin_op

_CUDAQ_MISSING_MSG = """\
CUDA-Q backend requires the `cudaq` package.
Install with `pip install qamomile[cudaq]`.
CUDA-Q currently supports Linux, macOS ARM64 (Apple silicon), and Windows via WSL2.
Native Windows is not supported.
See: https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html\
"""

# Mapping from public symbol name to the submodule that defines it.
_SYMBOL_TO_MODULE: dict[str, str] = {
    "CudaqTranspiler": "qamomile.cudaq.transpiler",
    "CudaqExecutor": "qamomile.cudaq.transpiler",
    "CudaqEmitPass": "qamomile.cudaq.transpiler",
    "BoundCudaqCircuit": "qamomile.cudaq.transpiler",
    "CudaqGateEmitter": "qamomile.cudaq.emitter",
    "CudaqCircuit": "qamomile.cudaq.emitter",
    "hamiltonian_to_cudaq_spin_op": "qamomile.cudaq.observable",
}

__all__ = [
    "CudaqTranspiler",
    "CudaqExecutor",
    "CudaqEmitPass",
    "BoundCudaqCircuit",
    "CudaqGateEmitter",
    "CudaqCircuit",
    "hamiltonian_to_cudaq_spin_op",
]


def __getattr__(name: str) -> object:
    """Lazily resolve public CUDA-Q symbols.

    Raises :class:`AttributeError` for unknown names and :class:`ImportError`
    with install guidance when the ``cudaq`` package is not present.

    Parameters
    ----------
    name:
        Attribute name being accessed on this module.

    Returns
    -------
    object
        The requested symbol, cached into module globals for subsequent access.

    Raises
    ------
    AttributeError
        If *name* is not a known public symbol of this module.
    ImportError
        If the ``cudaq`` package is not installed.
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
