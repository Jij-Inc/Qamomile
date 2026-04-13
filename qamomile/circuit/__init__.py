from __future__ import annotations

from typing import TYPE_CHECKING

# Frontend API
from .frontend.composite_gate import CompositeGate, composite_gate
from .frontend.constructors import bit, float_, qubit, qubit_array, uint
from .frontend.handle import (
    Bit,
    Dict,
    Float,
    Handle,
    Matrix,
    Observable,
    QFixed,
    Qubit,
    Tensor,
    Tuple,
    UInt,
    Vector,
)
from .frontend.operation.cast import cast
from .frontend.operation.control_flow import for_items, items, range
from .frontend.operation.controlled import controlled
from .frontend.operation.expval import expval
from .frontend.operation.measurement import measure
from .frontend.operation.pauli_evolve import pauli_evolve
from .frontend.operation.qubit_gates import (
    ccx,
    cp,
    cx,
    cz,
    h,
    p,
    rx,
    ry,
    rz,
    rzz,
    s,
    sdg,
    swap,
    t,
    tdg,
    x,
    y,
    z,
)
from .frontend.qkernel import QKernel, qkernel

# Algorithm building blocks
from .algorithm import (
    amplitude_encoding,
    compute_mottonen_amplitude_encoding_thetas,
    compute_mottonen_thetas,
)

# Standard library circuits
from .stdlib import iqft, qft, qpe

if TYPE_CHECKING:
    from .visualization import DEFAULT_STYLE, CircuitStyle, MatplotlibDrawer

_VISUALIZATION_NAMES = {"MatplotlibDrawer", "CircuitStyle", "DEFAULT_STYLE"}


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in _VISUALIZATION_NAMES:
        from .visualization import DEFAULT_STYLE, CircuitStyle, MatplotlibDrawer

        globals().update(
            {
                "MatplotlibDrawer": MatplotlibDrawer,
                "CircuitStyle": CircuitStyle,
                "DEFAULT_STYLE": DEFAULT_STYLE,
            }
        )
        return globals()[name]
    raise AttributeError(f"module 'qamomile.circuit' has no attribute {name!r}")


__all__ = [
    "qkernel",
    "composite_gate",
    "CompositeGate",
    "controlled",
    "cast",
    "bit",
    "float_",
    "qubit",
    "uint",
    "qubit_array",
    "h",
    "x",
    "y",
    "z",
    "t",
    "tdg",
    "s",
    "sdg",
    "cx",
    "cz",
    "ccx",
    "p",
    "rx",
    "ry",
    "rz",
    "rzz",
    "cp",
    "swap",
    "measure",
    "expval",
    "pauli_evolve",
    "for_items",
    "items",
    "range",
    "Bit",
    "Dict",
    "Float",
    "Handle",
    "Qubit",
    "QFixed",
    "Tuple",
    "UInt",
    "Vector",
    "Matrix",
    "Tensor",
    "Observable",
    # stdlib
    "qpe",
    "iqft",
    "qft",
    "QKernel",
    # Algorithm
    "amplitude_encoding",
    "compute_mottonen_amplitude_encoding_thetas",
    "compute_mottonen_thetas",
    # Visualization (lazy-loaded)
    "MatplotlibDrawer",
    "CircuitStyle",
    "DEFAULT_STYLE",
]
