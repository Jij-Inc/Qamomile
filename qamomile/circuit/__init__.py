from __future__ import annotations

from typing import TYPE_CHECKING

from .estimator import (
    CallResources,
    CostBasis,
    DepthResources,
    FixedResourceModel,
    GateResources,
    ResourceContext,
    ResourceEstimate,
    ResourceEstimator,
    ResourcePolicy,
    UnknownResourcePolicy,
    WidthResources,
    estimate_resources,
)

# Frontend API
from .frontend.callable_signature import CallableSignature
from .frontend.composite_gate import (
    CompositeGate as CompositeGate,
    composite as composite,
    composite_gate as composite_gate,
)
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
    VectorView,
)
from .frontend.operation.cast import cast
from .frontend.operation.control import control
from .frontend.operation.control_flow import for_items, items, range
from .frontend.operation.expval import expval
from .frontend.operation.inverse import inverse
from .frontend.operation.measurement import (
    measure,
    measure_reset,
    project_x,
    project_y,
    project_z,
    reset,
)
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
from .frontend.oracle import Oracle, opaque
from .frontend.qkernel import QKernel, qkernel

# Standard library circuits
from .stdlib import (
    grover_iteration_count,
    grover_search,
    iqft,
    modmul_const,
    qft,
    qpe,
    shor_order_finding,
)

# Execution result / job types (return values of ExecutableProgram.sample / run)
from .transpiler.job import (
    ExpvalJob,
    Job,
    JobStatus,
    RunJob,
    SampleJob,
    SampleResult,
)

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


# Imported after frontend symbols are initialized because these kernels use
# ``import qamomile.circuit as qmc`` in their implementation module.
from .algorithm.arithmetic.modular_incdec import (  # noqa: E402, I001
    modular_decrement,
    modular_increment,
)


__all__ = [
    "qkernel",
    "composite_gate",
    "Oracle",
    "opaque",
    "CallableSignature",
    "CallResources",
    "CostBasis",
    "DepthResources",
    "FixedResourceModel",
    "GateResources",
    "ResourceContext",
    "ResourceEstimate",
    "ResourceEstimator",
    "ResourcePolicy",
    "UnknownResourcePolicy",
    "WidthResources",
    "estimate_resources",
    "control",
    "inverse",
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
    "measure_reset",
    "project_x",
    "project_y",
    "project_z",
    "reset",
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
    "VectorView",
    "Matrix",
    "Tensor",
    "Observable",
    # stdlib
    "modular_decrement",
    "modular_increment",
    "qpe",
    "iqft",
    "qft",
    "modmul_const",
    "shor_order_finding",
    "grover_search",
    "grover_iteration_count",
    "QKernel",
    # Job / result types
    "Job",
    "JobStatus",
    "SampleResult",
    "SampleJob",
    "RunJob",
    "ExpvalJob",
    # Visualization (lazy-loaded)
    "MatplotlibDrawer",
    "CircuitStyle",
    "DEFAULT_STYLE",
]
