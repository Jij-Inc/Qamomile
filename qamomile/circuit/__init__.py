"""Compiler core of Qamomile and its public user-facing API surface.

Design center
-------------

The central abstraction is the ``qkernel`` decorator (``frontend/``):
users write quantum programs as plain Python functions, the frontend
traces them into an IR ``Block`` (``ir/``), the transpiler pipeline
(``transpiler/``) rewrites the IR through staged, ``BlockKind``-gated
passes, and a backend package emits an executable program. This module
re-exports everything a user program needs to be written: the decorator,
the handle types (``Qubit``, ``Vector``, ``Float``, ...), gate /
measurement / control-flow builders, meta-operations (``control`` /
``inverse``), stdlib and algorithm kernels (QFT, QPE, Grover, Shor, modular
arithmetic), scheme-specific descriptors returned by circuit factories,
symbolic resource estimation (``estimator/``), and the job / result types
returned by ``ExecutableProgram.sample`` / ``run``.

Dependency direction (hard constraint)
--------------------------------------

``qamomile.circuit`` is the design center of the whole project: every
other qamomile module depends on it, never the reverse —
``optimization → circuit ← backends`` (qiskit / quri_parts / cudaq /
...). Nothing under this package may import a backend package or SDK.
Backend-specific concretization (native gate sets, per-qubit instruction
encoding, runtime control-flow lowering) belongs in each backend's emit
pass / ``GateEmitter``; this package owns only the abstract IR, the
backend-agnostic pass pipeline, and the shared decomposition recipes
that backends may fall back on.

Module-local constraints
------------------------

- stdlib / algorithm kernels are imported *after* the frontend symbols
  because their implementation modules do
  ``import qamomile.circuit as qmc`` — keep new kernel imports at the
  bottom of this file.
- Visualization (``MatplotlibDrawer`` etc.) is lazy-loaded via module
  ``__getattr__`` so that importing ``qamomile.circuit`` does not pull
  in matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .estimator import (
    CallResources,
    DepthResources,
    EstimateQuality,
    GateBasis,
    GateResources,
    OpaqueCallContext,
    ResourceEstimate,
    ResourceEstimator,
    UnknownResourcePolicy,
    WidthResources,
    estimate_resources,
)

# Frontend API
from .frontend.callable_signature import CallableSignature
from .frontend.composite_gate import composite_gate as composite_gate
from .frontend.constructors import bit, bit_array, float_, qubit, qubit_array, uint
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
from .frontend.operation.global_phase import global_phase
from .frontend.operation.inverse import inverse
from .frontend.operation.math import ceil, log2
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
from .frontend.operation.select import select
from .frontend.oracle import Oracle, opaque
from .frontend.qkernel import QKernel, qkernel
from .frontend.struct import struct
from .ir.effect import KernelEffect
from .stdlib import (
    LCUBlockEncoding,
    PauliLCUBlockEncoding,
    PeriodicShiftLCUBlockEncoding,
    add_const,
    amplitude_encoding,
    amplitude_encoding_from_angles,
    computational_basis_state,
    controlled_add_const,
    controlled_modular_add,
    controlled_modular_add_const,
    controlled_modular_add_const_modulus,
    grover_iteration_count,
    grover_search,
    iqft,
    lookup_xor,
    mcx,
    modmul_const,
    modular_add,
    modular_add_const,
    modular_decrement,
    modular_increment,
    mottonen_amplitude_encoding,
    mottonen_amplitude_encoding_from_angles,
    multi_controlled_x,
    pauli_lcu_block_encoding,
    periodic_shift_lcu_block_encoding,
    qft,
    qpe,
    ripple_carry_add,
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
from .algorithm.shor import (  # noqa: E402, I001
    ekera_hastad_factoring,
    shor_order_finding,
)

__all__ = [
    "qkernel",
    "struct",
    "KernelEffect",
    "composite_gate",
    "Oracle",
    "opaque",
    "CallableSignature",
    "CallResources",
    "DepthResources",
    "EstimateQuality",
    "GateBasis",
    "GateResources",
    "OpaqueCallContext",
    "ResourceEstimate",
    "ResourceEstimator",
    "UnknownResourcePolicy",
    "WidthResources",
    "estimate_resources",
    "control",
    "select",
    "inverse",
    "global_phase",
    "cast",
    "ceil",
    "log2",
    "bit",
    "bit_array",
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
    "PeriodicShiftLCUBlockEncoding",
    "periodic_shift_lcu_block_encoding",
    "modular_decrement",
    "modular_increment",
    "qpe",
    "mcx",
    "multi_controlled_x",
    "LCUBlockEncoding",
    "PauliLCUBlockEncoding",
    "pauli_lcu_block_encoding",
    "computational_basis_state",
    "amplitude_encoding",
    "amplitude_encoding_from_angles",
    "mottonen_amplitude_encoding",
    "mottonen_amplitude_encoding_from_angles",
    "ripple_carry_add",
    "add_const",
    "controlled_add_const",
    "modular_add",
    "controlled_modular_add",
    "modular_add_const",
    "controlled_modular_add_const",
    "controlled_modular_add_const_modulus",
    "lookup_xor",
    "iqft",
    "qft",
    "modmul_const",
    "shor_order_finding",
    "ekera_hastad_factoring",
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
