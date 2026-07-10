"""Provide QFT and inverse-QFT as ordinary named qkernels."""

from __future__ import annotations

import math

import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.ir.operation.callable import CallPolicy, CompositeGateType


@qmc.composite_gate(name="qft")
def qft(qubits: Vector[Qubit]) -> Vector[Qubit]:
    """Apply the standard quantum Fourier transform.

    The Qamomile body remains attached to the named invocation for every
    register width. Backends may emit a native QFT, while other backends lower
    this same body during emission.

    Args:
        qubits (Vector[Qubit]): Register to transform.

    Returns:
        Vector[Qubit]: Transformed register.
    """
    n = qubits.shape[0]
    for offset in qmc.range(n):
        target = n - 1 - offset
        qubits[target] = qmc.h(qubits[target])
        for delta in qmc.range(target):
            control = target - 1 - delta
            angle = math.pi / (2 ** (target - control))
            qubits[target], qubits[control] = qmc.cp(
                qubits[target], qubits[control], angle
            )
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        qubits[index], qubits[mirror] = qmc.swap(qubits[index], qubits[mirror])
    return qubits


configure_composite(
    qft,
    namespace="qamomile.stdlib",
    gate_type=CompositeGateType.QFT,
    policy=CallPolicy.NATIVE_FIRST,
)


def _qft_resource_estimate(ctx: qmc.ResourceContext) -> qmc.ResourceEstimate:
    """Return the closed-form resources of QFT or inverse QFT.

    Args:
        ctx (qmc.ResourceContext): Invocation context carrying the target shape.

    Returns:
        qmc.ResourceEstimate: Exact logical gate and width formulas.
    """
    n = next(
        iter(ctx.operand_shapes.values()), sp.Symbol("n", integer=True, positive=True)
    )
    swaps = sp.floor(n / 2)
    controlled_phases = n * (n - 1) / 2
    return qmc.ResourceEstimate(
        gates=qmc.GateResources(
            total=n + controlled_phases + swaps,
            single_qubit=n,
            two_qubit=controlled_phases + swaps,
            rotation=controlled_phases,
            non_clifford=controlled_phases,
        ),
        depth=qmc.DepthResources(depth=n + controlled_phases + swaps),
    )


qft.resource_model(_qft_resource_estimate, estimate_kind="exact_decomposed")


@qmc.composite_gate(name="iqft")
def iqft(qubits: Vector[Qubit]) -> Vector[Qubit]:
    """Apply the inverse quantum Fourier transform.

    Args:
        qubits (Vector[Qubit]): Register to transform.

    Returns:
        Vector[Qubit]: Transformed register.
    """
    n = qubits.shape[0]
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        qubits[index], qubits[mirror] = qmc.swap(qubits[index], qubits[mirror])
    for target in qmc.range(n):
        for control in qmc.range(target):
            angle = -math.pi / (2 ** (target - control))
            qubits[target], qubits[control] = qmc.cp(
                qubits[target], qubits[control], angle
            )
        qubits[target] = qmc.h(qubits[target])
    return qubits


configure_composite(
    iqft,
    namespace="qamomile.stdlib",
    gate_type=CompositeGateType.IQFT,
    policy=CallPolicy.NATIVE_FIRST,
)

iqft.resource_model(_qft_resource_estimate, estimate_kind="exact_decomposed")


__all__ = ["qft", "iqft"]
