"""Provide a semantic multi-controlled X with a capability-driven fallback."""

from __future__ import annotations

from typing import Any, cast

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.ir.operation.callable import CallPolicy


@qmc.composite_gate(name="multi_controlled_x")
def multi_controlled_x(
    controls: Vector[Qubit],
    target: Qubit,
) -> tuple[Vector[Qubit], Qubit]:
    """Flip a target when every qubit in a control register is one.

    The function keeps the familiar qkernel call style while preserving one
    semantic operation for targets with an arbitrary-width controlled-X gate.
    Other targets execute the body through Qamomile's controlled-gate lowering
    when their declared control-width profile admits it; narrower targets
    reject the call at the capability boundary.

    Args:
        controls (Vector[Qubit]): Non-empty control register.
        target (Qubit): Target qubit to conditionally flip.

    Returns:
        tuple[Vector[Qubit], Qubit]: Updated controls and target.
    """
    operation = cast(Any, qmc.control(qmc.x, num_controls=controls.shape[0]))
    return operation(controls, target)


configure_composite(
    multi_controlled_x,
    namespace="qamomile.stdlib",
    policy=CallPolicy.NATIVE_FIRST,
)

mcx = multi_controlled_x
"""Short public alias for :func:`multi_controlled_x`."""

__all__ = ["mcx", "multi_controlled_x"]
