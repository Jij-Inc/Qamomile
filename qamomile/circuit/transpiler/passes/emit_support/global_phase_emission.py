"""Emit zero-qubit global-phase operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.transpiler.passes.emit_support.gate_emission import (
    resolve_angle_value,
)

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass


def emit_global_phase(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: GlobalPhaseOperation,
    bindings: dict[str, Any],
) -> None:
    """Emit a standalone zero-qubit global phase when the backend preserves it.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (GlobalPhaseOperation): Global-phase operation to emit.
        bindings (dict[str, Any]): Active emit bindings.

    Raises:
        EmitError: If a preserving backend cannot resolve the phase angle.
    """
    emit_hook = getattr(emit_pass._emitter, "emit_global_phase", None)
    if callable(emit_hook):
        emit_hook(circuit, resolve_angle_value(emit_pass, op.phase, bindings))


def emit_controlled_global_phase_operation(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: GlobalPhaseOperation,
    control_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit a global phase as a relative phase under accumulated controls.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (GlobalPhaseOperation): Global-phase operation to control.
        control_indices (list[int]): Accumulated physical control qubits.
        bindings (dict[str, Any]): Active emit bindings.

    Raises:
        EmitError: If the phase cannot be resolved or the backend cannot emit
            the accumulated control arity.
    """
    angle = resolve_angle_value(emit_pass, op.phase, bindings)
    emit_controlled_global_phase(emit_pass, circuit, control_indices, angle)


def emit_controlled_global_phase(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    control_indices: list[int],
    angle: Any,
) -> None:
    """Emit the relative phase produced by controlling a global phase.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        control_indices (list[int]): Physical controls gating the phase.
        angle (Any): Resolved phase angle.

    Raises:
        EmitError: If three or more controls are requested but the backend
            provides neither a native primitive nor the shared clean-ancilla
            decomposition.
    """
    num_controls = len(control_indices)
    if num_controls == 0:
        emit_hook = getattr(emit_pass._emitter, "emit_global_phase", None)
        if callable(emit_hook):
            emit_hook(circuit, angle)
        return

    emitter = emit_pass._emitter
    multi_controlled = getattr(emitter, "emit_multi_controlled_p", None)
    if callable(multi_controlled):
        multi_controlled(
            circuit,
            control_indices[1:],
            control_indices[0],
            angle,
        )
        return
    if num_controls == 1:
        emitter.emit_p(circuit, control_indices[0], angle)
        return
    if num_controls == 2:
        emitter.emit_cp(circuit, control_indices[0], control_indices[1], angle)
        return
    # Treat one of the phase controls as the P-gate target. The remaining
    # controls then form an ordinary multi-controlled P operation, allowing
    # backends such as QURI Parts to use StandardEmitPass's shared clean-
    # ancilla Toffoli cascade introduced by the multi-control decomposition.
    emit_pass._emit_irreducible_multi_controlled_gate(
        circuit,
        GateOperationType.P,
        control_indices[1:],
        control_indices[0],
        angle,
    )
