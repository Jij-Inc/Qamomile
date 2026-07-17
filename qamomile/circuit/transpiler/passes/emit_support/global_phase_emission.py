"""Emit zero-qubit global-phase operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.gate_emission import (
    resolve_angle_value,
)

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass


def _require_global_phase_hook(
    emit_pass: "StandardEmitPass",
) -> Callable[[Any, Any], None]:
    """Return the target hook that preserves a standalone phase.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.

    Returns:
        Callable[[Any, Any], None]: Callable target hook.

    Raises:
        EmitError: If the selected emitter cannot preserve the phase.
    """
    emit_hook = getattr(emit_pass._emitter, "emit_global_phase", None)
    if not callable(emit_hook):
        raise EmitError(
            "The selected emitter cannot preserve a standalone global phase.",
            operation="GlobalPhaseOperation",
        )
    return emit_hook


def emit_resolved_global_phase(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    angle: Any,
) -> None:
    """Emit an already resolved standalone phase without discarding it.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Circuit representation being emitted into.
        angle (Any): Resolved phase angle.

    Raises:
        EmitError: If the selected emitter cannot preserve the phase.
    """
    emit_hook = _require_global_phase_hook(emit_pass)
    emit_hook(circuit, angle)


def emit_global_phase(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: GlobalPhaseOperation,
    bindings: dict[str, Any],
) -> None:
    """Emit a standalone phase when the lowering adapter collects it.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Circuit representation being emitted into.
        op (GlobalPhaseOperation): Global-phase operation to emit.
        bindings (dict[str, Any]): Active emit bindings.

    Raises:
        EmitError: If the adapter cannot preserve or resolve the phase.
    """
    emit_hook = _require_global_phase_hook(emit_pass)
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
        circuit (Any): Circuit representation being emitted into.
        op (GlobalPhaseOperation): Global-phase operation to control.
        control_indices (list[int]): Accumulated physical control qubits.
        bindings (dict[str, Any]): Active emit bindings.

    Raises:
        EmitError: If the phase cannot be resolved or the adapter cannot emit
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
        emit_resolved_global_phase(emit_pass, circuit, angle)
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
    # representations with clean-ancilla support to use StandardEmitPass's
    # shared Toffoli cascade for multi-control decomposition.
    emit_pass._emit_irreducible_multi_controlled_gate(
        circuit,
        GateOperationType.P,
        control_indices[1:],
        control_indices[0],
        angle,
    )
