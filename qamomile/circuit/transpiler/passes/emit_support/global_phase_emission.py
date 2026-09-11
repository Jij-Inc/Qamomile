"""Emit zero-qubit global-phase operations."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.gate_emission import (
    resolve_angle_value,
)

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass


def is_exact_real_zero(value: Any) -> bool:
    """Return whether a value is a concrete real scalar equal to zero.

    The comparison intentionally has no tolerance: a tiny nonzero value must
    remain observable, while engine parameter expressions remain unresolved.

    Args:
        value (Any): Concrete numeric value or engine parameter expression.

    Returns:
        bool: ``True`` only for a non-boolean real scalar exactly equal to
        zero.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    symbolic_zero = getattr(value, "is_zero", None)
    if symbolic_zero is not None:
        return symbolic_zero is True
    try:
        return bool(value == 0)
    except (TypeError, ValueError):
        return False


def is_identity_phase_angle(angle: Any) -> bool:
    """Return whether a concrete angle is exactly zero modulo two pi.

    The test intentionally uses no tolerance so a tiny but nonzero phase is
    never discarded. Runtime parameter expressions remain non-identity until
    an engine resolves them to a concrete number.

    Args:
        angle (Any): Resolved numeric angle or engine parameter expression.

    Returns:
        bool: Whether ``angle`` is a finite real number exactly congruent to
        zero modulo ``2 * pi``.
    """
    if isinstance(angle, bool) or not isinstance(angle, Real):
        return False
    if is_exact_real_zero(angle):
        return True
    try:
        numeric = float(angle)
    except (OverflowError, TypeError, ValueError):
        return False
    if not math.isfinite(numeric) or numeric == 0.0:
        # A nonzero exact value that underflows during binary64 conversion is
        # still an observable phase and must fail closed.
        return False
    return is_exact_real_zero(math.fmod(numeric, math.tau))


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
        circuit (Any): Engine circuit being emitted into.
        control_indices (list[int]): Physical controls gating the phase.
        angle (Any): Resolved phase angle.

    Raises:
        EmitError: If three or more controls are requested but the engine
            provides neither a native primitive nor the shared clean-ancilla
            decomposition.
    """
    if is_identity_phase_angle(angle):
        return

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
