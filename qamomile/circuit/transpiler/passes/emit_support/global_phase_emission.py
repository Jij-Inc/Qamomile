"""Global-phase block operation emission.

Emits :class:`GlobalPhaseBlockOperation` by emitting the wrapped block's
body inline onto the target qubits and then adding the scalar global phase
through the backend's ``emit_global_phase`` hook. The hook is a no-op on
backends without a native circuit-level global phase (CUDA-Q, QURI Parts),
so a standalone global phase is correctly dropped there while Qiskit folds
it into ``QuantumCircuit.global_phase``.

The body emission delegates to ``emit_pass._emit_operations``, so this
single shared implementation works for every backend that subclasses
``StandardEmitPass`` -- no backend-specific standalone handler is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation.global_phase_block import GlobalPhaseBlockOperation
from qamomile.circuit.ir.value import ArrayValue
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _bind_and_populate_block_inputs,
    _expand_quantum_operands_to_phys,
    _prepare_nested_block_for_emit,
)
from qamomile.circuit.transpiler.passes.emit_support.gate_emission import (
    resolve_angle_value,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    ClbitMap,
    QubitAddress,
    QubitMap,
)

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass


def emit_global_phase_block(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: GlobalPhaseBlockOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a standalone global-phase block operation.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (GlobalPhaseBlockOperation): Global-phase block operation to emit.
        qubit_map (QubitMap): Current quantum value to physical qubit map.
        bindings (dict[str, Any]): Active emit bindings.
    """
    target_index_groups = [
        _expand_quantum_operands_to_phys(
            emit_pass,
            operand,
            qubit_map,
            bindings,
            operation="GlobalPhaseBlockOperation",
        )
        for operand in op.target_qubits
    ]
    target_indices = [index for group in target_index_groups for index in group]

    _emit_global_phase_body_inline(emit_pass, circuit, op, target_indices, bindings)

    # Fold the global phase into the backend's circuit-level accumulator when
    # it has one (Qiskit). Backends without a native global phase (CUDA-Q,
    # QURI Parts) simply lack ``emit_global_phase`` and correctly drop it --
    # a standalone global phase is unobservable. Resolved via ``getattr`` so
    # a new backend needs no global-phase code to stay correct.
    emit_global_phase = getattr(emit_pass._emitter, "emit_global_phase", None)
    if callable(emit_global_phase):
        angle = resolve_angle_value(emit_pass, op.phase, bindings)
        emit_global_phase(circuit, angle)

    _map_global_phase_results(op, target_index_groups, qubit_map)


def _emit_global_phase_body_inline(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: GlobalPhaseBlockOperation,
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit the wrapped block's body inline onto the target qubits.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (GlobalPhaseBlockOperation): Operation whose ``source_block`` is
            emitted as-is (no inversion).
        target_indices (list[int]): Physical target qubits.
        bindings (dict[str, Any]): Parent emit bindings.
    """
    source = _prepare_nested_block_for_emit(op.source_block, bindings)
    if source is None or not hasattr(source, "operations"):
        return

    local_qubit_map: QubitMap = {}
    local_clbit_map: ClbitMap = {}
    local_bindings = _bind_and_populate_block_inputs(
        emit_pass,
        source,
        [*op.target_qubits, *op.parameters],
        len(target_indices),
        bindings,
        local_qubit_map,
        parent_qubits=target_indices,
        operation_name="GlobalPhaseBlockOperation",
    )
    emit_pass._emit_operations(
        circuit,
        source.operations,
        local_qubit_map,
        local_clbit_map,
        local_bindings,
        force_unroll=True,
    )


def emit_global_phase_block_at_indices(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: GlobalPhaseBlockOperation,
    control_indices: list[int],
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit a global-phase block under already-resolved outer controls.

    Controls the wrapped block's body via the backend's controlled fallback,
    then adds the relative phase that the controlled global phase produces on
    the control qubits.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (GlobalPhaseBlockOperation): Operation to emit.
        control_indices (list[int]): Physical outer control qubits.
        target_indices (list[int]): Physical qubits for the wrapped block's
            targets.
        bindings (dict[str, Any]): Active emit bindings.
    """
    if not control_indices:
        _emit_global_phase_body_inline(emit_pass, circuit, op, target_indices, bindings)
        angle = resolve_angle_value(emit_pass, op.phase, bindings)
        emit_global_phase = getattr(emit_pass._emitter, "emit_global_phase", None)
        if callable(emit_global_phase):
            emit_global_phase(circuit, angle)
        return

    source = _prepare_nested_block_for_emit(op.source_block, bindings)
    local_qubit_map: QubitMap = {}
    local_bindings = _bind_and_populate_block_inputs(
        emit_pass,
        source,
        [*op.target_qubits, *op.parameters],
        len(target_indices),
        bindings,
        local_qubit_map,
        operation_name="GlobalPhaseBlockOperation",
    )
    emit_pass._emit_controlled_fallback(
        circuit,
        source,
        len(control_indices),
        control_indices,
        target_indices,
        1,
        local_bindings,
    )
    angle = resolve_angle_value(emit_pass, op.phase, bindings)
    emit_controlled_global_phase(emit_pass, circuit, control_indices, angle)


def emit_controlled_global_phase(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    control_indices: list[int],
    angle: Any,
) -> None:
    """Emit the relative phase produced by controlling a global phase.

    ``control(e^{i*angle} I)`` applies ``e^{i*angle}`` exactly when every
    control qubit is ``|1>``, which is a (multi-)controlled phase on the
    control qubits -- the projector-controlled-phase building block. This is
    the load-bearing observable case of a controlled global phase; the
    target qubits of the wrapped block are not involved.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        control_indices (list[int]): Physical control qubits gating the
            phase. An empty list emits nothing (an uncontrolled global phase
            is unobservable).
        angle (Any): Resolved phase angle (float or backend parameter).

    Raises:
        EmitError: If three or more controls are requested but the backend
            exposes no multi-controlled phase primitive.
    """
    n = len(control_indices)
    if n == 0:
        return
    emitter = emit_pass._emitter
    # Prefer a native multi-controlled phase when the backend has one
    # (CUDA-Q ``r1.ctrl``); it covers every arity including 1 and 2.
    mcp = getattr(emitter, "emit_multi_controlled_p", None)
    if callable(mcp):
        mcp(circuit, control_indices[1:], control_indices[0], angle)
        return
    if n == 1:
        emitter.emit_p(circuit, control_indices[0], angle)
        return
    if n == 2:
        emitter.emit_cp(circuit, control_indices[0], control_indices[1], angle)
        return
    raise EmitError(
        f"Controlled global phase with {n} controls needs a multi-controlled "
        f"phase primitive, but this backend only provides single- and "
        f"two-control phase gates. Run on a backend whose controlled-U path "
        f"builds a native controlled gate (Qiskit), or add an "
        f"``emit_multi_controlled_p`` primitive to this emitter.",
        operation="GlobalPhaseBlockOperation",
    )


def _map_global_phase_results(
    op: GlobalPhaseBlockOperation,
    target_index_groups: list[list[int]],
    qubit_map: QubitMap,
) -> None:
    """Map global-phase-block result values to physical qubits.

    The body passes its qubits through unchanged (a global phase moves no
    qubit), so each target result re-uses the physical qubits of its
    operand.

    Args:
        op (GlobalPhaseBlockOperation): Operation that was emitted.
        target_index_groups (list[list[int]]): Physical qubits for each
            target operand, preserving scalar/vector operand grouping.
        qubit_map (QubitMap): Mutable block-local qubit map.
    """
    target_results = [
        r for r in op.results[op.num_control_qubits :] if r.type.is_quantum()
    ]
    for result, indices in zip(target_results, target_index_groups):
        if isinstance(result, ArrayValue):
            for i, phys in enumerate(indices):
                qubit_map[QubitAddress(result.uuid, i)] = phys
            if indices:
                qubit_map[QubitAddress(result.uuid)] = indices[0]
        elif indices:
            qubit_map[QubitAddress(result.uuid)] = indices[0]
