"""Inverse block emission helpers extracted from StandardEmitPass.

Emits :class:`InverseBlockOperation` with a three-tier strategy: try the
backend-native inverse of the forward ``source_block`` first (reusable
gate + ``gate_inverse``), then a reusable gate built from the
``implementation_block`` fallback, and finally inline the fallback
gate by gate. Each function takes an ``emit_pass`` parameter (a
``StandardEmitPass`` instance) in place of ``self`` so subclass
overrides (e.g. backend emit passes) are respected, mirroring
:mod:`.controlled_emission`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.value import ArrayValue
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _bind_and_populate_block_inputs,
    _emitter_supports_reusable_gates,
    _expand_quantum_operands_to_phys,
    _gate_matches_qubit_count,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    ClbitMap,
    QubitAddress,
    QubitMap,
)

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass


def emit_inverse_block(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: InverseBlockOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a first-class inverse block operation.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (InverseBlockOperation): Inverse block operation to emit.
        qubit_map (QubitMap): Current quantum value to physical qubit map.
        bindings (dict[str, Any]): Active emit bindings.

    Raises:
        EmitError: If an inverse block has missing blocks, unresolved qubit
            operands, or no available native/fallback emission path.
    """
    control_index_groups = [
        _expand_quantum_operands_to_phys(
            emit_pass,
            operand,
            qubit_map,
            bindings,
            operation="InverseBlockOperation",
        )
        for operand in op.control_qubits
    ]
    target_index_groups = [
        _expand_quantum_operands_to_phys(
            emit_pass,
            operand,
            qubit_map,
            bindings,
            operation="InverseBlockOperation",
        )
        for operand in op.target_qubits
    ]

    control_indices = [index for group in control_index_groups for index in group]
    target_indices = [index for group in target_index_groups for index in group]
    emit_inverse_block_at_indices(
        emit_pass,
        circuit,
        op,
        control_indices,
        target_indices,
        bindings,
    )
    _map_inverse_block_results(
        op,
        control_index_groups,
        target_index_groups,
        qubit_map,
    )


def emit_inverse_block_at_indices(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: InverseBlockOperation,
    control_indices: list[int],
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit an inverse block after physical qubits have been resolved.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (InverseBlockOperation): Inverse block operation to emit.
        control_indices (list[int]): Physical control qubits.
        target_indices (list[int]): Physical target qubits.
        bindings (dict[str, Any]): Active emit bindings.

    Raises:
        EmitError: If required source/fallback blocks are missing or no
            emission path can represent the operation.
    """
    if op.source_block is None or op.implementation_block is None:
        raise EmitError(
            "Cannot emit inverse block without both source and implementation blocks.",
            operation="InverseBlockOperation",
        )
    if len(target_indices) != op.num_target_qubits:
        raise EmitError(
            "Inverse block target count does not match resolved qubits: "
            f"expected {op.num_target_qubits}, got {len(target_indices)}.",
            operation="InverseBlockOperation",
        )

    input_operands = [*op.target_qubits, *op.parameters]
    can_build_reusable_gate = _emitter_supports_reusable_gates(emit_pass._emitter)
    source_gate = (
        emit_pass._blockvalue_to_gate(
            op.source_block,
            len(target_indices),
            bindings,
            input_operands=input_operands,
            operation_name="InverseBlockOperation",
        )
        if can_build_reusable_gate
        and _emitter_supports_gate_inverse(emit_pass._emitter)
        else None
    )
    if source_gate is not None:
        inverse_gate = emit_pass._emitter.gate_inverse(source_gate)
        if inverse_gate is not None:
            if control_indices:
                inverse_gate = emit_pass._emitter.gate_controlled(
                    inverse_gate,
                    len(control_indices),
                )
            if inverse_gate is not None and _gate_matches_qubit_count(
                inverse_gate,
                len(control_indices) + len(target_indices),
            ):
                emit_pass._emitter.append_gate(
                    circuit,
                    inverse_gate,
                    [*control_indices, *target_indices],
                )
                return

    fallback_gate = (
        emit_pass._blockvalue_to_gate(
            op.implementation_block,
            len(target_indices),
            bindings,
            input_operands=input_operands,
            operation_name="InverseBlockOperation",
        )
        if can_build_reusable_gate
        else None
    )
    if fallback_gate is not None:
        if control_indices:
            fallback_gate = emit_pass._emitter.gate_controlled(
                fallback_gate,
                len(control_indices),
            )
        if fallback_gate is not None and _gate_matches_qubit_count(
            fallback_gate,
            len(control_indices) + len(target_indices),
        ):
            emit_pass._emitter.append_gate(
                circuit,
                fallback_gate,
                [*control_indices, *target_indices],
            )
            return

    if control_indices:
        local_bindings = _bind_inverse_block_inputs(
            emit_pass,
            op,
            len(target_indices),
            bindings,
        )
        emit_pass._emit_controlled_fallback(
            circuit,
            op.implementation_block,
            len(control_indices),
            control_indices,
            target_indices,
            1,
            local_bindings,
        )
        return

    _emit_inverse_block_inline(
        emit_pass,
        circuit,
        op.implementation_block,
        op,
        target_indices,
        bindings,
    )


def _bind_inverse_block_inputs(
    emit_pass: "StandardEmitPass",
    op: InverseBlockOperation,
    num_qubits: int,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    """Bind inverse block formal inputs to call-site operands.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        op (InverseBlockOperation): Inverse block operation being emitted.
        num_qubits (int): Number of target qubits available to the fallback.
        bindings (dict[str, Any]): Parent emit bindings.

    Returns:
        dict[str, Any]: Local bindings for the fallback implementation.

    Raises:
        EmitError: If the inverse operation has no fallback implementation
            block.
    """
    if op.implementation_block is None:
        raise EmitError(
            "Cannot bind inverse block inputs without an implementation block.",
            operation="InverseBlockOperation",
        )
    local_qubit_map: QubitMap = {}
    return _bind_and_populate_block_inputs(
        emit_pass,
        op.implementation_block,
        [*op.target_qubits, *op.parameters],
        num_qubits,
        bindings,
        local_qubit_map,
        operation_name="InverseBlockOperation",
    )


def _emit_inverse_block_inline(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    impl: Any,
    op: InverseBlockOperation,
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Inline an inverse fallback implementation into the parent circuit.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        impl (Any): Fallback inverse implementation block.
        op (InverseBlockOperation): Inverse block operation being emitted.
        target_indices (list[int]): Physical target qubits.
        bindings (dict[str, Any]): Local bindings for ``impl``.
    """
    local_qubit_map: QubitMap = {}
    local_clbit_map: ClbitMap = {}
    local_bindings = _bind_and_populate_block_inputs(
        emit_pass,
        impl,
        [*op.target_qubits, *op.parameters],
        len(target_indices),
        bindings,
        local_qubit_map,
        parent_qubits=target_indices,
        operation_name="InverseBlockOperation",
    )

    if hasattr(impl, "operations"):
        emit_pass._emit_operations(
            circuit,
            impl.operations,
            local_qubit_map,
            local_clbit_map,
            local_bindings,
            force_unroll=True,
        )


def _emitter_supports_gate_inverse(emitter: Any) -> bool:
    """Return whether an emitter can invert reusable gates.

    Args:
        emitter (Any): Backend gate emitter.

    Returns:
        bool: True when ``emitter`` advertises reusable-gate inversion.
    """
    supports = getattr(emitter, "supports_gate_inverse", None)
    return bool(supports()) if callable(supports) else False


def _map_inverse_block_results(
    op: InverseBlockOperation,
    control_index_groups: list[list[int]],
    target_index_groups: list[list[int]],
    qubit_map: QubitMap,
) -> None:
    """Map inverse-block result values to physical qubits.

    Args:
        op (InverseBlockOperation): Inverse operation that was emitted.
        control_index_groups (list[list[int]]): Physical qubits for each
            control operand.
        target_index_groups (list[list[int]]): Physical qubits for each
            target operand, preserving scalar/vector operand grouping.
        qubit_map (QubitMap): Mutable block-local qubit map.
    """
    control_results = op.results[: op.num_control_qubits]
    for result, indices in zip(control_results, control_index_groups):
        if indices:
            qubit_map[QubitAddress(result.uuid)] = indices[0]

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
