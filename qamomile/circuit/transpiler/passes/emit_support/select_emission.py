"""SELECT (quantum multiplexer) emission helpers.

Provides :func:`emit_select`, the emit-time lowering of
:class:`~qamomile.circuit.ir.operation.select.SelectOperation`. The op is
decomposed gate-by-gate into one controlled-U per case, each
controlled on the full index register with a mixed ``0``/``1`` pattern
equal to the big-endian binary expansion of the case index. The
``0``-controls reuse the same X-bracket realisation as ``qmc.control``'s
zero-control mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation.select import (
    SelectOperation,
    control_values_for_index,
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _bind_quantum_input_shapes,
    _emit_single_target_block_per_vector_element,
    _expand_quantum_operands_to_phys,
    _map_operand_result_groups,
    _prepare_nested_block_for_emit,
    _should_emit_single_target_block_per_vector_element,
    bracket_zero_controls,
    emit_controlled_block_at_indices,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import QubitMap

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass


def emit_select(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: SelectOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a :class:`SelectOperation`.

    Decomposes the op into one controlled-U per case. Case ``i`` is applied to the
    target register controlled on the index register reading the
    big-endian integer ``i``; the ``0`` bits of that pattern become
    anti-controls realised by X-bracketing the corresponding index
    qubits.

    Args:
        emit_pass (StandardEmitPass): The driving emit pass. Provides the
            ``_resolver`` / ``_emitter`` / ``_blockvalue_to_gate`` /
            ``_emit_controlled_fallback`` machinery.
        circuit (Any): The backend circuit being built.
        op (SelectOperation): The SELECT operation to emit.
        qubit_map (QubitMap): The active address -> physical qubit map,
            mutated in place with the result-side mappings.
        bindings (dict[str, Any]): Active emit bindings.

    Raises:
        EmitError: If an index operand cannot be resolved to a physical
            qubit, or (propagated from the controlled fallback) the
            backend cannot realise a per-case controlled-U (e.g. an
            unsupported multi-control shape).
    """
    _emit_select_with_outer_controls(
        emit_pass,
        circuit,
        op,
        [],
        qubit_map,
        bindings,
    )


def emit_controlled_select(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: SelectOperation,
    outer_control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a SELECT nested inside an enclosing controlled block.

    Each case is controlled on both the enclosing controls and the SELECT
    index pattern.

    Args:
        emit_pass (StandardEmitPass): The driving emit pass.
        circuit (Any): The backend circuit being built.
        op (SelectOperation): The nested SELECT operation.
        outer_control_indices (list[int]): Physical controls accumulated from
            enclosing controlled operations.
        qubit_map (QubitMap): Block-local address-to-physical-qubit map,
            updated with the SELECT results.
        bindings (dict[str, Any]): Bindings visible inside the enclosing block.

    Raises:
        EmitError: If an operand cannot be resolved or a composed controlled
            case cannot be emitted by the backend.
    """
    _emit_select_with_outer_controls(
        emit_pass,
        circuit,
        op,
        outer_control_indices,
        qubit_map,
        bindings,
    )


def _emit_select_with_outer_controls(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: SelectOperation,
    outer_control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a SELECT with an optional accumulated outer control set.

    Args:
        emit_pass (StandardEmitPass): The driving emit pass.
        circuit (Any): The backend circuit being built.
        op (SelectOperation): The SELECT operation to lower.
        outer_control_indices (list[int]): Physical controls inherited from
            enclosing controlled operations.
        qubit_map (QubitMap): Address-to-physical-qubit map, updated with
            result aliases.
        bindings (dict[str, Any]): Active emit bindings.
    Raises:
        EmitError: If operands cannot be resolved or the backend cannot emit a
            composed controlled case.
    """
    num_idx = op.num_index_qubits
    index_operands = op.operands[:num_idx]
    remaining_operands = op.operands[num_idx:]
    target_qubit_operands = [v for v in remaining_operands if v.type.is_quantum()]
    param_operands = [
        v for v in remaining_operands if v.type.is_classical() or v.type.is_object()
    ]

    # Resolve physical index qubits (mirrors ConcreteControlledU controls:
    # the frontend normalises ``operands[:num_index_qubits]`` to one scalar
    # per physical index qubit).
    index_indices: list[int] = []
    for q in index_operands:
        idx = emit_pass._resolver.resolve_qubit_index(q, qubit_map, bindings)
        if idx is not None:
            index_indices.append(idx)
    if len(index_indices) < len(index_operands):
        raise EmitError(
            f"SelectOperation: only {len(index_indices)}/"
            f"{len(index_operands)} index operand(s) could be resolved to "
            f"physical qubits. Emitting a partial- or zero-arity select "
            f"would silently miswire the circuit. This typically indicates "
            f"broken parent_array / slice metadata on the index operands.",
            operation="SelectOperation",
        )

    # Resolve physical target qubits once; every case shares them.
    target_indices: list[int] = []
    target_index_groups: list[list[int]] = []
    for q in target_qubit_operands:
        indices = _expand_quantum_operands_to_phys(emit_pass, q, qubit_map, bindings)
        target_index_groups.append(indices)
        target_indices.extend(indices)

    # Gate-by-gate fallback: one controlled-U per case with a mixed 0/1
    # control pattern (big-endian bits of the case index).
    for case_index, case_block in enumerate(op.case_blocks):
        control_values = control_values_for_index(case_index, num_idx)
        zero_phys = [
            index_indices[j] for j, value in enumerate(control_values) if value == 0
        ]
        composed_controls = [*outer_control_indices, *index_indices]

        local_bindings = emit_pass._resolver.bind_block_params(
            case_block, param_operands, bindings
        )
        _bind_quantum_input_shapes(
            emit_pass._resolver,
            case_block,
            target_qubit_operands,
            bindings,
            local_bindings,
        )
        prepared_block = _prepare_nested_block_for_emit(case_block, local_bindings)

        with bracket_zero_controls(emit_pass, circuit, zero_phys):
            if _should_emit_single_target_block_per_vector_element(
                prepared_block, target_qubit_operands, target_indices
            ):
                # Scalar single-qubit case broadcast over a Vector[Qubit]
                # target: apply the controlled scalar case to each element,
                # mirroring qmc.control's vector-broadcast convenience.
                _emit_single_target_block_per_vector_element(
                    emit_pass,
                    circuit,
                    prepared_block,
                    len(composed_controls),
                    composed_controls,
                    target_indices,
                    1,
                    local_bindings,
                )
            else:
                emit_controlled_block_at_indices(
                    emit_pass,
                    circuit,
                    prepared_block,
                    len(composed_controls),
                    composed_controls,
                    target_indices,
                    1,
                    local_bindings,
                )

    _map_select_results(
        op,
        num_idx,
        index_indices,
        target_index_groups,
        qubit_map,
    )


def _map_select_results(
    op: SelectOperation,
    num_index_qubits: int,
    index_indices: list[int],
    target_index_groups: list[list[int]],
    qubit_map: QubitMap,
) -> None:
    """Map a SELECT op's result ``Value`` UUIDs to physical qubits.

    Result layout mirrors the operand layout: ``op.results[:num_index]``
    are the index qubits (1:1 with ``index_indices``) and the quantum
    results that follow correspond to the target operands. A
    ``Vector[Qubit]`` target result registers a per-element address plus
    the base address so ``result[i]`` and whole-array lookups both
    resolve. Index and target qubits keep their physical slots — SELECT
    only adds relative phases / state changes under the index basis, it
    never moves qubits.

    Args:
        op (SelectOperation): The emitted SELECT operation.
        num_index_qubits (int): Number of physical index qubits.
        index_indices (list[int]): Physical index qubits.
        target_index_groups (list[list[int]]): Per-operand physical index
            groups from :func:`_expand_quantum_operands_to_phys`.
        qubit_map (QubitMap): Mutated in place with the new result
            addresses.
    """
    index_results = op.results[:num_index_qubits]
    _map_operand_result_groups(
        index_results,
        [[physical] for physical in index_indices],
        qubit_map,
    )

    target_results = [r for r in op.results[num_index_qubits:] if r.type.is_quantum()]
    _map_operand_result_groups(target_results, target_index_groups, qubit_map)
