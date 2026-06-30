"""Controlled operation emission helpers extracted from StandardEmitPass.

This module provides module-level functions for emitting controlled-U
operations, controlled gates, and related helpers. Each function takes an
``emit_pass`` parameter (a ``StandardEmitPass`` instance) in place of
``self``.

Note: ``emit_controlled_fallback`` and ``blockvalue_to_gate`` are called
via ``emit_pass._emit_controlled_fallback(...)`` and
``emit_pass._blockvalue_to_gate(...)`` respectively, so that subclass
overrides (e.g. CudaqEmitPass) are respected.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any, cast

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import (
    Operation,
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation, HasNestedOps
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.cast_binop_emission import (
    evaluate_binop,
)
from qamomile.circuit.transpiler.passes.emit_support.physical_index_map import (
    map_array_result_group,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    ClbitMap,
    QubitAddress,
    QubitMap,
)

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass


def emit_controlled_powers(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    block_value: Any,
    counting_indices: list[int],
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit controlled-U^(2^k) operations."""
    num_targets = len(target_indices)
    block_value = _prepare_nested_block_for_emit(block_value, bindings)
    unitary_gate = emit_pass._blockvalue_to_gate(block_value, num_targets, bindings)

    if unitary_gate is not None:
        for k, ctrl_idx in enumerate(counting_indices):
            power = 2**k
            powered_gate = emit_pass._emitter.gate_power(unitary_gate, power)
            controlled_powered_gate = emit_pass._emitter.gate_controlled(
                powered_gate, 1
            )
            emit_pass._emitter.append_gate(
                circuit, controlled_powered_gate, [ctrl_idx] + target_indices
            )
    else:
        for k, ctrl_idx in enumerate(counting_indices):
            power = 2**k
            for _ in range(power):
                emit_controlled_block(
                    emit_pass, circuit, block_value, ctrl_idx, target_indices, bindings
                )


def emit_controlled_block(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    block_value: Any,
    control_idx: int,
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit a controlled version of a block via the mapped walker.

    Builds a block-local qubit map from the block's formal quantum
    inputs to ``target_indices`` and walks the block body, so inner
    gates land on the physical qubit their operand actually refers to
    rather than on ``target_indices[0]``.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        block_value (Any): Inner block whose operations are controlled.
            Objects without ``operations`` are silently skipped.
        control_idx (int): Physical control qubit index.
        target_indices (list[int]): Physical target qubits covering the
            block's quantum inputs in declaration order.
        bindings (dict[str, Any]): Local block bindings.

    Raises:
        EmitError: If the block's quantum inputs cannot be mapped onto
            ``target_indices`` or an inner operation cannot be emitted.
    """
    if not hasattr(block_value, "operations"):
        return

    block_value = _prepare_nested_block_for_emit(block_value, bindings)
    qubit_map = build_controlled_block_qubit_map(
        emit_pass, block_value, target_indices, bindings
    )
    emit_controlled_operations(
        emit_pass,
        circuit,
        block_value.operations,
        [control_idx],
        qubit_map,
        bindings,
    )


def build_controlled_block_qubit_map(
    emit_pass: "StandardEmitPass",
    block_value: Any,
    target_indices: list[int],
    bindings: dict[str, Any],
) -> QubitMap:
    """Build a block-local qubit map backed by physical target indices.

    Seeds one entry per formal quantum input of ``block_value`` —
    scalar ``Qubit`` inputs map to one physical index,
    ``Vector[Qubit]`` inputs map per-element — positionally matching
    ``target_indices`` in declaration order.

    Args:
        emit_pass (StandardEmitPass): Emit pass used to resolve
            symbolic vector input shapes against ``bindings``.
        block_value (Any): Inner block whose ``input_values`` define the
            quantum formal arguments. Objects without ``input_values``
            yield an empty map.
        target_indices (list[int]): Physical qubit indices supplied at
            the controlled call site, one per flattened quantum input
            qubit.
        bindings (dict[str, Any]): Bindings used while resolving vector
            input shapes.

    Returns:
        QubitMap: Mapping from the inner block's formal quantum input
            addresses to physical parent-circuit qubit indices.

    Raises:
        EmitError: If a vector input length cannot be resolved, is
            negative, or the block's quantum input footprint exceeds
            ``len(target_indices)``.
    """
    local_map: QubitMap = {}
    _populate_input_qubit_map(
        emit_pass,
        getattr(block_value, "input_values", []),
        len(target_indices),
        bindings,
        local_map,
    )
    return {address: target_indices[slot] for address, slot in local_map.items()}


def emit_controlled_operations(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    operations: list[Operation],
    control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit controlled versions of operations with operand mapping.

    Walks ``operations`` under the accumulated ``control_indices``,
    resolving every quantum operand through ``qubit_map`` so gates land
    on the physical qubit their operand refers to. Nested
    ``ControlledUOperation``s are not rejected: their own controls are
    resolved and composed with the outer controls, lowering e.g. an
    inner ``qmc.control(qmc.x, num_controls=k)`` under one outer
    control to a ``(k+1)``-controlled X.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        operations (list[Operation]): Block operations to walk.
        control_indices (list[int]): Accumulated physical control
            qubits.
        qubit_map (QubitMap): Mutable block-local qubit map seeded by
            :func:`build_controlled_block_qubit_map`; updated in place
            with gate / nested-op result addresses.
        bindings (dict[str, Any]): Bindings visible inside the block,
            including loop-iteration values during unrolling.

    Raises:
        EmitError: If an operand cannot be resolved to a physical
            qubit, a ``ForOperation`` bound cannot be resolved at
            transpile time, or an operation kind is unsupported in
            controlled decomposition.
    """
    for op in operations:
        if isinstance(op, GateOperation):
            gate_targets = _resolve_controlled_gate_targets(
                emit_pass, op, qubit_map, bindings
            )
            emit_multi_controlled_gate(
                emit_pass, circuit, op, control_indices, gate_targets, bindings
            )
            _propagate_controlled_gate_results(op, gate_targets, qubit_map)
        elif isinstance(op, BinOp):
            evaluate_binop(emit_pass, op, bindings)
        elif isinstance(op, ControlledUOperation):
            _emit_nested_controlled_u(
                emit_pass, circuit, op, control_indices, qubit_map, bindings
            )
        elif isinstance(op, CompositeGateOperation):
            composite_control_groups = [
                _expand_quantum_operands_to_phys(
                    emit_pass,
                    operand,
                    qubit_map,
                    bindings,
                    operation="CompositeGateOperation",
                )
                for operand in op.control_qubits
            ]
            composite_target_groups = [
                _expand_quantum_operands_to_phys(
                    emit_pass,
                    operand,
                    qubit_map,
                    bindings,
                    operation="CompositeGateOperation",
                )
                for operand in op.target_qubits
            ]
            composite_controls = [
                i for group in composite_control_groups for i in group
            ]
            composite_targets = [i for group in composite_target_groups for i in group]
            emit_controlled_composite_at_indices(
                emit_pass,
                circuit,
                op,
                control_indices,
                [*composite_controls, *composite_targets],
                bindings,
            )
            _map_operand_result_groups(
                [r for r in op.results if r.type.is_quantum()],
                composite_control_groups + composite_target_groups,
                qubit_map,
            )
        elif isinstance(op, InverseBlockOperation):
            inverse_control_groups = [
                _expand_quantum_operands_to_phys(
                    emit_pass,
                    operand,
                    qubit_map,
                    bindings,
                    operation="InverseBlockOperation",
                )
                for operand in op.control_qubits
            ]
            inverse_target_groups = [
                _expand_quantum_operands_to_phys(
                    emit_pass,
                    operand,
                    qubit_map,
                    bindings,
                    operation="InverseBlockOperation",
                )
                for operand in op.target_qubits
            ]
            inner_controls = [i for group in inverse_control_groups for i in group]
            inner_targets = [i for group in inverse_target_groups for i in group]
            # Imported lazily: inverse_emission imports this module's
            # shared helpers at module level, so a top-level import here
            # would be circular.
            from qamomile.circuit.transpiler.passes.emit_support.inverse_emission import (  # noqa: I001
                _map_inverse_block_results,
                emit_inverse_block_at_indices,
            )

            emit_inverse_block_at_indices(
                emit_pass,
                circuit,
                op,
                [*control_indices, *inner_controls],
                inner_targets,
                bindings,
            )
            _map_inverse_block_results(
                op, inverse_control_groups, inverse_target_groups, qubit_map
            )
        elif isinstance(op, PauliEvolveOp):
            emit_controlled_pauli_evolve(
                emit_pass, circuit, op, control_indices, qubit_map, bindings
            )
        elif isinstance(op, ForOperation):
            from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
                _bind_loop_var,
                resolve_loop_bounds,
            )

            start, stop, step = resolve_loop_bounds(emit_pass._resolver, op, bindings)
            if start is not None and stop is not None and step is not None:
                for i in range(start, stop, step):
                    loop_bindings = bindings.copy()
                    _bind_loop_var(loop_bindings, op, i)
                    emit_controlled_operations(
                        emit_pass,
                        circuit,
                        op.operations,
                        control_indices,
                        qubit_map,
                        loop_bindings,
                    )
            else:
                raise EmitError(
                    "Cannot resolve ForOperation bounds in controlled block. "
                    "Loop bounds must be resolvable at transpile time."
                )
        elif isinstance(op, HasNestedOps):
            raise EmitError(
                f"Unsupported control flow {type(op).__name__} in controlled "
                f"block decomposition. Only ForOperation is supported.",
                operation="ControlledGate",
            )
        elif isinstance(op, ReturnOperation):
            continue
        else:
            raise EmitError(
                f"Unsupported operation {type(op).__name__} in controlled "
                f"block decomposition.",
                operation="ControlledGate",
            )


def _resolve_controlled_gate_targets(
    emit_pass: "StandardEmitPass",
    op: GateOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> list[int]:
    """Resolve a gate operation's quantum operands to physical qubits.

    Args:
        emit_pass (StandardEmitPass): Emit pass whose resolver maps IR
            values to physical qubit indices.
        op (GateOperation): Gate operation being emitted under controls.
        qubit_map (QubitMap): Current block-local qubit map.
        bindings (dict[str, Any]): Bindings used for array element and
            slice resolution.

    Returns:
        list[int]: Physical target qubit indices in operand order.

    Raises:
        EmitError: If any gate operand cannot be resolved. Falling back
            to a positional slot would silently route the gate to the
            wrong physical qubit.
    """
    target_indices: list[int] = []
    for operand in op.qubit_operands:
        index = emit_pass._resolver.resolve_qubit_index(operand, qubit_map, bindings)
        if index is None:
            gate_name = op.gate_type.name if op.gate_type is not None else "<unknown>"
            raise EmitError(
                f"Controlled fallback cannot resolve operand "
                f"{operand.name!r} (uuid {operand.uuid[:8]}...) of inner "
                f"gate {gate_name} to a physical qubit.",
                operation="ControlledGate",
            )
        target_indices.append(index)
    return target_indices


def _propagate_controlled_gate_results(
    op: GateOperation,
    target_indices: list[int],
    qubit_map: QubitMap,
) -> None:
    """Propagate gate result values to their unchanged physical slots.

    Args:
        op (GateOperation): Gate operation whose results were just
            emitted. Gates never move qubits, so each result keeps the
            physical slot of the operand it versions.
        target_indices (list[int]): Physical qubit indices resolved from
            ``op.qubit_operands``.
        qubit_map (QubitMap): Mutable block-local qubit map to update.
    """
    quantum_results = [result for result in op.results if result.type.is_quantum()]
    for result, index in zip(quantum_results, target_indices):
        qubit_map[QubitAddress(result.uuid)] = index


def _map_operand_result_groups(
    results: list[Any],
    index_groups: list[list[int]],
    qubit_map: QubitMap,
) -> None:
    """Map result values onto the physical slots of their operands.

    Controlled emission never moves qubits, so each quantum result
    occupies exactly the physical slots its corresponding operand
    resolved to. Scalar results get a scalar address; ``ArrayValue``
    results get one address per element plus a base address.

    Args:
        results (list[Any]): Quantum results, paired positionally with
            the operand index groups.
        index_groups (list[list[int]]): Physical indices per operand as
            returned by :func:`_expand_quantum_operands_to_phys`.
        qubit_map (QubitMap): Mutable qubit map updated in place.
    """
    from qamomile.circuit.ir.value import ArrayValue

    for result, group in zip(results, index_groups):
        if isinstance(result, ArrayValue):
            map_array_result_group(result.uuid, group, qubit_map)
        elif group:
            qubit_map[QubitAddress(result.uuid)] = group[0]


@dataclasses.dataclass
class ResolvedControlledU:
    """Physical resolution of one ``ControlledUOperation`` call site.

    Produced by :func:`resolve_controlled_u_call` for every concrete and
    symbolic operand layout, so walkers can lower nested controlled-U
    operations without shape-specific branching.

    Attributes:
        control_phys (list[int]): Active physical control qubits, in
            control-operand order (for the ``control_indices`` form,
            in listed-index order).
        control_operand_groups (list[list[int]]): Physical indices per
            control operand slot. For the ``control_indices`` form this
            is the full pool — every element keeps its slot, whether or
            not it acts as a control.
        target_qubit_operands (list[Any]): Quantum sub-kernel operands
            in declaration order.
        target_index_groups (list[list[int]]): Physical indices per
            target operand.
        target_phys (list[int]): Flattened physical target qubits.
        block (Any): The inner block to apply under the controls.
        local_bindings (dict[str, Any]): Inner-block bindings with
            classical params bound and vector input shapes seeded.
    """

    control_phys: list[int]
    control_operand_groups: list[list[int]]
    target_qubit_operands: list[Any]
    target_index_groups: list[list[int]]
    target_phys: list[int]
    block: Any
    local_bindings: dict[str, Any]


def resolve_controlled_u_call(
    emit_pass: "StandardEmitPass",
    op: ControlledUOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> ResolvedControlledU:
    """Resolve a (possibly nested) controlled-U call site to physical qubits.

    Handles all three operand layouts:

    - ``ConcreteControlledU``: one scalar operand per control qubit.
    - ``SymbolicControlledU`` with ``control_indices is None``: a
      control prefix of ``num_control_args`` scalar / vector operands
      whose flattened qubit count equals the resolved ``num_controls``.
    - ``SymbolicControlledU`` with ``control_indices`` set: a single
      control pool whose listed elements act as controls while the rest
      pass through.

    Args:
        emit_pass (StandardEmitPass): Active emit pass; provides the
            value resolver.
        op (ControlledUOperation): The controlled-U operation to
            resolve.
        qubit_map (QubitMap): Qubit map the operands resolve against
            (block-local when called from a controlled walker).
        bindings (dict[str, Any]): Bindings visible at the call site,
            including loop-iteration values during unrolling.

    Returns:
        ResolvedControlledU: Physical controls / targets and the
            inner-block bindings.

    Raises:
        EmitError: If the operation has no inner block, ``num_controls``
            or a ``control_indices`` entry cannot be resolved, indices
            repeat or fall outside the pool, the flattened control
            count does not match ``num_controls``, or an operand cannot
            be expanded to physical qubits.
    """
    if op.block is None:
        raise EmitError(
            "Cannot resolve a ControlledUOperation without an inner block.",
            operation="ControlledUOperation",
        )

    if isinstance(op, SymbolicControlledU):
        resolved_nc = emit_pass._resolver.resolve_classical_value(
            op.num_controls, bindings
        )
        if resolved_nc is None:
            raise EmitError(
                f"Cannot resolve num_controls Value "
                f"{op.num_controls.name!r} for nested controlled-U emit.",
                operation="ControlledUOperation",
            )
        nc = int(resolved_nc)
        if nc <= 0:
            raise EmitError(
                f"Nested SymbolicControlledU resolved num_controls={nc}; "
                f"must be a strictly positive integer.",
                operation="ControlledUOperation",
            )
        num_control_args = op.num_control_args
    else:
        assert isinstance(op, ConcreteControlledU)
        nc = op.num_controls
        num_control_args = nc

    control_operands = list(op.operands[:num_control_args])
    control_operand_groups = [
        _expand_quantum_operands_to_phys(emit_pass, operand, qubit_map, bindings)
        for operand in control_operands
    ]

    if isinstance(op, SymbolicControlledU) and op.control_indices is not None:
        # ``control_indices`` selects a subset of a single control pool, so the
        # frontend rejects combining it with a multi-arg control prefix. Guard
        # the invariant here too: a malformed op (e.g. one reconstructed from a
        # wire payload) with ``num_control_args > 1`` would otherwise treat
        # ``control_operand_groups[0]`` as the whole pool and silently drop the
        # remaining control operands.
        if len(control_operand_groups) != 1:
            raise EmitError(
                f"SymbolicControlledU with control_indices must carry exactly "
                f"one control-pool operand, but num_control_args="
                f"{num_control_args}. Combining control_indices with a "
                f"multi-arg control prefix is not a valid operand layout.",
                operation="ControlledUOperation",
            )
        pool_phys = control_operand_groups[0]
        resolved_indices: list[int] = []
        for index_value in op.control_indices:
            resolved_index = emit_pass._resolver.resolve_classical_value(
                index_value, bindings
            )
            if resolved_index is None:
                raise EmitError(
                    f"Cannot resolve control_indices entry "
                    f"{index_value.name!r} for nested controlled-U emit.",
                    operation="ControlledUOperation",
                )
            resolved_indices.append(int(resolved_index))
        if len(resolved_indices) != nc:
            raise EmitError(
                f"control_indices length ({len(resolved_indices)}) does "
                f"not match num_controls ({nc}).",
                operation="ControlledUOperation",
            )
        if len(set(resolved_indices)) != len(resolved_indices):
            raise EmitError(
                f"control_indices contains duplicate entries: {resolved_indices}.",
                operation="ControlledUOperation",
            )
        for index in resolved_indices:
            if index < 0 or index >= len(pool_phys):
                raise EmitError(
                    f"control_indices entry {index} out of bounds for "
                    f"control pool of length {len(pool_phys)}.",
                    operation="ControlledUOperation",
                )
        control_phys = [pool_phys[index] for index in resolved_indices]
    else:
        control_phys = [i for group in control_operand_groups for i in group]
        if len(control_phys) != nc:
            raise EmitError(
                f"Controlled-U control operands expanded to "
                f"{len(control_phys)} qubit(s), but num_controls "
                f"resolves to {nc}.",
                operation="ControlledUOperation",
            )

    remaining_operands = op.operands[num_control_args:]
    target_qubit_operands = [v for v in remaining_operands if v.type.is_quantum()]
    param_operands = [
        v for v in remaining_operands if v.type.is_classical() or v.type.is_object()
    ]

    target_index_groups = [
        _expand_quantum_operands_to_phys(emit_pass, operand, qubit_map, bindings)
        for operand in target_qubit_operands
    ]
    target_phys = [i for group in target_index_groups for i in group]

    local_bindings = emit_pass._resolver.bind_block_params(
        op.block, param_operands, bindings
    )
    _bind_quantum_input_shapes(
        emit_pass, op.block, target_qubit_operands, bindings, local_bindings
    )

    return ResolvedControlledU(
        control_phys=control_phys,
        control_operand_groups=control_operand_groups,
        target_qubit_operands=target_qubit_operands,
        target_index_groups=target_index_groups,
        target_phys=target_phys,
        block=op.block,
        local_bindings=local_bindings,
    )


def map_nested_controlled_u_results(
    op: ControlledUOperation,
    resolved: ResolvedControlledU,
    qubit_map: QubitMap,
) -> None:
    """Map a nested controlled-U's result values to physical qubits.

    The result layout mirrors the operand layout for every controlled-U
    shape: one result per control operand slot followed by one result
    per sub-kernel quantum operand. Controlled gates only add a
    relative phase — no qubit moves — so each result keeps the physical
    slots of its operand. For the ``control_indices`` pool form the
    whole pool passes through, selected controls included.

    Args:
        op (ControlledUOperation): The operation whose results need
            mapping.
        resolved (ResolvedControlledU): Physical resolution returned by
            :func:`resolve_controlled_u_call` for this operation.
        qubit_map (QubitMap): Mutable qubit map updated in place.
    """
    num_control_results = len(resolved.control_operand_groups)
    control_results = [
        r for r in op.results[:num_control_results] if r.type.is_quantum()
    ]
    _map_operand_result_groups(
        control_results,
        resolved.control_operand_groups,
        qubit_map,
    )
    target_results = [
        r for r in op.results[num_control_results:] if r.type.is_quantum()
    ]
    _map_operand_result_groups(
        target_results,
        resolved.target_index_groups,
        qubit_map,
    )


def _emit_nested_controlled_u(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: ControlledUOperation,
    outer_control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a nested controlled-U by composing outer and inner controls.

    Resolves the nested operation's own controls and targets through
    the block-local ``qubit_map``, prepends the outer controls, and
    lowers the result. Backends whose ``circuit_to_gate`` works get a
    single native multi-controlled gate; others recurse through the
    mapped walker with the composed control set.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (ControlledUOperation): Nested controlled-U operation.
        outer_control_indices (list[int]): Physical controls accumulated
            from enclosing controlled-U operations.
        qubit_map (QubitMap): Current block-local qubit map; updated in
            place with the nested operation's result addresses.
        bindings (dict[str, Any]): Bindings visible inside the current
            block.

    Raises:
        EmitError: If the nested operation cannot be resolved or its
            block contains operations the walker cannot lower under the
            composed controls.
    """
    resolved = resolve_controlled_u_call(emit_pass, op, qubit_map, bindings)
    composed_controls = [*outer_control_indices, *resolved.control_phys]
    block = _prepare_nested_block_for_emit(resolved.block, resolved.local_bindings)
    power = resolve_power(emit_pass, op, bindings)

    unitary_gate = emit_pass._blockvalue_to_gate(
        block, len(resolved.target_phys), resolved.local_bindings
    )
    if unitary_gate is not None:
        if power > 1:
            unitary_gate = emit_pass._emitter.gate_power(unitary_gate, power)
        controlled_gate = emit_pass._emitter.gate_controlled(
            unitary_gate, len(composed_controls)
        )
        emit_pass._emitter.append_gate(
            circuit, controlled_gate, composed_controls + resolved.target_phys
        )
    else:
        inner_map = build_controlled_block_qubit_map(
            emit_pass, block, resolved.target_phys, resolved.local_bindings
        )
        for _ in range(power):
            emit_controlled_operations(
                emit_pass,
                circuit,
                block.operations,
                composed_controls,
                inner_map,
                resolved.local_bindings,
            )

    map_nested_controlled_u_results(op, resolved, qubit_map)


def emit_controlled_gate(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: GateOperation,
    control_idx: int,
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit a controlled version of a gate."""
    if not target_indices:
        return

    target_idx = target_indices[0]

    match op.gate_type:
        case GateOperationType.H:
            emit_pass._emitter.emit_ch(circuit, control_idx, target_idx)
        case GateOperationType.X:
            emit_pass._emitter.emit_cx(circuit, control_idx, target_idx)
        case GateOperationType.Y:
            emit_pass._emitter.emit_cy(circuit, control_idx, target_idx)
        case GateOperationType.Z:
            emit_pass._emitter.emit_cz(circuit, control_idx, target_idx)
        case GateOperationType.P:
            angle = emit_pass._resolve_angle(op, bindings)
            emit_pass._emitter.emit_cp(circuit, control_idx, target_idx, angle)
        case GateOperationType.RX:
            angle = emit_pass._resolve_angle(op, bindings)
            emit_pass._emitter.emit_crx(circuit, control_idx, target_idx, angle)
        case GateOperationType.RY:
            angle = emit_pass._resolve_angle(op, bindings)
            emit_pass._emitter.emit_cry(circuit, control_idx, target_idx, angle)
        case GateOperationType.RZ:
            angle = emit_pass._resolve_angle(op, bindings)
            emit_pass._emitter.emit_crz(circuit, control_idx, target_idx, angle)
        case GateOperationType.S:
            emit_pass._emitter.emit_cp(circuit, control_idx, target_idx, math.pi / 2)
        case GateOperationType.T:
            emit_pass._emitter.emit_cp(circuit, control_idx, target_idx, math.pi / 4)
        case GateOperationType.SDG:
            emit_pass._emitter.emit_cp(circuit, control_idx, target_idx, -math.pi / 2)
        case GateOperationType.TDG:
            emit_pass._emitter.emit_cp(circuit, control_idx, target_idx, -math.pi / 4)
        case GateOperationType.SWAP:
            if len(target_indices) < 2:
                raise EmitError(
                    "Controlled-SWAP requires at least 2 target qubits.",
                    operation="ControlledGate",
                )
            tgt_a = target_indices[0]
            tgt_b = target_indices[1]
            # Fredkin gate decomposition:
            #   CNOT(tgt_b, tgt_a)
            #   Toffoli(ctrl, tgt_a, tgt_b)
            #   CNOT(tgt_b, tgt_a)
            emit_pass._emitter.emit_cx(circuit, tgt_b, tgt_a)
            emit_pass._emitter.emit_toffoli(circuit, control_idx, tgt_a, tgt_b)
            emit_pass._emitter.emit_cx(circuit, tgt_b, tgt_a)
        case _:
            raise EmitError(
                f"Unsupported gate type {op.gate_type!r} in controlled "
                f"block decomposition.",
                operation="ControlledGate",
            )


def emit_multi_controlled_gate(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: GateOperation,
    control_indices: list[int],
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit one gate under an arbitrary number of accumulated controls.

    Structurally reduces multi-qubit gate types to single-target forms
    by absorbing their own control qubits into the control set
    (``CX -> X``, ``CZ -> Z``, ``CP -> P``, ``TOFFOLI -> X``; ``SWAP``
    and ``RZZ`` via their standard CX conjugations), then emits:

    - one control: via :func:`emit_controlled_gate` (the existing
      single-control dispatch),
    - two controls on X / Z: via ``emit_toffoli`` (Z conjugated by H),
    - anything else: via the backend's
      ``_emit_irreducible_multi_controlled_gate`` hook, whose base
      implementation raises a descriptive ``EmitError``.

    Args:
        emit_pass (StandardEmitPass): Active emit pass. Subclass
            overrides of ``_emit_irreducible_multi_controlled_gate``
            are respected for the irreducible tail.
        circuit (Any): Backend circuit being emitted into.
        op (GateOperation): Gate operation whose operands are already
            resolved to ``target_indices``.
        control_indices (list[int]): Physical control qubits. Must be
            non-empty.
        target_indices (list[int]): Physical qubits for the gate's own
            operands, in operand order.
        bindings (dict[str, Any]): Bindings used to resolve rotation
            angles.

    Raises:
        EmitError: If ``control_indices`` is empty, the gate has fewer
            resolved targets than its type requires, or the reduction
            bottoms out on a backend without multi-control support.
    """
    gate_type = op.gate_type
    if not control_indices:
        raise EmitError(
            "emit_multi_controlled_gate requires at least one control.",
            operation="ControlledGate",
        )

    def _require_targets(count: int) -> None:
        """Validate that the gate received enough resolved targets.

        Args:
            count (int): Minimum number of physical targets required.

        Raises:
            EmitError: If fewer than ``count`` targets were resolved.
        """
        if len(target_indices) < count:
            gate_name = gate_type.name if gate_type is not None else "<unknown>"
            raise EmitError(
                f"Controlled-{gate_name} requires {count} target "
                f"qubit(s); got {len(target_indices)}.",
                operation="ControlledGate",
            )

    match gate_type:
        case GateOperationType.CX:
            _require_targets(2)
            _emit_mc_x(
                emit_pass,
                circuit,
                [*control_indices, target_indices[0]],
                target_indices[1],
            )
            return
        case GateOperationType.CZ:
            _require_targets(2)
            _emit_mc_z(
                emit_pass,
                circuit,
                [*control_indices, target_indices[0]],
                target_indices[1],
            )
            return
        case GateOperationType.TOFFOLI:
            _require_targets(3)
            _emit_mc_x(
                emit_pass,
                circuit,
                [*control_indices, target_indices[0], target_indices[1]],
                target_indices[2],
            )
            return
        case GateOperationType.CP:
            _require_targets(2)
            angle = emit_pass._resolve_angle(op, bindings)
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.P,
                [*control_indices, target_indices[0]],
                target_indices[1],
                angle,
            )
            return
        case GateOperationType.RZZ:
            _require_targets(2)
            angle = emit_pass._resolve_angle(op, bindings)
            # RZZ = CX(t0, t1) . RZ(t1) . CX(t0, t1); the CX pair is
            # self-inverse, so only the RZ needs the controls.
            emit_pass._emitter.emit_cx(circuit, target_indices[0], target_indices[1])
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.RZ,
                control_indices,
                target_indices[1],
                angle,
            )
            emit_pass._emitter.emit_cx(circuit, target_indices[0], target_indices[1])
            return
        case GateOperationType.SWAP:
            _require_targets(2)
            # Generalized Fredkin: SWAP(a, b) = CX(b, a) . CX(a, b)
            # . CX(b, a); controlling only the middle CX controls the
            # whole SWAP.
            emit_pass._emitter.emit_cx(circuit, target_indices[1], target_indices[0])
            _emit_mc_x(
                emit_pass,
                circuit,
                [*control_indices, target_indices[0]],
                target_indices[1],
            )
            emit_pass._emitter.emit_cx(circuit, target_indices[1], target_indices[0])
            return
        case _:
            pass

    if len(control_indices) == 1:
        emit_controlled_gate(
            emit_pass, circuit, op, control_indices[0], target_indices, bindings
        )
        return

    _require_targets(1)
    target_idx = target_indices[0]
    match gate_type:
        case GateOperationType.X:
            _emit_mc_x(emit_pass, circuit, control_indices, target_idx)
        case GateOperationType.Z:
            _emit_mc_z(emit_pass, circuit, control_indices, target_idx)
        case GateOperationType.S:
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.P,
                control_indices,
                target_idx,
                math.pi / 2,
            )
        case GateOperationType.T:
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.P,
                control_indices,
                target_idx,
                math.pi / 4,
            )
        case GateOperationType.SDG:
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.P,
                control_indices,
                target_idx,
                -math.pi / 2,
            )
        case GateOperationType.TDG:
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.P,
                control_indices,
                target_idx,
                -math.pi / 4,
            )
        case (
            GateOperationType.P
            | GateOperationType.RX
            | GateOperationType.RY
            | GateOperationType.RZ
        ):
            angle = emit_pass._resolve_angle(op, bindings)
            _emit_mc_rotation(
                emit_pass,
                circuit,
                gate_type,
                control_indices,
                target_idx,
                angle,
            )
        case _:
            _emit_irreducible(
                emit_pass, circuit, gate_type, control_indices, target_idx, None
            )


def _emit_mc_x(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    control_indices: list[int],
    target_idx: int,
) -> None:
    """Emit an X gate under one or more controls.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        control_indices (list[int]): Physical control qubits (>= 1).
        target_idx (int): Physical target qubit.

    Raises:
        EmitError: If three or more controls are required and the
            backend has no multi-controlled gate hook.
    """
    if len(control_indices) == 1:
        emit_pass._emitter.emit_cx(circuit, control_indices[0], target_idx)
    elif len(control_indices) == 2:
        emit_pass._emitter.emit_toffoli(
            circuit, control_indices[0], control_indices[1], target_idx
        )
    else:
        _emit_irreducible(
            emit_pass, circuit, GateOperationType.X, control_indices, target_idx, None
        )


def _emit_mc_z(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    control_indices: list[int],
    target_idx: int,
) -> None:
    """Emit a Z gate under one or more controls.

    Two-control Z is conjugated into a Toffoli by Hadamards on the
    target (``Z = H X H``).

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        control_indices (list[int]): Physical control qubits (>= 1).
        target_idx (int): Physical target qubit.

    Raises:
        EmitError: If three or more controls are required and the
            backend has no multi-controlled gate hook.
    """
    if len(control_indices) == 1:
        emit_pass._emitter.emit_cz(circuit, control_indices[0], target_idx)
    elif len(control_indices) == 2:
        emit_pass._emitter.emit_h(circuit, target_idx)
        emit_pass._emitter.emit_toffoli(
            circuit, control_indices[0], control_indices[1], target_idx
        )
        emit_pass._emitter.emit_h(circuit, target_idx)
    else:
        _emit_irreducible(
            emit_pass, circuit, GateOperationType.Z, control_indices, target_idx, None
        )


def _emit_mc_rotation(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    gate_type: GateOperationType,
    control_indices: list[int],
    target_idx: int,
    angle: Any,
) -> None:
    """Emit a rotation-like gate under one or more controls.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        gate_type (GateOperationType): One of ``P`` / ``RX`` / ``RY`` /
            ``RZ``.
        control_indices (list[int]): Physical control qubits (>= 1).
        target_idx (int): Physical target qubit.
        angle (Any): Resolved rotation angle (concrete float or backend
            parameter expression).

    Raises:
        EmitError: If two or more controls are required and the backend
            has no multi-controlled gate hook, or ``gate_type`` is not
            rotation-like.
    """
    if len(control_indices) == 1:
        control_idx = control_indices[0]
        match gate_type:
            case GateOperationType.P:
                emit_pass._emitter.emit_cp(circuit, control_idx, target_idx, angle)
            case GateOperationType.RX:
                emit_pass._emitter.emit_crx(circuit, control_idx, target_idx, angle)
            case GateOperationType.RY:
                emit_pass._emitter.emit_cry(circuit, control_idx, target_idx, angle)
            case GateOperationType.RZ:
                emit_pass._emitter.emit_crz(circuit, control_idx, target_idx, angle)
            case _:
                raise EmitError(
                    f"Gate type {gate_type!r} is not rotation-like.",
                    operation="ControlledGate",
                )
        return
    _emit_irreducible(emit_pass, circuit, gate_type, control_indices, target_idx, angle)


def _emit_irreducible(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    gate_type: GateOperationType | None,
    control_indices: list[int],
    target_idx: int,
    angle: Any,
) -> None:
    """Dispatch an irreducible multi-controlled gate to the backend hook.

    Args:
        emit_pass (StandardEmitPass): Active emit pass. Its
            ``_emit_irreducible_multi_controlled_gate`` method (base
            implementation raises; backends may override) receives the
            gate.
        circuit (Any): Backend circuit being emitted into.
        gate_type (GateOperationType | None): Single-qubit gate kind.
        control_indices (list[int]): Physical control qubits.
        target_idx (int): Physical target qubit.
        angle (Any): Resolved rotation angle, or ``None`` for fixed
            gates.

    Raises:
        EmitError: If ``gate_type`` is missing or the backend's hook
            rejects the gate.
    """
    if gate_type is None:
        raise EmitError(
            f"Cannot emit {len(control_indices)}-controlled gate without a gate type.",
            operation="ControlledGate",
        )
    emit_pass._emit_irreducible_multi_controlled_gate(
        circuit, gate_type, control_indices, target_idx, angle
    )


def emit_controlled_pauli_evolve(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: PauliEvolveOp,
    control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit ``exp(-i * gamma * H)`` under accumulated controls.

    Lowers a controlled Pauli evolution by exploiting that each Pauli
    term's evolution ``exp(-i * theta * P)`` factors as
    ``U_dagger * RZ(2 * theta) * U`` where ``U`` (the basis change onto
    the Z axis followed by the parity CX ladder) is Clifford and thus
    control-independent. Controlling a conjugated gate is
    ``U_dagger * C[RZ] * U`` with ``U`` / ``U_dagger`` applied
    unconditionally: when every control is ``0`` the central ``RZ``
    becomes the identity and ``U_dagger * I * U = I``. Only the central
    ``RZ`` therefore carries the controls, which keeps the dense
    multi-controlled cost to a single rotation per Hamiltonian term
    instead of controlling every basis-change and ladder gate.

    The basis-change (``H`` / ``SDG`` / ``S``) and CX-ladder gates are
    emitted uncontrolled through ``emit_pass._emitter``; the central
    ``RZ`` is routed through :func:`_emit_mc_rotation`, which dispatches
    to ``emit_crz`` for a single control and to the backend's
    ``_emit_irreducible_multi_controlled_gate`` hook for two or more.

    A constant (identity) Hamiltonian term ``c * I`` is a global phase
    ``exp(-i * gamma * c)`` for the uncontrolled evolution (correctly
    dropped by :func:`emit_pauli_evolve`), but under controls it becomes
    an *observable* relative phase on the all-controls-on subspace, so it
    MUST be emitted here. It is realized as a ``P(-gamma * c)`` on one
    control conditioned on the remaining controls (``emit_p`` for a single
    control, a controlled / dense ``P`` for more), matching Qiskit's
    native ``PauliEvolutionGate`` whose ``SparsePauliOp`` carries the
    constant.

    Args:
        emit_pass (StandardEmitPass): Active emit pass; provides the
            value resolver, gate emitter, and the multi-controlled
            rotation hook.
        circuit (Any): Backend circuit being emitted into.
        op (PauliEvolveOp): The Pauli evolution operation inside the
            controlled block.
        control_indices (list[int]): Accumulated physical control
            qubits. Must be non-empty.
        qubit_map (QubitMap): Block-local qubit map the operation's
            quantum operands resolve against; updated in place with the
            evolved-register result addresses.
        bindings (dict[str, Any]): Bindings visible inside the block,
            used to resolve the Hamiltonian, gamma, and register shape.

    Raises:
        EmitError: If ``control_indices`` is empty, the observable does
            not resolve to a Hamiltonian, gamma cannot be resolved, the
            Hamiltonian is non-Hermitian (a term or the constant has a
            non-real coefficient), the register size does not match the
            Hamiltonian, a term qubit cannot be resolved, or (for two or
            more controls, or a nonzero constant term) the angle is
            runtime-parametric and the backend's dense multi-controlled
            path requires a compile-time-numeric angle.
    """
    import qamomile.observable as qm_o
    from qamomile.circuit.transpiler.passes.emit_support.pauli_evolve_emission import (
        _resolve_gamma,
    )
    from qamomile.observable.hamiltonian import (
        HERMITIAN_IMAG_ATOL,
        PAULI_TERM_ZERO_ATOL,
    )

    if not control_indices:
        raise EmitError(
            "emit_controlled_pauli_evolve requires at least one control.",
            operation="PauliEvolveOp",
        )

    hamiltonian = emit_pass._resolver.resolve_bound_value(op.observable, bindings)
    if not isinstance(hamiltonian, qm_o.Hamiltonian):
        raise EmitError(
            f"PauliEvolveOp requires a Hamiltonian binding. "
            f"Observable '{op.observable.name}' not found or not a Hamiltonian.",
            operation="PauliEvolveOp",
        )

    gamma = _resolve_gamma(emit_pass, op, bindings)
    if gamma is None:
        raise EmitError(
            "Cannot resolve gamma parameter for PauliEvolveOp. "
            "gamma must be a concrete float binding or a declared "
            "parameter (scalar or array element).",
            operation="PauliEvolveOp",
        )

    def scaled_gamma(factor: float) -> Any:
        """Scale ``gamma`` by a real ``factor`` for a controlled rotation angle.

        ``gamma`` is either a concrete ``float`` (compile-time bound) or a
        backend runtime-parameter expression. ``factor`` is the term-specific
        real scale: ``2 * coeff`` for a Pauli term's central RZ, or
        ``-constant`` for the identity-term phase.

        Args:
            factor (float): Real multiplier applied to ``gamma``.

        Returns:
            Any: ``factor * gamma`` — a Python ``float`` for concrete gamma,
                or a backend parameter expression for a runtime-parametric one.

        Raises:
            EmitError: If ``gamma`` is a runtime parameter whose backend type
                exposes no Python arithmetic (e.g. QURI Parts' Rust-backed
                ``Parameter``), so the scaling cannot be expressed. The raw
                ``TypeError`` is converted into a clear compile-time error
                pointing at binding ``gamma`` to a concrete value. (A
                concrete ``float`` gamma never raises; two-or-more controls
                additionally need a numeric angle because the dense
                multi-controlled matrix bakes the angle in.)
        """
        try:
            return factor * gamma
        except TypeError as exc:
            raise EmitError(
                "Controlled Pauli evolution requires a compile-time-numeric "
                "gamma on this backend: its runtime parameter type does not "
                "support the angle scaling needed for the controlled "
                "rotations. Bind gamma to a concrete value before "
                "transpilation.",
                operation="PauliEvolveOp",
            ) from exc

    # Resolve the target register against the block-local qubit map.
    # ``_expand_quantum_operands_to_phys`` walks the operand's slice chain and
    # returns one physical qubit index per element — the controlled-emission
    # analogue of the ``resolve_slice_chain`` lookup the uncontrolled path
    # performs.
    qubit_indices = _expand_quantum_operands_to_phys(
        emit_pass, op.qubits, qubit_map, bindings, operation="PauliEvolveOp"
    )
    if len(qubit_indices) != hamiltonian.num_qubits:
        raise EmitError(
            f"PauliEvolveOp qubit count mismatch: qubit register has "
            f"{len(qubit_indices)} qubits but Hamiltonian acts on "
            f"{hamiltonian.num_qubits} qubits.",
            operation="PauliEvolveOp",
        )

    for operators, coeff in hamiltonian:
        # Validate Hermiticity of every term (including ones skipped below)
        # before emitting it, folded into the emission pass to avoid a second
        # walk over a large Hamiltonian.
        if abs(coeff.imag) > HERMITIAN_IMAG_ATOL:
            raise EmitError(
                f"PauliEvolveOp requires a Hermitian Hamiltonian "
                f"(real coefficients), but found complex coefficient "
                f"{coeff} on term {operators}.",
                operation="PauliEvolveOp",
            )
        if abs(coeff) < PAULI_TERM_ZERO_ATOL or len(operators) == 0:
            continue
        # RZ(theta) = exp(-i*theta*Z/2), so exp(-i*gamma*coeff*P) needs the
        # central rotation theta = 2*gamma*coeff.
        angle = scaled_gamma(2.0 * float(coeff.real))
        term_qubit_indices = [qubit_indices[item.index] for item in operators]
        pauli_types = [item.pauli for item in operators]

        # Basis change onto Z (uncontrolled; the Clifford ``U``).
        for qi, pi in zip(term_qubit_indices, pauli_types):
            if pi == qm_o.Pauli.X:
                emit_pass._emitter.emit_h(circuit, qi)
            elif pi == qm_o.Pauli.Y:
                emit_pass._emitter.emit_sdg(circuit, qi)
                emit_pass._emitter.emit_h(circuit, qi)
            # Z and I are already diagonal in the Z basis: no basis change.

        # CX ladder (uncontrolled) wrapping the controlled central RZ.
        if len(term_qubit_indices) == 1:
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.RZ,
                control_indices,
                term_qubit_indices[0],
                angle,
            )
        else:
            for step in range(len(term_qubit_indices) - 1):
                emit_pass._emitter.emit_cx(
                    circuit,
                    term_qubit_indices[step],
                    term_qubit_indices[step + 1],
                )
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.RZ,
                control_indices,
                term_qubit_indices[-1],
                angle,
            )
            for step in range(len(term_qubit_indices) - 2, -1, -1):
                emit_pass._emitter.emit_cx(
                    circuit,
                    term_qubit_indices[step],
                    term_qubit_indices[step + 1],
                )

        # Undo basis change (uncontrolled; the Clifford ``U_dagger``).
        for qi, pi in reversed(list(zip(term_qubit_indices, pauli_types))):
            if pi == qm_o.Pauli.X:
                emit_pass._emitter.emit_h(circuit, qi)
            elif pi == qm_o.Pauli.Y:
                emit_pass._emitter.emit_h(circuit, qi)
                emit_pass._emitter.emit_s(circuit, qi)
            # Z and I had no basis change to undo.

    # Controlled constant (identity) term. ``exp(-i*gamma*c*I)`` is a global
    # phase for the uncontrolled evolution, but under controls it is an
    # observable relative phase on the all-controls-on subspace, so it must
    # be emitted. ``P(lambda) = diag(1, e^{i*lambda})`` with lambda = -gamma*c
    # puts e^{-i*gamma*c} on the all-ones control subspace (no factor of two:
    # this is a direct phase, not an RZ). It is applied to one control,
    # conditioned on the rest.
    constant = hamiltonian.constant
    if abs(constant) > PAULI_TERM_ZERO_ATOL:
        if abs(constant.imag) > HERMITIAN_IMAG_ATOL:
            raise EmitError(
                f"PauliEvolveOp requires a Hermitian Hamiltonian (real "
                f"coefficients), but found a complex constant {constant}.",
                operation="PauliEvolveOp",
            )
        # exp(-i*gamma*c*I) -> e^{-i*gamma*c} on the all-controls-on subspace,
        # i.e. P(lambda) with lambda = -gamma*c (no factor of two: a direct
        # phase, not an RZ).
        phase = scaled_gamma(-float(constant.real))
        if len(control_indices) == 1:
            emit_pass._emitter.emit_p(circuit, control_indices[0], phase)
        else:
            _emit_mc_rotation(
                emit_pass,
                circuit,
                GateOperationType.P,
                control_indices[:-1],
                control_indices[-1],
                phase,
            )

    # PauliEvolve never moves qubits: the evolved register occupies the
    # same physical slots as the input register.
    _map_operand_result_groups([op.evolved_qubits], [qubit_indices], qubit_map)


def resolve_power(
    emit_pass: "StandardEmitPass",
    op: ControlledUOperation,
    bindings: dict[str, Any],
) -> int:
    """Resolve ``ControlledUOperation.power`` to a concrete ``int``."""
    power = op.power

    if isinstance(power, int):
        return power

    if isinstance(power, Value):
        resolved = emit_pass._resolver.resolve_classical_value(power, bindings)
        if resolved is None:
            raise EmitError(
                f"Cannot resolve ControlledU power '{power.name}'. "
                f"Ensure all parameters are bound before transpilation.",
                operation="ControlledUOperation",
            )
        return int(resolved)

    raise EmitError(
        f"ControlledU power has unexpected type "
        f"{type(power).__name__}. Expected int or Value.",
        operation="ControlledUOperation",
    )


def emit_controlled_u_with_symbolic_indices(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: SymbolicControlledU,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a ``SymbolicControlledU`` whose ``control_indices`` is set.

    Routed from :func:`emit_controlled_u` when the constant-folding
    pass has left the op as ``SymbolicControlledU`` because
    ``control_indices`` carries pass-through semantics that the
    ``ConcreteControlledU`` promotion cannot represent in its scalar
    control-operand layout (see §12.5 of the design).

    The function resolves the symbolic ``num_controls`` and every
    ``control_indices`` entry to concrete ints, walks the control
    pool's ``slice_of`` chain to look up physical qubits per element,
    builds ``control_phys`` from the selected pool slots, expands the
    sub-kernel quantum operands via the §12.1 helper
    ``_expand_quantum_operands_to_phys`` for the target side, and
    threads the rest through the standard ``gate_controlled`` +
    ``append_gate`` pipeline (with the per-gate fallback for backends
    whose ``circuit_to_gate`` returns ``None``).

    Args:
        emit_pass (StandardEmitPass): The emit pass driving the
            conversion; provides ``_resolver``, ``_emitter``,
            ``_blockvalue_to_gate``, and ``_emit_controlled_fallback``.
        circuit (Any): The backend circuit being built.
        op (SymbolicControlledU): The IR op with ``control_indices``
            **not** ``None``.  Callers must guarantee this; the
            ``control_indices is None`` branch is handled by the
            constant-folding promotion to ``ConcreteControlledU``.
        qubit_map (QubitMap): The active ``QubitAddress`` -> physical
            qubit map; mutated in place with the result-side mappings.
        bindings (dict[str, Any]): Caller bindings used to resolve
            ``num_controls``, ``control_indices`` Values, ``c_qs``
            length, and slice bounds.

    Raises:
        EmitError: Surfaces under any of the following conditions:

            * ``num_controls`` cannot be resolved or is ``<= 0``;
            * the control-vector length cannot be resolved, is shorter
              than ``num_controls``, or is shorter than the largest
              listed index;
            * a ``control_indices`` entry cannot be resolved, is
              negative, or repeats;
            * ``len(control_indices) != num_controls``;
            * a sub-quantum operand cannot be expanded to physical
              qubits (delegated to
              :func:`_expand_quantum_operands_to_phys`).
    """
    assert op.control_indices is not None, (
        "emit_controlled_u_with_symbolic_indices requires "
        "control_indices to be set; the None branch is handled by "
        "the constant-folding promotion to ConcreteControlledU."
    )

    resolved_nc = emit_pass._resolver.resolve_classical_value(op.num_controls, bindings)
    if resolved_nc is None:
        raise EmitError(
            f"Cannot resolve num_controls Value "
            f"{op.num_controls.name!r} for SymbolicControlledU emit.",
            operation="ControlledUOperation",
        )
    nc = int(resolved_nc)
    if nc <= 0:
        raise EmitError(
            f"SymbolicControlledU resolved num_controls={nc}; must be "
            f"a strictly positive integer.",
            operation="ControlledUOperation",
        )

    resolved_indices: list[int] = []
    for v in op.control_indices:
        idx = emit_pass._resolver.resolve_classical_value(v, bindings)
        if idx is None:
            raise EmitError(
                f"Cannot resolve control_indices entry {v.name!r} "
                f"for SymbolicControlledU emit.",
                operation="ControlledUOperation",
            )
        idx_int = int(idx)
        if idx_int < 0:
            raise EmitError(
                f"Negative control_indices entry ({idx_int}) is not allowed.",
                operation="ControlledUOperation",
            )
        resolved_indices.append(idx_int)

    if len(resolved_indices) != nc:
        raise EmitError(
            f"control_indices length ({len(resolved_indices)}) does "
            f"not match num_controls ({nc}).",
            operation="ControlledUOperation",
        )
    if len(set(resolved_indices)) != len(resolved_indices):
        raise EmitError(
            f"control_indices contains duplicate entries: {resolved_indices}.",
            operation="ControlledUOperation",
        )

    from qamomile.circuit.ir.value import ArrayValue as _ArrayValue

    vector_value = op.operands[0]
    if not isinstance(vector_value, _ArrayValue):
        raise EmitError(
            "SymbolicControlledU expects an ArrayValue as the first "
            "operand (the control pool).",
            operation="ControlledUOperation",
        )
    size_val = vector_value.shape[0]
    vector_size = emit_pass._resolver.resolve_int_value(size_val, bindings)
    if vector_size is None:
        raise EmitError(
            "Cannot resolve control vector size for SymbolicControlledU emit.",
            operation="ControlledUOperation",
        )
    if vector_size < nc:
        raise EmitError(
            f"SymbolicControlledU: control vector length ({vector_size}) "
            f"is smaller than num_controls ({nc}); the pool cannot "
            f"supply enough control qubits.",
            operation="ControlledUOperation",
        )
    for idx in resolved_indices:
        if idx >= vector_size:
            raise EmitError(
                f"control_indices entry {idx} out of bounds for "
                f"control vector of length {vector_size}.",
                operation="ControlledUOperation",
            )

    root_av, slice_start, slice_step = emit_pass._resolver.resolve_slice_chain(
        vector_value, bindings, operation="ControlledUOperation"
    )
    pool_phys: list[int] = []
    for i in range(vector_size):
        addr = QubitAddress(root_av.uuid, slice_start + slice_step * i)
        if addr not in qubit_map:
            raise EmitError(
                f"Expected qubit address {addr!s} for SymbolicControlledU "
                f"control element {i} not found in qubit_map.",
                operation="ControlledUOperation",
            )
        pool_phys.append(qubit_map[addr])
    control_phys = [pool_phys[i] for i in resolved_indices]

    remaining_operands = op.operands[1:]
    target_qubit_operands = [v for v in remaining_operands if v.type.is_quantum()]
    param_operands = [
        v for v in remaining_operands if v.type.is_classical() or v.type.is_object()
    ]

    target_indices: list[int] = []
    target_index_groups: list[list[int]] = []
    for q in target_qubit_operands:
        indices = _expand_quantum_operands_to_phys(emit_pass, q, qubit_map, bindings)
        target_index_groups.append(indices)
        target_indices.extend(indices)

    block_value = op.block
    local_bindings = emit_pass._resolver.bind_block_params(
        block_value, param_operands, bindings
    )
    _bind_quantum_input_shapes(
        emit_pass, block_value, target_qubit_operands, bindings, local_bindings
    )
    block_value = _prepare_nested_block_for_emit(block_value, local_bindings)

    num_targets = len(target_indices)
    unitary_gate = emit_pass._blockvalue_to_gate(
        block_value, num_targets, local_bindings
    )

    power_value = resolve_power(emit_pass, op, bindings)

    if unitary_gate is not None:
        if power_value > 1:
            unitary_gate = emit_pass._emitter.gate_power(unitary_gate, power_value)
        controlled_gate = emit_pass._emitter.gate_controlled(unitary_gate, nc)
        emit_pass._emitter.append_gate(
            circuit, controlled_gate, control_phys + target_indices
        )
    else:
        emit_pass._emit_controlled_fallback(
            circuit,
            block_value,
            nc,
            control_phys,
            target_indices,
            power_value,
            local_bindings,
        )

    # Map result ArrayValue (c_qs_out) in qubit_map.  Every input
    # element keeps its physical qubit, so per-element addresses map
    # 1:1 — pass-through elements stay where they were and controls
    # also occupy the same physical slot they came from (the controlled
    # gate only adds a relative phase under the on-state of those
    # qubits, it does not move them).
    vector_result = op.results[0]
    for i in range(vector_size):
        result_addr = QubitAddress(vector_result.uuid, i)
        input_addr = QubitAddress(root_av.uuid, slice_start + slice_step * i)
        if input_addr in qubit_map and result_addr not in qubit_map:
            qubit_map[result_addr] = qubit_map[input_addr]

    # Sub-quantum result bookkeeping mirrors the ConcreteControlledU
    # path so downstream lookups via ``view_out[i]`` resolve.
    sub_quantum_results = [r for r in op.results[1:] if r.type.is_quantum()]
    for i, result in enumerate(sub_quantum_results):
        if i >= len(target_index_groups):
            break
        indices = target_index_groups[i]
        if isinstance(result, _ArrayValue):
            map_array_result_group(result.uuid, indices, qubit_map)
        else:
            if indices:
                qubit_map[QubitAddress(result.uuid)] = indices[0]


def emit_controlled_u_multi_arg(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: SymbolicControlledU,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a ``SymbolicControlledU`` whose control prefix is multi-arg.

    The multi-arg form (``num_control_args > 1``) carries a
    heterogeneous control prefix: a mix of scalar ``Value``
    (single-qubit) and ``ArrayValue`` (whole ``Vector`` / slice
    ``VectorView``) operands whose qubit-count sum equals
    ``num_controls`` once ``bindings`` resolves the lengths.
    ``ConstantFoldingPass`` cannot promote this shape to
    ``ConcreteControlledU`` (the per-control-qubit operand layout
    would lose the array-grouping the resource allocator needs), so
    it survives intact and lands here.

    The function expands every control operand into physical qubit
    indices via the §12.1 helper, asserts the resulting count
    matches the resolved ``num_controls``, expands the target side
    the same way, and threads the rest through the standard
    ``gate_controlled`` + ``append_gate`` pipeline (with the
    per-gate fallback for backends whose ``circuit_to_gate`` returns
    ``None``).

    Args:
        emit_pass (StandardEmitPass): The emit pass driving the
            conversion.
        circuit (Any): The backend circuit being built.
        op (SymbolicControlledU): The IR op with
            ``num_control_args > 1`` and ``control_indices is
            None``.
        qubit_map (QubitMap): The active ``QubitAddress`` ->
            physical qubit map; mutated in place with the
            result-side mappings.
        bindings (dict[str, Any]): Caller bindings used to resolve
            ``num_controls`` and any symbolic-length control / target
            operand.

    Raises:
        EmitError: ``num_controls`` cannot be resolved, is non-
            positive, or does not match the sum of expanded control
            operand sizes.
    """
    resolved_nc = emit_pass._resolver.resolve_classical_value(op.num_controls, bindings)
    if resolved_nc is None:
        raise EmitError(
            f"Cannot resolve num_controls Value "
            f"{op.num_controls.name!r} for multi-arg SymbolicControlledU emit.",
            operation="ControlledUOperation",
        )
    nc = int(resolved_nc)
    if nc <= 0:
        raise EmitError(
            f"SymbolicControlledU resolved num_controls={nc}; must be "
            f"a strictly positive integer.",
            operation="ControlledUOperation",
        )

    # Expand the control prefix one operand at a time.  Each operand
    # may be a scalar Value (one physical qubit) or an ArrayValue
    # (one physical qubit per element).
    control_operands = op.operands[: op.num_control_args]
    control_index_groups = [
        _expand_quantum_operands_to_phys(emit_pass, q, qubit_map, bindings)
        for q in control_operands
    ]
    control_phys = [index for group in control_index_groups for index in group]

    if len(control_phys) != nc:
        raise EmitError(
            f"Multi-arg SymbolicControlledU: control operands expanded "
            f"to {len(control_phys)} qubit(s), but num_controls "
            f"resolves to {nc}.  The sum of qubit counts across the "
            f"control prefix args must equal num_controls.",
            operation="ControlledUOperation",
        )

    remaining_operands = op.operands[op.num_control_args :]
    target_qubit_operands = [v for v in remaining_operands if v.type.is_quantum()]
    param_operands = [
        v for v in remaining_operands if v.type.is_classical() or v.type.is_object()
    ]

    target_indices: list[int] = []
    target_index_groups: list[list[int]] = []
    for q in target_qubit_operands:
        indices = _expand_quantum_operands_to_phys(emit_pass, q, qubit_map, bindings)
        target_index_groups.append(indices)
        target_indices.extend(indices)

    block_value = op.block
    local_bindings = emit_pass._resolver.bind_block_params(
        block_value, param_operands, bindings
    )
    _bind_quantum_input_shapes(
        emit_pass, block_value, target_qubit_operands, bindings, local_bindings
    )
    block_value = _prepare_nested_block_for_emit(block_value, local_bindings)

    num_targets = len(target_indices)
    unitary_gate = emit_pass._blockvalue_to_gate(
        block_value, num_targets, local_bindings
    )

    power_value = resolve_power(emit_pass, op, bindings)

    if unitary_gate is not None:
        if power_value > 1:
            unitary_gate = emit_pass._emitter.gate_power(unitary_gate, power_value)
        controlled_gate = emit_pass._emitter.gate_controlled(unitary_gate, nc)
        emit_pass._emitter.append_gate(
            circuit, controlled_gate, control_phys + target_indices
        )
    else:
        emit_pass._emit_controlled_fallback(
            circuit,
            block_value,
            nc,
            control_phys,
            target_indices,
            power_value,
            local_bindings,
        )

    # Result-side bookkeeping: every control operand keeps its
    # physical qubits onto the corresponding result operand
    # (controlled gates only add a relative phase, they do not move
    # qubits).  Targets get their per-operand index groupings copied
    # to the result so downstream ``view_out[i]`` lookups resolve.
    from qamomile.circuit.ir.value import ArrayValue as _ArrayValue

    control_results = []
    control_result_groups = []
    for result, group in zip(op.results[: op.num_control_args], control_index_groups):
        if result.type.is_quantum():
            control_results.append(result)
            control_result_groups.append(group)
    _map_operand_result_groups(control_results, control_result_groups, qubit_map)

    sub_quantum_results = [
        r for r in op.results[op.num_control_args :] if r.type.is_quantum()
    ]
    for i, result in enumerate(sub_quantum_results):
        if i >= len(target_index_groups):
            break
        indices = target_index_groups[i]
        if isinstance(result, _ArrayValue):
            map_array_result_group(result.uuid, indices, qubit_map)
        else:
            if indices:
                qubit_map[QubitAddress(result.uuid)] = indices[0]


def emit_controlled_u(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: ControlledUOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a ControlledUOperation."""
    if isinstance(op, SymbolicControlledU):
        if op.control_indices is not None:
            emit_controlled_u_with_symbolic_indices(
                emit_pass, circuit, op, qubit_map, bindings
            )
            return
        # Every other symbolic shape — both the legacy single-pool
        # form (``num_control_args == 1``) and the new multi-arg
        # form (``num_control_args > 1``) — routes through the
        # multi-arg emit helper, which expands each control operand
        # to its physical qubits via ``_expand_quantum_operands_to_phys``
        # and matches the total against ``num_controls`` resolved
        # from ``bindings``.  This catches the loop-unrolling case
        # too (``num_controls = n - 1 - k`` inside ``qmc.range``):
        # ``ConstantFoldingPass`` cannot promote that op because the
        # loop variable is not bound yet, but each unrolled iteration
        # arrives at ``emit_controlled_u`` with a fully-resolvable
        # ``num_controls`` and the multi-arg helper handles it.
        emit_controlled_u_multi_arg(emit_pass, circuit, op, qubit_map, bindings)
        return
    assert isinstance(op, ConcreteControlledU)
    nc: int = op.num_controls
    block_value = op.block
    control_operands = op.control_operands
    remaining_operands = op.operands[nc:]

    target_qubit_operands = [v for v in remaining_operands if v.type.is_quantum()]
    param_operands = [
        v for v in remaining_operands if v.type.is_classical() or v.type.is_object()
    ]

    # Resolve controls via ``resolve_qubit_index``: frontend Step 2.a
    # normalises ``operands[:num_controls]`` to one scalar per physical
    # control qubit, so each control operand maps to a single physical
    # index.  ``Vector[Qubit]`` / ``VectorView`` controls are already
    # expanded into per-element scalars upstream.
    #
    # Historical note: total-failure previously took a silent-return
    # path because the ``SymbolicControlledU`` → ``ConcreteControlledU``
    # promotion in ``ConstantFoldingPass`` can produce an inconsistent
    # operand layout (the control Vector is not expanded to individual
    # qubits, so ``operands[:num_controls]`` picks up a target Value
    # rather than each control individually).  That promotion bug is
    # tracked separately; if it triggers post-this-change, the
    # ``EmitError`` here surfaces it instead of silently dropping the
    # gate.
    control_indices: list[int] = []
    for q in control_operands:
        idx = emit_pass._resolver.resolve_qubit_index(q, qubit_map, bindings)
        if idx is not None:
            control_indices.append(idx)

    if len(control_indices) < len(control_operands):
        raise EmitError(
            f"ControlledUOperation: only "
            f"{len(control_indices)}/{len(control_operands)} control "
            f"operand(s) could be resolved to physical qubits. "
            f"Emitting a partial- or zero-arity controlled gate would "
            f"silently miswire the circuit (or drop it entirely).  "
            f"This typically indicates broken parent_array / slice "
            f"metadata on the control operands, or a stale "
            f"``SymbolicControlledU`` → ``ConcreteControlledU`` "
            f"promotion in ``ConstantFoldingPass``.",
            operation="ControlledUOperation",
        )

    # Resolve sub-kernel quantum (target) operands.  Each operand may
    # be either a scalar ``Value`` (one physical qubit) or an
    # ``ArrayValue`` of quantum element type (a ``Vector[Qubit]`` arg
    # that contributes ``length`` physical qubits).  The shared §12.1
    # helper handles both cases uniformly; ``target_index_groups``
    # records the per-operand grouping so result-side bookkeeping can
    # re-attach the physical indices to each result UUID below.
    target_indices: list[int] = []
    target_index_groups: list[list[int]] = []
    for q in target_qubit_operands:
        indices = _expand_quantum_operands_to_phys(emit_pass, q, qubit_map, bindings)
        target_index_groups.append(indices)
        target_indices.extend(indices)

    local_bindings = emit_pass._resolver.bind_block_params(
        block_value, param_operands, bindings
    )
    _bind_quantum_input_shapes(
        emit_pass, block_value, target_qubit_operands, bindings, local_bindings
    )
    block_value = _prepare_nested_block_for_emit(block_value, local_bindings)

    power_value = resolve_power(emit_pass, op, bindings)
    if _should_emit_single_target_block_per_vector_element(
        block_value, target_qubit_operands, target_indices
    ):
        _emit_single_target_block_per_vector_element(
            emit_pass,
            circuit,
            block_value,
            nc,
            control_indices,
            target_indices,
            power_value,
            local_bindings,
        )
        _map_controlled_u_results(
            op,
            nc,
            control_indices,
            target_qubit_operands,
            target_index_groups,
            qubit_map,
        )
        return

    num_targets = len(target_indices)
    unitary_gate = emit_pass._blockvalue_to_gate(
        block_value, num_targets, local_bindings
    )

    if unitary_gate is not None:
        if power_value > 1:
            unitary_gate = emit_pass._emitter.gate_power(unitary_gate, power_value)
        controlled_gate = emit_pass._emitter.gate_controlled(unitary_gate, nc)
        emit_pass._emitter.append_gate(
            circuit, controlled_gate, control_indices + target_indices
        )
    else:
        emit_pass._emit_controlled_fallback(
            circuit,
            block_value,
            nc,
            control_indices,
            target_indices,
            power_value,
            local_bindings,
        )

    _map_controlled_u_results(
        op, nc, control_indices, target_qubit_operands, target_index_groups, qubit_map
    )


def _should_emit_single_target_block_per_vector_element(
    block_value: Any,
    target_qubit_operands: list[Any],
    target_indices: list[int],
) -> bool:
    """Check for scalar-target controlled-U applied to a vector target.

    Built-in gates such as ``qmc.x`` are wrapped as scalar-qbit
    qkernels for ``qmc.control``.  When the caller supplies a
    ``Vector[Qubit]`` or ``VectorView[Qubit]`` target, the natural
    broadcast meaning is to apply the same controlled scalar block to
    every physical target element.  This helper detects exactly that
    shape so the emitter does not attempt to turn a one-qubit block
    into a multi-target custom gate.

    Args:
        block_value (Any): Inner controlled block.
        target_qubit_operands (list[Any]): Quantum target operands
            supplied at the controlled-U call site.
        target_indices (list[int]): Flattened physical target qubits.

    Returns:
        bool: ``True`` when a single scalar formal target is being
            broadcast over a vector actual target.
    """
    from qamomile.circuit.ir.value import ArrayValue

    if len(target_qubit_operands) != 1 or len(target_indices) <= 1:
        return False
    if not isinstance(target_qubit_operands[0], ArrayValue):
        return False
    quantum_inputs = [
        input_value
        for input_value in getattr(block_value, "input_values", [])
        if hasattr(input_value, "type") and input_value.type.is_quantum()
    ]
    return len(quantum_inputs) == 1 and not isinstance(quantum_inputs[0], ArrayValue)


def _emit_single_target_block_per_vector_element(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    block_value: Any,
    num_controls: int,
    control_indices: list[int],
    target_indices: list[int],
    power: int,
    bindings: dict[str, Any],
) -> None:
    """Emit a scalar controlled-U once for each vector target element.

    Args:
        emit_pass (StandardEmitPass): Driving emit pass.
        circuit (Any): Backend circuit being emitted.
        block_value (Any): Single-target inner block.
        num_controls (int): Number of control qubits.
        control_indices (list[int]): Physical control qubits.
        target_indices (list[int]): Physical target qubits to receive
            the broadcasted controlled operation.
        power (int): Positive controlled-U power.
        bindings (dict[str, Any]): Local bindings for the inner block.

    Raises:
        EmitError: If the backend cannot convert the block to a gate and
            the fallback controlled decomposition does not support the
            block shape.
    """
    unitary_gate = emit_pass._blockvalue_to_gate(block_value, 1, bindings)
    if unitary_gate is not None:
        if power > 1:
            unitary_gate = emit_pass._emitter.gate_power(unitary_gate, power)
        controlled_gate = emit_pass._emitter.gate_controlled(unitary_gate, num_controls)
        for target_idx in target_indices:
            emit_pass._emitter.append_gate(
                circuit, controlled_gate, control_indices + [target_idx]
            )
        return

    for target_idx in target_indices:
        emit_pass._emit_controlled_fallback(
            circuit,
            block_value,
            num_controls,
            control_indices,
            [target_idx],
            power,
            bindings,
        )


def emit_controlled_fallback(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    block_value: Any,
    num_controls: int,
    control_indices: list[int],
    target_indices: list[int],
    power: int,
    bindings: dict[str, Any],
) -> None:
    """Fallback emission for controlled-U when gate conversion fails.

    Decomposes the block body gate-by-gate under the full control set.
    A block-local qubit map ties the block's formal quantum inputs to
    ``target_indices``, so every inner gate is emitted on the physical
    qubit its operand resolves to — multi-target inner blocks are
    supported. Nested ``ControlledUOperation``s compose their controls
    with the outer ones; irreducible multi-controlled single-qubit
    gates route through the backend's
    ``_emit_irreducible_multi_controlled_gate`` hook. Subclasses may
    still override this method to emit controlled blocks natively
    (e.g. CUDA-Q's ``cudaq.control`` helper kernels).

    Args:
        emit_pass: The StandardEmitPass instance.
        circuit: The backend circuit being built.
        block_value: The block value containing operations to control.
        num_controls: Number of control qubits.
        control_indices: Physical indices of control qubits.
        target_indices: Physical indices of target qubits.
        power: Number of times to repeat the controlled operation.
        bindings: Parameter bindings.

    Raises:
        EmitError: If ``num_controls`` disagrees with
            ``control_indices``, the block's quantum inputs cannot be
            mapped onto ``target_indices``, or an inner operation
            cannot be lowered under the accumulated controls (e.g. an
            irreducible multi-controlled gate on a backend without the
            multi-control hook).
    """
    if not target_indices:
        return
    if not hasattr(block_value, "operations"):
        raise EmitError(
            "Cannot emit controlled fallback: block has no operations.",
            operation="ControlledUOperation",
        )
    if num_controls != len(control_indices):
        raise EmitError(
            f"Controlled fallback received inconsistent control metadata: "
            f"num_controls={num_controls}, "
            f"control_indices={control_indices!r}.",
            operation="ControlledUOperation",
        )

    block_value = _prepare_nested_block_for_emit(block_value, bindings)
    qubit_map = build_controlled_block_qubit_map(
        emit_pass, block_value, target_indices, bindings
    )
    for _ in range(power):
        emit_controlled_operations(
            emit_pass,
            circuit,
            block_value.operations,
            control_indices,
            qubit_map,
            bindings,
        )


def emit_custom_composite(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: Any,
    impl: Any,
    qubit_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit a custom composite gate with implementation.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (Any): Composite gate operation.
        impl (Any): Fallback implementation block to emit.
        qubit_indices (list[int]): Physical qubits for the operation.
        bindings (dict[str, Any]): Active emit bindings.
    """
    num_qubits = len(qubit_indices)
    impl = _prepare_nested_block_for_emit(impl, bindings)
    custom_gate = emit_pass._blockvalue_to_gate(
        impl,
        num_qubits,
        bindings,
        input_operands=op.operands,
        operation_name="CompositeGateOperation",
    )

    if custom_gate is not None and _gate_matches_qubit_count(custom_gate, num_qubits):
        emit_pass._emitter.append_gate(circuit, custom_gate, qubit_indices)
    else:
        local_qubit_map: QubitMap = {}
        local_clbit_map: ClbitMap = {}
        local_bindings = _bind_and_populate_block_inputs(
            emit_pass,
            impl,
            op.operands,
            num_qubits,
            bindings,
            local_qubit_map,
            parent_qubits=qubit_indices,
            operation_name="CompositeGateOperation",
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


def emit_controlled_composite_at_indices(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: CompositeGateOperation,
    control_indices: list[int],
    qubit_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit a composite gate under already-resolved outer controls.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (CompositeGateOperation): Composite operation to emit.
        control_indices (list[int]): Physical outer control qubits.
        qubit_indices (list[int]): Physical qubits occupied by ``op``'s
            own control and target operands.
        bindings (dict[str, Any]): Active emit bindings.

    Returns:
        None.

    Raises:
        EmitError: If the composite has no implementation block or the
            fallback cannot represent the controlled composite.
    """
    impl = op.implementation
    if impl is None:
        raise EmitError(
            "Cannot emit controlled composite without an implementation block.",
            operation="CompositeGateOperation",
        )

    num_qubits = len(qubit_indices)
    custom_gate = emit_pass._blockvalue_to_gate(
        impl,
        num_qubits,
        bindings,
        input_operands=op.operands,
        operation_name="CompositeGateOperation",
    )
    if custom_gate is not None:
        controlled_gate = custom_gate
        if control_indices:
            controlled_gate = emit_pass._emitter.gate_controlled(
                custom_gate,
                len(control_indices),
            )
        if controlled_gate is not None and _gate_matches_qubit_count(
            controlled_gate,
            len(control_indices) + num_qubits,
        ):
            emit_pass._emitter.append_gate(
                circuit,
                controlled_gate,
                [*control_indices, *qubit_indices],
            )
            return

    local_qubit_map: QubitMap = {}
    local_bindings = _bind_and_populate_block_inputs(
        emit_pass,
        impl,
        op.operands,
        num_qubits,
        bindings,
        local_qubit_map,
        operation_name="CompositeGateOperation",
    )
    emit_pass._emit_controlled_fallback(
        circuit,
        impl,
        len(control_indices),
        control_indices,
        qubit_indices,
        1,
        local_bindings,
    )


def _emitter_supports_reusable_gates(emitter: Any) -> bool:
    """Return whether an emitter can build reusable gates.

    Args:
        emitter (Any): Backend gate emitter.

    Returns:
        bool: True when ``emitter`` advertises reusable-gate support.
    """
    supports = getattr(emitter, "supports_reusable_gates", None)
    return bool(supports()) if callable(supports) else False


def blockvalue_to_gate(
    emit_pass: "StandardEmitPass",
    block_value: Any,
    num_qubits: int,
    bindings: dict[str, Any],
    input_operands: list[Any] | None = None,
    operation_name: str = "ControlledUOperation",
) -> Any:
    """Convert a Block to a backend gate.

    Pre-populates ``local_qubit_map`` with one entry per quantum input in
    declaration order, then runs the allocator over the block's body.
    For ``Vector[Qubit]`` inputs (``ArrayValue`` with a quantum element
    type) the entry expands into per-element ``QubitAddress(uuid, i)``
    keys — the inner block has no ``QInitOperation`` for inputs, so
    without these the allocator's element-resolution assertion fires for
    every ``qs[i]`` reference in the body.

    Args:
        emit_pass (StandardEmitPass): The emit pass driving the
            conversion. Used for its ``_allocator``, ``_emitter``,
            ``_resolver``, and ``_emit_operations``.
        block_value (Any): The inner block to convert. Expected to expose
            ``operations`` and ``input_values``; anything else returns
            ``None`` (the caller falls back to gate-by-gate
            decomposition).
        num_qubits (int): Total physical qubits the resulting gate will
            occupy in the parent circuit. Used as the fallback length
            for a single ``Vector[Qubit]`` input whose shape is symbolic
            and unresolvable from ``bindings``, and as the sub-circuit
            width when the block has no qubit-producing operations.
        bindings (dict[str, Any]): Parameter bindings forwarded to the
            allocator and ``_emit_operations``. Also consulted when
            resolving ``Vector[Qubit]`` input shapes.
        input_operands (list[Any] | None): Optional call-site operands
            corresponding to `block_value.input_values`. Quantum operands
            propagate actual ``Vector[Qubit]`` shapes into the nested
            block before the vector-aware input qubit map is populated;
            classical operands are resolved into local bindings before the
            nested block is emitted. Defaults to None.
        operation_name (str): Operation name used in diagnostics when
            input binding fails. Defaults to ``"ControlledUOperation"``.

    Returns:
        Any: A backend gate object produced by
            ``emit_pass._emitter.circuit_to_gate``, or ``None`` when the
            conversion is unable to proceed (missing ``operations``,
            allocator / emitter exception, etc.) so the caller can fall
            back to gate-by-gate emission.
    """
    if not hasattr(block_value, "operations"):
        return None

    block_value = _prepare_nested_block_for_emit(block_value, bindings)

    try:
        local_qubit_map: QubitMap = {}
        local_clbit_map: ClbitMap = {}
        local_bindings = _bind_and_populate_block_inputs(
            emit_pass,
            block_value,
            input_operands,
            num_qubits,
            bindings,
            local_qubit_map,
            operation_name=operation_name,
        )

        local_qubit_map, local_clbit_map = emit_pass._allocator.allocate(
            block_value.operations,
            local_bindings,
            initial_qubit_map=local_qubit_map,
            initial_clbit_map=local_clbit_map,
        )

        qubit_count = (
            max(local_qubit_map.values()) + 1 if local_qubit_map else num_qubits
        )
        sub_circuit = emit_pass._emitter.create_circuit(qubit_count, 0)

        emit_pass._emit_operations(
            sub_circuit,
            block_value.operations,
            local_qubit_map,
            local_clbit_map,
            local_bindings,
            force_unroll=True,
        )

        return emit_pass._emitter.circuit_to_gate(sub_circuit, "U")

    except (AttributeError, TypeError, ValueError, KeyError, IndexError, RuntimeError):
        import logging

        logging.getLogger(__name__).debug(
            "blockvalue_to_gate: falling back to gate-by-gate decomposition",
            exc_info=True,
        )
        return None


def _bind_block_inputs(
    emit_pass: "StandardEmitPass",
    block_value: Any,
    input_operands: list[Any] | None,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    """Bind nested block classical inputs to call-site operands.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        block_value (Any): Nested block whose inputs are being emitted.
        input_operands (list[Any] | None): Call-site operands. Quantum
            operands are skipped here; callers should handle them through
            ``_bind_quantum_input_shapes`` and ``_populate_input_qubit_map``.
            Classical operands are resolved into local bindings. When None,
            no direct binding is performed.
        bindings (dict[str, Any]): Parent emit bindings.

    Returns:
        dict[str, Any]: Local bindings for nested emission.
    """
    local_bindings = dict(bindings)
    if input_operands is None or not hasattr(block_value, "input_values"):
        return local_bindings

    quantum_inputs = [
        formal
        for formal in block_value.input_values
        if hasattr(formal, "type") and formal.type.is_quantum()
    ]
    classical_inputs = [
        formal
        for formal in block_value.input_values
        if not (hasattr(formal, "type") and formal.type.is_quantum())
    ]
    classical_operands = list(input_operands[len(quantum_inputs) :])

    for formal, actual in zip(classical_inputs, classical_operands):
        if hasattr(formal, "uuid"):
            local_bindings[formal.uuid] = _resolve_call_operand(
                emit_pass,
                actual,
                bindings,
            )

    return local_bindings


def _bind_and_populate_block_inputs(
    emit_pass: "StandardEmitPass",
    block_value: Any,
    input_operands: list[Any] | None,
    num_qubits: int,
    bindings: dict[str, Any],
    qubit_map: QubitMap,
    parent_qubits: list[int] | None = None,
    operation_name: str = "ControlledUOperation",
) -> dict[str, Any]:
    """Bind nested block inputs and populate its quantum input map.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        block_value (Any): Nested block whose inputs are being emitted.
        input_operands (list[Any] | None): Call-site operands. Quantum
            operands are used to propagate ``Vector[Qubit]`` shapes;
            classical operands are resolved into local bindings. Defaults
            to None.
        num_qubits (int): Local qubit width available to ``block_value``.
        bindings (dict[str, Any]): Parent emit bindings.
        qubit_map (QubitMap): Local qubit map to mutate with scalar and
            per-element quantum input addresses.
        parent_qubits (list[int] | None): Optional parent-circuit physical
            qubits used to remap the local ``0..num_qubits-1`` indices after
            population. Defaults to None.
        operation_name (str): Operation label used in emitted errors.
            Defaults to ``"ControlledUOperation"``.

    Returns:
        dict[str, Any]: Local bindings for nested emission.

    Raises:
        EmitError: If vector inputs cannot fit in ``num_qubits`` or a local
            populated qubit index cannot be remapped to ``parent_qubits``.
    """
    local_bindings = _bind_block_inputs(
        emit_pass,
        block_value,
        input_operands,
        bindings,
    )
    quantum_operands = _quantum_input_operands(block_value, input_operands)
    _bind_quantum_input_shapes(
        emit_pass,
        block_value,
        quantum_operands,
        bindings,
        local_bindings,
    )
    if hasattr(block_value, "input_values"):
        _populate_input_qubit_map(
            emit_pass,
            block_value.input_values,
            num_qubits,
            local_bindings,
            qubit_map,
        )
    if parent_qubits is not None:
        _remap_local_qubit_map(
            qubit_map,
            parent_qubits,
            operation_name,
        )
    return local_bindings


def _quantum_input_operands(
    block_value: Any,
    input_operands: list[Any] | None,
) -> list[Any]:
    """Return call-site operands that correspond to quantum block inputs.

    Args:
        block_value (Any): Block whose ``input_values`` define the formal
            quantum/classical input split.
        input_operands (list[Any] | None): Call-site operands. Defaults to
            None.

    Returns:
        list[Any]: Quantum operands paired with formal quantum inputs in
            declaration order.
    """
    if input_operands is None or not hasattr(block_value, "input_values"):
        return []
    quantum_input_count = sum(
        1
        for input_value in block_value.input_values
        if hasattr(input_value, "type") and input_value.type.is_quantum()
    )
    return list(input_operands[:quantum_input_count])


def _remap_local_qubit_map(
    qubit_map: QubitMap,
    parent_qubits: list[int],
    operation_name: str,
) -> None:
    """Remap local block input qubit slots to parent physical qubits.

    Args:
        qubit_map (QubitMap): Map populated with local ``0..n-1`` slots.
            Mutated in place to parent-circuit physical qubit indices.
        parent_qubits (list[int]): Parent physical qubits indexed by local
            slot.
        operation_name (str): Operation label used in emitted errors.

    Raises:
        EmitError: If a local slot falls outside ``parent_qubits``.
    """
    for address, local_index in list(qubit_map.items()):
        if local_index < 0 or local_index >= len(parent_qubits):
            raise EmitError(
                f"{operation_name}: local input qubit index {local_index} "
                f"cannot be remapped through {len(parent_qubits)} parent "
                f"qubit(s).",
                operation=operation_name,
            )
        qubit_map[address] = parent_qubits[local_index]


def _gate_matches_qubit_count(gate: Any, num_qubits: int) -> bool:
    """Return whether a backend gate can be appended at the call site.

    Args:
        gate (Any): Backend gate candidate.
        num_qubits (int): Number of qubits supplied by the call site.

    Returns:
        bool: True when the backend exposes a qubit-count field and the
            field matches `num_qubits`.
    """
    gate_num_qubits = getattr(gate, "num_qubits", None)
    return gate_num_qubits == num_qubits


def _resolve_call_operand(
    emit_pass: "StandardEmitPass",
    actual: Any,
    bindings: dict[str, Any],
) -> Any:
    """Resolve a call-site classical operand for nested block emission.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        actual (Any): Call-site operand.
        bindings (dict[str, Any]): Parent emit bindings.

    Returns:
        Any: Concrete value, backend parameter, or backend expression.
    """
    if not hasattr(actual, "uuid"):
        return actual
    resolved = emit_pass._resolver.resolve_classical_value(
        actual,
        bindings,
    )
    if resolved is not None:
        return resolved
    param_key = emit_pass._resolver.get_parameter_key(actual, bindings)
    if param_key is not None:
        return emit_pass._get_or_create_parameter(param_key, actual.uuid)
    return actual


def _contains_slice_markers(
    operations: list[Operation],
    _seen: set[int] | None = None,
) -> bool:
    """Return whether ``operations`` contains slice borrow markers.

    Args:
        operations (list[Operation]): Operations to inspect recursively,
            including control-flow children and nested block-valued
            operations.

    Returns:
        bool: ``True`` when a ``SliceArrayOperation`` or
        ``ReleaseSliceViewOperation`` is present; otherwise ``False``.
    """
    seen = _seen if _seen is not None else set()
    for op in operations:
        if isinstance(op, (SliceArrayOperation, ReleaseSliceViewOperation)):
            return True
        if isinstance(op, HasNestedOps):
            if any(
                _contains_slice_markers(nested, seen) for nested in op.nested_op_lists()
            ):
                return True
        nested_block = getattr(op, "block", None)
        if isinstance(nested_block, Block):
            block_id = id(nested_block)
            if block_id not in seen:
                seen.add(block_id)
                if _contains_slice_markers(nested_block.operations, seen):
                    return True
        implementation_block = getattr(op, "implementation_block", None)
        if isinstance(implementation_block, Block):
            block_id = id(implementation_block)
            if block_id not in seen:
                seen.add(block_id)
                if _contains_slice_markers(implementation_block.operations, seen):
                    return True
    return False


def _prepare_nested_block_for_emit(
    block_value: Any,
    bindings: dict[str, Any],
    _seen: set[int] | None = None,
) -> Any:
    """Run nested-block slice checks and remove emit-only markers.

    Top-level blocks pass through ``SliceBorrowCheckPass`` and
    ``StripSliceArrayOpsPass`` before segmentation and emission, but a
    ``ControlledUOperation`` or custom composite carries its inner
    ``Block`` as an operation field rather than as regular control-flow
    children. Generic pass visitors therefore do not descend into that
    nested block. Apply the same invariant here before any nested block
    with slice markers is converted to a backend gate or emitted through
    a fallback path.

    Args:
        block_value (Any): Candidate nested block to normalize.
        bindings (dict[str, Any]): Compile-time bindings visible inside
            the nested block.

    Returns:
        Any: ``block_value`` after ``SliceBorrowCheckPass`` and marker
        stripping when it is a ``Block`` containing slice markers; otherwise
        the original object unchanged.

    Raises:
        EmitError: If the marker-bearing block is already past the stages
            that can be safely checked.
        ValidationError: If ``SliceBorrowCheckPass`` rejects the block
            stage.
        SliceBorrowViolationError: If nested slice borrows violate the
            same linearity rules enforced for top-level blocks.
    """
    if not isinstance(block_value, Block):
        return block_value

    seen = _seen if _seen is not None else set()
    block_id = id(block_value)
    if block_id in seen:
        return block_value
    seen.add(block_id)

    try:
        block_value = _prepare_nested_operation_blocks(block_value, bindings, seen)
        if not _contains_slice_markers(block_value.operations, seen):
            return block_value

        from qamomile.circuit.transpiler.passes.constant_fold import (
            ConstantFoldingPass,
        )
        from qamomile.circuit.transpiler.passes.slice_borrow_check import (
            SliceBorrowCheckPass,
        )
        from qamomile.circuit.transpiler.passes.strip_slice_ops import (
            StripSliceArrayOpsPass,
        )

        if block_value.kind == BlockKind.TRACED:
            block_value = dataclasses.replace(block_value, kind=BlockKind.HIERARCHICAL)
        elif block_value.kind not in (BlockKind.HIERARCHICAL, BlockKind.AFFINE):
            raise EmitError(
                f"Cannot normalize nested slice markers in {block_value.kind} block "
                f"{block_value.name!r}; slice markers must be checked before "
                f"analysis and emit.",
                operation="SliceArrayOperation",
            )
        folded = ConstantFoldingPass(bindings, strip_slice_ops=False).run(block_value)
        checked = SliceBorrowCheckPass().run(folded)
        return StripSliceArrayOpsPass().run(checked)
    finally:
        seen.remove(block_id)


def _prepare_nested_operation_blocks(
    block_value: Block,
    bindings: dict[str, Any],
    seen: set[int],
) -> Block:
    """Normalize block-valued operation fields inside ``block_value``.

    Args:
        block_value (Block): Parent block whose operations may carry
            nested ``block`` or ``implementation_block`` attributes.
        bindings (dict[str, Any]): Bindings visible while normalizing
            nested block-valued attributes.
        seen (set[int]): Identity set used to avoid following cyclic
            block references repeatedly.

    Returns:
        Block: ``block_value`` with any normalized nested block fields
        reattached to their owning operations.
    """
    new_ops, changed = _prepare_nested_operation_list_blocks(
        block_value.operations,
        bindings,
        seen,
    )
    if not changed:
        return block_value
    return dataclasses.replace(block_value, operations=new_ops)


def _prepare_nested_operation_list_blocks(
    operations: list[Operation],
    bindings: dict[str, Any],
    seen: set[int],
) -> tuple[list[Operation], bool]:
    """Normalize block-valued fields in an operation list.

    Args:
        operations (list[Operation]): Operations to inspect.
        bindings (dict[str, Any]): Bindings visible to nested blocks.
        seen (set[int]): Identity set used as a recursion guard.

    Returns:
        tuple[list[Operation], bool]: Rewritten operations and whether
        any operation changed.
    """
    new_ops: list[Operation] = []
    changed = False
    for op in operations:
        new_op = _prepare_nested_operation_block_fields(op, bindings, seen)
        if new_op is not op:
            changed = True
        if isinstance(new_op, HasNestedOps):
            nested_lists: list[list[Operation]] = []
            nested_changed = False
            for nested in new_op.nested_op_lists():
                new_nested, did_change = _prepare_nested_operation_list_blocks(
                    nested,
                    bindings,
                    seen,
                )
                nested_lists.append(new_nested)
                nested_changed = nested_changed or did_change
            if nested_changed:
                new_op = new_op.rebuild_nested(nested_lists)
                changed = True
        new_ops.append(new_op)
    return new_ops, changed


def _prepare_nested_operation_block_fields(
    op: Operation,
    bindings: dict[str, Any],
    seen: set[int],
) -> Operation:
    """Normalize ``block`` and ``implementation_block`` fields on ``op``.

    Args:
        op (Operation): Operation to inspect.
        bindings (dict[str, Any]): Bindings visible to nested blocks.
        seen (set[int]): Identity set used as a recursion guard.

    Returns:
        Operation: ``op`` with normalized nested block fields when
        needed; otherwise ``op`` unchanged.
    """
    updates: dict[str, Block] = {}
    for attr in ("block", "implementation_block"):
        nested = getattr(op, attr, None)
        if isinstance(nested, Block):
            normalized = _prepare_nested_block_for_emit(nested, bindings, seen)
            if normalized is not nested:
                updates[attr] = normalized
    if not updates:
        return op
    return dataclasses.replace(cast(Any, op), **updates)


def _populate_input_qubit_map(
    emit_pass: "StandardEmitPass",
    input_values: list[Any],
    num_qubits: int,
    bindings: dict[str, Any],
    qubit_map: QubitMap,
) -> None:
    """Pre-populate ``qubit_map`` with the inner block's quantum inputs.

    Scalar ``Qubit`` inputs consume one physical index each;
    ``Vector[Qubit]`` inputs (``ArrayValue`` with a quantum element type)
    consume ``length`` consecutive indices, one per element, keyed as
    ``QubitAddress(uuid, i)``. The base ``QubitAddress(uuid)`` is also
    registered for the vector so any downstream lookup that addresses
    the array as a whole resolves to its first element.

    Args:
        emit_pass (StandardEmitPass): Provides the value resolver used
            to resolve symbolic ``Vector[Qubit]`` shapes against
            ``bindings``.
        input_values (list[Any]): The inner block's ``input_values``.
            Non-quantum inputs (``Float``, ``UInt``, ...) are skipped.
        num_qubits (int): Total qubit count the resulting gate will
            occupy. Used as the fallback length for a single
            ``Vector[Qubit]`` input whose shape Value is not present in
            ``bindings``.
        bindings (dict[str, Any]): Parameter bindings consulted when
            resolving a vector's symbolic shape.
        qubit_map (QubitMap): Mapping mutated in place. Each registered
            entry maps a ``QubitAddress`` (scalar or array element) to a
            consecutive physical index starting at 0.

    Raises:
        EmitError: Four failure modes are surfaced loudly rather than
            silently mis-mapping qubits:

            * A resolved ``Vector[Qubit]`` length (from a constant shape
              or a binding) is negative. ``range(length)`` would be
              empty and ``qubit_idx += length`` would step the cursor
              backwards, producing overlapping physical assignments.
            * Multiple unresolvable ``Vector[Qubit]`` inputs (only one
              symbolic length can be inferred from the remaining
              ``num_qubits`` budget).
            * A single unresolvable ``Vector[Qubit]`` whose inferred
              length ``num_qubits - scalar_count - sum(resolved)`` is
              negative.
            * Total quantum-input footprint
              ``scalar_count + sum(resolved_vector_lengths)`` exceeds
              ``num_qubits`` (i.e., a resolved or bound vector shape
              overflows the gate's declared qubit width).
    """
    from qamomile.circuit.ir.value import ArrayValue

    quantum_inputs = [
        iv for iv in input_values if hasattr(iv, "type") and iv.type.is_quantum()
    ]

    scalar_count = sum(1 for iv in quantum_inputs if not isinstance(iv, ArrayValue))
    vector_inputs = [iv for iv in quantum_inputs if isinstance(iv, ArrayValue)]

    resolved_lengths: dict[str, int] = {}
    unresolved: list[ArrayValue] = []
    for iv in vector_inputs:
        length = _resolve_vector_input_length(emit_pass, iv, bindings)
        if length is None:
            unresolved.append(iv)
        elif length < 0:
            raise EmitError(
                f"Vector[Qubit] input {iv.name!r} resolved to a negative "
                f"length ({length}); shapes must be non-negative. Check "
                f"the binding for {iv.shape[0].name!r}.",
                operation="ControlledUOperation",
            )
        else:
            resolved_lengths[iv.uuid] = length

    # Fall back to ``num_qubits`` for the single unresolved Vector[Qubit]
    # input. The caller's ``num_qubits`` is the total qubit count of the
    # emitted gate, so subtracting scalars and any resolved vector
    # lengths leaves exactly the remaining vector's length.
    if len(unresolved) == 1:
        inferred = num_qubits - scalar_count - sum(resolved_lengths.values())
        if inferred < 0:
            raise EmitError(
                f"Vector[Qubit] input {unresolved[0].name!r} has an unresolved "
                f"length and the remaining qubit budget ({inferred}) is "
                f"negative; bind the vector's shape before transpilation.",
                operation="ControlledUOperation",
            )
        resolved_lengths[unresolved[0].uuid] = inferred
    elif len(unresolved) > 1:
        raise EmitError(
            f"Cannot resolve Vector[Qubit] input shapes for inner block "
            f"({[iv.name for iv in unresolved]!r}); only one symbolic "
            f"length can be inferred from the gate's qubit count. Bind "
            f"the remaining shapes before transpilation.",
            operation="ControlledUOperation",
        )

    # Sanity-check the total budget. The inferred-length branch above
    # cannot overflow (its length is computed to fit exactly), but a
    # vector whose length was resolved from a binding or a constant
    # shape can legitimately be larger than the gate's qubit width if
    # the caller misconfigured things. Reject upfront rather than
    # silently writing physical indices past ``num_qubits - 1``.
    total = scalar_count + sum(resolved_lengths.values())
    if total > num_qubits:
        raise EmitError(
            f"Inner block's quantum inputs require {total} physical "
            f"qubits (scalars={scalar_count}, vector lengths="
            f"{sorted(resolved_lengths.values())}) but the controlled "
            f"gate only provides {num_qubits}. Check the bound vector "
            f"shapes against the gate's expected target count.",
            operation="ControlledUOperation",
        )

    qubit_idx = 0
    for input_val in quantum_inputs:
        if isinstance(input_val, ArrayValue):
            length = resolved_lengths[input_val.uuid]
            for i in range(length):
                qubit_map[QubitAddress(input_val.uuid, i)] = qubit_idx + i
            base_addr = QubitAddress(input_val.uuid)
            if base_addr not in qubit_map and length > 0:
                qubit_map[base_addr] = qubit_idx
            qubit_idx += length
        else:
            qubit_map[QubitAddress(input_val.uuid)] = qubit_idx
            qubit_idx += 1


def _resolve_vector_input_length(
    emit_pass: "StandardEmitPass",
    input_val: Any,
    bindings: dict[str, Any],
) -> int | None:
    """Resolve a ``Vector[Qubit]`` input's length from its shape Value.

    Args:
        emit_pass (StandardEmitPass): Provides ``_resolver`` for shape
            resolution.
        input_val (Any): An ``ArrayValue`` representing a ``Vector[Qubit]``
            input. The first element of ``input_val.shape`` is the
            length Value (constant or symbolic).
        bindings (dict[str, Any]): Bindings consulted when the shape
            Value is symbolic.

    Returns:
        int | None: The resolved length, or ``None`` when ``input_val.shape``
            is empty (no dimension Value at all) or when the first
            dimension is symbolic and not present in ``bindings``. The
            caller treats both cases as "unresolved" and falls back to
            inferring the length from the surrounding qubit budget.
    """
    if not input_val.shape:
        return None
    size_val = input_val.shape[0]
    if size_val.is_constant():
        return int(size_val.get_const())
    return emit_pass._resolver.resolve_int_value(size_val, bindings)


def _expand_quantum_operands_to_phys(
    emit_pass: "StandardEmitPass",
    operand: Any,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
    *,
    operation: str = "ControlledUOperation",
) -> list[int]:
    """Expand one quantum operand into its per-element physical qubit indices.

    Scalar quantum ``Value`` s (the canonical ``Qubit`` operand kind)
    return a single-element list, identical to what
    ``_resolver.resolve_qubit_index`` produces.  ``ArrayValue`` operands
    whose element type is quantum (``Vector[Qubit]`` arguments) are
    expanded element-by-element: the shape is resolved against
    ``bindings``, the ``slice_of`` chain is walked to the root
    parent, and one physical index per covered slot is computed via
    the affine map ``root_idx = slice_start + slice_step * i``.

    Centralising the expansion lets ``emit_controlled_u`` accept
    ``Vector[Qubit]`` sub-kernel arguments (Step 2.b of the
    controlled-API redesign) without copy/pasting the
    length-resolution + slice-walk + per-element-lookup sequence into
    every controlled emit path.

    Args:
        emit_pass (StandardEmitPass): The emit pass driving the
            conversion; consulted for its ``_resolver`` (used to
            resolve array shapes and slice bounds).
        operand (Any): A scalar quantum ``Value`` or an ``ArrayValue``
            whose element type is quantum.  Other operand kinds are
            rejected.
        qubit_map (QubitMap): The current map from ``QubitAddress``
            to physical qubit index.
        bindings (dict[str, Any]): Caller bindings consulted when
            shapes or slice bounds are symbolic.
        operation (str): Operation name used as the ``EmitError``
            ``operation`` tag for any raised diagnostic.  Defaults to
            ``"ControlledUOperation"``.

    Returns:
        list[int]: One physical qubit index per covered slot, in
            declaration order for arrays.

    Raises:
        EmitError: Surfaces under any of the following conditions:

            * the operand is an ``ArrayValue`` with no shape;
            * the operand is an ``ArrayValue`` whose length cannot be
              resolved from ``bindings``;
            * a slice bound (``slice_start`` / ``slice_step``) along
              the ``slice_of`` chain cannot be resolved;
            * a per-element ``QubitAddress`` is missing from
              ``qubit_map``;
            * a scalar operand cannot be resolved to a physical
              qubit by ``resolve_qubit_index``.
    """
    from qamomile.circuit.ir.value import ArrayValue

    if isinstance(operand, ArrayValue):
        if not operand.shape:
            raise EmitError(
                f"Cannot expand ArrayValue {operand.name!r} without a shape.",
                operation=operation,
            )
        size = emit_pass._resolver.resolve_int_value(operand.shape[0], bindings)
        if size is None:
            raise EmitError(
                f"Cannot resolve Vector[Qubit] length for "
                f"{operand.name!r}; bind {operand.shape[0].name!r} "
                f"before transpilation.",
                operation=operation,
            )
        root_av, slice_start, slice_step = emit_pass._resolver.resolve_slice_chain(
            operand, bindings, operation=operation
        )
        phys: list[int] = []
        for i in range(size):
            addr = QubitAddress(root_av.uuid, slice_start + slice_step * i)
            if addr not in qubit_map:
                raise EmitError(
                    f"Expected qubit address {addr!s} for "
                    f"Vector[Qubit] {operand.name!r} element {i} "
                    f"not found in qubit_map.",
                    operation=operation,
                )
            phys.append(qubit_map[addr])
        return phys

    idx = emit_pass._resolver.resolve_qubit_index(operand, qubit_map, bindings)
    if idx is None:
        raise EmitError(
            f"Cannot resolve scalar quantum operand {operand.name!r} "
            f"(uuid {operand.uuid[:8]}...) to a physical qubit.",
            operation=operation,
        )
    return [idx]


def _bind_quantum_input_shapes(
    emit_pass: "StandardEmitPass",
    block_value: Any,
    actual_target_operands: list[Any],
    bindings: dict[str, Any],
    local_bindings: dict[str, Any],
) -> None:
    """Propagate ``Vector[Qubit]`` actual-arg lengths into the inner block's shape Values.

    ``bind_block_params`` only walks the inner block's classical input
    parameters, so the quantum side (``Vector[Qubit]`` sub-kernel args)
    contributes nothing to ``local_bindings``.  The inner block's body,
    however, may compute ``m = q.shape[0]`` against a *formal*
    ``Vector[Qubit]`` parameter and then use ``m`` as a loop bound,
    a slice length, etc.  Without seeding the formal shape Value's
    UUID (and name, for parity with the resolver's name-keyed fallback)
    to the actual operand's resolved length, ``resolve_int_value``
    cannot fold the formal ``shape[0]`` and downstream pieces like
    ``emit_controlled_operations``'s for-loop bounds resolution raise
    "Cannot resolve ForOperation bounds in controlled block".

    Args:
        emit_pass (StandardEmitPass): Driving emit pass; consulted for
            its ``_resolver`` to resolve actual operand sizes.
        block_value (Any): The inner block whose ``input_values`` we
            walk to find the quantum formal parameters.  Objects with
            no ``input_values`` attribute are silently skipped.
        actual_target_operands (list[Any]): The actual quantum
            operands (scalar ``Qubit`` ``Value`` s and / or
            ``ArrayValue`` s) supplied at the controlled-U call site,
            in declaration order matching the formal quantum inputs.
        bindings (dict[str, Any]): Caller bindings consulted when the
            actual operand's shape is itself symbolic.
        local_bindings (dict[str, Any]): The inner-block-local
            bindings dict to extend in place with the propagated
            shape entries (UUID-keyed and name-keyed).
    """
    from qamomile.circuit.ir.value import ArrayValue

    if not hasattr(block_value, "input_values"):
        return
    quantum_inputs = [
        iv
        for iv in block_value.input_values
        if hasattr(iv, "type") and iv.type.is_quantum()
    ]
    for actual, formal in zip(actual_target_operands, quantum_inputs):
        if not (isinstance(actual, ArrayValue) and isinstance(formal, ArrayValue)):
            continue
        if not (actual.shape and formal.shape):
            continue
        actual_size = emit_pass._resolver.resolve_int_value(actual.shape[0], bindings)
        if actual_size is None:
            continue
        formal_dim = formal.shape[0]
        local_bindings[formal_dim.uuid] = actual_size
        formal_dim_name = getattr(formal_dim, "name", "")
        if formal_dim_name:
            local_bindings[formal_dim_name] = actual_size


def _map_controlled_u_results(
    op: ConcreteControlledU,
    num_controls: int,
    control_indices: list[int],
    target_qubit_operands: list[Any],
    target_index_groups: list[list[int]],
    qubit_map: QubitMap,
) -> None:
    """Map a ``ConcreteControlledU``'s result ``Value`` UUIDs to physical qubits.

    Result layout (set up by the Step 2.a frontend, mirroring the
    operand layout):

    - ``op.results[:num_controls]`` — one scalar ``Value`` per physical
      control qubit; maps 1:1 to ``control_indices``.
    - ``op.results[num_controls:]`` — one entry per sub-kernel quantum
      operand, in the same order as ``target_qubit_operands``.  A
      scalar operand contributes a scalar result; a ``Vector[Qubit]``
      operand contributes an ``ArrayValue`` result, in which case
      ``QubitAddress(result.uuid, i)`` is registered for every covered
      element so downstream lookups via ``view_out[i]`` still resolve.

    Args:
        op (ConcreteControlledU): The IR operation being emitted.
        num_controls (int): Number of physical control qubits.
        control_indices (list[int]): Physical indices of the controls.
        target_qubit_operands (list[Any]): Quantum sub-kernel operands
            in declaration order (scalar ``Value`` or ``ArrayValue``).
        target_index_groups (list[list[int]]): Per-operand physical
            index groups returned by
            :func:`_expand_quantum_operands_to_phys`.
        qubit_map (QubitMap): Mutated in place with the new result
            ``QubitAddress`` entries.
    """
    from qamomile.circuit.ir.value import ArrayValue, resolve_root_qubit_address

    control_results = op.results[:num_controls]
    for i, result in enumerate(control_results):
        if i >= len(control_indices):
            break
        physical = control_indices[i]
        qubit_map[QubitAddress(result.uuid)] = physical
        # Whole-``Vector`` / ``VectorView`` controls expand into
        # per-element scalar results, but the user-facing output handle
        # wraps a single next-version ``ArrayValue`` that the frontend
        # re-parented these scalars onto (see
        # ``ControlledGate._build_control_results``).  Populate that
        # array's per-element root ``QubitAddress`` so a downstream
        # ``measure`` / element access of the returned control vector
        # resolves to the same physical qubit as the input element:
        # ``emit_measure_vector`` / ``resolve_slice_chain`` look the
        # element up by the root ``(array_uuid, start + step * i)`` key,
        # which mirrors ``_allocate_qubit_list``'s result-chain copy.
        # Standalone scalar ``Qubit`` controls have no ``parent_array``
        # so ``resolve_root_qubit_address`` returns ``None`` and they
        # fall through; array-element scalar controls resolve to their
        # already-registered input-array address (guarded no-op).  The
        # whole-array base key (no element index) is intentionally not
        # written here — no resolver path reads it for these results,
        # and writing it would add a base key to the *input* array for
        # array-element scalar controls.
        root = resolve_root_qubit_address(result)
        if root is not None:
            element_addr = QubitAddress(*root)
            if element_addr not in qubit_map:
                qubit_map[element_addr] = physical

    target_results = [r for r in op.results[num_controls:] if r.type.is_quantum()]
    for i, result in enumerate(target_results):
        if i >= len(target_index_groups):
            break
        indices = target_index_groups[i]
        if isinstance(result, ArrayValue):
            # Per-element addresses plus the base address so callers
            # that subscript the result via ``result[i]`` (which
            # produces ``Value(parent_array=result, element_indices=…)``)
            # and callers that address the whole array both resolve.
            map_array_result_group(result.uuid, indices, qubit_map)
        else:
            if indices:
                qubit_map[QubitAddress(result.uuid)] = indices[0]
