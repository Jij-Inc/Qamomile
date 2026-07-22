"""Controlled operation emission helpers extracted from StandardEmitPass.

This module provides module-level functions for emitting controlled-U
operations, controlled gates, and related helpers. Each function takes an
``emit_pass`` parameter (a ``StandardEmitPass`` instance) in place of
``self``.

Primitive multi-control decomposition lives in
``multi_control_gate_emission``. Reusable block preparation and operand
mapping live in ``controlled_block_support``; their names are re-exported
here for compatibility with existing internal extension points.

Note: ``emit_controlled_fallback`` and ``blockvalue_to_gate`` are called
via ``emit_pass._emit_controlled_fallback(...)`` and
``emit_pass._blockvalue_to_gate(...)`` respectively, so that subclass
overrides (e.g. CudaqEmitPass) are respected.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
)
from qamomile.circuit.ir.operation.callable import (
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    HasNestedOps,
    IfOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.cast_binop_emission import (
    _set_emit_value,
    evaluate_binop,
    evaluate_classical_predicate,
)
from qamomile.circuit.transpiler.passes.emit_support.condition_resolution import (
    remap_static_merge_outputs,
    resolve_if_condition,
)
from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
    register_classical_merge_aliases,
    resolve_loop_bounds,
)
from qamomile.circuit.transpiler.passes.emit_support.control_value_emission import (
    bracket_control_value,
)
from qamomile.circuit.transpiler.passes.emit_support.controlled_block_support import (
    _bind_and_populate_block_inputs as _bind_and_populate_block_inputs,
    _bind_block_inputs as _bind_block_inputs,
    _bind_quantum_input_shapes as _bind_quantum_input_shapes,
    _contains_slice_markers as _contains_slice_markers,
    _emitter_supports_reusable_gates as _emitter_supports_reusable_gates,
    _expand_quantum_operands_to_phys as _expand_quantum_operands_to_phys,
    _gate_matches_qubit_count as _gate_matches_qubit_count,
    _map_controlled_u_results as _map_controlled_u_results,
    _populate_input_qubit_map as _populate_input_qubit_map,
    _prepare_nested_block_for_emit as _prepare_nested_block_for_emit,
    _prepare_nested_operation_block_fields as _prepare_nested_operation_block_fields,
    _prepare_nested_operation_blocks as _prepare_nested_operation_blocks,
    _prepare_nested_operation_list_blocks as _prepare_nested_operation_list_blocks,
    _quantum_input_operands as _quantum_input_operands,
    _remap_local_qubit_map as _remap_local_qubit_map,
    _resolve_call_operand as _resolve_call_operand,
    _resolve_vector_input_length as _resolve_vector_input_length,
    blockvalue_to_gate as blockvalue_to_gate,
)
from qamomile.circuit.transpiler.passes.emit_support.gate_emission import (
    reject_duplicate_physical_indices,
)
from qamomile.circuit.transpiler.passes.emit_support.global_phase_emission import (
    emit_controlled_global_phase_operation,
)
from qamomile.circuit.transpiler.passes.emit_support.multi_control_gate_emission import (
    _and_ladder_steps as _and_ladder_steps,
    _emit_irreducible as _emit_irreducible,
    _emit_mc_rotation as _emit_mc_rotation,
    _emit_mc_x as _emit_mc_x,
    _emit_mc_z as _emit_mc_z,
    _emit_toffoli_steps as _emit_toffoli_steps,
    emit_controlled_gate as emit_controlled_gate,
    emit_multi_controlled_gate as emit_multi_controlled_gate,
    emit_multi_controlled_on_clean_ancillas as emit_multi_controlled_on_clean_ancillas,
    emit_single_controlled_primitive as emit_single_controlled_primitive,
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


def _checked_append_gate(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    gate: Any,
    qubit_indices: list[int],
    gate_label: str,
) -> None:
    """Append a controlled / composite gate after rejecting qubit aliasing.

    Every controlled or composite block reaches the backend through
    ``append_gate`` with a combined physical-index list (``control_phys +
    target_indices``). A controlled block is defined only on distinct qubits;
    when a symbolic control and target index coincide at runtime (e.g.
    ``qmc.control(x)(qs[i], qs[j])`` on the diagonal) the duplicate is visible
    only here at emit time. This wrapper runs the shared aliasing check before
    delegating to the backend, so the controlled path gets the same Qamomile
    ``QubitAliasError`` the native ``emit_gate`` path already raises, on every
    backend, instead of a raw ``CircuitError`` (Qiskit) or a silent
    compile-then-crash (CUDA-Q).

    Args:
        emit_pass (StandardEmitPass): Active emit pass (for its emitter).
        circuit (Any): Backend circuit being emitted into.
        gate (Any): The already-controlled/powered backend gate to append.
        qubit_indices (list[int]): Combined physical qubit indices the gate
            acts on (controls followed by targets).
        gate_label (str): Human-readable label for the aliasing diagnostic.

    Returns:
        None

    Raises:
        QubitAliasError: If two of ``qubit_indices`` are the same physical
            qubit.
    """
    reject_duplicate_physical_indices(gate_label, qubit_indices)
    emit_pass._emitter.append_gate(circuit, gate, qubit_indices)


def emit_controlled_powers(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    block_value: Any,
    counting_indices: list[int],
    target_indices: list[int],
    bindings: dict[str, Any],
    parent_qubit_map: QubitMap | None = None,
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
            _checked_append_gate(
                emit_pass,
                circuit,
                controlled_powered_gate,
                [ctrl_idx] + target_indices,
                "controlled power",
            )
    else:
        for k, ctrl_idx in enumerate(counting_indices):
            power = 2**k
            for _ in range(power):
                emit_controlled_block(
                    emit_pass,
                    circuit,
                    block_value,
                    ctrl_idx,
                    target_indices,
                    bindings,
                    parent_qubit_map=parent_qubit_map,
                )


def emit_controlled_block(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    block_value: Any,
    control_idx: int,
    target_indices: list[int],
    bindings: dict[str, Any],
    parent_qubit_map: QubitMap | None = None,
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
        parent_qubit_map (QubitMap | None): Parent-circuit allocation map.
            Internal fresh-allocation addresses are copied from it when
            present. Defaults to None.

    Raises:
        EmitError: If the block's quantum inputs cannot be mapped onto
            ``target_indices`` or an inner operation cannot be emitted.
    """
    if not hasattr(block_value, "operations"):
        return

    block_value = _prepare_nested_block_for_emit(block_value, bindings)
    qubit_map = build_controlled_block_qubit_map(
        emit_pass,
        block_value,
        target_indices,
        bindings,
        parent_qubit_map=(
            parent_qubit_map
            if parent_qubit_map is not None
            else getattr(emit_pass, "_active_qubit_map", None)
        ),
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
    parent_qubit_map: QubitMap | None = None,
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
        parent_qubit_map (QubitMap | None): Parent-circuit allocation map
            containing any nested fresh-workspace addresses. Defaults to
            None.

    Returns:
        QubitMap: Mapping from the inner block's formal quantum input
            addresses to physical parent-circuit qubit indices.

    Raises:
        EmitError: If a vector input length cannot be resolved, is
            negative, or the block's quantum input footprint exceeds
            ``len(target_indices)``.
    """
    # Nested fresh allocations were reserved on the parent circuit before
    # its width was fixed.  Keep those addresses visible while overriding
    # the block's formal inputs with this call site's actual targets.
    local_map: QubitMap = dict(parent_qubit_map or {})
    input_map: QubitMap = {}
    _populate_input_qubit_map(
        emit_pass,
        getattr(block_value, "input_values", []),
        len(target_indices),
        bindings,
        input_map,
    )
    local_map.update(
        {address: target_indices[slot] for address, slot in input_map.items()}
    )
    return local_map


def allocate_controlled_workspaces(
    emit_pass: "StandardEmitPass",
    operations: list[Operation],
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any],
) -> None:
    """Reserve parent-circuit wires for fresh allocations in controlled bodies.

    A reusable backend gate can only act on wires supplied by its call site.
    When a controlled qkernel allocates private workspace, its body must
    therefore be decomposed on the parent circuit and the workspace wires
    must be included before that circuit's width is fixed. This pre-emission
    walk allocates those nested ``QInitOperation`` resources and publishes
    their addresses in ``qubit_map`` for the controlled walker.

    Args:
        emit_pass (StandardEmitPass): Active emit pass and resource allocator.
        operations (list[Operation]): Segment operations to inspect recursively.
        qubit_map (QubitMap): Parent logical-to-physical map, mutated in place.
        clbit_map (ClbitMap): Parent classical map used to seed nested allocation.
        bindings (dict[str, Any]): Compile-time bindings visible in the segment.

    Raises:
        EmitError: If a controlled body's target footprint or workspace size
            cannot be resolved at transpile time.
    """
    for op in operations:
        if isinstance(op, ControlledUOperation) and op.block is not None:
            block = _prepare_nested_block_for_emit(op.block, bindings)
            local_bindings = emit_pass._resolver.bind_block_params(
                block,
                op.param_operands,
                bindings,
                parameter_factory=emit_pass._get_or_create_parameter,
            )
            if _body_allocates_workspace(block.operations):
                target_operands = [
                    operand
                    for operand in op.target_operands
                    if operand.type.is_quantum()
                ]
                target_groups = [
                    _expand_quantum_operands_to_phys(
                        emit_pass,
                        operand,
                        qubit_map,
                        bindings,
                        operation="ControlledUOperation",
                    )
                    for operand in target_operands
                ]
                target_indices = [index for group in target_groups for index in group]
                local_map: QubitMap = {}
                local_bindings = _bind_and_populate_block_inputs(
                    emit_pass,
                    block,
                    [*target_operands, *op.param_operands],
                    len(target_indices),
                    bindings,
                    local_map,
                    parent_qubits=target_indices,
                )
                allocation_seed = dict(qubit_map)
                allocation_seed.update(local_map)
                with emit_pass._allocator.preserving_analysis_state():
                    allocated_qubits, allocated_clbits = emit_pass._allocator.allocate(
                        block.operations,
                        local_bindings,
                        initial_qubit_map=allocation_seed,
                        initial_clbit_map=clbit_map,
                    )
                qubit_map.update(allocated_qubits)
                clbit_map.update(allocated_clbits)
            allocate_controlled_workspaces(
                emit_pass,
                block.operations,
                qubit_map,
                clbit_map,
                local_bindings,
            )
        if isinstance(op, HasNestedOps):
            for nested in op.nested_op_lists():
                allocate_controlled_workspaces(
                    emit_pass,
                    nested,
                    qubit_map,
                    clbit_map,
                    bindings,
                )


def _body_allocates_workspace(operations: list[Operation]) -> bool:
    """Return whether a controlled body directly allocates fresh qubits.

    Nested ordinary control-flow regions share the body's allocation scope and
    are included. A nested ``ControlledUOperation`` owns a separate callable
    scope and is intentionally left for ``allocate_controlled_workspaces``'s
    recursive call.

    Args:
        operations (list[Operation]): Controlled body operations.

    Returns:
        bool: True when a QInit operation occurs in this callable scope.
    """
    for operation in operations:
        if isinstance(operation, QInitOperation):
            return True
        if isinstance(operation, HasNestedOps) and any(
            _body_allocates_workspace(nested) for nested in operation.nested_op_lists()
        ):
            return True
    return False


_BATCH_MIN_WEIGHT = 2

# Gate types that are already cheap under two composed controls (a native
# Toffoli / Hadamard-conjugated Toffoli) or that QURI Parts lowers to a
# Toffoli pair even under a single control (RZZ). Batching a body made up
# only of these behind one AND ancilla at exactly two controls is a wash
# or a loss, so the ``num == 2`` fast path skips it.
_BATCH_NATIVE_AT_TWO_CONTROLS = frozenset(
    {
        GateOperationType.X,
        GateOperationType.Z,
        GateOperationType.CX,
        GateOperationType.CZ,
        GateOperationType.TOFFOLI,
        GateOperationType.RZZ,
    }
)


def _batch_op_weight(
    emit_pass: "StandardEmitPass",
    op: Operation,
    bindings: dict[str, Any],
) -> int:
    """Return how much a single controlled-body op argues for batching.

    Batching an AND ladder once for the whole body pays off only when the
    body actually emits two or more gates under the shared controls. This
    weight distinguishes real work from no-ops so a statically empty loop
    or a zero-power nested controlled-U does not trigger a wasted ladder.

    Args:
        emit_pass (StandardEmitPass): Active emit pass (for power / loop
            bound resolution).
        op (Operation): One operation of the controlled block body.
        bindings (dict[str, Any]): Bindings visible inside the block.

    Returns:
        int: 0 for ops that emit nothing, 1 for a single controlled gate,
            2 for constructs that on their own justify batching.
    """
    if isinstance(op, (BinOp, CompOp, CondOp, NotOp, ReturnOperation)):
        return 0
    if isinstance(op, GateOperation):
        return 1
    if isinstance(op, ControlledUOperation):
        try:
            power = resolve_power(emit_pass, op, bindings)
        except EmitError:
            # Unresolvable power fails identically on the non-batch path;
            # count it as real work so the estimate is not skewed.
            return 1
        return 1 if power > 0 else 0
    if isinstance(op, PauliEvolveOp):
        return 2
    if isinstance(op, ForOperation):
        return _for_batch_weight(emit_pass, op, bindings)
    if isinstance(op, IfOperation):
        resolved = resolve_if_condition(op.condition, bindings)
        if resolved is None:
            return 1
        selected = op.true_operations if resolved else op.false_operations
        return _controlled_body_batch_weight(emit_pass, selected, bindings)
    if isinstance(op, InvokeOperation):
        block = op.effective_body(backend=getattr(emit_pass, "backend_name", None))
        if block is None:
            return 0
        return _controlled_body_batch_weight(emit_pass, block.operations, bindings)
    if isinstance(op, InverseBlockOperation):
        block = (
            op.implementation_block
            if op.implementation_block is not None
            else op.source_block
        )
        if block is None:
            return 0
        return _controlled_body_batch_weight(emit_pass, block.operations, bindings)
    if isinstance(op, SelectOperation):
        return 1
    # Unsupported op kinds are rejected by the walker further down; if a
    # ladder is emitted before that failure the whole transpile aborts, so
    # counting them as real work here is harmless.
    return 1


def _for_batch_weight(
    emit_pass: "StandardEmitPass",
    op: ForOperation,
    bindings: dict[str, Any],
) -> int:
    """Return the batch weight of a for loop in a controlled block body.

    A statically empty loop emits nothing (weight 0); a single iteration
    contributes its body's weight; two or more iterations that emit any
    work justify batching on their own (weight 2), because the ladder is
    hoisted out of the loop and amortised over every iteration.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        op (ForOperation): Loop operation in the controlled body.
        bindings (dict[str, Any]): Bindings visible inside the block.

    Returns:
        int: The loop's batch weight (0, 1, or 2).
    """
    start, stop, step = resolve_loop_bounds(emit_pass._resolver, op, bindings)
    if start is None or stop is None or step is None or step == 0:
        # Unresolvable bounds make the non-batch walker raise the same
        # EmitError; do not emit a ladder ahead of that failure.
        return 0
    try:
        iteration_count = len(range(start, stop, step))
    except OverflowError:
        iteration_count = 2
    if iteration_count == 0:
        return 0
    body_weight = _controlled_body_batch_weight(emit_pass, op.operations, bindings)
    if iteration_count == 1:
        return body_weight
    return _BATCH_MIN_WEIGHT if body_weight >= 1 else 0


def _controlled_body_batch_weight(
    emit_pass: "StandardEmitPass",
    operations: list[Operation],
    bindings: dict[str, Any],
) -> int:
    """Sum the batch weights of a controlled block body, capped at the threshold.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        operations (list[Operation]): Controlled block body operations.
        bindings (dict[str, Any]): Bindings visible inside the block.

    Returns:
        int: The total weight, clamped to ``_BATCH_MIN_WEIGHT`` once reached
            (callers only compare against that threshold).
    """
    total = 0
    for op in operations:
        total += _batch_op_weight(emit_pass, op, bindings)
        if total >= _BATCH_MIN_WEIGHT:
            return _BATCH_MIN_WEIGHT
    return total


def _body_has_rotation_like_leaf(operations: list[Operation]) -> bool:
    """Return True when the body has a gate that batching helps at two controls.

    Used only for the ``num == 2`` guard: X / Z / CX / CZ / TOFFOLI / RZZ
    gain nothing (or lose) from a two-control AND ladder, but any other
    single-qubit rotation, or any nested construct that recurses into
    further controlled lowering, does benefit.

    Args:
        operations (list[Operation]): Controlled block body operations.

    Returns:
        bool: True if at least one op benefits from batching at two controls.
    """
    for op in operations:
        if isinstance(op, GateOperation):
            if op.gate_type not in _BATCH_NATIVE_AT_TWO_CONTROLS:
                return True
        elif isinstance(
            op,
            (
                ControlledUOperation,
                ForOperation,
                InvokeOperation,
                InverseBlockOperation,
                PauliEvolveOp,
                SelectOperation,
            ),
        ):
            return True
    return False


def try_emit_batched_controlled_operations(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    operations: list[Operation],
    control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
    walker: Callable[..., None],
) -> bool:
    """Emit a controlled block body behind a single shared AND ladder.

    When a block body emits several gates under the same composed
    controls, lowering each gate through its own Toffoli cascade rebuilds
    the same AND ladder per gate — the uncompute of one gate and the
    recompute of the next cancel. This helper instead ANDs the controls
    onto one ancilla once, walks the whole body under that single control,
    then uncomputes the ladder once. Nested controlled-U operations
    compose ``[and_ancilla]`` with their own controls and re-enter
    ``walker``, which batches again from the advanced pool offset; loops
    are batched before iteration expansion, so the ladder is hoisted out.

    Args:
        emit_pass (StandardEmitPass): Active emit pass (must hold a
            ``_mc_ancilla_pool``).
        circuit (Any): Backend circuit being emitted into.
        operations (list[Operation]): Controlled block body operations.
        control_indices (list[int]): Composed physical control qubits.
        qubit_map (QubitMap): Mutable block-local qubit map.
        bindings (dict[str, Any]): Bindings visible inside the block.
        walker (Callable[..., None]): The controlled-body walker to run
            under the single AND control (its own signature).

    Returns:
        bool: True when the body was batched, False when the caller should
            fall back to per-gate emission (fewer than two controls, no
            pool, insufficient weight, the two-control guard, or a pool
            that cannot spare the ladder).
    """
    num_controls = len(control_indices)
    if num_controls < 2:
        return False
    pool = emit_pass._mc_ancilla_pool
    if pool is None:
        return False
    if (
        _controlled_body_batch_weight(emit_pass, operations, bindings)
        < _BATCH_MIN_WEIGHT
    ):
        return False
    if num_controls == 2 and not _body_has_rotation_like_leaf(operations):
        return False
    with pool.try_hold(num_controls - 1) as ancillas:
        if ancillas is None:
            # The demand estimate legitimately reserved fewer ancillas than
            # a batch would want (e.g. a sibling whose demand dominates);
            # fall back to per-gate lowering rather than over-reserving.
            return False
        steps = _and_ladder_steps(control_indices, ancillas)
        _emit_toffoli_steps(emit_pass._emitter, circuit, steps)
        walker(
            emit_pass,
            circuit,
            operations,
            [ancillas[num_controls - 2]],
            qubit_map,
            bindings,
        )
        _emit_toffoli_steps(emit_pass._emitter, circuit, reversed(steps))
    return True


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
    if try_emit_batched_controlled_operations(
        emit_pass,
        circuit,
        operations,
        control_indices,
        qubit_map,
        bindings,
        walker=emit_controlled_operations,
    ):
        return
    for op in operations:
        if isinstance(op, QInitOperation):
            # Allocation itself is not a unitary instruction to control. Its
            # clean parent-circuit wire was reserved by
            # ``allocate_controlled_workspaces`` before circuit creation; all
            # subsequent preparation and uncomputation gates inherit the
            # outer controls through this walker.
            result = op.results[0]
            from qamomile.circuit.ir.value import ArrayValue

            if isinstance(result, ArrayValue):
                if result.shape:
                    size = emit_pass._resolver.resolve_int_value(
                        result.shape[0], bindings
                    )
                    if size is None:
                        raise EmitError(
                            "Cannot resolve controlled workspace array size.",
                            operation="QInitOperation",
                        )
                    missing = [
                        QubitAddress(result.uuid, index)
                        for index in range(size)
                        if QubitAddress(result.uuid, index) not in qubit_map
                    ]
                    if missing:
                        raise EmitError(
                            "Controlled workspace was not reserved before "
                            f"emission: {missing[0]!s}.",
                            operation="QInitOperation",
                        )
            elif QubitAddress(result.uuid) not in qubit_map:
                raise EmitError(
                    "Controlled scalar workspace was not reserved before emission.",
                    operation="QInitOperation",
                )
        elif isinstance(op, GateOperation):
            gate_targets = _resolve_controlled_gate_targets(
                emit_pass, op, qubit_map, bindings
            )
            emit_multi_controlled_gate(
                emit_pass, circuit, op, control_indices, gate_targets, bindings
            )
            _propagate_controlled_gate_results(op, gate_targets, qubit_map)
        elif isinstance(op, BinOp):
            evaluate_binop(emit_pass, op, bindings)
        elif isinstance(op, (CompOp, CondOp, NotOp)):
            evaluate_classical_predicate(emit_pass, op, bindings)
        elif isinstance(op, IfOperation):
            emit_static_controlled_if(
                emit_pass,
                circuit,
                op,
                control_indices,
                qubit_map,
                bindings,
                walker=emit_controlled_operations,
            )
        elif isinstance(op, ControlledUOperation):
            _emit_nested_controlled_u(
                emit_pass, circuit, op, control_indices, qubit_map, bindings
            )
        elif isinstance(op, InvokeOperation):
            composite_control_groups = [
                _expand_quantum_operands_to_phys(
                    emit_pass,
                    operand,
                    qubit_map,
                    bindings,
                    operation="InvokeOperation",
                )
                for operand in op.control_qubits
            ]
            composite_target_groups = [
                _expand_quantum_operands_to_phys(
                    emit_pass,
                    operand,
                    qubit_map,
                    bindings,
                    operation="InvokeOperation",
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
        elif isinstance(op, GlobalPhaseOperation):
            emit_controlled_global_phase_operation(
                emit_pass,
                circuit,
                op,
                control_indices,
                bindings,
            )
        elif isinstance(op, SelectOperation):
            emit_pass._emit_select(
                circuit,
                op,
                qubit_map,
                bindings,
                outer_control_indices=control_indices,
            )
        elif isinstance(op, PauliEvolveOp):
            emit_controlled_pauli_evolve(
                emit_pass, circuit, op, control_indices, qubit_map, bindings
            )
        elif isinstance(op, ForOperation):
            replay_controlled_for(
                emit_pass,
                circuit,
                op,
                control_indices,
                qubit_map,
                bindings,
                walker=emit_controlled_operations,
            )
        elif isinstance(op, HasNestedOps):
            raise EmitError(
                f"Unsupported control flow {type(op).__name__} in controlled "
                "block decomposition. Only compile-time-resolved IfOperation "
                "and statically bounded ForOperation bodies are supported.",
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


def replay_controlled_for(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: ForOperation,
    control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
    *,
    walker: Callable[..., None],
) -> None:
    """Replay one static range loop under accumulated quantum controls.

    Controlled blocks use a specialized operation walker because every
    quantum gate in the body must inherit the outer controls.  Reusing only
    the old loop-variable binding logic here skipped the loop's explicit
    ``RegionArg`` protocol, so a scalar recurrence such as ``index += 1``
    either stayed pinned to its initial value or became unresolved.  This
    helper shares the canonical emit-time carry primitives with ordinary
    loop emission and accepts the backend-specific controlled walker as a
    callback.  Nested range loops therefore replay recursively with the same
    ``init -> block_arg -> yielded -> result`` semantics on every backend.

    Args:
        emit_pass (StandardEmitPass): Active emit pass and value resolver.
        circuit (Any): Backend circuit being constructed.
        op (ForOperation): Static range loop to replay.
        control_indices (list[int]): Physical controls accumulated from the
            enclosing controlled operations.
        qubit_map (QubitMap): Mutable block-local qubit map.
        bindings (dict[str, Any]): Bindings visible before the loop.  Final
            RegionArg results and the final loop variable are published here.
        walker (Callable[..., None]): Controlled operation walker with the
            same leading arguments as :func:`emit_controlled_operations`.

    Returns:
        None: The callback appends operations to ``circuit`` in place.

    Raises:
        EmitError: If the bounds are unresolved or invalid, loop identities
            are malformed, or a RegionArg value cannot be resolved.
    """
    from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
        _advance_region_args,
        _bind_loop_var,
        _publish_region_results,
        _seed_region_args,
        resolve_loop_bounds,
        validated_loop_indexset,
    )

    if op.loop_var_value is None:
        raise EmitError(
            f"ForOperation '{op.loop_var or '<unnamed>'}' has no "
            "loop_var_value; the IR must be rebuilt with the current frontend.",
            operation="ForOperation",
        )
    start, stop, step = resolve_loop_bounds(emit_pass._resolver, op, bindings)
    if start is None or stop is None or step is None:
        raise EmitError(
            "Cannot resolve ForOperation bounds in controlled block. "
            "Loop bounds must be resolvable at transpile time.",
            operation="ControlledUOperation",
        )

    indexset = validated_loop_indexset(start, stop, step)
    carried = _seed_region_args(emit_pass, op, bindings)
    last_index: int | None = None
    for index in indexset:
        last_index = index
        loop_bindings = bindings.copy()
        for value_uuid, carried_value in carried.items():
            _set_emit_value(loop_bindings, value_uuid, carried_value)
        _bind_loop_var(loop_bindings, op, index)
        walker(
            emit_pass,
            circuit,
            op.operations,
            control_indices,
            qubit_map,
            loop_bindings,
        )
        _advance_region_args(emit_pass, op, carried, loop_bindings)

    _publish_region_results(op, carried, bindings)
    if last_index is not None:
        _bind_loop_var(bindings, op, last_index)


def emit_static_controlled_if(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: IfOperation,
    control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
    walker: Callable[..., None],
) -> None:
    """Emit the selected branch of a static if under accumulated controls.

    This is the common safety net for controlled-body walkers. Compile-time
    lowering normally removes these nodes earlier, but a loop induction value
    can make the condition concrete only while emit unrolls that iteration.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (IfOperation): Static branch to resolve and emit.
        control_indices (list[int]): Accumulated physical control qubits.
        qubit_map (QubitMap): Mutable block-local qubit map.
        bindings (dict[str, Any]): Bindings visible in the current iteration.
        walker (Callable[..., None]): Backend controlled-body walker used to
            emit the selected branch.

    Raises:
        EmitError: If the condition remains runtime-dependent.
    """
    resolved = resolve_if_condition(op.condition, bindings)
    if resolved is None:
        raise EmitError(
            "IfOperation in a controlled unitary must resolve from compile-time "
            "or loop-iteration bindings before fallback emission. Runtime "
            "measurement-backed control flow is not unitary.",
            operation="ControlledGate",
        )
    selected = op.true_operations if resolved else op.false_operations
    walker(
        emit_pass,
        circuit,
        selected,
        control_indices,
        qubit_map,
        bindings,
    )
    remap_static_merge_outputs(
        op,
        resolved,
        qubit_map,
        {},
        bindings=bindings,
        resolver=emit_pass._resolver,
    )
    register_classical_merge_aliases(
        emit_pass,
        op,
        bindings,
        resolved,
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

    target_qubit_operands = [
        operand for operand in op.target_operands if operand.type.is_quantum()
    ]
    param_operands = op.param_operands

    target_index_groups = [
        _expand_quantum_operands_to_phys(emit_pass, operand, qubit_map, bindings)
        for operand in target_qubit_operands
    ]
    target_phys = [i for group in target_index_groups for i in group]

    local_bindings = emit_pass._resolver.bind_block_params(
        op.block,
        param_operands,
        bindings,
        parameter_factory=emit_pass._get_or_create_parameter,
    )
    _bind_quantum_input_shapes(
        emit_pass._resolver, op.block, target_qubit_operands, bindings, local_bindings
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

    if power == 0:
        map_nested_controlled_u_results(op, resolved, qubit_map)
        return

    control_value = op.control_value if isinstance(op, ConcreteControlledU) else None
    with bracket_control_value(
        emit_pass,
        circuit,
        resolved.control_phys,
        control_value,
    ):
        unitary_gate = emit_pass._blockvalue_to_gate(
            block, len(resolved.target_phys), resolved.local_bindings
        )
        if unitary_gate is not None:
            if power > 1:
                unitary_gate = emit_pass._emitter.gate_power(unitary_gate, power)
            controlled_gate = emit_pass._emitter.gate_controlled(
                unitary_gate, len(composed_controls)
            )
            _checked_append_gate(
                emit_pass,
                circuit,
                controlled_gate,
                composed_controls + resolved.target_phys,
                "controlled gate",
            )
        else:
            inner_map = build_controlled_block_qubit_map(
                emit_pass,
                block,
                resolved.target_phys,
                resolved.local_bindings,
                parent_qubit_map=qubit_map,
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
    ``RZ`` therefore carries the controls, which keeps the
    multi-controlled cost to a single rotation per Hamiltonian term
    instead of controlling every basis-change and ladder gate.

    The basis-change (``H`` / ``SDG`` / ``S``) and CX-ladder gates are
    emitted uncontrolled through ``emit_pass._emitter``; the central
    ``RZ`` is routed through :func:`_emit_mc_rotation`, which dispatches
    to ``emit_crz`` for a single control and to the backend's
    ``_emit_irreducible_multi_controlled_gate`` hook for two or more.

    A constant (identity) Hamiltonian term ``c * I`` is the standalone phase
    ``exp(-i * gamma * c)`` for an uncontrolled evolution and is retained by
    :func:`emit_pauli_evolve`. Under controls it becomes an *observable*
    relative phase on the all-controls-on subspace, so it is realized here as
    a ``P(-gamma * c)`` on one control conditioned on the remaining controls
    (``emit_p`` for a single control, a multi-controlled ``P`` for more),
    matching Qiskit's native ``PauliEvolutionGate`` whose ``SparsePauliOp``
    carries the constant.

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
            non-real coefficient), the Hamiltonian is larger than the
            register, a term qubit cannot be resolved, or gamma is
            runtime-parametric and the backend's runtime parameter type
            does not support the required angle scaling (e.g. QURI
            Parts' ``Parameter``).
    """
    import qamomile.observable as qm_o
    from qamomile.circuit.transpiler.passes.emit_support.pauli_evolve_emission import (
        _resolve_gamma,
        validate_hamiltonian_within_register,
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
                concrete ``float`` gamma never raises.)
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
    # A Hamiltonian smaller than the register is embedded by acting only
    # on its declared qubits (the leading ``num_qubits`` entries of
    # ``qubit_indices``); the register-wide result mapping at the end of
    # this function keeps the untouched tail resolvable.
    validate_hamiltonian_within_register(hamiltonian.num_qubits, len(qubit_indices))

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
    if abs(constant.imag) > HERMITIAN_IMAG_ATOL:
        raise EmitError(
            f"PauliEvolveOp requires a Hermitian Hamiltonian (real "
            f"coefficients), but found a complex constant {constant}.",
            operation="PauliEvolveOp",
        )
    if constant.real:
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
        resolved_power = power

    elif isinstance(power, Value):
        resolved = emit_pass._resolver.resolve_classical_value(power, bindings)
        if resolved is None:
            raise EmitError(
                f"Cannot resolve ControlledU power '{power.name}'. "
                f"Ensure all parameters are bound before transpilation.",
                operation="ControlledUOperation",
            )
        resolved_power = int(resolved)

    else:
        raise EmitError(
            f"ControlledU power has unexpected type "
            f"{type(power).__name__}. Expected int or Value.",
            operation="ControlledUOperation",
        )

    if resolved_power < 0:
        raise EmitError(
            f"ControlledU power must be non-negative, got {resolved_power}.",
            operation="ControlledUOperation",
        )
    return resolved_power


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
    control-operand layout.

    The function resolves the symbolic ``num_controls`` and every
    ``control_indices`` entry to concrete ints, walks the control
    pool's ``slice_of`` chain to look up physical qubits per element,
    builds ``control_phys`` from the selected pool slots, expands the
    sub-kernel quantum operands via
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

    target_qubit_operands = [
        operand for operand in op.target_operands if operand.type.is_quantum()
    ]
    param_operands = op.param_operands

    target_indices: list[int] = []
    target_index_groups: list[list[int]] = []
    for q in target_qubit_operands:
        indices = _expand_quantum_operands_to_phys(emit_pass, q, qubit_map, bindings)
        target_index_groups.append(indices)
        target_indices.extend(indices)

    block_value = op.block
    local_bindings = emit_pass._resolver.bind_block_params(
        block_value,
        param_operands,
        bindings,
        parameter_factory=emit_pass._get_or_create_parameter,
    )
    _bind_quantum_input_shapes(
        emit_pass._resolver,
        block_value,
        target_qubit_operands,
        bindings,
        local_bindings,
    )
    block_value = _prepare_nested_block_for_emit(block_value, local_bindings)

    num_targets = len(target_indices)
    unitary_gate = emit_pass._blockvalue_to_gate(
        block_value, num_targets, local_bindings
    )

    power_value = resolve_power(emit_pass, op, bindings)

    if power_value != 0:
        if unitary_gate is not None:
            if power_value > 1:
                unitary_gate = emit_pass._emitter.gate_power(unitary_gate, power_value)
            controlled_gate = emit_pass._emitter.gate_controlled(unitary_gate, nc)
            _checked_append_gate(
                emit_pass,
                circuit,
                controlled_gate,
                control_phys + target_indices,
                "controlled gate",
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
    indices via ``_expand_quantum_operands_to_phys``, asserts the resulting count
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

    target_qubit_operands = [
        operand for operand in op.target_operands if operand.type.is_quantum()
    ]
    param_operands = op.param_operands

    target_indices: list[int] = []
    target_index_groups: list[list[int]] = []
    for q in target_qubit_operands:
        indices = _expand_quantum_operands_to_phys(emit_pass, q, qubit_map, bindings)
        target_index_groups.append(indices)
        target_indices.extend(indices)

    block_value = op.block
    local_bindings = emit_pass._resolver.bind_block_params(
        block_value,
        param_operands,
        bindings,
        parameter_factory=emit_pass._get_or_create_parameter,
    )
    _bind_quantum_input_shapes(
        emit_pass._resolver,
        block_value,
        target_qubit_operands,
        bindings,
        local_bindings,
    )
    block_value = _prepare_nested_block_for_emit(block_value, local_bindings)

    num_targets = len(target_indices)
    unitary_gate = emit_pass._blockvalue_to_gate(
        block_value, num_targets, local_bindings
    )

    power_value = resolve_power(emit_pass, op, bindings)

    if power_value != 0:
        if unitary_gate is not None:
            if power_value > 1:
                unitary_gate = emit_pass._emitter.gate_power(unitary_gate, power_value)
            controlled_gate = emit_pass._emitter.gate_controlled(unitary_gate, nc)
            _checked_append_gate(
                emit_pass,
                circuit,
                controlled_gate,
                control_phys + target_indices,
                "controlled gate",
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
    target_qubit_operands = [
        operand for operand in op.target_operands if operand.type.is_quantum()
    ]
    param_operands = op.param_operands

    # The frontend normalises ``operands[:num_controls]`` to one scalar per physical
    # control qubit, so each control operand maps to a single physical
    # index.  ``Vector[Qubit]`` / ``VectorView`` controls are already
    # expanded into per-element scalars upstream.
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
    # that contributes ``length`` physical qubits).  The shared expansion
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
        block_value,
        param_operands,
        bindings,
        parameter_factory=emit_pass._get_or_create_parameter,
    )
    _bind_quantum_input_shapes(
        emit_pass._resolver,
        block_value,
        target_qubit_operands,
        bindings,
        local_bindings,
    )
    block_value = _prepare_nested_block_for_emit(block_value, local_bindings)

    power_value = resolve_power(emit_pass, op, bindings)
    if power_value == 0:
        _map_controlled_u_results(
            op,
            nc,
            control_indices,
            target_qubit_operands,
            target_index_groups,
            qubit_map,
        )
        return

    with bracket_control_value(
        emit_pass,
        circuit,
        control_indices,
        op.control_value,
    ):
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
        else:
            num_targets = len(target_indices)
            unitary_gate = emit_pass._blockvalue_to_gate(
                block_value, num_targets, local_bindings
            )

            if unitary_gate is not None:
                if power_value > 1:
                    unitary_gate = emit_pass._emitter.gate_power(
                        unitary_gate, power_value
                    )
                controlled_gate = emit_pass._emitter.gate_controlled(unitary_gate, nc)
                _checked_append_gate(
                    emit_pass,
                    circuit,
                    controlled_gate,
                    control_indices + target_indices,
                    "controlled gate",
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
    if power == 0:
        return

    unitary_gate = emit_pass._blockvalue_to_gate(block_value, 1, bindings)
    if unitary_gate is not None:
        if power > 1:
            unitary_gate = emit_pass._emitter.gate_power(unitary_gate, power)
        controlled_gate = emit_pass._emitter.gate_controlled(unitary_gate, num_controls)
        for target_idx in target_indices:
            _checked_append_gate(
                emit_pass,
                circuit,
                controlled_gate,
                control_indices + [target_idx],
                "controlled gate",
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

    # The fallback (block decomposition) path does not go through
    # ``_checked_append_gate``, so re-run the shared aliasing check on the
    # combined control + target set here. A control that coincides with a
    # target (or a duplicated target) at runtime is physically ill-defined and
    # would otherwise decompose into gates acting twice on one qubit.
    reject_duplicate_physical_indices(
        "controlled gate (fallback)", control_indices + target_indices
    )

    block_value = _prepare_nested_block_for_emit(block_value, bindings)
    qubit_map = build_controlled_block_qubit_map(
        emit_pass,
        block_value,
        target_indices,
        bindings,
        parent_qubit_map=getattr(emit_pass, "_active_qubit_map", None),
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
        operation_name="InvokeOperation",
    )

    if custom_gate is not None and _gate_matches_qubit_count(custom_gate, num_qubits):
        _checked_append_gate(
            emit_pass, circuit, custom_gate, qubit_indices, "composite gate"
        )
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
            operation_name="InvokeOperation",
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
    op: InvokeOperation,
    control_indices: list[int],
    qubit_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit a composite gate under already-resolved outer controls.

    A non-default activation value applies only to the invocation's own
    leading controls. Enclosing controls remain ordinary all-ones controls;
    bracketing the inner controls around the complete call composes correctly
    even when the enclosing control is inactive.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (InvokeOperation): Composite or oracle invocation to emit.
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
    own_controls = qubit_indices[: op.num_control_qubits]
    with bracket_control_value(
        emit_pass,
        circuit,
        own_controls,
        op.control_value,
    ):
        _emit_all_ones_controlled_composite_at_indices(
            emit_pass,
            circuit,
            op,
            control_indices,
            qubit_indices,
            bindings,
        )


def _emit_all_ones_controlled_composite_at_indices(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: InvokeOperation,
    control_indices: list[int],
    qubit_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit an invocation after activation controls have been normalized.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted into.
        op (InvokeOperation): Composite or oracle invocation to emit.
        control_indices (list[int]): Physical outer control qubits.
        qubit_indices (list[int]): Physical qubits occupied by ``op``'s own
            control and target operands.
        bindings (dict[str, Any]): Active emit bindings.

    Returns:
        None.

    Raises:
        EmitError: If the composite has no implementation block or the
            fallback cannot represent the controlled composite.
    """
    selected_impl = op.implementation_for(
        backend=getattr(emit_pass, "backend_name", None)
    )
    if selected_impl is not None and selected_impl.body is not None:
        if not control_indices:
            emit_pass._emit_custom_composite(
                circuit,
                op,
                selected_impl.body,
                qubit_indices,
                bindings,
            )
            return
        # A transform-specific body already implements the invocation's own
        # control/inverse semantics. Treat all of its qubits as body targets
        # and apply only the controls accumulated from enclosing blocks.
        impl = selected_impl.body
        body_qubits = qubit_indices
        body_operands = op.operands
        all_controls = list(control_indices)
    elif op.transform is CallTransform.INVERSE:
        raise EmitError(
            f"Inverse callable '{op.target.name}' has no inverse "
            "implementation body for this backend. Bind structural "
            "parameters at compile time so the inverse can be "
            "materialized, or register an inverse implementation.",
            operation=f"InvokeOperation[{op.target.name}]",
        )
    else:
        own_control_count = (
            op.num_control_qubits if op.transform is CallTransform.CONTROLLED else 0
        )
        own_controls = qubit_indices[:own_control_count]
        body_qubits = qubit_indices[own_control_count:]
        body_operands = op.operands[own_control_count:]
        all_controls = [*control_indices, *own_controls]

        impl = op.body
        if impl is None:
            raise EmitError(
                "Cannot emit controlled invocation without an implementation block.",
                operation="InvokeOperation",
            )

    num_qubits = len(body_qubits)
    custom_gate = emit_pass._blockvalue_to_gate(
        impl,
        num_qubits,
        bindings,
        input_operands=body_operands,
        operation_name="InvokeOperation",
    )
    if custom_gate is not None:
        controlled_gate = custom_gate
        if all_controls:
            controlled_gate = emit_pass._emitter.gate_controlled(
                custom_gate,
                len(all_controls),
            )
        if controlled_gate is not None and _gate_matches_qubit_count(
            controlled_gate,
            len(all_controls) + num_qubits,
        ):
            _checked_append_gate(
                emit_pass,
                circuit,
                controlled_gate,
                [*all_controls, *body_qubits],
                "composite gate",
            )
            return

    local_qubit_map: QubitMap = {}
    local_bindings = _bind_and_populate_block_inputs(
        emit_pass,
        impl,
        body_operands,
        num_qubits,
        bindings,
        local_qubit_map,
        operation_name="InvokeOperation",
    )
    emit_pass._emit_controlled_fallback(
        circuit,
        impl,
        len(all_controls),
        all_controls,
        body_qubits,
        1,
        local_bindings,
    )
