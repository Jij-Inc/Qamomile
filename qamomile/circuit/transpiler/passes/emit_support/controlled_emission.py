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

import contextlib
import dataclasses
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from qamomile._utils import coerce_nonnegative_integral
from qamomile.circuit.ir._resource_contract import quantum_operand_widths
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    UnaryMathOp,
)
from qamomile.circuit.ir.operation.callable import (
    CallableBodySelection,
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import (
    DictGetItemOperation,
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    HasNestedOps,
    IfOperation,
)
from qamomile.circuit.ir.operation.control_value import control_pattern_for_value
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import CInitOperation, QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.operation.slice_array import (
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.cast_binop_emission import (
    _set_emit_value,
    evaluate_binop,
    evaluate_classical_predicate,
    evaluate_unary_math,
    handle_cast,
)
from qamomile.circuit.transpiler.passes.emit_support.clean_ancilla_toffoli import (
    clean_ancilla_toffoli_ladder,
)
from qamomile.circuit.transpiler.passes.emit_support.condition_resolution import (
    remap_static_merge_outputs,
    resolve_if_condition,
)
from qamomile.circuit.transpiler.passes.emit_support.control_batching import (
    CONTROL_BATCH_MIN_WEIGHT,
    ControlBatchProfile,
    combine_control_batch_profiles,
    should_batch_controlled_body,
    static_controlled_batch_profile,
)
from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
    evaluate_dict_getitem,
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
from qamomile.circuit.transpiler.passes.emit_support.counting_emitter import (
    CountingEmitter,
)
from qamomile.circuit.transpiler.passes.emit_support.gate_emission import (
    reject_duplicate_physical_indices,
    resolve_angle_value,
)
from qamomile.circuit.transpiler.passes.emit_support.global_phase_emission import (
    emit_controlled_global_phase_operation,
    is_identity_phase_angle,
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


@contextlib.contextmanager
def _parameter_probe_scope(
    emit_pass: "StandardEmitPass",
) -> Iterator[None]:
    """Enter the emit pass's analysis-only parameter scope when available.

    Small policy-test doubles predate parameter probing and have no ABI or
    emitter state to mutate. Production passes provide _parameter_probe; the
    fallback keeps resolver-only doubles usable without weakening the
    production boundary.

    Args:
        emit_pass (StandardEmitPass): Active pass or a resolver-only test
            double.

    Yields:
        None: Control returns to the analysis walk.
    """
    probe = getattr(emit_pass, "_parameter_probe", None)
    if callable(probe):
        with probe():
            yield
        return
    yield


@contextlib.contextmanager
def _zero_work_analysis_scope(
    emit_pass: "StandardEmitPass",
    control_indices: list[int],
    target_indices: list[int],
    bindings: dict[str, Any],
) -> Iterator[Any]:
    """Provide an isolated circuit for a zero-work semantic emit walk.

    Production passes delegate to their full analysis-emission transaction,
    which swaps every mutable pass field and restores ``bindings``. The small
    fallback exists only for resolver/emitter test doubles that do not own
    production ABI state; it still swaps in a no-op emitter and never exposes
    the real circuit to the walk.

    Args:
        emit_pass (StandardEmitPass): Active production pass or lightweight
            internal test double.
        control_indices (list[int]): Physical coherent-control slots.
        target_indices (list[int]): Physical target slots.
        bindings (dict[str, Any]): Body-local bindings.

    Yields:
        Any: Stateless circuit accepted by the ordinary controlled walker.
    """
    active_qubit_map = dict(getattr(emit_pass, "_active_qubit_map", None) or {})
    physical_indices = [
        *active_qubit_map.values(),
        *control_indices,
        *target_indices,
    ]
    data_qubit_count = max(physical_indices, default=-1) + 1
    transaction = getattr(emit_pass, "_analysis_emission_transaction", None)
    if callable(transaction):
        with transaction(
            data_qubit_count,
            active_qubit_map,
            {},
            bindings,
        ) as (analysis_circuit, _qubit_map, _clbit_map, _pool):
            yield analysis_circuit
        return

    saved_emitter = emit_pass._emitter
    saved_bindings = dict(bindings)
    emit_pass._emitter = CountingEmitter(saved_emitter)
    try:
        analysis_circuit = emit_pass._emitter.create_circuit(0, 0)
        with _parameter_probe_scope(emit_pass):
            yield analysis_circuit
    finally:
        emit_pass._emitter = saved_emitter
        bindings.clear()
        bindings.update(saved_bindings)


def _checked_append_gate(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    gate: Any,
    qubit_indices: list[int],
    gate_label: str,
) -> None:
    """Append a controlled / composite gate after rejecting qubit aliasing.

    Every controlled or composite block reaches the engine through
    ``append_gate`` with a combined physical-index list (``control_phys +
    target_indices``). A controlled block is defined only on distinct qubits;
    when a symbolic control and target index coincide at runtime (e.g.
    ``qmc.control(x)(qs[i], qs[j])`` on the diagonal) the duplicate is visible
    only here at emit time. This wrapper runs the shared aliasing check before
    delegating to the engine, so the controlled path gets the same Qamomile
    ``QubitAliasError`` the native ``emit_gate`` path already raises, on every
    engine, instead of a raw ``CircuitError`` (Qiskit) or a silent
    compile-then-crash (CUDA-Q).

    Args:
        emit_pass (StandardEmitPass): Active emit pass (for its emitter).
        circuit (Any): Engine circuit being emitted into.
        gate (Any): The already-controlled/powered engine gate to append.
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
        circuit (Any): Engine circuit being emitted into.
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
    *,
    _under_control: bool = False,
    _active_invoke_bodies: set[int] | None = None,
) -> None:
    """Reserve parent-circuit wires for fresh allocations in controlled bodies.

    A reusable engine gate can only act on wires supplied by its call site.
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
        _under_control (bool): Whether ``operations`` are already reached by
            a controlled-body walker. Defaults to False.
        _active_invoke_bodies (set[int] | None): Definition-body identities on
            the active recursive path. Defaults to a fresh set.

    Raises:
        EmitError: If a controlled body's target footprint or workspace size
            cannot be resolved at transpile time.
        ValueError: If a selected callable implementation body disagrees with
            its invocation contract.
    """
    active_invoke_bodies = (
        set() if _active_invoke_bodies is None else _active_invoke_bodies
    )
    for op in operations:
        if isinstance(op, ControlledUOperation) and op.block is not None:
            # A loop-local power can remain unresolved until the controlled
            # walker replays that iteration. Only that case is deferred;
            # invalid powers and resolver diagnostics must fail before width
            # allocation can obscure their source.
            power = _resolve_power_if_bound(emit_pass, op, bindings)
            if power == 0:
                # A zero-powered call is the identity. Its body never executes,
                # so neither direct nor recursively nested private workspaces
                # belong in the parent circuit.
                continue
            block = _prepare_nested_block_for_emit(op.block, bindings)
            local_bindings = emit_pass._resolver.bind_block_params(
                block,
                op.param_operands,
                bindings,
                parameter_factory=emit_pass._get_or_create_parameter,
            )
            target_operands = [
                operand for operand in op.target_operands if operand.type.is_quantum()
            ]
            body_allocates_workspace = _body_allocates_workspace(block.operations)
            scalar_vector_broadcast = _is_single_target_block_vector_broadcast(
                block,
                target_operands,
            )
            target_indices: list[int] | None = None
            if body_allocates_workspace or scalar_vector_broadcast:
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
            if scalar_vector_broadcast and not target_indices:
                # A scalar body broadcast over an empty vector is an identity:
                # neither its own workspace nor recursively nested workspace
                # exists in the emitted circuit.
                continue
            if body_allocates_workspace:
                assert target_indices is not None
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
                _under_control=True,
                _active_invoke_bodies=active_invoke_bodies,
            )
        if isinstance(op, InvokeOperation) and (
            _under_control or op.transform.is_controlled
        ):
            _allocate_selected_invoke_workspaces(
                emit_pass,
                op,
                qubit_map,
                clbit_map,
                bindings,
                active_invoke_bodies=active_invoke_bodies,
            )
        if isinstance(op, HasNestedOps):
            for nested in op.nested_op_lists():
                allocate_controlled_workspaces(
                    emit_pass,
                    nested,
                    qubit_map,
                    clbit_map,
                    bindings,
                    _under_control=_under_control,
                    _active_invoke_bodies=active_invoke_bodies,
                )


def _allocate_selected_invoke_workspaces(
    emit_pass: "StandardEmitPass",
    operation: InvokeOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any],
    *,
    active_invoke_bodies: set[int],
) -> None:
    """Reserve workspace reachable through one controlled invocation body.

    Args:
        emit_pass (StandardEmitPass): Active emit pass and allocator.
        operation (InvokeOperation): Invocation reached under coherent control.
        qubit_map (QubitMap): Parent logical-to-physical map, mutated in place.
        clbit_map (ClbitMap): Parent classical map, mutated in place.
        bindings (dict[str, Any]): Bindings visible at the call site.
        active_invoke_bodies (set[int]): Body identities on the active recursive
            call path.

    Raises:
        EmitError: If body operands or workspace sizes cannot be resolved.
        ValueError: If the selected body violates the invocation contract.
    """
    selection = _controlled_invoke_selection(
        operation,
        getattr(emit_pass, "engine_name", None),
    )
    body = selection.body
    if body is None or not body.operations:
        return
    body_identity = id(body)
    if body_identity in active_invoke_bodies:
        return

    body = _prepare_nested_block_for_emit(body, bindings)
    actual_operands = list(selection.operands)
    body_input_width = _selected_body_quantum_input_width(
        emit_pass,
        operation,
        selection,
        bindings,
    )
    local_map: QubitMap = {}
    local_bindings = _bind_and_populate_block_inputs(
        emit_pass,
        body,
        actual_operands,
        body_input_width,
        bindings,
        local_map,
        operation_name=f"InvokeOperation[{operation.target.name}]",
    )
    body_map = dict(qubit_map)
    body_map.update(local_map)
    if _body_allocates_workspace(body.operations):
        with emit_pass._allocator.preserving_analysis_state():
            allocated_qubits, allocated_clbits = emit_pass._allocator.allocate(
                body.operations,
                local_bindings,
                initial_qubit_map=body_map,
                initial_clbit_map=clbit_map,
            )
        qubit_map.update(allocated_qubits)
        clbit_map.update(allocated_clbits)
        body_map.update(allocated_qubits)

    active_invoke_bodies.add(body_identity)
    try:
        allocate_controlled_workspaces(
            emit_pass,
            body.operations,
            body_map,
            clbit_map,
            local_bindings,
            _under_control=True,
            _active_invoke_bodies=active_invoke_bodies,
        )
    finally:
        active_invoke_bodies.remove(body_identity)
    qubit_map.update(body_map)


def _selected_body_quantum_input_width(
    emit_pass: "StandardEmitPass",
    operation: InvokeOperation,
    selection: CallableBodySelection,
    bindings: dict[str, Any],
) -> int:
    """Resolve the scalar-qubit width passed to a selected callable body.

    Legacy ``num_target_qubits`` metadata is not reliable for every ordinary
    nested qkernel and can count a vector as one Python operand. Deriving the
    width from the aligned selection keeps workspace allocation consistent
    with the actual scalar and vector arguments.

    Args:
        emit_pass (StandardEmitPass): Active emit pass and value resolver.
        operation (InvokeOperation): Invocation used in diagnostics.
        selection (CallableBodySelection): Selected body and aligned operands.
        bindings (dict[str, Any]): Bindings used to resolve array dimensions.

    Returns:
        int: Total scalar-qubit width of the selected body's quantum inputs.

    Raises:
        EmitError: If resource-contract metadata is malformed, an array rank
            or dimension cannot be resolved safely, or a resolved width
            contradicts its exact resource contract.
    """
    operation_label = f"InvokeOperation[{operation.target.name}]"
    try:
        contract_widths = {
            entry.index: entry.width
            for entry in quantum_operand_widths(
                operation.attrs,
                source=operation.target.name,
            )
        }
    except ValueError as error:
        raise EmitError(str(error), operation=operation_label) from error

    contract_offset = (
        operation.num_body_external_control_qubits
        if selection.realized_transform.is_controlled
        else 0
    )
    width = 0
    quantum_index = 0
    for operand in selection.operands:
        if not operand.type.is_quantum():
            continue
        contract_index = quantum_index - contract_offset
        contracted_size = (
            contract_widths.get(contract_index) if contract_index >= 0 else None
        )
        if isinstance(operand, ArrayValue):
            if len(operand.shape) != 1:
                raise EmitError(
                    f"Callable '{operation.target.name}' selected body received "
                    f"a rank-{len(operand.shape)} quantum array; controlled "
                    "workspace allocation supports rank-1 arrays only.",
                    operation=operation_label,
                )
            size = emit_pass._resolver.resolve_int_value(operand.shape[0], bindings)
            if size is None:
                size = contracted_size
        else:
            size = 1
        if size is None or size < 0:
            raise EmitError(
                "Cannot resolve a non-negative quantum-array width for "
                f"callable '{operation.target.name}'.",
                operation=operation_label,
            )
        if contracted_size is not None and size != contracted_size:
            raise EmitError(
                f"Callable '{operation.target.name}' quantum operand "
                f"{contract_index} has width {size}, but its resource "
                f"contract requires {contracted_size}.",
                operation=operation_label,
            )
        width += size
        quantum_index += 1
    return width


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


def _batch_op_profile(
    emit_pass: "StandardEmitPass",
    op: Operation,
    bindings: dict[str, Any],
) -> ControlBatchProfile:
    """Return resolved batching information for one controlled-body operation.

    The profile preserves whether a nested callable selects the shared path at
    exactly two outer controls. This makes the decision invariant under call,
    inverse, loop, and branch boundaries. The enclosing body profiler uses a
    parameter-probe scope, so ordinary value resolution cannot mutate the
    emitted circuit's runtime parameter ABI.

    Args:
        emit_pass (StandardEmitPass): Active emit pass used to resolve
            compile-time values and callable implementations.
        op (Operation): One operation inside the controlled body.
        bindings (dict[str, Any]): Scratch bindings visible inside the block.
            Concrete classical operation results are added in program order.

    Returns:
        ControlBatchProfile: Resolved work and exact-two-control choice.

    Raises:
        EmitError: If resolving loop bounds or carried values, a phase angle,
            Pauli-evolution time, controlled-call power, nested operands, or
            control metadata finds an invalid compile-time value.
        ValueError: If a selected callable implementation body disagrees with
            its invocation contract.
    """
    if isinstance(op, BinOp):
        evaluate_binop(emit_pass, op, bindings)
        return ControlBatchProfile()
    if isinstance(op, (CompOp, CondOp, NotOp)):
        evaluate_classical_predicate(emit_pass, op, bindings)
        return ControlBatchProfile()
    if isinstance(op, UnaryMathOp):
        evaluate_unary_math(emit_pass, op, bindings)
        return ControlBatchProfile()
    if isinstance(op, DictGetItemOperation):
        evaluate_dict_getitem(emit_pass, op, bindings)
        return ControlBatchProfile()
    static_profile = static_controlled_batch_profile(op)
    if static_profile is not None:
        return static_profile
    if isinstance(op, GlobalPhaseOperation):
        angle = resolve_angle_value(emit_pass, op.phase, bindings)
        if is_identity_phase_angle(angle):
            return ControlBatchProfile()
        return ControlBatchProfile(weight=1)
    if isinstance(op, PauliEvolveOp):
        from qamomile.circuit.transpiler.passes.emit_support.pauli_evolve_emission import (
            _resolve_gamma,
            is_zero_evolution_time,
        )

        gamma = _resolve_gamma(emit_pass, op, bindings)
        if is_zero_evolution_time(gamma):
            return ControlBatchProfile()
        return ControlBatchProfile(
            weight=CONTROL_BATCH_MIN_WEIGHT,
            selects_exact_two=True,
        )
    if isinstance(op, ControlledUOperation):
        power = _resolve_power_if_bound(emit_pass, op, bindings)
        # A loop-local power is resolved when its iteration is replayed. Count
        # that unresolved operation as real work without hiding invalid values.
        if power == 0:
            return ControlBatchProfile()
        if op.block is None:
            return ControlBatchProfile(weight=1, selects_exact_two=True)
        body_operands = op.body_operands
        target_operands = [
            operand for operand in op.target_operands if operand.type.is_quantum()
        ]
        if (
            _is_single_target_block_vector_broadcast(op.block, target_operands)
            and _resolve_vector_input_length(
                emit_pass,
                target_operands[0],
                bindings,
            )
            == 0
        ):
            return ControlBatchProfile()
        local_bindings = _bind_block_inputs(
            emit_pass,
            op.block,
            body_operands,
            bindings,
        )
        body_profile = _controlled_body_batch_profile(
            emit_pass,
            op.block.operations,
            local_bindings,
        )
        if body_profile.weight == 0:
            return ControlBatchProfile()
        _controlled_u_num_controls_if_bound(
            emit_pass,
            op,
            bindings,
        )
        return ControlBatchProfile(
            weight=CONTROL_BATCH_MIN_WEIGHT,
            selects_exact_two=True,
        )
    if isinstance(op, ForOperation):
        return _for_batch_profile(emit_pass, op, bindings)
    if isinstance(op, IfOperation):
        resolved = resolve_if_condition(op.condition, bindings)
        if resolved is None:
            return ControlBatchProfile(weight=1)
        selected = op.true_operations if resolved else op.false_operations
        profile = _controlled_body_batch_profile(
            emit_pass,
            selected,
            bindings,
            isolate_bindings=False,
        )
        if profile.decision_complete:
            # No later operation can change the batching decision, so merge
            # values are deliberately not resolved on this short-circuit path.
            return profile
        register_classical_merge_aliases(
            emit_pass,
            op,
            bindings,
            resolved,
        )
        return profile
    if isinstance(op, InvokeOperation):
        engine_name = getattr(emit_pass, "engine_name", None)
        selection = _controlled_invoke_selection(
            op,
            engine_name,
        )
        block = selection.body
        if block is None:
            return ControlBatchProfile(weight=1, selects_exact_two=True)
        local_bindings = _bind_block_inputs(
            emit_pass,
            block,
            list(selection.operands),
            bindings,
        )
        body_profile = _controlled_body_batch_profile(
            emit_pass,
            block.operations,
            local_bindings,
        )
        own_controls = (
            op.num_body_external_control_qubits
            if op.transform.is_controlled
            and not selection.realized_transform.is_controlled
            else 0
        )
        if body_profile.weight and own_controls:
            return ControlBatchProfile(
                weight=CONTROL_BATCH_MIN_WEIGHT,
                selects_exact_two=True,
            )
        return body_profile
    if isinstance(op, SelectOperation):
        return _select_batch_profile(emit_pass, op, bindings)
    if isinstance(op, InverseBlockOperation):
        block = (
            op.implementation_block
            if op.implementation_block is not None
            else op.source_block
        )
        if block is None:
            return ControlBatchProfile(weight=1, selects_exact_two=True)
        local_bindings = _bind_block_inputs(
            emit_pass,
            block,
            [*op.target_qubits, *op.parameters],
            bindings,
        )
        body_profile = _controlled_body_batch_profile(
            emit_pass,
            block.operations,
            local_bindings,
        )
        if body_profile.weight and op.num_control_qubits:
            return ControlBatchProfile(
                weight=CONTROL_BATCH_MIN_WEIGHT,
                selects_exact_two=True,
            )
        return body_profile
    # Unsupported op kinds are rejected by the walker further down; if a
    # ladder is emitted before that failure the whole transpile aborts, so
    # counting them as real work here is harmless.
    return ControlBatchProfile(weight=1, selects_exact_two=True)


def _controlled_u_num_controls_if_bound(
    emit_pass: "StandardEmitPass",
    operation: ControlledUOperation,
    bindings: dict[str, Any],
) -> int | None:
    """Resolve a controlled-U's local control count for batch analysis.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        operation (ControlledUOperation): Nested controlled operation.
        bindings (dict[str, Any]): Bindings visible in the current scope.

    Returns:
        int | None: Positive local control count, or ``None`` while an
            otherwise valid symbolic value remains unresolved.

    Raises:
        EmitError: If a resolved control count is not a positive integer.
    """
    candidate: object
    if isinstance(operation.num_controls, Value):
        resolved = emit_pass._resolver.resolve_classical_value(
            operation.num_controls,
            bindings,
        )
        if resolved is None:
            return None
        candidate = resolved
    else:
        candidate = operation.num_controls
    try:
        count = coerce_nonnegative_integral(
            candidate,
            label="ControlledU num_controls",
        )
    except (TypeError, ValueError) as error:
        raise EmitError(
            str(error),
            operation="ControlledUOperation",
        ) from error
    if count == 0:
        raise EmitError(
            "ControlledU num_controls must be a positive integer.",
            operation="ControlledUOperation",
        )
    return count


def _select_batch_profile(
    emit_pass: "StandardEmitPass",
    operation: SelectOperation,
    bindings: dict[str, Any],
) -> ControlBatchProfile:
    """Return batching information for the active work in a SELECT.

    A SELECT case already inherits at least one index control. Therefore any
    active case selects the shared carrier when the enclosing body adds exactly
    two controls, independently of the case body's primitive gate kinds.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        operation (SelectOperation): SELECT operation to inspect.
        bindings (dict[str, Any]): Bindings visible at the SELECT call site.

    Returns:
        ControlBatchProfile: A heavy exact-two profile when a case is active,
            otherwise an empty profile.

    Raises:
        EmitError: If binding or inspecting a case finds invalid compile-time
            control metadata.
    """
    actual_operands = [
        *operation.target_operands,
        *operation.param_operands,
    ]
    for case_block in operation.case_blocks:
        local_bindings = _bind_block_inputs(
            emit_pass,
            case_block,
            actual_operands,
            bindings,
        )
        if (
            _controlled_body_batch_profile(
                emit_pass,
                case_block.operations,
                local_bindings,
            ).weight
            > 0
        ):
            return ControlBatchProfile(
                weight=CONTROL_BATCH_MIN_WEIGHT,
                selects_exact_two=True,
            )
    return ControlBatchProfile()


def _for_batch_profile(
    emit_pass: "StandardEmitPass",
    op: ForOperation,
    bindings: dict[str, Any],
) -> ControlBatchProfile:
    """Return batching information for a statically bounded range loop.

    Repeating a body increases its bounded work weight but does not change the
    exact-two-control decomposition selected by its leaves. In particular, a
    loop containing only X or Z gates remains on their direct two-control path.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        op (ForOperation): Range loop operation to inspect.
        bindings (dict[str, Any]): Bindings visible before the loop.

    Returns:
        ControlBatchProfile: Resolved loop work and propagated exact-two
            batching choice.

    Raises:
        EmitError: If resolving the loop or its body finds invalid compile-time
            control metadata.
    """
    from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
        _advance_region_args,
        _bind_loop_var,
        _publish_region_results,
        _seed_region_args,
        validated_loop_indexset,
    )

    start, stop, step = resolve_loop_bounds(emit_pass._resolver, op, bindings)
    if start is None or stop is None or step is None or step == 0:
        # Keep an invalid or unresolved loop visible so the controlled walker
        # reaches its normal EmitError instead of mistaking the whole call for
        # an identity. One unit avoids selecting a shared ladder by itself.
        return ControlBatchProfile(weight=1)
    indexset = validated_loop_indexset(start, stop, step)
    iteration_count = len(indexset)
    if (
        iteration_count > 0
        and not op.region_args
        and not op.loop_carried_rebinds
        and all(
            static_controlled_batch_profile(body_op) is not None
            for body_op in op.operations
        )
    ):
        loop_bindings = bindings.copy()
        _bind_loop_var(loop_bindings, op, indexset[0])
        iteration_profile = combine_control_batch_profiles(
            _batch_op_profile(emit_pass, body_op, loop_bindings)
            for body_op in op.operations
        )
        repeated_weight = min(
            CONTROL_BATCH_MIN_WEIGHT,
            iteration_profile.weight * iteration_count,
        )
        _bind_loop_var(bindings, op, indexset[-1])
        return ControlBatchProfile(
            weight=repeated_weight,
            selects_exact_two=iteration_profile.selects_exact_two,
        )

    profile = ControlBatchProfile()
    carried = _seed_region_args(emit_pass, op, bindings)
    last_index: int | None = None
    for index in indexset:
        last_index = index
        loop_bindings = bindings.copy()
        for value_uuid, carried_value in carried.items():
            _set_emit_value(loop_bindings, value_uuid, carried_value)
        _bind_loop_var(loop_bindings, op, index)
        iteration_profile = combine_control_batch_profiles(
            _batch_op_profile(emit_pass, body_op, loop_bindings)
            for body_op in op.operations
        )
        profile = combine_control_batch_profiles((profile, iteration_profile))
        if profile.decision_complete:
            # The caller will stop at this result too. Avoid replaying the
            # remaining loop solely to compute classical carries that can no
            # longer influence the batching choice.
            return profile
        _advance_region_args(emit_pass, op, carried, loop_bindings)

    _publish_region_results(op, carried, bindings)
    if last_index is not None:
        _bind_loop_var(bindings, op, last_index)
    return profile


def _controlled_body_batch_profile(
    emit_pass: "StandardEmitPass",
    operations: list[Operation],
    bindings: dict[str, Any],
    *,
    isolate_bindings: bool = True,
) -> ControlBatchProfile:
    """Combine resolved batching profiles for one controlled block body.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        operations (list[Operation]): Controlled block body operations.
        bindings (dict[str, Any]): Bindings visible inside the block.
        isolate_bindings (bool): Whether to evaluate with a private copy of
            ``bindings``. Set to False for a selected static branch whose
            classical results must feed its enclosing loop. Defaults to True.

    Returns:
        ControlBatchProfile: Capped work and exact-two-control choice for the
            whole body.

    Raises:
        EmitError: If resolving a nested operation finds an invalid control
            count.
    """
    local_bindings = bindings.copy() if isolate_bindings else bindings
    with _parameter_probe_scope(emit_pass):
        return combine_control_batch_profiles(
            _batch_op_profile(emit_pass, op, local_bindings) for op in operations
        )


def _controlled_invoke_selection(
    operation: InvokeOperation,
    engine_name: str | None,
) -> CallableBodySelection:
    """Select the body that generic controlled emission may execute.

    Estimation may count a direct body for an inverse call because reversing a
    known unitary does not change its abstract resource count. Emission cannot
    execute that forward body as its inverse, so this helper deliberately
    removes only that non-executable fallback while preserving its validated
    operand contract.

    Args:
        operation (InvokeOperation): Invocation whose controlled fallback body
            should be selected.
        engine_name (str | None): Active engine name.

    Returns:
        CallableBodySelection: Executable exact/partial selection, or a
        bodyless selection when structural inverse materialization is absent.

    Raises:
        ValueError: If the selected body violates the invocation contract.
    """
    selection = operation.select_body(engine=engine_name)
    if selection.realized_transform is operation.transform:
        return selection
    if (
        operation.transform is CallTransform.CONTROLLED_INVERSE
        and selection.realized_transform is CallTransform.INVERSE
    ):
        return selection
    if not operation.transform.is_inverse:
        return selection
    return dataclasses.replace(selection, body=None)


def _bind_prepared_controlled_body(
    emit_pass: "StandardEmitPass",
    block: Block,
    param_operands: list[Any],
    target_operands: list[Any],
    bindings: dict[str, Any],
) -> tuple[Block, dict[str, Any]]:
    """Bind and prepare one controlled body for analysis or real emission.

    The caller decides whether this runs inside
    :meth:`StandardEmitPass._parameter_probe`.  Analysis must probe; real
    emission must not, so only parameters that reach emitted instructions
    become part of the runtime ABI.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        block (Block): Unprepared controlled body.
        param_operands (list[Any]): Classical call operands.
        target_operands (list[Any]): Quantum call operands used to bind shapes.
        bindings (dict[str, Any]): Caller-visible bindings.

    Returns:
        tuple[Block, dict[str, Any]]: Prepared body and its local bindings.
    """
    local_bindings = emit_pass._resolver.bind_block_params(
        block,
        param_operands,
        bindings,
        parameter_factory=emit_pass._get_or_create_parameter,
    )
    _bind_quantum_input_shapes(
        emit_pass._resolver,
        block,
        target_operands,
        bindings,
        local_bindings,
    )
    return (
        _prepare_nested_block_for_emit(block, local_bindings),
        local_bindings,
    )


def _controlled_body_emission_plan(
    emit_pass: "StandardEmitPass",
    block: Block,
    bindings: dict[str, Any],
    *,
    power: int,
    scalar_vector_broadcast: bool,
    target_indices: list[int],
) -> tuple[bool, ControlBatchProfile | None]:
    """Plan no-op handling and reusable batch analysis for a controlled body.

    Args:
        emit_pass (StandardEmitPass): Active emit pass used to resolve body
            values.
        block (Block): Prepared controlled body.
        bindings (dict[str, Any]): Bindings visible inside ``block``.
        power (int): Resolved nonnegative body repetition count.
        scalar_vector_broadcast (bool): Whether one scalar body is broadcast
            over the target indices.
        target_indices (list[int]): Physical target qubits for this call.

    Returns:
        tuple[bool, ControlBatchProfile | None]: Whether emission should be
            skipped and any batch profile already resolved for identity or
            activation-pattern handling.

    Raises:
        EmitError: If phase or nested batching metadata cannot be resolved.
    """
    if (
        power == 0
        or (scalar_vector_broadcast and not target_indices)
        or not block.operations
    ):
        return True, None

    batch_profile = _controlled_body_batch_profile(
        emit_pass,
        block.operations,
        bindings,
    )
    return batch_profile.weight == 0, batch_profile


def _emit_zero_work_controlled_body_bookkeeping(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    block: Block,
    control_indices: list[int],
    target_indices: list[int],
    power: int,
    bindings: dict[str, Any],
    *,
    scalar_vector_broadcast: bool,
    batch_profile: ControlBatchProfile,
) -> None:
    """Run validation and alias updates for a zero-gate controlled body.

    The ordinary controlled walker runs inside a count-only transaction,
    without open-control brackets or runtime-parameter registration. The real
    circuit and pass state remain untouched, while deferred validation and
    classical value propagation still execute against isolated copies.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Real circuit deliberately withheld from the dry run.
        block (Block): Prepared zero-work controlled body.
        control_indices (list[int]): Resolved coherent controls.
        target_indices (list[int]): Flattened physical target indices.
        power (int): Number of body applications. A positive value executes
            the zero-work validation once because repeating an identical
            bookkeeping-only body cannot add quantum work or change its
            operand contract.
        bindings (dict[str, Any]): Body-local bindings.
        scalar_vector_broadcast (bool): Whether one scalar body is broadcast
            over every target index.
        batch_profile (ControlBatchProfile): Resolved zero-work profile.

    Raises:
        EmitError: If bookkeeping validation or operand mapping fails.
        RuntimeError: If an invalid slice marker reaches emission.
    """
    del circuit
    if power == 0:
        return

    validation_power = 1
    with _zero_work_analysis_scope(
        emit_pass,
        control_indices,
        target_indices,
        bindings,
    ) as analysis_circuit:
        if scalar_vector_broadcast:
            for target_index in target_indices:
                emit_controlled_fallback(
                    emit_pass,
                    analysis_circuit,
                    block,
                    len(control_indices),
                    control_indices,
                    [target_index],
                    validation_power,
                    bindings,
                    batch_profile=batch_profile,
                )
            return
        emit_controlled_fallback(
            emit_pass,
            analysis_circuit,
            block,
            len(control_indices),
            control_indices,
            target_indices,
            validation_power,
            bindings,
            batch_profile=batch_profile,
        )


def _is_resolved_identity_phase_block(
    emit_pass: "StandardEmitPass",
    block: Block,
    bindings: dict[str, Any],
    *,
    batch_profile: ControlBatchProfile | None = None,
) -> bool:
    """Return whether an explicit phase-only body resolves to identity.

    Inverse emission uses this narrower predicate before selecting a reusable
    implementation. General controlled-U emission uses
    :func:`_controlled_body_emission_plan`, which also recognizes
    bookkeeping-only identities.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        block (Block): Candidate inverse implementation body.
        bindings (dict[str, Any]): Bindings visible inside the body.
        batch_profile (ControlBatchProfile | None): Previously resolved body
            profile. Defaults to None.

    Returns:
        bool: Whether the body contains an explicit phase and has zero
            controlled quantum work.

    Raises:
        EmitError: If a phase or nested batching value cannot be resolved.
    """
    if not any(
        isinstance(operation, GlobalPhaseOperation) for operation in block.operations
    ):
        return False
    profile = (
        batch_profile
        if batch_profile is not None
        else _controlled_body_batch_profile(
            emit_pass,
            block.operations,
            bindings,
        )
    )
    return profile.weight == 0


def _has_zero_control(control_value: int | None, num_controls: int) -> bool:
    """Return whether an activation pattern needs an X bracket.

    Args:
        control_value (int | None): Requested activation value, or None for
            the ordinary all-ones pattern.
        num_controls (int): Number of controls described by the pattern.

    Returns:
        bool: True when at least one control is activated on zero.

    Raises:
        TypeError: If ``control_value`` is not a Python integer or None.
        ValueError: If ``control_value`` does not fit ``num_controls``.
    """
    return 0 in control_pattern_for_value(control_value, num_controls)


def try_emit_batched_controlled_operations(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    operations: list[Operation],
    control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
    walker: Callable[..., None],
    batch_profile: ControlBatchProfile | None = None,
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
        circuit (Any): Engine circuit being emitted into.
        operations (list[Operation]): Controlled block body operations.
        control_indices (list[int]): Composed physical control qubits.
        qubit_map (QubitMap): Mutable block-local qubit map.
        bindings (dict[str, Any]): Bindings visible inside the block.
        walker (Callable[..., None]): The controlled-body walker to run
            under the single AND control (its own signature).
        batch_profile (ControlBatchProfile | None): Previously resolved body
            profile for this exact block and binding scope. Defaults to
            ``None``, which computes the profile locally.

    Returns:
        bool: True when the body was batched, False when the caller should
            fall back to per-gate emission (fewer than two controls, no
            pool, insufficient weight, the two-control guard, or a pool
            that cannot spare the ladder).

    Raises:
        EmitError: If resolving nested batching metadata or walking the
            controlled body fails.
    """
    num_controls = len(control_indices)
    if num_controls < 2:
        return False
    pool = emit_pass._mc_ancilla_pool
    if pool is None:
        return False
    profile = (
        batch_profile
        if batch_profile is not None
        else _controlled_body_batch_profile(
            emit_pass,
            operations,
            bindings,
        )
    )
    if not should_batch_controlled_body(
        num_controls=num_controls,
        profile=profile,
    ):
        return False
    recipe = clean_ancilla_toffoli_ladder(num_controls)
    with pool.try_hold(recipe.clean_ancillas) as ancillas:
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
            [ancillas[recipe.clean_ancillas - 1]],
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
    batch_profile: ControlBatchProfile | None = None,
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
        circuit (Any): Engine circuit being emitted into.
        operations (list[Operation]): Block operations to walk.
        control_indices (list[int]): Accumulated physical control
            qubits.
        qubit_map (QubitMap): Mutable block-local qubit map seeded by
            :func:`build_controlled_block_qubit_map`; updated in place
            with gate / nested-op result addresses.
        bindings (dict[str, Any]): Bindings visible inside the block,
            including loop-iteration values during unrolling.
        batch_profile (ControlBatchProfile | None): Previously resolved body
            profile for this exact block and binding scope. Defaults to
            ``None``, which computes the profile only if batching is possible.

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
        batch_profile=batch_profile,
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
        elif isinstance(op, CInitOperation):
            continue
        elif isinstance(op, GateOperation):
            gate_targets = _resolve_controlled_gate_targets(
                emit_pass, op, qubit_map, bindings
            )
            emit_multi_controlled_gate(
                emit_pass, circuit, op, control_indices, gate_targets, bindings
            )
            _propagate_controlled_gate_results(op, gate_targets, qubit_map)
        elif isinstance(op, (SliceArrayOperation, ReleaseSliceViewOperation)):
            emit_pass._reject_slice_marker_at_emit(op)
        elif isinstance(op, StoreArrayElementOperation):
            emit_pass._reject_store_array_element_at_emit(op)
        elif isinstance(op, ReturnQuantumArrayElementOperation):
            emit_pass._validate_quantum_array_element_return(
                op,
                qubit_map,
                bindings,
            )
        elif isinstance(op, CastOperation):
            handle_cast(emit_pass, op, qubit_map)
        elif isinstance(op, BinOp):
            evaluate_binop(emit_pass, op, bindings)
        elif isinstance(op, DictGetItemOperation):
            evaluate_dict_getitem(emit_pass, op, bindings)
        elif isinstance(op, (CompOp, CondOp, NotOp)):
            evaluate_classical_predicate(emit_pass, op, bindings)
        elif isinstance(op, UnaryMathOp):
            evaluate_unary_math(emit_pass, op, bindings)
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
    loop emission and accepts the engine-specific controlled walker as a
    callback.  Nested range loops therefore replay recursively with the same
    ``init -> block_arg -> yielded -> result`` semantics on every engine.

    Args:
        emit_pass (StandardEmitPass): Active emit pass and value resolver.
        circuit (Any): Engine circuit being constructed.
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
        circuit (Any): Engine circuit being emitted into.
        op (IfOperation): Static branch to resolve and emit.
        control_indices (list[int]): Accumulated physical control qubits.
        qubit_map (QubitMap): Mutable block-local qubit map.
        bindings (dict[str, Any]): Bindings visible in the current iteration.
        walker (Callable[..., None]): Engine controlled-body walker used to
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
    *,
    bind_body: bool = True,
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
        bind_body (bool): Whether to bind the selected body's formal
            parameters and shapes. Set to False for a zero-powered identity.
            Defaults to True.

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

    control_operands = op.control_operands
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

    if bind_body:
        local_bindings = emit_pass._resolver.bind_block_params(
            op.block,
            param_operands,
            bindings,
            parameter_factory=emit_pass._get_or_create_parameter,
        )
        _bind_quantum_input_shapes(
            emit_pass._resolver,
            op.block,
            target_qubit_operands,
            bindings,
            local_bindings,
        )
    else:
        local_bindings = dict(bindings)

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
    lowers the result. Engines whose ``circuit_to_gate`` works get a
    single native multi-controlled gate; others recurse through the
    mapped walker with the composed control set.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Engine circuit being emitted into.
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
    power = resolve_power(emit_pass, op, bindings)
    resolved = resolve_controlled_u_call(
        emit_pass,
        op,
        qubit_map,
        bindings,
        bind_body=False,
    )
    if power == 0:
        map_nested_controlled_u_results(op, resolved, qubit_map)
        return
    composed_controls = [*outer_control_indices, *resolved.control_phys]
    target_operands = [
        operand for operand in op.target_operands if operand.type.is_quantum()
    ]
    control_value = op.control_value if isinstance(op, ConcreteControlledU) else None
    with _parameter_probe_scope(emit_pass):
        analysis_block, analysis_bindings = _bind_prepared_controlled_body(
            emit_pass,
            resolved.block,
            op.param_operands,
            target_operands,
            bindings,
        )
        scalar_vector_broadcast = _is_single_target_block_vector_broadcast(
            analysis_block,
            target_operands,
        )
        skip_emission, body_profile = _controlled_body_emission_plan(
            emit_pass,
            analysis_block,
            analysis_bindings,
            power=power,
            scalar_vector_broadcast=scalar_vector_broadcast,
            target_indices=resolved.target_phys,
        )
        if skip_emission and body_profile is not None:
            _emit_zero_work_controlled_body_bookkeeping(
                emit_pass,
                circuit,
                analysis_block,
                composed_controls,
                resolved.target_phys,
                power,
                analysis_bindings,
                scalar_vector_broadcast=scalar_vector_broadcast,
                batch_profile=body_profile,
            )
    if skip_emission:
        map_nested_controlled_u_results(op, resolved, qubit_map)
        return

    block, local_bindings = _bind_prepared_controlled_body(
        emit_pass,
        resolved.block,
        op.param_operands,
        target_operands,
        bindings,
    )
    resolved = dataclasses.replace(
        resolved,
        block=block,
        local_bindings=local_bindings,
    )

    with bracket_control_value(
        emit_pass,
        circuit,
        resolved.control_phys,
        control_value,
    ):
        if scalar_vector_broadcast:
            _emit_single_target_block_per_vector_element(
                emit_pass,
                circuit,
                block,
                len(composed_controls),
                composed_controls,
                resolved.target_phys,
                power,
                resolved.local_bindings,
                batch_profile=body_profile,
            )
        else:
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
                        batch_profile=body_profile,
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
    to ``emit_crz`` for a single control and to the engine's
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
        circuit (Any): Engine circuit being emitted into.
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
            runtime-parametric and the engine's runtime parameter type
            does not support the required angle scaling (e.g. QURI
            Parts' ``Parameter``).
    """
    import qamomile.observable as qm_o
    from qamomile.circuit.transpiler.passes.emit_support.pauli_evolve_emission import (
        _resolve_gamma,
        is_zero_evolution_time,
        validate_hamiltonian_within_register,
        validate_hermitian_hamiltonian,
    )
    from qamomile.observable.hamiltonian import PAULI_TERM_ZERO_ATOL

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
        engine runtime-parameter expression. ``factor`` is the term-specific
        real scale: ``2 * coeff`` for a Pauli term's central RZ, or
        ``-constant`` for the identity-term phase.

        Args:
            factor (float): Real multiplier applied to ``gamma``.

        Returns:
            Any: ``factor * gamma`` — a Python ``float`` for concrete gamma,
                or an engine parameter expression for a runtime-parametric one.

        Raises:
            EmitError: If ``gamma`` is a runtime parameter whose engine type
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
                "gamma on this engine: its runtime parameter type does not "
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

    # Complete validation before the zero-time shortcut or any circuit
    # mutation. The later emission pass is intentionally a second traversal.
    validate_hermitian_hamiltonian(hamiltonian)
    constant = hamiltonian.constant

    if is_zero_evolution_time(gamma):
        _map_operand_result_groups([op.evolved_qubits], [qubit_indices], qubit_map)
        return

    for operators, coeff in hamiltonian:
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


def _resolve_power_if_bound(
    emit_pass: "StandardEmitPass",
    op: ControlledUOperation,
    bindings: dict[str, Any],
) -> int | None:
    """Resolve a controlled power when its current scope binds the value.

    Args:
        emit_pass (StandardEmitPass): Active emit pass and value resolver.
        op (ControlledUOperation): Controlled operation owning the power.
        bindings (dict[str, Any]): Emit-time bindings visible in the current
            scope.

    Returns:
        int | None: Validated nonnegative power, or ``None`` only when a
            symbolic value is not yet bound in this scope.

    Raises:
        EmitError: If the power has an unexpected type, is negative, or its
            value resolver reports a deterministic error.
    """
    power = op.power

    if isinstance(power, Value):
        resolved = emit_pass._resolver.resolve_classical_value(power, bindings)
        if resolved is None:
            return None
        candidate: object = resolved
    else:
        candidate = power
    try:
        return coerce_nonnegative_integral(
            candidate,
            label="ControlledU power",
        )
    except (TypeError, ValueError) as error:
        raise EmitError(
            str(error),
            operation="ControlledUOperation",
        ) from error


def resolve_power(
    emit_pass: "StandardEmitPass",
    op: ControlledUOperation,
    bindings: dict[str, Any],
) -> int:
    """Resolve ``ControlledUOperation.power`` to a concrete integer.

    Args:
        emit_pass (StandardEmitPass): Active emit pass and value resolver.
        op (ControlledUOperation): Controlled operation owning the power.
        bindings (dict[str, Any]): Emit-time bindings visible in the current
            scope.

    Returns:
        int: Validated nonnegative power.

    Raises:
        EmitError: If the power is unresolved, has an unexpected type, is
            negative, or its value resolver reports a deterministic error.
    """
    resolved_power = _resolve_power_if_bound(emit_pass, op, bindings)
    if resolved_power is None:
        power = op.power
        assert isinstance(power, Value)
        raise EmitError(
            f"Cannot resolve ControlledU power '{power.name}'. "
            f"Ensure all parameters are bound before transpilation.",
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
    ``append_gate`` pipeline (with the per-gate fallback for engines
    whose ``circuit_to_gate`` returns ``None``).

    Args:
        emit_pass (StandardEmitPass): The emit pass driving the
            conversion; provides ``_resolver``, ``_emitter``,
            ``_blockvalue_to_gate``, and ``_emit_controlled_fallback``.
        circuit (Any): The engine circuit being built.
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

    power_value = resolve_power(emit_pass, op, bindings)
    if power_value == 0:
        _map_symbolic_indexed_controlled_results(
            op,
            vector_size=vector_size,
            root_array=root_av,
            slice_start=slice_start,
            slice_step=slice_step,
            target_index_groups=target_index_groups,
            qubit_map=qubit_map,
        )
        return

    block_value = op.block
    if block_value is None:
        raise EmitError(
            "Cannot emit a nonzero ControlledUOperation without an inner block.",
            operation="ControlledUOperation",
        )
    with _parameter_probe_scope(emit_pass):
        analysis_block, analysis_bindings = _bind_prepared_controlled_body(
            emit_pass,
            block_value,
            param_operands,
            target_qubit_operands,
            bindings,
        )
        scalar_vector_broadcast = _is_single_target_block_vector_broadcast(
            analysis_block,
            target_qubit_operands,
        )
        skip_emission, body_profile = _controlled_body_emission_plan(
            emit_pass,
            analysis_block,
            analysis_bindings,
            power=power_value,
            scalar_vector_broadcast=scalar_vector_broadcast,
            target_indices=target_indices,
        )
        if skip_emission and body_profile is not None:
            _emit_zero_work_controlled_body_bookkeeping(
                emit_pass,
                circuit,
                analysis_block,
                control_phys,
                target_indices,
                power_value,
                analysis_bindings,
                scalar_vector_broadcast=scalar_vector_broadcast,
                batch_profile=body_profile,
            )

    if not skip_emission:
        block_value, local_bindings = _bind_prepared_controlled_body(
            emit_pass,
            block_value,
            param_operands,
            target_qubit_operands,
            bindings,
        )
        if scalar_vector_broadcast:
            _emit_single_target_block_per_vector_element(
                emit_pass,
                circuit,
                block_value,
                nc,
                control_phys,
                target_indices,
                power_value,
                local_bindings,
                batch_profile=body_profile,
            )
        else:
            unitary_gate = emit_pass._blockvalue_to_gate(
                block_value,
                len(target_indices),
                local_bindings,
            )
            if unitary_gate is not None:
                if power_value > 1:
                    unitary_gate = emit_pass._emitter.gate_power(
                        unitary_gate,
                        power_value,
                    )
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
                    batch_profile=body_profile,
                )

    _map_symbolic_indexed_controlled_results(
        op,
        vector_size=vector_size,
        root_array=root_av,
        slice_start=slice_start,
        slice_step=slice_step,
        target_index_groups=target_index_groups,
        qubit_map=qubit_map,
    )


def _map_symbolic_indexed_controlled_results(
    operation: SymbolicControlledU,
    *,
    vector_size: int,
    root_array: Any,
    slice_start: int,
    slice_step: int,
    target_index_groups: list[list[int]],
    qubit_map: QubitMap,
) -> None:
    """Map indexed-control results without inspecting the controlled body.

    Args:
        operation (SymbolicControlledU): Indexed-control operation.
        vector_size (int): Resolved control-pool size.
        root_array (Any): Root control array after slice resolution.
        slice_start (int): Root index of the first control-pool element.
        slice_step (int): Root stride between control-pool elements.
        target_index_groups (list[list[int]]): Physical target indices grouped
            by quantum operand.
        qubit_map (QubitMap): Logical-to-physical map, mutated in place.
    """
    from qamomile.circuit.ir.value import ArrayValue

    vector_result = operation.results[0]
    for index in range(vector_size):
        result_address = QubitAddress(vector_result.uuid, index)
        input_address = QubitAddress(
            root_array.uuid,
            slice_start + slice_step * index,
        )
        if input_address in qubit_map and result_address not in qubit_map:
            qubit_map[result_address] = qubit_map[input_address]

    target_results = [
        result for result in operation.results[1:] if result.type.is_quantum()
    ]
    for result, indices in zip(
        target_results,
        target_index_groups,
        strict=False,
    ):
        if isinstance(result, ArrayValue):
            map_array_result_group(result.uuid, indices, qubit_map)
        elif indices:
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
    per-gate fallback for engines whose ``circuit_to_gate`` returns
    ``None``).

    Args:
        emit_pass (StandardEmitPass): The emit pass driving the
            conversion.
        circuit (Any): The engine circuit being built.
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
    control_operands = op.control_operands
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

    power_value = resolve_power(emit_pass, op, bindings)
    if power_value == 0:
        _map_symbolic_multi_arg_controlled_results(
            op,
            control_index_groups=control_index_groups,
            target_index_groups=target_index_groups,
            qubit_map=qubit_map,
        )
        return

    block_value = op.block
    if block_value is None:
        raise EmitError(
            "Cannot emit a nonzero ControlledUOperation without an inner block.",
            operation="ControlledUOperation",
        )
    with _parameter_probe_scope(emit_pass):
        analysis_block, analysis_bindings = _bind_prepared_controlled_body(
            emit_pass,
            block_value,
            param_operands,
            target_qubit_operands,
            bindings,
        )
        scalar_vector_broadcast = _is_single_target_block_vector_broadcast(
            analysis_block,
            target_qubit_operands,
        )
        skip_emission, body_profile = _controlled_body_emission_plan(
            emit_pass,
            analysis_block,
            analysis_bindings,
            power=power_value,
            scalar_vector_broadcast=scalar_vector_broadcast,
            target_indices=target_indices,
        )
        if skip_emission and body_profile is not None:
            _emit_zero_work_controlled_body_bookkeeping(
                emit_pass,
                circuit,
                analysis_block,
                control_phys,
                target_indices,
                power_value,
                analysis_bindings,
                scalar_vector_broadcast=scalar_vector_broadcast,
                batch_profile=body_profile,
            )

    if not skip_emission:
        block_value, local_bindings = _bind_prepared_controlled_body(
            emit_pass,
            block_value,
            param_operands,
            target_qubit_operands,
            bindings,
        )
        if scalar_vector_broadcast:
            _emit_single_target_block_per_vector_element(
                emit_pass,
                circuit,
                block_value,
                nc,
                control_phys,
                target_indices,
                power_value,
                local_bindings,
                batch_profile=body_profile,
            )
        else:
            unitary_gate = emit_pass._blockvalue_to_gate(
                block_value,
                len(target_indices),
                local_bindings,
            )
            if unitary_gate is not None:
                if power_value > 1:
                    unitary_gate = emit_pass._emitter.gate_power(
                        unitary_gate,
                        power_value,
                    )
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
                    batch_profile=body_profile,
                )

    _map_symbolic_multi_arg_controlled_results(
        op,
        control_index_groups=control_index_groups,
        target_index_groups=target_index_groups,
        qubit_map=qubit_map,
    )


def _map_symbolic_multi_arg_controlled_results(
    operation: SymbolicControlledU,
    *,
    control_index_groups: list[list[int]],
    target_index_groups: list[list[int]],
    qubit_map: QubitMap,
) -> None:
    """Map multi-argument controlled results without inspecting its body.

    Args:
        operation (SymbolicControlledU): Multi-argument controlled operation.
        control_index_groups (list[list[int]]): Physical indices grouped by
            control operand.
        target_index_groups (list[list[int]]): Physical indices grouped by
            target operand.
        qubit_map (QubitMap): Logical-to-physical map, mutated in place.
    """
    from qamomile.circuit.ir.value import ArrayValue

    control_results: list[Any] = []
    control_result_groups: list[list[int]] = []
    for result, group in zip(
        operation.results[: operation.num_control_args],
        control_index_groups,
        strict=False,
    ):
        if result.type.is_quantum():
            control_results.append(result)
            control_result_groups.append(group)
    _map_operand_result_groups(
        control_results,
        control_result_groups,
        qubit_map,
    )

    target_results = [
        result
        for result in operation.results[operation.num_control_args :]
        if result.type.is_quantum()
    ]
    for result, indices in zip(
        target_results,
        target_index_groups,
        strict=False,
    ):
        if isinstance(result, ArrayValue):
            map_array_result_group(result.uuid, indices, qubit_map)
        elif indices:
            qubit_map[QubitAddress(result.uuid)] = indices[0]


def emit_controlled_u(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: ControlledUOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a controlled operation for every concrete operand layout.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Engine circuit being emitted into.
        op (ControlledUOperation): Concrete or symbolic controlled operation.
        qubit_map (QubitMap): Logical-to-physical map, mutated with results.
        bindings (dict[str, Any]): Bindings visible at the call site.

    Raises:
        EmitError: If controls, targets, power, or body parameters cannot be
            resolved, or the engine cannot lower the controlled operation.
    """
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

    if block_value is None:
        raise EmitError(
            "Cannot emit a nonzero ControlledUOperation without an inner block.",
            operation="ControlledUOperation",
        )

    with _parameter_probe_scope(emit_pass):
        analysis_block, analysis_bindings = _bind_prepared_controlled_body(
            emit_pass,
            block_value,
            param_operands,
            target_qubit_operands,
            bindings,
        )
        scalar_vector_broadcast = _is_single_target_block_vector_broadcast(
            analysis_block,
            target_qubit_operands,
        )
        skip_emission, body_profile = _controlled_body_emission_plan(
            emit_pass,
            analysis_block,
            analysis_bindings,
            power=power_value,
            scalar_vector_broadcast=scalar_vector_broadcast,
            target_indices=target_indices,
        )
        if skip_emission and body_profile is not None:
            _emit_zero_work_controlled_body_bookkeeping(
                emit_pass,
                circuit,
                analysis_block,
                control_indices,
                target_indices,
                power_value,
                analysis_bindings,
                scalar_vector_broadcast=scalar_vector_broadcast,
                batch_profile=body_profile,
            )
    if skip_emission:
        _map_controlled_u_results(
            op,
            nc,
            control_indices,
            target_qubit_operands,
            target_index_groups,
            qubit_map,
        )
        return

    block_value, local_bindings = _bind_prepared_controlled_body(
        emit_pass,
        block_value,
        param_operands,
        target_qubit_operands,
        bindings,
    )

    with bracket_control_value(
        emit_pass,
        circuit,
        control_indices,
        op.control_value,
    ):
        if scalar_vector_broadcast:
            _emit_single_target_block_per_vector_element(
                emit_pass,
                circuit,
                block_value,
                nc,
                control_indices,
                target_indices,
                power_value,
                local_bindings,
                batch_profile=body_profile,
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
                    batch_profile=body_profile,
                )

    _map_controlled_u_results(
        op, nc, control_indices, target_qubit_operands, target_index_groups, qubit_map
    )


def _is_single_target_block_vector_broadcast(
    block_value: Any,
    target_qubit_operands: list[Any],
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

    Returns:
        bool: ``True`` when a single scalar formal target is being
            broadcast over a vector actual target.
    """
    from qamomile.circuit.ir.value import ArrayValue

    if len(target_qubit_operands) != 1:
        return False
    if not isinstance(target_qubit_operands[0], ArrayValue):
        return False
    quantum_inputs = [
        input_value
        for input_value in getattr(block_value, "input_values", [])
        if hasattr(input_value, "type") and input_value.type.is_quantum()
    ]
    return len(quantum_inputs) == 1 and not isinstance(quantum_inputs[0], ArrayValue)


def _fallback_batch_profile_if_needed(
    emit_pass: "StandardEmitPass",
    block_value: Any,
    num_controls: int,
    bindings: dict[str, Any],
    batch_profile: ControlBatchProfile | None,
) -> ControlBatchProfile | None:
    """Resolve a profile only when shared-ladder fallback can consume it.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        block_value (Any): Controlled block being lowered by fallback.
        num_controls (int): Concrete number of physical controls.
        bindings (dict[str, Any]): Bindings visible inside the block.
        batch_profile (ControlBatchProfile | None): Previously computed
            profile, if one is already available.

    Returns:
        ControlBatchProfile | None: Existing or newly resolved profile when a
            shared ladder is possible, otherwise None.
    """
    if (
        batch_profile is None
        and num_controls >= 2
        and getattr(emit_pass, "_mc_ancilla_pool", None) is not None
    ):
        return _controlled_body_batch_profile(
            emit_pass,
            block_value.operations,
            bindings,
        )
    return batch_profile


def _emit_single_target_block_per_vector_element(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    block_value: Any,
    num_controls: int,
    control_indices: list[int],
    target_indices: list[int],
    power: int,
    bindings: dict[str, Any],
    batch_profile: ControlBatchProfile | None = None,
) -> None:
    """Emit a scalar controlled-U once for each vector target element.

    Args:
        emit_pass (StandardEmitPass): Driving emit pass.
        circuit (Any): Engine circuit being emitted.
        block_value (Any): Single-target inner block.
        num_controls (int): Number of control qubits.
        control_indices (list[int]): Physical control qubits.
        target_indices (list[int]): Physical target qubits to receive
            the broadcasted controlled operation.
        power (int): Nonnegative controlled-U power.
        bindings (dict[str, Any]): Local bindings for the inner block.
        batch_profile (ControlBatchProfile | None): Previously resolved body
            profile for this exact block and binding scope. Defaults to
            ``None``, which lets each fallback resolve it.

    Raises:
        EmitError: If the engine cannot convert the block to a gate and
            the fallback controlled decomposition does not support the
            block shape.
    """
    if power == 0 or not target_indices:
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

    batch_profile = _fallback_batch_profile_if_needed(
        emit_pass,
        block_value,
        num_controls,
        bindings,
        batch_profile,
    )
    for target_idx in target_indices:
        emit_pass._emit_controlled_fallback(
            circuit,
            block_value,
            num_controls,
            control_indices,
            [target_idx],
            power,
            bindings,
            batch_profile=batch_profile,
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
    batch_profile: ControlBatchProfile | None = None,
) -> None:
    """Fallback emission for controlled-U when gate conversion fails.

    Decomposes the block body gate-by-gate under the full control set.
    A block-local qubit map ties the block's formal quantum inputs to
    ``target_indices``, so every inner gate is emitted on the physical
    qubit its operand resolves to — multi-target inner blocks are
    supported. Nested ``ControlledUOperation``s compose their controls
    with the outer ones; irreducible multi-controlled single-qubit
    gates route through the engine's
    ``_emit_irreducible_multi_controlled_gate`` hook. Subclasses may
    still override this method to emit controlled blocks natively
    (e.g. CUDA-Q's ``cudaq.control`` helper kernels).

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Engine circuit being built.
        block_value (Any): Block whose operations should be controlled.
        num_controls (int): Number of control qubits.
        control_indices (list[int]): Physical indices of control qubits.
        target_indices (list[int]): Physical indices of target qubits.
        power (int): Number of controlled body repetitions.
        bindings (dict[str, Any]): Parameter bindings visible in the body.
        batch_profile (ControlBatchProfile | None): Previously resolved body
            profile for this exact block and binding scope. Defaults to
            ``None``, which lets the controlled walker compute it.

    Raises:
        EmitError: If ``num_controls`` disagrees with
            ``control_indices``, the block's quantum inputs cannot be
            mapped onto ``target_indices``, or an inner operation
            cannot be lowered under the accumulated controls (e.g. an
            irreducible multi-controlled gate on an engine without the
            multi-control hook).
    """
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
    # Only the shared-ladder fallback needs this semantic walk. Resolve it
    # after reusable-gate attempts have failed and once per repeated call.
    batch_profile = _fallback_batch_profile_if_needed(
        emit_pass,
        block_value,
        num_controls,
        bindings,
        batch_profile,
    )
    for _ in range(power):
        emit_controlled_operations(
            emit_pass,
            circuit,
            block_value.operations,
            control_indices,
            qubit_map,
            bindings,
            batch_profile=batch_profile,
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
        circuit (Any): Engine circuit being emitted into.
        op (Any): Composite gate operation.
        impl (Any): Fallback implementation block to emit.
        qubit_indices (list[int]): Physical qubits for the operation.
        bindings (dict[str, Any]): Active emit bindings.

    Raises:
        EmitError: If body inputs, workspace addresses, or engine gate
            operands cannot be resolved safely.
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
        input_qubit_map: QubitMap = {}
        local_clbit_map: ClbitMap = {}
        local_bindings = _bind_and_populate_block_inputs(
            emit_pass,
            impl,
            op.operands,
            num_qubits,
            bindings,
            input_qubit_map,
            parent_qubits=qubit_indices,
            operation_name="InvokeOperation",
        )
        local_qubit_map: QubitMap = dict(
            getattr(emit_pass, "_active_qubit_map", None) or {}
        )
        local_qubit_map.update(input_qubit_map)

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
        circuit (Any): Engine circuit being emitted into.
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
        ValueError: If the selected implementation body disagrees with the
            invocation contract.
    """
    selection = _controlled_invoke_selection(
        op,
        getattr(emit_pass, "engine_name", None),
    )
    body = selection.body
    if body is not None:
        with _parameter_probe_scope(emit_pass):
            local_bindings = _bind_block_inputs(
                emit_pass,
                body,
                list(selection.operands),
                bindings,
            )
            body_profile = _controlled_body_batch_profile(
                emit_pass,
                body.operations,
                local_bindings,
            )
        if body_profile.weight == 0:
            # Zero quantum work is not permission to discard bookkeeping:
            # validate deferred returns and propagate aliases/classical values
            # through the structural fallback. Run that walk in the same
            # isolated transaction used by zero-work ControlledU bodies so it
            # cannot create runtime ABI entries or append a nested native gate
            # to the real circuit.
            with _zero_work_analysis_scope(
                emit_pass,
                control_indices,
                qubit_indices,
                bindings,
            ) as analysis_circuit:
                _emit_all_ones_controlled_composite_at_indices(
                    emit_pass,
                    analysis_circuit,
                    op,
                    control_indices,
                    qubit_indices,
                    bindings,
                    structural_only=True,
                    batch_profile=body_profile,
                )
            return

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
    *,
    structural_only: bool = False,
    batch_profile: ControlBatchProfile | None = None,
) -> None:
    """Emit an invocation after activation controls have been normalized.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Engine circuit being emitted into.
        op (InvokeOperation): Composite or oracle invocation to emit.
        control_indices (list[int]): Physical outer control qubits.
        qubit_indices (list[int]): Physical qubits occupied by ``op``'s own
            control and target operands.
        bindings (dict[str, Any]): Active emit bindings.
        structural_only (bool): Whether to skip reusable/native gate
            conversion and execute only the fallback body's bookkeeping.
            Defaults to False.
        batch_profile (ControlBatchProfile | None): Previously resolved body
            profile. Defaults to None.

    Returns:
        None.

    Raises:
        EmitError: If the composite has no implementation block or the
            fallback cannot represent the controlled composite.
        ValueError: If the selected implementation body disagrees with the
            invocation contract.
    """
    selection = _controlled_invoke_selection(
        op,
        getattr(emit_pass, "engine_name", None),
    )
    impl = selection.body
    body_implements_transform = selection.realized_transform is op.transform
    if impl is not None:
        if body_implements_transform and not control_indices and not structural_only:
            emit_pass._emit_custom_composite(
                circuit,
                op,
                impl,
                qubit_indices,
                bindings,
            )
            return
        if body_implements_transform:
            # A transform-specific body already implements the invocation's
            # own control/inverse semantics. Treat all of its qubits as body
            # targets and apply only controls accumulated from enclosing blocks.
            body_qubits = qubit_indices
            body_operands = list(selection.operands)
            all_controls = list(control_indices)
        else:
            # An INVERSE body selected for CONTROLLED_INVERSE already realizes
            # only the inverse component. Apply the invocation's own coherent
            # controls, plus any controls accumulated from enclosing blocks.
            own_control_count = op.num_body_external_control_qubits
            own_controls = qubit_indices[:own_control_count]
            body_qubits = qubit_indices[own_control_count:]
            body_operands = list(selection.operands)
            all_controls = [*control_indices, *own_controls]
    elif op.transform.is_inverse:
        raise EmitError(
            f"Inverse callable '{op.target.name}' has no inverse "
            "implementation body for this engine. Bind structural "
            "parameters at compile time so the inverse can be "
            "materialized, or register an inverse implementation.",
            operation=f"InvokeOperation[{op.target.name}]",
        )
    else:
        own_control_count = op.num_control_qubits if op.transform.is_controlled else 0
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
    if not structural_only:
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
    if structural_only:
        emit_controlled_fallback(
            emit_pass,
            circuit,
            impl,
            len(all_controls),
            all_controls,
            body_qubits,
            1,
            local_bindings,
            batch_profile=batch_profile,
        )
        return
    emit_pass._emit_controlled_fallback(
        circuit,
        impl,
        len(all_controls),
        all_controls,
        body_qubits,
        1,
        local_bindings,
        batch_profile=batch_profile,
    )
