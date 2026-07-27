"""Prepare reusable controlled blocks and map their quantum operands."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, cast

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import (
    Operation,
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.gate import ConcreteControlledU
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.block_parameter_binding import (
    block_parameter_binding_keys,
    pair_block_parameter_operands,
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.physical_index_map import (
    map_array_result_group,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    ClbitMap,
    QubitAddress,
    QubitMap,
)

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
        ValueResolver,
    )
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass


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

        # The nested allocate() below recomputes the allocator's
        # segment-level analysis state (measurement taint, safe-merge
        # allowlist, counters) for the SUB-block; snapshot it so later
        # iteration replays of the enclosing segment keep consulting
        # the segment's own sets.
        with emit_pass._allocator.preserving_analysis_state():
            local_qubit_map, local_clbit_map = emit_pass._allocator.allocate(
                block_value.operations,
                local_bindings,
                initial_qubit_map=local_qubit_map,
                initial_clbit_map=local_clbit_map,
            )

            qubit_count = (
                max(local_qubit_map.values()) + 1 if local_qubit_map else num_qubits
            )
            if qubit_count != num_qubits:
                # A backend gate cannot allocate private wires when it is
                # appended to its parent circuit: every wire in the gate
                # arity must be supplied by the call site.  Nested QInit
                # operations can make the temporary circuit wider than the
                # callable's declared quantum inputs.  Boxing that wider
                # circuit used to produce a ReusableCircuit whose arity did
                # not match its CallInstruction inputs, so circuit-IR
                # verification crashed only after the malformed call had
                # escaped this boundary.  Return ``None`` and let the
                # controlled-operation walker either lower the body safely or
                # reject the unsupported allocation with an EmitError.
                return None
            sub_circuit = emit_pass._emitter.create_circuit(qubit_count, 0)

            # The segment ancilla pool addresses the parent circuit; suspend
            # it so a multi-controlled gate inside this block cannot index it
            # against the narrower sub-circuit. If one is present, the shared
            # cascade raises EmitError (caught below) and the caller falls
            # back to gate-by-gate emission on the parent circuit.
            with emit_pass._suspended_mc_ancilla_pool():
                emit_pass._emit_operations(
                    sub_circuit,
                    block_value.operations,
                    local_qubit_map,
                    local_clbit_map,
                    local_bindings,
                    force_unroll=True,
                )

        return emit_pass._emitter.circuit_to_gate(sub_circuit, "U")

    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        EmitError,
    ):
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
    local_bindings = bindings.copy()
    if input_operands is None or not hasattr(block_value, "input_values"):
        return local_bindings

    for formal in block_value.input_values:
        if not (formal.type.is_classical() or formal.type.is_object()):
            continue
        for key in block_parameter_binding_keys(formal):
            local_bindings.pop(key, None)
    param_operands = [
        cast(Value, operand)
        for operand in input_operands
        if hasattr(operand, "type")
        and (operand.type.is_classical() or operand.type.is_object())
    ]
    for formal, actual in pair_block_parameter_operands(
        block_value,
        param_operands,
    ):
        inner_keys = block_parameter_binding_keys(formal)
        resolved = _resolve_call_operand(
            emit_pass,
            actual,
            bindings,
        )
        for key in inner_keys:
            local_bindings[key] = resolved

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
        emit_pass._resolver,
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
    return [
        operand
        for operand in input_operands
        if hasattr(operand, "type") and operand.type.is_quantum()
    ]


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
        ValidationError: If ``SliceBorrowCheckPass`` rejects the block stage
            or cannot represent nested ownership across control flow.
        QubitBorrowConflictError: If nested slice views have overlapping live
            ownership.
        QubitConsumedError: If the nested block accesses a qubit slot after a
            destructive operation consumed it.
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
        EmitError: Five failure modes are surfaced loudly rather than
            silently mis-mapping qubits:

            * A quantum input is a rank>1 register (its ``ArrayValue``
              has more than one shape dimension). Elements are keyed as
              ``QubitAddress(uuid, i)`` with a single flat index, so a
              higher-rank register would silently alias distinct
              elements onto the same physical qubit. The frontend
              rejects such registers at construction time; this guard
              covers hand-built or deserialized IR.
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

    # Rank guard in its own loop, before any length resolution: the
    # element addressing below keys each element as
    # ``QubitAddress(uuid, i)`` with a single flat index, so a rank>1
    # register cannot be mapped without aliasing distinct elements onto
    # the same physical qubit.
    for iv in vector_inputs:
        if len(iv.shape) > 1:
            raise EmitError(
                f"Inner block input {iv.name!r} is a rank-{len(iv.shape)} "
                f"quantum register: the qubit addressing path is rank-1, "
                f"so a higher-rank register would silently alias distinct "
                f"elements onto the same physical qubit. Use a 1-D "
                f"Vector[Qubit] with explicit index arithmetic instead.",
                operation="ControlledUOperation",
            )

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
    ``Vector[Qubit]`` sub-kernel arguments without duplicating the
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
    resolver: "ValueResolver",
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
        resolver (ValueResolver): Emit value resolver used to resolve
            actual operand sizes.
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
        actual_size = resolver.resolve_int_value(actual.shape[0], bindings)
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

    The frontend result layout mirrors the operand layout:

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
