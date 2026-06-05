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

import math
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import ForOperation, HasNestedOps
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import EmitError
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
    """Emit controlled version of a block."""
    if not hasattr(block_value, "operations"):
        return

    emit_controlled_operations(
        emit_pass,
        circuit,
        block_value.operations,
        control_idx,
        target_indices,
        bindings,
    )


def emit_controlled_operations(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    operations: list[Operation],
    control_idx: int,
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit controlled versions of operations."""
    for op in operations:
        if isinstance(op, GateOperation):
            emit_controlled_gate(
                emit_pass, circuit, op, control_idx, target_indices, bindings
            )
        elif isinstance(op, ControlledUOperation):
            raise EmitError(
                "Cannot decompose nested ControlledUOperation in a controlled "
                "block on this backend: the fallback path controls each "
                "top-level inner operation with a single outer control and "
                "cannot preserve the nested operation's own controls. Use a "
                "backend that can convert the inner block to a native "
                "controlled gate, or flatten the controls explicitly.",
                operation="ControlledUOperation",
            )
        elif isinstance(op, ForOperation):
            start = (
                emit_pass._resolver.resolve_int_value(op.operands[0], bindings)
                if len(op.operands) > 0
                else 0
            )
            stop = (
                emit_pass._resolver.resolve_int_value(op.operands[1], bindings)
                if len(op.operands) > 1
                else 1
            )
            step = (
                emit_pass._resolver.resolve_int_value(op.operands[2], bindings)
                if len(op.operands) > 2
                else 1
            )

            if start is not None and stop is not None and step is not None:
                from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
                    _bind_loop_var,
                )

                for i in range(start, stop, step):
                    loop_bindings = bindings.copy()
                    _bind_loop_var(loop_bindings, op, i)
                    emit_controlled_operations(
                        emit_pass,
                        circuit,
                        op.operations,
                        control_idx,
                        target_indices,
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
            for j, phys in enumerate(indices):
                qubit_map[QubitAddress(result.uuid, j)] = phys
            if indices:
                base_addr = QubitAddress(result.uuid)
                if base_addr not in qubit_map:
                    qubit_map[base_addr] = indices[0]
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
    control_phys: list[int] = []
    for q in control_operands:
        indices = _expand_quantum_operands_to_phys(emit_pass, q, qubit_map, bindings)
        control_phys.extend(indices)

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

    control_results = op.results[: op.num_control_args]
    for src, dst in zip(control_operands, control_results):
        if isinstance(src, _ArrayValue):
            for addr, idx in list(qubit_map.items()):
                if addr.matches_array(src.uuid):
                    result_addr = QubitAddress(dst.uuid, addr.element_index)
                    if result_addr not in qubit_map:
                        qubit_map[result_addr] = idx
        else:
            src_addr = QubitAddress(src.uuid)
            if src_addr in qubit_map:
                dst_addr = QubitAddress(dst.uuid)
                if dst_addr not in qubit_map:
                    qubit_map[dst_addr] = qubit_map[src_addr]

    sub_quantum_results = [
        r for r in op.results[op.num_control_args :] if r.type.is_quantum()
    ]
    for i, result in enumerate(sub_quantum_results):
        if i >= len(target_index_groups):
            break
        indices = target_index_groups[i]
        if isinstance(result, _ArrayValue):
            for j, phys in enumerate(indices):
                qubit_map[QubitAddress(result.uuid, j)] = phys
            if indices:
                base_addr = QubitAddress(result.uuid)
                if base_addr not in qubit_map:
                    qubit_map[base_addr] = indices[0]
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

    Decomposes the block body gate-by-gate with single-control emission.
    Subclasses may override to support multi-control natively.

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
        EmitError: When ``num_controls > 1`` (multi-control not
            supported in the default gate-by-gate decomposition), when
            the inner block contains a nested ``ControlledUOperation``,
            or when the inner block addresses more than a single target
            and is not the narrowly-supported "exactly one SWAP op"
            case.  ``emit_controlled_gate`` only carries a single
            ``target_idx = target_indices[0]`` mapping, so any inner
            gate that should land on ``target_indices[i > 0]`` would
            silently route to slot 0 instead.  SWAP is the lone
            exception because the SWAP branch in
            ``emit_controlled_gate`` explicitly reads
            ``target_indices[0]`` *and* ``target_indices[1]``.
            Subclasses that carry a proper operand-to-target mapping
            (the CUDA-Q transpiler's
            ``_build_block_qubit_map`` override) bypass this check by
            replacing the method outright.
    """
    if num_controls > 1:
        raise EmitError(
            f"Cannot decompose multi-controlled operation "
            f"(num_controls={num_controls}): "
            f"block-to-gate conversion failed and the "
            f"fallback decomposition only supports single control.",
            operation="ControlledUOperation",
        )
    nested_controlled = [
        o for o in block_value.operations if isinstance(o, ControlledUOperation)
    ]
    if nested_controlled:
        raise EmitError(
            "Cannot decompose controlled-U with nested ControlledUOperation "
            "on this backend: the fallback path controls each top-level "
            "inner operation with a single outer control and cannot preserve "
            "the nested operation's own controls.",
            operation="ControlledUOperation",
        )
    if len(target_indices) > 1:
        # Walk the inner block's top-level gate operations.  Per-gate
        # decomposition only routes correctly to ``target_indices[0]``,
        # except for the SWAP branch which special-cases
        # ``target_indices[1]``.  Any other shape is a silent
        # miscompile waiting to fire.
        inner_gates = [
            o for o in block_value.operations if isinstance(o, GateOperation)
        ]
        is_single_swap = (
            len(inner_gates) == 1 and inner_gates[0].gate_type == GateOperationType.SWAP
        )
        if not is_single_swap:
            raise EmitError(
                f"Cannot decompose controlled-U with multi-target inner block "
                f"(target_indices={target_indices}, block has "
                f"{len(inner_gates)} gate op(s)) on this backend: the "
                f"per-gate fallback only routes each inner op to "
                f"``target_indices[0]``, so any gate intended for "
                f"``target_indices[i > 0]`` would silently miscompile to "
                f"slot 0.  Either run this kernel on a backend whose "
                f"``circuit_to_gate`` succeeds (Qiskit produces a native "
                f"controlled custom gate), or rewrite the wrapped sub-kernel "
                f"to take individual ``Qubit`` arguments and apply the "
                f"control element-by-element at the call site.",
                operation="ControlledUOperation",
            )
    for _ in range(power):
        for ctrl_idx in control_indices:
            emit_controlled_block(
                emit_pass,
                circuit,
                block_value,
                ctrl_idx,
                target_indices,
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
    """Emit a custom composite gate with implementation."""
    num_qubits = len(qubit_indices)
    custom_gate = emit_pass._blockvalue_to_gate(impl, num_qubits, bindings)

    if custom_gate is not None:
        emit_pass._emitter.append_gate(circuit, custom_gate, qubit_indices)
    else:
        local_qubit_map: QubitMap = {}
        local_clbit_map: ClbitMap = {}

        if hasattr(impl, "input_values"):
            for i, input_val in enumerate(impl.input_values):
                if i < len(qubit_indices):
                    local_qubit_map[QubitAddress(input_val.uuid)] = qubit_indices[i]

        if hasattr(impl, "operations"):
            emit_pass._emit_operations(
                circuit,
                impl.operations,
                local_qubit_map,
                local_clbit_map,
                bindings,
                force_unroll=True,
            )


def blockvalue_to_gate(
    emit_pass: "StandardEmitPass",
    block_value: Any,
    num_qubits: int,
    bindings: dict[str, Any],
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

    Returns:
        Any: A backend gate object produced by
            ``emit_pass._emitter.circuit_to_gate``, or ``None`` when the
            conversion is unable to proceed (missing ``operations``,
            allocator / emitter exception, etc.) so the caller can fall
            back to gate-by-gate emission.
    """
    if not hasattr(block_value, "operations"):
        return None

    block_value = _strip_slice_markers_for_nested_emit(block_value)

    try:
        local_qubit_map: QubitMap = {}
        local_clbit_map: ClbitMap = {}

        if hasattr(block_value, "input_values"):
            _populate_input_qubit_map(
                emit_pass,
                block_value.input_values,
                num_qubits,
                bindings,
                local_qubit_map,
            )

        local_qubit_map, local_clbit_map = emit_pass._allocator.allocate(
            block_value.operations,
            bindings,
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
            bindings,
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


def _strip_slice_markers_for_nested_emit(block_value: Any) -> Any:
    """Remove slice marker operations from a nested block before emit.

    Top-level blocks pass through ``SliceBorrowCheckPass`` and
    ``StripSliceArrayOpsPass`` before segmentation and emission, but a
    ``ControlledUOperation`` or custom composite carries its inner
    ``Block`` as an operation field rather than as regular control-flow
    children. Generic pass visitors therefore do not descend into that
    nested block. Apply the same marker-stripping invariant here before
    converting the nested block into a backend gate.

    Args:
        block_value (Any): Candidate nested block to normalize.

    Returns:
        Any: ``block_value`` with ``SliceArrayOperation`` and
        ``ReleaseSliceViewOperation`` markers removed when it is a
        ``Block``; otherwise the original object unchanged.
    """
    if not isinstance(block_value, Block):
        return block_value

    from qamomile.circuit.transpiler.passes.strip_slice_ops import (
        StripSliceArrayOpsPass,
    )

    return StripSliceArrayOpsPass().run(block_value)


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
    from qamomile.circuit.ir.value import ArrayValue

    control_results = op.results[:num_controls]
    for i, result in enumerate(control_results):
        if i < len(control_indices):
            qubit_map[QubitAddress(result.uuid)] = control_indices[i]

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
            for j, phys in enumerate(indices):
                qubit_map[QubitAddress(result.uuid, j)] = phys
            if indices:
                base_addr = QubitAddress(result.uuid)
                if base_addr not in qubit_map:
                    qubit_map[base_addr] = indices[0]
        else:
            if indices:
                qubit_map[QubitAddress(result.uuid)] = indices[0]
