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

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import ForOperation, HasNestedOps
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    IndexSpecControlledU,
    SymbolicControlledU,
)
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
                for i in range(start, stop, step):
                    loop_bindings = bindings.copy()
                    loop_bindings[op.loop_var] = i
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


def emit_controlled_u_with_index_spec(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: ControlledUOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a ControlledUOperation with target_indices/controlled_indices."""
    # 1. Resolve num_controls
    nc: int
    if op.is_symbolic_num_controls:
        resolved_nc = emit_pass._resolver.resolve_classical_value(
            op.num_controls,
            bindings,  # type: ignore[arg-type]
        )
        if resolved_nc is None:
            raise EmitError(
                "Cannot resolve symbolic num_controls for index_spec emit.",
                operation="ControlledUOperation",
            )
        nc = int(resolved_nc)
    else:
        nc = int(op.num_controls)

    # 2. Resolve index lists
    resolved_ti: list[int] | None = None
    resolved_ci: list[int] | None = None
    if op.target_indices is not None:
        resolved_ti = []
        for v in op.target_indices:
            val = emit_pass._resolver.resolve_classical_value(v, bindings)
            if val is None:
                raise EmitError(
                    "Cannot resolve target index.",
                    operation="ControlledUOperation",
                )
            resolved_ti.append(int(val))
    elif op.controlled_indices is not None:
        resolved_ci = []
        for v in op.controlled_indices:
            val = emit_pass._resolver.resolve_classical_value(v, bindings)
            if val is None:
                raise EmitError(
                    "Cannot resolve controlled index.",
                    operation="ControlledUOperation",
                )
            resolved_ci.append(int(val))
    else:
        raise EmitError(
            "ControlledUOperation with index_spec must have "
            "target_indices or controlled_indices.",
            operation="ControlledUOperation",
        )

    # 3. Get Vector's physical qubit indices
    from qamomile.circuit.ir.value import ArrayValue as _ArrayValue

    vector_value = op.operands[0]
    if not isinstance(vector_value, _ArrayValue):
        raise EmitError(
            "ControlledUOperation with index_spec expects ArrayValue operand.",
            operation="ControlledUOperation",
        )
    size_val = vector_value.shape[0]
    vector_size = emit_pass._resolver.resolve_int_value(size_val, bindings)
    if vector_size is None:
        raise EmitError(
            "Cannot resolve Vector size for index_spec emit.",
            operation="ControlledUOperation",
        )

    all_phys_indices = []
    for i in range(vector_size):
        qubit_addr = QubitAddress(vector_value.uuid, i)
        if qubit_addr in qubit_map:
            all_phys_indices.append(qubit_map[qubit_addr])
        else:
            raise EmitError(
                f"Qubit {str(qubit_addr)} not found in qubit_map.",
                operation="ControlledUOperation",
            )

    # 4. Validate resolved indices
    # One of resolved_ti or resolved_ci is guaranteed non-None by the branch above
    resolved_indices: list[int] = (
        resolved_ti if resolved_ti is not None else resolved_ci
    )  # type: ignore[assignment]
    for idx in resolved_indices:
        if not (0 <= idx < vector_size):
            raise EmitError(
                f"Index {idx} out of bounds [0, {vector_size}) for ControlledU vector.",
                operation="ControlledUOperation",
            )
    if len(set(resolved_indices)) != len(resolved_indices):
        raise EmitError(
            f"Duplicate indices in ControlledU index spec: {resolved_indices}.",
            operation="ControlledUOperation",
        )

    # 5. Partition into control and target physical indices
    if resolved_ti is not None:
        target_set = set(resolved_ti)
        target_phys = [all_phys_indices[i] for i in resolved_ti]
        control_phys = [
            all_phys_indices[i] for i in range(vector_size) if i not in target_set
        ]
    else:
        assert resolved_ci is not None
        control_set = set(resolved_ci)
        control_phys = [all_phys_indices[i] for i in resolved_ci]
        target_phys = [
            all_phys_indices[i] for i in range(vector_size) if i not in control_set
        ]

    # 6. Validate num_controls consistency
    if len(control_phys) != nc:
        raise EmitError(
            f"num_controls ({nc}) does not match actual control count "
            f"({len(control_phys)}).",
            operation="ControlledUOperation",
        )

    # 7. Bind classical parameters
    block_value = op.block
    param_operands = [v for v in op.operands[1:] if v.type.is_classical()]
    local_bindings = emit_pass._resolver.bind_block_params(
        block_value, param_operands, bindings
    )

    # 8. Build and emit gate
    num_targets = len(target_phys)
    unitary_gate = emit_pass._blockvalue_to_gate(
        block_value, num_targets, local_bindings
    )

    power_value = resolve_power(emit_pass, op, bindings)

    if unitary_gate is not None:
        if power_value > 1:
            unitary_gate = emit_pass._emitter.gate_power(unitary_gate, power_value)
        controlled_gate = emit_pass._emitter.gate_controlled(unitary_gate, nc)
        emit_pass._emitter.append_gate(
            circuit, controlled_gate, control_phys + target_phys
        )
    else:
        emit_pass._emit_controlled_fallback(
            circuit,
            block_value,
            nc,
            control_phys,
            target_phys,
            power_value,
            local_bindings,
        )

    # 9. Map result ArrayValue in qubit_map
    vector_result = op.results[0]
    for i in range(vector_size):
        result_addr = QubitAddress(vector_result.uuid, i)
        input_addr = QubitAddress(vector_value.uuid, i)
        if input_addr in qubit_map and result_addr not in qubit_map:
            qubit_map[result_addr] = qubit_map[input_addr]


def emit_controlled_u(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: ControlledUOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a ControlledUOperation."""
    if isinstance(op, IndexSpecControlledU):
        emit_controlled_u_with_index_spec(emit_pass, circuit, op, qubit_map, bindings)
        return
    if isinstance(op, SymbolicControlledU):
        raise EmitError(
            "Cannot emit ControlledUOperation with symbolic num_controls. "
            "Bind parameters to concrete values before transpilation.",
            operation="ControlledUOperation",
        )
    assert isinstance(op, ConcreteControlledU)
    nc: int = op.num_controls
    block_value = op.block
    control_operands = op.control_operands
    remaining_operands = op.operands[nc:]

    target_qubit_operands = [v for v in remaining_operands if v.type.is_quantum()]
    param_operands = [v for v in remaining_operands if v.type.is_classical()]

    control_indices = []
    for q in control_operands:
        idx = emit_pass._resolver.resolve_qubit_index(q, qubit_map, bindings)
        if idx is not None:
            control_indices.append(idx)

    target_indices = []
    for q in target_qubit_operands:
        idx = emit_pass._resolver.resolve_qubit_index(q, qubit_map, bindings)
        if idx is not None:
            target_indices.append(idx)

    if not control_indices or not target_indices:
        return

    local_bindings = emit_pass._resolver.bind_block_params(
        block_value, param_operands, bindings
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

    all_input_indices = control_indices + target_indices
    for i, result in enumerate(op.results):
        if i < len(all_input_indices):
            qubit_map[QubitAddress(result.uuid)] = all_input_indices[i]


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
        EmitError: When num_controls > 1 (multi-control not supported
            in the default gate-by-gate decomposition).
    """
    if num_controls > 1:
        raise EmitError(
            f"Cannot decompose multi-controlled operation "
            f"(num_controls={num_controls}): "
            f"block-to-gate conversion failed and the "
            f"fallback decomposition only supports single control.",
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
    """Convert a Block to a backend gate."""
    if not hasattr(block_value, "operations"):
        return None

    try:
        local_qubit_map: QubitMap = {}
        local_clbit_map: ClbitMap = {}

        if hasattr(block_value, "input_values"):
            qubit_idx = 0
            for input_val in block_value.input_values:
                if hasattr(input_val, "type") and input_val.type.is_quantum():
                    local_qubit_map[QubitAddress(input_val.uuid)] = qubit_idx
                    qubit_idx += 1

        local_qubit_map, local_clbit_map = emit_pass._allocator.allocate(
            block_value.operations
        )

        # Remap to ensure input qubits come first
        if hasattr(block_value, "input_values"):
            for i, input_val in enumerate(block_value.input_values):
                if hasattr(input_val, "type") and input_val.type.is_quantum():
                    local_qubit_map[QubitAddress(input_val.uuid)] = i

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
