"""Standard emit pass using GateEmitter protocol.

This module provides StandardEmitPass, a reusable emit pass implementation
that uses the GateEmitter protocol for backend-specific gate emission.
"""

from __future__ import annotations

import math
import re
from typing import Any, Generic, TypeVar

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
    MeasureOperation,
    MeasureVectorOperation,
    MeasureQFixedOperation,
    ControlledUOperation,
)
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.value import Value

from qamomile.circuit.transpiler.passes.emit import EmitPass, CompositeGateEmitter
from qamomile.circuit.transpiler.passes.emit_base import (
    ResourceAllocator,
    ValueResolver,
    LoopAnalyzer,
    CompositeDecomposer,
    QubitResolutionResult,
)
from qamomile.circuit.transpiler.errors import (
    QubitIndexResolutionError,
    OperandResolutionInfo,
    ResolutionFailureReason,
)
from qamomile.circuit.transpiler.gate_emitter import GateEmitter
from qamomile.circuit.transpiler.executable import ParameterMetadata, ParameterInfo

T = TypeVar("T")  # Backend circuit type


class StandardEmitPass(EmitPass[T], Generic[T]):
    """Standard emit pass implementation using GateEmitter protocol.

    This class provides the orchestration logic for circuit emission
    while delegating backend-specific operations to a GateEmitter.

    Args:
        gate_emitter: Backend-specific gate emitter
        bindings: Parameter bindings for the circuit
        parameters: List of parameter names to preserve as backend parameters
        composite_emitters: Optional list of CompositeGateEmitter for native implementations
    """

    def __init__(
        self,
        gate_emitter: GateEmitter[T],
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
        composite_emitters: list[CompositeGateEmitter[T]] | None = None,
    ):
        super().__init__(bindings, parameters)
        self._emitter = gate_emitter
        self._composite_emitters = composite_emitters or []

        # Helper classes
        self._allocator = ResourceAllocator()
        self._resolver = ValueResolver(self.parameters)
        self._loop_analyzer = LoopAnalyzer()
        self._decomposer = CompositeDecomposer()

        # Cache for backend parameter objects
        self._parameter_map: dict[str, Any] = {}

    def _build_parameter_metadata(self) -> ParameterMetadata:
        """Build parameter metadata from created parameter objects."""
        params = []
        for name, backend_param in self._parameter_map.items():
            match = re.match(r"(\w+)\[(\d+)\]", name)
            if match:
                array_name = match.group(1)
                index = int(match.group(2))
            else:
                array_name = name
                index = None

            params.append(
                ParameterInfo(
                    name=name,
                    array_name=array_name,
                    index=index,
                    backend_param=backend_param,
                )
            )

        return ParameterMetadata(parameters=params)

    def _emit_quantum_segment(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> tuple[T, dict[str, int], dict[str, int]]:
        """Generate backend circuit from operations."""
        # First pass: allocate resources (pass bindings for dynamic size resolution)
        qubit_map, clbit_map = self._allocator.allocate(operations, bindings)

        # Count physical qubits/clbits
        qubit_count = max(qubit_map.values()) + 1 if qubit_map else 0
        clbit_count = max(clbit_map.values()) + 1 if clbit_map else 0

        # Create circuit
        circuit = self._emitter.create_circuit(qubit_count, clbit_count)

        # Second pass: emit gates
        self._emit_operations(circuit, operations, qubit_map, clbit_map, bindings)

        return circuit, qubit_map, clbit_map

    def _emit_operations(
        self,
        circuit: T,
        operations: list[Operation],
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
        force_unroll: bool = False,
    ) -> None:
        """Emit operations to the circuit."""
        for op in operations:
            if isinstance(op, QInitOperation):
                continue

            elif isinstance(op, GateOperation):
                self._emit_gate(circuit, op, qubit_map, bindings)

            elif isinstance(op, MeasureOperation):
                self._emit_measure(circuit, op, qubit_map, clbit_map)

            elif isinstance(op, MeasureVectorOperation):
                self._emit_measure_vector(circuit, op, qubit_map, clbit_map, bindings)

            elif isinstance(op, MeasureQFixedOperation):
                self._emit_measure_qfixed(circuit, op, qubit_map, clbit_map)

            elif isinstance(op, ForOperation):
                self._emit_for(
                    circuit, op, qubit_map, clbit_map, bindings, force_unroll
                )

            elif isinstance(op, IfOperation):
                self._emit_if(circuit, op, qubit_map, clbit_map, bindings)

            elif isinstance(op, WhileOperation):
                self._emit_while(circuit, op, qubit_map, clbit_map, bindings)

            elif isinstance(op, CompositeGateOperation):
                self._emit_composite_gate(circuit, op, qubit_map, bindings)

            elif isinstance(op, ControlledUOperation):
                self._emit_controlled_u(circuit, op, qubit_map, bindings)

            elif isinstance(op, CastOperation):
                self._handle_cast(op, qubit_map)

            elif isinstance(op, BinOp):
                self._evaluate_binop(op, bindings)

    def _emit_gate(
        self,
        circuit: T,
        op: GateOperation,
        qubit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit a single gate operation."""
        qubit_indices = []
        failed_operands: list[OperandResolutionInfo] = []

        for v in op.operands:
            result = self._resolver.resolve_qubit_index_detailed(v, qubit_map, bindings)
            if result.success:
                qubit_indices.append(result.index)
            else:
                # Collect detailed failure info
                element_indices_names = []
                if v.element_indices:
                    for idx in v.element_indices:
                        if idx.parent_array is not None:
                            element_indices_names.append(
                                f"{idx.parent_array.name}[{idx.element_indices[0].name if idx.element_indices else '?'}]"
                            )
                        else:
                            element_indices_names.append(idx.name)

                info = OperandResolutionInfo(
                    operand_name=v.name,
                    operand_uuid=v.uuid,
                    is_array_element=v.parent_array is not None,
                    parent_array_name=v.parent_array.name if v.parent_array else None,
                    element_indices_names=element_indices_names,
                    failure_reason=result.failure_reason or ResolutionFailureReason.UNKNOWN,
                    failure_details=result.failure_details,
                )
                failed_operands.append(info)

        if not qubit_indices or failed_operands:
            # If we have no successful resolutions or some failed, raise detailed error
            if not failed_operands:
                # All operands returned None without detailed failure info
                for v in op.operands:
                    element_indices_names = []
                    if v.element_indices:
                        for idx in v.element_indices:
                            if idx.parent_array is not None:
                                element_indices_names.append(
                                    f"{idx.parent_array.name}[{idx.element_indices[0].name if idx.element_indices else '?'}]"
                                )
                            else:
                                element_indices_names.append(idx.name)

                    failed_operands.append(
                        OperandResolutionInfo(
                            operand_name=v.name,
                            operand_uuid=v.uuid,
                            is_array_element=v.parent_array is not None,
                            parent_array_name=v.parent_array.name if v.parent_array else None,
                            element_indices_names=element_indices_names,
                            failure_reason=ResolutionFailureReason.UNKNOWN,
                            failure_details="No indices resolved but no specific failure recorded",
                        )
                    )

            raise QubitIndexResolutionError(
                gate_type=op.gate_type.name,
                operand_infos=failed_operands,
                available_bindings_keys=list(bindings.keys()),
                available_qubit_map_keys=list(qubit_map.keys()),
            )

        match op.gate_type:
            case GateOperationType.H:
                self._emitter.emit_h(circuit, qubit_indices[0])
            case GateOperationType.X:
                self._emitter.emit_x(circuit, qubit_indices[0])
            case GateOperationType.Y:
                self._emitter.emit_y(circuit, qubit_indices[0])
            case GateOperationType.Z:
                self._emitter.emit_z(circuit, qubit_indices[0])
            case GateOperationType.T:
                self._emitter.emit_t(circuit, qubit_indices[0])
            case GateOperationType.S:
                self._emitter.emit_s(circuit, qubit_indices[0])
            case GateOperationType.CX:
                self._emitter.emit_cx(circuit, qubit_indices[0], qubit_indices[1])
            case GateOperationType.CZ:
                self._emitter.emit_cz(circuit, qubit_indices[0], qubit_indices[1])
            case GateOperationType.SWAP:
                self._emitter.emit_swap(circuit, qubit_indices[0], qubit_indices[1])
            case GateOperationType.TOFFOLI:
                self._emitter.emit_toffoli(
                    circuit, qubit_indices[0], qubit_indices[1], qubit_indices[2]
                )
            case GateOperationType.P:
                angle = self._resolve_angle(op, bindings)
                self._emitter.emit_p(circuit, qubit_indices[0], angle)
            case GateOperationType.RX:
                angle = self._resolve_angle(op, bindings)
                self._emitter.emit_rx(circuit, qubit_indices[0], angle)
            case GateOperationType.RY:
                angle = self._resolve_angle(op, bindings)
                self._emitter.emit_ry(circuit, qubit_indices[0], angle)
            case GateOperationType.RZ:
                angle = self._resolve_angle(op, bindings)
                self._emitter.emit_rz(circuit, qubit_indices[0], angle)
            case GateOperationType.CP:
                angle = self._resolve_angle(op, bindings)
                self._emitter.emit_cp(
                    circuit, qubit_indices[0], qubit_indices[1], angle
                )
            case GateOperationType.RZZ:
                angle = self._resolve_angle(op, bindings)
                self._emitter.emit_rzz(
                    circuit, qubit_indices[0], qubit_indices[1], angle
                )
            case _:
                raise RuntimeError(f"Unsupported gate type: {op.gate_type}")

        # Update qubit_map for new versions
        for i, result in enumerate(op.results):
            if i < len(qubit_indices):
                qubit_map[result.uuid] = qubit_indices[i]

    def _emit_measure(
        self,
        circuit: T,
        op: MeasureOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
    ) -> None:
        """Emit a single measurement."""
        qubit_uuid = op.operands[0].uuid
        clbit_uuid = op.results[0].uuid
        if qubit_uuid in qubit_map and clbit_uuid in clbit_map:
            self._emitter.emit_measure(
                circuit, qubit_map[qubit_uuid], clbit_map[clbit_uuid]
            )

    def _emit_measure_vector(
        self,
        circuit: T,
        op: MeasureVectorOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit vector measurement."""
        from qamomile.circuit.ir.value import ArrayValue

        qubits_array = op.operands[0]
        bits_array = op.results[0]

        if isinstance(qubits_array, ArrayValue) and isinstance(bits_array, ArrayValue):
            element_uuids = qubits_array.params.get("element_uuids", None)

            shape = qubits_array.shape
            if shape:
                size_val = shape[0]
                # Use allocator's _resolve_size to handle dynamic sizes (e.g., hi_dim0)
                size = self._allocator._resolve_size(size_val, bindings)
                if size is not None:
                    for i in range(size):
                        if element_uuids and i < len(element_uuids):
                            qubit_uuid = element_uuids[i]
                        else:
                            qubit_uuid = f"{qubits_array.uuid}_{i}"

                        clbit_id = f"{bits_array.uuid}_{i}"
                        if qubit_uuid in qubit_map and clbit_id in clbit_map:
                            self._emitter.emit_measure(
                                circuit, qubit_map[qubit_uuid], clbit_map[clbit_id]
                            )

    def _emit_measure_qfixed(
        self,
        circuit: T,
        op: MeasureQFixedOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
    ) -> None:
        """Emit QFixed measurement."""
        qfixed = op.operands[0]
        qubit_uuids = qfixed.params.get("qubit_values", [])
        result = op.results[0]

        for i, qubit_uuid in enumerate(qubit_uuids):
            clbit_id = f"{result.uuid}_{i}"
            if qubit_uuid in qubit_map and clbit_id in clbit_map:
                self._emitter.emit_measure(
                    circuit, qubit_map[qubit_uuid], clbit_map[clbit_id]
                )

    def _emit_for(
        self,
        circuit: T,
        op: ForOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
        force_unroll: bool = False,
    ) -> None:
        """Emit a for loop."""
        start = (
            self._resolver.resolve_int_value(op.operands[0], bindings)
            if len(op.operands) > 0
            else 0
        )
        stop = (
            self._resolver.resolve_int_value(op.operands[1], bindings)
            if len(op.operands) > 1
            else 1
        )
        step = (
            self._resolver.resolve_int_value(op.operands[2], bindings)
            if len(op.operands) > 2
            else 1
        )

        if start is None or stop is None or step is None:
            self._emit_for_unrolled(circuit, op, qubit_map, clbit_map, bindings)
            return

        indexset = range(start, stop, step)
        if len(indexset) == 0:
            return

        if force_unroll:
            self._emit_for_unrolled(circuit, op, qubit_map, clbit_map, bindings)
            return

        if self._loop_analyzer.should_unroll(op, bindings):
            self._emit_for_unrolled(circuit, op, qubit_map, clbit_map, bindings)
            return

        # Try native for loop
        if self._emitter.supports_for_loop():
            loop_context = self._emitter.emit_for_loop_start(circuit, indexset)
            loop_bindings = bindings.copy()
            loop_bindings[op.loop_var] = loop_context
            self._emit_operations(
                circuit, op.operations, qubit_map, clbit_map, loop_bindings
            )
            self._emitter.emit_for_loop_end(circuit, loop_context)
        else:
            self._emit_for_unrolled(circuit, op, qubit_map, clbit_map, bindings)

    def _emit_for_unrolled(
        self,
        circuit: T,
        op: ForOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit for loop by unrolling."""
        start = (
            self._resolver.resolve_int_value(op.operands[0], bindings)
            if len(op.operands) > 0
            else 0
        )
        stop = (
            self._resolver.resolve_int_value(op.operands[1], bindings)
            if len(op.operands) > 1
            else 1
        )
        step = (
            self._resolver.resolve_int_value(op.operands[2], bindings)
            if len(op.operands) > 2
            else 1
        )

        if start is None or stop is None or step is None:
            raise ValueError(
                f"Cannot unroll loop: bounds could not be resolved. "
                f"start={start}, stop={stop}, step={step}"
            )

        for i in range(start, stop, step):
            loop_bindings = bindings.copy()
            loop_bindings[op.loop_var] = i
            self._emit_operations(
                circuit, op.operations, qubit_map, clbit_map, loop_bindings
            )

    def _emit_if(
        self,
        circuit: T,
        op: IfOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit if/else operation."""
        condition_uuid = op.condition.uuid

        if condition_uuid not in clbit_map:
            return

        clbit_idx = clbit_map[condition_uuid]

        if self._emitter.supports_if_else():
            context = self._emitter.emit_if_start(circuit, clbit_idx, 1)
            self._emit_operations(
                circuit, op.true_operations, qubit_map, clbit_map, bindings
            )
            self._emitter.emit_else_start(circuit, context)
            self._emit_operations(
                circuit, op.false_operations, qubit_map, clbit_map, bindings
            )
            self._emitter.emit_if_end(circuit, context)

    def _emit_while(
        self,
        circuit: T,
        op: WhileOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit while loop operation."""
        if not op.operands:
            raise ValueError("WhileOperation requires a condition operand")

        condition = op.operands[0]
        condition_value = condition.value if hasattr(condition, "value") else condition
        condition_uuid = (
            condition_value.uuid
            if hasattr(condition_value, "uuid")
            else str(condition_value)
        )

        if condition_uuid not in clbit_map:
            raise ValueError("While loop condition not found in classical bit map.")

        clbit_idx = clbit_map[condition_uuid]

        if self._emitter.supports_while_loop():
            context = self._emitter.emit_while_start(circuit, clbit_idx, 1)
            self._emit_operations(
                circuit, op.operations, qubit_map, clbit_map, bindings
            )
            self._emitter.emit_while_end(circuit, context)

    def _emit_composite_gate(
        self,
        circuit: T,
        op: CompositeGateOperation,
        qubit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit a composite gate operation."""
        all_qubits = op.control_qubits + op.target_qubits
        qubit_indices = [qubit_map[q.uuid] for q in all_qubits if q.uuid in qubit_map]

        # Try native emitters first
        for emitter in self._composite_emitters:
            if emitter.can_emit(op.gate_type):
                if emitter.emit(circuit, op, qubit_indices, bindings):
                    self._update_composite_result_mapping(op, qubit_indices, qubit_map)
                    return

        # Fall back to decomposition
        self._emit_composite_fallback(circuit, op, qubit_indices, bindings)
        self._update_composite_result_mapping(op, qubit_indices, qubit_map)

    def _emit_composite_fallback(
        self,
        circuit: T,
        op: CompositeGateOperation,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit composite gate using decomposition."""
        if op.gate_type == CompositeGateType.QPE:
            self._emit_qpe_manual(circuit, op, qubit_indices, bindings)
        elif op.gate_type == CompositeGateType.QFT:
            self._emit_qft_manual(circuit, qubit_indices)
        elif op.gate_type == CompositeGateType.IQFT:
            self._emit_iqft_manual(circuit, qubit_indices)
        elif not op.has_implementation:
            if qubit_indices:
                self._emitter.emit_barrier(circuit, qubit_indices)
        elif op.has_implementation:
            impl = op.implementation
            if impl is not None:
                self._emit_custom_composite(circuit, op, impl, qubit_indices, bindings)
            elif qubit_indices:
                self._emitter.emit_barrier(circuit, qubit_indices)

    def _emit_qft_manual(self, circuit: T, qubit_indices: list[int]) -> None:
        """Emit QFT using decomposition."""
        n = len(qubit_indices)
        if n == 0:
            return

        for i in range(n):
            self._emitter.emit_h(circuit, qubit_indices[i])
            for j in range(i + 1, n):
                k = j - i
                angle = math.pi / (2**k)
                self._emitter.emit_cp(
                    circuit, qubit_indices[j], qubit_indices[i], angle
                )

        for i in range(n // 2):
            self._emitter.emit_swap(circuit, qubit_indices[i], qubit_indices[n - 1 - i])

    def _emit_iqft_manual(self, circuit: T, qubit_indices: list[int]) -> None:
        """Emit inverse QFT using decomposition."""
        n = len(qubit_indices)
        if n == 0:
            return

        for i in range(n - 1, -1, -1):
            self._emitter.emit_h(circuit, qubit_indices[i])
            for j in range(i - 1, -1, -1):
                k = i - j
                angle = -math.pi / (2**k)
                self._emitter.emit_cp(
                    circuit, qubit_indices[j], qubit_indices[i], angle
                )

        for i in range(n // 2):
            self._emitter.emit_swap(circuit, qubit_indices[i], qubit_indices[n - 1 - i])

    def _emit_qpe_manual(
        self,
        circuit: T,
        op: CompositeGateOperation,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit QPE using manual decomposition."""
        num_counting = op.num_control_qubits
        num_targets = op.num_target_qubits

        if len(qubit_indices) < num_counting + num_targets:
            return

        counting_indices = qubit_indices[:num_counting]
        target_indices = qubit_indices[num_counting : num_counting + num_targets]

        # Step 1: Apply H to counting qubits
        for idx in counting_indices:
            self._emitter.emit_h(circuit, idx)

        # Step 2: Apply controlled-U^(2^k) operations
        if op.operands and hasattr(op.operands[0], "operations"):
            block_value = op.operands[0]

            local_bindings = bindings.copy()
            if hasattr(block_value, "input_values"):
                param_operands = op.parameters
                param_inputs = [
                    iv
                    for iv in block_value.input_values
                    if hasattr(iv, "type") and iv.type.is_classical()
                ]

                for param_input, param_operand in zip(param_inputs, param_operands):
                    param_name = param_input.name
                    if param_operand.is_constant():
                        local_bindings[param_name] = param_operand.get_const()
                    elif param_operand.is_parameter():
                        outer_name = param_operand.parameter_name()
                        if outer_name and outer_name in bindings:
                            local_bindings[param_name] = bindings[outer_name]

            self._emit_controlled_powers(
                circuit, block_value, counting_indices, target_indices, local_bindings
            )
        else:
            phase = self._extract_phase_from_params(op, bindings)
            if phase is not None:
                for k, ctrl_idx in enumerate(counting_indices):
                    power = 2**k
                    angle = phase * power
                    for tgt_idx in target_indices:
                        self._emitter.emit_cp(circuit, ctrl_idx, tgt_idx, angle)

        # Step 3: Apply inverse QFT
        self._emit_iqft_manual(circuit, counting_indices)

    def _emit_controlled_powers(
        self,
        circuit: T,
        block_value: Any,
        counting_indices: list[int],
        target_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit controlled-U^(2^k) operations."""
        num_targets = len(target_indices)
        unitary_gate = self._blockvalue_to_gate(block_value, num_targets, bindings)

        if unitary_gate is not None:
            for k, ctrl_idx in enumerate(counting_indices):
                power = 2**k
                powered_gate = self._emitter.gate_power(unitary_gate, power)
                controlled_powered_gate = self._emitter.gate_controlled(powered_gate, 1)
                self._emitter.append_gate(
                    circuit, controlled_powered_gate, [ctrl_idx] + target_indices
                )
        else:
            for k, ctrl_idx in enumerate(counting_indices):
                power = 2**k
                for _ in range(power):
                    self._emit_controlled_block(
                        circuit, block_value, ctrl_idx, target_indices, bindings
                    )

    def _emit_controlled_block(
        self,
        circuit: T,
        block_value: Any,
        control_idx: int,
        target_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit controlled version of a block."""
        if not hasattr(block_value, "operations"):
            return

        self._emit_controlled_operations(
            circuit, block_value.operations, control_idx, target_indices, bindings
        )

    def _emit_controlled_operations(
        self,
        circuit: T,
        operations: list[Operation],
        control_idx: int,
        target_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit controlled versions of operations."""
        for op in operations:
            if isinstance(op, GateOperation):
                self._emit_controlled_gate(
                    circuit, op, control_idx, target_indices, bindings
                )
            elif isinstance(op, ForOperation):
                start = (
                    self._resolver.resolve_int_value(op.operands[0], bindings)
                    if len(op.operands) > 0
                    else 0
                )
                stop = (
                    self._resolver.resolve_int_value(op.operands[1], bindings)
                    if len(op.operands) > 1
                    else 1
                )
                step = (
                    self._resolver.resolve_int_value(op.operands[2], bindings)
                    if len(op.operands) > 2
                    else 1
                )

                if start is not None and stop is not None and step is not None:
                    for i in range(start, stop, step):
                        loop_bindings = bindings.copy()
                        loop_bindings[op.loop_var] = i
                        self._emit_controlled_operations(
                            circuit,
                            op.operations,
                            control_idx,
                            target_indices,
                            loop_bindings,
                        )

    def _emit_controlled_gate(
        self,
        circuit: T,
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
                self._emitter.emit_ch(circuit, control_idx, target_idx)
            case GateOperationType.X:
                self._emitter.emit_cx(circuit, control_idx, target_idx)
            case GateOperationType.Y:
                self._emitter.emit_cy(circuit, control_idx, target_idx)
            case GateOperationType.Z:
                self._emitter.emit_cz(circuit, control_idx, target_idx)
            case GateOperationType.P:
                angle = self._resolve_angle(op, bindings)
                self._emitter.emit_cp(circuit, control_idx, target_idx, angle)
            case GateOperationType.RX:
                angle = self._resolve_angle(op, bindings)
                self._emitter.emit_crx(circuit, control_idx, target_idx, angle)
            case GateOperationType.RY:
                angle = self._resolve_angle(op, bindings)
                self._emitter.emit_cry(circuit, control_idx, target_idx, angle)
            case GateOperationType.RZ:
                angle = self._resolve_angle(op, bindings)
                self._emitter.emit_crz(circuit, control_idx, target_idx, angle)
            case GateOperationType.S:
                self._emitter.emit_cp(circuit, control_idx, target_idx, math.pi / 2)
            case GateOperationType.T:
                self._emitter.emit_cp(circuit, control_idx, target_idx, math.pi / 4)

    def _emit_controlled_u(
        self,
        circuit: T,
        op: ControlledUOperation,
        qubit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit a ControlledUOperation."""
        block_value = op.block
        control_operands = op.control_operands
        remaining_operands = op.operands[1 + op.num_controls :]

        target_qubit_operands = []
        param_operands = []
        for operand in remaining_operands:
            if hasattr(operand, "type") and operand.type.is_quantum():
                target_qubit_operands.append(operand)
            elif hasattr(operand, "type") and operand.type.is_classical():
                param_operands.append(operand)

        control_indices = []
        for q in control_operands:
            idx = self._resolver.resolve_qubit_index(q, qubit_map, bindings)
            if idx is not None:
                control_indices.append(idx)

        target_indices = []
        for q in target_qubit_operands:
            idx = self._resolver.resolve_qubit_index(q, qubit_map, bindings)
            if idx is not None:
                target_indices.append(idx)

        if not control_indices or not target_indices:
            return

        local_bindings = bindings.copy()

        if hasattr(block_value, "input_values"):
            param_inputs = [
                iv
                for iv in block_value.input_values
                if hasattr(iv, "type") and iv.type.is_classical()
            ]
            for i, operand in enumerate(param_operands):
                if i >= len(param_inputs):
                    break

                param_name = param_inputs[i].name

                if operand.is_constant():
                    local_bindings[param_name] = operand.get_const()
                elif operand.is_parameter():
                    outer_name = operand.parameter_name()
                    if outer_name and outer_name in bindings:
                        local_bindings[param_name] = bindings[outer_name]
                elif operand.name in bindings:
                    local_bindings[param_name] = bindings[operand.name]
                elif (
                    hasattr(operand, "params")
                    and operand.params
                    and "const" in operand.params
                ):
                    local_bindings[param_name] = operand.params["const"]

        num_targets = len(target_indices)
        unitary_gate = self._blockvalue_to_gate(
            block_value, num_targets, local_bindings
        )

        if unitary_gate is not None:
            if op.power > 1:
                unitary_gate = self._emitter.gate_power(unitary_gate, op.power)
            controlled_gate = self._emitter.gate_controlled(
                unitary_gate, op.num_controls
            )
            self._emitter.append_gate(
                circuit, controlled_gate, control_indices + target_indices
            )
        else:
            for _ in range(op.power):
                for ctrl_idx in control_indices:
                    self._emit_controlled_block(
                        circuit, block_value, ctrl_idx, target_indices, local_bindings
                    )

        all_input_indices = control_indices + target_indices
        for i, result in enumerate(op.results):
            if i < len(all_input_indices):
                qubit_map[result.uuid] = all_input_indices[i]

    def _emit_custom_composite(
        self,
        circuit: T,
        op: CompositeGateOperation,
        impl: Any,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit a custom composite gate with implementation."""
        num_qubits = len(qubit_indices)
        custom_gate = self._blockvalue_to_gate(impl, num_qubits, bindings)

        if custom_gate is not None:
            self._emitter.append_gate(circuit, custom_gate, qubit_indices)
        else:
            local_qubit_map: dict[str, int] = {}
            local_clbit_map: dict[str, int] = {}

            if hasattr(impl, "input_values"):
                for i, input_val in enumerate(impl.input_values):
                    if i < len(qubit_indices):
                        local_qubit_map[input_val.uuid] = qubit_indices[i]

            if hasattr(impl, "operations"):
                self._emit_operations(
                    circuit,
                    impl.operations,
                    local_qubit_map,
                    local_clbit_map,
                    bindings,
                    force_unroll=True,
                )

    def _blockvalue_to_gate(
        self,
        block_value: Any,
        num_qubits: int,
        bindings: dict[str, Any],
    ) -> Any:
        """Convert a BlockValue to a backend gate."""
        if not hasattr(block_value, "operations"):
            return None

        try:
            local_qubit_map: dict[str, int] = {}
            local_clbit_map: dict[str, int] = {}

            if hasattr(block_value, "input_values"):
                qubit_idx = 0
                for input_val in block_value.input_values:
                    if hasattr(input_val, "type") and input_val.type.is_quantum():
                        local_qubit_map[input_val.uuid] = qubit_idx
                        qubit_idx += 1

            local_qubit_map, local_clbit_map = self._allocator.allocate(
                block_value.operations
            )

            # Remap to ensure input qubits come first
            if hasattr(block_value, "input_values"):
                for i, input_val in enumerate(block_value.input_values):
                    if hasattr(input_val, "type") and input_val.type.is_quantum():
                        local_qubit_map[input_val.uuid] = i

            qubit_count = (
                max(local_qubit_map.values()) + 1 if local_qubit_map else num_qubits
            )
            sub_circuit = self._emitter.create_circuit(qubit_count, 0)

            self._emit_operations(
                sub_circuit,
                block_value.operations,
                local_qubit_map,
                local_clbit_map,
                bindings,
                force_unroll=True,
            )

            return self._emitter.circuit_to_gate(sub_circuit, "U")

        except Exception:
            return None

    def _update_composite_result_mapping(
        self,
        op: CompositeGateOperation,
        qubit_indices: list[int],
        qubit_map: dict[str, int],
    ) -> None:
        """Update qubit_map for composite gate results."""
        for i, result in enumerate(op.results):
            if i < len(qubit_indices):
                qubit_map[result.uuid] = qubit_indices[i]

    def _extract_phase_from_params(
        self,
        op: CompositeGateOperation,
        bindings: dict[str, Any],
    ) -> float | None:
        """Extract phase parameter from QPE operation."""
        for i, operand in enumerate(op.operands):
            if i < 1:
                continue
            if hasattr(operand, "type") and hasattr(operand.type, "is_classical"):
                if operand.type.is_classical():
                    if operand.is_constant():
                        const_val = operand.get_const()
                        if const_val is not None:
                            return float(const_val)
                    if operand.is_parameter():
                        param_name = operand.parameter_name()
                        if param_name and param_name in bindings:
                            return float(bindings[param_name])
                    if hasattr(operand, "name") and operand.name in bindings:
                        return float(bindings[operand.name])
            elif (
                hasattr(operand, "params")
                and operand.params
                and "const" in operand.params
            ):
                return float(operand.params["const"])

        return None

    def _handle_cast(self, op: CastOperation, qubit_map: dict[str, int]) -> None:
        """Handle CastOperation - update qubit_map without emitting gates."""
        result = op.results[0]

        for i, qubit_uuid in enumerate(op.qubit_mapping):
            if qubit_uuid in qubit_map:
                result_element_id = f"{result.uuid}_{i}"
                qubit_map[result_element_id] = qubit_map[qubit_uuid]

        if op.qubit_mapping and op.qubit_mapping[0] in qubit_map:
            qubit_map[result.uuid] = qubit_map[op.qubit_mapping[0]]

    def _evaluate_binop(self, op: BinOp, bindings: dict[str, Any]) -> None:
        """Evaluate a BinOp and store the result in bindings."""
        lhs = self._resolver.resolve_classical_value(op.lhs, bindings)
        rhs = self._resolver.resolve_classical_value(op.rhs, bindings)

        lhs_param_key = self._resolver.get_parameter_key(op.lhs, bindings)
        rhs_param_key = self._resolver.get_parameter_key(op.rhs, bindings)

        if lhs is None and lhs_param_key:
            if lhs_param_key not in self._parameter_map:
                self._parameter_map[lhs_param_key] = self._emitter.create_parameter(
                    lhs_param_key
                )
            lhs = self._parameter_map[lhs_param_key]
        if rhs is None and rhs_param_key:
            if rhs_param_key not in self._parameter_map:
                self._parameter_map[rhs_param_key] = self._emitter.create_parameter(
                    rhs_param_key
                )
            rhs = self._parameter_map[rhs_param_key]

        if lhs is None or rhs is None:
            return

        result = None
        match op.kind:
            case BinOpKind.ADD:
                result = lhs + rhs
            case BinOpKind.SUB:
                result = lhs - rhs
            case BinOpKind.MUL:
                result = lhs * rhs
            case BinOpKind.DIV:
                result = (
                    lhs / rhs
                    if (isinstance(rhs, (int, float)) and rhs != 0)
                    else (lhs / rhs if rhs != 0 else 0.0)
                )
            case BinOpKind.FLOORDIV:
                result = (
                    lhs // rhs if (isinstance(rhs, (int, float)) and rhs != 0) else 0
                )
            case BinOpKind.POW:
                result = lhs**rhs

        if result is not None and op.results:
            output = op.results[0]
            bindings[output.uuid] = result
            bindings[output.name] = result

    def _resolve_angle(
        self,
        op: GateOperation,
        bindings: dict[str, Any],
    ) -> float | Any:
        """Resolve angle parameter for rotation gates."""
        if hasattr(op, "theta") and op.theta is not None:
            theta = op.theta
            if isinstance(theta, (int, float)):
                return float(theta)
            elif isinstance(theta, Value):
                param_key = self._resolver.get_parameter_key(theta, bindings)
                if param_key:
                    if param_key not in self._parameter_map:
                        self._parameter_map[param_key] = self._emitter.create_parameter(
                            param_key
                        )
                    return self._parameter_map[param_key]

                resolved = self._resolver.resolve_classical_value(theta, bindings)
                if resolved is not None:
                    if not isinstance(resolved, (int, float)):
                        return resolved
                    return float(resolved)

        for operand in op.operands:
            if hasattr(operand, "type") and operand.type.is_classical():
                param_key = self._resolver.get_parameter_key(operand, bindings)
                if param_key:
                    if param_key not in self._parameter_map:
                        self._parameter_map[param_key] = self._emitter.create_parameter(
                            param_key
                        )
                    return self._parameter_map[param_key]

                resolved = self._resolver.resolve_classical_value(operand, bindings)
                if resolved is not None:
                    if not isinstance(resolved, (int, float)):
                        return resolved
                    return float(resolved)

        return 0.0
