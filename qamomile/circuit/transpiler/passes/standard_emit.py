"""Standard emit pass using GateEmitter protocol.

This module provides StandardEmitPass, a reusable emit pass implementation
that uses the GateEmitter protocol for backend-specific gate emission.

The actual emission logic is decomposed into focused modules under
``emit_support/``. This class serves as the orchestrator with thin
wrappers that delegate to those module functions while preserving
subclass override points (used by QiskitEmitPass, CudaqEmitPass).
"""

from __future__ import annotations

import re
from typing import Any, Generic, TypeVar

from qamomile.circuit.ir.operation import Operation, SliceArrayOperation
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.transpiler.executable import ParameterInfo, ParameterMetadata
from qamomile.circuit.transpiler.gate_emitter import GateEmitter
from qamomile.circuit.transpiler.passes.emit import CompositeGateEmitter, EmitPass
from qamomile.circuit.transpiler.passes.emit_support import (
    ClbitMap,
    CompositeDecomposer,
    LoopAnalyzer,
    QubitMap,
    ResourceAllocator,
)
from qamomile.circuit.transpiler.passes.emit_support.cast_binop_emission import (
    evaluate_binop,
    handle_cast,
)
from qamomile.circuit.transpiler.passes.emit_support.composite_gate_emission import (
    emit_composite_gate,
)
from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
    emit_for,
    emit_for_items,
    emit_if,
    emit_while,
)
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    blockvalue_to_gate,
    emit_controlled_fallback,
    emit_controlled_u,
)
from qamomile.circuit.transpiler.passes.emit_support.gate_emission import emit_gate
from qamomile.circuit.transpiler.passes.emit_support.measurement_emission import (
    emit_measure,
    emit_measure_qfixed,
    emit_measure_vector,
)
from qamomile.circuit.transpiler.passes.emit_support.pauli_evolve_emission import (
    emit_pauli_evolve,
)

T = TypeVar("T")  # Backend circuit type


class StandardEmitPass(EmitPass[T], Generic[T]):
    """Standard emit pass implementation using GateEmitter protocol.

    This class provides the orchestration logic for circuit emission
    while delegating backend-specific operations to a GateEmitter.

    Subclasses (QiskitEmitPass, CudaqEmitPass) override specific methods
    to provide native backend support. The thin wrappers here delegate to
    module functions in ``emit_support/`` by default.

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

        # Helper classes (``_resolver`` is built by ``EmitPass.__init__``).
        self._allocator = ResourceAllocator()
        self._loop_analyzer = LoopAnalyzer()
        self._decomposer = CompositeDecomposer()

        # Cache for backend parameter objects
        self._parameter_map: dict[str, Any] = {}
        self._parameter_sources: dict[str, str] = {}

        # Mapping from classical bit index to physical qubit index.
        # Populated during measurement emission to support backends
        # where emit_measure is a no-op (e.g., QURI Parts).
        self._measurement_qubit_map: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Core orchestration (stays on this class)
    # ------------------------------------------------------------------

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
                    source_ref=self._parameter_sources.get(name),
                )
            )

        return ParameterMetadata(parameters=params)

    def _get_or_create_parameter(
        self,
        name: str,
        source_ref: str | None = None,
    ) -> Any:
        """Get or create a backend parameter while tracking its IR source."""
        if name not in self._parameter_map:
            self._parameter_map[name] = self._emitter.create_parameter(name)
        if source_ref is not None:
            self._parameter_sources.setdefault(name, source_ref)
        return self._parameter_map[name]

    def _emit_quantum_segment(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> tuple[T, QubitMap, ClbitMap]:
        """Generate backend circuit from operations."""
        qubit_map, clbit_map = self._allocator.allocate(operations, bindings)

        qubit_count = max(qubit_map.values()) + 1 if qubit_map else 0
        clbit_count = max(clbit_map.values()) + 1 if clbit_map else 0

        circuit = self._emitter.create_circuit(qubit_count, clbit_count)
        self._measurement_qubit_map.clear()

        self._emit_operations(circuit, operations, qubit_map, clbit_map, bindings)

        return circuit, qubit_map, clbit_map

    def _emit_operations(
        self,
        circuit: T,
        operations: list[Operation],
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
        force_unroll: bool = False,
    ) -> None:
        """Emit operations to the circuit (dispatcher)."""
        for op in operations:
            if isinstance(op, QInitOperation):
                continue
            elif isinstance(op, SliceArrayOperation):
                # ConstantFoldingPass is expected to strip this op
                # before emit; reaching here is a compiler-internal
                # invariant violation.  Fail loudly rather than
                # silently emitting nothing.
                raise RuntimeError(
                    "SliceArrayOperation reached emit — ConstantFoldingPass "
                    "should have stripped it.  This is a compiler bug; "
                    "please report it."
                )
            elif isinstance(op, GateOperation):
                emit_gate(self, circuit, op, qubit_map, bindings)
            elif isinstance(op, MeasureOperation):
                emit_measure(self, circuit, op, qubit_map, clbit_map, bindings)
            elif isinstance(op, MeasureVectorOperation):
                emit_measure_vector(self, circuit, op, qubit_map, clbit_map, bindings)
            elif isinstance(op, MeasureQFixedOperation):
                emit_measure_qfixed(self, circuit, op, qubit_map, clbit_map)
            elif isinstance(op, ForOperation):
                self._emit_for(
                    circuit, op, qubit_map, clbit_map, bindings, force_unroll
                )
            elif isinstance(op, ForItemsOperation):
                emit_for_items(self, circuit, op, qubit_map, clbit_map, bindings)
            elif isinstance(op, IfOperation):
                self._emit_if(circuit, op, qubit_map, clbit_map, bindings)
            elif isinstance(op, WhileOperation):
                self._emit_while(circuit, op, qubit_map, clbit_map, bindings)
            elif isinstance(op, CompositeGateOperation):
                emit_composite_gate(self, circuit, op, qubit_map, bindings)
            elif isinstance(op, ControlledUOperation):
                emit_controlled_u(self, circuit, op, qubit_map, bindings)
            elif isinstance(op, PauliEvolveOp):
                self._emit_pauli_evolve(circuit, op, qubit_map, bindings)
            elif isinstance(op, CastOperation):
                handle_cast(self, op, qubit_map)
            elif isinstance(op, BinOp):
                evaluate_binop(self, op, bindings)
            elif isinstance(op, HasNestedOps):
                raise NotImplementedError(
                    f"Unhandled control flow: {type(op).__name__}"
                )

    # ------------------------------------------------------------------
    # Methods overridden by backend subclasses (QiskitEmitPass, CudaqEmitPass).
    # Only methods with actual overrides are kept as instance methods.
    # ------------------------------------------------------------------

    def _emit_for(
        self,
        circuit: T,
        op: ForOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
        force_unroll: bool = False,
    ) -> None:
        emit_for(self, circuit, op, qubit_map, clbit_map, bindings, force_unroll)

    def _emit_for_unrolled(
        self,
        circuit: T,
        op: ForOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
            emit_for_unrolled,
        )

        emit_for_unrolled(self, circuit, op, qubit_map, clbit_map, bindings)

    def _emit_if(
        self,
        circuit: T,
        op: IfOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        emit_if(self, circuit, op, qubit_map, clbit_map, bindings)

    def _emit_while(
        self,
        circuit: T,
        op: WhileOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        emit_while(self, circuit, op, qubit_map, clbit_map, bindings)

    def _emit_pauli_evolve(
        self,
        circuit: T,
        op: PauliEvolveOp,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        emit_pauli_evolve(self, circuit, op, qubit_map, bindings)

    def _emit_controlled_fallback(
        self,
        circuit: T,
        block_value: Any,
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        power: int,
        bindings: dict[str, Any],
    ) -> None:
        emit_controlled_fallback(
            self,
            circuit,
            block_value,
            num_controls,
            control_indices,
            target_indices,
            power,
            bindings,
        )

    def _blockvalue_to_gate(
        self, block_value: Any, num_qubits: int, bindings: dict[str, Any]
    ) -> Any:
        return blockvalue_to_gate(self, block_value, num_qubits, bindings)

    # ------------------------------------------------------------------
    # Helpers called from emit_support modules
    # ------------------------------------------------------------------

    def _resolve_angle(self, op: GateOperation, bindings: dict[str, Any]) -> Any:
        from qamomile.circuit.transpiler.passes.emit_support.gate_emission import (
            resolve_angle,
        )

        return resolve_angle(self, op, bindings)

    def _emit_custom_composite(
        self,
        circuit: T,
        op: Any,
        impl: Any,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
            emit_custom_composite,
        )

        emit_custom_composite(self, circuit, op, impl, qubit_indices, bindings)

    def _emit_controlled_powers(
        self,
        circuit: T,
        block_value: Any,
        counting_indices: list[int],
        target_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
            emit_controlled_powers,
        )

        emit_controlled_powers(
            self, circuit, block_value, counting_indices, target_indices, bindings
        )
