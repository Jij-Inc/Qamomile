"""Qiskit backend transpiler implementation."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
    MeasureOperation,
    ControlledUOperation,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation, IfOperation
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.executable import (
    ExecutionContext,
    QuantumExecutor,
    CompiledQuantumSegment,
)


class QiskitEmitPass(EmitPass["QuantumCircuit"]):
    """Qiskit-specific emission pass."""

    def _emit_quantum_segment(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> tuple["QuantumCircuit", dict[str, int], dict[str, int]]:
        """Generate Qiskit QuantumCircuit from operations."""
        from qiskit import QuantumCircuit

        # Count qubits and clbits needed
        qubit_map: dict[str, int] = {}
        clbit_map: dict[str, int] = {}
        qubit_count = 0
        clbit_count = 0

        # First pass: allocate qubits/clbits
        self._allocate_resources(
            operations, qubit_map, clbit_map
        )
        qubit_count = len(qubit_map)
        clbit_count = len(clbit_map)

        # Create circuit
        circuit = QuantumCircuit(qubit_count, clbit_count)

        # Second pass: emit gates
        self._emit_operations(circuit, operations, qubit_map, clbit_map, bindings)

        return circuit, qubit_map, clbit_map

    def _allocate_resources(
        self,
        operations: list[Operation],
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
    ) -> None:
        """Allocate qubit and clbit indices from operations."""
        for op in operations:
            if isinstance(op, QInitOperation):
                result = op.results[0]
                if result.uuid not in qubit_map:
                    qubit_map[result.uuid] = len(qubit_map)
            elif isinstance(op, MeasureOperation):
                result = op.results[0]
                if result.uuid not in clbit_map:
                    clbit_map[result.uuid] = len(clbit_map)
            elif isinstance(op, GateOperation):
                # Track qubit versions
                for operand in op.operands:
                    if operand.uuid not in qubit_map:
                        qubit_map[operand.uuid] = len(qubit_map)
                for result in op.results:
                    if result.uuid not in qubit_map:
                        # Same physical qubit, new version
                        # Use the operand's index
                        if op.operands:
                            qubit_map[result.uuid] = qubit_map.get(
                                op.operands[0].uuid, len(qubit_map)
                            )
            elif isinstance(op, ForOperation):
                self._allocate_resources(op.operations, qubit_map, clbit_map)
            elif isinstance(op, IfOperation):
                self._allocate_resources(op.true_operations, qubit_map, clbit_map)
                self._allocate_resources(op.false_operations, qubit_map, clbit_map)

    def _emit_operations(
        self,
        circuit: QuantumCircuit,
        operations: list[Operation],
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit operations to the circuit."""
        for op in operations:
            if isinstance(op, QInitOperation):
                continue  # Already handled in allocation

            elif isinstance(op, GateOperation):
                self._emit_gate(circuit, op, qubit_map, bindings)

            elif isinstance(op, MeasureOperation):
                qubit_uuid = op.operands[0].uuid
                clbit_uuid = op.results[0].uuid
                if qubit_uuid in qubit_map and clbit_uuid in clbit_map:
                    qubit_idx = qubit_map[qubit_uuid]
                    clbit_idx = clbit_map[clbit_uuid]
                    circuit.measure(qubit_idx, clbit_idx)

            elif isinstance(op, ForOperation):
                self._emit_for(circuit, op, qubit_map, clbit_map, bindings)

            elif isinstance(op, IfOperation):
                self._emit_if(circuit, op, qubit_map, clbit_map, bindings)

    def _emit_gate(
        self,
        circuit: QuantumCircuit,
        op: GateOperation,
        qubit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit a single gate operation."""
        # Get qubit indices from operands
        qubit_indices = []
        for v in op.operands:
            if v.uuid in qubit_map:
                qubit_indices.append(qubit_map[v.uuid])

        if not qubit_indices:
            return

        match op.gate_type:
            case GateOperationType.H:
                circuit.h(qubit_indices[0])
            case GateOperationType.X:
                circuit.x(qubit_indices[0])
            case GateOperationType.Y:
                circuit.y(qubit_indices[0])
            case GateOperationType.Z:
                circuit.z(qubit_indices[0])
            case GateOperationType.T:
                circuit.t(qubit_indices[0])
            case GateOperationType.S:
                circuit.s(qubit_indices[0])
            case GateOperationType.CX:
                if len(qubit_indices) >= 2:
                    circuit.cx(qubit_indices[0], qubit_indices[1])
            case GateOperationType.CZ:
                if len(qubit_indices) >= 2:
                    circuit.cz(qubit_indices[0], qubit_indices[1])
            case GateOperationType.SWAP:
                if len(qubit_indices) >= 2:
                    circuit.swap(qubit_indices[0], qubit_indices[1])
            case GateOperationType.TOFFOLI:
                if len(qubit_indices) >= 3:
                    circuit.ccx(qubit_indices[0], qubit_indices[1], qubit_indices[2])
            case GateOperationType.P:
                angle = self._resolve_angle(op, bindings)
                circuit.p(angle, qubit_indices[0])
            case GateOperationType.RX:
                angle = self._resolve_angle(op, bindings)
                circuit.rx(angle, qubit_indices[0])
            case GateOperationType.RY:
                angle = self._resolve_angle(op, bindings)
                circuit.ry(angle, qubit_indices[0])
            case GateOperationType.RZ:
                angle = self._resolve_angle(op, bindings)
                circuit.rz(angle, qubit_indices[0])
            case GateOperationType.CP:
                if len(qubit_indices) >= 2:
                    angle = self._resolve_angle(op, bindings)
                    circuit.cp(angle, qubit_indices[0], qubit_indices[1])
            case GateOperationType.RZZ:
                if len(qubit_indices) >= 2:
                    angle = self._resolve_angle(op, bindings)
                    circuit.rzz(angle, qubit_indices[0], qubit_indices[1])

        # Update qubit_map for new versions (results)
        for i, result in enumerate(op.results):
            if i < len(qubit_indices):
                qubit_map[result.uuid] = qubit_indices[i]

    def _resolve_angle(
        self,
        op: GateOperation,
        bindings: dict[str, Any],
    ) -> float:
        """Resolve angle parameter for rotation gates.

        The angle is typically stored in the last operand as a classical value.
        """
        # Check if there's an angle operand (classical value)
        for operand in op.operands:
            if operand.type.is_classical():
                # Check if it's a constant
                if operand.is_constant():
                    return float(operand.get_const())
                # Check if it's a parameter
                if operand.is_parameter():
                    param_name = operand.parameter_name()
                    if param_name in bindings:
                        return float(bindings[param_name])
                # Check by name in bindings
                if operand.name in bindings:
                    return float(bindings[operand.name])

        return 0.0

    def _emit_for(
        self,
        circuit: QuantumCircuit,
        op: ForOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit a for loop.

        For now, we unroll the loop if we can determine the count.
        """
        # Get loop count from operand
        loop_count = 1
        if op.operands:
            loop_val = op.operands[0]
            if loop_val.is_constant():
                loop_count = int(loop_val.get_const())
            elif loop_val.is_parameter():
                param_name = loop_val.parameter_name()
                loop_count = int(bindings.get(param_name, 1))
            elif loop_val.name in bindings:
                loop_count = int(bindings[loop_val.name])

        # Unroll the loop
        for _ in range(loop_count):
            self._emit_operations(
                circuit, op.operations, qubit_map, clbit_map, bindings
            )

    def _emit_if(
        self,
        circuit: QuantumCircuit,
        op: IfOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit an if/else using Qiskit's conditional operations."""
        condition = op.condition
        condition_uuid = condition.uuid

        if condition_uuid in clbit_map:
            clbit_idx = clbit_map[condition_uuid]
            # Use Qiskit's if_test for dynamic circuits
            with circuit.if_test((circuit.clbits[clbit_idx], 1)) as else_:
                self._emit_operations(
                    circuit, op.true_operations, qubit_map, clbit_map, bindings
                )
            with else_:
                self._emit_operations(
                    circuit, op.false_operations, qubit_map, clbit_map, bindings
                )


class QiskitExecutor(QuantumExecutor["QuantumCircuit"]):
    """Qiskit quantum executor using AerSimulator or other backends."""

    def __init__(self, backend=None, shots: int = 1024):
        """Initialize executor with backend.

        Args:
            backend: Qiskit backend (defaults to AerSimulator if available)
            shots: Number of measurement shots
        """
        self.backend = backend
        self.shots = shots

        if self.backend is None:
            try:
                from qiskit_aer import AerSimulator
                self.backend = AerSimulator()
            except ImportError:
                pass

    def submit(
        self,
        compiled_segment: CompiledQuantumSegment["QuantumCircuit"],
        context: ExecutionContext,
    ) -> Any:
        """Submit circuit for execution."""
        from qiskit import transpile

        circuit = compiled_segment.circuit

        if self.backend is not None:
            transpiled = transpile(circuit, self.backend)
            job = self.backend.run(transpiled, shots=self.shots)
            return job
        return None

    def get_result(self, job: Any) -> dict[str, Any]:
        """Extract results from execution."""
        if job is None:
            return {}

        result = job.result()
        counts = result.get_counts()
        return {"counts": counts}


class QiskitTranspiler(Transpiler["QuantumCircuit"]):
    """Qiskit backend transpiler.

    Converts Qamomile QKernels into Qiskit QuantumCircuits.

    Example:
        from qamomile.qiskit import QiskitTranspiler
        import qamomile as qm

        @qm.qkernel
        def bell_state(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Bit, qm.Bit]:
            q0 = qm.h(q0)
            q0, q1 = qm.cx(q0, q1)
            return qm.measure(q0), qm.measure(q1)

        transpiler = QiskitTranspiler()
        circuit = transpiler.to_circuit(bell_state)
        print(circuit.draw())
    """

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
    ) -> EmitPass["QuantumCircuit"]:
        return QiskitEmitPass(bindings)

    def executor(
        self,
        backend=None,
        shots: int = 1024,
    ) -> QiskitExecutor:
        """Create a Qiskit executor.

        Args:
            backend: Qiskit backend (defaults to AerSimulator)
            shots: Number of measurement shots

        Returns:
            QiskitExecutor configured with the backend
        """
        return QiskitExecutor(backend, shots)
