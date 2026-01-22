"""Qiskit GateEmitter implementation.

This module provides QiskitGateEmitter, which implements the GateEmitter
protocol for Qiskit backends.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate


class QiskitGateEmitter:
    """GateEmitter implementation for Qiskit.

    Emits individual quantum gates to Qiskit QuantumCircuit objects.
    """

    def create_circuit(self, num_qubits: int, num_clbits: int) -> "QuantumCircuit":
        """Create a new Qiskit QuantumCircuit."""
        from qiskit import QuantumCircuit

        return QuantumCircuit(num_qubits, num_clbits)

    def create_parameter(self, name: str) -> Any:
        """Create a Qiskit Parameter object."""
        from qiskit.circuit import Parameter

        return Parameter(name)

    # Single-qubit gates
    def emit_h(self, circuit: "QuantumCircuit", qubit: int) -> None:
        circuit.h(qubit)

    def emit_x(self, circuit: "QuantumCircuit", qubit: int) -> None:
        circuit.x(qubit)

    def emit_y(self, circuit: "QuantumCircuit", qubit: int) -> None:
        circuit.y(qubit)

    def emit_z(self, circuit: "QuantumCircuit", qubit: int) -> None:
        circuit.z(qubit)

    def emit_s(self, circuit: "QuantumCircuit", qubit: int) -> None:
        circuit.s(qubit)

    def emit_t(self, circuit: "QuantumCircuit", qubit: int) -> None:
        circuit.t(qubit)

    # Single-qubit rotation gates
    def emit_rx(
        self, circuit: "QuantumCircuit", qubit: int, angle: float | Any
    ) -> None:
        circuit.rx(angle, qubit)

    def emit_ry(
        self, circuit: "QuantumCircuit", qubit: int, angle: float | Any
    ) -> None:
        circuit.ry(angle, qubit)

    def emit_rz(
        self, circuit: "QuantumCircuit", qubit: int, angle: float | Any
    ) -> None:
        circuit.rz(angle, qubit)

    def emit_p(self, circuit: "QuantumCircuit", qubit: int, angle: float | Any) -> None:
        circuit.p(angle, qubit)

    # Two-qubit gates
    def emit_cx(self, circuit: "QuantumCircuit", control: int, target: int) -> None:
        circuit.cx(control, target)

    def emit_cz(self, circuit: "QuantumCircuit", control: int, target: int) -> None:
        circuit.cz(control, target)

    def emit_swap(self, circuit: "QuantumCircuit", qubit1: int, qubit2: int) -> None:
        circuit.swap(qubit1, qubit2)

    # Two-qubit rotation gates
    def emit_cp(
        self, circuit: "QuantumCircuit", control: int, target: int, angle: float | Any
    ) -> None:
        circuit.cp(angle, control, target)

    def emit_rzz(
        self, circuit: "QuantumCircuit", qubit1: int, qubit2: int, angle: float | Any
    ) -> None:
        circuit.rzz(angle, qubit1, qubit2)

    # Three-qubit gates
    def emit_toffoli(
        self, circuit: "QuantumCircuit", control1: int, control2: int, target: int
    ) -> None:
        circuit.ccx(control1, control2, target)

    # Controlled single-qubit gates
    def emit_ch(self, circuit: "QuantumCircuit", control: int, target: int) -> None:
        circuit.ch(control, target)

    def emit_cy(self, circuit: "QuantumCircuit", control: int, target: int) -> None:
        circuit.cy(control, target)

    def emit_crx(
        self, circuit: "QuantumCircuit", control: int, target: int, angle: float | Any
    ) -> None:
        circuit.crx(angle, control, target)

    def emit_cry(
        self, circuit: "QuantumCircuit", control: int, target: int, angle: float | Any
    ) -> None:
        circuit.cry(angle, control, target)

    def emit_crz(
        self, circuit: "QuantumCircuit", control: int, target: int, angle: float | Any
    ) -> None:
        circuit.crz(angle, control, target)

    # Measurement
    def emit_measure(self, circuit: "QuantumCircuit", qubit: int, clbit: int) -> None:
        circuit.measure(qubit, clbit)

    # Barrier
    def emit_barrier(self, circuit: "QuantumCircuit", qubits: list[int]) -> None:
        if qubits:
            circuit.barrier(qubits)

    # Sub-circuit support
    def circuit_to_gate(
        self, circuit: "QuantumCircuit", name: str = "U"
    ) -> "Gate | None":
        """Convert circuit to a reusable gate."""
        try:
            return circuit.to_gate(label=name)
        except Exception:
            return None

    def append_gate(
        self,
        circuit: "QuantumCircuit",
        gate: Any,
        qubits: list[int],
    ) -> None:
        """Append a gate to the circuit."""
        circuit.append(gate, qubits)

    def gate_power(self, gate: Any, power: int) -> Any:
        """Create gate raised to a power."""
        return gate.power(power)

    def gate_controlled(self, gate: Any, num_controls: int) -> Any:
        """Create controlled version of a gate."""
        return gate.control(num_controls)

    # Control flow support
    def supports_for_loop(self) -> bool:
        """Qiskit supports native for loops."""
        return True

    def emit_for_loop_start(
        self,
        circuit: "QuantumCircuit",
        indexset: range,
    ) -> Any:
        """Start a native for loop using Qiskit's for_loop context manager.

        Note: The context manager must be used differently than a simple return.
        This method returns the indexset for use with _QiskitForLoopContext.
        """
        return _QiskitForLoopContext(circuit, indexset)

    def emit_for_loop_end(self, circuit: "QuantumCircuit", context: Any) -> None:
        """End is handled by context manager exit."""
        pass

    def supports_if_else(self) -> bool:
        """Qiskit supports native if/else."""
        return True

    def emit_if_start(
        self,
        circuit: "QuantumCircuit",
        clbit: int,
        value: int = 1,
    ) -> Any:
        """Start a native if using Qiskit's if_test context manager."""
        return _QiskitIfContext(circuit, clbit, value)

    def emit_else_start(self, circuit: "QuantumCircuit", context: Any) -> None:
        """Handled by context manager."""
        pass

    def emit_if_end(self, circuit: "QuantumCircuit", context: Any) -> None:
        """Handled by context manager."""
        pass

    def supports_while_loop(self) -> bool:
        """Qiskit supports native while loops."""
        return True

    def emit_while_start(
        self,
        circuit: "QuantumCircuit",
        clbit: int,
        value: int = 1,
    ) -> Any:
        """Start a native while loop."""
        return _QiskitWhileContext(circuit, clbit, value)

    def emit_while_end(self, circuit: "QuantumCircuit", context: Any) -> None:
        """Handled by context manager."""
        pass


class _QiskitForLoopContext:
    """Helper for Qiskit for_loop context management.

    Since StandardEmitPass needs to emit operations inside the loop,
    we need to handle the context manager properly.
    """

    def __init__(self, circuit: "QuantumCircuit", indexset: range):
        self.circuit = circuit
        self.indexset = indexset
        self._context = None
        self._loop_param = None

    def __enter__(self):
        self._context = self.circuit.for_loop(self.indexset)
        self._loop_param = self._context.__enter__()
        return self._loop_param

    def __exit__(self, *args):
        if self._context:
            return self._context.__exit__(*args)
        return False


class _QiskitIfContext:
    """Helper for Qiskit if_test context management."""

    def __init__(self, circuit: "QuantumCircuit", clbit: int, value: int = 1):
        self.circuit = circuit
        self.clbit = clbit
        self.value = value
        self._context = None
        self._else_context = None

    def __enter__(self):
        self._context = self.circuit.if_test(
            (self.circuit.clbits[self.clbit], self.value)
        )
        self._else_context = self._context.__enter__()
        return self

    def start_else(self):
        """Start the else branch."""
        if self._else_context:
            return self._else_context.__enter__()
        return None

    def __exit__(self, *args):
        if self._context:
            return self._context.__exit__(*args)
        return False


class _QiskitWhileContext:
    """Helper for Qiskit while_loop context management."""

    def __init__(self, circuit: "QuantumCircuit", clbit: int, value: int = 1):
        self.circuit = circuit
        self.clbit = clbit
        self.value = value
        self._context = None

    def __enter__(self):
        self._context = self.circuit.while_loop(
            (self.circuit.clbits[self.clbit], self.value)
        )
        self._context.__enter__()
        return self

    def __exit__(self, *args):
        if self._context:
            return self._context.__exit__(*args)
        return False
