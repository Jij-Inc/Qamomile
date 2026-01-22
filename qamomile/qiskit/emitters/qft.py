"""Qiskit QFT/IQFT emitter using native qiskit.circuit.library.QFT."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)


class QiskitQFTEmitter:
    """Emitter for QFT and IQFT using Qiskit's native library.

    Uses qiskit.circuit.library.QFT which is a stable, non-deprecated API.
    This provides optimized implementations that benefit from Qiskit's
    circuit optimization passes.

    Supports:
    - CompositeGateType.QFT: Quantum Fourier Transform
    - CompositeGateType.IQFT: Inverse Quantum Fourier Transform
    """

    def can_emit(self, gate_type: CompositeGateType) -> bool:
        """Check if this emitter can handle the given gate type.

        Args:
            gate_type: The CompositeGateType to check

        Returns:
            True for QFT and IQFT, False otherwise
        """
        return gate_type in (CompositeGateType.QFT, CompositeGateType.IQFT)

    def emit(
        self,
        circuit: "QuantumCircuit",
        op: CompositeGateOperation,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> bool:
        """Emit QFT or IQFT to the circuit using Qiskit's native library.

        Args:
            circuit: The Qiskit QuantumCircuit to emit to
            op: The CompositeGateOperation (QFT or IQFT)
            qubit_indices: Physical qubit indices for the operation
            bindings: Parameter bindings (unused for QFT/IQFT)

        Returns:
            True if emission succeeded, False to fall back to manual
        """
        if not qubit_indices:
            return False

        try:
            from qiskit.circuit.library import QFT as QiskitQFT
        except ImportError:
            return False  # Qiskit not available, fall back to manual

        num_qubits = len(qubit_indices)

        try:
            if op.gate_type == CompositeGateType.QFT:
                qft = QiskitQFT(num_qubits)
                circuit.compose(qft, qubit_indices, inplace=True)
                return True

            elif op.gate_type == CompositeGateType.IQFT:
                iqft = QiskitQFT(num_qubits).inverse()
                circuit.compose(iqft, qubit_indices, inplace=True)
                return True

        except Exception:
            return False  # Any error, fall back to manual

        return False

    def emit_raw_iqft(
        self,
        circuit: "QuantumCircuit",
        qubit_indices: list[int],
    ) -> bool:
        """Emit IQFT directly without CompositeGateOperation.

        Used for recursive native emit within other composite gates (e.g., QPE).

        Args:
            circuit: The Qiskit QuantumCircuit to emit to
            qubit_indices: Physical qubit indices for the IQFT

        Returns:
            True if emission succeeded, False to fall back to manual
        """
        if not qubit_indices:
            return False

        try:
            from qiskit.circuit.library import QFT as QiskitQFT

            iqft = QiskitQFT(len(qubit_indices)).inverse()
            circuit.compose(iqft, qubit_indices, inplace=True)
            return True
        except Exception:
            return False

    def emit_raw_qft(
        self,
        circuit: "QuantumCircuit",
        qubit_indices: list[int],
    ) -> bool:
        """Emit QFT directly without CompositeGateOperation.

        Used for recursive native emit within other composite gates.

        Args:
            circuit: The Qiskit QuantumCircuit to emit to
            qubit_indices: Physical qubit indices for the QFT

        Returns:
            True if emission succeeded, False to fall back to manual
        """
        if not qubit_indices:
            return False

        try:
            from qiskit.circuit.library import QFT as QiskitQFT

            qft = QiskitQFT(len(qubit_indices))
            circuit.compose(qft, qubit_indices, inplace=True)
            return True
        except Exception:
            return False
