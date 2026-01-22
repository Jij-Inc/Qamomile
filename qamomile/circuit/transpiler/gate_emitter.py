"""GateEmitter protocol for backend-agnostic gate emission.

This module defines the GateEmitter protocol that backends implement
to emit individual quantum gates. The StandardEmitPass uses this protocol
to orchestrate circuit generation without backend-specific code.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")  # Backend circuit type


class GateKind(Enum):
    """Classification of gates for emission."""

    # Single-qubit gates (no parameters)
    H = auto()
    X = auto()
    Y = auto()
    Z = auto()
    S = auto()
    T = auto()

    # Single-qubit rotation gates (with angle parameter)
    RX = auto()
    RY = auto()
    RZ = auto()
    P = auto()  # Phase gate

    # Two-qubit gates (no parameters)
    CX = auto()
    CZ = auto()
    SWAP = auto()

    # Two-qubit rotation gates (with angle parameter)
    CP = auto()  # Controlled phase
    RZZ = auto()

    # Three-qubit gates
    TOFFOLI = auto()

    # Controlled versions of single-qubit gates
    CH = auto()
    CY = auto()
    CRX = auto()
    CRY = auto()
    CRZ = auto()

    # Measurement
    MEASURE = auto()


@dataclass
class GateSpec:
    """Specification for a gate type.

    Attributes:
        kind: The gate kind
        num_qubits: Number of qubit operands
        has_angle: Whether the gate has an angle parameter
        num_controls: Number of control qubits (for controlled gates)
    """

    kind: GateKind
    num_qubits: int
    has_angle: bool = False
    num_controls: int = 0


# Registry of gate specifications
GATE_SPECS: dict[GateKind, GateSpec] = {
    # Single-qubit gates
    GateKind.H: GateSpec(GateKind.H, num_qubits=1),
    GateKind.X: GateSpec(GateKind.X, num_qubits=1),
    GateKind.Y: GateSpec(GateKind.Y, num_qubits=1),
    GateKind.Z: GateSpec(GateKind.Z, num_qubits=1),
    GateKind.S: GateSpec(GateKind.S, num_qubits=1),
    GateKind.T: GateSpec(GateKind.T, num_qubits=1),
    # Single-qubit rotation gates
    GateKind.RX: GateSpec(GateKind.RX, num_qubits=1, has_angle=True),
    GateKind.RY: GateSpec(GateKind.RY, num_qubits=1, has_angle=True),
    GateKind.RZ: GateSpec(GateKind.RZ, num_qubits=1, has_angle=True),
    GateKind.P: GateSpec(GateKind.P, num_qubits=1, has_angle=True),
    # Two-qubit gates
    GateKind.CX: GateSpec(GateKind.CX, num_qubits=2),
    GateKind.CZ: GateSpec(GateKind.CZ, num_qubits=2),
    GateKind.SWAP: GateSpec(GateKind.SWAP, num_qubits=2),
    # Two-qubit rotation gates
    GateKind.CP: GateSpec(GateKind.CP, num_qubits=2, has_angle=True),
    GateKind.RZZ: GateSpec(GateKind.RZZ, num_qubits=2, has_angle=True),
    # Three-qubit gates
    GateKind.TOFFOLI: GateSpec(GateKind.TOFFOLI, num_qubits=3),
    # Controlled single-qubit gates
    GateKind.CH: GateSpec(GateKind.CH, num_qubits=2, num_controls=1),
    GateKind.CY: GateSpec(GateKind.CY, num_qubits=2, num_controls=1),
    GateKind.CRX: GateSpec(GateKind.CRX, num_qubits=2, has_angle=True, num_controls=1),
    GateKind.CRY: GateSpec(GateKind.CRY, num_qubits=2, has_angle=True, num_controls=1),
    GateKind.CRZ: GateSpec(GateKind.CRZ, num_qubits=2, has_angle=True, num_controls=1),
    # Measurement
    GateKind.MEASURE: GateSpec(GateKind.MEASURE, num_qubits=1),
}


@runtime_checkable
class GateEmitter(Protocol[T]):
    """Protocol for backend-specific gate emission.

    Each backend implements this protocol to emit individual gates
    to their circuit representation.

    Type parameter T is the backend's circuit type (e.g., QuantumCircuit).
    """

    @abstractmethod
    def create_circuit(self, num_qubits: int, num_clbits: int) -> T:
        """Create a new empty circuit.

        Args:
            num_qubits: Number of qubits in the circuit
            num_clbits: Number of classical bits in the circuit

        Returns:
            A new backend-specific circuit object
        """
        ...

    @abstractmethod
    def create_parameter(self, name: str) -> Any:
        """Create a symbolic parameter for the backend.

        Args:
            name: Parameter name (e.g., "gammas[0]")

        Returns:
            Backend-specific parameter object
        """
        ...

    # Single-qubit gates (no parameters)
    @abstractmethod
    def emit_h(self, circuit: T, qubit: int) -> None:
        """Emit Hadamard gate."""
        ...

    @abstractmethod
    def emit_x(self, circuit: T, qubit: int) -> None:
        """Emit Pauli-X gate."""
        ...

    @abstractmethod
    def emit_y(self, circuit: T, qubit: int) -> None:
        """Emit Pauli-Y gate."""
        ...

    @abstractmethod
    def emit_z(self, circuit: T, qubit: int) -> None:
        """Emit Pauli-Z gate."""
        ...

    @abstractmethod
    def emit_s(self, circuit: T, qubit: int) -> None:
        """Emit S gate (√Z)."""
        ...

    @abstractmethod
    def emit_t(self, circuit: T, qubit: int) -> None:
        """Emit T gate (√S)."""
        ...

    # Single-qubit rotation gates
    @abstractmethod
    def emit_rx(self, circuit: T, qubit: int, angle: float | Any) -> None:
        """Emit RX rotation gate.

        Args:
            circuit: The circuit to emit to
            qubit: Target qubit index
            angle: Rotation angle (float or backend parameter)
        """
        ...

    @abstractmethod
    def emit_ry(self, circuit: T, qubit: int, angle: float | Any) -> None:
        """Emit RY rotation gate."""
        ...

    @abstractmethod
    def emit_rz(self, circuit: T, qubit: int, angle: float | Any) -> None:
        """Emit RZ rotation gate."""
        ...

    @abstractmethod
    def emit_p(self, circuit: T, qubit: int, angle: float | Any) -> None:
        """Emit Phase gate (P(θ) = diag(1, e^(iθ)))."""
        ...

    # Two-qubit gates
    @abstractmethod
    def emit_cx(self, circuit: T, control: int, target: int) -> None:
        """Emit CNOT gate."""
        ...

    @abstractmethod
    def emit_cz(self, circuit: T, control: int, target: int) -> None:
        """Emit CZ gate."""
        ...

    @abstractmethod
    def emit_swap(self, circuit: T, qubit1: int, qubit2: int) -> None:
        """Emit SWAP gate."""
        ...

    # Two-qubit rotation gates
    @abstractmethod
    def emit_cp(
        self, circuit: T, control: int, target: int, angle: float | Any
    ) -> None:
        """Emit controlled-Phase gate."""
        ...

    @abstractmethod
    def emit_rzz(
        self, circuit: T, qubit1: int, qubit2: int, angle: float | Any
    ) -> None:
        """Emit RZZ gate (exp(-i * θ/2 * Z⊗Z))."""
        ...

    # Three-qubit gates
    @abstractmethod
    def emit_toffoli(
        self, circuit: T, control1: int, control2: int, target: int
    ) -> None:
        """Emit Toffoli (CCX) gate."""
        ...

    # Controlled single-qubit gates
    @abstractmethod
    def emit_ch(self, circuit: T, control: int, target: int) -> None:
        """Emit controlled-Hadamard gate."""
        ...

    @abstractmethod
    def emit_cy(self, circuit: T, control: int, target: int) -> None:
        """Emit controlled-Y gate."""
        ...

    @abstractmethod
    def emit_crx(
        self, circuit: T, control: int, target: int, angle: float | Any
    ) -> None:
        """Emit controlled-RX gate."""
        ...

    @abstractmethod
    def emit_cry(
        self, circuit: T, control: int, target: int, angle: float | Any
    ) -> None:
        """Emit controlled-RY gate."""
        ...

    @abstractmethod
    def emit_crz(
        self, circuit: T, control: int, target: int, angle: float | Any
    ) -> None:
        """Emit controlled-RZ gate."""
        ...

    # Measurement
    @abstractmethod
    def emit_measure(self, circuit: T, qubit: int, clbit: int) -> None:
        """Emit measurement operation."""
        ...

    # Barrier (optional, for visual separation)
    @abstractmethod
    def emit_barrier(self, circuit: T, qubits: list[int]) -> None:
        """Emit barrier on specified qubits."""
        ...

    # Sub-circuit support
    @abstractmethod
    def circuit_to_gate(self, circuit: T, name: str = "U") -> Any:
        """Convert a circuit to a reusable gate.

        Args:
            circuit: The circuit to convert
            name: Label for the gate

        Returns:
            Backend-specific gate object, or None if not supported
        """
        ...

    @abstractmethod
    def append_gate(
        self,
        circuit: T,
        gate: Any,
        qubits: list[int],
    ) -> None:
        """Append a gate to the circuit.

        Args:
            circuit: The circuit to append to
            gate: The gate to append (from circuit_to_gate)
            qubits: Target qubit indices
        """
        ...

    @abstractmethod
    def gate_power(self, gate: Any, power: int) -> Any:
        """Create gate raised to a power (U^n).

        Args:
            gate: The gate to raise to a power
            power: The power to raise to

        Returns:
            New gate representing gate^power
        """
        ...

    @abstractmethod
    def gate_controlled(self, gate: Any, num_controls: int) -> Any:
        """Create controlled version of a gate.

        Args:
            gate: The gate to control
            num_controls: Number of control qubits

        Returns:
            New controlled gate
        """
        ...

    # Control flow support (optional - backends can return False to fall back)
    def supports_for_loop(self) -> bool:
        """Check if backend supports native for loops."""
        return False

    def emit_for_loop_start(
        self,
        circuit: T,
        indexset: range,
    ) -> Any:
        """Start a native for loop context.

        Returns a context manager or loop parameter, depending on backend.
        """
        raise NotImplementedError("Backend does not support native for loops")

    def emit_for_loop_end(self, circuit: T, context: Any) -> None:
        """End a native for loop context."""
        raise NotImplementedError("Backend does not support native for loops")

    def supports_if_else(self) -> bool:
        """Check if backend supports native if/else."""
        return False

    def emit_if_start(
        self,
        circuit: T,
        clbit: int,
        value: int = 1,
    ) -> Any:
        """Start a native if context.

        Returns context for the if/else block.
        """
        raise NotImplementedError("Backend does not support native if/else")

    def emit_else_start(self, circuit: T, context: Any) -> None:
        """Start the else branch."""
        raise NotImplementedError("Backend does not support native if/else")

    def emit_if_end(self, circuit: T, context: Any) -> None:
        """End the if/else block."""
        raise NotImplementedError("Backend does not support native if/else")

    def supports_while_loop(self) -> bool:
        """Check if backend supports native while loops."""
        return False

    def emit_while_start(
        self,
        circuit: T,
        clbit: int,
        value: int = 1,
    ) -> Any:
        """Start a native while loop context."""
        raise NotImplementedError("Backend does not support native while loops")

    def emit_while_end(self, circuit: T, context: Any) -> None:
        """End the while loop context."""
        raise NotImplementedError("Backend does not support native while loops")
