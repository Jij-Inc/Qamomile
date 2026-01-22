"""Base test class for transpiler testing.

This module provides TranspilerTestSuite, a base test class that backends
inherit from to get comprehensive gate testing with minimal configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pytest

from .gate_test_specs import (
    GATE_SPECS,
    TEST_ANGLES,
    GateCategory,
    all_zeros_state,
    computational_basis_state,
    compute_expected_statevector,
    statevectors_equal,
    tensor_product,
    identity,
)


class TranspilerTestSuite(ABC):
    """Base test suite for transpiler testing.

    Inherit from this class to get comprehensive gate tests for your backend.
    Override the abstract methods and set the class attributes.

    Class Attributes:
        backend_name: Name of the backend (e.g., "qiskit")
        unsupported_gates: Set of gate names that this backend doesn't support
        requires_classical_bits: Set to True if backend requires classical bits

    Example:
        class TestQiskitTranspiler(TranspilerTestSuite):
            backend_name = "qiskit"
            unsupported_gates = set()

            @classmethod
            def get_emitter(cls):
                from qamomile.qiskit.emitter import QiskitGateEmitter
                return QiskitGateEmitter()

            @classmethod
            def get_simulator(cls):
                from qiskit_aer import AerSimulator
                return AerSimulator(method='statevector')

            @classmethod
            def extract_statevector(cls, result):
                return np.array(result.get_statevector())
    """

    backend_name: str
    unsupported_gates: set[str] = set()
    requires_classical_bits: bool = False

    @classmethod
    @abstractmethod
    def get_emitter(cls) -> Any:
        """Get the GateEmitter instance for this backend.

        Returns:
            Backend-specific GateEmitter instance
        """
        ...

    @classmethod
    @abstractmethod
    def get_simulator(cls) -> Any:
        """Get a simulator for statevector execution.

        Returns:
            Simulator instance for running circuits
        """
        ...

    @classmethod
    @abstractmethod
    def run_circuit_statevector(cls, circuit: Any) -> np.ndarray:
        """Run a circuit and extract statevector.

        Args:
            circuit: Backend-specific circuit

        Returns:
            Statevector as numpy array
        """
        ...

    @classmethod
    def skip_if_unsupported(cls, gate_name: str) -> None:
        """Skip test if gate is unsupported by this backend."""
        if gate_name in cls.unsupported_gates:
            pytest.skip(f"{gate_name} is not supported by {cls.backend_name}")

    # =========================================================================
    # Single-qubit gate tests
    # =========================================================================

    @pytest.mark.parametrize("gate", ["H", "X", "Y", "Z", "S", "T"])
    def test_single_qubit_gate(self, gate: str) -> None:
        """Test single-qubit gates produce correct statevector."""
        self.skip_if_unsupported(gate)
        spec = GATE_SPECS[gate]

        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=1, num_clbits=0)

        # Apply the gate
        emit_method = getattr(emitter, f"emit_{gate.lower()}")
        emit_method(circuit, 0)

        # Get actual statevector
        actual_sv = self.run_circuit_statevector(circuit)

        # Compute expected statevector
        initial_state = all_zeros_state(1)
        expected_sv = compute_expected_statevector(initial_state, spec.matrix_fn())

        assert statevectors_equal(actual_sv, expected_sv), (
            f"{gate} gate produced incorrect statevector.\n"
            f"Expected: {expected_sv}\n"
            f"Actual: {actual_sv}"
        )

    @pytest.mark.parametrize("gate", ["H", "X", "Y", "Z", "S", "T"])
    def test_single_qubit_gate_on_state_one(self, gate: str) -> None:
        """Test single-qubit gates on |1> state."""
        self.skip_if_unsupported(gate)
        spec = GATE_SPECS[gate]

        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=1, num_clbits=0)

        # Prepare |1> state
        emitter.emit_x(circuit, 0)

        # Apply the gate
        emit_method = getattr(emitter, f"emit_{gate.lower()}")
        emit_method(circuit, 0)

        # Get actual statevector
        actual_sv = self.run_circuit_statevector(circuit)

        # Compute expected statevector: X|0> then gate
        initial_state = all_zeros_state(1)
        x_matrix = GATE_SPECS["X"].matrix_fn()
        after_x = compute_expected_statevector(initial_state, x_matrix)
        expected_sv = compute_expected_statevector(after_x, spec.matrix_fn())

        assert statevectors_equal(actual_sv, expected_sv), (
            f"{gate} gate on |1> produced incorrect statevector.\n"
            f"Expected: {expected_sv}\n"
            f"Actual: {actual_sv}"
        )

    # =========================================================================
    # Single-qubit rotation gate tests
    # =========================================================================

    @pytest.mark.parametrize("gate", ["RX", "RY", "RZ", "P"])
    @pytest.mark.parametrize("angle", TEST_ANGLES)
    def test_rotation_gate(self, gate: str, angle: float) -> None:
        """Test single-qubit rotation gates with various angles."""
        self.skip_if_unsupported(gate)
        spec = GATE_SPECS[gate]

        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=1, num_clbits=0)

        # Apply the rotation gate
        emit_method = getattr(emitter, f"emit_{gate.lower()}")
        emit_method(circuit, 0, angle)

        # Get actual statevector
        actual_sv = self.run_circuit_statevector(circuit)

        # Compute expected statevector
        initial_state = all_zeros_state(1)
        expected_sv = compute_expected_statevector(initial_state, spec.matrix_fn(angle))

        assert statevectors_equal(actual_sv, expected_sv), (
            f"{gate}({angle}) produced incorrect statevector.\n"
            f"Expected: {expected_sv}\n"
            f"Actual: {actual_sv}"
        )

    @pytest.mark.parametrize("gate", ["RX", "RY", "RZ", "P"])
    @pytest.mark.parametrize("angle", [np.pi / 4, np.pi / 2])
    def test_rotation_gate_on_superposition(self, gate: str, angle: float) -> None:
        """Test rotation gates on |+> state (superposition)."""
        self.skip_if_unsupported(gate)
        self.skip_if_unsupported("H")
        spec = GATE_SPECS[gate]

        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=1, num_clbits=0)

        # Prepare |+> state
        emitter.emit_h(circuit, 0)

        # Apply the rotation gate
        emit_method = getattr(emitter, f"emit_{gate.lower()}")
        emit_method(circuit, 0, angle)

        # Get actual statevector
        actual_sv = self.run_circuit_statevector(circuit)

        # Compute expected statevector
        initial_state = all_zeros_state(1)
        h_matrix = GATE_SPECS["H"].matrix_fn()
        after_h = compute_expected_statevector(initial_state, h_matrix)
        expected_sv = compute_expected_statevector(after_h, spec.matrix_fn(angle))

        assert statevectors_equal(actual_sv, expected_sv), (
            f"{gate}({angle}) on |+> produced incorrect statevector.\n"
            f"Expected: {expected_sv}\n"
            f"Actual: {actual_sv}"
        )

    # =========================================================================
    # Two-qubit gate tests
    # =========================================================================

    @pytest.mark.parametrize("gate", ["CX", "CZ", "SWAP"])
    @pytest.mark.parametrize(
        "initial_state_index",
        [0, 1, 2, 3],  # |00>, |01>, |10>, |11>
    )
    def test_two_qubit_gate(self, gate: str, initial_state_index: int) -> None:
        """Test two-qubit gates on all computational basis states."""
        self.skip_if_unsupported(gate)
        spec = GATE_SPECS[gate]

        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=2, num_clbits=0)

        # Prepare initial state
        if initial_state_index & 1:  # qubit 0 is |1>
            emitter.emit_x(circuit, 0)
        if initial_state_index & 2:  # qubit 1 is |1>
            emitter.emit_x(circuit, 1)

        # Apply the gate
        emit_method = getattr(emitter, f"emit_{gate.lower()}")
        emit_method(circuit, 0, 1)

        # Get actual statevector
        actual_sv = self.run_circuit_statevector(circuit)

        # Compute expected statevector
        initial_state = computational_basis_state(2, initial_state_index)
        expected_sv = compute_expected_statevector(initial_state, spec.matrix_fn())

        assert statevectors_equal(actual_sv, expected_sv), (
            f"{gate} on |{initial_state_index:02b}> produced incorrect statevector.\n"
            f"Expected: {expected_sv}\n"
            f"Actual: {actual_sv}"
        )

    def test_cx_creates_entanglement(self) -> None:
        """Test that CX creates a Bell state from H|0>|0>."""
        self.skip_if_unsupported("CX")
        self.skip_if_unsupported("H")

        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=2, num_clbits=0)

        # Create Bell state: H on qubit 0, then CNOT
        emitter.emit_h(circuit, 0)
        emitter.emit_cx(circuit, 0, 1)

        # Get actual statevector
        actual_sv = self.run_circuit_statevector(circuit)

        # Expected: (|00> + |11>) / sqrt(2)
        expected_sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

        assert statevectors_equal(actual_sv, expected_sv), (
            f"CX did not create correct Bell state.\n"
            f"Expected: {expected_sv}\n"
            f"Actual: {actual_sv}"
        )

    # =========================================================================
    # Two-qubit rotation gate tests
    # =========================================================================

    @pytest.mark.parametrize("gate", ["CP", "RZZ"])
    @pytest.mark.parametrize("angle", TEST_ANGLES)
    def test_two_qubit_rotation_gate(self, gate: str, angle: float) -> None:
        """Test two-qubit rotation gates."""
        self.skip_if_unsupported(gate)
        spec = GATE_SPECS[gate]

        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=2, num_clbits=0)

        # Apply the gate on |11> to see the rotation effect
        emitter.emit_x(circuit, 0)
        emitter.emit_x(circuit, 1)

        emit_method = getattr(emitter, f"emit_{gate.lower()}")
        emit_method(circuit, 0, 1, angle)

        # Get actual statevector
        actual_sv = self.run_circuit_statevector(circuit)

        # Compute expected statevector
        initial_state = computational_basis_state(2, 3)  # |11>
        expected_sv = compute_expected_statevector(initial_state, spec.matrix_fn(angle))

        assert statevectors_equal(actual_sv, expected_sv), (
            f"{gate}({angle}) on |11> produced incorrect statevector.\n"
            f"Expected: {expected_sv}\n"
            f"Actual: {actual_sv}"
        )

    # =========================================================================
    # Three-qubit gate tests
    # =========================================================================

    @pytest.mark.parametrize(
        "initial_state_index",
        [0, 1, 2, 3, 4, 5, 6, 7],  # All 3-qubit computational basis states
    )
    def test_toffoli_gate(self, initial_state_index: int) -> None:
        """Test Toffoli gate on all computational basis states."""
        self.skip_if_unsupported("TOFFOLI")
        spec = GATE_SPECS["TOFFOLI"]

        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=3, num_clbits=0)

        # Prepare initial state
        if initial_state_index & 1:  # qubit 0 is |1>
            emitter.emit_x(circuit, 0)
        if initial_state_index & 2:  # qubit 1 is |1>
            emitter.emit_x(circuit, 1)
        if initial_state_index & 4:  # qubit 2 is |1>
            emitter.emit_x(circuit, 2)

        # Apply Toffoli: control1=0, control2=1, target=2
        emitter.emit_toffoli(circuit, 0, 1, 2)

        # Get actual statevector
        actual_sv = self.run_circuit_statevector(circuit)

        # Compute expected statevector
        initial_state = computational_basis_state(3, initial_state_index)
        expected_sv = compute_expected_statevector(initial_state, spec.matrix_fn())

        assert statevectors_equal(actual_sv, expected_sv), (
            f"TOFFOLI on |{initial_state_index:03b}> produced incorrect statevector.\n"
            f"Expected: {expected_sv}\n"
            f"Actual: {actual_sv}"
        )

    # =========================================================================
    # Controlled single-qubit gate tests
    # =========================================================================

    @pytest.mark.parametrize("gate", ["CH", "CY"])
    @pytest.mark.parametrize(
        "initial_state_index",
        [0, 1, 2, 3],  # |00>, |01>, |10>, |11>
    )
    def test_controlled_gate(self, gate: str, initial_state_index: int) -> None:
        """Test controlled single-qubit gates."""
        self.skip_if_unsupported(gate)
        spec = GATE_SPECS[gate]

        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=2, num_clbits=0)

        # Prepare initial state
        if initial_state_index & 1:
            emitter.emit_x(circuit, 0)
        if initial_state_index & 2:
            emitter.emit_x(circuit, 1)

        emit_method = getattr(emitter, f"emit_{gate.lower()}")
        emit_method(circuit, 0, 1)

        # Get actual statevector
        actual_sv = self.run_circuit_statevector(circuit)

        # Compute expected statevector
        initial_state = computational_basis_state(2, initial_state_index)
        expected_sv = compute_expected_statevector(initial_state, spec.matrix_fn())

        assert statevectors_equal(actual_sv, expected_sv), (
            f"{gate} on |{initial_state_index:02b}> produced incorrect statevector.\n"
            f"Expected: {expected_sv}\n"
            f"Actual: {actual_sv}"
        )

    @pytest.mark.parametrize("gate", ["CRX", "CRY", "CRZ"])
    @pytest.mark.parametrize("angle", [np.pi / 4, np.pi / 2, np.pi])
    def test_controlled_rotation_gate(self, gate: str, angle: float) -> None:
        """Test controlled rotation gates."""
        self.skip_if_unsupported(gate)
        spec = GATE_SPECS[gate]

        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=2, num_clbits=0)

        # Test on |10> state (control=1, so rotation should apply)
        emitter.emit_x(circuit, 0)

        emit_method = getattr(emitter, f"emit_{gate.lower()}")
        emit_method(circuit, 0, 1, angle)

        # Get actual statevector
        actual_sv = self.run_circuit_statevector(circuit)

        # Compute expected statevector
        initial_state = computational_basis_state(2, 1)  # |01> in our ordering
        expected_sv = compute_expected_statevector(initial_state, spec.matrix_fn(angle))

        assert statevectors_equal(actual_sv, expected_sv), (
            f"{gate}({angle}) on |10> produced incorrect statevector.\n"
            f"Expected: {expected_sv}\n"
            f"Actual: {actual_sv}"
        )

    # =========================================================================
    # Multi-qubit circuit tests
    # =========================================================================

    def test_multi_qubit_sequence(self) -> None:
        """Test a simple sequence of gates on multiple qubits.

        Tests: H(0), CX(0,1) which should produce a Bell state.
        This is a simpler test that doesn't require complex tensor products.
        """
        self.skip_if_unsupported("H")
        self.skip_if_unsupported("CX")

        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=2, num_clbits=0)

        # Build circuit: H(0), CX(0,1) - creates Bell state
        emitter.emit_h(circuit, 0)
        emitter.emit_cx(circuit, 0, 1)

        # Get actual statevector
        actual_sv = self.run_circuit_statevector(circuit)

        # Expected: Bell state (|00> + |11>) / sqrt(2)
        # In little-endian: indices 0 and 3 have equal amplitude
        expected_sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

        assert statevectors_equal(actual_sv, expected_sv), (
            f"Multi-qubit sequence produced incorrect statevector.\n"
            f"Expected: {expected_sv}\n"
            f"Actual: {actual_sv}"
        )

    # =========================================================================
    # Circuit creation tests
    # =========================================================================

    def test_create_empty_circuit(self) -> None:
        """Test creating an empty circuit."""
        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=2, num_clbits=1)
        # Just verify it doesn't raise an error
        assert circuit is not None

    def test_barrier_no_error(self) -> None:
        """Test that barrier doesn't cause errors."""
        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=2, num_clbits=0)

        emitter.emit_h(circuit, 0)
        emitter.emit_barrier(circuit, [0, 1])
        emitter.emit_h(circuit, 1)

        # Verify circuit still works
        actual_sv = self.run_circuit_statevector(circuit)
        assert actual_sv is not None


class HamiltonianTestMixin:
    """Mixin for testing Hamiltonian transpilation.

    Add this mixin to test classes that support Hamiltonian transpilation.
    Requires implementation of `get_transpiler` method.
    Note: Tests will be skipped if the transpiler doesn't have `transpile_hamiltonian`.
    """

    @classmethod
    @abstractmethod
    def get_transpiler(cls) -> Any:
        """Get the transpiler instance.

        Returns:
            Backend-specific transpiler instance
        """
        ...

    @classmethod
    def _skip_if_no_hamiltonian_support(cls) -> None:
        """Skip test if transpiler doesn't support Hamiltonian transpilation."""
        transpiler = cls.get_transpiler()
        if not hasattr(transpiler, "transpile_hamiltonian"):
            pytest.skip(
                f"{cls.backend_name} transpiler does not support transpile_hamiltonian"
            )

    def test_simple_hamiltonian(self) -> None:
        """Test transpiling a simple single-qubit Hamiltonian."""
        self._skip_if_no_hamiltonian_support()
        from qamomile.core.operator import Hamiltonian, X, Z

        h = Hamiltonian()
        h += 1.0 * X(0)
        h += 2.0 * Z(1)

        transpiler = self.get_transpiler()
        result = transpiler.transpile_hamiltonian(h)
        assert result is not None

    def test_multi_qubit_hamiltonian(self) -> None:
        """Test transpiling a multi-qubit Hamiltonian."""
        self._skip_if_no_hamiltonian_support()
        from qamomile.core.operator import Hamiltonian, X, Y, Z

        h = Hamiltonian()
        h += X(0) * Z(1)
        h += Y(0) * Y(1)
        h += 0.5 * Z(0) * Z(1) * Z(2)

        transpiler = self.get_transpiler()
        result = transpiler.transpile_hamiltonian(h)
        assert result is not None

    def test_hamiltonian_with_constant(self) -> None:
        """Test transpiling a Hamiltonian with a constant term."""
        self._skip_if_no_hamiltonian_support()
        from qamomile.core.operator import Hamiltonian, Z

        h = Hamiltonian()
        h += 1.5 * Z(0)
        h.constant = 0.5

        transpiler = self.get_transpiler()
        result = transpiler.transpile_hamiltonian(h)
        assert result is not None

    def test_empty_hamiltonian(self) -> None:
        """Test transpiling an empty Hamiltonian."""
        self._skip_if_no_hamiltonian_support()
        from qamomile.core.operator import Hamiltonian

        h = Hamiltonian()

        transpiler = self.get_transpiler()
        result = transpiler.transpile_hamiltonian(h)
        assert result is not None
