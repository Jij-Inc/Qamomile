"""Tests for QuriPartsExecutor.

Covers bind_parameters (including error case) and estimate_expectation.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.quri_parts

# Skip entire module if required QURI Parts components are not installed.
pytest.importorskip("quri_parts.circuit")
pytest.importorskip("quri_parts.core.operator")
pytest.importorskip("quri_parts.qulacs")

from qamomile.circuit.transpiler.executable import (  # noqa: E402
    ParameterInfo,
    ParameterMetadata,
)
from qamomile.quri_parts import (  # noqa: E402
    QuriPartsExecutor,
    QuriPartsGateEmitter,
)
from qamomile.quri_parts.exceptions import QamomileQuriPartsTranspileError  # noqa: E402


class TestBindParameters:
    """Tests for QuriPartsExecutor.bind_parameters."""

    def _make_parametric_circuit(self):
        """Create a simple parametric circuit with one parameter 'theta'.

        Returns:
            tuple: (circuit, ParameterMetadata)
        """
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        param = emitter.create_parameter("theta")
        emitter.emit_rx(circuit, 0, param)
        metadata = ParameterMetadata(
            parameters=[
                ParameterInfo(
                    name="theta",
                    array_name="theta",
                    index=None,
                    backend_param=param,
                )
            ]
        )
        return circuit, metadata

    def test_bind_parameters_valid(self) -> None:
        """Verify binding a valid parameter produces a bound circuit."""
        circuit, metadata = self._make_parametric_circuit()
        executor = QuriPartsExecutor()

        bound = executor.bind_parameters(circuit, {"theta": 1.0}, metadata)
        # Bound circuit should be usable (qubit_count preserved)
        assert bound.qubit_count == 1

    @pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi, 2 * np.pi])
    def test_bind_parameters_various_angles(self, angle: float) -> None:
        """Verify binding works for various angle values including boundaries."""
        circuit, metadata = self._make_parametric_circuit()
        executor = QuriPartsExecutor()

        bound = executor.bind_parameters(circuit, {"theta": angle}, metadata)
        assert bound.qubit_count == 1

    def test_bind_parameters_missing_raises(self) -> None:
        """Verify missing parameter raises QamomileQuriPartsTranspileError."""
        circuit, metadata = self._make_parametric_circuit()
        executor = QuriPartsExecutor()

        with pytest.raises(QamomileQuriPartsTranspileError, match="Missing binding"):
            executor.bind_parameters(circuit, {}, metadata)

    def test_bind_parameters_wrong_name_raises(self) -> None:
        """Verify incorrect parameter name raises QamomileQuriPartsTranspileError."""
        circuit, metadata = self._make_parametric_circuit()
        executor = QuriPartsExecutor()

        with pytest.raises(QamomileQuriPartsTranspileError, match="theta"):
            executor.bind_parameters(circuit, {"phi": 1.0}, metadata)


class TestExecute:
    """Tests for QuriPartsExecutor.execute."""

    def test_execute_simple_circuit(self) -> None:
        """Verify executing a trivial circuit returns valid bitstring counts."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)

        # |0> state, should always measure 0
        bound = circuit.bind_parameters([])
        executor = QuriPartsExecutor()
        counts = executor.execute(bound, shots=100)

        assert "0" in counts
        assert counts["0"] == 100

    def test_execute_x_gate(self) -> None:
        """Verify X gate flips qubit from |0> to |1>."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        emitter.emit_x(circuit, 0)

        bound = circuit.bind_parameters([])
        executor = QuriPartsExecutor()
        counts = executor.execute(bound, shots=100)

        assert "1" in counts
        assert counts["1"] == 100

    def test_execute_bell_state(self) -> None:
        """Verify Bell state produces roughly equal |00> and |11> counts."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)
        emitter.emit_h(circuit, 0)
        emitter.emit_cx(circuit, 0, 1)

        bound = circuit.bind_parameters([])
        executor = QuriPartsExecutor()
        counts = executor.execute(bound, shots=10000)

        # Bell state should produce only |00> and |11>
        for key in counts:
            assert key in ("00", "11")
        assert counts.get("00", 0) > 0
        assert counts.get("11", 0) > 0


class TestEstimateExpectation:
    """Tests for QuriPartsExecutor.estimate_expectation."""

    def test_z_expectation_on_zero_state(self) -> None:
        """Verify <0|Z|0> = 1.0."""
        import quri_parts.core.operator as qp_o

        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        # No gates -> |0> state

        hamiltonian = qp_o.Operator({qp_o.pauli_label([(0, 3)]): 1.0})
        executor = QuriPartsExecutor()

        result = executor.estimate_expectation(circuit, hamiltonian, [])
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_z_expectation_on_one_state(self) -> None:
        """Verify <1|Z|1> = -1.0."""
        import quri_parts.core.operator as qp_o

        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        emitter.emit_x(circuit, 0)

        hamiltonian = qp_o.Operator({qp_o.pauli_label([(0, 3)]): 1.0})
        executor = QuriPartsExecutor()

        result = executor.estimate_expectation(circuit, hamiltonian, [])
        assert np.isclose(result, -1.0, atol=1e-10)

    def test_x_expectation_on_plus_state(self) -> None:
        """Verify <+|X|+> = 1.0."""
        import quri_parts.core.operator as qp_o

        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        emitter.emit_h(circuit, 0)

        hamiltonian = qp_o.Operator({qp_o.pauli_label([(0, 1)]): 1.0})
        executor = QuriPartsExecutor()

        result = executor.estimate_expectation(circuit, hamiltonian, [])
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_estimate_with_qamomile_hamiltonian(self) -> None:
        """Verify estimate() auto-converts qamomile Hamiltonian to Operator."""
        import qamomile.observable as qm_o

        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        # No gates -> |0> state; <0|Z|0> = 1.0

        hamiltonian = qm_o.Z(0)
        executor = QuriPartsExecutor()

        result = executor.estimate(circuit, hamiltonian, params=[])
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_estimate_zero_hamiltonian(self) -> None:
        """Verify estimate() with empty Hamiltonian returns 0.0."""
        import qamomile.observable as qm_o

        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)

        hamiltonian = qm_o.Hamiltonian()  # empty — no terms, zero constant
        executor = QuriPartsExecutor()

        result = executor.estimate(circuit, hamiltonian, params=[])
        assert np.isclose(result, 0.0, atol=1e-10)
