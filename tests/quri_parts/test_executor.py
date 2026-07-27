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

    def test_estimate_accepts_numpy_parameter_array(self) -> None:
        """NumPy parameter arrays do not enter ambiguous truth-value logic."""
        import qamomile.observable as qm_o

        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        theta = emitter.create_parameter("theta")
        emitter.emit_rx(circuit, 0, theta)

        result = QuriPartsExecutor().estimate(
            circuit,
            qm_o.Z(0),
            params=np.array([0.3]),
        )

        assert np.isclose(result, np.cos(0.3), atol=1e-10)

    def test_custom_estimator_is_used_after_parameter_binding(self) -> None:
        """Binding a circuit cannot silently replace an explicit estimator."""
        import quri_parts.core.operator as qp_o

        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        theta = emitter.create_parameter("theta")
        emitter.emit_rx(circuit, 0, theta)
        metadata = ParameterMetadata(
            parameters=[ParameterInfo("theta", "theta", None, theta)]
        )
        bound = QuriPartsExecutor().bind_parameters(
            circuit,
            {"theta": 0.3},
            metadata,
        )
        calls: list[tuple[object, object]] = []

        class Estimate:
            """Minimal QURI-compatible estimate result."""

            value = complex(7.0)

        def estimator(operator: object, state: object) -> Estimate:
            """Record the configured non-parametric estimator call."""
            calls.append((operator, state))
            return Estimate()

        operator = qp_o.Operator({qp_o.pauli_label([(0, 3)]): 1.0})
        result = QuriPartsExecutor(bound_estimator=estimator).estimate_expectation(
            bound,
            operator,
            [],
        )

        assert result == 7.0
        assert len(calls) == 1

    def test_parametric_estimator_on_bound_state_has_clear_error(self) -> None:
        """A three-argument estimator is not called with a bound state."""
        import quri_parts.core.operator as qp_o
        from quri_parts.qulacs.estimator import (
            create_qulacs_vector_parametric_estimator,
        )

        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        theta = emitter.create_parameter("theta")
        emitter.emit_rx(circuit, 0, theta)
        operator = qp_o.Operator({qp_o.pauli_label([(0, 3)]): 1.0})
        executor = QuriPartsExecutor(
            estimator=create_qulacs_vector_parametric_estimator()
        )

        with pytest.raises(
            QamomileQuriPartsTranspileError,
            match="requires a two-argument non-parametric estimator",
        ):
            executor.estimate_expectation(
                circuit.bind_parameters([0.3]),
                operator,
                [],
            )

    def test_default_parametric_estimator_does_not_shadow_bound_path(self) -> None:
        """Lazy parametric-estimator creation preserves later bound estimates."""
        import quri_parts.core.operator as qp_o

        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        theta = emitter.create_parameter("theta")
        emitter.emit_rx(circuit, 0, theta)
        operator = qp_o.Operator({qp_o.pauli_label([(0, 3)]): 1.0})
        executor = QuriPartsExecutor()

        unbound_result = executor.estimate_expectation(circuit, operator, [0.3])
        bound_result = executor.estimate_expectation(
            circuit.bind_parameters([0.3]),
            operator,
            [],
        )

        assert np.isclose(unbound_result, np.cos(0.3), atol=1e-10)
        assert np.isclose(bound_result, unbound_result, atol=1e-10)
