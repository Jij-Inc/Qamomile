"""Tests for QuriPartsTranspiler (new qamomile.circuit API)."""

import math

import numpy as np
import pytest

pytestmark = pytest.mark.quri_parts

# Skip entire module if QURI Parts circuit support is not installed.
pytest.importorskip("quri_parts.circuit")

import qamomile.circuit as qmc  # noqa: E402
from qamomile.circuit.transpiler.circuit_ir import (  # noqa: E402
    CircuitBuilder,
    ParameterExpr,
    ReusableCircuit,
)
from qamomile.circuit.transpiler.errors import EmitError  # noqa: E402
from qamomile.circuit.transpiler.gate_emitter import GateKind  # noqa: E402
from qamomile.quri_parts import (  # noqa: E402
    QuriPartsExecutor,
    QuriPartsGateEmitter,
)
from qamomile.quri_parts.materializer import QuriPartsMaterializer  # noqa: E402


def _phase_only_call(
    *,
    controls: int,
    inverse: bool = False,
    power: int = 1,
) -> CircuitBuilder:
    """Build a controlled reusable identity with a symbolic global phase."""
    body = CircuitBuilder(1, 0, name="phase-only")
    body.add_global_phase(ParameterExpr("theta"))
    caller = CircuitBuilder(controls + 1, 0, name="phase-caller")
    for control in range(controls):
        caller.append_gate(GateKind.H, (control,))
    caller.append_call(
        ReusableCircuit(
            body.freeze(),
            "phase-only",
            controls=controls,
            inverse=inverse,
            power=power,
        ),
        tuple(range(controls + 1)),
    )
    return caller


def _run_statevector(circuit, parameter_values: list[float]) -> np.ndarray:
    """Evaluate a possibly parametric QURI Parts circuit with Qulacs."""
    pytest.importorskip("quri_parts.qulacs")
    from quri_parts.core.state import GeneralCircuitQuantumState
    from quri_parts.qulacs.simulator import evaluate_state_to_vector

    bound = circuit.bind_parameters(parameter_values)
    state = GeneralCircuitQuantumState(bound.qubit_count, bound)
    return np.asarray(evaluate_state_to_vector(state).vector)


def _expected_phase_kickback(
    controls: int,
    phase: float,
) -> np.ndarray:
    """Return the phase-kickback state on controls with an idle target."""
    state = np.zeros(2 ** (controls + 1), dtype=np.complex128)
    state[: 2**controls] = 1.0 / np.sqrt(2**controls)
    state[(1 << controls) - 1] *= np.exp(1j * phase)
    return state


class TestQuriPartsGateEmitter:
    """Test the gate emitter directly."""

    def test_create_circuit(self) -> None:
        """Test circuit creation."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(3, 2)
        assert circuit.qubit_count == 3
        # Note: QURI Parts doesn't track classical bits in the circuit

    def test_single_qubit_gates(self) -> None:
        """Test single qubit gate emission."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_h(circuit, 0)
        emitter.emit_x(circuit, 1)
        emitter.emit_y(circuit, 0)
        emitter.emit_z(circuit, 1)
        emitter.emit_s(circuit, 0)
        emitter.emit_t(circuit, 1)

        gates = list(circuit.gates)
        assert len(gates) == 6

    def test_rotation_gates_with_float(self) -> None:
        """Test rotation gates with float angles."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)

        emitter.emit_rx(circuit, 0, math.pi / 2)
        emitter.emit_ry(circuit, 0, math.pi / 4)
        emitter.emit_rz(circuit, 0, math.pi / 8)

        gates = list(circuit.gates)
        assert len(gates) == 3

    def test_rotation_gates_with_parameter(self) -> None:
        """Test rotation gates with parametric angles."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)

        param = emitter.create_parameter("theta")
        emitter.emit_rx(circuit, 0, param)
        emitter.emit_ry(circuit, 0, param)

        # Circuit should have parametric gates
        assert circuit.parameter_count == 1

    def test_two_qubit_gates(self) -> None:
        """Test two qubit gate emission."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_cx(circuit, 0, 1)
        emitter.emit_cz(circuit, 0, 1)
        emitter.emit_swap(circuit, 0, 1)

        gates = list(circuit.gates)
        assert len(gates) == 3

    def test_controlled_phase_decomposition(self) -> None:
        """Test the exact CP decomposition, including its global factor."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_cp(circuit, 0, 1, math.pi / 2)

        # The final native U1 combines RZ(control, theta / 2) with the
        # decomposition's otherwise missing exp(i * theta / 4) factor.
        gates = list(circuit.gates)
        assert len(gates) == 5

        state = _run_statevector(circuit, [])
        assert np.allclose(
            state,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
            rtol=0.0,
            atol=1e-10,
        )

    def test_parametric_phase_gate_preserves_its_global_factor(self) -> None:
        """Parametric P remains exact when QURI lowers it through RZ."""
        emitter = QuriPartsGateEmitter(phase_carrier=1)
        circuit = emitter.create_circuit(2, 0)
        theta = emitter.create_parameter("theta")

        emitter.emit_p(circuit, 0, theta)

        angle = 0.37
        state = _run_statevector(circuit, [angle])
        assert np.allclose(
            state,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
            rtol=0.0,
            atol=1e-10,
        )

    def test_concrete_global_phase_uses_an_existing_qubit(self) -> None:
        """A U1 and RZ pair phases an arbitrary state without a carrier."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)

        emitter.emit_h(circuit, 0)
        emitter.emit_global_phase(circuit, 0.25, carrier=0)

        assert len(circuit.gates) == 3
        state = _run_statevector(circuit, [])
        assert np.allclose(
            state,
            np.exp(0.25j) * np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0),
            rtol=0.0,
            atol=1e-10,
        )

    def test_global_phase_without_any_carrier_fails(self) -> None:
        """Scalar phase emission fails closed when no carrier is available."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(0, 0)

        with pytest.raises(EmitError, match="phase-carrier"):
            emitter.emit_global_phase(circuit, 0.25)

    def test_rzz_gate(self) -> None:
        """Test RZZ gate emission."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_rzz(circuit, 0, 1, math.pi / 4)

        gates = list(circuit.gates)
        assert len(gates) == 1

    def test_toffoli_gate(self) -> None:
        """Test Toffoli gate emission."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(3, 0)

        emitter.emit_toffoli(circuit, 0, 1, 2)

        gates = list(circuit.gates)
        assert len(gates) == 1

    def test_controlled_single_qubit_decomposition(self) -> None:
        """Test controlled single-qubit gate decompositions."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_ch(circuit, 0, 1)
        emitter.emit_cy(circuit, 0, 1)
        emitter.emit_crx(circuit, 0, 1, math.pi / 2)
        emitter.emit_cry(circuit, 0, 1, math.pi / 2)
        emitter.emit_crz(circuit, 0, 1, math.pi / 2)

        # All controlled gates should decompose
        gates = list(circuit.gates)
        assert len(gates) > 5  # Each gate decomposes to multiple gates

    def test_measure_is_noop(self) -> None:
        """Test that measure is a no-op (doesn't add gates)."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 1)

        emitter.emit_h(circuit, 0)
        emitter.emit_measure(circuit, 0, 0)

        gates = list(circuit.gates)
        # Only H gate, measure is no-op
        assert len(gates) == 1

    def test_barrier_is_noop(self) -> None:
        """Test that barrier is a no-op (doesn't add gates)."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_h(circuit, 0)
        emitter.emit_barrier(circuit, [0, 1])
        emitter.emit_h(circuit, 1)

        gates = list(circuit.gates)
        # Only 2 H gates, barrier is no-op
        assert len(gates) == 2

    def test_supports_no_native_control_flow(self) -> None:
        """Test that emitter reports no native control flow support."""
        emitter = QuriPartsGateEmitter()
        assert not emitter.supports_for_loop()
        assert not emitter.supports_if_else()
        assert not emitter.supports_while_loop()

    def test_circuit_to_gate_returns_none(self) -> None:
        """Test that circuit_to_gate returns None (not supported)."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        emitter.emit_h(circuit, 0)

        gate = emitter.circuit_to_gate(circuit, "test")
        assert gate is None

    def test_gate_power_returns_none(self) -> None:
        """Test that gate_power returns None (not supported)."""
        emitter = QuriPartsGateEmitter()
        result = emitter.gate_power(None, 2)
        assert result is None

    def test_gate_controlled_returns_none(self) -> None:
        """Test that gate_controlled returns None (not supported)."""
        emitter = QuriPartsGateEmitter()
        result = emitter.gate_controlled(None, 1)
        assert result is None


class TestQuriPartsTranspiler:
    """Test the transpiler configuration."""

    def test_transpiler_creation(self) -> None:
        """Test transpiler can be created."""
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        assert transpiler is not None

    def test_executor_creation(self) -> None:
        """Test executor can be created from transpiler."""
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        executor = transpiler.executor()
        assert executor is not None

    def test_executor_with_custom_sampler(self) -> None:
        """Test executor with custom sampler."""
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        # Custom sampler can be passed to executor
        executor = transpiler.executor(sampler=None, estimator=None)
        assert executor is not None

    def test_standalone_phase_is_preserved_without_changing_the_abi(self) -> None:
        """A root phase reaches QURI exactly and keeps its parameter ABI."""
        builder = CircuitBuilder(1, 0, name="standalone-phase")
        theta = ParameterExpr("theta")
        builder.add_global_phase(theta)

        materialized = QuriPartsMaterializer().materialize(
            builder.freeze(),
            parameter_names=("theta",),
        )

        assert tuple(materialized.parameters) == ("theta",)
        assert materialized.parameter_order == ("theta",)
        assert materialized.artifact.parameter_count == 1
        assert materialized.artifact.qubit_count == 2
        assert materialized.implicit_output_qubit_indices == (0,)

        angle = 0.41
        state = _run_statevector(materialized.artifact, [angle])
        assert np.allclose(
            state,
            np.array(
                [np.exp(1j * angle), 0.0, 0.0, 0.0],
                dtype=np.complex128,
            ),
            rtol=0.0,
            atol=1e-10,
        )

    def test_zero_qubit_standalone_phase_uses_only_an_internal_carrier(self) -> None:
        """A zero-qubit phase remains exact while its logical ABI stays empty."""
        builder = CircuitBuilder(0, 0, name="zero-qubit-phase")
        builder.add_global_phase(0.25)

        materialized = QuriPartsMaterializer().materialize(builder.freeze())

        assert materialized.artifact.qubit_count == 1
        assert materialized.implicit_output_qubit_indices == ()
        state = _run_statevector(materialized.artifact, [])
        assert np.allclose(
            state,
            np.array([np.exp(0.25j), 0.0], dtype=np.complex128),
            rtol=0.0,
            atol=1e-10,
        )

    def test_concrete_standalone_phase_reuses_a_logical_qubit(self) -> None:
        """Concrete phase avoids a dedicated carrier when data is available."""
        builder = CircuitBuilder(1, 0, name="canonical-identity-phase")
        builder.add_global_phase(0.25)

        circuit = QuriPartsMaterializer().materialize(builder.freeze()).artifact

        assert circuit.qubit_count == 1
        assert len(circuit.gates) == 2
        assert all(gate.target_indices == (0,) for gate in circuit.gates)

    def test_phase_carrier_is_hidden_from_implicit_execution_outputs(self) -> None:
        """QURI sampling and run expose only logical qubits, not the carrier."""
        from qamomile.quri_parts import QuriPartsTranspiler

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> None:
            """Apply a symbolic phase gate without an explicit return value."""
            q = qmc.qubit("q")
            q = qmc.p(q, theta)

        transpiler = QuriPartsTranspiler()
        executable = transpiler.transpile(circuit, parameters=["theta"])
        executor = transpiler.executor(seed=123)

        assert executable.compiled_quantum[0].implicit_output_qubit_indices == (0,)
        assert executable.compiled_quantum[0].circuit.qubit_count == 2
        assert executable.sample(
            executor,
            shots=8,
            bindings={"theta": 0.37},
        ).result().results == [((0,), 8)]
        assert executable.run(
            executor,
            bindings={"theta": 0.37},
        ).result() == (0,)

    @pytest.mark.parametrize(
        ("controls", "inverse", "power", "phase_factor"),
        [
            (1, False, 1, 1),
            (2, True, 1, -1),
            (3, False, 2, 2),
        ],
    )
    def test_transformed_phase_only_call_preserves_relative_phase(
        self,
        controls: int,
        inverse: bool,
        power: int,
        phase_factor: int,
    ) -> None:
        """Phase-only calls retain transforms and uncompute phase ancillas."""
        theta = 0.41
        materialized = QuriPartsMaterializer().materialize(
            _phase_only_call(
                controls=controls,
                inverse=inverse,
                power=power,
            ).freeze(),
            parameter_names=("theta",),
        )
        circuit = materialized.artifact
        data_qubits = controls + 1
        expected_ancillas = max(0, controls - 2) + 1

        assert circuit.qubit_count == data_qubits + expected_ancillas
        assert materialized.implicit_output_qubit_indices == tuple(range(data_qubits))
        assert tuple(materialized.parameters) == ("theta",)
        state = _run_statevector(circuit, [theta])
        assert np.allclose(state[2**data_qubits :], 0.0, rtol=0.0, atol=1e-10)
        data_state = state[: 2**data_qubits]
        expected = _expected_phase_kickback(
            controls,
            phase_factor * theta,
        )
        assert abs(np.vdot(expected, data_state)) == pytest.approx(1.0, abs=1e-10)

    def test_tiny_controlled_phase_is_not_optimized_away(self) -> None:
        """Every exact nonzero controlled phase emits a relative-phase gate."""
        body = CircuitBuilder(1, 0, name="tiny-phase")
        body.add_global_phase(1e-16)
        caller = CircuitBuilder(2, 0)
        caller.append_call(
            ReusableCircuit(body.freeze(), "tiny-phase", controls=1),
            (0, 1),
        )

        circuit = QuriPartsMaterializer().materialize(caller.freeze()).artifact

        assert len(circuit.gates) == 1

    def test_controlled_cp_with_runtime_parameter_transpiles(self) -> None:
        """Controlled CP fallback preserves a symbolic QURI Parts angle."""
        from qamomile.quri_parts import QuriPartsTranspiler

        @qmc.qkernel
        def cp_layer(
            control: qmc.Qubit,
            target: qmc.Qubit,
            theta: qmc.Float,
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Apply a symbolic controlled-phase body."""
            control, target = qmc.cp(control, target, theta)
            return control, target

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Apply a controlled CP body with a runtime angle."""
            qs = qmc.qubit_array(3, "qs")
            qs[0], qs[1], qs[2] = qmc.control(cp_layer)(
                qs[0],
                qs[1],
                qs[2],
                theta=theta,
            )
            return qmc.measure(qs)

        transpiler = QuriPartsTranspiler()
        executable = transpiler.transpile(circuit, parameters=["theta"])
        quri_circuit = executable.compiled_quantum[0].circuit

        assert quri_circuit.parameter_count == 1
        assert len(quri_circuit.gates) > 0

    def test_controlled_inverse_with_runtime_parameter_transpiles(self) -> None:
        """Controlled inverse fallback evaluates symbolic angle negation."""
        from qamomile.quri_parts import QuriPartsTranspiler

        @qmc.qkernel
        def phase_layer(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            """Apply a runtime phase rotation."""
            q = qmc.rz(q, theta)
            return q

        @qmc.qkernel
        def inverse_body(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            """Apply an inverse runtime phase inside a controlled body."""
            q = qmc.inverse(phase_layer)(q, theta)
            return q

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Apply a controlled inverse body with a runtime angle."""
            qs = qmc.qubit_array(2, "qs")
            qs[0] = qmc.x(qs[0])
            controlled = qmc.control(inverse_body)
            qs[0], qs[1] = controlled(qs[0], qs[1], theta)
            return qmc.measure(qs)

        transpiler = QuriPartsTranspiler()
        executable = transpiler.transpile(circuit, parameters=["theta"])
        quri_circuit = executable.compiled_quantum[0].circuit

        assert quri_circuit.parameter_count == 1
        assert len(quri_circuit.gates) > 0

    def test_controlled_toffoli_cascade_path_executes(self) -> None:
        """Controlled-Toffoli cascade flips only the intended target qubit."""
        pytest.importorskip("quri_parts.qulacs")

        from qamomile.quri_parts import QuriPartsTranspiler

        @qmc.qkernel
        def toffoli_layer(
            left: qmc.Qubit,
            right: qmc.Qubit,
            target: qmc.Qubit,
        ) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
            """Apply a Toffoli gate inside a controlled body."""
            left, right, target = qmc.ccx(left, right, target)
            return left, right, target

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            """Prepare three controls and apply a controlled Toffoli."""
            qs = qmc.qubit_array(4, "qs")
            qs[0] = qmc.x(qs[0])
            qs[1] = qmc.x(qs[1])
            qs[2] = qmc.x(qs[2])
            qs[0], qs[1], qs[2], qs[3] = qmc.control(toffoli_layer)(
                qs[0],
                qs[1],
                qs[2],
                qs[3],
            )
            return qmc.measure(qs)

        transpiler = QuriPartsTranspiler()
        executable = transpiler.transpile(circuit)
        result = executable.sample(QuriPartsExecutor(seed=123), shots=16).result()

        assert result.results == [((True, True, True, True), 16)]
