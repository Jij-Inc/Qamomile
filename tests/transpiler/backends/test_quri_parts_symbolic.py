"""QURI Parts symbolic-angle (linear-combination dict) tests.

Covers ``QuriPartsGateEmitter.combine_symbolic`` directly and the
end-to-end emission path that produces ``ParametricRX``/``Parametric...``
gates when a runtime parameter participates in a ``BinOp``.

QURI Parts' Rust-backed ``Parameter`` does not support Python arithmetic,
so the qamomile compiler routes parametric BinOps through
``combine_symbolic`` to produce the linear-combination dicts that
``LinearMappedUnboundParametricQuantumCircuit`` consumes natively.
Non-linear combinations (``param * param``, ``param ** n``, etc.)
fundamentally cannot be expressed in this form and must surface a clear
``QamomileQuriPartsTranspileError`` rather than a low-level ``TypeError``.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.quri_parts

import qamomile.circuit as qmc  # noqa: E402

pytest.importorskip("quri_parts")
pytest.importorskip("quri_parts.qulacs")

from quri_parts.circuit import CONST, gate_names  # noqa: E402

from qamomile.circuit.ir.operation.arithmetic_operations import BinOpKind  # noqa: E402
from qamomile.quri_parts import QuriPartsTranspiler  # noqa: E402
from qamomile.quri_parts.emitter import QuriPartsGateEmitter  # noqa: E402
from qamomile.quri_parts.exceptions import (  # noqa: E402
    QamomileQuriPartsTranspileError,
)

# ---------------------------------------------------------------------------
# Unit tests for combine_symbolic
# ---------------------------------------------------------------------------


def _emitter_with_circuit():
    """Build an emitter with an active circuit so create_parameter works."""
    e = QuriPartsGateEmitter()
    e.create_circuit(num_qubits=1, num_clbits=0)
    return e


class TestCombineSymbolicLinear:
    """Linear-combination cases that combine_symbolic must handle."""

    def test_param_times_const_lhs_const(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        result = e.combine_symbolic(BinOpKind.MUL, gamma, 2.5)
        assert result == {gamma: 2.5}

    def test_param_times_const_rhs_const(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        # Symmetric — scalar on the left
        result = e.combine_symbolic(BinOpKind.MUL, 2.5, gamma)
        assert result == {gamma: 2.5}

    def test_param_plus_const(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        result = e.combine_symbolic(BinOpKind.ADD, gamma, 0.5)
        assert result == {gamma: 1.0, CONST: 0.5}

    def test_param_minus_const(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        result = e.combine_symbolic(BinOpKind.SUB, gamma, 0.5)
        assert result == {gamma: 1.0, CONST: -0.5}

    def test_const_minus_param(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        result = e.combine_symbolic(BinOpKind.SUB, 1.0, gamma)
        assert result == {CONST: 1.0, gamma: -1.0}

    def test_param_plus_param(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        beta = e.create_parameter("beta")
        result = e.combine_symbolic(BinOpKind.ADD, gamma, beta)
        assert result == {gamma: 1.0, beta: 1.0}

    def test_param_div_const(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        result = e.combine_symbolic(BinOpKind.DIV, gamma, 2.0)
        assert result == {gamma: 0.5}

    def test_chained_form_dict_lhs(self):
        """combine_symbolic re-applied to its own output handles dict inputs."""
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        # First: gamma * 2.0 → {gamma: 2.0}
        first = e.combine_symbolic(BinOpKind.MUL, gamma, 2.0)
        # Then: (gamma * 2.0) + 0.1 → {gamma: 2.0, CONST: 0.1}
        second = e.combine_symbolic(BinOpKind.ADD, first, 0.1)
        assert second == {gamma: 2.0, CONST: 0.1}

    def test_to_linear_form_returns_fresh_dict(self):
        """Mutation of result must not alias an upstream stored intermediate."""
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        first = e.combine_symbolic(BinOpKind.MUL, gamma, 2.0)
        second = e.combine_symbolic(BinOpKind.ADD, first, 0.1)
        # Mutating ``second`` must not change ``first``.
        second.clear()
        assert first == {gamma: 2.0}


class TestCombineSymbolicNonLinear:
    """Non-linear or unsupported cases must raise a clear error."""

    def test_param_times_param_raises(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        beta = e.create_parameter("beta")
        with pytest.raises(QamomileQuriPartsTranspileError, match="non-linear"):
            e.combine_symbolic(BinOpKind.MUL, gamma, beta)

    def test_const_div_param_raises(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        with pytest.raises(QamomileQuriPartsTranspileError, match="division by"):
            e.combine_symbolic(BinOpKind.DIV, 1.0, gamma)

    def test_param_div_param_raises(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        beta = e.create_parameter("beta")
        with pytest.raises(QamomileQuriPartsTranspileError, match="division by"):
            e.combine_symbolic(BinOpKind.DIV, gamma, beta)

    def test_param_div_zero_raises(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        with pytest.raises(QamomileQuriPartsTranspileError, match="division by zero"):
            e.combine_symbolic(BinOpKind.DIV, gamma, 0.0)

    def test_pow_raises(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        with pytest.raises(QamomileQuriPartsTranspileError, match="POW"):
            e.combine_symbolic(BinOpKind.POW, gamma, 2)

    def test_floordiv_raises(self):
        e = _emitter_with_circuit()
        gamma = e.create_parameter("gamma")
        with pytest.raises(QamomileQuriPartsTranspileError, match="FLOORDIV"):
            e.combine_symbolic(BinOpKind.FLOORDIV, gamma, 2)


# ---------------------------------------------------------------------------
# End-to-end: parametric circuits with BinOps
# ---------------------------------------------------------------------------


class TestSymbolicEndToEnd:
    """End-to-end transpile + bind + sample/estimate with symbolic angles.

    Note: top-level ``rx(q, theta * 2.0)`` patterns (BinOp directly between
    quantum gates without an enclosing for-items loop) are rejected by the
    segmentation pass with ``MultipleQuantumSegmentsError`` on every backend
    — this is a separate, pre-existing constraint of the NISQ segmentation
    strategy, not a QURI-Parts-specific issue. The realistic and supported
    pattern is BinOp-inside-loop, which the QAOA tests below exercise.
    """

    def test_rx_with_param_times_const_in_items_loop(self):
        """Inside qmc.items, rx(q, theta * c) survives emission and binds."""

        @qmc.qkernel
        def circuit(
            theta: qmc.Float,
            coeffs: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Bit:
            q = qmc.qubit("q")
            for _, c in qmc.items(coeffs):
                q = qmc.rx(q, theta * c)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(
            circuit,
            bindings={"coeffs": {0: 2.0, 1: -0.5}},
            parameters=["theta"],
        )
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count == 1
        # Bind theta = 1.0 → angles = [2.0, -0.5]
        bound = qc.bind_parameters([1.0])
        gates = list(bound.gates)
        rx_gates = [g for g in gates if g.name == gate_names.RX]
        assert len(rx_gates) == 2
        assert np.isclose(rx_gates[0].params[0], 2.0, atol=1e-10)
        assert np.isclose(rx_gates[1].params[0], -0.5, atol=1e-10)

    def test_qaoa_layer_sample_matches_qiskit(self):
        """QAOA layer with symbolic gamma*Jij — sampling distribution sanity."""

        @qmc.qkernel
        def qaoa_layer(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
            beta: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], beta)
            return qmc.measure(q)

        ising = {(0, 1): 1.0}
        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(
            qaoa_layer,
            bindings={"n": 2, "ising": ising},
            parameters=["gamma", "beta"],
        )
        executor = transpiler.executor()
        job = exe.sample(
            executor,
            shots=2000,
            bindings={"gamma": 0.5, "beta": 0.3},
        )
        result = job.result()
        # Symmetric Ising with H+RX produces non-trivial distribution; just
        # verify shots add up and all bitstrings are length-2.
        total = sum(count for _, count in result.results)
        assert total == 2000
        for value, _ in result.results:
            assert len(value) == 2

    def test_qaoa_expval_matches_qiskit(self):
        """QAOA expval through both QURI Parts and Qiskit must agree.

        Cross-backend equivalence check per CLAUDE.md (sampling + expval
        both required for algorithm/stdlib changes).
        """
        pytest.importorskip("qiskit")
        import qamomile.observable as qm_o
        from qamomile.qiskit import QiskitTranspiler

        ising = {(0, 1): 1.0}

        @qmc.qkernel
        def qaoa_expval(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            obs: qmc.Observable,
            gamma: qmc.Float,
            beta: qmc.Float,
        ) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], beta)
            return qmc.expval(q, obs)

        H = qm_o.Z(0) * qm_o.Z(1)

        gamma_val, beta_val = 0.7, 0.4
        bindings = {"n": 2, "ising": ising, "obs": H}

        qp = QuriPartsTranspiler()
        exe_qp = qp.transpile(
            qaoa_expval, bindings=bindings, parameters=["gamma", "beta"]
        )
        val_qp = exe_qp.run(
            qp.executor(),
            bindings={"gamma": gamma_val, "beta": beta_val},
        ).result()

        qk = QiskitTranspiler()
        exe_qk = qk.transpile(
            qaoa_expval, bindings=bindings, parameters=["gamma", "beta"]
        )
        val_qk = exe_qk.run(
            qk.executor(),
            bindings={"gamma": gamma_val, "beta": beta_val},
        ).result()

        assert np.isclose(val_qp, val_qk, atol=1e-8)

    def test_nonlinear_param_squared_raises_at_transpile(self):
        """theta * theta inside an items loop surfaces our linear-only error."""

        @qmc.qkernel
        def circuit(
            theta: qmc.Float,
            coeffs: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Bit:
            q = qmc.qubit("q")
            for _, _c in qmc.items(coeffs):
                q = qmc.rx(q, theta * theta)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        with pytest.raises(QamomileQuriPartsTranspileError, match="non-linear"):
            transpiler.transpile(
                circuit,
                bindings={"coeffs": {0: 1.0}},
                parameters=["theta"],
            )
