"""Tests for single-qubit gate broadcasting over `Vector[Qubit]`.

Verifies that calling a single-qubit gate (``h``, ``x``, ``y``, ``z``,
``s``, ``t``, ``sdg``, ``tdg``, ``rx``, ``ry``, ``rz``, ``p``) with a
`Vector[Qubit]` argument produces the same IR shape and the same backend
behaviour as a hand-written ``for i in qmc.range(n): qs[i] = gate(qs[i])``
loop. Cross-backend execution is exercised on every supported SDK
(Qiskit, QuriParts, CUDA-Q) per ``CLAUDE.md``'s test policy.

Note: Do NOT use ``from __future__ import annotations`` in this file.
The @qkernel AST transformer relies on resolved type annotations.
"""

import math

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType

# ---------------------------------------------------------------------------
# IR-shape assertions (no backend required)
# ---------------------------------------------------------------------------


class TestBroadcastIRShape:
    """Broadcast lowers to the same `ForOperation` as an explicit loop."""

    @staticmethod
    @qmc.qkernel
    def h_broadcast() -> qmc.Vector[qmc.Bit]:
        """Apply H to a 4-qubit array via broadcast."""
        qs = qmc.qubit_array(4, "qs")
        qs = qmc.h(qs)
        return qmc.measure(qs)

    @staticmethod
    @qmc.qkernel
    def h_explicit_loop() -> qmc.Vector[qmc.Bit]:
        """Apply H to a 4-qubit array via explicit ``for i in qmc.range(n)``."""
        qs = qmc.qubit_array(4, "qs")
        n = qs.shape[0]
        for i in qmc.range(n):
            qs[i] = qmc.h(qs[i])
        return qmc.measure(qs)

    @staticmethod
    @qmc.qkernel
    def rx_broadcast(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
        """Apply RX(theta) to a 3-qubit array via broadcast."""
        qs = qmc.qubit_array(3, "qs")
        qs = qmc.rx(qs, theta)
        return qmc.measure(qs)

    @staticmethod
    @qmc.qkernel
    def rx_explicit_loop(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
        """Apply RX(theta) to a 3-qubit array via explicit ``for i in qmc.range(n)``."""
        qs = qmc.qubit_array(3, "qs")
        n = qs.shape[0]
        for i in qmc.range(n):
            qs[i] = qmc.rx(qs[i], theta)
        return qmc.measure(qs)

    def _find_for_op(self, block) -> ForOperation:
        """Return the (single) top-level ForOperation in the block."""
        fors = [op for op in block.operations if isinstance(op, ForOperation)]
        assert len(fors) == 1, (
            f"Expected exactly one top-level ForOperation, found {len(fors)}"
        )
        return fors[0]

    def test_h_broadcast_emits_for_operation_with_h_gate(self):
        block = self.h_broadcast.block
        for_op = self._find_for_op(block)
        gate_ops = [op for op in for_op.operations if isinstance(op, GateOperation)]
        assert len(gate_ops) == 1, "Expected one gate inside broadcast loop"
        assert gate_ops[0].gate_type is GateOperationType.H

    def test_h_broadcast_loop_structure_matches_explicit(self):
        bc = self._find_for_op(self.h_broadcast.block)
        ex = self._find_for_op(self.h_explicit_loop.block)

        # Same number and kind of nested operations.
        assert len(bc.operations) == len(ex.operations)
        for a, b in zip(bc.operations, ex.operations):
            assert type(a) is type(b)
        bc_gate = next(op for op in bc.operations if isinstance(op, GateOperation))
        ex_gate = next(op for op in ex.operations if isinstance(op, GateOperation))
        assert bc_gate.gate_type is ex_gate.gate_type

        # Loop bound IR values agree on constness/value.
        for bc_v, ex_v in zip(bc.operands, ex.operands):
            assert bc_v.is_constant() == ex_v.is_constant()
            if bc_v.is_constant():
                assert bc_v.get_const() == ex_v.get_const()

    def test_rx_broadcast_emits_for_operation_with_rx_gate(self):
        block = self.rx_broadcast.block
        for_op = self._find_for_op(block)
        gate_ops = [op for op in for_op.operations if isinstance(op, GateOperation)]
        assert len(gate_ops) == 1
        assert gate_ops[0].gate_type is GateOperationType.RX


# ---------------------------------------------------------------------------
# Backend execution: Qiskit and QuriParts (sampling + expval)
# ---------------------------------------------------------------------------

# QuriParts is the more permissive of the two for symbolic shapes;
# exercise both on every test where the SDKs are available.
pytest.importorskip("qiskit")
pytest.importorskip("quri_parts")
pytest.importorskip("quri_parts.qulacs")

import qamomile.observable as qm_o  # noqa: E402
from qamomile.qiskit import QiskitTranspiler  # noqa: E402
from qamomile.quri_parts import QuriPartsTranspiler  # noqa: E402

# CUDA-Q is optional — gate the parametrize entry on whether the SDK is
# present, mirroring how other cross-backend test suites handle it.
_HAS_CUDAQ = True
try:  # pragma: no cover - presence check, not behaviour
    import cudaq  # noqa: F401

    from qamomile.cudaq import CudaqTranspiler  # noqa: E402
except ImportError:  # pragma: no cover - covered when cudaq is absent
    _HAS_CUDAQ = False
    CudaqTranspiler = None  # type: ignore[assignment]

BACKENDS = [
    pytest.param(QiskitTranspiler, id="qiskit"),
    pytest.param(QuriPartsTranspiler, id="quri_parts"),
    pytest.param(
        CudaqTranspiler,
        id="cudaq",
        marks=pytest.mark.skipif(not _HAS_CUDAQ, reason="cudaq not installed"),
    ),
]

# Fixed n-set spans 1, 2, 3, 5 — covers single-qubit, paired, odd, and >4 cases.
N_VALUES = [1, 2, 3, 5]
SEEDS = [0, 1, 2, 42]


def _kernel_x_broadcast(n: int):
    """Build a kernel that applies X to all qubits via broadcast and measures."""

    @qmc.qkernel
    def _circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        qs = qmc.x(qs)
        return qmc.measure(qs)

    return _circuit


def _kernel_h_broadcast(n: int):
    """Build a kernel that applies H to all qubits via broadcast and measures."""

    @qmc.qkernel
    def _circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        qs = qmc.h(qs)
        return qmc.measure(qs)

    return _circuit


def _kernel_rx_broadcast(n: int):
    """Build a kernel that applies RX(theta) to all qubits and measures."""

    @qmc.qkernel
    def _circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        qs = qmc.rx(qs, theta)
        return qmc.measure(qs)

    return _circuit


def _kernel_rx_loop(n: int):
    """Build a kernel that applies RX(theta) via explicit ``for i in qmc.range(n)``."""

    @qmc.qkernel
    def _circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        m = qs.shape[0]
        for i in qmc.range(m):
            qs[i] = qmc.rx(qs[i], theta)
        return qmc.measure(qs)

    return _circuit


def _kernel_z_expval(n: int):
    """Build a kernel that prepares |+>^n via H broadcast, then computes a Hamiltonian expval."""

    @qmc.qkernel
    def _circuit(obs: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(n, "qs")
        qs = qmc.h(qs)
        return qmc.expval(qs, obs)

    return _circuit


def _kernel_rx_expval(n: int):
    """Build a kernel that applies RX(theta) broadcast, then computes a Hamiltonian expval."""

    @qmc.qkernel
    def _circuit(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(n, "qs")
        qs = qmc.rx(qs, theta)
        return qmc.expval(qs, obs)

    return _circuit


def _sum_z_hamiltonian(n: int) -> "qm_o.Hamiltonian":
    """Build H = sum_{i=0}^{n-1} Z_i, with num_qubits=n so single-qubit registers padded."""
    H = qm_o.Hamiltonian.zero(num_qubits=n)
    for i in range(n):
        H += qm_o.Z(i)
    return H


class TestBroadcastSampling:
    """Broadcast-X / broadcast-H produce expected sample distributions."""

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    @pytest.mark.parametrize("n", N_VALUES)
    def test_x_broadcast_flips_all_qubits(self, transpiler_factory, n):
        """X broadcast over n qubits → every shot is the all-ones bitstring."""
        kernel = _kernel_x_broadcast(n)
        t = transpiler_factory()
        exe = t.transpile(kernel)
        results = exe.sample(t.executor(), shots=128).result().results
        expected = (1,) * n
        for value, _count in results:
            assert value == expected, (
                f"[{transpiler_factory.__name__}, n={n}] expected {expected}, got {value}"
            )

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    @pytest.mark.parametrize("n", [3, 5])
    def test_h_broadcast_uniform_distribution(self, transpiler_factory, n):
        """H broadcast over n qubits → roughly uniform over 2^n outcomes."""
        kernel = _kernel_h_broadcast(n)
        t = transpiler_factory()
        exe = t.transpile(kernel)
        shots = 4096
        results = exe.sample(t.executor(), shots=shots).result().results
        # Uniform → each outcome's probability ~1/2^n.  Verify total shots
        # match and at least 2^(n-1) distinct outcomes appeared (loose bound
        # so the test is robust to shot noise on small registers).
        total = sum(count for _val, count in results)
        assert total == shots
        assert len(results) >= 2 ** (n - 1)


class TestBroadcastVsExplicitLoopEquivalence:
    """Broadcast and explicit-loop forms produce statistically identical results."""

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    @pytest.mark.parametrize("n", N_VALUES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_rx_broadcast_matches_loop_expval(self, transpiler_factory, n, seed):
        """<sum_i Z_i> after RX(theta) broadcast equals the same expval after an explicit RX loop."""
        rng = np.random.default_rng(seed)
        # Cover boundary inputs (0, π, 2π) along with random samples.
        boundary_choices = [0.0, math.pi, 2 * math.pi]
        theta = float(rng.choice(boundary_choices + [rng.uniform(-math.pi, math.pi)]))

        # H = sum_i Z_i over all n qubits.  After RX(theta) broadcast on
        # |0>^n every <Z_i> = cos(theta), so <H> = n * cos(theta).
        H = qm_o.Hamiltonian.zero(num_qubits=n)
        for i in range(n):
            H += qm_o.Z(i)

        @qmc.qkernel
        def _broadcast(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
            qs = qmc.qubit_array(n, "qs")
            qs = qmc.rx(qs, theta)
            return qmc.expval(qs, obs)

        @qmc.qkernel
        def _loop(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
            qs = qmc.qubit_array(n, "qs")
            m = qs.shape[0]
            for i in qmc.range(m):
                qs[i] = qmc.rx(qs[i], theta)
            return qmc.expval(qs, obs)

        t = transpiler_factory()
        exe_b = t.transpile(_broadcast, parameters=["theta"], bindings={"obs": H})
        exe_l = t.transpile(_loop, parameters=["theta"], bindings={"obs": H})

        executor = t.executor()
        out_b = exe_b.run(executor, bindings={"theta": theta}).result()
        out_l = exe_l.run(executor, bindings={"theta": theta}).result()

        assert np.isclose(out_b, out_l, atol=1e-6), (
            f"[{transpiler_factory.__name__}, n={n}, seed={seed}] "
            f"broadcast={out_b}, loop={out_l}, theta={theta}"
        )
        # Cross-check against the analytic expression <H> = n * cos(theta).
        assert np.isclose(out_b, n * math.cos(theta), atol=1e-6)


class TestBroadcastExpval:
    """Broadcast-prepared states give the analytic expectation values."""

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    @pytest.mark.parametrize("n", N_VALUES)
    def test_h_broadcast_then_sum_z_expval_is_zero(self, transpiler_factory, n):
        """H broadcast prepares |+>^n; <sum_i Z_i> on |+>^n is 0."""
        H = _sum_z_hamiltonian(n)
        kernel = _kernel_z_expval(n)
        t = transpiler_factory()
        exe = t.transpile(kernel, bindings={"obs": H})
        out = exe.run(t.executor()).result()
        assert np.isclose(out, 0.0, atol=1e-6)

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    @pytest.mark.parametrize("n", N_VALUES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_rx_broadcast_expval_matches_n_cos(self, transpiler_factory, n, seed):
        """<sum_i Z_i> after RX(theta) broadcast over n qubits equals n*cos(theta)."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-2 * math.pi, 2 * math.pi))
        H = _sum_z_hamiltonian(n)
        kernel = _kernel_rx_expval(n)
        t = transpiler_factory()
        exe = t.transpile(kernel, parameters=["theta"], bindings={"obs": H})
        out = exe.run(t.executor(), bindings={"theta": theta}).result()
        assert np.isclose(out, n * math.cos(theta), atol=1e-6), (
            f"[{transpiler_factory.__name__}, n={n}, seed={seed}, theta={theta}] "
            f"expected n*cos(theta)={n * math.cos(theta)}, got {out}"
        )


# ---------------------------------------------------------------------------
# Coverage of all single-qubit gate primitives (sampling smoke + IR check)
# ---------------------------------------------------------------------------


class TestAllPrimitivesBroadcast:
    """Each non-parametric and parametric single-qubit gate accepts Vector[Qubit]."""

    @pytest.mark.parametrize(
        "gate_func, gate_type",
        [
            (qmc.h, GateOperationType.H),
            (qmc.x, GateOperationType.X),
            (qmc.y, GateOperationType.Y),
            (qmc.z, GateOperationType.Z),
            (qmc.s, GateOperationType.S),
            (qmc.sdg, GateOperationType.SDG),
            (qmc.t, GateOperationType.T),
            (qmc.tdg, GateOperationType.TDG),
        ],
    )
    def test_non_parametric_broadcast_emits_correct_gate(self, gate_func, gate_type):
        """Non-parametric gate broadcast over a 3-qubit array emits one matching gate per loop body."""

        @qmc.qkernel
        def _circuit() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            qs = gate_func(qs)
            return qmc.measure(qs)

        block = _circuit.block
        fors = [op for op in block.operations if isinstance(op, ForOperation)]
        assert len(fors) == 1
        gates = [op for op in fors[0].operations if isinstance(op, GateOperation)]
        assert len(gates) == 1
        assert gates[0].gate_type is gate_type

    @pytest.mark.parametrize(
        "gate_func, gate_type",
        [
            (qmc.rx, GateOperationType.RX),
            (qmc.ry, GateOperationType.RY),
            (qmc.rz, GateOperationType.RZ),
            (qmc.p, GateOperationType.P),
        ],
    )
    def test_parametric_broadcast_emits_correct_gate(self, gate_func, gate_type):
        """Parametric gate broadcast emits one matching gate per loop body."""

        @qmc.qkernel
        def _circuit() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            qs = gate_func(qs, 0.5)
            return qmc.measure(qs)

        block = _circuit.block
        fors = [op for op in block.operations if isinstance(op, ForOperation)]
        assert len(fors) == 1
        gates = [op for op in fors[0].operations if isinstance(op, GateOperation)]
        assert len(gates) == 1
        assert gates[0].gate_type is gate_type


class TestBroadcastInputValidation:
    """Broadcast dispatch rejects non-Qubit / non-Vector[Qubit] inputs."""

    def test_broadcast_rejects_non_qubit_target(self):
        """Calling ``h`` with a plain int raises TypeError."""
        with pytest.raises(TypeError, match="Expected Qubit or Vector"):
            qmc.h(123)  # type: ignore[arg-type]

    def test_rx_broadcast_rejects_non_qubit_target(self):
        """Calling ``rx`` with a non-Qubit target raises TypeError."""
        with pytest.raises(TypeError, match="Expected Qubit or Vector"):
            qmc.rx("not a qubit", 0.5)  # type: ignore[arg-type]
