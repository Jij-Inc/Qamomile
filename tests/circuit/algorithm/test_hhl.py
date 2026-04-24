"""Tests for qamomile/circuit/algorithm/hhl.py — HHL algorithm."""

import math

import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.hhl import (
    _decode_eigenvalue_from_clock,
    _normalize_supported_raw_bins,
    hhl,
    reciprocal_rotation,
)
from qamomile.qiskit.transpiler import QiskitTranspiler

# ---------------------------------------------------------------------------
# Helper: count gates by name in a Qiskit circuit
# ---------------------------------------------------------------------------


def _gate_counts(qc):
    """Return a dict of {gate_name: count} from a Qiskit QuantumCircuit."""
    counts = {}
    for inst in qc.data:
        name = inst.operation.name
        counts[name] = counts.get(name, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Test unitaries: phase gate and its inverse (1-qubit system register)
# ---------------------------------------------------------------------------


@qmc.qkernel
def _phase_u(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """U = P(theta) — a simple unitary for testing (1-qubit register)."""
    return qmc.p(q, theta)


@qmc.qkernel
def _phase_u_inv(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """U-dagger = P(-theta)."""
    return qmc.p(q, -1.0 * theta)


# ---------------------------------------------------------------------------
# Test unitaries: 2-qubit system register
# ---------------------------------------------------------------------------


@qmc.qkernel
def _phase_u_2q(
    q0: qmc.Qubit,
    q1: qmc.Qubit,
    theta: qmc.Float,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """U on 2-qubit register: P(theta) on q0, identity on q1."""
    q0 = qmc.p(q0, theta)
    return q0, q1


@qmc.qkernel
def _phase_u_inv_2q(
    q0: qmc.Qubit,
    q1: qmc.Qubit,
    theta: qmc.Float,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """U-dagger on 2-qubit register."""
    q0 = qmc.p(q0, -1.0 * theta)
    return q0, q1


# ---------------------------------------------------------------------------
# Wrapper qkernels for transpilation (1-qubit system)
# ---------------------------------------------------------------------------

# Use phase_scale = 2*pi so that eigenvalue bins are lambda_hat = k/N * 2*pi.
# With scaling=0.25 and 2 clock qubits (N=4):
#   raw=1 -> lambda_hat = pi/2,   ratio = 0.25/(pi/2) ~ 0.159  (ok)
#   raw=2 -> lambda_hat = pi,     ratio = 0.25/pi ~ 0.080      (ok)
#   raw=3 -> lambda_hat = 3*pi/2, ratio = 0.25/(3*pi/2) ~ 0.053 (ok)


@qmc.qkernel
def _wrap_hhl_phase(theta: qmc.Float) -> qmc.Bit:
    """Full HHL circuit using phase gate unitary, 2 clock qubits."""
    sys = qmc.qubit_array(1, name="sys")
    sys[0] = qmc.x(sys[0])  # Prepare |b> = |1>
    clock = qmc.qubit_array(2, name="clock")
    anc = qmc.qubit("anc")

    sys, clock, anc = hhl(
        sys,
        clock,
        anc,
        unitary=_phase_u,
        inv_unitary=_phase_u_inv,
        scaling=0.25,
        phase_scale=2.0 * math.pi,
        theta=theta,
    )
    return qmc.measure(anc)


@qmc.qkernel
def _wrap_hhl_3clock(theta: qmc.Float) -> qmc.Bit:
    """HHL circuit with 3 clock qubits."""
    sys = qmc.qubit_array(1, name="sys")
    sys[0] = qmc.x(sys[0])
    clock = qmc.qubit_array(3, name="clock")
    anc = qmc.qubit("anc")

    sys, clock, anc = hhl(
        sys,
        clock,
        anc,
        unitary=_phase_u,
        inv_unitary=_phase_u_inv,
        scaling=0.1,
        phase_scale=2.0 * math.pi,
        theta=theta,
    )
    return qmc.measure(anc)


@qmc.qkernel
def _wrap_reciprocal_only() -> qmc.Bit:
    """Reciprocal rotation in isolation for testing."""
    clock = qmc.qubit_array(2, name="clock")
    anc = qmc.qubit("anc")
    clock, anc = reciprocal_rotation(
        clock, anc, scaling=0.25, phase_scale=2.0 * math.pi
    )
    return qmc.measure(anc)


# ---------------------------------------------------------------------------
# Tests — eigenvalue decoding
# ---------------------------------------------------------------------------


class TestEigenvalueDecode:
    """Test the eigenvalue-decoding helper."""

    def test_unsigned_decode(self):
        """Unsigned mode: lambda_hat = phase_scale * raw / 2^m."""
        # 2 clock qubits, phase_scale=2*pi
        assert _decode_eigenvalue_from_clock(0, 2, 2 * math.pi, False) == 0.0
        assert math.isclose(
            _decode_eigenvalue_from_clock(1, 2, 2 * math.pi, False),
            math.pi / 2,
        )
        assert math.isclose(
            _decode_eigenvalue_from_clock(2, 2, 2 * math.pi, False),
            math.pi,
        )
        assert math.isclose(
            _decode_eigenvalue_from_clock(3, 2, 2 * math.pi, False),
            3 * math.pi / 2,
        )

    def test_signed_decode(self):
        """Signed mode: two's complement interpretation."""
        # 2 clock qubits, phase_scale=2*pi, N=4
        # raw=0 -> 0, raw=1 -> pi/2, raw=2 -> -pi, raw=3 -> -pi/2
        assert _decode_eigenvalue_from_clock(0, 2, 2 * math.pi, True) == 0.0
        assert math.isclose(
            _decode_eigenvalue_from_clock(1, 2, 2 * math.pi, True),
            math.pi / 2,
        )
        assert math.isclose(
            _decode_eigenvalue_from_clock(2, 2, 2 * math.pi, True),
            -math.pi,
        )
        assert math.isclose(
            _decode_eigenvalue_from_clock(3, 2, 2 * math.pi, True),
            -math.pi / 2,
        )


# ---------------------------------------------------------------------------
# Tests — circuit build / transpile
# ---------------------------------------------------------------------------


class TestHHLCircuitBuild:
    """Test that HHL circuits build and transpile without errors."""

    def test_hhl_2clock_transpiles(self):
        """HHL with 2 clock qubits should transpile successfully."""
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_hhl_phase, bindings={"theta": math.pi / 4})
        qc = exe.compiled_quantum[0].circuit
        # 1 system + 2 clock + 1 ancilla = 4 qubits
        assert qc.num_qubits == 4

    def test_hhl_3clock_transpiles(self):
        """HHL with 3 clock qubits should transpile successfully."""
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_hhl_3clock, bindings={"theta": math.pi / 4})
        qc = exe.compiled_quantum[0].circuit
        # 1 system + 3 clock + 1 ancilla = 5 qubits
        assert qc.num_qubits == 5

    def test_reciprocal_rotation_standalone(self):
        """Reciprocal rotation should transpile on its own."""
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_reciprocal_only)
        qc = exe.compiled_quantum[0].circuit
        # 2 clock + 1 ancilla = 3 qubits
        assert qc.num_qubits == 3

    def test_hhl_2qubit_system_transpiles(self):
        """HHL with 2-qubit system register should transpile successfully."""

        @qmc.qkernel
        def _wrap_2q(theta: qmc.Float) -> qmc.Bit:
            sys = qmc.qubit_array(2, name="sys")
            sys[0] = qmc.x(sys[0])
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            sys, clock, anc = hhl(
                sys,
                clock,
                anc,
                unitary=_phase_u_2q,
                inv_unitary=_phase_u_inv_2q,
                scaling=0.25,
                phase_scale=2.0 * math.pi,
                theta=theta,
            )
            return qmc.measure(anc)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_2q, bindings={"theta": math.pi})
        qc = exe.compiled_quantum[0].circuit
        # 2 system + 2 clock + 1 ancilla = 5 qubits
        assert qc.num_qubits == 5


# ---------------------------------------------------------------------------
# Tests — gate structure
# ---------------------------------------------------------------------------


class TestHHLGateStructure:
    """Test structural properties of the HHL circuit."""

    def test_hhl_contains_hadamard(self):
        """HHL circuit should contain Hadamard gates for QPE."""
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_hhl_phase, bindings={"theta": math.pi / 4})
        qc = exe.compiled_quantum[0].circuit
        counts = _gate_counts(qc)
        # QPE forward: 2 H, QPE inverse: 2 H => >= 4 H
        assert counts.get("h", 0) >= 4

    def test_hhl_contains_x_gates_for_reciprocal(self):
        """HHL circuit should contain X gates from reciprocal rotation bit flips."""
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_hhl_phase, bindings={"theta": math.pi / 4})
        qc = exe.compiled_quantum[0].circuit
        counts = _gate_counts(qc)
        # At least 1 X for state prep + X gates from reciprocal rotation
        assert counts.get("x", 0) >= 1

    def test_reciprocal_rotation_gate_structure(self):
        """Reciprocal rotation with 2 clock qubits should produce X gates."""
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_reciprocal_only)
        qc = exe.compiled_quantum[0].circuit
        counts = _gate_counts(qc)
        # Should contain X gates for the bit-flip pattern
        assert counts.get("x", 0) >= 1


# ---------------------------------------------------------------------------
# Tests — parameters and error handling
# ---------------------------------------------------------------------------


class TestHHLParameters:
    """Test that HHL parameters are handled correctly."""

    def test_strict_rejects_invalid_ratio(self):
        """strict=True + supported_raw_bins should raise when |C/lambda_hat| > 1."""

        @qmc.qkernel
        def _wrap_bad() -> qmc.Bit:
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            # scaling=10 with phase_scale=1 means for raw=1:
            #   lambda_hat = 1*1/4 = 0.25,  ratio = 10/0.25 = 40  (> 1)
            # strict fires only when supported_raw_bins is provided.
            clock, anc = reciprocal_rotation(
                clock,
                anc,
                scaling=10.0,
                phase_scale=1.0,
                strict=True,
                supported_raw_bins=(1, 2, 3),
            )
            return qmc.measure(anc)

        transpiler = QiskitTranspiler()
        with pytest.raises(ValueError, match="Invalid reciprocal rotation"):
            transpiler.transpile(_wrap_bad)

    def test_strict_without_bins_does_not_raise(self):
        """strict=True without supported_raw_bins should NOT raise."""

        @qmc.qkernel
        def _wrap_no_bins() -> qmc.Bit:
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            # Same params that would fail with supported_raw_bins,
            # but without bins declared, strict is silent.
            clock, anc = reciprocal_rotation(
                clock, anc, scaling=10.0, phase_scale=1.0, strict=True
            )
            return qmc.measure(anc)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_no_bins)
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 3

    def test_nonstrict_skips_invalid_ratio(self):
        """strict=False should silently skip bins with |C/lambda_hat| > 1."""

        @qmc.qkernel
        def _wrap_skip() -> qmc.Bit:
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            clock, anc = reciprocal_rotation(
                clock, anc, scaling=10.0, phase_scale=1.0, strict=False
            )
            return qmc.measure(anc)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_skip)
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 3

    def test_negative_scaling_raises(self):
        """scaling <= 0 should raise ValueError."""

        @qmc.qkernel
        def _wrap_neg() -> qmc.Bit:
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            clock, anc = reciprocal_rotation(
                clock, anc, scaling=-0.5, phase_scale=2.0 * math.pi
            )
            return qmc.measure(anc)

        transpiler = QiskitTranspiler()
        with pytest.raises(ValueError, match="scaling must be positive"):
            transpiler.transpile(_wrap_neg)

    def test_different_theta_values(self):
        """HHL circuit should build for different theta values."""
        transpiler = QiskitTranspiler()
        for theta in [0.1, math.pi / 4, math.pi / 2, math.pi]:
            exe = transpiler.transpile(_wrap_hhl_phase, bindings={"theta": theta})
            qc = exe.compiled_quantum[0].circuit
            assert qc.num_qubits == 4

    def test_signed_mode_transpiles(self):
        """HHL with signed=True should transpile successfully."""

        @qmc.qkernel
        def _wrap_signed(theta: qmc.Float) -> qmc.Bit:
            sys = qmc.qubit_array(1, name="sys")
            sys[0] = qmc.x(sys[0])
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            sys, clock, anc = hhl(
                sys,
                clock,
                anc,
                unitary=_phase_u,
                inv_unitary=_phase_u_inv,
                scaling=0.1,
                phase_scale=2.0 * math.pi,
                signed=True,
                theta=theta,
            )
            return qmc.measure(anc)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_signed, bindings={"theta": math.pi / 4})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 4

    def test_big_endian_clock_transpiles(self):
        """HHL with little_endian_clock=False should transpile successfully."""

        @qmc.qkernel
        def _wrap_big(theta: qmc.Float) -> qmc.Bit:
            sys = qmc.qubit_array(1, name="sys")
            sys[0] = qmc.x(sys[0])
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            sys, clock, anc = hhl(
                sys,
                clock,
                anc,
                unitary=_phase_u,
                inv_unitary=_phase_u_inv,
                scaling=0.25,
                phase_scale=2.0 * math.pi,
                little_endian_clock=False,
                theta=theta,
            )
            return qmc.measure(anc)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_big, bindings={"theta": math.pi / 4})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 4

    def test_supported_raw_bins_transpiles(self):
        """HHL with supported_raw_bins should transpile successfully."""

        @qmc.qkernel
        def _wrap_bins(theta: qmc.Float) -> qmc.Bit:
            sys = qmc.qubit_array(1, name="sys")
            sys[0] = qmc.x(sys[0])
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            sys, clock, anc = hhl(
                sys,
                clock,
                anc,
                unitary=_phase_u,
                inv_unitary=_phase_u_inv,
                scaling=0.25,
                phase_scale=2.0 * math.pi,
                supported_raw_bins=(2,),
                theta=theta,
            )
            return qmc.measure(anc)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_bins, bindings={"theta": math.pi})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 4

    def test_iqft_includes_swaps_transpiles(self):
        """HHL with iqft_includes_swaps=True should transpile successfully."""

        @qmc.qkernel
        def _wrap_swaps(theta: qmc.Float) -> qmc.Bit:
            sys = qmc.qubit_array(1, name="sys")
            sys[0] = qmc.x(sys[0])
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            sys, clock, anc = hhl(
                sys,
                clock,
                anc,
                unitary=_phase_u,
                inv_unitary=_phase_u_inv,
                scaling=0.25,
                phase_scale=2.0 * math.pi,
                iqft_includes_swaps=True,
                theta=theta,
            )
            return qmc.measure(anc)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(_wrap_swaps, bindings={"theta": math.pi / 4})
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 4


# ---------------------------------------------------------------------------
# Tests — supported_raw_bins validation
# ---------------------------------------------------------------------------


class TestNormalizeSupportedRawBins:
    """Test the _normalize_supported_raw_bins helper."""

    def test_none_returns_none(self):
        assert _normalize_supported_raw_bins(None, 2) is None

    def test_valid_bins(self):
        result = _normalize_supported_raw_bins([1, 2, 3], 2)
        assert result == frozenset({1, 2, 3})

    def test_empty_collection_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _normalize_supported_raw_bins([], 2)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out-of-range"):
            _normalize_supported_raw_bins([4], 2)  # valid: 0..3

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="out-of-range"):
            _normalize_supported_raw_bins([-1, 1], 2)

    def test_zero_bin_raises(self):
        """Zero bin is not invertible and should be rejected."""
        with pytest.raises(ValueError, match="must not contain 0"):
            _normalize_supported_raw_bins([0, 1], 2)

    def test_bool_raises(self):
        """bool values should be rejected even though bool is subclass of int."""
        with pytest.raises(TypeError, match="integers only"):
            _normalize_supported_raw_bins([True, 2], 2)


# ---------------------------------------------------------------------------
# Tests — _decode_eigenvalue_from_clock validation
# ---------------------------------------------------------------------------


class TestDecodeEigenvalueValidation:
    """Test validation in _decode_eigenvalue_from_clock."""

    def test_n_clock_zero_raises(self):
        with pytest.raises(ValueError, match="n_clock must be >= 1"):
            _decode_eigenvalue_from_clock(0, 0, 2 * math.pi, False)

    def test_phase_scale_zero_raises(self):
        with pytest.raises(ValueError, match="phase_scale must be positive"):
            _decode_eigenvalue_from_clock(0, 2, 0.0, False)

    def test_phase_scale_negative_raises(self):
        with pytest.raises(ValueError, match="phase_scale must be positive"):
            _decode_eigenvalue_from_clock(0, 2, -1.0, False)


# ---------------------------------------------------------------------------
# Tests — Block-level operation inspection
# ---------------------------------------------------------------------------


class TestHHLBlockStructure:
    """Verify HHL circuit structure via qkernel.build() graph inspection."""

    def _build_hhl_graph(self, n_clock=2):
        """Build an HHL kernel and return its traced Graph."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            sys = qmc.qubit_array(1, name="sys")
            clock = qmc.qubit_array(n_clock, name="clock")
            anc = qmc.qubit("anc")
            sys, clock, anc = hhl(
                sys,
                clock,
                anc,
                unitary=_phase_u,
                inv_unitary=_phase_u_inv,
                scaling=0.25,
                phase_scale=2.0 * math.pi,
                supported_raw_bins=(2,),
                strict=True,
                theta=theta,
            )
            return qmc.measure(anc)

        return circuit.build()

    def test_graph_has_operations(self) -> None:
        """The built graph should contain a non-empty operation list."""
        graph = self._build_hhl_graph()
        assert len(graph.operations) > 0

    def test_graph_contains_h_gates(self) -> None:
        """HHL uses H gates for clock superposition and uncomputation."""
        from qamomile.circuit.ir.operation.gate import (
            GateOperation,
            GateOperationType,
        )

        graph = self._build_hhl_graph(n_clock=2)
        h_ops = [
            op
            for op in graph.operations
            if isinstance(op, GateOperation) and op.gate_type == GateOperationType.H
        ]
        # Forward QPE: 2 H gates, Inverse QPE: 2 H gates => at least 4
        assert len(h_ops) >= 4, f"Expected >= 4 H gates, got {len(h_ops)}"

    def test_graph_contains_controlled_u_ops(self) -> None:
        """HHL uses controlled-U for QPE and controlled-U-dagger for uncompute."""
        from qamomile.circuit.ir.operation.gate import ControlledUOperation

        graph = self._build_hhl_graph(n_clock=2)
        cu_ops = [op for op in graph.operations if isinstance(op, ControlledUOperation)]
        # 2 clock qubits: 2 forward CU + 2 inverse CU + multi-controlled RY
        assert len(cu_ops) >= 4, f"Expected >= 4 CU ops, got {len(cu_ops)}"

    def test_graph_contains_composite_gates(self) -> None:
        """HHL uses IQFT and QFT as composite gate operations."""
        from qamomile.circuit.ir.operation.composite_gate import (
            CompositeGateOperation,
            CompositeGateType,
        )

        graph = self._build_hhl_graph(n_clock=2)
        composite_ops = [
            op for op in graph.operations if isinstance(op, CompositeGateOperation)
        ]
        gate_types = {op.gate_type for op in composite_ops}
        assert CompositeGateType.IQFT in gate_types, (
            f"Expected IQFT in composite gates, got {gate_types}"
        )
        assert CompositeGateType.QFT in gate_types, (
            f"Expected QFT in composite gates, got {gate_types}"
        )

    def test_graph_has_measurement(self) -> None:
        """HHL kernel ends with a measurement of the ancilla."""
        from qamomile.circuit.ir.operation.gate import MeasureOperation

        graph = self._build_hhl_graph()
        measure_ops = [
            op for op in graph.operations if isinstance(op, MeasureOperation)
        ]
        assert len(measure_ops) >= 1, "Expected at least one MeasureOperation"

    def test_graph_qubit_init_count(self) -> None:
        """Three qubit allocations: system array, clock array, ancilla."""
        from qamomile.circuit.ir.operation.operation import QInitOperation

        for n_clock in [2, 3]:
            graph = self._build_hhl_graph(n_clock=n_clock)
            qinit_ops = [
                op for op in graph.operations if isinstance(op, QInitOperation)
            ]
            # 3 allocations: qubit_array(1), qubit_array(n_clock), qubit()
            assert len(qinit_ops) == 3, (
                f"n_clock={n_clock}: expected 3 qubit allocations, got {len(qinit_ops)}"
            )
