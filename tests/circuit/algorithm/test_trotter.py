"""Tests for qamomile/circuit/algorithm/trotter.py — TrotterCircuit."""

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.trotter import (
    TrotterCircuit,
    _full_sequence,
    _product_formula,
    _split_hamiltonian,
)
from qamomile.observable.hamiltonian import Hamiltonian

# -----------------------------------------------------------------------
# Helper: build common Hamiltonians
# -----------------------------------------------------------------------


def _make_2term_h() -> Hamiltonian:
    """H = Z0 Z1 + X0."""
    return qm_o.Z(0) * qm_o.Z(1) + qm_o.X(0)


def _make_3term_h() -> Hamiltonian:
    """H = Z0 Z1 + 0.5 X0 + 0.5 X1."""
    return qm_o.Z(0) * qm_o.Z(1) + 0.5 * qm_o.X(0) + 0.5 * qm_o.X(1)


def _gate_counts(qc):
    """Return a dict of {gate_name: count} from a Qiskit QuantumCircuit."""
    counts: dict[str, int] = {}
    for inst in qc.data:
        name = inst.operation.name
        counts[name] = counts.get(name, 0) + 1
    return counts


# -----------------------------------------------------------------------
# Validation tests
# -----------------------------------------------------------------------


class TestValidation:
    def test_step_zero_raises(self):
        with pytest.raises(ValueError, match="step must be a positive integer"):
            TrotterCircuit(_make_2term_h(), time=1.0, step=0, order=1)

    def test_step_negative_raises(self):
        with pytest.raises(ValueError, match="step must be a positive integer"):
            TrotterCircuit(_make_2term_h(), time=1.0, step=-1, order=1)

    def test_time_negative_raises(self):
        with pytest.raises(ValueError, match="time must be non-negative"):
            TrotterCircuit(_make_2term_h(), time=-0.5, step=1, order=1)

    def test_time_zero_allowed(self):
        tc = TrotterCircuit(_make_2term_h(), time=0.0, step=1, order=1)
        assert tc.time == 0.0

    def test_order_zero_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            TrotterCircuit(_make_2term_h(), time=1.0, step=1, order=0)

    def test_order_odd_3_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            TrotterCircuit(_make_2term_h(), time=1.0, step=1, order=3)

    def test_order_odd_5_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            TrotterCircuit(_make_2term_h(), time=1.0, step=1, order=5)

    def test_order_negative_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            TrotterCircuit(_make_2term_h(), time=1.0, step=1, order=-2)

    def test_single_term_hamiltonian_raises(self):
        h = qm_o.Z(0)
        with pytest.raises(ValueError, match="at least 2 terms"):
            TrotterCircuit(h, time=1.0, step=1, order=1)

    def test_order_1_allowed(self):
        tc = TrotterCircuit(_make_2term_h(), time=1.0, step=1, order=1)
        assert tc.order == 1

    def test_order_2_allowed(self):
        tc = TrotterCircuit(_make_2term_h(), time=1.0, step=1, order=2)
        assert tc.order == 2

    def test_order_4_allowed(self):
        tc = TrotterCircuit(_make_2term_h(), time=1.0, step=1, order=4)
        assert tc.order == 4


# -----------------------------------------------------------------------
# _split_hamiltonian tests
# -----------------------------------------------------------------------


class TestSplitHamiltonian:
    def test_two_terms(self):
        h = _make_2term_h()
        terms = _split_hamiltonian(h)
        assert len(terms) == 2
        for t in terms:
            assert isinstance(t, Hamiltonian)
            assert len(t) == 1
            assert t.num_qubits == h.num_qubits

    def test_three_terms(self):
        h = _make_3term_h()
        terms = _split_hamiltonian(h)
        assert len(terms) == 3

    def test_coefficients_preserved(self):
        h = _make_3term_h()
        terms = _split_hamiltonian(h)
        reconstructed = terms[0]
        for t in terms[1:]:
            reconstructed = reconstructed + t
        assert reconstructed.terms == h.terms


# -----------------------------------------------------------------------
# Product formula sequence tests
# -----------------------------------------------------------------------


class TestProductFormula:
    def test_order1_sequence_length(self):
        seq = _product_formula(n_terms=3, order=1, dt_frac=0.5)
        assert len(seq) == 3

    def test_order1_term_order(self):
        seq = _product_formula(n_terms=3, order=1, dt_frac=0.5)
        assert [idx for idx, _ in seq] == [0, 1, 2]

    def test_order1_coefficients(self):
        seq = _product_formula(n_terms=2, order=1, dt_frac=0.25)
        for _, frac in seq:
            assert abs(frac - 0.25) < 1e-15

    def test_order2_sequence_length(self):
        seq = _product_formula(n_terms=3, order=2, dt_frac=0.5)
        assert len(seq) == 6

    def test_order2_palindrome_structure(self):
        seq = _product_formula(n_terms=3, order=2, dt_frac=0.5)
        indices = [idx for idx, _ in seq]
        assert indices == [0, 1, 2, 2, 1, 0]

    def test_order2_coefficients_half_dt(self):
        seq = _product_formula(n_terms=2, order=2, dt_frac=0.5)
        for _, frac in seq:
            assert abs(frac - 0.25) < 1e-15

    def test_order4_structure(self):
        """order=4: 5 blocks of order=2, each with 2*m operations."""
        seq = _product_formula(n_terms=2, order=4, dt_frac=1.0)
        assert len(seq) == 20

    def test_order4_coefficient_sum(self):
        seq = _product_formula(n_terms=2, order=4, dt_frac=1.0)
        for term_idx in range(2):
            total = sum(frac for idx, frac in seq if idx == term_idx)
            assert abs(total - 1.0) < 1e-12

    def test_order2_coefficient_sum(self):
        seq = _product_formula(n_terms=3, order=2, dt_frac=0.5)
        for term_idx in range(3):
            total = sum(frac for idx, frac in seq if idx == term_idx)
            assert abs(total - 0.5) < 1e-15

    def test_order1_coefficient_sum(self):
        seq = _product_formula(n_terms=3, order=1, dt_frac=0.5)
        for term_idx in range(3):
            total = sum(frac for idx, frac in seq if idx == term_idx)
            assert abs(total - 0.5) < 1e-15

    def test_order6_coefficient_sum(self):
        seq = _product_formula(n_terms=2, order=6, dt_frac=1.0)
        for term_idx in range(2):
            total = sum(frac for idx, frac in seq if idx == term_idx)
            assert abs(total - 1.0) < 1e-10


class TestFullSequence:
    def test_steps_multiply_length(self):
        one = _product_formula(n_terms=2, order=1, dt_frac=1.0)
        full = _full_sequence(n_terms=2, order=1, step=3)
        assert len(full) == len(one) * 3

    def test_full_coefficient_sum(self):
        full = _full_sequence(n_terms=3, order=2, step=4)
        for term_idx in range(3):
            total = sum(frac for idx, frac in full if idx == term_idx)
            assert abs(total - 1.0) < 1e-12


# -----------------------------------------------------------------------
# TrotterCircuit properties
# -----------------------------------------------------------------------


class TestTrotterCircuitProperties:
    def test_num_qubits(self):
        tc = TrotterCircuit(_make_2term_h(), time=1.0, step=1, order=1)
        assert tc.num_qubits == 2

    def test_sub_hamiltonians_count(self):
        tc = TrotterCircuit(_make_3term_h(), time=1.0, step=1, order=1)
        assert len(tc.sub_hamiltonians) == 3

    def test_num_terms(self):
        tc = TrotterCircuit(_make_3term_h(), time=1.0, step=1, order=1)
        assert tc.num_terms == 3

    def test_sequence_absolute_coefficients(self):
        tc = TrotterCircuit(_make_2term_h(), time=2.0, step=1, order=1)
        for _, coeff in tc.sequence:
            assert abs(coeff - 2.0) < 1e-15

    def test_bindings_keys(self):
        tc = TrotterCircuit(_make_3term_h(), time=1.5, step=1, order=1)
        b = tc.bindings()
        assert b["time"] == 1.5
        assert "H_0" in b and "H_1" in b and "H_2" in b
        assert len(b) == 4

    def test_bindings_custom_time_key(self):
        tc = TrotterCircuit(_make_2term_h(), time=1.0, step=1, order=1)
        b = tc.bindings(time_key="t")
        assert "t" in b
        assert "time" not in b


# -----------------------------------------------------------------------
# Integration: evolve() + transpile via Qiskit
#
# Each test defines its own @qkernel wrapper so that the TrotterCircuit
# instance is captured via closure (avoids AST-transform namespace issues).
# -----------------------------------------------------------------------

pytest.importorskip("qiskit")

from qamomile.qiskit.transpiler import QiskitTranspiler  # noqa: E402


class TestTranspileOrder1:
    def test_pauli_evolve_count(self):
        """order=1, step=1, 2 terms → 2 PauliEvolution gates."""
        h = _make_2term_h()
        tc = TrotterCircuit(h, time=1.0, step=1, order=1)

        @qmc.qkernel
        def wrapper(
            n: qmc.UInt,
            time: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = tc.evolve(q, [H_0, H_1], time)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            wrapper, bindings={"n": h.num_qubits, **tc.bindings()}
        )
        qc = exe.compiled_quantum[0].circuit
        assert _gate_counts(qc).get("PauliEvolution", 0) == 2

    def test_step2_doubles_gates(self):
        """step=2 should double the PauliEvolution count vs step=1."""
        h = _make_2term_h()
        tr = QiskitTranspiler()

        tc1 = TrotterCircuit(h, time=1.0, step=1, order=1)

        @qmc.qkernel
        def wrapper1(
            n: qmc.UInt,
            time: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = tc1.evolve(q, [H_0, H_1], time)
            return qmc.measure(q)

        exe1 = tr.transpile(
            wrapper1, bindings={"n": h.num_qubits, **tc1.bindings()}
        )
        n1 = _gate_counts(exe1.compiled_quantum[0].circuit).get(
            "PauliEvolution", 0
        )

        tc2 = TrotterCircuit(h, time=1.0, step=2, order=1)

        @qmc.qkernel
        def wrapper2(
            n: qmc.UInt,
            time: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = tc2.evolve(q, [H_0, H_1], time)
            return qmc.measure(q)

        exe2 = tr.transpile(
            wrapper2, bindings={"n": h.num_qubits, **tc2.bindings()}
        )
        n2 = _gate_counts(exe2.compiled_quantum[0].circuit).get(
            "PauliEvolution", 0
        )

        assert n2 == 2 * n1


class TestTranspileOrder2:
    def test_pauli_evolve_count_2term(self):
        """order=2, step=1, 2 terms → 4 PauliEvolution gates."""
        h = _make_2term_h()
        tc = TrotterCircuit(h, time=1.0, step=1, order=2)

        @qmc.qkernel
        def wrapper(
            n: qmc.UInt,
            time: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = tc.evolve(q, [H_0, H_1], time)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            wrapper, bindings={"n": h.num_qubits, **tc.bindings()}
        )
        qc = exe.compiled_quantum[0].circuit
        assert _gate_counts(qc).get("PauliEvolution", 0) == 4

    def test_pauli_evolve_count_3term(self):
        """order=2, step=1, 3 terms → 6 PauliEvolution gates."""
        h = _make_3term_h()
        tc = TrotterCircuit(h, time=0.5, step=1, order=2)

        @qmc.qkernel
        def wrapper(
            n: qmc.UInt,
            time: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
            H_2: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = tc.evolve(q, [H_0, H_1, H_2], time)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            wrapper, bindings={"n": h.num_qubits, **tc.bindings()}
        )
        qc = exe.compiled_quantum[0].circuit
        assert _gate_counts(qc).get("PauliEvolution", 0) == 6


class TestTranspileHigherOrder:
    def test_order4_pauli_evolve_count(self):
        """order=4, step=1, 2 terms → 20 PauliEvolution gates."""
        h = _make_2term_h()
        tc = TrotterCircuit(h, time=1.0, step=1, order=4)

        @qmc.qkernel
        def wrapper(
            n: qmc.UInt,
            time: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = tc.evolve(q, [H_0, H_1], time)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            wrapper, bindings={"n": h.num_qubits, **tc.bindings()}
        )
        qc = exe.compiled_quantum[0].circuit
        assert _gate_counts(qc).get("PauliEvolution", 0) == 20


class TestTranspileTimeZero:
    def test_time_zero_compiles(self):
        """time=0 should produce a valid circuit (all angles 0)."""
        h = _make_2term_h()
        tc = TrotterCircuit(h, time=0.0, step=1, order=1)

        @qmc.qkernel
        def wrapper(
            n: qmc.UInt,
            time: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = tc.evolve(q, [H_0, H_1], time)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            wrapper, bindings={"n": h.num_qubits, **tc.bindings()}
        )
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits >= 1
