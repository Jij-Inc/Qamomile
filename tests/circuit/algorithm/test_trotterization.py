"""Tests for qamomile/circuit/algorithm/trotterization.py."""

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.trotterization import (
    _full_sequence,
    _validate_inputs,
    product_formula,
    trotterized_time_evolution,
)
from qamomile.observable.hamiltonian import Hamiltonian

# -----------------------------------------------------------------------
# Helper: build common Hamiltonians and split them
# -----------------------------------------------------------------------


def _split_hamiltonian(hamiltonian: Hamiltonian) -> list[Hamiltonian]:
    """Split a Hamiltonian into individual single-term Hamiltonians."""
    terms: list[Hamiltonian] = []
    for ops, coeff in hamiltonian:
        h = Hamiltonian(num_qubits=hamiltonian.num_qubits)
        h.add_term(ops, coeff)
        terms.append(h)
    return terms


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
            _validate_inputs(n_terms=2, step=0, order=1)

    def test_step_negative_raises(self):
        with pytest.raises(ValueError, match="step must be a positive integer"):
            _validate_inputs(n_terms=2, step=-1, order=1)

    def test_order_zero_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            _validate_inputs(n_terms=2, step=1, order=0)

    def test_order_odd_3_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            _validate_inputs(n_terms=2, step=1, order=3)

    def test_order_odd_5_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            _validate_inputs(n_terms=2, step=1, order=5)

    def test_order_negative_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            _validate_inputs(n_terms=2, step=1, order=-2)

    def test_single_term_raises(self):
        with pytest.raises(ValueError, match="at least 2 terms"):
            _validate_inputs(n_terms=1, step=1, order=1)

    def test_order_1_allowed(self):
        _validate_inputs(n_terms=2, step=1, order=1)

    def test_order_2_allowed(self):
        _validate_inputs(n_terms=2, step=1, order=2)

    def test_order_4_allowed(self):
        _validate_inputs(n_terms=2, step=1, order=4)

    def test_step_bool_true_raises(self):
        with pytest.raises(ValueError, match="step must be a positive integer"):
            _validate_inputs(n_terms=2, step=True, order=1)

    def test_step_bool_false_raises(self):
        with pytest.raises(ValueError, match="step must be a positive integer"):
            _validate_inputs(n_terms=2, step=False, order=1)

    def test_order_bool_true_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            _validate_inputs(n_terms=2, step=1, order=True)

    def test_order_bool_false_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            _validate_inputs(n_terms=2, step=1, order=False)


# -----------------------------------------------------------------------
# Product formula sequence tests
# -----------------------------------------------------------------------


class TestProductFormulaValidation:
    def test_order_odd_3_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            product_formula(n_terms=2, order=3, dt_frac=1.0)

    def test_order_odd_5_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            product_formula(n_terms=2, order=5, dt_frac=1.0)

    def test_order_zero_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            product_formula(n_terms=2, order=0, dt_frac=1.0)

    def test_order_negative_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            product_formula(n_terms=2, order=-2, dt_frac=1.0)

    def test_n_terms_one_raises(self):
        with pytest.raises(ValueError, match="n_terms must be at least 2"):
            product_formula(n_terms=1, order=1, dt_frac=1.0)

    def test_n_terms_zero_raises(self):
        with pytest.raises(ValueError, match="n_terms must be at least 2"):
            product_formula(n_terms=0, order=1, dt_frac=1.0)

    def test_valid_order1_does_not_raise(self):
        product_formula(n_terms=2, order=1, dt_frac=1.0)

    def test_valid_order2_does_not_raise(self):
        product_formula(n_terms=2, order=2, dt_frac=1.0)

    def test_order_bool_true_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            product_formula(n_terms=2, order=True, dt_frac=1.0)

    def test_order_bool_false_raises(self):
        with pytest.raises(ValueError, match="order must be 1 or a positive even"):
            product_formula(n_terms=2, order=False, dt_frac=1.0)


class TestProductFormula:
    def test_order1_sequence_length(self):
        seq = product_formula(n_terms=3, order=1, dt_frac=0.5)
        assert len(seq) == 3

    def test_order1_term_order(self):
        seq = product_formula(n_terms=3, order=1, dt_frac=0.5)
        assert [idx for idx, _ in seq] == [0, 1, 2]

    def test_order1_coefficients(self):
        seq = product_formula(n_terms=2, order=1, dt_frac=0.25)
        for _, frac in seq:
            assert abs(frac - 0.25) < 1e-15

    def test_order2_sequence_length(self):
        seq = product_formula(n_terms=3, order=2, dt_frac=0.5)
        assert len(seq) == 6

    def test_order2_palindrome_structure(self):
        seq = product_formula(n_terms=3, order=2, dt_frac=0.5)
        indices = [idx for idx, _ in seq]
        assert indices == [0, 1, 2, 2, 1, 0]

    def test_order2_coefficients_half_dt(self):
        seq = product_formula(n_terms=2, order=2, dt_frac=0.5)
        for _, frac in seq:
            assert abs(frac - 0.25) < 1e-15

    def test_order4_structure(self):
        """order=4: 5 blocks of order=2, each with 2*m operations."""
        seq = product_formula(n_terms=2, order=4, dt_frac=1.0)
        assert len(seq) == 20

    def test_order4_coefficient_sum(self):
        seq = product_formula(n_terms=2, order=4, dt_frac=1.0)
        for term_idx in range(2):
            total = sum(frac for idx, frac in seq if idx == term_idx)
            assert abs(total - 1.0) < 1e-12

    def test_order2_coefficient_sum(self):
        seq = product_formula(n_terms=3, order=2, dt_frac=0.5)
        for term_idx in range(3):
            total = sum(frac for idx, frac in seq if idx == term_idx)
            assert abs(total - 0.5) < 1e-15

    def test_order1_coefficient_sum(self):
        seq = product_formula(n_terms=3, order=1, dt_frac=0.5)
        for term_idx in range(3):
            total = sum(frac for idx, frac in seq if idx == term_idx)
            assert abs(total - 0.5) < 1e-15

    def test_order6_coefficient_sum(self):
        seq = product_formula(n_terms=2, order=6, dt_frac=1.0)
        for term_idx in range(2):
            total = sum(frac for idx, frac in seq if idx == term_idx)
            assert abs(total - 1.0) < 1e-10


class TestFullSequence:
    def test_steps_multiply_length(self):
        one = product_formula(n_terms=2, order=1, dt_frac=1.0)
        full = list(_full_sequence(n_terms=2, order=1, step=3))
        assert len(full) == len(one) * 3

    def test_full_coefficient_sum(self):
        full = list(_full_sequence(n_terms=3, order=2, step=4))
        for term_idx in range(3):
            total = sum(frac for idx, frac in full if idx == term_idx)
            assert abs(total - 1.0) < 1e-12


# -----------------------------------------------------------------------
# Integration: trotterized_time_evolution() + transpile via Qiskit
# -----------------------------------------------------------------------

try:
    from qamomile.qiskit.transpiler import QiskitTranspiler

    _has_qiskit = True
except ImportError:
    _has_qiskit = False

_requires_qiskit = pytest.mark.skipif(not _has_qiskit, reason="Qiskit not installed")


@_requires_qiskit
class TestTranspileOrder1:
    def test_pauli_evolve_count(self):
        """order=1, step=1, 2 terms → 2 PauliEvolution gates."""
        h = _make_2term_h()
        sub_hs = _split_hamiltonian(h)

        @qmc.qkernel
        def wrapper(
            n: qmc.UInt,
            t: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = trotterized_time_evolution(q, [H_0, H_1], t, step=1, order=1)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            wrapper,
            bindings={"n": h.num_qubits, "t": 1.0, "H_0": sub_hs[0], "H_1": sub_hs[1]},
        )
        qc = exe.compiled_quantum[0].circuit
        assert _gate_counts(qc).get("PauliEvolution", 0) == 2

    def test_step2_doubles_gates(self):
        """step=2 should double the PauliEvolution count vs step=1."""
        h = _make_2term_h()
        sub_hs = _split_hamiltonian(h)
        tr = QiskitTranspiler()

        @qmc.qkernel
        def wrapper1(
            n: qmc.UInt,
            t: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = trotterized_time_evolution(q, [H_0, H_1], t, step=1, order=1)
            return qmc.measure(q)

        exe1 = tr.transpile(
            wrapper1,
            bindings={"n": h.num_qubits, "t": 1.0, "H_0": sub_hs[0], "H_1": sub_hs[1]},
        )
        n1 = _gate_counts(exe1.compiled_quantum[0].circuit).get("PauliEvolution", 0)

        @qmc.qkernel
        def wrapper2(
            n: qmc.UInt,
            t: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = trotterized_time_evolution(q, [H_0, H_1], t, step=2, order=1)
            return qmc.measure(q)

        exe2 = tr.transpile(
            wrapper2,
            bindings={"n": h.num_qubits, "t": 1.0, "H_0": sub_hs[0], "H_1": sub_hs[1]},
        )
        n2 = _gate_counts(exe2.compiled_quantum[0].circuit).get("PauliEvolution", 0)

        assert n2 == 2 * n1


@_requires_qiskit
class TestTranspileOrder2:
    def test_pauli_evolve_count_2term(self):
        """order=2, step=1, 2 terms → 4 PauliEvolution gates."""
        h = _make_2term_h()
        sub_hs = _split_hamiltonian(h)

        @qmc.qkernel
        def wrapper(
            n: qmc.UInt,
            t: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = trotterized_time_evolution(q, [H_0, H_1], t, step=1, order=2)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            wrapper,
            bindings={"n": h.num_qubits, "t": 1.0, "H_0": sub_hs[0], "H_1": sub_hs[1]},
        )
        qc = exe.compiled_quantum[0].circuit
        assert _gate_counts(qc).get("PauliEvolution", 0) == 4

    def test_pauli_evolve_count_3term(self):
        """order=2, step=1, 3 terms → 6 PauliEvolution gates."""
        h = _make_3term_h()
        sub_hs = _split_hamiltonian(h)

        @qmc.qkernel
        def wrapper(
            n: qmc.UInt,
            t: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
            H_2: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = trotterized_time_evolution(q, [H_0, H_1, H_2], t, step=1, order=2)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            wrapper,
            bindings={
                "n": h.num_qubits,
                "t": 0.5,
                "H_0": sub_hs[0],
                "H_1": sub_hs[1],
                "H_2": sub_hs[2],
            },
        )
        qc = exe.compiled_quantum[0].circuit
        assert _gate_counts(qc).get("PauliEvolution", 0) == 6


@_requires_qiskit
class TestTranspileHigherOrder:
    def test_order4_pauli_evolve_count(self):
        """order=4, step=1, 2 terms → 20 PauliEvolution gates."""
        h = _make_2term_h()
        sub_hs = _split_hamiltonian(h)

        @qmc.qkernel
        def wrapper(
            n: qmc.UInt,
            t: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = trotterized_time_evolution(q, [H_0, H_1], t, step=1, order=4)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            wrapper,
            bindings={"n": h.num_qubits, "t": 1.0, "H_0": sub_hs[0], "H_1": sub_hs[1]},
        )
        qc = exe.compiled_quantum[0].circuit
        assert _gate_counts(qc).get("PauliEvolution", 0) == 20


@_requires_qiskit
class TestTranspileTimeZero:
    def test_time_zero_compiles(self):
        """time=0 should produce a valid circuit (all angles 0)."""
        h = _make_2term_h()
        sub_hs = _split_hamiltonian(h)

        @qmc.qkernel
        def wrapper(
            n: qmc.UInt,
            t: qmc.Float,
            H_0: qmc.Observable,
            H_1: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = trotterized_time_evolution(q, [H_0, H_1], t, step=1, order=1)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            wrapper,
            bindings={"n": h.num_qubits, "t": 0.0, "H_0": sub_hs[0], "H_1": sub_hs[1]},
        )
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits >= 1
