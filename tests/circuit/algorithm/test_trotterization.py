"""Tests for qamomile/circuit/algorithm/trotterization.py."""

import numpy as np
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


def _gate_counts(qc) -> dict[str, int]:
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
    @pytest.mark.parametrize(
        "step, order, match",
        [
            (0, 1, "step must be a positive integer"),
            (-1, 1, "step must be a positive integer"),
            (True, 1, "step must be a positive integer"),
            (False, 1, "step must be a positive integer"),
            (1, 0, "order must be 1 or a positive even"),
            (1, 3, "order must be 1 or a positive even"),
            (1, 5, "order must be 1 or a positive even"),
            (1, -2, "order must be 1 or a positive even"),
            (1, True, "order must be 1 or a positive even"),
            (1, False, "order must be 1 or a positive even"),
        ],
    )
    def test_invalid_inputs_raise(self, step, order, match):
        """Invalid step or order raises ValueError."""
        with pytest.raises(ValueError, match=match):
            _validate_inputs(n_terms=2, step=step, order=order)

    def test_single_term_raises(self):
        """n_terms < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_terms must be at least 2"):
            _validate_inputs(n_terms=1, step=1, order=1)

    @pytest.mark.parametrize("order", [1, 2, 4, 6])
    def test_valid_order_allowed(self, order):
        """Valid orders pass without error."""
        _validate_inputs(n_terms=2, step=1, order=order)


# -----------------------------------------------------------------------
# Product formula sequence tests
# -----------------------------------------------------------------------


class TestProductFormulaValidation:
    @pytest.mark.parametrize(
        "n_terms, order, match",
        [
            (2, 3, "order must be 1 or a positive even"),
            (2, 5, "order must be 1 or a positive even"),
            (2, 0, "order must be 1 or a positive even"),
            (2, -2, "order must be 1 or a positive even"),
            (2, True, "order must be 1 or a positive even"),
            (2, False, "order must be 1 or a positive even"),
            (1, 1, "n_terms must be at least 2"),
            (0, 1, "n_terms must be at least 2"),
        ],
    )
    def test_invalid_inputs_raise(self, n_terms, order, match):
        """Invalid n_terms or order raises ValueError."""
        with pytest.raises(ValueError, match=match):
            product_formula(n_terms=n_terms, dt_frac=1.0, order=order)

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_valid_inputs_do_not_raise(self, order):
        """Valid inputs pass without error."""
        product_formula(n_terms=2, dt_frac=1.0, order=order)


class TestProductFormula:
    def test_order1_sequence_length(self):
        """Order-1 produces exactly n_terms entries."""
        seq = product_formula(n_terms=3, dt_frac=0.5, order=1)
        assert len(seq) == 3

    def test_order1_term_order(self):
        """Order-1 iterates terms in forward order."""
        seq = product_formula(n_terms=3, dt_frac=0.5, order=1)
        assert [idx for idx, _ in seq] == [0, 1, 2]

    def test_order1_coefficients(self):
        """Order-1 assigns dt_frac to each term."""
        seq = product_formula(n_terms=2, dt_frac=0.25, order=1)
        for _, frac in seq:
            assert abs(frac - 0.25) < 1e-15

    def test_order2_sequence_length(self):
        """Order-2 produces 2 * n_terms entries (forward + backward)."""
        seq = product_formula(n_terms=3, dt_frac=0.5, order=2)
        assert len(seq) == 6

    def test_order2_palindrome_structure(self):
        """Order-2 has palindrome index structure."""
        seq = product_formula(n_terms=3, dt_frac=0.5, order=2)
        indices = [idx for idx, _ in seq]
        assert indices == [0, 1, 2, 2, 1, 0]

    def test_order2_coefficients_half_dt(self):
        """Order-2 assigns dt_frac/2 to each entry."""
        seq = product_formula(n_terms=2, dt_frac=0.5, order=2)
        for _, frac in seq:
            assert abs(frac - 0.25) < 1e-15

    def test_order4_structure(self):
        """Order-4 produces 5 blocks of order-2 = 20 entries for 2 terms."""
        seq = product_formula(n_terms=2, dt_frac=1.0, order=4)
        assert len(seq) == 20

    @pytest.mark.parametrize(
        "n_terms, dt_frac, order, atol",
        [
            (2, 1.0, 1, 1e-15),
            (3, 0.5, 1, 1e-15),
            (3, 0.5, 2, 1e-15),
            (2, 1.0, 4, 1e-12),
            (2, 1.0, 6, 1e-10),
        ],
    )
    def test_coefficient_sum(self, n_terms, dt_frac, order, atol):
        """Per-term coefficient sum equals dt_frac."""
        seq = product_formula(n_terms=n_terms, dt_frac=dt_frac, order=order)
        for term_idx in range(n_terms):
            total = sum(frac for idx, frac in seq if idx == term_idx)
            assert abs(total - dt_frac) < atol

    @pytest.mark.parametrize("seed", range(5))
    def test_coefficient_sum_random(self, seed):
        """Per-term coefficient sum equals dt_frac for random inputs."""
        rng = np.random.default_rng(seed)
        n_terms = int(rng.integers(2, 6))
        dt_frac = float(rng.uniform(0.01, 2.0))
        for order in [1, 2, 4]:
            seq = product_formula(n_terms=n_terms, dt_frac=dt_frac, order=order)
            for term_idx in range(n_terms):
                total = sum(frac for idx, frac in seq if idx == term_idx)
                assert abs(total - dt_frac) < 1e-10

    @pytest.mark.parametrize("n_terms", [2, 5, 10])
    def test_scaling_with_n_terms(self, n_terms):
        """Sequence length and index range scale correctly with n_terms."""
        seq_o1 = product_formula(n_terms=n_terms, dt_frac=1.0, order=1)
        assert len(seq_o1) == n_terms
        assert set(idx for idx, _ in seq_o1) == set(range(n_terms))

        seq_o2 = product_formula(n_terms=n_terms, dt_frac=1.0, order=2)
        assert len(seq_o2) == 2 * n_terms
        # Palindrome structure
        indices = [idx for idx, _ in seq_o2]
        assert indices == list(range(n_terms)) + list(range(n_terms - 1, -1, -1))


class TestFullSequence:
    def test_steps_multiply_length(self):
        """Full sequence length equals one_step length * step."""
        one = product_formula(n_terms=2, dt_frac=1.0, order=1)
        full = list(_full_sequence(n_terms=2, step=3, order=1))
        assert len(full) == len(one) * 3

    def test_full_coefficient_sum(self):
        """Full sequence coefficients sum to 1.0 for each term."""
        full = list(_full_sequence(n_terms=3, step=4, order=2))
        for term_idx in range(3):
            total = sum(frac for idx, frac in full if idx == term_idx)
            assert abs(total - 1.0) < 1e-12


# -----------------------------------------------------------------------
# Integration: trotterized_time_evolution() + transpile via Qiskit
#
# @qkernel wrappers defined at module level for inspect.getsource().
# -----------------------------------------------------------------------

pytest.importorskip("qiskit")

from qamomile.qiskit.transpiler import QiskitTranspiler  # noqa: E402


@qmc.qkernel
def _wrap_trotter_2term_o1_s1(
    n: qmc.UInt,
    t: qmc.Float,
    H_0: qmc.Observable,
    H_1: qmc.Observable,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = trotterized_time_evolution(q, [H_0, H_1], t, step=1, order=1)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_trotter_2term_o1_s2(
    n: qmc.UInt,
    t: qmc.Float,
    H_0: qmc.Observable,
    H_1: qmc.Observable,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = trotterized_time_evolution(q, [H_0, H_1], t, step=2, order=1)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_trotter_2term_o2(
    n: qmc.UInt,
    t: qmc.Float,
    H_0: qmc.Observable,
    H_1: qmc.Observable,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = trotterized_time_evolution(q, [H_0, H_1], t, step=1, order=2)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_trotter_3term_o2(
    n: qmc.UInt,
    t: qmc.Float,
    H_0: qmc.Observable,
    H_1: qmc.Observable,
    H_2: qmc.Observable,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = trotterized_time_evolution(q, [H_0, H_1, H_2], t, step=1, order=2)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_trotter_2term_o4(
    n: qmc.UInt,
    t: qmc.Float,
    H_0: qmc.Observable,
    H_1: qmc.Observable,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = trotterized_time_evolution(q, [H_0, H_1], t, step=1, order=4)
    return qmc.measure(q)


class TestTranspileOrder1:
    def test_pauli_evolve_count(self):
        """order=1, step=1, 2 terms produces 2 PauliEvolution gates."""
        h = _make_2term_h()
        sub_hs = _split_hamiltonian(h)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            _wrap_trotter_2term_o1_s1,
            bindings={"n": h.num_qubits, "t": 1.0, "H_0": sub_hs[0], "H_1": sub_hs[1]},
        )
        qc = exe.compiled_quantum[0].circuit
        assert _gate_counts(qc).get("PauliEvolution", 0) == 2

    def test_step2_doubles_gates(self):
        """step=2 doubles the PauliEvolution count vs step=1."""
        h = _make_2term_h()
        sub_hs = _split_hamiltonian(h)
        tr = QiskitTranspiler()
        bindings = {"n": h.num_qubits, "t": 1.0, "H_0": sub_hs[0], "H_1": sub_hs[1]}

        exe1 = tr.transpile(_wrap_trotter_2term_o1_s1, bindings=bindings)
        n1 = _gate_counts(exe1.compiled_quantum[0].circuit).get("PauliEvolution", 0)

        exe2 = tr.transpile(_wrap_trotter_2term_o1_s2, bindings=bindings)
        n2 = _gate_counts(exe2.compiled_quantum[0].circuit).get("PauliEvolution", 0)

        assert n2 == 2 * n1


class TestTranspileOrder2:
    def test_pauli_evolve_count_2term(self):
        """order=2, step=1, 2 terms produces 4 PauliEvolution gates."""
        h = _make_2term_h()
        sub_hs = _split_hamiltonian(h)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            _wrap_trotter_2term_o2,
            bindings={"n": h.num_qubits, "t": 1.0, "H_0": sub_hs[0], "H_1": sub_hs[1]},
        )
        qc = exe.compiled_quantum[0].circuit
        assert _gate_counts(qc).get("PauliEvolution", 0) == 4

    def test_pauli_evolve_count_3term(self):
        """order=2, step=1, 3 terms produces 6 PauliEvolution gates."""
        h = _make_3term_h()
        sub_hs = _split_hamiltonian(h)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            _wrap_trotter_3term_o2,
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


class TestTranspileHigherOrder:
    def test_order4_pauli_evolve_count(self):
        """order=4, step=1, 2 terms produces 20 PauliEvolution gates."""
        h = _make_2term_h()
        sub_hs = _split_hamiltonian(h)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            _wrap_trotter_2term_o4,
            bindings={"n": h.num_qubits, "t": 1.0, "H_0": sub_hs[0], "H_1": sub_hs[1]},
        )
        qc = exe.compiled_quantum[0].circuit
        assert _gate_counts(qc).get("PauliEvolution", 0) == 20


class TestTranspileTimeZero:
    def test_time_zero_compiles(self):
        """time=0 produces a valid circuit (all angles 0)."""
        h = _make_2term_h()
        sub_hs = _split_hamiltonian(h)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            _wrap_trotter_2term_o1_s1,
            bindings={"n": h.num_qubits, "t": 0.0, "H_0": sub_hs[0], "H_1": sub_hs[1]},
        )
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits >= 1
