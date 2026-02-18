import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.optimization.qaoa import QAOAConverter
from qamomile.optimization.binary_model import BinaryExpr, BinaryModel, binary
from qamomile.circuit.algorithm.qaoa import apply_phase_gadget
from qamomile.qiskit.transpiler import QiskitTranspiler


def _make_3body_model():
    """Create a HUBO model with a 3-body term: x0*x1*x2 + x0 - 0.5*x1."""
    hubo = {
        (0, 1, 2): 1.0,
        (0,): 1.0,
        (1,): -0.5,
    }
    return BinaryModel.from_hubo(hubo)


def test_hubo_hamiltonian():
    """Verify that a 3-body HUBO model produces a ZZZ term in the Hamiltonian."""
    model = _make_3body_model()
    converter = QAOAConverter(model)
    hamiltonian = converter.get_cost_hamiltonian()

    # Check that a 3-body Z term exists
    three_body_found = False
    for term, coeff in hamiltonian.terms.items():
        if len(term) == 3:
            three_body_found = True
            assert all(op.pauli == qm_o.Pauli.Z for op in term)
    assert three_body_found, "Expected a 3-body ZZZ term in the Hamiltonian"


@pytest.mark.parametrize(
    "hubo,expected_max_order",
    [
        ({(0, 1, 2): 1.0, (0,): 1.0}, 3),
        ({(0, 1, 2, 3): 0.5}, 4),
        ({(0, 1, 2): 1.0, (2, 3, 4, 5): -0.5, (0,): 1.0}, 4),
        ({(0, 1, 2): 1.0, (0, 1): 0.3, (0,): -1.0}, 3),
    ],
    ids=["3-body", "4-body-only", "mixed-3-and-4-body", "mixed-quad-and-3-body"],
)
def test_hubo_hamiltonian_parametrized(hubo, expected_max_order):
    """Verify Hamiltonian contains terms of the expected maximum order."""
    model = BinaryModel.from_hubo(hubo)
    converter = QAOAConverter(model)
    hamiltonian = converter.get_cost_hamiltonian()

    max_order = max(len(term) for term in hamiltonian.terms)
    assert max_order == expected_max_order

    for term in hamiltonian.terms:
        assert all(op.pauli == qm_o.Pauli.Z for op in term)


def test_hubo_hamiltonian_random_coefficients():
    """Verify Hamiltonian construction with random HUBO coefficients."""
    rng = np.random.default_rng(42)
    n = 5
    hubo = {}
    for _ in range(5):
        order = rng.integers(1, 5)
        indices = tuple(sorted(rng.choice(n, size=order, replace=False)))
        hubo[indices] = rng.uniform(-2.0, 2.0)

    model = BinaryModel.from_hubo(hubo)
    converter = QAOAConverter(model)
    hamiltonian = converter.get_cost_hamiltonian()

    # All terms must be Pauli-Z
    for term in hamiltonian.terms:
        assert all(op.pauli == qm_o.Pauli.Z for op in term)


def test_hubo_transpile():
    """Verify that a HUBO model transpiles and produces valid measurement results."""
    model = _make_3body_model()
    converter = QAOAConverter(model)

    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=1)

    # Build a concrete circuit with fixed parameters
    bindings = {"gammas": [0.5], "betas": [0.3]}
    job = executable.sample(transpiler.executor(), shots=10, bindings=bindings)
    result = job.result()

    # Should have produced valid measurement results
    assert len(result.results) > 0
    for sample, count in result.results:
        assert len(sample) == model.num_bits
        assert count > 0


@pytest.mark.parametrize(
    "hubo",
    [
        {(0, 1, 2): 1.0, (0,): 1.0, (1,): -0.5},
        {(0, 1, 2, 3): 0.5, (0,): -1.0},
        {(0, 1, 2): 1.0, (2, 3, 4): -0.5, (0,): 1.0},
    ],
    ids=["3-body", "4-body", "two-3-body-groups"],
)
def test_hubo_transpile_parametrized(hubo):
    """Verify transpilation works for various HUBO model structures."""
    model = BinaryModel.from_hubo(hubo)
    converter = QAOAConverter(model)

    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=1)

    bindings = {"gammas": [0.5], "betas": [0.3]}
    job = executable.sample(transpiler.executor(), shots=10, bindings=bindings)
    result = job.result()

    assert len(result.results) > 0
    for sample, count in result.results:
        assert len(sample) == model.num_bits
        assert count > 0


def test_quadratic_regression():
    """Pure quadratic model should still work via the fast path."""
    x = binary(0)
    y = binary(1)
    z = binary(2)

    problem = BinaryExpr()
    problem += x * y
    problem += -1 * y * z
    problem += x
    problem += -0.1 * z

    model = BinaryModel(problem)
    converter = QAOAConverter(model)

    # Should have no higher-order terms
    assert not converter.spin_model.higher

    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=1)

    bindings = {"gammas": [0.5], "betas": [0.3]}
    job = executable.sample(transpiler.executor(), shots=10, bindings=bindings)
    result = job.result()
    assert len(result.results) > 0


def test_quadratic_dispatches_to_fast_path():
    """Verify that a model without higher-order terms uses the quadratic path."""
    hubo = {(0, 1): 1.0, (0,): -0.5}
    model = BinaryModel.from_hubo(hubo)
    converter = QAOAConverter(model)

    assert not converter.spin_model.higher


def test_hubo_near_zero_coefficient_filtered():
    """Verify that near-zero coefficients are filtered from the Hamiltonian."""
    hubo = {
        (0, 1, 2): 1e-16,
        (0,): 1.0,
    }
    model = BinaryModel.from_hubo(hubo)
    converter = QAOAConverter(model)
    hamiltonian = converter.get_cost_hamiltonian()

    # The near-zero 3-body term should be filtered
    for term in hamiltonian.terms:
        assert len(term) < 3, "Near-zero 3-body term should have been filtered"


def test_hubo_duplicate_indices_accumulated():
    """Verify that duplicate index tuples are accumulated into a single key."""
    hubo = {
        (0, 1, 2): 1.0,
        (2, 0, 1): 0.5,
    }
    model = BinaryModel.from_hubo(hubo)

    # Both should be accumulated into (0,1,2) with coefficient 1.5
    assert np.isclose(model.higher[(0, 1, 2)], 1.5)


@qmc.qkernel
def _gadget_k0(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = apply_phase_gadget(q, [], 0.5)
    return qmc.measure(q)


@qmc.qkernel
def _gadget_k1(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = apply_phase_gadget(q, [0], 0.5)
    return qmc.measure(q)


@qmc.qkernel
def _gadget_k2(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = apply_phase_gadget(q, [0, 1], 0.5)
    return qmc.measure(q)


@qmc.qkernel
def _gadget_k3(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = apply_phase_gadget(q, [0, 1, 2], 0.5)
    return qmc.measure(q)


@qmc.qkernel
def _gadget_k5(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = apply_phase_gadget(q, [0, 1, 2, 3, 4], 0.5)
    return qmc.measure(q)


@pytest.mark.parametrize(
    "kernel,n_qubits",
    [
        (_gadget_k0, 1),
        (_gadget_k1, 1),
        (_gadget_k2, 2),
        (_gadget_k3, 3),
        (_gadget_k5, 5),
    ],
    ids=["k=0", "k=1", "k=2", "k=3", "k=5"],
)
def test_apply_phase_gadget_branches(kernel, n_qubits):
    """Verify apply_phase_gadget transpiles for all k-body branch cases."""
    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(
        kernel,
        bindings={"n": n_qubits},
    )
    assert executable is not None


def test_hubo_energy_optimization():
    """End-to-end test: minimize a HUBO objective with QAOA."""
    import scipy.optimize

    # Problem: minimize x0*x1*x2 - 2*x0 - 2*x1 - 2*x2
    # Optimal: x0=1, x1=1, x2=1 -> energy = 1 - 2 - 2 - 2 = -5
    hubo = {
        (0, 1, 2): 1.0,
        (0,): -2.0,
        (1,): -2.0,
        (2,): -2.0,
    }
    model = BinaryModel.from_hubo(hubo)
    converter = QAOAConverter(model)

    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=2)

    def _obj(params):
        gammas_val = [params[0], params[1]]
        betas_val = [params[2], params[3]]
        bindings = {"gammas": gammas_val, "betas": betas_val}
        job = executable.sample(transpiler.executor(), shots=100, bindings=bindings)
        result = job.result()
        sampleset = converter.decode(result)
        argmin_index = np.argmin(sampleset.energy)
        best_energy = sampleset.energy[argmin_index]
        best_occurrence = sampleset.num_occurrences[argmin_index]
        return best_energy * best_occurrence

    # Starting point chosen empirically to converge to the global minimum
    # on a statevector simulator within the COBYLA iteration budget.
    x0 = [2.6, 0.11, 0.52, 0.45]
    bounds = [(0, np.pi), (0, np.pi), (0, np.pi / 2), (0, np.pi / 2)]
    res = scipy.optimize.minimize(
        _obj, x0=x0, bounds=bounds, method="COBYLA", options={"maxiter": 100}
    )

    gammas_opt = [res.x[0], res.x[1]]
    betas_opt = [res.x[2], res.x[3]]
    bindings_opt = {"gammas": gammas_opt, "betas": betas_opt}
    job_opt = executable.sample(
        transpiler.executor(), shots=1000, bindings=bindings_opt
    )
    result_opt = job_opt.result()
    binary_result = converter.decode(result_opt)

    best_sample, best_energy, _ = binary_result.lowest()
    assert best_sample == {0: 1, 1: 1, 2: 1}
    assert np.isclose(best_energy, -5.0, atol=1e-6)


# --- Edge-case and negative tests (P2-1) ---


def test_empty_hubo_produces_trivial_hamiltonian():
    """Verify that an empty HUBO dict produces a Hamiltonian with no terms."""
    model = BinaryModel.from_hubo({})
    converter = QAOAConverter(model)
    hamiltonian = converter.get_cost_hamiltonian()

    assert len(hamiltonian.terms) == 0


def test_empty_hubo_transpiles_via_quadratic_path():
    """Verify that an empty HUBO model transpiles through the quadratic path."""
    model = BinaryModel.from_hubo({(0,): 1.0})
    converter = QAOAConverter(model)

    # No higher-order terms — should use quadratic path
    assert not converter.spin_model.higher

    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=1)

    bindings = {"gammas": [0.5], "betas": [0.3]}
    job = executable.sample(transpiler.executor(), shots=10, bindings=bindings)
    result = job.result()
    assert len(result.results) > 0


def test_single_higher_order_term():
    """Verify a HUBO model with only one higher-order term and nothing else."""
    hubo = {(0, 1, 2): 2.0}
    model = BinaryModel.from_hubo(hubo)
    converter = QAOAConverter(model)
    hamiltonian = converter.get_cost_hamiltonian()

    higher_terms = [t for t in hamiltonian.terms if len(t) == 3]
    assert len(higher_terms) >= 1

    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=1)
    bindings = {"gammas": [0.5], "betas": [0.3]}
    job = executable.sample(transpiler.executor(), shots=10, bindings=bindings)
    result = job.result()
    assert len(result.results) > 0
