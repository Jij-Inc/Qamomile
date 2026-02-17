import numpy as np

import qamomile.observable as qm_o
from qamomile.optimization.qaoa import QAOAConverter
from qamomile.optimization.binary_model import BinaryExpr, BinaryModel, binary
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


def test_hubo_transpile():
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


def test_hubo_energy_optimization():
    """End-to-end test: minimize a HUBO objective with QAOA."""
    import scipy.optimize

    # Problem: minimize x0*x1*x2 - 2*x0 - 2*x1 - 2*x2
    # Optimal: x0=1, x1=1, x2=1 → energy = 1 - 2 - 2 - 2 = -5
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
    assert abs(best_energy - (-5.0)) < 1e-6
