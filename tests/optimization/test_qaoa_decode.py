from qamomile.optimization.qaoa import QAOAConverter
from qamomile.qiskit.transpiler import QiskitTranspiler
from qamomile.optimization.binary_model import binary, BinaryExpr, BinaryModel
import jijmodeling as jm
from math import isclose


def test_simple_qaoa_decode():
    x = binary(0)
    y = binary(1)
    z = binary(2)

    problem = BinaryExpr()
    problem += x * y
    problem += -1 * y * z
    problem += x
    problem += -0.1 * z

    model = BinaryModel(problem)

    # x y - y z + x
    # binary to spin: x = (1 - s_x) / 2
    # x = (1 - s_x) / 2
    # x y = (1 - s_x) / 2 * (1 - s_y) / 2 = 1/4(1 - s_x - s_y + s_x s_y)
    # - y z = - (1 - s_y) / 2 * (1 - s_z) / 2 = -1/4(1 - s_y - s_z + s_y s_z)
    # --------------------------------------------
    # total
    # = 1/2-1/2*0.1 -3/4 s_x + 1/4 s_x s_y + (1/4 + 1/2*0.1) s_z - 1/4 s_y s_z

    converter = QAOAConverter(model)

    assert isclose(converter.spin_model.constant, 1 / 2 - 1 / 2 * 0.1)
    assert set(converter.spin_model.linear) == {0, 2}
    assert isclose(converter.spin_model.linear[0], -3 / 4)
    assert isclose(converter.spin_model.linear[2], 1 / 4 + 1 / 2 * 0.1)
    assert set(converter.spin_model.quad) == {(0, 1), (1, 2)}
    assert isclose(converter.spin_model.quad[(0, 1)], 1 / 4)
    assert isclose(converter.spin_model.quad[(1, 2)], -1 / 4)

    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=2)  # p=2に変更してより表現力を高める

    import scipy

    def _obj(params):
        gammas_val = [params[0], params[1]]
        betas_val = [params[2], params[3]]
        bindings = {"gammas": gammas_val, "betas": betas_val}
        job = executable.sample(
            transpiler.executor(), shots=100, bindings=bindings
        )  # さらにshotsを増やす
        result = job.result()

        sampleset = converter.decode(result)
        argmin_index = np.argmin(sampleset.energy)
        best_energy = sampleset.energy[argmin_index]
        best_occurrence = sampleset.num_occurrences[argmin_index]
        return best_energy * best_occurrence

    import numpy as np

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
    print(binary_result)

    # Optimal solution: x=0, y=1, z=1 with energy = 0*1 - 1*1 + 0 - 0.1*1 = -1.1
    best_sample, best_energy, _ = binary_result.lowest()
    assert best_sample == {0: 0, 1: 1, 2: 1}
    assert abs(best_energy - (-1.1)) < 1e-6


def test_qaoa_decode():
    problem = jm.Problem("sample")

    @problem.update
    def _(problem: jm.DecoratedProblem):
        x = problem.BinaryVar()
        y = problem.BinaryVar()
        z = problem.BinaryVar()

        problem += x * y
        problem += -1 * y * z
        problem += x
        problem += -0.1 * z

    instance = problem.eval({})

    converter = QAOAConverter(instance)

    # x y - y z + x
    # binary to spin: x = (1 - s_x) / 2
    # x = (1 - s_x) / 2
    # x y = (1 - s_x) / 2 * (1 - s_y) / 2 = 1/4(1 - s_x - s_y + s_x s_y)
    # - y z = - (1 - s_y) / 2 * (1 - s_z) / 2 = -1/4(1 - s_y - s_z + s_y s_z)
    # --------------------------------------------
    # total
    # = 1/2-1/2*0.1 -3/4 s_x + 1/4 s_x s_y + (1/4 + 1/2*0.1) s_z - 1/4 s_y s_z

    # Check spin model values (index assignment may vary in jijmodeling v2)
    assert isclose(converter.spin_model.constant, 1 / 2 - 1 / 2 * 0.1)
    expected_linear = sorted([-3 / 4, 1 / 4 + 1 / 2 * 0.1])
    actual_linear = sorted(converter.spin_model.linear.values())
    assert all(isclose(a, b) for a, b in zip(actual_linear, expected_linear))
    expected_quad = sorted([1 / 4, -1 / 4])
    actual_quad = sorted(converter.spin_model.quad.values())
    assert all(isclose(a, b) for a, b in zip(actual_quad, expected_quad))

    # Check structural relationships:
    # The variable with linear=-0.75 (x) and the one with no linear term (y) form
    # a quadratic pair with coefficient 0.25.
    # The variable with no linear term (y) and the one with linear=0.3 (z) form
    # a quadratic pair with coefficient -0.25.
    linear = converter.spin_model.linear
    quad = converter.spin_model.quad
    idx_x = [k for k, v in linear.items() if isclose(v, -3 / 4)][0]
    idx_z = [k for k, v in linear.items() if abs(v - 0.3) < 1e-10][0]
    all_indices = set()
    for pair in quad:
        all_indices.update(pair)
    idx_y = (all_indices - {idx_x, idx_z}).pop()
    assert isclose(quad[(min(idx_x, idx_y), max(idx_x, idx_y))], 1 / 4)
    assert isclose(quad[(min(idx_y, idx_z), max(idx_y, idx_z))], -1 / 4)

    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=2)

    import scipy
    import numpy as np

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

    # Optimal solution: x=0, y=1, z=1 with energy = 0*1 - 1*1 + 0 - 0.1*1 = -1.1
    best_sample, best_energy, _ = binary_result.lowest()
    assert best_sample == {idx_x: 0, idx_y: 1, idx_z: 1}
    assert abs(best_energy - (-1.1)) < 1e-6
