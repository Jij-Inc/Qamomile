import qamomile.circuit as qmc
from qamomile.optimization.qaoa import QAOAConverter
from qamomile.qiskit.transpiler import QiskitTranspiler
from qamomile.optimization.binary_model import binary, BinaryExpr, BinaryModel
import jijmodeling as jm


def test_simple_qaoa_decode():
    x = binary(0)
    y = binary(1)
    z = binary(2)

    problem = BinaryExpr()
    problem += x * y
    problem += -1 * y * z
    problem += x
    problem += -0.1*z

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

    assert converter.spin_model.constant == 1/2 - 1/2*0.1
    assert converter.spin_model.linear == {0: -3/4, 2: 1/4 + 1/2*0.1}
    assert converter.spin_model.quad == {(0, 1): 1/4, (1, 2): -1/4}


    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=2)  # p=2に変更してより表現力を高める

    import scipy

    def _obj(params):
        gammas_val = [params[0], params[1]]
        betas_val = [params[2], params[3]]
        bindings = {"gammas": gammas_val, "betas": betas_val}
        job = executable.sample(transpiler.executor(), shots=100, bindings=bindings)  # さらにshotsを増やす
        result = job.result()

        sampleset = converter.decode(result)
        argmin_index = np.argmin(sampleset.energy)
        best_energy = sampleset.energy[argmin_index]
        best_occurrence = sampleset.num_occurrences[argmin_index]
        return best_energy * best_occurrence

    import numpy as np
    x0 = [2.6, 0.11, 0.52, 0.45]
    bounds = [(0, np.pi), (0, np.pi), (0, np.pi/2), (0, np.pi/2)]
    res = scipy.optimize.minimize(_obj, x0=x0, bounds=bounds, method='COBYLA', options={'maxiter': 100})

    gammas_opt = [res.x[0], res.x[1]]
    betas_opt = [res.x[2], res.x[3]]
    bindings_opt = {"gammas": gammas_opt, "betas": betas_opt}
    job_opt = executable.sample(transpiler.executor(), shots=1000, bindings=bindings_opt)
    result_opt = job_opt.result()
    binary_result = converter.decode(result_opt)
    print(binary_result)

    # Optimal solution: x=0, y=1, z=1 with energy = 0*1 - 1*1 + 0 - 0.1*1 = -1.1
    best_sample, best_energy, _ = binary_result.lowest()
    assert best_sample == {0: 0, 1: 1, 2: 1}
    assert abs(best_energy - (-1.1)) < 1e-6



def test_qaoa_decode():
    x = jm.BinaryVar("x")
    y = jm.BinaryVar("y")
    z = jm.BinaryVar("z")

    problem = jm.Problem("sample")
    problem += x * y
    problem += -1 * y * z
    problem += x
    problem += -0.1*z

    interpreter = jm.Interpreter({})
    instance = interpreter.eval_problem(problem)

    converter = QAOAConverter(instance)

    # x y - y z + x
    # binary to spin: x = (1 - s_x) / 2
    # x = (1 - s_x) / 2
    # x y = (1 - s_x) / 2 * (1 - s_y) / 2 = 1/4(1 - s_x - s_y + s_x s_y)
    # - y z = - (1 - s_y) / 2 * (1 - s_z) / 2 = -1/4(1 - s_y - s_z + s_y s_z)
    # --------------------------------------------
    # total 
    # = 1/2-1/2*0.1 -3/4 s_x + 1/4 s_x s_y + (1/4 + 1/2*0.1) s_z - 1/4 s_y s_z

    assert converter.spin_model.constant == 1/2 - 1/2*0.1
    assert converter.spin_model.linear == {0: -3/4, 2: 1/4 + 1/2*0.1}
    assert converter.spin_model.quad == {(0, 1): 1/4, (1, 2): -1/4}


    transpiler = QiskitTranspiler()
    executable = converter.transpile(transpiler, p=2)  # p=2に変更してより表現力を高める

    import scipy

    def _obj(params):
        gammas_val = [params[0], params[1]]
        betas_val = [params[2], params[3]]
        bindings = {"gammas": gammas_val, "betas": betas_val}
        job = executable.sample(transpiler.executor(), shots=100, bindings=bindings)  # さらにshotsを増やす
        result = job.result()

        sampleset = converter.decode(result)
        argmin_index = np.argmin(sampleset.energy)
        best_energy = sampleset.energy[argmin_index]
        best_occurrence = sampleset.num_occurrences[argmin_index]
        return best_energy * best_occurrence

    import numpy as np
    x0 = [2.6, 0.11, 0.52, 0.45]
    bounds = [(0, np.pi), (0, np.pi), (0, np.pi/2), (0, np.pi/2)]
    res = scipy.optimize.minimize(_obj, x0=x0, bounds=bounds, method='COBYLA', options={'maxiter': 100})

    gammas_opt = [res.x[0], res.x[1]]
    betas_opt = [res.x[2], res.x[3]]
    bindings_opt = {"gammas": gammas_opt, "betas": betas_opt}
    job_opt = executable.sample(transpiler.executor(), shots=1000, bindings=bindings_opt)
    result_opt = job_opt.result()

    binary_result = converter.decode(result_opt)

    # Optimal solution: x=0, y=1, z=1 with energy = 0*1 - 1*1 + 0 - 0.1*1 = -1.1
    best_sample, best_energy, _ = binary_result.lowest()
    assert best_sample == {0: 0, 1: 1, 2: 1}
    assert abs(best_energy - (-1.1)) < 1e-6
