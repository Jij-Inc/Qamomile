import qiskit as qk

import jijmodeling  as jm
import jijmodeling.transpiler as jmt
from jijtranspiler_qiskit.qaoa.to_qaoa import transpile_to_qaoa_ansatz


def test_qaoa_onehot():
    n = jm.Placeholder("n")
    x = jm.Binary("x", shape=n)
    i = jm.Element("i", n)
    problem = jm.Problem("sample")
    problem += jm.Sum(i, x[i])
    problem += jm.Constraint("onehot", jm.Sum(i, x[i]) == 1)

    compiled_instance = jmt.core.compile_model(problem, {"n": 3})
    qaoa_builder = transpile_to_qaoa_ansatz(compiled_instance)

    qaoa_ansatz, cost_func, constant = qaoa_builder.get_qaoa_ansatz(p=1)

