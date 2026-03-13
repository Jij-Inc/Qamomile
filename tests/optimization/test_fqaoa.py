import pytest
import jijmodeling as jm

from qamomile.optimization.fqaoa import FQAOAConverter
from qamomile.optimization.binary_model import BinaryModel
from qamomile.qiskit import QiskitTranspiler

from tests.utils import Utils


@pytest.fixture
def simple_problem():
    problem = jm.Problem("qubo")

    @problem.update
    def _(problem: jm.DecoratedProblem):
        J = problem.Float(ndim=2)
        n = J.len_at(0, latex="n")
        D = problem.Dim()
        x = problem.BinaryVar(shape=(n, D))

        # Quadratic objective
        problem += J.ndenumerate().map(
            lambda ij_v: ij_v[1] * x[ij_v[0][0]].sum() * x[ij_v[0][1]].sum()
        ).sum()

        # Equality constraint
        problem += problem.Constraint("constraint", x.sum() == 4)

    instance_data = {
        "J": [
            [0.0, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.0],
            [0.0, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "D": 2,
    }
    instance = problem.eval(instance_data)

    return instance


def test_initialization(simple_problem):
    fqaoa_converter = FQAOAConverter(simple_problem, num_fermions=4)

    assert fqaoa_converter.num_integers == 4
    assert fqaoa_converter.num_bits == 2
    assert fqaoa_converter.num_fermions == 4
    assert isinstance(fqaoa_converter.var_map, dict)
    assert isinstance(fqaoa_converter.spin_model, BinaryModel)
    assert fqaoa_converter.num_qubits == 8


def test_cyclic_mapping(simple_problem):
    fqaoa_converter = FQAOAConverter(simple_problem, num_fermions=4)

    assert fqaoa_converter.var_map == {
        (0, 0): 0,
        (1, 0): 1,
        (2, 0): 2,
        (3, 0): 3,
        (0, 1): 4,
        (1, 1): 5,
        (2, 1): 6,
        (3, 1): 7,
    }


def test_get_fqaoa_ansatz_transpile(simple_problem):
    fqaoa_converter = FQAOAConverter(simple_problem, num_fermions=4)
    transpiler = QiskitTranspiler()
    executable = fqaoa_converter.transpile(transpiler, p=2)
    circuit = executable.get_first_circuit()

    assert circuit.num_qubits == fqaoa_converter.num_qubits
    assert len(circuit.parameters) == 4


@pytest.mark.parametrize(
    "instance_data",
    [
        {"N": 3, "a": [-1.0, 1.0, -1.0]},
        {"N": 4, "a": [0.5, -0.5, 0.5, -0.5]},
    ],
)
def test_n_body_problem_with_constraints(instance_data):
    """Create FQAOAConverter with a HUBO problem.

    Check if
    - ValueError is raised.
    """
    n_body_problem = Utils.get_n_body_problem()
    instance = n_body_problem.eval(instance_data)

    with pytest.raises(ValueError):
        FQAOAConverter(instance, num_fermions=0)
