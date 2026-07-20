import jijmodeling as jm
import numpy as np
import ommx.v1
import pytest

from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.fqaoa import FQAOAConverter
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
        problem += (
            J.ndenumerate()
            .map(lambda ij_v: ij_v[1] * x[ij_v[0][0]].sum() * x[ij_v[0][1]].sum())
            .sum()
        )

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


def test_odd_fermion_orbitals_are_orthonormal():
    """Odd-fermion orbitals retain the constant mode and orthonormal rows."""
    variables = [
        ommx.v1.DecisionVariable.binary(
            site * 2 + orbital,
            name="x",
            subscripts=[site, orbital],
        )
        for site in range(4)
        for orbital in range(2)
    ]
    instance = ommx.v1.Instance.from_components(
        decision_variables=variables,
        objective=0.0,
        constraints=[(sum(variables) == 3).set_id(0)],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    converter = FQAOAConverter(instance, num_fermions=3)

    orbital = converter.get_fermi_orbital()

    np.testing.assert_allclose(orbital @ orbital.T, np.eye(3), atol=1e-12)


def test_flatten_givens_data_reverses_elimination_order(simple_problem):
    """State preparation replays decomposition rotations in inverse order."""
    converter = FQAOAConverter(simple_problem, num_fermions=4)
    givens = [[(0, 1), 0.25], [(2, 3), -0.5], [(1, 2), 0.75]]

    indices, angles = converter._flatten_givens_data(givens)

    np.testing.assert_array_equal(indices, [[1, 2], [2, 3], [0, 1]])
    assert angles == [0.75, -0.5, 0.25]


def test_fqaoa_rejects_variables_without_two_subscripts():
    """Malformed variable layout fails with an actionable ValueError."""
    variable = ommx.v1.DecisionVariable.binary(0, name="x", subscripts=[0])
    instance = ommx.v1.Instance.from_components(
        decision_variables=[variable],
        objective=variable,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )

    with pytest.raises(ValueError, match="exactly two"):
        FQAOAConverter(instance, num_fermions=1)


def test_fqaoa_preserves_non_particle_constraints_in_cost():
    """Constraints beyond fixed particle count remain penalty terms."""
    variables = [
        ommx.v1.DecisionVariable.binary(
            site * 2 + orbital,
            name="x",
            subscripts=[site, orbital],
        )
        for site in range(2)
        for orbital in range(2)
    ]
    instance = ommx.v1.Instance.from_components(
        decision_variables=variables,
        objective=0.0,
        constraints=[
            (sum(variables) == 1).set_id(0),
            (variables[0] == 0).set_id(1),
        ],
        sense=ommx.v1.Instance.MINIMIZE,
    )
    converter = FQAOAConverter(
        instance,
        num_fermions=1,
        uniform_penalty_weight=5.0,
    )

    forbidden = converter.spin_model.calc_energy([-1, 1, 1, 1])
    allowed = converter.spin_model.calc_energy([1, -1, 1, 1])

    assert forbidden > allowed


def test_fqaoa_rejects_mismatched_cardinality_constraint():
    """The prepared fermion sector must match an explicit cardinality RHS."""
    variables = [
        ommx.v1.DecisionVariable.binary(
            site * 2 + orbital,
            name="x",
            subscripts=[site, orbital],
        )
        for site in range(2)
        for orbital in range(2)
    ]
    instance = ommx.v1.Instance.from_components(
        decision_variables=variables,
        objective=0.0,
        constraints=[(sum(variables) == 2).set_id(17)],
        sense=ommx.v1.Instance.MINIMIZE,
    )

    with pytest.raises(ValueError, match="cardinality constraint 17"):
        FQAOAConverter(instance, num_fermions=1)


@pytest.mark.quri_parts
def test_fqaoa_quri_parts_runtime_parameter_sampling(simple_problem):
    """FQAOA sampling binds runtime gammas and betas on QURI Parts."""
    pytest.importorskip("quri_parts")
    pytest.importorskip("quri_parts.qulacs")
    from qamomile.quri_parts import QuriPartsTranspiler

    fqaoa_converter = FQAOAConverter(simple_problem, num_fermions=4)
    transpiler = QuriPartsTranspiler()
    executable = fqaoa_converter.transpile(transpiler, p=1)

    result = executable.sample(
        transpiler.executor(),
        shots=32,
        bindings={"gammas": [0.2], "betas": [0.4]},
    ).result()

    assert sum(count for _, count in result.results) == 32
    assert all(len(value) == fqaoa_converter.num_qubits for value, _ in result.results)
    assert all(
        sum(value) == fqaoa_converter.num_fermions for value, _ in result.results
    )


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
