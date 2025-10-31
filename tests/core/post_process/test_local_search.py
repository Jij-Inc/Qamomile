import pytest
import jijmodeling as jm
import ommx.v1
from qamomile.core.post_process.local_search import IsingMatrix, LocalSearch
from qamomile.core.converters.qaoa import QAOAConverter
import numpy as np


@pytest.fixture
def qaoa_converter() -> QAOAConverter:
    x = jm.BinaryVar("x", shape=(3,))
    problem = jm.Problem("qubo")
    problem += (
        8 * x[0] * x[1] + 8 * x[1] * x[2] - 12 * x[0] - 8 * x[1] - 4 * x[2] + 8
    )  # in order to generate IsingModel({(0, 1): 2.0 ,(1, 2): 2.0}, {0: 4.0}, 0)
    compiled_instance = jm.Interpreter({}).eval_problem(problem)
    qaoa_converter = QAOAConverter(compiled_instance)

    return qaoa_converter


@pytest.fixture
def ising_matrix():
    return IsingMatrix()


@pytest.fixture
def local_search_instance(qaoa_converter):
    return LocalSearch(qaoa_converter)


def test_to_ising_matrix(ising_matrix, qaoa_converter):
    qaoa_encoded = qaoa_converter.ising_encode()
    ising_matrix.to_ising_matrix(qaoa_encoded)
    size = qaoa_encoded.num_bits

    assert isinstance(ising_matrix.quad, np.ndarray)
    assert isinstance(ising_matrix.linear, np.ndarray)
    assert ising_matrix.quad.shape == (size, size)
    assert ising_matrix.linear.shape == (size,)


def test_calc_E_diff(ising_matrix, qaoa_converter):
    ising_ex = ising_ex = qaoa_converter.ising_encode()
    ising_matrix.to_ising_matrix(ising_ex)
    state = np.array([1, 1, -1])
    delta_E = ising_matrix.calc_E_diff(state, 0)

    assert delta_E == -12.0


def test_run(local_search_instance, qaoa_converter):
    size = qaoa_converter.ising_encode().num_bits
    state = np.random.choice([-1, 1], size=size)

    invalid_method = "non_existent_method"
    with pytest.raises(ValueError):
        local_search_instance.run(state, local_search_method=invalid_method)


def test_run_local_search(local_search_instance, qaoa_converter):
    local_search_instance.ising = qaoa_converter.ising_encode()
    state = np.array([1, -1, -1])
    new_state = local_search_instance._run_local_search(
        method=local_search_instance.best_improvement, initial_state=state, max_iter=-1
    )

    assert np.array_equal(new_state, np.array([-1, 1, -1]))

    new_state_2 = local_search_instance._run_local_search(
        method=local_search_instance.best_improvement, initial_state=state, max_iter=1
    )

    assert np.array_equal(new_state_2, np.array([-1, -1, -1]))


def test_best_improvement(local_search_instance, ising_matrix, qaoa_converter):
    ising_ex = qaoa_converter.ising_encode()
    ising_matrix.to_ising_matrix(ising_ex)
    size = 3
    state = np.array([1, 1, -1])

    new_state = local_search_instance.best_improvement(ising_matrix, state, size)

    assert np.array_equal(new_state, np.array([-1, 1, -1]))
    assert ising_ex.calc_energy(new_state) == -8


def test_first_improvement(local_search_instance, ising_matrix, qaoa_converter):
    ising_ex = qaoa_converter.ising_encode()
    ising_matrix.to_ising_matrix(ising_ex)
    size = 3
    state = np.array([1, -1, -1])

    new_state = local_search_instance.first_improvement(ising_matrix, state, size)

    assert np.array_equal(new_state, np.array([-1, 1, -1]))
    assert ising_ex.calc_energy(new_state) == -8


def test_decode(local_search_instance):
    state = np.array([-1, 1, -1])
    result = local_search_instance.decode(state)
    state_dict = result.extract_decision_variables("x", sample_id=0)
    expected_state_dict = {(2,): 1.0, (0,): 1.0, (1,): 0.0}
    obj = result.objectives[0]

    assert isinstance(result, ommx.v1.SampleSet)
    assert state_dict == expected_state_dict
    assert obj == -8
