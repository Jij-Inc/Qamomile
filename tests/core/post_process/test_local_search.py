import pytest
import jijmodeling as jm
import jijmodeling_transpiler.core as jmt
from qamomile.core.post_process.local_search import IsingMatrix, LocalSearch
from qamomile.core.converters.qaoa import QAOAConverter
from qamomile.core.converters.converter import QuantumConverter
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def qaoa_converter() -> QAOAConverter:
    x = jm.BinaryVar("x", shape=(3,))
    problem = jm.Problem("qubo")
    problem += -x[0] * x[1] + x[1] * x[2] + x[2] * x[0]

    compiled_instance = jmt.compile_model(problem, {})
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
    size = qaoa_encoded.num_bits()

    assert isinstance(ising_matrix.quad, np.ndarray)
    assert isinstance(ising_matrix.linear, np.ndarray)
    assert ising_matrix.quad.shape == (size, size)
    assert ising_matrix.linear.shape == (size,)


def test_calc_E_diff(ising_matrix, qaoa_converter):
    qaoa_encoded = qaoa_converter.ising_encode()
    ising_matrix.to_ising_matrix(qaoa_encoded)
    size = qaoa_encoded.num_bits()
    state = np.random.choice([-1, 1], size=size)
    random_index = np.random.randint(0, size)
    delta_E = ising_matrix.calc_E_diff(state, random_index)

    assert isinstance(delta_E, float)


def test_run(local_search_instance, qaoa_converter):
    size = qaoa_converter.ising_encode().num_bits()
    state = np.random.choice([-1, 1], size=size)

    invalid_method = "non_existent_method"
    with pytest.raises(ValueError):
        local_search_instance.run(state, local_search_method=invalid_method)


def test_run_local_search(local_search_instance, qaoa_converter):
    size = qaoa_converter.ising_encode().num_bits()
    state = np.random.choice([-1, 1], size=size)

    mock_method = MagicMock()

    def side_effect(_, current_state, __):
        if mock_method.call_count < 5:
            new_state = current_state.copy()
            new_state[0] *= -1  
        else:
            new_state = (
                current_state.copy()
            )  
        return new_state

    mock_method.side_effect = side_effect
    final_state = local_search_instance._run_local_search(
        mock_method, state, max_iter=10
    )

    assert mock_method.call_count == 5
    assert len(state) == len(final_state)
    assert all(val in [-1, 1] for val in final_state)

    mock_method.reset_mock()
    final_state_2 = local_search_instance._run_local_search(
        mock_method, state, max_iter=3
    )
    assert mock_method.call_count == 3
    assert len(state) == len(final_state_2)
    assert all(val in [-1, 1] for val in final_state_2)


def test_best_improvement(local_search_instance, ising_matrix, qaoa_converter):
    qaoa_encoded = qaoa_converter.ising_encode()
    ising_matrix.to_ising_matrix(qaoa_encoded)
    size = qaoa_encoded.num_bits()
    state = np.random.choice([-1, 1], size=size)

    new_state = local_search_instance.best_improvement(ising_matrix, state, size)

    assert len(state) == len(new_state)
    assert all(val in [-1, 1] for val in new_state)


def test_first_improvement(local_search_instance, ising_matrix, qaoa_converter):
    qaoa_encoded = qaoa_converter.ising_encode()
    ising_matrix.to_ising_matrix(qaoa_encoded)
    size = qaoa_encoded.num_bits()
    state = np.random.choice([-1, 1], size=size)

    new_state = local_search_instance.first_improvement(ising_matrix, state, size)

    assert len(state) == len(new_state)
    assert all(val in [-1, 1] for val in new_state)


def test_decode(local_search_instance, qaoa_converter):
    size = qaoa_converter.ising_encode().num_bits()
    state = np.random.choice([-1, 1], size=size)
    sampleset = local_search_instance.decode(state)

    assert isinstance(sampleset, jm.experimental.SampleSet)
