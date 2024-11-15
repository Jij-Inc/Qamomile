"""
qamomile/core/post_process.py

This module provides functionality for Local Search Algorithm. Local Search Algorithm is one of the simplest 
heuristic algorithms for the optimization problem. Let's take the Ising Hamiltonian as objective
function to be minimized. 

    .. math::
        \sum_{ij} J_{ij} z_i z_j + \sum_i h_i z_i, ~\text{s.t.}~z_i \in \{-1, 1\}
            
Key Features:
- Implementation of first- and best-improvement local search algorithms
- Construction of cost Hamiltonians for the Ising model
- Conversion of Ising interactions from dictionary format to matrix form
"""

from qamomile.core.ising_qubo import IsingModel
from qamomile.core.converters.converter import QuantumConverter
import qamomile.core.operator as qm_o
from qamomile.core.converters.utils import is_close_zero
from qamomile.core.bitssample import BitsSample, BitsSampleSet
import numpy as np
import jijmodeling as jm
from typing import Callable

class IsingMatrix:
    def __init__(self, quad: np.array, linear: np.array):
        self.quad = quad
        self.linear = linear

class LocalSearch:
    def __init__(self, converter: QuantumConverter):
        self.converter = converter
        self.ising = converter.ising_encode()
        
    def decode(self, result) -> jm.experimental.SampleSet:
        sample = BitsSample(1,np.where(result == -1, 0, 1).tolist())
        sample_set = BitsSampleSet(bitarrays=[sample])
        decoded_sampleset=self.converter.decode_bits_to_sampleset(sample_set)

        return decoded_sampleset

    def to_ising_matrix(self, ising: IsingModel) -> IsingMatrix:
        size = max(max(i, j) for i, j in ising.quad.keys()) + 1
        quad_matrix = np.zeros((size, size))
        for (i, j), value in ising.quad.items():
            quad_matrix[i, j] = value
            quad_matrix[j, i] = value

        linear_vector = np.zeros(size)
        for i, value in ising.linear.items():
            linear_vector[i] = value

        return IsingMatrix(quad=quad_matrix, linear=linear_vector)

    def calc_E_diff(self, ising: IsingMatrix, state: np.ndarray, l: int) -> float:
        delta_E = -2 * state[l] * (ising.quad[:, l] @ state - ising.linear[l])
        return delta_E

    def first_improvement(self, ising_matrix: IsingMatrix, current_state: np.ndarray, N: int) -> np.ndarray:

        for i in range(N):
            delta_E_i = self.calc_E_diff(ising_matrix, current_state, i)
            if delta_E_i < 0:
                current_state[i] = -current_state[i]
            
        return current_state

    def best_improvement(self, ising_matrix: IsingMatrix, current_state: np.ndarray, N: int) -> np.ndarray:
        delta_E = np.array([self.calc_E_diff(ising_matrix, current_state, i) for i in range(N)])
        best_index = np.argmin(delta_E)
        best_delta_E = delta_E[best_index]
        
        if best_delta_E < 0:
            current_state[best_index] = -current_state[best_index]
        
        return current_state
    
    def run(self, initial_state: np.array, max_iter: int, local_search_method: str = "best_improvement") -> jm.experimental.SampleSet:
        method_map = {
        "best_improvement": self.best_improvement,
        "first_improvement": self.first_improvement
        }
        if local_search_method not in method_map:
            raise ValueError(f"Invalid local_search_method: {local_search_method}. Choose from {list(method_map.keys())}.")

        method = method_map[local_search_method]
        result = self._run_local_search(method, initial_state, max_iter)
        decoded_sampleset = self.decode(result)
        return decoded_sampleset

    def _run_local_search(self, method: Callable, initial_state: np.array, max_iter: int) -> np.ndarray:
        current_state = initial_state.copy()
        ising_matrix = self.to_ising_matrix(self.ising)
        N = len(current_state)
        counter = 0

        while counter < max_iter:
            previous_state = current_state.copy()
            current_state = method(ising_matrix, current_state, N)

            if np.array_equal(previous_state, current_state):
                break
            
            counter += 1

        return current_state

