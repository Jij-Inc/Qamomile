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
import numpy as np

class IsingMatrix:
    def __init__(self, quad: np.array, linear: np.array):
        self.quad = quad
        self.linear = linear


class IsingConverter(QuantumConverter):
    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """
        Construct the cost Hamiltonian.

        Returns:
            qm_o.Hamiltonian: The cost Hamiltonian.
        """
        hamiltonian = qm_o.Hamiltonian()
        ising = self.get_ising()

        # Add linear terms
        for i, hi in ising.linear.items():
            if not is_close_zero(hi):
                hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, i),), hi)

        # Add quadratic terms
        for (i, j), Jij in ising.quad.items():
            if not is_close_zero(Jij):
                hamiltonian.add_term(
                    (
                        qm_o.PauliOperator(qm_o.Pauli.Z, i),
                        qm_o.PauliOperator(qm_o.Pauli.Z, j),
                    ),
                    Jij,
                )

        hamiltonian.constant = ising.constant
        return hamiltonian

	
def to_ising_matrix(ising: IsingModel):
    size = max(max(i, j) for i, j in ising.quad.keys()) + 1
    
    quad_matrix = np.zeros((size, size))
    for (i, j), value in ising.quad.items():
        quad_matrix[i, j] = value
        quad_matrix[j, i] = value
    
    linear_vector = np.zeros(size)
    for i, value in ising.linear.items():
        linear_vector[i] = value
    
    return IsingMatrix(quad=quad_matrix, linear=linear_vector)

def first_improvement(ising: IsingModel, initial_state: np.array):
    ising_matrix = to_ising_matrix(ising)
    N = len(initial_state)
    current_state = initial_state.copy()

    for i in range(N): 
        delta_E_i = calc_E_diff(ising_matrix, current_state, i)
        
        if delta_E_i < 0:
            current_state[i] = -current_state[i]
            break  

    return current_state

import numpy as np

def best_improvement(ising: IsingModel, initial_state: np.array):
    ising_matrix = to_ising_matrix(ising)
    N = len(initial_state)
    current_state = initial_state.copy() 

    best_delta_E = 0
    best_index = -1

    for i in range(N):
        delta_E_i = calc_E_diff(ising_matrix, current_state, i)
        if delta_E_i < best_delta_E:  
            best_delta_E = delta_E_i
            best_index = i
    

    if best_index != -1:
        current_state[best_index] = -current_state[best_index]

    return current_state


def calc_E_diff(ising: IsingMatrix, state: np.array, l: int):
    delta_E = -2 * state[l] * (ising.quad[:, l] @ state - ising.linear[l])
    return delta_E

def run(local_search_method, ising: IsingModel, initial_state: np.array):
    current_state = initial_state.copy()
    
    while True:
        previous_state = current_state 
        current_state = local_search_method(ising, current_state)
        
        if np.array_equal(previous_state, current_state):
            break  

    return current_state
