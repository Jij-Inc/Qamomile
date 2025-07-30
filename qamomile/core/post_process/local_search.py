"""
qamomile/core/post_process.py

This module provides functionality for Local Search Algorithm. Local Search Algorithm is one of the simplest
heuristic algorithms for the optimization problem. You can apply the local search to the following Ising Hamiltonian.

    .. math::
        \sum_{ij} J_{ij} z_i z_j + \sum_i h_i z_i, \\\\
        \\text{s.t.}~z_i \in \{-1, 1\}

Key Features:

- Implementation of first- and best-improvement local search algorithms

- Construction of cost Hamiltonians for the Ising model

- Conversion of Ising interactions from dictionary format to matrix form

"""

from typing import Callable, Optional

import jijmodeling as jm
from ommx.v1 import SampleSet
import numpy as np
from qamomile.core.bitssample import BitsSample, BitsSampleSet
from qamomile.core.converters.converter import QuantumConverter
from qamomile.core.ising_qubo import IsingModel


class IsingMatrix:
    """
    A class to represent the Ising model in matrix form.

    Attributes:
        quad (np.ndarray): The interaction terms between spin variables in the Ising model.
        linear (np.ndarray): The external magnetic fields or self-interaction applied to each spin in the Ising model

    Methods:
        to_ising_matrix: Converts an Ising model into its corresponding matrix form.
        calc_E_diff: Calculates the change in energy if a specific bit is flipped.
    """

    def __init__(
        self, quad: Optional[np.ndarray] = None, linear: Optional[np.ndarray] = None
    ):
        """
        Initializes an IsingMatrix instance.

        Args:
            quad (Optional[np.ndarray]): Quadratic term matrix representing the interaction term.
            linear (Optional[np.ndarray]): Linear term vector representing the self-interaction term.
        """
        if quad is None:
            self.quad = np.array([])
        else:
            self.quad = quad

        if linear is None:
            self.linear = np.array([])
        else:
            self.linear = linear

    def to_ising_matrix(self, ising: "IsingModel") -> "IsingMatrix":
        """
        Converts an IsingModel into matrix form with quadratic and linear components.

        Args:
            ising (IsingModel): The Ising model to be converted.

        Returns:
            IsingMatrix: The converted Ising model in matrix form.
        """
        size = ising.num_bits()

        quad_matrix = np.zeros((size, size))
        for (i, j), value in ising.quad.items():
            quad_matrix[i, j] = value
            quad_matrix[j, i] = value

        linear_vector = np.zeros(size)
        for i, value in ising.linear.items():
            linear_vector[i] = value

        self.quad = quad_matrix
        self.linear = linear_vector
        return self

    def calc_E_diff(self, state: np.ndarray, l: int) -> float:
        """
        Calculates the energy difference when flipping a bit in the Ising state.

        Args:
            state (np.ndarray): The current state of the Ising model.
            l (int): Index of the bit to be flipped.

        Returns:
            float: The difference in energy due to flipping the specified bit.
        """
        delta_E = -2 * state[l] * (self.quad[:, l] @ state + self.linear[l])
        return delta_E


class LocalSearch:
    """
    A class to perform local search algorithms on Ising models.

    Attributes:
        converter (QuantumConverter): The converter should be an implementation of the `QuantumConverter` abstract class, allowing the optimization problem
            to be encoded as an Ising model.
        ising (IsingModel): The encoded Ising model to be optimized.

    Methods:
        decode: Decodes the final solution from Ising spin state to jijmodeling SampleSet.
        first_improvement: Performs first-improvement local search.
        best_improvement: Performs best-improvement local search.
        run: Runs the local search algorithm with a specified method.
    """

    def __init__(self, converter: QuantumConverter):
        """
        Initializes a LocalSearch instance with a quantum converter.

        Args:
            converter (QuantumConverter): A converter used to encode optimization problems to Ising form.
        """
        self.converter = converter
        self.ising = converter.ising_encode()

    def decode(self, result) -> SampleSet:
        """
        Decodes the result obtained from a local search into a jijmodeling SampleSet.

        Args:
            result (np.ndarray): The final state of the Ising model after local search.

        Returns:
            SampleSet: The decoded results.
        """
        sample = BitsSample(1, np.where(result == -1, 1, 0).tolist())
        sample_set = BitsSampleSet(bitarrays=[sample])
        decoded_sampleset = self.converter.decode_bits_to_sampleset(sample_set)

        return decoded_sampleset

    def first_improvement(
        self, ising_matrix: IsingMatrix, current_state: np.ndarray, N: int
    ) -> np.ndarray:
        """
        Performs first-improvement local search on the Ising model.

        The first-improvement local search method iteratively examines each bit in the current state
        of the Ising model to determine if flipping that bit will reduce the system's energy.
        If a bit flip results in an energy decrease, the flip is accepted immediately, and the algorithm moves to the next bit.
        This process continues until all bits have been evaluated once.

        Args:
            ising_matrix (IsingMatrix): The Ising matrix representation of the problem.
            current_state (np.ndarray): The current state of the Ising model.
            N (int): Number of bits in the state.

        Returns:
            np.ndarray: The updated state after attempting to find an improving move.
        """
        for i in range(N):
            delta_E_i = ising_matrix.calc_E_diff(current_state, i)
            if delta_E_i < 0:
                current_state[i] = -current_state[i]

        return current_state

    def best_improvement(
        self, ising_matrix: IsingMatrix, current_state: np.ndarray, N: int
    ) -> np.ndarray:
        """
        Performs best-improvement local search on the Ising model.

        The best-improvement local search method examines all possible bit flips in the current state of
        the Ising model to determine which single bit flip would result in the greatest decrease in energy.
        It then flips the bit that leads to the most significant energy reduction, provided there is at least one such improvement.
        If no flip results in a reduction in energy, the state remains unchanged.

        Args:
            ising_matrix (IsingMatrix): The Ising matrix representation of the problem.
            current_state (np.ndarray): The current state of the Ising model.
            N (int): Number of bits in the state.

        Returns:
            np.ndarray: The updated state after attempting to find the best improving move.
        """
        delta_E = np.array(
            [ising_matrix.calc_E_diff(current_state, i) for i in range(N)]
        )
        best_index = np.argmin(delta_E)
        best_delta_E = delta_E[best_index]

        if best_delta_E < 0:
            current_state[best_index] = -current_state[best_index]

        return current_state

    def run(
        self,
        initial_state: np.ndarray,
        max_iter: int = -1,
        local_search_method: str = "best_improvement",
    ) -> SampleSet:
        """
        Runs the local search algorithm until convergence or until a maximum number of iterations.

        Args:
            initial_state (np.ndarray): The initial state to start the local search from.
            max_iter (int): Maximum number of iterations to run the local search. Defaults to -1 (no limit).
            local_search_method (str): Method of local search ("best_improvement" or "first_improvement").
                Defaults to "best_improvement".

        Returns:
            SampleSet: The decoded solution after the local search.
        """
        method_map = {
            "best_improvement": self.best_improvement,
            "first_improvement": self.first_improvement,
        }
        if local_search_method not in method_map:
            raise ValueError(
                f"Invalid local_search_method: {local_search_method}. Choose from {list(method_map.keys())}."
            )

        method = method_map[local_search_method]
        result = self._run_local_search(method, initial_state, max_iter)
        decoded_sampleset = self.decode(result)
        return decoded_sampleset

    def _run_local_search(
        self, method: Callable, initial_state: np.ndarray, max_iter: int
    ) -> np.ndarray:
        """
        Internal method to perform the local search on the Ising model.

        Args:
            method (Callable): The local search method (either best or first improvement).
            initial_state (np.ndarray): The initial state to start the search from.
            max_iter (int): The maximum number of iterations for the local search.

        Returns:
            np.ndarray: The final state after convergence or reaching the iteration limit.
        """
        current_state = initial_state.copy()
        ising_matrix = IsingMatrix()
        ising_matrix.to_ising_matrix(self.ising)
        N = len(current_state)
        counter = 0

        while max_iter == -1 or counter < max_iter:
            previous_state = current_state.copy()
            current_state = method(ising_matrix, current_state, N)

            if np.array_equal(previous_state, current_state):
                break

            counter += 1

        return current_state
