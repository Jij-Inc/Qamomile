import jijmodeling as jm
from typing import Callable, Final

import numpy as np


class Utils:
    """A utility class for tests"""

    KET_0: Final[np.ndarray] = np.array([[1], [0]])
    KET_1: Final[np.ndarray] = np.array([[0], [1]])

    MATRIX_0: Final[np.ndarray] = np.array([[1, 0], [0, 0]])  # |0><0|
    MATRIX_1: Final[np.ndarray] = np.array([[0, 0], [0, 1]])  # |1><1|

    I_MATRIX: Final[np.ndarray] = np.identity(2)
    H_MATRIX: Final[np.ndarray] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    X_MATRIX: Final[np.ndarray] = np.array([[0, 1], [1, 0]])
    Y_MATRIX: Final[np.ndarray] = np.array([[0, -1j], [1j, 0]])
    Z_MATRIX: Final[np.ndarray] = np.array([[1, 0], [0, -1]])
    S_MATRIX: Final[np.ndarray] = np.array([[1, 0], [0, 1j]])
    T_MATRIX: Final[np.ndarray] = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

    RX_MATRIX: Final[Callable] = lambda theta: np.array(
        [
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )
    RY_MATRIX: Final[Callable] = lambda theta: np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )
    RZ_MATRIX: Final[Callable] = lambda theta: np.array(
        [
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)],
        ]
    )

    RXX_MATRIX: Final[Callable] = lambda theta: np.array(
        [
            [np.cos(theta / 2), 0, 0, -1j * np.sin(theta / 2)],
            [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
            [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
            [-1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
        ]
    )
    RYY_MATRIX: Final[Callable] = lambda theta: np.array(
        [
            [np.cos(theta / 2), 0, 0, 1j * np.sin(theta / 2)],
            [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
            [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
            [1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
        ]
    )
    RZZ_MATRIX: Final[Callable] = lambda theta: np.array(
        [
            [np.exp(-1j * theta / 2), 0, 0, 0],
            [0, np.exp(1j * theta / 2), 0, 0],
            [0, 0, np.exp(1j * theta / 2), 0],
            [0, 0, 0, np.exp(-1j * theta / 2)],
        ]
    )

    @staticmethod
    def take_tensor_product(matrices: list[np.ndarray]) -> np.ndarray:
        """Computes the tensor product of multiple matrices or vectors.

        Returns:
            np.ndarray: a matrix or vector that is the tensor product of the input matrices or vectors.
        """
        result = matrices[0]
        for mat in matrices[1:]:
            result = np.kron(result, mat)

        return result

    @staticmethod
    def get_n_body_problem():
        N = jm.Placeholder("N")
        x = jm.BinaryVar("x", shape=(N,))
        a = jm.Placeholder("a", shape=(N,))
        i = jm.Element("i", belong_to=(0, N))

        problem = jm.Problem("N-body problem")

        problem += jm.prod(i, x[i])
        problem += jm.sum(i, a[i] * x[i])
        return problem

    @staticmethod
    def get_n_body_problem_with_constraints():
        N = jm.Placeholder("N")
        x = jm.BinaryVar("x", shape=(N,))
        a = jm.Placeholder("a", shape=(N,))
        i = jm.Element("i", belong_to=(0, N))

        problem = jm.Problem("N-body problem")

        problem += jm.prod(i, x[i])
        problem += jm.sum(i, a[i] * x[i])

        problem += jm.Constraint("constraint", jm.sum(i, x[i]) > 0)
        return problem
