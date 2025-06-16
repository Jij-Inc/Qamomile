from typing import Callable, Final

import numpy as np


KET_0: Final[np.ndarray] = np.array([[1], [0]])
KET_1: Final[np.ndarray] = np.array([[0], [1]])

MATRIX_0: Final[np.ndarray] = np.array([[1, 0], [0, 0]])  # |0><0|
MATRIX_1: Final[np.ndarray] = np.array([[0, 0], [0, 1]])  # |1><1|


I_MATRIX: Final[np.ndarray] = np.identity(2)
H_MATRIX: Final[np.ndarray] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
X_MATRIX: Final[np.ndarray] = np.array([[0, 1], [1, 0]])

CX_MATRIX: Final[np.ndarray] = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
)

CCX_MATRIX: Final[np.ndarray] = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]
)


RX_MATRIX: Final[Callable] = lambda theta: np.array(
    [
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)],
    ]
)


def take_tensor_product(*args: np.ndarray) -> np.ndarray:
    """Computes the tensor product of multiple matrices or vectors.

    Returns:
        np.ndarray: a matrix or vector that is the tensor product of the input matrices or vectors.
    """
    result_state = args[0]
    for mat in args[1:]:
        result_state = np.kron(result_state, mat)

    return result_state
