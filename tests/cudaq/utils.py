from typing import Callable, Final

import numpy as np
import scipy

from tests.utils import Utils


class CudaqUtils:

    # Define the matrices for the gates.
    # Note that, they assume the qubits applied to are next to each other such as control is 1 and target is 2.
    CX_MATRIX: Final[np.ndarray] = Utils.take_tensor_product(
        [Utils.I_MATRIX, Utils.MATRIX_0]
    ) + Utils.take_tensor_product([Utils.X_MATRIX, Utils.MATRIX_1])
    CZ_MATRIX: Final[np.ndarray] = Utils.take_tensor_product(
        [Utils.I_MATRIX, Utils.MATRIX_0]
    ) + Utils.take_tensor_product([Utils.Z_MATRIX, Utils.MATRIX_1])
    CRX_MATRIX: Final[Callable] = lambda theta: Utils.take_tensor_product(
        [Utils.I_MATRIX, Utils.MATRIX_0]
    ) + Utils.take_tensor_product([Utils.RX_MATRIX(theta), Utils.MATRIX_1])
    CRY_MATRIX: Final[Callable] = lambda theta: Utils.take_tensor_product(
        [Utils.I_MATRIX, Utils.MATRIX_0]
    ) + Utils.take_tensor_product([Utils.RY_MATRIX(theta), Utils.MATRIX_1])
    CRZ_MATRIX: Final[Callable] = lambda theta: Utils.take_tensor_product(
        [Utils.I_MATRIX, Utils.MATRIX_0]
    ) + Utils.take_tensor_product([Utils.RZ_MATRIX(theta), Utils.MATRIX_1])
    CCX_MATRIX: Final[np.ndarray] = Utils.take_tensor_product(
        [Utils.I_MATRIX, Utils.I_MATRIX, Utils.MATRIX_0]
    ) + Utils.take_tensor_product([CX_MATRIX, Utils.MATRIX_1])
    EXP_PAULI_MATRIX: Final[Callable] = lambda theta, hamiltonian: scipy.linalg.expm(
        -1j * theta * hamiltonian
    )
