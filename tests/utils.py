from typing import Final

import numpy as np

import qamomile.core.operator as qm_o


class Utils:
    I_MATRIX: Final[np.ndarray] = np.identity(2)
    X_MATRIX: Final[np.ndarray] = np.array([[0, 1], [1, 0]])
    Y_MATRIX: Final[np.ndarray] = np.array([[0, -1j], [1j, 0]])
    Z_MATRIX: Final[np.ndarray] = np.array([[1, 0], [0, -1]])

    # See e.g. https://en.wikipedia.org/wiki/Pauli_matrices or https://mathworld.wolfram.com/PauliMatrices.html
    PAULI_PRODUCT_TABLE: Final[
        dict[tuple[qm_o.Pauli, qm_o.Pauli], (qm_o.Pauli, complex)]
    ] = {
        (qm_o.Pauli.X, qm_o.Pauli.X): (qm_o.Pauli.I, 1.0),
        (qm_o.Pauli.X, qm_o.Pauli.Y): (qm_o.Pauli.Z, 1.0j),
        (qm_o.Pauli.X, qm_o.Pauli.Z): (qm_o.Pauli.Y, -1.0j),
        (qm_o.Pauli.X, qm_o.Pauli.I): (qm_o.Pauli.X, 1.0),
        (qm_o.Pauli.Y, qm_o.Pauli.X): (qm_o.Pauli.Z, -1.0j),
        (qm_o.Pauli.Y, qm_o.Pauli.Y): (qm_o.Pauli.I, 1.0),
        (qm_o.Pauli.Y, qm_o.Pauli.Z): (qm_o.Pauli.X, 1.0j),
        (qm_o.Pauli.Y, qm_o.Pauli.I): (qm_o.Pauli.Y, 1.0),
        (qm_o.Pauli.Z, qm_o.Pauli.X): (qm_o.Pauli.Y, 1.0j),
        (qm_o.Pauli.Z, qm_o.Pauli.Y): (qm_o.Pauli.X, -1.0j),
        (qm_o.Pauli.Z, qm_o.Pauli.Z): (qm_o.Pauli.I, 1.0),
        (qm_o.Pauli.Z, qm_o.Pauli.I): (qm_o.Pauli.Z, 1.0),
        (qm_o.Pauli.I, qm_o.Pauli.X): (qm_o.Pauli.X, 1.0),
        (qm_o.Pauli.I, qm_o.Pauli.Y): (qm_o.Pauli.Y, 1.0),
        (qm_o.Pauli.I, qm_o.Pauli.Z): (qm_o.Pauli.Z, 1.0),
        (qm_o.Pauli.I, qm_o.Pauli.I): (qm_o.Pauli.I, 1.0),
    }

    @staticmethod
    def get_pauli_string(pauli: qm_o.Pauli) -> str:
        """Get the string representation of a Pauli operator."""
        match pauli:
            case qm_o.Pauli.I:
                return "I"
            case qm_o.Pauli.X:
                return "X"
            case qm_o.Pauli.Y:
                return "Y"
            case qm_o.Pauli.Z:
                return "Z"
            case _:
                raise ValueError(f"Unknown Pauli operator: {pauli}")
