from __future__ import annotations

import itertools
import typing as typ
from collections.abc import Callable

import numpy as np
import qiskit as qk
import qiskit.quantum_info as qk_info
from qiskit.primitives import Estimator


def define_pauli_op(
    num_register_bits: int, ancilla: bool = False
) -> list[qk_info.SparsePauliOp]:
    """Function to define pauli operators

    Args:
        num_register_bits (int): number of register bits
        ancilla (bool, optional): whether to add ancilla qubit |1> or not. Defaults to False.

    Raises:
        ValueError: if num_register_bits < 1

    Returns:
        list[qk_info.SparsePauliOp]: list of pauli operators
    """

    if num_register_bits < 1:
        raise ValueError("num_register_bits must be greater than 0")

    z = [[False], [True]]
    x = [[False], [False]]
    zero_op = qk_info.SparsePauliOp(
        qk_info.PauliList.from_symplectic(z, x), coeffs=[1 / 2, 1 / 2]
    )

    one_op = qk_info.SparsePauliOp(
        qk_info.PauliList.from_symplectic(z, x), coeffs=[1 / 2, -1 / 2]
    )

    identity_op = qk_info.SparsePauliOp(qk_info.Pauli(([0], [0])))

    ancilla_operator = one_op if ancilla else identity_op

    pauli_ops = []

    if num_register_bits == 1:
        for _op in [one_op, zero_op]:
            pauli_ops.append(_op.tensor(ancilla_operator))
    else:
        for val in itertools.product(
            [one_op, zero_op], repeat=num_register_bits
        ):
            pauli_op = val[0]
            for _op in val[1:]:
                pauli_op = pauli_op.tensor(_op)

            pauli_ops.append(pauli_op.tensor(ancilla_operator))

    return pauli_ops


def initialize_cost_function(
    qubo: dict[tuple[int, int], float], num_cbits: int
) -> Callable[[np.array], float]:
    """Function to initialize cost function for minimal encoding

    Args:
        qubo (dict[tuple[int, int], float]): QUBO Matrix
        num_cbit (int): number of classical bits

    Returns:
        Callable[[np.array], float]: cost function for minimal encoding
    """

    # define cost function
    def cost_function(one_coeffs: np.array) -> float:
        """Function to compute value of cost function for minimal encoding

        Args:
            one_coeffs (np.array): the ratio of the expectation value of each pauli operator for when ancilla qubit is |1> and the expectation value of each pauli operator ignoring ancilla qubit

        Returns:
            float: value of cost function
        """

        cost = 0
        for i in range(num_cbits):
            for j in range(i, num_cbits):
                if i != j:
                    cost += 2 * qubo[(i, j)] * one_coeffs[i] * one_coeffs[j]
                else:
                    cost += qubo[(i, j)] * one_coeffs[i]

        return cost

    return cost_function
