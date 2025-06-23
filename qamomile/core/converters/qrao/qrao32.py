"""
This modeule implements Quantum Random Access Optimization (QRAO) 
using :math:`(3,2,p)`-QRAC :cite:`teramoto2023quantum`.

The Ising Hamiltonian

.. math::
    H = \sum_{ij} J_{ij} Z_i Z_j + \sum_{i} h_i Z_i 

is converted into a relaxed Hamiltonian 
using the :math:`(3,2,p)`-QRAC method and the relaxed Hamiltonian becomes

.. math::
    \\tilde{H} = \sum_{ij} 6J_{ij} P_{f(i)} P_{f(j)} + \sum_{i} \sqrt{6}h_i P_{f(i)},\ P_{k,\mu} \in \{X', Y', Z'\}

Here, :math:`X'`, :math:`Y'`, and :math:`Z'` are the 2-local Pauli operator which defined as follows:

.. math::
        X' = \\frac{1}{\sqrt{6}}\left( \\frac{1}{2}X_1X_2 + \\frac{1}{2}X_1Z_2 + Z_1I_2 \\right), \\\\
        Y' = \\frac{1}{\sqrt{6}}\left(\\frac{1}{2}I_1X_2 + I_1Z_2 + \\frac{1}{2}Y_1Y_2 \\right), \\\\
        Z' = \\frac{1}{\sqrt{6}}\left(Z_1Z_2 - \\frac{1}{2}X_1I_2 - \\frac{1}{2}Z_1X_2 \\right).

The basic steps for constructing the Hamiltonian are the same as those for :math:`(3,1,p)`-QRAO.

:math:`i` th variable is mapped into Pauli :math:`\mu` operator of the :math:`k` th qubit by :math:`f(i)`.
For example, if :math:`f(i) = (2,0)`, the :math:`i` th variable is mapped into the Pauli :math:`Z` operator of the 2nd qubit.
If :math:`f(i) = (0,1)`, the :math:`i` th variable is mapped into the Pauli :math:`X` operator of the 0th qubit.
If :math:`f(i) = (4,2)`, the :math:`i` th variable is mapped into the Pauli :math:`Y` operator of the 4th qubit
The assignment of variables to qubits is determined by solving the graph coloring problem on the interaction graph 
so that :math:`f(i)` and :math:`f(j)` are assigned to different qubits.


This module provides functionality to convert optimization problems which written by `jijmodeling`
into relaxed Hamiltonians using :math:`(3,2,p)`-QRAO.

The `QRAC32Converter` class extends the `QuantumConverter` base class, specializing in
:math:`(3,2,p)` -QRAO-specific operations such as relaxed Hamiltonian generation and result decoding.


Key Features:
    - Generation of relaxed Hamiltonians for :math:`(3,2,p)`-QRAO
    - Graph coloring algorithm for qubit assignment
    - Retrieve an encoded Pauli operators list for Pauli Rounding
    - Decoding of rounding results into classical optimization solutions    
    
Attention:
    Currently, this module does not provide the rounding algorithm.

Note:
    This module requires `jijmodeling` for problem representation
    and decoding functionalities.
    

.. bibliography::
    :filter: docname in docnames

"""

from __future__ import annotations
import typing as typ
import numpy as np
from qamomile.core.converters.converter import QuantumConverter
from qamomile.core.converters.utils import is_close_zero
from qamomile.core.ising_qubo import IsingModel
import qamomile.core.operator as qm_o
from .qrao31 import color_group_to_qrac_encode
from .graph_coloring import greedy_graph_coloring, check_linear_term


def create_x_prime(idx: int) -> qm_o.PauliOperator:
    """
    Creates a X' operator for the given index.

    .. math::
            X' = \\frac{1}{\sqrt{6}}\left( \\frac{1}{2}X_1X_2 + \\frac{1}{2}X_1Z_2 + Z_1I_2\\right)

    Parameters:
    idx (int): The index of the first qubit.

    Returns:
    qm_o.PauliOperator: The X' operator for the given index.
    """
    Xi0 = qm_o.X(idx)
    Xi1 = qm_o.X(idx + 1)
    Zi0 = qm_o.Z(idx)
    Zi1 = qm_o.Z(idx + 1)
    return 1 / np.sqrt(6) * (1 / 2 * (Xi0 * Xi1) + 1 / 2 * (Xi0 * Zi1) + Zi0)


def create_y_prime(idx: int) -> qm_o.Hamiltonian:
    """
    Creates a Y' operator for the given index.

    .. math::
            Y' = \\frac{1}{\sqrt{6}}\left( \\frac{1}{2}I_1X_2 + I_1Z_2 + \\frac{1}{2}Y_1Y_2\\right)

    Parameters:
    idx (int): The index of the first qubit.

    Returns:
    qm_o.PauliOperator: The Y' operator for the given index.
    """
    Xi1 = qm_o.X(idx + 1)
    Zi1 = qm_o.Z(idx + 1)
    Yi0 = qm_o.Y(idx)
    Yi1 = qm_o.Y(idx + 1)
    return 1 / np.sqrt(6) * (1 / 2 * Xi1 + Zi1 + 1 / 2 * (Yi0 * Yi1))


def create_z_prime(idx: int) -> qm_o.Hamiltonian:
    """
    Creates a Z' operator for the given index.

    .. math::
            Z' = \\frac{1}{\sqrt{6}}\left( Z_1Z_2 - \\frac{1}{2}X_1I_2 - \\frac{1}{2}Z_1X_2\\right)

    Parameters:
    idx (int): The index of the first qubit.

    Returns:
    qm_o.PauliOperator: The Z' operator for the given index.
    """
    Xi0 = qm_o.X(idx)
    Xi1 = qm_o.X(idx + 1)
    Zi0 = qm_o.Z(idx)
    Zi1 = qm_o.Z(idx + 1)
    return 1 / np.sqrt(6) * (Zi0 * Zi1 - 1 / 2 * Xi0 - 1 / 2 * (Zi0 * Xi1))


def create_prime_operator(pauli_op: qm_o.PauliOperator) -> qm_o.Hamiltonian:
    if pauli_op.pauli == qm_o.Pauli.X:
        return create_x_prime(2 * pauli_op.index)
    elif pauli_op.pauli == qm_o.Pauli.Y:
        return create_y_prime(2 * pauli_op.index)
    elif pauli_op.pauli == qm_o.Pauli.Z:
        return create_z_prime(2 * pauli_op.index)
    else:
        raise ValueError("Invalid Pauli operator")


def qrac32_encode_ising(
    ising: IsingModel, color_group: dict[int, list[int]]
) -> tuple[qm_o.Hamiltonian, dict[int, qm_o.PauliOperator]]:
    encoded_ope = color_group_to_qrac_encode(color_group)

    offset = ising.constant

    hamiltonian = qm_o.Hamiltonian()
    hamiltonian.constant = offset

    # convert linear parts of the objective function into Hamiltonian.
    for idx, coeff in ising.linear.items():
        if is_close_zero(coeff):
            continue

        pauli = encoded_ope[idx]
        prime_i = create_prime_operator(pauli)
        hamiltonian += np.sqrt(6) * coeff * prime_i

    # create quad terms
    for (i, j), coeff in ising.quad.items():
        if is_close_zero(coeff):
            continue

        if i == j:
            hamiltonian.constant += coeff
            continue

        pauli_i = encoded_ope[i]
        prime_i = create_prime_operator(pauli_i)

        pauli_j = encoded_ope[j]
        prime_j = create_prime_operator(pauli_j)

        hamiltonian += 6 * coeff * prime_i * prime_j

    return hamiltonian, encoded_ope


class QRAC32Converter(QuantumConverter):
    """
    :math:`(3,2,p)`-QRAO (Quantum Random Access Optimization) converter class.

    This class provides methods to convert optimization problems into :math:`(3,2,p)`-QRAO
    relaxed Hamiltonians, and decode quantum computation results.

    Examples:

        .. code::

            from qamomile.core.converters.qrao.qrao32 import QRAC32Converter

            # Initialize with a compiled optimization problem instance
            qrao_converter = QRAC32Converter(compiled_instance)

            # Generate relaxed Hamiltonian
            cost_hamiltonian = qrao_converter.get_cost_hamiltonian()

    """

    max_color_group_size = 3

    def ising_encode(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[
            dict[str, dict[tuple[int, ...], tuple[float, float]]]
        ] = None,
    ) -> IsingModel:
        ising = super().ising_encode(multipliers, detail_parameters)

        _, color_group = greedy_graph_coloring(
            ising.quad.keys(), self.max_color_group_size
        )
        color_group = check_linear_term(
            color_group, list(ising.linear.keys()), self.max_color_group_size
        )

        self.color_group = color_group
        return ising

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """
        Construct the relaxed Hamiltonian for :math:`(3,2,p)`-QRAO.

        Returns:
            qm_o.Hamiltonian: The relaxed Hamiltonian.
        """
        ising = self.get_ising()

        hamiltonian, pauli_encoding = qrac32_encode_ising(ising, self.color_group)
        self.pauli_encoding = pauli_encoding
        return hamiltonian

    def get_encoded_pauli_list(self) -> list[qm_o.Hamiltonian]:
        """
        Get the encoded Pauli operators as a list of Hamiltonians.

        This method returns the Pauli Operators which correspond
        to the each variable in the Ising model.

        Returns:
            list[qm_o.Hamiltonian]: A list of the encoded Pauli operators.
        """
        # return the encoded Pauli operators as list
        ising = self.get_ising()
        num_qubits = len(self.color_group)
        zero_pauli = qm_o.Hamiltonian(num_qubits=num_qubits * 2)
        pauli_operators = [zero_pauli] * ising.num_bits()
        for idx, pauli in self.pauli_encoding.items():
            observable = create_prime_operator(pauli)
            pauli_operators[idx] = observable
        return pauli_operators
