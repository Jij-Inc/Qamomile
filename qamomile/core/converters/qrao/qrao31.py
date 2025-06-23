"""
This modeule implements Quantum Random Access Optimization (QRAO)
using :math:`(3,1,p)`-QRAC :cite:`fuller2024approximate`.

The Ising Hamiltonian

.. math::
    H = \sum_{ij} J_{ij} Z_i Z_j + \sum_{i} h_i Z_i

is converted into a relaxed Hamiltonian
using the :math:`(3,1,p)`-QRAC method and the relaxed Hamiltonian becomes

.. math::
    \\tilde{H} = \sum_{ij} 3J_{ij} P_{f(i)} P_{f(j)} + \sum_{i} \sqrt{3}h_i P_{f(i)},\ P_{k,\mu} \in \{X, Y, Z\}

where :math:`i` th variable is mapped into Pauli :math:`\mu` operator of the :math:`k` th qubit by :math:`f(i)`.
For example, if :math:`f(i) = (2,0)`, the :math:`i` th variable is mapped into the Pauli :math:`Z` operator of the 2nd qubit.
If :math:`f(i) = (0,1)`, the :math:`i` th variable is mapped into the Pauli :math:`X` operator of the 0th qubit.
If :math:`f(i) = (4,2)`, the :math:`i` th variable is mapped into the Pauli :math:`Y` operator of the 4th qubit
The assignment of variables to qubits is determined by solving the graph coloring problem on the interaction graph
so that :math:`f(i)` and :math:`f(j)` are assigned to different qubits.


This module provides functionality to convert optimization problems which written by `jijmodeling`
into relaxed Hamiltonians using :math:`(3,1,p)`-QRAO.

The `QRAC31Converter` class extends the `QuantumConverter` base class, specializing in
:math:`(3,1,p)` -QRAO-specific operations such as relaxed Hamiltonian generation and result decoding.


Key Features:
    - Generation of relaxed Hamiltonians for :math:`(3,1,p)`-QRAO
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
from .graph_coloring import greedy_graph_coloring, check_linear_term


def color_group_to_qrac_encode(
    color_group: dict[int, list[int]],
) -> dict[int, qm_o.PauliOperator]:
    """qrac encode

    Args:
        color_group (dict[int, list[int]]): key is color (qubit's index). value is list of bit's index.

    Returns:
        dict[int, tuple[int, Pauli]]: key is bit's index. value is tuple of qubit's index and Pauli operator kind.

    Examples:
        >>> color_group = {0: [0, 1, 2], 1: [3, 4], 2: [6,]}
        >>> color_group_to_qrac_encode(color_group)
        {0: Z0, 1: X0, 2: Y0, 3: Z1, 4: X1, 6: Z2}

    """
    qrac31 = {}
    paulis = [qm_o.Pauli.Z, qm_o.Pauli.X, qm_o.Pauli.Y]
    for color, group in color_group.items():
        for ope_idx, bit_index in enumerate(group):
            qrac31[bit_index] = qm_o.PauliOperator(paulis[ope_idx], color)
    return qrac31


def qrac31_encode_ising(
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
        hamiltonian.add_term((pauli,), np.sqrt(3) * coeff)

    # create Pauli terms
    for (i, j), coeff in ising.quad.items():
        if is_close_zero(coeff):
            continue

        if i == j:
            hamiltonian.constant += coeff
            continue

        pauli_i = encoded_ope[i]

        pauli_j = encoded_ope[j]

        hamiltonian.add_term((pauli_i, pauli_j), 3 * coeff)

    return hamiltonian, encoded_ope


class QRAC31Converter(QuantumConverter):
    """
    :math:`(3,1,p)`-QRAO (Quantum Random Access Optimization) converter class.

    This class provides methods to convert optimization problems into :math:`(3,1,p)`-QRAO
    relaxed Hamiltonians, and decode quantum computation results.

    Examples:

        .. code::

            from qamomile.core.converters.qrao.qrao31 import QRAC31Converter

            # Initialize with a compiled optimization problem instance
            qrao_converter = QRAC31Converter(compiled_instance)

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
        Construct the relaxed Hamiltonian for :math:`(3,1,p)`-QRAO.

        Returns:
            qm_o.Hamiltonian: The relaxed Hamiltonian.
        """
        ising = self.get_ising()

        hamiltonian, pauli_encoding = qrac31_encode_ising(ising, self.color_group)
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
        zero_pauli = qm_o.Hamiltonian(num_qubits=num_qubits)
        pauli_operators = [zero_pauli] * ising.num_bits()
        for idx, pauli in self.pauli_encoding.items():
            observable = qm_o.Hamiltonian(num_qubits=num_qubits)
            observable.add_term((pauli,), 1.0)
            pauli_operators[idx] = observable
        return pauli_operators
