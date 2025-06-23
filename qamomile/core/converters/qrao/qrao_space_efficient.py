"""
This modeule implements Space efficient Quantum Random Access Optimization (QRAO) :cite:`teramoto2023quantum`.
This method is referred to as Space Compression Ratio Preserving Quantum Relaxation in the paper.
With this method, the compression ratio can always be kept at 2.

The Ising Hamiltonian

.. math::
    H = \sum_{ij} J_{ij} Z_i Z_j + \sum_{i} h_i Z_i 

is converted into a relaxed Hamiltonian as follows:

.. math::
    \\tilde{H} = \sum_{ij} J_{ij} O_{f(i),f(j)} + \sum_{i} h_i P_{f(i)},\ P_{k,\mu} \in \{X, Y, Z\}

where :math:`i` th variable is mapped into Pauli :math:`\mu` operator of the :math:`k` th qubit by :math:`f(i)`.

This Hamiltonian is constructed using the following procedure.

First, divide all the nodes into two groups.
Then, assign Pauli :math:`X` and :math:`Y` to each group.
So, :math:`f(i)` always provides the Pauli :math:`X` or :math:`Y` operator on :math:`k` th qubit
and :math:`O_{f(i),f(j)}` becomes :math:`P_{f(i)} P_{f(j)}`.

On the other hand, if the nodes next to each other on the interaction graph are assigned to the same qubit 
(:math:`f(i)` and :math:`f(j)` are assigned to same :math:`k` th qubit), 
:math:`O_{f(i),f(j)}` becomes :math:`Z_k`.

In summary, :math:`O_{f(i),f(j)}` becomes as follows:

.. math::
    O_{f(i),f(j)} = \\begin{cases}
        3P_{f(i)} P_{f(j)} & \\text{if } f(i) \\text{ and } f(j) \\text{ map on different qubit} \\\\
        \sqrt{3}Z_k & \\text{if } f(i) \\text{ and } f(j) \\text{ map on same qubit}
    \end{cases}

This module provides functionality to convert optimization problems which written by `jijmodeling`
into relaxed Hamiltonians using above procedure.

The `QRACSpaceEfficientConverter` class extends the `QuantumConverter` base class, specializing in
Space efficient QRAO-specific operations such as relaxed Hamiltonian generation and result decoding.


Key Features:
    - Generation of relaxed Hamiltonians for Space efficient QRAO
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
from qamomile.core.ising_qubo import IsingModel
import qamomile.core.operator as qm_o
from .graph_coloring import greedy_graph_coloring, check_linear_term
from qamomile.core.converters.utils import is_close_zero


def numbering_space_efficient_encode(
    ising: IsingModel,
) -> dict[int, qm_o.PauliOperator]:
    """
    Encodes the Ising model into a space efficient and provides corresponding Pauli operator.

    Args:
        ising (IsingModel): The Ising model to be encoded.

    Returns:
        dict[int, qm_o.PauliOperator]: A dictionary mapping qubit indices to Pauli operators.
    """
    max_quad_index = max(max(t) for t in ising.quad.keys())
    max_linear_index = max(ising.linear.keys())
    num_vars = max(max_quad_index, max_linear_index) + 1

    encode = {}
    pauli_ope = [qm_o.Pauli.X, qm_o.Pauli.Y]
    for i in range(num_vars):
        qubit_index = i // 2
        color = i % 2
        encode[i] = qm_o.PauliOperator(pauli_ope[color], qubit_index)
    return encode


def qrac_space_efficient_encode_ising(
    ising: IsingModel,
) -> tuple[qm_o.Hamiltonian, dict[int, qm_o.PauliOperator]]:
    encoded_ope = numbering_space_efficient_encode(ising)

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

        if pauli_i.index == pauli_j.index:
            hamiltonian.add_term(
                (qm_o.PauliOperator(qm_o.Pauli.Z, pauli_i.index),), np.sqrt(3) * coeff
            )
        else:
            hamiltonian.add_term((pauli_i, pauli_j), 3 * coeff)

    return hamiltonian, encoded_ope


class QRACSpaceEfficientConverter(QuantumConverter):
    """
    Space efficient  QRAO (Quantum Random Access Optimization) converter class.

    This class provides methods to convert optimization problems into Space efficient  QRAO
    relaxed Hamiltonians, and decode quantum computation results.

    Examples:

        .. code::

            from qamomile.core.converters.qrao.qrao_space_efficient import QRACSpaceEfficientConverter

            # Initialize with a compiled optimization problem instance
            qrao_converter = QRACSpaceEfficientConverter(compiled_instance)

            # Generate relaxed Hamiltonian
            cost_hamiltonian = qrao_converter.get_cost_hamiltonian()
    """

    def ising_encode(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[
            dict[str, dict[tuple[int, ...], tuple[float, float]]]
        ] = None,
    ) -> IsingModel:
        ising = super().ising_encode(multipliers, detail_parameters)
        return ising

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """
        Construct the relaxed Hamiltonian for Space Efficient QRAO.

        Returns:
            qm_o.Hamiltonian: The relaxed Hamiltonian.
        """
        ising = self.get_ising()

        hamiltonian, pauli_encoding = qrac_space_efficient_encode_ising(ising)
        self.pauli_encoding = pauli_encoding
        self.num_qubits = hamiltonian.num_qubits
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

        zero_pauli = qm_o.Hamiltonian(num_qubits=self.num_qubits)
        pauli_operators = [zero_pauli] * ising.num_bits()
        for idx, pauli in self.pauli_encoding.items():
            observable = qm_o.Hamiltonian(num_qubits=self.num_qubits)
            observable.add_term((pauli,), 1.0)
            pauli_operators[idx] = observable
        return pauli_operators
