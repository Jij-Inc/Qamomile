"""
This module provides the intermediate representation of Hamiltonian for quantum systems.

It defines classes and functions to create and manipulate Pauli operators and Hamiltonians,
which are fundamental in quantum mechanics and quantum computing.

Key Components:
- Pauli: An enumeration of Pauli operators (X, Y, Z).
- PauliOperator: A class representing a single Pauli operator acting on a specific qubit.
- Hamiltonian: A class representing a quantum Hamiltonian as a sum of Pauli operator products.

Usage:
    from qamomile.operator.hamiltonian import X, Y, Z, Hamiltonian

    # Create Pauli operators
    X0 = X(0)  # Pauli X operator on qubit 0
    Y1 = Y(1)  # Pauli Y operator on qubit 1

    # Create a Hamiltonian
    H = Hamiltonian()
    H.add_term((X0, Y1), 0.5)  # Add term 0.5 * X0 * Y1 to the Hamiltonian
    H.add_term((Z(2),), 1.0)   # Add term 1.0 * Z2 to the Hamiltonian

    # Access Hamiltonian properties
    print(H.terms)
    print(H.num_qubits)
"""

import dataclasses
import enum
from typing import Dict, Tuple, Union, Optional


class Pauli(enum.Enum):
    """
    Enum class for Pauli operators.

    Attributes:
        X (int): Pauli X operator, represented by 0.
        Y (int): Pauli Y operator, represented by 1.
        Z (int): Pauli Z operator, represented by 2.
    """

    X = 0
    Y = 1
    Z = 2


@dataclasses.dataclass
class PauliOperator:
    """
    Represents a single Pauli operator acting on a specific qubit.

    Attributes:
        pauli (Pauli): The type of Pauli operator (X, Y, or Z).
        index (int): The index of the qubit on which this operator acts.

    Example:
        >>> X0 = PauliOperator(Pauli.X, 0)
        >>> print(X0)
        X0
    """

    pauli: Pauli
    index: int

    def __hash__(self) -> int:
        """
        Makes this class hashable for use as a dictionary key in the `Hamiltonian` class.

        Returns:
            int: A hash value based on the Pauli type and qubit index.
        """
        return hash((self.pauli, self.index))

    def __repr__(self) -> str:
        """
        Provides a string representation of the PauliOperator.

        Returns:
            str: A string in the format "PauliTypeQubitIndex" (e.g., "X0" for Pauli X on qubit 0).
        """
        return f"{self.pauli.name}{self.index}"


def X(index: int) -> PauliOperator:
    """
    Creates a Pauli X operator for a specified qubit.

    Args:
        index (int): The index of the qubit.

    Returns:
        PauliOperator: A Pauli X operator acting on the specified qubit.

    Example:
        >>> X0 = X(0)
        >>> print(X0)
        X0
    """
    return PauliOperator(Pauli.X, index)


def Y(index: int) -> PauliOperator:
    """
    Creates a Pauli Y operator for a specified qubit.

    Args:
        index (int): The index of the qubit.

    Returns:
        PauliOperator: A Pauli Y operator acting on the specified qubit.

    Example:
        >>> Y1 = Y(1)
        >>> print(Y1)
        Y1
    """
    return PauliOperator(Pauli.Y, index)


def Z(index: int) -> PauliOperator:
    """
    Creates a Pauli Z operator for a specified qubit.

    Args:
        index (int): The index of the qubit.

    Returns:
        PauliOperator: A Pauli Z operator acting on the specified qubit.

    Example:
        >>> Z2 = Z(2)
        >>> print(Z2)
        Z2
    """
    return PauliOperator(Pauli.Z, index)


class Hamiltonian:
    """
    Represents a quantum Hamiltonian as a sum of Pauli operator products.

    The Hamiltonian is stored as a dictionary where keys are tuples of PauliOperators
    and values are their corresponding coefficients.

    Attributes:
        _terms (Dict[Tuple[PauliOperator, ...], complex]): The terms of the Hamiltonian.
        constant (float): A constant term added to the Hamiltonian.

    Example:
        >>> H = Hamiltonian()
        >>> H.add_term((X(0), Y(1)), 0.5)
        >>> H.add_term((Z(2),), 1.0)
        >>> print(H.terms)
        {(X0, Y1): 0.5, (Z2,): 1.0}
    """

    def __init__(self, num_qubits: Optional[int] = None) -> None:
        self._terms: Dict[Tuple[PauliOperator, ...], complex] = {}
        self.constant: float = 0.0
        self._num_qubits = num_qubits

    @property
    def terms(self) -> Dict[Tuple[PauliOperator, ...], complex]:
        """
        Getter for the terms of the Hamiltonian.

        Returns:
            Dict[Tuple[PauliOperator, ...], complex]: A dictionary representing the Hamiltonian terms.

        Example:
            >>> H = Hamiltonian()
            >>> H.add_term((X(0), Y(1)), 0.5)
            >>> print(H.terms)
            {(X0, Y1): 0.5}
        """
        return self._terms

    def add_term(
        self, operators: Tuple[PauliOperator, ...], coeff: Union[float, complex]
    ):
        """
        Adds a term to the Hamiltonian.

        This method adds a product of Pauli operators with a given coefficient to the Hamiltonian.
        If the term already exists, the coefficients are summed.

        Args:
            operators (Tuple[PauliOperator, ...]): A tuple of PauliOperators representing the term.
            coeff (Union[float, complex]): The coefficient of the term.

        Example:
            >>> H = Hamiltonian()
            >>> H.add_term((X(0), Y(1)), 0.5)
            >>> H.add_term((X(0), Y(1)), 0.5j)
            >>> print(H.terms)
            {(X0, Y1): (0.5+0.5j)}
        """
        # Sort the operators to ensure consistent representation
        operators = tuple(sorted(operators, key=lambda x: x.index * 10 + x.pauli.value))
        if operators in self._terms:
            self._terms[operators] += coeff
        else:
            self._terms[operators] = coeff

    @property
    def num_qubits(self) -> int:
        """
        Calculates the number of qubits in the Hamiltonian.

        Returns:
            int: The number of qubits, which is the highest qubit index plus one.

        Example:
            >>> H = Hamiltonian()
            >>> H.add_term((X(0), Y(3)), 1.0)
            >>> print(H.num_qubits)
            4
        """
        if self._num_qubits is not None:
            return self._num_qubits
        if not self._terms:
            return 0
        return max(op.index for term in self.terms.keys() for op in term) + 1

    def __repr__(self) -> str:
        """
        Provides a string representation of the Hamiltonian.

        Returns:
            str: A string representation of the Hamiltonian terms.

        Example:
            >>> H = Hamiltonian()
            >>> H.add_term((X(0), Y(1)), 0.5)
            >>> H.add_term((Z(2),), 1.0)
            >>> print(H)
            Hamiltonian((X0, Y1): 0.5, (Z2,): 1.0)
        """
        terms_str = ", ".join(
            f"{operators}: {coeff}" for operators, coeff in self._terms.items()
        )
        return f"Hamiltonian({terms_str})"
