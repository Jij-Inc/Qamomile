"""
This module provides the intermediate representation of Hamiltonian for quantum systems.

It defines classes and functions to create and manipulate Pauli operators and Hamiltonians,
which are fundamental in quantum mechanics and quantum computing.

Key Components:
- Pauli: An enumeration of Pauli operators (X, Y, Z, I).
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

import math
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
        I (int): Identity operator, represented by 3.
    """

    X = 0
    Y = 1
    Z = 2
    I = 3


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
        >>> H.add_term((PauliOperator(Pauli.X, 0), PauliOperator(Pauli.Y, 1)), 0.5)
        >>> H.add_term((PauliOperator(Pauli.Z, 2),), 1.0)
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
            >>> H.add_term((PauliOperator(Pauli.X, 0), PauliOperator(Pauli.Y, 1)), 0.5)
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
            >>> H.add_term((PauliOperator(Pauli.X, 0), PauliOperator(Pauli.Y, 1)), 0.5)
            >>> H.add_term((PauliOperator(Pauli.X, 0), PauliOperator(Pauli.Y, 1)), 0.5j)
            >>> print(H.terms)
            {(X0, Y1): (0.5+0.5j)}
        """

        operators, phase = simplify_pauliop_terms(operators)
        if operators:
            # Sort the operators to ensure consistent representation
            operators = tuple(
                sorted(operators, key=lambda x: x.index * 10 + x.pauli.value)
            )
            if operators in self._terms:
                self._terms[operators] += phase * coeff
            else:
                self._terms[operators] = phase * coeff
        else:
            self.constant += phase * coeff

    @property
    def num_qubits(self) -> int:
        """
        Calculates the number of qubits in the Hamiltonian.

        Returns:
            int: The number of qubits, which is the highest qubit index plus one.

        Example:
            >>> H = Hamiltonian()
            >>> H.add_term((PauliOperator(Pauli.X, 0), PauliOperator(Pauli.Y, 3)), 1.0)
            >>> print(H.num_qubits)
            4
        """
        # Get the maximum qubit index from the terms.
        #    If the terms is {(X0, Y1): 0.5, (Z3,): 1.0}, then it will be 3.
        max_qubit_index = -1  # If no terms, assume no qubits are used
        if self._terms:
            max_qubit_index = max(op.index for term in self.terms.keys() for op in term)
        # Compute the number of qubits to realise all those terms.
        num_qubits_to_realise_terms = max_qubit_index + 1

        if self._num_qubits is None:
            # Return the number of qubits to realise all terms if the user did not specify the number of qubits.
            return num_qubits_to_realise_terms
        else:
            # Return the maximum of the two if the user specified the number of qubits.
            return max(num_qubits_to_realise_terms, self._num_qubits)

    def to_latex(self) -> str:
        """
        Converts the Hamiltonian to a LaTeX representation.

        This function does not add constant term when we show the Hamiltonian.
        This function does not add $ symbols.

        Returns:
            str:
                A LaTeX representation of the Hamiltonian.

        .. code::

            import qamomile.core.operator as qm_o
            import IPython.display as ipd

            h = qm_o.Hamiltonian()
            h += -qm_o.X(0) * qm_o.Y(1) - 2.0 * qm_o.Z(0) * qm_o.Z(1)

            # Show the Hamiltonian in LaTeX at Jupyter Notebook
            ipd.display(ipd.Latex("$" + h.to_latex() + "$"))
        """
        h_str = ""
        counter = 0

        pauli_map = {Pauli.X: "X", Pauli.Y: "Y", Pauli.Z: "Z"}

        for term, coeff in self.terms.items():
            term_str = ""

            for op in term:
                if op.pauli == Pauli.I:
                    continue

                pauli_str = pauli_map.get(op.pauli, "")
                term_str += f"{pauli_str}_{{{op.index}}}"

            # At first term or h_str is still empty, we don't need to add a sign
            if counter == 0 or h_str == "":
                if abs(coeff) == 1:
                    h_str += term_str if coeff > 0 else "-" + term_str
                else:
                    h_str += f"{coeff}{term_str}"
            else:
                if term_str == "":
                    # This means the term is just a constant. We can skip this.
                    continue
                if abs(coeff) == 1:
                    h_str += "+" + term_str if coeff > 0 else "-" + term_str
                else:
                    h_str += (
                        f"+{coeff}" + term_str if coeff > 0 else f"{coeff}" + term_str
                    )

            counter += 1

        return h_str

    def __repr__(self) -> str:
        """
        Provides a string representation of the Hamiltonian.

        Returns:
            str: A string representation of the Hamiltonian terms.

        Example:
            >>> H = Hamiltonian()
            >>> H.add_term((PauliOperator(Pauli.X, 0), PauliOperator(Pauli.Y, 1)), 0.5)
            >>> H.add_term((PauliOperator(Pauli.Z, 2),), 1.0)
            >>> print(H)
            Hamiltonian((X0, Y1): 0.5, (Z2,): 1.0)
        """
        terms_str = ", ".join(
            f"{operators}: {coeff}" for operators, coeff in self._terms.items()
        )
        return f"Hamiltonian({terms_str})"

    def __eq__(self, other):
        if not isinstance(other, Hamiltonian):
            return False
        return self.terms == other.terms and self.constant == other.constant

    def __add__(self, other):
        if isinstance(other, Hamiltonian):
            h = Hamiltonian()
            h._terms = self._terms.copy()
            h.constant = self.constant
            for term, coeff in other.terms.items():
                h.add_term(term, coeff)
            h.constant += other.constant

            if h.num_qubits < self.num_qubits:
                h._num_qubits = self.num_qubits

            return h
        elif isinstance(other, (int, float, complex)):
            h = Hamiltonian(num_qubits=self.num_qubits)
            h._terms = self._terms.copy()
            h.constant = self.constant + other

            return h
        else:
            raise ValueError("Unsupported addition operation.")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-1.0 * other)

    def __rsub__(self, other):
        return -1.0 * self + other

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            h = Hamiltonian(num_qubits=self.num_qubits)
            for term, coeff in self.terms.items():
                h.add_term(term, coeff * other)
            h.constant = self.constant * other
            return h
        elif isinstance(other, Hamiltonian):
            h = Hamiltonian()
            for term1, coeff1 in self.terms.items():
                for term2, coeff2 in other.terms.items():
                    term, phase = simplify_pauliop_terms(term1 + term2)
                    if term:
                        h.add_term(term, phase * coeff1 * coeff2)
                    else:
                        h.constant += phase * coeff1 * coeff2

            if not math.isclose(abs(other.constant), 0.0, abs_tol=1e-15):
                for terms, coeff1 in self.terms.items():
                    h.add_term(terms, coeff1 * other.constant)

            if not math.isclose(abs(self.constant), 0.0, abs_tol=1e-15):
                for terms, coeff2 in other.terms.items():
                    h.add_term(terms, coeff2 * self.constant)

            h.constant += self.constant * other.constant

            if h.num_qubits < self.num_qubits:
                h._num_qubits = self.num_qubits

            return h
        else:
            raise ValueError("Unsupported multiplication operation.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return -1.0 * self


def X(index: int) -> Hamiltonian:
    """
    Creates a Pauli X operator for a specified qubit.

    Args:
        index (int): The index of the qubit.

    Returns:
        Hamiltonian: A Pauli X Hamiltonian operator acting on the specified qubit.

    Example:
        >>> X0 = X(0)
        >>> print(X0)
        Hamiltonian((X0,): 1.0)
    """
    h = Hamiltonian()
    h.add_term((PauliOperator(Pauli.X, index),), 1.0)
    return h


def Y(index: int) -> Hamiltonian:
    """
    Creates a Pauli Y operator for a specified qubit.

    Args:
        index (int): The index of the qubit.

    Returns:
        Hamiltonian: A Pauli Y Hamiltonian operator acting on the specified qubit.

    Example:
        >>> Y1 = Y(1)
        >>> print(Y1)
        Hamiltonian((Y1,): 1.0)
    """
    h = Hamiltonian()
    h.add_term((PauliOperator(Pauli.Y, index),), 1.0)
    return h


def Z(index: int) -> Hamiltonian:
    """
    Creates a Pauli Z operator for a specified qubit.

    Args:
        index (int): The index of the qubit.

    Returns:
        Hamiltonian: A Pauli Z Hamiltonian operator acting on the specified qubit.

    Example:
        >>> Z2 = Z(2)
        >>> print(Z2)
        Hamiltonian((Z2,): 1.0)
    """
    h = Hamiltonian()
    h.add_term((PauliOperator(Pauli.Z, index),), 1.0)
    return h


def multiply_pauli_same_qubit(
    pauli1: PauliOperator, pauli2: PauliOperator
) -> tuple[PauliOperator, complex]:
    """
    Multiplies two Pauli operators acting on the same qubit.

    Args:
        pauli1 (PauliOperator): The first Pauli operator.
        pauli2 (PauliOperator): The second Pauli operator.

    Returns:
        tuple[PauliOperator, complex]: A tuple containing the resulting Pauli operator and the complex coefficient.

    Raises:
        ValueError: If the Pauli operators act on different qubits.

    Example:
        >>> X0 = PauliOperator(Pauli.X, 0)
        >>> Y0 = PauliOperator(Pauli.Y, 0)
        >>> Z0 = PauliOperator(Pauli.Z, 0)
        >>> multiply_pauli_same_qubit(X0, Y0)
        (Z0, 1j)
    """

    if pauli1.index == pauli2.index:
        if pauli1.pauli == pauli2.pauli:
            return PauliOperator(Pauli.I, pauli1.index), 1.0

        elif pauli1.pauli == Pauli.X:
            if pauli2.pauli == Pauli.Y:
                return PauliOperator(Pauli.Z, pauli1.index), 1.0j

            elif pauli2.pauli == Pauli.Z:
                return PauliOperator(Pauli.Y, pauli1.index), -1.0j

            elif pauli2.pauli == Pauli.I:
                return PauliOperator(Pauli.X, pauli1.index), 1.0

        elif pauli1.pauli == Pauli.Y:
            if pauli2.pauli == Pauli.X:
                return PauliOperator(Pauli.Z, pauli1.index), -1.0j

            elif pauli2.pauli == Pauli.Z:
                return PauliOperator(Pauli.X, pauli1.index), 1.0j

            elif pauli2.pauli == Pauli.I:
                return PauliOperator(Pauli.Y, pauli1.index), 1.0

        elif pauli1.pauli == Pauli.Z:
            if pauli2.pauli == Pauli.X:
                return PauliOperator(Pauli.Y, pauli1.index), 1.0j

            elif pauli2.pauli == Pauli.Y:
                return PauliOperator(Pauli.X, pauli1.index), -1.0j

            elif pauli2.pauli == Pauli.I:
                return PauliOperator(Pauli.Z, pauli1.index), 1.0

        elif pauli1.pauli == Pauli.I:
            return pauli2, 1.0

    else:
        raise ValueError("Pauli operators act on different qubits.")


def simplify_pauliop_terms(
    term: tuple[PauliOperator],
) -> tuple[tuple[PauliOperator, ...], complex]:
    """
    Simplifies a tuple of Pauli operators by combining operators acting on the same qubit.

    Args:
        term (tuple[PauliOperator]): A tuple of Pauli operators.

    Returns:
        tuple[tuple[PauliOperator,...],complex]: A tuple containing the simplified Pauli operators and the phase factor.

    Example:
        >>> X0 = PauliOperator(Pauli.X, 0)
        >>> Y0 = PauliOperator(Pauli.Y, 0)
        >>> Z1 = PauliOperator(Pauli.Z, 1)
        >>> simplify_pauliop_terms((X0, Y0, Z1))
        ((Z0, Z1), 1j)
    """
    phase = 1.0
    paulis = {}

    for op in term:
        if op.index in paulis:
            paulis[op.index].append(op)
        else:
            paulis[op.index] = [op]

    for qubit_index, _pauli_list in paulis.items():
        if len(_pauli_list) == 1:
            continue
        else:
            op = _pauli_list[0]
            for i in range(1, len(_pauli_list)):
                op, _phase = multiply_pauli_same_qubit(op, _pauli_list[i])
                phase *= _phase

            paulis[qubit_index] = [op]

    pauli_list = []

    for _pauli_list in paulis.values():
        if _pauli_list[0].pauli != Pauli.I:
            pauli_list.append(_pauli_list[0])

    return tuple(pauli_list), phase
