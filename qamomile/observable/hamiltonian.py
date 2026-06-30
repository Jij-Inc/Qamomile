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

Features:
- Factory methods for common Hamiltonians (zero, identity, single_pauli)
- Iterator protocol for looping over terms
- Qubit index remapping for circuit integration
- Modern type annotations (PEP 604)
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Iterator

import numpy as np

# Numerical tolerances for interpreting Hamiltonian coefficients. They live
# with the Hamiltonian (rather than in any backend / emit module) because they
# describe properties of the operator itself and are shared across consumers:
# Hamiltonian arithmetic here, observable conversion, and the Pauli-evolution
# emit paths.
#
# A coefficient whose magnitude is at or below this is treated as zero (the
# term is dropped / not emitted). The slack keeps coefficients that cancel to
# ~0 during arithmetic from lingering or emitting spurious gates.
PAULI_TERM_ZERO_ATOL = 1e-15
# A coefficient whose imaginary part exceeds this fails the Hermiticity
# requirement: a Hamiltonian is Hermitian (real coefficients) only within this
# slack, which absorbs floating-point imaginary residue from complex
# arithmetic. ``exp(-i * gamma * H)`` is unitary only for a Hermitian ``H``.
HERMITIAN_IMAG_ATOL = 1e-10


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
    I = 3  # noqa: E741


# Dense 2x2 matrices for each single-qubit Pauli. Hoisted to module level
# so ``Hamiltonian.to_numpy()`` does not rebuild them on every call.
_PAULI_MATRICES: dict[Pauli, np.ndarray] = {
    Pauli.I: np.eye(2, dtype=np.complex128),
    Pauli.X: np.array([[0, 1], [1, 0]], dtype=np.complex128),
    Pauli.Y: np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    Pauli.Z: np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


# Pauli multiplication table: (P1, P2) -> (Result, Phase)
# Used for efficient Pauli operator multiplication
_PAULI_MUL_TABLE: dict[tuple[Pauli, Pauli], tuple[Pauli, complex]] = {
    (Pauli.I, Pauli.I): (Pauli.I, 1),
    (Pauli.I, Pauli.X): (Pauli.X, 1),
    (Pauli.I, Pauli.Y): (Pauli.Y, 1),
    (Pauli.I, Pauli.Z): (Pauli.Z, 1),
    (Pauli.X, Pauli.I): (Pauli.X, 1),
    (Pauli.Y, Pauli.I): (Pauli.Y, 1),
    (Pauli.Z, Pauli.I): (Pauli.Z, 1),
    (Pauli.X, Pauli.X): (Pauli.I, 1),
    (Pauli.Y, Pauli.Y): (Pauli.I, 1),
    (Pauli.Z, Pauli.Z): (Pauli.I, 1),
    (Pauli.X, Pauli.Y): (Pauli.Z, 1j),
    (Pauli.Y, Pauli.X): (Pauli.Z, -1j),
    (Pauli.Y, Pauli.Z): (Pauli.X, 1j),
    (Pauli.Z, Pauli.Y): (Pauli.X, -1j),
    (Pauli.Z, Pauli.X): (Pauli.Y, 1j),
    (Pauli.X, Pauli.Z): (Pauli.Y, -1j),
}


@dataclasses.dataclass(frozen=True)
class PauliOperator:
    """
    Represents a single Pauli operator acting on a specific qubit.

    Frozen so that ``Hamiltonian.copy()`` can share term operator
    references across the original and the copy without risk of one
    side mutating an operator the other still observes.  None of the
    existing callers mutate ``pauli`` or ``index`` after construction,
    so freezing is a no-op behaviourally but makes the immutability
    claim that ``Hamiltonian.copy()``'s docstring relies on real.

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
        _terms (dict[tuple[PauliOperator, ...], complex]): The terms of the Hamiltonian.
        constant (float | complex): A constant term added to the Hamiltonian.

    Example:
        >>> H = Hamiltonian()
        >>> H.add_term((PauliOperator(Pauli.X, 0), PauliOperator(Pauli.Y, 1)), 0.5)
        >>> H.add_term((PauliOperator(Pauli.Z, 2),), 1.0)
        >>> print(H.terms)
        {(X0, Y1): 0.5, (Z2,): 1.0}
    """

    def __init__(self, num_qubits: int | None = None) -> None:
        self._terms: dict[tuple[PauliOperator, ...], complex] = {}
        self.constant: float | complex = 0.0
        self._num_qubits = num_qubits

    @property
    def terms(self) -> dict[tuple[PauliOperator, ...], complex]:
        """
        Getter for the terms of the Hamiltonian.

        Returns:
            dict[tuple[PauliOperator, ...], complex]: A dictionary representing the Hamiltonian terms.

        Example:
            >>> H = Hamiltonian()
            >>> H.add_term((PauliOperator(Pauli.X, 0), PauliOperator(Pauli.Y, 1)), 0.5)
            >>> print(H.terms)
            {(X0, Y1): 0.5}
        """
        return self._terms

    @classmethod
    def zero(cls, num_qubits: int | None = None) -> Hamiltonian:
        """Create a zero Hamiltonian."""
        return cls(num_qubits=num_qubits)

    @classmethod
    def identity(
        cls, coeff: float | complex = 1.0, num_qubits: int | None = None
    ) -> Hamiltonian:
        """Create a scalar times identity Hamiltonian."""
        h = cls(num_qubits=num_qubits)
        h.constant = coeff
        return h

    @classmethod
    def single_pauli(
        cls, pauli: Pauli, index: int, coeff: float | complex = 1.0
    ) -> Hamiltonian:
        """Create a single Pauli term Hamiltonian."""
        h = cls()
        h.add_term((PauliOperator(pauli, index),), coeff)
        return h

    def add_term(self, operators: tuple[PauliOperator, ...], coeff: float | complex):
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
            self._add_simplified_term(operators, phase * coeff)
        else:
            self.constant += phase * coeff

    def _add_simplified_term(
        self,
        operators: tuple[PauliOperator, ...],
        coeff: float | complex,
    ) -> None:
        """Add a pre-simplified Pauli term, skipping a second simplification pass.

        Fast path for callers that have already run
        ``simplify_pauliop_terms`` and know ``operators`` is a canonical
        non-empty Pauli string — at most one operator per qubit, no
        identities. The helper only sorts the operators into the
        canonical ``(index, pauli.value)`` order and updates ``_terms``;
        it does NOT re-run ``simplify_pauliop_terms`` or fold any phase,
        so the caller is responsible for passing the fully resolved
        coefficient.

        Args:
            operators (tuple[PauliOperator, ...]): A pre-simplified,
                non-empty Pauli string. Empty tuples must be routed by
                the caller to ``self.constant`` directly.
            coeff (float | complex): The fully resolved coefficient to
                add. Summed with any existing coefficient for the same
                canonicalized term.

        Returns:
            None: This method mutates ``self._terms`` in place.
        """
        sorted_ops = tuple(sorted(operators, key=lambda x: (x.index, x.pauli.value)))
        if sorted_ops in self._terms:
            self._terms[sorted_ops] += coeff
        else:
            self._terms[sorted_ops] = coeff

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
                    h_str += term_str if coeff.real > 0 else "-" + term_str
                else:
                    h_str += f"{coeff}{term_str}"
            else:
                if term_str == "":
                    # This means the term is just a constant. We can skip this.
                    continue
                if abs(coeff) == 1:
                    h_str += "+" + term_str if coeff.real > 0 else "-" + term_str
                else:
                    h_str += (
                        f"+{coeff}" + term_str
                        if coeff.real > 0
                        else f"{coeff}" + term_str
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hamiltonian):
            return False
        return self.terms == other.terms and self.constant == other.constant

    def __iter__(self) -> Iterator[tuple[tuple[PauliOperator, ...], complex]]:
        """Iterate over (operators, coefficient) pairs."""
        return iter(self._terms.items())

    def __len__(self) -> int:
        """Return the number of terms in the Hamiltonian."""
        return len(self._terms)

    def copy(self) -> Hamiltonian:
        """Return an independent copy sharing no mutable state with ``self``.

        Produces a new ``Hamiltonian`` with the same terms, constant,
        and declared ``_num_qubits``.  The underlying ``_terms`` dict
        is fresh, so subsequent ``add_term`` / ``constant`` mutations
        on either instance do not affect the other.  ``PauliOperator``
        instances inside the term tuples are reused — they are
        ``dataclass(frozen=True)`` values and safely shared.

        Returns:
            A shallow-cloned ``Hamiltonian`` instance.

        Example:
            >>> H = Hamiltonian()
            >>> H.add_term((PauliOperator(Pauli.Z, 0),), 1.0)
            >>> H2 = H.copy()
            >>> H2.add_term((PauliOperator(Pauli.X, 1),), 0.5)
            >>> H.num_qubits  # unchanged by H2's mutation
            1
        """
        clone = Hamiltonian(num_qubits=self._num_qubits)
        clone.constant = self.constant
        for operators, coeff in self._terms.items():
            clone.add_term(operators, coeff)
        return clone

    def remap_qubits(self, qubit_map: dict[int, int]) -> Hamiltonian:
        """Remap qubit indices according to the given mapping.

        This is used to translate Pauli indices (logical indices within an
        expval call) to physical qubit indices in the actual quantum circuit.

        Args:
            qubit_map: Mapping from logical index to physical index.
                       e.g., {0: 5, 1: 3} maps logical index 0 → physical qubit 5

        Returns:
            New Hamiltonian with remapped qubit indices.
        """
        if not qubit_map:
            return self

        h = Hamiltonian(num_qubits=self._num_qubits)
        h.constant = self.constant

        for operators, coeff in self._terms.items():
            remapped_ops = tuple(
                PauliOperator(op.pauli, qubit_map.get(op.index, op.index))
                for op in operators
            )
            h.add_term(remapped_ops, coeff)

        return h

    def to_numpy(self) -> np.ndarray:
        """Convert the Hamiltonian to a dense NumPy matrix.

        Qubit 0 is mapped to the least-significant bit of computational-basis
        indices, matching :meth:`qamomile.linalg.HermitianMatrix.to_hamiltonian`.
        The returned array has shape ``(2**n, 2**n)`` where ``n`` is
        :attr:`num_qubits`.
        """
        num_qubits = self.num_qubits
        dim = 1 << num_qubits
        result = np.asarray(self.constant, dtype=np.complex128) * np.eye(
            dim, dtype=np.complex128
        )

        for operators, coeff in self._terms.items():
            factors = [_PAULI_MATRICES[Pauli.I]] * num_qubits
            for op in operators:
                factors[op.index] = _PAULI_MATRICES[op.pauli]

            term = np.array([[1.0 + 0.0j]], dtype=np.complex128)
            for factor in reversed(factors):
                term = np.kron(term, factor)

            result = result + np.asarray(coeff, dtype=np.complex128) * term

        return result

    def __add__(self, other):
        if isinstance(other, Hamiltonian):
            h = Hamiltonian()
            h._terms = self._terms.copy()
            h.constant = self.constant
            for term, coeff in other.terms.items():
                h.add_term(term, coeff)
            h.constant += other.constant

            # Preserve the qubit register from BOTH operands.  The previous
            # logic only kept ``self.num_qubits``, so e.g.
            # ``Hamiltonian.identity(1, num_qubits=2) + Hamiltonian.identity(1, num_qubits=5)``
            # silently lost the right-hand register.
            declared = max(self.num_qubits, other.num_qubits)
            if h.num_qubits < declared:
                h._num_qubits = declared

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

            if not math.isclose(abs(other.constant), 0.0, abs_tol=PAULI_TERM_ZERO_ATOL):
                for terms, coeff1 in self.terms.items():
                    h.add_term(terms, coeff1 * other.constant)

            if not math.isclose(abs(self.constant), 0.0, abs_tol=PAULI_TERM_ZERO_ATOL):
                for terms, coeff2 in other.terms.items():
                    h.add_term(terms, coeff2 * self.constant)

            h.constant += self.constant * other.constant

            # Preserve the qubit register from BOTH operands.  The previous
            # logic only kept ``self.num_qubits``, so e.g.
            # ``Hamiltonian.identity(1, num_qubits=2) * Hamiltonian.identity(1, num_qubits=5)``
            # silently lost the right-hand register.
            declared = max(self.num_qubits, other.num_qubits)
            if h.num_qubits < declared:
                h._num_qubits = declared

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
    if pauli1.index != pauli2.index:
        raise ValueError("Pauli operators act on different qubits.")

    result_pauli, phase = _PAULI_MUL_TABLE[(pauli1.pauli, pauli2.pauli)]
    return PauliOperator(result_pauli, pauli1.index), phase


def simplify_pauliop_terms(
    term: tuple[PauliOperator, ...],
) -> tuple[tuple[PauliOperator, ...], complex]:
    """
    Simplifies a tuple of Pauli operators by combining operators acting on the same qubit.

    This function performs canonicalization by:
    1. Multiplying Pauli operators on the same qubit using the Pauli algebra
    2. Removing identity operators from the result
    3. Tracking the accumulated phase factor from the multiplications

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


def _pauli_strings_anticommute(
    term1: tuple[PauliOperator, ...],
    term2: tuple[PauliOperator, ...],
) -> bool:
    """Decide whether two Pauli strings anticommute.

    Two Pauli strings ``P`` and ``Q`` anticommute iff the number of
    qubits on which both act with different non-identity Pauli
    operators is odd. On every other qubit ``P`` and ``Q`` commute
    trivially (one is identity, or the two Paulis are equal), so the
    overall sign of ``PQ`` versus ``QP`` is ``(-1)`` raised to that
    count.

    Both inputs MUST be sorted by qubit index, free of ``Pauli.I``
    entries, and carry at most one Pauli per qubit. Term tuples stored
    on ``Hamiltonian._terms`` satisfy this contract because
    ``Hamiltonian.add_term`` runs ``simplify_pauliop_terms`` *and then*
    sorts by ``(index, pauli.value)`` before insertion. Note that
    ``simplify_pauliop_terms`` by itself does NOT sort its returned
    tuple — it preserves the insertion order of its internal per-qubit
    dict — so external callers reusing this helper on the raw output of
    ``simplify_pauliop_terms`` must sort the tuple themselves.

    Under the canonical invariant the parity check is a single linear
    merge over the two tuples — no per-call dict / set allocation is
    needed, which matters because this helper is the inner-loop filter
    of ``commutator``. Passing a non-canonical string (unsorted, or with
    explicit ``Pauli.I`` entries) is undefined behavior.

    Args:
        term1 (tuple[PauliOperator, ...]): The first canonical Pauli
            string (one non-identity operator per qubit, sorted by
            ``index``).
        term2 (tuple[PauliOperator, ...]): The second canonical Pauli
            string, in the same canonical form as ``term1``.

    Returns:
        bool: True if the two strings anticommute (``PQ = -QP``),
            False if they commute (``PQ = QP``).

    Example:
        >>> X0 = PauliOperator(Pauli.X, 0)
        >>> Y0 = PauliOperator(Pauli.Y, 0)
        >>> Z1 = PauliOperator(Pauli.Z, 1)
        >>> _pauli_strings_anticommute((X0,), (Y0,))
        True
        >>> _pauli_strings_anticommute((X0,), (Z1,))
        False
    """
    i, j = 0, 0
    n1, n2 = len(term1), len(term2)
    anticommute_count = 0
    while i < n1 and j < n2:
        op1 = term1[i]
        op2 = term2[j]
        if op1.index < op2.index:
            i += 1
        elif op1.index > op2.index:
            j += 1
        else:
            # Same qubit; canonical form rules out identities, so two
            # distinct Paulis anticommute on this qubit.
            if op1.pauli != op2.pauli:
                anticommute_count += 1
            i += 1
            j += 1

    return anticommute_count % 2 == 1


def commutator(a: Hamiltonian, b: Hamiltonian) -> Hamiltonian:
    """Compute the commutator ``[A, B] = A B - B A`` of two Hamiltonians.

    Iterates over the Pauli-string terms of `a` and `b` once and uses
    the fact that two Pauli strings either commute (``[P, Q] = 0``) or
    anticommute (``[P, Q] = 2 P Q``). Commuting pairs are skipped
    entirely, so for sparse or nearly-commuting Hamiltonians this is
    cheaper than expanding ``a * b - b * a`` and then cancelling.
    The asymptotic cost is still O(|a| * |b|) Pauli-string pairs in
    the worst case where every pair anticommutes.

    The constant parts of `a` and `b` commute with every Pauli string
    and with each other, so they contribute nothing to the commutator
    and are ignored.

    Args:
        a (Hamiltonian): The left operand of the commutator.
        b (Hamiltonian): The right operand of the commutator.

    Returns:
        Hamiltonian: A new Hamiltonian equal to ``a * b - b * a``.
            Its `num_qubits` is the maximum of `a.num_qubits` and
            `b.num_qubits` so the qubit register is preserved even
            when the commutator collapses to zero on a subset of
            qubits.

    Example:
        >>> from qamomile.observable.hamiltonian import X, Y, commutator
        >>> commutator(X(0), Y(0))
        Hamiltonian((Z0,): 2j)
        >>> commutator(X(0), X(1))
        Hamiltonian()
    """
    result = Hamiltonian(num_qubits=max(a.num_qubits, b.num_qubits))

    for term_a, coeff_a in a.terms.items():
        for term_b, coeff_b in b.terms.items():
            if not _pauli_strings_anticommute(term_a, term_b):
                continue
            product, phase = simplify_pauliop_terms(term_a + term_b)
            scaled = 2.0 * phase * coeff_a * coeff_b
            if product:
                # ``product`` is already simplified, so route around
                # the second simplify pass inside ``add_term``.
                result._add_simplified_term(product, scaled)
            else:
                result.constant += scaled

    return result
