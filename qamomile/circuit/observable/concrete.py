"""Concrete Hamiltonian representation after graph evaluation.

This module provides ConcreteHamiltonian, which represents a Hamiltonian
as a sum of Pauli strings with complex coefficients. This is the evaluated
form of the HamiltonianExpr operation graph.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Tuple, Iterator

from qamomile.circuit.ir.types.hamiltonian import PauliKind


# Type alias for a Pauli string: tuple of (PauliKind, qubit_index) pairs
# Sorted by qubit index for canonicalization
PauliString = Tuple[Tuple[PauliKind, int], ...]


# Pauli multiplication table with phase factors
# (P1, P2) -> (result_pauli, phase_factor)
# Phase factors: 1, -1, 1j, -1j
_PAULI_MUL_TABLE: Dict[Tuple[PauliKind, PauliKind], Tuple[PauliKind, complex]] = {
    # I * anything = anything
    (PauliKind.I, PauliKind.I): (PauliKind.I, 1),
    (PauliKind.I, PauliKind.X): (PauliKind.X, 1),
    (PauliKind.I, PauliKind.Y): (PauliKind.Y, 1),
    (PauliKind.I, PauliKind.Z): (PauliKind.Z, 1),
    # anything * I = anything
    (PauliKind.X, PauliKind.I): (PauliKind.X, 1),
    (PauliKind.Y, PauliKind.I): (PauliKind.Y, 1),
    (PauliKind.Z, PauliKind.I): (PauliKind.Z, 1),
    # XX = YY = ZZ = I
    (PauliKind.X, PauliKind.X): (PauliKind.I, 1),
    (PauliKind.Y, PauliKind.Y): (PauliKind.I, 1),
    (PauliKind.Z, PauliKind.Z): (PauliKind.I, 1),
    # XY = iZ, YX = -iZ
    (PauliKind.X, PauliKind.Y): (PauliKind.Z, 1j),
    (PauliKind.Y, PauliKind.X): (PauliKind.Z, -1j),
    # YZ = iX, ZY = -iX
    (PauliKind.Y, PauliKind.Z): (PauliKind.X, 1j),
    (PauliKind.Z, PauliKind.Y): (PauliKind.X, -1j),
    # ZX = iY, XZ = -iY
    (PauliKind.Z, PauliKind.X): (PauliKind.Y, 1j),
    (PauliKind.X, PauliKind.Z): (PauliKind.Y, -1j),
}


def _multiply_paulis_same_qubit(
    p1: PauliKind, p2: PauliKind
) -> Tuple[PauliKind, complex]:
    """Multiply two Paulis acting on the same qubit.

    Returns:
        (result_pauli, phase_factor)
    """
    return _PAULI_MUL_TABLE[(p1, p2)]


def _canonicalize_pauli_string(
    paulis: list[tuple[PauliKind, int]]
) -> Tuple[PauliString, complex]:
    """Canonicalize a Pauli string by sorting and applying Pauli algebra.

    Args:
        paulis: List of (PauliKind, qubit_index) pairs

    Returns:
        (canonical_pauli_string, accumulated_phase)
    """
    if not paulis:
        return (), 1

    # Group by qubit index
    by_qubit: Dict[int, PauliKind] = {}
    phase = complex(1)

    for pauli_kind, qubit_idx in paulis:
        if qubit_idx in by_qubit:
            # Apply Pauli algebra: multiply with existing
            existing = by_qubit[qubit_idx]
            result_pauli, mul_phase = _multiply_paulis_same_qubit(existing, pauli_kind)
            phase *= mul_phase
            if result_pauli == PauliKind.I:
                del by_qubit[qubit_idx]  # Identity cancels
            else:
                by_qubit[qubit_idx] = result_pauli
        else:
            if pauli_kind != PauliKind.I:  # Skip identity
                by_qubit[qubit_idx] = pauli_kind

    # Sort by qubit index to create canonical form
    result = tuple(sorted(by_qubit.items(), key=lambda x: x[0]))
    # Convert to (PauliKind, qubit_idx) format
    canonical = tuple((pauli, idx) for idx, pauli in result)
    return canonical, phase


@dataclasses.dataclass
class ConcreteHamiltonian:
    """Concrete Hamiltonian after graph evaluation.

    A Hamiltonian is represented as a sum of Pauli strings with complex coefficients:
        H = sum_i coeff_i * P_i

    where each P_i is a tensor product of Pauli operators on specific qubits.

    The `terms` dictionary maps Pauli strings (as tuples) to their coefficients.
    An empty Pauli string () represents the identity term (constant).

    Attributes:
        terms: Dict mapping PauliString -> coefficient
        _num_qubits: Cached number of qubits (computed lazily)
    """

    terms: Dict[PauliString, complex] = dataclasses.field(default_factory=dict)
    _num_qubits: int | None = dataclasses.field(default=None, repr=False)

    @classmethod
    def zero(cls) -> "ConcreteHamiltonian":
        """Create a zero Hamiltonian."""
        return cls()

    @classmethod
    def identity(cls, coeff: complex = 1.0) -> "ConcreteHamiltonian":
        """Create a scalar times identity Hamiltonian."""
        if coeff == 0:
            return cls()
        return cls(terms={(): coeff})

    @classmethod
    def single_pauli(
        cls, pauli_kind: PauliKind, qubit_idx: int, coeff: complex = 1.0
    ) -> "ConcreteHamiltonian":
        """Create a single Pauli term Hamiltonian."""
        if coeff == 0:
            return cls()
        if pauli_kind == PauliKind.I:
            return cls(terms={(): coeff})
        return cls(terms={((pauli_kind, qubit_idx),): coeff})

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits this Hamiltonian acts on.

        Returns the maximum qubit index + 1 across all terms.
        """
        if self._num_qubits is not None:
            return self._num_qubits

        max_idx = -1
        for pauli_string in self.terms:
            for _, qubit_idx in pauli_string:
                max_idx = max(max_idx, qubit_idx)

        self._num_qubits = max_idx + 1 if max_idx >= 0 else 0
        return self._num_qubits

    @property
    def constant(self) -> complex:
        """Return the constant (identity) term coefficient."""
        return self.terms.get((), 0)

    @constant.setter
    def constant(self, value: complex) -> None:
        """Set the constant (identity) term coefficient."""
        if value == 0:
            self.terms.pop((), None)
        else:
            self.terms[()] = value

    def add_term(
        self,
        paulis: list[tuple[PauliKind, int]] | PauliString,
        coeff: complex,
    ) -> "ConcreteHamiltonian":
        """Add a Pauli term to the Hamiltonian (returns new Hamiltonian).

        Args:
            paulis: List of (PauliKind, qubit_index) pairs
            coeff: Coefficient for the term

        Returns:
            New ConcreteHamiltonian with the term added
        """
        if coeff == 0:
            return self

        # Canonicalize the Pauli string
        canonical, phase = _canonicalize_pauli_string(list(paulis))
        final_coeff = coeff * phase

        # Add to terms
        new_terms = dict(self.terms)
        if canonical in new_terms:
            new_terms[canonical] += final_coeff
            if abs(new_terms[canonical]) < 1e-15:
                del new_terms[canonical]
        else:
            new_terms[canonical] = final_coeff

        return ConcreteHamiltonian(terms=new_terms)

    def __add__(self, other: "ConcreteHamiltonian") -> "ConcreteHamiltonian":
        """Add two Hamiltonians."""
        new_terms = dict(self.terms)

        for pauli_string, coeff in other.terms.items():
            if pauli_string in new_terms:
                new_terms[pauli_string] += coeff
                if abs(new_terms[pauli_string]) < 1e-15:
                    del new_terms[pauli_string]
            else:
                new_terms[pauli_string] = coeff

        return ConcreteHamiltonian(terms=new_terms)

    def __neg__(self) -> "ConcreteHamiltonian":
        """Negate the Hamiltonian."""
        new_terms = {ps: -coeff for ps, coeff in self.terms.items()}
        return ConcreteHamiltonian(terms=new_terms)

    def __sub__(self, other: "ConcreteHamiltonian") -> "ConcreteHamiltonian":
        """Subtract two Hamiltonians."""
        return self + (-other)

    def __mul__(
        self, other: "ConcreteHamiltonian | complex | float | int"
    ) -> "ConcreteHamiltonian":
        """Multiply Hamiltonian by scalar or another Hamiltonian."""
        if isinstance(other, (complex, float, int)):
            # Scalar multiplication
            if other == 0:
                return ConcreteHamiltonian()
            new_terms = {ps: coeff * other for ps, coeff in self.terms.items()}
            return ConcreteHamiltonian(terms=new_terms)

        if isinstance(other, ConcreteHamiltonian):
            # Hamiltonian multiplication (tensor product)
            result = ConcreteHamiltonian()

            for ps1, coeff1 in self.terms.items():
                for ps2, coeff2 in other.terms.items():
                    # Combine Pauli strings
                    combined = list(ps1) + list(ps2)
                    result = result.add_term(combined, coeff1 * coeff2)

            return result

        return NotImplemented

    def __rmul__(
        self, other: "complex | float | int"
    ) -> "ConcreteHamiltonian":
        """Reverse scalar multiplication."""
        return self.__mul__(other)

    def simplify(self, tol: float = 1e-15) -> "ConcreteHamiltonian":
        """Remove terms with coefficients below tolerance.

        Args:
            tol: Coefficient magnitude below which terms are removed

        Returns:
            Simplified ConcreteHamiltonian
        """
        new_terms = {
            ps: coeff for ps, coeff in self.terms.items() if abs(coeff) >= tol
        }
        return ConcreteHamiltonian(terms=new_terms)

    def __iter__(self) -> Iterator[Tuple[PauliString, complex]]:
        """Iterate over (pauli_string, coefficient) pairs."""
        return iter(self.terms.items())

    def __len__(self) -> int:
        """Return the number of terms."""
        return len(self.terms)

    def __repr__(self) -> str:
        """String representation."""
        if not self.terms:
            return "ConcreteHamiltonian(0)"

        term_strs = []
        for pauli_string, coeff in sorted(self.terms.items(), key=lambda x: len(x[0])):
            if not pauli_string:
                term_strs.append(f"{coeff:.4g}")
            else:
                pauli_str = " ".join(
                    f"{p.name}({q})" for p, q in pauli_string
                )
                term_strs.append(f"{coeff:.4g} * {pauli_str}")

        return f"ConcreteHamiltonian({' + '.join(term_strs)})"

    def to_sparse_dict(self) -> Dict[str, complex]:
        """Convert to a sparse dictionary representation.

        Returns:
            Dict mapping Pauli string notation (e.g., "IXYZ") to coefficients
        """
        result = {}
        n = self.num_qubits

        for pauli_string, coeff in self.terms.items():
            # Build Pauli string in standard notation
            paulis = ["I"] * n
            for pauli_kind, qubit_idx in pauli_string:
                paulis[qubit_idx] = pauli_kind.name

            key = "".join(paulis)
            result[key] = coeff

        return result

    def remap_qubits(self, qubit_map: Dict[int, int]) -> "ConcreteHamiltonian":
        """Remap qubit indices according to the given mapping.

        This is used to translate Pauli indices (logical indices within an
        expval call) to physical qubit indices in the actual quantum circuit.

        Args:
            qubit_map: Mapping from logical index to physical index.
                       e.g., {0: 5, 1: 3} maps logical index 0 → physical qubit 5

        Returns:
            New ConcreteHamiltonian with remapped qubit indices.
        """
        if not qubit_map:
            return self

        new_terms: Dict[PauliString, complex] = {}
        for pauli_string, coeff in self.terms.items():
            # Remap each qubit index in the Pauli string
            new_string = tuple(
                (kind, qubit_map.get(idx, idx)) for kind, idx in pauli_string
            )
            # Canonicalize to handle any reordering
            canonical, phase = _canonicalize_pauli_string(list(new_string))
            final_coeff = coeff * phase

            if canonical in new_terms:
                new_terms[canonical] += final_coeff
                if abs(new_terms[canonical]) < 1e-15:
                    del new_terms[canonical]
            else:
                new_terms[canonical] = final_coeff

        return ConcreteHamiltonian(terms=new_terms)
