"""Represent general complex matrices as linear combinations of Paulis."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from numpy.typing import ArrayLike

from qamomile.linalg._fwht import fwht_complex_pauli_coefficients
from qamomile.linalg._validation import (
    coerce_finite_complex_scalar,
    coerce_nonnegative_finite_real,
)
from qamomile.observable import Pauli, PauliOperator


@dataclasses.dataclass(frozen=True, slots=True)
class PauliLCUTerm:
    """Store one nonzero complex coefficient and one sparse Pauli word.

    The empty operator tuple denotes the identity. Non-identity operators are
    canonicalized into increasing qubit-index order.

    Args:
        coefficient (complex): Finite, nonzero coefficient multiplying the
            Pauli word.
        operators (tuple[PauliOperator, ...]): Sparse Pauli word. Identity
            operators must be omitted; use an empty tuple for the all-identity
            word.

    Raises:
        TypeError: If the coefficient is not a numeric scalar, an operator is
            not a ``PauliOperator``, its Pauli value is invalid, or its index
            is not an integer.
        ValueError: If the coefficient is zero or non-finite, an identity
            operator is stored explicitly, or multiple operators address the
            same qubit.
    """

    coefficient: complex
    operators: tuple[PauliOperator, ...]

    def __post_init__(self) -> None:
        """Validate and canonicalize the immutable term.

        Raises:
            TypeError: If the coefficient is not a numeric scalar, an operator
                is not a ``PauliOperator``, its Pauli value is invalid, or its
                index is not an integer.
            ValueError: If the coefficient or Pauli word is invalid.
        """
        coefficient = coerce_finite_complex_scalar(
            self.coefficient,
            "PauliLCUTerm coefficient",
        )
        if not coefficient:
            raise ValueError("PauliLCUTerm coefficient must be nonzero.")

        operators = tuple(self.operators)
        for operator in operators:
            if not isinstance(operator, PauliOperator):
                raise TypeError(
                    "PauliLCUTerm operators must all be PauliOperator values."
                )
            if not isinstance(operator.pauli, Pauli):
                raise TypeError("PauliOperator.pauli must be a Pauli value.")
            if isinstance(operator.index, (bool, np.bool_)) or not isinstance(
                operator.index, (int, np.integer)
            ):
                raise TypeError("Pauli operator indices must be integers.")
            if operator.index < 0:
                raise ValueError("Pauli operator indices must be non-negative.")
            if operator.pauli is Pauli.I:
                raise ValueError(
                    "Identity operators must be omitted from a sparse Pauli word."
                )

        canonical = tuple(
            sorted(
                (
                    PauliOperator(operator.pauli, int(operator.index))
                    for operator in operators
                ),
                key=lambda operator: operator.index,
            )
        )
        indices = [int(operator.index) for operator in canonical]
        if len(indices) != len(set(indices)):
            raise ValueError("A Pauli word cannot address one qubit more than once.")

        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "operators", canonical)


@dataclasses.dataclass(frozen=True, slots=True)
class PauliLCU:
    """Store an immutable Pauli linear-combination decomposition.

    Args:
        num_qubits (int): Number of system qubits represented by the Pauli
            words. Zero is valid for a scalar ``1 x 1`` matrix.
        terms (tuple[PauliLCUTerm, ...]): Ordered nonzero Pauli terms. The
            empty tuple represents the zero operator.
        truncation_error_bound (float): Upper bound on the spectral-norm error
            caused only by coefficients intentionally removed by
            :meth:`from_matrix`. Defaults to ``0.0``. It does not include
            floating-point FWHT roundoff.

    Raises:
        TypeError: If ``num_qubits``, ``terms``, or
            ``truncation_error_bound`` has an invalid type.
        ValueError: If a term addresses a qubit outside the system, duplicate
            Pauli words are present, the LCU normalization exceeds the finite
            float range, or the error bound is negative or non-finite.
    """

    num_qubits: int
    terms: tuple[PauliLCUTerm, ...]
    truncation_error_bound: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalize the immutable decomposition.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If a term or the truncation error bound is invalid.
        """
        if isinstance(self.num_qubits, (bool, np.bool_)) or not isinstance(
            self.num_qubits, (int, np.integer)
        ):
            raise TypeError("PauliLCU num_qubits must be an integer.")
        num_qubits = int(self.num_qubits)
        if num_qubits < 0:
            raise ValueError("PauliLCU num_qubits must be non-negative.")

        try:
            terms = tuple(self.terms)
        except TypeError as exc:
            raise TypeError("PauliLCU terms must be an iterable of terms.") from exc
        keys: set[tuple[tuple[int, Pauli], ...]] = set()
        for term in terms:
            if not isinstance(term, PauliLCUTerm):
                raise TypeError("PauliLCU terms must all be PauliLCUTerm values.")
            if any(operator.index >= num_qubits for operator in term.operators):
                raise ValueError("PauliLCU term addresses a qubit outside num_qubits.")
            key = tuple(
                (int(operator.index), operator.pauli) for operator in term.operators
            )
            if key in keys:
                raise ValueError("PauliLCU cannot contain duplicate Pauli words.")
            keys.add(key)

        try:
            alpha = math.fsum(abs(term.coefficient) for term in terms)
        except OverflowError as exc:
            raise ValueError(
                "PauliLCU normalization exceeds the finite float range."
            ) from exc
        if not math.isfinite(alpha):
            raise ValueError("PauliLCU normalization must be finite.")

        error_bound = coerce_nonnegative_finite_real(
            self.truncation_error_bound,
            "truncation_error_bound",
        )

        object.__setattr__(self, "num_qubits", num_qubits)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "truncation_error_bound", error_bound)

    @classmethod
    def from_matrix(
        cls,
        matrix: ArrayLike,
        *,
        atol: float = 0.0,
    ) -> PauliLCU:
        """Decompose a general complex matrix into the Pauli basis.

        Qubit zero is the least-significant bit of the matrix's computational
        basis index. Coefficients whose magnitude is at most ``atol`` are
        omitted, and their absolute values are accumulated into
        :attr:`truncation_error_bound`.

        Args:
            matrix (ArrayLike): Finite square matrix with dimension ``2**n``.
                Hermiticity is not required.
            atol (float): Non-negative absolute coefficient threshold.
                Defaults to ``0.0``, which removes exact zeros only.

        Returns:
            PauliLCU: Immutable ordered Pauli decomposition.

        Raises:
            TypeError: If ``atol`` is not a real number.
            ValueError: If ``atol`` is negative or non-finite, or ``matrix``
                is not a finite square numeric matrix with power-of-two
                dimension, or the resulting LCU normalization exceeds the
                finite float range.
        """
        threshold = coerce_nonnegative_finite_real(atol, "atol")
        coeffs, num_qubits = fwht_complex_pauli_coefficients(matrix)
        dim = 1 << num_qubits
        pauli_for_bits = {
            (1, 0): Pauli.X,
            (0, 1): Pauli.Z,
            (1, 1): Pauli.Y,
        }

        flat_coeffs = coeffs.reshape(-1)
        if threshold:
            with np.errstate(over="ignore"):
                # ``np.hypot`` matches Python's prior ``abs(complex)``
                # threshold decisions while retaining vectorized discovery.
                magnitudes = np.hypot(flat_coeffs.real, flat_coeffs.imag)
            retained_indices = np.flatnonzero(magnitudes > threshold)
            dropped_magnitudes = magnitudes[
                (magnitudes > 0.0) & (magnitudes <= threshold)
            ]
        else:
            retained_indices = np.flatnonzero(flat_coeffs)
            dropped_magnitudes = np.empty(0, dtype=np.float64)

        terms: list[PauliLCUTerm] = []
        for flat_index in retained_indices:
            x_mask, z_mask = divmod(int(flat_index), dim)
            coefficient = complex(flat_coeffs[flat_index])
            operators = tuple(
                PauliOperator(
                    pauli_for_bits[((x_mask >> qubit) & 1, (z_mask >> qubit) & 1)],
                    qubit,
                )
                for qubit in range(num_qubits)
                if ((x_mask >> qubit) & 1, (z_mask >> qubit) & 1) != (0, 0)
            )
            terms.append(
                PauliLCUTerm(
                    coefficient=coefficient,
                    operators=operators,
                )
            )

        try:
            truncation_error_bound = math.fsum(
                float(magnitude) for magnitude in dropped_magnitudes
            )
        except OverflowError as exc:
            raise ValueError(
                "PauliLCU truncation error bound exceeds the finite float range."
            ) from exc

        return cls(
            num_qubits=num_qubits,
            terms=tuple(terms),
            truncation_error_bound=truncation_error_bound,
        )

    @property
    def alpha(self) -> float:
        """Return the LCU one-norm normalization.

        Returns:
            float: Stable sum ``sum(abs(term.coefficient))`` over retained
                terms. The zero operator has normalization ``0.0``.
        """
        return math.fsum(abs(term.coefficient) for term in self.terms)

    @property
    def num_terms(self) -> int:
        """Return the number of retained Pauli terms.

        Returns:
            int: Number of nonzero terms.
        """
        return len(self.terms)


__all__ = ["PauliLCU", "PauliLCUTerm"]
