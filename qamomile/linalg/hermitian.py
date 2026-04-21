"""Dense Hermitian matrix wrapper with Pauli decomposition via FWHT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qamomile._utils import is_close_zero
from qamomile.linalg._fwht import fwht_pauli_coefficients, is_power_of_two

if TYPE_CHECKING:
    from qamomile.observable import Hamiltonian


class HermitianMatrix:
    """Dense Hermitian operator on ``n`` qubits (a ``2**n x 2**n`` matrix).

    Wraps a NumPy array and validates on construction that it is 2D square,
    has a power-of-two dimension, and is Hermitian up to ``atol``. The
    Hermitian check can be skipped via ``validate=False`` for internal use
    (for example, when the caller already knows a result is Hermitian by
    construction).

    The main entry point for downstream quantum code is
    :meth:`to_hamiltonian`, which returns the exact Pauli decomposition as
    a :class:`qamomile.observable.Hamiltonian` via the Fast Walsh-Hadamard
    Transform in ``O(n * 4**n)`` time.

    Example:
        >>> import numpy as np
        >>> from qamomile.linalg import HermitianMatrix
        >>> X = np.array([[0, 1], [1, 0]], dtype=complex)
        >>> Z = np.array([[1, 0], [0, -1]], dtype=complex)
        >>> H = HermitianMatrix(np.kron(X, Z) + 0.5 * np.kron(Z, np.eye(2)))
        >>> H.num_qubits
        2
        >>> ham = H.to_hamiltonian()
    """

    def __init__(
        self,
        matrix: np.ndarray,
        *,
        validate: bool = True,
        atol: float = 1e-10,
    ) -> None:
        m = np.asarray(matrix, dtype=np.complex128)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(
                f"HermitianMatrix requires a 2D square array; got shape {m.shape}."
            )
        if not is_power_of_two(m.shape[0]):
            raise ValueError(
                f"HermitianMatrix dimension must be a power of 2; got {m.shape[0]}."
            )
        if validate and not np.allclose(m, m.conj().T, atol=atol):
            raise ValueError(f"Input matrix is not Hermitian (atol={atol}).")
        self._matrix = m
        self._num_qubits = m.shape[0].bit_length() - 1

    @property
    def matrix(self) -> np.ndarray:
        """Return a read-only view of the underlying array."""
        view = self._matrix.view()
        view.flags.writeable = False
        return view

    @property
    def num_qubits(self) -> int:
        """Number of qubits (``log2`` of the matrix dimension).

        Returns:
            The qubit count ``n`` such that the matrix is ``2**n x 2**n``.
        """
        return self._num_qubits

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the underlying matrix as ``(2**n, 2**n)``.

        Returns:
            A 2-tuple of equal integers.
        """
        return self._matrix.shape

    def to_hamiltonian(self, *, tol: float = 1e-12) -> Hamiltonian:
        """Return the exact Pauli decomposition as a ``Hamiltonian``.

        Uses FWHT internally. Pauli terms with ``|coeff| <= tol`` are
        dropped. The all-identity component is accumulated into
        :attr:`qamomile.observable.Hamiltonian.constant`. Qubit 0
        corresponds to the least-significant bit of the matrix's
        computational-basis indices.

        Args:
            tol: Absolute tolerance below which a Pauli coefficient is
                treated as zero and omitted from the result.

        Returns:
            A :class:`~qamomile.observable.Hamiltonian` containing the
            Pauli decomposition.
        """
        from qamomile.observable import Hamiltonian, Pauli, PauliOperator

        pauli_for_bits = {
            (1, 0): Pauli.X,
            (0, 1): Pauli.Z,
            (1, 1): Pauli.Y,
        }

        coeffs, n = fwht_pauli_coefficients(self._matrix)
        h = Hamiltonian(num_qubits=n)
        dim = 1 << n

        for x_mask in range(dim):
            for z_mask in range(dim):
                c = float(coeffs[x_mask, z_mask])
                if abs(c) <= tol:
                    continue
                ops: list[PauliOperator] = []
                for k in range(n):
                    x_bit = (x_mask >> k) & 1
                    z_bit = (z_mask >> k) & 1
                    if x_bit == 0 and z_bit == 0:
                        continue
                    ops.append(PauliOperator(pauli_for_bits[(x_bit, z_bit)], k))
                if not ops:
                    h.constant += c
                else:
                    h.add_term(tuple(ops), c)

        return h

    def __add__(self, other: object) -> HermitianMatrix:
        """Element-wise addition of two Hermitian matrices.

        Args:
            other: Another ``HermitianMatrix`` of the same shape.

        Returns:
            A new ``HermitianMatrix`` equal to ``self + other``.

        Raises:
            ValueError: If the shapes do not match.
        """
        if not isinstance(other, HermitianMatrix):
            return NotImplemented
        if self._matrix.shape != other._matrix.shape:
            raise ValueError(
                f"Shape mismatch: {self._matrix.shape} vs {other._matrix.shape}."
            )
        return HermitianMatrix(self._matrix + other._matrix, validate=False)

    def __sub__(self, other: object) -> HermitianMatrix:
        """Element-wise subtraction of two Hermitian matrices.

        Args:
            other: Another ``HermitianMatrix`` of the same shape.

        Returns:
            A new ``HermitianMatrix`` equal to ``self - other``.

        Raises:
            ValueError: If the shapes do not match.
        """
        if not isinstance(other, HermitianMatrix):
            return NotImplemented
        if self._matrix.shape != other._matrix.shape:
            raise ValueError(
                f"Shape mismatch: {self._matrix.shape} vs {other._matrix.shape}."
            )
        return HermitianMatrix(self._matrix - other._matrix, validate=False)

    def __mul__(self, scalar: int | float | complex) -> HermitianMatrix:
        """Multiply by a real scalar.

        Args:
            scalar: A real number. A ``complex`` value whose imaginary
                part is close to zero is also accepted.

        Returns:
            A new ``HermitianMatrix`` scaled by *scalar*.

        Raises:
            TypeError: If *scalar* has a nonzero imaginary part.
        """
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float, complex)):
            return NotImplemented
        if isinstance(scalar, complex) and not is_close_zero(scalar.imag):
            raise TypeError(
                "HermitianMatrix only supports multiplication by real scalars "
                "(a nonzero imaginary scalar would break the Hermitian property)."
            )
        real_scalar = float(scalar.real if isinstance(scalar, complex) else scalar)
        return HermitianMatrix(self._matrix * real_scalar, validate=False)

    __rmul__ = __mul__

    def __truediv__(self, scalar: int | float | complex) -> HermitianMatrix:
        """Divide by a real scalar.

        Args:
            scalar: A real number. A ``complex`` value whose imaginary
                part is close to zero is also accepted.

        Returns:
            A new ``HermitianMatrix`` divided by *scalar*.

        Raises:
            TypeError: If *scalar* has a nonzero imaginary part.
            ZeroDivisionError: If *scalar* is zero or within the
                :func:`~qamomile._utils.is_close_zero` tolerance of zero.
        """
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float, complex)):
            return NotImplemented
        if isinstance(scalar, complex) and not is_close_zero(scalar.imag):
            raise TypeError(
                "HermitianMatrix only supports division by real scalars "
                "(a nonzero imaginary scalar would break the Hermitian property)."
            )
        real_scalar = float(scalar.real if isinstance(scalar, complex) else scalar)
        if is_close_zero(real_scalar):
            raise ZeroDivisionError(
                "HermitianMatrix division by a scalar too close to zero."
            )
        return HermitianMatrix(self._matrix / real_scalar, validate=False)

    def __neg__(self) -> HermitianMatrix:
        """Negate the matrix element-wise.

        Returns:
            A new ``HermitianMatrix`` equal to ``-self``.
        """
        return HermitianMatrix(-self._matrix, validate=False)

    def __eq__(self, other: object) -> bool:
        """Element-wise exact equality.

        Args:
            other: Another ``HermitianMatrix``.

        Returns:
            ``True`` if both matrices are element-wise identical.
        """
        if not isinstance(other, HermitianMatrix):
            return NotImplemented
        return np.array_equal(self._matrix, other._matrix)

    def __repr__(self) -> str:
        return f"HermitianMatrix(num_qubits={self._num_qubits}, shape={self.shape})"
