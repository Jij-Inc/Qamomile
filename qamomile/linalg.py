"""Classical linear algebra at the boundary of Qamomile programs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from types import NotImplementedType
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from ._native import compiler as native
from .observable import (
    Hamiltonian,
    Pauli,
    _hamiltonian_from_native_header,
)

Samples: TypeAlias = Sequence[Sequence[int]] | npt.NDArray[np.integer[Any]]
Basis: TypeAlias = Pauli | int
Bases: TypeAlias = Sequence[Basis] | Sequence[Sequence[Basis]] | npt.NDArray[Any]


class HermitianMatrix:
    """Immutable dense Hermitian operator on a power-of-two Hilbert space.

    Construction converts array-like numeric input to owned ``complex128``
    storage. Rust validates its finite square shape and proves Hermiticity
    using only the requested absolute tolerance. The main
    :meth:`to_hamiltonian` operation performs the Pauli FWHT in Rust and
    returns the same canonical :class:`Hamiltonian` used by every target.

    Args:
        matrix (numpy.typing.ArrayLike): Dense matrix-like numeric input.
        atol (float): Non-negative absolute Hermitian tolerance.

    Raises:
        TypeError: If the input cannot be converted to complex numeric data or
            the required native matrix boundary is unavailable.
        ValueError: If the matrix shape, values, tolerance, or Hermiticity is
            invalid, or the exponential decomposition budget is exceeded.
    """

    __slots__ = ("_entries", "_matrix", "_num_qubits")

    def __init__(
        self,
        matrix: npt.ArrayLike,
        *,
        atol: float = 1e-10,
    ) -> None:
        """Initialize an immutable Hermitian matrix.

        Args:
            matrix (numpy.typing.ArrayLike): Dense square matrix whose dimension
                is a power of two.
            atol (float): Nonnegative absolute tolerance used to verify
                Hermiticity.

        Raises:
            TypeError: If ``matrix`` cannot be converted to complex numeric
                data or the native matrix boundary is unavailable.
            ValueError: If the shape, values, tolerance, or Hermiticity is
                invalid.
        """

        converted, entries, num_qubits = _validated_dense_storage(
            matrix,
            require_hermitian=True,
            atol=atol,
        )
        self._matrix = converted
        self._entries = entries
        self._num_qubits = num_qubits

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """Return a read-only view of the owned dense matrix."""

        view = self._matrix.view()
        view.flags.writeable = False
        return view

    @property
    def num_qubits(self) -> int:
        """Return the logical width ``log2(matrix.shape[0])``."""

        return self._num_qubits

    @property
    def shape(self) -> tuple[int, int]:
        """Return the square matrix shape."""

        rows, columns = self._matrix.shape
        return int(rows), int(columns)

    def to_hamiltonian(self, *, tol: float = 1e-12) -> Hamiltonian:
        """Return the canonical real Pauli decomposition.

        Qubit zero is the least-significant bit of each computational-basis
        matrix index. Coefficients whose absolute value is at most ``tol`` are
        omitted, while the identity coefficient becomes ``constant``.

        Args:
            tol (float): Non-negative finite absolute truncation threshold.

        Returns:
            Hamiltonian: Compiler-normalized Pauli sum.

        Raises:
            TypeError: If the required native decomposition is unavailable or
                violates its result ABI.
            ValueError: If the tolerance is invalid or the matrix has a
                significant anti-Hermitian component.
        """

        decompose = getattr(native, "decompose_hermitian_matrix", None)
        if not callable(decompose):
            raise TypeError("Q7400: native dense-matrix algebra is unavailable")
        return _hamiltonian_from_native_header(
            decompose(self.shape, self._entries, tol)
        )

    def __add__(self, other: object) -> HermitianMatrix:
        """Add a same-shaped Hermitian matrix."""

        if not isinstance(other, HermitianMatrix):
            return NotImplemented
        add = getattr(native, "add_scaled_hermitian_matrices", None)
        if not callable(add):
            raise TypeError("Q7400: native dense-matrix algebra is unavailable")
        return _hermitian_matrix_from_native(
            self.shape,
            add(self.shape, self._entries, other.shape, other._entries, 1.0),
        )

    def __sub__(self, other: object) -> HermitianMatrix:
        """Subtract a same-shaped Hermitian matrix."""

        if not isinstance(other, HermitianMatrix):
            return NotImplemented
        add = getattr(native, "add_scaled_hermitian_matrices", None)
        if not callable(add):
            raise TypeError("Q7400: native dense-matrix algebra is unavailable")
        return _hermitian_matrix_from_native(
            self.shape,
            add(self.shape, self._entries, other.shape, other._entries, -1.0),
        )

    def __mul__(self, scalar: object) -> HermitianMatrix:
        """Multiply by a finite real scalar."""

        normalized = _real_matrix_scalar(scalar)
        if normalized is NotImplemented:
            return NotImplemented
        scale = getattr(native, "scale_hermitian_matrix", None)
        if not callable(scale):
            raise TypeError("Q7400: native dense-matrix algebra is unavailable")
        return _hermitian_matrix_from_native(
            self.shape,
            scale(self.shape, self._entries, normalized, False),
        )

    __rmul__ = __mul__

    def __truediv__(self, scalar: object) -> HermitianMatrix:
        """Divide by a finite nonzero real scalar."""

        normalized = _real_matrix_scalar(scalar)
        if normalized is NotImplemented:
            return NotImplemented
        scale = getattr(native, "scale_hermitian_matrix", None)
        if not callable(scale):
            raise TypeError("Q7400: native dense-matrix algebra is unavailable")
        return _hermitian_matrix_from_native(
            self.shape,
            scale(self.shape, self._entries, normalized, True),
        )

    def __neg__(self) -> HermitianMatrix:
        """Negate every matrix entry."""

        scale = getattr(native, "scale_hermitian_matrix", None)
        if not callable(scale):
            raise TypeError("Q7400: native dense-matrix algebra is unavailable")
        return _hermitian_matrix_from_native(
            self.shape,
            scale(self.shape, self._entries, -1.0, False),
        )

    def __eq__(self, other: object) -> bool:
        """Return exact element-wise equality for another wrapper."""

        if not isinstance(other, HermitianMatrix):
            return NotImplemented
        return bool(np.array_equal(self._matrix, other._matrix))

    def __repr__(self) -> str:
        """Return a compact shape-oriented representation."""

        return f"HermitianMatrix(num_qubits={self.num_qubits}, shape={self.shape})"


@dataclass(frozen=True, slots=True, init=False)
class PauliDecomposition:
    """Canonical complex Pauli expansion of one dense matrix.

    Args:
        operator (Hamiltonian): Existing canonical Pauli-sum value. It may be
            non-Hermitian when the source matrix is a general linear operator.
        truncation_error_bound (float): Conservative operator-norm bound from
            coefficients deliberately omitted by the requested threshold.
    """

    operator: Hamiltonian
    truncation_error_bound: float

    @classmethod
    def _from_native(
        cls,
        operator: Hamiltonian,
        truncation_error_bound: float,
    ) -> PauliDecomposition:
        """Build one result from a checked native decomposition projection."""

        if (
            not isinstance(operator, Hamiltonian)
            or type(truncation_error_bound) is not float
            or not math.isfinite(truncation_error_bound)
            or truncation_error_bound < 0.0
        ):
            raise TypeError("native dense-matrix algebra violates the Python ABI")
        result = object.__new__(cls)
        object.__setattr__(result, "operator", operator)
        object.__setattr__(
            result,
            "truncation_error_bound",
            truncation_error_bound,
        )
        return result

    @property
    def alpha(self) -> float:
        """Return the retained LCU one-norm normalization."""

        derive = getattr(native, "observable_lcu_one_norm", None)
        if not callable(derive):
            raise TypeError("Q7400: native Pauli one-norm algebra is unavailable")
        value = derive(self.operator._native_literal)
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise TypeError("native Pauli one-norm algebra violates the Python ABI")
        return value

    @property
    def num_terms(self) -> int:
        """Return the retained count, including a nonzero identity word."""

        return len(self.operator.terms) + int(bool(self.operator.constant))


def decompose_pauli_matrix(
    matrix: npt.ArrayLike,
    *,
    atol: float = 0.0,
) -> PauliDecomposition:
    """Decompose an arbitrary finite dense matrix into complex Pauli words.

    The Rust FWHT uses bit zero as logical qubit zero and removes coefficients
    whose magnitude is at most ``atol``. The returned ``operator`` passes
    directly to :func:`qamomile.stdlib.pauli_block_encoding`; no separate LCU
    term or producer hierarchy is required.

    Args:
        matrix (numpy.typing.ArrayLike): Finite power-of-two square matrix.
        atol (float): Non-negative finite coefficient threshold.

    Returns:
        PauliDecomposition: Canonical operator and truncation evidence.

    Raises:
        TypeError: If the native dense-matrix boundary is unavailable or
            violates its result ABI.
        ValueError: If the input, threshold, or bounded output is invalid.
    """

    converted, entries, _ = _validated_dense_storage(
        matrix,
        require_hermitian=False,
        atol=0.0,
    )
    decompose = getattr(native, "decompose_pauli_matrix", None)
    if not callable(decompose):
        raise TypeError("Q7400: native dense-matrix algebra is unavailable")
    projection = decompose(converted.shape, entries, atol)
    if (
        not isinstance(projection, tuple)
        or len(projection) != 3
        or not isinstance(projection[0], bytes)
        or type(projection[1]) is not bool
        or not isinstance(projection[2], float)
        or not math.isfinite(projection[2])
        or projection[2] < 0.0
    ):
        raise TypeError("native dense-matrix algebra violates the Python ABI")
    return PauliDecomposition._from_native(
        _hamiltonian_from_native_header((projection[0], projection[1])),
        projection[2],
    )


def _validated_dense_storage(
    matrix: npt.ArrayLike,
    *,
    require_hermitian: bool,
    atol: float,
) -> tuple[
    npt.NDArray[np.complex128],
    tuple[tuple[float, float], ...],
    int,
]:
    """Copy and validate one dense matrix through the Rust boundary.

    Args:
        matrix (numpy.typing.ArrayLike): Candidate dense numeric matrix.
        require_hermitian (bool): Whether Rust must prove Hermiticity.
        atol (float): Absolute Hermitian tolerance.

    Returns:
        tuple: Owned read-only array, native complex-coordinate row, and width.

    Raises:
        TypeError: If native support is missing or malformed.
        ValueError: If Rust rejects the matrix domain.
    """

    converted = np.array(matrix, dtype=np.complex128, copy=True, order="C")
    inspect = getattr(native, "validate_hermitian_matrix", None)
    matrix_budget = getattr(native, "dense_matrix_entry_budget", None)
    if not callable(inspect) or not callable(matrix_budget):
        raise TypeError("Q7400: native dense-matrix algebra is unavailable")
    budget = matrix_budget()
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise TypeError("native dense-matrix algebra violates the Python ABI")
    if converted.size > budget:
        inspect(converted.shape, (), require_hermitian, atol)
        raise TypeError(
            "native dense-matrix algebra accepted an over-budget matrix shape"
        )
    entries = tuple((float(value.real), float(value.imag)) for value in converted.flat)
    num_qubits = inspect(
        converted.shape,
        entries,
        require_hermitian,
        atol,
    )
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("native dense-matrix algebra violates the Python ABI")
    converted.flags.writeable = False
    return converted, entries, num_qubits


def _hermitian_matrix_from_native(
    shape: tuple[int, int],
    value: object,
) -> HermitianMatrix:
    """Restore one compiler-validated arithmetic result without recomputation.

    Args:
        shape (tuple[int, int]): Expected result shape inherited from the left
            operand.
        value (object): Native row-major complex-coordinate projection.

    Returns:
        HermitianMatrix: Immutable wrapper around the native result.

    Raises:
        TypeError: If the native result violates its Python ABI.
    """

    entry_count = shape[0] * shape[1]
    if not isinstance(value, list) or len(value) != entry_count:
        raise TypeError("native dense-matrix algebra violates the Python ABI")
    entries: list[tuple[float, float]] = []
    complex_values: list[complex] = []
    for coordinate in value:
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 2
            or type(coordinate[0]) is not float
            or type(coordinate[1]) is not float
            or not math.isfinite(coordinate[0])
            or not math.isfinite(coordinate[1])
        ):
            raise TypeError("native dense-matrix algebra violates the Python ABI")
        entries.append(coordinate)
        complex_values.append(complex(*coordinate))
    matrix = np.asarray(complex_values, dtype=np.complex128).reshape(shape)
    matrix.flags.writeable = False
    result = object.__new__(HermitianMatrix)
    result._matrix = matrix
    result._entries = tuple(entries)
    result._num_qubits = shape[0].bit_length() - 1
    return result


def _real_matrix_scalar(value: object) -> float | NotImplementedType:
    """Normalize one Python or NumPy real scalar.

    Args:
        value (object): Candidate scalar.

    Returns:
        float | types.NotImplementedType: Real value, or
        ``NotImplemented`` for unsupported operand types.

    Raises:
        TypeError: If a numeric scalar has a significant imaginary part.
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, complex, np.number)
    ):
        return NotImplemented
    scalar = complex(value)
    if abs(scalar.imag) > 1e-15:
        raise TypeError("HermitianMatrix only supports multiplication by real scalars")
    return float(scalar.real)


def subspace_hamiltonian(
    samples: Samples,
    hamiltonian: Hamiltonian,
) -> npt.NDArray[np.complex128]:
    """Project a Pauli Hamiltonian onto Z-basis product states.

    Args:
        samples (Sequence[Sequence[int]]): Computational-basis bitstrings in
            logical-qubit order. Repeated rows are allowed.
        hamiltonian (Hamiltonian): Observable whose matrix elements are built.

    Returns:
        numpy.ndarray: Complex Hermitian subspace matrix when the input
        Hamiltonian is Hermitian.

    Raises:
        TypeError: If samples or their entries do not have integral sequence
            form, or if ``hamiltonian`` has the wrong type.
        ValueError: If a sample has the wrong width or contains a non-bit value.
    """

    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("hamiltonian must be a Hamiltonian")
    hamiltonian_matrix, _ = generalized_subspace_matrices(
        samples,
        (Pauli.Z,) * hamiltonian.num_qubits,
        hamiltonian,
    )
    return hamiltonian_matrix


def generalized_subspace_matrices(
    samples: Samples,
    bases: Bases,
    hamiltonian: Hamiltonian,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Project a Hamiltonian and identity onto Pauli product states.

    Each basis entry is ``Pauli.X``, ``Pauli.Y``, ``Pauli.Z`` or its integer
    code ``0``, ``1``, ``2``. A one-dimensional basis row is broadcast across
    samples; a two-dimensional value selects a basis independently per sample.

    Args:
        samples (Sequence[Sequence[int]]): Pauli eigenvalue bits in logical
            qubit order, with zero denoting the positive eigenstate.
        bases (Bases): Broadcast or per-sample X/Y/Z basis labels.
        hamiltonian (Hamiltonian): Canonical Pauli operator to project.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Projected Hamiltonian and overlap
        matrices in source sample order.

    Raises:
        TypeError: If a container, entry, Hamiltonian, or native result has an
            invalid type.
        ValueError: If a shape, bit, basis code, coefficient, or projection
            budget is invalid.
    """

    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("hamiltonian must be a Hamiltonian")
    sample_rows = _validated_samples(samples, hamiltonian.num_qubits)
    basis_rows = _validated_bases(
        bases,
        sample_count=len(sample_rows),
        width=hamiltonian.num_qubits,
    )
    return _project_pauli_subspace(sample_rows, basis_rows, hamiltonian)


def solve_subspace(
    samples: Samples,
    hamiltonian: Hamiltonian,
    *,
    bases: Bases | None = None,
    overlap_tol: float = 1e-12,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """Solve a Hermitian generalized eigenproblem in sampled Pauli states.

    Args:
        samples (Sequence[Sequence[int]]): Pauli eigenvalue bitstrings.
        hamiltonian (Hamiltonian): Hermitian observable to diagonalize.
        bases (Bases | None): Optional broadcast or per-sample Pauli bases.
            ``None`` selects the Z basis.
        overlap_tol (float): Non-negative finite cutoff for null overlap
            directions.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Ascending eigenvalues and column
        eigenvectors in the supplied sample order. Empty input returns arrays
        with shapes ``(0,)`` and ``(0, 0)``.

    Raises:
        TypeError: If inputs have invalid types.
        ValueError: If inputs are malformed, the Hamiltonian is non-Hermitian,
            or no overlap direction survives the cutoff.
    """

    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("hamiltonian must be a Hamiltonian")
    if not hamiltonian.is_hermitian:
        raise ValueError("solve_subspace requires a Hermitian Hamiltonian")
    if isinstance(overlap_tol, bool) or not isinstance(overlap_tol, (int, float)):
        raise TypeError("overlap_tol must be a real scalar")
    overlap_tol = float(overlap_tol)
    if not math.isfinite(overlap_tol) or overlap_tol < 0.0:
        raise ValueError("overlap_tol must be finite and non-negative")
    if bases is None:
        bases = (Pauli.Z,) * hamiltonian.num_qubits
    projected, overlap = generalized_subspace_matrices(samples, bases, hamiltonian)
    if projected.shape == (0, 0):
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.complex128),
        )
    projected = 0.5 * (projected + projected.conjugate().T)
    overlap = 0.5 * (overlap + overlap.conjugate().T)
    overlap_values, overlap_vectors = np.linalg.eigh(overlap)
    keep = overlap_values > overlap_tol
    if not keep.any():
        raise ValueError("all overlap eigenvalues are below overlap_tol")
    retained = overlap_vectors[:, keep]
    inverse_sqrt = retained / np.sqrt(overlap_values[keep])[None, :]
    whitened = inverse_sqrt.conjugate().T @ projected @ inverse_sqrt
    whitened = 0.5 * (whitened + whitened.conjugate().T)
    eigenvalues, whitened_vectors = np.linalg.eigh(whitened)
    eigenvectors = inverse_sqrt @ whitened_vectors
    return eigenvalues.astype(np.float64, copy=False), eigenvectors


def _validated_samples(
    samples: object,
    width: int,
) -> tuple[tuple[int, ...], ...]:
    """Validate fixed-width bitstrings for the native projection.

    Args:
        samples (Sequence[Sequence[int]]): Candidate computational basis.
        width (int): Required logical width.

    Returns:
        tuple[tuple[int, ...], ...]: Canonical source-ordered bit rows.

    Raises:
        TypeError: If samples or entries do not have integral sequence form.
        ValueError: If a row has the wrong width or contains a non-bit value.
    """

    if isinstance(samples, (str, bytes)) or not isinstance(
        samples, (Sequence, np.ndarray)
    ):
        raise TypeError("samples must be a sequence of bit sequences")
    rows: list[tuple[int, ...]] = []
    for row in samples:
        if isinstance(row, (str, bytes)) or not isinstance(row, (Sequence, np.ndarray)):
            raise TypeError("each sample must be a bit sequence")
        if len(row) != width:
            raise ValueError(f"each sample must contain exactly {width} bits")
        normalized: list[int] = []
        for bit in row:
            if isinstance(bit, bool) or not isinstance(bit, Integral):
                raise TypeError("sample entries must be integers")
            value = int(bit)
            if value not in (0, 1):
                raise ValueError("sample entries must be zero or one")
            normalized.append(value)
        rows.append(tuple(normalized))
    return tuple(rows)


def _validated_bases(
    bases: object,
    *,
    sample_count: int,
    width: int,
) -> tuple[tuple[int, ...], ...]:
    """Normalize broadcast or per-sample Pauli basis labels.

    Args:
        bases (object): Candidate one- or two-dimensional basis data.
        sample_count (int): Required output row count.
        width (int): Required logical width.

    Returns:
        tuple[tuple[int, ...], ...]: Canonical basis-code rows.

    Raises:
        TypeError: If the container or a label has an invalid type.
        ValueError: If the shape or a basis code is invalid.
    """

    if isinstance(bases, (str, bytes)) or not isinstance(bases, (Sequence, np.ndarray)):
        raise TypeError("bases must be a sequence")
    raw = list(bases)
    if len(raw) == width and all(
        isinstance(value, Pauli) or not isinstance(value, (Sequence, np.ndarray))
        for value in raw
    ):
        row = tuple(_basis_code(value) for value in raw)
        return (row,) * sample_count
    if len(raw) != sample_count:
        raise ValueError("bases must be one row or one row per sample")
    rows: list[tuple[int, ...]] = []
    for row in raw:
        if isinstance(row, (str, bytes)) or not isinstance(row, (Sequence, np.ndarray)):
            raise TypeError("each basis row must be a sequence")
        if len(row) != width:
            raise ValueError(f"each basis row must contain exactly {width} labels")
        rows.append(tuple(_basis_code(value) for value in row))
    return tuple(rows)


def _basis_code(value: object) -> int:
    """Return the native code for one X/Y/Z basis label.

    Args:
        value (object): Pauli enum or integer code.

    Returns:
        int: Zero for X, one for Y, or two for Z.

    Raises:
        TypeError: If the label has the wrong host type.
        ValueError: If an integer code lies outside the basis domain.
    """

    if isinstance(value, Pauli):
        return {Pauli.X: 0, Pauli.Y: 1, Pauli.Z: 2}[value]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("basis entries must be Pauli.X/Y/Z or integer codes")
    code = int(value)
    if code not in (0, 1, 2):
        raise ValueError("basis codes must be zero, one, or two")
    return code


def _project_pauli_subspace(
    samples: tuple[tuple[int, ...], ...],
    bases: tuple[tuple[int, ...], ...],
    hamiltonian: Hamiltonian,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Call and validate the Rust sampled-product-state projection.

    Args:
        samples (tuple[tuple[int, ...], ...]): Canonical bit rows.
        bases (tuple[tuple[int, ...], ...]): Canonical basis-code rows.
        hamiltonian (Hamiltonian): Canonical observable.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Hamiltonian and overlap matrices.

    Raises:
        TypeError: If native support is absent or violates its result ABI.
        ValueError: If Rust rejects the projection domain.
    """

    project = getattr(native, "project_pauli_subspace", None)
    if not callable(project):
        raise TypeError("Q7400: native Pauli subspace projection is unavailable")
    result = project(
        hamiltonian._native_literal,
        samples,
        bases,
    )
    if (
        not isinstance(result, tuple)
        or len(result) != 3
        or isinstance(result[0], bool)
        or not isinstance(result[0], int)
        or not isinstance(result[1], list)
        or not isinstance(result[2], list)
    ):
        raise TypeError("native Pauli subspace projection violates the Python ABI")
    size = result[0]
    if (
        size != len(samples)
        or len(result[1]) != size * size
        or len(result[2]) != size * size
    ):
        raise TypeError("native Pauli subspace projection violates the Python ABI")
    matrices: list[npt.NDArray[np.complex128]] = []
    for values in (result[1], result[2]):
        coordinates: list[complex] = []
        for value in values:
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or not isinstance(value[0], float)
                or not isinstance(value[1], float)
                or not math.isfinite(value[0])
                or not math.isfinite(value[1])
            ):
                raise TypeError(
                    "native Pauli subspace projection violates the Python ABI"
                )
            coordinates.append(complex(value[0], value[1]))
        matrices.append(
            np.asarray(coordinates, dtype=np.complex128).reshape(size, size)
        )
    return matrices[0], matrices[1]


__all__ = [
    "HermitianMatrix",
    "PauliDecomposition",
    "decompose_pauli_matrix",
    "generalized_subspace_matrices",
    "solve_subspace",
    "subspace_hamiltonian",
]
