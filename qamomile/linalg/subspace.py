"""Sample-based partial diagonalization in a Pauli-eigenbasis subspace.

Given a finite collection of computational- or Pauli-basis product states
``{|s_i; β_i⟩}`` and a Hamiltonian ``H``, build the projected Hamiltonian
and overlap matrices

    H_sub[i, j] = ⟨s_i; β_i| H |s_j; β_j⟩,
    S_sub[i, j] = ⟨s_i; β_i|     s_j; β_j⟩,

and (optionally) solve the generalized eigenvalue problem
``H_sub v = λ S_sub v`` to recover variational eigenpairs in the
sampled subspace. When every sample is in the Z computational basis,
``S_sub`` reduces to the identity (modulo duplicates) and the function
``subspace_hamiltonian`` provides a vectorised XOR/parity fast path.

Conventions:

* **Qubit ordering**: qubit ``0`` is the least-significant bit of the
  computational-basis index, matching
  :meth:`qamomile.observable.Hamiltonian.to_numpy` and
  :class:`qamomile.linalg.HermitianMatrix`.
* **Y-eigenstate phase**: ``|Y, 0⟩ = |+i⟩ = (|0⟩ + i|1⟩)/√2`` with
  ``Y |+i⟩ = +|+i⟩``. This is consistent with the basis-rotation pulse
  ``S† H`` used to map a Y measurement onto the Z basis.
* **Basis codes**: ``X = 0``, ``Y = 1``, ``Z = 2``. Both raw integer
  codes and ``qamomile.observable.Pauli`` enum members are accepted on
  input. ``Pauli.I`` is rejected — every sample must lie in a definite
  X/Y/Z eigenbasis.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
import scipy.linalg

import qamomile.observable as qm_o

__all__ = [
    "subspace_hamiltonian",
    "generalized_subspace_matrices",
    "solve_subspace",
]


_BASIS_X = 0
_BASIS_Y = 1
_BASIS_Z = 2

_PAULI_TO_BASIS_CODE: dict[qm_o.Pauli, int] = {
    qm_o.Pauli.X: _BASIS_X,
    qm_o.Pauli.Y: _BASIS_Y,
    qm_o.Pauli.Z: _BASIS_Z,
}

BasisLike = Union[int, qm_o.Pauli]


def _build_local_overlap_table() -> np.ndarray:
    """Build the 1-qubit Pauli-eigenbasis overlap table.

    The returned array indexes as
    ``overlap[bra_basis, bra_bit, ket_basis, ket_bit]`` and stores
    ``⟨bra_basis, bra_bit | ket_basis, ket_bit⟩`` for the convention
    documented at the module level.

    Returns:
        Complex ``ndarray`` of shape ``(3, 2, 3, 2)``.
    """
    overlap = np.empty((3, 2, 3, 2), dtype=np.complex128)
    sqrt2_inv = 1.0 / np.sqrt(2.0)

    for basis in (_BASIS_X, _BASIS_Y, _BASIS_Z):
        overlap[basis, 0, basis, 0] = 1.0
        overlap[basis, 0, basis, 1] = 0.0
        overlap[basis, 1, basis, 0] = 0.0
        overlap[basis, 1, basis, 1] = 1.0

    # Z-X block: <0|+> = <0|-> = <1|+> = 1/sqrt(2), <1|-> = -1/sqrt(2).
    overlap[_BASIS_Z, 0, _BASIS_X, 0] = sqrt2_inv
    overlap[_BASIS_Z, 0, _BASIS_X, 1] = sqrt2_inv
    overlap[_BASIS_Z, 1, _BASIS_X, 0] = sqrt2_inv
    overlap[_BASIS_Z, 1, _BASIS_X, 1] = -sqrt2_inv

    # Z-Y block with the |Y,0> = |+i> convention.
    overlap[_BASIS_Z, 0, _BASIS_Y, 0] = sqrt2_inv
    overlap[_BASIS_Z, 0, _BASIS_Y, 1] = sqrt2_inv
    overlap[_BASIS_Z, 1, _BASIS_Y, 0] = 1.0j * sqrt2_inv
    overlap[_BASIS_Z, 1, _BASIS_Y, 1] = -1.0j * sqrt2_inv

    # X-Y block.
    overlap[_BASIS_X, 0, _BASIS_Y, 0] = 0.5 * (1.0 + 1.0j)
    overlap[_BASIS_X, 0, _BASIS_Y, 1] = 0.5 * (1.0 - 1.0j)
    overlap[_BASIS_X, 1, _BASIS_Y, 0] = 0.5 * (1.0 - 1.0j)
    overlap[_BASIS_X, 1, _BASIS_Y, 1] = 0.5 * (1.0 + 1.0j)

    # Each remaining cross-basis block is the Hermitian conjugate of the
    # one set above, transposed in the (bit-of-bra, bit-of-ket) plane.
    for src_bra, src_ket in (
        (_BASIS_Z, _BASIS_X),
        (_BASIS_Z, _BASIS_Y),
        (_BASIS_X, _BASIS_Y),
    ):
        for src_bb in (0, 1):
            for src_kb in (0, 1):
                overlap[src_ket, src_kb, src_bra, src_bb] = np.conjugate(
                    overlap[src_bra, src_bb, src_ket, src_kb]
                )

    return overlap


def _apply_pauli_to_basis_state(
    pauli: int, ket_basis: int, ket_bit: int
) -> tuple[complex, int, int]:
    """Apply a single-qubit Pauli to a basis eigenstate.

    Args:
        pauli: Pauli code (``0`` for X, ``1`` for Y, ``2`` for Z).
        ket_basis: Basis code of the ket eigenstate.
        ket_bit: Eigenvalue index of the ket (``0`` for the ``+1``
            eigenstate, ``1`` for the ``-1`` eigenstate).

    Returns:
        Tuple ``(phase, new_basis, new_bit)`` such that
        ``P |ket_basis, ket_bit⟩ = phase * |new_basis, new_bit⟩``.

    Raises:
        ValueError: If any argument is out of the documented range.
    """
    if ket_bit not in (0, 1):
        raise ValueError(f"ket_bit must be 0 or 1, got {ket_bit}")
    if pauli not in (_BASIS_X, _BASIS_Y, _BASIS_Z):
        raise ValueError(f"pauli must be 0/1/2, got {pauli}")
    if ket_basis not in (_BASIS_X, _BASIS_Y, _BASIS_Z):
        raise ValueError(f"ket_basis must be 0/1/2, got {ket_basis}")

    if pauli == _BASIS_X:
        if ket_basis == _BASIS_X:
            return ((1.0 if ket_bit == 0 else -1.0), _BASIS_X, ket_bit)
        if ket_basis == _BASIS_Y:
            return (
                (1.0j if ket_bit == 0 else -1.0j),
                _BASIS_Y,
                1 - ket_bit,
            )
        return (1.0, _BASIS_Z, 1 - ket_bit)

    if pauli == _BASIS_Y:
        if ket_basis == _BASIS_X:
            return (
                (-1.0j if ket_bit == 0 else 1.0j),
                _BASIS_X,
                1 - ket_bit,
            )
        if ket_basis == _BASIS_Y:
            return ((1.0 if ket_bit == 0 else -1.0), _BASIS_Y, ket_bit)
        return (
            (1.0j if ket_bit == 0 else -1.0j),
            _BASIS_Z,
            1 - ket_bit,
        )

    # pauli == Z
    if ket_basis == _BASIS_X:
        return (1.0, _BASIS_X, 1 - ket_bit)
    if ket_basis == _BASIS_Y:
        return (1.0, _BASIS_Y, 1 - ket_bit)
    return ((1.0 if ket_bit == 0 else -1.0), _BASIS_Z, ket_bit)


def _build_local_pauli_table(overlap: np.ndarray) -> np.ndarray:
    """Build the 1-qubit Pauli matrix-element table.

    The returned array indexes as
    ``pauli[pauli_code, bra_basis, bra_bit, ket_basis, ket_bit]`` and
    stores ``⟨bra_basis, bra_bit | P | ket_basis, ket_bit⟩``.

    Args:
        overlap: The basis-overlap table built by
            :func:`_build_local_overlap_table`.

    Returns:
        Complex ``ndarray`` of shape ``(3, 3, 2, 3, 2)``.
    """
    table = np.empty((3, 3, 2, 3, 2), dtype=np.complex128)
    for p in (_BASIS_X, _BASIS_Y, _BASIS_Z):
        for bb in (_BASIS_X, _BASIS_Y, _BASIS_Z):
            for bbit in (0, 1):
                for kb in (_BASIS_X, _BASIS_Y, _BASIS_Z):
                    for kbit in (0, 1):
                        phase, new_kb, new_kbit = _apply_pauli_to_basis_state(
                            p, kb, kbit
                        )
                        table[p, bb, bbit, kb, kbit] = (
                            phase * overlap[bb, bbit, new_kb, new_kbit]
                        )
    return table


_LOCAL_OVERLAP = _build_local_overlap_table()
_LOCAL_PAULI = _build_local_pauli_table(_LOCAL_OVERLAP)


def _coerce_basis_value(b: BasisLike) -> int:
    """Map a user-supplied basis label to an internal basis code.

    Args:
        b: Either an integer code (``0`` / ``1`` / ``2``) or a
            :class:`qamomile.observable.Pauli` member ``X`` / ``Y`` /
            ``Z``.

    Returns:
        The basis code in ``{0, 1, 2}``.

    Raises:
        ValueError: If the value is not a recognised basis label.
    """
    if isinstance(b, qm_o.Pauli):
        if b not in _PAULI_TO_BASIS_CODE:
            raise ValueError(
                f"basis must be Pauli.X/Y/Z (got {b}); "
                "Pauli.I is not a measurement basis"
            )
        return _PAULI_TO_BASIS_CODE[b]
    if isinstance(b, (int, np.integer)):
        if int(b) not in (0, 1, 2):
            raise ValueError(
                f"basis code must be 0/1/2 (got {int(b)})"
            )
        return int(b)
    raise ValueError(f"unsupported basis label: {b!r}")


def _coerce_samples(
    samples: Sequence[Sequence[int]] | np.ndarray, num_qubits: int
) -> np.ndarray:
    """Validate and pack ``samples`` into an ``int8`` ``ndarray``.

    Args:
        samples: Iterable of bitstrings, each of length at least
            ``num_qubits``. Excess columns are truncated; entries must
            be in ``{0, 1}``.
        num_qubits: Number of qubits expected per sample.

    Returns:
        ``ndarray`` of shape ``(n_samples, num_qubits)`` and dtype
        ``int8``. When ``samples`` is empty the return shape is
        ``(0, num_qubits)``.

    Raises:
        ValueError: On wrong rank, insufficient width, or out-of-range
            entries.
    """
    bits = np.asarray(samples, dtype=np.int8)
    if bits.size == 0:
        return bits.reshape(0, num_qubits)
    if bits.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {bits.shape}")
    if bits.shape[1] < num_qubits:
        raise ValueError(
            f"samples must have at least {num_qubits} columns, "
            f"got shape {bits.shape}"
        )
    bits = bits[:, :num_qubits]
    if ((bits != 0) & (bits != 1)).any():
        raise ValueError("samples entries must all be 0 or 1")
    return bits


def _coerce_bases(
    bases: Sequence[Sequence[BasisLike]] | Sequence[BasisLike] | np.ndarray,
    n_samples: int,
    num_qubits: int,
) -> np.ndarray:
    """Validate and pack ``bases`` into an ``int8`` ``ndarray``.

    Accepts either a per-qubit shape ``(num_qubits,)`` (broadcast across
    samples) or a per-sample shape ``(n_samples, num_qubits)``.

    Args:
        bases: Per-qubit or per-sample basis labels.
        n_samples: Expected number of samples.
        num_qubits: Expected number of qubits per sample.

    Returns:
        ``ndarray`` of shape ``(n_samples, num_qubits)`` and dtype
        ``int8`` containing basis codes.

    Raises:
        ValueError: On rank/length mismatch or unrecognised labels.
    """
    if isinstance(bases, np.ndarray) and np.issubdtype(bases.dtype, np.integer):
        # Fast path: integer ndarray taken at face value as basis codes.
        arr = bases.astype(np.int8, copy=False)
    else:
        bases_list = list(bases)
        if not bases_list:
            arr = np.zeros((0,), dtype=np.int8)
        else:
            first = bases_list[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                arr = np.asarray(
                    [
                        [_coerce_basis_value(b) for b in row]  # type: ignore[union-attr]
                        for row in bases_list
                    ],
                    dtype=np.int8,
                )
            else:
                arr = np.asarray(
                    [_coerce_basis_value(b) for b in bases_list],  # type: ignore[arg-type]
                    dtype=np.int8,
                )

    if arr.ndim == 1:
        if arr.shape[0] != num_qubits:
            raise ValueError(
                f"bases of shape {arr.shape} does not match "
                f"num_qubits={num_qubits}"
            )
        arr = np.broadcast_to(arr, (n_samples, num_qubits)).copy()
    elif arr.ndim == 2:
        if arr.shape != (n_samples, num_qubits):
            raise ValueError(
                f"bases shape {arr.shape} does not match "
                f"({n_samples}, {num_qubits})"
            )
    else:
        raise ValueError(f"bases must be 1D or 2D, got shape {arr.shape}")

    arr = arr.astype(np.int8, copy=False)
    if ((arr < 0) | (arr > 2)).any():
        raise ValueError("bases entries must all be in {0, 1, 2}")
    return arr


def subspace_hamiltonian(
    samples: Sequence[Sequence[int]] | np.ndarray,
    hamiltonian: qm_o.Hamiltonian,
) -> np.ndarray:
    """Build ``H_sub[i, j] = ⟨s_i| H |s_j⟩`` for Z-basis bitstring samples.

    Vectorised over all ``(i, j)`` pairs using XOR masks to detect the
    X/Y support of each Pauli term and Z/Y parities for the sign and
    phase. Identical bitstrings produce a degenerate row/column; dedupe
    upstream if the resulting subspace must be non-degenerate.

    Args:
        samples: Iterable of length-``num_qubits`` bitstrings, with
            ``samples[k][q]`` the Z-eigenvalue index (``0`` for ``|0⟩``,
            ``1`` for ``|1⟩``) of qubit ``q``. Excess columns are
            truncated.
        hamiltonian: The Hamiltonian whose matrix elements are taken.
            ``hamiltonian.constant`` contributes ``c · I`` on the
            diagonal of the projected matrix.

    Returns:
        Complex ``ndarray`` of shape ``(n_samples, n_samples)`` holding
        ``⟨s_i|H|s_j⟩``.

    Raises:
        ValueError: If ``samples`` has the wrong rank, fewer columns
            than ``hamiltonian.num_qubits``, or out-of-range entries.

    Example:
        >>> import qamomile.observable as qm_o
        >>> from qamomile.linalg import subspace_hamiltonian
        >>> H = qm_o.Z(0) + qm_o.X(1)
        >>> # All four Z-basis bitstrings on 2 qubits
        >>> samples = [(0, 0), (1, 0), (0, 1), (1, 1)]
        >>> H_sub = subspace_hamiltonian(samples, H)
        >>> H_sub.shape
        (4, 4)
    """
    num_qubits = hamiltonian.num_qubits
    bits = _coerce_samples(samples, num_qubits)
    n = bits.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.complex128)

    sub_h = np.zeros((n, n), dtype=np.complex128)
    constant = hamiltonian.constant
    if constant:
        sub_h += complex(constant) * np.eye(n, dtype=np.complex128)

    for pauli_ops, coeff in hamiltonian.terms.items():
        flip_mask = np.zeros(num_qubits, dtype=np.int8)
        phase_sites: list[int] = []
        y_count = 0
        for op in pauli_ops:
            if op.pauli == qm_o.Pauli.X:
                flip_mask[op.index] = 1
            elif op.pauli == qm_o.Pauli.Y:
                flip_mask[op.index] = 1
                phase_sites.append(op.index)
                y_count += 1
            elif op.pauli == qm_o.Pauli.Z:
                phase_sites.append(op.index)
            # Pauli.I contributes no flip and no phase.

        diff = bits[:, None, :] ^ bits[None, :, :]
        pair_mask = np.all(diff == flip_mask, axis=-1)
        if not pair_mask.any():
            continue

        if phase_sites:
            parity = bits[:, phase_sites].sum(axis=1) & 1
            signs_j = 1.0 - 2.0 * parity.astype(np.float64)
        else:
            signs_j = np.ones(n, dtype=np.float64)

        sub_h += (
            complex(coeff)
            * (1j**y_count)
            * pair_mask
            * signs_j[None, :]
        )

    return sub_h


def generalized_subspace_matrices(
    samples: Sequence[Sequence[int]] | np.ndarray,
    bases: Sequence[Sequence[BasisLike]] | Sequence[BasisLike] | np.ndarray,
    hamiltonian: qm_o.Hamiltonian,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(H_sub, S_sub)`` for samples in mixed X/Y/Z eigenbases.

    Each sample ``|s_i; β_i⟩ = ⊗_q |β_{i,q}, s_{i,q}⟩`` is a tensor
    product of single-qubit Pauli eigenstates with per-qubit basis
    ``β_{i,q}`` and bit ``s_{i,q}``. The returned matrices are

        H_sub[i, j] = ⟨s_i; β_i| H |s_j; β_j⟩,
        S_sub[i, j] = ⟨s_i; β_i|     s_j; β_j⟩,

    so the variational eigenpairs are recovered by solving the
    generalized eigenvalue problem ``H_sub v = λ S_sub v`` (see
    :func:`solve_subspace`).

    Args:
        samples: Iterable of bitstrings, with ``samples[k][q]`` the
            eigenvalue index for qubit ``q`` in basis ``bases[k][q]``
            (``0`` for the ``+1`` eigenstate, ``1`` for the ``-1``
            eigenstate).
        bases: Per-qubit basis labels. Either a 1-D sequence of length
            ``num_qubits`` (broadcast across all samples) or a 2-D
            ``(n_samples, num_qubits)`` array. Each entry is either a
            ``Pauli.X / Y / Z`` member or the corresponding integer
            code ``0`` / ``1`` / ``2``.
        hamiltonian: The Hamiltonian whose matrix elements are taken.

    Returns:
        Tuple ``(H_sub, S_sub)`` of complex ``ndarray`` s with shape
        ``(n_samples, n_samples)``.

    Raises:
        ValueError: On shape, range, or label mismatches in ``samples``
            or ``bases``.

    Example:
        Mixed-basis subspace with one Z-basis sample (|0⟩) and one
        X-basis sample (|+⟩) for H = Z:

        >>> import qamomile.observable as qm_o
        >>> from qamomile.linalg import generalized_subspace_matrices
        >>> H = qm_o.Z(0)
        >>> samples = [(0,), (0,)]  # bit=0 in each basis
        >>> bases = [(qm_o.Pauli.Z,), (qm_o.Pauli.X,)]  # |0⟩ and |+⟩
        >>> H_sub, S_sub = generalized_subspace_matrices(samples, bases, H)
        >>> S_sub  # off-diagonal ⟨0|+⟩ = 1/√2, not identity
        array([[1.    +0.j, 0.7071+0.j],
               [0.7071+0.j, 1.    +0.j]])
        >>> H_sub  # ⟨0|Z|0⟩=1, ⟨+|Z|+⟩=0, ⟨0|Z|+⟩=1/√2
        array([[1.    +0.j, 0.7071+0.j],
               [0.7071+0.j, 0.    +0.j]])
    """
    num_qubits = hamiltonian.num_qubits
    bits = _coerce_samples(samples, num_qubits)
    n = bits.shape[0]
    if n == 0:
        empty = np.zeros((0, 0), dtype=np.complex128)
        return empty, empty.copy()

    base_codes = _coerce_bases(bases, n, num_qubits)

    # Broadcasted indices for fancy lookups.
    bra_basis = base_codes[:, None, :]
    bra_bit = bits[:, None, :]
    ket_basis = base_codes[None, :, :]
    ket_bit = bits[None, :, :]

    overlap_per_q = _LOCAL_OVERLAP[bra_basis, bra_bit, ket_basis, ket_bit]
    s_sub = overlap_per_q.prod(axis=-1)

    h_sub = complex(hamiltonian.constant) * s_sub

    for pauli_ops, coeff in hamiltonian.terms.items():
        factors = overlap_per_q.copy()
        for op in pauli_ops:
            if op.pauli == qm_o.Pauli.I:
                continue
            q = op.index
            p_code = _PAULI_TO_BASIS_CODE[op.pauli]
            factors[:, :, q] = _LOCAL_PAULI[
                p_code,
                bra_basis[:, :, q],
                bra_bit[:, :, q],
                ket_basis[:, :, q],
                ket_bit[:, :, q],
            ]
        h_sub = h_sub + complex(coeff) * factors.prod(axis=-1)

    return h_sub, s_sub


def solve_subspace(
    samples: Sequence[Sequence[int]] | np.ndarray,
    hamiltonian: qm_o.Hamiltonian,
    *,
    bases: Sequence[Sequence[BasisLike]]
    | Sequence[BasisLike]
    | np.ndarray
    | None = None,
    overlap_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the (generalized) subspace eigenvalue problem.

    When ``bases`` is ``None`` every sample is treated as a Z-basis
    bitstring, ``S_sub`` is the identity (assuming distinct samples),
    and a standard Hermitian eigendecomposition of
    :func:`subspace_hamiltonian` is returned. Otherwise ``H_sub v =
    λ S_sub v`` is solved via :func:`scipy.linalg.eigh`. If ``S_sub``
    is rank-deficient — duplicate or near-duplicate samples make the
    overlap singular — the routine falls back to projecting onto the
    range of ``S_sub`` using ``overlap_tol`` as the cutoff on the
    overlap eigenvalues.

    Args:
        samples: Iterable of length-``num_qubits`` bitstrings.
        hamiltonian: The Hamiltonian to project.
        bases: Optional per-qubit or per-sample basis labels (see
            :func:`generalized_subspace_matrices`). Defaults to None,
            which means every qubit of every sample is in the Z basis.
        overlap_tol: Threshold below which an eigenvalue of ``S_sub``
            is treated as zero in the rank-deficient fallback. Must be
            non-negative. Defaults to ``1e-12``.

    Returns:
        Tuple ``(eigvals, eigvecs)``. ``eigvals`` is a real 1-D array
        sorted in ascending order; ``eigvecs[:, k]`` is the variational
        coefficient vector of the ``k``-th eigenstate in the basis of
        the supplied samples. In the rank-deficient fallback path the
        number of returned eigenpairs is the effective rank of
        ``S_sub``.

    Raises:
        ValueError: If ``overlap_tol`` is negative or any input fails
            validation in :func:`subspace_hamiltonian` /
            :func:`generalized_subspace_matrices`.

    Example:
        >>> import qamomile.observable as qm_o
        >>> from qamomile.linalg import solve_subspace
        >>> H = -qm_o.X(0)
        >>> # Sampling the |+⟩ eigenstate of X recovers the ground state.
        >>> eigvals, _ = solve_subspace(
        ...     [(0,)], H, bases=(qm_o.Pauli.X,)
        ... )
        >>> float(eigvals[0])
        -1.0
    """
    if overlap_tol < 0:
        raise ValueError(f"overlap_tol must be non-negative, got {overlap_tol}")

    if bases is None:
        h_sub = subspace_hamiltonian(samples, hamiltonian)
        n = h_sub.shape[0]
        if n == 0:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros((0, 0), dtype=np.complex128),
            )
        eigvals, eigvecs = np.linalg.eigh(h_sub)
        return eigvals, eigvecs

    h_sub, s_sub = generalized_subspace_matrices(samples, bases, hamiltonian)
    n = h_sub.shape[0]
    if n == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.complex128),
        )

    # Symmetrise to absorb numerical noise before the Hermitian solver.
    h_sym = 0.5 * (h_sub + h_sub.conj().T)
    s_sym = 0.5 * (s_sub + s_sub.conj().T)

    overlap_eigvals, overlap_eigvecs = np.linalg.eigh(s_sym)
    keep = overlap_eigvals > overlap_tol
    if keep.all():
        eigvals, eigvecs = scipy.linalg.eigh(h_sym, s_sym)
        return eigvals, eigvecs

    # Project H onto range(S) using the kept overlap eigenvectors.
    u = overlap_eigvecs[:, keep]
    s_kept = overlap_eigvals[keep]
    inv_sqrt = u / np.sqrt(s_kept)[None, :]
    h_proj = inv_sqrt.conj().T @ h_sym @ inv_sqrt
    h_proj = 0.5 * (h_proj + h_proj.conj().T)
    eigvals, w = np.linalg.eigh(h_proj)
    eigvecs = inv_sqrt @ w
    return eigvals, eigvecs
