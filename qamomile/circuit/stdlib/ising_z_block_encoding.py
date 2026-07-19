"""Build exact block encodings of diagonal Ising-Z operators."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.operation.global_phase import global_phase
from qamomile.circuit.frontend.operation.qubit_gates import z
from qamomile.circuit.frontend.qkernel import QKernel, qkernel
from qamomile.circuit.ir.operation.callable import CallPolicy
from qamomile.circuit.stdlib.lcu_block_encoding import (
    LCUBlockEncoding,
    _BlockEncodingUnitary,
    _build_lcu_block_encoding_unitary,
    _coefficient_phase,
    _identity_vector,
    _lcu_num_signal_qubits,
    _register_lcu_block_encoding_static_binding,
    _validate_positive_integer,
)

_CanonicalTerm = tuple[tuple[int, ...], complex]


@dataclass(frozen=True, slots=True, eq=False)
class IsingZBlockEncoding(LCUBlockEncoding):
    r"""Describe one static exact block encoding of an Ising-Z operator.

    The inherited ``unitary`` field has the quantum ABI
    ``unitary(signal, system) -> (signal, system)`` and no classical
    arguments. For the isometry ``V0`` that initializes the complete signal
    register to zero, it satisfies

    .. math::

        V_0^\dagger U V_0 = A / \mathtt{normalization}

    including the complex phases of the Ising-Z coefficients. The unitary
    allocates no hidden logical qubits and preserves the order of all input
    wires. Descriptor equality and hashing use object identity.

    This producer-specific subtype adds no qkernel-visible fields. Reusable
    qkernels should annotate inputs with :class:`LCUBlockEncoding` so Ising-Z,
    Pauli, and recursively composed descriptors share one static binding slot.

    Args:
        unitary (QKernel): QKernel implementing the static
            block-encoding unitary.
        normalization (float): Finite positive LCU coefficient one-norm. The
            exact zero operator uses ``1.0``.
        num_signal_qubits (int): Positive physical signal-register width,
            including pass-through padding.
        num_system_qubits (int): Positive ordered system-register width.

    Raises:
        TypeError: If a field has an invalid runtime type or ``unitary`` does
            not have the required static block-encoding ABI.
        ValueError: If normalization is non-finite or non-positive, or a
            register width is non-positive.
    """


def ising_z_block_encoding(
    coefficients: Mapping[tuple[int, ...], complex],
    num_system_qubits: int,
) -> IsingZBlockEncoding:
    r"""Create an exact block encoding of a diagonal Ising-Z operator.

    Each mapping key is a product of Z operators at the listed system-qubit
    indices. Repeated indices cancel pairwise because ``Z**2 = I``; the empty
    tuple denotes identity. Algebraically equivalent words are aggregated
    before constructing

    .. math::

        A = \sum_j c_j Z_{S_j}, \qquad
        \alpha = \sum_j |c_j|.

    A multi-term encoding uses PREPARE amplitudes
    ``sqrt(abs(c_j) / alpha)`` and SELECT cases
    ``exp(1j * arg(c_j)) * Z_{S_j}``. The returned encoding can be used as a
    child of another block encoding without converting the operator to a
    ``PauliLCU``.

    Args:
        coefficients (Mapping[tuple[int, ...], complex]): Ising-Z
            coefficients keyed by products of zero-based system-qubit
            indices. Coefficients must convert to finite built-in complex
            values.
        num_system_qubits (int): Positive number of ordered system qubits.

    Returns:
        IsingZBlockEncoding: Frozen non-callable block-encoding descriptor.

    Raises:
        TypeError: If ``coefficients`` is not a mapping, a word is not a
            tuple, an index is not an integer, or a coefficient cannot be
            converted to ``complex``.
        ValueError: If a width is non-positive, an index is out of range, a
            coefficient or aggregate is non-finite, or finite coefficient
            aggregation overflows.

    Example:
        >>> import qamomile.circuit as qmc
        >>> encoding = qmc.ising_z_block_encoding(
        ...     {(): 0.25, (0,): -1.0, (0, 1): 0.5j},
        ...     num_system_qubits=2,
        ... )
        >>> @qmc.qkernel
        ... def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        ...     signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        ...     system = qmc.qubit_array(encoding.num_system_qubits, "system")
        ...     return encoding.unitary(signal, system)
    """
    width = _validate_positive_integer(num_system_qubits, "num_system_qubits")
    terms = _canonicalize_coefficients(coefficients, width)
    normalization = _coefficient_one_norm(terms) if terms else 1.0
    signal_width = _lcu_num_signal_qubits(len(terms))
    unitary = _build_lcu_block_encoding_unitary(
        tuple(_build_phased_z_case(term) for term in terms),
        tuple(coefficient for _, coefficient in terms),
        width,
        description="Ising-Z block encoding",
        preparation_name="ising_z_prepare",
    )

    _configure_block_encoding(
        unitary,
        terms,
        width,
        signal_width,
        normalization,
    )
    return IsingZBlockEncoding(
        unitary=unitary,
        normalization=normalization,
        num_signal_qubits=signal_width,
        num_system_qubits=width,
    )


def _canonicalize_coefficients(
    coefficients: Mapping[tuple[int, ...], complex],
    num_system_qubits: int,
) -> tuple[_CanonicalTerm, ...]:
    """Canonicalize and deterministically aggregate Ising-Z coefficients.

    Args:
        coefficients (Mapping[tuple[int, ...], complex]): Input word-to-
            coefficient mapping.
        num_system_qubits (int): Valid exclusive upper bound for indices.

    Returns:
        tuple[_CanonicalTerm, ...]: Nonzero canonical terms sorted by word.

    Raises:
        TypeError: If the input is not a mapping, a word is not a tuple, an
            index is not an integer, or a coefficient cannot be converted to
            ``complex``.
        ValueError: If an index is out of range, a coefficient is non-finite,
            or aggregation overflows or produces a non-finite value.
    """
    if not isinstance(coefficients, Mapping):
        raise TypeError("coefficients must be a mapping.")

    buckets: dict[tuple[int, ...], tuple[list[float], list[float]]] = {}
    for word, value in coefficients.items():
        canonical_word = _canonicalize_word(word, num_system_qubits)
        coefficient = _coerce_finite_complex(value, canonical_word)
        real_parts, imaginary_parts = buckets.setdefault(canonical_word, ([], []))
        real_parts.append(_canonicalize_zero(coefficient.real))
        imaginary_parts.append(_canonicalize_zero(coefficient.imag))

    terms: list[_CanonicalTerm] = []
    for word in sorted(buckets):
        real_parts, imaginary_parts = buckets[word]
        try:
            real = math.fsum(sorted(real_parts, key=_component_sort_key))
            imaginary = math.fsum(sorted(imaginary_parts, key=_component_sort_key))
        except OverflowError as exc:
            raise ValueError(
                f"coefficient aggregation overflowed for canonical word {word}."
            ) from exc
        real = _canonicalize_zero(real)
        imaginary = _canonicalize_zero(imaginary)
        if not math.isfinite(real) or not math.isfinite(imaginary):
            raise ValueError(
                f"coefficient aggregate must be finite for canonical word {word}."
            )
        if real == 0.0 and imaginary == 0.0:
            continue
        terms.append((word, complex(real, imaginary)))
    return tuple(terms)


def _canonicalize_word(
    word: object,
    num_system_qubits: int,
) -> tuple[int, ...]:
    """Reduce one Ising-Z word by index parity and sort it.

    Args:
        word (object): Candidate tuple of system-qubit indices.
        num_system_qubits (int): Valid exclusive upper bound for indices.

    Returns:
        tuple[int, ...]: Sorted indices occurring an odd number of times.

    Raises:
        TypeError: If ``word`` is not a tuple or an index is not a non-boolean
            integer.
        ValueError: If an index is outside the system register.
    """
    if not isinstance(word, tuple):
        raise TypeError("each Ising-Z word must be a tuple of qubit indices.")

    odd_indices: set[int] = set()
    for raw_index in word:
        if isinstance(raw_index, (bool, np.bool_)) or not isinstance(
            raw_index,
            (numbers.Integral, np.integer),
        ):
            raise TypeError("Ising-Z word indices must be integers, not booleans.")
        index = int(raw_index)
        if not 0 <= index < num_system_qubits:
            raise ValueError(
                f"Ising-Z word index {index} is outside [0, {num_system_qubits})."
            )
        if index in odd_indices:
            odd_indices.remove(index)
        else:
            odd_indices.add(index)
    return tuple(sorted(odd_indices))


def _coerce_finite_complex(value: object, word: tuple[int, ...]) -> complex:
    """Convert one coefficient to a finite built-in complex value.

    Args:
        value (object): Candidate coefficient.
        word (tuple[int, ...]): Canonical word used in diagnostics.

    Returns:
        complex: Finite built-in complex coefficient with signed zeros removed.

    Raises:
        TypeError: If ``value`` cannot be converted to ``complex``.
        ValueError: If conversion overflows or produces a non-finite component.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (numbers.Complex, np.integer, np.floating, np.complexfloating),
    ):
        raise TypeError(
            f"coefficient for canonical word {word} must be a numeric scalar."
        )
    try:
        coefficient = complex(value)  # type: ignore[call-overload]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"coefficient for canonical word {word} must convert to complex."
        ) from exc
    except OverflowError as exc:
        raise ValueError(
            f"coefficient conversion overflowed for canonical word {word}."
        ) from exc
    if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
        raise ValueError(f"coefficient for canonical word {word} must be finite.")
    return complex(
        _canonicalize_zero(coefficient.real),
        _canonicalize_zero(coefficient.imag),
    )


def _canonicalize_zero(value: float) -> float:
    """Replace either signed floating-point zero with positive zero.

    Args:
        value (float): Floating-point component.

    Returns:
        float: ``0.0`` for either zero sign, otherwise ``value`` unchanged.
    """
    return 0.0 if value == 0.0 else value


def _component_sort_key(value: float) -> tuple[float, float]:
    """Return the deterministic summation-order key for one component.

    Args:
        value (float): Finite real or imaginary coefficient component.

    Returns:
        tuple[float, float]: Magnitude-first ordering key with value tie-break.
    """
    return abs(value), value


def _coefficient_one_norm(terms: tuple[_CanonicalTerm, ...]) -> float:
    """Compute the finite positive coefficient one-norm.

    Args:
        terms (tuple[_CanonicalTerm, ...]): Nonzero canonical terms.

    Returns:
        float: Finite positive sum of coefficient magnitudes.

    Raises:
        ValueError: If summation overflows or produces a non-finite value.
    """
    try:
        normalization = math.fsum(abs(coefficient) for _, coefficient in terms)
    except OverflowError as exc:
        raise ValueError("Ising-Z normalization overflowed.") from exc
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Ising-Z normalization must be finite and positive.")
    return normalization


def _build_phased_z_case(
    term: _CanonicalTerm,
) -> QKernel[..., Vector[Qubit]]:
    """Build one explicit SELECT case for a complex Ising-Z term.

    Args:
        term (_CanonicalTerm): Nonzero canonical Ising-Z term.

    Returns:
        QKernel[..., Vector[Qubit]]: Kernel implementing the phased Z word.
    """
    word, coefficient = term
    phase = _coefficient_phase(coefficient)

    if not phase:

        @qkernel
        def real_case(system: Vector[Qubit]) -> Vector[Qubit]:
            """Apply one positive-real Ising-Z SELECT case.

            Args:
                system (Vector[Qubit]): Shared SELECT target register.

            Returns:
                Vector[Qubit]: Transformed target register.
            """
            _apply_z_word(system, word)
            return system

        return real_case

    @qkernel
    def phased_case(system: Vector[Qubit]) -> Vector[Qubit]:
        """Apply one complex-phased Ising-Z SELECT case.

        Args:
            system (Vector[Qubit]): Shared SELECT target register.

        Returns:
            Vector[Qubit]: Transformed target register.
        """
        _apply_z_word(system, word)
        return global_phase(_identity_vector, phase)(system)

    return phased_case


def _apply_z_word(system: Vector[Qubit], word: tuple[int, ...]) -> None:
    """Emit one canonical Z word into a target register.

    Args:
        system (Vector[Qubit]): Target system register mutated in place.
        word (tuple[int, ...]): Sorted unique system-qubit indices.
    """
    for index in word:
        system[index] = z(system[index])


def _configure_block_encoding(
    unitary: _BlockEncodingUnitary,
    terms: tuple[_CanonicalTerm, ...],
    num_system_qubits: int,
    num_signal_qubits: int,
    normalization: float,
) -> _BlockEncodingUnitary:
    """Attach stable compiler identity and canonical Ising-Z metadata.

    Args:
        unitary (_BlockEncodingUnitary): Composite qkernel to configure.
        terms (tuple[_CanonicalTerm, ...]): Canonical nonzero Ising-Z terms.
        num_system_qubits (int): Ordered system-register width.
        num_signal_qubits (int): Complete physical signal-register width.
        normalization (float): Positive block normalization.

    Returns:
        _BlockEncodingUnitary: The same configured qkernel.
    """
    semantic_arguments = _semantic_arguments(
        terms,
        num_system_qubits,
        num_signal_qubits,
        normalization,
    )
    digest_input = json.dumps(
        semantic_arguments,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()[:16]
    configure_composite(
        unitary,
        name="ising_z_block_encoding",
        namespace=f"qamomile.stdlib.ising_z.{digest}",
        policy=CallPolicy.PRESERVE_BOX,
        semantic_arguments=semantic_arguments,
    )
    return unitary


def _semantic_arguments(
    terms: tuple[_CanonicalTerm, ...],
    num_system_qubits: int,
    num_signal_qubits: int,
    normalization: float,
) -> dict[str, Any]:
    """Return serializer-safe arguments determining an Ising-Z unitary.

    Args:
        terms (tuple[_CanonicalTerm, ...]): Canonical nonzero Ising-Z terms.
        num_system_qubits (int): Ordered system-register width.
        num_signal_qubits (int): Complete physical signal-register width.
        normalization (float): Positive block normalization.

    Returns:
        dict[str, Any]: Stable primitive semantic argument mapping.
    """
    return {
        "normalization": normalization,
        "num_signal_qubits": num_signal_qubits,
        "num_system_qubits": num_system_qubits,
        "terms": [
            {
                "word": list(word),
                "coefficient": [
                    float(coefficient.real),
                    float(coefficient.imag),
                ],
            }
            for word, coefficient in terms
        ],
    }


_register_lcu_block_encoding_static_binding(
    IsingZBlockEncoding,
    "qamomile.stdlib.ising_z_block_encoding",
)


__all__ = [
    "IsingZBlockEncoding",
    "ising_z_block_encoding",
]
