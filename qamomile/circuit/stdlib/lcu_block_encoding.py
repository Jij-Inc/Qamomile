"""Compose exact block encodings with a recursive linear combination."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.operation.global_phase import (
    GlobalPhaseGate,
    global_phase,
)
from qamomile.circuit.frontend.operation.inverse import inverse
from qamomile.circuit.frontend.operation.qubit_gates import x
from qamomile.circuit.frontend.operation.select import select
from qamomile.circuit.frontend.qkernel import QKernel, qkernel
from qamomile.circuit.stdlib._block_encoding import (
    _BlockEncodingKernel,
    _BlockEncodingLike,
    _validate_block_encoding_kernel,
    _validate_positive_integer,
    _validate_positive_real,
    _validate_register_widths,
    _validated_block_encoding,
)
from qamomile.circuit.stdlib.state_preparation.mottonen_amplitude_encoding import (
    _mottonen_composite,
)


@dataclass(frozen=True, slots=True, eq=False)
class LCUBlockEncoding:
    r"""Describe one static exact block encoding produced by LCU composition.

    ``kernel`` has no classical arguments and uses the quantum ABI
    ``kernel(signal, system) -> (signal, system)``. ``signal`` contains every
    source-level selector, child ancilla, workspace, and pass-through padding
    qubit owned by this LCU. Its all-zero state defines the success projector,
    while ``system`` is the ordered logical register on which the encoded
    matrix acts. Both registers are returned with the same logical wires in
    the same order.

    For the all-zero signal isometry ``V0``, the descriptor represents the
    exact ideal-logical relation

    .. math::

        V_0^\dagger U V_0 = A / \mathtt{normalization}

    including phase. This is the scheme-specific result descriptor for
    :func:`lcu_block_encoding`, not a universal base class or protocol.
    Independent block-encoding schemes remain free to define their own
    descriptors; the composer consumes their common shape structurally.
    Exactness refers to ideal logical semantics and exact arithmetic; host
    floating-point evaluation and backend synthesis roundoff are not exposed
    as an algorithmic ``error_bound`` by this descriptor.

    Instances intentionally use object-identity equality. Frozen fields only
    prevent rebinding; value equality and value hashing are not semantic
    contracts for LCU block encodings.

    Args:
        kernel (QKernel): Exact LCU block-encoding qkernel with the static
            ``(signal, system)`` ABI.
        normalization (float): Finite positive normalization multiplying the
            projected block.
        num_signal_qubits (int): Concrete positive width of the complete LCU
            signal register, including pass-through padding.
        num_system_qubits (int): Concrete positive width of the ordered system
            register.

    Raises:
        TypeError: If the qkernel ABI, normalization type, or width type is
            invalid.
        ValueError: If normalization is non-finite or non-positive, or a width
            is non-positive.
    """

    kernel: _BlockEncodingKernel
    normalization: float
    num_signal_qubits: int
    num_system_qubits: int

    def __post_init__(self) -> None:
        """Validate and normalize the frozen descriptor fields.

        Raises:
            TypeError: If the qkernel ABI, normalization type, or width type is
                invalid.
            ValueError: If normalization is non-finite or non-positive, or a
                width is non-positive.
        """
        _validate_block_encoding_kernel(self.kernel, "kernel")
        object.__setattr__(
            self,
            "normalization",
            _validate_positive_real(self.normalization, "normalization"),
        )
        object.__setattr__(
            self,
            "num_signal_qubits",
            _validate_positive_integer(
                self.num_signal_qubits,
                "num_signal_qubits",
            ),
        )
        object.__setattr__(
            self,
            "num_system_qubits",
            _validate_positive_integer(
                self.num_system_qubits,
                "num_system_qubits",
            ),
        )


def identity_block_encoding(num_system_qubits: int) -> LCUBlockEncoding:
    r"""Create an exact identity LCU encoding with one pass-through signal.

    The returned kernel acts as identity on both registers and satisfies
    ``V0^dagger U V0 = I`` with normalization ``1.0``. One signal qubit is
    retained so the result can be used wherever the current reusable-circuit,
    SELECT, and all-zero-reflection ABI requires a positive signal width.

    Args:
        num_system_qubits (int): Concrete positive system-register width.

    Returns:
        LCUBlockEncoding: Exact identity encoding with one signal qubit.

    Raises:
        TypeError: If ``num_system_qubits`` is not an integer.
        ValueError: If ``num_system_qubits`` is non-positive.
    """
    system_width = _validate_positive_integer(
        num_system_qubits,
        "num_system_qubits",
    )

    @qkernel
    def kernel(
        signal: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Apply the identity block-encoding body.

        Args:
            signal (Vector[Qubit]): One-qubit pass-through signal register.
            system (Vector[Qubit]): Pass-through system register.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Unchanged signal and system
                registers.

        Raises:
            TypeError: If either argument is not a vector register.
            ValueError: If either concrete register width is incorrect.
        """
        _validate_register_widths(
            signal,
            system,
            expected_signal=1,
            expected_system=system_width,
            owner="Identity block encoding",
        )
        return signal, system

    # The name is diagnostic only and does not define semantic identity.
    kernel.name = "identity_block_encoding"
    return LCUBlockEncoding(
        kernel=kernel,
        normalization=1.0,
        num_signal_qubits=1,
        num_system_qubits=system_width,
    )


@dataclass(frozen=True, slots=True, eq=False)
class LCUBlockEncodingTerm:
    r"""Pair a logical LCU coefficient with one child block encoding.

    ``coefficient`` multiplies the child target matrix ``A_j``, not the
    normalized projected block ``A_j / alpha_j``. Consequently the composer
    weights this term by ``abs(coefficient) * child.normalization``. The child
    may be an :class:`LCUBlockEncoding` or an independent scheme-specific
    descriptor exposing the same structural fields.

    Instances intentionally use object-identity equality. Frozen fields only
    prevent rebinding; value equality and value hashing are not semantic
    contracts for LCU terms.

    Args:
        coefficient (complex): Finite logical coefficient. Zero terms are
            removed by :func:`lcu_block_encoding` before nonzero circuit
            construction. A nonempty list containing only zero terms creates
            an exact zero encoding after its child system widths are checked.
        encoding (object): Exact child descriptor exposing ``kernel``,
            ``normalization``, ``num_signal_qubits``, and
            ``num_system_qubits``.

    Raises:
        TypeError: If the coefficient type or child descriptor structure is
            invalid.
        ValueError: If the coefficient is non-finite or child metadata is
            outside its finite positive range.
    """

    coefficient: complex
    encoding: object

    def __post_init__(self) -> None:
        """Validate and normalize the frozen term fields.

        Raises:
            TypeError: If the coefficient type or child descriptor structure
                is invalid.
            ValueError: If the coefficient is non-finite or child metadata is
                outside its finite positive range.
        """
        object.__setattr__(
            self,
            "coefficient",
            _validate_finite_complex(self.coefficient, "coefficient"),
        )
        _validated_block_encoding(self.encoding, "encoding")


@dataclass(frozen=True, slots=True)
class _ValidatedTerm:
    """Store one normalized term for construction-time lowering.

    Args:
        coefficient (complex): Finite logical coefficient.
        encoding (_BlockEncodingLike): Structurally validated child.
        normalization (float): Child normalization.
        signal_width (int): Child signal-register width.
        system_width (int): Child system-register width.
    """

    coefficient: complex
    encoding: _BlockEncodingLike
    normalization: float
    signal_width: int
    system_width: int


def lcu_block_encoding(
    terms: Sequence[LCUBlockEncodingTerm],
) -> LCUBlockEncoding:
    r"""Compose an ordered LCU of exact child block encodings.

    Given child encodings ``U_j`` satisfying

    .. math::

        V_{0,j}^\dagger U_j V_{0,j} = A_j / \alpha_j,

    and logical coefficients ``c_j``, this factory encodes

    .. math::

        A = \sum_j c_j A_j,
        \qquad
        \Lambda = \sum_j |c_j| \alpha_j.

    PREPARE amplitudes are ``sqrt(abs(c_j) * alpha_j / Lambda)`` and SELECT
    case ``j`` is ``exp(1j * arg(c_j)) * U_j``. Child signal registers share a
    pool whose width is the maximum child width. Each uniform-signature case
    routes only the leading child-sized slice and acts exactly as identity on
    unused padding. The parent signal layout is the private ordered lowering
    ``[outer selector | shared child pool]``; callers need only allocate the
    reported flat ``num_signal_qubits`` register and project all of it onto
    zero.

    Zero-coefficient terms do not participate in normalization, PREPARE, or
    SELECT. A nonempty term sequence containing only zeros creates an exact
    zero encoding with normalization ``1.0`` and one signal qubit; all child
    system widths are checked because they determine that zero operator's
    domain. An empty sequence is rejected because its system width is
    unknowable.

    The returned descriptor can itself be supplied as a later term, enabling
    recursive block encodings without forwarding child angles, coefficient
    arrays, register widths, or other classical construction arguments through
    the public qkernel ABI. Children must denote unitary qkernels whose IR can
    be derived under inverse and control. The stated block equality is an
    ideal-logical, exact-arithmetic contract; host floating-point evaluation
    and backend synthesis roundoff are outside this exact composer contract.

    Args:
        terms (Sequence[LCUBlockEncodingTerm]): Nonempty ordered child terms.
            Every active child must use the same ordered system-register width.
            If every coefficient is zero, every child must use the same width.

    Returns:
        LCUBlockEncoding: Exact recursively composable LCU block encoding.

    Raises:
        TypeError: If ``terms`` is not an ordered sequence or contains a
            non-term value.
        ValueError: If ``terms`` is empty, relevant child system widths differ,
            or the composed normalization overflows or is non-finite.

    Example:
        >>> import qamomile.circuit as qmc
        >>> identity = qmc.identity_block_encoding(1)
        >>> encoding = qmc.lcu_block_encoding(
        ...     [
        ...         qmc.LCUBlockEncodingTerm(2.0, identity),
        ...         qmc.LCUBlockEncodingTerm(-0.5j, identity),
        ...     ]
        ... )
        >>> encoding.normalization
        2.5
    """
    validated_terms = _validate_terms(terms)
    active_terms = tuple(term for term in validated_terms if term.coefficient != 0.0j)
    domain_terms = active_terms if active_terms else validated_terms
    system_width = _common_system_width(domain_terms)

    if not active_terms:
        normalization = 1.0
        signal_width = 1
        kernel = _build_zero_encoding(system_width)
    else:
        normalization = _lcu_normalization(active_terms)
        if len(active_terms) == 1:
            signal_width = active_terms[0].signal_width
            kernel = _build_single_term_encoding(active_terms[0])
        else:
            selector_width = (len(active_terms) - 1).bit_length()
            child_pool_width = max(term.signal_width for term in active_terms)
            signal_width = selector_width + child_pool_width
            kernel = _build_multi_term_encoding(
                active_terms,
                normalization,
                signal_width,
                system_width,
            )

    # This display name is diagnostic only; it is not stable semantic identity.
    kernel.name = "lcu_block_encoding"
    return LCUBlockEncoding(
        kernel=kernel,
        normalization=normalization,
        num_signal_qubits=signal_width,
        num_system_qubits=system_width,
    )


def _validate_terms(
    terms: object,
) -> tuple[_ValidatedTerm, ...]:
    """Validate a nonempty ordered term collection without dropping zeros.

    Args:
        terms (object): Candidate ordered terms.

    Returns:
        tuple[_ValidatedTerm, ...]: Validated terms in their original order.

    Raises:
        TypeError: If ``terms`` is not an ordered sequence or contains a
            non-term value.
        ValueError: If the sequence is empty.
    """
    if not isinstance(terms, Sequence) or isinstance(
        terms,
        (str, bytes, bytearray),
    ):
        raise TypeError("terms must be an ordered sequence of LCU terms.")
    candidates = tuple(terms)
    if not candidates:
        raise ValueError(
            "LCU block encoding requires at least one term so the system "
            "width can be determined."
        )

    normalized: list[_ValidatedTerm] = []
    for index, term in enumerate(candidates):
        if not isinstance(term, LCUBlockEncodingTerm):
            raise TypeError(
                "terms must contain only LCUBlockEncodingTerm values; "
                f"term {index} is {type(term).__name__}."
            )
        encoding, alpha, signal_width, system_width = _validated_block_encoding(
            term.encoding,
            f"terms[{index}].encoding",
        )
        normalized.append(
            _ValidatedTerm(
                coefficient=term.coefficient,
                encoding=encoding,
                normalization=alpha,
                signal_width=signal_width,
                system_width=system_width,
            )
        )
    return tuple(normalized)


def _common_system_width(terms: tuple[_ValidatedTerm, ...]) -> int:
    """Return the common ordered system width for relevant terms.

    Args:
        terms (tuple[_ValidatedTerm, ...]): Nonempty active terms, or all terms
            when constructing a zero encoding.

    Returns:
        int: Common positive system-register width.

    Raises:
        ValueError: If two terms use different system widths.
    """
    system_width = terms[0].system_width
    for index, term in enumerate(terms[1:], start=1):
        if term.system_width != system_width:
            raise ValueError(
                "LCU block encoding requires every relevant child to use the "
                f"same system width; term 0 uses {system_width} qubits but "
                f"term {index} uses {term.system_width}."
            )
    return system_width


def _lcu_normalization(terms: tuple[_ValidatedTerm, ...]) -> float:
    """Compute the finite positive weighted child normalization.

    Args:
        terms (tuple[_ValidatedTerm, ...]): Validated nonzero terms.

    Returns:
        float: ``sum(abs(c_j) * alpha_j)``.

    Raises:
        ValueError: If a weighted contribution or final sum overflows or is
            non-finite.
    """
    contributions: list[float] = []
    for index, term in enumerate(terms):
        try:
            contribution = abs(term.coefficient) * term.normalization
        except OverflowError as exc:
            raise ValueError(
                f"LCU block-encoding normalization overflowed at term {index}."
            ) from exc
        if not math.isfinite(contribution):
            raise ValueError(
                f"LCU block-encoding normalization is non-finite at term {index}."
            )
        contributions.append(contribution)
    try:
        normalization = math.fsum(contributions)
    except OverflowError as exc:
        raise ValueError("LCU block-encoding normalization overflowed.") from exc
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError(
            "LCU block-encoding normalization must be finite and positive."
        )
    return normalization


def _build_zero_encoding(system_width: int) -> _BlockEncodingKernel:
    """Build an exact zero block using one signal-qubit bit flip.

    Args:
        system_width (int): Positive ordered system-register width.

    Returns:
        _BlockEncodingKernel: Zero block-encoding kernel.
    """

    @qkernel
    def kernel(
        signal: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Apply the exact zero block-encoding body.

        Args:
            signal (Vector[Qubit]): One-qubit signal register.
            system (Vector[Qubit]): Pass-through system register.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Orthogonalized signal and
                unchanged system registers.

        Raises:
            TypeError: If either argument is not a vector register.
            ValueError: If either concrete register width is incorrect.
        """
        _validate_register_widths(
            signal,
            system,
            expected_signal=1,
            expected_system=system_width,
            owner="LCU block encoding",
        )
        signal[0] = x(signal[0])
        return signal, system

    return kernel


def _build_single_term_encoding(term: _ValidatedTerm) -> _BlockEncodingKernel:
    """Build a selector-free phased child encoding for one active term.

    Args:
        term (_ValidatedTerm): Sole active LCU term.

    Returns:
        _BlockEncodingKernel: Parent kernel with the child's signal width.
    """
    child = _phased_child(term)
    signal_width = term.signal_width
    system_width = term.system_width

    @qkernel
    def kernel(
        signal: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Apply one phased child without PREPARE or SELECT.

        Args:
            signal (Vector[Qubit]): Child signal register.
            system (Vector[Qubit]): Shared system register.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Updated signal and system
                registers.

        Raises:
            TypeError: If either argument is not a vector register.
            ValueError: If either concrete register width is incorrect.
        """
        _validate_register_widths(
            signal,
            system,
            expected_signal=signal_width,
            expected_system=system_width,
            owner="LCU block encoding",
        )
        return child(signal, system)

    return kernel


def _build_multi_term_encoding(
    terms: tuple[_ValidatedTerm, ...],
    normalization: float,
    signal_width: int,
    system_width: int,
) -> _BlockEncodingKernel:
    """Build a recursive PREPARE-SELECT-PREPARE-dagger encoding.

    Args:
        terms (tuple[_ValidatedTerm, ...]): Ordered active child terms.
        normalization (float): Weighted parent normalization.
        signal_width (int): Complete parent signal width.
        system_width (int): Shared child system width.

    Returns:
        _BlockEncodingKernel: Recursive multi-term LCU kernel.

    Raises:
        RuntimeError: If Möttönen preparation reports an unexpected width.
    """
    selector_width = (len(terms) - 1).bit_length()
    child_pool_width = max(term.signal_width for term in terms)
    amplitudes = np.zeros(1 << selector_width, dtype=np.float64)
    sqrt_normalization = math.sqrt(normalization)
    for index, term in enumerate(terms):
        amplitudes[index] = (
            math.sqrt(abs(term.coefficient) * term.normalization) / sqrt_normalization
        )

    preparation, required_width = _mottonen_composite(
        amplitudes,
        preserve_unitary_completion=True,
    )
    if required_width != selector_width:
        raise RuntimeError(
            "Möttönen preparation width disagrees with the LCU selector width."
        )
    unprepare = inverse(preparation)
    selector = select(
        tuple(_build_uniform_case(term, child_pool_width) for term in terms),
        num_index_qubits=selector_width,
    )

    @qkernel
    def kernel(
        signal: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Apply the recursive multi-term LCU body.

        Args:
            signal (Vector[Qubit]): Flat selector-plus-child-pool register.
            system (Vector[Qubit]): Shared ordered system register.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Updated flat signal and system
                registers.

        Raises:
            TypeError: If either argument is not a vector register.
            ValueError: If either concrete register width is incorrect.
        """
        _validate_register_widths(
            signal,
            system,
            expected_signal=signal_width,
            expected_system=system_width,
            owner="LCU block encoding",
        )
        signal[:selector_width] = preparation(signal[:selector_width])
        signal[:selector_width], signal[selector_width:], system = selector(
            signal[:selector_width],
            signal[selector_width:],
            system,
        )
        signal[:selector_width] = unprepare(signal[:selector_width])
        return signal, system

    return kernel


def _build_uniform_case(
    term: _ValidatedTerm,
    child_pool_width: int,
) -> QKernel[..., tuple[Vector[Qubit], Vector[Qubit]]]:
    """Wrap one child in the uniform shared-pool SELECT signature.

    Args:
        term (_ValidatedTerm): Child term to route.
        child_pool_width (int): Width shared by every SELECT case.

    Returns:
        QKernel[..., tuple[Vector[Qubit], Vector[Qubit]]]: Wrapper acting on the
            child's leading signal slice and preserving unused padding.
    """
    child = _phased_child(term)
    child_width = term.signal_width

    @qkernel
    def case(
        signal: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Route one child through its leading shared-pool slice.

        Args:
            signal (Vector[Qubit]): Uniform shared child-signal pool.
            system (Vector[Qubit]): Shared ordered system register.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Updated pool and system,
                preserving every unused pool qubit exactly.

        Raises:
            TypeError: If ``signal`` is not a vector register.
            ValueError: If the concrete shared pool has an unexpected width.
        """
        _validate_pool_width(signal, child_pool_width)
        signal[:child_width], system = child(
            signal[:child_width],
            system,
        )
        return signal, system

    return case


def _phased_child(
    term: _ValidatedTerm,
) -> _BlockEncodingKernel | GlobalPhaseGate:
    """Return the child unitary with its logical coefficient phase.

    Args:
        term (_ValidatedTerm): Nonzero term whose phase should be applied.

    Returns:
        _BlockEncodingKernel | GlobalPhaseGate: Child kernel multiplied by
            ``coefficient / abs(coefficient)``.
    """
    phase = math.atan2(term.coefficient.imag, term.coefficient.real)
    if not phase:
        return term.encoding.kernel
    return global_phase(term.encoding.kernel, phase)


def _validate_pool_width(child_pool: Vector[Qubit], expected: int) -> None:
    """Validate a concrete shared child-pool width.

    Args:
        child_pool (Vector[Qubit]): Pool passed to one SELECT case.
        expected (int): Required uniform pool width.

    Raises:
        TypeError: If ``child_pool`` is not a vector register.
        ValueError: If the concrete pool width differs from ``expected``.
    """
    from qamomile.circuit.frontend.handle.utils import get_size

    try:
        actual = get_size(child_pool)
    except ValueError:
        return
    if actual != expected:
        unit = "qubit" if expected == 1 else "qubits"
        raise ValueError(
            "LCU block-encoding SELECT case requires "
            f"{expected} child-pool {unit}, got {actual}."
        )


def _validate_finite_complex(value: object, name: str) -> complex:
    """Validate one finite non-boolean complex scalar.

    Args:
        value (object): Candidate coefficient.
        name (str): Argument name used in diagnostics.

    Returns:
        complex: Equivalent Python complex with signed zeros canonicalized.

    Raises:
        TypeError: If ``value`` is not a complex numeric scalar.
        ValueError: If conversion overflows or either component is non-finite.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (numbers.Complex, np.integer, np.floating, np.complexfloating),
    ):
        raise TypeError(f"{name} must be a complex numeric scalar.")
    try:
        normalized = complex(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(normalized.real) or not math.isfinite(normalized.imag):
        raise ValueError(f"{name} must be finite.")
    real = normalized.real if normalized.real else 0.0
    imag = normalized.imag if normalized.imag else 0.0
    return complex(real, imag)


__all__ = [
    "LCUBlockEncoding",
    "LCUBlockEncodingTerm",
    "identity_block_encoding",
    "lcu_block_encoding",
]
