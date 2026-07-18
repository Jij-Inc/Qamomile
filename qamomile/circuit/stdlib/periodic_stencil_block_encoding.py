r"""LCU block encodings for constant-coefficient periodic stencils.

For axis registers with widths ``register_sizes``, this module represents

.. math::

    A = \sum_k c_k T_k,

where ``T_k`` adds the integer offset tuple ``k`` modulo the corresponding
power-of-two axis sizes.  The construction prepares real amplitudes
``sqrt(abs(c_k) / lambda)``, applies
``exp(1j * arg(c_k)) * T_k`` through ``qmc.select``, and unprepares the signal
register.  Consequently the all-zero signal block is ``A / lambda`` for
``lambda = sum(abs(c_k))``.

Only periodic boundaries, constant coefficients, and power-of-two axis sizes
are in scope.  Non-periodic boundary corrections and position-dependent
coefficients require different block-encoding constructions.  Constant shifts
are emitted as repeated ancilla-free unit increments or decrements, so this
factory targets local stencils rather than dense circulant kernels with large
displacements.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Number
from typing import Any, cast

import numpy as np

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.ir.operation.callable import CallPolicy
from qamomile.circuit.stdlib.lcu_block_encoding import (
    LCUBlockEncoding,
    _BlockEncodingUnitary,
)
from qamomile.circuit.stdlib.modular_incdec import (
    _apply_fixed_window_periodic_shift,
)
from qamomile.circuit.stdlib.state_preparation.mottonen_amplitude_encoding import (
    _mottonen_composite,
)

Offset = int | tuple[int, ...]
_NORMALIZATION_REL_TOLERANCE = 1e-12


def _is_integer(value: Any) -> bool:
    """Return whether a runtime value is an integer but not a boolean.

    Args:
        value (Any): Runtime value to inspect.

    Returns:
        bool: Whether the value implements ``numbers.Integral`` without being
        the Boolean subclass of ``int``.
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, Integral)


@dataclass(frozen=True, slots=True, eq=False)
class PeriodicStencilBlockEncoding(LCUBlockEncoding):
    r"""Describe one static exact periodic-stencil block encoding.

    ``unitary`` is the qkernel implementing the larger unitary ``U``; it is
    neither the encoded stencil matrix ``A`` nor a dense matrix value. It has
    no classical arguments and its quantum ABI is
    ``unitary(signal, system) -> (signal, system)``. ``system`` is the ordered
    flattened data register on which ``A`` acts. ``signal`` is the complete
    source-level ancilla bundle whose all-zero state selects the encoded
    block. The unitary returns the same logical wires in the same order and
    acts unitarily for arbitrary signal inputs; signal need not return to zero
    after one application.

    For the all-zero signal isometry ``V0``, the encoded discrete periodic
    stencil satisfies

    .. math::

        V_0^\dagger U V_0 = A / \mathtt{normalization}

    including coefficient phase. The construction is exact in ideal logical
    arithmetic; host floating-point roundoff in state-preparation angles and
    backend gate synthesis are outside this semantic equality. This producer
    allocates no hidden source-level logical qubits. Backend decomposition
    scratch is permitted only when resource-accounted and exactly uncomputed
    for every input, including under inverse and control. Descriptor comparison
    and hashing use object identity rather than field values.

    The inherited fields form the qkernel-visible static LCU contract.
    ``register_sizes``, ``offsets``, and ``coefficients`` are deeply immutable
    producer metadata for host-side inspection only. Reusable qkernels should
    annotate descriptor arguments with :class:`LCUBlockEncoding`, allowing the
    same serialized template to accept this and other exact LCU producers.

    Args:
        unitary (qmc.QKernel): QKernel implementing the block-encoding unitary
            ``U`` with the static ``(signal, system)`` ABI.
        normalization (float): Finite positive LCU normalization
            ``sum(abs(coefficients))`` after equivalent periodic offsets are
            combined. Direct construction accepts relative disagreement up to
            ``1e-12`` for host rounding and stores the canonical
            coefficient-derived sum.
        num_signal_qubits (int): Concrete positive width of the complete signal
            register, including selector padding.
        num_system_qubits (int): Concrete positive width of the ordered flat
            system register.
        register_sizes (tuple[int, ...]): Qubit widths of the flattened system
            register's axes.
        offsets (tuple[tuple[int, ...], ...]): Canonical modular offsets, in
            SELECT case order.
        coefficients (tuple[complex, ...]): Nonzero combined coefficients, in
            SELECT case order.

    Raises:
        TypeError: If the common block-encoding fields or method-specific
            metadata have invalid runtime types.
        ValueError: If normalization or a width is invalid, method-specific
            metadata is inconsistent, or offsets are not canonical.
    """

    register_sizes: tuple[int, ...]
    offsets: tuple[tuple[int, ...], ...]
    coefficients: tuple[complex, ...]

    def __post_init__(self) -> None:
        """Validate and normalize the immutable descriptor fields.

        Raises:
            TypeError: If a common field or method-specific metadata value has
                an invalid runtime type.
            ValueError: If a common field is outside its valid range or the
                method-specific metadata is inconsistent.
        """
        LCUBlockEncoding.__post_init__(self)
        (
            register_sizes,
            offsets,
            coefficients,
            normalization,
        ) = _validate_descriptor_metadata(
            self.register_sizes,
            self.offsets,
            self.coefficients,
            self.normalization,
            self.num_signal_qubits,
            self.num_system_qubits,
        )
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "register_sizes", register_sizes)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "coefficients", coefficients)


def _validate_register_sizes(register_sizes: Sequence[Any]) -> tuple[int, ...]:
    """Validate and freeze the per-axis qubit widths.

    Args:
        register_sizes (Sequence[int]): Qubit width for each periodic axis.

    Returns:
        tuple[int, ...]: Validated positive widths.

    Raises:
        ValueError: If no axis is supplied or an axis has non-positive width.
        TypeError: If a width is not a non-Boolean integer.
    """
    raw_sizes = tuple(register_sizes)
    if not raw_sizes:
        raise ValueError("register_sizes must contain at least one axis")
    sizes: list[int] = []
    for axis, width in enumerate(raw_sizes):
        if not _is_integer(width):
            raise TypeError(
                f"register_sizes[{axis}] must be an int, got {type(width).__name__}"
            )
        normalized = int(width)
        if normalized <= 0:
            raise ValueError(
                f"register_sizes[{axis}] must be positive, got {normalized}"
            )
        sizes.append(normalized)
    return tuple(sizes)


def _validate_descriptor_metadata(
    register_sizes: object,
    offsets: object,
    coefficients: object,
    normalization: float,
    num_signal_qubits: int,
    num_system_qubits: int,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[complex, ...],
    float,
]:
    """Validate periodic metadata stored alongside one generated unitary.

    Args:
        register_sizes (object): Candidate tuple of per-axis qubit widths.
        offsets (object): Candidate canonical SELECT-order offset tuple.
        coefficients (object): Candidate combined coefficient tuple.
        normalization (float): Validated common block normalization.
        num_signal_qubits (int): Validated common signal width.
        num_system_qubits (int): Validated common system width.

    Returns:
        tuple: Validated register sizes, offsets, coefficients, and canonical
            coefficient-derived normalization, in that order.

    Raises:
        TypeError: If metadata is not represented by the documented tuple and
            scalar types.
        ValueError: If metadata is empty, noncanonical, or inconsistent with
            the common descriptor fields.
    """
    if not isinstance(register_sizes, tuple):
        raise TypeError("register_sizes must be a tuple of plain integers.")
    sizes = _validate_register_sizes(register_sizes)
    if num_system_qubits != sum(sizes):
        raise ValueError("num_system_qubits must equal the sum of register_sizes.")
    if not isinstance(offsets, tuple):
        raise TypeError("offsets must be a tuple of canonical offset tuples.")
    if not isinstance(coefficients, tuple):
        raise TypeError("coefficients must be a tuple of nonzero complex values.")
    if not offsets or len(offsets) != len(coefficients):
        raise ValueError("offsets and coefficients must have the same nonzero length.")

    canonical_offsets: list[tuple[int, ...]] = []
    canonical_coefficients: list[complex] = []
    for index, (offset, coefficient) in enumerate(
        zip(offsets, coefficients, strict=True)
    ):
        if not isinstance(offset, tuple) or len(offset) != len(sizes):
            raise ValueError(
                f"offsets[{index}] must contain one component per register axis."
            )
        for axis, (component, width) in enumerate(zip(offset, sizes, strict=True)):
            if type(component) is not int:
                raise TypeError(f"offsets[{index}][{axis}] must be a plain integer.")
            if component < 0 or component >= 1 << width:
                raise ValueError(
                    f"offsets[{index}][{axis}] is not a canonical modular residue."
                )
        value = _coerce_coefficient(coefficient, offset)
        if not value:
            raise ValueError("descriptor coefficients must be nonzero.")
        canonical_offsets.append(offset)
        canonical_coefficients.append(value)

    frozen_offsets = tuple(canonical_offsets)
    if frozen_offsets != tuple(sorted(frozen_offsets)) or len(
        set(frozen_offsets)
    ) != len(frozen_offsets):
        raise ValueError("descriptor offsets must be unique and canonically sorted.")
    frozen_coefficients = tuple(canonical_coefficients)
    expected_signal = max(1, (len(frozen_offsets) - 1).bit_length())
    if num_signal_qubits != expected_signal:
        raise ValueError(
            "num_signal_qubits does not match the number of canonical terms."
        )
    canonical_normalization = _normalization(
        tuple(zip(frozen_offsets, frozen_coefficients, strict=True))
    )
    if not math.isclose(
        normalization,
        canonical_normalization,
        rel_tol=_NORMALIZATION_REL_TOLERANCE,
        abs_tol=0.0,
    ):
        raise ValueError(
            "normalization must equal the sum of absolute canonical coefficients."
        )
    return sizes, frozen_offsets, frozen_coefficients, canonical_normalization


def _canonical_offset(
    offset: Any,
    register_sizes: tuple[int, ...],
) -> tuple[int, ...]:
    """Convert one user offset to canonical modular residues.

    Args:
        offset (int | tuple[int, ...]): Integer for a one-dimensional stencil,
            or one integer per axis for a multidimensional stencil.
        register_sizes (tuple[int, ...]): Validated axis qubit widths.

    Returns:
        tuple[int, ...]: Residues modulo each axis dimension.

    Raises:
        TypeError: If the offset form or one component is not an integer.
        ValueError: If a tuple has the wrong dimensionality.
    """
    dimensions = len(register_sizes)
    if dimensions == 1 and _is_integer(offset):
        components = (int(offset),)
    elif isinstance(offset, tuple):
        if len(offset) != dimensions:
            raise ValueError(
                f"offset {offset!r} has {len(offset)} dimensions; expected {dimensions}"
            )
        components = offset
    else:
        expected = (
            "an int or one-element tuple"
            if dimensions == 1
            else (f"a {dimensions}-element tuple")
        )
        raise TypeError(f"offset must be {expected}, got {offset!r}")

    residues: list[int] = []
    for axis, (component, width) in enumerate(
        zip(components, register_sizes, strict=True)
    ):
        if not _is_integer(component):
            raise TypeError(
                f"offset component {axis} must be an int, got "
                f"{type(component).__name__}"
            )
        residues.append(int(component) % (1 << width))
    return tuple(residues)


def _canonical_terms(
    coefficients: Mapping[Any, Any],
    register_sizes: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], complex], ...]:
    """Combine equivalent periodic offsets and discard exact cancellations.

    Args:
        coefficients (Mapping[int | tuple[int, ...], complex]): Stencil
            coefficients indexed by offsets.
        register_sizes (tuple[int, ...]): Validated axis qubit widths.

    Returns:
        tuple[tuple[tuple[int, ...], complex], ...]: Sorted nonzero canonical
        offset-coefficient pairs.

    Raises:
        ValueError: If the mapping is empty, all terms cancel, or a coefficient
            is non-finite.
        TypeError: If a coefficient is not numeric or an offset is invalid.
    """
    if not coefficients:
        raise ValueError("coefficients must contain at least one stencil term")

    grouped: dict[tuple[int, ...], list[complex]] = {}
    for offset, coefficient in coefficients.items():
        value = _coerce_coefficient(coefficient, offset)
        canonical = _canonical_offset(offset, register_sizes)
        grouped.setdefault(canonical, []).append(value)

    combined = tuple(
        (offset, _sum_complex_values(values, offset))
        for offset, values in sorted(grouped.items())
    )
    terms = tuple(
        (offset, coefficient) for offset, coefficient in combined if coefficient
    )
    if not terms:
        raise ValueError("periodic stencil is the zero operator after combining terms")
    return terms


def _coerce_coefficient(coefficient: Any, offset: Any) -> complex:
    """Validate one coefficient and canonicalize signed zero components.

    Args:
        coefficient (Any): Candidate numeric stencil coefficient.
        offset (Any): Source offset used in diagnostics.

    Returns:
        complex: Finite complex coefficient with positive signed zeros.

    Raises:
        TypeError: If the coefficient is Boolean, nonnumeric, or cannot be
            converted to complex.
        ValueError: If conversion overflows or a component is non-finite.
    """
    if isinstance(coefficient, bool) or not isinstance(coefficient, Number):
        raise TypeError(
            f"coefficient for offset {offset!r} must be numeric, got "
            f"{type(coefficient).__name__}"
        )
    try:
        value = complex(cast(Any, coefficient))
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"coefficient for offset {offset!r} cannot be converted to complex"
        ) from error
    except OverflowError as error:
        raise ValueError(
            f"coefficient for offset {offset!r} must be representable as a "
            "finite complex number"
        ) from error
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError(
            f"coefficient for offset {offset!r} must be finite, got {coefficient!r}"
        )
    real = value.real if value.real else 0.0
    imag = value.imag if value.imag else 0.0
    return complex(real, imag)


def _sum_complex_values(
    values: Sequence[complex],
    offset: tuple[int, ...],
) -> complex:
    """Combine equivalent-offset coefficients in canonical numeric order.

    Args:
        values (Sequence[complex]): Finite values assigned to one residue.
        offset (tuple[int, ...]): Canonical residue used in diagnostics.

    Returns:
        complex: Accurately summed coefficient with signed zero canonicalized.

    Raises:
        ValueError: If finite inputs overflow during canonical summation.
    """
    ordered = sorted(values, key=lambda value: (value.real, value.imag))
    try:
        real = math.fsum(value.real for value in ordered)
        imag = math.fsum(value.imag for value in ordered)
    except OverflowError as error:
        raise ValueError(
            f"combined coefficient for offset {offset!r} must be finite"
        ) from error
    if not math.isfinite(real) or not math.isfinite(imag):
        raise ValueError(f"combined coefficient for offset {offset!r} must be finite")
    return complex(real if real else 0.0, imag if imag else 0.0)


def _normalization(
    terms: Sequence[tuple[tuple[int, ...], complex]],
) -> float:
    """Compute the finite positive LCU normalization of canonical terms.

    Args:
        terms (Sequence[tuple[tuple[int, ...], complex]]): Canonically ordered
            nonzero offset-coefficient pairs.

    Returns:
        float: Sum of coefficient magnitudes.

    Raises:
        ValueError: If the magnitude sum overflows or is not finite and
            positive.
    """
    try:
        normalization = math.fsum(abs(coefficient) for _, coefficient in terms)
    except OverflowError as error:
        raise ValueError("periodic stencil normalization must be finite") from error
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("periodic stencil normalization must be finite and positive")
    return normalization


def _shortest_signed_shift(residue: int, width: int) -> int:
    """Return the shortest signed representative of a modular offset.

    Args:
        residue (int): Canonical offset in ``range(2**width)``.
        width (int): Axis register width in qubits.

    Returns:
        int: Positive increment count or negative decrement count.
    """
    modulus = 1 << width
    return residue if residue <= modulus // 2 else residue - modulus


def _apply_constant_periodic_shift(
    axis: qmc.Vector[qmc.Qubit] | qmc.VectorView[qmc.Qubit],
    shift: int,
) -> qmc.Vector[qmc.Qubit] | qmc.VectorView[qmc.Qubit]:
    """Apply a compile-time constant periodic shift to one axis register.

    Args:
        axis (qmc.Vector[qmc.Qubit] | qmc.VectorView[qmc.Qubit]): Axis register
            interpreted as an LSB-first unsigned integer.
        shift (int): Signed modular displacement. Positive values increment;
            negative values decrement.

    Returns:
        qmc.Vector[qmc.Qubit] | qmc.VectorView[qmc.Qubit]: Updated axis handle.
    """
    primitive = qmc.modular_increment if shift >= 0 else qmc.modular_decrement
    for _ in range(abs(shift)):
        axis = primitive(axis)
    return axis


def _apply_multidimensional_shift(
    system: qmc.Vector[qmc.Qubit],
    register_sizes: tuple[int, ...],
    signed_shifts: tuple[int, ...],
) -> qmc.Vector[qmc.Qubit]:
    """Apply captured constant shifts to a flattened multidimensional register.

    This helper is ordinary trace-time Python rather than a qkernel.  Its loop
    therefore unrolls the fixed axis layout while the emitted operations remain
    ordinary Qamomile qkernel calls.

    Args:
        system (qmc.Vector[qmc.Qubit]): Flattened axis register.
        register_sizes (tuple[int, ...]): Qubit width of every axis.
        signed_shifts (tuple[int, ...]): Shortest signed displacement per axis.

    Returns:
        qmc.Vector[qmc.Qubit]: Shifted flattened register.
    """
    if len(register_sizes) == 1:
        return _apply_constant_periodic_shift(system, signed_shifts[0])  # type: ignore[return-value]

    start = 0
    for width, shift in zip(register_sizes, signed_shifts, strict=True):
        stop = start + width
        if shift:
            system = _apply_fixed_window_periodic_shift(system, start, width, shift)
        start = stop
    return system


def _make_shift_case(
    offset: tuple[int, ...],
    coefficient: complex,
    register_sizes: tuple[int, ...],
) -> qmc.QKernel:
    """Build one phased multidimensional shift qkernel for SELECT.

    Args:
        offset (tuple[int, ...]): Canonical modular offset per axis.
        coefficient (complex): Nonzero coefficient whose argument becomes the
            case global phase.
        register_sizes (tuple[int, ...]): Axis widths within the flat system
            register.

    Returns:
        qmc.QKernel: Qkernel implementing
        ``exp(1j * arg(coefficient)) * T_offset``.
    """
    signed_shifts = tuple(
        _shortest_signed_shift(residue, width)
        for residue, width in zip(offset, register_sizes, strict=True)
    )

    @qmc.qkernel
    def shift_case(
        system: qmc.Vector[qmc.Qubit],
    ) -> qmc.Vector[qmc.Qubit]:
        """Apply the captured tensor product of periodic shifts.

        Args:
            system (qmc.Vector[qmc.Qubit]): Flattened axis register.

        Returns:
            qmc.Vector[qmc.Qubit]: Shifted system register.
        """
        return _apply_multidimensional_shift(
            system,
            register_sizes,
            signed_shifts,
        )

    phase = _coefficient_phase(coefficient)
    if not phase:
        return shift_case

    phased_shift = qmc.global_phase(shift_case, phase)

    @qmc.qkernel
    def phased_shift_case(
        system: qmc.Vector[qmc.Qubit],
    ) -> qmc.Vector[qmc.Qubit]:
        """Apply the captured shift and its coefficient phase.

        Args:
            system (qmc.Vector[qmc.Qubit]): Flattened axis register.

        Returns:
            qmc.Vector[qmc.Qubit]: Phased and shifted system register.
        """
        return phased_shift(system)

    return phased_shift_case


def _coefficient_phase(coefficient: complex) -> float:
    """Return one nonzero coefficient phase with signed zero canonicalized.

    Args:
        coefficient (complex): Nonzero canonical stencil coefficient.

    Returns:
        float: Principal coefficient phase, using positive zero for a
            positive-real coefficient.
    """
    phase = math.atan2(coefficient.imag, coefficient.real)
    return phase if phase else 0.0


def _validate_register_widths(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
    expected_signal: int,
    expected_system: int,
) -> None:
    """Validate concrete registers while allowing symbolic cache traces.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Complete public signal register.
        system (qmc.Vector[qmc.Qubit]): Ordered flat system register.
        expected_signal (int): Required signal width.
        expected_system (int): Required system width.

    Raises:
        TypeError: If either argument is not a qubit vector.
        ValueError: If a concrete register width differs from its requirement.
    """
    _validate_register_width(signal, expected_signal, "signal")
    _validate_register_width(system, expected_system, "system")


def _validate_register_width(
    register: qmc.Vector[qmc.Qubit],
    expected: int,
    name: str,
) -> None:
    """Validate one concrete vector width and defer symbolic-width checks.

    Args:
        register (qmc.Vector[qmc.Qubit]): Register to inspect.
        expected (int): Required width.
        name (str): Register name used in diagnostics.

    Raises:
        TypeError: If ``register`` is not a vector.
        ValueError: If its concrete width differs from ``expected``.
    """
    try:
        actual = get_size(register)
    except ValueError:
        return
    if actual != expected:
        unit = "qubit" if expected == 1 else "qubits"
        raise ValueError(
            "periodic stencil block encoding requires "
            f"{expected} {name} {unit}, got {actual}."
        )


def _build_single_term_encoding(
    term: tuple[tuple[int, ...], complex],
    register_sizes: tuple[int, ...],
    num_signal_qubits: int,
) -> _BlockEncodingUnitary:
    """Build an unconditional phased-shift encoding for one stencil term.

    Args:
        term (tuple[tuple[int, ...], complex]): Canonical nonzero stencil term.
        register_sizes (tuple[int, ...]): Per-axis system-register widths.
        num_signal_qubits (int): Positive pass-through signal width.

    Returns:
        _BlockEncodingUnitary: Single-term periodic block-encoding unitary.
    """
    offset, coefficient = term
    phased_shift = _make_shift_case(offset, coefficient, register_sizes)
    num_system_qubits = sum(register_sizes)

    @qmc.qkernel
    def unitary(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Apply one captured phased periodic shift.

        Args:
            signal (qmc.Vector[qmc.Qubit]): Pass-through signal register.
            system (qmc.Vector[qmc.Qubit]): Flattened system register.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Preserved
                signal and shifted system registers.
        """
        _validate_register_widths(
            signal,
            system,
            num_signal_qubits,
            num_system_qubits,
        )
        system = phased_shift(system)
        return signal, system

    return unitary


def _build_multi_term_encoding(
    terms: tuple[tuple[tuple[int, ...], complex], ...],
    register_sizes: tuple[int, ...],
    normalization: float,
    num_signal_qubits: int,
) -> _BlockEncodingUnitary:
    """Build PREPARE-SELECT-PREPARE-dagger for multiple stencil terms.

    Args:
        terms (tuple[tuple[tuple[int, ...], complex], ...]): Canonical nonzero
            periodic stencil terms.
        register_sizes (tuple[int, ...]): Per-axis system-register widths.
        normalization (float): Finite positive sum of coefficient magnitudes.
        num_signal_qubits (int): Required SELECT index width.

    Returns:
        _BlockEncodingUnitary: Multi-term periodic block-encoding unitary.

    Raises:
        RuntimeError: If state preparation disagrees with the SELECT width.
    """
    amplitudes = np.zeros(1 << num_signal_qubits, dtype=np.float64)
    sqrt_normalization = math.sqrt(normalization)
    for index, (_, coefficient) in enumerate(terms):
        amplitudes[index] = math.sqrt(abs(coefficient)) / sqrt_normalization

    preparation, required_signal_qubits = _mottonen_composite(
        amplitudes,
        name="periodic_stencil_prepare",
        policy=CallPolicy.PRESERVE_BOX,
    )
    if required_signal_qubits != num_signal_qubits:
        raise RuntimeError(
            "Möttönen preparation width disagrees with the stencil SELECT width."
        )
    unprepare = qmc.inverse(preparation)
    selector = qmc.select(
        tuple(
            _make_shift_case(offset, coefficient, register_sizes)
            for offset, coefficient in terms
        ),
        num_index_qubits=qmc.uint(num_signal_qubits),
    )
    num_system_qubits = sum(register_sizes)

    @qmc.qkernel
    def unitary(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Apply PREPARE, phased shift SELECT, and inverse PREPARE.

        Args:
            signal (qmc.Vector[qmc.Qubit]): All-zero block signal register.
            system (qmc.Vector[qmc.Qubit]): Flattened system register.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated signal
                and system registers.
        """
        _validate_register_widths(
            signal,
            system,
            num_signal_qubits,
            num_system_qubits,
        )
        signal = preparation(signal)
        signal, system = selector(signal, system)
        signal = unprepare(signal)
        return signal, system

    return unitary


def periodic_stencil_block_encoding(
    coefficients: Mapping[Offset, complex],
    register_sizes: Sequence[int],
) -> PeriodicStencilBlockEncoding:
    """Build an LCU block encoding of a periodic constant-coefficient stencil.

    The system is a flattened concatenation of LSB-first axis registers.  An
    integer offset is accepted for a one-dimensional stencil; multidimensional
    offsets are tuples.  Axis ``j`` wraps modulo ``2**register_sizes[j]``.
    Equivalent offsets are combined before computing the normalization.
    A shift uses the shorter of repeated modular increments or decrements, so
    its gate cost is linear in that shortest displacement.
    Only exact cancellations are removed before ``lambda`` and PREPARE are
    constructed, so every representable nonzero coefficient is preserved.

    The returned descriptor's ``unitary`` expects an all-zero signal register.
    Projecting that register onto all zero before and after the unitary yields
    ``A / lambda``, where ``lambda`` is available as ``result.normalization``.
    A single term retains one pass-through signal qubit for composition but
    omits PREPARE and SELECT entirely.

    Args:
        coefficients (Mapping[int | tuple[int, ...], complex]): Constant
            stencil coefficients keyed by signed or unsigned periodic offsets.
        register_sizes (Sequence[int]): Positive qubit width of every axis in
            flattened-system order.

    Returns:
        PeriodicStencilBlockEncoding: Frozen non-callable descriptor containing
            the generated unitary and method-specific canonical metadata.

    Raises:
        ValueError: If no axes or terms are supplied, an axis width is not
            positive, an offset has the wrong dimension, a coefficient or
            normalization is non-finite, or all terms cancel modulo the axis
            sizes.
        TypeError: If widths or offsets are not integers, or coefficients are
            not numeric.

    Example:
        >>> import qamomile.circuit as qmc
        >>> encoding = qmc.periodic_stencil_block_encoding(
        ...     {-1: 1.0, 0: -2.0, 1: 1.0},
        ...     register_sizes=(3,),
        ... )
        >>> encoding.normalization
        4.0
        >>> @qmc.qkernel
        ... def apply() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        ...     signal = qmc.qubit_array(encoding.num_signal_qubits)
        ...     system = qmc.qubit_array(encoding.num_system_qubits)
        ...     return encoding.unitary(signal, system)
    """
    sizes = _validate_register_sizes(register_sizes)
    terms = _canonical_terms(coefficients, sizes)
    normalization = _normalization(terms)
    num_signal_qubits = max(1, (len(terms) - 1).bit_length())
    if len(terms) == 1:
        unitary = _build_single_term_encoding(
            terms[0],
            sizes,
            num_signal_qubits,
        )
    else:
        unitary = _build_multi_term_encoding(
            terms,
            sizes,
            normalization,
            num_signal_qubits,
        )
    # This display name is diagnostic only. Stable boxed callable identity is
    # intentionally deferred until boxed inverse lowering supports this body.
    unitary.name = "periodic_stencil_block_encoding"
    return PeriodicStencilBlockEncoding(
        unitary=unitary,
        normalization=normalization,
        num_signal_qubits=num_signal_qubits,
        num_system_qubits=sum(sizes),
        register_sizes=sizes,
        offsets=tuple(offset for offset, _ in terms),
        coefficients=tuple(coefficient for _, coefficient in terms),
    )


__all__ = [
    "PeriodicStencilBlockEncoding",
    "periodic_stencil_block_encoding",
]
