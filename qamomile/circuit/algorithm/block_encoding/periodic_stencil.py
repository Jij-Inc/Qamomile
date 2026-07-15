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

import cmath
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Number
from typing import Any, cast

import numpy as np

import qamomile.circuit as qmc
from qamomile.circuit.stdlib.state_preparation.mottonen_amplitude_encoding import (
    _mottonen_composite,
)

Offset = int | tuple[int, ...]


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


@dataclass(frozen=True)
class PeriodicStencilEncoding:
    """Describe one compiled periodic-stencil block-encoding construction.

    Args:
        kernel (qmc.QKernel): Qkernel taking ``(signal, system)`` registers and
            returning them in the same order after applying the block encoding.
        normalization (float): Positive LCU normalization
            ``sum(abs(coefficients))`` after equivalent periodic offsets are
            combined.
        num_signal_qubits (int): Required all-zero signal-register width.
        register_sizes (tuple[int, ...]): Qubit widths of the flattened system
            register's axes.
        offsets (tuple[tuple[int, ...], ...]): Canonical modular offsets, in
            SELECT case order.
        coefficients (tuple[complex, ...]): Nonzero combined coefficients, in
            SELECT case order.
    """

    kernel: qmc.QKernel
    normalization: float
    num_signal_qubits: int
    register_sizes: tuple[int, ...]
    offsets: tuple[tuple[int, ...], ...]
    coefficients: tuple[complex, ...]

    @property
    def num_system_qubits(self) -> int:
        """Return the required flattened system-register width.

        Returns:
            int: Sum of the per-axis qubit widths.
        """
        return sum(self.register_sizes)

    def __call__(
        self,
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Apply the generated block-encoding qkernel.

        Args:
            signal (qmc.Vector[qmc.Qubit]): All-zero signal register with
                ``num_signal_qubits`` qubits.
            system (qmc.Vector[qmc.Qubit]): Flattened axis registers with
                ``num_system_qubits`` qubits in total.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated signal
            and system registers.

        Raises:
            ValueError: If a concretely sized signal or system register does
                not match the factory's required width.
        """
        _validate_register_width(signal, self.num_signal_qubits, "signal")
        _validate_register_width(system, self.num_system_qubits, "system")
        return self.kernel(signal, system)


def _concrete_register_width(register: qmc.Vector[qmc.Qubit]) -> int | None:
    """Return a vector's concrete width when available during tracing.

    Args:
        register (qmc.Vector[qmc.Qubit]): Quantum register to inspect.

    Returns:
        int | None: Concrete width, or ``None`` while the shape is symbolic.
    """
    if not register.shape:
        return None
    dimension = register.shape[0]
    if isinstance(dimension, int):
        return dimension
    value = getattr(dimension, "value", None)
    if value is not None and value.is_constant():
        return int(value.get_const())
    return None


def _validate_register_width(
    register: qmc.Vector[qmc.Qubit],
    expected: int,
    role: str,
) -> None:
    """Reject a concretely known register width that violates the factory ABI.

    Args:
        register (qmc.Vector[qmc.Qubit]): Quantum register supplied by a caller.
        expected (int): Width required by the generated construction.
        role (str): Register role used in the diagnostic.

    Raises:
        ValueError: If the concrete width differs from ``expected``.
    """
    actual = _concrete_register_width(register)
    if actual is not None and actual != expected:
        raise ValueError(
            f"periodic stencil block encoding requires {expected} {role} "
            f"qubits, got {actual}"
        )


def _validate_register_sizes(register_sizes: Sequence[Any]) -> tuple[int, ...]:
    """Validate and freeze the per-axis qubit widths.

    Args:
        register_sizes (Sequence[int]): Qubit width for each periodic axis.

    Returns:
        tuple[int, ...]: Validated positive widths.

    Raises:
        ValueError: If no axis is supplied or an axis has non-positive width.
        TypeError: If a width is not a plain integer.
    """
    sizes = tuple(register_sizes)
    if not sizes:
        raise ValueError("register_sizes must contain at least one axis")
    for axis, width in enumerate(sizes):
        if not _is_integer(width):
            raise TypeError(
                f"register_sizes[{axis}] must be an int, got {type(width).__name__}"
            )
        if int(width) <= 0:
            raise ValueError(
                f"register_sizes[{axis}] must be positive, got {int(width)}"
            )
    return tuple(int(width) for width in sizes)


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

    combined: dict[tuple[int, ...], complex] = {}
    for offset, coefficient in coefficients.items():
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
        canonical = _canonical_offset(offset, register_sizes)
        combined[canonical] = combined.get(canonical, 0.0j) + value

    terms = tuple(
        (offset, coefficient)
        for offset, coefficient in sorted(combined.items())
        if coefficient
    )
    if not terms:
        raise ValueError("periodic stencil is the zero operator after combining terms")
    return terms


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
            system = _apply_flat_axis_shift(system, start, width, shift)
        start = stop
    return system


def _apply_flat_axis_shift(
    system: qmc.Vector[qmc.Qubit],
    start: int,
    width: int,
    shift: int,
) -> qmc.Vector[qmc.Qubit]:
    """Shift one fixed subregister without creating symbolic nested slices.

    The existing modular increment/decrement kernels are used for a one-axis
    system.  A multidimensional flat register needs this fixed-boundary form
    because passing a concrete axis view into those symbolic-width kernels
    would create a second, symbolic view under the same live borrow.  This
    helper emits the same increment/decrement gate ladders with concrete
    prefix bounds.

    Args:
        system (qmc.Vector[qmc.Qubit]): Flattened multidimensional register.
        start (int): First qubit of the axis.
        width (int): Number of qubits in the axis.
        shift (int): Signed number of unit modular shifts.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated flattened register.
    """
    increment = shift > 0
    for _ in range(abs(shift)):
        if not increment:
            system[start] = qmc.x(system[start])

        target_indices = range(width - 1, 0, -1) if increment else range(1, width)
        for target_index in target_indices:
            controls = system[start : start + target_index]
            target = system[start + target_index]
            controls, target = qmc.mcx(controls, target)
            system[start : start + target_index] = controls
            system[start + target_index] = target

        if increment:
            system[start] = qmc.x(system[start])
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

    phased_shift = qmc.global_phase(shift_case, cmath.phase(coefficient))

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


def periodic_stencil_block_encoding(
    coefficients: Mapping[Offset, complex],
    register_sizes: Sequence[int],
) -> PeriodicStencilEncoding:
    """Build an LCU block encoding of a periodic constant-coefficient stencil.

    The system is a flattened concatenation of LSB-first axis registers.  An
    integer offset is accepted for a one-dimensional stencil; multidimensional
    offsets are tuples.  Axis ``j`` wraps modulo ``2**register_sizes[j]``.
    Equivalent offsets are combined before computing the normalization.
    A shift uses the shorter of repeated modular increments or decrements, so
    its gate cost is linear in that shortest displacement.
    Only exact cancellations are removed before ``lambda`` and PREPARE are
    constructed, so every representable nonzero coefficient is preserved.

    The returned qkernel expects an all-zero signal register.  Projecting that
    register onto all zero before and after the qkernel yields ``A / lambda``,
    where ``lambda`` is available as ``result.normalization``.

    Args:
        coefficients (Mapping[int | tuple[int, ...], complex]): Constant
            stencil coefficients keyed by signed or unsigned periodic offsets.
        register_sizes (Sequence[int]): Positive qubit width of every axis in
            flattened-system order.

    Returns:
        PeriodicStencilEncoding: Method-specific kernel and metadata.

    Raises:
        ValueError: If no axes or terms are supplied, an axis width is not
            positive, an offset has the wrong dimension, a coefficient or
            normalization is non-finite, or all terms cancel modulo the axis
            sizes.
        TypeError: If widths or offsets are not integers, or coefficients are
            not numeric.

    Example:
        >>> import qamomile.circuit as qmc
        >>> encoding = periodic_stencil_block_encoding(
        ...     {-1: 1.0, 0: -2.0, 1: 1.0},
        ...     register_sizes=(3,),
        ... )
        >>> encoding.normalization
        4.0
        >>> @qmc.qkernel
        ... def apply():
        ...     signal = qmc.qubit_array(encoding.num_signal_qubits)
        ...     system = qmc.qubit_array(encoding.num_system_qubits)
        ...     return encoding(signal, system)
    """
    sizes = _validate_register_sizes(register_sizes)
    terms = _canonical_terms(coefficients, sizes)
    normalization = float(sum(abs(coefficient) for _, coefficient in terms))
    if not math.isfinite(normalization):
        raise ValueError("periodic stencil normalization must be finite")

    num_terms = len(terms)
    num_signal_qubits = max(1, (num_terms - 1).bit_length())
    prepare_amplitudes = np.zeros(1 << num_signal_qubits, dtype=np.float64)
    for index, (_, coefficient) in enumerate(terms):
        prepare_amplitudes[index] = math.sqrt(abs(coefficient) / normalization)

    cases = [
        _make_shift_case(offset, coefficient, sizes) for offset, coefficient in terms
    ]
    if len(cases) == 1:
        cases.append(_make_shift_case(tuple(0 for _ in sizes), 1.0 + 0.0j, sizes))

    prepare, required_signal_qubits = _mottonen_composite(prepare_amplitudes)
    if required_signal_qubits != num_signal_qubits:
        raise AssertionError("state-preparation width disagrees with SELECT width")
    unprepare = qmc.inverse(prepare)
    select_gate = qmc.select(cases)

    @qmc.qkernel
    def kernel(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Apply PREPARE, phased shift SELECT, and inverse PREPARE.

        Args:
            signal (qmc.Vector[qmc.Qubit]): All-zero signal register.
            system (qmc.Vector[qmc.Qubit]): Flattened system register.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated signal
            and system registers.
        """
        signal = prepare(signal)
        signal, system = select_gate(signal, system)
        signal = unprepare(signal)
        return signal, system

    kernel.name = "periodic_stencil_block_encoding"
    return PeriodicStencilEncoding(
        kernel=kernel,
        normalization=normalization,
        num_signal_qubits=num_signal_qubits,
        register_sizes=sizes,
        offsets=tuple(offset for offset, _ in terms),
        coefficients=tuple(coefficient for _, coefficient in terms),
    )


__all__ = [
    "PeriodicStencilEncoding",
    "periodic_stencil_block_encoding",
]
