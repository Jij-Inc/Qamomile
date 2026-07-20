r"""Build LCU block encodings from constant-coefficient periodic shifts.

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
coefficients require different block-encoding constructions. Each constant
shift is decomposed into ancilla-free increment or decrement ladders for the
set bits of its displacement. A dense circulant kernel may still require
exponentially many LCU terms, so this factory is intended for shift-sparse
decompositions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import qamomile.circuit as qmc
from qamomile.circuit.stdlib.arithmetic import (
    _apply_fixed_window_periodic_shift,
)
from qamomile.circuit.stdlib.lcu_block_encoding import (
    LCUBlockEncoding,
    _build_lcu_block_encoding_unitary,
    _coefficient_phase,
    _identity_vector,
    _lcu_num_signal_qubits,
    _register_lcu_block_encoding_static_binding,
)
from qamomile.linalg import PeriodicShiftLCU, PeriodicShiftLCUTerm

_NORMALIZATION_REL_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True, eq=False)
class PeriodicShiftLCUBlockEncoding(LCUBlockEncoding):
    r"""Describe one static exact periodic-shift LCU block encoding.

    ``unitary`` is the qkernel implementing the larger unitary ``U``; it is
    neither the encoded periodic-shift matrix ``A`` nor a dense matrix value. It has
    no classical arguments and its quantum ABI is
    ``unitary(signal, system) -> (signal, system)``. ``system`` is the ordered
    flattened data register on which ``A`` acts. ``signal`` is the complete
    source-level ancilla bundle whose all-zero state selects the encoded
    block. The unitary returns the same logical wires in the same order and
    acts unitarily for arbitrary signal inputs; signal need not return to zero
    after one application.

    For the all-zero signal isometry ``V0``, the encoded periodic-shift matrix
    satisfies

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
    producer metadata for host-side inspection only. A producer-specific
    qkernel may annotate an argument with this class. Reusable qkernels should
    instead annotate descriptor arguments with :class:`LCUBlockEncoding`,
    allowing the same serialized template to accept this and other exact LCU
    producers.

    Args:
        unitary (qmc.QKernel): QKernel implementing the block-encoding unitary
            ``U`` with the static ``(signal, system)`` ABI.
        normalization (float): Finite positive LCU normalization
            ``sum(abs(coefficients))`` after equivalent periodic offsets are
            combined. Direct construction accepts relative disagreement up to
            ``1e-12`` for host rounding and stores the canonical
            coefficient-derived sum. The empty zero-operator representation
            instead uses ``1.0``.
        num_signal_qubits (int): Concrete positive width of the complete signal
            register, including selector padding.
        num_system_qubits (int): Concrete positive width of the ordered flat
            system register.
        register_sizes (tuple[int, ...]): Qubit widths of the flattened system
            register's axes.
        offsets (tuple[tuple[int, ...], ...]): Canonical modular offsets, in
            SELECT case order. The empty tuple represents the zero operator.
        coefficients (tuple[complex, ...]): Nonzero combined coefficients, in
            SELECT case order. The empty tuple represents the zero operator.

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
        ValueError: If metadata is noncanonical or inconsistent with the
            common descriptor fields.
    """
    if not isinstance(register_sizes, tuple):
        raise TypeError("register_sizes must be a tuple of integers.")
    if not isinstance(offsets, tuple):
        raise TypeError("offsets must be a tuple of canonical offset tuples.")
    if not isinstance(coefficients, tuple):
        raise TypeError("coefficients must be a tuple of nonzero complex values.")
    if len(offsets) != len(coefficients):
        raise ValueError("offsets and coefficients must have the same length.")

    terms: list[PeriodicShiftLCUTerm] = []
    for index, (offset, coefficient) in enumerate(
        zip(offsets, coefficients, strict=True)
    ):
        if not isinstance(offset, tuple):
            raise TypeError(f"offsets[{index}] must be a tuple.")
        try:
            terms.append(PeriodicShiftLCUTerm(coefficient, offset))
        except (TypeError, ValueError) as error:
            raise type(error)(f"invalid descriptor term {index}: {error}") from error

    supplied_offsets = tuple(term.offset for term in terms)
    if len(set(supplied_offsets)) != len(supplied_offsets):
        raise ValueError("descriptor offsets must be unique.")
    if supplied_offsets != tuple(sorted(supplied_offsets)):
        raise ValueError("descriptor offsets must be canonically sorted.")

    lcu = PeriodicShiftLCU(
        register_sizes=register_sizes,
        terms=tuple(terms),
    )
    if num_system_qubits != lcu.num_qubits:
        raise ValueError("num_system_qubits must equal the sum of register_sizes.")
    frozen_offsets = tuple(term.offset for term in lcu.terms)
    frozen_coefficients = tuple(term.coefficient for term in terms)
    expected_signal = _lcu_num_signal_qubits(len(frozen_offsets))
    if num_signal_qubits != expected_signal:
        raise ValueError(
            "num_signal_qubits does not match the number of canonical terms."
        )
    canonical_normalization = lcu.alpha if lcu.terms else 1.0
    if not math.isclose(
        normalization,
        canonical_normalization,
        rel_tol=_NORMALIZATION_REL_TOLERANCE,
        abs_tol=0.0,
    ):
        raise ValueError(
            "normalization must equal the sum of absolute canonical coefficients."
        )
    return (
        lcu.register_sizes,
        frozen_offsets,
        frozen_coefficients,
        canonical_normalization,
    )


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


def _apply_multidimensional_shift(
    system: qmc.Vector[qmc.Qubit],
    register_sizes: tuple[int, ...],
    signed_shifts: tuple[int, ...],
) -> qmc.Vector[qmc.Qubit]:
    """Apply captured constant shifts to a flattened multidimensional register.

    This helper is ordinary trace-time Python rather than a qkernel. Its loop
    unrolls the fixed axis layout, and every axis uses the same fixed-window
    implementation regardless of the number of dimensions.

    Args:
        system (qmc.Vector[qmc.Qubit]): Flattened axis register.
        register_sizes (tuple[int, ...]): Qubit width of every axis.
        signed_shifts (tuple[int, ...]): Shortest signed displacement per axis.

    Returns:
        qmc.Vector[qmc.Qubit]: Shifted flattened register.
    """
    start = 0
    for width, shift in zip(register_sizes, signed_shifts, strict=True):
        if shift:
            system = _apply_fixed_window_periodic_shift(system, start, width, shift)
        start += width
    return system


def _make_shift_case(
    offset: tuple[int, ...],
    coefficient: complex,
    register_sizes: tuple[int, ...],
) -> qmc.QKernel[..., qmc.Vector[qmc.Qubit]]:
    """Build one phased multidimensional shift qkernel for SELECT.

    Args:
        offset (tuple[int, ...]): Canonical modular offset per axis.
        coefficient (complex): Nonzero coefficient whose argument becomes the
            case global phase.
        register_sizes (tuple[int, ...]): Axis widths within the flat system
            register.

    Returns:
        qmc.QKernel[..., qmc.Vector[qmc.Qubit]]: Qkernel implementing
            ``exp(1j * arg(coefficient)) * T_offset``.
    """
    signed_shifts = tuple(
        _shortest_signed_shift(residue, width)
        for residue, width in zip(offset, register_sizes, strict=True)
    )

    phase = _coefficient_phase(coefficient)
    if not phase:

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

        return shift_case

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
        system = _apply_multidimensional_shift(
            system,
            register_sizes,
            signed_shifts,
        )
        return qmc.global_phase(_identity_vector, phase)(system)

    return phased_shift_case


def periodic_shift_lcu_block_encoding(
    lcu: PeriodicShiftLCU,
) -> PeriodicShiftLCUBlockEncoding:
    """Build an LCU block encoding from a periodic-shift decomposition.

    ``lcu`` defines ``A = sum_k c_k T_k``, where ``T_k`` is the modular
    translation for one canonical offset tuple. The system is the flattened
    concatenation of the decomposition's LSB-first axis registers. A shift
    uses the shorter signed displacement, then emits one ancilla-free
    increment or decrement ladder for each set bit of its magnitude. This
    bounds the number of Qamomile-level X and multi-controlled-X operations by
    a quadratic function of an axis register's width; backend elementary-gate
    cost depends on how that backend decomposes multi-controlled X operations.

    The returned descriptor's ``unitary`` acts on arbitrary signal states.
    Projecting its signal register onto all zero before and after the unitary
    yields ``A / lambda``, where ``lambda`` is available as
    ``result.normalization``.
    The zero operator uses one signal qubit and an ``X`` gate, giving an exact
    zero all-zero block with normalization ``1.0`` while ``lcu.alpha`` remains
    ``0.0``.
    A single term retains one pass-through signal qubit for composition but
    omits PREPARE and SELECT entirely.

    When :meth:`PeriodicShiftLCU.from_matrix` or
    :meth:`PeriodicShiftLCU.from_coefficients` pruned coefficients, the source
    error remains available as ``lcu.truncation_error_bound`` and is not an
    error in the retained unitary.

    Args:
        lcu (PeriodicShiftLCU): Immutable retained periodic-shift
            decomposition. It must describe at least one system qubit.

    Returns:
        PeriodicShiftLCUBlockEncoding: Frozen non-callable descriptor containing
            the generated shift-LCU unitary and method-specific canonical
            metadata.

    Raises:
        TypeError: If ``lcu`` is not a :class:`PeriodicShiftLCU`.
        ValueError: If ``lcu`` represents a scalar zero-qubit system.

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.serialization import deserialize, serialize
        >>> from qamomile.linalg import PeriodicShiftLCU
        >>> from qamomile.qiskit import QiskitTranspiler
        >>> @qmc.qkernel
        ... def apply_encoding(
        ...     encoding: qmc.LCUBlockEncoding,
        ... ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        ...     signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        ...     system = qmc.qubit_array(encoding.num_system_qubits, "system")
        ...     return encoding.unitary(signal, system)
        >>> payload = serialize(apply_encoding)
        >>> received = deserialize(payload)
        >>> lcu = PeriodicShiftLCU.from_coefficients(
        ...     {-1: 1.0, 0: -2.0, 1: 1.0},
        ...     register_sizes=(3,),
        ... )
        >>> neighbor_difference = qmc.periodic_shift_lcu_block_encoding(lcu)
        >>> isinstance(neighbor_difference, qmc.LCUBlockEncoding)
        True
        >>> neighbor_difference.normalization
        4.0
        >>> executable = QiskitTranspiler().transpile(
        ...     received,
        ...     bindings={"encoding": neighbor_difference},
        ... )
    """
    if not isinstance(lcu, PeriodicShiftLCU):
        raise TypeError("lcu must be a PeriodicShiftLCU.")
    if lcu.num_qubits == 0:
        raise ValueError(
            "periodic_shift_lcu_block_encoding requires at least one system "
            "qubit; a 1 x 1 scalar matrix has no Qamomile system register."
        )

    sizes = lcu.register_sizes
    terms = tuple((term.offset, term.coefficient) for term in lcu.terms)
    normalization = lcu.alpha if terms else 1.0
    num_signal_qubits = _lcu_num_signal_qubits(len(terms))
    unitary = _build_lcu_block_encoding_unitary(
        tuple(
            _make_shift_case(offset, coefficient, sizes)
            for offset, coefficient in terms
        ),
        tuple(coefficient for _, coefficient in terms),
        sum(sizes),
        description="periodic shift LCU block encoding",
        preparation_name="periodic_shift_lcu_prepare",
    )
    # This display name is diagnostic only. Stable boxed callable identity is
    # intentionally deferred until boxed inverse lowering supports this body.
    unitary.name = "periodic_shift_lcu_block_encoding"
    return PeriodicShiftLCUBlockEncoding(
        unitary=unitary,
        normalization=normalization,
        num_signal_qubits=num_signal_qubits,
        num_system_qubits=sum(sizes),
        register_sizes=sizes,
        offsets=tuple(offset for offset, _ in terms),
        coefficients=tuple(coefficient for _, coefficient in terms),
    )


_register_lcu_block_encoding_static_binding(
    PeriodicShiftLCUBlockEncoding,
    "qamomile.stdlib.periodic_shift_lcu_block_encoding",
)


__all__ = [
    "PeriodicShiftLCUBlockEncoding",
    "periodic_shift_lcu_block_encoding",
]
