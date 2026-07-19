"""Define the common static descriptor contract for exact LCU encodings."""

from __future__ import annotations

import inspect
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from qamomile.circuit.frontend.constructors import uint
from qamomile.circuit.frontend.handle import Float, Qubit, UInt, Vector
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.frontend.operation.global_phase import global_phase
from qamomile.circuit.frontend.operation.inverse import inverse
from qamomile.circuit.frontend.operation.qubit_gates import x
from qamomile.circuit.frontend.operation.select import select
from qamomile.circuit.frontend.qkernel import QKernel, qkernel
from qamomile.circuit.frontend.static_binding import (
    StaticBindingFieldSpec,
    StaticBindingMemberSpec,
    StaticBindingSpec,
    register_static_binding,
)
from qamomile.circuit.ir.operation.callable import CallPolicy
from qamomile.circuit.stdlib.state_preparation.mottonen_amplitude_encoding import (
    _mottonen_composite,
)

_BlockEncodingUnitary = QKernel[
    ...,
    tuple[Vector[Qubit], Vector[Qubit]],
]
_LCUCase = QKernel[..., Vector[Qubit]]


@dataclass(frozen=True, slots=True, eq=False)
class LCUBlockEncoding:
    r"""Describe one static exact LCU block encoding.

    ``unitary`` is the qkernel implementing the larger unitary ``U``; it is
    neither the encoded matrix ``A`` nor a dense matrix value. It has no
    classical arguments and its quantum ABI is
    ``unitary(signal, system) -> (signal, system)``. ``system`` is the ordered
    logical data register on which ``A`` acts. ``signal`` is the complete
    source-level ancilla bundle whose all-zero state selects the encoded block.
    The unitary returns the same logical wires in the same order and acts
    unitarily for arbitrary signal inputs; after one application, the signal
    may have non-zero components rather than returning entirely to zero.

    For the all-zero signal isometry ``V0``, the producer must guarantee

    .. math::

        V_0^\dagger U V_0 = A / \mathtt{normalization}

    including coefficient phase. ``normalization`` is finite and positive;
    an encoding of the zero operator uses ``1.0``. Implementations allocate no
    hidden source-level logical qubits. A backend may still use temporary
    decomposition scratch that is resource-accounted, exactly uncomputed for
    every public input, and preserved under inverse and control. Descriptor
    comparison and hashing use object identity rather than field values.

    This common descriptor deliberately excludes decomposition-specific
    metadata. Reusable qkernels should annotate an encoding argument with this
    class so descriptors produced by Pauli and future LCU factories can occupy
    the same compile-time binding slot.

    Args:
        unitary (QKernel): QKernel implementing the block-encoding unitary
            ``U`` with the static ``(signal, system)`` ABI.
        normalization (float): Finite positive block normalization.
        num_signal_qubits (int): Concrete positive width of the complete
            signal register, including selectors, logical workspace, and
            padding required by the producer.
        num_system_qubits (int): Concrete positive width of the ordered system
            register.

    Raises:
        TypeError: If ``unitary`` is not a ``QKernel`` with the exact static
            positional ABI above, normalization is not a real scalar, or
            either width is not an integer.
        ValueError: If normalization is non-finite or non-positive, or either
            width is non-positive.
    """

    unitary: _BlockEncodingUnitary
    normalization: float
    num_signal_qubits: int
    num_system_qubits: int

    def __post_init__(self) -> None:
        """Validate and normalize the immutable descriptor fields.

        Raises:
            TypeError: If a field has an invalid runtime type or ``unitary``
                does not have the exact static positional block-encoding ABI.
            ValueError: If normalization or a register width is outside its
                valid finite positive range.
        """
        if not isinstance(self.unitary, QKernel):
            raise TypeError("unitary must be a QKernel.")
        parameters = tuple(self.unitary.signature.parameters.values())
        expected_inputs = {
            "signal": Vector[Qubit],
            "system": Vector[Qubit],
        }
        expected_outputs = [Vector[Qubit], Vector[Qubit]]
        if (
            tuple(parameter.name for parameter in parameters) != ("signal", "system")
            or any(
                parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
                or parameter.default is not inspect.Parameter.empty
                for parameter in parameters
            )
            or self.unitary.input_types != expected_inputs
            or self.unitary.output_types != expected_outputs
        ):
            raise TypeError(
                "unitary must have signature "
                "(signal: Vector[Qubit], system: Vector[Qubit]) -> "
                "tuple[Vector[Qubit], Vector[Qubit]]."
            )

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


def _validate_positive_real(value: object, name: str) -> float:
    """Validate one finite positive real scalar.

    Args:
        value (object): Candidate scalar value.
        name (str): Field name used in diagnostics.

    Returns:
        float: Equivalent finite positive Python float.

    Raises:
        TypeError: If ``value`` is not a non-boolean real scalar.
        ValueError: If conversion overflows or the value is non-finite or
            non-positive.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (numbers.Real, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a real numeric scalar.")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and positive.") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


def _validate_positive_integer(value: object, name: str) -> int:
    """Validate one concrete positive integer width.

    Args:
        value (object): Candidate register width.
        name (str): Field name used in diagnostics.

    Returns:
        int: Equivalent positive Python integer.

    Raises:
        TypeError: If ``value`` is not a non-boolean integer.
        ValueError: If ``value`` is non-positive.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


@qkernel
def _identity_vector(qubits: Vector[Qubit]) -> Vector[Qubit]:
    """Return a quantum register unchanged.

    Args:
        qubits (Vector[Qubit]): Register to preserve.

    Returns:
        Vector[Qubit]: The same logical register.
    """
    return qubits


def _coefficient_phase(coefficient: complex) -> float:
    """Return a coefficient phase with signed zero canonicalized.

    Args:
        coefficient (complex): Nonzero finite complex coefficient.

    Returns:
        float: Principal coefficient phase, using positive zero for a
            positive-real coefficient.
    """
    phase = math.atan2(coefficient.imag, coefficient.real)
    return phase if phase else 0.0


def _lcu_num_signal_qubits(num_terms: int) -> int:
    """Return the positive selector width for a retained LCU term count.

    Zero- and single-term encodings retain one pass-through signal qubit so
    every exact LCU producer has the same two-register ABI under nested SELECT.

    Args:
        num_terms (int): Non-negative number of retained nonzero LCU terms.

    Returns:
        int: Positive signal-register width.

    Raises:
        TypeError: If ``num_terms`` is not a plain integer.
        ValueError: If ``num_terms`` is negative.
    """
    if type(num_terms) is not int:
        raise TypeError("num_terms must be an integer.")
    if num_terms < 0:
        raise ValueError("num_terms must be non-negative.")
    return max(1, (num_terms - 1).bit_length())


def _validate_register_widths(
    signal: Vector[Qubit],
    system: Vector[Qubit],
    expected_signal: int,
    expected_system: int,
    description: str,
) -> None:
    """Validate concrete LCU registers while allowing symbolic cache traces.

    Args:
        signal (Vector[Qubit]): Complete signal register.
        system (Vector[Qubit]): Ordered system register.
        expected_signal (int): Required signal width.
        expected_system (int): Required system width.
        description (str): Producer description used in diagnostics.

    Raises:
        ValueError: If a concrete register width differs from its requirement.
    """
    _validate_register_width(signal, expected_signal, "signal", description)
    _validate_register_width(system, expected_system, "system", description)


def _validate_register_width(
    register: Vector[Qubit],
    expected: int,
    name: str,
    description: str,
) -> None:
    """Validate one concrete vector width and defer symbolic-width checks.

    Args:
        register (Vector[Qubit]): Register to inspect.
        expected (int): Required width.
        name (str): Register name used in diagnostics.
        description (str): Producer description used in diagnostics.

    Raises:
        ValueError: If the concrete width differs from ``expected``.
    """
    try:
        actual = get_size(register)
    except ValueError:
        return
    if actual != expected:
        unit = "qubit" if expected == 1 else "qubits"
        raise ValueError(
            f"{description} requires {expected} {name} {unit}, got {actual}."
        )


def _build_lcu_block_encoding_unitary(
    cases: tuple[_LCUCase, ...],
    coefficients: tuple[complex, ...],
    num_system_qubits: int,
    *,
    description: str,
    preparation_name: str,
) -> _BlockEncodingUnitary:
    """Build the shared zero-, single-, or multi-term exact LCU unitary.

    Each supplied case must already include the phase and unitary action for
    its matching coefficient. The builder owns the common signal-width rule,
    zero-block bit flip, and PREPARE-SELECT-PREPARE-dagger construction.

    Args:
        cases (tuple[_LCUCase, ...]): Uniform one-register SELECT cases in
            retained coefficient order.
        coefficients (tuple[complex, ...]): Matching nonzero LCU coefficients.
        num_system_qubits (int): Positive width of the ordered system register.
        description (str): Producer description used in width diagnostics.
        preparation_name (str): Compiler identity name for the Möttönen
            PREPARE callable.

    Returns:
        _BlockEncodingUnitary: Shared-signature block-encoding unitary.

    Raises:
        ValueError: If the case and coefficient counts differ.
        RuntimeError: If state preparation returns an unexpected signal width.
    """
    if len(cases) != len(coefficients):
        raise ValueError("LCU cases and coefficients must have the same length.")
    signal_width = _lcu_num_signal_qubits(len(cases))
    normalization = math.fsum(abs(coefficient) for coefficient in coefficients)

    if not cases:

        @qkernel
        def unitary(
            signal: Vector[Qubit],
            system: Vector[Qubit],
        ) -> tuple[Vector[Qubit], Vector[Qubit]]:
            """Apply the shared exact-zero block-encoding body.

            Args:
                signal (Vector[Qubit]): One-qubit signal register.
                system (Vector[Qubit]): Ordered system register.

            Returns:
                tuple[Vector[Qubit], Vector[Qubit]]: Updated signal and
                    preserved system registers.
            """
            _validate_register_widths(
                signal,
                system,
                signal_width,
                num_system_qubits,
                description,
            )
            signal[0] = x(signal[0])
            return signal, system

    elif len(cases) == 1:
        case = cases[0]

        @qkernel
        def unitary(
            signal: Vector[Qubit],
            system: Vector[Qubit],
        ) -> tuple[Vector[Qubit], Vector[Qubit]]:
            """Apply one unconditional phased LCU case.

            Args:
                signal (Vector[Qubit]): Pass-through signal register.
                system (Vector[Qubit]): Ordered system register.

            Returns:
                tuple[Vector[Qubit], Vector[Qubit]]: Preserved signal and
                    transformed system registers.
            """
            _validate_register_widths(
                signal,
                system,
                signal_width,
                num_system_qubits,
                description,
            )
            system = case(system)
            return signal, system

    else:
        amplitudes = np.zeros(1 << signal_width, dtype=np.float64)
        sqrt_normalization = math.sqrt(normalization)
        for index, coefficient in enumerate(coefficients):
            amplitudes[index] = math.sqrt(abs(coefficient)) / sqrt_normalization

        preparation, required_signal_qubits = _mottonen_composite(
            amplitudes,
            name=preparation_name,
            policy=CallPolicy.PRESERVE_BOX,
        )
        if required_signal_qubits != signal_width:
            raise RuntimeError(
                "Möttönen preparation width disagrees with the LCU signal width."
            )
        unprepare = inverse(preparation)
        selector = select(cases, num_index_qubits=uint(signal_width))

        @qkernel
        def unitary(
            signal: Vector[Qubit],
            system: Vector[Qubit],
        ) -> tuple[Vector[Qubit], Vector[Qubit]]:
            """Apply shared PREPARE, SELECT, and inverse PREPARE operations.

            Args:
                signal (Vector[Qubit]): All-zero block signal register.
                system (Vector[Qubit]): Ordered system register.

            Returns:
                tuple[Vector[Qubit], Vector[Qubit]]: Updated signal and system
                    registers.
            """
            _validate_register_widths(
                signal,
                system,
                signal_width,
                num_system_qubits,
                description,
            )
            signal = preparation(signal)
            signal, system = selector(signal, system)
            signal = unprepare(signal)
            return signal, system

    return cast(_BlockEncodingUnitary, unitary)


def _register_lcu_block_encoding_static_binding(
    annotation: type[LCUBlockEncoding],
    type_key: str,
) -> None:
    """Register one nominal LCU descriptor type for static qkernel binding.

    Args:
        annotation (type[LCUBlockEncoding]): Common descriptor class or one
            scheme-specific subclass accepted by the binding slot.
        type_key (str): Stable serialization key for qkernels annotated with
            ``annotation``.

    Raises:
        TypeError: If the annotation or generated adapter contract is invalid.
        ValueError: If the annotation or type key is already registered.
    """
    register_static_binding(
        StaticBindingSpec(
            annotation=annotation,
            type_key=type_key,
            fields={
                "normalization": StaticBindingFieldSpec(
                    handle_type=Float,
                    getter=lambda encoding: encoding.normalization,
                ),
                "num_signal_qubits": StaticBindingFieldSpec(
                    handle_type=UInt,
                    getter=lambda encoding: encoding.num_signal_qubits,
                ),
                "num_system_qubits": StaticBindingFieldSpec(
                    handle_type=UInt,
                    getter=lambda encoding: encoding.num_system_qubits,
                ),
            },
            members={
                "unitary": StaticBindingMemberSpec(
                    input_types={
                        "signal": Vector[Qubit],
                        "system": Vector[Qubit],
                    },
                    output_types=(Vector[Qubit], Vector[Qubit]),
                    return_annotation=tuple[Vector[Qubit], Vector[Qubit]],
                    getter=lambda encoding: encoding.unitary,
                    qubit_width_fields={
                        "signal": "num_signal_qubits",
                        "system": "num_system_qubits",
                    },
                ),
            },
        )
    )


_register_lcu_block_encoding_static_binding(
    LCUBlockEncoding,
    "qamomile.stdlib.lcu_block_encoding",
)


@dataclass(frozen=True, slots=True, eq=False)
class LCUBlockEncodingTerm:
    r"""Pair a logical coefficient with one exact child block encoding.

    ``coefficient`` multiplies the child target matrix ``A_j``, not its
    normalized projected block ``A_j / alpha_j``. The recursive composer
    therefore assigns the term weight ``abs(coefficient) *
    encoding.normalization``. Children are concrete construction-time
    descriptors; the completed parent, rather than unresolved children, is
    the object intended for qkernel static binding.

    Args:
        coefficient (complex): Finite logical coefficient. Zero terms are
            removed before nonzero circuit construction.
        encoding (LCUBlockEncoding): Concrete exact child descriptor. Producer
            subtypes such as ``PauliLCUBlockEncoding`` are accepted through
            the nominal common base class.

    Raises:
        TypeError: If the coefficient is not a non-boolean complex numeric
            scalar or ``encoding`` is not an ``LCUBlockEncoding``.
        ValueError: If either coefficient component is non-finite.
    """

    coefficient: complex
    encoding: LCUBlockEncoding

    def __post_init__(self) -> None:
        """Validate and normalize the frozen term fields.

        Raises:
            TypeError: If the coefficient or child descriptor type is
                invalid.
            ValueError: If either coefficient component is non-finite.
        """
        object.__setattr__(
            self,
            "coefficient",
            _validate_finite_complex(self.coefficient, "coefficient"),
        )
        if not isinstance(self.encoding, LCUBlockEncoding):
            raise TypeError("encoding must be an LCUBlockEncoding.")


@dataclass(frozen=True, slots=True)
class _ValidatedTerm:
    """Store one normalized construction-time LCU term.

    Args:
        coefficient (complex): Finite logical coefficient.
        encoding (LCUBlockEncoding): Validated nominal child descriptor.
        normalization (float): Child block normalization.
        signal_width (int): Child signal-register width.
        system_width (int): Child system-register width.
    """

    coefficient: complex
    encoding: LCUBlockEncoding
    normalization: float
    signal_width: int
    system_width: int


def identity_block_encoding(num_system_qubits: int) -> LCUBlockEncoding:
    r"""Create an exact identity encoding with one pass-through signal.

    The unitary acts as identity on both registers and satisfies
    ``V0^dagger U V0 = I`` with normalization ``1.0``. One signal qubit is
    retained so the descriptor has the same positive-width ABI required by
    reusable circuits, nested SELECT, and all-zero projector consumers.

    Args:
        num_system_qubits (int): Concrete positive system-register width.

    Returns:
        LCUBlockEncoding: Exact identity descriptor with one signal qubit.

    Raises:
        TypeError: If ``num_system_qubits`` is not an integer.
        ValueError: If ``num_system_qubits`` is non-positive.
    """
    system_width = _validate_positive_integer(
        num_system_qubits,
        "num_system_qubits",
    )
    unitary = _build_lcu_block_encoding_unitary(
        (_identity_vector,),
        (1.0 + 0.0j,),
        system_width,
        description="identity block encoding",
        preparation_name="identity_block_encoding_prepare",
    )
    # The name is diagnostic only; stable generated semantic identity remains
    # deferred until the compiler can derive it from the owned callable body.
    unitary.name = "identity_block_encoding"
    return LCUBlockEncoding(
        unitary=unitary,
        normalization=1.0,
        num_signal_qubits=1,
        num_system_qubits=system_width,
    )


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
    case ``j`` applies ``exp(1j * arg(c_j)) * U_j``. Heterogeneous child
    signal registers share a pool whose width is the maximum child width.
    Every uniform-signature case routes only its leading child-sized slice and
    acts exactly as identity on unused padding for arbitrary pool states. The
    private parent lowering is ``[outer selector | shared child pool]``;
    consumers only allocate the reported flat signal register and project all
    of it onto zero.

    Child descriptors are concrete when this factory runs. Their coefficient
    data, angles, and layouts do not become public arguments of the completed
    parent unitary. The returned common descriptor may itself be supplied as a
    child of another composition or bound later to a reusable qkernel argument
    annotated with :class:`LCUBlockEncoding`.

    Children must interpret the ordered system wires in the same logical basis;
    the common descriptor can validate their widths but cannot infer basis
    conventions. Generated callable semantic identity across separately
    constructed descriptors is not part of this API yet. Concrete children are
    captured when the factory runs, and the completed descriptor supports
    nesting and compile-time static binding without exposing those children as
    public qkernel arguments. Qiskit, QuriParts, and CUDA-Q inline these
    generated bodies. The HUGR target does not currently support recursive LCU
    composition because SELECT lowering and multiple inline callable-body
    variants sharing one source-derived reference are not yet supported.

    Zero-coefficient terms are removed before normalization and SELECT
    construction. A nonempty sequence containing only zero coefficients
    creates an exact zero encoding with normalization ``1.0`` and one signal
    qubit; its child system widths must agree so the zero operator's domain is
    defined. An empty sequence is rejected because its system width is
    unknowable.

    Args:
        terms (Sequence[LCUBlockEncodingTerm]): Nonempty ordered child terms.
            Every active child must use the same ordered system-register
            width. If all coefficients are zero, every child must use one
            common width.

    Returns:
        LCUBlockEncoding: Exact recursively composable parent descriptor.

    Raises:
        TypeError: If ``terms`` is not an ordered sequence or contains a
            non-term value.
        ValueError: If ``terms`` is empty, relevant child system widths
            differ, or the composed normalization overflows or is non-finite.

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
    active_terms = tuple(term for term in validated_terms if term.coefficient)
    domain_terms = active_terms if active_terms else validated_terms
    system_width = _common_system_width(domain_terms)

    if not active_terms:
        normalization = 1.0
        signal_width = 1
        unitary = _build_lcu_block_encoding_unitary(
            (),
            (),
            system_width,
            description="LCU block encoding",
            preparation_name="recursive_lcu_prepare",
        )
    else:
        normalization = _lcu_normalization(active_terms)
        if len(active_terms) == 1:
            signal_width = active_terms[0].signal_width
            unitary = _build_single_term_encoding(active_terms[0])
        else:
            selector_width = (len(active_terms) - 1).bit_length()
            child_pool_width = max(term.signal_width for term in active_terms)
            signal_width = selector_width + child_pool_width
            unitary = _build_multi_term_encoding(
                active_terms,
                normalization,
                signal_width,
                system_width,
            )

    # The name is diagnostic only; stable generated semantic identity remains
    # deferred until the compiler can derive it from the owned callable body.
    unitary.name = "lcu_block_encoding"
    return LCUBlockEncoding(
        unitary=unitary,
        normalization=normalization,
        num_signal_qubits=signal_width,
        num_system_qubits=system_width,
    )


def _validate_terms(terms: object) -> tuple[_ValidatedTerm, ...]:
    """Validate a nonempty ordered term collection without dropping zeros.

    Args:
        terms (object): Candidate ordered terms.

    Returns:
        tuple[_ValidatedTerm, ...]: Validated terms in original order.

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
        normalized.append(
            _ValidatedTerm(
                coefficient=term.coefficient,
                encoding=term.encoding,
                normalization=term.encoding.normalization,
                signal_width=term.encoding.num_signal_qubits,
                system_width=term.encoding.num_system_qubits,
            )
        )
    return tuple(normalized)


def _common_system_width(terms: tuple[_ValidatedTerm, ...]) -> int:
    """Return the common ordered system width for relevant terms.

    Args:
        terms (tuple[_ValidatedTerm, ...]): Nonempty active terms, or all
            terms when constructing a zero encoding.

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


def _build_single_term_encoding(
    term: _ValidatedTerm,
) -> _BlockEncodingUnitary:
    """Build a selector-free phased child encoding for one active term.

    Args:
        term (_ValidatedTerm): Sole active LCU term.

    Returns:
        _BlockEncodingUnitary: Parent unitary with the child's signal width.
    """
    child = term.encoding.unitary
    phase = _coefficient_phase(term.coefficient)
    applied_child = global_phase(child, phase) if phase else child
    signal_width = term.signal_width
    system_width = term.system_width

    @qkernel
    def unitary(
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
            description="LCU block encoding",
        )
        return applied_child(signal, system)

    return unitary


def _build_multi_term_encoding(
    terms: tuple[_ValidatedTerm, ...],
    normalization: float,
    signal_width: int,
    system_width: int,
) -> _BlockEncodingUnitary:
    """Build a recursive PREPARE-SELECT-PREPARE-dagger encoding.

    Args:
        terms (tuple[_ValidatedTerm, ...]): Ordered active child terms.
        normalization (float): Weighted parent normalization.
        signal_width (int): Complete parent signal width.
        system_width (int): Shared child system width.

    Returns:
        _BlockEncodingUnitary: Recursive multi-term LCU unitary.

    Raises:
        RuntimeError: If Mottönen preparation reports an unexpected width.
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
        name="mottonen_amplitude_encoding",
        policy=CallPolicy.PRESERVE_BOX,
    )
    if required_width != selector_width:
        raise RuntimeError(
            "Mottönen preparation width disagrees with the LCU selector width."
        )
    unprepare = inverse(preparation)
    selector = select(
        tuple(_build_uniform_case(term, child_pool_width) for term in terms),
        num_index_qubits=uint(selector_width),
    )

    @qkernel
    def unitary(
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
            description="LCU block encoding",
        )
        signal[:selector_width] = preparation(signal[:selector_width])
        signal[:selector_width], signal[selector_width:], system = selector(
            signal[:selector_width],
            signal[selector_width:],
            system,
        )
        signal[:selector_width] = unprepare(signal[:selector_width])
        return signal, system

    return unitary


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
    child = term.encoding.unitary
    child_width = term.signal_width
    phase = _coefficient_phase(term.coefficient)
    applied_child = global_phase(child, phase) if phase else child

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
        signal[:child_width], system = applied_child(
            signal[:child_width],
            system,
        )
        return signal, system

    return case


def _validate_pool_width(child_pool: Vector[Qubit], expected: int) -> None:
    """Validate a concrete shared child-pool width.

    Args:
        child_pool (Vector[Qubit]): Pool passed to one SELECT case.
        expected (int): Required uniform pool width.

    Raises:
        TypeError: If ``child_pool`` is not a vector register.
        ValueError: If the concrete pool width differs from ``expected``.
    """
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
