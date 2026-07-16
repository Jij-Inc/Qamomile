"""Build block encodings from complex Pauli linear combinations."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np

from qamomile.circuit.frontend.composite_gate import (
    composite_gate,
    configure_composite,
)
from qamomile.circuit.frontend.constructors import uint
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.frontend.operation.global_phase import global_phase
from qamomile.circuit.frontend.operation.inverse import inverse
from qamomile.circuit.frontend.operation.qubit_gates import x, y, z
from qamomile.circuit.frontend.operation.select import select
from qamomile.circuit.frontend.qkernel import QKernel, qkernel
from qamomile.circuit.ir.operation.callable import CallPolicy
from qamomile.circuit.stdlib.state_preparation.mottonen_amplitude_encoding import (
    _mottonen_composite,
)
from qamomile.linalg import PauliLCU, PauliLCUTerm
from qamomile.observable import Pauli, PauliOperator

_BlockEncodingUnitary = QKernel[
    ...,
    tuple[Vector[Qubit], Vector[Qubit]],
]


@dataclass(frozen=True, slots=True, eq=False)
class PauliLCUBlockEncoding:
    r"""Describe one static exact block encoding of a retained Pauli LCU.

    ``unitary`` is the qkernel implementing the larger unitary ``U``; it is
    neither the encoded matrix ``A`` nor a dense matrix value. It has no
    classical arguments and its quantum ABI is
    ``unitary(signal, system) -> (signal, system)``. ``system`` is the ordered
    logical data register on which ``A`` acts. ``signal`` is the complete
    source-level ancilla bundle whose all-zero state selects the encoded block.
    The unitary returns the same logical wires in the same order and acts
    unitarily for arbitrary signal inputs; the signal may contain failure
    components rather than returning to zero after one application.

    For the all-zero signal isometry ``V0``, the exact retained LCU satisfies

    .. math::

        V_0^\dagger U V_0 = A / \mathtt{normalization}

    including coefficient phase. ``normalization`` is finite and positive;
    it is ``1.0`` for the zero operator. This implementation allocates no
    hidden source-level logical qubits. A backend may still use temporary
    decomposition scratch that is resource-accounted, exactly uncomputed for
    every public input, and preserved under inverse and control.
    Descriptor comparison and hashing use object identity rather than field
    values.

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


def _pauli_lcu_num_signal_qubits(lcu: PauliLCU) -> int:
    """Return the signal-register width required by the block encoding.

    At least one qubit is retained for the zero- and single-term paths so the
    returned unitary has one stable two-register signature and remains valid as
    a nested SELECT case.

    Args:
        lcu (PauliLCU): Pauli linear combination to encode.

    Returns:
        int: Required positive number of signal qubits.
    """
    if lcu.num_terms <= 1:
        return 1
    return (lcu.num_terms - 1).bit_length()


def pauli_lcu_block_encoding(lcu: PauliLCU) -> PauliLCUBlockEncoding:
    r"""Create a static exact block encoding of a complex Pauli LCU.

    For a nonzero decomposition

    .. math::

        A = \sum_j c_j P_j, \qquad \alpha = \sum_j |c_j|,

    the descriptor's qkernel ``unitary`` implements ``U`` and satisfies

    .. math::

        (\langle 0|^{\otimes a} \otimes I) U
        (|0\rangle^{\otimes a} \otimes I) = A / \alpha.

    Multi-term encodings use real-amplitude PREPARE weights
    ``sqrt(abs(c_j) / alpha)`` and SELECT cases
    ``exp(1j * arg(c_j)) * P_j``. The identity Pauli word is a normal case, so
    its coefficient phase is retained. The zero operator uses one signal
    qubit and an ``X`` gate, giving an exact zero all-zero block with
    normalization ``1.0``; its ``PauliLCU.alpha`` remains ``0.0``.

    The retained Pauli LCU is square, static, and encoded exactly. When
    :meth:`PauliLCU.from_matrix` truncated coefficients, its source-to-retained
    error remains available as ``lcu.truncation_error_bound`` and is not an
    error in this unitary. The unitary accepts arbitrary signal states, returns
    the same signal and system wires in the same order, supports
    :func:`~qamomile.circuit.inverse`, and allocates no hidden source-level
    logical workspace. Backend-only decomposition scratch is permitted only
    when resource-accounted and exactly uncomputed for all inputs, including
    under inverse and control.

    Args:
        lcu (PauliLCU): Immutable retained Pauli decomposition. It must
            describe at least one system qubit.

    Returns:
        PauliLCUBlockEncoding: Frozen non-callable descriptor. Allocate its
            registers with ``num_signal_qubits`` and ``num_system_qubits``,
            then invoke ``unitary(signal, system)``.

    Raises:
        TypeError: If ``lcu`` is not a ``PauliLCU``.
        ValueError: If ``lcu`` represents a scalar zero-qubit system.

    Example:
        >>> import numpy as np
        >>> import qamomile.circuit as qmc
        >>> from qamomile.linalg import PauliLCU
        >>> lcu = PauliLCU.from_matrix(np.array([[0, 1], [0, 0]], complex))
        >>> encoding = qmc.pauli_lcu_block_encoding(lcu)
        >>> @qmc.qkernel
        ... def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        ...     signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        ...     system = qmc.qubit_array(encoding.num_system_qubits, "system")
        ...     return encoding.unitary(signal, system)
    """
    if not isinstance(lcu, PauliLCU):
        raise TypeError("lcu must be a PauliLCU.")
    if lcu.num_qubits == 0:
        raise ValueError(
            "pauli_lcu_block_encoding requires at least one system qubit; "
            "a 1 x 1 scalar matrix has no Qamomile system register."
        )

    if lcu.num_terms == 0:
        unitary = _build_zero_encoding(lcu)
    elif lcu.num_terms == 1:
        unitary = _build_single_term_encoding(lcu)
    else:
        unitary = _build_multi_term_encoding(lcu)
    unitary = _configure_block_encoding(unitary, lcu)
    return PauliLCUBlockEncoding(
        unitary=unitary,
        normalization=lcu.alpha if lcu.num_terms else 1.0,
        num_signal_qubits=_pauli_lcu_num_signal_qubits(lcu),
        num_system_qubits=lcu.num_qubits,
    )


def _build_zero_encoding(lcu: PauliLCU) -> _BlockEncodingUnitary:
    """Build the exact zero block using one signal-qubit bit flip.

    Args:
        lcu (PauliLCU): Zero Pauli linear combination.

    Returns:
        _BlockEncodingUnitary: Zero block-encoding unitary.
    """
    signal_width = _pauli_lcu_num_signal_qubits(lcu)

    @composite_gate(name="pauli_lcu_block_encoding")
    def unitary(
        signal: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Apply the zero block-encoding body.

        Args:
            signal (Vector[Qubit]): One-qubit signal register.
            system (Vector[Qubit]): System register.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Updated signal and system
                registers.
        """
        _validate_register_widths(signal, system, signal_width, lcu.num_qubits)
        signal[0] = x(signal[0])
        return signal, system

    return unitary


def _build_single_term_encoding(
    lcu: PauliLCU,
) -> _BlockEncodingUnitary:
    """Build an unconditional phased-Pauli encoding for one term.

    Args:
        lcu (PauliLCU): One-term Pauli linear combination.

    Returns:
        _BlockEncodingUnitary: Single-term block-encoding unitary.
    """
    signal_width = _pauli_lcu_num_signal_qubits(lcu)
    term = lcu.terms[0]
    phase = _coefficient_phase(term.coefficient)

    @composite_gate(name="pauli_lcu_block_encoding")
    def unitary(
        signal: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Apply the single phased-Pauli body.

        Args:
            signal (Vector[Qubit]): One-qubit signal register, preserved
                unchanged.
            system (Vector[Qubit]): System register receiving the Pauli word.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Preserved signal register
                and transformed system register.
        """
        _validate_register_widths(signal, system, signal_width, lcu.num_qubits)
        _apply_pauli_word(system, term.operators)
        system = global_phase(_identity_vector, phase)(system)
        return signal, system

    return unitary


def _build_multi_term_encoding(
    lcu: PauliLCU,
) -> _BlockEncodingUnitary:
    """Build a PREPARE-SELECT-PREPARE-dagger encoding.

    Args:
        lcu (PauliLCU): Pauli linear combination containing at least two
            terms.

    Returns:
        _BlockEncodingUnitary: Multi-term block-encoding unitary.
    """
    signal_width = _pauli_lcu_num_signal_qubits(lcu)
    amplitudes = np.zeros(1 << signal_width, dtype=np.float64)
    alpha = lcu.alpha
    sqrt_alpha = math.sqrt(alpha)
    for index, term in enumerate(lcu.terms):
        amplitudes[index] = math.sqrt(abs(term.coefficient)) / sqrt_alpha

    preparation, required_width = _mottonen_composite(amplitudes)
    if required_width != signal_width:
        raise RuntimeError(
            "Möttönen preparation width disagrees with the LCU signal width."
        )
    unprepare = inverse(preparation)
    # A constant UInt keeps the unspecialized composite's signal Vector
    # traceable while lowering still resolves the exact fixed LCU width.
    selector = select(
        tuple(_build_phased_pauli_case(term) for term in lcu.terms),
        num_index_qubits=uint(signal_width),
    )

    @composite_gate(name="pauli_lcu_block_encoding")
    def unitary(
        signal: Vector[Qubit],
        system: Vector[Qubit],
    ) -> tuple[Vector[Qubit], Vector[Qubit]]:
        """Apply the multi-term Pauli LCU block encoding.

        Args:
            signal (Vector[Qubit]): Signal register whose all-zero state selects
                the encoded block.
            system (Vector[Qubit]): Arbitrary system register.

        Returns:
            tuple[Vector[Qubit], Vector[Qubit]]: Updated signal and system
                registers.
        """
        _validate_register_widths(signal, system, signal_width, lcu.num_qubits)
        signal = preparation(signal)
        signal, system = selector(signal, system)
        signal = unprepare(signal)
        return signal, system

    return unitary


def _build_phased_pauli_case(
    term: PauliLCUTerm,
) -> QKernel[..., Vector[Qubit]]:
    """Build one explicit QKernel SELECT case for a complex Pauli term.

    Args:
        term (PauliLCUTerm): Nonzero Pauli LCU term.

    Returns:
        QKernel[..., Vector[Qubit]]: QKernel implementing
            ``exp(1j * arg(coefficient)) * P``.
    """
    operators = term.operators
    phase = _coefficient_phase(term.coefficient)

    if not phase:

        @qkernel
        def real_case(system: Vector[Qubit]) -> Vector[Qubit]:
            """Apply one real-positive sparse Pauli word.

            Args:
                system (Vector[Qubit]): Shared SELECT target register.

            Returns:
                Vector[Qubit]: Transformed target register.
            """
            _apply_pauli_word(system, operators)
            return system

        return real_case

    @qkernel
    def phased_case(system: Vector[Qubit]) -> Vector[Qubit]:
        """Apply one phased sparse Pauli word.

        Args:
            system (Vector[Qubit]): Shared SELECT target register.

        Returns:
            Vector[Qubit]: Transformed target register.
        """
        _apply_pauli_word(system, operators)
        return global_phase(_identity_vector, phase)(system)

    return phased_case


def _apply_pauli_word(
    system: Vector[Qubit],
    operators: tuple[PauliOperator, ...],
) -> None:
    """Emit one sparse Pauli word into a target register.

    Args:
        system (Vector[Qubit]): Target register mutated in place.
        operators (tuple[PauliOperator, ...]): Canonical sparse Pauli word.

    Raises:
        ValueError: If an operator contains an unsupported Pauli value.
    """
    for operator in operators:
        if operator.pauli is Pauli.X:
            system[operator.index] = x(system[operator.index])
        elif operator.pauli is Pauli.Y:
            system[operator.index] = y(system[operator.index])
        elif operator.pauli is Pauli.Z:
            system[operator.index] = z(system[operator.index])
        else:
            raise ValueError(f"Unsupported sparse Pauli value: {operator.pauli!r}.")


def _coefficient_phase(coefficient: complex) -> float:
    """Return the coefficient phase with signed zero canonicalized.

    Args:
        coefficient (complex): Nonzero complex Pauli coefficient.

    Returns:
        float: Coefficient phase, using positive zero for a positive-real
            coefficient.
    """
    phase = math.atan2(coefficient.imag, coefficient.real)
    return phase if phase else 0.0


def _validate_register_widths(
    signal: Vector[Qubit],
    system: Vector[Qubit],
    expected_signal: int,
    expected_system: int,
) -> None:
    """Validate concrete register widths while allowing symbolic cache traces.

    Args:
        signal (Vector[Qubit]): Complete signal register.
        system (Vector[Qubit]): System register.
        expected_signal (int): Required signal width.
        expected_system (int): Required system width.

    Raises:
        TypeError: If either argument is not a vector register.
        ValueError: If a concrete register width differs from its requirement.
    """
    _validate_register_width(signal, expected_signal, "signal")
    _validate_register_width(system, expected_system, "system")


def _validate_register_width(
    register: Vector[Qubit],
    expected: int,
    name: str,
) -> None:
    """Validate one concrete vector width and defer symbolic-width checks.

    Args:
        register (Vector[Qubit]): Register to inspect.
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
            f"Pauli LCU block encoding requires {expected} {name} {unit}, got {actual}."
        )


def _configure_block_encoding(
    unitary: _BlockEncodingUnitary,
    lcu: PauliLCU,
) -> _BlockEncodingUnitary:
    """Attach stable compiler identity and semantic LCU metadata.

    Args:
        unitary (_BlockEncodingUnitary): Composite qkernel to configure.
        lcu (PauliLCU): Decomposition determining the composite unitary.

    Returns:
        _BlockEncodingUnitary: The same configured qkernel.
    """
    semantic_arguments = _semantic_arguments(lcu)
    digest_input = json.dumps(
        semantic_arguments,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()[:16]
    configure_composite(
        unitary,
        name="pauli_lcu_block_encoding",
        namespace=f"qamomile.stdlib.pauli_lcu.{digest}",
        policy=CallPolicy.PRESERVE_BOX,
        semantic_arguments=semantic_arguments,
    )
    return unitary


def _semantic_arguments(lcu: PauliLCU) -> dict[str, Any]:
    """Return serializer-safe arguments that determine the encoded unitary.

    Args:
        lcu (PauliLCU): Pauli decomposition to serialize.

    Returns:
        dict[str, Any]: Stable semantic argument mapping.
    """
    return {
        "num_qubits": lcu.num_qubits,
        "signal_qubits": _pauli_lcu_num_signal_qubits(lcu),
        "terms": [
            {
                "coefficient": [
                    float(term.coefficient.real),
                    float(term.coefficient.imag),
                ],
                "operators": [
                    [operator.pauli.name, int(operator.index)]
                    for operator in term.operators
                ],
            }
            for term in lcu.terms
        ],
    }


__all__ = [
    "PauliLCUBlockEncoding",
    "pauli_lcu_block_encoding",
]
