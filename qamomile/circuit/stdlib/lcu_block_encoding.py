"""Define the common static descriptor contract for exact LCU encodings."""

from __future__ import annotations

import inspect
import math
import numbers
from dataclasses import dataclass

import numpy as np

from qamomile.circuit.frontend.handle import Float, Qubit, UInt, Vector
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.frontend.static_binding import (
    StaticBindingFieldSpec,
    StaticBindingMemberSpec,
    StaticBindingSpec,
    register_static_binding,
)

_BlockEncodingUnitary = QKernel[
    ...,
    tuple[Vector[Qubit], Vector[Qubit]],
]


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


__all__ = ["LCUBlockEncoding"]
