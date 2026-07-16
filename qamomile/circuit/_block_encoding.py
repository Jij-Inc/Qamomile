"""Validate private contracts shared by block-encoding producers."""

from __future__ import annotations

import inspect
import math
import numbers

import numpy as np

from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.frontend.qkernel import QKernel

BlockEncodingUnitary = QKernel[
    ...,
    tuple[Vector[Qubit], Vector[Qubit]],
]


def validate_block_encoding_fields(
    unitary: object,
    normalization: object,
    num_signal_qubits: object,
    num_system_qubits: object,
) -> tuple[BlockEncodingUnitary, float, int, int]:
    """Validate the common fields of one static block-encoding descriptor.

    This helper deliberately validates a structural contract without defining
    a public base class or protocol for block encodings.

    Args:
        unitary (object): Candidate qkernel implementing the encoding unitary.
        normalization (object): Candidate finite positive block normalization.
        num_signal_qubits (object): Candidate concrete signal-register width.
        num_system_qubits (object): Candidate concrete system-register width.

    Returns:
        tuple[BlockEncodingUnitary, float, int, int]: The validated unitary
            (``BlockEncodingUnitary``), normalized scale (``float``), signal
            width (``int``), and system width (``int``), in that order.

    Raises:
        TypeError: If the unitary does not have the exact static two-register
            ABI, normalization is not real, or a width is not an integer.
        ValueError: If normalization is non-finite or non-positive, or a width
            is non-positive.
    """
    if not isinstance(unitary, QKernel):
        raise TypeError("unitary must be a QKernel.")
    parameters = tuple(unitary.signature.parameters.values())
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
        or unitary.input_types != expected_inputs
        or unitary.output_types != expected_outputs
    ):
        raise TypeError(
            "unitary must have signature "
            "(signal: Vector[Qubit], system: Vector[Qubit]) -> "
            "tuple[Vector[Qubit], Vector[Qubit]]."
        )

    return (
        unitary,
        _validate_positive_real(normalization, "normalization"),
        _validate_positive_integer(num_signal_qubits, "num_signal_qubits"),
        _validate_positive_integer(num_system_qubits, "num_system_qubits"),
    )


def validate_block_encoding_registers(
    signal: Vector[Qubit],
    system: Vector[Qubit],
    expected_signal: int,
    expected_system: int,
    display_name: str,
) -> None:
    """Validate concrete call-register widths for a block-encoding unitary.

    Symbolic widths are left to later shape validation, while every concrete
    specialization receives the same diagnostic through plain, inverse,
    controlled, and nested-SELECT calls.

    Args:
        signal (Vector[Qubit]): Complete public signal register.
        system (Vector[Qubit]): Ordered logical system register.
        expected_signal (int): Required concrete signal width.
        expected_system (int): Required concrete system width.
        display_name (str): Producer name used in diagnostics.

    Raises:
        TypeError: If either register is not a qubit vector.
        ValueError: If a concrete register width differs from its requirement.
    """
    _validate_register_width(signal, expected_signal, "signal", display_name)
    _validate_register_width(system, expected_system, "system", display_name)


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
    """Validate and normalize one concrete positive integer width.

    Args:
        value (object): Candidate register width.
        name (str): Field name used in diagnostics.

    Returns:
        int: Equivalent positive Python integer.

    Raises:
        TypeError: If ``value`` is not a non-Boolean integer.
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


def _validate_register_width(
    register: Vector[Qubit],
    expected: int,
    role: str,
    display_name: str,
) -> None:
    """Validate one concrete vector width and defer symbolic-width checks.

    Args:
        register (Vector[Qubit]): Register to inspect.
        expected (int): Required width.
        role (str): Register role used in diagnostics.
        display_name (str): Producer name used in diagnostics.

    Raises:
        TypeError: If ``register`` is not a qubit vector.
        ValueError: If its concrete width differs from ``expected``.
    """
    try:
        actual = get_size(register)
    except ValueError:
        return
    if actual != expected:
        unit = "qubit" if expected == 1 else "qubits"
        raise ValueError(
            f"{display_name} requires {expected} {role} {unit}, got {actual}."
        )


__all__: list[str] = []
