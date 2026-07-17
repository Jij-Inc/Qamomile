"""Define private structural validation for block-encoding composers."""

from __future__ import annotations

import inspect
import math
import numbers
from typing import Protocol, cast

import numpy as np

from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.frontend.qkernel import QKernel

_BlockEncodingKernel = QKernel[
    ...,
    tuple[Vector[Qubit], Vector[Qubit]],
]


class _BlockEncodingLike(Protocol):
    """Describe the private structural fields consumed by composers."""

    kernel: _BlockEncodingKernel
    normalization: float
    num_signal_qubits: int
    num_system_qubits: int


def _validated_block_encoding(
    value: object,
    name: str,
) -> tuple[_BlockEncodingLike, float, int, int]:
    """Validate and normalize one structural block-encoding descriptor.

    Args:
        value (object): Candidate descriptor exposing the four structural
            fields consumed by a block-encoding composer.
        name (str): Argument path used in diagnostics.

    Returns:
        tuple[_BlockEncodingLike, float, int, int]: Original descriptor,
            normalized normalization, signal width, and system width.

    Raises:
        TypeError: If a required field, qkernel ABI, scalar type, or width type
            is invalid.
        ValueError: If normalization is non-finite or non-positive, or a width
            is non-positive.
    """
    try:
        kernel = getattr(value, "kernel")
        normalization = getattr(value, "normalization")
        signal_width = getattr(value, "num_signal_qubits")
        system_width = getattr(value, "num_system_qubits")
    except AttributeError as exc:
        raise TypeError(
            f"{name} must expose kernel, normalization, "
            "num_signal_qubits, and num_system_qubits."
        ) from exc

    _validate_block_encoding_kernel(kernel, f"{name}.kernel")
    normalized = _validate_positive_real(
        normalization,
        f"{name}.normalization",
    )
    normalized_signal = _validate_positive_integer(
        signal_width,
        f"{name}.num_signal_qubits",
    )
    normalized_system = _validate_positive_integer(
        system_width,
        f"{name}.num_system_qubits",
    )
    return (
        cast(_BlockEncodingLike, value),
        normalized,
        normalized_signal,
        normalized_system,
    )


def _validate_block_encoding_kernel(value: object, name: str) -> None:
    """Validate the exact static qkernel ABI used by block encodings.

    Args:
        value (object): Candidate qkernel.
        name (str): Argument path used in diagnostics.

    Raises:
        TypeError: If ``value`` is not a qkernel with the exact positional
            ``(signal, system)`` vector ABI and two vector outputs.
    """
    if not isinstance(value, QKernel):
        raise TypeError(f"{name} must be a QKernel.")
    parameters = tuple(value.signature.parameters.values())
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
        or value.input_types != expected_inputs
        or value.output_types != expected_outputs
    ):
        raise TypeError(
            f"{name} must have signature "
            "(signal: Vector[Qubit], system: Vector[Qubit]) -> "
            "tuple[Vector[Qubit], Vector[Qubit]]."
        )


def _validate_positive_real(value: object, name: str) -> float:
    """Validate one finite positive real scalar.

    Args:
        value (object): Candidate scalar.
        name (str): Argument path used in diagnostics.

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
        value (object): Candidate width.
        name (str): Argument path used in diagnostics.

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


def _validate_register_widths(
    signal: Vector[Qubit],
    system: Vector[Qubit],
    expected_signal: int,
    expected_system: int,
    owner: str,
) -> None:
    """Validate concrete block-register widths and defer symbolic widths.

    Args:
        signal (Vector[Qubit]): Complete signal register.
        system (Vector[Qubit]): Ordered system register.
        expected_signal (int): Required signal width.
        expected_system (int): Required system width.
        owner (str): Encoding name used in diagnostics.

    Raises:
        TypeError: If either argument is not a vector register.
        ValueError: If either concrete register width is incorrect.
    """
    _validate_register_width(signal, expected_signal, "signal", owner)
    _validate_register_width(system, expected_system, "system", owner)


def _validate_register_width(
    register: Vector[Qubit],
    expected: int,
    name: str,
    owner: str,
) -> None:
    """Validate one concrete vector width while allowing symbolic traces.

    Args:
        register (Vector[Qubit]): Register to inspect.
        expected (int): Required width.
        name (str): Register role used in diagnostics.
        owner (str): Encoding name used in diagnostics.

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
        raise ValueError(f"{owner} requires {expected} {name} {unit}, got {actual}.")
