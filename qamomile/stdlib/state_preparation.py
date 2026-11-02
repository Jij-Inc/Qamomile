# ruff: noqa: F821 -- ``width`` is closed into a concrete dependent shape.

"""Functional amplitude preparation from an immutable classical vector."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import qamomile as qm
from qamomile._native import compiler as native


@dataclass(frozen=True, slots=True)
class AmplitudeAngles:
    """Exact-phase Möttönen coordinates for one normalized state vector.

    Attributes:
        width: Number of coherent bits in the prepared state.
        ry: Gray-walk uniformly controlled Y-rotation angles.
        rz: Gray-walk uniformly controlled Z-rotation angles.
        phase: Retained global phase of the prepared state.
    """

    width: int
    ry: tuple[float, ...]
    rz: tuple[float, ...]
    phase: float

    def __post_init__(self) -> None:
        """Validate the complete, finite fixed-width coordinate vector."""

        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise TypeError("width must be an integer")
        if self.width <= 0:
            raise ValueError("width must be positive")
        count = (1 << self.width) - 1
        if len(self.ry) != count or len(self.rz) != count:
            raise ValueError(f"ry and rz must each contain {count} angles")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in (*self.ry, *self.rz, self.phase)
        ):
            raise ValueError("amplitude angles must be finite real numbers")

    def bindings(self) -> dict[str, object]:
        """Return launch bindings for the standard parametric preparation ABI."""

        return {"ry_angles": self.ry, "rz_angles": self.rz, "phase": self.phase}


def amplitude_angles(amplitudes: Sequence[complex]) -> AmplitudeAngles:
    """Convert one amplitude vector into reusable exact-phase coordinates.

    Args:
        amplitudes: Finite real or complex amplitudes. The one-dimensional
            length must be a power of two and at least two. Normalization is
            performed before angle synthesis.

    Returns:
        Immutable Möttönen coordinates suitable for static closure capture or
        runtime binding to :func:`amplitude_encoding_from_angles`.

    Raises:
        TypeError: If the input is not a sequence of numeric scalars.
        ValueError: If its length is invalid, an element is nonfinite, or the
            vector is zero.
    """

    coordinates = _native_amplitude_coordinates(amplitudes)
    try:
        width, ry, rz, phase = coordinates
        return AmplitudeAngles(width, tuple(ry), tuple(rz), phase)
    except (TypeError, ValueError) as error:
        raise TypeError("Q7506: malformed native amplitude coordinates") from error


def amplitude_encoding(amplitudes: Sequence[complex]) -> qm.QKernel:
    """Construct a unitary that prepares one normalized amplitude vector.

    The returned function maps the all-zero state to the normalized input
    vector using the Möttönen Gray-walk decomposition. Classical angle
    synthesis happens once at factory time; the closed Qamomile body contains
    only ordinary ``RY``, ``RZ``, ``CX``, and global-phase operations. The
    global phase is retained, so the callable has a deterministic exact-phase
    convention and remains safe under adjoint and coherent composition.

    Args:
        amplitudes: Finite real or complex amplitudes. The one-dimensional
            length must be a power of two and at least two. Normalization is
            performed by the factory.

    Returns:
        A unary unitary over ``log2(len(amplitudes))`` coherent bits.

    Raises:
        TypeError: If the input is not a sequence of numeric scalars.
        ValueError: If its length is invalid, an element is nonfinite, or the
            vector is zero.
    """

    angles = amplitude_angles(amplitudes)
    width = angles.width
    ry_angles = angles.ry
    rz_angles = angles.rz
    phase = angles.phase
    schedule = _controlled_rotation_schedule(width)
    ry_operations = tuple(
        (ry_angles[index], target, control) for index, target, control in schedule
    )
    rz_operations = tuple(
        (rz_angles[index], target, control) for index, target, control in schedule
    )
    has_relative_phase = any(rz_angles)
    has_global_phase = bool(phase)

    if width == 1:

        @qm.qkernel
        def encode_one(
            state: qm.Q[qm.Bits[1]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
            """Prepare the configured one-component quantum state.

            Args:
                state (qm.Q[qm.Bits[1]]): Owned one-component zero state.

            Returns:
                qm.Q[qm.Bits[1]]: Prepared state with the configured amplitudes.
            """

            state = qm.ry(state, ry_angles[0])
            if has_relative_phase:
                state = qm.rz(state, rz_angles[0])
            if has_global_phase:
                state = qm.global_phase(state, phase)
            return state

        return encode_one

    @qm.qkernel
    def encode(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Prepare the configured multi-component quantum state.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned zero state to prepare.

        Returns:
            qm.Q[qm.Bits[width]]: Prepared state with the configured amplitudes.
        """

        state = qm.ry(state, ry_angles[0], target=width - 1)
        for operation in ry_operations:
            angle, target, control = operation
            state = qm.ry(state, angle, target=target)
            state = qm.cx(
                state,
                control=control,
                target=target,
            )
        if has_relative_phase:
            state = qm.rz(state, rz_angles[0], target=width - 1)
            for operation in rz_operations:
                angle, target, control = operation
                state = qm.rz(state, angle, target=target)
                state = qm.cx(
                    state,
                    control=control,
                    target=target,
                )
        if has_global_phase:
            state = qm.global_phase(state, phase)
        return state

    return encode


def amplitude_encoding_from_angles(width: int) -> qm.QKernel:
    """Construct a reusable runtime-parametric state-preparation unitary.

    The circuit topology depends only on ``width``. One compiled artifact can
    therefore prepare different vectors by binding coordinates returned from
    :func:`amplitude_angles` without recompilation. The aggregate ``Vector``
    parameters remain visible in Qamomile's ABI; circuit backends share one
    structural scalar-leaf projection rather than implementing this operation
    specially.

    Args:
        width: Positive number of coherent bits in the prepared state.

    Returns:
        A unitary accepting ``state``, ``ry_angles``, ``rz_angles``, and
        ``phase`` in that order.

    Raises:
        TypeError: If ``width`` is not an integer.
        ValueError: If ``width`` is not positive or exceeds the native
            state-preparation budget.
    """

    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if width <= 0:
        raise ValueError("width must be positive")
    schedule = _controlled_rotation_schedule(width)
    angle_count = (1 << width) - 1

    if width == 1:

        @qm.qkernel
        def encode_one_from_angles(
            state: qm.Q[qm.Bits[1]],  # type: ignore[valid-type]
            ry_angles: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
            rz_angles: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
            phase: qm.Param[qm.Real],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
            """Prepare one component from runtime rotation coordinates.

            Args:
                state (qm.Q[qm.Bits[1]]): Owned one-component zero state.
                ry_angles (qm.Param[qm.Vector[qm.Real, 1]]): Runtime Y-rotation
                    coordinates.
                rz_angles (qm.Param[qm.Vector[qm.Real, 1]]): Runtime Z-rotation
                    coordinates.
                phase (qm.Param[qm.Real]): Runtime global phase.

            Returns:
                qm.Q[qm.Bits[1]]: Prepared one-component state.
            """

            state = qm.ry(state, ry_angles[0])
            state = qm.rz(state, rz_angles[0])
            return qm.global_phase(state, phase)

        return encode_one_from_angles

    @qm.qkernel
    def encode_from_angles(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
        ry_angles: qm.Param[qm.Vector[qm.Real, angle_count]],  # type: ignore[valid-type]
        rz_angles: qm.Param[qm.Vector[qm.Real, angle_count]],  # type: ignore[valid-type]
        phase: qm.Param[qm.Real],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Prepare a state from runtime rotation coordinates.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned zero state to prepare.
            ry_angles (qm.Param[qm.Vector[qm.Real, angle_count]]): Runtime
                Y-rotation coordinates.
            rz_angles (qm.Param[qm.Vector[qm.Real, angle_count]]): Runtime
                Z-rotation coordinates.
            phase (qm.Param[qm.Real]): Runtime global phase.

        Returns:
            qm.Q[qm.Bits[width]]: Prepared state.
        """

        state = qm.ry(state, ry_angles[0], target=width - 1)
        for operation in schedule:
            state = qm.ry(
                state,
                ry_angles[operation[0]],
                target=operation[1],
            )
            state = qm.cx(
                state,
                control=operation[2],
                target=operation[1],
            )
        state = qm.rz(state, rz_angles[0], target=width - 1)
        for operation in schedule:
            state = qm.rz(
                state,
                rz_angles[operation[0]],
                target=operation[1],
            )
            state = qm.cx(
                state,
                control=operation[2],
                target=operation[1],
            )
        return qm.global_phase(state, phase)

    return encode_from_angles


def _native_amplitude_coordinates(
    amplitudes: object,
) -> tuple[int, Sequence[float], Sequence[float], float]:
    """Cross the required native coordinate-synthesis boundary."""

    if isinstance(amplitudes, (str, bytes)) or not isinstance(amplitudes, Sequence):
        raise TypeError("amplitudes must be a sequence of numeric scalars")
    coordinates: list[tuple[float, float]] = []
    for amplitude in amplitudes:
        if isinstance(amplitude, bool) or not isinstance(amplitude, numbers.Complex):
            raise TypeError("each amplitude must be a real or complex scalar")
        value = complex(amplitude)
        coordinates.append((value.real, value.imag))
    derive = getattr(native, "mottonen_amplitude_coordinates", None)
    if derive is None:
        raise TypeError("Q7500: native amplitude coordinate synthesis is unavailable")
    result = derive(coordinates)
    if not isinstance(result, tuple) or len(result) != 4:
        raise TypeError("Q7506: malformed native amplitude coordinates")
    return cast(tuple[int, Sequence[float], Sequence[float], float], result)


def _controlled_rotation_schedule(width: int) -> tuple[tuple[int, int, int], ...]:
    """Return angle indices and wires for the width-dependent Gray walk."""

    derive = getattr(native, "mottonen_rotation_schedule", None)
    if derive is None:
        raise TypeError("Q7500: native amplitude schedule synthesis is unavailable")
    result = derive(width)
    expected_count = (1 << width) - 2
    if (
        not isinstance(result, list)
        or len(result) != expected_count
        or any(
            not isinstance(operation, tuple)
            or len(operation) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in operation
            )
            or not 1 <= operation[0] <= expected_count
            or not 0 <= operation[1] < width
            or not 0 <= operation[2] < width
            or operation[1] == operation[2]
            for operation in result
        )
    ):
        raise TypeError("Q7506: malformed native amplitude schedule")
    return tuple(result)


__all__ = [
    "AmplitudeAngles",
    "amplitude_angles",
    "amplitude_encoding",
    "amplitude_encoding_from_angles",
]
