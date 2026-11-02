# ruff: noqa: F821 -- widths are closed into dependent source annotations.

"""Uniformly controlled rotations as ordinary functional composition."""

from __future__ import annotations

from collections.abc import Sequence

import qamomile as qm
from qamomile._native import compiler as native


def uniformly_controlled_angles(branch_angles: Sequence[float]) -> tuple[float, ...]:
    """Convert address-ordered RY angles to Gray-walk coordinates.

    Args:
        branch_angles: One desired finite rotation per computational-basis
            address. The nonempty length must be a power of two.

    Returns:
        Normalized Walsh-Hadamard coordinates in cyclic Gray order.

    Raises:
        TypeError: If the input is not a sequence of real scalars.
        ValueError: If its length or one scalar is invalid.
    """

    if not isinstance(branch_angles, Sequence):
        raise TypeError("branch_angles must be a sequence of real scalars")
    values: list[float] = []
    for angle in branch_angles:
        if isinstance(angle, bool) or not isinstance(angle, (int, float)):
            raise TypeError("branch_angles must contain real scalars")
        values.append(float(angle))
    convert = getattr(native, "uniformly_controlled_rotation_coordinates", None)
    if convert is None:
        raise TypeError(
            "Q7510: native uniformly controlled coordinates are unavailable"
        )
    coordinates = convert(values)
    if (
        not isinstance(coordinates, list)
        or len(coordinates) != len(values)
        or any(not isinstance(angle, float) for angle in coordinates)
    ):
        raise TypeError("Q7515: malformed native uniformly controlled coordinates")
    return tuple(coordinates)


def uniformly_controlled_ry(branch_angles: Sequence[float]) -> qm.QKernel:
    """Construct an address-controlled Y rotation from desired branch angles.

    The leading ``log2(len(branch_angles))`` components form the retained
    little-endian address and the final component is the target. Host-side
    angle conversion is closed into an ordinary RY/CX body, so no multiplexer
    operation or backend implementation is introduced.

    Args:
        branch_angles: One desired rotation per computational-basis address.

    Returns:
        A unary unitary over the address and one target component.

    Raises:
        TypeError: If the angle input is invalid.
        ValueError: If its length or one scalar is invalid.
    """

    angles = uniformly_controlled_angles(branch_angles)
    count = len(angles)
    control_width = count.bit_length() - 1
    controls = _gray_controls(control_width)
    if control_width == 0:

        @qm.qkernel
        def rotate_one(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
            """Apply the sole configured branch rotation.

            Args:
                state (qm.Q[qm.Bits[1]]): Owned target component.

            Returns:
                qm.Q[qm.Bits[1]]: Rotated target component.
            """

            return qm.ry(state, angles[0])

        return rotate_one
    total_width = control_width + 1

    @qm.qkernel
    def identity(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Return the address and target state unchanged.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Owned complete state.

        Returns:
            qm.Q[qm.Bits[total_width]]: Unchanged state.
        """

        return state

    current = identity
    for index, control in enumerate(controls):
        previous = current
        angle = angles[index]

        @qm.qkernel
        def append_step(
            state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
            """Append one configured Gray-walk rotation step.

            Args:
                state (qm.Q[qm.Bits[total_width]]): Owned complete state.

            Returns:
                qm.Q[qm.Bits[total_width]]: State after the accumulated step.
            """

            state = previous(state)
            state = qm.ry(state, angle, target=control_width)
            return qm.cx(state, control=control, target=control_width)

        current = append_step
    return current


def uniformly_controlled_ry_from_angles(control_width: int) -> qm.QKernel:
    """Construct a compile-once Gray-walk RY multiplexer.

    This is the runtime-coordinate companion to
    :func:`uniformly_controlled_ry`. Its ``angles`` parameter accepts the
    output of :func:`uniformly_controlled_angles` through the common aggregate
    scalar-leaf ABI.

    Args:
        control_width: Nonnegative number of leading address components.

    Returns:
        A unitary accepting ``state`` and a runtime angle vector.

    Raises:
        TypeError: If ``control_width`` is not an integer.
        ValueError: If ``control_width`` is negative or exceeds the native
            uniformly controlled rotation budget.
    """

    if isinstance(control_width, bool) or not isinstance(control_width, int):
        raise TypeError("control_width must be an integer")
    if control_width < 0:
        raise ValueError("control_width must be nonnegative")
    controls = _gray_controls(control_width)
    count = 1 << control_width
    total_width = control_width + 1
    if control_width == 0:

        @qm.qkernel
        def rotate_one(
            state: qm.Q[qm.Bits[1]],  # type: ignore[valid-type]
            angles: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
            """Apply one runtime branch rotation.

            Args:
                state (qm.Q[qm.Bits[1]]): Owned target component.
                angles (qm.Param[qm.Vector[qm.Real, 1]]): Runtime rotation
                    coordinate.

            Returns:
                qm.Q[qm.Bits[1]]: Rotated target component.
            """

            return qm.ry(state, angles[0])

        return rotate_one

    @qm.qkernel
    def identity(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
        angles: qm.Param[qm.Vector[qm.Real, count]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Return the state unchanged while accepting runtime coordinates.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Owned complete state.
            angles (qm.Param[qm.Vector[qm.Real, count]]): Runtime rotation
                coordinates.

        Returns:
            qm.Q[qm.Bits[total_width]]: Unchanged state.
        """

        return state

    current = identity
    for index, control in enumerate(controls):
        previous = current

        @qm.qkernel
        def append_step(
            state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
            angles: qm.Param[qm.Vector[qm.Real, count]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
            """Append one runtime Gray-walk rotation step.

            Args:
                state (qm.Q[qm.Bits[total_width]]): Owned complete state.
                angles (qm.Param[qm.Vector[qm.Real, count]]): Runtime rotation
                    coordinates.

            Returns:
                qm.Q[qm.Bits[total_width]]: State after the accumulated step.
            """

            state = previous(state, angles)
            state = qm.ry(state, angles[index], target=control_width)
            return qm.cx(state, control=control, target=control_width)

        current = append_step
    return current


def _gray_controls(width: int) -> tuple[int, ...]:
    """Return control components for one closed cyclic Gray walk."""

    derive = getattr(native, "cyclic_gray_control_schedule", None)
    if derive is None:
        raise TypeError("Q7510: native uniformly controlled schedule is unavailable")
    result = derive(width)
    expected_count = 0 if width == 0 else 1 << width
    if (
        not isinstance(result, list)
        or len(result) != expected_count
        or any(
            isinstance(control, bool)
            or not isinstance(control, int)
            or not 0 <= control < width
            for control in result
        )
    ):
        raise TypeError("Q7515: malformed native uniformly controlled schedule")
    return tuple(result)


__all__ = [
    "uniformly_controlled_angles",
    "uniformly_controlled_ry",
    "uniformly_controlled_ry_from_angles",
]
