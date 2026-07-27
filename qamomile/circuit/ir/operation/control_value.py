"""Shared activation-value semantics for coherent quantum controls."""

from __future__ import annotations

from qamomile._utils import is_plain_int


def normalize_control_value(
    control_value: int | None,
    num_controls: int,
) -> int | None:
    """Normalize an integer activation state for a control register.

    Control qubits follow Qamomile's LSB-first integer convention: bit ``j``
    of ``control_value`` describes the ``j``-th flattened control operand.
    ``None`` and the all-ones value are the canonical ordinary-control state.

    Args:
        control_value (int | None): Required computational-basis value, or
            ``None`` for the ordinary all-ones control state.
        num_controls (int): Concrete positive control-register width.

    Returns:
        int | None: A non-default activation value, or ``None`` for all-ones.

    Raises:
        TypeError: If ``control_value`` is not a Python ``int`` or ``None``.
        ValueError: If ``num_controls`` is not positive, or if
            ``control_value`` does not fit in the control-register width.
    """
    if not is_plain_int(num_controls) or num_controls < 1:
        raise ValueError(
            f"num_controls must be a positive Python int, got {num_controls!r}."
        )
    if control_value is None:
        return None
    if not is_plain_int(control_value):
        raise TypeError(
            "control_value must be a Python int or None, "
            f"got {type(control_value).__name__}."
        )
    maximum = (1 << num_controls) - 1
    if not 0 <= control_value <= maximum:
        raise ValueError(
            f"control_value {control_value} does not fit in num_controls="
            f"{num_controls} (valid range 0..{maximum})."
        )
    return None if control_value == maximum else control_value


def control_pattern_for_value(
    control_value: int | None,
    num_controls: int,
) -> tuple[int, ...]:
    """Return the LSB-first activation pattern for a control value.

    Args:
        control_value (int | None): Required computational-basis value.
            ``None`` means the ordinary all-ones control state.
        num_controls (int): Concrete positive control-register width.

    Returns:
        tuple[int, ...]: One ``0``/``1`` activation bit per flattened control,
            with the first control represented by bit zero.

    Raises:
        TypeError: If ``control_value`` is not a Python ``int`` or ``None``.
        ValueError: If the width or activation value is invalid.

    Example:
        >>> control_pattern_for_value(2, 2)
        (0, 1)
        >>> control_pattern_for_value(None, 2)
        (1, 1)
    """
    normalized = normalize_control_value(control_value, num_controls)
    value = (1 << num_controls) - 1 if normalized is None else normalized
    return tuple((value >> bit) & 1 for bit in range(num_controls))


__all__ = ["control_pattern_for_value", "normalize_control_value"]
