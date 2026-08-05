import math
from numbers import Integral, Real
from typing import Any, cast


def is_close_zero(value: float, abs_tol: float = 1e-15) -> bool:
    """
    Check if a given floating-point value is close to zero within a small tolerance.

    Args:
        value (float): The floating-point value to check.
        abs_tol (float): Absolute tolerance passed to :func:`math.isclose`.
            Defaults to ``1e-15``.

    Returns:
        bool: True if the value is close to zero, False otherwise.
    """
    return math.isclose(value, 0.0, abs_tol=abs_tol)


def is_plain_int(value: object) -> bool:
    """Return True if ``value`` is a Python ``int`` but not a ``bool``.

    ``bool`` is a subclass of ``int`` in Python, so ``isinstance(True, int)``
    is ``True``. This helper distinguishes a genuine integer from a boolean,
    which matters wherever a boolean must be rejected in an integer slot — for
    example, validating decoded wire data or a register width.

    Args:
        value (object): The value to test.

    Returns:
        bool: ``True`` when ``value`` is an ``int`` and not a ``bool``.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def coerce_nonnegative_integral(value: object, *, label: str) -> int:
    """Normalize a real scalar with an integer value to a nonnegative integer.

    Args:
        value (object): Candidate Python, NumPy, or SymPy real scalar.
        label (str): User-facing field label used in diagnostics.

    Returns:
        int: Equivalent nonnegative Python integer.

    Raises:
        TypeError: If ``value`` is Boolean, is not a real scalar, is not
            finite, or does not have an integer value.
        ValueError: If the normalized integer is negative.
    """
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a nonnegative integer, got bool ({value}).")
    if isinstance(value, Integral):
        normalized = int(cast(Any, value))
    elif isinstance(value, Real):
        try:
            normalized = int(cast(Any, value))
        except (OverflowError, TypeError, ValueError) as error:
            raise TypeError(
                f"{label} must be a finite integer, got {value}."
            ) from error
        try:
            remainder = value - normalized
            symbolic_zero = getattr(remainder, "is_zero", None)
            is_integral = symbolic_zero is True or (
                symbolic_zero is None and bool(remainder == 0)
            )
        except (ArithmeticError, TypeError):
            is_integral = False
        if not is_integral:
            raise TypeError(
                f"{label} must be an integer, got non-integer float-like "
                f"real value {value}."
            )
    else:
        raise TypeError(
            f"{label} must be a nonnegative integer, got {type(value).__name__}."
        )
    if normalized < 0:
        raise ValueError(f"{label} must be nonnegative, got {normalized}.")
    return normalized
