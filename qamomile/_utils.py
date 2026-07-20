import math


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
