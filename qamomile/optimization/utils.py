import math


def is_close_zero(value: float, abs_tol=1e-15) -> bool:
    """
    Check if a given floating-point value is close to zero within a small tolerance.

    Args:
        value (float): The floating-point value to check.

    Returns:
        bool: True if the value is close to zero, False otherwise.
    """
    return math.isclose(value, 0.0, abs_tol=abs_tol)
