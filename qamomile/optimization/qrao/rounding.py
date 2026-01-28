"""Rounding algorithms for QRAO.

This module provides rounding algorithms that convert Pauli expectation
values into discrete spin values (+1 or -1).
"""

from __future__ import annotations


class SignRounder:
    """Simple sign-based rounding.

    Rounds Pauli expectation values to spin values using the sign function:
    - <P_i> >= 0 → s_i = +1
    - <P_i> < 0  → s_i = -1

    Example:
        >>> rounder = SignRounder()
        >>> expectations = [0.8, -0.3, 0.1]  # Pauli expectation values
        >>> spins = rounder.round(expectations)
        >>> print(spins)  # [1, -1, 1]
    """

    def round(self, expectations: list[float]) -> list[int]:
        """Round Pauli expectations to spin values.

        Args:
            expectations: List of Pauli expectation values for each variable.
                         Each value should be in range [-1, 1].

        Returns:
            List of spin values (+1 or -1) for each variable
        """
        return [1 if exp >= 0 else -1 for exp in expectations]
