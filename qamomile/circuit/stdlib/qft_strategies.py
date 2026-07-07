"""Decomposition strategies for QFT and IQFT.

This module provides multiple decomposition strategies for the Quantum Fourier
Transform, allowing trade-offs between precision and gate count.

Strategies:
    StandardQFTStrategy: Full precision QFT with O(n^2) gates
    ApproximateQFTStrategy: Truncated rotations with O(n*k) gates

Example:
    from qamomile.circuit.stdlib.qft_strategies import (
        StandardQFTStrategy,
        ApproximateQFTStrategy,
    )
    from qamomile.circuit.stdlib.qft import QFT

    # Register strategies
    QFT.register_strategy("standard", StandardQFTStrategy())
    QFT.register_strategy("approximate", ApproximateQFTStrategy(truncation_depth=3))

    # Use approximate strategy
    qft_gate = QFT(5)
    result = qft_gate(q0, q1, q2, q3, q4, strategy="approximate")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import qamomile.circuit as qmc

if TYPE_CHECKING:
    from qamomile.circuit.frontend.handle.primitives import Qubit


@dataclass
class StandardQFTStrategy:
    """Standard QFT decomposition: H + CP + SWAP (full precision).

    This strategy implements the standard QFT decomposition with:
    - n Hadamard gates
    - n(n-1)/2 controlled phase gates
    - n/2 SWAP gates for bit reversal

    Total gate count: O(n^2)
    """

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return "standard"

    def decompose(self, qubits: tuple["Qubit", ...]) -> tuple["Qubit", ...]:
        """Decompose QFT into elementary gates.

        Args:
            qubits: Input qubits

        Returns:
            Output qubits after QFT transformation
        """
        n = len(qubits)
        qubits_list = list(qubits)

        # Apply QFT rotations (from highest index to lowest)
        for j in range(n - 1, -1, -1):
            # Apply H gate
            qubits_list[j] = qmc.h(qubits_list[j])

            # Apply controlled phase rotations
            for k in range(j - 1, -1, -1):
                angle = math.pi / (2 ** (j - k))
                qubits_list[j], qubits_list[k] = qmc.cp(
                    qubits_list[j], qubits_list[k], angle
                )

        # Swap qubits to reverse order
        for j in range(n // 2):
            qubits_list[j], qubits_list[n - j - 1] = qmc.swap(
                qubits_list[j], qubits_list[n - j - 1]
            )

        return tuple(qubits_list)

@dataclass
class ApproximateQFTStrategy:
    """Approximate QFT decomposition with truncated rotations.

    This strategy truncates small-angle rotations to reduce gate count
    while maintaining acceptable precision for many applications.

    For a given truncation_depth k:
    - Only controlled phases with angle >= pi/2^k are applied
    - Gate count is O(n*k) instead of O(n^2)
    - Error scales as O(n/2^k)

    Attributes:
        truncation_depth: Maximum exponent for controlled phase gates.
            Larger values give higher precision but more gates.
            Default is 3 (angles >= pi/8).
    """

    truncation_depth: int = 3

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return f"approximate_k{self.truncation_depth}"

    def decompose(self, qubits: tuple["Qubit", ...]) -> tuple["Qubit", ...]:
        """Decompose QFT with truncated rotations.

        Args:
            qubits: Input qubits

        Returns:
            Output qubits after approximate QFT transformation
        """
        n = len(qubits)
        k = self.truncation_depth
        qubits_list = list(qubits)

        # Apply QFT rotations with truncation
        for j in range(n - 1, -1, -1):
            # Apply H gate
            qubits_list[j] = qmc.h(qubits_list[j])

            # Apply controlled phase rotations (truncated)
            # Only include rotations where exponent <= truncation_depth
            for m in range(j - 1, max(j - k - 1, -1), -1):
                exponent = j - m
                if exponent <= k:
                    angle = math.pi / (2**exponent)
                    qubits_list[j], qubits_list[m] = qmc.cp(
                        qubits_list[j], qubits_list[m], angle
                    )

        # Swap qubits to reverse order (same as standard)
        for j in range(n // 2):
            qubits_list[j], qubits_list[n - j - 1] = qmc.swap(
                qubits_list[j], qubits_list[n - j - 1]
            )

        return tuple(qubits_list)

@dataclass
class StandardIQFTStrategy:
    """Standard inverse QFT decomposition (full precision).

    This strategy implements the standard IQFT decomposition, which is
    the inverse of the standard QFT.
    """

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return "standard"

    def decompose(self, qubits: tuple["Qubit", ...]) -> tuple["Qubit", ...]:
        """Decompose IQFT into elementary gates.

        Args:
            qubits: Input qubits

        Returns:
            Output qubits after IQFT transformation
        """
        n = len(qubits)
        qubits_list = list(qubits)

        # Swap qubits to reverse order (first)
        for j in range(n // 2):
            qubits_list[j], qubits_list[n - j - 1] = qmc.swap(
                qubits_list[j], qubits_list[n - j - 1]
            )

        # Apply inverse QFT rotations (from lowest index to highest)
        for j in range(n):
            # Apply inverse controlled phase rotations first
            for k in range(j):
                angle = -math.pi / (2 ** (j - k))
                qubits_list[j], qubits_list[k] = qmc.cp(
                    qubits_list[j], qubits_list[k], angle
                )
            # Apply H gate
            qubits_list[j] = qmc.h(qubits_list[j])

        return tuple(qubits_list)

@dataclass
class ApproximateIQFTStrategy:
    """Approximate inverse QFT decomposition with truncated rotations.

    This strategy truncates small-angle rotations in the IQFT,
    mirroring the ApproximateQFTStrategy.

    Attributes:
        truncation_depth: Maximum exponent for controlled phase gates.
    """

    truncation_depth: int = 3

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return f"approximate_k{self.truncation_depth}"

    def decompose(self, qubits: tuple["Qubit", ...]) -> tuple["Qubit", ...]:
        """Decompose IQFT with truncated rotations.

        Args:
            qubits: Input qubits

        Returns:
            Output qubits after approximate IQFT transformation
        """
        n = len(qubits)
        k = self.truncation_depth
        qubits_list = list(qubits)

        # Swap qubits to reverse order (first)
        for j in range(n // 2):
            qubits_list[j], qubits_list[n - j - 1] = qmc.swap(
                qubits_list[j], qubits_list[n - j - 1]
            )

        # Apply inverse QFT rotations with truncation
        for j in range(n):
            # Apply inverse controlled phase rotations (truncated)
            for m in range(max(0, j - k), j):
                exponent = j - m
                if exponent <= k:
                    angle = -math.pi / (2**exponent)
                    qubits_list[j], qubits_list[m] = qmc.cp(
                        qubits_list[j], qubits_list[m], angle
                    )
            # Apply H gate
            qubits_list[j] = qmc.h(qubits_list[j])

        return tuple(qubits_list)
