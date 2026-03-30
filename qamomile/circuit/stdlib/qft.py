"""Quantum Fourier Transform implementation using CompositeGate.

This module provides QFT and IQFT as CompositeGate classes, serving as
a reference for implementing custom composite gates using the frontend API.

Multiple decomposition strategies are available:
    - "standard": Full precision QFT with O(n^2) gates
    - "approximate": Truncated rotations with O(n*k) gates (default k=3)
    - "approximate_k2": Truncated rotations with k=2

Example:
    from qamomile.circuit.stdlib.qft import QFT, IQFT, qft, iqft

    @qmc.qkernel
    def my_algorithm(qubits: Vector[Qubit]) -> Vector[Qubit]:
        # Using factory function
        qubits = qft(qubits)
        # ... some operations ...
        qubits = iqft(qubits)
        return qubits

    # Or using class directly with strategy selection
    qft_gate = QFT(3)
    result = qft_gate(q0, q1, q2, strategy="approximate")

    # Compare resources
    standard_resources = qft_gate.get_resources_for_strategy("standard")
    approx_resources = qft_gate.get_resources_for_strategy("approximate")
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateType,
    ResourceMetadata,
)

# Import strategies
from qamomile.circuit.stdlib.qft_strategies import (
    ApproximateIQFTStrategy,
    ApproximateQFTStrategy,
    StandardIQFTStrategy,
    StandardQFTStrategy,
)

if TYPE_CHECKING:
    pass


class QFT(CompositeGate):
    """Quantum Fourier Transform composite gate.

    The QFT is the quantum analog of the discrete Fourier transform.
    It's a key component of many quantum algorithms.

    Available strategies:
        - "standard": Full precision QFT (default)
        - "approximate": Truncated rotations (k=3)
        - "approximate_k2": Truncated rotations (k=2)

    Example:
        # Create QFT for 3 qubits
        qft_gate = QFT(3)

        # Apply to qubits (uses default strategy)
        result = qft_gate(q0, q1, q2)

        # Apply with specific strategy
        result = qft_gate(q0, q1, q2, strategy="approximate")

        # Compare resources between strategies
        standard = qft_gate.get_resources_for_strategy("standard")
        approx = qft_gate.get_resources_for_strategy("approximate")

        # Or use the factory function
        qubits = qft(qubit_vector)
    """

    gate_type = CompositeGateType.QFT
    custom_name = "qft"

    # Initialize strategy registry for QFT
    _strategies = {
        "standard": StandardQFTStrategy(),
        "approximate": ApproximateQFTStrategy(truncation_depth=3),
        "approximate_k2": ApproximateQFTStrategy(truncation_depth=2),
        "approximate_k4": ApproximateQFTStrategy(truncation_depth=4),
    }
    _default_strategy = "standard"

    def __init__(self, num_qubits: int):
        """Initialize QFT gate.

        Args:
            num_qubits: Number of qubits for the QFT
        """
        self._num_qubits = num_qubits

    @property
    def num_target_qubits(self) -> int:
        """Return the number of target qubits."""
        return self._num_qubits

    def _decompose(
        self,
        qubits: tuple[Qubit, ...],
    ) -> tuple[Qubit, ...]:
        """Decompose QFT into elementary gates.

        Args:
            qubits: Tuple of input qubits

        Returns:
            Tuple of output qubits after QFT transformation
        """
        n = self._num_qubits
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

    def _resources(self) -> ResourceMetadata:
        """Return resource metadata for QFT.

        QFT uses O(n^2) gates but no T gates in the standard decomposition.
        """
        n = self._num_qubits
        num_h = n
        num_cp = n * (n - 1) // 2
        num_swap = n // 2
        return ResourceMetadata(
            t_gates=0,
            total_gates=num_h + num_cp + num_swap,
            single_qubit_gates=num_h,
            two_qubit_gates=num_cp + num_swap,
            clifford_gates=num_h + num_swap,
            rotation_gates=num_cp,
            custom_metadata={
                "num_h_gates": num_h,
                "num_cp_gates": num_cp,
                "num_swap_gates": num_swap,
                "total_gates": num_h + num_cp + num_swap,
            },
        )


class IQFT(CompositeGate):
    """Inverse Quantum Fourier Transform composite gate.

    The IQFT is the inverse of the QFT. It's a key component of:
    - Quantum Phase Estimation (QPE)
    - Shor's algorithm
    - Quantum counting

    Available strategies:
        - "standard": Full precision IQFT (default)
        - "approximate": Truncated rotations (k=3)
        - "approximate_k2": Truncated rotations (k=2)

    Example:
        # Create IQFT for 3 qubits
        iqft_gate = IQFT(3)

        # Apply to qubits
        result = iqft_gate(q0, q1, q2)

        # Apply with specific strategy
        result = iqft_gate(q0, q1, q2, strategy="approximate")

        # Or use the factory function
        qubits = iqft(qubit_vector)
    """

    gate_type = CompositeGateType.IQFT
    custom_name = "iqft"

    # Initialize strategy registry for IQFT
    _strategies = {
        "standard": StandardIQFTStrategy(),
        "approximate": ApproximateIQFTStrategy(truncation_depth=3),
        "approximate_k2": ApproximateIQFTStrategy(truncation_depth=2),
        "approximate_k4": ApproximateIQFTStrategy(truncation_depth=4),
    }
    _default_strategy = "standard"

    def __init__(self, num_qubits: int):
        """Initialize IQFT gate.

        Args:
            num_qubits: Number of qubits for the IQFT
        """
        self._num_qubits = num_qubits

    @property
    def num_target_qubits(self) -> int:
        """Return the number of target qubits."""
        return self._num_qubits

    def _decompose(
        self,
        qubits: tuple[Qubit, ...],
    ) -> tuple[Qubit, ...]:
        """Decompose IQFT into elementary gates.

        Args:
            qubits: Tuple of input qubits

        Returns:
            Tuple of output qubits after IQFT transformation
        """
        n = self._num_qubits
        qubits_list = list(qubits)

        # Swap qubits to reverse order
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

    def _resources(self) -> ResourceMetadata:
        """Return resource metadata for IQFT.

        IQFT uses the same resources as QFT.
        """
        n = self._num_qubits
        num_h = n
        num_cp = n * (n - 1) // 2
        num_swap = n // 2
        return ResourceMetadata(
            t_gates=0,
            total_gates=num_h + num_cp + num_swap,
            single_qubit_gates=num_h,
            two_qubit_gates=num_cp + num_swap,
            clifford_gates=num_h + num_swap,
            rotation_gates=num_cp,
            custom_metadata={
                "num_h_gates": num_h,
                "num_cp_gates": num_cp,
                "num_swap_gates": num_swap,
                "total_gates": num_h + num_cp + num_swap,
            },
        )


def _get_size(arr: Vector[Qubit]) -> int:
    """Get array size as Python int.

    Args:
        arr: A Vector of Qubits

    Returns:
        The size of the array as an integer

    Raises:
        ValueError: If the array doesn't have a fixed size
    """
    size = arr.shape[0]
    if isinstance(size, int):
        return size
    if hasattr(size, "value") and size.value.is_constant():
        val = size.value.get_const()
        if val is not None:
            return int(val)
    if hasattr(size, "init_value"):
        return int(size.init_value)
    raise ValueError("Array must have fixed size")


def qft(qubits: Vector[Qubit]) -> Vector[Qubit]:
    """Apply Quantum Fourier Transform to a vector of qubits.

    This is a convenience factory function that creates a QFT gate
    and applies it to the qubits.

    Args:
        qubits: Vector of qubits to transform

    Returns:
        Transformed qubits (same vector, modified in place)

    Example:
        @qmc.qkernel
        def my_algorithm(qubits: Vector[Qubit]) -> Vector[Qubit]:
            qubits = qft(qubits)
            return qubits
    """
    n = _get_size(qubits)
    qft_gate = QFT(n)

    # Get individual qubits from vector
    qubit_list = [qubits[i] for i in range(n)]

    # Apply QFT gate
    result = qft_gate(*qubit_list)

    # Write results back to vector
    for i in range(n):
        qubits[i] = result[i]

    return qubits


def iqft(qubits: Vector[Qubit]) -> Vector[Qubit]:
    """Apply Inverse Quantum Fourier Transform to a vector of qubits.

    This is a convenience factory function that creates an IQFT gate
    and applies it to the qubits.

    Args:
        qubits: Vector of qubits to transform

    Returns:
        Transformed qubits (same vector, modified in place)

    Example:
        @qmc.qkernel
        def my_algorithm(qubits: Vector[Qubit]) -> Vector[Qubit]:
            qubits = iqft(qubits)
            return qubits
    """
    n = _get_size(qubits)
    iqft_gate = IQFT(n)

    # Get individual qubits from vector
    qubit_list = [qubits[i] for i in range(n)]

    # Apply IQFT gate
    result = iqft_gate(*qubit_list)

    # Write results back to vector
    for i in range(n):
        qubits[i] = result[i]

    return qubits
