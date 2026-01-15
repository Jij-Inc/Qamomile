"""Backend capabilities for transpilation."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class BackendCapabilities:
    """Describes what features a quantum backend supports.

    Used by SeparatePass to determine what transformations are needed.
    """

    # Control flow support
    supports_dynamic_circuits: bool = False
    """Whether the backend can execute if/else based on measurement results."""

    # Quantum register types
    supports_qfixed: bool = False
    """Whether the backend natively supports QFixed (fixed-point quantum registers)."""

    supports_quint: bool = False
    """Whether the backend natively supports QUInt (quantum unsigned integers)."""

    # Hardware constraints
    max_qubits: int | None = None
    """Maximum number of qubits available (None = unlimited)."""

    max_classical_bits: int | None = None
    """Maximum number of classical bits available (None = unlimited)."""

    # Loop support
    supports_for_loops: bool = False
    """Whether the backend supports for loops natively (vs unrolling required)."""

    supports_while_loops: bool = False
    """Whether the backend supports while loops natively."""
