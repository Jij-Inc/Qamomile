"""Standard library of quantum algorithms.

Available:
    Classes:
        - QFT: Quantum Fourier Transform (CompositeGate class)
        - IQFT: Inverse Quantum Fourier Transform (CompositeGate class)

    Strategies:
        - StandardQFTStrategy: Full precision QFT
        - ApproximateQFTStrategy: Truncated rotations QFT
        - StandardIQFTStrategy: Full precision IQFT
        - ApproximateIQFTStrategy: Truncated rotations IQFT

    Functions:
        - qft: Apply QFT to a vector of qubits
        - iqft: Apply IQFT to a vector of qubits
        - qpe: Quantum Phase Estimation

Example:
    # Using class-based API (recommended for custom gates)
    from qamomile.circuit.stdlib import QFT, IQFT

    class MyCustomGate(CompositeGate):
        def _decompose(self, qubits):
            # Use QFT as building block
            ...

    # Using function-based API (recommended for kernels)
    from qamomile.circuit.stdlib import qft, iqft

    @qmc.qkernel
    def my_algorithm(qubits: Vector[Qubit]) -> Vector[Qubit]:
        qubits = qft(qubits)
        return qubits

    # Using strategies
    qft_gate = QFT(5)
    result = qft_gate(q0, q1, q2, q3, q4, strategy="approximate")
"""

# Class-based API (new)
from .qft import QFT, IQFT

# Function-based API (kept for compatibility, using new class-based impl)
from .qft import qft, iqft

# Strategies
from .qft_strategies import (
    StandardQFTStrategy,
    ApproximateQFTStrategy,
    StandardIQFTStrategy,
    ApproximateIQFTStrategy,
)

from .qpe import qpe

__all__ = [
    # Classes
    "QFT",
    "IQFT",
    # Strategies
    "StandardQFTStrategy",
    "ApproximateQFTStrategy",
    "StandardIQFTStrategy",
    "ApproximateIQFTStrategy",
    # Functions
    "qft",
    "iqft",
    "qpe",
]
