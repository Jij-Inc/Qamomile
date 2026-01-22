"""Standard library of quantum algorithms.

Available:
    Classes:
        - QFT: Quantum Fourier Transform (CompositeGate class)
        - IQFT: Inverse Quantum Fourier Transform (CompositeGate class)

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
"""

# Class-based API (new)
from .qft import QFT, IQFT

# Function-based API (kept for compatibility, using new class-based impl)
from .qft import qft, iqft

from .qpe import qpe

__all__ = [
    # Classes
    "QFT",
    "IQFT",
    # Functions
    "qft",
    "iqft",
    "qpe",
]
