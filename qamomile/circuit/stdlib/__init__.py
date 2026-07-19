"""Expose standard-library quantum callables.

The reader-facing circuit API is function-oriented: use :func:`qft`,
:func:`iqft`, :func:`qpe`, state-preparation helpers, arithmetic helpers, and
:func:`mcx` inside qkernels. Factories that must also expose algorithm metadata
may return frozen non-callable descriptors; invoke the descriptor's documented
qkernel field rather than the descriptor itself. Internally these functions
emit named callables with Qamomile bodies and optional backend-native
implementations.

Standard composites use the same ``composite_gate`` mechanism as user
callables; there is no separate class-based gate hierarchy.

Example:
    ```python
    import qamomile.circuit as qmc

    @qmc.qkernel
    def my_algorithm(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        qubits = qmc.qft(qubits)
        return qmc.iqft(qubits)
    ```
"""

from .arithmetic import (
    controlled_modular_add,
    modmul_const,
    modular_add,
    modular_decrement,
    modular_increment,
    ripple_carry_add,
)
from .grover import grover_iteration_count, grover_search
from .ising_z_block_encoding import IsingZBlockEncoding, ising_z_block_encoding
from .lcu_block_encoding import (
    LCUBlockEncoding,
    LCUBlockEncodingTerm,
    identity_block_encoding,
    lcu_block_encoding,
)
from .multi_controlled_x import mcx, multi_controlled_x
from .pauli_lcu_block_encoding import (
    PauliLCUBlockEncoding,
    pauli_lcu_block_encoding,
)
from .periodic_shift_lcu_block_encoding import (
    PeriodicShiftLCUBlockEncoding,
    periodic_shift_lcu_block_encoding,
)
from .qft import iqft, qft
from .qpe import qpe
from .shor import shor_order_finding
from .state_preparation import (
    amplitude_encoding,
    amplitude_encoding_from_angles,
    computational_basis_state,
    mottonen_amplitude_encoding,
    mottonen_amplitude_encoding_from_angles,
)

__all__ = [
    "qft",
    "iqft",
    "qpe",
    "mcx",
    "multi_controlled_x",
    "LCUBlockEncoding",
    "LCUBlockEncodingTerm",
    "identity_block_encoding",
    "lcu_block_encoding",
    "IsingZBlockEncoding",
    "ising_z_block_encoding",
    "PauliLCUBlockEncoding",
    "pauli_lcu_block_encoding",
    "PeriodicShiftLCUBlockEncoding",
    "periodic_shift_lcu_block_encoding",
    "computational_basis_state",
    "amplitude_encoding",
    "amplitude_encoding_from_angles",
    "mottonen_amplitude_encoding",
    "mottonen_amplitude_encoding_from_angles",
    # Arithmetic
    "ripple_carry_add",
    "modular_increment",
    "modular_decrement",
    "modular_add",
    "controlled_modular_add",
    "modmul_const",
    "shor_order_finding",
    "grover_search",
    "grover_iteration_count",
]
