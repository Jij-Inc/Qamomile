"""Expose standard-library quantum callables.

The reader-facing circuit API is function-oriented: use :func:`qft`,
:func:`iqft`, :func:`qpe`, :func:`qsvt`, state-preparation helpers, arithmetic
helpers, and :func:`mcx` inside qkernels. Factories that must also expose
algorithm metadata may return frozen non-callable descriptors; invoke the
descriptor's documented qkernel field rather than the descriptor itself.
Internally these functions emit named callables with Qamomile bodies and
optional backend-native implementations.

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
    add_const,
    controlled_add_const,
    controlled_modular_add,
    controlled_modular_add_const,
    controlled_modular_add_const_modulus,
    lookup_xor,
    modmul_const,
    modular_add,
    modular_add_const,
    modular_decrement,
    modular_increment,
    ripple_carry_add,
)
from .block_encoding import (
    IsingZBlockEncoding,
    LCUBlockEncoding,
    LCUBlockEncodingTerm,
    PauliLCUBlockEncoding,
    PeriodicShiftLCUBlockEncoding,
    identity_block_encoding,
    ising_z_block_encoding,
    lcu_block_encoding,
    pauli_lcu_block_encoding,
    periodic_shift_lcu_block_encoding,
)
from .grover import grover_iteration_count, grover_search
from .multi_controlled_x import mcx, multi_controlled_x
from .qft import iqft, qft
from .qpe import qpe
from .qsvt import qsvt
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
    "qsvt",
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
    "add_const",
    "controlled_add_const",
    "modular_increment",
    "modular_decrement",
    "modular_add",
    "controlled_modular_add",
    "modular_add_const",
    "controlled_modular_add_const",
    "controlled_modular_add_const_modulus",
    "lookup_xor",
    "modmul_const",
    "grover_search",
    "grover_iteration_count",
]
