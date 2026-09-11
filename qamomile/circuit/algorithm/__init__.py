"""Algorithm building blocks for quantum circuits."""

from .basic import (
    cx_entangling_layer,
    cz_entangling_layer,
    rx_layer,
    ry_layer,
    rz_layer,
    superposition_vector,
)
from .fqaoa import (
    cost_layer,
    fqaoa_layers,
    fqaoa_state,
    givens_rotation,
    givens_rotations,
    hopping_gate,
    initial_occupations,
    mixer_layer,
)
from .gas import (
    apply_function_preparation_qubo,
    apply_function_preparation_qubo_dagger,
    diffusion_op,
    first_degree_qft_encoding,
    function_preparation_qubo,
    function_preparation_qubo_dagger,
    grover_algorithm,
    grover_operator,
    qft_encoding,
    second_degree_qft_encoding,
    zero_degree_qft_encoding,
)
from .qaoa import (
    hubo_ising_cost,
    hubo_qaoa_layers,
    hubo_qaoa_state,
    ising_cost,
    qaoa_layers,
    qaoa_state,
    x_mixer,
)
from .shor import ekera_hastad_factoring, shor_order_finding
from .trotter import trotterized_time_evolution

__all__ = [
    # Factoring
    "shor_order_finding",
    "ekera_hastad_factoring",
    # QAOA
    "ising_cost",
    "x_mixer",
    "qaoa_layers",
    "qaoa_state",
    "hubo_ising_cost",
    "hubo_qaoa_layers",
    "hubo_qaoa_state",
    # Basic layers
    "rx_layer",
    "ry_layer",
    "rz_layer",
    "cz_entangling_layer",
    "cx_entangling_layer",
    "superposition_vector",
    # FQAOA
    "initial_occupations",
    "givens_rotation",
    "givens_rotations",
    "hopping_gate",
    "mixer_layer",
    "cost_layer",
    "fqaoa_layers",
    "fqaoa_state",
    # Trotterization
    "trotterized_time_evolution",
    # GAS primitives
    "qft_encoding",
    "zero_degree_qft_encoding",
    "first_degree_qft_encoding",
    "second_degree_qft_encoding",
    "apply_function_preparation_qubo",
    "apply_function_preparation_qubo_dagger",
    "diffusion_op",
    "function_preparation_qubo",
    "function_preparation_qubo_dagger",
    "grover_operator",
    "grover_algorithm",
]
