"""Algorithm building blocks for quantum circuits."""

from .aoa import (
    aoa_layers,
    aoa_state_basis_state,
    aoa_state_dicke,
    aoa_state_superposition,
    basis_state_preparation,
    hubo_aoa_layers,
    hubo_aoa_state_basis_state,
    hubo_aoa_state_dicke,
    hubo_aoa_state_superposition,
    xy_mixer,
    xy_pair_rotation,
)
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
from .qaoa import (
    hubo_ising_cost,
    hubo_qaoa_layers,
    hubo_qaoa_state,
    ising_cost,
    qaoa_layers,
    qaoa_state,
    x_mixer,
)
from .state_preparation import (
    MottonenAmplitudeEncoding,
    amplitude_encoding,
    amplitude_encoding_from_angles,
)
from .trotter import trotterized_time_evolution

__all__ = [
    # QAOA
    "ising_cost",
    "x_mixer",
    "qaoa_layers",
    "qaoa_state",
    "hubo_ising_cost",
    "hubo_qaoa_layers",
    "hubo_qaoa_state",
    # AOA
    "basis_state_preparation",
    "aoa_layers",
    "aoa_state_superposition",
    "aoa_state_dicke",
    "aoa_state_basis_state",
    "hubo_aoa_layers",
    "hubo_aoa_state_superposition",
    "hubo_aoa_state_dicke",
    "hubo_aoa_state_basis_state",
    "xy_mixer",
    "xy_pair_rotation",
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
    # State preparation
    "MottonenAmplitudeEncoding",
    "amplitude_encoding",
    "amplitude_encoding_from_angles",
    # Trotterization
    "trotterized_time_evolution",
]
