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
from .hhl import (
    hhl,
    reciprocal_rotation,
)
from .mottonen_amplitude_encoding import (
    MottonenAmplitudeEncoding,
    amplitude_encoding,
    compute_mottonen_amplitude_encoding_thetas,
    compute_mottonen_thetas,
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

__all__ = [
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
    # HHL
    "hhl",
    "reciprocal_rotation",
    # State preparation
    "MottonenAmplitudeEncoding",
    "amplitude_encoding",
    "compute_mottonen_amplitude_encoding_thetas",
    "compute_mottonen_thetas",
]
