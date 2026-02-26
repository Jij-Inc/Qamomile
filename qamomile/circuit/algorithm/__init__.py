"""Algorithm building blocks for quantum circuits."""

from .basic import (
    cz_entangling_layer,
    rx_layer,
    ry_layer,
    rz_layer,
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
from .mottonen_amplitude_encode import (
    MottonenAmplitudeEncode,
    amplitude_encoding,
    compute_mottonen_thetas,
    parametric_amplitude_encoding,
)
from .qaoa import (
    ising_cost_circuit,
    qaoa_circuit,
    qaoa_state,
    superposition_vector,
    x_mixier_circuit,
)

__all__ = [
    # QAOA
    "ising_cost_circuit",
    "x_mixier_circuit",
    "qaoa_circuit",
    "superposition_vector",
    "qaoa_state",
    # Basic layers
    "rx_layer",
    "ry_layer",
    "rz_layer",
    "cz_entangling_layer",
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
    "MottonenAmplitudeEncode",
    "amplitude_encoding",
    "compute_mottonen_thetas",
    "parametric_amplitude_encoding",
]
