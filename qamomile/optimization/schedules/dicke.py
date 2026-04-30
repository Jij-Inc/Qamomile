""" This helpers contains the helpers that precompute the indexes of the SCS blocks for the Dicke state preparation algorithm.

"""

import numpy as np

def _scs_schedule(n_dicke: int, k_dicke: int):
    """
    Build one SCS column for an n_dicke-qubit register with weight k_dicke.

    Args:
        n_dicke (int): Number of qubits in the Dicke state.
        k_dicke (int): Hamming weight of the Dicke state.
    
    Returns:
        pair_indices (np.ndarray): Indices for the 2-qubit SCS block.
        triplets_indices (np.ndarray): Indices for the 3-qubit SCS blocks.
        pair_angles (np.ndarray): Rotation angles for the 2-qubit SCS block.
        triplets_angles (np.ndarray): Rotation angles for the 3-qubit SCS blocks.
    """
    if not (0 <= k_dicke <= n_dicke):
        raise ValueError("Require 0 <= k <= n")

    n = n_dicke - 1

    pair_indices = []
    triplets_indices = []
    pair_angles = []
    triplets_angles = []

    pair_indices.append([n - 1, n])
    pair_angles.append(2 * np.arccos(1 / np.sqrt(n_dicke)))

    # Clamp upper bound to n so that n-l never goes negative.
    for l in range(2, min(k_dicke + 1, n + 1)):
        triplets_indices.append([n - l, n - l + 1, n])
        triplets_angles.append(2 * np.arccos(np.sqrt(l / n_dicke)))

    return (
        np.array(pair_indices, dtype=np.uint32),
        np.array(triplets_indices, dtype=np.uint32),
        np.array(pair_angles, dtype=float),
        np.array(triplets_angles, dtype=float),
    )

def bartschi_eidenbenz_schedule(n_dicke: int, k_dicke: int):
    """
    Build the Bartschi-Eidenbenz schedule for an n_dicke-qubit register with weight k_dicke.

    Args:
        n_dicke (int): Number of qubits in the Dicke state.
        k_dicke (int): Hamming weight of the Dicke state.

    Returns:
        pair_indices (np.ndarray): Indices for the 2-qubit SCS blocks.
        triplets_indices (np.ndarray): Indices for the 3-qubit SCS blocks.
        pair_angles (np.ndarray): Rotation angles for the 2-qubit SCS blocks.
        triplets_angles (np.ndarray): Rotation angles for the 3-qubit SCS blocks.
    """

    pair_indices = []
    triplets_indices = []
    pair_angles = []
    triplets_angles = []

    for n in range(n_dicke, 1, -1):
        # At column n, at most n-1 excitations can flow to the qubits below.
        k_sub = min(k_dicke, n - 1)
        p, tr, pa, ta = _scs_schedule(n, k_sub)
        pair_indices += p.tolist()
        triplets_indices += tr.tolist()
        pair_angles += pa.tolist()
        triplets_angles += ta.tolist()

    return (
        np.array(pair_indices, dtype=np.uint32),
        np.array(triplets_indices, dtype=np.uint32),
        np.array(pair_angles, dtype=float),
        np.array(triplets_angles, dtype=float),
    )


def dicke_state_composition_schedule(
    n_qubits: int,
    block_size: int,
    hamming_weight: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a global schedule for a product of identical Dicke states on blocks.

    The register is partitioned into contiguous blocks of size ``block_size``.
    The same Dicke weight ``hamming_weight`` is used for every block.

    Args:
        n_qubits: Total number of qubits in the full register.
        block_size: Number of qubits per block.
        hamming_weight: Dicke Hamming weight ``k`` for each block ``|D^block_size_k>``.

    Returns:
        initial_ones: Global qubit indices initialized in ``|1>``.
        pair_indices: Global 2-qubit SCS schedule.
        triplets_indices: Global 3-qubit SCS schedule.
        pair_angles: Angles for 2-qubit SCS schedule.
        triplets_angles: Angles for 3-qubit SCS schedule.

    Raises:
        ValueError: If inputs are invalid.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be > 0.")
    if block_size <= 0:
        raise ValueError("block_size must be > 0.")
    if n_qubits % block_size != 0:
        raise ValueError("n_qubits must be divisible by block_size.")
    if not (0 <= hamming_weight <= block_size):
        raise ValueError("Require 0 <= hamming_weight <= block_size.")

    num_blocks = n_qubits // block_size

    all_initial_ones: list[int] = []
    all_pair_indices: list[list[int]] = []
    all_triplets_indices: list[list[int]] = []
    all_pair_angles: list[float] = []
    all_triplets_angles: list[float] = []

    for block_idx in range(num_blocks):
        start = block_idx * block_size

        # Use the same basis-state convention as other Dicke utilities:
        # initialize the last k qubits of each block to |1>.
        if hamming_weight > 0:
            local_initial_ones = np.arange(block_size - hamming_weight, block_size, dtype=np.uint32)
            all_initial_ones.extend((local_initial_ones + start).tolist())

        local_pairs, local_triplets, local_pair_angles, local_triplets_angles = (
            bartschi_eidenbenz_schedule(block_size, hamming_weight)
        )

        if local_pairs.size > 0:
            all_pair_indices.extend((local_pairs + start).tolist())
            all_pair_angles.extend(local_pair_angles.tolist())
        if local_triplets.size > 0:
            all_triplets_indices.extend((local_triplets + start).tolist())
            all_triplets_angles.extend(local_triplets_angles.tolist())

    return (
        np.array(all_initial_ones, dtype=np.uint32),
        np.array(all_pair_indices, dtype=np.uint32).reshape(-1, 2),
        np.array(all_triplets_indices, dtype=np.uint32).reshape(-1, 3),
        np.array(all_pair_angles, dtype=float),
        np.array(all_triplets_angles, dtype=float),
    )