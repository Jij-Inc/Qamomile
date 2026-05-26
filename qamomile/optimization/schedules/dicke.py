"""Helpers to precompute the indices of the SCS blocks for the Dicke state preparation algorithm."""

import numpy as np


def _scs_schedule(
    n_dicke: int, k_dicke: int
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int, int], float]]:
    """Build one SCS column for an n_dicke-qubit register with weight k_dicke.

    Args:
        n_dicke (int): Number of qubits in the Dicke state.
        k_dicke (int): Hamming weight of the Dicke state.

    Returns:
        tuple[dict[tuple[int, int], float], dict[tuple[int, int, int], float]]:
        ``(pairs, triplets)`` where ``pairs`` maps ``(t, c)`` qubit index pairs
        to rotation angles and ``triplets`` maps ``(t, c1, c2)`` qubit index
        triples to rotation angles.

    Raises:
        ValueError: If ``k_dicke`` is outside ``[0, n_dicke]``.
    """
    if not (0 <= k_dicke <= n_dicke):
        raise ValueError("Require 0 <= k_dicke <= n_dicke.")

    n = n_dicke - 1

    pairs: dict[tuple[int, int], float] = {}
    triplets: dict[tuple[int, int, int], float] = {}

    pairs[(n - 1, n)] = 2 * np.arccos(1 / np.sqrt(n_dicke))

    # Clamp upper bound to n so that n-k never goes negative.
    for k in range(2, min(k_dicke + 1, n + 1)):
        triplets[(n - k, n - k + 1, n)] = 2 * np.arccos(np.sqrt(k / n_dicke))

    return pairs, triplets


def bartschi_eidenbenz_schedule(
    n_dicke: int, k_dicke: int
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int, int], float]]:
    """Build the Bartschi-Eidenbenz schedule for an n_dicke-qubit register with weight k_dicke.

    For the degenerate weights ``k_dicke == 0`` and ``k_dicke == n_dicke``,
    the Dicke state is already a computational basis state (all-zero or
    all-one) and no SCS rotation is needed; empty dicts are returned so
    that no gates are emitted.

    Args:
        n_dicke (int): Number of qubits in the Dicke state.
        k_dicke (int): Hamming weight of the Dicke state.

    Returns:
        tuple[dict[tuple[int, int], float], dict[tuple[int, int, int], float]]:
        ``(pairs, triplets)`` — stacked SCS column outputs in descending column
        order, suitable for direct use with
        :func:`~qamomile.circuit.algorithm.state_preparation.dicke.prepare_dicke`.
        ``pairs`` maps ``(t, c)`` qubit pairs to rotation angles and
        ``triplets`` maps ``(t, c1, c2)`` qubit triples to rotation angles.

    Raises:
        ValueError: If ``k_dicke`` is outside ``[0, n_dicke]``.
    """
    if not (0 <= k_dicke <= n_dicke):
        raise ValueError("Require 0 <= k_dicke <= n_dicke.")

    if k_dicke == 0 or k_dicke == n_dicke:
        return {}, {}

    pairs: dict[tuple[int, int], float] = {}
    triplets: dict[tuple[int, int, int], float] = {}

    for n in range(n_dicke, 1, -1):
        # At column n, at most n-1 excitations can flow to the qubits below.
        k_sub = min(k_dicke, n - 1)
        col_pairs, col_triplets = _scs_schedule(n, k_sub)
        pairs.update(col_pairs)
        triplets.update(col_triplets)

    return pairs, triplets


def dicke_state_composition_schedule(
    n_qubits: int,
    block_size: int,
    hamming_weight: int = 1,
) -> tuple[np.ndarray, dict[tuple[int, int], float], dict[tuple[int, int, int], float]]:
    """Build a global schedule for a product of identical Dicke states on blocks.

    The register is partitioned into contiguous blocks of size ``block_size``.
    The same Dicke weight ``hamming_weight`` is used for every block.

    Args:
        n_qubits (int): Total number of qubits in the full register.
        block_size (int): Number of qubits per block.
        hamming_weight (int): Dicke Hamming weight ``k`` for each block
            ``|D^block_size_k>``. Defaults to ``1``.

    Returns:
        tuple[np.ndarray, dict[tuple[int, int], float], dict[tuple[int, int, int], float]]:
        ``(initial_ones, pairs, triplets)`` — global qubit indices and SCS
        rotation parameters for the full register, ready for use with
        :func:`~qamomile.circuit.algorithm.state_preparation.dicke.prepare_dicke`.
        ``pairs`` maps ``(t, c)`` qubit pairs to rotation angles and
        ``triplets`` maps ``(t, c1, c2)`` qubit triples to rotation angles.

    Raises:
        ValueError: If ``n_qubits <= 0``; if ``block_size <= 0``; if
            ``n_qubits`` is not divisible by ``block_size``; or if
            ``hamming_weight`` is outside ``[0, block_size]``.
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
    pairs: dict[tuple[int, int], float] = {}
    triplets: dict[tuple[int, int, int], float] = {}

    for block_idx in range(num_blocks):
        start = block_idx * block_size

        # Use the same basis-state convention as other Dicke utilities:
        # initialize the last k qubits of each block to |1>.
        if hamming_weight > 0:
            local_initial_ones = np.arange(
                block_size - hamming_weight, block_size, dtype=np.uint32
            )
            all_initial_ones.extend((local_initial_ones + start).tolist())

        local_pairs, local_triplets = bartschi_eidenbenz_schedule(
            block_size, hamming_weight
        )

        for (t, c), angle in local_pairs.items():
            pairs[(t + start, c + start)] = angle
        for (t, c1, c2), angle in local_triplets.items():
            triplets[(t + start, c1 + start, c2 + start)] = angle

    return (
        np.array(all_initial_ones, dtype=np.uint32),
        pairs,
        triplets,
    )
