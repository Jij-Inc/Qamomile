"""Helpers to precompute the indices of the SCS blocks for the Dicke state preparation algorithm."""

import numpy as np


def _scs_schedule(n_dicke: int, k_dicke: int) -> dict[tuple[int, int, int], float]:
    """Build one SCS column for an n_dicke-qubit register with weight k_dicke.

    Gates are returned in application order: the 2-qubit SCS pair gate first,
    followed by the 3-qubit SCS triplet gates for this column.  Pair gates are
    encoded with a repeated control qubit ``(t, c, c)`` so that all entries
    share the same 3-tuple key type.  Triplet gates are encoded as
    ``(t, c1, c2)`` with ``c1 != c2``; because ``c1 = n-k+1 < n = c2`` for
    k >= 2, the two encodings never collide.

    Args:
        n_dicke (int): Number of qubits in the Dicke state.
        k_dicke (int): Hamming weight of the Dicke state.

    Returns:
        dict[tuple[int, int, int], float]: Ordered gate schedule mapping
        3-tuple qubit indices to rotation angles.  Pair entries satisfy
        ``key[1] == key[2]``; triplet entries satisfy ``key[1] != key[2]``.

    Raises:
        ValueError: If ``k_dicke`` is outside ``[0, n_dicke]``.
    """
    if not (0 <= k_dicke <= n_dicke):
        raise ValueError("Require 0 <= k_dicke <= n_dicke.")

    n = n_dicke - 1
    schedule: dict[tuple[int, int, int], float] = {}

    # Pair gate encoded as (t, c, c) — c1 == c2 signals a 2-qubit SCS gate.
    schedule[(n - 1, n, n)] = 2 * np.arccos(1 / np.sqrt(n_dicke))

    # Triplet gates follow immediately; c1 < c2 always holds here (k >= 2).
    for k in range(2, min(k_dicke + 1, n + 1)):
        schedule[(n - k, n - k + 1, n)] = 2 * np.arccos(np.sqrt(k / n_dicke))

    return schedule


def bartschi_eidenbenz_schedule(
    n_dicke: int, k_dicke: int
) -> dict[tuple[int, int, int], float]:
    """Build the Bartschi-Eidenbenz schedule for an n_dicke-qubit register with weight k_dicke.

    For the degenerate weights ``k_dicke == 0`` and ``k_dicke == n_dicke``,
    the Dicke state is already a computational basis state (all-zero or
    all-one) and no SCS rotation is needed; an empty dict is returned so
    that no gates are emitted.

    Gates are returned in application order: columns are stacked in descending
    order (largest column first), and within each column the pair gate precedes
    its triplet gates.  The encoding uses 3-tuple keys throughout: pair gates
    satisfy ``key[1] == key[2]``; triplet gates satisfy ``key[1] != key[2]``.

    Args:
        n_dicke (int): Number of qubits in the Dicke state.
        k_dicke (int): Hamming weight of the Dicke state.

    Returns:
        dict[tuple[int, int, int], float]: Ordered gate schedule ready for
        direct use with
        :func:`~qamomile.circuit.algorithm.state_preparation.dicke.prepare_dicke`.

    Raises:
        ValueError: If ``k_dicke`` is outside ``[0, n_dicke]``.
    """
    if not (0 <= k_dicke <= n_dicke):
        raise ValueError("Require 0 <= k_dicke <= n_dicke.")

    if k_dicke == 0 or k_dicke == n_dicke:
        return {}

    schedule: dict[tuple[int, int, int], float] = {}

    for n in range(n_dicke, 1, -1):
        k_sub = min(k_dicke, n - 1)
        schedule.update(_scs_schedule(n, k_sub))

    return schedule


def dicke_state_composition_schedule(
    n_qubits: int,
    block_size: int,
    hamming_weight: int = 1,
) -> tuple[np.ndarray, dict[tuple[int, int, int], float]]:
    """Build a global schedule for a product of identical Dicke states on blocks.

    The register is partitioned into contiguous blocks of size ``block_size``.
    The same Dicke weight ``hamming_weight`` is used for every block.

    Args:
        n_qubits (int): Total number of qubits in the full register.
        block_size (int): Number of qubits per block.
        hamming_weight (int): Dicke Hamming weight ``k`` for each block
            ``|D^block_size_k>``. Defaults to ``1``.

    Returns:
        tuple[np.ndarray, dict[tuple[int, int, int], float]]:
        ``(initial_ones, schedule)`` — global qubit indices and SCS gate
        schedule for the full register, ready for use with
        :func:`~qamomile.circuit.algorithm.state_preparation.dicke.prepare_dicke`.
        ``schedule`` maps 3-tuple qubit indices to rotation angles; pair
        entries satisfy ``key[1] == key[2]`` and triplet entries satisfy
        ``key[1] != key[2]``.

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
    schedule: dict[tuple[int, int, int], float] = {}

    for block_idx in range(num_blocks):
        start = block_idx * block_size

        if hamming_weight > 0:
            local_initial_ones = np.arange(
                block_size - hamming_weight, block_size, dtype=np.uint32
            )
            all_initial_ones.extend((local_initial_ones + start).tolist())

        local_schedule = bartschi_eidenbenz_schedule(block_size, hamming_weight)
        for (t, c1, c2), angle in local_schedule.items():
            schedule[(t + start, c1 + start, c2 + start)] = angle

    return (
        np.array(all_initial_ones, dtype=np.uint32),
        schedule,
    )
