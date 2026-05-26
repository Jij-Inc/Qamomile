"""General Dicke state preparation algorithm from the Bartschi-Eidenbenz paper.

This module includes the main Dicke state preparation algorithm, as well as the necessary components.
The method relies on Split & Cyclic Shift (SCS) blocks

All functions are decorated with ``@qmc.qkernel`` and use Handle-typed
parameters so they can be composed inside other ``@qkernel`` functions.

References:
- Bartschi, A., & Eidenbenz, S. (2019). Deterministic preparation of Dicke states. arXiv preprint arXiv:1904.07358

"""

import qamomile.circuit as qmc


@qmc.qkernel
def scs_gate_2q(
    q: qmc.Vector[qmc.Qubit],
    t: qmc.UInt,
    c: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the 2-qubit SCS block for the (n, n-1) case.

    Uses one controlled-RY decomposed into RY + CNOT.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        t (qmc.UInt): Target qubit index.
        c (qmc.UInt): Control qubit index.
        theta (qmc.Float): Rotation angle.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after applying the SCS block.
    """
    q[t], q[c] = qmc.cx(q[t], q[c])

    # Controlled-RY(theta) with c as control, t as target.
    q[t] = qmc.ry(q[t], 0.5 * theta)
    q[c], q[t] = qmc.cx(q[c], q[t])
    q[t] = qmc.ry(q[t], -0.5 * theta)
    q[c], q[t] = qmc.cx(q[c], q[t])

    q[t], q[c] = qmc.cx(q[t], q[c])

    return q


@qmc.qkernel
def scs_gate_3q(
    q: qmc.Vector[qmc.Qubit],
    t: qmc.UInt,
    c1: qmc.UInt,
    c2: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the 3-qubit SCS block for the non-extremal case.

    Uses CNOT gates and one doubly-controlled RY decomposed into RY + CNOT.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        t (qmc.UInt): Target qubit index.
        c1 (qmc.UInt): First control qubit index.
        c2 (qmc.UInt): Second control qubit index.
        theta (qmc.Float): Rotation angle.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register after applying the SCS block
    """
    q[t], q[c2] = qmc.cx(q[t], q[c2])

    # Doubly-controlled RY(theta): applies RY(theta) on t iff c1 == 1 AND c2 == 1.
    # 4-CNOT decomposition: net angle on t is (+1-1+1-1)*theta/4 = 0 unless both
    # CNOTs fire, where X*RY(-theta/4)*X = RY(+theta/4) flips the sign and all
    # four terms sum to theta.
    q[t] = qmc.ry(q[t], 0.25 * theta)
    q[c2], q[t] = qmc.cx(q[c2], q[t])
    q[t] = qmc.ry(q[t], -0.25 * theta)
    q[c1], q[t] = qmc.cx(q[c1], q[t])
    q[t] = qmc.ry(q[t], 0.25 * theta)
    q[c2], q[t] = qmc.cx(q[c2], q[t])
    q[t] = qmc.ry(q[t], -0.25 * theta)
    q[c1], q[t] = qmc.cx(q[c1], q[t])

    q[t], q[c2] = qmc.cx(q[t], q[c2])

    return q


@qmc.qkernel
def _apply_triplet_if_stage_matches(
    q: qmc.Vector[qmc.Qubit],
    indices: qmc.Vector[qmc.UInt],
    angle3: qmc.Float,
    stage_c: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a 3-qubit SCS gate only when the triplet belongs to the current stage.

    A triplet with key ``[t, c1, c2]`` belongs to the stage whose pair gate has
    control qubit ``c``. The match condition is ``c2 == c``. This helper exists
    as a separate qkernel so that the conditional is compiled in an isolated
    scope that contains no Dict handles — avoiding a phi-merge failure that
    would occur if the ``if`` were placed directly inside ``qmc.items(triplets)``.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register.
        indices (qmc.Vector[qmc.UInt]): Length-3 index vector ``[t, c1, c2]``.
        angle3 (qmc.Float): Rotation angle for the 3-qubit SCS gate.
        stage_c (qmc.UInt): Control qubit index of the current pair stage.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register (unchanged when ``indices[2] != stage_c``).
    """
    if indices[2] == stage_c:
        q = scs_gate_3q(q, indices[0], indices[1], indices[2], angle3)
    return q


@qmc.qkernel
def prepare_dicke(
    n: qmc.UInt,
    initial_ones: qmc.Vector[qmc.UInt],
    pairs: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    triplets: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Prepare a Dicke state using the Bartschi-Eidenbenz SCS construction.

    The schedule dicts must be precomputed with
    :func:`~qamomile.optimization.schedules.dicke.bartschi_eidenbenz_schedule`
    (single block) or
    :func:`~qamomile.optimization.schedules.dicke.dicke_state_composition_schedule`
    (multi-block).

    Args:
        n (qmc.UInt): Number of qubits in the register.
        initial_ones (qmc.Vector[qmc.UInt]): Indices of the qubits that are
            initially in the ``|1>`` state.
        pairs (qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]): Precomputed
            mapping from ``(t, c)`` qubit index pairs to rotation angles for the
            2-qubit SCS blocks.
        triplets (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]): Precomputed mapping
            from ``(t, c1, c2)`` qubit index triples to rotation angles for the
            3-qubit SCS blocks. Keys are length-3 index vectors.

    Returns:
        qmc.Vector[qmc.Qubit]: Qubit register prepared in the Dicke state.

    Example:
        >>> from qamomile.optimization.schedules.dicke import dicke_state_composition_schedule
        >>> initial_ones, pairs, triplets = dicke_state_composition_schedule(
        ...     n_qubits=4, block_size=4, hamming_weight=1
        ... )
        >>> q = prepare_dicke(4, initial_ones, pairs, triplets)
    """
    q = qmc.qubit_array(n, name="q")

    # Start from a computational basis state with Hamming weight k.
    for idx in qmc.range(initial_ones.shape[0]):
        qubit_index = initial_ones[idx]
        q[qubit_index] = qmc.x(q[qubit_index])

    # Apply SCS stages in the same order as bartschi_eidenbenz_schedule.
    # Each pair (t, c) defines one stage; triplets with matching c2 == c
    # belong to that stage and are applied immediately after the pair gate.
    # The conditional is isolated in _apply_triplet_if_stage_matches so that
    # the Dict handle for triplets never enters the phi-variable list of an if.
    for pair, angle in qmc.items(pairs):
        t = pair[0]
        c = pair[1]
        q = scs_gate_2q(q, t, c, angle)
        for indices, angle3 in qmc.items(triplets):
            q = _apply_triplet_if_stage_matches(q, indices, angle3, c)

    return q
