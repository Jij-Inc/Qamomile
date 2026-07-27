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
def prepare_dicke(
    n: qmc.UInt,
    initial_ones: qmc.Vector[qmc.UInt],
    schedule: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Prepare a Dicke state using the Bartschi-Eidenbenz SCS construction.

    The schedule must be precomputed with
    :func:`~qamomile.optimization.schedules.dicke.bartschi_eidenbenz_schedule`
    (single block) or
    :func:`~qamomile.optimization.schedules.dicke.dicke_state_composition_schedule`
    (multi-block).  Both functions return a single ordered dict that encodes
    pair gates as ``(t, c, c)`` (``key[1] == key[2]``) and triplet gates as
    ``(t, c1, c2)`` (``key[1] != key[2]``).

    Args:
        n (qmc.UInt): Number of qubits in the register.
        initial_ones (qmc.Vector[qmc.UInt]): Indices of the qubits that are
            initially in the ``|1>`` state.
        schedule (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]):
            Ordered gate schedule from a schedule utility.  Pair entries have
            ``key[1] == key[2]`` and triplet entries have ``key[1] != key[2]``.
            Because ``schedule`` is always passed via ``bindings``, the
            ``key[1] == key[2]`` branch is resolved at compile time by
            ``CompileTimeIfLoweringPass`` — no runtime check is emitted.

    Returns:
        qmc.Vector[qmc.Qubit]: Qubit register prepared in the Dicke state.

    Example:
        >>> from qamomile.optimization.schedules.dicke import dicke_state_composition_schedule
        >>> initial_ones, schedule = dicke_state_composition_schedule(
        ...     n_qubits=4, block_size=4, hamming_weight=1
        ... )
        >>> q = prepare_dicke(4, initial_ones, schedule)
    """
    q = qmc.qubit_array(n, name="q")

    for idx in qmc.range(initial_ones.shape[0]):
        qubit_index = initial_ones[idx]
        q[qubit_index] = qmc.x(q[qubit_index])

    # Pair entries have key[1] == key[2]; triplet entries have key[1] != key[2].
    # The schedule is compile-time bound so CompileTimeIfLoweringPass resolves
    # the branch for each iteration — the if disappears from the emitted circuit.
    for key, angle in qmc.items(schedule):
        if key[1] == key[2]:
            q = scs_gate_2q(q, key[0], key[1], angle)
        else:
            q = scs_gate_3q(q, key[0], key[1], key[2], angle)

    return q
