"""General Dicke state preparation algorithm from the Bartschi-Eidenbenz paper.

This module includes the main Dicke state preparation algorithm, as well as the necessary components.
The method relies on Split & Cyclic Shift (SCS) blocks

All functions are decorated with ``@qm_c.qkernel`` and use Handle-typed
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
    pair_indices: qmc.Matrix[qmc.UInt],
    triplets_indices: qmc.Matrix[qmc.UInt],
    pair_angles: qmc.Vector[qmc.Float],
    triplets_angles: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Prepare a Dicke state using the Bartschi-Eidenbenz SCS construction.

    Args:
        n (qmc.UInt): Number of qubits in the register.
        initial_ones (qmc.Vector[qmc.UInt]): Indices of the qubits that are initially in the |1> state.
        pair_indices (qmc.Matrix[qmc.UInt]): Precomputed indices for the 2-qubit SCS blocks.
        triplets_indices (qmc.Matrix[qmc.UInt]): Precomputed indices for the 3-qubit SCS blocks.
        pair_angles (qmc.Vector[qmc.Float]): Precomputed angles for the 2-qubit SCS blocks.
        triplets_angles (qmc.Vector[qmc.Float]): Precomputed angles for the 3-qubit SCS blocks.

    Returns:
        qmc.Vector[qmc.Qubit]: Qubit register prepared in the Dicke state.
    """
    q = qmc.qubit_array(n, name="q")

    # Start from a computational basis state with Hamming weight k.
    for idx in qmc.range(initial_ones.shape[0]):
        qubit_index = initial_ones[idx]
        q[qubit_index] = qmc.x(q[qubit_index])

    # Apply SCS stages in the same order as bartschi_eidenbenz_schedule.
    # Each stage consists of one 2-qubit SCS gate and (k-1) 3-qubit SCS gates.
    # Schedule invariant: triplets belonging to one stage share
    # triplets_indices[row, 2] == pair_indices[stage, 1] (same c2 wire).
    for stage_idx in qmc.range(pair_indices.shape[0]):
        q = scs_gate_2q(
            q,
            pair_indices[stage_idx, 0],
            pair_indices[stage_idx, 1],
            pair_angles[stage_idx],
        )

        stage_c2 = pair_indices[stage_idx, 1]
        for triplet_row in qmc.range(triplets_indices.shape[0]):
            if triplets_indices[triplet_row, 2] == stage_c2:
                q = scs_gate_3q(
                    q,
                    triplets_indices[triplet_row, 0],  # t (target)
                    triplets_indices[triplet_row, 1],  # c1 (first control)
                    triplets_indices[triplet_row, 2],  # c2 (second control)
                    triplets_angles[triplet_row],
                )

    return q
