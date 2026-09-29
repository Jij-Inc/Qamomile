from __future__ import annotations

import math
from typing import cast

import qamomile.circuit as qmc


@qmc.qkernel
def _prepare_basis_bit(target: qmc.Qubit, bit: qmc.UInt) -> qmc.Qubit:
    """Apply an RX-based bit flip before its exact phase compensation.

    Args:
        target (qmc.Qubit): Qubit to update.
        bit (qmc.UInt): Zero or one selecting identity or an RX(pi).

    Returns:
        qmc.Qubit: Updated qubit, including the RX global phase.
    """
    angle = cast(qmc.Float, math.pi * bit)
    return qmc.rx(target, angle)


@qmc.qkernel
def _apply_exact_basis_bit(target: qmc.Qubit, bit: qmc.UInt) -> qmc.Qubit:
    """Apply identity or exact X according to a classical bit.

    Args:
        target (qmc.Qubit): Qubit to update.
        bit (qmc.UInt): Zero or one selecting identity or X.

    Returns:
        qmc.Qubit: Updated qubit with the RX phase compensated.
    """
    exact_x_power = qmc.global_phase(
        _prepare_basis_bit,
        cast(qmc.Float, 0.5 * math.pi * bit),
    )
    return exact_x_power(target, bit)


@qmc.qkernel
def computational_basis_state(
    q: qmc.Vector[qmc.Qubit],
    bits: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Prepare the computational basis state labeled by ``bits``.

    Applies an exact ``X ** bits[i]`` to each qubit. The implementation uses
    ``Rx(pi * bits[i])`` plus its compensating ``exp(+i*pi*bits[i]/2)``
    global phase, so controlling this kernel preserves the intended unitary
    rather than turning the RX phase into an observable relative phase.

    Assumes ``q`` starts in :math:`\\lvert 0 \\rangle^{\\otimes n}` and
    ``q.shape[0] == bits.shape[0]``.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register, expected to start in |0>^n.
        bits (qmc.Vector[qmc.UInt]): Classical bit register specifying the target state.

    Returns:
        qmc.Vector[qmc.Qubit]: Qubit register prepared in the exact |bits>
        state convention.
    """
    for i in qmc.range(q.shape[0]):
        bit = bits[i]
        q[i] = _apply_exact_basis_bit(q[i], bit)
    return q


@qmc.qkernel
def apply_x_at_indices(
    q: qmc.Vector[qmc.Qubit],
    indices: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Apply ``X`` to every qubit of ``q`` listed in ``indices``.

    Takes an existing register rather than allocating one, so callers that
    allocate their own register (such as
    :func:`~qamomile.circuit.stdlib.state_preparation.dicke.prepare_dicke`)
    can share this step.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to update.
        indices (qmc.Vector[qmc.UInt]): Indices of the qubits to flip.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    for idx in qmc.range(indices.shape[0]):
        qubit_index = indices[idx]
        q[qubit_index] = qmc.x(q[qubit_index])
    return q


@qmc.qkernel
def basis_state_preparation(
    n: qmc.UInt,
    initial_ones: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Prepare a computational basis state from a list of ``|1>`` indices.

    Allocates an ``n``-qubit register in :math:`|0\\rangle^{\\otimes n}` and
    applies ``X`` to every qubit listed in ``initial_ones``. Unlike
    :func:`computational_basis_state`, the state is given by the positions
    of its ones rather than by a dense bit vector.

    Args:
        n (qmc.UInt): Number of qubits.
        initial_ones (qmc.Vector[qmc.UInt]): Indices initialized to ``|1>``.

    Returns:
        qmc.Vector[qmc.Qubit]: Qubit register in the requested basis state.
    """
    q = qmc.qubit_array(n, name="q")
    return apply_x_at_indices(q, initial_ones)
