"""Quantum Phase Estimation implementation.

Example:
    @qmc.qkernel
    def p_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
        return qmc.p(q, theta)

    @qmc.qkernel
    def circuit(theta: float) -> qmc.Float:
        counting = qmc.qubit_array(3, name="counting")
        target = qmc.qubit(name="target")
        target = qmc.x(target)

        phase = qmc.qpe(target, counting, p_gate, theta=theta)
        return qmc.measure(phase)
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import QFixed, Qubit, Vector
from qamomile.circuit.frontend.operation.control_flow import for_loop

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel_like import QKernelLike


def qpe(
    target: Qubit,
    counting: Vector[Qubit],
    unitary: "QKernelLike",
    **params: Any,
) -> QFixed:
    """Quantum Phase Estimation.

    Estimates the phase φ where U|ψ> = e^{2πiφ}|ψ>.

    Args:
        target (Qubit): Eigenstate ``|psi>`` of the unitary.
        counting (Vector[Qubit]): Register that stores the phase estimate.
        unitary (QKernelLike): Unitary qkernel to control.
        **params (Any): Classical parameters forwarded to the unitary.

    Returns:
        QFixed: Phase register as quantum fixed-point number
    """
    n = counting.shape[0]  # UInt handle (symbolic or concrete)
    controlled_u = cast(Callable[..., tuple[Any, ...]], qmc.control(unitary))

    # 1. Hadamard gates on counting qubits
    with for_loop(0, n, var_name="i") as i:
        counting[i] = qmc.h(counting[i])

    # 2. Controlled-U^(2^k) operations
    # Uses power parameter to emit Controlled(U^(2^k)), NOT Controlled(U)^(2^k)
    with for_loop(0, n, var_name="k") as k:
        counting[k], target = controlled_u(counting[k], target, power=2**k, **params)

    # 3. Inverse QFT (emit as InvokeOperation for native backend support)
    # Returns QFixed directly (bypassing cast) to ensure correct UUID mapping
    return _emit_iqft_and_cast_to_qfixed(counting)


def _emit_iqft_and_cast_to_qfixed(qubits: Vector[Qubit]) -> QFixed:
    """Apply the named IQFT qkernel and cast its register to QFixed.

    Args:
        qubits (Vector[Qubit]): Counting register.

    Returns:
        QFixed: Fixed-point alias of the transformed register.
    """
    from qamomile.circuit.frontend.operation.cast import cast as cast_handle
    from qamomile.circuit.stdlib.qft import iqft

    return cast_handle(iqft(qubits), QFixed, int_bits=0)
