"""Expectation value operation for computing <psi|H|psi>.

This module provides the `expval()` function for computing the expectation
value of a Hamiltonian observable with respect to a quantum state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qamomile.circuit.frontend.handle import Float, Qubit, Vector, HamiltonianExpr
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import Value, ArrayValue

if TYPE_CHECKING:
    pass


def expval(
    qubits: Vector[Qubit] | tuple[Qubit, ...],
    hamiltonian: HamiltonianExpr,
) -> Float:
    """Compute the expectation value of an observable on a quantum state.

    This function computes <psi|H|psi> where psi is the quantum state
    represented by the qubits and H is the Hamiltonian observable.

    The quantum state (qubits) is NOT consumed by this operation - the
    qubits can still be used for further operations after expval.

    Args:
        qubits: The quantum register holding the prepared state.
            Can be a Vector[Qubit] or a tuple of individual Qubits.
        hamiltonian: The HamiltonianExpr observable to measure.

    Returns:
        Float containing the expectation value.

    Example:
        ```python
        @qm.qkernel
        def vqe_step(q: qm.Vector[qm.Qubit], theta: qm.Float) -> qm.Float:
            # Ansatz
            q[0] = qm.ry(q[0], theta)
            q[0], q[1] = qm.cx(q[0], q[1])

            # Observable definition
            H = qm.pauli.Z(0) * qm.pauli.Z(1) + 0.5 * (qm.pauli.X(0) + qm.pauli.X(1))

            # Expectation value -> Float
            exp_val = qm.expval(q, H)
            return exp_val
        ```
    """
    # Convert qubits to Value
    if isinstance(qubits, tuple):
        # Tuple of individual Qubits - collect their values
        # For now, we create a pseudo-ArrayValue to group them
        # The emitter will handle unpacking
        qubit_values = [q.value for q in qubits]
        qubits_value = ArrayValue(
            type=qubit_values[0].type,
            name="expval_qubits",
            shape=tuple(),
            params={"qubit_values": qubit_values},
        )
    elif isinstance(qubits, Vector):
        qubits_value = qubits.value
    else:
        # Single qubit - wrap in array-like structure
        qubits_value = qubits.value

    # Create result Float value
    result_value = Value(type=FloatType(), name="expval_result")

    # Create ExpvalOp
    op = ExpvalOp(
        operands=[qubits_value, hamiltonian.value],
        results=[result_value],
    )

    # Emit to tracer
    tracer = get_current_tracer()
    tracer.add_operation(op)

    return Float(value=result_value)
