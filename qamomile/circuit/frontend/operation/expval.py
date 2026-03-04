"""Expectation value operation for computing <psi|H|psi>.

This module provides the `expval()` function for computing the expectation
value of a Hamiltonian observable with respect to a quantum state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qamomile.circuit.frontend.handle import Float, Qubit, Vector, Observable
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import Value, ArrayValue

if TYPE_CHECKING:
    pass


def expval(
    qubits: Vector[Qubit] | tuple[Qubit, ...],
    hamiltonian: Observable,
) -> Float:
    """Compute the expectation value of an observable on a quantum state.

    This function computes <psi|H|psi> where psi is the quantum state
    represented by the qubits and H is the Hamiltonian observable.

    The quantum state (qubits) is NOT consumed by this operation - the
    qubits can still be used for further operations after expval.

    Args:
        qubits: The quantum register holding the prepared state.
            Can be a Vector[Qubit] or a tuple of individual Qubits.
        hamiltonian: The Observable parameter representing the Hamiltonian.
            The actual qamomile.observable.Hamiltonian is provided via bindings.

    Returns:
        Float containing the expectation value.

    Example:
        ```python
        import qamomile.circuit as qm
        import qamomile.observable as qm_o

        # Build Hamiltonian in Python
        H = qm_o.Z(0) * qm_o.Z(1) + 0.5 * (qm_o.X(0) + qm_o.X(1))

        @qm.qkernel
        def vqe_step(q: qm.Vector[qm.Qubit], H: qm.Observable) -> qm.Float:
            # Ansatz
            q[0] = qm.ry(q[0], theta)
            q[0], q[1] = qm.cx(q[0], q[1])

            # Expectation value -> Float
            return qm.expval(q, H)

        # Pass Hamiltonian via bindings
        executable = transpiler.transpile(vqe_step, bindings={"H": H})
        ```
    """
    # Convert qubits to Value
    if isinstance(qubits, tuple):
        if len(qubits) == 0:
            raise ValueError("expval requires at least one qubit")
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
