"""Expectation value operation for computing <psi|H|psi>.

This module provides the `expval()` function for computing the expectation
value of a Hamiltonian observable with respect to a quantum state.
"""

from __future__ import annotations

from qamomile.circuit.frontend.handle import Float, Qubit, Vector, Observable
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import Value, ArrayValue
from qamomile.circuit.transpiler.errors import QubitConsumedError


def expval(
    qubits: Vector[Qubit] | tuple[Qubit, ...],
    hamiltonian: Observable,
) -> Float:
    """Compute the expectation value of an observable on a quantum state.

    This function computes <psi|H|psi> where psi is the quantum state
    represented by the qubits and H is the Hamiltonian observable.

    The quantum state (qubits) is consumed by this operation. The qubits
    cannot be used for further quantum operations after expval.
    Only a single top-level expval is supported per kernel.

    Args:
        qubits: The quantum register holding the prepared state.
            Can be a Vector[Qubit] or a tuple of individual Qubits.
            For Vector[Qubit], all borrowed elements must be returned first.
            For tuple[Qubit, ...], each qubit is consumed individually;
            duplicate qubits will raise QubitConsumedError.
        hamiltonian: The Observable parameter representing the Hamiltonian.
            The actual qamomile.observable.Hamiltonian is provided via bindings.

    Returns:
        Float containing the expectation value.

    Raises:
        QubitConsumedError: If any qubit in the tuple has already been consumed,
            or if a duplicate qubit appears in the tuple.
        UnreturnedBorrowError: If a Vector[Qubit] has unreturned borrowed elements.

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

            # Expectation value -> Float (consumes q)
            return qm.expval(q, H)

        # Pass Hamiltonian via bindings
        executable = transpiler.transpile(vqe_step, bindings={"H": H})
        ```
    """
    # Convert qubits to Value, consuming the quantum resource
    if isinstance(qubits, tuple):
        if len(qubits) == 0:
            raise ValueError(
                "expval requires at least one qubit. Got an empty tuple."
            )
        # Tuple of individual Qubits - consume each qubit
        # Duplicate detection: track consumed qubit IDs
        seen_ids: set[str] = set()
        consumed_qubits: list[Qubit] = []
        for q in qubits:
            if q.id in seen_ids:
                raise QubitConsumedError(
                    f"Duplicate qubit in expval tuple: qubit '{q.value.name}' "
                    f"appears more than once.\n\n"
                    f"Each qubit in the expval tuple must be distinct.",
                    handle_name=q.value.name,
                    operation_name="expval",
                    first_use_location="expval (earlier in same tuple)",
                )
            seen_ids.add(q.id)
            consumed_qubits.append(q.consume(operation_name="expval"))

        qubit_values = [q.value for q in consumed_qubits]
        qubits_value = ArrayValue(
            type=qubit_values[0].type,
            name="expval_qubits",
            shape=tuple(),
            params={"qubit_values": qubit_values},
        )
    elif isinstance(qubits, Vector):
        # Vector[Qubit] - consume (includes validation of returned borrows)
        qubits = qubits.consume(operation_name="expval")
        qubits_value = qubits.value
    else:
        # Single qubit - consume it
        qubits = qubits.consume(operation_name="expval")
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
