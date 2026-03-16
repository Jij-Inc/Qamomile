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
from qamomile.circuit.errors import QubitConsumedError


def _consume_tuple_qubits(qubits: tuple[Qubit, ...]) -> ArrayValue:
    """Consume a tuple of distinct qubits and package it as a synthetic array."""
    if len(qubits) == 0:
        raise ValueError("expval requires at least one qubit. Got an empty tuple.")

    # Validate all members are Qubit before consuming or checking duplicates.
    for i, member in enumerate(qubits):
        if not isinstance(member, Qubit):
            raise TypeError(
                f"expval tuple expects only Qubit elements, "
                f"got {type(member).__name__} at index {i}"
            )

    seen_ids: set[str] = set()
    qubit_values: list[Value] = []

    for qubit in qubits:
        if qubit.id in seen_ids:
            raise QubitConsumedError(
                f"Duplicate qubit in expval tuple: qubit '{qubit.value.name}' "
                f"appears more than once.\n\n"
                f"Each qubit in the expval tuple must be distinct.",
                handle_name=qubit.value.name,
                operation_name="expval",
                first_use_location="expval (earlier in same tuple)",
            )
        seen_ids.add(qubit.id)
        qubit_values.append(qubit.consume(operation_name="expval").value)

    return ArrayValue(
        type=qubit_values[0].type,
        name="expval_qubits",
        shape=tuple(),
        params={"qubit_values": qubit_values},
    )


def _consume_expval_target(
    qubits: Qubit | Vector[Qubit] | tuple[Qubit, ...],
) -> Value | ArrayValue:
    """Consume an expval target and normalize it to the IR operand form."""
    if isinstance(qubits, tuple):
        return _consume_tuple_qubits(qubits)
    if isinstance(qubits, Vector):
        return qubits.consume(operation_name="expval").value
    if isinstance(qubits, Qubit):
        return qubits.consume(operation_name="expval").value

    raise TypeError(
        f"expval expects Qubit, Vector[Qubit], or tuple[Qubit, ...], "
        f"got {type(qubits).__name__}"
    )


def expval(
    qubits: Qubit | Vector[Qubit] | tuple[Qubit, ...],
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
            Can be a single Qubit, a Vector[Qubit], or a tuple of individual Qubits.
            For Vector[Qubit], all borrowed elements must be returned first.
            For tuple[Qubit, ...], each qubit is consumed individually;
            duplicate qubits will raise QubitConsumedError.
        hamiltonian: The Observable parameter representing the Hamiltonian.
            The actual qamomile.observable.Hamiltonian is provided via bindings.

    Returns:
        Float containing the expectation value.

    Raises:
        TypeError: If the target is not Qubit, Vector[Qubit], or tuple[Qubit, ...],
            or if any element in a tuple is not a Qubit.
        ValueError: If the tuple is empty.
        QubitConsumedError: If any qubit has already been consumed,
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
    qubits_value = _consume_expval_target(qubits)

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
