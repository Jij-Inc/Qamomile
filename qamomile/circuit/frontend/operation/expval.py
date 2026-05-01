"""Expectation value operation for computing <psi|H|psi>.

This module provides the `expval()` function for computing the expectation
value of a Hamiltonian observable with respect to a quantum state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qamomile.circuit.frontend.handle import Float, Observable, Qubit, Vector
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import ArrayValue, Value

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
        # Tuple of individual Qubits - collect their values
        # For now, we create a pseudo-ArrayValue to group them
        # The emitter will handle unpacking
        qubit_values = [q.value for q in qubits]
        qubits_value = ArrayValue(
            type=qubit_values[0].type,
            name="expval_qubits",
            shape=tuple(),
        ).with_array_runtime_metadata(
            element_uuids=tuple(q.uuid for q in qubit_values),
            element_logical_ids=tuple(q.logical_id for q in qubit_values),
        )
    else:
        # Guard for Vector[Qubit] operands: if any slot of the array was
        # physically destroyed by a prior destructive view operation
        # (e.g. ``measure(q[1::2])``), using the whole array in
        # ``expval`` would try to estimate over a partially-collapsed
        # quantum state.  Detect this at trace time so the error is
        # surfaced before reaching the backend.
        #
        # We only call this on ``Vector`` (which is an ``ArrayBase``
        # subclass and has ``_check_no_consumed_slots``).  A bare
        # ``Qubit`` handle — supported for back-compat even though the
        # public type signature requires ``Vector | tuple`` — cannot
        # carry consumed-slot markers and is skipped.
        if isinstance(qubits, Vector):
            qubits._check_no_consumed_slots("expval")
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
