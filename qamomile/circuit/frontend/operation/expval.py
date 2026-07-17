"""Expectation value operation for computing <psi|H|psi>.

This module provides the `expval()` function for computing the expectation
value of a Hamiltonian observable with respect to a quantum state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qamomile.circuit.frontend.handle import Float, Observable, Qubit, Vector
from qamomile.circuit.frontend.qkernel_utils import reject_aliased_quantum_args
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import ArrayValue, Value, resolve_root_qubit_address

if TYPE_CHECKING:
    pass


def expval(
    qubits: Qubit | Vector[Qubit] | tuple[Qubit, ...],
    hamiltonian: Observable,
) -> Float:
    """Compute the expectation value of an observable on a quantum state.

    This function computes ``<psi|H|psi>`` where ``psi`` is the quantum
    state represented by ``qubits`` and ``H`` is the Hamiltonian
    observable.

    The quantum state is **consumed** by this operation: ``expval``
    classifies as :attr:`ConsumeMode.DESTRUCTIVE`, the same category as
    ``measure`` / ``cast``.  Conceptually an Estimator runs many shots
    of the state to estimate the expectation, so the qubits cannot be
    reused afterwards.  Any attempt to access the same qubits / view
    slots after ``expval`` is rejected as use-after-destroy, both at
    trace time and post-fold in the IR.

    Args:
        qubits (Qubit | Vector[Qubit] | tuple[Qubit, ...]): The quantum
            register holding the prepared state. A single ``Qubit``
            handle is accepted for 1-qubit observables. When a
            ``Vector`` is passed all previously-borrowed elements must
            have been returned (the strict-return policy is enforced
            by ``consume`` here). When a slice view (``VectorView``)
            is passed its covered parent slots become consumed-slot
            markers so the parent cannot reuse them later.
        hamiltonian (Observable): The Observable parameter
            representing the Hamiltonian.  The actual
            ``qamomile.observable.Hamiltonian`` is provided via
            ``transpile(..., bindings={...})``.

    Returns:
        Float: A scalar handle holding the expectation value, suitable
            for use as the kernel return value or as an operand to
            further classical operations.

    Raises:
        RuntimeError: If no qkernel tracer is active.
        QubitConsumedError: If ``qubits`` was already consumed (e.g.
            measured / cast earlier in the kernel), or if any covered
            slot of a passed view was destroyed by a prior destructive
            view operation.
        UnreturnedBorrowError: If ``qubits`` is a ``Vector`` with
            outstanding element or slice-view borrows that have not
            been returned.

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

            # Expectation value -> Float (q is consumed here)
            return qm.expval(q, H)

        # Pass Hamiltonian via bindings
        executable = transpiler.transpile(vqe_step, bindings={"H": H})
        ```
    """
    tracer = get_current_tracer()

    # Convert qubits to Value without consuming ownership until the complete
    # ExpvalOp has been constructed successfully.
    if isinstance(qubits, tuple):
        reject_aliased_quantum_args(
            "expval",
            {f"qubits[{index}]": qubit for index, qubit in enumerate(qubits)},
            caller="expval",
        )
        for qubit in qubits:
            qubit.validate_consumable("expval")
        qubit_values = [q.value for q in qubits]
        # Snapshot each element's root ``(array_uuid, index)`` so emit can map
        # the observable's Pauli index to the physical qubit registered under
        # the root array's QInit key, even for a Vector element whose own UUID
        # was never registered in the quantum segment (e.g. an ungated ancilla,
        # or an element produced as a gate/composite result).  A standalone
        # qubit has no ``parent_array`` and resolves to ``None``; it is recorded
        # with the ``("", -1)`` sentinel so emit falls back to the flat UUID
        # lookup that already resolves standalone qubits.
        # We must resolve the root address HERE (at trace time) because the
        # pseudo-ArrayValue below flattens the elements into bare UUID tuples and
        # drops their ``parent_array`` -- emit could not chain-walk later. (The
        # non-tuple branch keeps the real element Value, so it resolves at emit
        # instead; see ``_build_qubit_map``.)
        parent_addrs = [resolve_root_qubit_address(v) for v in qubit_values]
        qubits_value = ArrayValue(
            type=qubit_values[0].type,
            name="expval_qubits",
            shape=tuple(),
        ).with_array_runtime_metadata(
            element_uuids=tuple(q.uuid for q in qubit_values),
            element_logical_ids=tuple(q.logical_id for q in qubit_values),
            # Encode each resolved root as (uuid, idx); ``None`` (standalone
            # qubit, or unresolved element) becomes the ``("", -1)`` sentinel
            # that ``get_element_parent_addresses()`` decodes back to ``None``.
            # Kept as two parallel tuples (not one tuple of pairs) so they ride
            # the existing ArrayRuntimeMetadata serialize / canonical paths.
            element_parent_uuids=tuple(
                addr[0] if addr is not None else "" for addr in parent_addrs
            ),
            element_parent_indices=tuple(
                addr[1] if addr is not None else -1 for addr in parent_addrs
            ),
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
        # ``Qubit`` handle — accepted by the public signature for
        # 1-qubit observables — cannot carry consumed-slot markers and
        # is skipped.
        if isinstance(qubits, Vector):
            qubits._check_no_consumed_slots("expval")
        qubits.validate_consumable("expval")
        qubits_value = qubits.value

    # Create result Float value
    result_value = Value(type=FloatType(), name="expval_result")

    # Create ExpvalOp
    op = ExpvalOp(
        operands=[qubits_value, hamiltonian.value],
        results=[result_value],
    )

    if isinstance(qubits, tuple):
        for qubit in qubits:
            qubit.consume(operation_name="expval")
    else:
        # Destructive consume marks the scalar, whole register, or view slots
        # unavailable only after every fallible construction step above.
        qubits.consume(operation_name="expval")

    tracer.add_operation(op)

    return Float(value=result_value)
