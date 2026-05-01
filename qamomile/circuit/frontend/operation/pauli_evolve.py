"""Pauli evolution operation for applying exp(-i * gamma * H).

This module provides the `pauli_evolve()` function for applying the
time evolution operator of a Pauli Hamiltonian to a quantum state.
"""

from __future__ import annotations

from typing import cast

from qamomile.circuit.frontend.handle import Float, Observable, Qubit, Vector
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import ArrayValue


def pauli_evolve(
    q: Vector[Qubit],
    hamiltonian: Observable,
    gamma: Float,
) -> Vector[Qubit]:
    """Apply exp(-i * gamma * H) to a qubit register.

    Implements Hamiltonian time evolution using the Pauli gadget technique.
    The actual Hamiltonian is provided via bindings at transpile time.

    Each backend can use native implementations:
    - Qiskit: PauliEvolutionGate
    - QuriParts: PauliRotation gates
    - Others: fallback decomposition (basis change + CNOT ladder + RZ)

    Args:
        q: The quantum register to evolve.
        hamiltonian: Observable parameter referencing the Hamiltonian.
            The actual qamomile.observable.Hamiltonian is provided via bindings.
        gamma: Evolution time / variational parameter.

    Returns:
        Vector[Qubit]: The evolved qubit register.

    Example:
        ```python
        import qamomile.circuit as qmc
        import qamomile.observable as qm_o

        H = 0.5 * qm_o.X(0) * qm_o.Z(1) + qm_o.Z(0)

        @qmc.qkernel
        def cost_layer(
            q: qmc.Vector[qmc.Qubit],
            H: qmc.Observable,
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            q = qmc.pauli_evolve(q, H, gamma)
            return q

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(cost_layer, bindings={"H": H, "gamma": 0.5})
        ```
    """
    # Consume the qubit array (affine type enforcement)
    consumed = q.consume("pauli_evolve")
    qubits_value = consumed.value

    # Create new result qubit array value.  When the input is a
    # sliced view (``pauli_evolve(q[1::2], H, gamma)``), forward its
    # ``slice_of`` chain onto the result so that downstream
    # operations — in particular ``expval(result, Z(0))`` — walk the
    # same chain back to the root parent's physical qubits instead
    # of landing on qubit 0 by default.  ``pauli_evolve`` is
    # in-place at the physical-qubit level; preserving the chain is
    # the semantically correct mapping.  For a non-view input
    # ``slice_of`` is ``None`` and the result remains a plain array
    # (no behavioural change for the existing non-view path).
    result_array = ArrayValue(
        type=QubitType(),
        name=f"{qubits_value.name}_evolved",
        shape=qubits_value.shape,
        slice_of=qubits_value.slice_of,
        slice_start=qubits_value.slice_start,
        slice_step=qubits_value.slice_step,
    )

    # Create PauliEvolveOp
    op = PauliEvolveOp(
        operands=[qubits_value, hamiltonian.value, gamma.value],
        results=[result_array],
    )

    # Emit to tracer
    tracer = get_current_tracer()
    tracer.add_operation(op)

    # Return new Vector[Qubit] wrapping the result
    result_vector = cast(
        Vector[Qubit],
        Vector._create_from_value(value=result_array, shape=consumed._shape),
    )
    return result_vector
