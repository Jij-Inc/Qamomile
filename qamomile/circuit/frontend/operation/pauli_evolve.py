"""Pauli evolution operation for applying exp(-i * gamma * H).

This module provides the `pauli_evolve()` function for applying the
time evolution operator of a Pauli Hamiltonian to a quantum state.
"""

from __future__ import annotations

from typing import cast

from qamomile.circuit.frontend.handle import Float, Observable, Qubit, Vector
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp


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

    # Bump the SSA version of the input array. ``next_version`` preserves
    # ``logical_id`` and ``shape`` so the result is recognised as the same
    # logical register across the IR (resource allocation, inline-pass
    # remapping for nested @qkernel calls, visualization, etc.) -- creating
    # a fresh ArrayValue here would mint a new logical_id and drop the
    # caller-side qubit identity, breaking measurement after a nested
    # @qkernel that uses pauli_evolve internally (issue #354).
    result_array = qubits_value.next_version()

    op = PauliEvolveOp(
        operands=[qubits_value, hamiltonian.value, gamma.value],
        results=[result_array],
    )

    tracer = get_current_tracer()
    tracer.add_operation(op)

    result_vector = cast(
        Vector[Qubit],
        Vector._create_from_value(value=result_array, shape=consumed._shape),
    )
    return result_vector
