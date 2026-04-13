"""Pauli evolution emission helper for StandardEmitPass.

Extracted from ``standard_emit.py`` to isolate the Hamiltonian
decomposition logic from the main emit dispatch.

The ``emit_pauli_evolve`` function is the **default** implementation.
Backend-specific emit passes (e.g., ``QiskitEmitPass``) may override
the corresponding ``_emit_pauli_evolve`` method; calling
``super()._emit_pauli_evolve(...)`` will ultimately delegate here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass

from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.value import ArrayValue
from qamomile.circuit.transpiler.errors import EmitError

from .qubit_address import QubitAddress, QubitMap


def _resolve_gamma(
    emit_pass: "StandardEmitPass",
    op: PauliEvolveOp,
    bindings: dict[str, Any],
) -> float | Any | None:
    """Resolve a PauliEvolveOp gamma operand to a concrete float or backend Parameter.

    Resolution order:
        1. **parameter array element fast path** — when ``gamma`` is
           ``arr[idx]`` and ``arr`` is in ``parameters``, return a
           backend Parameter. Takes priority over concrete bindings
           because users often pass concrete arrays as a shape hint.
        2. constant / concrete binding / scalar parameter → Python
           float or backend Parameter as appropriate.

    Returns ``None`` when none apply; the caller converts that into
    an ``EmitError``.
    """
    theta = op.gamma
    parameters = emit_pass._resolver.parameters

    # Fast path: ``arr[idx]`` where ``arr`` is a declared parameter.
    if theta.parent_array is not None and theta.parent_array.name in parameters:
        param_key = emit_pass._resolver.get_parameter_key(theta, bindings)
        if param_key:
            return emit_pass._get_or_create_parameter(param_key, theta.uuid)

    # Scalar declared parameter.
    if theta.name in parameters:
        param_key = emit_pass._resolver.get_parameter_key(theta, bindings)
        if param_key:
            return emit_pass._get_or_create_parameter(param_key, theta.uuid)

    # Constant / concrete binding.
    concrete = emit_pass._resolver.resolve_classical_value(theta, bindings)
    if concrete is not None and isinstance(concrete, (int, float)):
        return float(concrete)

    return None


def emit_pauli_evolve(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: PauliEvolveOp,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit Pauli evolution: exp(-i * gamma * H).

    Resolves the Hamiltonian from bindings and decomposes each term
    using the Pauli gadget technique:
    1. Basis change per qubit (X->H, Y->Sdg*H, Z->identity)
    2. CNOT ladder + RZ
    3. Undo basis change

    Subclasses can override this for backend-native implementations
    (e.g., Qiskit PauliEvolutionGate).
    """
    import qamomile.observable as qm_o

    # Resolve Hamiltonian from bindings
    obs_value = op.observable
    hamiltonian = None
    if hasattr(obs_value, "name") and obs_value.name in bindings:
        hamiltonian = bindings[obs_value.name]
    if hamiltonian is None and hasattr(obs_value, "uuid"):
        hamiltonian = bindings.get(obs_value.uuid)
    if not isinstance(hamiltonian, qm_o.Hamiltonian):
        raise EmitError(
            f"PauliEvolveOp requires a Hamiltonian binding. "
            f"Observable '{getattr(obs_value, 'name', '?')}' not found or "
            f"not a Hamiltonian.",
            operation="PauliEvolveOp",
        )

    # Resolve gamma. When gamma is a parameter (scalar or array element),
    # obtain a backend Parameter so that the per-term RZ angles are
    # emitted as parametric expressions (`2 * coeff * backend_param`),
    # matching how ``ising_cost`` handles parametric gamma directly.
    gamma = _resolve_gamma(emit_pass, op, bindings)
    if gamma is None:
        raise EmitError(
            "Cannot resolve gamma parameter for PauliEvolveOp. "
            "gamma must be a concrete float binding or a declared "
            "parameter (scalar or array element).",
            operation="PauliEvolveOp",
        )

    # Validate qubit count: logical array size vs Hamiltonian
    input_array = op.qubits
    num_h_qubits = hamiltonian.num_qubits
    if isinstance(input_array, ArrayValue) and input_array.shape:
        n_resolved = emit_pass._resolver.resolve_int_value(
            input_array.shape[0], bindings
        )
        if n_resolved is not None and n_resolved != num_h_qubits:
            raise EmitError(
                f"PauliEvolveOp qubit count mismatch: "
                f"qubit register has {n_resolved} qubits but "
                f"Hamiltonian acts on {num_h_qubits} qubits.",
                operation="PauliEvolveOp",
            )

    # Validate Hermitian (real coefficients)
    for operators, coeff in hamiltonian:
        if abs(coeff.imag) > 1e-10:
            raise EmitError(
                f"PauliEvolveOp requires a Hermitian Hamiltonian "
                f"(real coefficients), but found complex coefficient "
                f"{coeff} on term {operators}.",
                operation="PauliEvolveOp",
            )

    # Resolve qubit indices from the input array
    qubit_indices: list[int] = []
    for i in range(num_h_qubits):
        addr = QubitAddress(input_array.uuid, i)
        if addr in qubit_map:
            qubit_indices.append(qubit_map[addr])
        else:
            raise EmitError(
                f"Cannot resolve qubit index {i} for PauliEvolveOp. "
                f"Key '{str(addr)}' not found in qubit_map.",
                operation="PauliEvolveOp",
            )

    # Emit each Hamiltonian term using the Pauli gadget technique
    for operators, coeff in hamiltonian:
        if abs(coeff) < 1e-15:
            continue
        # RZ(theta) = exp(-i*theta*Z/2), so to get exp(-i*gamma*c*P)
        # we need theta = 2*gamma*c. Works for both concrete gamma
        # (float * float) and parametric gamma (float * Parameter),
        # relying on backend Parameter arithmetic.
        angle: Any
        if isinstance(gamma, (int, float)):
            angle = 2.0 * float(coeff.real * gamma)
        else:
            angle = (2.0 * float(coeff.real)) * gamma
        term_qubit_indices = [qubit_indices[op_item.index] for op_item in operators]
        pauli_types = [op_item.pauli for op_item in operators]

        if len(operators) == 0:
            continue

        # Step 1: Basis change
        for qi, pi in zip(term_qubit_indices, pauli_types):
            if pi == qm_o.Pauli.X:
                emit_pass._emitter.emit_h(circuit, qi)
            elif pi == qm_o.Pauli.Y:
                emit_pass._emitter.emit_sdg(circuit, qi)
                emit_pass._emitter.emit_h(circuit, qi)
            # Z and I: no basis change

        # Step 2: CNOT ladder + RZ (phase gadget)
        if len(term_qubit_indices) == 1:
            emit_pass._emitter.emit_rz(circuit, term_qubit_indices[0], angle)
        else:
            # Forward CNOT ladder
            for step in range(len(term_qubit_indices) - 1):
                emit_pass._emitter.emit_cx(
                    circuit,
                    term_qubit_indices[step],
                    term_qubit_indices[step + 1],
                )
            # RZ on last qubit
            emit_pass._emitter.emit_rz(circuit, term_qubit_indices[-1], angle)
            # Reverse CNOT ladder
            for step in range(len(term_qubit_indices) - 2, -1, -1):
                emit_pass._emitter.emit_cx(
                    circuit,
                    term_qubit_indices[step],
                    term_qubit_indices[step + 1],
                )

        # Step 3: Undo basis change (reverse order)
        for qi, pi in reversed(list(zip(term_qubit_indices, pauli_types))):
            if pi == qm_o.Pauli.X:
                emit_pass._emitter.emit_h(circuit, qi)
            elif pi == qm_o.Pauli.Y:
                emit_pass._emitter.emit_h(circuit, qi)
                emit_pass._emitter.emit_s(circuit, qi)
            # Z and I: no basis change

    # Map result array to same physical qubits
    result_array = op.evolved_qubits
    for i, phys_idx in enumerate(qubit_indices):
        result_addr = QubitAddress(result_array.uuid, i)
        if result_addr not in qubit_map:
            qubit_map[result_addr] = phys_idx
