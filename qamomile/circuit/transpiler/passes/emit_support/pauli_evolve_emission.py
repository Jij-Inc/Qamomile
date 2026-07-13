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
from qamomile.observable.hamiltonian import HERMITIAN_IMAG_ATOL, PAULI_TERM_ZERO_ATOL

from .gate_emission import resolve_angle_value
from .qubit_address import QubitAddress, QubitMap


def _resolve_gamma(
    emit_pass: "StandardEmitPass",
    op: PauliEvolveOp,
    bindings: dict[str, Any],
) -> Any:
    """Resolve gamma with the canonical angle-resolution contract.

    Pauli evolution accepts the same concrete, declared-parameter, and
    emit-time symbolic expressions as rotation gates and global phase. In
    particular, a loop-carried gamma may already be a backend expression and
    must not be rejected merely because it is not a Python ``float``.

    Args:
        emit_pass (StandardEmitPass): Active emit pass providing angle
            resolution and backend parameter construction.
        op (PauliEvolveOp): Pauli evolution whose gamma is resolved.
        bindings (dict[str, Any]): Active emit-time bindings.

    Returns:
        Any: Concrete float or backend symbolic angle expression.

    Raises:
        EmitError: If gamma cannot be represented as an angle.
    """
    return resolve_angle_value(emit_pass, op.gamma, bindings)


def validate_hamiltonian_within_register(
    num_h_qubits: int,
    register_size: int,
) -> None:
    """Validate that a Hamiltonian fits within the target qubit register.

    A Hamiltonian acting on fewer qubits than the register is embedded
    into the register's qubit space (identity on the untouched qubits)
    by acting only on its declared qubits; only a Hamiltonian *larger*
    than the register is a genuine error. Every ``PauliEvolveOp`` emit
    path (shared, backend-native, and controlled) must apply this same
    rule through this helper so the size contract cannot drift between
    backends.

    Args:
        num_h_qubits (int): Number of qubits the Hamiltonian acts on
            (``Hamiltonian.num_qubits``).
        register_size (int): Resolved element count of the target qubit
            register.

    Raises:
        EmitError: If the Hamiltonian acts on more qubits than the
            register provides.
    """
    if num_h_qubits > register_size:
        raise EmitError(
            f"PauliEvolveOp qubit count mismatch: "
            f"qubit register has {register_size} qubits but "
            f"Hamiltonian acts on {num_h_qubits} qubits. "
            f"The Hamiltonian must not be larger than the register.",
            operation="PauliEvolveOp",
        )


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

    # Resolve Hamiltonian from bindings via the shared resolver.
    obs_value = op.observable
    hamiltonian = emit_pass._resolver.resolve_bound_value(obs_value, bindings)
    if not isinstance(hamiltonian, qm_o.Hamiltonian):
        raise EmitError(
            f"PauliEvolveOp requires a Hamiltonian binding. "
            f"Observable '{obs_value.name}' not found or not a Hamiltonian.",
            operation="PauliEvolveOp",
        )

    # Resolve gamma. When gamma is a parameter (scalar or array element),
    # obtain a backend Parameter so that the per-term RZ angles are
    # emitted as parametric expressions (`2 * coeff * backend_param`),
    # matching how ``ising_cost`` handles parametric gamma directly.
    gamma = _resolve_gamma(emit_pass, op, bindings)

    # Validate qubit count: logical array size vs Hamiltonian. A smaller
    # Hamiltonian is embedded by acting only on its declared qubits below.
    input_array = op.qubits
    assert isinstance(input_array, ArrayValue)
    num_h_qubits = hamiltonian.num_qubits
    if input_array.shape:
        n_resolved = emit_pass._resolver.resolve_int_value(
            input_array.shape[0], bindings
        )
        if n_resolved is not None:
            validate_hamiltonian_within_register(num_h_qubits, n_resolved)

    # Validate Hermitian (real coefficients)
    for operators, coeff in hamiltonian:
        if abs(coeff.imag) > HERMITIAN_IMAG_ATOL:
            raise EmitError(
                f"PauliEvolveOp requires a Hermitian Hamiltonian "
                f"(real coefficients), but found complex coefficient "
                f"{coeff} on term {operators}.",
                operation="PauliEvolveOp",
            )

    # Resolve qubit indices from the input array. For a sliced view
    # (``pauli_evolve(q[1::2], H, gamma)``) walk the ``slice_of`` chain
    # to the root ArrayValue so every view-local index ``i`` maps to
    # ``start + step * i`` in the root array's physical qubit space.
    root_av, slice_start, slice_step = emit_pass._resolver.resolve_slice_chain(
        input_array, bindings, operation="PauliEvolveOp"
    )
    qubit_indices: list[int] = []
    for i in range(num_h_qubits):
        addr = QubitAddress(root_av.uuid, slice_start + slice_step * i)
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
        if abs(coeff) < PAULI_TERM_ZERO_ATOL:
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

    # Map result array to same physical qubits. Resolve the result's
    # own slice chain so downstream ``resolve_qubit_index_detailed``
    # callers that walk to the root find the registered mapping, while
    # direct lookups via the result array's own uuid also still work.
    result_array = op.evolved_qubits
    assert isinstance(result_array, ArrayValue)
    result_root, result_start, result_step = emit_pass._resolver.resolve_slice_chain(
        result_array, bindings, operation="PauliEvolveOp"
    )
    for i, phys_idx in enumerate(qubit_indices):
        direct_addr = QubitAddress(result_array.uuid, i)
        if direct_addr not in qubit_map:
            qubit_map[direct_addr] = phys_idx
        root_addr = QubitAddress(result_root.uuid, result_start + result_step * i)
        if root_addr not in qubit_map:
            qubit_map[root_addr] = phys_idx
