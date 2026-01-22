"""Quantum Phase Estimation implementation.

Example:
    @qmc.qkernel
    def p_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
        return qmc.p(q, theta)

    @qmc.qkernel
    def circuit(theta: float) -> qmc.Float:
        counting = qmc.qubit_array(3, name="counting")
        target = qmc.qubit(name="target")
        target = qmc.x(target)

        phase = qmc.qpe(target, counting, p_gate, theta=theta)
        return qmc.measure(phase)
"""

from typing import TYPE_CHECKING, Any

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import Qubit, Vector, QFixed
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.types import QFixedType
from qamomile.circuit.ir.value import Value

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel


def qpe(
    target: Qubit,
    counting: Vector[Qubit],
    unitary: "QKernel",
    **params: Any,
) -> QFixed:
    """Quantum Phase Estimation.

    Estimates the phase φ where U|ψ> = e^{2πiφ}|ψ>.

    Args:
        target: The eigenstate |ψ> of the unitary U
        counting: Vector of qubits for phase estimation result
        unitary: The unitary operation U as a QKernel
        **params: Parameters to pass to the unitary

    Returns:
        QFixed: Phase register as quantum fixed-point number
    """
    n = _get_size(counting)
    controlled_u = qmc.controlled(unitary)

    # 1. Hadamard gates on counting qubits
    for i in range(n):
        counting[i] = qmc.h(counting[i])

    # 2. Controlled-U^(2^k) operations
    # Uses power parameter to emit Controlled(U^(2^k)), NOT Controlled(U)^(2^k)
    for k in range(n):
        counting[k], target = controlled_u(counting[k], target, power=2**k, **params)

    # 3. Inverse QFT (emit as CompositeGateOperation for native backend support)
    # Returns QFixed directly (bypassing cast) to ensure correct UUID mapping
    return _emit_iqft_and_cast_to_qfixed(counting)


def _emit_iqft_and_cast_to_qfixed(qubits: Vector[Qubit]) -> QFixed:
    """Emit IQFT and create QFixed directly with correct UUID mapping.

    This function combines IQFT emission and QFixed creation to ensure
    the qubit UUIDs are correctly tracked through the transformation.

    Instead of calling the iqft qkernel (which would be inlined into
    individual H, CP, SWAP gates), this emits a single CompositeGateOperation
    that the Qiskit emitter can render as native Qiskit QFT.inverse().
    """
    n = _get_size(qubits)

    # Collect input qubit values - borrow elements once
    qubit_handles = [qubits[i] for i in range(n)]
    operands = [h.value for h in qubit_handles]

    # Create result values (new versions) - these will be the qubit UUIDs for QFixed
    iqft_results = [v.next_version() for v in operands]

    # Return borrowed elements by assigning back to the array
    for i, h in enumerate(qubit_handles):
        qubits[i] = h

    # Create CompositeGateOperation for IQFT
    iqft_op = CompositeGateOperation(
        operands=operands,
        results=iqft_results,
        gate_type=CompositeGateType.IQFT,
        num_control_qubits=0,
        num_target_qubits=n,
        has_implementation=False,  # Use native backend
    )

    tracer = get_current_tracer()
    tracer.add_operation(iqft_op)

    # Now create QFixed directly using the IQFT result UUIDs
    int_bits = 0
    qubit_uuids = [r.uuid for r in iqft_results]
    qubit_logical_ids = [r.logical_id for r in iqft_results]

    result_type = QFixedType()
    result_value = Value(
        type=result_type,
        name=f"{qubits.value.name}_as_qfixed",
        params={
            "cast_source_uuid": qubits.value.uuid,
            "cast_source_logical_id": qubits.value.logical_id,
            "cast_qubit_uuids": qubit_uuids,
            "cast_qubit_logical_ids": qubit_logical_ids,
            "num_bits": n,
            "int_bits": int_bits,
            "qubit_values": qubit_uuids,  # This is what MeasureQFixedOperation uses
        },
    )

    # Create CastOperation (for IR completeness, though we've already set up UUIDs)
    cast_op = CastOperation(
        operands=[qubits.value],
        results=[result_value],
        source_type=qubits.value.type,
        target_type=result_type,
        qubit_mapping=qubit_uuids,
    )
    tracer.add_operation(cast_op)

    return QFixed(value=result_value)


def _get_size(arr: Vector[Qubit]) -> int:
    """Get array size as Python int."""
    size = arr.shape[0]
    if isinstance(size, int):
        return size
    if hasattr(size, "value") and size.value.is_constant():
        val = size.value.get_const()
        if val is not None:
            return int(val)
    if hasattr(size, "init_value"):
        return int(size.init_value)
    raise ValueError("Array must have fixed size")
