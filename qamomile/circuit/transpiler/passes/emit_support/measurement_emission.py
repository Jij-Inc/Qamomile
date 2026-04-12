"""Measurement emission helpers extracted from StandardEmitPass.

Provides ``emit_measure``, ``emit_measure_vector`` and
``emit_measure_qfixed`` as module-level functions.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass

from qamomile.circuit.ir.operation.gate import (
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.value import ArrayValue
from qamomile.circuit.transpiler.gate_emitter import MeasurementMode

from .qubit_address import ClbitMap, QubitAddress, QubitMap


def emit_measure(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: MeasureOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any] | None = None,
) -> None:
    """Emit a single measurement.

    Resolves the qubit operand using the full resolver (handles both
    scalar qubits and array element qubits with composite keys).

    Raises:
        warnings.warn: If the qubit or clbit cannot be resolved, a
            warning is emitted instead of silently dropping the
            measurement.
    """
    qubit_val = op.operands[0]
    clbit_uuid = op.results[0].uuid
    clbit_addr = QubitAddress(clbit_uuid)

    # Use the resolver for proper array element handling
    result = emit_pass._resolver.resolve_qubit_index_detailed(
        qubit_val, qubit_map, bindings or {}
    )
    qubit_idx = result.index if result.success else None

    # Fallback to direct UUID lookup
    if qubit_idx is None:
        qubit_addr = QubitAddress(qubit_val.uuid)
        if qubit_addr in qubit_map:
            qubit_idx = qubit_map[qubit_addr]

    if qubit_idx is not None and clbit_addr in clbit_map:
        clbit_idx = clbit_map[clbit_addr]
        emit_pass._emitter.emit_measure(circuit, qubit_idx, clbit_idx)
        if emit_pass._emitter.measurement_mode == MeasurementMode.STATIC:
            emit_pass._measurement_qubit_map[clbit_idx] = qubit_idx
    else:
        details: list[str] = []
        if qubit_idx is None:
            details.append(
                f"qubit '{qubit_val.name}' (uuid: {qubit_val.uuid[:8]}...) "
                f"could not be resolved to a physical qubit index"
            )
        if clbit_addr not in clbit_map:
            details.append(
                f"clbit (uuid: {clbit_uuid[:8]}...) not found in clbit_map"
            )
        warnings.warn(
            f"Measurement dropped: {'; '.join(details)}.",
            stacklevel=2,
        )


def emit_measure_vector(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: MeasureVectorOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit vector measurement."""
    qubits_array = op.operands[0]
    bits_array = op.results[0]

    if isinstance(qubits_array, ArrayValue) and isinstance(bits_array, ArrayValue):
        element_uuids = qubits_array.get_element_uuids()

        shape = qubits_array.shape
        if shape:
            size_val = shape[0]
            # Use allocator's _resolve_size to handle dynamic sizes (e.g., hi_dim0)
            size = emit_pass._allocator._resolve_size(size_val, bindings)
            if size is not None:
                for i in range(size):
                    if element_uuids and i < len(element_uuids):
                        qubit_addr = QubitAddress.from_composite_key(
                            element_uuids[i]
                        )
                    else:
                        qubit_addr = QubitAddress(qubits_array.uuid, i)

                    clbit_addr = QubitAddress(bits_array.uuid, i)
                    if qubit_addr in qubit_map and clbit_addr in clbit_map:
                        q_idx = qubit_map[qubit_addr]
                        c_idx = clbit_map[clbit_addr]
                        emit_pass._emitter.emit_measure(circuit, q_idx, c_idx)
                        if emit_pass._emitter.measurement_mode == MeasurementMode.STATIC:
                            emit_pass._measurement_qubit_map[c_idx] = q_idx


def emit_measure_qfixed(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: MeasureQFixedOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
) -> None:
    """Emit QFixed measurement."""
    qfixed = op.operands[0]
    qubit_uuids = qfixed.get_qfixed_qubit_uuids()
    result = op.results[0]

    for i, qubit_uuid in enumerate(qubit_uuids):
        qubit_addr = QubitAddress.from_composite_key(qubit_uuid)
        clbit_addr = QubitAddress(result.uuid, i)
        if qubit_addr in qubit_map and clbit_addr in clbit_map:
            q_idx = qubit_map[qubit_addr]
            c_idx = clbit_map[clbit_addr]
            emit_pass._emitter.emit_measure(circuit, q_idx, c_idx)
            if emit_pass._emitter.measurement_mode == MeasurementMode.STATIC:
                emit_pass._measurement_qubit_map[c_idx] = q_idx
