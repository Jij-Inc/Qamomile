"""Measurement emission helpers extracted from StandardEmitPass.

Provides ``emit_measure``, ``emit_measure_vector`` and
``emit_measure_qfixed`` as module-level functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass

from qamomile.circuit.ir.operation.gate import (
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.value import ArrayValue
from qamomile.circuit.transpiler.errors import EmitError
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
        EmitError: If the qubit or clbit cannot be resolved to a physical
            address. A dropped measurement is never recoverable inside a
            quantum segment — the emitted circuit would silently return
            results with a measured bit missing — so this is a hard error
            rather than a warning. The most common trigger is a
            ``measure(q[i])`` whose index ``i`` is an unresolved native-loop
            parameter; ``LoopAnalyzer.should_unroll`` is expected to have
            forced such loops to unroll before emit, so reaching this path
            indicates a real resolution failure.
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
            details.append(f"clbit (uuid: {clbit_uuid[:8]}...) not found in clbit_map")
        raise EmitError(
            f"Measurement could not be emitted: {'; '.join(details)}. "
            f"A measurement inside a native backend loop indexed by the loop "
            f"variable (e.g. `measure(q[i])`) must be unrolled; if you reached "
            f"this error the loop was not unrolled or the qubit was never "
            f"allocated.",
            operation="MeasureOperation",
        )


def emit_measure_vector(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: MeasureVectorOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit vector measurement.

    Resolves the qubit operand's slice_of chain so that measuring a
    view (``measure(q[1::2])``) correctly targets the root parent's
    physical qubits.

    Behaviour for unresolvable inputs differs by case:

    * Unknown vector length (``_resolve_size`` returns ``None``) — the
      function defers silently. This handles transpile-only paths
      (e.g. unbound parameter-size arrays) where downstream late-
      binding callers still handle the emit correctly.
    * Resolvable length but a sliced view that resolves to **zero**
      physical qubits — raises ``EmitError``. Previously this produced
      executions returning ``[(None, N)]``, a data-integrity hazard,
      so the silent path is intentionally narrowed to view-on-empty
      cases only.
    * Resolvable length but slice bounds are themselves symbolic —
      raises ``EmitError`` from ``resolve_slice_chain``.

    Raises:
        EmitError: When a sliced view resolves to zero physical qubits,
            or when slice bounds remain symbolic at emit time.
    """
    qubits_array = op.operands[0]
    bits_array = op.results[0]

    if not (
        isinstance(qubits_array, ArrayValue) and isinstance(bits_array, ArrayValue)
    ):
        return

    element_uuids = qubits_array.get_element_uuids()
    shape = qubits_array.shape
    if not shape:
        return

    size_val = shape[0]
    # Use allocator's _resolve_size to handle dynamic sizes (e.g., hi_dim0).
    # ``None`` means the length is not yet resolvable at this emit site
    # (e.g. transpile-only paths with unbound parameter-size arrays);
    # silently defer — the prior behaviour — rather than raising, so
    # downstream passes that handle late binding still succeed.
    size = emit_pass._allocator._resolve_size(size_val, bindings)
    if size is None:
        return

    # Walk slice chain once up front. For a root array this is
    # (root=qubits_array, start=0, step=1) and the loop body is identical
    # to the pre-view behaviour; for a view the (start, step) translate
    # view-local indices to root-space positions.
    root_av, start, step = emit_pass._resolver.resolve_slice_chain(
        qubits_array, bindings, operation="MeasureVectorOperation"
    )
    is_view = root_av is not qubits_array

    resolved_measurements: list[tuple[int, int]] = []
    for i in range(size):
        if not is_view and element_uuids and i < len(element_uuids):
            # Non-view fast path: preserve the composite-key lookup so
            # arrays whose elements were registered under explicit UUIDs
            # (e.g. qfixed, function returns) keep resolving correctly.
            qubit_addr = QubitAddress.from_composite_key(element_uuids[i])
        else:
            qubit_addr = QubitAddress(root_av.uuid, start + step * i)

        clbit_addr = QubitAddress(bits_array.uuid, i)
        if qubit_addr in qubit_map and clbit_addr in clbit_map:
            q_idx = qubit_map[qubit_addr]
            c_idx = clbit_map[clbit_addr]
            resolved_measurements.append((q_idx, c_idx))

    vector_emitter = getattr(emit_pass._emitter, "emit_measure_vector", None)
    if resolved_measurements and callable(vector_emitter):
        vector_emitter(
            circuit,
            tuple(qubit for qubit, _ in resolved_measurements),
            tuple(clbit for _, clbit in resolved_measurements),
        )
    else:
        for qubit, clbit in resolved_measurements:
            emit_pass._emitter.emit_measure(circuit, qubit, clbit)

    if emit_pass._emitter.measurement_mode == MeasurementMode.STATIC:
        for qubit, clbit in resolved_measurements:
            emit_pass._measurement_qubit_map[clbit] = qubit

    # Raise on the specific "view produced zero measurements" case that
    # previously silently returned ``[(None, shots)]`` — a data-integrity
    # hazard.  For non-view arrays we preserve the legacy silent-skip
    # behaviour: their elements may legitimately be registered through
    # other paths (pauli_evolve result, qfixed, sub-kernel returns) and
    # raising here would regress established tests without catching new
    # bugs.
    if is_view and not resolved_measurements and size > 0:
        raise EmitError(
            f"MeasureVectorOperation on view '{qubits_array.name}' emitted "
            f"no measurements: none of the {size} element(s) resolved to a "
            f"physical qubit. This typically indicates an unsupported "
            f"slice or a missing allocator registration for the root "
            f"parent '{root_av.name}'.",
            operation="MeasureVectorOperation",
        )


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
