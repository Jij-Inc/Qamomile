"""Boxed callable emission helpers extracted from StandardEmitPass.

This module provides module-level functions for emitting boxed callables
(QFT, IQFT, QPE, custom composites, and opaque oracles) and their approximate
variants. Each function takes an
``emit_pass`` parameter (a ``StandardEmitPass`` instance) in place of
``self``.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation.callable import (
    CallTransform,
    CompositeGateType,
    InvokeOperation,
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.control_value_emission import (
    bracket_control_value,
)
from qamomile.circuit.transpiler.passes.emit_support.physical_index_map import (
    map_array_result_group,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    QubitAddress,
    QubitMap,
)

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass


def _ensure_composite_gate_type(
    op: Any,
    expected: CompositeGateType,
    helper_name: str,
) -> None:
    """Validate that a specialized composite helper received its gate type.

    Args:
        op (Any): Operation passed to the specialized helper.
        expected (CompositeGateType): Gate type that the helper knows how to
            process.
        helper_name (str): Name of the helper reporting the validation error.

    Raises:
        EmitError: If ``op`` is not the expected composite gate type.
    """
    if isinstance(op, InvokeOperation) and op.gate_type == expected:
        return

    if not isinstance(op, InvokeOperation):
        got_name = type(op).__name__
        raise EmitError(
            f"{helper_name} only supports {expected.name} composite gates, "
            f"got {got_name}.",
            operation=got_name,
        )

    raise EmitError(
        f"{helper_name} only supports {expected.name} composite gates, "
        f"got {op.gate_type.name}.",
        operation=f"InvokeOperation[{op.gate_type.name}]",
    )


def emit_composite_gate(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: InvokeOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a boxed composite/oracle invocation.

    Resolves each qubit operand through ``resolve_qubit_index_detailed``
    so that view-local operands (``qft(q[1::2])``) walk the ``slice_of``
    chain and map to the root parent's physical qubits. Raises
    ``EmitError`` if any operand cannot be resolved, rather than
    silently dropping it (previously ``qft(view)`` emitted zero gates).

    Raises:
        EmitError: If any control or target qubit operand fails to
            resolve to a physical qubit index.
    """
    from qamomile.circuit.transpiler.passes.emit_support.controlled_block_support import (
        _expand_quantum_operands_to_phys,
    )

    all_qubits = op.control_qubits + op.target_qubits
    qubit_groups = [
        _expand_quantum_operands_to_phys(
            emit_pass,
            q,
            qubit_map,
            bindings,
            operation=f"InvokeOperation[{op.gate_type.name}]",
        )
        for q in all_qubits
    ]
    qubit_indices: list[int] = []
    for group in qubit_groups:
        qubit_indices.extend(group)

    if emit_callable_implementation_emitter(
        emit_pass,
        circuit,
        op,
        qubit_indices,
        bindings,
    ):
        update_composite_result_mapping(op, qubit_groups, qubit_map)
        return

    if op.transform is CallTransform.CONTROLLED:
        from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
            emit_controlled_composite_at_indices,
        )

        emit_controlled_composite_at_indices(
            emit_pass,
            circuit,
            op,
            [],
            qubit_indices,
            bindings,
        )
        update_composite_result_mapping(op, qubit_groups, qubit_map)
        return

    if op.transform is CallTransform.INVERSE:
        implementation = op.implementation_for(
            backend=getattr(emit_pass, "backend_name", None)
        )
        if implementation is None or implementation.body is None:
            raise EmitError(
                f"Inverse callable '{op.target.name}' has no inverse "
                "implementation body for this backend. Bind structural "
                "parameters at compile time so the inverse can be "
                "materialized, or register an inverse implementation.",
                operation=f"InvokeOperation[{op.target.name}]",
            )
        emit_pass._emit_custom_composite(
            circuit,
            op,
            implementation.body,
            qubit_indices,
            bindings,
        )
        update_composite_result_mapping(op, qubit_groups, qubit_map)
        return

    # Try backend-global native emitters after callable-specific implementations.
    for emitter in emit_pass._composite_emitters:
        if emitter.can_emit(op.gate_type):
            if emitter.emit(circuit, op, qubit_indices, bindings):
                update_composite_result_mapping(op, qubit_groups, qubit_map)
                return

    # Fall back to decomposition
    emit_composite_fallback(emit_pass, circuit, op, qubit_indices, bindings)
    update_composite_result_mapping(op, qubit_groups, qubit_map)


def emit_invoke_operation(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: InvokeOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit an invoke operation.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted.
        op (InvokeOperation): Invocation to emit.
        qubit_map (QubitMap): Current qubit allocation map.
        bindings (dict[str, Any]): Active emit bindings.

    Raises:
        EmitError: If the invocation is opaque and has neither an executable
            body nor a selected native emitter.
    """
    backend_name = getattr(emit_pass, "backend_name", None)
    body = op.effective_body(backend=backend_name)
    impl = op.implementation_for(backend=backend_name)
    has_native_emitter = impl is not None and impl.emitter is not None
    if body is None and op.attrs.get("kind") == "oracle" and not has_native_emitter:
        raise EmitError(
            f"Oracle '{op.target.name}' has no implementation for execution.",
            operation=f"InvokeOperation[{op.target.name}]",
        )

    # Reject a manually constructed model-only definition instead of silently
    # emitting it as an identity barrier.
    definition = op.definition
    opaque_cost = getattr(definition, "opaque_cost", None) if definition else None
    if (
        body is None
        and not has_native_emitter
        and op.gate_type is CompositeGateType.CUSTOM
        and opaque_cost is not None
    ):
        raise EmitError(
            f"Composite '{op.target.name}' has an opaque cost for estimation "
            "but no executable body or native emitter for this backend; it "
            "cannot be transpiled to an executable circuit.",
            operation=f"InvokeOperation[{op.target.name}]",
        )

    emit_composite_gate(emit_pass, circuit, op, qubit_map, bindings)


def emit_composite_fallback(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: InvokeOperation,
    qubit_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit composite gate using decomposition.

    If the operation has a strategy name, resolve it from the callable's own
    implementation candidates.
    """
    if op.gate_type == CompositeGateType.QPE:
        emit_qpe_manual(emit_pass, circuit, op, qubit_indices, bindings)
    elif op.gate_type == CompositeGateType.QFT:
        emit_qft_with_strategy(emit_pass, circuit, op, qubit_indices)
    elif op.gate_type == CompositeGateType.IQFT:
        emit_iqft_with_strategy(emit_pass, circuit, op, qubit_indices)
    else:
        backend_name = getattr(emit_pass, "backend_name", None)
        impl = op.effective_body(backend=backend_name)
        if impl is not None:
            # _emit_custom_composite lives in controlled_emission module;
            # call via emit_pass so CudaqEmitPass overrides are respected.
            emit_pass._emit_custom_composite(circuit, op, impl, qubit_indices, bindings)
        elif qubit_indices:
            emit_pass._emitter.emit_barrier(circuit, qubit_indices)


def emit_callable_implementation_emitter(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: InvokeOperation,
    qubit_indices: list[int],
    bindings: dict[str, Any],
) -> bool:
    """Emit using a selected callable implementation emitter, if present.

    Args:
        emit_pass (StandardEmitPass): Active emit pass.
        circuit (Any): Backend circuit being emitted.
        op (InvokeOperation): Invocation to emit.
        qubit_indices (list[int]): Physical qubit indices in operand order.
        bindings (dict[str, Any]): Active emit bindings.

    Returns:
        bool: ``True`` when the implementation emitter handled the operation.

    Raises:
        EmitError: If the selected emitter object does not provide a callable
            ``emit`` method.
    """
    backend_name = getattr(emit_pass, "backend_name", None)
    impl = op.implementation_for(backend=backend_name)
    if impl is None or impl.emitter is None:
        return False

    emit = getattr(impl.emitter, "emit", None)
    if not callable(emit):
        raise EmitError(
            f"Callable implementation for '{op.target.name}' has a native "
            "emitter without an emit() method.",
            operation=f"InvokeOperation[{op.target.name}]",
        )
    own_controls = qubit_indices[: op.num_control_qubits]
    normalized_op = op
    if op.control_value is not None:
        normalized_attrs = dict(op.attrs)
        normalized_attrs.pop("control_value", None)
        normalized_op = dataclasses.replace(op, attrs=normalized_attrs)
    with bracket_control_value(
        emit_pass,
        circuit,
        own_controls,
        op.control_value,
    ):
        return bool(emit(circuit, normalized_op, qubit_indices, bindings))


def emit_qft_with_strategy(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: InvokeOperation,
    qubit_indices: list[int],
) -> None:
    """Emit QFT considering strategy selection.

    If ``approximate_kN`` is selected, omits rotations beyond depth ``N``.
    Otherwise falls back to standard QFT.

    Args:
        emit_pass (StandardEmitPass): The active emit pass whose emitter
            should receive decomposed QFT gates.
        circuit (Any): Backend circuit being emitted.
        op (InvokeOperation): Invocation expected to be a QFT.
        qubit_indices (list[int]): Physical qubit indices for the QFT target
            register.

    Raises:
        EmitError: If ``op`` is not a QFT composite operation.
    """
    _ensure_composite_gate_type(op, CompositeGateType.QFT, "emit_qft_with_strategy")

    truncation_depth = _qft_truncation_depth(op.strategy_name, "QFT")
    if truncation_depth is not None:
        emit_approximate_qft(emit_pass, circuit, qubit_indices, truncation_depth)
    else:
        emit_qft_manual(emit_pass, circuit, qubit_indices)


def emit_iqft_with_strategy(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: InvokeOperation,
    qubit_indices: list[int],
) -> None:
    """Emit IQFT considering strategy selection.

    If ``approximate_kN`` is selected, omits rotations beyond depth ``N``.
    Otherwise falls back to standard IQFT.

    Args:
        emit_pass (StandardEmitPass): The active emit pass whose emitter
            should receive decomposed IQFT gates.
        circuit (Any): Backend circuit being emitted.
        op (InvokeOperation): Invocation expected to be an IQFT.
        qubit_indices (list[int]): Physical qubit indices for the IQFT target
            register.

    Raises:
        EmitError: If ``op`` is not an IQFT composite operation.
    """
    _ensure_composite_gate_type(op, CompositeGateType.IQFT, "emit_iqft_with_strategy")

    truncation_depth = _qft_truncation_depth(op.strategy_name, "IQFT")
    if truncation_depth is not None:
        emit_approximate_iqft(emit_pass, circuit, qubit_indices, truncation_depth)
    else:
        emit_iqft_manual(emit_pass, circuit, qubit_indices)


def _qft_truncation_depth(strategy_name: str | None, operation: str) -> int | None:
    """Validate a QFT strategy and return its truncation depth.

    Args:
        strategy_name (str | None): Strategy metadata from the invocation.
        operation (str): ``"QFT"`` or ``"IQFT"`` for diagnostics.

    Returns:
        int | None: Positive approximate depth, or ``None`` for exact
        emission.

    Raises:
        EmitError: If the strategy is not ``exact`` or
            ``approximate_k<positive integer>``.
    """
    if strategy_name in (None, "exact"):
        return None
    match = re.fullmatch(r"approximate_k([1-9][0-9]*)", strategy_name)
    if match is None:
        raise EmitError(
            f"Invalid {operation} strategy {strategy_name!r}; expected "
            "'exact' or 'approximate_k<positive integer>'.",
            operation=f"InvokeOperation[{operation}]",
        )
    return int(match.group(1))


def emit_approximate_qft(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    qubit_indices: list[int],
    truncation_depth: int,
) -> None:
    """Emit approximate QFT with truncated rotations matching stdlib convention.

    Processes qubits from highest index to lowest (same as exact QFT), applying
    H then controlled-phase rotations with lower-indexed qubits. Rotations with
    exponent > truncation_depth are omitted. Finishes with bit-reversal SWAPs.

    Args:
        emit_pass: The StandardEmitPass instance.
        circuit: Target circuit
        qubit_indices: Qubit indices
        truncation_depth: Maximum exponent for controlled phase gates
    """
    n = len(qubit_indices)
    if n == 0:
        return

    k = truncation_depth

    for j in range(n - 1, -1, -1):
        emit_pass._emitter.emit_h(circuit, qubit_indices[j])
        for m in range(j - 1, max(j - k - 1, -1), -1):
            exponent = j - m
            if exponent <= k:
                angle = math.pi / (2**exponent)
                emit_pass._emitter.emit_cp(
                    circuit, qubit_indices[j], qubit_indices[m], angle
                )

    for j in range(n // 2):
        emit_pass._emitter.emit_swap(
            circuit, qubit_indices[j], qubit_indices[n - 1 - j]
        )


def emit_approximate_iqft(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    qubit_indices: list[int],
    truncation_depth: int,
) -> None:
    """Emit approximate IQFT with truncated rotations matching stdlib convention.

    IQFT = QFT†. SWAP comes first (undoing QFT's trailing SWAP), then for each
    qubit j (low-to-high), applies inverse controlled-phase rotations with
    lower-indexed qubits first (omitting exponents > truncation_depth), then H.

    Args:
        emit_pass: The StandardEmitPass instance.
        circuit: Target circuit
        qubit_indices: Qubit indices
        truncation_depth: Maximum exponent for controlled phase gates
    """
    n = len(qubit_indices)
    if n == 0:
        return

    k = truncation_depth

    # Bit-reversal SWAP first (inverse of QFT's trailing SWAP)
    for j in range(n // 2):
        emit_pass._emitter.emit_swap(
            circuit, qubit_indices[j], qubit_indices[n - 1 - j]
        )

    for j in range(n):
        for m in range(max(0, j - k), j):
            exponent = j - m
            if exponent <= k:
                angle = -math.pi / (2**exponent)
                emit_pass._emitter.emit_cp(
                    circuit, qubit_indices[j], qubit_indices[m], angle
                )
        emit_pass._emitter.emit_h(circuit, qubit_indices[j])


def emit_qft_manual(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    qubit_indices: list[int],
) -> None:
    """Emit QFT using decomposition matching stdlib convention.

    Processes qubits from highest index to lowest: for each qubit j, applies H
    then controlled-phase rotations with all lower-indexed qubits k (angle =
    π/2^(j-k)). Finishes with bit-reversal SWAPs.
    """
    n = len(qubit_indices)
    if n == 0:
        return

    for j in range(n - 1, -1, -1):
        emit_pass._emitter.emit_h(circuit, qubit_indices[j])
        for k in range(j - 1, -1, -1):
            angle = math.pi / (2 ** (j - k))
            emit_pass._emitter.emit_cp(
                circuit, qubit_indices[j], qubit_indices[k], angle
            )

    for j in range(n // 2):
        emit_pass._emitter.emit_swap(
            circuit, qubit_indices[j], qubit_indices[n - 1 - j]
        )


def emit_iqft_manual(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    qubit_indices: list[int],
) -> None:
    """Emit inverse QFT using decomposition matching stdlib convention.

    IQFT = QFT†. SWAP comes first (undoing QFT's trailing SWAP), then for each
    qubit j (low-to-high), applies inverse controlled-phase rotations with all
    lower-indexed qubits k (angle = -π/2^(j-k)) first, then H.
    """
    n = len(qubit_indices)
    if n == 0:
        return

    # Bit-reversal SWAP first (inverse of QFT's trailing SWAP)
    for j in range(n // 2):
        emit_pass._emitter.emit_swap(
            circuit, qubit_indices[j], qubit_indices[n - 1 - j]
        )

    for j in range(n):
        for k in range(j):
            angle = -math.pi / (2 ** (j - k))
            emit_pass._emitter.emit_cp(
                circuit, qubit_indices[j], qubit_indices[k], angle
            )
        emit_pass._emitter.emit_h(circuit, qubit_indices[j])


def emit_qpe_manual(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: InvokeOperation,
    qubit_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit QPE using manual decomposition.

    Args:
        emit_pass (StandardEmitPass): The active emit pass whose emitter
            should receive decomposed QPE gates.
        circuit (Any): Backend circuit being emitted.
        op (InvokeOperation): Invocation expected to be a QPE.
        qubit_indices (list[int]): Physical qubit indices for counting and
            target registers, in operation operand order.
        bindings (dict[str, Any]): Emit-time concrete bindings used to resolve
            QPE block parameters or phase operands.

    Raises:
        EmitError: If ``op`` is not a QPE composite operation.
    """
    _ensure_composite_gate_type(op, CompositeGateType.QPE, "emit_qpe_manual")

    num_counting = op.num_control_qubits
    num_targets = op.num_target_qubits

    if len(qubit_indices) < num_counting + num_targets:
        return

    counting_indices = qubit_indices[:num_counting]
    target_indices = qubit_indices[num_counting : num_counting + num_targets]

    # Step 1: Apply H to counting qubits
    for idx in counting_indices:
        emit_pass._emitter.emit_h(circuit, idx)

    # Step 2: Apply controlled-U^(2^k) operations
    if op.operands and hasattr(op.operands[0], "operations"):
        block_value = op.operands[0]

        local_bindings = emit_pass._resolver.bind_block_params(
            block_value,
            op.parameters,
            bindings,
            parameter_factory=emit_pass._get_or_create_parameter,
        )

        # _emit_controlled_powers lives in controlled_emission module;
        # call via emit_pass so subclass overrides are respected.
        emit_pass._emit_controlled_powers(
            circuit, block_value, counting_indices, target_indices, local_bindings
        )
    else:
        phase = extract_phase_from_params(emit_pass, op, bindings)
        if phase is not None:
            for k, ctrl_idx in enumerate(counting_indices):
                power = 2**k
                angle = phase * power
                for tgt_idx in target_indices:
                    emit_pass._emitter.emit_cp(circuit, ctrl_idx, tgt_idx, angle)

    # Step 3: Apply inverse QFT
    emit_iqft_manual(emit_pass, circuit, counting_indices)


def update_composite_result_mapping(
    op: InvokeOperation,
    qubit_groups: list[list[int]],
    qubit_map: QubitMap,
) -> None:
    """Update qubit_map for composite gate results.

    Args:
        op (InvokeOperation): Invocation whose quantum results should be
            mapped onto the unchanged physical slots of its operands.
        qubit_groups (list[list[int]]): Physical indices grouped per quantum
            operand. A scalar qubit contributes one index; a vector operand
            contributes one index per element.
        qubit_map (QubitMap): Mutable map updated in place.
    """
    from qamomile.circuit.ir.value import ArrayValue

    quantum_results = [result for result in op.results if result.type.is_quantum()]
    for result, group in zip(quantum_results, qubit_groups, strict=False):
        if isinstance(result, ArrayValue):
            map_array_result_group(result.uuid, group, qubit_map)
        elif group:
            qubit_map[QubitAddress(result.uuid)] = group[0]


def extract_phase_from_params(
    emit_pass: "StandardEmitPass",
    op: InvokeOperation,
    bindings: dict[str, Any],
) -> float | None:
    """Extract a concrete phase parameter from a QPE operation.

    Scans the operation's parameter operands. The block-free QPE fallback
    models a single phase parameter, so more than one concrete numeric
    parameter is ambiguous and rejected instead of guessed.

    Args:
        emit_pass (StandardEmitPass): The active emit pass whose resolver
            should resolve bound scalar and array-element operands.
        op (InvokeOperation): The QPE invocation.
        bindings (dict[str, Any]): Emit-time concrete bindings keyed by
            parameter name, value UUID, or loop-local variable name.

    Returns:
        float | None: The resolved phase angle, or ``None`` when the phase
            operand remains symbolic.

    Raises:
        EmitError: If ``op`` is not a QPE composite operation, or if multiple
            concrete numeric phase parameters are present.
    """
    _ensure_composite_gate_type(op, CompositeGateType.QPE, "extract_phase_from_params")

    phase: float | None = None
    for operand in op.parameters:
        if hasattr(operand, "type") and hasattr(operand.type, "is_classical"):
            if operand.type.is_classical():
                resolved = _resolve_phase_operand(emit_pass, operand, bindings)
                if resolved is None:
                    continue
                if phase is not None:
                    raise EmitError(
                        "QPE manual fallback requires exactly one concrete "
                        "phase parameter, but multiple numeric parameters "
                        "were resolved.",
                        operation="InvokeOperation[QPE]",
                    )
                phase = resolved
        elif operand.is_constant():
            const_val = operand.get_const()
            if const_val is not None:
                if phase is not None:
                    raise EmitError(
                        "QPE manual fallback requires exactly one concrete "
                        "phase parameter, but multiple numeric parameters "
                        "were resolved.",
                        operation="InvokeOperation[QPE]",
                    )
                phase = float(const_val)

    return phase


def _resolve_phase_operand(
    emit_pass: "StandardEmitPass",
    operand: Any,
    bindings: dict[str, Any],
) -> float | None:
    """Resolve a QPE phase operand to a concrete float.

    Symbolic array indices or slice bounds return ``None`` and preserve the
    existing manual-QPE fallback behavior, where no phase kickback gates are
    emitted when no concrete phase can be extracted.

    Args:
        emit_pass (StandardEmitPass): The active emit pass whose resolver
            should perform ordinary scalar and bound-array lookup.
        operand (Any): Candidate classical phase operand.
        bindings (dict[str, Any]): Emit-time concrete bindings.

    Returns:
        float | None: The resolved phase angle, or ``None`` when the operand
            is symbolic or its array index cannot be resolved.
    """
    resolved = emit_pass._resolver.resolve_classical_value(operand, bindings)
    if resolved is not None:
        return float(resolved)
    return None
