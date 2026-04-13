"""Control flow emission helpers for StandardEmitPass.

Extracted from ``standard_emit.py`` to keep the main class focused on
gate-level dispatch.  Each function mirrors the original method but takes
an explicit ``emit_pass`` parameter instead of ``self``.

These are the **default** implementations.  Backend-specific emit passes
(e.g., ``QiskitEmitPass``) may override the corresponding methods on
``StandardEmitPass``; calling ``super()._emit_for(...)`` etc. will
ultimately delegate here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass

from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import EmitError

from .condition_resolution import (
    map_phi_outputs,
    remap_static_phi_outputs,
    resolve_if_condition,
)
from .qubit_address import ClbitMap, QubitAddress, QubitMap


def resolve_loop_bounds(
    resolver: Any,
    op: ForOperation,
    bindings: dict[str, Any],
) -> tuple[int | None, int | None, int | None]:
    """Resolve for-loop bounds (start, stop, step) from operands.

    Missing operands use defaults: start=0, stop=1, step=1.
    Returns None for bounds that cannot be resolved to concrete ints.
    """
    start = (
        resolver.resolve_int_value(op.operands[0], bindings)
        if len(op.operands) > 0
        else 0
    )
    stop = (
        resolver.resolve_int_value(op.operands[1], bindings)
        if len(op.operands) > 1
        else 1
    )
    step = (
        resolver.resolve_int_value(op.operands[2], bindings)
        if len(op.operands) > 2
        else 1
    )
    return start, stop, step


# ---------------------------------------------------------------------------
# For loop
# ---------------------------------------------------------------------------


def emit_for(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: ForOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any],
    force_unroll: bool = False,
) -> None:
    """Emit a for loop."""
    start, stop, step = resolve_loop_bounds(emit_pass._resolver, op, bindings)

    if start is None or stop is None or step is None:
        emit_for_unrolled(emit_pass, circuit, op, qubit_map, clbit_map, bindings)
        return

    indexset = range(start, stop, step)
    if len(indexset) == 0:
        return

    if force_unroll:
        emit_for_unrolled(emit_pass, circuit, op, qubit_map, clbit_map, bindings)
        return

    if emit_pass._loop_analyzer.should_unroll(op, bindings):
        emit_for_unrolled(emit_pass, circuit, op, qubit_map, clbit_map, bindings)
        return

    # Try native for loop
    if emit_pass._emitter.supports_for_loop():
        loop_context = emit_pass._emitter.emit_for_loop_start(circuit, indexset)
        loop_bindings = bindings.copy()
        loop_bindings[op.loop_var] = loop_context
        emit_pass._emit_operations(
            circuit, op.operations, qubit_map, clbit_map, loop_bindings
        )
        emit_pass._emitter.emit_for_loop_end(circuit, loop_context)
    else:
        emit_for_unrolled(emit_pass, circuit, op, qubit_map, clbit_map, bindings)


# ---------------------------------------------------------------------------
# For loop (unrolled)
# ---------------------------------------------------------------------------


def emit_for_unrolled(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: ForOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit for loop by unrolling."""
    start, stop, step = resolve_loop_bounds(emit_pass._resolver, op, bindings)

    if start is None or stop is None or step is None:
        raise ValueError(
            f"Cannot unroll loop: bounds could not be resolved. "
            f"start={start}, stop={stop}, step={step}"
        )

    for i in range(start, stop, step):
        loop_bindings = bindings.copy()
        loop_bindings[op.loop_var] = i
        emit_pass._emit_operations(
            circuit, op.operations, qubit_map, clbit_map, loop_bindings
        )


# ---------------------------------------------------------------------------
# For-items loop
# ---------------------------------------------------------------------------


def emit_for_items(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: ForItemsOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit for-items loop (always unrolled).

    This handles iteration over Dict items, e.g.:
        for (i, j), Jij in qmc.items(ising):
            ...
    """
    if not op.operands:
        return

    dict_value = op.operands[0]
    entries = resolve_dict_entries(emit_pass, dict_value, bindings)

    if entries is None:
        raise ValueError(
            f"Cannot unroll for-items loop: dict entries could not be resolved. "
            f"Dict value: {dict_value.name if hasattr(dict_value, 'name') else dict_value}"
        )

    for key, value in entries:
        loop_bindings = bindings.copy()

        # Bind key variables (tuple unpacking)
        if len(op.key_vars) > 1:
            # Key is a tuple, unpack to multiple variables
            for i, kv_name in enumerate(op.key_vars):
                loop_bindings[kv_name] = key[i] if hasattr(key, "__getitem__") else key
        elif len(op.key_vars) == 1:
            # Single key variable
            loop_bindings[op.key_vars[0]] = key
            if op.key_is_vector:
                # Provide key length for Vector[UInt] shape resolution
                loop_bindings[f"{op.key_vars[0]}_dim0"] = len(key)

        # Bind value variable
        loop_bindings[op.value_var] = value

        emit_pass._emit_operations(
            circuit, op.operations, qubit_map, clbit_map, loop_bindings
        )


# ---------------------------------------------------------------------------
# Dict entry resolution (helper for emit_for_items)
# ---------------------------------------------------------------------------


def resolve_dict_entries(
    emit_pass: "StandardEmitPass",
    dict_value: Any,
    bindings: dict[str, Any],
) -> list[tuple[Any, Any]] | None:
    """Resolve DictValue to concrete (key, value) pairs.

    Args:
        emit_pass: The StandardEmitPass instance (for resolver access)
        dict_value: The DictValue IR node being iterated
        bindings: Current parameter bindings

    Returns:
        List of (key, value) tuples, or None if cannot be resolved
    """
    bound_items = dict_value.get_bound_data_items()
    if bound_items:
        return list(bound_items)

    # Check if dict_value is a parameter that should be bound
    if hasattr(dict_value, "is_parameter") and dict_value.is_parameter():
        param_name = dict_value.parameter_name()
        if param_name and param_name in bindings:
            bound = bindings[param_name]
            if isinstance(bound, dict):
                return list(bound.items())
            elif hasattr(bound, "items"):
                return list(bound.items())
            return bound

    # Check by name in bindings
    if hasattr(dict_value, "name") and dict_value.name in bindings:
        bound = bindings[dict_value.name]
        if isinstance(bound, dict):
            return list(bound.items())
        elif hasattr(bound, "items"):
            return list(bound.items())
        return bound

    # Check if dict_value has entries directly (from IR)
    if hasattr(dict_value, "entries") and dict_value.entries:
        # Resolve each entry
        resolved_entries = []
        for key_val, value_val in dict_value.entries:
            key = emit_pass._resolver.resolve_classical_value(key_val, bindings)
            value = emit_pass._resolver.resolve_classical_value(value_val, bindings)
            if key is not None and value is not None:
                resolved_entries.append((key, value))
        if resolved_entries:
            return resolved_entries

    return None


# ---------------------------------------------------------------------------
# If / else
# ---------------------------------------------------------------------------


def emit_if(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: IfOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit if/else operation.

    Handles two condition types:

    1. **Compile-time constant** (plain Python ``int``/``bool`` from
       ``@qkernel`` AST transformer closure variables, constant-folded
       Values, or Values resolvable via ``bindings``): the active
       branch is emitted unconditionally, the inactive branch is
       discarded.  No backend ``c_if`` / ``if_test`` is needed.
    2. **Runtime condition** (measurement ``Value`` that cannot be
       resolved at compile time): delegates to the backend's
       ``emit_if_start`` / ``emit_else_start`` / ``emit_if_end``
       protocol.
    """
    condition = op.condition
    resolved = resolve_if_condition(condition, bindings)

    # Compile-time constant condition: emit only the selected branch.
    # Use remap_static_phi_outputs (not the runtime map_phi_outputs
    # validator) so that dead-branch array quantum phi outputs are
    # aliased from the selected branch only.
    if resolved is not None:
        if resolved:
            emit_pass._emit_operations(
                circuit, op.true_operations, qubit_map, clbit_map, bindings
            )
        else:
            emit_pass._emit_operations(
                circuit, op.false_operations, qubit_map, clbit_map, bindings
            )
        remap_static_phi_outputs(op.phi_ops, resolved, qubit_map, clbit_map)
        return

    condition_addr = QubitAddress(condition.uuid)

    if condition_addr not in clbit_map:
        raise EmitError(
            "Runtime if-conditions must come from measurement results "
            "or be bound before transpilation. The condition value was "
            "neither resolved at compile time nor backed by a "
            "measurement result."
        )

    clbit_idx = clbit_map[condition_addr]

    if emit_pass._emitter.supports_if_else():
        context = emit_pass._emitter.emit_if_start(circuit, clbit_idx, 1)
        emit_pass._emit_operations(
            circuit, op.true_operations, qubit_map, clbit_map, bindings
        )
        if op.false_operations:
            emit_pass._emitter.emit_else_start(circuit, context)
            emit_pass._emit_operations(
                circuit, op.false_operations, qubit_map, clbit_map, bindings
            )
        emit_pass._emitter.emit_if_end(circuit, context)

        # Register phi output UUIDs so subsequent operations
        # (e.g., measure) can resolve the merged values.
        register_phi_outputs(emit_pass, op, qubit_map, clbit_map, bindings)
    else:
        raise EmitError(
            "Backend does not support native if/else control flow. "
            "Cannot emit IfOperation."
        )


# ---------------------------------------------------------------------------
# Phi output registration (helper for emit_if)
# ---------------------------------------------------------------------------


def register_phi_outputs(
    emit_pass: "StandardEmitPass",
    op: IfOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any] | None = None,
) -> None:
    """Register phi output UUIDs via the shared ``map_phi_outputs`` utility.

    Uses the full ``ValueResolver.resolve_qubit_index_detailed`` for
    scalar qubit resolution (handles array element operands).
    """
    resolver_bindings = bindings or {}

    def _resolve_scalar(source: Value, qmap: QubitMap) -> int | None:
        result = emit_pass._resolver.resolve_qubit_index_detailed(
            source, qmap, resolver_bindings
        )
        return result.index if result.success else None

    map_phi_outputs(op.phi_ops, qubit_map, clbit_map, _resolve_scalar)


# ---------------------------------------------------------------------------
# While loop
# ---------------------------------------------------------------------------


def emit_while(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: WhileOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit while loop operation."""
    if not op.operands:
        raise ValueError("WhileOperation requires a condition operand")

    condition = op.operands[0]
    condition_value = condition.value if hasattr(condition, "value") else condition
    condition_uuid = (
        condition_value.uuid
        if hasattr(condition_value, "uuid")
        else str(condition_value)
    )
    condition_addr = QubitAddress(condition_uuid)

    if condition_addr not in clbit_map:
        raise ValueError("While loop condition not found in classical bit map.")

    clbit_idx = clbit_map[condition_addr]

    if emit_pass._emitter.supports_while_loop():
        context = emit_pass._emitter.emit_while_start(circuit, clbit_idx, 1)
        emit_pass._emit_operations(
            circuit, op.operations, qubit_map, clbit_map, bindings
        )
        emit_pass._emitter.emit_while_end(circuit, context)
    else:
        raise EmitError(
            "Backend does not support native while loop control flow. "
            "Cannot emit WhileOperation."
        )
