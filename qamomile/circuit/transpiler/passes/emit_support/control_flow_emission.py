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
        _bind_loop_var(loop_bindings, op, loop_context)
        emit_pass._emit_operations(
            circuit, op.operations, qubit_map, clbit_map, loop_bindings
        )
        emit_pass._emitter.emit_for_loop_end(circuit, loop_context)
    else:
        emit_for_unrolled(emit_pass, circuit, op, qubit_map, clbit_map, bindings)


def _bind_loop_var(
    loop_bindings: Any,
    op: ForOperation,
    value: Any,
) -> None:
    """Push the loop variable binding into ``loop_bindings``.

    Single source of truth for "how is a loop iteration variable bound
    into the emit context". Always writes UUID-keyed (the canonical
    identity). Nested loops with identical display names (e.g. outer
    and inner ``for i``) coexist safely because each ``ForOperation``
    carries its own ``loop_var_value`` with a distinct UUID.

    Args:
        loop_bindings: The emit context to receive the binding.
            ``EmitContext`` instances are written via ``push_loop_var``;
            plain dicts get a UUID-keyed write.
        op: The ``ForOperation`` whose iteration variable is being bound.
            ``op.loop_var_value`` must not be None.
        value: The bound iteration value (int / Hamiltonian item /
            backend loop parameter / etc.).

    Raises:
        EmitError: If ``op.loop_var_value`` is None — the IR predates
            the UUID-keyed binding migration. Such IR cannot be emitted
            correctly because the loop variable has no stable identity.
    """
    if op.loop_var_value is None:
        raise EmitError(
            f"ForOperation '{op.loop_var or '<unnamed>'}' has no "
            "loop_var_value; cannot bind the iteration variable. The IR "
            "must be re-built with the current frontend.",
            operation="ForOperation",
        )
    uuid = op.loop_var_value.uuid
    push_loop_var = getattr(loop_bindings, "push_loop_var", None)
    if callable(push_loop_var):
        push_loop_var(uuid, value, display_name=op.loop_var or None)
    else:
        loop_bindings[uuid] = value


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
        # Identify the unresolved operand(s) so the user can act on it.
        labels = ("start", "stop", "step")
        values = (start, stop, step)
        operands = op.operands
        unresolved_details: list[str] = []
        for label, resolved, operand in zip(labels, values, operands):
            if resolved is None:
                name = getattr(operand, "name", None) or "<anonymous>"
                unresolved_details.append(f"{label}={name!r}")
        details = ", ".join(unresolved_details)
        raise ValueError(
            "Cannot unroll loop: bounds could not be resolved at compile "
            f"time ({details}). Likely causes: (1) a parameter array shape "
            "dimension reached emit without being folded — bind the array "
            "concretely in transpile(bindings={...}), or use a separate "
            "compile-time loop counter (e.g. bindings={'p': p} with "
            "qmc.range(p)); (2) a non-parameter symbolic value slipped "
            "through the pipeline (report this as a compiler bug)."
        )

    for i in range(start, stop, step):
        loop_bindings = bindings.copy()
        _bind_loop_var(loop_bindings, op, i)
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

    key_var_values = op.key_var_values
    value_var_value = op.value_var_value

    for key, value in entries:
        loop_bindings = bindings.copy()
        push = getattr(loop_bindings, "push_loop_var", None)

        def _bind(
            uuid: str | None,
            display_name: str,
            v: Any,
            _push=push,
            _ctx=loop_bindings,
        ) -> None:
            """Bind a for-items key/value variable.

            For-items key variables and dim0-shape entries are read by
            name from ``_index_into_array`` (the container lookup is
            ``bindings.get(parent.name)``), so the name-keyed write
            stays here even though scalar ``for`` loop variables no
            longer write by name. UUID-keyed writes happen alongside
            for canonical identity tracking.
            """
            if callable(_push) and uuid is not None:
                _push(uuid, v, display_name=display_name or None)
            else:
                if uuid is not None:
                    _ctx[uuid] = v
            if display_name:
                _ctx[display_name] = v

        # Bind key variables (tuple unpacking)
        if len(op.key_vars) > 1:
            for i, kv_name in enumerate(op.key_vars):
                kv_uuid = (
                    key_var_values[i].uuid
                    if key_var_values is not None and i < len(key_var_values)
                    else None
                )
                _bind(
                    kv_uuid,
                    kv_name,
                    key[i] if hasattr(key, "__getitem__") else key,
                )
        elif len(op.key_vars) == 1:
            kv_uuid = (
                key_var_values[0].uuid
                if key_var_values is not None and len(key_var_values) >= 1
                else None
            )
            _bind(kv_uuid, op.key_vars[0], key)
            if op.key_is_vector:
                # The dim0 Value is a child of the ArrayValue stored in
                # key_var_values[0]; for now bind the legacy name only
                # (Vector key dim is rarely accessed by the loop body).
                _bind(None, f"{op.key_vars[0]}_dim0", len(key))

        value_uuid = value_var_value.uuid if value_var_value is not None else None
        _bind(value_uuid, op.value_var, value)

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
    if hasattr(dict_value, "name") and dict_value.name and dict_value.name in bindings:
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
        register_classical_phi_aliases(emit_pass, op.phi_ops, bindings, resolved)
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
        register_classical_phi_aliases(emit_pass, op.phi_ops, bindings, None)
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


def register_classical_phi_aliases(
    emit_pass: "StandardEmitPass",
    phi_ops: list,
    bindings: dict[str, Any],
    resolved: bool | None,
) -> None:
    """Bind classical phi outputs to a concrete value when resolvable.

    The frontend creates a phi for *every* variable referenced in an
    if-branch, including read-only ones (e.g. a for-loop index ``j`` that
    is read but not assigned in the branch). These read-only phis have
    ``true_value is false_value`` — both inputs reference the same IR
    Value — so the phi output is deterministically equal to that input.

    For classical types (UInt / Float / Bit) the phi outputs are not
    captured by ``map_phi_outputs`` / ``remap_static_phi_outputs`` (which
    only handle qubit / clbit phys-resource mapping). Without this
    binding, downstream uses like ``data[j_phi_4]`` cannot resolve the
    index and emit fails with ``symbolic_index_not_bound``.

    The alias is written to ``bindings`` by both UUID and (when present)
    name, mirroring the pattern used by ``emit_for_unrolled`` for the
    original loop variable.

    Args:
        emit_pass: The active emit pass (for resolver access).
        phi_ops: ``IfOperation.phi_ops``.
        bindings: Current bindings; mutated in place to bind phi outputs.
        resolved: ``True`` / ``False`` if the if was compile-time resolved
            (use the selected branch's input); ``None`` if it was a
            runtime if (only bind when both inputs resolve to the same
            value).

    Returns:
        None.
    """
    from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp

    for phi in phi_ops:
        if not isinstance(phi, PhiOp):
            continue
        output = phi.results[0]
        # Only handle classical types; quantum/bit phi physical mapping
        # is the responsibility of map_phi_outputs / remap_static_phi_outputs.
        if output.type.is_quantum() or hasattr(output.type, "_is_bit_marker"):
            continue
        true_v = phi.true_value
        false_v = phi.false_value

        # Compile-time path: bind to selected branch's input.
        if resolved is True:
            value = emit_pass._resolver.resolve_classical_value(true_v, bindings)
        elif resolved is False:
            value = emit_pass._resolver.resolve_classical_value(false_v, bindings)
        else:
            # Runtime path: only bind when both inputs are the same Value
            # (read-only variable) — otherwise the phi output truly depends
            # on the runtime branch and we can't pre-bind.
            if not (
                hasattr(true_v, "uuid")
                and hasattr(false_v, "uuid")
                and true_v.uuid == false_v.uuid
            ):
                continue
            value = emit_pass._resolver.resolve_classical_value(true_v, bindings)

        if value is None:
            continue
        # Phi output is a UUID-identified intermediate; route through the
        # typed slot when the bindings is an EmitContext.
        set_value = getattr(bindings, "set_value", None)
        if callable(set_value):
            set_value(output.uuid, value)
        else:
            bindings[output.uuid] = value
        if output.name:
            # Phi output names are like ``"j_phi_4"`` — unique per phi op
            # within an if, so name-keyed writes don't collide the way
            # generic ``"uint_tmp"`` tmp names do. Stored under the flat
            # dict for legacy lookups; not a separate semantic slot.
            bindings[output.name] = value


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
