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

from qamomile.circuit.ir.operation.classical_ops import DictGetItemOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    RegionArg,
    WhileOperation,
)
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.param_keys import dict_param_key

from .cast_binop_emission import _set_emit_value
from .condition_resolution import (
    map_merge_outputs,
    remap_static_merge_outputs,
    resolve_condition_address_detailed,
    resolve_if_condition,
)
from .qubit_address import ClbitMap, QubitAddress, QubitMap
from .value_resolver import ValueResolver


def resolve_condition_address(
    condition: Value,
    bindings: dict[str, Any],
    resolver: ValueResolver | None,
) -> QubitAddress:
    """Resolve a runtime control-flow condition to its ``clbit_map`` key.

    Scalar measurement results carry their own UUID and the clbit allocator
    registers them under ``QubitAddress(bit.uuid)``. ``Vector[Bit]`` element
    accesses (``s[i]`` where ``s = qmc.measure(register)``) instead live
    under ``QubitAddress(root_array.uuid, root_index)``. The element index
    and every ``slice_start`` / ``slice_step`` along the parent's
    ``slice_of`` chain are resolved the same way — taken directly when
    constant, otherwise folded through ``bindings`` via ``resolver`` so
    that loop-variable indices and runtime-valued slice bounds (``s[j:k]``
    where ``j``/``k`` are loop variables) both work. The chain composes
    into a root-space index via the standard affine map
    ``root_index = start + step * view_local_index`` repeated along the
    chain — matching ``ResourceAllocator._resolve_root_qubit_address`` /
    ``ValueResolver.resolve_slice_chain``. Falls back to the scalar
    address when no parent array is set, or when the index or any slice
    bound cannot be resolved to a concrete int (e.g. a backend runtime
    parameter, which cannot index a static classical register, or any
    symbolic value with no ``resolver``), deferring the diagnostic to the
    caller's ``clbit_map`` lookup. Used by both the default if/while
    emission path and the Qiskit / CUDA-Q backends when looking up a
    measurement-derived clbit for a runtime predicate.

    Args:
        condition (Value): Condition operand of an ``IfOperation`` or
            ``WhileOperation``, or an operand of a measurement-derived
            classical predicate (e.g. inside ``RuntimeClassicalExpr``).
        bindings (dict[str, Any]): Active emit-time bindings used to
            resolve symbolic indices and slice bounds (loop variables,
            compile-time-bound parameters).
        resolver (ValueResolver | None): The active ``ValueResolver``
            exposing ``resolve_int_value``. ``None`` is accepted for
            early-emit pre-scans (e.g. CUDA-Q's loop-carried clbit
            collector) that run before runtime bindings exist — only the
            constant path is taken in that case; symbolic indices and
            symbolic slice bounds fall through to the scalar UUID.

    Returns:
        QubitAddress: Key suitable for looking up the condition in
            ``clbit_map``.

    Raises:
        No exceptions for any well-formed IR. See
        ``resolve_condition_address_detailed`` for the resolution contract.
    """
    return resolve_condition_address_detailed(condition, bindings, resolver)[0]


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
# Loop-carried classical values (emit-time threading)
# ---------------------------------------------------------------------------


def _validated_region_args(
    op: ForOperation | ForItemsOperation | WhileOperation,
) -> tuple[RegionArg, ...]:
    """Validate and return a loop operation's region arguments.

    Region-argument results are definitions owned by the loop.  Requiring
    them to align with ``op.results`` prevents malformed or legacy IR from
    silently publishing a carried value under the wrong SSA identity.

    Args:
        op (ForOperation | ForItemsOperation | WhileOperation): The loop
            operation to validate.

    Returns:
        tuple[RegionArg, ...]: The validated region arguments.

    Raises:
        EmitError: If result counts or UUIDs do not align, block/result
            UUIDs are duplicated, or the four values of an argument have
            different types.
    """
    region_args = op.region_args
    if len(region_args) != len(op.results):
        raise EmitError(
            "Loop RegionArg data is inconsistent: "
            f"{len(region_args)} region args for {len(op.results)} results.",
            operation=type(op).__name__,
        )

    block_arg_uuids: set[str] = set()
    result_uuids: set[str] = set()
    for index, (arg, result) in enumerate(zip(region_args, op.results, strict=True)):
        if arg.result.uuid != result.uuid:
            raise EmitError(
                "Loop RegionArg data is inconsistent: "
                f"region_args[{index}].result does not match results[{index}].",
                operation=type(op).__name__,
            )
        if not (
            arg.init.type == arg.block_arg.type == arg.yielded.type == arg.result.type
        ):
            raise EmitError(
                f"Loop RegionArg '{arg.var_name}' has mismatched value types.",
                operation=type(op).__name__,
            )
        if arg.block_arg.uuid in block_arg_uuids or arg.result.uuid in result_uuids:
            raise EmitError(
                "Loop RegionArg data contains duplicate block/result identities.",
                operation=type(op).__name__,
            )
        block_arg_uuids.add(arg.block_arg.uuid)
        result_uuids.add(arg.result.uuid)
    return region_args


# ---------------------------------------------------------------------------
# For loop
# ---------------------------------------------------------------------------


def _seed_region_args(
    emit_pass: "StandardEmitPass",
    op: ForOperation | ForItemsOperation,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    """Resolve each region argument's init value into a carried-value map.

    Args:
        emit_pass (StandardEmitPass): The emit pass (for resolver access).
        op (ForOperation | ForItemsOperation): The loop carrying
            ``region_args``.
        bindings (dict[str, Any]): Current emit-time bindings.

    Returns:
        dict[str, Any]: ``block_arg.uuid`` → concrete (or backend
            symbolic ``Parameter``) init value for every region argument.

    Raises:
        EmitError: If an init value can be neither resolved concretely
            nor represented as a backend runtime parameter, or if the
            RegionArg identities are inconsistent.
    """
    carried: dict[str, Any] = {}
    for arg in _validated_region_args(op):
        init = emit_pass._resolver.resolve_classical_value(arg.init, bindings)
        if init is None:
            param_key = emit_pass._resolver.get_parameter_key(arg.init, bindings)
            if param_key:
                init = emit_pass._get_or_create_parameter(param_key, arg.init.uuid)
        if init is None:
            raise EmitError(
                f"Loop-carried value '{arg.var_name}' has an initial value "
                f"that cannot be resolved at emit time. Loop-carried "
                f"classical scalars must start from a compile-time-"
                f"resolvable value or a declared runtime parameter.",
                operation=type(op).__name__,
            )
        carried[arg.block_arg.uuid] = init
    return carried


def _advance_region_args(
    emit_pass: "StandardEmitPass",
    op: ForOperation | ForItemsOperation,
    carried: dict[str, Any],
    loop_bindings: dict[str, Any],
) -> None:
    """Carry each region argument's yielded value into the next iteration.

    The body emission has already evaluated classical ops into
    ``loop_bindings`` (``evaluate_binop`` writes results by UUID), so the
    yielded value is read from there first and only falls back to the
    resolver for pass-through shapes (identity carries, constants).

    Args:
        emit_pass (StandardEmitPass): The emit pass (for resolver access).
        op (ForOperation | ForItemsOperation): The loop carrying
            ``region_args``.
        carried (dict[str, Any]): The carried-value map from
            ``_seed_region_args``; updated in place.
        loop_bindings (dict[str, Any]): The just-emitted iteration's
            bindings.

    Raises:
        EmitError: If a yielded value cannot be resolved — e.g. it
            depends on a mid-circuit measurement outcome, which has no
            emit-time value.
    """
    # Resolve all yields before rebinding any block argument.  This is
    # observable for simultaneous updates such as ``a, b = b, a``.
    advanced: dict[str, Any] = {}
    for arg in _validated_region_args(op):
        nxt = loop_bindings.get(arg.yielded.uuid)
        if nxt is None:
            nxt = emit_pass._resolver.resolve_classical_value(
                arg.yielded, loop_bindings
            )
        if nxt is None:
            raise EmitError(
                f"Loop-carried value '{arg.var_name}' could not be computed "
                f"for the next iteration at emit time. Carried classical "
                f"scalars inside a quantum loop must be computable from "
                f"compile-time values, runtime parameters, and the loop "
                f"variable; values derived from measurement results cannot "
                f"be carried across unrolled iterations.",
                operation=type(op).__name__,
            )
        advanced[arg.block_arg.uuid] = nxt
    carried.update(advanced)


def _publish_region_results(
    op: ForOperation | ForItemsOperation,
    carried: dict[str, Any],
    bindings: dict[str, Any],
) -> None:
    """Expose each region argument's final carried value as the loop result.

    Args:
        op (ForOperation | ForItemsOperation): The emitted loop.
        carried (dict[str, Any]): The final carried-value map (init
            values for a zero-trip loop).
        bindings (dict[str, Any]): The enclosing emit bindings; mutated
            so post-loop operations resolve the loop results.
    """
    for arg in _validated_region_args(op):
        _set_emit_value(bindings, arg.result.uuid, carried[arg.block_arg.uuid])


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
    _validated_region_args(op)
    if op.loop_var_value is None:
        raise EmitError(
            f"ForOperation '{op.loop_var or '<unnamed>'}' has no "
            "loop_var_value; the IR must be rebuilt with the current "
            "frontend.",
            operation="ForOperation",
        )
    start, stop, step = resolve_loop_bounds(emit_pass._resolver, op, bindings)

    if start is None or stop is None or step is None:
        emit_for_unrolled(emit_pass, circuit, op, qubit_map, clbit_map, bindings)
        return

    indexset = range(start, stop, step)
    if len(indexset) == 0:
        # Zero-trip loop: region results still need publishing (they
        # pass the init values through).
        if op.region_args:
            carried = _seed_region_args(emit_pass, op, bindings)
            _publish_region_results(op, carried, bindings)
        return

    if force_unroll or op.region_args:
        # Region-carried values must be threaded iteration by iteration,
        # which only the unrolled path can do — a native backend loop
        # re-executes one body and cannot rebind a per-iteration
        # classical carried value.
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
            circuit,
            op.operations,
            qubit_map,
            clbit_map,
            loop_bindings,
            emit_qinit_reset=True,
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
    _validated_region_args(op)
    if op.loop_var_value is None:
        raise EmitError(
            f"ForOperation '{op.loop_var or '<unnamed>'}' has no "
            "loop_var_value; the IR must be rebuilt with the current "
            "frontend.",
            operation="ForOperation",
        )
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
        carry_note = (
            " Note: this loop carries classical values "
            f"({', '.join(arg.var_name for arg in op.region_args)}) "
            "between iterations; "
            "loop-carried classical values require "
            "compile-time-resolvable loop bounds, so a native runtime "
            "loop is not an option here."
            if op.region_args
            else ""
        )
        raise ValueError(
            "Cannot unroll loop: bounds could not be resolved at compile "
            f"time ({details}). Likely causes: (1) a parameter array shape "
            "dimension reached emit without being folded — bind the array "
            "concretely in transpile(bindings={...}), or use a separate "
            "compile-time loop counter (e.g. bindings={'p': p} with "
            f"qmc.range(p)); (2) a non-parameter symbolic value slipped "
            f"through the pipeline (report this as a compiler bug).{carry_note}"
        )

    carried = _seed_region_args(emit_pass, op, bindings)
    for i in range(start, stop, step):
        loop_bindings = bindings.copy()
        for uuid, value in carried.items():
            _set_emit_value(loop_bindings, uuid, value)
        _bind_loop_var(loop_bindings, op, i)
        emit_pass._emit_operations(
            circuit,
            op.operations,
            qubit_map,
            clbit_map,
            loop_bindings,
            emit_qinit_reset=True,
        )
        _advance_region_args(emit_pass, op, carried, loop_bindings)
    _publish_region_results(op, carried, bindings)


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

    Args:
        emit_pass (StandardEmitPass): The emit pass (for resolver access).
        circuit (Any): The backend circuit being built.
        op (ForItemsOperation): The for-items loop being unrolled.
        qubit_map (QubitMap): Current qubit address mapping.
        clbit_map (ClbitMap): Current classical bit address mapping.
        bindings (dict[str, Any]): Current emit-time bindings.

    Raises:
        EmitError: If the iterated dict is a runtime parameter — its key
            structure is unknown at compile time, so the loop cannot be
            unrolled.
        ValueError: If the dict entries cannot be resolved for any other
            reason.
    """
    if not op.operands:
        raise EmitError(
            "ForItemsOperation requires an iterable operand.",
            operation="ForItemsOperation",
        )

    key_var_values = op.key_var_values
    value_var_value = op.value_var_value
    if key_var_values is None or len(key_var_values) != len(op.key_vars):
        raise EmitError(
            "ForItemsOperation key identities are missing or inconsistent; "
            "the IR must be rebuilt with the current frontend.",
            operation="ForItemsOperation",
        )
    if value_var_value is None:
        raise EmitError(
            "ForItemsOperation value identity is missing; the IR must be "
            "rebuilt with the current frontend.",
            operation="ForItemsOperation",
        )
    vector_dim_value = None
    if op.key_is_vector:
        if (
            not key_var_values
            or not isinstance(key_var_values[0], ArrayValue)
            or not key_var_values[0].shape
        ):
            raise EmitError(
                "ForItemsOperation vector-key shape identity is missing; "
                "the IR must be rebuilt with the current frontend.",
                operation="ForItemsOperation",
            )
        vector_dim_value = key_var_values[0].shape[0]

    dict_value = op.operands[0]
    entries = resolve_dict_entries(emit_pass, dict_value, bindings)

    if entries is None:
        dict_name = getattr(dict_value, "name", None)
        if dict_name and dict_name in emit_pass._resolver.parameters:
            raise EmitError(
                f"Dict '{dict_name}' is a runtime parameter, so its key "
                f"structure is unknown at compile time and an items() "
                f"loop cannot be unrolled. Bind the dict via "
                f"bindings={{...}} instead, or restrict the kernel to "
                f"constant-key subscript lookups (d[key])."
            )
        raise ValueError(
            f"Cannot unroll for-items loop: dict entries could not be resolved. "
            f"Dict value: {dict_value.name if hasattr(dict_value, 'name') else dict_value}"
        )

    carried = _seed_region_args(emit_pass, op, bindings)
    for key, value in entries:
        loop_bindings = bindings.copy()
        for carried_uuid, carried_value in carried.items():
            _set_emit_value(loop_bindings, carried_uuid, carried_value)
        push = getattr(loop_bindings, "push_loop_var", None)

        def _bind(
            uuid: str,
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
            if callable(_push):
                _push(uuid, v, display_name=display_name or None)
            else:
                _ctx[uuid] = v
            if display_name:
                _ctx[display_name] = v

        # Bind key variables (tuple unpacking)
        if len(op.key_vars) > 1:
            for i, kv_name in enumerate(op.key_vars):
                _bind(
                    key_var_values[i].uuid,
                    kv_name,
                    key[i] if hasattr(key, "__getitem__") else key,
                )
        elif len(op.key_vars) == 1:
            _bind(key_var_values[0].uuid, op.key_vars[0], key)
            if op.key_is_vector:
                assert vector_dim_value is not None
                _bind(
                    vector_dim_value.uuid,
                    f"{op.key_vars[0]}_dim0",
                    len(key),
                )

        _bind(value_var_value.uuid, op.value_var, value)

        # ``emit_qinit_reset=True`` mirrors ``emit_for_unrolled`` and the
        # native ``for`` / ``if`` / ``while`` branches. Like those paths,
        # for-items re-emits the *same* ``op.operations`` once per dict entry
        # without UUID remapping — the ``ResourceAllocator`` registers each
        # body ``QInitOperation`` exactly once — so a fresh ``qmc.qubit(...)``
        # allocated in the body maps to one shared physical qubit reused
        # across entries. Without an explicit reset the second and later
        # entries would silently reuse it in its post-measurement state.
        emit_pass._emit_operations(
            circuit,
            op.operations,
            qubit_map,
            clbit_map,
            loop_bindings,
            emit_qinit_reset=True,
        )
        _advance_region_args(emit_pass, op, carried, loop_bindings)
    _publish_region_results(op, carried, bindings)


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
    # A dict carrying dict_runtime metadata is compile-time-bound and
    # authoritative even when its data is empty: presence of the metadata
    # (not non-emptiness of the data) is what marks it bound, mirroring
    # Dict.__getitem__ / __len__. Returning its (possibly empty) items lets
    # an empty bound dict — e.g. a Python signature default of ``{}`` —
    # unroll to a zero-iteration loop instead of falling through to the
    # "entries could not be resolved" error path below.
    metadata = getattr(dict_value, "metadata", None)
    if getattr(metadata, "dict_runtime", None) is not None:
        return list(dict_value.get_bound_data_items())

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
# Dict getitem (symbolic key lookup)
# ---------------------------------------------------------------------------


def evaluate_dict_getitem(
    emit_pass: "StandardEmitPass",
    op: DictGetItemOperation,
    bindings: dict[str, Any],
) -> None:
    """Evaluate a DictGetItemOperation and store the result in bindings.

    Resolves each key component (typically a for-items loop variable
    bound by UUID in ``bindings``), looks the key up in the dict's
    resolved entries, and writes the value into ``bindings`` under the
    result UUID so downstream ops (BinOps feeding gate angles) can
    consume it.

    When the dict is a declared runtime parameter
    (``transpile(..., parameters=["coeffs"])``), there are no entries to
    look up; the resolved key instead names one backend parameter
    (``coeffs[3]`` / ``coeffs[(0, 1)]``) which is stored under the
    result UUID. Repeated lookups of the same key share one backend
    parameter via ``_get_or_create_parameter``.

    Args:
        emit_pass (StandardEmitPass): The emit pass (for resolver access).
        op (DictGetItemOperation): The lookup op being evaluated.
        bindings (dict[str, Any]): Current emit-time bindings.

    Raises:
        EmitError: If a key component or the dict entries cannot be
            resolved, or the key is not present in the dict.
    """
    dict_value = op.operands[0]
    resolved_key: list[int] = []
    for key_value in op.operands[1:]:
        resolved = emit_pass._resolver.resolve_classical_value(key_value, bindings)
        if resolved is None:
            raise EmitError(
                f"Dict lookup key '{key_value.name}' could not be resolved "
                f"at emit time (dict '{getattr(dict_value, 'name', '?')}')"
            )
        resolved_key.append(int(resolved))

    lookup_key: Any = tuple(resolved_key) if op.key_arity > 1 else resolved_key[0]

    # Runtime-parameter dict: there is no bound data to look the key up
    # in — each resolved key becomes one backend parameter instead. The
    # key itself is still fully concrete here (loop unrolling has bound
    # any symbolic components), so the circuit structure stays static
    # while the looked-up value remains symbolic until execution time.
    dict_name = getattr(dict_value, "name", None)
    if dict_name and dict_name in emit_pass._resolver.parameters:
        param = emit_pass._get_or_create_parameter(
            dict_param_key(dict_name, lookup_key), op.results[0].uuid
        )
        _set_emit_value(bindings, op.results[0].uuid, param)
        return

    entries = resolve_dict_entries(emit_pass, dict_value, bindings)
    if entries is None:
        raise EmitError(
            f"Dict '{getattr(dict_value, 'name', '?')}' entries could not "
            f"be resolved for subscript lookup"
        )

    for entry_key, entry_value in entries:
        if isinstance(entry_key, (tuple, list)):
            entry_key = tuple(entry_key)
        if entry_key == lookup_key:
            _set_emit_value(bindings, op.results[0].uuid, entry_value)
            return
    raise EmitError(
        f"Key {lookup_key!r} not found in dict "
        f"'{getattr(dict_value, 'name', '?')}' during emit"
    )


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
    # Use remap_static_merge_outputs (not the runtime map_merge_outputs
    # validator) so that dead-branch array quantum merge outputs are
    # aliased from the selected branch only.
    if resolved is not None:
        if resolved:
            emit_pass._emit_operations(
                circuit,
                op.true_operations,
                qubit_map,
                clbit_map,
                bindings,
                emit_qinit_reset=True,
            )
        else:
            emit_pass._emit_operations(
                circuit,
                op.false_operations,
                qubit_map,
                clbit_map,
                bindings,
                emit_qinit_reset=True,
            )
        remap_static_merge_outputs(
            op,
            resolved,
            qubit_map,
            clbit_map,
            bindings=bindings,
            resolver=emit_pass._resolver,
        )
        register_classical_merge_aliases(emit_pass, op, bindings, resolved)
        return

    condition_addr = resolve_condition_address(condition, bindings, emit_pass._resolver)

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
            circuit,
            op.true_operations,
            qubit_map,
            clbit_map,
            bindings,
            emit_qinit_reset=True,
        )
        if op.false_operations:
            emit_pass._emitter.emit_else_start(circuit, context)
            emit_pass._emit_operations(
                circuit,
                op.false_operations,
                qubit_map,
                clbit_map,
                bindings,
                emit_qinit_reset=True,
            )
        emit_pass._emitter.emit_if_end(circuit, context)

        # Register merge output UUIDs so subsequent operations
        # (e.g., measure) can resolve the merged values.
        register_merge_outputs(emit_pass, op, qubit_map, clbit_map, bindings)
        register_classical_merge_aliases(emit_pass, op, bindings, None)
    else:
        raise EmitError(
            "Backend does not support native if/else control flow. "
            "Cannot emit IfOperation."
        )


# ---------------------------------------------------------------------------
# Merge output registration (helper for emit_if)
# ---------------------------------------------------------------------------


def register_merge_outputs(
    emit_pass: "StandardEmitPass",
    op: IfOperation,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    bindings: dict[str, Any] | None = None,
) -> None:
    """Register merge output UUIDs via the shared ``map_merge_outputs`` utility.

    Uses the full ``ValueResolver.resolve_qubit_index_detailed`` for
    scalar qubit resolution (handles array element operands). Runs at emit
    time with ``reject_runtime_bit_mux=True`` so an unrepresentable runtime
    multiplexing of two pre-existing measured bits fails loudly rather than
    silently binding the merge to the true branch.

    Args:
        emit_pass (StandardEmitPass): The active emit pass, providing the
            ``ValueResolver`` used for scalar / array-element resolution.
        op (IfOperation): The runtime if-else whose merged outputs are
            registered onto their physical clbits / qubits.
        qubit_map (QubitMap): Address-to-physical-qubit map, mutated in
            place.
        clbit_map (ClbitMap): Address-to-physical-clbit map, mutated in
            place.
        bindings (dict[str, Any] | None): Active emit-time bindings used to
            fold a merge source's symbolic ``Vector[Bit]`` element index
            (e.g. an unrolled loop variable). Defaults to None (empty).

    Raises:
        EmitError: If a quantum merge's branches resolve to different
            physical resources, or a runtime scalar ``Bit`` merge
            multiplexes two distinct pre-existing measured clbits (see
            ``map_merge_outputs``).
    """
    resolver_bindings = bindings or {}

    def _resolve_scalar(source: Value, qmap: QubitMap) -> int | None:
        result = emit_pass._resolver.resolve_qubit_index_detailed(
            source, qmap, resolver_bindings
        )
        return result.index if result.success else None

    map_merge_outputs(
        op,
        qubit_map,
        clbit_map,
        _resolve_scalar,
        bindings=resolver_bindings,
        resolver=emit_pass._resolver,
        # Emit-time only: by now representable Bit merges (branch-local
        # fresh measurements, while-loop-carried conditions) have been
        # aliased onto a single shared clbit, so a still-distinct pair of
        # source clbits marks the unrepresentable runtime pre-measured
        # multiplexing shape and must fail loudly instead of silently
        # binding the merge to the true branch.
        reject_runtime_bit_mux=True,
    )


def register_classical_merge_aliases(
    emit_pass: "StandardEmitPass",
    op: IfOperation,
    bindings: dict[str, Any],
    resolved: bool | None,
) -> None:
    """Bind classical merge outputs to a concrete value when resolvable.

    The frontend creates a merge for *every* variable referenced in an
    if-branch, including read-only ones (e.g. a for-loop index ``j`` that
    is read but not assigned in the branch). These read-only merges are
    identity merges — both inputs reference the same IR Value — so the
    merge output is deterministically equal to that input.

    For classical types (UInt / Float / Bit) the merge outputs are not
    captured by ``map_merge_outputs`` / ``remap_static_merge_outputs`` (which
    only handle qubit / clbit phys-resource mapping). Without this
    binding, downstream uses like ``data[j_merge_4]`` cannot resolve the
    index and emit fails with ``symbolic_index_not_bound``.

    The alias is written to ``bindings`` by both UUID and (when present)
    name, mirroring the pattern used by ``emit_for_unrolled`` for the
    original loop variable.

    Args:
        emit_pass (StandardEmitPass): The active emit pass (for resolver
            access).
        op (IfOperation): The if-else whose merged classical outputs
            should be bound; merges are read through ``iter_merges``.
        bindings (dict[str, Any]): Current bindings; mutated in place to
            bind merge outputs.
        resolved (bool | None): ``True`` / ``False`` if the if was
            compile-time resolved (use the selected branch's input);
            ``None`` if it was a runtime if (only bind identity merges).

    Returns:
        None.
    """
    for merge in op.iter_merges():
        output = merge.result
        # Only handle classical types; quantum/bit merge physical mapping
        # is the responsibility of map_merge_outputs / remap_static_merge_outputs.
        if output.type.is_quantum() or hasattr(output.type, "_is_bit_marker"):
            continue

        # Compile-time path: bind to selected branch's input.
        if resolved is not None:
            value = emit_pass._resolver.resolve_classical_value(
                merge.select(resolved), bindings
            )
        else:
            # Runtime path: only bind identity merges (read-only variable)
            # — otherwise the merge output truly depends on the runtime
            # branch and we can't pre-bind.
            if not merge.is_identity:
                continue
            value = emit_pass._resolver.resolve_classical_value(
                merge.true_value, bindings
            )

        if value is None:
            continue
        # Merge output is a UUID-identified intermediate; route through the
        # typed slot when the bindings is an EmitContext.
        set_value = getattr(bindings, "set_value", None)
        if callable(set_value):
            set_value(output.uuid, value)
        else:
            bindings[output.uuid] = value
        if output.name:
            # Merge output names are like ``"j_merge_4"`` — unique per merge op
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
    region_args = _validated_region_args(op)
    if region_args:
        # Backstop for the transpile-time rejection: a while loop's trip
        # count is a runtime measurement outcome, so the loop must stay
        # a runtime loop, and no backend can carry a classical value
        # between runtime-loop iterations.
        raise EmitError(
            "Loop-carried classical values in a while loop cannot be "
            f"emitted ({', '.join(arg.var_name for arg in region_args)}): "
            "a runtime loop "
            "re-executes one static body and cannot thread a classical "
            "value between iterations.",
            operation="WhileOperation",
        )
    if not op.operands:
        raise EmitError(
            "WhileOperation requires a condition operand.",
            operation="WhileOperation",
        )

    condition = op.operands[0]
    condition_value = condition.value if hasattr(condition, "value") else condition
    if isinstance(condition_value, Value):
        condition_addr = resolve_condition_address(
            condition_value, bindings, emit_pass._resolver
        )
    else:
        condition_addr = QubitAddress(str(condition_value))

    if condition_addr not in clbit_map:
        raise EmitError(
            "Runtime while-conditions must come from measurement results "
            "or be bound before transpilation. The condition value was "
            "neither resolved at compile time nor backed by a "
            "measurement result.",
            operation="WhileOperation",
        )

    clbit_idx = clbit_map[condition_addr]

    if emit_pass._emitter.supports_while_loop():
        context = emit_pass._emitter.emit_while_start(circuit, clbit_idx, 1)
        emit_pass._emit_operations(
            circuit,
            op.operations,
            qubit_map,
            clbit_map,
            bindings,
            emit_qinit_reset=True,
        )
        emit_pass._emitter.emit_while_end(circuit, context)
    else:
        raise EmitError(
            "Backend does not support native while loop control flow. "
            "Cannot emit WhileOperation."
        )
