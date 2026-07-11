import builtins
import contextlib
import contextvars
import copy
import dataclasses
import typing

from qamomile.circuit.frontend.func_to_block import is_array_type
from qamomile.circuit.frontend.handle.array import ArrayBase, Vector
from qamomile.circuit.frontend.handle.containers import Dict, DictItemsIterator
from qamomile.circuit.frontend.handle.primitives import (
    Float,
    Handle,
    UInt,
)
from qamomile.circuit.frontend.tracer import Tracer, get_current_tracer, trace
from qamomile.circuit.ir.operation.control_flow import (
    BranchRebind,
    ForItemsOperation,
    ForOperation,
    IfOperation,
    LoopCarriedRebind,
    RegionArg,
    WhileOperation,
)
from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value


class WhileLoop:
    pass


@contextlib.contextmanager
def while_loop(cond: typing.Callable) -> typing.Generator[WhileLoop, None, None]:
    """Create a while loop whose condition is a measurement result.

    The condition must be a ``Bit`` produced by ``qmc.measure()``.
    Non-measurement conditions (classical variables, constants,
    comparisons) are accepted at build time but will be rejected by
    ``ValidateWhileContractPass`` during transpilation.

    Args:
        cond: A callable (lambda) that returns the loop condition.
            Must return a ``Bit`` handle originating from ``qmc.measure()``.

    Yields:
        WhileLoop: A marker object for the while loop context.

    Example::

        @qm.qkernel
        def repeat_until_zero() -> qm.Bit:
            q = qm.qubit("q")
            q = qm.h(q)
            bit = qm.measure(q)
            while bit:
                q2 = qm.qubit("q2")
                q2 = qm.h(q2)
                bit = qm.measure(q2)
            return bit

    The body register is a body-local name (``q2``), not a rebind of the
    pre-loop ``q``: rebinding a pre-existing quantum variable to a
    register allocated in the body is rejected by the transpiler's
    control-flow discard check, because the runtime loop re-executes its
    body on one persistent register without reset and cannot realize
    "fresh per iteration" semantics for the rebound name.
    """
    # 1. Get the PARENT tracer (the one active before entering the while loop)
    parent_tracer = get_current_tracer()

    # 2. Evaluate the condition lambda to get the condition expression
    # The lambda returns a Handle (e.g., result of i < n comparison).
    # Normalize to an IR Value: WhileOperation operands must never hold
    # frontend handles (the serializer reads operand UUIDs) or raw
    # Python primitives (``while True:`` becomes a const Bit Value that
    # ValidateWhileContractPass rejects with a typed error).
    condition_result = cond()
    condition_value = _value_to_ir_value(condition_result, "while_cond")

    # 3. Create a new tracer for capturing body operations
    body_tracer = Tracer()

    # 4. Yield inside the body tracer context so body operations get captured
    with trace(body_tracer):
        yield WhileLoop()

    # 5. After the with block exits, body_tracer.operations contains the body
    # 6. Re-evaluate the condition lambda to capture loop-carried updates.
    #    Python closures capture variables by reference, so if the body
    #    reassigned the condition variable (e.g., bit = qmc.measure(q)),
    #    calling cond() again returns the NEW handle.  This lets us detect
    #    that the body produces an updated condition and alias both to the
    #    same classical bit during resource allocation.
    # Re-evaluate in a temporary tracer to discard any side-effect operations
    # (e.g., CompOp from comparison conditions like `while i < n:`).
    temp_tracer = Tracer()
    with trace(temp_tracer):
        condition_after = cond()
    condition_after_value = _value_to_ir_value(condition_after, "while_cond")

    # 7. Create WhileOperation with captured body operations
    while_op = WhileOperation(
        operations=body_tracer.operations,
        loop_carried_rebinds=body_tracer.loop_carried_rebinds,
    )
    # operands[0]: initial condition (checked at loop entry)
    while_op.operands.append(condition_value)
    # operands[1]: loop-carried condition (updated inside body)
    # Only append if the body actually produced a different handle.
    if condition_after is not condition_result:
        while_op.operands.append(condition_after_value)

    # 8. Add the WhileOperation to the PARENT tracer (not a local one)
    parent_tracer.add_operation(while_op)


@contextlib.contextmanager
def for_loop(
    start, stop, step=1, var_name: str = "_loop_idx"
) -> typing.Generator[UInt, None, None]:
    """Builder function to create a for loop in Qamomile frontend.

    Args:
        start: Loop start value (can be Handle or int)
        stop: Loop stop value (can be Handle or int)
        step: Loop step value (default=1)
        var_name: Name of the loop variable (default="_loop_idx")

    Yields:
        UInt: The loop iteration variable (can be used as array index)

    Example:
        ```python
        @QKernel
        def my_kernel(qubits: Array[Qubit, Literal[3]]) -> Array[Qubit, Literal[3]]:
            for i in qm.range(3):
                qubits[i] = h(qubits[i])
            return qubits

        @QKernel
        def my_kernel2(qubits: Array[Qubit, Literal[5]]) -> Array[Qubit, Literal[5]]:
            for i in qm.range(1, 4):  # i = 1, 2, 3
                qubits[i] = h(qubits[i])
            return qubits
        ```
    """
    parent_tracer = get_current_tracer()
    body_tracer = Tracer()

    # Create a UInt to represent the loop variable (can be used as array index)
    # ``loop_var_value`` is the IR identity carrier (UUID); ``var_name`` is
    # display-only and survives only for printers / error messages.
    loop_var_value = Value(type=UIntType(), name=var_name)
    loop_var = UInt(
        value=loop_var_value,
        init_value=0,  # Placeholder, actual value is symbolic during tracing
    )

    with trace(body_tracer):
        yield loop_var

    # Convert pending region entries (loop-carried classical scalars)
    # into explicit RegionArg records and publish post-loop result
    # handles for the AST-injected loop_region_result assignments.
    region_args, region_results = _close_region_entries(body_tracer, parent_tracer)

    # Create ForOperation
    # operands: [start, stop, step]
    for_op = ForOperation(
        loop_var=var_name,
        loop_var_value=loop_var_value,
        operations=body_tracer.operations,
        loop_carried_rebinds=body_tracer.loop_carried_rebinds,
        region_args=region_args,
    )
    for_op.results.extend(region_results)
    for_op.operands.append(_value_to_ir_value(start, "start"))
    for_op.operands.append(_value_to_ir_value(stop, "stop"))
    for_op.operands.append(_value_to_ir_value(step, "step"))

    parent_tracer.add_operation(for_op)


def _fresh_handle_copy_for_tracing(h: typing.Any) -> typing.Any:
    """Create a branch-independent Handle copy for if-else branch tracing.

    This function intentionally accesses Handle's private ``_consumed`` /
    ``_consumed_by`` / ``_consumed_at`` / ``_consumed_pre_branch`` attributes.
    Direct consumed-state manipulation is confined to the phi-merge machinery
    in this module: here (giving each branch its own consumption state so
    mutually-exclusive branches trace independently) and in
    ``_mark_conditionally_consumed`` (re-marking the merged handle when a
    branch consumed the value — the conditional-move rule).  Keeping this off
    the public Handle surface preserves the affine-type enforcement that
    prevents qubit reuse bugs.

    Branch independence means each branch gets its own consumption state so
    that consuming a handle inside one branch does not poison tracing of the
    mutually-exclusive branch.  It does NOT mean resurrection: a handle that
    was already consumed BEFORE the branch stays consumed in both copies, so
    using it inside a branch raises ``QubitConsumedError`` at trace time with
    the original consumer's location.  Such copies are marked with
    ``_consumed_pre_branch`` so the phi-merge machinery can still recognize
    the slot as untouched by the branch (elision stays valid and the
    conditional-move rule does not misfire — see ``_consumed_in_branch``).

    Scope note: this preserves *scalar* pre-branch consumption. The array
    element borrow table is still reset to empty per branch, because the
    phi-merge element-borrow union (``_merge_branch_element_borrows``)
    assumes each branch's leftover borrows are in-branch only; seeding it
    with pre-branch borrows would double-count them onto the merged handle.

    Non-Handle values (int, float, etc.) are returned unchanged.

    Args:
        h (typing.Any): Variable captured for the branch bodies — a Handle
            or a plain Python value.

    Returns:
        typing.Any: A branch-local shallow copy for Handles, or ``h``
            unchanged for non-Handles.
    """
    if not isinstance(h, Handle):
        return h
    c = copy.copy(h)
    if c._consumed:
        # Consumed before the branch: keep the consumed state (no
        # resurrection) but mark it so the merge machinery can tell it
        # apart from in-branch consumption.
        c._consumed_pre_branch = True
    else:
        c._consumed_by = None
        c._consumed_at = None
    # Reset borrowed-element tracking for ArrayBase instances so that
    # each branch starts with an empty borrow set.  Without this,
    # shallow copy shares the same _borrowed_indices dict and borrowing
    # an element in one branch would cause QubitConsumedError in the other.
    if isinstance(c, ArrayBase):
        c._borrowed_indices = {}
    return c


# Marker phrase embedded in the ``_consumed_by`` message a conditional
# consumption produces. Used to detect a nested-if phrase and avoid
# re-wrapping it (which would duplicate the suffix).
_COND_CONSUME_MARKER = "of the preceding if/else"


def _consumed_in_branch(val: typing.Any) -> bool:
    """Check whether a branch-returned value was consumed INSIDE the branch.

    Branch-tracing copies of handles that were already consumed before the
    if-else carry ``_consumed_pre_branch`` (see
    ``_fresh_handle_copy_for_tracing``). Their ``_consumed`` flag is True, but
    the branch itself provably did not touch them, so they must not block
    merge elision nor trigger the conditional-move rule
    (``_mark_conditionally_consumed``) — only values consumed by an operation
    *inside* the branch should. This helper is the single source of truth for
    "the branch consumed this value".

    Args:
        val (typing.Any): Value a branch returned for a variable slot
            (Handle, Value, or primitive).

    Returns:
        bool: ``True`` when the value was consumed by an operation inside the
            branch body (as opposed to before the if-else).
    """
    return getattr(val, "_consumed", False) and not getattr(
        val, "_consumed_pre_branch", False
    )


def _merge_branch_element_borrows(
    merged_handle: typing.Any,
    true_val: typing.Any,
    false_val: typing.Any,
) -> None:
    """Carry unreturned branch element borrows onto the phi-merged array.

    Each branch traces against a fresh handle copy whose ``_borrowed_indices``
    starts empty (so mutually-exclusive branches don't conflict). Any entry
    still outstanding at the end of a branch is an element that was borrowed
    and never returned on that path — most importantly an element
    destructively consumed inside the branch (``if sel: _ = measure(qs[0])``).
    Before this merge, that per-element state was silently dropped: a post-if
    ``qs[0] = h(qs[0])`` compiled even though on the taken path it acts on a
    measured wire — the exact program that raises when written without the
    enclosing ``if``. Union-merging both branches' leftover borrow entries
    onto the merged handle routes the post-if element access into the
    existing borrow / destroyed-slot enforcement in
    ``ArrayBase.__getitem__``, which raises at trace time.

    A borrow that IS returned inside its branch (``if sel: qs[0] =
    h(qs[0])``) releases its entry before the branch ends and is unaffected.
    Dropping the array (no post-if element access) also stays legal — the
    entries only bite on reuse.

    Args:
        merged_handle (typing.Any): The phi-result handle for this variable.
        true_val (typing.Any): The true-branch handle/value (its leftover
            ``_borrowed_indices`` are merged when it is an ``ArrayBase``).
        false_val (typing.Any): The false-branch handle/value.

    Returns:
        None
    """
    if not isinstance(merged_handle, ArrayBase):
        return
    value = getattr(merged_handle, "value", None)
    if value is None or not value.type.is_quantum():
        return
    for branch_val in (true_val, false_val):
        if isinstance(branch_val, ArrayBase):
            merged_handle._borrowed_indices.update(branch_val._borrowed_indices)


def _mark_conditionally_consumed(
    merged_handle: typing.Any,
    true_val: typing.Any,
    false_val: typing.Any,
    true_consumed: bool,
    false_consumed: bool,
) -> None:
    """Mark a phi-merged quantum handle consumed if a branch consumed it.

    Implements the conditional-move rule (as in Rust's borrow checker /
    ``maybe-init`` dataflow): a quantum value consumed on *any* branch of an
    if/else is treated as consumed after the merge. Without this, a variable
    consumed on one path (e.g. ``if c: _ = qmc.measure(q)``) but not rebound
    would still surface after the if as a live handle wrapping the phi result,
    so a later ``q = qmc.h(q)`` compiled silently — the exact program that
    raises ``QubitConsumedError`` when written without the enclosing ``if``.
    Marking the merged handle consumed makes the reuse raise at trace time
    with an actionable message, restoring affine soundness.

    This intentionally accesses the Handle's private ``_consumed`` /
    ``_consumed_by`` fields, mirroring ``_fresh_handle_copy_for_tracing``:
    phi-merge is the one place that must adjust consumed state directly. Only
    quantum handles are marked (classical values are non-affine and may be
    reused freely). Dropping the value (never using it after the if) stays
    legal — affinity permits drop; only *reuse* now raises.

    Args:
        merged_handle (typing.Any): The phi-result handle returned to the
            traced Python code for this variable.
        true_val (typing.Any): The true-branch handle/value for this variable
            (carries ``_consumed_by`` when it was consumed).
        false_val (typing.Any): The false-branch handle/value for this
            variable.
        true_consumed (bool): Whether the true branch consumed this variable.
        false_consumed (bool): Whether the false branch consumed this
            variable.

    Returns:
        None
    """
    _merge_branch_element_borrows(merged_handle, true_val, false_val)
    if not (true_consumed or false_consumed):
        return
    value = getattr(merged_handle, "value", None)
    if value is None or not value.type.is_quantum():
        return
    consuming_op = getattr(true_val, "_consumed_by", None) if true_consumed else None
    if consuming_op is None:
        consuming_op = getattr(false_val, "_consumed_by", None)
    # Carry the consuming branch's source location through the merge so a
    # later reuse reports where inside the branch the value was consumed.
    # Each fallback is guarded by that branch's consumed flag so a
    # non-consuming branch's handle can never contribute a stale location.
    consumed_at = getattr(true_val, "_consumed_at", None) if true_consumed else None
    if consumed_at is None and false_consumed:
        consumed_at = getattr(false_val, "_consumed_at", None)
    merged_handle._consumed_at = consumed_at
    # Nested-if case: the branch value was consumed by a deeper merge that
    # already produced a full conditional-consumption phrase (it contains the
    # marker below). Re-wrapping it would duplicate the suffix
    # ("...of the preceding if/else ... of the preceding if/else ..."), so the
    # inner phrase — which already conveys that the value was conditionally
    # consumed — is kept verbatim.
    if consuming_op and _COND_CONSUME_MARKER in consuming_op:
        merged_handle._consumed = True
        merged_handle._consumed_by = consuming_op
        return
    if true_consumed and false_consumed:
        which = "both branches"
    elif true_consumed:
        which = "the true branch"
    else:
        which = "the false branch"
    op_desc = consuming_op if consuming_op else "a measurement/gate"
    # ``_consumed_by`` is rendered inside single quotes by the
    # ``QubitConsumedError`` template, so the phrase is not self-quoted here.
    merged_handle._consumed = True
    merged_handle._consumed_by = (
        f"{op_desc} in {which} {_COND_CONSUME_MARKER} "
        f"(a quantum value consumed on one branch cannot be used after the if)"
    )


def _value_to_ir_value(val: typing.Any, name_prefix: str = "const") -> Value:
    """Convert a Python value or Handle to an IR Value.

    Args:
        val: Python primitive, Handle, or Value to convert
        name_prefix: Prefix for generated value name

    Returns:
        IR Value object

    Raises:
        TypeError: If value type is not supported
    """
    # Already a Value
    if isinstance(val, Value):
        return val

    # Extract Value from Handle
    if hasattr(val, "value") and isinstance(val.value, Value):
        return val.value

    # Convert primitive to Value
    if isinstance(val, (int, float, bool)):
        if isinstance(val, bool):
            return Value(type=BitType(), name=name_prefix).with_const(val)
        elif isinstance(val, float):
            return Value(type=FloatType(), name=name_prefix).with_const(val)
        else:  # int
            return Value(type=UIntType(), name=name_prefix).with_const(val)

    # Unsupported type
    raise TypeError(f"Cannot convert {type(val)} to IR Value")


def _const_int_for_loop_bound(val: typing.Any) -> int | None:
    """Return a concrete loop-bound integer when available.

    Args:
        val: Python integer, ``UInt`` handle, or IR ``Value`` used as a
            ``qmc.range`` bound.

    Returns:
        Concrete ``int`` when ``val`` is statically known, otherwise
        ``None``.
    """
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, UInt):
        const_value = val.value.get_const()
    elif isinstance(val, Value):
        const_value = val.get_const()
    else:
        return None
    if isinstance(const_value, bool):
        return int(const_value)
    if isinstance(const_value, int):
        return const_value
    return None


def should_trace_for_loop(
    start: typing.Any, stop: typing.Any, step: typing.Any
) -> bool:
    """Decide whether a ``qmc.range`` body must be traced.

    The frontend executes loop bodies once to capture a ``ForOperation``.
    When all bounds are concrete and Python's ``range`` would execute
    zero times, tracing the body would incorrectly leak borrow /
    destructive-consume state into the enclosing scope.  Symbolic or
    invalid bounds stay conservative and trace the body so the normal
    compiler validation path reports any errors.

    Args:
        start: Loop start bound.
        stop: Loop stop bound.
        step: Loop step bound.

    Returns:
        ``False`` only for statically-known zero-trip loops; ``True``
        otherwise.
    """
    start_int = _const_int_for_loop_bound(start)
    stop_int = _const_int_for_loop_bound(stop)
    step_int = _const_int_for_loop_bound(step)
    if start_int is None or stop_int is None or step_int is None:
        return True
    try:
        return bool(builtins.range(start_int, stop_int, step_int))
    except ValueError:
        return True


def should_trace_items_loop(mapping: typing.Any) -> bool:
    """Decide whether a ``qmc.items`` body must be traced.

    The frontend executes loop bodies once to capture a
    ``ForItemsOperation``. When the mapping is a ``Dict`` handle whose
    bound contents are compile-time-known and EMPTY, Python's iteration
    would execute zero times, so tracing the body would incorrectly
    leak bindings (and rebind records) into the enclosing scope — the
    ``qmc.range`` zero-trip guard's exact analogue
    (:func:`should_trace_for_loop`). Symbolic or unbound mappings stay
    conservative and trace the body so the normal compiler validation
    path reports any errors.

    Args:
        mapping (typing.Any): The iterated mapping — normally a ``Dict``
            handle; anything without bound dict metadata is treated as
            symbolic.

    Returns:
        bool: ``False`` only for a mapping with present-and-empty bound
        dict contents; ``True`` otherwise.
    """
    value = getattr(mapping, "value", None)
    if value is None:
        return True
    metadata = getattr(value, "metadata", None)
    dict_runtime = getattr(metadata, "dict_runtime", None)
    if dict_runtime is None:
        return True
    return len(dict_runtime.bound_data) > 0


@dataclasses.dataclass
class _PendingRegionArg:
    """Working record for a loop region argument during body tracing.

    Created by :func:`loop_region_enter` when the loop body's
    read-before-write classical scalar candidates are rebound to fresh
    region-argument handles at body entry, completed by
    :func:`record_loop_rebinds` (which fills ``yielded``), and consumed
    by the loop builders, which convert it into an IR ``RegionArg`` and
    publish the post-loop result handle.

    Attributes:
        var_name (str): The carried Python variable name.
        init (Value): The pre-loop IR value entering iteration 0
            (synthesized as a constant for plain Python numbers).
        block_arg (Value): The region-argument value the body reads.
        entry_handle (Handle): The frontend handle wrapping
            ``block_arg`` that the AST-injected entry assignment bound
            into the loop body's frame.
        yielded (Value | None): The body's final IR value for the
            variable, set by ``record_loop_rebinds``. ``None`` means the
            body produced no representable update (identity carry).
        publish_result (bool): Whether the loop builder should publish
            the post-loop result handle for this variable. ``False``
            when the body's final Python value is not representable in
            IR (an opaque object), in which case the Python binding must
            keep the opaque value.
    """

    var_name: str
    init: Value
    block_arg: Value
    entry_handle: Handle
    yielded: Value | None = None
    publish_result: bool = True


def _region_scalar_handle(value: Value, template: Handle | None) -> Handle:
    """Wrap an IR scalar value in the matching frontend handle.

    Args:
        value (Value): The IR value to wrap (``UIntType`` or
            ``FloatType``).
        template (Handle | None): The handle whose family should be
            preserved, or ``None`` to dispatch on ``value.type``.

    Returns:
        Handle: A ``UInt`` or ``Float`` handle wrapping ``value``.

    Raises:
        TypeError: If ``value`` is neither UInt- nor Float-typed.
    """
    if isinstance(template, UInt) or (
        template is None and isinstance(value.type, UIntType)
    ):
        return UInt(value=value, init_value=0)
    if isinstance(template, Float) or (
        template is None and isinstance(value.type, FloatType)
    ):
        return Float(value=value, init_value=0.0)
    raise TypeError(
        f"Loop region arguments support UInt / Float scalars; got value type "
        f"{value.type!r}"
    )


def loop_region_enter(
    snapshot: dict[str, typing.Any],
    name: str,
) -> typing.Any:
    """Bind a loop-carried classical scalar to a fresh region argument.

    Called from AST-injected code at the top of a ``for`` /
    ``for-items`` loop body (immediately after ``loop_rebind_snapshot``)
    for each read-before-write classical candidate: ``total =
    loop_region_enter(_qm_rebind_snap_N, "total")``. When the pre-loop
    binding is a classical ``UInt`` / ``Float`` scalar (or a plain
    Python ``int`` / ``float``), the body is given a fresh
    region-argument handle so its reads become explicit loop-carried
    reads instead of stale pre-loop reads — the MLIR ``iter_args``
    model. The loop builder later converts the pending entry into a
    ``RegionArg`` on the loop operation.

    Non-scalar and quantum bindings (arrays, dicts, ``Qubit``, ``Bit``,
    opaque Python objects, ``bool``) are returned unchanged, so those
    shapes keep their existing tracing behavior — quantum rebinds keep
    feeding the discard check, and measurement-backed ``Bit`` carries
    keep their targeted rejection.

    Args:
        snapshot (dict[str, typing.Any]): Pre-loop-body bindings from
            ``loop_rebind_snapshot``.
        name (str): The candidate variable name.

    Returns:
        typing.Any: A fresh region-argument handle for supported scalar
            bindings, or the original binding unchanged.

    Raises:
        NameError: If ``name`` resolves nowhere — mirroring the
            ``NameError`` the body's first read would have raised.
    """
    if name not in snapshot:
        raise NameError(f"name '{name}' is not defined")
    resolved = snapshot[name]
    tracer = get_current_tracer()

    init_value: Value | None = None
    template: Handle | None = None
    if isinstance(resolved, (UInt, Float)):
        candidate = resolved.value
        if isinstance(candidate, Value) and not candidate.type.is_quantum():
            init_value = candidate
            template = resolved
    elif not isinstance(resolved, bool) and isinstance(resolved, (int, float)):
        init_value = _value_to_ir_value(resolved, name)
    if init_value is None:
        return resolved

    block_arg = Value(type=init_value.type, name=name)
    entry_handle = _region_scalar_handle(block_arg, template)
    tracer.region_entries[name] = _PendingRegionArg(
        var_name=name,
        init=init_value,
        block_arg=block_arg,
        entry_handle=entry_handle,
    )
    return entry_handle


def loop_region_result(name: str, current: typing.Any) -> typing.Any:
    """Rebind a loop-carried variable to its post-loop result handle.

    Called from AST-injected code immediately after a ``for`` /
    ``for-items`` loop's ``with`` block: ``total =
    loop_region_result("total", total)``. Consumes the result handle
    the loop builder published for ``name`` (if any) so post-loop reads
    reference the loop operation's result value instead of the body's
    final yielded value.

    Args:
        name (str): The carried variable name.
        current (typing.Any): The variable's current binding (the body's
            final handle), returned unchanged when the closed loop
            published no result for ``name``.

    Returns:
        typing.Any: The published result handle, or ``current``.
    """
    tracer = get_current_tracer()
    return tracer.loop_region_results.pop(name, current)


def _close_region_entries(
    body_tracer: Tracer,
    parent_tracer: Tracer,
) -> tuple[tuple[RegionArg, ...], list[Value]]:
    """Convert pending region entries into IR ``RegionArg`` records.

    Called by the ``for`` / ``for-items`` loop builders after the body
    trace completes. For each pending entry, synthesizes the loop-result
    ``Value``, builds the ``RegionArg``, and publishes the post-loop
    result handle on ``parent_tracer.loop_region_results`` (replacing
    any leftovers from a previously closed loop) for the AST-injected
    ``loop_region_result`` assignments to consume.

    Args:
        body_tracer (Tracer): The tracer that captured the loop body
            (carries ``region_entries``).
        parent_tracer (Tracer): The tracer the loop operation is being
            appended to (receives ``loop_region_results``).

    Returns:
        tuple[tuple[RegionArg, ...], list[Value]]: The region-argument
            records for the loop operation and the result values to
            append to its ``results`` list, in entry order.
    """
    region_args: list[RegionArg] = []
    results: list[Value] = []
    result_handles: dict[str, typing.Any] = {}
    for entry in body_tracer.region_entries.values():
        yielded = entry.yielded if entry.yielded is not None else entry.block_arg
        result = Value(type=entry.block_arg.type, name=entry.var_name)
        region_args.append(
            RegionArg(
                var_name=entry.var_name,
                init=entry.init,
                block_arg=entry.block_arg,
                yielded=yielded,
                result=result,
            )
        )
        results.append(result)
        if entry.publish_result:
            result_handles[entry.var_name] = _region_scalar_handle(
                result, entry.entry_handle
            )
    parent_tracer.loop_region_results = result_handles
    return tuple(region_args), results


def loop_rebind_snapshot(
    frame_locals: dict[str, typing.Any],
    names: tuple[str, ...],
) -> dict[str, typing.Any]:
    """Snapshot pre-loop variable handles for rebind detection.

    Called from AST-injected probe code as the first statement of a traced
    loop body. The snapshot records which handle each candidate variable
    name pointed at before the body ran, so ``record_loop_rebinds`` can
    detect rebinds by comparing IR value identity afterwards.

    Candidates are resolved through :func:`branch_rebind_pre_bindings`:
    the caller's frame locals first, then the enclosing if-branch
    pre-binding stack. The fallback matters inside an if branch — a
    variable the branch only *stores* (via the loop) is not a branch
    input parameter, so it is unbound in the branch's frame at loop
    entry, yet its pre-branch handle is exactly the state a loop-body
    rebind would discard. Names bound nowhere are silently omitted.

    Args:
        frame_locals (dict[str, typing.Any]): The caller's ``locals()``
            at loop entry.
        names (tuple[str, ...]): Candidate variable names to snapshot.

    Returns:
        dict[str, typing.Any]: The resolvable candidate names mapped to
            their pre-loop-body handles (or plain Python values).
    """
    return branch_rebind_pre_bindings(frame_locals, names)


def record_loop_rebinds(
    snapshot: dict[str, typing.Any],
    frame_locals: dict[str, typing.Any],
    names: tuple[str, ...],
    classical_names: tuple[str, ...],
) -> None:
    """Record classical and quantum rebinds on the current loop-body tracer.

    Called from AST-injected probe code as the last statement of a traced
    loop body. Two families of rebinds are recorded as
    :class:`LoopCarriedRebind` entries on the active body tracer (the
    loop builders copy them onto the loop operation, where the
    transpiler's rejection passes read them):

    - **Quantum** (any candidate name): the variable's pre-body value is
      quantum and its post-body value carries a different ``logical_id``
      — a substitution to a different quantum resource (fresh allocation
      or another register) rather than a gate self-update, which keeps
      the logical_id. These feed the transpiler's loop-body quantum
      discard check.
    - **Classical scalar** (only names in ``classical_names``, the
      read-before-write candidates): the post-body value is a classical
      scalar with a different IR identity — the ``total = total + i``
      shape. Candidates that were region-bound at body entry
      (``loop_region_enter``) complete their pending ``RegionArg``
      instead of producing a record — the carry is then explicit and
      supported. A record is created only for the shapes region binding
      declined (``while`` bodies, measurement-backed ``Bit`` carries),
      which the transpiler still rejects. Store-only classical
      reassignments (a value recomputed from scratch each iteration)
      are deliberately not recorded; they trace correctly.

    No IR operations are emitted; this only annotates the tracer.

    Args:
        snapshot (dict[str, typing.Any]): Pre-loop-body handles from
            ``loop_rebind_snapshot``.
        frame_locals (dict[str, typing.Any]): The caller's ``locals()``
            at the end of the loop body.
        names (tuple[str, ...]): All candidate variable names.
        classical_names (tuple[str, ...]): The subset of ``names`` the
            loop body reads before writing; only these may produce
            classical records.
    """
    post_bindings = branch_rebind_pre_bindings(frame_locals, names)
    classical_candidates = set(classical_names)
    tracer = get_current_tracer()
    records: list[LoopCarriedRebind] = []
    for name in names:
        entry = tracer.region_entries.get(name)
        if entry is not None:
            # Region-bound classical scalar: fill in the yielded value
            # instead of recording a rebind — the carried update is now
            # an explicit, supported RegionArg, not a staleness hazard.
            if name not in post_bindings:
                continue
            after = post_bindings[name]
            if after is entry.entry_handle:
                # Never actually stored on the traced path: identity
                # carry (yielded defaults to the block argument).
                continue
            after_value = _ir_value_from_handle_like(after)
            if after_value is None and isinstance(after, (bool, int, float)):
                after_value = _value_to_ir_value(after, name)
            if (
                after_value is None
                or isinstance(after_value, ArrayValue)
                or after_value.type.is_quantum()
            ):
                # The body's final Python value is not representable as
                # a carried classical scalar (opaque object / array /
                # quantum). Keep the identity carry for the body's
                # block-argument reads, but let the Python binding keep
                # the body's final value after the loop.
                entry.publish_result = False
                continue
            entry.yielded = after_value
            continue
        if name not in snapshot or name not in post_bindings:
            continue
        before = snapshot[name]
        after = post_bindings[name]
        if before is after:
            continue
        before_value = _ir_value_from_handle_like(before)
        after_value = _ir_value_from_handle_like(after)
        if before_value is not None and before_value.type.is_quantum():
            # Quantum rebind: compare logical_id, not uuid — a gate
            # self-update keeps the wire identity and is not a rebind.
            # A post-body handle that is no IR value at all (a plain
            # Python constant, ``None``, or an opaque classical call
            # result — the shape the decoration-time analyzer forbids at
            # top level) drops the wire just the same: synthesize a
            # classical placeholder so the record exists for the
            # transpiler's discard check (mirroring the if-branch
            # records).
            if after_value is None:
                if isinstance(after, (bool, int, float)):
                    after_value = _value_to_ir_value(after, name)
                else:
                    after_value = Value(type=UIntType(), name=name)
                records.append(
                    LoopCarriedRebind(
                        var_name=name,
                        before=before_value,
                        after=after_value,
                    )
                )
                continue
            if after_value.logical_id != before_value.logical_id:
                records.append(
                    LoopCarriedRebind(
                        var_name=name,
                        before=before_value,
                        after=after_value,
                    )
                )
            continue
        if name not in classical_candidates:
            continue
        if after_value is None or isinstance(after_value, ArrayValue):
            continue
        if after_value.type.is_quantum():
            continue
        before_synthesized = False
        if before_value is None:
            if isinstance(before, (bool, int, float)):
                before_value = _value_to_ir_value(before, name)
                before_synthesized = True
            else:
                continue
        if isinstance(before_value, ArrayValue) or before_value.type.is_quantum():
            continue
        if before_value.uuid == after_value.uuid:
            continue
        records.append(
            LoopCarriedRebind(
                var_name=name,
                before=before_value,
                after=after_value,
                before_synthesized=before_synthesized,
            )
        )
    if records:
        tracer.loop_carried_rebinds = tracer.loop_carried_rebinds + tuple(records)


def _create_merge_for_values(
    true_val: typing.Any,
    false_val: typing.Any,
    if_operation: IfOperation,
) -> Handle:
    """Create a branch-merge slot for a pair of branch values.

    Args:
        true_val (typing.Any): The true branch's value. Must be a
            frontend ``Handle`` at runtime — its family determines how
            the merge output is wrapped (via ``_wrap_merge_result``); any
            other object is rejected.
        false_val (typing.Any): The false branch's value (Handle, Value,
            or primitive); only its IR value participates in the merge.
        if_operation (IfOperation): The if-else the merge is added to via
            ``add_merge``. Its condition operand must already be attached.

    Returns:
        Handle: The frontend handle wrapping the merged IR output value.
            The merge output value itself is already registered on
            ``if_operation`` via ``add_merge`` and is reachable as
            ``merged_handle.value``.

    Raises:
        TypeError: If the branch value types differ, if ``true_val`` is
            not a frontend ``Handle``, or if its handle family does not
            support merging.
    """
    # Convert both values to IR Values
    true_v = _value_to_ir_value(true_val, "true_const")
    false_v = _value_to_ir_value(false_val, "false_const")

    # Type mismatch check
    if true_v.type != false_v.type:
        raise TypeError(
            f"Type mismatch in if-else branches: "
            f"true branch has {true_v.type}, false branch has {false_v.type}"
        )

    # Create merge output value (indexed to avoid name collisions)
    merge_index = len(if_operation.results)
    if isinstance(true_v, ArrayValue):
        merge_output = ArrayValue(
            type=true_v.type,
            name=f"{true_v.name}_merge_{merge_index}",
            shape=true_v.shape,
            slice_of=true_v.slice_of,
            slice_start=true_v.slice_start,
            slice_step=true_v.slice_step,
        )
    else:
        merge_output = Value(
            type=true_v.type, name=f"{true_v.name}_merge_{merge_index}"
        )

    # Wrap the merge output in the true-branch handle's family. The wrap
    # may rebuild the output value with copied metadata (QFixed carriers),
    # so the handle's value — not the bare merge_output — is what the merge
    # must record.
    if not isinstance(true_val, Handle):
        raise TypeError(
            "Unsupported Handle type for if-else merge: "
            f"{type(true_val).__name__}. Add explicit handle wrapping support "
            "before merging this handle type."
        )
    merged_handle = true_val._wrap_merge_result(merge_output, false_v)

    # Store the merge in the IfOperation through its official accessor
    if_operation.add_merge(true_v, false_v, merged_handle.value)
    _refresh_slice_merge_owner(true_val, false_val, merged_handle)

    return merged_handle


def _slice_view_coverage(view: typing.Any) -> tuple[int, ...] | None:
    """Return concrete root coverage for a slice view when available.

    Args:
        view (typing.Any): Candidate ``VectorView`` handle.

    Returns:
        tuple[int, ...] | None: Concrete root-space slots covered by the view,
        or ``None`` when ``view`` is not a slice view or its bounds are still
        symbolic.
    """
    from qamomile.circuit.frontend.handle.array import (
        VectorView,
        _coverage_from_array_value,
    )

    if not isinstance(view, VectorView):
        return None
    covered = view._slice_covered_indices
    if covered is not None:
        return covered
    return _coverage_from_array_value(view.value)


def _refresh_slice_merge_owner(
    true_val: typing.Any,
    false_val: typing.Any,
    merged_handle: typing.Any,
) -> None:
    """Transfer same-slice branch ownership to a merged slice handle.

    Runtime ``if`` tracing uses branch-local handle copies.  When both branches
    return the same parent slice lineage, the merged view is the handle that
    should be assignable after the ``if``.  Refreshing the parent borrow table
    here keeps the frontend owner identity aligned with that merged handle.

    Args:
        true_val (typing.Any): True-branch value participating in the merge.
        false_val (typing.Any): False-branch value participating in the merge.
        merged_handle (typing.Any): Handle wrapping the merge result.
    """
    from qamomile.circuit.frontend.handle.array import VectorView

    if not (
        isinstance(true_val, VectorView)
        and isinstance(false_val, VectorView)
        and isinstance(merged_handle, VectorView)
    ):
        return
    if true_val._slice_parent is not false_val._slice_parent:
        return

    coverage = _slice_view_coverage(merged_handle)
    if (
        coverage is None
        or _slice_view_coverage(true_val) != coverage
        or _slice_view_coverage(false_val) != coverage
    ):
        return

    parent = true_val._slice_parent
    for idx in coverage:
        key = (f"const:{idx}",)
        owner = parent._borrowed_indices.get(key)
        if isinstance(owner, VectorView) and _slice_view_coverage(owner) == coverage:
            parent._borrowed_indices[key] = merged_handle


def _trace_branch(
    branch_func: typing.Callable,
    variables: list,
) -> typing.Tuple[Tracer, tuple]:
    """Trace a conditional branch and return its tracer and results.

    Args:
        branch_func: Function to execute for this branch
        variables: List of variables passed to the function

    Returns:
        Tuple of (tracer, normalized_result_tuple)
    """
    tracer = Tracer()
    with trace(tracer):
        result = branch_func(*variables)

    # Normalize result to tuple
    if not isinstance(result, tuple):
        result = (result,) if result is not None else ()

    return tracer, result


def _operation_touches_array_element(op: typing.Any, array_value: ArrayValue) -> bool:
    """Return whether an operation touches an element of an array.

    Element-level updates such as ``qs[0] = qmc.x(qs[0])`` do not replace the
    outer ``ArrayValue`` handle, but the gate operands/results still carry
    ``parent_array`` metadata.  ``emit_if`` uses this to avoid eliding an array
    merge merely because both branch handles still point at the same outer array.

    Args:
        op (typing.Any): Operation-like object to inspect.  It may be a normal
            IR operation or a control-flow operation with nested operation lists.
        array_value (ArrayValue): Array whose element-level usage is being
            detected.

    Returns:
        bool: ``True`` if ``op`` or any nested operation reads/writes an element
        whose ``parent_array`` is ``array_value``; ``False`` otherwise.
    """
    # ``all_input_values`` (falling back to ``operands`` for op-like test
    # doubles) keeps subclass-extra references — a nested if's merge
    # yields in particular — counting as touches, matching the
    # conservative reach this prover had when merges were stored as
    # nested merge pseudo-operations.
    if hasattr(op, "all_input_values"):
        inputs = op.all_input_values()
    else:
        inputs = getattr(op, "operands", ())
    values = [*inputs, *getattr(op, "results", ())]
    for value in values:
        parent_array = getattr(value, "parent_array", None)
        if (
            isinstance(parent_array, ArrayValue)
            and parent_array.uuid == array_value.uuid
        ):
            return True

    nested_op_lists = getattr(op, "nested_op_lists", None)
    if nested_op_lists is None:
        return False
    return any(
        _operation_touches_array_element(nested_op, array_value)
        for nested_ops in nested_op_lists()
        for nested_op in nested_ops
    )


def _branch_touches_array_elements(
    tracer: Tracer,
    array_value: ArrayValue,
) -> bool:
    """Return whether a traced branch touches elements of an array.

    Args:
        tracer (Tracer): Branch tracer whose captured operations are inspected.
        array_value (ArrayValue): Array whose element-level usage is being
            detected.

    Returns:
        bool: ``True`` if any captured operation touches an element of
        ``array_value``; ``False`` otherwise.
    """
    return any(
        _operation_touches_array_element(op, array_value) for op in tracer.operations
    )


def _ir_value_from_handle_like(value: typing.Any) -> Value | None:
    """Extract the IR value carried by a handle-like object.

    Args:
        value (typing.Any): Candidate handle or IR value.

    Returns:
        Value | None: The contained IR ``Value`` when available, otherwise
        ``None``.
    """
    if isinstance(value, Value):
        return value
    ir_value = getattr(value, "value", None)
    if isinstance(ir_value, Value):
        return ir_value
    return None


def _find_original_handle_for_result(
    result: typing.Any,
    variables: list,
) -> typing.Any | None:
    """Find the input handle that still owns a pass-through result value.

    ``visit_If`` may return values that were not passed as inputs, such as new
    locals defined in both branches.  For array no-op merge elision we still want
    to reuse the original outer handle when the result is merely a branch copy
    of an existing input array, because that original handle carries live borrow
    state.  Matching by UUID keeps this independent of the differing input and
    output variable orders.

    Args:
        result (typing.Any): Branch result being merged.
        variables (list): Handles passed into ``emit_if`` as branch inputs.

    Returns:
        typing.Any | None: The matching original input handle when one exists,
        otherwise ``None``.
    """
    result_value = _ir_value_from_handle_like(result)
    if result_value is None:
        return None
    for variable in variables:
        variable_value = _ir_value_from_handle_like(variable)
        if variable_value is not None and variable_value.uuid == result_value.uuid:
            return variable
    return None


def _can_elide_scalar_merge(true_val: typing.Any, false_val: typing.Any) -> bool:
    """Decide whether a scalar branch-merge slot can be omitted at trace time.

    The frontend skips a merge only when the trace PROVES it would be a
    no-op: both branches returned the *same* IR ``Value`` object and
    neither branch consumed it. Skipping keeps the IR SSA-minimal and
    avoids generating merge-versioned aliases (e.g. ``j_merge_4``) for
    read-only scalar loop variables, which downstream emit-time loop
    unrolling cannot bind.

    Invariant (the belt-and-braces contract with emit): elision is an
    optimization, never a correctness requirement. Every merge that is
    NOT elided — including identity merges this prover cannot see, such
    as consumed-in-branch values or metadata-divergent handles — must be
    handled first-class by emit (an identity merge trivially resolves to
    one physical resource). Emit must never assume the frontend elided
    every identity merge.

    Args:
        true_val (typing.Any): Value the true branch returned for this
            variable slot (Handle, Value, or primitive).
        false_val (typing.Any): Value the false branch returned for the
            same slot.

    Returns:
        bool: ``True`` when the merge slot is provably a no-op and may
            be skipped.
    """
    from qamomile.circuit.frontend.handle.array import ArrayBase

    if isinstance(true_val, ArrayBase) or isinstance(false_val, ArrayBase):
        return False
    true_v = true_val.value if hasattr(true_val, "value") else true_val
    false_v = false_val.value if hasattr(false_val, "value") else false_val
    return (
        isinstance(true_v, Value)
        and isinstance(false_v, Value)
        and true_v is false_v
        and not getattr(true_val, "_consumed", False)
        and not getattr(false_val, "_consumed", False)
    )


def _can_elide_array_merge(
    true_val: typing.Any,
    false_val: typing.Any,
    true_tracer: Tracer,
    false_tracer: Tracer,
) -> bool:
    """Decide whether an array branch-merge slot can be omitted at trace time.

    Array handles normally need a merge because whole-array ops can
    return a fresh ``ArrayValue``. When both branches still point at the
    exact same quantum ``ArrayValue``, neither branch consumed it, and
    neither branch touched any of its elements, the array handle
    provably did not change; the caller then reuses the original outer
    handle so parent borrow tables (for live slice views captured
    outside the branch) are not replaced by a merge-root with empty
    state.

    The same invariant as :func:`_can_elide_scalar_merge` applies:
    elision is an optimization only, and emit handles every non-elided
    merge — identity merges included — first-class.

    Args:
        true_val (typing.Any): Value the true branch returned for this
            variable slot.
        false_val (typing.Any): Value the false branch returned for the
            same slot.
        true_tracer (Tracer): Tracer holding the true branch's captured
            operations, scanned for element-level touches.
        false_tracer (Tracer): Tracer holding the false branch's
            captured operations.

    Returns:
        bool: ``True`` when the merge slot is provably a no-op and may
            be skipped.
    """
    from qamomile.circuit.frontend.handle.array import ArrayBase

    if not (isinstance(true_val, ArrayBase) or isinstance(false_val, ArrayBase)):
        return False
    true_v = true_val.value if hasattr(true_val, "value") else true_val
    false_v = false_val.value if hasattr(false_val, "value") else false_val
    return (
        isinstance(true_v, ArrayValue)
        and isinstance(false_v, ArrayValue)
        and true_v.type.is_quantum()
        and true_v is false_v
        and not getattr(true_val, "_consumed", False)
        and not getattr(false_val, "_consumed", False)
        and not _branch_touches_array_elements(true_tracer, true_v)
        and not _branch_touches_array_elements(false_tracer, false_v)
    )


# Sentinel returned by ``dead_rebind_binding`` when a probed name is not
# bound in the branch body (the branch did not store it and it is not an
# input parameter). Compared by identity in ``_collect_branch_rebinds``.
_DEAD_REBIND_UNBOUND: typing.Any = object()


def dead_rebind_binding(
    frame_locals: dict[str, typing.Any],
    name: str,
) -> typing.Any:
    """Probe a branch body's post-branch binding of a dead-after variable.

    Called from AST-injected code as an extra element of a branch body's
    return tuple. A variable reassigned in a branch but never read after
    the if is dead-store-eliminated from the branch outputs, so its
    post-branch binding is not otherwise observable by ``emit_if``; this
    probe reads it from the body's ``locals()``. In the branch that does
    not store the variable the name is unbound in the body scope, so the
    probe returns a sentinel instead of raising ``NameError``.

    Args:
        frame_locals (dict[str, typing.Any]): The body's ``locals()`` at
            the return point.
        name (str): The probed variable name.

    Returns:
        typing.Any: The post-branch handle, or the unbound sentinel when
            the body never bound the name.
    """
    return frame_locals.get(name, _DEAD_REBIND_UNBOUND)


# Stack of the pre-binding maps of the emit_if calls currently tracing
# their branches. A nested emit_if call site lives inside a generated
# branch-body function whose scope only contains that if's inputs, so a
# dead-after variable defined further out is not in the call site's
# ``locals()``; the enclosing emit_if's captured pre-bindings supply it.
# Context-local (like the tracer's ``_current_tracer``) so concurrent
# traces in different threads / async tasks cannot leak bindings into
# each other; stored as an immutable tuple and updated via set/reset
# tokens in ``emit_if``.
_ACTIVE_REBIND_PRE_BINDINGS: contextvars.ContextVar[
    tuple[dict[str, typing.Any], ...]
] = contextvars.ContextVar("qamomile_active_rebind_pre_bindings", default=())


def branch_rebind_pre_bindings(
    frame_locals: dict[str, typing.Any],
    names: tuple,
) -> dict[str, typing.Any]:
    """Capture pre-branch bindings for if-rebind records.

    Called from AST-injected code at the ``emit_if`` call site with the
    caller's ``locals()``. A name missing from the call site's locals is
    resolved through the enclosing emit_if calls' captured pre-bindings
    (innermost first): a dead-after variable never enters the generated
    branch-body scopes, so for a nested if only the enclosing capture
    still knows its pre-branch handle. The transformer's candidate
    analysis is lexical, so a name may genuinely be unbound everywhere
    (a preceding pure-store if can be dead-store-eliminated from its
    outputs); such names are silently skipped instead of raising
    ``UnboundLocalError`` at the call site.

    Args:
        frame_locals (dict[str, typing.Any]): The caller's ``locals()``.
        names (tuple): Candidate variable names to capture.

    Returns:
        dict[str, typing.Any]: The resolvable candidate names mapped to
            their pre-branch handles.
    """
    captured: dict[str, typing.Any] = {}
    for name in names:
        if name in frame_locals:
            captured[name] = frame_locals[name]
            continue
        for enclosing in reversed(_ACTIVE_REBIND_PRE_BINDINGS.get()):
            if name in enclosing:
                captured[name] = enclosing[name]
                break
    return captured


def _rebound_from(pre_value: Value, post_handle: typing.Any) -> bool:
    """Decide whether a post-branch binding rebinds away from a pre value.

    Compares ``logical_id``, not ``uuid``: a gate self-update
    (``q = qmc.x(q)``) produces a new SSA version with a fresh uuid but
    the SAME logical_id (same quantum wire), which is not a discard. A
    substitution to a different quantum resource — a fresh allocation or
    another existing register — changes the logical_id, and a post-branch
    binding that is no IR value at all (a plain Python constant, ``None``,
    or an opaque classical call result — the shape the decoration-time
    analyzer forbids at top level as an unknown-call overwrite) drops the
    wire just the same; both are recorded.

    Args:
        pre_value (Value): The variable's IR value at branch entry.
        post_handle (typing.Any): The variable's post-branch handle (or
            raw value, or the unbound sentinel for dead-rebind probes of
            a branch that never bound the name).

    Returns:
        bool: True when the branch left the variable bound to a
            different quantum wire or to a non-IR value.
    """
    if post_handle is _DEAD_REBIND_UNBOUND:
        return False
    post_value = post_handle.value if hasattr(post_handle, "value") else post_handle
    if not isinstance(post_value, Value):
        return True
    return post_value.logical_id != pre_value.logical_id


def _collect_branch_rebinds(
    output_names: tuple,
    rebind_pre_bindings: dict | None,
    true_result: tuple,
    false_result: tuple,
    dead_names: tuple,
    dead_true: tuple,
    dead_false: tuple,
) -> tuple[BranchRebind, ...]:
    """Record quantum variables whose binding changed in an if branch.

    Compares each candidate variable's pre-branch IR value (captured at
    the ``emit_if`` call site by the AST transformer) with the value the
    variable holds after each branch. The merge only carries the
    *new* branch values — and a reassigned variable whose old value is
    dead is not even passed into the branches — so the pre-branch value
    can disappear from the ``IfOperation`` entirely; these records
    preserve it for the transpiler's branch-discard check. Live
    candidates are matched positionally through ``output_names``;
    dead-after candidates (dead-store-eliminated from the outputs) are
    matched through the probe tails the branch bodies append after their
    ordinary return values. Classical variables are not recorded —
    classical rebinds are ordinary merged dataflow.

    Args:
        output_names (tuple): Variable names positionally aligned with
            the merged branch-result prefixes (the AST transformer's
            output list).
        rebind_pre_bindings (dict | None): Candidate variable names (live
            and dead) mapped to their pre-branch handles, captured at the
            call site. None or empty when no pre-existing variable is
            reassigned in a branch.
        true_result (tuple): Merged-output values returned by the true
            branch, positionally aligned with ``output_names``.
        false_result (tuple): Merged-output values returned by the false
            branch, positionally aligned with ``output_names``.
        dead_names (tuple): Dead-after candidate names, positionally
            aligned with the probe tails.
        dead_true (tuple): The true branch's probe tail (post-branch
            handles or unbound sentinels).
        dead_false (tuple): The false branch's probe tail.

    Returns:
        tuple[BranchRebind, ...]: One record per quantum variable whose
            binding changed in at least one branch. Empty when there are
            no rebind candidates at all.

    Raises:
        AssertionError: If the branch result tuples do not align
            positionally with the name lists. This is an internal
            invariant the AST transformer guarantees; raising loudly
            keeps a future alignment regression from silently disabling
            the branch-discard check (whose sole input is these records).
    """
    if not rebind_pre_bindings:
        return ()
    # These records are the branch-discard check's SOLE input, so a
    # silent empty return on misalignment would fail open — it would
    # disable discard detection and let a rejectable kernel compile
    # silently. The AST transformer builds the branch return tuples and
    # ``emit_if`` slices them to match ``output_names`` / ``dead_names``
    # by construction, so a mismatch is an internal invariant break, not
    # a legitimate input; fail loud so a future change that breaks the
    # alignment is caught in tests rather than reopening the hole.
    if not (len(output_names) == len(true_result) == len(false_result)):
        raise AssertionError(
            "Branch output alignment broken in _collect_branch_rebinds: "
            "expected len(output_names) == len(true_result) == "
            f"len(false_result), got {len(output_names)} / "
            f"{len(true_result)} / {len(false_result)}."
        )
    if not (len(dead_names) == len(dead_true) == len(dead_false)):
        raise AssertionError(
            "Dead-probe alignment broken in _collect_branch_rebinds: "
            "expected len(dead_names) == len(dead_true) == len(dead_false), "
            f"got {len(dead_names)} / {len(dead_true)} / {len(dead_false)}."
        )
    positions = {name: index for index, name in enumerate(output_names)}
    dead_positions = {name: index for index, name in enumerate(dead_names)}
    records: list[BranchRebind] = []
    for name, pre_handle in rebind_pre_bindings.items():
        pre_value = pre_handle.value if hasattr(pre_handle, "value") else pre_handle
        if not isinstance(pre_value, Value) or not pre_value.type.is_quantum():
            continue
        index = positions.get(name)
        dead_index = dead_positions.get(name)
        if index is not None:
            post_true: typing.Any = true_result[index]
            post_false: typing.Any = false_result[index]
        elif dead_index is not None:
            post_true = dead_true[dead_index]
            post_false = dead_false[dead_index]
        else:
            continue
        rebound_in_true = _rebound_from(pre_value, post_true)
        rebound_in_false = _rebound_from(pre_value, post_false)
        if rebound_in_true or rebound_in_false:
            records.append(
                BranchRebind(
                    var_name=name,
                    before=pre_value,
                    rebound_in_true=rebound_in_true,
                    rebound_in_false=rebound_in_false,
                )
            )
    return tuple(records)


def emit_if(
    cond_func: typing.Callable,
    true_func: typing.Callable,
    false_func: typing.Callable,
    variables: list,
    output_names: tuple = (),
    rebind_pre_bindings: dict | None = None,
    dead_names: tuple = (),
) -> typing.Any:
    """Builder function for if-else conditional with merge function merging.

    This function is called from AST-transformed code. The AST transformer
    converts:
        if condition:
            true_body
        else:
            false_body

    Into:
        def _cond_N(vars): return condition
        def _body_N(vars): true_body; return vars
        def _body_N+1(vars): false_body; return vars
        result = emit_if(_cond_N, _body_N, _body_N+1, [var_list])

    Args:
        cond_func: Function returning the condition (Bit or bool-like Handle)
        true_func: Function executing true branch, returns updated variables
        false_func: Function executing false branch, returns updated variables
        variables: List of variables used in the branches
        output_names (tuple): Variable names positionally aligned with the
            branch return tuples, used for branch-rebind records. Empty
            when the transformer found no rebind candidates.
        rebind_pre_bindings (dict | None): Pre-branch handles of every
            pre-existing variable reassigned in a branch, keyed by name;
            captured at the call site by the AST transformer. None when
            there are no candidates.
        dead_names (tuple): Names of dead-after rebind candidates whose
            post-branch bindings the branch bodies append as a probe tail
            after their ordinary return values (see
            ``dead_rebind_binding``). The tail is consumed for rebind
            records only and never merged or returned. Empty when there
            are no dead candidates.

    Returns:
        Merged variable values after conditional execution (using merge functions)

    Example:
        ```python
        @qkernel
        def my_kernel(q: Qubit) -> Qubit:
            result = measure(q)
            if result:
                q = z(q)
            return q
        ```
    """
    parent_tracer = get_current_tracer()

    # 1. Evaluate condition using the ORIGINAL variables (before copying).
    #    The AST transformer guarantees that the condition function only
    #    produces comparison operations and never applies quantum gates,
    #    so it is safe to pass the original (unconsumed) handles here.
    condition_result = cond_func(*variables)
    condition_value = (
        condition_result.value
        if hasattr(condition_result, "value")
        else condition_result
    )

    # 2. Trace both branches (fresh copies avoid consumed conflicts)
    true_vars = [_fresh_handle_copy_for_tracing(v) for v in variables]
    false_vars = [_fresh_handle_copy_for_tracing(v) for v in variables]
    # Expose this call's captured pre-bindings to nested emit_if call
    # sites while the branch bodies trace (see
    # ``branch_rebind_pre_bindings``).
    stack_token = _ACTIVE_REBIND_PRE_BINDINGS.set(
        _ACTIVE_REBIND_PRE_BINDINGS.get() + (dict(rebind_pre_bindings or {}),)
    )
    try:
        true_tracer, true_result = _trace_branch(true_func, true_vars)
        false_tracer, false_result = _trace_branch(false_func, false_vars)
    finally:
        _ACTIVE_REBIND_PRE_BINDINGS.reset(stack_token)

    # 3. Create IfOperation
    if_op = IfOperation(
        true_operations=true_tracer.operations,
        false_operations=false_tracer.operations,
    )
    if_op.operands.append(condition_value)

    # 4. Create merge functions for each variable to merge branches
    # Note: The AST transformer guarantees both branches return the same
    # variable list in the same order, so true_val and false_val always
    # have the same type.
    if len(true_result) != len(false_result):
        raise ValueError(
            f"Branch result length mismatch: true={len(true_result)}, false={len(false_result)}"
        )
    # Split off the dead-rebind probe tail (never merged or returned).
    n_merged = len(true_result) - len(dead_names)
    if n_merged < 0:
        raise ValueError(
            f"Branch result shorter than dead-rebind probe tail: "
            f"len(result)={len(true_result)}, len(dead_names)={len(dead_names)}"
        )
    if_op.branch_rebinds = _collect_branch_rebinds(
        output_names,
        rebind_pre_bindings,
        true_result[:n_merged],
        false_result[:n_merged],
        dead_names,
        true_result[n_merged:],
        false_result[n_merged:],
    )
    true_result = true_result[:n_merged]
    false_result = false_result[:n_merged]
    merged_results = []
    for true_val, false_val in zip(true_result, false_result, strict=True):
        if isinstance(true_val, (Handle, Value)):
            if not isinstance(false_val, (Handle, Value)):
                raise TypeError(
                    f"Branch value mismatch in merge: "
                    f"true branch returned {type(true_val).__name__}, "
                    f"but false branch returned {type(false_val).__name__}. "
                    f"Both branches of an if-else must return the same variables."
                )
            # Only in-branch consumption counts: a handle consumed BEFORE the
            # if (``_consumed_pre_branch``) must not block elision or trigger
            # the conditional-move rule — its reuse inside a branch already
            # raised at trace time (see ``_fresh_handle_copy_for_tracing``).
            true_consumed = _consumed_in_branch(true_val)
            false_consumed = _consumed_in_branch(false_val)
            # Merge minimization: skip slots the trace proves are no-ops
            # (see the elision predicates for the exact conditions and
            # the belt-and-braces invariant they share with emit).
            if _can_elide_scalar_merge(true_val, false_val):
                merged_results.append(true_val)
                continue
            if _can_elide_array_merge(true_val, false_val, true_tracer, false_tracer):
                original_val = _find_original_handle_for_result(true_val, variables)
                merged_results.append(
                    original_val if original_val is not None else true_val
                )
                continue
            merged_handle = _create_merge_for_values(true_val, false_val, if_op)
            # Conditional-move rule: if the value was consumed on either
            # branch, the merged handle is consumed after the if. Reusing it
            # then raises at trace time instead of carrying a consumed value
            # forward through the branch merge.
            _mark_conditionally_consumed(
                merged_handle, true_val, false_val, true_consumed, false_consumed
            )
            merged_results.append(merged_handle)
        elif isinstance(false_val, (Handle, Value)):
            raise TypeError(
                f"Branch value mismatch in merge: "
                f"false branch returned {type(false_val).__name__}, "
                f"but true branch returned {type(true_val).__name__}. "
                f"Both branches of an if-else must return the same variables."
            )
        else:
            # Non-Handle/Value values (int, float, etc.) don't need merge
            merged_results.append(true_val)

    # 5. Add IfOperation to parent tracer
    parent_tracer.add_operation(if_op)

    # 6. Return merged results
    if len(merged_results) == 0:
        return None
    elif len(merged_results) == 1:
        return merged_results[0]
    else:
        return tuple(merged_results)


def range(
    stop_or_start: "int | UInt",
    stop: "int | UInt | None" = None,
    step: "int | UInt" = 1,
) -> typing.Iterator[UInt]:
    """Symbolic range for use in qkernel for-loops.

    This function accepts UInt (symbolic) values and is transformed
    by the AST transformer into for_loop() calls.

    Example:
        ```python
        for i in qmc.range(n):          # 0 to n-1
        for i in qmc.range(start, stop):  # start to stop-1
        for i in qmc.range(start, stop, step):
        ```

    Note:
        This function is a placeholder - the actual looping is handled by
        the AST transformer which converts range() calls to for_loop().
    """
    # This is a dummy implementation - AST transformer replaces this with for_loop()
    # The function signature accepts UInt for type checking purposes
    return iter([])


def items(d: Dict) -> DictItemsIterator:
    """Iterate over dictionary key-value pairs.

    This function returns an iterator over (key, value) pairs from a Dict.
    Used for iterating over Ising coefficients and similar data structures.

    Example:
        ```python
        for (i, j), Jij in qmc.items(ising):
            q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
        ```

    Args:
        d: A Dict handle to iterate over

    Returns:
        DictItemsIterator yielding (key, value) pairs
    """
    return d.items()


@contextlib.contextmanager
def for_items(
    d: Dict,
    key_var_names: list[str],
    value_var_name: str,
) -> typing.Generator[tuple[typing.Any, typing.Any], None, None]:
    """Builder function to create a for-items loop in Qamomile frontend.

    This context manager creates a ForItemsOperation that iterates over
    dictionary (key, value) pairs. The operation is always unrolled at
    transpile time since quantum backends cannot natively iterate over
    classical data structures.

    Args:
        d: Dict handle to iterate over
        key_var_names: Names of key unpacking variables (e.g., ["i", "j"] for tuple keys)
        value_var_name: Name of value variable (e.g., "Jij")

    Yields:
        Tuple of (key_handles, value_handle) for use in loop body

    Raises:
        TypeError: If ``d`` is a runtime-parameter Dict (declared via
            ``parameters=[...]`` without bound data). Its key structure
            is unknown at compile time, so an items() loop cannot be
            unrolled; only constant-key subscript lookups (``d[key]``)
            are supported for runtime-parameter dicts.

    Example:
        ```python
        @qkernel
        def ising_cost(
            q: Vector[Qubit],
            ising: Dict[Tuple[UInt, UInt], Float],
            gamma: Float,
        ) -> Vector[Qubit]:
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            return q
        ```
    """
    # Runtime-parameter dicts carry no bound data, so the loop would
    # silently unroll over an unknown key set at emit time. Fail at
    # trace time with an actionable message instead. The Handle-level
    # flag (set only by ``parameters=[...]`` input creation) is what
    # distinguishes these from visualization / inner-kernel dummy
    # inputs, whose entries connect at inline/emit time; dicts reaching
    # a sub-kernel indirectly are caught by the emit-time check in
    # ``emit_for_items`` instead.
    if getattr(d, "_runtime_parameter", False):
        raise TypeError(
            f"Dict '{d.value.name}' is a runtime parameter; items() "
            f"iteration requires the key structure at compile time. "
            f"Bind the dict via bindings={{...}} instead, or use "
            f"constant-key subscript lookups (d[key])."
        )

    parent_tracer = get_current_tracer()
    body_tracer = Tracer()

    # Check if Dict key type is a vector (e.g., Dict[Vector[UInt], Float])
    key_type = getattr(d, "_key_type", None)
    _key_is_vector = (
        key_type is not None and is_array_type(key_type) and len(key_var_names) == 1
    )

    # Track the IR Values for each key/value variable so we can store them
    # on the ForItemsOperation. Identity (for emit-time bindings + loop
    # body refs) flows through these UUIDs; the string names are
    # display-only.
    key_var_values: list[Value] = []
    if _key_is_vector:
        # Create a symbolic Vector[UInt] handle for the key variable
        kv_name = key_var_names[0]
        dim0_value = Value(type=UIntType(), name=f"{kv_name}_dim0")
        array_value = ArrayValue(
            type=UIntType(),
            name=kv_name,
            shape=(dim0_value,),
        )
        key_var_values.append(array_value)
        dim0_handle = UInt(value=dim0_value)
        key_result = object.__new__(Vector)
        key_result.value = array_value
        key_result._shape = (dim0_handle,)
        key_result._borrowed_indices = {}
        key_result.parent = None
        key_result.indices = ()
        key_result.name = kv_name
        key_result.id = str(id(key_result))
        key_result._consumed = False
        key_result.element_type = UInt  # type: ignore[assignment]
    else:
        # Create symbolic key handles (UInt for each key variable)
        key_handles = []
        for kv_name in key_var_names:
            kv_value = Value(type=UIntType(), name=kv_name)
            key_var_values.append(kv_value)
            key_handle = UInt(
                value=kv_value,
                init_value=0,  # Placeholder, actual value bound at emit time
            )
            key_handles.append(key_handle)

        # Package key handles: tuple for multiple keys, single handle otherwise
        if len(key_handles) == 1:
            key_result = key_handles[0]
        else:
            key_result = tuple(key_handles)

    # Create symbolic value handle (Float for Ising coefficients)
    value_var_value = Value(type=FloatType(), name=value_var_name)
    value_handle = Float(
        value=value_var_value,
        init_value=0.0,  # Placeholder, actual value bound at emit time
    )

    with trace(body_tracer):
        yield (key_result, value_handle)

    # Convert pending region entries (loop-carried classical scalars)
    # into explicit RegionArg records and publish post-loop result
    # handles for the AST-injected loop_region_result assignments.
    region_args, region_results = _close_region_entries(body_tracer, parent_tracer)

    # Create ForItemsOperation with captured body operations
    for_items_op = ForItemsOperation(
        key_vars=key_var_names,
        value_var=value_var_name,
        key_is_vector=_key_is_vector,
        key_var_values=tuple(key_var_values),
        value_var_value=value_var_value,
        operations=body_tracer.operations,
        loop_carried_rebinds=body_tracer.loop_carried_rebinds,
        region_args=region_args,
    )
    for_items_op.results.extend(region_results)
    for_items_op.operands.append(d.value)  # type: ignore[arg-type]  # DictValue is not Value but stored as operand

    parent_tracer.add_operation(for_items_op)
