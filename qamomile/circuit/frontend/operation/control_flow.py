import builtins
import contextlib
import contextvars
import copy
import dataclasses
import struct
import typing

from qamomile.circuit.frontend.func_to_block import handle_type_map, is_array_type
from qamomile.circuit.frontend.handle.array import ArrayBase, Vector
from qamomile.circuit.frontend.handle.containers import Dict, DictItemsIterator
from qamomile.circuit.frontend.handle.primitives import (
    Bit,
    Float,
    Handle,
    UInt,
)
from qamomile.circuit.frontend.qkernel_utils import (
    array_extents_equal,
    array_resource_identity,
    array_resources_equal,
    const_int,
    is_full_reslice_of_input,
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
    validate_region_args,
)
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    TupleType,
    UIntType,
    ValueType,
)
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase


class WhileLoop:
    """Mark the body of a traced Qamomile while loop."""

    pass


@contextlib.contextmanager
def while_loop(cond: typing.Callable) -> typing.Generator[WhileLoop, None, None]:
    """Create a while loop whose condition is a measurement result.

    The condition must be a ``Bit`` produced by ``qmc.measure()``.
    Non-measurement conditions (classical variables, constants,
    comparisons) are accepted at build time but will be rejected by
    ``ValidateWhileContractPass`` during transpilation.

    Args:
        cond (typing.Callable): A callable (lambda) that returns the loop condition.
            Must return a ``Bit`` handle originating from ``qmc.measure()``.

    Yields:
        WhileLoop: A marker object for the while loop context.

    Raises:
        TypeError: If either condition evaluation cannot be represented as a
            scalar IR value.
        ValueError: If the constructed loop has inconsistent region-result
            metadata.

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

    Classical scalar updates (``count = count + 1``) are likewise
    rejected on while loops: the trip count is a runtime measurement
    outcome, so the loop cannot unroll, and no backend can thread a
    classical value between runtime-loop iterations. Use a
    compile-time-bounded ``qmc.range`` loop for carried reductions.
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
    validate_region_args(while_op)
    parent_tracer.add_operation(while_op)


@contextlib.contextmanager
def for_loop(
    start, stop, step=1, var_name: str = "_loop_idx"
) -> typing.Generator[UInt, None, None]:
    """Create a traced for loop in the Qamomile frontend.

    Args:
        start (typing.Any): Inclusive loop start as an integer or ``UInt``.
        stop (typing.Any): Exclusive loop stop as an integer or ``UInt``.
        step (typing.Any): Nonzero loop step as an integer or ``UInt``.
            Defaults to 1.
        var_name (str): Display name of the loop variable. Defaults to
            ``"_loop_idx"``.

    Yields:
        UInt: The loop iteration variable (can be used as array index)

    Raises:
        TypeError: If a bound cannot be represented as a scalar IR value.
        ValueError: If the constructed loop has inconsistent region-result
            metadata.

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

    Classical scalar updates (``total = total + i``) become explicit
    ``RegionArg`` records on the ``ForOperation``: the loop enters with
    the initializer, each iteration reads the previous iteration's
    value, and post-loop code reads the loop result.
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
        results=list(region_results),
        loop_var=var_name,
        loop_var_value=loop_var_value,
        operations=body_tracer.operations,
        loop_carried_rebinds=body_tracer.loop_carried_rebinds,
        region_args=region_args,
    )
    for_op.operands.append(_value_to_ir_value(start, "start"))
    for_op.operands.append(_value_to_ir_value(stop, "stop"))
    for_op.operands.append(_value_to_ir_value(step, "step"))

    validate_region_args(for_op)
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

    For arrays, this single-handle helper first installs an independent empty
    borrow table. ``_fresh_handle_copies_for_tracing`` then reconstructs the
    complete pre-branch ownership graph with branch-local owner copies. Live
    element handles and slice owners remain visible in both branches without
    either branch sharing mutable borrow state with the other or with the
    original graph. Destructively consumed element owners remain in that graph
    as destroyed-slot markers, so a runtime branch cannot revive them.

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
    # Break the shallow-copy alias first. The graph-copy helper repopulates
    # this table with branch-local copies of every pre-branch owner.
    if isinstance(c, ArrayBase):
        c._borrowed_indices = {}
    return c


def _fresh_handle_copies_for_tracing(
    values: list,
) -> tuple[list, dict[int, tuple[Handle, Handle]]]:
    """Copy a branch-local handle graph while preserving its ownership.

    Args:
        values (list): Values captured by the generated branch function.

    Returns:
        tuple[list, dict[int, tuple[Handle, Handle]]]: Branch-local values and
            an original-identity map containing every copied ownership node.
    """
    from qamomile.circuit.frontend.handle.array import VectorView

    copies: dict[int, Handle] = {}
    originals: dict[int, Handle] = {}

    def copy_handle(handle: Handle) -> Handle:
        """Copy one ownership node and recursively reconnect its graph.

        Args:
            handle (Handle): Original frontend handle to copy.

        Returns:
            Handle: The unique branch-local copy for ``handle``.
        """
        existing = copies.get(id(handle))
        if existing is not None:
            return existing

        copied = _fresh_handle_copy_for_tracing(handle)
        copies[id(handle)] = copied
        originals[id(handle)] = handle

        if handle.parent is not None:
            copied.parent = typing.cast(ArrayBase, copy_handle(handle.parent))

        if isinstance(handle, VectorView):
            copied._slice_parent = typing.cast(
                Vector, copy_handle(handle._slice_parent)
            )
            copied._slice_outer_view = (
                typing.cast(VectorView, copy_handle(handle._slice_outer_view))
                if handle._slice_outer_view is not None
                else None
            )

        if isinstance(handle, ArrayBase):
            copied._borrowed_indices = {
                key: copy_handle(owner) if isinstance(owner, Handle) else owner
                for key, owner in handle._borrowed_indices.items()
            }
        return copied

    result = [
        copy_handle(value) if isinstance(value, Handle) else value for value in values
    ]

    graph = {key: (originals[key], copied) for key, copied in copies.items()}
    return result, graph


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

    Each branch traces against a separately copied ownership graph. Existing
    pre-branch borrows are reconstructed with branch-local owners, and later
    borrow-table mutations remain independent between mutually exclusive
    branches. Any entry still outstanding at the end of a branch is an
    element whose ownership is still active on that path — including a
    pre-branch borrow or an element destructively consumed inside the branch
    (``if sel: _ = measure(qs[0])``). Union-merging both branches' entries
    onto the merged handle routes post-if element access into the existing
    borrow / destroyed-slot enforcement in ``ArrayBase.__getitem__``.

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
        val (typing.Any): Python primitive, frontend handle, or IR value to
            convert.
        name_prefix (str): Prefix for a generated value name. Defaults to
            ``"const"``.

    Returns:
        Value: Scalar IR value carrying the original identity or constant.

    Raises:
        TypeError: If ``val`` cannot be represented as a scalar IR value.
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


def _scalar_handle_class(value_type: ValueType) -> type[Handle]:
    """Return the frontend handle class for a scalar IR type.

    Args:
        value_type (ValueType): UInt, Float, or Bit IR type.

    Returns:
        type[Handle]: Matching frontend handle class.

    Raises:
        TypeError: If ``value_type`` is not a supported scalar type.
    """
    if isinstance(value_type, UIntType):
        return UInt
    if isinstance(value_type, FloatType):
        return Float
    if isinstance(value_type, BitType):
        return Bit
    raise TypeError(
        "Only UInt, Float, and Bit values have classical scalar handles; "
        f"got {value_type!r}."
    )


def _scalar_handle_for_value(value: Value) -> Handle:
    """Wrap a classical scalar IR value in its matching frontend handle.

    Args:
        value (Value): UInt-, Float-, or Bit-typed scalar value.

    Returns:
        Handle: A ``UInt``, ``Float``, or ``Bit`` carrying ``value``.

    Raises:
        TypeError: If ``value`` has a non-scalar or unsupported type.
    """
    handle_class = _scalar_handle_class(value.type)
    return handle_class(value=value)


def _promote_branch_scalar(value: typing.Any) -> typing.Any:
    """Promote a Python branch scalar to a mergeable frontend handle.

    Runtime branch bodies execute once during tracing. Returning a bare
    Python scalar from each body must therefore create an IR merge just like
    returning ``qmc.uint`` / ``qmc.float_`` / ``qmc.bit`` handles; selecting
    the true Python value unconditionally would freeze the branch outcome at
    trace time.

    Args:
        value (typing.Any): Branch result value to normalize.

    Returns:
        typing.Any: A scalar handle for bool/int/float and scalar ``Value``
            inputs, or ``value`` unchanged for existing handles and unsupported
            opaque objects.
    """
    if isinstance(value, Handle):
        return value
    if isinstance(value, Value) and isinstance(
        value.type, (UIntType, FloatType, BitType)
    ):
        return _scalar_handle_for_value(value)
    if isinstance(value, (bool, int, float)):
        return _scalar_handle_for_value(_value_to_ir_value(value, "branch_const"))
    return value


def _same_plain_scalar(true_val: typing.Any, false_val: typing.Any) -> bool:
    """Report whether both branch results are the identical plain scalar.

    A plain Python ``bool`` / ``int`` / ``float`` that both branches
    yield with the exact same type and value cannot diverge at runtime,
    so it needs no merge and must NOT be promoted: promotion would hand
    the tracing frame a symbolic handle where the user wrote a plain
    Python value, breaking trace-time interop (list indexing, builtin
    ``range``, format strings). Divergent plain scalars still promote —
    that is the represented-merge fix for the trace-time freeze bug.

    Args:
        true_val (typing.Any): True branch result slot.
        false_val (typing.Any): False branch result slot.

    Returns:
        bool: True when both values are plain scalars of the exact same
            type and representation. Floats compare bit-exactly (matching
            ``_same_exact_typed_constant`` on the lowering side): NaNs with
            the same payload pass through, while distinct NaN payloads and a
            ``0.0`` / ``-0.0`` pair still promote so no payload or sign bit is
            frozen to one branch's value.
    """
    if isinstance(true_val, (Handle, Value)) or isinstance(false_val, (Handle, Value)):
        return False
    if not isinstance(true_val, (bool, int, float)):
        return False
    if type(true_val) is not type(false_val):
        return False
    if isinstance(true_val, float):
        return struct.pack("!d", true_val) == struct.pack("!d", false_val)
    return true_val == false_val


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
        original_binding (typing.Any): Exact pre-loop Python binding. Identity
            carries restore this object so plain scalars stay available to
            ordinary Python operations after the traced loop.
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
    original_binding: typing.Any
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
        original_binding=resolved,
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
    trace completes. For every pending entry, synthesizes the loop-result
    ``Value`` and builds the ``RegionArg``. Genuine recurrences publish that
    result handle after the loop. Exact identity carries instead publish the
    original Python binding so plain scalars stay usable by ordinary Python
    code; their well-formed RegionArgs remain in the IR until compile-time
    lowering removes them through the normal identity-carry path.

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
            if _is_identity_region_entry(entry):
                result_handles[entry.var_name] = entry.original_binding
            else:
                result_handles[entry.var_name] = _region_scalar_handle(
                    result, entry.entry_handle
                )
    parent_tracer.loop_region_results = result_handles
    return tuple(region_args), results


def _is_identity_region_entry(entry: _PendingRegionArg) -> bool:
    """Return whether a pending carry preserves its exact initializer.

    Args:
        entry (_PendingRegionArg): Pending frontend region argument.

    Returns:
        bool: True when the body yields its own block argument, or an exact
            typed constant equal to the initializer.
    """
    if entry.yielded is None or entry.yielded.uuid in {
        entry.block_arg.uuid,
        entry.init.uuid,
    }:
        return True
    if entry.init.type != entry.yielded.type:
        return False
    if not entry.init.is_constant() or not entry.yielded.is_constant():
        return False
    return _same_plain_scalar(entry.init.get_const(), entry.yielded.get_const())


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


def _same_quantum_resource_binding(
    before_value: Value,
    after_handle: typing.Any,
) -> bool:
    """Return whether two bindings denote the same quantum resource.

    Args:
        before_value (Value): Quantum value bound before control flow.
        after_handle (typing.Any): Frontend binding observed afterwards.

    Returns:
        bool: ``True`` when scalar logical identity or canonical whole-array
            identity is preserved through exact full reslices.
    """
    after_value = _ir_value_from_handle_like(after_handle)
    if not isinstance(after_value, Value) or not after_value.type.is_quantum():
        return False
    if isinstance(before_value, ArrayValue) or isinstance(after_value, ArrayValue):
        if not isinstance(before_value, ArrayValue) or not isinstance(
            after_value, ArrayValue
        ):
            return False
        if array_resources_equal(before_value, after_value):
            return True

        from qamomile.circuit.frontend.handle.array import VectorView

        if not isinstance(after_handle, VectorView):
            return False
        current = after_handle
        seen: set[int] = set()
        while current._slice_outer_view is not None:
            if id(current) in seen:
                return False
            seen.add(id(current))
            outer = current._slice_outer_view
            if (
                len(current.value.shape) != 1
                or len(outer.value.shape) != 1
                or not array_extents_equal(
                    current.value.shape[0],
                    outer.value.shape[0],
                )
            ):
                return False
            if array_resources_equal(before_value, outer.value):
                return True
            current = outer
        return False
    return before_value.logical_id == after_value.logical_id


def _normalize_loop_full_reslice_lineage(
    before_handle: typing.Any,
    after_handle: typing.Any,
) -> None:
    """Collapse a loop-carried full-reslice alias to its pre-loop nesting.

    Loop tracing executes the body once, so ``view = view[:]`` leaves the
    trace-time post-loop handle nested under the pre-loop ``view``. At runtime
    the slice is only another representation of the same quantum resource;
    retaining that extra frontend nesting level incorrectly requires callers
    to return through a handle that is no longer bound to any Python name.

    Args:
        before_handle (typing.Any): Binding captured at loop entry.
        after_handle (typing.Any): Binding observed at the end of the traced
            loop body.

    Returns:
        None
    """
    from qamomile.circuit.frontend.handle.array import VectorView

    if not isinstance(before_handle, VectorView) or not isinstance(
        after_handle, VectorView
    ):
        return
    if getattr(after_handle, "_slice_divergent_merge", False):
        return
    if not array_resources_equal(before_handle.value, after_handle.value):
        return
    if not array_resources_equal(
        before_handle._slice_parent.value,
        after_handle._slice_parent.value,
    ):
        return

    current = after_handle
    seen: set[int] = set()
    reaches_pre_loop_view = False
    while current._slice_outer_view is not None:
        if id(current) in seen:
            return
        seen.add(id(current))
        outer = current._slice_outer_view
        if not array_resources_equal(current.value, outer.value):
            return
        if outer is before_handle or outer.value.uuid == before_handle.value.uuid:
            reaches_pre_loop_view = True
            break
        current = outer
    if not reaches_pre_loop_view:
        return

    # Preserve the post-body ArrayValue and borrow owner; only remove the
    # representation-only nesting introduced while tracing the loop body.
    # Partial reslices have a distinct resource identity and return above.
    after_handle._slice_parent = before_handle._slice_parent
    after_handle._slice_outer_view = before_handle._slice_outer_view


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
      quantum and its post-body value denotes a different resource — a fresh
      allocation or another register rather than a gate self-update or exact
      full reslice. These feed the transpiler's loop-body quantum discard
      check.
    - **Classical values** (only names in ``classical_names``): supported
      same-type ``UInt`` / ``Float`` values that were region-bound at body
      entry complete their pending ``RegionArg`` instead of producing a
      record. Every other representable rebind produces a residual record,
      including measurement-backed ``Bit`` values and containers. The
      transpiler rejects shapes that need unsupported routing; a store-only
      scalar ``Bit`` is accepted only for a statically non-empty unrolled loop,
      where reusing the measurement-result UUID correctly selects the last
      iteration and no zero-trip initializer must be routed.
      ``classical_names`` includes both read-before-write carries and
      store-only values that are live after the loop.

    No IR operations are emitted; this only annotates the tracer.

    Args:
        snapshot (dict[str, typing.Any]): Pre-loop-body handles from
            ``loop_rebind_snapshot``.
        frame_locals (dict[str, typing.Any]): The caller's ``locals()``
            at the end of the loop body.
        names (tuple[str, ...]): All candidate variable names.
        classical_names (tuple[str, ...]): The subset of ``names`` the
            loop body either reads before writing or overwrites and exposes
            after the loop; only these may produce classical records or
            complete pending region arguments.
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
                not isinstance(after_value, Value)
                or after_value.type.is_quantum()
                or after_value.type != entry.init.type
            ):
                # The body's final value is not a same-type classical scalar.
                # Keep the identity RegionArg for reads already traced via the
                # block argument, but retain a residual record so validation
                # rejects the unsupported post-loop binding instead of leaking
                # the one trace-time body value.
                entry.publish_result = False
                if isinstance(after_value, ValueBase):
                    records.append(
                        LoopCarriedRebind(
                            var_name=name,
                            before=entry.init,
                            after=after_value,
                        )
                    )
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
        if isinstance(before_value, Value) and before_value.type.is_quantum():
            # Quantum rebind: compare canonical resource identity, not UUID.
            # A gate self-update and an exact full reslice keep the resource
            # identity and are not rebinds.
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
            if not _same_quantum_resource_binding(before_value, after):
                records.append(
                    LoopCarriedRebind(
                        var_name=name,
                        before=before_value,
                        after=after_value,
                    )
                )
            else:
                _normalize_loop_full_reslice_lineage(before, after)
            continue
        if name not in classical_candidates:
            continue
        if after_value is None:
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
        if before_value.type.is_quantum():
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


def _merged_logical_identity(
    true_value: Value,
    false_value: Value,
) -> str | None:
    """Return the shared resource identity for a branch merge.

    Args:
        true_value (Value): Value returned by the true branch.
        false_value (Value): Value returned by the false branch.

    Returns:
        str | None: Shared logical identity, or None for distinct resources.
    """
    if isinstance(true_value, ArrayValue):
        if not isinstance(false_value, ArrayValue):
            return None
        if not array_resources_equal(true_value, false_value):
            return None
        return array_resource_identity(true_value)
    elif isinstance(false_value, ArrayValue):
        return None
    if true_value.logical_id == false_value.logical_id:
        return true_value.logical_id
    return None


def _common_slice_merge_template(
    true_val: typing.Any,
    false_val: typing.Any,
) -> tuple[Handle, str] | None:
    """Return a common pre-reslice view and its canonical resource identity.

    Args:
        true_val (typing.Any): Value returned by the true branch.
        false_val (typing.Any): Value returned by the false branch.

    Returns:
        tuple[Handle, str] | None: A structural merge template and shared
            resource identity, or ``None`` when the branch views are not
            equivalent through exact full reslices.
    """
    from qamomile.circuit.frontend.handle.array import VectorView

    if not isinstance(true_val, VectorView) or not isinstance(false_val, VectorView):
        return None

    def full_reslice_ancestor(view: VectorView) -> VectorView | None:
        """Find the first ancestor not hidden by exact full reslices.

        Args:
            view (VectorView): Branch-local view to walk toward its outer
                lineage.

        Returns:
            VectorView | None: First structurally significant ancestor, or
                ``None`` when the outer-view chain contains a cycle.
        """
        current = view
        seen: set[int] = set()
        while current._slice_outer_view is not None:
            if id(current) in seen:
                return None
            seen.add(id(current))
            outer = current._slice_outer_view
            if (
                len(current.value.shape) != 1
                or len(outer.value.shape) != 1
                or not array_extents_equal(
                    current.value.shape[0],
                    outer.value.shape[0],
                )
            ):
                break
            current = outer
        return current

    true_ancestor = full_reslice_ancestor(true_val)
    false_ancestor = full_reslice_ancestor(false_val)
    if (
        true_ancestor is None
        or false_ancestor is None
        or not array_resources_equal(
            true_ancestor.value,
            false_ancestor.value,
        )
    ):
        return None
    identity = array_resource_identity(true_ancestor.value)
    return (true_ancestor, identity) if identity is not None else None


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
            f"Branch value mismatch: Type mismatch in if-else branches: "
            f"true branch has {true_v.type}, false branch has {false_v.type}"
        )

    # Exact full re-slices are representation-only aliases. Prefer the branch
    # value they slice from so merge behaviour does not depend on whether the
    # direct value appeared in the true or false branch.
    template_val = true_val
    template_v = true_v
    common_slice = _common_slice_merge_template(true_val, false_val)
    slice_identity: str | None = None
    if common_slice is not None:
        template_val, slice_identity = common_slice
        template_v = template_val.value
    elif (
        isinstance(true_v, ArrayValue)
        and isinstance(false_v, ArrayValue)
        and isinstance(false_val, Handle)
        and is_full_reslice_of_input(true_v, false_v)
    ):
        template_val = false_val
        template_v = false_v

    # Create merge output value (indexed to avoid name collisions)
    merge_index = len(if_operation.results)
    if isinstance(template_v, ArrayValue):
        merge_output = ArrayValue(
            type=template_v.type,
            name=f"{template_v.name}_merge_{merge_index}",
            shape=template_v.shape,
            slice_of=template_v.slice_of,
            slice_start=template_v.slice_start,
            slice_step=template_v.slice_step,
        )
    else:
        merge_output = Value(
            type=template_v.type, name=f"{template_v.name}_merge_{merge_index}"
        )
    merged_logical_id = _merged_logical_identity(true_v, false_v) or slice_identity
    if merged_logical_id is not None:
        merge_output = dataclasses.replace(
            merge_output,
            logical_id=merged_logical_id,
            version=max(true_v.version, false_v.version) + 1,
        )

    # Wrap the merge output in the true-branch handle's family. The wrap
    # may rebuild the output value with copied metadata (QFixed carriers),
    # so the handle's value — not the bare merge_output — is what the merge
    # must record.
    if not isinstance(template_val, Handle):
        raise TypeError(
            "Unsupported Handle type for if-else merge: "
            f"{type(template_val).__name__}. Add explicit handle wrapping support "
            "before merging this handle type."
        )
    other_v = true_v if template_val is false_val else false_v
    merged_handle = template_val._wrap_merge_result(merge_output, other_v)

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


def _reconnect_merge_owners(
    merges: typing.Sequence[tuple[typing.Any, typing.Any, Handle]],
    merged_by_branch_pair: dict[tuple[int, int], Handle],
    canonical_by_branch_id: dict[int, Handle],
    if_operation: IfOperation,
) -> None:
    """Reconnect merged element and view handles to canonical ownership nodes.

    A view merge is wrapped from one branch's structural template. Its parent
    and outer-view fields therefore point into that branch's copied handle
    graph. Normalize those template fields directly instead of reconstructing
    them from the two branch results: exact full reslices can give those
    results different immediate outer-view depths even though they represent
    the same resource.

    Args:
        merges (typing.Sequence[tuple[typing.Any, typing.Any, Handle]]): Branch
            value pairs and the handles created to merge them.
        merged_by_branch_pair (dict[tuple[int, int], Handle]): Canonical handle
            selected for each pair of true- and false-branch handles.
        canonical_by_branch_id (dict[int, Handle]): Canonical handle selected
            for each copied branch handle identity.
        if_operation (IfOperation): Conditional receiving any auxiliary
            source-index merge needed to defer a quantum return check.

    Returns:
        None
    """
    from qamomile.circuit.frontend.handle.array import VectorView

    def reconnect_array_borrows(
        branch_array: ArrayBase,
        merged_array: ArrayBase,
    ) -> None:
        """Copy one branch table through canonical post-If owners.

        Args:
            branch_array (ArrayBase): Branch-local array whose outstanding
                owners are being carried across the merge.
            merged_array (ArrayBase): Canonical array selected after the If.

        Returns:
            None.
        """
        for key, branch_owner in branch_array._borrowed_indices.items():
            owner = branch_owner
            if isinstance(branch_owner, Handle):
                owner = canonical_by_branch_id.get(id(branch_owner), branch_owner)
                if not isinstance(branch_owner, ArrayBase):
                    branch_parent = branch_owner.parent
                    canonical_parent = canonical_by_branch_id.get(id(branch_parent))
                    if isinstance(canonical_parent, ArrayBase):
                        owner.parent = canonical_parent
                        owner.indices = branch_owner.indices
            merged_array._borrowed_indices[key] = owner

    # Normalize array tables first. Scalar/view merge reconnection below is
    # more specific and must win when both passes update the same slot.
    for true_val, false_val, merged_handle in merges:
        if isinstance(merged_handle, ArrayBase):
            if isinstance(true_val, ArrayBase):
                reconnect_array_borrows(true_val, merged_handle)
            if isinstance(false_val, ArrayBase):
                reconnect_array_borrows(false_val, merged_handle)

    for true_val, false_val, merged_handle in merges:
        true_parent = getattr(true_val, "parent", None)
        false_parent = getattr(false_val, "parent", None)
        true_indices = getattr(true_val, "indices", ())
        false_indices = getattr(false_val, "indices", ())
        if isinstance(true_parent, ArrayBase) and isinstance(false_parent, ArrayBase):
            template_parent = getattr(merged_handle, "parent", None)
            merged_parent = canonical_by_branch_id.get(id(template_parent))
            if merged_parent is None:
                merged_parent = merged_by_branch_pair.get(
                    (id(true_parent), id(false_parent))
                )
            if isinstance(merged_parent, ArrayBase):
                merged_parent._borrowed_indices.update(true_parent._borrowed_indices)
                merged_parent._borrowed_indices.update(false_parent._borrowed_indices)
                if len(true_indices) == len(false_indices) and all(
                    left.value.uuid == right.value.uuid
                    or (
                        const_int(left.value) is not None
                        and const_int(left.value) == const_int(right.value)
                    )
                    for left, right in zip(
                        true_indices,
                        false_indices,
                        strict=True,
                    )
                ):
                    merged_handle.parent = merged_parent
                    merged_handle.indices = true_indices
                elif (
                    len(true_indices) == len(false_indices)
                    and len(true_indices) > 0
                    and merged_handle.value.type.is_quantum()
                ):
                    merged_indices: list[UInt] = []
                    for index, (true_index, false_index) in enumerate(
                        zip(true_indices, false_indices, strict=True)
                    ):
                        merge_value = Value(
                            type=UIntType(),
                            name=f"quantum_return_index_merge_{index}",
                        )
                        if_operation.add_merge(
                            true_index.value,
                            false_index.value,
                            merge_value,
                        )
                        merged_indices.append(UInt(value=merge_value))

                    for branch_indices in (true_indices, false_indices):
                        branch_key = merged_parent._make_indices_key(branch_indices)
                        merged_parent._borrowed_indices.pop(branch_key, None)
                    merged_handle.parent = merged_parent
                    merged_handle.indices = tuple(merged_indices)
                    merged_parent._borrowed_indices[
                        merged_parent._make_indices_key(merged_handle.indices)
                    ] = merged_handle

        if not (
            isinstance(true_val, VectorView)
            and isinstance(false_val, VectorView)
            and isinstance(merged_handle, VectorView)
        ):
            continue

        true_parent = true_val._slice_parent
        false_parent = false_val._slice_parent
        template_parent = merged_handle._slice_parent
        merged_parent = canonical_by_branch_id.get(id(template_parent))
        if merged_parent is None:
            merged_parent = merged_by_branch_pair.get(
                (id(true_parent), id(false_parent))
            )
        if merged_parent is None and true_parent is false_parent:
            merged_parent = true_parent
        if not isinstance(merged_parent, Vector):
            continue

        merged_handle._slice_parent = merged_parent
        template_outer = merged_handle._slice_outer_view
        if template_outer is None:
            merged_handle._slice_outer_view = None
        else:
            merged_outer = canonical_by_branch_id.get(id(template_outer))
            if isinstance(merged_outer, VectorView):
                merged_handle._slice_outer_view = merged_outer

        coverage = _slice_view_coverage(merged_handle)
        if (
            coverage is None
            or _slice_view_coverage(true_val) != coverage
            or _slice_view_coverage(false_val) != coverage
        ):
            continue
        for idx in coverage:
            merged_parent._borrowed_indices[(f"const:{idx}",)] = merged_handle


def _canonical_branch_handles(
    true_handle_graph: dict[int, tuple[Handle, Handle]],
    false_handle_graph: dict[int, tuple[Handle, Handle]],
    merged_by_branch_pair: dict[tuple[int, int], Handle],
) -> dict[int, Handle]:
    """Map copied branch handles to their canonical post-merge handles.

    Alias pairs can share one side when branch-local names diverge, so a side
    is accepted from those pairs only when every candidate agrees. The two
    copies of the same pre-branch handle are authoritative and override any
    ambiguous alias candidate.

    Args:
        true_handle_graph (dict[int, tuple[Handle, Handle]]): Original handles
            and their true-branch copies, keyed by original identity.
        false_handle_graph (dict[int, tuple[Handle, Handle]]): Original handles
            and their false-branch copies, keyed by original identity.
        merged_by_branch_pair (dict[tuple[int, int], Handle]): Canonical handle
            selected for each known true/false branch pair.

    Returns:
        dict[int, Handle]: Branch-copy identities mapped to canonical handles.
    """
    candidates: dict[int, dict[int, Handle]] = {}
    for (true_id, false_id), canonical in merged_by_branch_pair.items():
        for branch_id in (true_id, false_id):
            candidates.setdefault(branch_id, {})[id(canonical)] = canonical

    canonical_by_branch_id = {
        branch_id: next(iter(branch_candidates.values()))
        for branch_id, branch_candidates in candidates.items()
        if len(branch_candidates) == 1
    }

    for key in true_handle_graph.keys() & false_handle_graph.keys():
        true_copy = true_handle_graph[key][1]
        false_copy = false_handle_graph[key][1]
        pair = (id(true_copy), id(false_copy))
        canonical = merged_by_branch_pair[pair]
        canonical_by_branch_id[id(true_copy)] = canonical
        canonical_by_branch_id[id(false_copy)] = canonical

    for canonical in merged_by_branch_pair.values():
        canonical_by_branch_id.setdefault(id(canonical), canonical)
    return canonical_by_branch_id


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


def _ir_value_from_handle_like(value: typing.Any) -> ValueBase | None:
    """Extract the IR value carried by a handle-like object.

    Args:
        value (typing.Any): Candidate handle or IR value.

    Returns:
        ValueBase | None: The contained IR value when available, otherwise
        ``None``.
    """
    if isinstance(value, ValueBase):
        return value
    ir_value = getattr(value, "value", None)
    if isinstance(ir_value, ValueBase):
        return ir_value
    return None


def _find_original_handle_for_result(
    result: typing.Any,
    variables: list,
) -> typing.Any | None:
    """Find the input handle that still owns a pass-through result value.

    ``visit_If`` may return values that were not passed as inputs, such as new
    locals defined in both branches. For no-op merge elision we still want to
    reuse the original handle when the result is merely a branch copy of an
    existing input. Arrays carry live borrow state, while scalar elements carry
    the canonical parent that owns their borrow. Matching by UUID keeps this
    independent of the differing input and output variable orders.

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
    return not _same_quantum_resource_binding(pre_value, post_handle)


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
    """Trace an if/else conditional and merge its branch results.

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
        cond_func (typing.Callable): Function returning the condition as a
            ``Bit`` or bool-like handle.
        true_func (typing.Callable): Function tracing the true branch and
            returning its updated variables.
        false_func (typing.Callable): Function tracing the false branch and
            returning its updated variables.
        variables (list): Variables captured by the two branch functions.
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
        typing.Any: The sole merged value, a tuple of merged values, or None
            when the branches return no values.

    Raises:
        TypeError: If corresponding branch values have incompatible types or
            divergent values with no Qamomile IR representation.
        ValueError: If branch result lengths or probe-tail lengths disagree.

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
    true_vars, true_handle_graph = _fresh_handle_copies_for_tracing(variables)
    false_vars, false_handle_graph = _fresh_handle_copies_for_tracing(variables)
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
    merged_aliases: dict[tuple[int, int], Handle] = {}
    owner_merges: list[tuple[typing.Any, typing.Any, Handle]] = []
    for true_val, false_val in zip(true_result, false_result, strict=True):
        if _same_plain_scalar(true_val, false_val):
            # Identical plain scalar on both sides: no runtime divergence
            # to represent, and the surrounding Python code may rely on
            # it staying a genuine Python value after the if.
            merged_results.append(true_val)
            continue
        true_val = _promote_branch_scalar(true_val)
        false_val = _promote_branch_scalar(false_val)
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
            alias_key = (id(true_val), id(false_val))
            if _can_elide_scalar_merge(true_val, false_val):
                original_val = _find_original_handle_for_result(true_val, variables)
                merged_handle = original_val if original_val is not None else true_val
                merged_results.append(merged_handle)
                if isinstance(merged_handle, Handle):
                    merged_aliases[alias_key] = merged_handle
                    owner_merges.append((true_val, false_val, merged_handle))
                continue
            if _can_elide_array_merge(true_val, false_val, true_tracer, false_tracer):
                original_val = _find_original_handle_for_result(true_val, variables)
                merged_handle = original_val if original_val is not None else true_val
                merged_results.append(merged_handle)
                if isinstance(merged_handle, Handle):
                    merged_aliases[alias_key] = merged_handle
                    owner_merges.append((true_val, false_val, merged_handle))
                continue
            merged_handle = merged_aliases.get(alias_key)
            if merged_handle is None:
                merged_handle = _create_merge_for_values(true_val, false_val, if_op)
                merged_aliases[alias_key] = merged_handle
            owner_merges.append((true_val, false_val, merged_handle))
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
            # Opaque pass-through objects are safe only when both branches
            # return the exact same object. Divergent opaque values have no IR
            # representation and must fail instead of silently selecting the
            # true branch's trace-time value.
            if true_val is not false_val:
                raise TypeError(
                    "Branch values cannot be merged because they have no "
                    "Qamomile scalar/handle representation: "
                    f"true={type(true_val).__name__}, "
                    f"false={type(false_val).__name__}."
                )
            merged_results.append(true_val)

    ownership_targets = {
        (
            id(true_handle_graph[key][1]),
            id(false_handle_graph[key][1]),
        ): true_handle_graph[key][0]
        for key in true_handle_graph.keys() & false_handle_graph.keys()
    }
    ownership_targets.update(merged_aliases)
    canonical_by_branch_id = _canonical_branch_handles(
        true_handle_graph,
        false_handle_graph,
        ownership_targets,
    )
    _reconnect_merge_owners(
        owner_merges,
        ownership_targets,
        canonical_by_branch_id,
        if_op,
    )

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


def _for_items_key_ir_types(
    d: Dict,
    key_is_vector: bool,
    key_var_names: list[str],
) -> tuple[ValueType, ...]:
    """Resolve the IR type of each for-items key binding.

    Args:
        d (Dict): Iterated dict handle carrying its ``Dict[K, V]``
            annotation.
        key_is_vector (bool): Whether one binding represents a Vector key.
        key_var_names (list[str]): Flattened key-binding names from the AST.

    Returns:
        tuple[ValueType, ...]: One scalar element type for a Vector key, or
            one scalar type per unpacked key variable.

    Raises:
        TypeError: If the declared key type is unsupported or its tuple arity
            disagrees with the loop target.
    """
    key_annotation = getattr(d, "_key_type", None)
    if key_annotation is None:
        # Legacy Dict handles created without a generic annotation used UInt
        # keys. Preserve that compatibility while typed handles take the path
        # below.
        return tuple(UIntType() for _ in key_var_names)

    mapped = handle_type_map(key_annotation)
    if key_is_vector:
        key_types = (mapped,)
    elif isinstance(mapped, TupleType):
        key_types = mapped.element_types
    else:
        key_types = (mapped,)

    if len(key_types) != len(key_var_names):
        raise TypeError(
            "For-items key binding arity does not match the declared Dict key "
            f"type: {len(key_var_names)} names for {len(key_types)} values."
        )
    for key_type in key_types:
        _scalar_handle_class(key_type)
    return key_types


@contextlib.contextmanager
def for_items(
    d: Dict,
    key_var_names: list[str],
    value_var_name: str,
) -> typing.Generator[tuple[typing.Any, typing.Any], None, None]:
    """Create a traced for-items loop in the Qamomile frontend.

    This context manager creates a ForItemsOperation that iterates over
    dictionary (key, value) pairs. The operation is always unrolled at
    transpile time since quantum backends cannot natively iterate over
    classical data structures.

    Args:
        d (Dict): Dict handle whose compile-time-known entries are iterated.
        key_var_names (list[str]): Names of key-unpacking variables, for
            example ``["i", "j"]`` for tuple keys.
        value_var_name (str): Display name of the item-value variable.

    Yields:
        tuple[typing.Any, typing.Any]: Key handle(s) and the typed scalar
            value handle used while tracing the loop body.

    Raises:
        TypeError: If ``d`` is a runtime-parameter Dict (declared via
            ``parameters=[...]`` without bound data), or its key annotation
            cannot be represented by the loop target. A runtime Dict's key
            structure is unknown at compile time, so an items() loop cannot
            be unrolled; only constant-key subscript lookups (``d[key]``)
            are supported for runtime-parameter dicts.
        NotImplementedError: If the Dict value annotation is a container or
            another type without a scalar frontend handle.
        ValueError: If the constructed loop has inconsistent region-result
            metadata.

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
    key_ir_types = _for_items_key_ir_types(d, _key_is_vector, key_var_names)

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
            type=key_ir_types[0],
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
        key_result.element_type = _scalar_handle_class(  # type: ignore[assignment]
            key_ir_types[0]
        )
    else:
        # Create symbolic key handles according to the declared Dict key type.
        key_handles = []
        for kv_name, key_ir_type in zip(key_var_names, key_ir_types, strict=True):
            kv_value = Value(type=key_ir_type, name=kv_name)
            key_var_values.append(kv_value)
            key_handle = _scalar_handle_for_value(kv_value)
            key_handles.append(key_handle)

        # Package key handles: tuple for multiple keys, single handle otherwise
        if len(key_handles) == 1:
            key_result = key_handles[0]
        else:
            key_result = tuple(key_handles)

    # Create the symbolic item-value handle from Dict[K, V], preserving UInt,
    # Float, and Bit rather than treating every coefficient as Float.
    value_ir_type, _value_handle_class, _value_coercer = d._result_spec()
    value_var_value = Value(type=value_ir_type, name=value_var_name)
    value_handle = _scalar_handle_for_value(value_var_value)

    with trace(body_tracer):
        yield (key_result, value_handle)

    # Convert pending region entries (loop-carried classical scalars)
    # into explicit RegionArg records and publish post-loop result
    # handles for the AST-injected loop_region_result assignments.
    region_args, region_results = _close_region_entries(body_tracer, parent_tracer)

    # Create ForItemsOperation with captured body operations
    for_items_op = ForItemsOperation(
        results=list(region_results),
        key_vars=key_var_names,
        value_var=value_var_name,
        key_is_vector=_key_is_vector,
        key_var_values=tuple(key_var_values),
        value_var_value=value_var_value,
        operations=body_tracer.operations,
        loop_carried_rebinds=body_tracer.loop_carried_rebinds,
        region_args=region_args,
    )
    for_items_op.operands.append(d.value)  # type: ignore[arg-type]  # DictValue is not Value but stored as operand

    validate_region_args(for_items_op)
    parent_tracer.add_operation(for_items_op)
