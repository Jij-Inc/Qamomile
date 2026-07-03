import builtins
import contextlib
import copy
import typing

from qamomile.circuit.frontend.func_to_block import is_array_type
from qamomile.circuit.frontend.handle.array import ArrayBase, Vector
from qamomile.circuit.frontend.handle.containers import Dict, DictItemsIterator
from qamomile.circuit.frontend.handle.hamiltonian import Observable
from qamomile.circuit.frontend.handle.primitives import (
    Bit,
    Float,
    Handle,
    QFixed,
    Qubit,
    UInt,
)
from qamomile.circuit.frontend.tracer import Tracer, get_current_tracer, trace
from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.control_flow import (
    BranchRebind,
    ForItemsOperation,
    ForOperation,
    IfOperation,
    LoopCarriedRebind,
    WhileOperation,
)
from qamomile.circuit.ir.types import QFixedType
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
                q = qm.qubit("q2")
                q = qm.h(q)
                bit = qm.measure(q)
            return bit
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

    # Create ForOperation
    # operands: [start, stop, step]
    for_op = ForOperation(
        loop_var=var_name,
        loop_var_value=loop_var_value,
        operations=body_tracer.operations,
        loop_carried_rebinds=body_tracer.loop_carried_rebinds,
    )
    for_op.operands.append(_value_to_ir_value(start, "start"))
    for_op.operands.append(_value_to_ir_value(stop, "stop"))
    for_op.operands.append(_value_to_ir_value(step, "step"))

    parent_tracer.add_operation(for_op)


def _create_handle_from_value(value: Value, template_handle: Handle) -> Handle:
    """Wrap a phi result in the same frontend handle family.

    Args:
        value (Value): IR phi result value to expose to the traced Python code.
        template_handle (Handle): Branch handle whose frontend type should be
            preserved for the merged value.

    Returns:
        Handle: Frontend handle of the same supported family as
            ``template_handle``.

    Raises:
        TypeError: If ``template_handle`` is not a supported phi-merge handle
            type.
    """
    if isinstance(template_handle, Qubit):
        return Qubit(value=value)
    elif isinstance(template_handle, UInt):
        return UInt(value=value)
    elif isinstance(template_handle, Float):
        return Float(value=value)
    elif isinstance(template_handle, Bit):
        return Bit(value=value)
    elif isinstance(template_handle, QFixed):
        return QFixed(value=value)
    elif isinstance(template_handle, Observable):
        return Observable(value=value)
    elif isinstance(template_handle, ArrayBase):
        from qamomile.circuit.frontend.handle.array import VectorView

        if isinstance(template_handle, VectorView):
            assert isinstance(value, ArrayValue)
            view = VectorView._wrap_unregistered(
                parent=template_handle._slice_parent,
                sliced_av=value,
                length=template_handle._shape[0],
                start_uint=template_handle._slice_start,
                step_uint=template_handle._slice_step,
            )
            view._slice_covered_indices = template_handle._slice_covered_indices
            view._slice_outer_view = template_handle._slice_outer_view
            return view
        cls = type(template_handle)
        assert isinstance(value, ArrayValue)
        return cls._create_from_value(value=value, shape=template_handle._shape)
    raise TypeError(
        "Unsupported Handle type for if-else phi merge: "
        f"{type(template_handle).__name__}. Add explicit handle wrapping support "
        "before merging this handle type."
    )


def _copy_qfixed_phi_metadata(
    phi_output: Value,
    true_v: Value,
    false_v: Value,
) -> Value:
    """Copy QFixed carrier metadata onto a compatible phi result.

    QFixed is a scalar quantum handle backed by multiple physical qubit
    carriers recorded in metadata. A merged QFixed can only reuse that metadata
    when both branches describe the exact same carrier layout; otherwise the
    frontend cannot represent the condition-dependent carrier set safely.

    Args:
        phi_output (Value): Newly-created phi result value.
        true_v (Value): True-branch QFixed value.
        false_v (Value): False-branch QFixed value.

    Returns:
        Value: ``phi_output`` with QFixed metadata copied when applicable.

    Raises:
        TypeError: If either branch lacks QFixed metadata or the branch carrier
            layouts differ.
    """
    if not isinstance(true_v.type, QFixedType):
        return phi_output

    true_meta = true_v.metadata.qfixed
    false_meta = false_v.metadata.qfixed
    if true_meta is None or false_meta is None:
        raise TypeError(
            "QFixed if-else phi merge requires QFixed metadata on both branches."
        )
    if (
        true_meta.qubit_uuids != false_meta.qubit_uuids
        or true_meta.num_bits != false_meta.num_bits
        or true_meta.int_bits != false_meta.int_bits
    ):
        raise TypeError(
            "QFixed if-else phi merge requires identical carrier qubits and "
            "fixed-point layout across branches."
        )
    return phi_output.with_qfixed_metadata(
        qubit_uuids=true_meta.qubit_uuids,
        num_bits=true_meta.num_bits,
        int_bits=true_meta.int_bits,
    )


def _fresh_handle_copy_for_tracing(h: typing.Any) -> typing.Any:
    """Create a Handle copy with consumed state reset for branch tracing.

    This function intentionally accesses Handle's private ``_consumed`` and
    ``_consumed_by`` attributes.  This is the **only** place where such access
    is acceptable: if-else branches are mutually exclusive, so both must be
    traceable independently.  Exposing a general-purpose copy method on Handle
    would undermine the affine-type enforcement that prevents qubit reuse bugs.

    Non-Handle values (int, float, etc.) are returned unchanged.
    """
    if not isinstance(h, Handle):
        return h
    c = copy.copy(h)
    c._consumed = False
    c._consumed_by = None
    # Reset borrowed-element tracking for ArrayBase instances so that
    # each branch starts with an empty borrow set.  Without this,
    # shallow copy shares the same _borrowed_indices dict and borrowing
    # an element in one branch would cause QubitConsumedError in the other.
    if isinstance(c, ArrayBase):
        c._borrowed_indices = {}
    return c


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


def loop_rebind_snapshot(named_handles: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Snapshot pre-loop variable handles for rebind detection.

    Called from AST-injected probe code as the first statement of a traced
    loop body. The snapshot records which handle each candidate variable
    name pointed at before the body ran, so ``record_loop_rebinds`` can
    detect rebinds by comparing IR value identity afterwards.

    Args:
        named_handles (dict[str, typing.Any]): Candidate variable names
            mapped to their current handles (or plain Python values).

    Returns:
        dict[str, typing.Any]: A shallow copy of ``named_handles``.
    """
    return dict(named_handles)


def record_loop_rebinds(
    snapshot: dict[str, typing.Any],
    named_handles: dict[str, typing.Any],
) -> None:
    """Record classical scalar rebinds on the current loop-body tracer.

    Called from AST-injected probe code as the last statement of a traced
    loop body. For each candidate variable whose handle now carries a
    different IR value than the pre-loop snapshot, and whose post-body
    value is a classical scalar (``UInt`` / ``Float`` / ``Bit``), a
    :class:`LoopCarriedRebind` record is stashed on the active body
    tracer. The loop builders copy the records onto the loop operation,
    where the transpiler's rejection pass reads them.

    No IR operations are emitted; this only annotates the tracer.

    Args:
        snapshot (dict[str, typing.Any]): Pre-loop handles from
            ``loop_rebind_snapshot``.
        named_handles (dict[str, typing.Any]): The same variable names
            mapped to their post-body handles.
    """
    records: list[LoopCarriedRebind] = []
    for name, after in named_handles.items():
        if name not in snapshot:
            continue
        before = snapshot[name]
        if before is after:
            continue
        after_value = _ir_value_from_handle_like(after)
        if after_value is None or isinstance(after_value, ArrayValue):
            continue
        if after_value.type.is_quantum():
            continue
        before_synthesized = False
        before_value = _ir_value_from_handle_like(before)
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
        tracer = get_current_tracer()
        tracer.loop_carried_rebinds = tracer.loop_carried_rebinds + tuple(records)


def _create_phi_for_values(
    condition_value: Value,
    true_val: typing.Any,
    false_val: typing.Any,
    if_operation: IfOperation,
) -> typing.Tuple[Value, Handle]:
    """Create a Phi operation for merging branch values.

    Args:
        condition_value: The condition Value (from if statement)
        true_val: Value from true branch (Handle, Value, or primitive)
        false_val: Value from false branch (Handle, Value, or primitive)
        if_operation: The IfOperation to add Phi result to

    Returns:
        Tuple of (phi_output_value, merged_handle)
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

    # Create Phi output value (indexed to avoid name collisions)
    phi_index = len(if_operation.results)
    if isinstance(true_v, ArrayValue):
        phi_output = ArrayValue(
            type=true_v.type,
            name=f"{true_v.name}_phi_{phi_index}",
            shape=true_v.shape,
            slice_of=true_v.slice_of,
            slice_start=true_v.slice_start,
            slice_step=true_v.slice_step,
        )
    else:
        phi_output = Value(type=true_v.type, name=f"{true_v.name}_phi_{phi_index}")
        phi_output = _copy_qfixed_phi_metadata(phi_output, true_v, false_v)

    # Create PhiOp and store in IfOperation
    _phi_op = PhiOp(operands=[condition_value, true_v, false_v], results=[phi_output])
    if_operation.phi_ops.append(_phi_op)
    if_operation.results.append(phi_output)

    # Create appropriate Handle type for the merged value
    merged_handle = _create_handle_from_value(phi_output, true_val)
    _refresh_slice_phi_owner(true_val, false_val, merged_handle)

    return phi_output, merged_handle


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


def _refresh_slice_phi_owner(
    true_val: typing.Any,
    false_val: typing.Any,
    merged_handle: typing.Any,
) -> None:
    """Transfer same-slice branch ownership to a merged slice phi handle.

    Runtime ``if`` tracing uses branch-local handle copies.  When both branches
    return the same parent slice lineage, the merged phi view is the handle that
    should be assignable after the ``if``.  Refreshing the parent borrow table
    here keeps the frontend owner identity aligned with that merged handle.

    Args:
        true_val (typing.Any): True-branch value participating in the phi.
        false_val (typing.Any): False-branch value participating in the phi.
        merged_handle (typing.Any): Handle wrapping the phi result.
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
    phi merely because both branch handles still point at the same outer array.

    Args:
        op (typing.Any): Operation-like object to inspect.  It may be a normal
            IR operation or a control-flow operation with nested operation lists.
        array_value (ArrayValue): Array whose element-level usage is being
            detected.

    Returns:
        bool: ``True`` if ``op`` or any nested operation reads/writes an element
        whose ``parent_array`` is ``array_value``; ``False`` otherwise.
    """
    values = [*getattr(op, "operands", ()), *getattr(op, "results", ())]
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
    locals defined in both branches.  For array no-op phi elision we still want
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


def branch_rebind_pre_bindings(
    frame_locals: dict[str, typing.Any],
    names: tuple,
) -> dict[str, typing.Any]:
    """Capture pre-branch bindings for if-rebind records.

    Called from AST-injected code at the ``emit_if`` call site with the
    caller's ``locals()``. The transformer's rebind-candidate analysis is
    lexical, so a candidate name may not actually be bound at runtime
    (a preceding pure-store if can be dead-store-eliminated from its
    outputs); unbound names are silently skipped instead of raising
    ``UnboundLocalError`` at the call site.

    Args:
        frame_locals (dict[str, typing.Any]): The caller's ``locals()``.
        names (tuple): Candidate variable names to capture.

    Returns:
        dict[str, typing.Any]: The bound candidate names mapped to their
            current (pre-branch) handles.
    """
    return {name: frame_locals[name] for name in names if name in frame_locals}


def _collect_branch_rebinds(
    output_names: tuple,
    rebind_pre_bindings: dict | None,
    true_result: tuple,
    false_result: tuple,
) -> tuple[BranchRebind, ...]:
    """Record quantum variables whose binding changed in an if branch.

    Compares each candidate variable's pre-branch IR value (captured at
    the ``emit_if`` call site by the AST transformer) with the value the
    variable holds after each branch. The phi merge only carries the
    *new* branch values — and a reassigned variable whose old value is
    dead is not even passed into the branches — so the pre-branch value
    can disappear from the ``IfOperation`` entirely; these records
    preserve it for the transpiler's branch-discard check. Classical
    variables are not recorded — classical rebinds are ordinary
    phi-merged dataflow.

    Args:
        output_names (tuple): Variable names positionally aligned with
            the branch-result tuples (the AST transformer's output list).
        rebind_pre_bindings (dict | None): Candidate variable names
            mapped to their pre-branch handles, captured at the call
            site. None or empty when no pre-existing variable is
            reassigned in a branch.
        true_result (tuple): Variable values returned by the true branch,
            positionally aligned with ``output_names``.
        false_result (tuple): Variable values returned by the false
            branch, positionally aligned with ``output_names``.

    Returns:
        tuple[BranchRebind, ...]: One record per quantum variable whose
            binding changed in at least one branch. Empty when the
            result tuples do not align with ``output_names`` (defensive —
            the AST transformer guarantees alignment).
    """
    if not rebind_pre_bindings:
        return ()
    if not (len(output_names) == len(true_result) == len(false_result)):
        return ()
    positions = {name: index for index, name in enumerate(output_names)}
    records: list[BranchRebind] = []
    for name, pre_handle in rebind_pre_bindings.items():
        index = positions.get(name)
        if index is None:
            continue
        pre_value = pre_handle.value if hasattr(pre_handle, "value") else pre_handle
        if not isinstance(pre_value, Value) or not pre_value.type.is_quantum():
            continue
        true_val = true_result[index]
        false_val = false_result[index]
        true_value = true_val.value if hasattr(true_val, "value") else true_val
        false_value = false_val.value if hasattr(false_val, "value") else false_val
        if not isinstance(true_value, Value) or not isinstance(false_value, Value):
            continue
        # Compare logical_id, not uuid: a gate self-update (``q = qmc.x(q)``)
        # produces a new SSA version with a fresh uuid but the SAME
        # logical_id (same quantum wire), which is not a discard. Only a
        # substitution to a different quantum resource — a fresh allocation
        # or another existing register — changes the logical_id, and only
        # those are genuine branch rebinds worth recording.
        rebound_in_true = true_value.logical_id != pre_value.logical_id
        rebound_in_false = false_value.logical_id != pre_value.logical_id
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
) -> typing.Any:
    """Builder function for if-else conditional with Phi function merging.

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

    Returns:
        Merged variable values after conditional execution (using Phi functions)

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
    true_tracer, true_result = _trace_branch(true_func, true_vars)
    false_tracer, false_result = _trace_branch(false_func, false_vars)

    # 3. Create IfOperation
    if_op = IfOperation(
        true_operations=true_tracer.operations,
        false_operations=false_tracer.operations,
    )
    if_op.operands.append(condition_value)

    # 4. Create Phi functions for each variable to merge branches
    # Note: The AST transformer guarantees both branches return the same
    # variable list in the same order, so true_val and false_val always
    # have the same type.
    if len(true_result) != len(false_result):
        raise ValueError(
            f"Branch result length mismatch: true={len(true_result)}, false={len(false_result)}"
        )
    if_op.branch_rebinds = _collect_branch_rebinds(
        output_names, rebind_pre_bindings, true_result, false_result
    )
    merged_results = []
    for true_val, false_val in zip(true_result, false_result, strict=True):
        if isinstance(true_val, (Handle, Value)):
            if not isinstance(false_val, (Handle, Value)):
                raise TypeError(
                    f"Branch value mismatch in phi merge: "
                    f"true branch returned {type(true_val).__name__}, "
                    f"but false branch returned {type(false_val).__name__}. "
                    f"Both branches of an if-else must return the same variables."
                )
            # Phi minimization (scalar handles only): when both branches
            # return the *same* IR Value AND neither branch consumed it,
            # the phi would be a no-op merge. Skipping it keeps the IR
            # SSA-minimal and avoids generating phi-versioned aliases
            # (e.g. ``j_phi_4``) for read-only scalar loop variables,
            # which downstream emit-time loop unrolling cannot bind.
            #
            # Array handles normally need a phi because whole-array ops
            # can return a fresh ``ArrayValue``.  When both branches
            # still point at the exact same ``ArrayValue`` and neither
            # branch consumed it, however, the array handle itself did
            # not change.  Reuse the original outer handle so parent
            # borrow tables (for live slice views captured outside the
            # branch) are not replaced by a phi-root with empty state.
            from qamomile.circuit.frontend.handle.array import ArrayBase

            true_v = true_val.value if hasattr(true_val, "value") else true_val
            false_v = false_val.value if hasattr(false_val, "value") else false_val
            true_consumed = getattr(true_val, "_consumed", False)
            false_consumed = getattr(false_val, "_consumed", False)
            is_array_handle = isinstance(true_val, ArrayBase) or isinstance(
                false_val, ArrayBase
            )
            if (
                not is_array_handle
                and isinstance(true_v, Value)
                and isinstance(false_v, Value)
                and true_v is false_v
                and not true_consumed
                and not false_consumed
            ):
                merged_results.append(true_val)
                continue
            if (
                is_array_handle
                and isinstance(true_v, ArrayValue)
                and isinstance(false_v, ArrayValue)
                and true_v.type.is_quantum()
                and true_v is false_v
                and not true_consumed
                and not false_consumed
                and not _branch_touches_array_elements(true_tracer, true_v)
                and not _branch_touches_array_elements(false_tracer, false_v)
            ):
                original_val = _find_original_handle_for_result(true_val, variables)
                merged_results.append(
                    original_val if original_val is not None else true_val
                )
                continue
            phi_output, merged_handle = _create_phi_for_values(
                condition_value, true_val, false_val, if_op
            )
            merged_results.append(merged_handle)
        elif isinstance(false_val, (Handle, Value)):
            raise TypeError(
                f"Branch value mismatch in phi merge: "
                f"false branch returned {type(false_val).__name__}, "
                f"but true branch returned {type(true_val).__name__}. "
                f"Both branches of an if-else must return the same variables."
            )
        else:
            # Non-Handle/Value values (int, float, etc.) don't need phi
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

    # Create ForItemsOperation with captured body operations
    for_items_op = ForItemsOperation(
        key_vars=key_var_names,
        value_var=value_var_name,
        key_is_vector=_key_is_vector,
        key_var_values=tuple(key_var_values),
        value_var_value=value_var_value,
        operations=body_tracer.operations,
        loop_carried_rebinds=body_tracer.loop_carried_rebinds,
    )
    for_items_op.operands.append(d.value)  # type: ignore[arg-type]  # DictValue is not Value but stored as operand

    parent_tracer.add_operation(for_items_op)
