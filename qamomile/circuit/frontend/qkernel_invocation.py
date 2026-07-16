"""Call-time invocation logic for ``QKernel`` objects."""

from __future__ import annotations

from typing import Any, cast

from qamomile.circuit.frontend.func_to_block import (
    is_array_type,
    is_dict_type,
    is_tuple_type,
)
from qamomile.circuit.frontend.handle.array import ArrayBase, VectorView
from qamomile.circuit.frontend.handle.containers import Dict, Tuple
from qamomile.circuit.frontend.handle.primitives import Handle, UInt
from qamomile.circuit.frontend.param_validation import _validate_bound_handles
from qamomile.circuit.frontend.qkernel_callable import qkernel_invoke_block
from qamomile.circuit.frontend.qkernel_self_call import emit_self_call_forward_ref
from qamomile.circuit.frontend.qkernel_specialization import (
    select_specialized_block,
)
from qamomile.circuit.frontend.qkernel_utils import (
    is_full_reslice_of_input,
    promote_literal_to_handle,
    reject_aliased_quantum_args,
    reject_consumed_view_arg,
    view_result_value_for_full_reslice,
)
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueLike,
)

InputViewKey = tuple[str, str | None, str | None, str | None]


def _collect_input_view_metas(arguments: dict[str, Any]) -> dict[InputViewKey, Any]:
    """Collect live ``VectorView`` metadata before the call consumes inputs.

    Args:
        arguments (dict[str, Any]): Bound qkernel call arguments.

    Returns:
        dict[InputViewKey, Any]: Mapping from slice identity to the caller-side
        ``VectorView`` object.
    """
    input_view_metas: dict[InputViewKey, Any] = {}
    for handle in arguments.values():
        if isinstance(handle, VectorView) and handle._should_enforce_linear():
            root_av = handle.value
            while root_av.slice_of is not None:
                root_av = root_av.slice_of
            start_uuid = handle._slice_start.value.uuid if handle._slice_start else None
            step_uuid = handle._slice_step.value.uuid if handle._slice_step else None
            length_uuid = handle.value.shape[0].uuid if handle.value.shape else None
            input_view_metas[
                (root_av.logical_id, start_uuid, step_uuid, length_uuid)
            ] = handle
    return input_view_metas


def _prepare_call_inputs(
    kernel: Any,
    arguments: dict[str, Any],
    reject_consumed_view_arg: Any,
) -> tuple[
    dict[str, ValueLike],
    dict[str, tuple[Any, tuple, Handle]],
    dict[InputViewKey, Any],
]:
    """Validate qkernel call arguments and collect input metadata.

    Args:
        kernel (Any): ``QKernel`` instance being invoked.
        arguments (dict[str, Any]): Bound and literal-promoted arguments.
        reject_consumed_view_arg (Any): Helper that rejects stale view
            arguments.

    Returns:
        tuple[dict[str, ValueLike], dict[str, tuple[Any, tuple, Handle]],
        dict[InputViewKey, Any]]: Input values by parameter name,
        scalar-borrow provenance with each current owner by logical id, and
        vector-view metadata by slice key.

    Raises:
        TypeError: If an argument is not a frontend ``Handle``.
    """
    inputs_map: dict[str, ValueLike] = {}
    provenance_map: dict[str, tuple[Any, tuple, Handle]] = {}

    for name, handle in arguments.items():
        if not isinstance(handle, Handle):
            raise TypeError(
                f"Argument '{name}' must be a Handle instance, got {type(handle)}"
            )
        if (
            handle.parent is not None
            and not is_array_type(type(handle))
            and handle._should_enforce_linear()
        ):
            provenance_map[handle.value.logical_id] = (
                handle.parent,
                handle.indices,
                handle,
            )

    input_view_metas = _collect_input_view_metas(arguments)

    for name, handle in arguments.items():
        if not isinstance(handle, Handle):
            continue
        if isinstance(handle, VectorView) and handle._should_enforce_linear():
            reject_consumed_view_arg(kernel.name, handle)
        if handle._should_enforce_linear():
            handle.validate_consumable(operation_name=f"QKernel[{kernel.name}]")
        inputs_map[name] = handle.value

    return inputs_map, provenance_map, input_view_metas


def _commit_call_inputs(
    kernel: Any,
    arguments: dict[str, Any],
    provenance_map: dict[str, tuple[Any, tuple, Handle]],
) -> None:
    """Commit affine ownership only after call construction succeeds.

    Args:
        kernel (Any): ``QKernel`` instance being invoked.
        arguments (dict[str, Any]): Validated bound call arguments.
        provenance_map (dict[str, tuple[Any, tuple, Handle]]): Direct scalar
            borrow provenance to update with each consume successor.

    Returns:
        None.
    """
    operation_name = f"QKernel[{kernel.name}]"
    for handle in arguments.values():
        if (
            not isinstance(handle, Handle)
            or not handle._should_enforce_linear()
            or isinstance(handle, VectorView)
        ):
            continue
        consumed = handle.consume(operation_name=operation_name)
        provenance = provenance_map.get(handle.value.logical_id)
        if provenance is not None:
            parent, indices, _ = provenance
            provenance_map[handle.value.logical_id] = (parent, indices, consumed)


def _select_call_operation(
    kernel: Any,
    arguments: dict[str, Any],
    inputs_map: dict[str, ValueLike],
    invoke_block_factory: Any | None = None,
) -> tuple[Any, Block | None, dict[str, tuple[ArrayValue, Any]]]:
    """Build the invocation operation for a direct or self-recursive call.

    Args:
        kernel (Any): ``QKernel`` instance being invoked.
        arguments (dict[str, Any]): Bound frontend arguments.
        inputs_map (dict[str, ValueLike]): Validated input values by parameter
            name. Ownership is not committed yet.
        invoke_block_factory (Any | None): Optional callable used to create
            the invocation for a selected block. Defaults to a qkernel
            ``InvokeOperation`` factory.
    Returns:
        tuple[Any, Block | None, dict[str, tuple[ArrayValue, Any]]]: Invocation
            operation, selected block if available, and caller view metadata
            for formal inputs.
    """
    if kernel._block_building:
        return emit_self_call_forward_ref(kernel, inputs_map), None, {}

    block_ir = select_specialized_block(kernel, arguments)

    formal_input_views: dict[str, tuple[ArrayValue, Any]] = {}
    for label, formal_input in zip(block_ir.label_args, block_ir.input_values):
        actual_handle = arguments.get(label)
        if (
            isinstance(actual_handle, VectorView)
            and isinstance(formal_input, ArrayValue)
            and actual_handle._should_enforce_linear()
        ):
            formal_input_views[formal_input.logical_id] = (formal_input, actual_handle)

    if invoke_block_factory is None:

        def invoke_block_factory(block: Block, inputs: dict[str, ValueLike]) -> Any:
            """Create the default qkernel invocation operation.

            Args:
                block (Block): Selected qkernel implementation block.
                inputs (dict[str, ValueLike]): Actual input values keyed by
                    callee parameter name.

            Returns:
                Any: Inline-by-default qkernel invocation operation.
            """
            return qkernel_invoke_block(kernel, block, inputs)

    return invoke_block_factory(block_ir, inputs_map), block_ir, formal_input_views


def _wrap_array_result(
    *,
    kernel: Any,
    call_op: Any,
    result_idx: int,
    val: Any,
    handle_type: Any,
    block_ir_for_call: Block | None,
    formal_input_views: dict[str, tuple[ArrayValue, Any]],
    input_view_metas: dict[InputViewKey, Any],
) -> Any:
    """Wrap an array result value back into a frontend handle.

    Args:
        kernel (Any): ``QKernel`` instance being invoked.
        call_op (Any): Invocation operation whose result may be adjusted.
        result_idx (int): Result position being wrapped.
        val (Any): IR result value.
        handle_type (Any): Declared frontend result handle type.
        block_ir_for_call (Block | None): Selected callee block.
        formal_input_views (dict[str, tuple[ArrayValue, Any]]): Formal input
            views keyed by logical id.
        input_view_metas (dict[InputViewKey, Any]): Caller view metadata.

    Returns:
        Any: Wrapped frontend array handle.

    Raises:
        RuntimeError: If a sliced result lacks shape metadata.
    """
    actual_class = cast(
        type[ArrayBase[Any]],
        getattr(handle_type, "__origin__", handle_type),
    )
    assert isinstance(val, ArrayValue)

    formal_output = (
        block_ir_for_call.output_values[result_idx]
        if block_ir_for_call is not None
        and result_idx < len(block_ir_for_call.output_values)
        else None
    )
    if isinstance(formal_output, ArrayValue) and formal_output.slice_of is not None:
        for formal_input, input_view in formal_input_views.values():
            if is_full_reslice_of_input(formal_output, formal_input):
                val = view_result_value_for_full_reslice(val, input_view)
                call_op.results[result_idx] = val
                break

    shape = tuple(UInt(value=dim_val) for dim_val in val.shape) if val.shape else ()
    if val.slice_of is None:
        return actual_class._create_from_value(value=val, shape=shape)

    result_root_av = val
    while result_root_av.slice_of is not None:
        result_root_av = result_root_av.slice_of
    result_start_uuid = val.slice_start.uuid if val.slice_start else None
    result_step_uuid = val.slice_step.uuid if val.slice_step else None
    result_length_uuid = val.shape[0].uuid if val.shape else None
    meta_key = (
        result_root_av.logical_id,
        result_start_uuid,
        result_step_uuid,
        result_length_uuid,
    )
    in_view = input_view_metas.get(meta_key)
    if in_view is not None and in_view._slice_parent is not None:
        if shape:
            length = shape[0]
        elif val.shape:
            length = UInt(value=val.shape[0])
        else:
            raise RuntimeError(f"Slice result '{val.name}' has no shape metadata.")
        new_view = VectorView._wrap_unregistered(
            parent=in_view._slice_parent,
            sliced_av=val,
            length=length,
            start_uint=in_view._slice_start,
            step_uint=in_view._slice_step,
        )
        in_view._transfer_borrow_to(new_view, f"QKernel[{kernel.name}]")
        return new_view

    return actual_class._create_from_value(value=val, shape=shape)


def _wrap_nested_array_result(value: ArrayValue, handle_type: Any) -> ArrayBase[Any]:
    """Wrap a classical array nested inside a structural call result.

    Nested quantum arrays and views need ownership transfer through their
    containing structural handle. That ownership contract is not yet modeled,
    so this helper rejects them instead of producing an unsound alias.

    Args:
        value (ArrayValue): Nested array IR value.
        handle_type (Any): Declared array handle annotation.

    Returns:
        ArrayBase[Any]: Frontend handle for a plain classical array.

    Raises:
        RuntimeError: If the nested array is quantum-backed or is a view.
    """
    if value.type.is_quantum() or value.slice_of is not None:
        raise RuntimeError(
            "Quantum arrays and array views nested inside qmc.Tuple or "
            "qmc.Dict call results are not supported yet."
        )
    actual_class = cast(
        type[ArrayBase[Any]],
        getattr(handle_type, "__origin__", handle_type),
    )
    shape = tuple(UInt(value=dimension) for dimension in value.shape)
    return actual_class._create_from_value(value=value, shape=shape)


def _wrap_call_value(
    value: ValueLike,
    handle_type: Any,
    provenance_map: dict[str, tuple[Any, tuple, Handle]],
) -> Handle:
    """Recursively wrap one invocation result value as a frontend handle.

    Args:
        value (ValueLike): Caller-local invocation result value.
        handle_type (Any): Declared frontend handle annotation.
        provenance_map (dict[str, tuple[Any, tuple, Handle]]): Scalar borrow
            provenance and committed owner keyed by logical ID.

    Returns:
        Handle: Frontend handle matching ``handle_type``.

    Raises:
        RuntimeError: If the IR value graph does not match the declared return
            annotation or contains an unsupported nested quantum array/view.
    """
    if is_tuple_type(handle_type):
        if not isinstance(value, TupleValue):
            raise RuntimeError(
                f"Expected TupleValue for return type {handle_type!r}, "
                f"got {type(value).__name__}."
            )
        element_types = getattr(handle_type, "__args__", ())
        if len(value.elements) != len(element_types):
            raise RuntimeError(
                "Tuple call result does not match its return annotation: "
                f"expected {len(element_types)} elements, got "
                f"{len(value.elements)}."
            )
        elements = tuple(
            _wrap_call_value(element, element_type, provenance_map)
            for element, element_type in zip(
                value.elements,
                element_types,
                strict=True,
            )
        )
        actual_class = cast(
            type[Tuple[Any, Any]],
            getattr(handle_type, "__origin__", handle_type),
        )
        return actual_class(value=value, _elements=elements)

    if is_dict_type(handle_type):
        if not isinstance(value, DictValue):
            raise RuntimeError(
                f"Expected DictValue for return type {handle_type!r}, "
                f"got {type(value).__name__}."
            )
        key_type, entry_type = getattr(handle_type, "__args__", (None, None))
        entries = [
            (
                _wrap_call_value(key, key_type, provenance_map),
                _wrap_call_value(entry_value, entry_type, provenance_map),
            )
            for key, entry_value in value.entries
        ]
        actual_class = cast(
            type[Dict[Any, Any]],
            getattr(handle_type, "__origin__", handle_type),
        )
        return actual_class(
            value=value,
            _entries=entries,
            _key_type=key_type,
            _value_type=entry_type,
        )

    if is_array_type(handle_type):
        if not isinstance(value, ArrayValue):
            raise RuntimeError(
                f"Expected ArrayValue for return type {handle_type!r}, "
                f"got {type(value).__name__}."
            )
        return _wrap_nested_array_result(value, handle_type)

    if not isinstance(value, Value):
        raise RuntimeError(
            f"Expected scalar Value for return type {handle_type!r}, "
            f"got {type(value).__name__}."
        )
    if value.logical_id in provenance_map:
        parent, indices, intermediate = provenance_map.pop(value.logical_id)
        output = handle_type(value=value, parent=parent, indices=indices)
        intermediate._handoff_direct_borrow_to(output)
        return output
    return handle_type(value=value)


def _wrap_call_results(
    *,
    kernel: Any,
    call_op: Any,
    provenance_map: dict[str, tuple[Any, tuple, Handle]],
    input_view_metas: dict[InputViewKey, Any],
    block_ir_for_call: Block | None,
    formal_input_views: dict[str, tuple[ArrayValue, Any]],
) -> list[Any]:
    """Wrap invocation results back into frontend handles.

    Args:
        kernel (Any): ``QKernel`` instance being invoked.
        call_op (Any): Invocation operation.
        provenance_map (dict[str, tuple[Any, tuple, Handle]]): Scalar borrow
            provenance and committed owner keyed by logical id.
        input_view_metas (dict[InputViewKey, Any]): Caller view metadata.
        block_ir_for_call (Block | None): Selected callee block.
        formal_input_views (dict[str, tuple[ArrayValue, Any]]): Formal input
            view metadata.

    Returns:
        list[Any]: Wrapped frontend results.

    Raises:
        RuntimeError: If the operation result count mismatches the qkernel
            annotation.
    """
    results = call_op.results
    if len(results) != len(kernel.output_types):
        raise RuntimeError(
            f"Mismatch in return values: expected {len(kernel.output_types)}, "
            f"got {len(results)}"
        )

    wrapped_results: list[Any] = []
    for result_idx, (val, handle_type) in enumerate(zip(results, kernel.output_types)):
        if is_array_type(handle_type):
            wrapped_results.append(
                _wrap_array_result(
                    kernel=kernel,
                    call_op=call_op,
                    result_idx=result_idx,
                    val=val,
                    handle_type=handle_type,
                    block_ir_for_call=block_ir_for_call,
                    formal_input_views=formal_input_views,
                    input_view_metas=input_view_metas,
                )
            )
            continue

        wrapped_results.append(_wrap_call_value(val, handle_type, provenance_map))

    return wrapped_results


def invoke_qkernel_with_operation(
    kernel: Any,
    invoke_block_factory: Any | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Invoke a ``QKernel`` using a custom operation factory.

    Args:
        kernel (Any): ``QKernel`` instance.
        invoke_block_factory (Any | None): Optional callable that receives
            ``(block, inputs_map)`` and returns the invocation operation.
        *args (Any): Positional qkernel call arguments.
        **kwargs (Any): Keyword qkernel call arguments.

    Returns:
        Any: A single frontend handle or a tuple of frontend handles matching
        the qkernel return annotation.

    Raises:
        TypeError: If an argument is not a frontend handle after literal
            promotion.
        RuntimeError: If no qkernel tracer is active, or if the generated
            invocation result count does not match the qkernel return
            annotation.
    """
    tracer = get_current_tracer()
    bound_args = kernel.signature.bind(*args, **kwargs)
    bound_args.apply_defaults()

    for arg_name, arg_value in list(bound_args.arguments.items()):
        expected_type = kernel.input_types.get(arg_name)
        if expected_type is not None:
            bound_args.arguments[arg_name] = promote_literal_to_handle(
                arg_value, expected_type
            )

    _validate_bound_handles(
        kernel.input_types,
        bound_args.arguments,
        context=f"{kernel.name}()",
    )
    reject_aliased_quantum_args(kernel.name, bound_args.arguments)

    inputs_map, provenance_map, input_view_metas = _prepare_call_inputs(
        kernel,
        bound_args.arguments,
        reject_consumed_view_arg,
    )
    call_op, block_ir_for_call, formal_input_views = _select_call_operation(
        kernel,
        bound_args.arguments,
        inputs_map,
        invoke_block_factory,
    )

    if len(call_op.results) != len(kernel.output_types):
        raise RuntimeError(
            f"Mismatch in return values: expected {len(kernel.output_types)}, "
            f"got {len(call_op.results)}"
        )

    _commit_call_inputs(kernel, bound_args.arguments, provenance_map)

    tracer.add_operation(call_op)

    wrapped_results = _wrap_call_results(
        kernel=kernel,
        call_op=call_op,
        provenance_map=provenance_map,
        input_view_metas=input_view_metas,
        block_ir_for_call=block_ir_for_call,
        formal_input_views=formal_input_views,
    )

    for _parent, _indices, intermediate in provenance_map.values():
        intermediate.consume(operation_name="qkernel call (scalar dropped)")

    for in_view in input_view_metas.values():
        if not in_view._consumed:
            in_view.consume(operation_name="qkernel call (view dropped)")

    if len(wrapped_results) == 1:
        return wrapped_results[0]
    return tuple(wrapped_results)


def invoke_qkernel(kernel: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke a ``QKernel`` inside a tracing context.

    Args:
        kernel (Any): ``QKernel`` instance.
        *args (Any): Positional qkernel call arguments.
        **kwargs (Any): Keyword qkernel call arguments.

    Returns:
        Any: A single frontend handle or a tuple of frontend handles matching
        the qkernel return annotation.

    Raises:
        TypeError: If an argument is not a frontend handle after literal
            promotion.
        RuntimeError: If no qkernel tracer is active, or if the generated
            invocation result count does not match the qkernel return
            annotation.
    """
    return invoke_qkernel_with_operation(kernel, None, *args, **kwargs)
