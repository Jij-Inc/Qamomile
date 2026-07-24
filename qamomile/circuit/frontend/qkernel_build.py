"""Build and tracing helpers for QKernel objects."""

from __future__ import annotations

import inspect
from typing import Any, cast

from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.func_to_block import (
    _validate_returned_arrays,
    build_param_slots,
    create_dummy_input,
    is_array_type,
)
from qamomile.circuit.frontend.handle import Observable
from qamomile.circuit.frontend.param_validation import (
    validate_bindings_parameters_disjoint,
)
from qamomile.circuit.frontend.qkernel_definition import (
    refresh_qkernel_function_namespace,
)
from qamomile.circuit.frontend.qkernel_inputs import (
    auto_detect_parameters,
    create_bound_input,
    create_parameter_input,
    validate_kwargs,
    validate_parameters,
)
from qamomile.circuit.frontend.qkernel_metadata import extract_return_names
from qamomile.circuit.frontend.qkernel_utils import get_array_element_type
from qamomile.circuit.frontend.static_binding import (
    is_static_binding_annotation,
    validate_static_binding_argument,
)
from qamomile.circuit.frontend.tracer import Tracer, trace
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.value import Value, ValueLike


def build_specialized_block(
    kernel: Any,
    *,
    parameters: list[str],
    bindings: dict[str, Any],
    qubit_sizes: dict[str, int],
) -> Block:
    """Trace a specialized sub-block for a call site.

    Args:
        kernel (Any): QKernel-like object to trace.
        parameters (list[str]): Classical argument names that remain symbolic
            in the specialized block.
        bindings (dict[str, Any]): Concrete Python values for classical
            arguments and caller-owned proxies for unresolved static bindings.
        qubit_sizes (dict[str, int]): First-axis sizes for ``Vector[Qubit]``
            arguments supplied by the caller.

    Returns:
        Block: Specialized hierarchical block ready to be invoked from the
        caller's trace.
    """
    validate_parameters(kernel.input_types, parameters)
    block = create_traced_block(
        kernel,
        parameters,
        bindings,
        qubit_sizes=qubit_sizes,
        emit_qubit_init=False,
        emit_return_op=True,
    )
    block.output_names = extract_return_names(kernel) or []
    return block


def create_traced_block(
    kernel: Any,
    parameters: list[str],
    kwargs: dict[str, Any],
    qubit_sizes: dict[str, int] | None = None,
    *,
    emit_qubit_init: bool = True,
    emit_return_op: bool = False,
) -> Block:
    """Trace a kernel and return a ``Block``.

    Args:
        kernel (Any): QKernel-like object to trace.
        parameters (list[str]): Argument names to keep as unbound parameters.
        kwargs (dict[str, Any]): Concrete values for non-parameter arguments
            and caller-owned proxies for unresolved static bindings.
        qubit_sizes (dict[str, int] | None): Optional mapping from
            ``Vector[Qubit]`` parameter names to integer sizes. Defaults to
            ``None``.
        emit_qubit_init (bool): Whether quantum-array size entries should emit
            ``QInitOperation``. Defaults to ``True``.
        emit_return_op (bool): Whether to append an explicit
            ``ReturnOperation`` for inline-call specialization. Defaults to
            ``False``.

    Returns:
        Block: Traced block with label arguments, inputs, outputs, and
        parameter slots populated.

    Raises:
        UnreturnedBorrowError: If a returned quantum array still has
            outstanding element/view borrows at trace end (whether or
            not the borrowed element is co-returned with it).
        TypeError: If a static binding declares a default, a concrete static
            binding has the wrong registered type, or a symbolic static
            binding does not preserve the parameter's slot identity.
        ValueError: If a required static binding is absent from ``kwargs``.
    """
    if qubit_sizes is None:
        qubit_sizes = {}

    tracer = Tracer()
    tracked_parameters: dict[str, Value] = {}

    refresh_qkernel_function_namespace(kernel)

    with trace(tracer):
        dummy_inputs: dict[str, Any] = {}

        for name, param in kernel.signature.parameters.items():
            param_type = kernel.input_types.get(name, param.annotation)

            if is_static_binding_annotation(param_type):
                if param.default is not inspect.Parameter.empty:
                    raise TypeError(
                        f"Static binding parameter {name!r} cannot have a "
                        "default value; provide it through bindings when "
                        "building or transpiling the qkernel."
                    )
                if name not in kwargs:
                    raise ValueError(
                        f"Static binding argument {name!r} must be provided "
                        "through bindings."
                    )
                handle = validate_static_binding_argument(
                    param_type,
                    name,
                    kwargs[name],
                )
            elif param_type is Observable or (
                is_array_type(param_type)
                and get_array_element_type(param_type) is Observable
                and name not in kwargs
            ):
                handle = create_parameter_input(param_type, name)
                tracked_parameters[name] = handle.value
            elif name in parameters:
                if qubit_sizes and is_array_type(param_type):
                    handle = create_dummy_input(param_type, name)
                else:
                    handle = create_parameter_input(param_type, name)
                tracked_parameters[name] = handle.value
            elif name in qubit_sizes:
                if emit_qubit_init:
                    handle = qubit_array(qubit_sizes[name], name)
                else:
                    handle = create_dummy_input(
                        param_type,
                        name,
                        emit_init=False,
                        shape=(qubit_sizes[name],),
                    )
            elif name in kwargs:
                handle = create_bound_input(param_type, name, kwargs[name])
            elif param.default is not inspect.Parameter.empty:
                handle = create_bound_input(param_type, name, param.default)
            else:
                handle = create_dummy_input(param_type, name, emit_init=emit_qubit_init)

            dummy_inputs[name] = handle

        result = kernel.func(**dummy_inputs)
        # Same trace-end borrow validation as ``func_to_block``: this is
        # the second of the two trace paths (``.build()`` / call-time
        # specialization), and without the check here a sub-kernel could
        # return a parent register with outstanding element borrows —
        # including a still-borrowed element returned ALONGSIDE its
        # parent, which hands the caller two handles onto one physical
        # slot. Co-returning the borrowed element does not exempt the
        # parent: the alias escapes either way.
        _validate_returned_arrays(result)
        output_values = _extract_output_values(result)
        if emit_return_op:
            tracer.add_operation(
                ReturnOperation(
                    operands=cast(list[Value], output_values),
                    results=[],
                )
            )

    ordinary_inputs = {
        name: handle
        for name, handle in dummy_inputs.items()
        if not is_static_binding_annotation(kernel.input_types[name])
    }
    input_values: list[ValueLike] = [
        cast(ValueLike, handle.value) for handle in ordinary_inputs.values()
    ]
    param_slots = build_param_slots(
        signature=kernel.signature,
        input_types=kernel.input_types,
        parameters=parameters,
        kwargs=kwargs,
        qubit_sizes=qubit_sizes,
        bind_defaults=True,
    )

    return Block(
        operations=tracer.operations,
        label_args=list(ordinary_inputs),
        input_values=input_values,
        output_values=output_values,
        name=kernel.name,
        parameters=tracked_parameters,
        kind=BlockKind.TRACED,
        param_slots=param_slots,
    )


def build_qkernel(
    kernel: Any,
    parameters: list[str] | None = None,
    **kwargs: Any,
) -> Block:
    """Build a traced block from a qkernel.

    Args:
        kernel (Any): QKernel-like object to trace.
        parameters (list[str] | None): Argument names to preserve as runtime
            parameters. Defaults to ``None``, which auto-detects parameters.
        **kwargs (Any): Concrete values for non-parameter arguments.

    Returns:
        Block: Traced block ready for transpilation, estimation, or
        visualization.

    Raises:
        TypeError: If a non-parameterizable type is listed as a parameter.
        ValueError: If required arguments are missing, or if a name appears in
            both ``parameters`` and ``kwargs`` (the compile-time-bound values),
            which violates the bindings/parameters disjointness rule.
    """
    # Enforce the bindings/parameters disjointness rule against the
    # *user-provided* ``parameters`` before auto-detection. Auto-detect only
    # ever picks names absent from ``kwargs``, so it can never introduce an
    # overlap; the ambiguous case is exactly a name the caller listed in
    # ``parameters`` while also passing a concrete value for it in ``kwargs``.
    validate_bindings_parameters_disjoint(kwargs, parameters)

    if parameters is None:
        parameters = auto_detect_parameters(
            kernel.signature, kernel.input_types, kwargs
        )

    validate_parameters(kernel.input_types, parameters)
    validate_kwargs(kernel.signature, kernel.input_types, parameters, kwargs)

    block = create_traced_block(kernel, parameters, kwargs)
    block.output_names = extract_return_names(kernel) or []
    return block


def _extract_output_values(result: Any) -> list[ValueLike]:
    """Extract IR output values from a qkernel return object.

    Args:
        result (Any): Return value from the traced qkernel body.

    Returns:
        list[ValueLike]: Values carried by handle-like return objects.
    """
    output_values: list[ValueLike] = []
    if result is None:
        return output_values
    if isinstance(result, tuple):
        for item in result:
            if hasattr(item, "value"):
                output_values.append(cast(ValueLike, item.value))
        return output_values
    if hasattr(result, "value"):
        output_values.append(cast(ValueLike, result.value))
    return output_values
