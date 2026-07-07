"""Build and tracing helpers for QKernel objects."""

from __future__ import annotations

import inspect
from typing import Any

from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.func_to_block import (
    build_param_slots,
    create_dummy_input,
    is_array_type,
)
from qamomile.circuit.frontend.handle import Observable
from qamomile.circuit.frontend.handle.primitives import Handle
from qamomile.circuit.frontend.qkernel_inputs import (
    auto_detect_parameters,
    create_bound_input,
    create_parameter_input,
    validate_kwargs,
    validate_parameters,
)
from qamomile.circuit.frontend.qkernel_metadata import extract_return_names
from qamomile.circuit.frontend.qkernel_utils import get_array_element_type
from qamomile.circuit.frontend.tracer import Tracer, trace
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.value import Value


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
            arguments.
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
        kwargs (dict[str, Any]): Concrete values for non-parameter arguments.
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
    """
    if qubit_sizes is None:
        qubit_sizes = {}

    tracer = Tracer()
    tracked_parameters: dict[str, Value] = {}

    with trace(tracer):
        dummy_inputs: dict[str, Handle] = {}

        for name, param in kernel.signature.parameters.items():
            param_type = kernel.input_types.get(name, param.annotation)

            is_scalar_observable = param_type is Observable
            is_unbound_observable_array = (
                is_array_type(param_type)
                and get_array_element_type(param_type) is Observable
                and name not in kwargs
            )
            if is_scalar_observable or is_unbound_observable_array:
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
        output_values = _extract_output_values(result)
        if emit_return_op:
            tracer.add_operation(ReturnOperation(operands=output_values, results=[]))

    input_values = [handle.value for handle in dummy_inputs.values()]
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
        label_args=list(dummy_inputs.keys()),
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
        ValueError: If required arguments are missing.
    """
    if parameters is None:
        parameters = auto_detect_parameters(
            kernel.signature, kernel.input_types, kwargs
        )

    validate_parameters(kernel.input_types, parameters)
    validate_kwargs(kernel.signature, kernel.input_types, parameters, kwargs)

    block = create_traced_block(kernel, parameters, kwargs)
    block.output_names = extract_return_names(kernel) or []
    return block


def _extract_output_values(result: Any) -> list[Value]:
    """Extract IR output values from a qkernel return object.

    Args:
        result (Any): Return value from the traced qkernel body.

    Returns:
        list[Value]: Values carried by handle-like return objects.
    """
    output_values: list[Value] = []
    if result is None:
        return output_values
    if isinstance(result, tuple):
        for item in result:
            if hasattr(item, "value"):
                output_values.append(item.value)
        return output_values
    if hasattr(result, "value"):
        output_values.append(result.value)
    return output_values
