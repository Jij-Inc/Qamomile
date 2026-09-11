"""Call-time specialization extraction for qkernel calls."""

from __future__ import annotations

from typing import Any, cast

from qamomile.circuit.frontend.func_to_block import (
    is_array_type,
    is_dict_type,
    is_tuple_type,
)
from qamomile.circuit.frontend.handle import Observable, QInt, Qubit
from qamomile.circuit.frontend.handle.array import Vector
from qamomile.circuit.frontend.handle.primitives import Bit, Float, Handle, UInt
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.frontend.qkernel_build import build_specialized_block
from qamomile.circuit.frontend.qkernel_inputs import is_parameterizable_type
from qamomile.circuit.frontend.qkernel_utils import get_array_element_type
from qamomile.circuit.frontend.static_binding import (
    is_static_binding_annotation,
    validate_static_binding_argument,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.types import QUIntType
from qamomile.circuit.ir.value import packed_register_type_width


def extract_calltime_specialization(
    kernel: Any,
    arguments: dict[str, Any],
) -> tuple[list[str], dict[str, Any], dict[str, int]] | None:
    """Extract specialization inputs for a qkernel call site.

    Args:
        kernel (Any): ``QKernel``-like object with ``signature`` and
            ``input_types`` attributes.
        arguments (dict[str, Any]): Bound call arguments after literal
            promotion and frontend validation. Registered static bindings
            remain concrete Python objects.

    Returns:
        tuple[list[str], dict[str, Any], dict[str, int]] | None: Runtime
        parameter names, compile-time bindings, and concrete qubit-array or
        QInt widths when specialization would change the callee trace; otherwise
        ``None``.
    """
    parameters: list[str] = []
    bindings: dict[str, Any] = {}
    qubit_sizes: dict[str, int] = {}
    has_qint = False

    for name, param in kernel.signature.parameters.items():
        param_type = kernel.input_types.get(name, param.annotation)
        argument = arguments.get(name)
        if is_static_binding_annotation(param_type):
            bindings[name] = validate_static_binding_argument(
                param_type,
                name,
                argument,
            )
            continue
        handle = argument
        assert isinstance(handle, Handle), (
            f"Internal invariant violated: argument {name!r} should already "
            f"be a Handle by the time extract_calltime_specialization runs."
        )

        if param_type is Qubit:
            continue
        if param_type is QInt:
            has_qint = True
            width = packed_register_type_width(handle.value.type)
            if width is not None:
                qubit_sizes[name] = width
            continue
        if param_type is Observable:
            continue
        if (
            is_array_type(param_type)
            and get_array_element_type(param_type) is Observable
        ):
            const_array = handle.value.get_const_array()
            if const_array is not None:
                bindings[name] = const_array
            continue

        if is_array_type(param_type) and get_array_element_type(param_type) is Qubit:
            if getattr(param_type, "__origin__", param_type) is not Vector:
                continue
            try:
                size = get_size(cast(Vector[Qubit], handle))
            except ValueError:
                continue
            qubit_sizes[name] = size
            continue

        if param_type in (int, UInt, float, Float, bool, Bit):
            const_value = handle.value.get_const()
            if const_value is not None:
                bindings[name] = const_value
            elif is_parameterizable_type(param_type):
                parameters.append(name)
            continue

        if is_array_type(param_type):
            const_array = handle.value.get_const_array()
            if const_array is not None:
                bindings[name] = const_array
            elif is_parameterizable_type(param_type):
                parameters.append(name)
            continue

        if is_dict_type(param_type):
            if handle.value.metadata.dict_runtime is None:
                continue
            bindings[name] = handle.value.get_bound_data()
            continue

        if is_tuple_type(param_type):
            continue

        return None

    if not bindings and not qubit_sizes and not has_qint:
        return None

    return parameters, bindings, qubit_sizes


def select_specialized_block(
    kernel: Any,
    arguments: dict[str, Any],
    *,
    require_handles: bool = True,
) -> Block:
    """Select the block implementation for a qkernel call site.

    Centralizes call-site specialization so plain qkernel calls, controlled
    calls, and inverse calls use the same rule. When concrete argument values
    would change the callee trace (for example a concrete ``Vector[Qubit]``
    size or a bound structural classical value), the function returns a
    temporary specialized block. QInt inputs also retain the caller's width
    type so symbolic dimensions stay connected across callable boundaries.
    Otherwise it returns the kernel's cached block.

    Args:
        kernel (Any): ``QKernel``-like object whose block should be selected.
        arguments (dict[str, Any]): Bound call arguments after literal
            promotion and frontend validation. Registered static bindings may
            remain concrete Python objects when ``require_handles`` is false.
        require_handles (bool): If ``True``, specialization is skipped unless
            every argument is a frontend ``Handle``. Defaults to ``True``.

    Returns:
        Block: Specialized call-site block or the cached kernel block.
    """
    if getattr(kernel, "_specializing", False):
        return kernel.block
    if require_handles and not all(
        isinstance(arg, Handle) for arg in arguments.values()
    ):
        return kernel.block

    spec = extract_calltime_specialization(kernel, arguments)
    qint_types = {
        name: argument.value.type
        for name, argument in arguments.items()
        if kernel.input_types.get(name) is QInt
        and isinstance(argument, QInt)
        and isinstance(argument.value.type, QUIntType)
    }
    if spec is None and not qint_types:
        return kernel.block

    sub_parameters, sub_bindings, sub_qubit_sizes = spec or ([], {}, {})
    kernel._specializing = True
    try:
        return build_specialized_block(
            kernel,
            parameters=sub_parameters,
            bindings=sub_bindings,
            qubit_sizes=sub_qubit_sizes,
            qint_types=qint_types,
        )
    finally:
        kernel._specializing = False
