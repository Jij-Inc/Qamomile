"""Call-time specialization extraction for qkernel calls."""

from __future__ import annotations

from typing import Any, cast

from qamomile.circuit.frontend.func_to_block import (
    is_array_type,
    is_dict_type,
    is_tuple_type,
)
from qamomile.circuit.frontend.handle import Observable, Qubit
from qamomile.circuit.frontend.handle.array import Vector
from qamomile.circuit.frontend.handle.primitives import Bit, Float, Handle, UInt
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.frontend.qkernel_build import build_specialized_block
from qamomile.circuit.frontend.qkernel_inputs import is_parameterizable_type
from qamomile.circuit.frontend.qkernel_utils import get_array_element_type
from qamomile.circuit.ir.block import Block


def extract_calltime_specialization(
    kernel: Any,
    arguments: dict[str, Any],
) -> tuple[list[str], dict[str, Any], dict[str, int]] | None:
    """Extract specialization inputs for a qkernel call site.

    Args:
        kernel (Any): ``QKernel``-like object with ``signature`` and
            ``input_types`` attributes.
        arguments (dict[str, Any]): Bound call arguments after literal
            promotion and frontend-handle validation.

    Returns:
        tuple[list[str], dict[str, Any], dict[str, int]] | None: Runtime
        parameter names, compile-time bindings, and concrete qubit-array
        sizes when specialization would change the callee trace; otherwise
        ``None``.
    """
    parameters: list[str] = []
    bindings: dict[str, Any] = {}
    qubit_sizes: dict[str, int] = {}

    for name, param in kernel.signature.parameters.items():
        param_type = kernel.input_types.get(name, param.annotation)
        handle = arguments.get(name)
        assert isinstance(handle, Handle), (
            f"Internal invariant violated: argument {name!r} should already "
            f"be a Handle by the time extract_calltime_specialization runs."
        )

        if param_type is Qubit:
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

    if not bindings and not qubit_sizes:
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
    temporary specialized block. Otherwise it returns the kernel's cached
    block.

    Args:
        kernel (Any): ``QKernel``-like object whose block should be selected.
        arguments (dict[str, Any]): Bound call arguments after literal
            promotion and frontend-handle validation.
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
    if spec is None:
        return kernel.block

    sub_parameters, sub_bindings, sub_qubit_sizes = spec
    kernel._specializing = True
    try:
        return build_specialized_block(
            kernel,
            parameters=sub_parameters,
            bindings=sub_bindings,
            qubit_sizes=sub_qubit_sizes,
        )
    finally:
        kernel._specializing = False
