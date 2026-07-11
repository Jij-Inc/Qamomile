"""Visualization helpers for QKernel objects."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.frontend.func_to_block import _get_ndim, is_array_type
from qamomile.circuit.frontend.handle import Qubit
from qamomile.circuit.frontend.qkernel_build import create_traced_block
from qamomile.circuit.frontend.qkernel_inputs import (
    auto_detect_parameters,
    validate_parameters,
)
from qamomile.circuit.frontend.qkernel_metadata import extract_return_names
from qamomile.circuit.frontend.qkernel_utils import get_array_element_type
from qamomile.circuit.ir.block import Block


def has_qubit_array_params(kernel: Any) -> bool:
    """Return whether a kernel declares quantum-array parameters.

    Args:
        kernel (Any): QKernel-like object with ``signature`` and
            ``input_types`` attributes.

    Returns:
        bool: ``True`` when any parameter is a ``Vector[Qubit]``-style
        quantum array.
    """
    for name, param in kernel.signature.parameters.items():
        param_type = kernel.input_types.get(name, param.annotation)
        if is_array_type(param_type) and get_array_element_type(param_type) is Qubit:
            return True
    return False


def build_graph_for_visualization(kernel: Any, **kwargs: Any) -> Block:
    """Build a traced block suitable for visualization.

    Args:
        kernel (Any): QKernel-like object to trace.
        **kwargs (Any): Concrete values for kernel arguments. For
            ``Vector[Qubit]`` parameters, pass an integer size.

    Returns:
        Block: Traced block with output names populated.
    """
    if has_qubit_array_params(kernel):
        graph = build_graph_with_qubit_arrays(kernel, kwargs)
    else:
        graph = kernel.build(parameters=None, **kwargs)
    graph.output_names = extract_return_names(kernel) or []
    return graph


def build_graph_with_qubit_arrays(kernel: Any, kwargs: dict[str, Any]) -> Block:
    """Build a traced block with concrete ``Vector[Qubit]`` sizes.

    Args:
        kernel (Any): QKernel-like object to trace.
        kwargs (dict[str, Any]): Concrete values for kernel arguments.
            Integer values for ``Vector[Qubit]`` parameters are interpreted
            as register sizes.

    Returns:
        Block: Traced block with quantum-array parameters realized as
        concrete 1-D registers.

    Raises:
        NotImplementedError: If the kernel declares a rank greater than one
            quantum array parameter.
        ValueError: If a quantum-array parameter is missing its integer size.
    """
    for name, param in kernel.signature.parameters.items():
        param_type = kernel.input_types.get(name, param.annotation)
        if is_array_type(param_type) and get_array_element_type(param_type) is Qubit:
            ndim = _get_ndim(param_type)
            if ndim > 1:
                raise NotImplementedError(
                    f"Parameter {name!r} is a rank-{ndim} quantum "
                    f"register: the quantum addressing path is rank-1, "
                    f"so a higher-rank register would silently alias "
                    f"distinct elements onto the same physical qubit. "
                    f"Declare a 1-D Vector[Qubit] parameter and compute "
                    f"flat indices explicitly instead "
                    f"(e.g. q[i * ncols + j])."
                )

    qubit_sizes: dict[str, int] = {}
    build_kwargs: dict[str, Any] = {}
    for key, val in kwargs.items():
        if key in kernel.signature.parameters:
            param_type = kernel.input_types.get(
                key, kernel.signature.parameters[key].annotation
            )
            if is_array_type(param_type):
                elem = get_array_element_type(param_type)
                if elem is Qubit and isinstance(val, int):
                    qubit_sizes[key] = val
                    continue
        build_kwargs[key] = val

    missing = []
    for name, param in kernel.signature.parameters.items():
        param_type = kernel.input_types.get(name, param.annotation)
        if is_array_type(param_type):
            elem = get_array_element_type(param_type)
            if elem is Qubit and name not in qubit_sizes:
                missing.append(name)
    if missing:
        names = ", ".join(f"'{name}'" for name in missing)
        raise ValueError(
            f"Vector[Qubit] parameter(s) {names} require an integer size "
            f"for visualization. Example: draw({missing[0]}=3)"
        )

    parameters = auto_detect_parameters(
        kernel.signature,
        kernel.input_types,
        build_kwargs,
    )
    validate_parameters(kernel.input_types, parameters)

    return create_traced_block(
        kernel, parameters, build_kwargs, qubit_sizes=qubit_sizes
    )


def draw_qkernel(
    kernel: Any,
    *,
    inline: bool = False,
    fold_loops: bool = True,
    expand_composite: bool = False,
    inline_depth: int | None = None,
    fold_ifs: bool = False,
    **kwargs: Any,
) -> Any:
    """Visualize a qkernel using the Matplotlib drawer.

    Args:
        kernel (Any): QKernel-like object to draw.
        inline (bool): Whether inline callable contents should be expanded.
            Defaults to ``False``.
        fold_loops (bool): Whether loops should be shown as folded blocks.
            Defaults to ``True``.
        expand_composite (bool): Whether boxed composite calls should be
            expanded. Defaults to ``False``.
        inline_depth (int | None): Maximum nesting depth for inline expansion.
            Defaults to ``None``.
        fold_ifs (bool): Whether if/else branches should be folded. Defaults
            to ``False``.
        **kwargs (Any): Concrete values for kernel arguments.

    Returns:
        Any: Matplotlib figure object.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If visualization requires a missing concrete register
            size.
    """
    from qamomile.circuit.visualization import MatplotlibDrawer

    return MatplotlibDrawer.draw_kernel(
        kernel,
        inline=inline,
        fold_loops=fold_loops,
        fold_ifs=fold_ifs,
        expand_composite=expand_composite,
        inline_depth=inline_depth,
        **kwargs,
    )
