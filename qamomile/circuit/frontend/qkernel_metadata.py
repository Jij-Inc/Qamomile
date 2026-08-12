"""Metadata helpers for QKernel convenience APIs."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qamomile.circuit.estimator import (
        ControlDecomposition,
        ResourceEstimate,
        UnknownResourcePolicy,
    )
    from qamomile.circuit.frontend.qkernel import QKernel


def extract_return_names(kernel: "QKernel[Any, Any]") -> list[str] | None:
    """Extract display names from the kernel's return statement.

    Args:
        kernel (QKernel[Any, Any]): Kernel whose raw Python source should be
            inspected.

    Returns:
        list[str] | None: Return expression labels when a top-level return can
        be parsed, otherwise ``None``.
    """
    source = inspect.getsource(kernel.raw_func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    return _return_value_names(stmt.value)

    return None


def _return_value_names(return_value: ast.expr) -> list[str]:
    """Render return expression labels.

    Args:
        return_value (ast.expr): AST expression from a return statement.

    Returns:
        list[str]: One label per returned expression.
    """
    if isinstance(return_value, ast.Tuple):
        return [_return_element_name(elt) for elt in return_value.elts]
    return [_return_element_name(return_value)]


def _return_element_name(element: ast.expr) -> str:
    """Render one returned expression as a display label.

    Args:
        element (ast.expr): Returned expression element.

    Returns:
        str: Source-shaped display label.
    """
    if isinstance(element, ast.Name):
        return element.id
    return ast.unparse(element)


def estimate_qkernel_resources(
    kernel: "QKernel[Any, Any]",
    *,
    inputs: dict[str, Any] | None = None,
    strategies: dict[str, str] | None = None,
    trace: bool = False,
    unknown_policy: str | UnknownResourcePolicy | None = None,
    control_decomposition: str | ControlDecomposition | None = None,
) -> "ResourceEstimate":
    """Estimate resources for a kernel.

    Args:
        kernel (QKernel[Any, Any]): Kernel to estimate.
        inputs (dict[str, Any] | None): QKernel input values used to specialize
            the symbolic estimate. Exact one-dimensional root quantum-port
            widths declared by callable resource metadata are inferred when
            omitted. Defaults to ``None``.
        strategies (dict[str, str] | None): Callable strategy overrides.
            Defaults to ``None``.
        trace (bool): Whether to retain the explanation tree. Defaults to
            ``False``.
        unknown_policy (str | UnknownResourcePolicy | None): Policy for
            bodyless callables without explicit costs. Defaults to ``None``,
            which uses the estimator default.
        control_decomposition (str | ControlDecomposition | None):
            Coherent-control model override. Defaults to ``None``, which uses
            the clean-ancilla Toffoli model.

    Returns:
        ResourceEstimate: Estimated width, gate, measurement, reset, depth,
            call, and parameter resources.

    Raises:
        RuntimeError: If a fixed or callback-provided opaque cost contains
            public metrics or metadata that disagree with retained canonical
            provenance.
        ValueError: If an input, estimation configuration, callable resource
            contract, or structural requirement is invalid.
        TypeError: If the qkernel cannot be built as an estimator input.
        NotImplementedError: If the qkernel contains a construct not supported
            by resource estimation.
    """
    from qamomile.circuit.estimator.resource_estimator import estimate_resources

    estimator_options: dict[str, Any] = {}
    if unknown_policy is not None:
        estimator_options["unknown_policy"] = unknown_policy
    if control_decomposition is not None:
        estimator_options["control_decomposition"] = control_decomposition

    return estimate_resources(
        kernel,
        inputs=inputs,
        strategies=strategies,
        trace=trace,
        **estimator_options,
    )
