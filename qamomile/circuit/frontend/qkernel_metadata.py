"""Metadata helpers for QKernel convenience APIs."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qamomile.circuit.estimator.resource_estimator import ResourceEstimate
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
    bindings: dict[str, Any] | None = None,
    substitutions: dict[str, Any] | None = None,
    parameters: list[str] | None = None,
    policy: Any = None,
    cost_basis: Any = None,
    strategies: dict[str, str] | None = None,
    trace: bool = False,
    unknown_policy: Any = None,
) -> "ResourceEstimate":
    """Estimate resources for a kernel.

    Args:
        kernel (QKernel[Any, Any]): Kernel to estimate.
        bindings (dict[str, Any] | None): Optional concrete parameter
            bindings baked into the built circuit. Defaults to ``None``.
        substitutions (dict[str, Any] | None): Estimation-only substitutions
            applied to the symbolic estimate (e.g. ``substitutions={"n":
            2048}``) without constructing a large circuit. Defaults to
            ``None``.
        parameters (list[str] | None): Runtime parameter names to preserve
            during kernel build. Defaults to ``None``.
        policy (Any): Optional ``ResourcePolicy`` override. Defaults to
            ``None``.
        cost_basis (Any): Optional ``CostBasis`` override. Defaults to
            ``None``.
        strategies (dict[str, str] | None): Callable strategy overrides.
            Defaults to ``None``.
        trace (bool): Whether to retain the explanation tree. Defaults to
            ``False``.
        unknown_policy (Any): Optional ``UnknownResourcePolicy`` override.
            Defaults to ``None``.

    Returns:
        ResourceEstimate: Estimated qubit, gate, and parameter resources.
    """
    from qamomile.circuit.estimator.resource_estimator import (
        CostBasis,
        ResourcePolicy,
        UnknownResourcePolicy,
        estimate_resources,
    )

    return estimate_resources(
        kernel,
        bindings=bindings,
        substitutions=substitutions,
        parameters=parameters,
        policy=policy or ResourcePolicy.MODEL_IF_AVAILABLE,
        cost_basis=cost_basis or CostBasis.LOGICAL_GATES,
        strategies=strategies,
        trace=trace,
        unknown_policy=unknown_policy or UnknownResourcePolicy.ERROR,
    )
