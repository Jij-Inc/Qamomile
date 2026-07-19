"""Definition-time helpers for QKernel construction."""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

from qamomile.circuit.frontend.ast_transform import (
    collect_quantum_rebind_violations,
    transform_control_flow,
)
from qamomile.circuit.frontend.handle.primitives import Handle
from qamomile.circuit.frontend.qkernel_rebind import format_rebind_violation
from qamomile.circuit.frontend.qkernel_utils import quantum_param_names
from qamomile.circuit.transpiler.errors import (
    FrontendTransformError,
    QubitRebindError,
)


def transform_qkernel_function(func: Callable[..., Any]) -> Callable[..., Any]:
    """Transform a Python function into the frontend DSL function.

    Args:
        func (Callable[..., Any]): Raw user function decorated as a qkernel.

    Returns:
        Callable[..., Any]: AST-transformed function.

    Raises:
        FrontendTransformError: If the transform reports an unsupported
            frontend construct.
        SyntaxError: If the transform detects invalid syntax-level DSL usage.
    """
    try:
        return transform_control_flow(func)
    except SyntaxError:
        raise
    except NotImplementedError as e:
        raise FrontendTransformError(
            f"AST transformation failed for function '{func.__name__}': {e}"
        )


def refresh_qkernel_function_namespace(kernel: Any) -> None:
    """Refresh an AST-transformed qkernel's live Python name bindings.

    The transformed function is compiled into a private globals dictionary so
    generated control-flow helpers do not pollute the user's module. Python
    module globals and closure values must nevertheless retain normal
    call-time lookup semantics, so this function synchronizes them immediately
    before each trace.

    Args:
        kernel (Any): QKernel-like object exposing ``raw_func``, ``func``,
            and ``name`` attributes.

    Raises:
        FrontendTransformError: If a closure cell required by the transformed
            function is empty at trace time.
    """
    raw_func = kernel.raw_func
    transformed_func = kernel.func
    namespace = transformed_func.__globals__
    namespace.update(raw_func.__globals__)

    generated_globals = getattr(
        transformed_func,
        "__qamomile_generated_globals__",
        {},
    )
    namespace.update(generated_globals)

    if raw_func.__closure__ is not None:
        for name, cell in zip(raw_func.__code__.co_freevars, raw_func.__closure__):
            try:
                namespace[name] = cell.cell_contents
            except ValueError as error:
                raise FrontendTransformError(
                    f"Closure variable '{name}' in @qkernel '{kernel.name}' "
                    "is not bound at trace time."
                ) from error

    # The live module dictionary normally contains the decorated QKernel under
    # this name. Set it explicitly as well for nested/local definitions and
    # self-recursive kernels whose local binding is not module-global.
    namespace[kernel.name] = kernel


def resolve_kernel_io_types(
    func: Callable[..., Any],
    signature: inspect.Signature,
) -> tuple[dict[str, type[Handle]], list[type[Handle]]]:
    """Resolve and validate qkernel input/output handle annotations.

    Args:
        func (Callable[..., Any]): Raw user function.
        signature (inspect.Signature): Function signature.

    Returns:
        tuple[dict[str, type[Handle]], list[type[Handle]]]: Input handle types
        keyed by parameter name and output handle types by position.

    Raises:
        TypeError: If any parameter or return type is missing an annotation.
    """
    type_hints = _resolve_type_hints(func, signature)
    input_types: dict[str, type[Handle]] = {}
    for param in signature.parameters.values():
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter '{param.name}' must have a type annotation")
        input_types[param.name] = type_hints.get(param.name, param.annotation)

    if signature.return_annotation is inspect.Signature.empty:
        raise TypeError("Return type must have a type annotation")

    output_types: list[type[Handle]] = []
    return_type = type_hints.get("return", signature.return_annotation)
    if getattr(return_type, "__origin__", None) is tuple:
        output_types.extend(return_type.__args__)
    else:
        output_types.append(return_type)

    return input_types, output_types


def _resolve_type_hints(
    func: Callable[..., Any],
    signature: inspect.Signature,
) -> dict[str, Any]:
    """Resolve annotations, falling back to raw annotations on failure.

    Args:
        func (Callable[..., Any]): Raw user function.
        signature (inspect.Signature): Function signature.

    Returns:
        dict[str, Any]: Resolved or raw type hints keyed by parameter name,
        with ``"return"`` for the return annotation when present.
    """
    try:
        func_globals = getattr(func, "__globals__", {})
        return get_type_hints(func, globalns=func_globals, localns=None)
    except Exception:
        type_hints: dict[str, Any] = {}
        for param in signature.parameters.values():
            if param.annotation is not inspect.Parameter.empty:
                type_hints[param.name] = param.annotation
        if signature.return_annotation is not inspect.Signature.empty:
            type_hints["return"] = signature.return_annotation
        return type_hints


def validate_quantum_rebinds(
    func: Callable[..., Any],
    *,
    kernel_name: str,
    input_types: dict[str, type[Handle]],
) -> None:
    """Reject illegal quantum variable rebindings in a qkernel body.

    Args:
        func (Callable[..., Any]): Raw user function.
        kernel_name (str): User-visible qkernel name for diagnostics.
        input_types (dict[str, type[Handle]]): Resolved input annotations.

    Raises:
        QubitRebindError: If the AST analyzer finds a forbidden quantum
            variable reassignment.
    """
    violations = collect_quantum_rebind_violations(
        func,
        quantum_param_names(input_types),
    )
    if not violations:
        return

    v = violations[0]
    pattern, reason, fix = format_rebind_violation(v)
    raise QubitRebindError(
        f"Kernel '{kernel_name}': forbidden quantum variable reassignment "
        f"at body line {v.lineno} (counting the first statement of "
        f"the function body as line 1): "
        f"'{pattern}' overwrites quantum variable '{v.target_name}' "
        f"with {reason}.\n\nTo fix:\n{fix}",
        handle_name=v.target_name,
        operation_name="assignment_rebind",
    )
