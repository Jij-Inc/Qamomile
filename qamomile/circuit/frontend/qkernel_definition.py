"""Definition-time helpers for QKernel construction."""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

from qamomile.circuit.frontend.ast_transform import (
    collect_quantum_rebind_violations,
    transform_control_flow,
)
from qamomile.circuit.frontend.qkernel_rebind import format_rebind_violation
from qamomile.circuit.frontend.qkernel_utils import quantum_param_names
from qamomile.circuit.frontend.region_analysis import RegionLocation, RegionSignature
from qamomile.circuit.transpiler.errors import (
    FrontendTransformError,
    QubitRebindError,
)


def transform_qkernel_function(
    func: Callable[..., Any],
    region_signatures: dict[RegionLocation, RegionSignature] | None = None,
) -> Callable[..., Any]:
    """Transform a Python function into the frontend DSL function.

    Args:
        func (Callable[..., Any]): Raw user function decorated as a qkernel.
        region_signatures (dict[RegionLocation, RegionSignature] | None):
            Precomputed explicit control-flow interfaces. Defaults to ``None``.

    Returns:
        Callable[..., Any]: AST-transformed function.

    Raises:
        FrontendTransformError: If the transform reports an unsupported
            frontend construct.
        SyntaxError: If the transform detects invalid syntax-level DSL usage.
    """
    try:
        return transform_control_flow(func, region_signatures=region_signatures)
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
) -> tuple[dict[str, Any], list[Any]]:
    """Resolve and validate qkernel input/output handle annotations.

    Args:
        func (Callable[..., Any]): Raw user function.
        signature (inspect.Signature): Function signature.

    Returns:
        tuple[dict[str, Any], list[Any]]: Resolved annotations or raw deferred
        fallbacks keyed by parameter name, and output annotations by position.

    Raises:
        TypeError: If any parameter or return type is missing an annotation.
    """
    input_types = resolve_kernel_input_types(func, signature)
    return_type = resolve_kernel_return_type(func, signature)
    return input_types, flatten_kernel_return_type(return_type)


def resolve_kernel_input_types(
    func: Callable[..., Any],
    signature: inspect.Signature,
) -> dict[str, Any]:
    """Resolve qkernel input annotations independently by parameter.

    Resolving each annotation separately prevents one deferred forward
    reference from reverting otherwise valid sibling annotations to strings.

    Args:
        func (Callable[..., Any]): Raw user function.
        signature (inspect.Signature): Function signature.

    Returns:
        dict[str, Any]: Resolved annotations or raw deferred fallbacks keyed by
        parameter name.

    Raises:
        TypeError: If any parameter is missing an annotation.
    """
    return try_resolve_kernel_input_types(func, signature)[0]


def try_resolve_kernel_input_types(
    func: Callable[..., Any],
    signature: inspect.Signature,
) -> tuple[dict[str, Any], dict[str, Exception]]:
    """Resolve each qkernel input annotation independently.

    Args:
        func (Callable[..., Any]): Raw user function.
        signature (inspect.Signature): Function signature.

    Returns:
        tuple[dict[str, Any], dict[str, Exception]]: Resolved annotations or raw
        fallbacks by parameter, plus resolution errors for deferred
        annotations.

    Raises:
        TypeError: If any parameter is missing an annotation.
    """
    input_types: dict[str, Any] = {}
    errors: dict[str, Exception] = {}
    for param in signature.parameters.values():
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter '{param.name}' must have a type annotation")
        annotation, resolved, error = _try_resolve_annotation(
            func,
            param.name,
            param.annotation,
        )
        input_types[param.name] = annotation
        if not resolved:
            assert error is not None
            errors[param.name] = error
    return input_types, errors


def flatten_kernel_return_type(return_type: Any) -> list[Any]:
    """Flatten a qkernel return annotation into its output-slot types.

    A Python tuple denotes multiple ABI results, while every other annotation,
    including the structural ``Tuple`` handle, denotes one result. A ``None``
    annotation denotes no result slots.

    Args:
        return_type (Any): Complete qkernel return annotation.

    Returns:
        list[Any]: Frontend annotations ordered by output slot.

    Raises:
        TypeError: If a variable-length Python tuple is declared because its
            result arity cannot be represented by the qkernel ABI.
    """
    if return_type is None or return_type is type(None):
        return []
    if getattr(return_type, "__origin__", None) is tuple:
        result_types = return_type.__args__
        if any(result_type is Ellipsis for result_type in result_types):
            raise TypeError(
                "Variable-length Python tuple return annotations are not "
                "supported; declare a fixed-length tuple instead."
            )
        return list(result_types)
    return [return_type]


def resolve_kernel_return_type(
    func: Callable[..., Any],
    signature: inspect.Signature,
) -> Any:
    """Resolve a qkernel's complete return annotation.

    Unlike the flattened ``output_types`` list, the complete annotation
    preserves the distinction between a Python tuple of results and one
    structural ``Tuple`` handle.

    Args:
        func (Callable[..., Any]): Raw user function.
        signature (inspect.Signature): Function signature.

    Returns:
        Any: Complete resolved return annotation.

    Raises:
        TypeError: If the return type is missing an annotation.
    """
    return try_resolve_kernel_return_type(func, signature)[0]


def resolve_qkernel_like_return_type(kernel: Any) -> Any:
    """Return a qkernel-like object's complete resolved return annotation.

    Decorator-created kernels expose a frozen ``return_type`` property. Legacy
    qkernel-like objects instead expose only a signature and original function,
    so postponed string annotations must be resolved before ABI decisions.

    Args:
        kernel (Any): QKernel-like object exposing a signature and, when its
            annotation is postponed, the original ``raw_func``.

    Returns:
        Any: Complete resolved return annotation.

    Raises:
        TypeError: If the annotation is missing or cannot be resolved without
            a frozen ``return_type`` contract.
    """
    missing_return_type = object()
    annotation = getattr(kernel, "return_type", missing_return_type)
    if annotation is not missing_return_type and not isinstance(annotation, str):
        return annotation

    if annotation is missing_return_type:
        annotation = kernel.signature.return_annotation
    if annotation is inspect.Signature.empty:
        raise TypeError("Return type must have a type annotation")
    if not isinstance(annotation, str):
        return annotation

    raw_func = getattr(kernel, "raw_func", None)
    if not callable(raw_func):
        raise TypeError(
            f"Cannot resolve deferred return annotation {annotation!r} "
            "without the original qkernel function."
        )
    resolved, succeeded, error = try_resolve_kernel_return_type(
        raw_func,
        kernel.signature,
    )
    if not succeeded:
        raise TypeError(
            f"Cannot resolve deferred return annotation {annotation!r}."
        ) from error
    return resolved


def try_resolve_kernel_return_type(
    func: Callable[..., Any],
    signature: inspect.Signature,
) -> tuple[Any, bool, Exception | None]:
    """Resolve one return annotation independently from parameter hints.

    An unresolved forward reference is retained so a live ``QKernel`` can
    retry it at the first compilation entry point. Once resolution succeeds,
    the kernel freezes that result as its return contract.

    Args:
        func (Callable[..., Any]): Raw user function.
        signature (inspect.Signature): Function signature.

    Returns:
        tuple[Any, bool, Exception | None]: Annotation or raw fallback,
        whether resolution succeeded, and the resolution error when it did
        not.

    Raises:
        TypeError: If the return type is missing an annotation.
    """
    annotation = signature.return_annotation
    if annotation is inspect.Signature.empty:
        raise TypeError("Return type must have a type annotation")

    return _try_resolve_annotation(func, "return", annotation)


def _try_resolve_annotation(
    func: Callable[..., Any],
    name: str,
    annotation: Any,
) -> tuple[Any, bool, Exception | None]:
    """Resolve one annotation without depending on any sibling hint.

    Args:
        func (Callable[..., Any]): Raw user function providing the namespace.
        name (str): Parameter name or ``"return"``.
        annotation (Any): Raw annotation to resolve.

    Returns:
        tuple[Any, bool, Exception | None]: Annotation or raw fallback,
        whether resolution succeeded, and the resolution error when it did
        not.
    """
    carrier = type(
        "_QKernelAnnotation",
        (),
        {"__annotations__": {name: annotation}},
    )
    try:
        resolved = get_type_hints(
            carrier,
            globalns=getattr(func, "__globals__", {}),
            localns=None,
        )[name]
    except (NameError, TypeError) as error:
        return annotation, False, error
    return resolved, True, None


def validate_quantum_rebinds(
    func: Callable[..., Any],
    *,
    kernel_name: str,
    input_types: dict[str, Any],
) -> None:
    """Reject illegal quantum variable rebindings in a qkernel body.

    Args:
        func (Callable[..., Any]): Raw user function.
        kernel_name (str): User-visible qkernel name for diagnostics.
        input_types (dict[str, Any]): Resolved annotations or raw deferred
            fallbacks keyed by parameter name.

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


def get_quantum_rebind_error(
    func: Callable[..., Any],
    *,
    kernel_name: str,
    input_types: dict[str, Any],
) -> QubitRebindError | None:
    """Capture an illegal quantum rebind for deferred input validation.

    Args:
        func (Callable[..., Any]): Raw user function.
        kernel_name (str): User-visible qkernel name for diagnostics.
        input_types (dict[str, Any]): Resolved annotations or raw deferred
            fallbacks keyed by parameter name.

    Returns:
        QubitRebindError | None: Validation error, or ``None`` when the body is
        valid for the resolved input types.
    """
    try:
        validate_quantum_rebinds(
            func,
            kernel_name=kernel_name,
            input_types=input_types,
        )
    except QubitRebindError as error:
        return error
    return None
