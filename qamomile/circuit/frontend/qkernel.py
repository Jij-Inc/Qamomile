from __future__ import annotations

import inspect
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
)

from qamomile.circuit.frontend.qkernel_api import (
    QKernelBuildMixin,
    QKernelVisualizationMixin,
)
from qamomile.circuit.frontend.qkernel_block import get_or_build_block
from qamomile.circuit.frontend.qkernel_definition import (
    _ANNOTATION_LOCALNS_ATTR,
    flatten_kernel_return_type,
    get_quantum_rebind_error,
    transform_qkernel_function,
    try_resolve_kernel_input_types,
    try_resolve_kernel_return_type,
    validate_quantum_rebinds,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.effect import KernelEffect
from qamomile.circuit.ir.operation.callable import CallPolicy, CompositeGateType

if TYPE_CHECKING:
    from qamomile.circuit.ir.operation.callable import InvokeOperation

P = ParamSpec("P")
R = TypeVar("R")


class QKernel(QKernelBuildMixin, QKernelVisualizationMixin, Generic[P, R]):
    """Decorator class for Qamomile quantum kernels."""

    def __init__(self, func: Callable[P, R]) -> None:
        # Hold a function where AST transformation has replaced control flow (if/while) with builder function calls
        self.raw_func = func
        self.func = transform_qkernel_function(func)

        # transform_control_flow's exec namespace binds `func.__name__` to
        # the raw AST-transformed DSL function.  If the user body contains
        # a self-reference (e.g. for a recursive kernel), letting that name
        # resolve to the DSL function bypasses __call__ entirely: argument
        # validation, affine-type consumption, and InvokeOperation emission
        # are all skipped, and the call becomes a direct in-place trace that
        # re-enters the same body forever.  Rebinding to the QKernel so that
        # self-calls always go through __call__ — where __call__ accesses
        # self.block, which then detects in-flight construction — fixes this.
        self.func.__globals__[func.__name__] = self

        self.name = func.__name__
        self.signature = inspect.signature(func)
        self._annotation_lock = threading.RLock()
        self._input_type_validation_error: Exception | None = None
        self._input_types, self._input_type_resolution_errors = (
            try_resolve_kernel_input_types(func, self.signature)
        )
        self._input_types_resolved = not self._input_type_resolution_errors
        if self._input_types_resolved:
            self._freeze_input_types()
        (
            self._return_type,
            self._return_type_resolved,
            self._return_type_resolution_error,
        ) = try_resolve_kernel_return_type(func, self.signature)
        self._output_types = flatten_kernel_return_type(self._return_type)
        if self._return_type_resolved:
            self._freeze_return_type()
        self._release_annotation_localns_if_resolved()

        # Lazy initialization for hierarchical Block
        self._block: Block | None = None
        self._block_building: bool = False
        # Serialize first access to the lazy block. An RLock is required:
        # same-thread re-entry must reach the explicit recursion diagnostic,
        # while another thread waits and then receives the shared cached block.
        self._block_lock = threading.RLock()
        # Reentry guard for :meth:`__call__`'s call-time specialization
        # path. While the specialized re-trace runs the kernel body,
        # any self-call must fall back to the cached ``self.block`` to
        # avoid unbounded re-tracing of self-recursive kernels.
        self._specializing: bool = False
        # Self-recursive InvokeOperations emitted during the build get their
        # definition back-patched to ``self._block`` once ``func_to_block``
        # returns.  See _finalize_pending_self_calls.
        self._pending_self_calls: list[InvokeOperation] = []

        # Every frontend callable is a QKernel. ``@composite_gate`` only changes
        # this compiler metadata; it does not wrap the object or replace its
        # Python call contract.
        self._callable_kind = "qkernel"
        self._callable_name = self.name
        self._callable_namespace: str | None = None
        self._callable_policy = CallPolicy.INLINE
        self._callable_gate_type = CompositeGateType.CUSTOM
        self._callable_implementations: tuple[Any, ...] = ()
        self._callable_semantic_arguments: dict[str, Any] = {}

        # AST-level quantum rebind analysis: a violation is a structural error
        # in the kernel definition itself, so raise eagerly at decoration time
        # rather than deferring to .block / build().
        #
        # The analyzer is run unconditionally rather than only when the kernel
        # has quantum-typed parameters: its constructor-tracking logic (LHS of
        # ``q = qm.qubit(...)`` / ``qm.qubit_array(...)``) seeds new origins
        # from inside the body, so kernels that derive all of their quantum
        # state from internal allocations are still subject to rebind checks.
        validate_quantum_rebinds(
            self.raw_func,
            kernel_name=self.name,
            input_types=self._input_types,
        )

    def _freeze_input_types(self) -> None:
        """Pin resolved input annotations to the transformed function."""
        setattr(
            self.func,
            "__qamomile_resolved_input_types__",
            dict(self._input_types),
        )
        self.func.__annotations__.update(self._input_types)

    def _freeze_return_type(self) -> None:
        """Pin the resolved return annotation to the transformed function."""
        self.func.__annotations__["return"] = self._return_type
        setattr(
            self.func,
            "__qamomile_resolved_return_type__",
            self._return_type,
        )

    def _release_annotation_localns_if_resolved(self) -> None:
        """Release captured defining locals after the interface is frozen."""
        if self._input_types_resolved and self._return_type_resolved:
            self.raw_func.__dict__.pop(_ANNOTATION_LOCALNS_ATTR, None)

    def _resolve_pending_annotation_types(
        self,
    ) -> tuple[dict[str, Exception], Exception | None]:
        """Resolve and freeze every currently available interface annotation.

        Returns:
            tuple[dict[str, Exception], Exception | None]: Remaining input
            resolution errors and the remaining return resolution error.

        Raises:
            QubitRebindError: If a newly resolved quantum input exposes an
                illegal quantum rebind in the kernel body.
        """
        with self._annotation_lock:
            if self._input_type_validation_error is not None:
                raise self._input_type_validation_error
            if self._input_types_resolved and self._return_type_resolved:
                return {}, None

            input_updates: dict[str, Any] = {}
            input_errors: dict[str, Exception] = {}
            if not self._input_types_resolved:
                unresolved_input_names = self._input_type_resolution_errors
            else:
                unresolved_input_names = {}
            for name in unresolved_input_names:
                parameter = self.signature.parameters[name]
                annotations, errors = try_resolve_kernel_input_types(
                    self.raw_func,
                    self.signature.replace(parameters=[parameter]),
                )
                if errors:
                    input_errors[name] = errors[name]
                else:
                    input_updates[name] = annotations[name]

            return_annotation = self._return_type
            return_resolved = self._return_type_resolved
            return_error = self._return_type_resolution_error
            if not return_resolved:
                (
                    return_annotation,
                    return_resolved,
                    return_error,
                ) = try_resolve_kernel_return_type(
                    self.raw_func,
                    self.signature,
                )

            if input_updates:
                self._input_types = {**self._input_types, **input_updates}
            self._input_type_resolution_errors = input_errors

            if return_resolved and not self._return_type_resolved:
                self._return_type = return_annotation
                self._output_types = flatten_kernel_return_type(return_annotation)
                self._return_type_resolution_error = None
                self._freeze_return_type()
                self._return_type_resolved = True
            elif not return_resolved:
                self._return_type_resolution_error = return_error

            if input_updates:
                validation_error = get_quantum_rebind_error(
                    self.raw_func,
                    kernel_name=self.name,
                    input_types=self._input_types,
                )
                if validation_error is not None:
                    self._input_type_validation_error = validation_error
                    raise validation_error

            if not input_errors and not self._input_types_resolved:
                self._freeze_input_types()
                self._input_types_resolved = True

            self._release_annotation_localns_if_resolved()
            return input_errors, return_error if not return_resolved else None

    def _ensure_annotation_types_resolved(self) -> None:
        """Require the complete qkernel interface to be resolved and frozen.

        Raises:
            TypeError: If an input or return annotation cannot be resolved.
            QubitRebindError: If a newly resolved quantum input exposes an
                illegal quantum rebind in the kernel body.
        """
        input_errors, return_error = self._resolve_pending_annotation_types()
        if not self._input_types_resolved:
            name, error = next(iter(input_errors.items()))
            annotation = self.signature.parameters[name].annotation
            raise TypeError(
                f"Cannot resolve annotation {annotation!r} for parameter "
                f"{name!r} of qkernel {self.name!r}."
            ) from error
        if not self._return_type_resolved:
            raise TypeError(
                f"Cannot resolve return annotation {self._return_type!r} for "
                f"qkernel {self.name!r}."
            ) from return_error

    def _ensure_input_types_resolved(self) -> None:
        """Resolve and freeze deferred qkernel input annotations.

        Raises:
            TypeError: If an input annotation cannot be resolved.
            QubitRebindError: If a newly resolved quantum input exposes an
                illegal quantum rebind in the kernel body.
        """
        with self._annotation_lock:
            if self._input_type_validation_error is not None:
                raise self._input_type_validation_error
            if self._input_types_resolved:
                return
        input_errors, _ = self._resolve_pending_annotation_types()
        if not input_errors:
            return
        name, error = next(iter(input_errors.items()))
        annotation = self.signature.parameters[name].annotation
        raise TypeError(
            f"Cannot resolve annotation {annotation!r} for parameter "
            f"{name!r} of qkernel {self.name!r}."
        ) from error

    @property
    def input_types(self) -> dict[str, Any]:
        """Return resolved and frozen frontend input annotations.

        Returns:
            dict[str, Any]: Input annotations keyed by parameter name.

        Raises:
            TypeError: If a deferred input annotation cannot be resolved.
        """
        self._ensure_input_types_resolved()
        return dict(self._input_types)

    @input_types.setter
    def input_types(self, value: dict[str, Any]) -> None:
        """Replace the mutable frontend input ABI.

        Args:
            value (dict[str, Any]): Replacement annotations keyed by parameter
                name.

        Raises:
            TypeError: If a deferred source annotation cannot be resolved.
            QubitRebindError: If the source or replacement annotations expose
                an illegal quantum rebind in the kernel body.
        """
        with self._annotation_lock:
            self._ensure_input_types_resolved()
            replacement = dict(value)
            validate_quantum_rebinds(
                self.raw_func,
                kernel_name=self.name,
                input_types=replacement,
            )
            self._input_types = replacement

    @property
    def block(self) -> Block:
        """Compile the function to a hierarchical Block if not already compiled."""
        return get_or_build_block(self)

    def _ensure_return_type_resolved(self) -> None:
        """Resolve and freeze the deferred qkernel return annotation.

        Raises:
            TypeError: If the return annotation cannot be resolved.
        """
        with self._annotation_lock:
            if self._return_type_resolved:
                return
        _, return_error = self._resolve_pending_annotation_types()
        if return_error is None:
            return
        raise TypeError(
            f"Cannot resolve return annotation {self._return_type!r} for "
            f"qkernel {self.name!r}."
        ) from return_error

    @property
    def return_type(self) -> Any:
        """Return the resolved and frozen complete return annotation.

        Returns:
            Any: Scalar, array, container, or Python tuple annotation.

        Raises:
            TypeError: If a deferred return annotation cannot be resolved.
        """
        self._ensure_return_type_resolved()
        return self._return_type

    @property
    def output_types(self) -> list[Any]:
        """Return the resolved frontend annotation for every output slot.

        Returns:
            list[Any]: Output annotations in ABI order.

        Raises:
            TypeError: If a deferred return annotation cannot be resolved.
        """
        self._ensure_return_type_resolved()
        return list(self._output_types)

    @property
    def effects(self) -> KernelEffect:
        """Return cached semantic effects of this qkernel.

        Returns:
            KernelEffect: Effects aggregated while building ``self.block``.
        """
        return self.block.effects

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Invoke this qkernel in the active tracing context.

        Args:
            *args (P.args): Positional frontend handles or scalar literals.
            **kwargs (P.kwargs): Keyword frontend handles or scalar literals.

        Returns:
            R: Frontend handle result matching the decorated function's return
            annotation.
        """
        from qamomile.circuit.frontend.qkernel_invocation import invoke_qkernel

        return cast(R, invoke_qkernel(self, *args, **kwargs))


def _defining_local_namespace(func: Callable[..., Any]) -> dict[str, Any]:
    """Copy the live local namespace that lexically defines a function.

    Args:
        func (Callable[..., Any]): Function whose defining frame may still be
            active while a decorator wrapper constructs its QKernel.

    Returns:
        dict[str, Any]: Snapshot of the defining frame's locals, or an empty
            dictionary when the frame is module-global or no longer active.
    """
    code = getattr(func, "__code__", None)
    if code is None or "." not in code.co_qualname:
        return {}

    defining_qualname = code.co_qualname.rsplit(".", 1)[0]
    if defining_qualname.endswith(".<locals>"):
        defining_qualname = defining_qualname.removesuffix(".<locals>")

    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            if frame.f_code.co_qualname == defining_qualname:
                if frame.f_locals is func.__globals__:
                    return {}
                return dict(frame.f_locals)
            frame = frame.f_back
    finally:
        del frame
    return {}


def qkernel(func: Callable[P, R]) -> QKernel[P, R]:
    """Decorator to define a Qamomile quantum kernel.

    Args:
        func (Callable[P, R]): Function to decorate.

    Returns:
        QKernel[P, R]: QKernel wrapping the function.
    """
    annotation_localns = _defining_local_namespace(func)
    if not annotation_localns:
        return QKernel(func)

    had_previous_localns = hasattr(func, _ANNOTATION_LOCALNS_ATTR)
    previous_localns = getattr(func, _ANNOTATION_LOCALNS_ATTR, {})
    if isinstance(previous_localns, dict):
        annotation_localns = {**previous_localns, **annotation_localns}
    setattr(func, _ANNOTATION_LOCALNS_ATTR, annotation_localns)
    constructed = False
    try:
        kernel = QKernel(func)
        constructed = True
    finally:
        if not constructed:
            if had_previous_localns:
                setattr(func, _ANNOTATION_LOCALNS_ATTR, previous_localns)
            else:
                func.__dict__.pop(_ANNOTATION_LOCALNS_ATTR, None)
    return kernel
