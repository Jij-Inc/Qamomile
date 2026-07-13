from __future__ import annotations

import inspect
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
    resolve_kernel_io_types,
    transform_qkernel_function,
    validate_quantum_rebinds,
)
from qamomile.circuit.ir.block import Block
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
        self.input_types, self.output_types = resolve_kernel_io_types(
            func,
            self.signature,
        )

        # Lazy initialization for hierarchical Block
        self._block: Block | None = None
        self._block_building: bool = False
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
            input_types=self.input_types,
        )

    @property
    def block(self) -> Block:
        """Compile the function to a hierarchical Block if not already compiled."""
        return get_or_build_block(self)

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


def qkernel(func: Callable[P, R]) -> QKernel[P, R]:
    """Decorator to define a Qamomile quantum kernel.

    Args:
        func: The function to decorate.

    Returns:
        An instance of QKernel wrapping the function.
    """
    return QKernel(func)
