"""QKernel-backed composite gate decorator helpers."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence, cast, overload

from qamomile.circuit.estimator.resource_estimator import ResourceEstimate
from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.handle.primitives import Qubit
from qamomile.circuit.frontend.qkernel_callable import block_call_operands_and_results
from qamomile.circuit.frontend.qkernel_utils import get_array_element_type
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableImplementation,
    CallableRef,
    CallPolicy,
    CompositeGateType,
    InvokeOperation,
    ResourceModelBinding,
    signature_from_block,
)
from qamomile.circuit.ir.value import ArrayValue, Value

_QKERNEL_DELEGATED_ATTRS = frozenset(
    {
        "raw_func",
        "func",
        "signature",
        "input_types",
        "output_types",
        "block",
    }
)


@dataclasses.dataclass
class _WrappedCompositeGate(CompositeGate):
    """Wrap a QKernel as a box-preserving composite callable.

    Args:
        _gate_type (CompositeGateType): Composite classification.
        _custom_name (str): Public callable name.
        _num_targets (int): Number of scalar target qubits for fixed-arity
            calls.
        _num_controls (int): Number of scalar control qubits for fixed-arity
            calls.
        _qkernel (Any): Wrapped ``QKernel`` instance.
        _resource_models (tuple[ResourceModelBinding, ...]): Context-aware
            resource models attached to the boxed callable.
        _implementations (tuple[CallableImplementation, ...]): Optional
            implementation candidates attached by the decorator.
    """

    _gate_type: CompositeGateType = CompositeGateType.CUSTOM
    _custom_name: str = ""
    _num_targets: int = 0
    _num_controls: int = 0
    _qkernel: Any = None
    _resource_models: tuple[ResourceModelBinding, ...] = ()
    _implementations: tuple[CallableImplementation, ...] = ()

    @property
    def qkernel(self) -> Any:
        """Return the wrapped qkernel object.

        Returns:
            Any: Wrapped ``QKernel`` instance.
        """
        return self._qkernel

    def __getattr__(self, name: str) -> Any:
        """Delegate qkernel-like introspection attributes to the wrapped kernel.

        Args:
            name (str): Attribute name requested by the caller.

        Returns:
            Any: Matching attribute from the wrapped ``QKernel``.

        Raises:
            AttributeError: If the attribute is not part of the qkernel-like
                surface delegated by this wrapper.
        """
        if name in _QKERNEL_DELEGATED_ATTRS and self._qkernel is not None:
            return getattr(self._qkernel, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    @property
    def name(self) -> str:
        """Return the qkernel-facing callable name.

        Returns:
            str: Public name used for qkernel-like introspection.
        """
        return self.custom_name or self._qkernel.name

    @property
    def gate_type(self) -> CompositeGateType:  # type: ignore[override]
        """Return the wrapped composite classification.

        Returns:
            CompositeGateType: Composite gate type recorded for this wrapper.
        """
        return self._gate_type

    @property
    def custom_name(self) -> str:  # type: ignore[override]
        """Return the public boxed callable name.

        Returns:
            str: User-visible composite name.
        """
        return self._custom_name

    @property
    def num_target_qubits(self) -> int:
        """Return the fixed scalar target arity.

        Returns:
            int: Number of scalar target qubits required by fixed-arity calls.
        """
        return self._num_targets

    @property
    def num_control_qubits(self) -> int:
        """Return the fixed scalar control arity.

        Returns:
            int: Number of scalar control qubits required by fixed-arity calls.
        """
        return self._num_controls

    def get_implementation(self) -> Block | None:
        """Return the wrapped QKernel body.

        Returns:
            Block | None: Cached QKernel block, or ``None`` when this wrapper has
            no kernel.
        """
        if self._qkernel is None:
            return None
        return self._qkernel.block

    def build(
        self,
        parameters: list[str] | None = None,
        **kwargs: Any,
    ) -> Block:
        """Build the wrapped qkernel body.

        This keeps decorator-created composites usable anywhere a qkernel-like
        object is inspected, while calls from another qkernel still emit a
        preserve-box composite invocation.

        Args:
            parameters (list[str] | None): Runtime parameter names to preserve.
                Defaults to ``None``, matching ``QKernel.build``.
            **kwargs (Any): Compile-time bindings for non-parameter arguments.

        Returns:
            Block: Traced body block from the wrapped qkernel.
        """
        return self._qkernel.build(parameters=parameters, **kwargs)

    def estimate_resources(
        self,
        *,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
        policy: Any = None,
        cost_basis: Any = None,
        strategies: dict[str, str] | None = None,
        unknown_policy: Any = None,
    ) -> ResourceEstimate:
        """Estimate resources for the wrapped qkernel body.

        Args:
            bindings (dict[str, Any] | None): Optional compile-time bindings.
                Defaults to ``None``.
            parameters (list[str] | None): Runtime parameter names to
                preserve. Defaults to ``None``.
            policy (Any): Optional ``ResourcePolicy`` override. Defaults to
                ``None``.
            cost_basis (Any): Optional ``CostBasis`` override. Defaults to
                ``None``.
            strategies (dict[str, str] | None): Strategy overrides. Defaults
                to ``None``.
            unknown_policy (Any): Optional unknown-callable policy. Defaults
                to ``None``.

        Returns:
            ResourceEstimate: Resource estimate for the wrapped qkernel body.
        """
        return self._qkernel.estimate_resources(
            bindings=bindings,
            parameters=parameters,
            policy=policy,
            cost_basis=cost_basis,
            strategies=strategies,
            unknown_policy=unknown_policy,
        )

    def resource_model(
        self,
        func: Callable | None = None,
        *,
        strategy: str | None = None,
        transform: Any | None = None,
        estimate_kind: str = "strategy_model",
    ) -> Callable:
        """Attach a context-aware resource model to this composite callable.

        Args:
            func (Callable | None): Function implementing
                ``estimate(ResourceContext)`` when used as a decorator.
                Defaults to ``None`` for decorator-with-arguments use.
            strategy (str | None): Optional strategy name. Defaults to
                ``None``.
            transform (Any | None): Optional ``CallTransform``. Defaults to
                ``None``.
            estimate_kind (str): Informational estimate-kind tag. Defaults to
                ``"strategy_model"``.

        Returns:
            Callable: The original model function or a decorator.
        """

        def decorator(model_func: Callable) -> Callable:
            """Register the supplied resource model function.

            Args:
                model_func (Callable): Function accepting ``ResourceContext``.

            Returns:
                Callable: The same function for normal decorator behavior.
            """
            self._resource_models = (
                *self._resource_models,
                ResourceModelBinding(
                    model=_FunctionResourceModel(model_func),
                    strategy=strategy,
                    transform=transform,
                    estimate_kind=estimate_kind,
                ),
            )
            return model_func

        if func is None:
            return decorator
        return decorator(func)

    def draw(
        self,
        inline: bool = False,
        fold_loops: bool = True,
        expand_composite: bool = False,
        inline_depth: int | None = None,
        fold_ifs: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Visualize the wrapped qkernel body.

        Args:
            inline (bool): Whether to expand inline callable contents.
                Defaults to ``False``.
            fold_loops (bool): Whether to fold loop bodies in the diagram.
                Defaults to ``True``.
            expand_composite (bool): Whether to expand boxed callable bodies.
                Defaults to ``False``.
            inline_depth (int | None): Optional maximum inline expansion depth.
                Defaults to ``None``.
            fold_ifs (bool): Whether to fold if bodies in the diagram.
                Defaults to ``False``.
            **kwargs (Any): Visualization-time bindings.

        Returns:
            Any: Matplotlib figure object from the wrapped qkernel.
        """
        return self._qkernel.draw(
            inline=inline,
            fold_loops=fold_loops,
            expand_composite=expand_composite,
            inline_depth=inline_depth,
            fold_ifs=fold_ifs,
            **kwargs,
        )

    def _callable_implementations(self) -> tuple[CallableImplementation, ...]:
        """Return decorator-supplied implementation candidates.

        Returns:
            tuple[CallableImplementation, ...]: Additional implementation
            candidates to attach to this composite callable.
        """
        return self._implementations

    def _has_qubit_array_signature(self) -> bool:
        """Return whether the wrapped qkernel has a qubit-array argument.

        Returns:
            bool: ``True`` when any input is annotated as an array of ``Qubit``
            handles.
        """
        from qamomile.circuit.frontend.func_to_block import is_array_type

        if self._qkernel is None:
            return False
        return any(
            is_array_type(param_type) and get_array_element_type(param_type) is Qubit
            for param_type in self._qkernel.input_types.values()
        )

    def _invoke_qkernel_block(
        self,
        block: Block,
        inputs_map: dict[str, Value],
        *,
        strategy_name: str | None = None,
    ) -> InvokeOperation:
        """Create a preserve-box invocation for the wrapped qkernel.

        Args:
            block (Block): Selected qkernel implementation body.
            inputs_map (dict[str, Value]): Actual argument values keyed by
                callee label.
            strategy_name (str | None): Optional implementation strategy
                selected at the call site. Defaults to ``None``.

        Returns:
            InvokeOperation: Box-preserving composite invocation.
        """
        inputs, results = block_call_operands_and_results(block, inputs_map)

        num_target_qubits = 0
        for value in inputs:
            if isinstance(value, ArrayValue):
                if value.type.is_quantum() and value.shape:
                    const = value.shape[0].get_const()
                    if const is not None:
                        num_target_qubits += int(const)
                continue
            if value.type.is_quantum():
                num_target_qubits += 1

        gate_ref = CallableRef(
            namespace="user.composite",
            name=self.custom_name
            or (self._qkernel.name if self._qkernel else "custom"),
        )
        attrs = {
            "kind": "composite",
            "gate_type": self.gate_type.name,
            "num_control_qubits": self.num_control_qubits,
            "num_target_qubits": num_target_qubits,
            "custom_name": self.custom_name,
            "strategy_name": strategy_name,
            "default_policy": CallPolicy.PRESERVE_BOX.name,
        }
        return InvokeOperation(
            operands=inputs,
            results=results,
            target=gate_ref,
            attrs=attrs,
            definition=CallableDef(
                ref=gate_ref,
                signature=signature_from_block(block),
                body=block,
                implementations=list(self._implementations),
                resource_models=list(self._resource_models),
                default_policy=CallPolicy.PRESERVE_BOX,
                attrs=attrs,
            ),
        )

    def __call__(
        self,
        *target_qubits: Any,
        controls: Sequence[Qubit] = (),
        strategy: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply the wrapped composite qkernel.

        Args:
            *target_qubits (Any): Positional qkernel arguments.
            controls (Sequence[Qubit]): Explicit controls for fixed-arity
                scalar composite gates. Defaults to an empty sequence.
            strategy (str | None): Optional decomposition strategy for
                fixed-arity scalar composite gates. Defaults to ``None``.
            **kwargs (Any): Keyword qkernel arguments for vector-signature
                wrappers.

        Returns:
            Any: The wrapped qkernel-shaped return handle for normal calls.
            Explicit controlled calls keep the legacy fixed-arity composite
            return shape.

        Raises:
            TypeError: If controls are supplied to a signature-aware vector
                wrapper, or if a controlled fixed-arity call also supplies
                keyword qkernel arguments.
        """
        from qamomile.circuit.frontend.qkernel_invocation import (
            invoke_qkernel_with_operation,
        )

        if not controls and self.num_control_qubits > 0:
            raise ValueError(
                f"Composite callable '{self.name}' declares "
                f"{self.num_control_qubits} control qubits. Pass controls=... "
                "or use qmc.control(...) for qkernel-like controlled calls."
            )

        if not controls:
            return invoke_qkernel_with_operation(
                self._qkernel,
                lambda block, inputs: self._invoke_qkernel_block(
                    block,
                    inputs,
                    strategy_name=strategy,
                ),
                *target_qubits,
                **kwargs,
            )
        if self._has_qubit_array_signature():
            raise TypeError("Vector composite wrappers do not accept controls yet.")
        if kwargs:
            raise TypeError(
                "Controlled fixed-arity composite calls do not accept keyword "
                "qkernel arguments."
            )
        return super().__call__(
            *target_qubits,
            controls=controls,
            strategy=strategy,
        )


@overload
def composite_gate(
    func: Callable,
) -> _WrappedCompositeGate:
    """Decorate a qkernel directly as a boxed callable."""
    ...


@overload
def composite_gate(
    *,
    name: str = "",
    num_controls: int = 0,
    gate_type: Any | None = None,
    resource_model: Any | None = None,
    resource_models: Sequence[Any] | None = None,
    implementations: Sequence[Any] | None = None,
) -> Callable[[Callable], _WrappedCompositeGate]:
    """Return a qkernel-to-boxed-callable decorator."""
    ...


def composite_gate(
    func: Callable | None = None,
    *,
    name: str = "",
    num_controls: int = 0,
    gate_type: Any | None = None,
    resource_model: Any | None = None,
    resource_models: Sequence[Any] | None = None,
    implementations: Sequence[Any] | None = None,
) -> _WrappedCompositeGate | Callable[[Callable], _WrappedCompositeGate]:
    """Create a boxed callable from a qkernel function.

    This is the public decorator for defining a named callable box with a
    standard qkernel body. Calls remain visible to later compiler stages as
    ``InvokeOperation`` objects, allowing backend-native lowering, strategy
    selection, and resource estimation to share one implementation-resolution
    path.

    Args:
        func (Callable | None): The qkernel function when used without
            arguments. Defaults to ``None`` for decorator-with-arguments use.
        name (str): Public callable name. Defaults to the qkernel name.
        num_controls (int): Number of explicit control qubits. Defaults to
            ``0``.
        gate_type (Any | None): Advanced internal composite classification.
            Defaults to ``None``, which records a custom composite.
        resource_model (Any | None): Optional context-aware resource model.
            Defaults to ``None``.
        resource_models (Sequence[Any] | None): Optional additional resource
            model bindings or model objects. Defaults to ``None``.
        implementations (Sequence[Any] | None): Optional advanced
            implementation candidates for backend, transform, or strategy
            selection. Defaults to ``None``.

    Returns:
        _WrappedCompositeGate | Callable[[Callable], _WrappedCompositeGate]:
        Boxed callable or decorator.

    Raises:
        TypeError: If the decorator is applied to an object that cannot be
            converted into a ``QKernel``.
    """
    from qamomile.circuit.frontend.qkernel import QKernel

    def decorator(
        kernel_or_func: Callable,
    ) -> _WrappedCompositeGate:
        """Wrap a decorated qkernel as a boxed callable.

        Args:
            kernel_or_func (Callable): QKernel produced by the qkernel
                decorator, or a raw Python function to convert into one.

        Returns:
            _WrappedCompositeGate: Boxed-callable wrapper around the qkernel.

        Raises:
            TypeError: If ``kernel_or_func`` cannot be converted into a
                ``QKernel``.
        """
        if not isinstance(kernel_or_func, QKernel):
            if not callable(kernel_or_func):
                raise TypeError(
                    "composite_gate decorator must be applied to a function "
                    "or qkernel. Use Oracle for opaque resource-only callables."
                )
            kernel_or_func = QKernel(kernel_or_func)

        qkernel_instance = kernel_or_func
        num_targets = _count_scalar_target_qubits(qkernel_instance)
        gate_name = name or qkernel_instance.name
        normalized_gate_type = _normalize_gate_type(gate_type)
        normalized_implementations = _normalize_implementations(implementations)
        normalized_resource_models = _normalize_resource_models(
            resource_model,
            resource_models,
        )

        return _WrappedCompositeGate(
            _gate_type=normalized_gate_type,
            _custom_name=gate_name,
            _num_targets=num_targets,
            _num_controls=num_controls,
            _qkernel=qkernel_instance,
            _resource_models=normalized_resource_models,
            _implementations=normalized_implementations,
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def composite(
    func: Callable,
) -> _WrappedCompositeGate:
    """Decorate a qkernel directly as a boxed composite callable."""
    ...


@overload
def composite(
    *,
    name: str = "",
    num_controls: int = 0,
    gate_type: Any | None = None,
    resource_model: Any | None = None,
    resource_models: Sequence[Any] | None = None,
    implementations: Sequence[Any] | None = None,
) -> Callable[[Callable], _WrappedCompositeGate]:
    """Return a qkernel-to-composite decorator."""
    ...


def composite(
    func: Callable | None = None,
    *,
    name: str = "",
    num_controls: int = 0,
    gate_type: Any | None = None,
    resource_model: Any | None = None,
    resource_models: Sequence[Any] | None = None,
    implementations: Sequence[Any] | None = None,
) -> _WrappedCompositeGate | Callable[[Callable], _WrappedCompositeGate]:
    """Create a boxed composite callable from a qkernel.

    This is a shorter alias for :func:`composite_gate`. It exists for code that
    prefers the callable terminology, while ``composite_gate`` remains the
    explicit public spelling used by the migration design.

    Args:
        func (Callable | None): The qkernel to wrap when used without
            arguments. Defaults to ``None`` for decorator-with-arguments use.
        name (str): Public box name. Defaults to the wrapped qkernel name.
        num_controls (int): Number of explicit control qubits. Defaults to
            ``0``.
        gate_type (Any | None): Advanced internal composite classification.
            Defaults to ``None``, which records a custom composite.
        resource_model (Any | None): Optional context-aware resource model.
            Defaults to ``None``.
        resource_models (Sequence[Any] | None): Optional additional resource
            model bindings or model objects. Defaults to ``None``.
        implementations (Sequence[Any] | None): Optional advanced
            implementation candidates for backend, transform, or strategy
            selection. Defaults to ``None``.

    Returns:
        _WrappedCompositeGate | Callable[[Callable], _WrappedCompositeGate]:
        Composite callable or decorator.
    """
    if func is None:
        return composite_gate(
            name=name,
            num_controls=num_controls,
            gate_type=gate_type,
            resource_model=resource_model,
            resource_models=resource_models,
            implementations=implementations,
        )
    decorator = composite_gate(
        name=name,
        num_controls=num_controls,
        gate_type=gate_type,
        resource_model=resource_model,
        resource_models=resource_models,
        implementations=implementations,
    )
    return decorator(func)


@dataclasses.dataclass(frozen=True)
class _FunctionResourceModel:
    """Adapt a function into the resource-model protocol.

    Args:
        func (Callable): Function accepting a ``ResourceContext`` and returning
            a ``ResourceEstimate``.
    """

    func: Callable

    def estimate(self, ctx: Any) -> ResourceEstimate:
        """Estimate resources by calling the wrapped function.

        Args:
            ctx (Any): Resource context passed by the estimator.

        Returns:
            ResourceEstimate: Function result.
        """
        return cast(ResourceEstimate, self.func(ctx))


def _normalize_gate_type(gate_type: Any | None) -> CompositeGateType:
    """Normalize an advanced composite classification argument.

    Args:
        gate_type (Any | None): Optional internal composite classification.

    Returns:
        CompositeGateType: Normalized gate type.

    Raises:
        TypeError: If ``gate_type`` is not a ``CompositeGateType``.
    """
    if gate_type is None:
        return CompositeGateType.CUSTOM
    if isinstance(gate_type, CompositeGateType):
        return gate_type
    raise TypeError("gate_type must be a CompositeGateType when provided.")


def _normalize_implementations(
    implementations: Sequence[Any] | None,
) -> tuple[CallableImplementation, ...]:
    """Normalize advanced implementation candidates.

    Args:
        implementations (Sequence[Any] | None): Optional implementation
            candidate sequence supplied to ``composite`` or ``composite_gate``.

    Returns:
        tuple[CallableImplementation, ...]: Validated implementation
        candidates.

    Raises:
        TypeError: If any candidate is not a ``CallableImplementation``.
    """
    if implementations is None:
        return ()
    for impl in implementations:
        if not isinstance(impl, CallableImplementation):
            raise TypeError(
                "implementations must contain CallableImplementation objects."
            )
    return cast(tuple[CallableImplementation, ...], tuple(implementations))


def _normalize_resource_models(
    resource_model: Any | None,
    resource_models: Sequence[Any] | None,
) -> tuple[ResourceModelBinding, ...]:
    """Normalize resource model arguments for a composite callable.

    Args:
        resource_model (Any | None): Single resource model object.
        resource_models (Sequence[Any] | None): Additional model objects or
            ``ResourceModelBinding`` instances.

    Returns:
        tuple[ResourceModelBinding, ...]: Normalized model bindings.
    """
    normalized: list[ResourceModelBinding] = []
    if resource_model is not None:
        normalized.append(ResourceModelBinding(model=resource_model))
    for model in resource_models or ():
        if isinstance(model, ResourceModelBinding):
            normalized.append(model)
        else:
            normalized.append(ResourceModelBinding(model=model))
    return tuple(normalized)


def _count_scalar_target_qubits(qkernel_instance: Any) -> int:
    """Count fixed scalar Qubit inputs in a qkernel signature.

    Args:
        qkernel_instance (Any): QKernel whose annotations are inspected.

    Returns:
        int: Number of scalar qubit arguments used by fixed-arity composite
        calls. Vector qubit arguments are handled by the signature-aware path
        and do not contribute here.
    """
    from qamomile.circuit.frontend.func_to_block import is_array_type

    num_targets = 0
    for param_type in qkernel_instance.input_types.values():
        if param_type is Qubit:
            num_targets += 1
        elif hasattr(param_type, "__origin__") and param_type.__origin__ is tuple:
            args = getattr(param_type, "__args__", ())
            num_targets += sum(1 for arg in args if arg is Qubit)
        elif is_array_type(param_type):
            args = getattr(param_type, "__args__", ())
            if args and args[0] is Qubit:  # type: ignore[misc]
                continue
    return num_targets
