"""Define named composite qkernels without a parallel frontend hierarchy."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, cast, overload

from qamomile.circuit.estimator.resource_estimator import EstimateKind, ResourceEstimate
from qamomile.circuit.frontend.qkernel import QKernel, qkernel
from qamomile.circuit.ir.operation.callable import (
    CallableImplementation,
    CallPolicy,
    CompositeGateType,
    ResourceModelBinding,
)


@dataclasses.dataclass(frozen=True)
class _FunctionResourceModel:
    """Adapt a function to the resource-model protocol.

    Args:
        func (Callable[..., ResourceEstimate]): Model function accepting a
            resource context.
    """

    func: Callable[..., ResourceEstimate]

    def estimate(self, ctx: Any) -> ResourceEstimate:
        """Evaluate the wrapped resource model.

        Args:
            ctx (Any): Resource context supplied by the estimator.

        Returns:
            ResourceEstimate: Resource model result.
        """
        return self.func(ctx)


def _normalize_resource_models(
    resource_model: Any | None,
    resource_models: Sequence[Any] | None,
) -> tuple[ResourceModelBinding, ...]:
    """Normalize decorator resource models.

    Args:
        resource_model (Any | None): Optional primary resource model.
        resource_models (Sequence[Any] | None): Optional additional models or
            bindings.

    Returns:
        tuple[ResourceModelBinding, ...]: Normalized model bindings.
    """
    values = ([resource_model] if resource_model is not None else []) + list(
        resource_models or ()
    )
    return tuple(
        value
        if isinstance(value, ResourceModelBinding)
        else ResourceModelBinding(model=value)
        for value in values
    )


def _validate_default_estimate_kind(default_estimate_kind: str | None) -> None:
    """Validate an optional default estimate-kind tag.

    Args:
        default_estimate_kind (str | None): Tag to validate.

    Raises:
        ValueError: If the tag is not a recognized estimate kind.
    """
    if default_estimate_kind is None:
        return
    valid = {kind.value for kind in EstimateKind}
    if default_estimate_kind not in valid:
        raise ValueError(
            f"default_estimate_kind {default_estimate_kind!r} is not a recognized "
            "estimate kind; "
            f"expected one of {sorted(valid)}."
        )


def configure_composite(
    kernel: QKernel[..., Any],
    *,
    name: str | None = None,
    namespace: str = "user.composite",
    gate_type: CompositeGateType = CompositeGateType.CUSTOM,
    policy: CallPolicy = CallPolicy.PRESERVE_BOX,
    resource_model: Any | None = None,
    resource_models: Sequence[Any] | None = None,
    implementations: Sequence[CallableImplementation] | None = None,
    default_estimate_kind: str | None = None,
) -> QKernel[..., Any]:
    """Configure a QKernel to remain visible as a named composite call.

    This mutates and returns the same QKernel object. No wrapper class or
    alternate call protocol is introduced.

    Args:
        kernel (QKernel[..., Any]): Kernel to configure.
        name (str | None): Public callable name. Defaults to the kernel name.
        namespace (str): Stable callable namespace. Defaults to
            ``"user.composite"``.
        gate_type (CompositeGateType): Internal stdlib classification.
            Defaults to ``CUSTOM``.
        policy (CallPolicy): Lowering policy. Defaults to ``PRESERVE_BOX``.
        resource_model (Any | None): Optional primary resource model.
        resource_models (Sequence[Any] | None): Optional additional resource
            models or bindings.
        implementations (Sequence[CallableImplementation] | None): Optional
            implementation candidates.
        default_estimate_kind (str | None): Preferred resource-model kind.

    Returns:
        QKernel[..., Any]: The same configured kernel instance.

    Raises:
        ValueError: If ``default_estimate_kind`` is invalid.
    """
    _validate_default_estimate_kind(default_estimate_kind)
    kernel._callable_kind = "composite"
    kernel._callable_name = name or kernel.name
    kernel._callable_namespace = namespace
    kernel._callable_policy = policy
    kernel._callable_gate_type = gate_type
    kernel._callable_resource_models = _normalize_resource_models(
        resource_model, resource_models
    )
    kernel._callable_implementations = tuple(implementations or ())
    kernel._default_estimate_kind = default_estimate_kind
    return kernel


def attach_resource_model(
    kernel: QKernel[..., Any],
    func: Callable[..., ResourceEstimate] | None = None,
    *,
    strategy: str | None = None,
    transform: Any | None = None,
    estimate_kind: str = "strategy_model",
) -> Callable[..., Any]:
    """Attach a resource model to a QKernel.

    Args:
        kernel (QKernel[..., Any]): Kernel receiving the model.
        func (Callable[..., ResourceEstimate] | None): Model function, or
            ``None`` for decorator-with-arguments use.
        strategy (str | None): Optional strategy constraint.
        transform (Any | None): Optional transform constraint.
        estimate_kind (str): Estimate source tag. Defaults to
            ``"strategy_model"``.

    Returns:
        Callable[..., Any]: Registered model function or decorator.
    """

    def decorator(model_func: Callable[..., ResourceEstimate]) -> Callable[..., Any]:
        """Register one model function.

        Args:
            model_func (Callable[..., ResourceEstimate]): Model to register.

        Returns:
            Callable[..., Any]: The unchanged model function.
        """
        kernel._callable_resource_models = (
            *kernel._callable_resource_models,
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


@overload
def composite_gate(func: Callable[..., Any]) -> QKernel[..., Any]: ...


@overload
def composite_gate(
    *,
    name: str = "",
    resource_model: Any | None = None,
    resource_models: Sequence[Any] | None = None,
    implementations: Sequence[CallableImplementation] | None = None,
    default_estimate_kind: str | None = None,
) -> Callable[[Callable[..., Any]], QKernel[..., Any]]: ...


def composite_gate(
    func: Callable[..., Any] | None = None,
    *,
    name: str = "",
    resource_model: Any | None = None,
    resource_models: Sequence[Any] | None = None,
    implementations: Sequence[CallableImplementation] | None = None,
    default_estimate_kind: str | None = None,
) -> QKernel[..., Any] | Callable[[Callable[..., Any]], QKernel[..., Any]]:
    """Define a named composite using the normal qkernel programming model.

    The decorated object is a ``QKernel``. Calls keep their named box in the IR,
    while ``build()``, ``draw()``, ``estimate_resources()``, ``control()``, and
    ``inverse()`` use the same interface as every other qkernel.

    Args:
        func (Callable[..., Any] | None): Function or qkernel to decorate.
            Defaults to ``None`` for decorator-with-arguments use.
        name (str): Public callable name. Defaults to the function name.
        resource_model (Any | None): Optional primary resource model.
        resource_models (Sequence[Any] | None): Optional additional resource
            models or bindings.
        implementations (Sequence[CallableImplementation] | None): Optional
            compiler implementation candidates.
        default_estimate_kind (str | None): Preferred resource-model kind.

    Returns:
        QKernel[..., Any] | Callable[[Callable[..., Any]], QKernel[..., Any]]:
            Configured qkernel or decorator.

    Raises:
        TypeError: If the decorator target is not callable.
        ValueError: If ``default_estimate_kind`` is invalid.

    Example:
        >>> import qamomile.circuit as qmc
        >>> @qmc.composite_gate(name="bell_pair")
        ... def bell_pair(
        ...     a: qmc.Qubit, b: qmc.Qubit
        ... ) -> tuple[qmc.Qubit, qmc.Qubit]:
        ...     a = qmc.h(a)
        ...     return qmc.cx(a, b)
    """

    def decorator(target: Callable[..., Any]) -> QKernel[..., Any]:
        """Convert and configure one decorator target.

        Args:
            target (Callable[..., Any]): Raw function or QKernel.

        Returns:
            QKernel[..., Any]: Configured QKernel.

        Raises:
            TypeError: If ``target`` is not callable.
        """
        if isinstance(target, QKernel):
            kernel = target
        elif callable(target):
            kernel = qkernel(target)
        else:
            raise TypeError("composite_gate must decorate a function or QKernel.")
        return configure_composite(
            cast(QKernel[..., Any], kernel),
            name=name or None,
            resource_model=resource_model,
            resource_models=resource_models,
            implementations=implementations,
            default_estimate_kind=default_estimate_kind,
        )

    if func is None:
        return decorator
    return decorator(func)


__all__ = ["composite_gate"]
