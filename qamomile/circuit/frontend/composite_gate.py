"""Define named composite qkernels without a parallel frontend hierarchy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast, overload

from qamomile.circuit.frontend.qkernel import QKernel, qkernel
from qamomile.circuit.ir.operation.callable import (
    CallableImplementation,
    CallPolicy,
    CompositeGateType,
)


def configure_composite(
    kernel: QKernel[..., Any],
    *,
    name: str | None = None,
    namespace: str | None = None,
    gate_type: CompositeGateType = CompositeGateType.CUSTOM,
    policy: CallPolicy = CallPolicy.PRESERVE_BOX,
    implementations: Sequence[CallableImplementation] | None = None,
) -> QKernel[..., Any]:
    """Configure a QKernel to remain visible as a named composite call.

    This mutates and returns the same QKernel object. No wrapper class or
    alternate call protocol is introduced.

    Args:
        kernel (QKernel[..., Any]): Kernel to configure.
        name (str | None): Public callable name. Defaults to the kernel name.
        namespace (str | None): Explicit stable callable namespace. ``None``
            derives one from the decorated function's module, qualified name,
            and source location. Defaults to ``None``.
        gate_type (CompositeGateType): Internal stdlib classification.
            Defaults to ``CUSTOM``.
        policy (CallPolicy): Lowering policy. Defaults to ``PRESERVE_BOX``.
        implementations (Sequence[CallableImplementation] | None): Optional
            implementation candidates.

    Returns:
        QKernel[..., Any]: The same configured kernel instance.
    """
    kernel._callable_kind = "composite"
    kernel._callable_name = name or kernel.name
    kernel._callable_namespace = namespace
    kernel._callable_policy = policy
    kernel._callable_gate_type = gate_type
    kernel._callable_implementations = tuple(implementations or ())
    return kernel


@overload
def composite_gate(func: Callable[..., Any]) -> QKernel[..., Any]: ...


@overload
def composite_gate(
    *,
    name: str = "",
    implementations: Sequence[CallableImplementation] | None = None,
) -> Callable[[Callable[..., Any]], QKernel[..., Any]]: ...


def composite_gate(
    func: Callable[..., Any] | None = None,
    *,
    name: str = "",
    implementations: Sequence[CallableImplementation] | None = None,
) -> QKernel[..., Any] | Callable[[Callable[..., Any]], QKernel[..., Any]]:
    """Define a named composite using the normal qkernel programming model.

    The decorated object is a ``QKernel``. Calls keep their named box in the IR,
    while ``build()``, ``draw()``, ``estimate_resources()``, ``control()``, and
    ``inverse()`` use the same interface as every other qkernel.

    Args:
        func (Callable[..., Any] | None): Function or qkernel to decorate.
            Defaults to ``None`` for decorator-with-arguments use.
        name (str): Public callable name. Defaults to the function name.
        implementations (Sequence[CallableImplementation] | None): Optional
            compiler implementation candidates.

    Returns:
        QKernel[..., Any] | Callable[[Callable[..., Any]], QKernel[..., Any]]:
            Configured qkernel or decorator.

    Raises:
        TypeError: If the decorator target is not callable.

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
            implementations=implementations,
        )

    if func is None:
        return decorator
    return decorator(func)


__all__ = ["composite_gate"]
