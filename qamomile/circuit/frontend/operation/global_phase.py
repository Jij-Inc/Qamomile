"""Provide the ``qmc.global_phase`` qkernel combinator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qamomile.circuit.frontend.handle import Handle
from qamomile.circuit.frontend.handle.primitives import Float
from qamomile.circuit.frontend.operation.control import _qkernel_for_callable
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation import GlobalPhaseOperation
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import Value

# A phase angle may be supplied as a frontend ``Float`` handle (symbolic /
# runtime parameter) or a plain Python number (compile-time constant).
PhaseValue = float | int | Float


def _phase_to_value(
    phase: PhaseValue,
    *,
    caller: str = "global_phase()",
) -> Value:
    """Coerce a user-supplied phase angle into a scalar IR value.

    Args:
        phase (float | int | Float): Phase angle in radians, as a Qamomile
            ``Float`` handle or a Python numeric literal.
        caller (str): API context used in validation diagnostics. Defaults to
            ``"global_phase()"``.

    Returns:
        Value: Scalar ``FloatType`` phase value.

    Raises:
        TypeError: If ``phase`` is a boolean, a non-``Float`` handle, or a
            non-numeric object.
    """
    if isinstance(phase, Float):
        return phase.value
    if isinstance(phase, Handle):
        raise TypeError(
            f"{caller}: phase must be a Float handle or a number, got "
            f"{type(phase).__name__}. Pass a qmc.Float (or a Python float)."
        )
    if isinstance(phase, bool):
        raise TypeError(
            f"{caller}: phase must be a number, not bool. Pass a numeric "
            "angle in radians (or a qmc.Float handle)."
        )
    if isinstance(phase, (int, float)):
        return Value(type=FloatType(), name="global_phase").with_const(float(phase))
    raise TypeError(
        f"{caller}: phase must be a Float handle or a number, got "
        f"{type(phase).__name__}."
    )


class GlobalPhaseGate:
    """Apply a wrapped qkernel followed by a zero-qubit global phase.

    Args:
        qkernel (QKernel): QKernel whose call is followed by the phase.
        phase (float | int | Float): Phase angle in radians.

    Example:
        >>> import qamomile.circuit as qmc
        >>> @qmc.qkernel
        ... def layer(q: qmc.Qubit) -> qmc.Qubit:
        ...     return qmc.h(q)
        >>> @qmc.qkernel
        ... def circuit(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
        ...     return qmc.global_phase(layer, theta)(q)
    """

    def __init__(self, qkernel: QKernel, phase: PhaseValue) -> None:
        """Initialize the global-phase wrapper.

        Args:
            qkernel (QKernel): QKernel whose call is followed by the phase.
            phase (float | int | Float): Phase angle in radians.
        """
        self._qkernel = qkernel
        self._phase = phase

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the wrapped qkernel and append its global phase.

        The wrapped call uses QKernel's canonical call path, including argument
        validation, specialization, affine consumption, output permutation,
        and ``VectorView`` borrow transfer. A standalone phase imposes no
        reversibility constraint. Transforms such as control and inverse
        enforce their own constraints when the transformed qkernel is compiled.

        Args:
            *args (Any): Positional arguments for the wrapped qkernel.
            **kwargs (Any): Keyword arguments for the wrapped qkernel.

        Returns:
            Any: Wrapped qkernel outputs in their original order and handle
                types.

        Raises:
            TypeError: If the phase or wrapped-call arguments are invalid.
            QubitConsumedError: If a quantum argument was already consumed.
            FrontendTransformError: If the wrapped call cannot be specialized.
            RuntimeError: If no qkernel tracer is active, the wrapped qkernel's
                emitted result count differs from its declared outputs, or a
                returned slice lacks metadata.
        """
        phase = _phase_to_value(self._phase)
        outputs = self._qkernel(*args, **kwargs)
        get_current_tracer().add_operation(
            GlobalPhaseOperation(operands=[phase], results=[])
        )
        return outputs


def global_phase(
    target: QKernel | Callable[..., Any],
    phase: PhaseValue,
) -> GlobalPhaseGate:
    """Apply a qkernel call followed by ``exp(i * phase)``.

    The phase is represented as a zero-qubit operation and is retained even
    when it is not observable in the surrounding program. A reversible
    qkernel containing the operation acquires an observable relative phase
    when it is coherently controlled. Measurement, reset, allocation,
    classical outputs, and classical-only qkernels remain valid for ordinary
    standalone use.

    Args:
        target (QKernel | Callable[..., Any]): QKernel or gate-like callable
            whose call is followed by the global phase.
        phase (float | int | Float): Phase angle in radians, supplied as a
            Qamomile ``Float`` handle or Python numeric literal.

    Returns:
        GlobalPhaseGate: Callable wrapper with the target's call interface.

    Raises:
        TypeError: If ``target`` cannot be interpreted as a gate-like callable.

    Example:
        >>> import qamomile.circuit as qmc
        >>> @qmc.qkernel
        ... def step(q: qmc.Qubit) -> qmc.Qubit:
        ...     return qmc.x(q)
        >>> @qmc.qkernel
        ... def phased_step(q: qmc.Qubit) -> qmc.Qubit:
        ...     return qmc.global_phase(step, 0.7)(q)
    """
    qkernel = _qkernel_for_callable(target, caller="global_phase")
    return GlobalPhaseGate(qkernel, phase)
