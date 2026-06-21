"""Frontend combinator for applying a global phase to a qkernel.

``qmc.global_phase(kernel, theta)`` returns a callable that applies the
wrapped kernel's unitary *and* multiplies the whole state by ``e^{i*theta}``.
It mirrors the higher-order combinator family (:func:`qmc.control`,
:func:`qmc.inverse`): the phase attaches to a whole unitary (a kernel), not
to a single qubit, which matches how every major SDK models global phase
(Qiskit ``GlobalPhaseGate`` / circuit ``global_phase``, PennyLane
``qml.GlobalPhase``, Qualtran ``GlobalPhase`` bloq).

Standalone, the global phase is physically unobservable and is folded into a
native circuit-level accumulator (Qiskit) or dropped (CUDA-Q / QURI Parts).
Under ``qmc.control`` it becomes a *relative* phase on the control qubit(s)
-- the case that matters for projector-controlled phase gates and QSVT.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from qamomile.circuit.frontend.handle import Handle
from qamomile.circuit.frontend.handle.array import ArrayBase, VectorView
from qamomile.circuit.frontend.handle.primitives import Float
from qamomile.circuit.frontend.operation.control import _qkernel_for_callable
from qamomile.circuit.frontend.operation.inverse import (
    _as_value,
    _InputBinding,
    _static_quantum_width,
    _validate_input_shape,
)
from qamomile.circuit.frontend.qkernel import QKernel, _promote_literal_to_handle
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.global_phase_block import GlobalPhaseBlockOperation
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase

# A phase angle may be supplied as a frontend ``Float`` handle (symbolic /
# runtime parameter) or a plain Python number (compile-time constant).
PhaseValue = float | int | Float


def _phase_to_value(phase: PhaseValue) -> Value:
    """Coerce a user-supplied phase angle into a classical ``FloatType`` Value.

    Args:
        phase (float | int | Float): Phase angle in radians, as a Qamomile
            ``Float`` handle (symbolic) or a Python number (constant).

    Returns:
        Value: The underlying ``Float`` handle Value, or a fresh constant
            ``FloatType`` Value for numeric input.

    Raises:
        TypeError: If ``phase`` is a ``bool``, a non-``Float`` handle, or
            any other non-numeric type.
    """
    if isinstance(phase, Float):
        return phase.value
    if isinstance(phase, Handle):
        raise TypeError(
            f"global_phase(): phase must be a Float handle or a number, got "
            f"{type(phase).__name__}. Pass a qmc.Float (or a Python float)."
        )
    if isinstance(phase, bool):
        raise TypeError(
            "global_phase(): phase must be a number, not bool. Pass a numeric "
            "angle in radians (or a qmc.Float handle)."
        )
    if isinstance(phase, (int, float)):
        return Value(type=FloatType(), name="global_phase").with_const(float(phase))
    raise TypeError(
        f"global_phase(): phase must be a Float handle or a number, got "
        f"{type(phase).__name__}."
    )


class GlobalPhaseGate:
    """Callable wrapper that applies a QKernel's unitary plus a global phase.

    Created by :func:`global_phase`. The wrapper is called with the same
    quantum / classical arguments as the wrapped kernel; the phase angle is
    fixed at construction time.

    Args:
        qkernel (QKernel): Kernel whose unitary should be emitted.
        phase (float | int | Float): Global phase angle in radians.

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
            qkernel (QKernel): Kernel whose unitary should be emitted.
            phase (float | int | Float): Global phase angle in radians.
        """
        self._qkernel = qkernel
        self._phase = phase

    def _bind_arguments(self, *args: Any, **kwargs: Any) -> Any:
        """Bind and literal-promote call arguments to the kernel signature.

        Args:
            *args (Any): Positional arguments supplied by the caller.
            **kwargs (Any): Keyword arguments supplied by the caller.

        Returns:
            inspect.BoundArguments: Bound, default-filled argument mapping.

        Raises:
            TypeError: If any final argument is not a frontend ``Handle``.
        """
        bound_args = self._qkernel.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for name, value in list(bound_args.arguments.items()):
            expected_type = self._qkernel.input_types.get(name)
            if expected_type is not None:
                bound_args.arguments[name] = _promote_literal_to_handle(
                    value, expected_type
                )
        for name, value in bound_args.arguments.items():
            if not isinstance(value, Handle):
                raise TypeError(
                    f"global_phase(): argument {name!r} must be a Handle "
                    f"instance, got {type(value).__name__}."
                )
        return bound_args

    def _select_block(self, arguments: dict[str, Any]) -> Block:
        """Select a cached or call-time-specialized block.

        Args:
            arguments (dict[str, Any]): Bound call arguments.

        Returns:
            Block: Block whose operations should be emitted.
        """
        block_ir = None
        if not self._qkernel._specializing:
            spec = self._qkernel._extract_calltime_specialization(arguments)
            if spec is not None:
                sub_parameters, sub_bindings, sub_qubit_sizes = spec
                self._qkernel._specializing = True
                try:
                    block_ir = self._qkernel._build_specialized(
                        parameters=sub_parameters,
                        bindings=sub_bindings,
                        qubit_sizes=sub_qubit_sizes,
                    )
                finally:
                    self._qkernel._specializing = False
        if block_ir is None:
            block_ir = self._qkernel.block
        return block_ir

    def _prepare_inputs(
        self, block: Block, arguments: dict[str, Any]
    ) -> list[_InputBinding]:
        """Consume quantum arguments and pair them with block inputs.

        Args:
            block (Block): Selected block.
            arguments (dict[str, Any]): Bound call arguments.

        Returns:
            list[_InputBinding]: One binding per block input.
        """
        bindings: list[_InputBinding] = []
        for name, block_input in zip(block.label_args, block.input_values):
            handle = cast(Handle, arguments[name])
            active_handle = handle
            if handle._should_enforce_linear() and not isinstance(handle, VectorView):
                active_handle = handle.consume(
                    operation_name=f"GlobalPhase[{self._qkernel.name}]"
                )
            _validate_input_shape(name, block_input, active_handle.value)
            bindings.append(
                _InputBinding(
                    name=name,
                    handle=handle,
                    active_handle=active_handle,
                    block_input=block_input,
                )
            )
        return bindings

    def _wrap_quantum_result(self, binding: _InputBinding, value: ValueBase) -> Handle:
        """Wrap an output value as a frontend handle preserving input kind.

        Args:
            binding (_InputBinding): Original input binding.
            value (ValueBase): Final output value for this operand.

        Returns:
            Handle: Frontend handle carrying ``value`` with the same kind
                (scalar / Vector / VectorView) and array write-back metadata
                as the input handle.

        Raises:
            TypeError: If an array input maps to a scalar output or vice
                versa.
        """
        active = binding.active_handle
        if isinstance(active, VectorView):
            if not isinstance(value, ArrayValue):
                raise TypeError(
                    "global_phase(): VectorView input produced scalar output."
                )
            new_view = VectorView._wrap_unregistered(
                parent=active._slice_parent,
                sliced_av=value,
                length=active._shape[0],
                start_uint=active._slice_start,
                step_uint=active._slice_step,
            )
            active._transfer_borrow_to(new_view, f"GlobalPhase[{self._qkernel.name}]")
            return new_view
        if isinstance(active, ArrayBase):
            if not isinstance(value, ArrayValue):
                raise TypeError("global_phase(): array input produced scalar output.")
            return type(active)._create_from_value(
                value=value,
                shape=active.shape,
                name=active.value.name,
            )
        if not isinstance(value, Value) or isinstance(value, ArrayValue):
            raise TypeError("global_phase(): scalar input produced array output.")
        return type(active)(
            value=value,
            parent=active.parent,
            indices=active.indices,
            name=active.name,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Apply ``e^{i*theta} * U_kernel`` at the current trace site.

        Args:
            *args (Any): Positional arguments for the wrapped kernel.
            **kwargs (Any): Keyword arguments for the wrapped kernel.

        Returns:
            Any: Quantum output handle, or a tuple of handles when the
                wrapped kernel has multiple quantum outputs.
        """
        bound_args = self._bind_arguments(*args, **kwargs)
        block = self._select_block(bound_args.arguments)
        bindings = self._prepare_inputs(block, bound_args.arguments)

        quantum_bindings = [binding for binding in bindings if binding.is_quantum]
        quantum_values = [
            _as_value(binding.active_handle.value, "global_phase qkernel input")
            for binding in quantum_bindings
        ]
        # A Vector[Qubit] input contributes many scalar qubits to the backend
        # width but remains a single operand/result value. Symbolic-width
        # inputs contribute ``None`` and are re-derived from resolved qubits
        # at emit time.
        target_width = sum(
            width
            for value in quantum_values
            if (width := _static_quantum_width(value)) is not None
        )
        parameter_values = [
            binding.active_handle.value
            for binding in bindings
            if not binding.is_quantum
        ]
        result_values = [value.next_version() for value in quantum_values]

        op = GlobalPhaseBlockOperation(
            operands=cast("list[Value]", [*quantum_values, *parameter_values]),
            results=result_values,
            num_control_qubits=0,
            num_target_qubits=target_width,
            custom_name=f"{block.name}_global_phase",
            source_block=block,
            phase=_phase_to_value(self._phase),
        )
        get_current_tracer().add_operation(op)

        outputs = [
            self._wrap_quantum_result(binding, value)
            for binding, value in zip(quantum_bindings, result_values)
        ]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def global_phase(
    target: QKernel | Callable[..., Any], phase: PhaseValue
) -> GlobalPhaseGate:
    """Apply a global phase ``e^{i*phase}`` on top of a kernel's unitary.

    A higher-order combinator (like :func:`qmc.control` / :func:`qmc.inverse`)
    that wraps a kernel so its emitted unitary is multiplied by the scalar
    global phase ``e^{i*phase}``. The phase attaches to the whole unitary,
    not to any qubit. Standalone it is unobservable; controlling the result
    (``qmc.control(qmc.global_phase(k, theta))``) turns it into an
    observable relative phase on the control -- the projector-controlled
    phase / QSVT building block.

    Args:
        target (QKernel | Callable[..., Any]): Kernel (or native gate
            callable) whose unitary should carry the global phase.
        phase (float | int | Float): Global phase angle in radians, as a
            Qamomile ``Float`` handle (symbolic / runtime parameter) or a
            Python number (compile-time constant).

    Returns:
        GlobalPhaseGate: A callable wrapper invoked with the wrapped
            kernel's arguments.

    Raises:
        TypeError: If ``phase`` is not a ``Float`` handle or a number, or
            if ``target`` cannot be interpreted as a gate-like callable.

    Example:
        >>> import qamomile.circuit as qmc
        >>> @qmc.qkernel
        ... def step(q: qmc.Qubit) -> qmc.Qubit:
        ...     return qmc.x(q)
        >>> @qmc.qkernel
        ... def hadamard_test(ctrl: qmc.Qubit, q: qmc.Qubit) -> tuple:
        ...     ctrl = qmc.h(ctrl)
        ...     ctrl, q = qmc.control(qmc.global_phase(step, 0.7))(ctrl, q)
        ...     ctrl = qmc.h(ctrl)
        ...     return ctrl, q
    """
    qkernel = _qkernel_for_callable(target, caller="global_phase")
    return GlobalPhaseGate(qkernel, phase)
