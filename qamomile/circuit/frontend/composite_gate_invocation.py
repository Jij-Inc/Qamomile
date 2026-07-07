"""Invocation helpers for class-based composite gates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qamomile.circuit.frontend.handle.primitives import Qubit
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableImplementation,
    CallableRef,
    CallPolicy,
    CallTransform,
    CompositeGateType,
    signature_from_values,
)
from qamomile.circuit.ir.value import Value


def invoke_composite_gate(
    gate: Any,
    *target_qubits: Qubit,
    controls: Sequence[Qubit] = (),
    strategy: str | None = None,
) -> tuple[Qubit, ...]:
    """Apply a class-based composite gate to frontend qubit handles.

    Args:
        gate (Any): Composite gate instance.
        *target_qubits (Qubit): Target qubits for the gate.
        controls (Sequence[Qubit]): Optional control qubits. Defaults to an
            empty sequence.
        strategy (str | None): Optional strategy name for decomposition.
            Defaults to ``None``.

    Returns:
        tuple[Qubit, ...]: Output qubits, with controls first followed by
        targets.

    Raises:
        ValueError: If arity is invalid or the composite has no implementation
            body.
    """
    _validate_arity(gate, target_qubits, controls)

    impl = gate._build_decomposition_block(target_qubits, strategy)
    if impl is None:
        impl = gate.get_implementation()
    if impl is None:
        gate_name = gate.custom_name or gate.gate_type.value
        raise ValueError(
            f"CompositeGate '{gate_name}' has no implementation. "
            "Use Oracle for opaque resource-only callables."
        )

    gate_name = gate.custom_name or gate.gate_type.value
    consumed_controls = [
        c.consume(operation_name=f"{gate_name}[control]") for c in controls
    ]
    consumed_targets = [
        t.consume(operation_name=f"{gate_name}[target]") for t in target_qubits
    ]

    operands = [q.value for q in [*consumed_controls, *consumed_targets]]
    results = [q.value.next_version() for q in [*consumed_controls, *consumed_targets]]
    op = _build_invoke_operation(
        gate=gate,
        impl=impl,
        operands=operands,
        results=results,
        controls=controls,
        targets=target_qubits,
        strategy=strategy,
        has_controls=bool(consumed_controls),
    )

    get_current_tracer().add_operation(op)
    return _wrap_output_qubits(consumed_controls, consumed_targets, results)


def _validate_arity(
    gate: Any,
    target_qubits: tuple[Qubit, ...],
    controls: Sequence[Qubit],
) -> None:
    """Validate target and control arity for a composite call.

    Args:
        gate (Any): Composite gate instance.
        target_qubits (tuple[Qubit, ...]): Target arguments.
        controls (Sequence[Qubit]): Control arguments.

    Raises:
        ValueError: If either arity does not match the gate declaration.
    """
    gate_name = gate.custom_name or gate.gate_type.value
    if len(target_qubits) != gate.num_target_qubits:
        raise ValueError(
            f"{gate_name} requires {gate.num_target_qubits} target qubits, "
            f"got {len(target_qubits)}"
        )
    if len(controls) != gate.num_control_qubits:
        raise ValueError(
            f"{gate_name} requires {gate.num_control_qubits} control qubits, "
            f"got {len(controls)}"
        )


def _build_invoke_operation(
    *,
    gate: Any,
    impl: Block,
    operands: list[Value],
    results: list[Value],
    controls: Sequence[Qubit],
    targets: tuple[Qubit, ...],
    strategy: str | None,
    has_controls: bool,
) -> Any:
    """Build the IR invoke operation for a composite call.

    Args:
        gate (Any): Composite gate instance.
        impl (Block): Selected implementation body.
        operands (list[Value]): Consumed input values.
        results (list[Value]): Produced output values.
        controls (Sequence[Qubit]): Control handles supplied at the call site.
        targets (tuple[Qubit, ...]): Target handles supplied at the call site.
        strategy (str | None): Selected strategy name.
        has_controls (bool): Whether this call has explicit controls.

    Returns:
        Any: InvokeOperation instance. Kept as ``Any`` to avoid importing the
        concrete operation type into user-facing annotations.
    """
    from qamomile.circuit.ir.operation.callable import InvokeOperation

    is_stdlib_gate = gate.gate_type is not CompositeGateType.CUSTOM
    default_policy = (
        CallPolicy.NATIVE_FIRST if is_stdlib_gate else CallPolicy.PRESERVE_BOX
    )
    gate_ref = CallableRef(
        namespace="qamomile.stdlib" if is_stdlib_gate else "user.composite",
        name=gate.custom_name or gate.gate_type.value,
    )
    attrs = {
        "kind": "composite",
        "gate_type": gate.gate_type.name,
        "num_control_qubits": len(controls),
        "num_target_qubits": len(targets),
        "custom_name": gate.custom_name,
        "strategy_name": strategy,
        "default_policy": default_policy.name,
    }
    transform = CallTransform.CONTROLLED if has_controls else CallTransform.DIRECT
    implementations = list(_extra_callable_implementations(gate))
    implementations.append(
        CallableImplementation(
            transform=CallTransform.DIRECT,
            strategy=strategy,
            body=impl,
        )
    )
    if transform is CallTransform.CONTROLLED:
        implementations.append(
            CallableImplementation(
                transform=CallTransform.CONTROLLED,
                strategy=strategy,
                body=impl,
            )
        )
    return InvokeOperation(
        operands=operands,
        results=results,
        target=gate_ref,
        transform=transform,
        attrs=attrs,
        definition=CallableDef(
            ref=gate_ref,
            signature=signature_from_values(operands, results),
            body=impl,
            implementations=implementations,
            default_policy=default_policy,
            attrs=attrs,
        ),
    )


def _extra_callable_implementations(gate: Any) -> tuple[CallableImplementation, ...]:
    """Return optional implementation candidates supplied by a gate object.

    Args:
        gate (Any): Composite gate object, possibly a decorator wrapper.

    Returns:
        tuple[CallableImplementation, ...]: Additional implementation
        candidates supplied by the object, or an empty tuple.
    """
    getter = getattr(gate, "_callable_implementations", None)
    if getter is None:
        return ()
    implementations = getter()
    if implementations is None:
        return ()
    return tuple(implementations)


def _wrap_output_qubits(
    consumed_controls: list[Qubit],
    consumed_targets: list[Qubit],
    results: list[Value],
) -> tuple[Qubit, ...]:
    """Wrap invoke result values back into frontend qubit handles.

    Args:
        consumed_controls (list[Qubit]): Consumed control handles.
        consumed_targets (list[Qubit]): Consumed target handles.
        results (list[Value]): Result values in control-then-target order.

    Returns:
        tuple[Qubit, ...]: Output handles preserving parent/index provenance.
    """
    output_qubits: list[Qubit] = []
    for i, control in enumerate(consumed_controls):
        output_qubits.append(
            Qubit(
                value=results[i],
                parent=control.parent,
                indices=control.indices,
            )
        )
    offset = len(consumed_controls)
    for i, target in enumerate(consumed_targets):
        output_qubits.append(
            Qubit(
                value=results[offset + i],
                parent=target.parent,
                indices=target.indices,
            )
        )
    return tuple(output_qubits)
