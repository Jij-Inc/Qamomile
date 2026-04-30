from typing import Any, Sequence, Union, overload

from qamomile.circuit.frontend.handle import Float, Qubit, Vector
from qamomile.circuit.frontend.handle.array import Vector as VectorClass
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.gate import (
    GateOperation as IRGateOperation,
    GateOperationType,
)
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import QubitAliasError


def _check_qubit_alias(
    qubits: Sequence[tuple[Any, str]],
    operation_name: str,
) -> None:
    """Raise QubitAliasError if any two qubits share the same logical_id."""
    seen: dict[str, str] = {}  # logical_id -> first role_name
    for q, role in qubits:
        lid = q.value.logical_id
        if lid in seen:
            q_name = q.name or "unnamed"
            raise QubitAliasError(
                f"Cannot use the same qubit as both {seen[lid]} and {role} "
                f"in {operation_name}.\n"
                f"Qubit '{q_name}' appears in both positions.\n\n"
                f"Fix: Use distinct qubits for {seen[lid]} and {role}.",
                handle_name=q_name,
                operation_name=operation_name,
            )
        seen[lid] = role


def _to_theta_value(angle: float | Float) -> Value:
    """Convert a frontend angle to an IR Value for operands."""
    if isinstance(angle, Float):
        return angle.value
    # Raw float → wrap as constant Value
    return Value(type=FloatType(), name="theta").with_const(angle)


def _apply_single_qubit_gate(qubit: Qubit, gate_type: GateOperationType) -> Qubit:
    """Apply a single-qubit gate and return the output qubit."""
    # Consume the input handle (enforces affine type)
    qubit = qubit.consume(operation_name=gate_type.name)

    output_value = qubit.value.next_version()
    output_qubit = Qubit(value=output_value, parent=qubit.parent, indices=qubit.indices)

    gate_op = IRGateOperation(
        gate_type=gate_type, operands=[qubit.value], results=[output_value]
    )
    tracer = get_current_tracer()
    tracer.add_operation(gate_op)
    return output_qubit


def _broadcast_single_qubit_gate(
    qubits: Vector[Qubit], gate_type: GateOperationType
) -> Vector[Qubit]:
    """Broadcast a non-parametric single-qubit gate over every element of a qubit array.

    Lowers to the same `ForOperation` IR a hand-written
    ``for i in qmc.range(n): qs[i] = gate(qs[i])`` would produce, so all
    transpiler passes, resource estimation, and visualization continue to
    work without any broadcast-specific handling.

    Args:
        qubits: The `Vector[Qubit]` to apply the gate to. Must have all
            previously-borrowed elements returned; otherwise the loop body's
            element borrow will fail. The handle itself is not consumed —
            the same instance is returned with element borrows released.
        gate_type: The `GateOperationType` to apply to each element. Must
            correspond to a non-parametric single-qubit gate
            (e.g. `H`, `X`, `Y`, `Z`, `S`, `T`, `SDG`, `TDG`).

    Returns:
        The same `Vector[Qubit]` handle that was passed in, after the
        broadcast loop has been emitted into the active tracer.

    Raises:
        UnreturnedBorrowError: If any element of `qubits` is still borrowed
            when broadcasting begins.
    """
    # Local import to avoid a frontend.operation circular import chain.
    from qamomile.circuit.frontend.operation.control_flow import for_loop

    qubits.validate_all_returned()
    n = qubits.shape[0]
    # ``var_name`` is display-only — the IR uses UUIDs for identity. Use ``i``
    # so the visualization matches a hand-written ``for i in qmc.range(n)``.
    with for_loop(0, n, var_name="i") as i:
        qubits[i] = _apply_single_qubit_gate(qubits[i], gate_type)
    return qubits


def _broadcast_rotation_gate(
    qubits: Vector[Qubit],
    angle: float | Float,
    gate_type: GateOperationType,
) -> Vector[Qubit]:
    """Broadcast a parametric single-qubit rotation gate over a qubit array.

    Like `_broadcast_single_qubit_gate`, but for rotation gates that take a
    shared `angle`. The same scalar `angle` is applied to every qubit;
    per-qubit angle arrays remain a per-element-loop concern (e.g.
    `rx_layer`).

    Args:
        qubits: The `Vector[Qubit]` to apply the rotation to.
        angle: The rotation angle in radians, shared across all qubits in
            the broadcast. Accepts a Python `float` or a `Float` handle.
        gate_type: The rotation gate kind (`RX`, `RY`, `RZ`, or `P`).

    Returns:
        The same `Vector[Qubit]` handle that was passed in.

    Raises:
        UnreturnedBorrowError: If any element of `qubits` is still borrowed
            when broadcasting begins.
    """
    from qamomile.circuit.frontend.operation.control_flow import for_loop

    qubits.validate_all_returned()
    n = qubits.shape[0]
    with for_loop(0, n, var_name="i") as i:
        qubits[i] = _apply_rotation_gate(qubits[i], angle, gate_type)
    return qubits


def _apply_two_qubit_gate(
    control: Qubit, target: Qubit, gate_type: GateOperationType
) -> tuple[Qubit, Qubit]:
    """Apply a two-qubit gate and return the output qubits."""
    _check_qubit_alias([(control, "control"), (target, "target")], gate_type.name)

    # Consume both input handles (enforces affine type)
    control = control.consume(operation_name=f"{gate_type.name}[control]")
    target = target.consume(operation_name=f"{gate_type.name}[target]")

    ctrl_out_value = control.value.next_version()
    tgt_out_value = target.value.next_version()

    ctrl_out = Qubit(
        value=ctrl_out_value, parent=control.parent, indices=control.indices
    )
    tgt_out = Qubit(value=tgt_out_value, parent=target.parent, indices=target.indices)

    gate_op = IRGateOperation(
        gate_type=gate_type,
        operands=[control.value, target.value],
        results=[ctrl_out_value, tgt_out_value],
    )
    tracer = get_current_tracer()
    tracer.add_operation(gate_op)
    return ctrl_out, tgt_out


def _apply_three_qubit_gate(
    control1: Qubit, control2: Qubit, target: Qubit, gate_type: GateOperationType
) -> tuple[Qubit, Qubit, Qubit]:
    """Apply a three-qubit gate and return the output qubits."""
    _check_qubit_alias(
        [(control1, "control1"), (control2, "control2"), (target, "target")],
        gate_type.name,
    )

    # Consume all three input handles (enforces affine type)
    control1 = control1.consume(operation_name=f"{gate_type.name}[control1]")
    control2 = control2.consume(operation_name=f"{gate_type.name}[control2]")
    target = target.consume(operation_name=f"{gate_type.name}[target]")

    ctrl1_out_value = control1.value.next_version()
    ctrl2_out_value = control2.value.next_version()
    tgt_out_value = target.value.next_version()

    ctrl1_out = Qubit(
        value=ctrl1_out_value, parent=control1.parent, indices=control1.indices
    )
    ctrl2_out = Qubit(
        value=ctrl2_out_value, parent=control2.parent, indices=control2.indices
    )
    tgt_out = Qubit(value=tgt_out_value, parent=target.parent, indices=target.indices)

    gate_op = IRGateOperation(
        gate_type=gate_type,
        operands=[control1.value, control2.value, target.value],
        results=[ctrl1_out_value, ctrl2_out_value, tgt_out_value],
    )
    tracer = get_current_tracer()
    tracer.add_operation(gate_op)
    return ctrl1_out, ctrl2_out, tgt_out


def _dispatch_single_qubit_gate(
    target: Union[Qubit, Vector[Qubit]],
    gate_type: GateOperationType,
) -> Union[Qubit, Vector[Qubit]]:
    """Apply a non-parametric single-qubit gate to a qubit or qubit array.

    Mirrors the dispatch pattern used by ``measure``: a `Vector[Qubit]`
    input lowers to a broadcast `ForOperation`, while a scalar `Qubit`
    takes the single-element path.

    Args:
        target: The qubit or qubit array to apply the gate to.
        gate_type: The gate kind to apply.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a
            `Vector[Qubit]`.
    """
    if isinstance(target, VectorClass) and target.element_type == Qubit:
        return _broadcast_single_qubit_gate(target, gate_type)
    if isinstance(target, Qubit):
        return _apply_single_qubit_gate(target, gate_type)
    raise TypeError(
        f"Unsupported target type for {gate_type.name}: {type(target).__name__}. "
        "Expected Qubit or Vector[Qubit]."
    )


@overload
def h(target: Qubit) -> Qubit: ...
@overload
def h(target: Vector[Qubit]) -> Vector[Qubit]: ...
def h(target: Union[Qubit, Vector[Qubit]]) -> Union[Qubit, Vector[Qubit]]:
    """Hadamard gate.

    Applied to a single `Qubit` it returns the transformed qubit. Applied
    to a `Vector[Qubit]` it broadcasts the gate over every element via a
    transpile-time loop, equivalent to
    ``for i in qmc.range(n): qs[i] = h(qs[i])``.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]` to apply H to.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a `Vector[Qubit]`.
    """
    return _dispatch_single_qubit_gate(target, GateOperationType.H)


@overload
def x(target: Qubit) -> Qubit: ...
@overload
def x(target: Vector[Qubit]) -> Vector[Qubit]: ...
def x(target: Union[Qubit, Vector[Qubit]]) -> Union[Qubit, Vector[Qubit]]:
    """Pauli-X gate (NOT gate).

    Broadcasts over a `Vector[Qubit]` when applied to one.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]`.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a `Vector[Qubit]`.
    """
    return _dispatch_single_qubit_gate(target, GateOperationType.X)


@overload
def y(target: Qubit) -> Qubit: ...
@overload
def y(target: Vector[Qubit]) -> Vector[Qubit]: ...
def y(target: Union[Qubit, Vector[Qubit]]) -> Union[Qubit, Vector[Qubit]]:
    """Pauli-Y gate.

    Broadcasts over a `Vector[Qubit]` when applied to one.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]`.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a `Vector[Qubit]`.
    """
    return _dispatch_single_qubit_gate(target, GateOperationType.Y)


@overload
def z(target: Qubit) -> Qubit: ...
@overload
def z(target: Vector[Qubit]) -> Vector[Qubit]: ...
def z(target: Union[Qubit, Vector[Qubit]]) -> Union[Qubit, Vector[Qubit]]:
    """Pauli-Z gate.

    Broadcasts over a `Vector[Qubit]` when applied to one.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]`.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a `Vector[Qubit]`.
    """
    return _dispatch_single_qubit_gate(target, GateOperationType.Z)


@overload
def t(target: Qubit) -> Qubit: ...
@overload
def t(target: Vector[Qubit]) -> Vector[Qubit]: ...
def t(target: Union[Qubit, Vector[Qubit]]) -> Union[Qubit, Vector[Qubit]]:
    """T gate (fourth root of Z).

    Broadcasts over a `Vector[Qubit]` when applied to one.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]`.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a `Vector[Qubit]`.
    """
    return _dispatch_single_qubit_gate(target, GateOperationType.T)


@overload
def s(target: Qubit) -> Qubit: ...
@overload
def s(target: Vector[Qubit]) -> Vector[Qubit]: ...
def s(target: Union[Qubit, Vector[Qubit]]) -> Union[Qubit, Vector[Qubit]]:
    """S gate (square root of Z).

    Broadcasts over a `Vector[Qubit]` when applied to one.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]`.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a `Vector[Qubit]`.
    """
    return _dispatch_single_qubit_gate(target, GateOperationType.S)


@overload
def sdg(target: Qubit) -> Qubit: ...
@overload
def sdg(target: Vector[Qubit]) -> Vector[Qubit]: ...
def sdg(target: Union[Qubit, Vector[Qubit]]) -> Union[Qubit, Vector[Qubit]]:
    """S-dagger gate (inverse of S gate).

    Broadcasts over a `Vector[Qubit]` when applied to one.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]`.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a `Vector[Qubit]`.
    """
    return _dispatch_single_qubit_gate(target, GateOperationType.SDG)


@overload
def tdg(target: Qubit) -> Qubit: ...
@overload
def tdg(target: Vector[Qubit]) -> Vector[Qubit]: ...
def tdg(target: Union[Qubit, Vector[Qubit]]) -> Union[Qubit, Vector[Qubit]]:
    """T-dagger gate (inverse of T gate).

    Broadcasts over a `Vector[Qubit]` when applied to one.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]`.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a `Vector[Qubit]`.
    """
    return _dispatch_single_qubit_gate(target, GateOperationType.TDG)


def cx(control: Qubit, target: Qubit) -> tuple[Qubit, Qubit]:
    """CNOT (Controlled-X) gate."""
    return _apply_two_qubit_gate(control, target, GateOperationType.CX)


def cz(control: Qubit, target: Qubit) -> tuple[Qubit, Qubit]:
    """CZ (Controlled-Z) gate."""
    return _apply_two_qubit_gate(control, target, GateOperationType.CZ)


def _apply_phase_gate(qubit: Qubit, theta: float | Float) -> Qubit:
    """Apply the phase gate ``P(theta)`` to a single qubit.

    Args:
        qubit: The qubit to apply the phase to.
        theta: Phase angle in radians, as a Python float or a `Float` handle.

    Returns:
        The output qubit handle after the phase rotation.
    """
    # Consume the input handle (enforces affine type)
    qubit = qubit.consume(operation_name="P")

    output_value = qubit.value.next_version()
    output_qubit = Qubit(value=output_value, parent=qubit.parent, indices=qubit.indices)

    theta_value = _to_theta_value(theta)

    p_op = IRGateOperation.rotation(
        gate_type=GateOperationType.P,
        qubits=[qubit.value],
        theta=theta_value,
        results=[output_value],
    )

    tracer = get_current_tracer()
    tracer.add_operation(p_op)
    return output_qubit


def _broadcast_phase_gate(qubits: Vector[Qubit], theta: float | Float) -> Vector[Qubit]:
    """Broadcast the phase gate ``P(theta)`` over every element of a qubit array.

    Lowers to a `ForOperation` so that downstream passes see the same IR
    they would for an explicit hand-written loop.

    Args:
        qubits: The `Vector[Qubit]` to apply the phase gate to.
        theta: Phase angle in radians, shared across all qubits.

    Returns:
        The same `Vector[Qubit]` handle, with the broadcast loop emitted
        into the active tracer.

    Raises:
        UnreturnedBorrowError: If any element of `qubits` is still borrowed
            when broadcasting begins.
    """
    from qamomile.circuit.frontend.operation.control_flow import for_loop

    qubits.validate_all_returned()
    n = qubits.shape[0]
    with for_loop(0, n, var_name="i") as i:
        qubits[i] = _apply_phase_gate(qubits[i], theta)
    return qubits


@overload
def p(target: Qubit, theta: float | Float) -> Qubit: ...
@overload
def p(target: Vector[Qubit], theta: float | Float) -> Vector[Qubit]: ...
def p(
    target: Union[Qubit, Vector[Qubit]], theta: float | Float
) -> Union[Qubit, Vector[Qubit]]:
    """Phase gate: ``P(theta)|1> = e^{i*theta}|1>``.

    Broadcasts the same `theta` over every qubit when called with a
    `Vector[Qubit]`.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]` to apply the phase to.
        theta: Phase angle in radians.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a `Vector[Qubit]`.
    """
    if isinstance(target, VectorClass) and target.element_type == Qubit:
        return _broadcast_phase_gate(target, theta)
    if isinstance(target, Qubit):
        return _apply_phase_gate(target, theta)
    raise TypeError(
        f"Unsupported target type for P: {type(target).__name__}. "
        "Expected Qubit or Vector[Qubit]."
    )


def cp(control: Qubit, target: Qubit, theta: float | Float) -> tuple[Qubit, Qubit]:
    """Controlled-Phase gate."""
    _check_qubit_alias([(control, "control"), (target, "target")], "CP")

    # Consume both input handles (enforces affine type)
    control = control.consume(operation_name="CP[control]")
    target = target.consume(operation_name="CP[target]")

    ctrl_out_value = control.value.next_version()
    tgt_out_value = target.value.next_version()

    ctrl_out = Qubit(
        value=ctrl_out_value, parent=control.parent, indices=control.indices
    )
    tgt_out = Qubit(value=tgt_out_value, parent=target.parent, indices=target.indices)

    theta_value = _to_theta_value(theta)

    cp_op = IRGateOperation.rotation(
        gate_type=GateOperationType.CP,
        qubits=[control.value, target.value],
        theta=theta_value,
        results=[ctrl_out_value, tgt_out_value],
    )

    tracer = get_current_tracer()
    tracer.add_operation(cp_op)
    return ctrl_out, tgt_out


def _apply_rotation_gate(
    qubit: Qubit, angle: float | Float, gate_type: GateOperationType
) -> Qubit:
    """Apply a rotation gate with angle."""
    # Consume the input handle (enforces affine type)
    qubit = qubit.consume(operation_name=gate_type.name)

    output_value = qubit.value.next_version()
    output_qubit = Qubit(value=output_value, parent=qubit.parent, indices=qubit.indices)

    theta_value = _to_theta_value(angle)

    gate_op = IRGateOperation.rotation(
        gate_type=gate_type,
        qubits=[qubit.value],
        theta=theta_value,
        results=[output_value],
    )

    tracer = get_current_tracer()
    tracer.add_operation(gate_op)
    return output_qubit


def _dispatch_rotation_gate(
    target: Union[Qubit, Vector[Qubit]],
    angle: float | Float,
    gate_type: GateOperationType,
) -> Union[Qubit, Vector[Qubit]]:
    """Apply a rotation gate to a qubit or broadcast over a qubit array.

    A `Vector[Qubit]` input lowers to a broadcast `ForOperation` that
    applies the same `angle` to every element.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]`.
        angle: Rotation angle in radians, as a Python float or `Float`
            handle. The same scalar is shared across all qubits in the
            broadcast case.
        gate_type: The rotation gate kind (`RX`, `RY`, or `RZ`).

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a
            `Vector[Qubit]`.
    """
    if isinstance(target, VectorClass) and target.element_type == Qubit:
        return _broadcast_rotation_gate(target, angle, gate_type)
    if isinstance(target, Qubit):
        return _apply_rotation_gate(target, angle, gate_type)
    raise TypeError(
        f"Unsupported target type for {gate_type.name}: {type(target).__name__}. "
        "Expected Qubit or Vector[Qubit]."
    )


@overload
def rx(target: Qubit, angle: float | Float) -> Qubit: ...
@overload
def rx(target: Vector[Qubit], angle: float | Float) -> Vector[Qubit]: ...
def rx(
    target: Union[Qubit, Vector[Qubit]], angle: float | Float
) -> Union[Qubit, Vector[Qubit]]:
    """Rotation around X-axis: ``RX(angle) = exp(-i * angle/2 * X)``.

    Broadcasts the same `angle` over every qubit when called with a
    `Vector[Qubit]`.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]`.
        angle: Rotation angle in radians.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a `Vector[Qubit]`.
    """
    return _dispatch_rotation_gate(target, angle, GateOperationType.RX)


@overload
def ry(target: Qubit, angle: float | Float) -> Qubit: ...
@overload
def ry(target: Vector[Qubit], angle: float | Float) -> Vector[Qubit]: ...
def ry(
    target: Union[Qubit, Vector[Qubit]], angle: float | Float
) -> Union[Qubit, Vector[Qubit]]:
    """Rotation around Y-axis: ``RY(angle) = exp(-i * angle/2 * Y)``.

    Broadcasts the same `angle` over every qubit when called with a
    `Vector[Qubit]`.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]`.
        angle: Rotation angle in radians.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a `Vector[Qubit]`.
    """
    return _dispatch_rotation_gate(target, angle, GateOperationType.RY)


@overload
def rz(target: Qubit, angle: float | Float) -> Qubit: ...
@overload
def rz(target: Vector[Qubit], angle: float | Float) -> Vector[Qubit]: ...
def rz(
    target: Union[Qubit, Vector[Qubit]], angle: float | Float
) -> Union[Qubit, Vector[Qubit]]:
    """Rotation around Z-axis: ``RZ(angle) = exp(-i * angle/2 * Z)``.

    Broadcasts the same `angle` over every qubit when called with a
    `Vector[Qubit]`.

    Args:
        target: A single `Qubit` or a `Vector[Qubit]`.
        angle: Rotation angle in radians.

    Returns:
        A `Qubit` for scalar input, a `Vector[Qubit]` for array input.

    Raises:
        TypeError: If `target` is neither a `Qubit` nor a `Vector[Qubit]`.
    """
    return _dispatch_rotation_gate(target, angle, GateOperationType.RZ)


def rzz(qubit_0: Qubit, qubit_1: Qubit, angle: float | Float) -> tuple[Qubit, Qubit]:
    """RZZ gate: exp(-i * angle/2 * Z ⊗ Z).

    The RZZ gate applies a rotation around the ZZ axis on two qubits.

    Args:
        qubit_0: First qubit.
        qubit_1: Second qubit.
        angle: Rotation angle in radians.

    Returns:
        Tuple of (qubit_0_out, qubit_1_out) after RZZ.
    """
    _check_qubit_alias([(qubit_0, "qubit_0"), (qubit_1, "qubit_1")], "RZZ")

    # Consume both input handles (enforces affine type)
    qubit_0 = qubit_0.consume(operation_name="RZZ[0]")
    qubit_1 = qubit_1.consume(operation_name="RZZ[1]")

    q0_out_value = qubit_0.value.next_version()
    q1_out_value = qubit_1.value.next_version()

    q0_out = Qubit(value=q0_out_value, parent=qubit_0.parent, indices=qubit_0.indices)
    q1_out = Qubit(value=q1_out_value, parent=qubit_1.parent, indices=qubit_1.indices)

    theta_value = _to_theta_value(angle)

    rzz_op = IRGateOperation.rotation(
        gate_type=GateOperationType.RZZ,
        qubits=[qubit_0.value, qubit_1.value],
        theta=theta_value,
        results=[q0_out_value, q1_out_value],
    )

    tracer = get_current_tracer()
    tracer.add_operation(rzz_op)
    return q0_out, q1_out


def swap(qubit_0: Qubit, qubit_1: Qubit) -> tuple[Qubit, Qubit]:
    """SWAP gate: exchanges two qubits.

    The SWAP gate swaps the states of two qubits.

    Args:
        qubit_0: First qubit.
        qubit_1: Second qubit.

    Returns:
        Tuple of (qubit_0_out, qubit_1_out) after SWAP.
    """
    return _apply_two_qubit_gate(qubit_0, qubit_1, GateOperationType.SWAP)


def ccx(control1: Qubit, control2: Qubit, target: Qubit) -> tuple[Qubit, Qubit, Qubit]:
    """Toffoli (CCX) gate: flips target when both controls are |1>.

    Args:
        control1: First control qubit.
        control2: Second control qubit.
        target: Target qubit.

    Returns:
        Tuple of (control1_out, control2_out, target_out) after CCX.
    """
    return _apply_three_qubit_gate(
        control1, control2, target, GateOperationType.TOFFOLI
    )
