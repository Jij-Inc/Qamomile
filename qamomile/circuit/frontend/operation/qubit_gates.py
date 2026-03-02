from qamomile.circuit.frontend.handle import Float, Qubit
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.gate import GateOperation as IRGateOperation
from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.transpiler.errors import QubitAliasError


def _apply_single_qubit_gate(qubit: Qubit, gate_type: GateOperationType) -> Qubit:
    """Apply a single-qubit gate and return the output qubit."""
    # Consume the input handle (enforces linear type)
    qubit = qubit.consume(operation_name=gate_type.name)

    output_value = qubit.value.next_version()
    output_qubit = Qubit(value=output_value, parent=qubit.parent, indices=qubit.indices)

    gate_op = IRGateOperation(
        gate_type=gate_type, operands=[qubit.value], results=[output_value]
    )
    tracer = get_current_tracer()
    tracer.add_operation(gate_op)
    return output_qubit


def _apply_two_qubit_gate(
    control: Qubit, target: Qubit, gate_type: GateOperationType
) -> tuple[Qubit, Qubit]:
    """Apply a two-qubit gate and return the output qubits."""
    # Check for aliasing (same physical qubit used twice)
    # Use logical_id to track physical qubit identity across SSA versions
    if control.value.logical_id == target.value.logical_id:
        ctrl_name = control.name or "unnamed"
        raise QubitAliasError(
            f"Cannot use the same qubit as both control and target in {gate_type.name}.\n"
            f"Qubit '{ctrl_name}' appears in both positions.\n\n"
            f"Fix: Use distinct qubits for control and target.",
            handle_name=ctrl_name,
            operation_name=gate_type.name,
        )

    # Consume both input handles (enforces linear type)
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
    # Check for aliasing - all 3 pairs
    pairs = [
        (control1, control2, "control1", "control2"),
        (control1, target, "control1", "target"),
        (control2, target, "control2", "target"),
    ]
    for q_a, q_b, name_a, name_b in pairs:
        if q_a.value.logical_id == q_b.value.logical_id:
            q_name = q_a.name or "unnamed"
            raise QubitAliasError(
                f"Cannot use the same qubit as both {name_a} and {name_b} in {gate_type.name}.\n"
                f"Qubit '{q_name}' appears in both positions.\n\n"
                f"Fix: Use distinct qubits for {name_a} and {name_b}.",
                handle_name=q_name,
                operation_name=gate_type.name,
            )

    # Consume all three input handles (enforces linear type)
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


def h(qubit: Qubit) -> Qubit:
    """Hadamard gate."""
    return _apply_single_qubit_gate(qubit, GateOperationType.H)


def x(qubit: Qubit) -> Qubit:
    """Pauli-X gate (NOT gate)."""
    return _apply_single_qubit_gate(qubit, GateOperationType.X)


def z(qubit: Qubit) -> Qubit:
    """Pauli-Z gate."""
    return _apply_single_qubit_gate(qubit, GateOperationType.Z)


def cx(control: Qubit, target: Qubit) -> tuple[Qubit, Qubit]:
    """CNOT (Controlled-X) gate."""
    return _apply_two_qubit_gate(control, target, GateOperationType.CX)


def cz(control: Qubit, target: Qubit) -> tuple[Qubit, Qubit]:
    """CZ (Controlled-Z) gate."""
    return _apply_two_qubit_gate(control, target, GateOperationType.CZ)


def p(qubit: Qubit, theta: float | Float) -> Qubit:
    """Phase gate: P(theta)|1> = e^{i*theta}|1>."""
    # Consume the input handle (enforces linear type)
    qubit = qubit.consume(operation_name="P")

    output_value = qubit.value.next_version()
    output_qubit = Qubit(value=output_value, parent=qubit.parent, indices=qubit.indices)

    theta_value = theta.value if isinstance(theta, Float) else theta

    p_op = IRGateOperation(
        gate_type=GateOperationType.P, operands=[qubit.value], results=[output_value]
    )
    p_op.theta = theta_value

    tracer = get_current_tracer()
    tracer.add_operation(p_op)
    return output_qubit


def cp(control: Qubit, target: Qubit, theta: float | Float) -> tuple[Qubit, Qubit]:
    """Controlled-Phase gate."""
    # Check for aliasing (same physical qubit used twice)
    # Use logical_id to track physical qubit identity across SSA versions
    if control.value.logical_id == target.value.logical_id:
        ctrl_name = control.name or "unnamed"
        raise QubitAliasError(
            f"Cannot use the same qubit as both control and target in CP.\n"
            f"Qubit '{ctrl_name}' appears in both positions.\n\n"
            f"Fix: Use distinct qubits for control and target.",
            handle_name=ctrl_name,
            operation_name="CP",
        )

    # Consume both input handles (enforces linear type)
    control = control.consume(operation_name="CP[control]")
    target = target.consume(operation_name="CP[target]")

    ctrl_out_value = control.value.next_version()
    tgt_out_value = target.value.next_version()

    ctrl_out = Qubit(
        value=ctrl_out_value, parent=control.parent, indices=control.indices
    )
    tgt_out = Qubit(value=tgt_out_value, parent=target.parent, indices=target.indices)

    theta_value = theta.value if isinstance(theta, Float) else theta

    cp_op = IRGateOperation(
        gate_type=GateOperationType.CP,
        operands=[control.value, target.value],
        results=[ctrl_out_value, tgt_out_value],
    )
    cp_op.theta = theta_value

    tracer = get_current_tracer()
    tracer.add_operation(cp_op)
    return ctrl_out, tgt_out


def _apply_rotation_gate(
    qubit: Qubit, angle: float | Float, gate_type: GateOperationType
) -> Qubit:
    """Apply a rotation gate with angle."""
    # Consume the input handle (enforces linear type)
    qubit = qubit.consume(operation_name=gate_type.name)

    output_value = qubit.value.next_version()
    output_qubit = Qubit(value=output_value, parent=qubit.parent, indices=qubit.indices)

    angle_value = angle.value if isinstance(angle, Float) else angle

    gate_op = IRGateOperation(
        gate_type=gate_type, operands=[qubit.value], results=[output_value]
    )
    gate_op.theta = angle_value

    tracer = get_current_tracer()
    tracer.add_operation(gate_op)
    return output_qubit


def rx(qubit: Qubit, angle: float | Float) -> Qubit:
    """Rotation around X-axis: RX(angle) = exp(-i * angle/2 * X)."""
    return _apply_rotation_gate(qubit, angle, GateOperationType.RX)


def ry(qubit: Qubit, angle: float | Float) -> Qubit:
    """Rotation around Y-axis: RY(angle) = exp(-i * angle/2 * Y)."""
    return _apply_rotation_gate(qubit, angle, GateOperationType.RY)


def rz(qubit: Qubit, angle: float | Float) -> Qubit:
    """Rotation around Z-axis: RZ(angle) = exp(-i * angle/2 * Z)."""
    return _apply_rotation_gate(qubit, angle, GateOperationType.RZ)


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
    # Check for aliasing (same physical qubit used twice)
    # Use logical_id to track physical qubit identity across SSA versions
    if qubit_0.value.logical_id == qubit_1.value.logical_id:
        q0_name = qubit_0.name or "unnamed"
        raise QubitAliasError(
            f"Cannot use the same qubit twice in RZZ.\n"
            f"Qubit '{q0_name}' appears in both positions.\n\n"
            f"Fix: Use distinct qubits for the two inputs.",
            handle_name=q0_name,
            operation_name="RZZ",
        )

    # Consume both input handles (enforces linear type)
    qubit_0 = qubit_0.consume(operation_name="RZZ[0]")
    qubit_1 = qubit_1.consume(operation_name="RZZ[1]")

    q0_out_value = qubit_0.value.next_version()
    q1_out_value = qubit_1.value.next_version()

    q0_out = Qubit(value=q0_out_value, parent=qubit_0.parent, indices=qubit_0.indices)
    q1_out = Qubit(value=q1_out_value, parent=qubit_1.parent, indices=qubit_1.indices)

    angle_value = angle.value if isinstance(angle, Float) else angle

    rzz_op = IRGateOperation(
        gate_type=GateOperationType.RZZ,
        operands=[qubit_0.value, qubit_1.value],
        results=[q0_out_value, q1_out_value],
    )
    rzz_op.theta = angle_value

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


def ccx(
    control1: Qubit, control2: Qubit, target: Qubit
) -> tuple[Qubit, Qubit, Qubit]:
    """Toffoli (CCX) gate: flips target when both controls are |1>.

    Args:
        control1: First control qubit.
        control2: Second control qubit.
        target: Target qubit.

    Returns:
        Tuple of (control1_out, control2_out, target_out) after CCX.
    """
    return _apply_three_qubit_gate(control1, control2, target, GateOperationType.TOFFOLI)
