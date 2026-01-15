from qamomile.circuit.frontend.handle import Float, Qubit
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.gate import GateOperation as IRGateOperation
from qamomile.circuit.ir.operation.gate import GateOperationType


def _apply_single_qubit_gate(qubit: Qubit, gate_type: GateOperationType) -> Qubit:
    """Apply a single-qubit gate and return the output qubit."""
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


def h(qubit: Qubit) -> Qubit:
    """Hadamard gate."""
    return _apply_single_qubit_gate(qubit, GateOperationType.H)


def x(qubit: Qubit) -> Qubit:
    """Pauli-X gate (NOT gate)."""
    return _apply_single_qubit_gate(qubit, GateOperationType.X)


def cx(control: Qubit, target: Qubit) -> tuple[Qubit, Qubit]:
    """CNOT (Controlled-X) gate."""
    return _apply_two_qubit_gate(control, target, GateOperationType.CX)


def p(qubit: Qubit, theta: float | Float) -> Qubit:
    """Phase gate: P(theta)|1> = e^{i*theta}|1>."""
    output_value = qubit.value.next_version()
    output_qubit = Qubit(value=output_value, parent=qubit.parent, indices=qubit.indices)

    theta_value = theta.value if isinstance(theta, Float) else theta

    p_op = IRGateOperation(
        gate_type=GateOperationType.P, operands=[qubit.value], results=[output_value]
    )
    p_op.theta = theta_value  # type: ignore

    tracer = get_current_tracer()
    tracer.add_operation(p_op)
    return output_qubit


def cp(control: Qubit, target: Qubit, theta: float | Float) -> tuple[Qubit, Qubit]:
    """Controlled-Phase gate."""
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
    cp_op.theta = theta_value  # type: ignore

    tracer = get_current_tracer()
    tracer.add_operation(cp_op)
    return ctrl_out, tgt_out


def _apply_rotation_gate(
    qubit: Qubit, theta: float | Float, gate_type: GateOperationType
) -> Qubit:
    """Apply a rotation gate with angle theta."""
    output_value = qubit.value.next_version()
    output_qubit = Qubit(value=output_value, parent=qubit.parent, indices=qubit.indices)

    theta_value = theta.value if isinstance(theta, Float) else theta

    gate_op = IRGateOperation(
        gate_type=gate_type, operands=[qubit.value], results=[output_value]
    )
    gate_op.theta = theta_value  # type: ignore

    tracer = get_current_tracer()
    tracer.add_operation(gate_op)
    return output_qubit


def rx(qubit: Qubit, theta: float | Float) -> Qubit:
    """Rotation around X-axis: RX(theta) = exp(-i * theta/2 * X)."""
    return _apply_rotation_gate(qubit, theta, GateOperationType.RX)


def ry(qubit: Qubit, theta: float | Float) -> Qubit:
    """Rotation around Y-axis: RY(theta) = exp(-i * theta/2 * Y)."""
    return _apply_rotation_gate(qubit, theta, GateOperationType.RY)


def rz(qubit: Qubit, theta: float | Float) -> Qubit:
    """Rotation around Z-axis: RZ(theta) = exp(-i * theta/2 * Z)."""
    return _apply_rotation_gate(qubit, theta, GateOperationType.RZ)


def rzz(qubit_0: Qubit, qubit_1: Qubit, theta: float | Float) -> tuple[Qubit, Qubit]:
    """RZZ gate: exp(-i * theta/2 * Z ⊗ Z).

    The RZZ gate applies a rotation around the ZZ axis on two qubits.

    Args:
        qubit_0: First qubit.
        qubit_1: Second qubit.
        theta: Rotation angle in radians.

    Returns:
        Tuple of (qubit_0_out, qubit_1_out) after RZZ.
    """
    q0_out_value = qubit_0.value.next_version()
    q1_out_value = qubit_1.value.next_version()

    q0_out = Qubit(value=q0_out_value, parent=qubit_0.parent, indices=qubit_0.indices)
    q1_out = Qubit(value=q1_out_value, parent=qubit_1.parent, indices=qubit_1.indices)

    theta_value = theta.value if isinstance(theta, Float) else theta

    rzz_op = IRGateOperation(
        gate_type=GateOperationType.RZZ,
        operands=[qubit_0.value, qubit_1.value],
        results=[q0_out_value, q1_out_value],
    )
    rzz_op.theta = theta_value  # type: ignore

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
