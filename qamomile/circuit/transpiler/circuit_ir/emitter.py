"""Gate-emitter adapter that lowers the existing circuit walk into circuit IR."""

from __future__ import annotations

import dataclasses
from typing import Any

from qamomile.circuit.ir.operation.arithmetic_operations import BinOpKind
from qamomile.circuit.transpiler.circuit_ir.model import (
    BinaryExpr,
    BinaryOperator,
    CircuitBuilder,
    CircuitProgram,
    ClassicalBitExpr,
    ParameterExpr,
    ReusableCircuit,
    ScalarExpr,
    as_scalar_expr,
)
from qamomile.circuit.transpiler.gate_emitter import GateKind, MeasurementMode

_BINARY_OPERATORS = {
    BinOpKind.ADD: BinaryOperator.ADD,
    BinOpKind.SUB: BinaryOperator.SUB,
    BinOpKind.MUL: BinaryOperator.MUL,
    BinOpKind.DIV: BinaryOperator.DIV,
    BinOpKind.FLOORDIV: BinaryOperator.FLOORDIV,
    BinOpKind.MOD: BinaryOperator.MOD,
    BinOpKind.POW: BinaryOperator.POW,
}


class CircuitGateEmitter:
    """Emit primitive operations into backend-neutral circuit IR."""

    @property
    def measurement_mode(self) -> MeasurementMode:
        """Return native measurement mode for explicit circuit IR.

        Returns:
            MeasurementMode: Always :attr:`MeasurementMode.NATIVE`.
        """
        return MeasurementMode.NATIVE

    def create_circuit(self, num_qubits: int, num_clbits: int) -> CircuitBuilder:
        """Create an empty circuit IR builder.

        Args:
            num_qubits (int): Number of virtual qubit slots.
            num_clbits (int): Number of classical bit slots.

        Returns:
            CircuitBuilder: Empty backend-neutral builder.
        """
        return CircuitBuilder(num_qubits, num_clbits)

    def create_parameter(self, name: str) -> ParameterExpr:
        """Create a target-neutral runtime parameter expression.

        Args:
            name (str): External parameter name.

        Returns:
            ParameterExpr: Parameter reference preserved until materialization.
        """
        return ParameterExpr(name)

    def combine_symbolic(
        self,
        kind: BinOpKind,
        lhs: ScalarExpr | bool | int | float,
        rhs: ScalarExpr | bool | int | float,
    ) -> BinaryExpr | None:
        """Combine symbolic operands without creating backend expressions.

        Args:
            kind (BinOpKind): Qamomile arithmetic operation.
            lhs (ScalarExpr | bool | int | float): Left operand.
            rhs (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr | None: Target-neutral expression, or ``None`` for an
                unsupported operation kind.
        """
        operator = _BINARY_OPERATORS.get(kind)
        if operator is None:
            return None
        return BinaryExpr(operator, as_scalar_expr(lhs), as_scalar_expr(rhs))

    def _emit_gate(
        self,
        circuit: CircuitBuilder,
        kind: GateKind,
        qubits: tuple[int, ...],
        parameters: tuple[ScalarExpr | bool | int | float, ...] = (),
    ) -> None:
        """Append one primitive gate after normalizing scalar parameters.

        Args:
            circuit (CircuitBuilder): Builder receiving the gate.
            kind (GateKind): Primitive gate kind.
            qubits (tuple[int, ...]): Participating qubit slots.
            parameters (tuple[ScalarExpr | bool | int | float, ...]): Gate
                parameters. Defaults to an empty tuple.
        """
        circuit.append_gate(
            kind,
            qubits,
            tuple(as_scalar_expr(parameter) for parameter in parameters),
        )

    def emit_h(self, circuit: CircuitBuilder, qubit: int) -> None:
        """Emit a Hadamard gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.H, (qubit,))

    def emit_x(self, circuit: CircuitBuilder, qubit: int) -> None:
        """Emit a Pauli-X gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.X, (qubit,))

    def emit_y(self, circuit: CircuitBuilder, qubit: int) -> None:
        """Emit a Pauli-Y gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.Y, (qubit,))

    def emit_z(self, circuit: CircuitBuilder, qubit: int) -> None:
        """Emit a Pauli-Z gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.Z, (qubit,))

    def emit_s(self, circuit: CircuitBuilder, qubit: int) -> None:
        """Emit an S gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.S, (qubit,))

    def emit_sdg(self, circuit: CircuitBuilder, qubit: int) -> None:
        """Emit an inverse-S gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.SDG, (qubit,))

    def emit_t(self, circuit: CircuitBuilder, qubit: int) -> None:
        """Emit a T gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.T, (qubit,))

    def emit_tdg(self, circuit: CircuitBuilder, qubit: int) -> None:
        """Emit an inverse-T gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.TDG, (qubit,))

    def emit_rx(
        self,
        circuit: CircuitBuilder,
        qubit: int,
        angle: ScalarExpr | float,
    ) -> None:
        """Emit an RX rotation.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Target slot.
            angle (ScalarExpr | float): Rotation angle in radians.
        """
        self._emit_gate(circuit, GateKind.RX, (qubit,), (angle,))

    def emit_ry(
        self,
        circuit: CircuitBuilder,
        qubit: int,
        angle: ScalarExpr | float,
    ) -> None:
        """Emit an RY rotation.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Target slot.
            angle (ScalarExpr | float): Rotation angle in radians.
        """
        self._emit_gate(circuit, GateKind.RY, (qubit,), (angle,))

    def emit_rz(
        self,
        circuit: CircuitBuilder,
        qubit: int,
        angle: ScalarExpr | float,
    ) -> None:
        """Emit an RZ rotation.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Target slot.
            angle (ScalarExpr | float): Rotation angle in radians.
        """
        self._emit_gate(circuit, GateKind.RZ, (qubit,), (angle,))

    def emit_p(
        self,
        circuit: CircuitBuilder,
        qubit: int,
        angle: ScalarExpr | float,
    ) -> None:
        """Emit a phase rotation.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Target slot.
            angle (ScalarExpr | float): Phase angle in radians.
        """
        self._emit_gate(circuit, GateKind.P, (qubit,), (angle,))

    def emit_cx(self, circuit: CircuitBuilder, control: int, target: int) -> None:
        """Emit a controlled-X gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            control (int): Control slot.
            target (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.CX, (control, target))

    def emit_cz(self, circuit: CircuitBuilder, control: int, target: int) -> None:
        """Emit a controlled-Z gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            control (int): Control slot.
            target (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.CZ, (control, target))

    def emit_swap(self, circuit: CircuitBuilder, qubit1: int, qubit2: int) -> None:
        """Emit a SWAP gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit1 (int): First slot.
            qubit2 (int): Second slot.
        """
        self._emit_gate(circuit, GateKind.SWAP, (qubit1, qubit2))

    def emit_cp(
        self,
        circuit: CircuitBuilder,
        control: int,
        target: int,
        angle: ScalarExpr | float,
    ) -> None:
        """Emit a controlled-phase rotation.

        Args:
            circuit (CircuitBuilder): Destination builder.
            control (int): Control slot.
            target (int): Target slot.
            angle (ScalarExpr | float): Phase angle in radians.
        """
        self._emit_gate(circuit, GateKind.CP, (control, target), (angle,))

    def emit_rzz(
        self,
        circuit: CircuitBuilder,
        qubit1: int,
        qubit2: int,
        angle: ScalarExpr | float,
    ) -> None:
        """Emit an RZZ rotation.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit1 (int): First slot.
            qubit2 (int): Second slot.
            angle (ScalarExpr | float): Rotation angle in radians.
        """
        self._emit_gate(circuit, GateKind.RZZ, (qubit1, qubit2), (angle,))

    def emit_toffoli(
        self,
        circuit: CircuitBuilder,
        control1: int,
        control2: int,
        target: int,
    ) -> None:
        """Emit a Toffoli gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            control1 (int): First control slot.
            control2 (int): Second control slot.
            target (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.TOFFOLI, (control1, control2, target))

    def emit_ch(self, circuit: CircuitBuilder, control: int, target: int) -> None:
        """Emit a controlled-H gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            control (int): Control slot.
            target (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.CH, (control, target))

    def emit_cy(self, circuit: CircuitBuilder, control: int, target: int) -> None:
        """Emit a controlled-Y gate.

        Args:
            circuit (CircuitBuilder): Destination builder.
            control (int): Control slot.
            target (int): Target slot.
        """
        self._emit_gate(circuit, GateKind.CY, (control, target))

    def emit_crx(
        self,
        circuit: CircuitBuilder,
        control: int,
        target: int,
        angle: ScalarExpr | float,
    ) -> None:
        """Emit a controlled-RX rotation.

        Args:
            circuit (CircuitBuilder): Destination builder.
            control (int): Control slot.
            target (int): Target slot.
            angle (ScalarExpr | float): Rotation angle in radians.
        """
        self._emit_gate(circuit, GateKind.CRX, (control, target), (angle,))

    def emit_cry(
        self,
        circuit: CircuitBuilder,
        control: int,
        target: int,
        angle: ScalarExpr | float,
    ) -> None:
        """Emit a controlled-RY rotation.

        Args:
            circuit (CircuitBuilder): Destination builder.
            control (int): Control slot.
            target (int): Target slot.
            angle (ScalarExpr | float): Rotation angle in radians.
        """
        self._emit_gate(circuit, GateKind.CRY, (control, target), (angle,))

    def emit_crz(
        self,
        circuit: CircuitBuilder,
        control: int,
        target: int,
        angle: ScalarExpr | float,
    ) -> None:
        """Emit a controlled-RZ rotation.

        Args:
            circuit (CircuitBuilder): Destination builder.
            control (int): Control slot.
            target (int): Target slot.
            angle (ScalarExpr | float): Rotation angle in radians.
        """
        self._emit_gate(circuit, GateKind.CRZ, (control, target), (angle,))

    def emit_measure(self, circuit: CircuitBuilder, qubit: int, clbit: int) -> None:
        """Emit a measurement into a classical slot.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Measured qubit slot.
            clbit (int): Destination classical slot.
        """
        circuit.append_measure(qubit, clbit)

    def emit_measure_vector(
        self,
        circuit: CircuitBuilder,
        qubits: tuple[int, ...],
        clbits: tuple[int, ...],
    ) -> None:
        """Preserve an ordered vector measurement as one instruction.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubits (tuple[int, ...]): Measured qubit slots in result order.
            clbits (tuple[int, ...]): Destination classical slots.
        """
        circuit.append_measure_vector(qubits, clbits)

    def emit_reset(self, circuit: CircuitBuilder, qubit: int) -> None:
        """Emit a reset-to-zero operation.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubit (int): Reset qubit slot.
        """
        circuit.append_reset(qubit)

    def emit_barrier(self, circuit: CircuitBuilder, qubits: list[int]) -> None:
        """Emit a scheduling barrier.

        Args:
            circuit (CircuitBuilder): Destination builder.
            qubits (list[int]): Participating slots.
        """
        circuit.append_barrier(tuple(qubits))

    def emit_global_phase(
        self,
        circuit: CircuitBuilder,
        angle: ScalarExpr | float,
    ) -> None:
        """Accumulate a phase in the builder's current lexical region.

        Args:
            circuit (CircuitBuilder): Destination builder.
            angle (ScalarExpr | float): Phase angle in radians.
        """
        circuit.add_global_phase(angle)

    def circuit_to_gate(
        self,
        circuit: CircuitBuilder | CircuitProgram,
        name: str = "U",
    ) -> ReusableCircuit:
        """Freeze a circuit as a reusable circuit value.

        Args:
            circuit (CircuitBuilder | CircuitProgram): Circuit body.
            name (str): Reusable circuit name. Defaults to ``"U"``.

        Returns:
            ReusableCircuit: Reusable body without target-native state.
        """
        body = circuit.freeze() if isinstance(circuit, CircuitBuilder) else circuit
        return ReusableCircuit(
            body=body,
            name=name,
            operand_widths=(body.num_qubits,),
        )

    def supports_reusable_gates(self) -> bool:
        """Report support for deferred reusable circuit calls.

        Returns:
            bool: Always ``True`` because :class:`ReusableCircuit` carries a
                target-neutral body and transforms until legalization or
                materialization.
        """
        return True

    def append_gate(
        self,
        circuit: CircuitBuilder,
        gate: ReusableCircuit,
        qubits: list[int],
    ) -> None:
        """Append a reusable circuit call.

        Args:
            circuit (CircuitBuilder): Destination builder.
            gate (ReusableCircuit): Reusable circuit value.
            qubits (list[int]): Participating slots.
        """
        circuit.append_call(gate, tuple(qubits))

    def gate_power(self, gate: ReusableCircuit, power: int) -> ReusableCircuit:
        """Apply an integral power transform to a reusable circuit.

        Args:
            gate (ReusableCircuit): Reusable circuit value.
            power (int): Integral repetition count.

        Returns:
            ReusableCircuit: Transformed reusable circuit.
        """
        return dataclasses.replace(gate, power=gate.power * power)

    def gate_controlled(
        self,
        gate: ReusableCircuit,
        num_controls: int,
    ) -> ReusableCircuit:
        """Add control wires to a reusable circuit.

        Args:
            gate (ReusableCircuit): Reusable circuit value.
            num_controls (int): Number of controls to add.

        Returns:
            ReusableCircuit: Controlled reusable circuit.
        """
        return dataclasses.replace(gate, controls=gate.controls + num_controls)

    def gate_inverse(self, gate: ReusableCircuit) -> ReusableCircuit:
        """Toggle the inverse transform on a reusable circuit.

        Args:
            gate (ReusableCircuit): Reusable circuit value.

        Returns:
            ReusableCircuit: Inverse reusable circuit.
        """
        return dataclasses.replace(gate, inverse=not gate.inverse)

    def supports_gate_inverse(self) -> bool:
        """Report support for deferred inverse transforms.

        Returns:
            bool: Always ``True`` for circuit IR.
        """
        return True

    def supports_for_loop(self) -> bool:
        """Report support for structured for loops.

        Returns:
            bool: Always ``True`` for circuit IR.
        """
        return True

    def emit_for_loop_start(
        self,
        circuit: CircuitBuilder,
        indexset: range,
    ) -> ScalarExpr:
        """Open a structured for-loop body.

        Args:
            circuit (CircuitBuilder): Destination builder.
            indexset (range): Concrete iteration range.

        Returns:
            ScalarExpr: Target-neutral induction expression.
        """
        return circuit.begin_for(indexset)

    def emit_for_loop_end(self, circuit: CircuitBuilder, context: Any) -> None:
        """Close a structured for-loop body.

        Args:
            circuit (CircuitBuilder): Destination builder.
            context (Any): Induction expression returned at loop start.
        """
        del context
        circuit.end_for()

    def supports_if_else(self) -> bool:
        """Report support for structured conditionals.

        Returns:
            bool: Always ``True`` for circuit IR.
        """
        return True

    def emit_if_start(
        self,
        circuit: CircuitBuilder,
        clbit: int,
        value: int = 1,
    ) -> Any:
        """Open a structured conditional true branch.

        Args:
            circuit (CircuitBuilder): Destination builder.
            clbit (int): Predicate classical bit slot.
            value (int): Required bit value. Defaults to one.

        Returns:
            Any: Opaque conditional builder context.
        """
        condition: ScalarExpr = ClassicalBitExpr(clbit)
        if value != 1:
            condition = BinaryExpr(
                BinaryOperator.EQ,
                condition,
                as_scalar_expr(value),
            )
        return circuit.begin_if(condition)

    def emit_else_start(self, circuit: CircuitBuilder, context: Any) -> None:
        """Switch an open conditional to its false branch.

        Args:
            circuit (CircuitBuilder): Destination builder.
            context (Any): Opaque conditional context.
        """
        circuit.begin_else(context)

    def emit_if_end(self, circuit: CircuitBuilder, context: Any) -> None:
        """Close a structured conditional.

        Args:
            circuit (CircuitBuilder): Destination builder.
            context (Any): Opaque conditional context.
        """
        circuit.end_if(context)

    def supports_while_loop(self) -> bool:
        """Report support for structured while loops.

        Returns:
            bool: Always ``True`` for circuit IR.
        """
        return True

    def emit_while_start(
        self,
        circuit: CircuitBuilder,
        clbit: int,
        value: int = 1,
    ) -> Any:
        """Open a structured while-loop body.

        Args:
            circuit (CircuitBuilder): Destination builder.
            clbit (int): Predicate classical bit slot.
            value (int): Required bit value. Defaults to one.

        Returns:
            Any: Opaque while-loop builder context.
        """
        condition: ScalarExpr = ClassicalBitExpr(clbit)
        if value != 1:
            condition = BinaryExpr(
                BinaryOperator.EQ,
                condition,
                as_scalar_expr(value),
            )
        return circuit.begin_while(condition)

    def emit_while_end(self, circuit: CircuitBuilder, context: Any) -> None:
        """Close a structured while-loop body.

        Args:
            circuit (CircuitBuilder): Destination builder.
            context (Any): Opaque while-loop context.
        """
        circuit.end_while(context)
