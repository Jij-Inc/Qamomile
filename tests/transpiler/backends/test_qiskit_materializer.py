"""Tests for CircuitProgram to Qiskit materialization."""

from __future__ import annotations

from qiskit.circuit import ForLoopOp, IfElseOp

from qamomile.circuit.transpiler.circuit_ir import (
    CircuitBuilder,
    ClassicalBitExpr,
    ParameterExpr,
    ReusableCircuit,
)
from qamomile.circuit.transpiler.gate_emitter import GateKind
from qamomile.qiskit.materializer import QiskitMaterializer


def test_qiskit_materializer_emits_primitive_circuit() -> None:
    """Primitive gates and measurements preserve circuit slot ordering."""
    builder = CircuitBuilder(2, 2, name="bell")
    builder.append_gate(GateKind.H, (0,))
    builder.append_gate(GateKind.CX, (0, 1))
    builder.append_measure(0, 0)
    builder.append_measure(1, 1)

    circuit = QiskitMaterializer().materialize(builder.freeze()).artifact

    assert circuit.name == "bell"
    assert [instruction.operation.name for instruction in circuit.data] == [
        "h",
        "cx",
        "measure",
        "measure",
    ]


def test_qiskit_materializer_preserves_symbolic_expression() -> None:
    """Runtime expressions become native Qiskit ParameterExpression values."""
    builder = CircuitBuilder(1, 0)
    theta = ParameterExpr("theta")
    builder.append_gate(GateKind.RY, (0,), (theta * 2.0,))

    circuit = QiskitMaterializer().materialize(builder.freeze()).artifact

    assert {parameter.name for parameter in circuit.parameters} == {"theta"}
    assert str(circuit.data[0].operation.params[0]) == "2*theta"


def test_qiskit_materializer_preserves_structured_for_loop() -> None:
    """Circuit for regions become native Qiskit ForLoopOp instructions."""
    builder = CircuitBuilder(1, 0)
    induction = builder.begin_for(range(1, 4))
    builder.append_gate(GateKind.RZ, (0,), (induction,))
    builder.end_for()

    circuit = QiskitMaterializer().materialize(builder.freeze()).artifact

    [instruction] = circuit.data
    assert isinstance(instruction.operation, ForLoopOp)
    assert instruction.operation.params[0] == range(1, 4)


def test_qiskit_materializer_preserves_structured_if() -> None:
    """Circuit condition regions become native Qiskit IfElseOp instructions."""
    builder = CircuitBuilder(1, 1)
    context = builder.begin_if(ClassicalBitExpr(0))
    builder.append_gate(GateKind.X, (0,))
    builder.begin_else(context)
    builder.append_gate(GateKind.Z, (0,))
    builder.end_if(context)

    circuit = QiskitMaterializer().materialize(builder.freeze()).artifact

    [instruction] = circuit.data
    assert isinstance(instruction.operation, IfElseOp)


def test_qiskit_materializer_materializes_reusable_call() -> None:
    """Reusable circuit calls retain a named gate boundary."""
    body = CircuitBuilder(1, 0, name="helper")
    body.append_gate(GateKind.H, (0,))
    caller = CircuitBuilder(1, 0)
    caller.append_call(ReusableCircuit(body.freeze(), "helper"), (0,))

    circuit = QiskitMaterializer().materialize(caller.freeze()).artifact

    [instruction] = circuit.data
    assert instruction.operation.label == "helper"
