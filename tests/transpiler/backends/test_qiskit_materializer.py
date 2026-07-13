"""Tests for CircuitProgram to Qiskit materialization."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import ForLoopOp, IfElseOp
from qiskit.quantum_info import Operator

from qamomile.circuit.transpiler.circuit_ir import (
    CircuitBuilder,
    ClassicalBitExpr,
    ParameterExpr,
    ReusableCircuit,
)
from qamomile.circuit.transpiler.gate_emitter import GateKind
from qamomile.qiskit.materializer import QiskitMaterializer


def _phase_only_call(
    *,
    controls: int,
    inverse: bool = False,
    power: int = 1,
) -> CircuitBuilder:
    """Build a reusable identity whose only effect is a symbolic phase."""
    body = CircuitBuilder(1, 0, name="phase-only")
    body.add_global_phase(ParameterExpr("theta"))
    caller = CircuitBuilder(controls + 1, 0, name="phase-caller")
    caller.append_call(
        ReusableCircuit(
            body.freeze(),
            "phase-only",
            controls=controls,
            inverse=inverse,
            power=power,
        ),
        tuple(range(controls + 1)),
    )
    return caller


def _projector_phase_matrix(
    controls: int,
    phase: float,
) -> np.ndarray:
    """Return a phase conditioned on all low-order control qubits."""
    dimension = 2 ** (controls + 1)
    mask = (1 << controls) - 1
    diagonal = np.ones(dimension, dtype=np.complex128)
    for basis in range(dimension):
        if basis & mask == mask:
            diagonal[basis] = np.exp(1j * phase)
    return np.diag(diagonal)


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


def test_qiskit_materializer_lowers_vector_measurement_at_sdk_boundary() -> None:
    """One vector instruction becomes Qiskit's ordered measurement API call."""
    builder = CircuitBuilder(3, 3, name="vector-measure")
    builder.append_measure_vector((2, 0, 1), (0, 1, 2))

    circuit = QiskitMaterializer().materialize(builder.freeze()).artifact

    assert [instruction.operation.name for instruction in circuit.data] == [
        "measure",
        "measure",
        "measure",
    ]
    assert [
        circuit.find_bit(instruction.qubits[0]).index for instruction in circuit.data
    ] == [
        2,
        0,
        1,
    ]


def test_qiskit_materializer_preserves_symbolic_expression() -> None:
    """Runtime expressions become native Qiskit ParameterExpression values."""
    builder = CircuitBuilder(1, 0)
    theta = ParameterExpr("theta")
    builder.append_gate(GateKind.RY, (0,), (theta * 2.0,))

    circuit = QiskitMaterializer().materialize(builder.freeze()).artifact

    assert {parameter.name for parameter in circuit.parameters} == {"theta"}
    assert str(circuit.data[0].operation.params[0]) == "2*theta"


def test_qiskit_materializer_anchors_unused_abi_parameter() -> None:
    """An unused ABI parameter remains bindable through a zero phase term."""
    builder = CircuitBuilder(1, 0)

    materialized = QiskitMaterializer().materialize(
        builder.freeze(),
        parameter_names=("theta",),
    )

    assert tuple(materialized.parameters) == ("theta",)
    parameter = materialized.parameters["theta"]
    assert parameter in materialized.artifact.parameters

    bound = materialized.artifact.assign_parameters({parameter: 0.37})

    assert not bound.parameters
    assert float(bound.global_phase) == pytest.approx(0.0)
    assert np.allclose(Operator(bound).data, np.eye(2), atol=1e-10)


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


def test_qiskit_materializer_reuses_parameters_across_reusable_calls() -> None:
    """Repeated parameterized calls share one Qiskit Parameter identity."""
    body = CircuitBuilder(1, 0, name="rotation")
    body.append_gate(GateKind.RY, (0,), (ParameterExpr("theta"),))
    reusable = ReusableCircuit(body.freeze(), "rotation")
    caller = CircuitBuilder(1, 0)
    caller.append_call(reusable, (0,))
    caller.append_call(reusable, (0,))

    circuit = QiskitMaterializer().materialize(caller.freeze()).artifact

    assert [instruction.operation.label for instruction in circuit.data] == [
        "rotation",
        "rotation",
    ]
    assert {parameter.name for parameter in circuit.parameters} == {"theta"}


@pytest.mark.parametrize(
    ("controls", "inverse", "power", "phase_factor"),
    [
        (1, False, 1, 1),
        (2, True, 1, -1),
        (2, False, 3, 3),
    ],
)
def test_qiskit_materializer_preserves_transformed_phase_only_calls(
    controls: int,
    inverse: bool,
    power: int,
    phase_factor: int,
) -> None:
    """Control, inverse, and power retain a reusable body's global phase."""
    theta = 0.37
    materialized = QiskitMaterializer().materialize(
        _phase_only_call(
            controls=controls,
            inverse=inverse,
            power=power,
        ).freeze(),
        parameter_names=("theta",),
    )

    assert tuple(materialized.parameters) == ("theta",)
    parameter = materialized.parameters["theta"]
    bound = materialized.artifact.assign_parameters({parameter: theta})

    assert np.allclose(
        Operator(bound).data,
        _projector_phase_matrix(controls, phase_factor * theta),
        atol=1e-10,
    )


def test_qiskit_materializer_shares_parameters_through_nested_inverse_calls() -> None:
    """Nested custom-gate definitions bind one shared parameter identity."""
    theta = 0.37
    controlled_phase = _phase_only_call(controls=1).freeze()
    caller = CircuitBuilder(2, 0, name="nested-inverse")
    caller.append_call(
        ReusableCircuit(
            controlled_phase,
            "controlled-phase-wrapper",
            inverse=True,
        ),
        (0, 1),
    )

    materialized = QiskitMaterializer().materialize(
        caller.freeze(),
        parameter_names=("theta",),
    )
    parameter = materialized.parameters["theta"]
    bound = materialized.artifact.assign_parameters({parameter: theta})

    assert not bound.parameters
    assert np.allclose(
        Operator(bound).data,
        _projector_phase_matrix(1, -theta),
        atol=1e-10,
    )
