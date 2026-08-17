"""Tests for final CircuitIR runtime-parameter ABI reconciliation."""

from __future__ import annotations

from typing import Any

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.circuit_ir.model import (
    BarrierInstruction,
    BinaryExpr,
    BinaryOperator,
    CallInstruction,
    CircuitProgram,
    ForInstruction,
    GateInstruction,
    IfInstruction,
    LiteralExpr,
    LoopVariableExpr,
    MeasureInstruction,
    MeasureVectorInstruction,
    ParameterExpr,
    PauliEvolutionInstruction,
    ResetInstruction,
    ReusableCircuit,
    UnaryExpr,
    UnaryOperator,
    WhileInstruction,
    WireId,
)
from qamomile.circuit.transpiler.circuit_ir.parameter_usage import (
    collect_program_parameter_names,
    collect_scalar_parameter_names,
    reconcile_parameter_metadata,
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.gate_emitter import GateKind
from qamomile.circuit.transpiler.parameter_binding import (
    ParameterArrayInfo,
    ParameterContainerKind,
    ParameterInfo,
    ParameterMetadata,
)
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def _unused_parameter_x(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Apply nonzero quantum work without using the formal scalar parameter."""
    return qmc.x(qubit)


@qmc.qkernel
def _unused_parameter_x_via_invoke(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Forward an unused formal parameter through a nested invocation."""
    return _unused_parameter_x(qubit, theta)


@qmc.qkernel
def _used_parameter_rx(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Use the formal scalar parameter in a rotation gate."""
    return qmc.rx(qubit, theta)


@qmc.qkernel
def _controlled_unused_parameter(theta: qmc.Float) -> qmc.Bit:
    """Control a nonempty body that does not use its scalar argument."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_unused_parameter_x)(control, target, theta)
    return qmc.measure(target)


@qmc.qkernel
def _controlled_nested_unused_parameter(theta: qmc.Float) -> qmc.Bit:
    """Control a nested nonempty body that does not use its scalar argument."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_unused_parameter_x_via_invoke)(
        control,
        target,
        theta,
    )
    return qmc.measure(target)


@qmc.qkernel
def _inverse_controlled_unused_parameter(theta: qmc.Float) -> qmc.Bit:
    """Invert a controlled nonempty body with an unused scalar argument."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.inverse(qmc.control(_unused_parameter_x))(
        control,
        target,
        theta,
    )
    return qmc.measure(target)


@qmc.qkernel
def _controlled_used_parameter(theta: qmc.Float) -> qmc.Bit:
    """Control a body that uses its scalar argument in a rotation."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_used_parameter_rx)(control, target, theta)
    return qmc.measure(target)


def _empty_program(
    *,
    name: str = "program",
    global_phase: Any = LiteralExpr(0.0),
    operations: tuple[Any, ...] = (),
) -> CircuitProgram:
    """Build a minimal synthetic CircuitProgram for parameter-use tests."""
    wire = WireId(0)
    return CircuitProgram(
        name=name,
        num_qubits=1,
        num_clbits=1,
        input_wires=(wire,),
        output_wires=(wire,),
        operations=operations,
        global_phase=global_phase,
    )


def _gate_parameter(name: str) -> GateInstruction:
    """Build one synthetic parameterized gate instruction."""
    wire = WireId(0)
    return GateInstruction(
        GateKind.RX,
        (wire,),
        (wire,),
        (ParameterExpr(name),),
    )


def test_parameter_visitor_covers_every_scalar_bearing_instruction_field() -> None:
    """The explicit visitor reaches all root, nested, and scoped expressions."""
    wire = WireId(0)
    shared_body = _empty_program(
        name="shared",
        global_phase=ParameterExpr("call_phase"),
        operations=(_gate_parameter("call_gate"),),
    )
    shared_call = CallInstruction(
        ReusableCircuit(shared_body, "shared"),
        (wire,),
        (wire,),
    )
    program = _empty_program(
        global_phase=ParameterExpr("root_phase"),
        operations=(
            GateInstruction(
                GateKind.RX,
                (wire,),
                (wire,),
                (
                    BinaryExpr(
                        BinaryOperator.ADD,
                        ParameterExpr("gate_left"),
                        ParameterExpr("gate_right"),
                    ),
                    LoopVariableExpr("i"),
                ),
            ),
            PauliEvolutionInstruction(
                hamiltonian=object(),
                time=ParameterExpr("pauli_time"),
                inputs=(wire,),
                outputs=(wire,),
            ),
            shared_call,
            shared_call,
            ForInstruction(
                range(2),
                LoopVariableExpr("index"),
                (wire,),
                (_gate_parameter("for_gate"),),
                (wire,),
                (wire,),
            ),
            IfInstruction(
                condition=ParameterExpr("if_condition"),
                inputs=(wire,),
                true_body=(_gate_parameter("if_true_gate"),),
                false_body=(_gate_parameter("if_false_gate"),),
                true_outputs=(wire,),
                false_outputs=(wire,),
                outputs=(wire,),
                true_global_phase=UnaryExpr(
                    UnaryOperator.NEG,
                    ParameterExpr("if_true_phase"),
                ),
                false_global_phase=ParameterExpr("if_false_phase"),
            ),
            WhileInstruction(
                condition=ParameterExpr("while_condition"),
                inputs=(wire,),
                body=(_gate_parameter("while_gate"),),
                body_outputs=(wire,),
                outputs=(wire,),
                body_global_phase=ParameterExpr("while_phase"),
            ),
            MeasureInstruction(wire, wire, 0),
            MeasureVectorInstruction((wire,), (wire,), (0,)),
            ResetInstruction(wire, wire),
            BarrierInstruction((wire,)),
        ),
    )

    assert collect_program_parameter_names(program) == {
        "call_gate",
        "call_phase",
        "for_gate",
        "gate_left",
        "gate_right",
        "if_condition",
        "if_false_gate",
        "if_false_phase",
        "if_true_gate",
        "if_true_phase",
        "pauli_time",
        "root_phase",
        "while_condition",
        "while_gate",
        "while_phase",
    }


def test_parameter_visitor_rejects_unknown_nodes() -> None:
    """Unknown scalar and instruction nodes fail closed."""
    with pytest.raises(EmitError, match="Unknown CircuitIR scalar"):
        collect_scalar_parameter_names(object())  # type: ignore[arg-type]

    malformed = _empty_program(operations=(object(),))
    with pytest.raises(EmitError, match="Unknown CircuitIR instruction"):
        collect_program_parameter_names(malformed)


def test_parameter_visitor_rejects_reusable_call_cycles() -> None:
    """Malformed cyclic reusable call graphs fail instead of recursing."""
    wire = WireId(0)
    program = _empty_program(name="cycle")
    call = CallInstruction(
        ReusableCircuit(program, "cycle"),
        (wire,),
        (wire,),
    )
    object.__setattr__(program, "operations", (call,))

    with pytest.raises(EmitError, match="contains a cycle"):
        collect_program_parameter_names(program)


def test_parameter_visitor_skips_zero_trip_loop_body() -> None:
    """A parameter used only in an unreachable loop body is absent."""
    wire = WireId(0)
    program = _empty_program(
        operations=(
            ForInstruction(
                range(0),
                LoopVariableExpr("index"),
                (wire,),
                (_gate_parameter("unreachable"),),
                (wire,),
                (wire,),
            ),
        )
    )

    assert collect_program_parameter_names(program) == set()


def test_parameter_visitor_handles_arbitrary_precision_range_cardinality() -> None:
    """A nonempty range larger than Py_ssize_t keeps body parameter use."""
    wire = WireId(0)
    program = _empty_program(
        operations=(
            ForInstruction(
                range(2**100),
                LoopVariableExpr("index"),
                (wire,),
                (_gate_parameter("large_range_parameter"),),
                (wire,),
                (wire,),
            ),
        )
    )

    assert collect_program_parameter_names(program) == {"large_range_parameter"}


def test_reconciliation_preserves_used_container_metadata_and_order() -> None:
    """Filtering retains used array and dict slot metadata without stale roots."""
    angle_zero = ParameterInfo(
        name="angles[0]",
        array_name="angles",
        index=0,
        indices=(0,),
        backend_param=object(),
        source_ref="angle-zero",
        container_kind=ParameterContainerKind.ARRAY,
    )
    angle_one = ParameterInfo(
        name="angles[1]",
        array_name="angles",
        index=1,
        indices=(1,),
        backend_param=object(),
        source_ref="angle-one",
        container_kind=ParameterContainerKind.ARRAY,
    )
    weight = ParameterInfo(
        name="weights[3]",
        array_name="weights",
        index=None,
        backend_param=object(),
        source_ref="weight-three",
        container_kind=ParameterContainerKind.DICT,
    )
    unused = ParameterInfo(
        name="unused",
        array_name="unused",
        index=None,
        backend_param=object(),
    )
    precise_array = ParameterArrayInfo("angles", 1, (2,))
    metadata = ParameterMetadata(
        parameters=[angle_zero, angle_one, weight, unused],
        arrays={"angles": precise_array},
    )
    wire = WireId(0)
    program = _empty_program(
        operations=(
            GateInstruction(
                GateKind.RX,
                (wire,),
                (wire,),
                (ParameterExpr("angles[0]"), ParameterExpr("weights[3]")),
            ),
        )
    )

    filtered = reconcile_parameter_metadata(program, metadata)

    assert filtered.parameters == [angle_zero, weight]
    assert filtered.parameters[0] is angle_zero
    assert filtered.parameters[1] is weight
    assert filtered.arrays == {"angles": precise_array}


def test_reconciliation_rejects_parameter_missing_from_candidate_metadata() -> None:
    """CircuitIR use without provisional metadata is a compiler invariant error."""
    program = _empty_program(global_phase=ParameterExpr("missing"))

    with pytest.raises(EmitError, match="missing.*compiled ABI metadata"):
        reconcile_parameter_metadata(program, ParameterMetadata())


def test_reconciliation_filters_each_segment_independently() -> None:
    """Separate CircuitPrograms retain only their own runtime parameter subset."""
    first = ParameterInfo("first", "first", None, object())
    second = ParameterInfo("second", "second", None, object())
    metadata = ParameterMetadata(parameters=[first, second])

    first_result = reconcile_parameter_metadata(
        _empty_program(global_phase=ParameterExpr("first")),
        metadata,
    )
    second_result = reconcile_parameter_metadata(
        _empty_program(global_phase=ParameterExpr("second")),
        metadata,
    )

    assert first_result.parameters == [first]
    assert second_result.parameters == [second]


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(_controlled_unused_parameter, id="controlled"),
        pytest.param(_controlled_nested_unused_parameter, id="nested-invoke"),
        pytest.param(
            _inverse_controlled_unused_parameter,
            id="inverse-controlled",
        ),
    ],
)
def test_nonempty_unused_controlled_parameter_is_absent_from_materialized_abi(
    kernel: Any,
) -> None:
    """A formal parameter unused by emitted nonzero work never enters the ABI."""
    executable = QiskitTranspiler().transpile(kernel, parameters=["theta"])
    segment = executable.compiled_quantum[0]

    assert segment.parameter_metadata.parameters == []
    assert not segment.circuit.parameters


def test_used_controlled_parameter_remains_in_materialized_abi() -> None:
    """A parameter referenced by controlled rotation remains bindable."""
    executable = QiskitTranspiler().transpile(
        _controlled_used_parameter,
        parameters=["theta"],
    )
    segment = executable.compiled_quantum[0]

    assert [parameter.name for parameter in segment.parameter_metadata.parameters] == [
        "theta"
    ]
    assert {parameter.name for parameter in segment.circuit.parameters} == {"theta"}
