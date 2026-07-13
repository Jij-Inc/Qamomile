"""Test flat global-phase operations across generic IR pipeline stages."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from qamomile.circuit.ir import content_hash, pretty_print_block
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import (
    DecodeQFixedOperation,
    GlobalPhaseOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.serialize import (
    dump_json,
    dump_msgpack,
    load_json,
    load_msgpack,
    to_dict,
)
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import DependencyError
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    CompileTimeIfLoweringPass,
)
from qamomile.circuit.transpiler.passes.constant_fold import ConstantFoldingPass
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.passes.value_mapping import ValueSubstitutor
from qamomile.circuit.transpiler.segments import QuantumStep


def _float_value(name: str, *, const: float | None = None) -> Value:
    """Create a scalar float value, optionally with constant metadata.

    Args:
        name (str): Display name for the value.
        const (float | None): Optional compile-time constant. Defaults to None.

    Returns:
        Value: Scalar ``FloatType`` value with the requested metadata.
    """
    value = Value(type=FloatType(), name=name)
    return value.with_const(const) if const is not None else value


def _phase_only_block(angle: float) -> Block:
    """Create a minimal affine block containing one zero-result phase op.

    Args:
        angle (float): Constant phase angle in radians.

    Returns:
        Block: Affine block containing one ``GlobalPhaseOperation``.
    """
    phase = _float_value("phase", const=angle)
    return Block(
        name="phase_only",
        kind=BlockKind.AFFINE,
        operations=[GlobalPhaseOperation(operands=[phase], results=[])],
    )


def test_content_hash_includes_global_phase_operand() -> None:
    """Changing only a zero-result phase operand changes canonical content."""
    first = _phase_only_block(0.25)
    equivalent = _phase_only_block(0.25)
    different = _phase_only_block(0.5)

    assert content_hash(first) == content_hash(equivalent)
    assert content_hash(first) != content_hash(different)


@pytest.mark.parametrize(
    ("dump", "load"),
    [(dump_json, load_json), (dump_msgpack, load_msgpack)],
    ids=["json", "msgpack"],
)
def test_global_phase_round_trip_preserves_zero_result_operand(
    dump: Callable[[Block], bytes],
    load: Callable[[bytes], Block],
) -> None:
    """JSON and msgpack preserve phase data and the zero-result layout."""
    block = _phase_only_block(0.375)
    restored = load(dump(block))

    assert to_dict(restored) == to_dict(block)
    assert content_hash(restored) == content_hash(block)
    assert len(restored.operations) == 1
    phase_op = restored.operations[0]
    assert isinstance(phase_op, GlobalPhaseOperation)
    assert phase_op.results == []
    assert phase_op.phase.get_const() == pytest.approx(0.375)


def test_constant_folding_substitutes_binop_result_into_phase() -> None:
    """A folded phase expression replaces the op operand before emission."""
    angle = _float_value("angle").with_parameter("angle")
    two = _float_value("two", const=2.0)
    product = _float_value("product")
    multiply = BinOp(
        operands=[angle, two],
        results=[product],
        kind=BinOpKind.MUL,
    )
    block = Block(
        name="computed_phase",
        kind=BlockKind.AFFINE,
        operations=[
            multiply,
            GlobalPhaseOperation(operands=[product], results=[]),
        ],
    )

    folded = ConstantFoldingPass(bindings={"angle": 0.5}).run(block)

    assert len(folded.operations) == 1
    phase_op = folded.operations[0]
    assert isinstance(phase_op, GlobalPhaseOperation)
    assert phase_op.phase.is_constant()
    assert phase_op.phase.get_const() == pytest.approx(1.0)


def test_value_substitution_replaces_global_phase_operand() -> None:
    """Generic value substitution reaches a flat global-phase operand."""
    old_phase = _float_value("old_phase")
    new_phase = _float_value("new_phase", const=0.625)
    phase_op = GlobalPhaseOperation(operands=[old_phase], results=[])

    substituted = ValueSubstitutor({old_phase.uuid: new_phase}).substitute_operation(
        phase_op
    )

    assert isinstance(substituted, GlobalPhaseOperation)
    assert substituted.phase.uuid == new_phase.uuid
    assert substituted.phase.get_const() == pytest.approx(0.625)


def test_segmentation_exposes_runtime_phase_in_segment_and_abi() -> None:
    """A runtime phase is both a quantum-segment input and a public ABI input."""
    angle = _float_value("angle").with_parameter("angle")
    phase_op = GlobalPhaseOperation(operands=[angle], results=[])
    block = Block(
        name="runtime_phase",
        label_args=["angle"],
        input_values=[angle],
        kind=BlockKind.ANALYZED,
        parameters={"angle": angle},
        operations=[phase_op],
    )

    plan = SegmentationPass().run(block)
    quantum_step = next(step for step in plan.steps if isinstance(step, QuantumStep))

    assert quantum_step.segment.operations == [phase_op]
    assert angle.uuid in quantum_step.segment.input_refs
    assert quantum_step.segment.output_refs == []
    assert plan.abi.public_inputs["angle"].uuid == angle.uuid
    assert plan.abi.output_values == []
    assert quantum_step.segment.operations[0].phase.uuid == angle.uuid
    assert quantum_step.segment.operations[0].results == []


def test_printer_formats_global_phase_without_assignment() -> None:
    """The zero-result operation prints as a statement, not an assignment."""
    rendered = pretty_print_block(_phase_only_block(0.125))
    phase_lines = [
        line.strip() for line in rendered.splitlines() if "global_phase" in line
    ]

    assert len(phase_lines) == 1
    assert phase_lines[0].startswith("global_phase(")
    assert " = " not in phase_lines[0]


@pytest.mark.parametrize(("condition", "expected_count"), [(True, 1), (False, 0)])
def test_compile_time_if_keeps_only_live_global_phase(
    condition: bool,
    expected_count: int,
) -> None:
    """Compile-time lowering retains a taken phase and removes a dead one."""
    condition_value = Value(type=BitType(), name="condition").with_const(condition)
    phase_op = GlobalPhaseOperation(
        operands=[_float_value("phase", const=0.75)],
        results=[],
    )
    if_op = IfOperation(
        operands=[condition_value],
        results=[],
        true_operations=[phase_op],
        false_operations=[],
    )
    block = Block(
        name="conditional_phase",
        kind=BlockKind.AFFINE,
        operations=[if_op],
    )

    lowered = CompileTimeIfLoweringPass().run(block)

    assert (
        sum(isinstance(op, GlobalPhaseOperation) for op in lowered.operations)
        == expected_count
    )
    assert not any(isinstance(op, IfOperation) for op in lowered.operations)


def test_measurement_derived_phase_raises_dependency_error() -> None:
    """Analyze rejects a phase decoded from measurement results."""
    size = Value(type=UIntType(), name="size").with_const(1)
    qubits = ArrayValue(type=QubitType(), name="qubits", shape=(size,))
    bits = ArrayValue(type=BitType(), name="bits", shape=(size,))
    decoded_phase = _float_value("decoded_phase")
    block = Block(
        name="measurement_phase",
        kind=BlockKind.AFFINE,
        output_values=[bits],
        operations=[
            QInitOperation(operands=[], results=[qubits]),
            MeasureVectorOperation(operands=[qubits], results=[bits]),
            DecodeQFixedOperation(
                operands=[bits],
                results=[decoded_phase],
                num_bits=1,
                int_bits=0,
            ),
            GlobalPhaseOperation(operands=[decoded_phase], results=[]),
        ],
    )

    with pytest.raises(DependencyError) as exc_info:
        AnalyzePass().run(block)

    assert exc_info.value.quantum_op == "GlobalPhaseOperation"
    assert exc_info.value.classical_value == "decoded_phase"
