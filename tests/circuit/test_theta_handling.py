"""Tests for GateOperation.theta handling across transpiler passes.

Verifies that UUIDRemapper, ValueSubstitutor, ConstantFoldingPass,
ValueCollector, DependencyGraphBuilder, QuantumDependencyValidator,
and SegmentIOCollector correctly handle the GateOperation.theta field,
which is stored outside the standard operands list.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
    MeasureOperation,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.types.primitives import BitType, FloatType, QubitType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import DependencyError
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.constant_fold import ConstantFoldingPass
from qamomile.circuit.transpiler.passes.control_flow_visitor import ValueCollector
from qamomile.circuit.transpiler.passes.separate import SeparatePass
from qamomile.circuit.transpiler.passes.value_mapping import (
    UUIDRemapper,
    ValueSubstitutor,
)


def _make_value(name: str, type_cls: type = FloatType, **params: object) -> Value:
    """Create a Value with the given name and optional params."""
    return Value(type=type_cls(), name=name, params=dict(params))


def _make_ry_gate(
    qubit: Value,
    theta: Value | float,
) -> GateOperation:
    """Create an RY gate with the given qubit and theta."""
    q_out = qubit.next_version()
    return GateOperation(
        operands=[qubit],
        results=[q_out],
        gate_type=GateOperationType.RY,
        theta=theta,
    )


# ---------------------------------------------------------------------------
# UUIDRemapper tests
# ---------------------------------------------------------------------------


class TestUUIDRemapperTheta:
    """Tests for UUIDRemapper cloning GateOperation.theta."""

    def test_clone_gate_with_theta_value(self) -> None:
        """Cloned GateOperation has a fresh UUID for theta."""
        q = _make_value("q", QubitType)
        theta = _make_value("angle", FloatType)
        gate = _make_ry_gate(q, theta)

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(gate)

        assert isinstance(cloned, GateOperation)
        assert isinstance(cloned.theta, Value)
        assert cloned.theta.uuid != theta.uuid
        assert cloned.theta.name == theta.name

    def test_clone_gate_with_float_theta(self) -> None:
        """Cloned GateOperation with float theta preserves it unchanged."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        gate = GateOperation(
            operands=[q],
            results=[q_out],
            gate_type=GateOperationType.RY,
            theta=1.5,
        )

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(gate)

        assert isinstance(cloned, GateOperation)
        assert cloned.theta == 1.5

    def test_clone_gate_without_theta(self) -> None:
        """Cloned GateOperation without theta (H gate) keeps theta=None."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        gate = GateOperation(
            operands=[q],
            results=[q_out],
            gate_type=GateOperationType.H,
            theta=None,
        )

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(gate)

        assert isinstance(cloned, GateOperation)
        assert cloned.theta is None

    def test_cloned_theta_registered_in_uuid_remap(self) -> None:
        """Cloned theta UUID is registered in the remapper's UUID map."""
        q = _make_value("q", QubitType)
        theta = _make_value("angle", FloatType)
        gate = _make_ry_gate(q, theta)

        remapper = UUIDRemapper()
        remapper.clone_operation(gate)

        assert theta.uuid in remapper.uuid_remap
        new_uuid = remapper.uuid_remap[theta.uuid]
        assert new_uuid != theta.uuid

    @pytest.mark.parametrize("seed", range(5))
    def test_clone_gate_with_random_float_theta(self, seed: int) -> None:
        """Cloned GateOperation preserves random float theta unchanged."""
        rng = np.random.default_rng(seed)
        theta_val = rng.uniform(-2 * np.pi, 2 * np.pi)

        q = _make_value("q", QubitType)
        q_out = q.next_version()
        gate = GateOperation(
            operands=[q],
            results=[q_out],
            gate_type=GateOperationType.RY,
            theta=float(theta_val),
        )

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(gate)

        assert isinstance(cloned, GateOperation)
        assert cloned.theta == float(theta_val)

    @pytest.mark.parametrize(
        "theta_val",
        [0.0, math.pi, -math.pi, 2 * math.pi, 1e-15, 1e10],
        ids=["zero", "pi", "neg_pi", "two_pi", "tiny", "large"],
    )
    def test_clone_gate_preserves_boundary_theta(self, theta_val: float) -> None:
        """Cloned GateOperation preserves boundary float theta values."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        gate = GateOperation(
            operands=[q],
            results=[q_out],
            gate_type=GateOperationType.RY,
            theta=theta_val,
        )

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(gate)

        assert isinstance(cloned, GateOperation)
        assert cloned.theta == theta_val


# ---------------------------------------------------------------------------
# ValueSubstitutor tests
# ---------------------------------------------------------------------------


class TestValueSubstitutorTheta:
    """Tests for ValueSubstitutor replacing GateOperation.theta."""

    def test_substitute_theta_when_mapped(self) -> None:
        """Theta is substituted when its UUID is in the value map."""
        q = _make_value("q", QubitType)
        callee_theta = _make_value("callee_angle", FloatType)
        caller_theta = _make_value("caller_angle", FloatType, const=0.5)
        gate = _make_ry_gate(q, callee_theta)

        value_map = {callee_theta.uuid: caller_theta}
        substitutor = ValueSubstitutor(value_map)
        substituted = substitutor.substitute_operation(gate)

        assert isinstance(substituted, GateOperation)
        assert isinstance(substituted.theta, Value)
        assert substituted.theta.uuid == caller_theta.uuid
        assert substituted.theta.name == "caller_angle"

    def test_theta_unchanged_when_not_mapped(self) -> None:
        """Theta is unchanged when its UUID is not in the value map."""
        q = _make_value("q", QubitType)
        theta = _make_value("angle", FloatType)
        gate = _make_ry_gate(q, theta)

        value_map: dict[str, Value] = {}
        substitutor = ValueSubstitutor(value_map)
        substituted = substitutor.substitute_operation(gate)

        assert isinstance(substituted, GateOperation)
        assert isinstance(substituted.theta, Value)
        assert substituted.theta.uuid == theta.uuid

    def test_float_theta_not_substituted(self) -> None:
        """Float theta is preserved (not a Value, no substitution)."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        gate = GateOperation(
            operands=[q],
            results=[q_out],
            gate_type=GateOperationType.RY,
            theta=3.14,
        )

        value_map: dict[str, Value] = {}
        substitutor = ValueSubstitutor(value_map)
        substituted = substitutor.substitute_operation(gate)

        assert isinstance(substituted, GateOperation)
        assert substituted.theta == 3.14

    @pytest.mark.parametrize("seed", range(5))
    def test_substitute_theta_with_random_const(self, seed: int) -> None:
        """Theta is substituted with a random constant value."""
        rng = np.random.default_rng(seed)
        const_val = float(rng.uniform(-2 * np.pi, 2 * np.pi))

        q = _make_value("q", QubitType)
        callee_theta = _make_value("callee_angle", FloatType)
        caller_theta = _make_value("caller_angle", FloatType, const=const_val)
        gate = _make_ry_gate(q, callee_theta)

        value_map = {callee_theta.uuid: caller_theta}
        substitutor = ValueSubstitutor(value_map)
        substituted = substitutor.substitute_operation(gate)

        assert isinstance(substituted, GateOperation)
        assert isinstance(substituted.theta, Value)
        assert substituted.theta.params.get("const") == const_val


# ---------------------------------------------------------------------------
# ConstantFoldingPass tests
# ---------------------------------------------------------------------------


class TestConstantFoldTheta:
    """Tests for ConstantFoldingPass propagating folded values into theta."""

    def test_folded_binop_propagated_to_theta(self) -> None:
        """When a BinOp result is used as theta, folding propagates the constant."""
        # Create: offset + i = const 3 (pre-folded), used as theta for RY
        binop_result = _make_value("binop_result", FloatType)
        folded_value = dataclasses.replace(
            binop_result,
            name="folded_binop_result",
            params={"const": 3.0},
        )

        q = _make_value("q", QubitType)
        gate = _make_ry_gate(q, binop_result)

        # Simulate: the BinOp was folded, so its result uuid maps to a constant
        folded_values = {binop_result.uuid: folded_value}

        pass_instance = ConstantFoldingPass()
        result = pass_instance._substitute_folded_operands(gate, folded_values)

        assert isinstance(result, GateOperation)
        assert isinstance(result.theta, Value)
        assert result.theta.params.get("const") == 3.0

    def test_theta_unchanged_when_not_folded(self) -> None:
        """Theta is unchanged when it is not in folded_values."""
        q = _make_value("q", QubitType)
        theta = _make_value("angle", FloatType, const=1.0)
        gate = _make_ry_gate(q, theta)

        folded_values: dict[str, Value] = {}

        pass_instance = ConstantFoldingPass()
        result = pass_instance._substitute_folded_operands(gate, folded_values)

        assert isinstance(result, GateOperation)
        assert isinstance(result.theta, Value)
        assert result.theta.uuid == theta.uuid

    def test_float_theta_not_affected_by_folding(self) -> None:
        """Float theta (not a Value) is not affected by folding."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        gate = GateOperation(
            operands=[q],
            results=[q_out],
            gate_type=GateOperationType.RY,
            theta=2.0,
        )

        folded_values: dict[str, Value] = {}

        pass_instance = ConstantFoldingPass()
        result = pass_instance._substitute_folded_operands(gate, folded_values)

        assert isinstance(result, GateOperation)
        assert result.theta == 2.0

    @pytest.mark.parametrize("seed", range(5))
    def test_folded_random_constant_propagated_to_theta(self, seed: int) -> None:
        """Random folded constant is propagated into theta."""
        rng = np.random.default_rng(seed)
        const_val = float(rng.uniform(-10.0, 10.0))

        binop_result = _make_value("binop_result", FloatType)
        folded_value = dataclasses.replace(
            binop_result,
            name="folded",
            params={"const": const_val},
        )

        q = _make_value("q", QubitType)
        gate = _make_ry_gate(q, binop_result)
        folded_values = {binop_result.uuid: folded_value}

        pass_instance = ConstantFoldingPass()
        result = pass_instance._substitute_folded_operands(gate, folded_values)

        assert isinstance(result, GateOperation)
        assert isinstance(result.theta, Value)
        assert result.theta.params.get("const") == const_val


# ---------------------------------------------------------------------------
# ValueCollector tests
# ---------------------------------------------------------------------------


class TestValueCollectorTheta:
    """Tests for ValueCollector including GateOperation.theta."""

    def test_theta_value_collected_as_operand(self) -> None:
        """ValueCollector includes theta UUID in operand_uuids."""
        q = _make_value("q", QubitType)
        theta = _make_value("angle", FloatType)
        gate = _make_ry_gate(q, theta)

        collector = ValueCollector()
        collector.visit_operations([gate])

        assert theta.uuid in collector.operand_uuids
        assert q.uuid in collector.operand_uuids

    def test_float_theta_not_collected(self) -> None:
        """Float theta does not appear in operand_uuids."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        gate = GateOperation(
            operands=[q],
            results=[q_out],
            gate_type=GateOperationType.RY,
            theta=1.0,
        )

        collector = ValueCollector()
        collector.visit_operations([gate])

        # Only the qubit operand should be collected
        assert q.uuid in collector.operand_uuids
        assert len(collector.operand_uuids) == 1

    def test_none_theta_not_collected(self) -> None:
        """None theta (H gate) does not appear in operand_uuids."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        gate = GateOperation(
            operands=[q],
            results=[q_out],
            gate_type=GateOperationType.H,
            theta=None,
        )

        collector = ValueCollector()
        collector.visit_operations([gate])

        assert q.uuid in collector.operand_uuids
        assert len(collector.operand_uuids) == 1


# ---------------------------------------------------------------------------
# DependencyGraphBuilder tests (analyze.py)
# ---------------------------------------------------------------------------


class TestDependencyGraphBuilderTheta:
    """Tests for DependencyGraphBuilder including theta in dependency graph."""

    def test_theta_value_in_dependency_graph(self) -> None:
        """Gate result depends on theta Value UUID in the dependency graph."""
        q = _make_value("q", QubitType)
        theta = _make_value("angle", FloatType)
        gate = _make_ry_gate(q, theta)

        analyze = AnalyzePass()
        graph = analyze._build_dependency_graph([gate])

        q_out_uuid = gate.results[0].uuid
        assert q_out_uuid in graph
        assert theta.uuid in graph[q_out_uuid]
        assert q.uuid in graph[q_out_uuid]

    def test_float_theta_not_in_dependency_graph(self) -> None:
        """Float theta does not contribute to dependency graph operand set."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        gate = GateOperation(
            operands=[q],
            results=[q_out],
            gate_type=GateOperationType.RY,
            theta=1.5,
        )

        analyze = AnalyzePass()
        graph = analyze._build_dependency_graph([gate])

        assert q_out.uuid in graph
        # Only the qubit operand should appear as a dependency
        assert graph[q_out.uuid] == {q.uuid}

    def test_none_theta_not_in_dependency_graph(self) -> None:
        """None theta (H gate) does not contribute to dependency graph."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        gate = GateOperation(
            operands=[q],
            results=[q_out],
            gate_type=GateOperationType.H,
            theta=None,
        )

        analyze = AnalyzePass()
        graph = analyze._build_dependency_graph([gate])

        assert q_out.uuid in graph
        assert graph[q_out.uuid] == {q.uuid}


# ---------------------------------------------------------------------------
# QuantumDependencyValidator tests (analyze.py)
# ---------------------------------------------------------------------------


class TestQuantumDependencyValidatorTheta:
    """Tests for QuantumDependencyValidator catching theta→measurement deps."""

    def test_theta_depending_on_measurement_raises(self) -> None:
        """Gate with theta derived from measurement raises DependencyError."""
        # Build a minimal block: QInit → H → Measure → BinOp → RY(theta=binop_result)
        q_init = _make_value("q", QubitType)
        q_init_out = q_init.next_version()
        qinit_op = QInitOperation(operands=[], results=[q_init_out])

        q_after_h = q_init_out.next_version()
        h_gate = GateOperation(
            operands=[q_init_out],
            results=[q_after_h],
            gate_type=GateOperationType.H,
        )

        bit_result = _make_value("bit", BitType)
        measure_op = MeasureOperation(operands=[q_after_h], results=[bit_result])

        # BinOp: bit_result + 0.5 -> binop_result (depends on measurement)
        const_val = _make_value("const", FloatType, const=0.5)
        binop_result = _make_value("binop_result", FloatType)
        binop = BinOp(
            operands=[bit_result, const_val],
            results=[binop_result],
            kind=BinOpKind.ADD,
        )

        # Second qubit for the RY gate that uses measurement-derived theta
        q2_init = _make_value("q2", QubitType)
        q2_init_out = q2_init.next_version()
        qinit_op2 = QInitOperation(operands=[], results=[q2_init_out])

        q2_after_ry = q2_init_out.next_version()
        ry_gate = GateOperation(
            operands=[q2_init_out],
            results=[q2_after_ry],
            gate_type=GateOperationType.RY,
            theta=binop_result,  # theta depends on measurement
        )

        operations = [qinit_op, h_gate, measure_op, binop, qinit_op2, ry_gate]

        block = Block(
            name="test",
            operations=operations,
            kind=BlockKind.LINEAR,
            input_values=[],
            output_values=[],
            parameters={},
        )

        analyze = AnalyzePass()

        with pytest.raises(DependencyError, match="measurement result"):
            analyze.run(block)

    def test_theta_with_parameter_value_passes(self) -> None:
        """Gate with theta as a parameter Value passes validation."""
        q_init = _make_value("q", QubitType)
        q_init_out = q_init.next_version()
        qinit_op = QInitOperation(operands=[], results=[q_init_out])

        param_theta = _make_value("theta_param", FloatType, parameter="theta_param")
        q_after_ry = q_init_out.next_version()
        ry_gate = GateOperation(
            operands=[q_init_out],
            results=[q_after_ry],
            gate_type=GateOperationType.RY,
            theta=param_theta,
        )

        operations = [qinit_op, ry_gate]

        block = Block(
            name="test",
            operations=operations,
            kind=BlockKind.LINEAR,
            input_values=[],
            output_values=[],
            parameters={"theta_param": param_theta},
        )

        analyze = AnalyzePass()
        result = analyze.run(block)

        # Should pass without raising
        assert result.kind == BlockKind.ANALYZED


# ---------------------------------------------------------------------------
# SegmentIOCollector tests (separate.py)
# ---------------------------------------------------------------------------


class TestSegmentIOCollectorTheta:
    """Tests for SegmentIOCollector tracking theta as segment input."""

    def test_theta_from_parameter_tracked_as_segment_input(self) -> None:
        """Theta defined as a parameter is collected as segment input."""
        q_init = _make_value("q", QubitType)
        q_init_out = q_init.next_version()
        qinit_op = QInitOperation(operands=[], results=[q_init_out])

        param_theta = _make_value("theta_param", FloatType, parameter="theta_param")
        q_after_ry = q_init_out.next_version()
        ry_gate = GateOperation(
            operands=[q_init_out],
            results=[q_after_ry],
            gate_type=GateOperationType.RY,
            theta=param_theta,
        )

        # Measure to create quantum + classical segments
        bit_result = _make_value("bit", BitType)
        measure_op = MeasureOperation(operands=[q_after_ry], results=[bit_result])

        operations = [qinit_op, ry_gate, measure_op]

        block = Block(
            name="test",
            operations=operations,
            kind=BlockKind.LINEAR,
            input_values=[],
            output_values=[bit_result],
            parameters={"theta_param": param_theta},
        )

        separate = SeparatePass()
        program = separate.run(block)

        # The quantum segment should have theta as an input
        assert param_theta.uuid in program.quantum.input_refs

    def test_float_theta_not_in_segment_inputs(self) -> None:
        """Float theta does not appear in segment input_refs."""
        q_init = _make_value("q", QubitType)
        q_init_out = q_init.next_version()
        qinit_op = QInitOperation(operands=[], results=[q_init_out])

        q_after_ry = q_init_out.next_version()
        ry_gate = GateOperation(
            operands=[q_init_out],
            results=[q_after_ry],
            gate_type=GateOperationType.RY,
            theta=1.57,  # float, not a Value
        )

        bit_result = _make_value("bit", BitType)
        measure_op = MeasureOperation(operands=[q_after_ry], results=[bit_result])

        operations = [qinit_op, ry_gate, measure_op]

        block = Block(
            name="test",
            operations=operations,
            kind=BlockKind.LINEAR,
            input_values=[],
            output_values=[bit_result],
            parameters={},
        )

        separate = SeparatePass()
        program = separate.run(block)

        # Float theta should not appear in any input_refs
        # (input_refs only contains Value UUIDs)
        for ref in program.quantum.input_refs:
            assert isinstance(ref, str)


# ---------------------------------------------------------------------------
# Boundary theta value tests (cross-cutting)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "theta_val",
    [0.0, math.pi, -math.pi, 2 * math.pi, 1e-15, 1e10],
    ids=["zero", "pi", "neg_pi", "two_pi", "tiny", "large"],
)
def test_constant_fold_preserves_boundary_float_theta(theta_val: float) -> None:
    """Constant folding preserves boundary float theta values."""
    q = _make_value("q", QubitType)
    q_out = q.next_version()
    gate = GateOperation(
        operands=[q],
        results=[q_out],
        gate_type=GateOperationType.RY,
        theta=theta_val,
    )

    folded_values: dict[str, Value] = {}
    pass_instance = ConstantFoldingPass()
    result = pass_instance._substitute_folded_operands(gate, folded_values)

    assert isinstance(result, GateOperation)
    assert result.theta == theta_val


@pytest.mark.parametrize(
    "theta_val",
    [0.0, math.pi, -math.pi, 2 * math.pi, 1e-15, 1e10],
    ids=["zero", "pi", "neg_pi", "two_pi", "tiny", "large"],
)
def test_substitutor_preserves_boundary_float_theta(theta_val: float) -> None:
    """ValueSubstitutor preserves boundary float theta values."""
    q = _make_value("q", QubitType)
    q_out = q.next_version()
    gate = GateOperation(
        operands=[q],
        results=[q_out],
        gate_type=GateOperationType.RY,
        theta=theta_val,
    )

    value_map: dict[str, Value] = {}
    substitutor = ValueSubstitutor(value_map)
    substituted = substitutor.substitute_operation(gate)

    assert isinstance(substituted, GateOperation)
    assert substituted.theta == theta_val
