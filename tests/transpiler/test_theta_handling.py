"""Tests for GateOperation.theta handling across transpiler passes.

GateOperation stores its angle parameter in a `theta` field separate from
`operands`.  Several transpiler passes must explicitly handle this field:
constant folding, UUID remapping, value substitution, and loop analysis.
These tests verify that theta is correctly propagated through each pass.
"""

from __future__ import annotations

import pytest

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
)
from qamomile.circuit.ir.types.primitives import FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.passes.constant_fold import ConstantFoldingPass
from qamomile.circuit.transpiler.passes.emit_base import LoopAnalyzer
from qamomile.circuit.transpiler.passes.value_mapping import (
    UUIDRemapper,
    ValueSubstitutor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qubit(name: str = "q") -> Value:
    """Create a qubit Value."""
    return Value(type=QubitType(), name=name)


def _float_val(name: str = "theta", *, const: float | None = None) -> Value:
    """Create a float Value, optionally constant."""
    params: dict[str, object] = {}
    if const is not None:
        params["const"] = const
    return Value(type=FloatType(), name=name, params=params)


def _uint_val(name: str, *, const: int | None = None) -> Value:
    """Create a UInt Value, optionally constant."""
    params: dict[str, object] = {}
    if const is not None:
        params["const"] = const
    return Value(type=UIntType(), name=name, params=params)


def _make_binop(
    lhs: Value,
    rhs: Value,
    kind: BinOpKind,
    result_name: str = "binop_result",
) -> tuple[BinOp, Value]:
    """Create a BinOp and its result Value. Returns (op, result)."""
    result = _float_val(result_name)
    op = BinOp(
        operands=[lhs, rhs],
        results=[result],
        kind=kind,
    )
    return op, result


def _make_gate(
    gate_type: GateOperationType,
    qubits: list[Value],
    theta: float | Value | None = None,
) -> GateOperation:
    """Create a GateOperation with given qubits and theta."""
    results = [q.next_version() for q in qubits]
    return GateOperation(
        operands=qubits,
        results=results,
        gate_type=gate_type,
        theta=theta,
    )


def _make_block(operations: list) -> Block:
    """Create a minimal LINEAR block wrapping given operations."""
    return Block(
        name="test_block",
        operations=operations,
        kind=BlockKind.LINEAR,
    )


# ===========================================================================
# ConstantFoldingPass — theta handling
# ===========================================================================


class TestConstantFoldingTheta:
    """Tests for ConstantFoldingPass handling of GateOperation.theta."""

    def test_theta_value_is_folded_when_referencing_binop_result(self) -> None:
        """Theta referencing a folded BinOp result gets the folded constant."""
        # BinOp: 0.5 * 2.0 = 1.0
        lhs = _float_val("a", const=0.5)
        rhs = _float_val("b", const=2.0)
        binop, binop_result = _make_binop(lhs, rhs, BinOpKind.MUL)

        # Gate whose theta references the BinOp result
        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=binop_result)

        block = _make_block([binop, gate])
        folded_block = ConstantFoldingPass().run(block)

        # BinOp should be removed (folded)
        assert len(folded_block.operations) == 1
        folded_gate = folded_block.operations[0]
        assert isinstance(folded_gate, GateOperation)

        # theta should now be the folded constant Value
        assert isinstance(folded_gate.theta, Value)
        assert folded_gate.theta.is_constant()
        assert folded_gate.theta.get_const() == pytest.approx(1.0)

    def test_theta_float_is_unchanged(self) -> None:
        """Theta stored as a plain float should not be affected by folding."""
        q = _qubit()
        gate = _make_gate(GateOperationType.RX, [q], theta=0.42)
        block = _make_block([gate])

        folded_block = ConstantFoldingPass().run(block)
        assert len(folded_block.operations) == 1
        folded_gate = folded_block.operations[0]
        assert isinstance(folded_gate, GateOperation)
        assert folded_gate.theta == 0.42

    def test_theta_none_is_unchanged(self) -> None:
        """Gate with no theta (e.g. H gate) should pass through unchanged."""
        q = _qubit()
        gate = _make_gate(GateOperationType.H, [q], theta=None)
        block = _make_block([gate])

        folded_block = ConstantFoldingPass().run(block)
        assert len(folded_block.operations) == 1
        assert folded_block.operations[0].theta is None

    def test_theta_value_not_foldable_is_unchanged(self) -> None:
        """Theta referencing a non-constant Value should remain as-is."""
        theta_val = _float_val("dynamic_param")  # no const, not foldable
        q = _qubit()
        gate = _make_gate(GateOperationType.RY, [q], theta=theta_val)
        block = _make_block([gate])

        folded_block = ConstantFoldingPass().run(block)
        assert len(folded_block.operations) == 1
        folded_gate = folded_block.operations[0]
        assert isinstance(folded_gate, GateOperation)
        assert isinstance(folded_gate.theta, Value)
        assert folded_gate.theta.uuid == theta_val.uuid

    def test_theta_with_bound_parameter_is_folded(self) -> None:
        """BinOp using a bound parameter folds, and theta references the result."""
        param = Value(
            type=FloatType(), name="phase", params={"parameter": "phase"}
        )
        two = _float_val("two", const=2.0)
        binop, binop_result = _make_binop(param, two, BinOpKind.MUL)

        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=binop_result)
        block = _make_block([binop, gate])

        folded_block = ConstantFoldingPass(bindings={"phase": 0.3}).run(block)

        assert len(folded_block.operations) == 1
        folded_gate = folded_block.operations[0]
        assert isinstance(folded_gate, GateOperation)
        assert isinstance(folded_gate.theta, Value)
        assert folded_gate.theta.is_constant()
        assert folded_gate.theta.get_const() == pytest.approx(0.6)

    def test_operand_folding_still_works_alongside_theta(self) -> None:
        """Both operand substitution and theta folding work in the same gate."""
        const_a = _float_val("a", const=3.0)
        const_b = _float_val("b", const=4.0)
        binop, binop_result = _make_binop(const_a, const_b, BinOpKind.ADD)

        q = _qubit()
        gate = _make_gate(GateOperationType.RX, [q], theta=binop_result)
        block = _make_block([binop, gate])

        folded_block = ConstantFoldingPass().run(block)

        assert len(folded_block.operations) == 1
        folded_gate = folded_block.operations[0]
        assert isinstance(folded_gate, GateOperation)
        assert isinstance(folded_gate.theta, Value)
        assert folded_gate.theta.get_const() == pytest.approx(7.0)

    def test_theta_element_indices_are_folded(self) -> None:
        """Theta as array element params[offset+i] has its index BinOp folded."""
        # BinOp: offset(2) + i(3) = 5
        offset = _uint_val("offset", const=2)
        i_val = _uint_val("i", const=3)
        binop, binop_result = _make_binop(offset, i_val, BinOpKind.ADD)

        # theta = params[binop_result] — an array element whose index is a BinOp
        params_array = ArrayValue(type=FloatType(), name="params")
        theta_elem = Value(
            type=FloatType(),
            name="params_idx",
            parent_array=params_array,
            element_indices=(binop_result,),
        )

        q = _qubit()
        gate = _make_gate(GateOperationType.RY, [q], theta=theta_elem)
        block = _make_block([binop, gate])

        folded_block = ConstantFoldingPass().run(block)

        # BinOp should be removed (folded)
        assert len(folded_block.operations) == 1
        folded_gate = folded_block.operations[0]
        assert isinstance(folded_gate, GateOperation)
        assert isinstance(folded_gate.theta, Value)

        # element_indices should now contain the folded constant (5)
        assert len(folded_gate.theta.element_indices) == 1
        folded_idx = folded_gate.theta.element_indices[0]
        assert folded_idx.is_constant()
        assert folded_idx.get_const() == 5

    def test_theta_element_indices_partial_fold(self) -> None:
        """Only foldable indices in theta.element_indices are replaced."""
        # BinOp: 1 + 2 = 3 (foldable)
        a = _uint_val("a", const=1)
        b = _uint_val("b", const=2)
        binop, binop_result = _make_binop(a, b, BinOpKind.ADD)

        # Non-foldable index
        dynamic_idx = _uint_val("dynamic")

        params_array = ArrayValue(type=FloatType(), name="params")
        theta_elem = Value(
            type=FloatType(),
            name="params_ij",
            parent_array=params_array,
            element_indices=(binop_result, dynamic_idx),
        )

        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=theta_elem)
        block = _make_block([binop, gate])

        folded_block = ConstantFoldingPass().run(block)

        assert len(folded_block.operations) == 1
        folded_gate = folded_block.operations[0]
        assert isinstance(folded_gate, GateOperation)
        assert isinstance(folded_gate.theta, Value)

        # First index folded, second unchanged
        assert len(folded_gate.theta.element_indices) == 2
        assert folded_gate.theta.element_indices[0].is_constant()
        assert folded_gate.theta.element_indices[0].get_const() == 3
        assert folded_gate.theta.element_indices[1].uuid == dynamic_idx.uuid


# ===========================================================================
# UUIDRemapper — theta cloning
# ===========================================================================


class TestUUIDRemapperTheta:
    """Tests for UUIDRemapper handling of GateOperation.theta."""

    def test_theta_value_gets_fresh_uuid(self) -> None:
        """Cloning a GateOperation should give theta a fresh UUID."""
        theta_val = _float_val("angle")
        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=theta_val)

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(gate)

        assert isinstance(cloned, GateOperation)
        assert isinstance(cloned.theta, Value)
        assert cloned.theta.uuid != theta_val.uuid
        assert cloned.theta.name == theta_val.name

    def test_theta_float_is_preserved(self) -> None:
        """Cloning should preserve theta when it's a plain float."""
        q = _qubit()
        gate = _make_gate(GateOperationType.RX, [q], theta=1.57)

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(gate)

        assert isinstance(cloned, GateOperation)
        assert cloned.theta == 1.57

    def test_theta_none_is_preserved(self) -> None:
        """Cloning should preserve theta=None."""
        q = _qubit()
        gate = _make_gate(GateOperationType.H, [q], theta=None)

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(gate)

        assert isinstance(cloned, GateOperation)
        assert cloned.theta is None

    def test_theta_uuid_appears_in_remap_table(self) -> None:
        """The remapper should record the theta Value's UUID mapping."""
        theta_val = _float_val("angle")
        q = _qubit()
        gate = _make_gate(GateOperationType.RY, [q], theta=theta_val)

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(gate)

        assert theta_val.uuid in remapper.uuid_remap
        assert isinstance(cloned, GateOperation)
        assert isinstance(cloned.theta, Value)
        assert remapper.uuid_remap[theta_val.uuid] == cloned.theta.uuid

    def test_operands_and_theta_get_independent_uuids(self) -> None:
        """Operand UUIDs and theta UUID should all be distinct after cloning."""
        theta_val = _float_val("angle")
        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=theta_val)

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(gate)

        assert isinstance(cloned, GateOperation)
        assert isinstance(cloned.theta, Value)
        cloned_uuids = {v.uuid for v in cloned.operands} | {v.uuid for v in cloned.results}
        assert cloned.theta.uuid not in cloned_uuids


# ===========================================================================
# ValueSubstitutor — theta substitution
# ===========================================================================


class TestValueSubstitutorTheta:
    """Tests for ValueSubstitutor handling of GateOperation.theta."""

    def test_theta_value_is_substituted(self) -> None:
        """Theta referencing a mapped Value gets substituted."""
        old_theta = _float_val("old_angle")
        new_theta = _float_val("new_angle", const=0.99)

        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=old_theta)

        sub = ValueSubstitutor({old_theta.uuid: new_theta})
        result = sub.substitute_operation(gate)

        assert isinstance(result, GateOperation)
        assert isinstance(result.theta, Value)
        assert result.theta.uuid == new_theta.uuid
        assert result.theta.get_const() == pytest.approx(0.99)

    def test_theta_not_in_map_is_unchanged(self) -> None:
        """Theta whose UUID is not in the map should remain as-is."""
        theta_val = _float_val("angle")
        q = _qubit()
        gate = _make_gate(GateOperationType.RX, [q], theta=theta_val)

        sub = ValueSubstitutor({})  # empty map
        result = sub.substitute_operation(gate)

        assert isinstance(result, GateOperation)
        assert isinstance(result.theta, Value)
        assert result.theta.uuid == theta_val.uuid

    def test_theta_float_is_unchanged(self) -> None:
        """Plain float theta should not be affected by substitution."""
        q = _qubit()
        gate = _make_gate(GateOperationType.RY, [q], theta=2.71)

        sub = ValueSubstitutor({})
        result = sub.substitute_operation(gate)

        assert isinstance(result, GateOperation)
        assert result.theta == 2.71

    def test_theta_and_operands_both_substituted(self) -> None:
        """Both operands and theta can be substituted in the same operation."""
        old_theta = _float_val("old_angle")
        new_theta = _float_val("new_angle", const=1.0)
        old_q = _qubit("old_q")
        new_q = _qubit("new_q")

        gate = _make_gate(GateOperationType.RZ, [old_q], theta=old_theta)

        sub = ValueSubstitutor({
            old_theta.uuid: new_theta,
            old_q.uuid: new_q,
        })
        result = sub.substitute_operation(gate)

        assert isinstance(result, GateOperation)
        # Theta substituted
        assert isinstance(result.theta, Value)
        assert result.theta.uuid == new_theta.uuid
        # Operand substituted
        assert result.operands[0].uuid == new_q.uuid


# ===========================================================================
# LoopAnalyzer._has_loop_var_binop
# ===========================================================================


class TestLoopAnalyzerBinOp:
    """Tests for LoopAnalyzer detecting BinOps dependent on loop variables."""

    def setup_method(self) -> None:
        self.analyzer = LoopAnalyzer()

    def test_direct_binop_dependency_triggers_unroll(self) -> None:
        """A BinOp using the loop variable directly should trigger unrolling."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=0.5)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.MUL)

        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=0.1)

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=5)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[binop, gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_no_binop_no_unroll(self) -> None:
        """A loop with no BinOps and no array access should not unroll."""
        q = _qubit()
        gate = _make_gate(GateOperationType.H, [q])

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is False

    def test_binop_not_using_loop_var_no_unroll(self) -> None:
        """A BinOp not referencing the loop variable should not trigger unrolling."""
        a = _float_val("a", const=1.0)
        b = _float_val("b", const=2.0)
        binop, _ = _make_binop(a, b, BinOpKind.ADD)

        q = _qubit()
        gate = _make_gate(GateOperationType.H, [q])

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[binop, gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is False

    def test_binop_in_nested_for_triggers_unroll(self) -> None:
        """A BinOp inside a nested ForOperation referencing the outer loop var."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=0.1)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.MUL)

        inner_start = _uint_val("is", const=0)
        inner_stop = _uint_val("ie", const=2)
        inner_step = _uint_val("ist", const=1)

        inner_for = ForOperation(
            operands=[inner_start, inner_stop, inner_step],
            results=[],
            loop_var="j",
            operations=[binop],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[inner_for],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_binop_in_if_true_branch_triggers_unroll(self) -> None:
        """A BinOp in the true branch of an IfOperation triggers unrolling."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=2.0)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.ADD)

        cond = Value(type=FloatType(), name="cond")  # placeholder condition
        if_op = IfOperation(
            operands=[cond],
            results=[],
            true_operations=[binop],
            false_operations=[],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[if_op],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_binop_in_if_false_branch_triggers_unroll(self) -> None:
        """A BinOp in the false branch of an IfOperation triggers unrolling."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=3.0)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.SUB)

        cond = Value(type=FloatType(), name="cond")
        if_op = IfOperation(
            operands=[cond],
            results=[],
            true_operations=[],
            false_operations=[binop],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[if_op],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_binop_in_while_triggers_unroll(self) -> None:
        """A BinOp inside a WhileOperation referencing loop var triggers unrolling."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=1.0)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.MUL)

        cond = Value(type=FloatType(), name="cond")
        while_op = WhileOperation(
            operands=[cond],
            results=[],
            operations=[binop],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[while_op],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_binop_in_for_items_triggers_unroll(self) -> None:
        """A BinOp inside ForItemsOperation referencing outer loop var."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=0.5)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.ADD)

        dict_val = Value(type=FloatType(), name="d")  # placeholder
        for_items_op = ForItemsOperation(
            operands=[dict_val],
            results=[],
            key_vars=["k"],
            value_var="v",
            operations=[binop],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[for_items_op],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True


# ===========================================================================
# LoopAnalyzer — theta array element access
# ===========================================================================


class TestLoopAnalyzerThetaArrayAccess:
    """Tests that theta array element access with loop var triggers unrolling."""

    def setup_method(self) -> None:
        self.analyzer = LoopAnalyzer()

    def test_theta_array_element_with_loop_var_triggers_unroll(self) -> None:
        """Gate with theta = gammas[i] where i is the loop var should unroll."""
        from qamomile.circuit.ir.value import ArrayValue

        gammas_array = ArrayValue(type=FloatType(), name="gammas")
        loop_idx = Value(type=UIntType(), name="i")
        theta_elem = Value(
            type=FloatType(),
            name="gammas_i",
            parent_array=gammas_array,
            element_indices=(loop_idx,),
        )

        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=theta_elem)

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_theta_array_element_without_loop_var_no_unroll(self) -> None:
        """Gate with theta = gammas[0] (constant index) should not unroll."""
        from qamomile.circuit.ir.value import ArrayValue

        gammas_array = ArrayValue(type=FloatType(), name="gammas")
        const_idx = _uint_val("idx", const=0)
        theta_elem = Value(
            type=FloatType(),
            name="gammas_0",
            parent_array=gammas_array,
            element_indices=(const_idx,),
        )

        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=theta_elem)

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[gate],
        )

        # No array element access using loop var, no binop using loop var
        assert self.analyzer.should_unroll(for_op, {}) is False
