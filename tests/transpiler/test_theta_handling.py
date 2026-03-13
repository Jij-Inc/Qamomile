"""Tests for GateOperation.theta handling across transpiler passes.

GateOperation stores its angle parameter in a ``theta`` field separate from
``operands``.  Several transpiler passes must explicitly handle this field:
constant folding, UUID remapping, and value substitution.
These tests verify that theta is correctly propagated through each pass.

LoopAnalyzer tests (BinOp dependency detection and theta array-element
access) live in ``test_emit_base.py``.
"""

from __future__ import annotations

import pytest

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
)
from qamomile.circuit.ir.types.primitives import FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.passes.constant_fold import ConstantFoldingPass
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
    """Create a BinOp and its result Value.

    The result type is inferred from the left-hand operand so that index
    arithmetic with ``UIntType`` operands produces ``UIntType`` results.
    """
    result = Value(type=lhs.type, name=result_name)
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
        param = Value(type=FloatType(), name="phase", params={"parameter": "phase"})
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

    def test_nested_element_indices_are_folded_in_operands(self) -> None:
        """Operand q[indices[last]] has nested index folded when last is a BinOp.

        This is the pattern used by phase_gadget: last = k - 1, then
        q[indices[last]] as operand to RZ.
        """
        k = _uint_val("k", const=3)
        one = _uint_val("one", const=1)
        binop, last = _make_binop(k, one, BinOpKind.SUB, result_name="last")

        # indices is an array; indices[last] is a nested index
        indices_array = ArrayValue(type=UIntType(), name="indices")
        indices_last = Value(
            type=UIntType(),
            name="indices_elem",
            parent_array=indices_array,
            element_indices=(last,),
        )

        # q[indices[last]] — a qubit with nested element_indices
        q_array = ArrayValue(type=QubitType(), name="q")
        q_operand = Value(
            type=QubitType(),
            name="q_elem",
            parent_array=q_array,
            element_indices=(indices_last,),
        )

        gate = _make_gate(GateOperationType.RZ, [q_operand], theta=0.5)
        block = _make_block([binop, gate])

        folded_block = ConstantFoldingPass().run(block)

        assert len(folded_block.operations) == 1
        folded_gate = folded_block.operations[0]
        assert isinstance(folded_gate, GateOperation)

        # Check operand: q[indices[folded_last]]
        q_op = folded_gate.operands[0]
        assert isinstance(q_op, Value)
        assert len(q_op.element_indices) == 1
        inner = q_op.element_indices[0]
        assert isinstance(inner, Value)
        assert len(inner.element_indices) == 1
        folded_last = inner.element_indices[0]
        assert folded_last.is_constant()
        assert folded_last.get_const() == 2  # k(3) - 1 = 2

    def test_nested_element_indices_are_folded_in_results(self) -> None:
        """Result q[indices[last]] also has nested index folded."""
        k = _uint_val("k", const=3)
        one = _uint_val("one", const=1)
        binop, last = _make_binop(k, one, BinOpKind.SUB, result_name="last")

        indices_array = ArrayValue(type=UIntType(), name="indices")
        indices_last = Value(
            type=UIntType(),
            name="indices_elem",
            parent_array=indices_array,
            element_indices=(last,),
        )

        q_array = ArrayValue(type=QubitType(), name="q")
        q_operand = Value(
            type=QubitType(),
            name="q_elem",
            parent_array=q_array,
            element_indices=(indices_last,),
        )

        gate = _make_gate(GateOperationType.RZ, [q_operand], theta=0.5)
        block = _make_block([binop, gate])

        folded_block = ConstantFoldingPass().run(block)

        assert len(folded_block.operations) == 1
        folded_gate = folded_block.operations[0]
        assert isinstance(folded_gate, GateOperation)

        # Check result: q[indices[folded_last]]
        q_res = folded_gate.results[0]
        assert isinstance(q_res, Value)
        assert len(q_res.element_indices) == 1
        inner = q_res.element_indices[0]
        assert isinstance(inner, Value)
        assert len(inner.element_indices) == 1
        folded_last = inner.element_indices[0]
        assert folded_last.is_constant()
        assert folded_last.get_const() == 2  # k(3) - 1 = 2

    def test_stale_nested_index_does_not_survive_fold(self) -> None:
        """After fold, no stale uint_tmp reference remains in the gate."""
        k = _uint_val("k", const=5)
        one = _uint_val("one", const=1)
        binop, last = _make_binop(k, one, BinOpKind.SUB, result_name="uint_tmp")

        indices_array = ArrayValue(type=UIntType(), name="indices")
        indices_last = Value(
            type=UIntType(),
            name="indices_elem",
            parent_array=indices_array,
            element_indices=(last,),
        )

        q_array = ArrayValue(type=QubitType(), name="q")
        q_operand = Value(
            type=QubitType(),
            name="q_elem",
            parent_array=q_array,
            element_indices=(indices_last,),
        )

        gate = _make_gate(GateOperationType.RZ, [q_operand], theta=0.5)
        block = _make_block([binop, gate])

        folded_block = ConstantFoldingPass().run(block)

        folded_gate = folded_block.operations[0]
        assert isinstance(folded_gate, GateOperation)

        # Verify the deepest index is now a folded constant (not the
        # original unresolved BinOp result) in both operands and results.
        for label, val_list in [
            ("operands", folded_gate.operands),
            ("results", folded_gate.results),
        ]:
            q_val = val_list[0]
            assert isinstance(q_val, Value)
            inner = q_val.element_indices[0]
            assert isinstance(inner, Value)
            deepest = inner.element_indices[0]
            assert deepest.is_constant(), (
                f"Deepest index in {label} is not constant: {deepest}"
            )
            assert deepest.get_const() == 4, (  # k(5) - 1 = 4
                f"Deepest index in {label} has wrong value: {deepest.get_const()}"
            )


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
        cloned_uuids = {v.uuid for v in cloned.operands} | {
            v.uuid for v in cloned.results
        }
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

        sub = ValueSubstitutor(
            {
                old_theta.uuid: new_theta,
                old_q.uuid: new_q,
            }
        )
        result = sub.substitute_operation(gate)

        assert isinstance(result, GateOperation)
        # Theta substituted
        assert isinstance(result.theta, Value)
        assert result.theta.uuid == new_theta.uuid
        # Operand substituted
        assert result.operands[0].uuid == new_q.uuid
