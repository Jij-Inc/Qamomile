"""Tests for emit_base pass — LoopAnalyzer BinOp dependency detection.

These tests verify that ``LoopAnalyzer.should_unroll`` correctly identifies
``ForOperation`` loops containing BinOps that depend on the loop variable
(directly or inside nested control-flow), and that theta array-element
access referencing the loop variable triggers unrolling.
"""

from __future__ import annotations

import pytest

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
from qamomile.circuit.transpiler.passes.emit_base import LoopAnalyzer


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

    The result type is inferred from the left-hand operand.
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

        cond = Value(type=FloatType(), name="cond")
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

        dict_val = Value(type=FloatType(), name="d")
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

        assert self.analyzer.should_unroll(for_op, {}) is False
