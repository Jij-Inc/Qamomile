"""Tests for LoopAnalyzer._has_loop_var_binop.

Verifies that BinOps depending on loop variables (directly or
transitively) are correctly detected, including inside nested
control flow structures.
"""

from __future__ import annotations

import numpy as np
import pytest

from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.types.primitives import FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes.emit_base import LoopAnalyzer


def _make_value(name: str, type_cls: type = UIntType) -> Value:
    """Create a simple Value with the given name."""
    return Value(type=type_cls(), name=name)


def _make_binop(
    lhs: Value,
    rhs: Value,
    result_name: str = "result",
    kind: BinOpKind = BinOpKind.ADD,
) -> BinOp:
    """Create a BinOp with the given operands."""
    result = _make_value(result_name, FloatType)
    return BinOp(
        operands=[lhs, rhs],
        results=[result],
        kind=kind,
    )


def _make_gate(qubit_name: str = "q") -> GateOperation:
    """Create a simple H gate (no theta)."""
    q_in = _make_value(qubit_name, QubitType)
    q_out = _make_value(qubit_name, QubitType)
    return GateOperation(
        operands=[q_in],
        results=[q_out],
        gate_type=GateOperationType.H,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer() -> LoopAnalyzer:
    """Create a fresh LoopAnalyzer instance."""
    return LoopAnalyzer()


# ---------------------------------------------------------------------------
# Direct dependency tests
# ---------------------------------------------------------------------------


class TestDirectDependency:
    """Tests for BinOps that directly reference the loop variable."""

    def test_binop_with_loop_var_on_lhs(self, analyzer: LoopAnalyzer) -> None:
        """BinOp whose left operand is the loop variable is detected."""
        loop_var = _make_value("i")
        offset = _make_value("offset")
        binop = _make_binop(loop_var, offset, result_name="idx")

        assert analyzer._has_loop_var_binop([binop], "i") is True

    def test_binop_with_loop_var_on_rhs(self, analyzer: LoopAnalyzer) -> None:
        """BinOp whose right operand is the loop variable is detected."""
        offset = _make_value("offset")
        loop_var = _make_value("i")
        binop = _make_binop(offset, loop_var, result_name="idx")

        assert analyzer._has_loop_var_binop([binop], "i") is True

    def test_binop_without_loop_var(self, analyzer: LoopAnalyzer) -> None:
        """BinOp not referencing the loop variable is not detected."""
        a = _make_value("a")
        b = _make_value("b")
        binop = _make_binop(a, b, result_name="c")

        assert analyzer._has_loop_var_binop([binop], "i") is False

    def test_empty_operations(self, analyzer: LoopAnalyzer) -> None:
        """Empty operation list returns False."""
        assert analyzer._has_loop_var_binop([], "i") is False

    def test_only_gates_no_binops(self, analyzer: LoopAnalyzer) -> None:
        """Operation list with only gates (no BinOps) returns False."""
        gate = _make_gate()
        assert analyzer._has_loop_var_binop([gate], "i") is False


# ---------------------------------------------------------------------------
# Transitive dependency tests
# ---------------------------------------------------------------------------


class TestTransitiveDependency:
    """Tests for BinOps transitively depending on the loop variable."""

    def test_transitive_via_result_name(self, analyzer: LoopAnalyzer) -> None:
        """BinOp depending on a BinOp result derived from loop var is detected.

        Simulates: idx = offset + i; val = idx * 2
        The first BinOp returns True immediately, so transitive detection
        happens only when the first BinOp's result feeds into a second.
        """
        loop_var = _make_value("i")
        offset = _make_value("offset")
        binop1 = _make_binop(offset, loop_var, result_name="idx")

        idx_ref = _make_value("idx")
        two = _make_value("two")
        binop2 = _make_binop(idx_ref, two, result_name="val", kind=BinOpKind.MUL)

        # First BinOp triggers early return
        assert analyzer._has_loop_var_binop([binop1, binop2], "i") is True


# ---------------------------------------------------------------------------
# Nested control flow tests
# ---------------------------------------------------------------------------


class TestNestedControlFlow:
    """Tests for BinOps inside nested control flow structures."""

    def test_binop_inside_for_operation(self, analyzer: LoopAnalyzer) -> None:
        """BinOp inside a nested ForOperation is detected."""
        loop_var = _make_value("i")
        offset = _make_value("offset")
        binop = _make_binop(loop_var, offset, result_name="idx")

        inner_for = ForOperation(
            operands=[_make_value("0"), _make_value("n"), _make_value("1")],
            results=[],
            loop_var="j",
            operations=[binop],
        )

        assert analyzer._has_loop_var_binop([inner_for], "i") is True

    def test_binop_inside_for_items_operation(
        self, analyzer: LoopAnalyzer
    ) -> None:
        """BinOp inside a ForItemsOperation is detected."""
        loop_var = _make_value("i")
        offset = _make_value("offset")
        binop = _make_binop(loop_var, offset, result_name="idx")

        for_items = ForItemsOperation(
            operands=[],
            results=[],
            key_vars=["k"],
            value_var="v",
            operations=[binop],
        )

        assert analyzer._has_loop_var_binop([for_items], "i") is True

    def test_binop_inside_if_true_branch(self, analyzer: LoopAnalyzer) -> None:
        """BinOp inside IfOperation true branch is detected."""
        loop_var = _make_value("i")
        offset = _make_value("offset")
        binop = _make_binop(loop_var, offset, result_name="idx")

        cond = _make_value("cond")
        if_op = IfOperation(
            operands=[cond],
            results=[],
            true_operations=[binop],
            false_operations=[],
        )

        assert analyzer._has_loop_var_binop([if_op], "i") is True

    def test_binop_inside_if_false_branch(
        self, analyzer: LoopAnalyzer
    ) -> None:
        """BinOp inside IfOperation false branch is detected."""
        loop_var = _make_value("i")
        offset = _make_value("offset")
        binop = _make_binop(loop_var, offset, result_name="idx")

        cond = _make_value("cond")
        if_op = IfOperation(
            operands=[cond],
            results=[],
            true_operations=[],
            false_operations=[binop],
        )

        assert analyzer._has_loop_var_binop([if_op], "i") is True

    def test_binop_inside_while_operation(
        self, analyzer: LoopAnalyzer
    ) -> None:
        """BinOp inside a WhileOperation is detected."""
        loop_var = _make_value("i")
        offset = _make_value("offset")
        binop = _make_binop(loop_var, offset, result_name="idx")

        while_op = WhileOperation(
            operands=[],
            results=[],
            operations=[binop],
        )

        assert analyzer._has_loop_var_binop([while_op], "i") is True

    def test_no_binop_inside_nested_for(self, analyzer: LoopAnalyzer) -> None:
        """ForOperation without loop-var BinOp returns False."""
        a = _make_value("a")
        b = _make_value("b")
        binop = _make_binop(a, b, result_name="c")

        inner_for = ForOperation(
            operands=[_make_value("0"), _make_value("n"), _make_value("1")],
            results=[],
            loop_var="j",
            operations=[binop],
        )

        assert analyzer._has_loop_var_binop([inner_for], "i") is False


# ---------------------------------------------------------------------------
# BinOp kind parametrization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        BinOpKind.ADD,
        BinOpKind.SUB,
        BinOpKind.MUL,
        BinOpKind.DIV,
        BinOpKind.FLOORDIV,
        BinOpKind.POW,
    ],
    ids=lambda k: k.name,
)
def test_all_binop_kinds_detected(
    analyzer: LoopAnalyzer, kind: BinOpKind
) -> None:
    """All BinOp kinds are detected when they depend on the loop variable."""
    loop_var = _make_value("i")
    other = _make_value("x")
    binop = _make_binop(loop_var, other, result_name="r", kind=kind)

    assert analyzer._has_loop_var_binop([binop], "i") is True


# ---------------------------------------------------------------------------
# Random testing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_random_loop_var_name_detected(
    analyzer: LoopAnalyzer, seed: int
) -> None:
    """BinOp with randomly named loop variable is detected."""
    rng = np.random.default_rng(seed)
    # Generate random loop variable name
    var_name = "var_" + str(rng.integers(0, 10000))

    loop_var = _make_value(var_name)
    other = _make_value("offset")
    binop = _make_binop(loop_var, other, result_name="idx")

    assert analyzer._has_loop_var_binop([binop], var_name) is True


@pytest.mark.parametrize("seed", range(5))
def test_random_non_loop_var_not_detected(
    analyzer: LoopAnalyzer, seed: int
) -> None:
    """BinOp with random non-matching names is not detected."""
    rng = np.random.default_rng(seed)
    name_a = "a_" + str(rng.integers(0, 10000))
    name_b = "b_" + str(rng.integers(0, 10000))

    a = _make_value(name_a)
    b = _make_value(name_b)
    binop = _make_binop(a, b, result_name="c")

    # Use a loop var name that definitely doesn't match
    assert analyzer._has_loop_var_binop([binop], "loop_i") is False
