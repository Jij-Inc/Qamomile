"""Shared assertion helpers for resource estimation tests."""

from __future__ import annotations

import sympy as sp

from qamomile.circuit.estimator import CircuitDepth, GateCount


def assert_expr_equal(actual: sp.Expr, expected: sp.Expr, msg: str = "") -> None:
    """Assert two SymPy expressions are symbolically equal."""
    diff = sp.simplify(actual - expected)
    assert diff == 0, (
        f"SymPy expressions not equal{' (' + msg + ')' if msg else ''}: "
        f"actual={actual}, expected={expected}, diff={diff}"
    )


def assert_gate_counts(actual: GateCount, expected: GateCount) -> None:
    """Assert all gate count fields match."""
    assert_expr_equal(actual.total, expected.total, "total")
    assert_expr_equal(actual.single_qubit, expected.single_qubit, "single_qubit")
    assert_expr_equal(actual.two_qubit, expected.two_qubit, "two_qubit")
    assert_expr_equal(actual.multi_qubit, expected.multi_qubit, "multi_qubit")
    assert_expr_equal(actual.t_gates, expected.t_gates, "t_gates")
    assert_expr_equal(actual.clifford_gates, expected.clifford_gates, "clifford_gates")
    assert_expr_equal(actual.rotation_gates, expected.rotation_gates, "rotation_gates")
    for name, expected_count in expected.oracle_calls.items():
        assert name in actual.oracle_calls, f"Missing oracle call: {name}"
        assert_expr_equal(
            actual.oracle_calls[name], expected_count, f"oracle_calls[{name}]"
        )
    assert set(actual.oracle_calls.keys()) == set(expected.oracle_calls.keys()), (
        f"oracle_calls keys mismatch: "
        f"actual={set(actual.oracle_calls.keys())}, "
        f"expected={set(expected.oracle_calls.keys())}"
    )
    # Only check oracle_queries when expected specifies them
    if expected.oracle_queries:
        for name, expected_count in expected.oracle_queries.items():
            assert name in actual.oracle_queries, f"Missing oracle query: {name}"
            assert_expr_equal(
                actual.oracle_queries[name],
                expected_count,
                f"oracle_queries[{name}]",
            )
        assert set(actual.oracle_queries.keys()) == set(
            expected.oracle_queries.keys()
        ), (
            f"oracle_queries keys mismatch: "
            f"actual={set(actual.oracle_queries.keys())}, "
            f"expected={set(expected.oracle_queries.keys())}"
        )


def assert_depth(actual: CircuitDepth, expected: CircuitDepth) -> None:
    """Assert all depth fields match."""
    assert_expr_equal(actual.total_depth, expected.total_depth, "total_depth")
    assert_expr_equal(actual.t_depth, expected.t_depth, "t_depth")
    assert_expr_equal(
        actual.two_qubit_depth, expected.two_qubit_depth, "two_qubit_depth"
    )
    assert_expr_equal(
        actual.multi_qubit_depth, expected.multi_qubit_depth, "multi_qubit_depth"
    )
    assert_expr_equal(actual.rotation_depth, expected.rotation_depth, "rotation_depth")
