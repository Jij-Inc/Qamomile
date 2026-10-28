"""Regression tests for safe finite-sum resource substitution."""

from unittest.mock import Mock

import pytest
import sympy as sp

import qamomile.circuit.estimator._resource_expressions as expressions_module
from qamomile.circuit.estimator._resource_bounds import (
    _conservative_maximum_sum_bound,
    _maximum_expr_over_range,
)


def test_boolean_condition_expands_ordered_piecewise_branches() -> None:
    """Boolean Piecewise values retain branch semantics instead of becoming Ne."""
    count = sp.Symbol("count", integer=True, nonnegative=True)
    value = sp.Symbol("value", real=True)
    condition = sp.Piecewise(
        (sp.false, sp.Eq(count, 0)),
        (value > 0, True),
    )

    normalized = expressions_module._boolean_condition(condition)

    assert sp.simplify_logic(normalized ^ (sp.Ne(count, 0) & (value > 0))) is sp.false
    assert normalized.subs(count, 0) is sp.false


def test_loop_maximum_folding_keeps_nested_sum_index_bound() -> None:
    """Piecewise folding cannot release a nested Sum's dummy index."""
    outer = sp.Symbol("outer", integer=True, nonnegative=True)
    inner = sp.Symbol("inner", integer=True, nonnegative=True)
    outer_count = sp.Symbol("outer_count", integer=True, nonnegative=True)
    inner_count = sp.Symbol("inner_count", integer=True, nonnegative=True)
    cutoff = sp.Symbol("cutoff", integer=True, nonnegative=True)
    nested_sum = expressions_module._sum_expr(
        sp.Piecewise((1, inner < cutoff), (0, True)),
        inner,
        sp.Integer(0),
        sp.Integer(1),
        inner_count,
    )
    expression = nested_sum + sp.Piecewise((outer, outer < 2), (0, True))

    maximum, _guard = _maximum_expr_over_range(
        expression,
        outer,
        sp.Integer(0),
        sp.Integer(1),
        outer_count,
    )
    upper_bound, _upper_guard = _conservative_maximum_sum_bound(
        nested_sum + sp.Piecewise((outer**2, outer < cutoff), (0, True)),
        outer,
        sp.Integer(0),
        sp.Integer(1),
        outer_count,
    )

    public_symbols = {outer_count, inner_count, cutoff}
    assert maximum.free_symbols <= public_symbols
    assert upper_bound.free_symbols <= public_symbols
    specialized = expressions_module._substitute_resource_expr(
        maximum,
        {
            outer_count: sp.Integer(3),
            inner_count: sp.Integer(4),
            cutoff: sp.Integer(2),
        },
    )
    assert specialized == 3


@pytest.mark.parametrize(
    ("iterations", "first_active"),
    [(129, 64), (1_100, 550)],
)
def test_sum_expr_closes_large_constant_piecewise_without_generic_evaluation(
    iterations: int,
    first_active: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large linear guards close exactly without generic finite summation."""
    index = sp.Symbol("index", integer=True, nonnegative=True)
    summand = sp.Piecewise((1, index >= first_active), (0, True))
    generic_evaluation = Mock(
        side_effect=AssertionError("constant Piecewise sum used generic evaluation")
    )
    monkeypatch.setattr(sp.Sum, "doit", generic_evaluation)

    result = expressions_module._sum_expr(
        summand,
        index,
        sp.Integer(0),
        sp.Integer(1),
        sp.Integer(iterations),
    )

    assert result == iterations - first_active
    generic_evaluation.assert_not_called()


def test_sum_expr_retains_unsupported_large_piecewise_without_generic_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported large guard retains an exact unevaluated finite Sum."""
    index = sp.Symbol("index", integer=True, nonnegative=True)
    summand = sp.Piecewise((1, sp.Eq(index**2, 4)), (0, True))
    generic_evaluation = Mock(
        side_effect=AssertionError("unsupported large Sum used generic evaluation")
    )
    monkeypatch.setattr(sp.Sum, "doit", generic_evaluation)

    result = expressions_module._sum_expr(
        summand,
        index,
        sp.Integer(0),
        sp.Integer(1),
        sp.Integer(1_100),
    )

    assert isinstance(result, sp.Sum)
    assert expressions_module._has_large_concrete_sum(result)
    generic_evaluation.assert_not_called()


@pytest.mark.parametrize("upper", [101, 2**100])
def test_constant_piecewise_sum_substitution_closes_exactly(
    upper: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constant Piecewise sums close exactly at ordinary and huge bounds."""
    index = sp.Symbol("index", integer=True)
    bound = sp.Symbol("bound", integer=True, nonnegative=True)
    expression = sp.Sum(
        sp.Piecewise((1, index > 0), (0, True)),
        (index, 0, bound),
    )
    generic_evaluation = Mock(
        side_effect=AssertionError("constant Piecewise sum used generic evaluation")
    )
    monkeypatch.setattr(sp.Sum, "doit", generic_evaluation)

    result = expressions_module._substitute_resource_expr(
        expression,
        {bound: sp.Integer(upper)},
    )

    assert result == sp.Integer(upper)
    generic_evaluation.assert_not_called()


def test_unsupported_huge_piecewise_sum_inside_max_stays_unevaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An index-valued huge Sum bypasses Max comparison and generic evaluation."""
    index = sp.Symbol("index", integer=True)
    bound = sp.Symbol("bound", integer=True, nonnegative=True)
    upper = sp.Integer(2**100)
    summand = sp.Piecewise((index, index > 0), (0, True))
    expression = sp.Max(
        1,
        sp.Sum(summand, (index, 0, bound)),
        evaluate=False,
    )
    expected = sp.Max(
        1,
        sp.Sum(summand, (index, 0, upper)),
        evaluate=False,
    )
    max_evaluation = Mock(
        side_effect=AssertionError("Max attempted to evaluate a huge Sum")
    )
    generic_evaluation = Mock(
        side_effect=AssertionError("huge Sum used generic evaluation")
    )
    monkeypatch.setattr(
        sp.Max,
        "_new_args_filter",
        staticmethod(max_evaluation),
    )
    monkeypatch.setattr(sp.Sum, "doit", generic_evaluation)

    result = expressions_module._substitute_resource_expr(
        expression,
        {bound: upper},
    )

    assert result == expected
    assert result.has(sp.Sum)
    max_evaluation.assert_not_called()
    generic_evaluation.assert_not_called()


def test_piecewise_sum_without_terminal_branch_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-exhaustive Piecewise remains a Sum instead of a partial total."""
    index = sp.Symbol("index", integer=True)
    bound = sp.Symbol("bound", integer=True, nonnegative=True)
    upper = sp.Integer(2**100)
    summand = sp.Piecewise((1, index > 0))
    expression = sp.Sum(summand, (index, 0, bound))
    expected = sp.Sum(summand, (index, 0, upper))
    generic_evaluation = Mock(
        side_effect=AssertionError("non-exhaustive Piecewise used generic evaluation")
    )
    monkeypatch.setattr(sp.Sum, "doit", generic_evaluation)

    result = expressions_module._substitute_resource_expr(
        expression,
        {bound: upper},
    )

    assert result == expected
    generic_evaluation.assert_not_called()


def test_nonaffine_piecewise_sum_guard_skips_set_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonlinear branch predicate falls back before invoking ``as_set``."""
    index = sp.Symbol("index", integer=True)
    bound = sp.Symbol("bound", integer=True, nonnegative=True)
    upper = sp.Integer(2**100)
    summand = sp.Piecewise((1, sp.Eq(index**2, 4)), (0, True))
    expression = sp.Sum(summand, (index, 0, bound))
    expected = sp.Sum(summand, (index, 0, upper))
    set_solver = Mock(
        side_effect=AssertionError("nonlinear predicate invoked the set solver")
    )
    generic_evaluation = Mock(
        side_effect=AssertionError("nonlinear Piecewise used generic evaluation")
    )
    monkeypatch.setattr(sp.logic.boolalg.Boolean, "as_set", set_solver)
    monkeypatch.setattr(sp.Sum, "doit", generic_evaluation)

    result = expressions_module._substitute_resource_expr(
        expression,
        {bound: upper},
    )

    assert result == expected
    set_solver.assert_not_called()
    generic_evaluation.assert_not_called()


def test_multilimit_sum_over_budget_stays_unevaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined work across multiple finite limits retains the exact Sum."""
    first = sp.Symbol("first", integer=True)
    second = sp.Symbol("second", integer=True)
    bound = sp.Symbol("bound", integer=True, nonnegative=True)
    upper = sp.Integer(32)
    expression = sp.Sum(
        first + second,
        (first, 0, bound),
        (second, 0, bound),
    )
    expected = sp.Sum(
        first + second,
        (first, 0, upper),
        (second, 0, upper),
    )
    generic_evaluation = Mock(
        side_effect=AssertionError("multi-limit Sum used generic evaluation")
    )
    monkeypatch.setattr(sp.Sum, "doit", generic_evaluation)

    result = expressions_module._substitute_resource_expr(
        expression,
        {bound: upper},
    )

    assert expressions_module._has_large_concrete_sum(expected)
    assert result == expected
    generic_evaluation.assert_not_called()


def test_reversed_huge_sum_limit_gets_lazy_nonnegative_clamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A huge reversed finite limit cannot produce a negative resource."""
    index = sp.Symbol("index", integer=True)
    bound = sp.Symbol("bound", integer=True, nonnegative=True)
    lower = sp.Integer(2**100)
    expression = sp.Sum(index, (index, bound, 0))
    substituted_sum = sp.Sum(index, (index, lower, 0))
    expected = sp.Max(0, substituted_sum, evaluate=False)
    generic_evaluation = Mock(
        side_effect=AssertionError("reversed huge Sum used generic evaluation")
    )
    monkeypatch.setattr(sp.Sum, "doit", generic_evaluation)

    result = expressions_module._substitute_resource_expr(
        expression,
        {bound: lower},
    )

    assert expressions_module._has_large_concrete_sum(substituted_sum)
    assert result == expected
    generic_evaluation.assert_not_called()


def test_explicitly_negative_huge_sum_wrapper_gets_lazy_nonnegative_clamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolved negative wrapper is clamped without evaluating its Sum."""
    index = sp.Dummy("index", integer=True, nonnegative=True)
    bound = sp.Symbol("bound", integer=True, nonnegative=True)
    upper = sp.Integer(2**100)
    summation = sp.Sum(
        sp.ceiling(sp.log(index + 1, 2)),
        (index, 0, bound - 1),
    )
    expression = sp.Mul(
        -1,
        sp.floor(summation, evaluate=False),
        evaluate=False,
    )
    substituted_sum = sp.Sum(
        summation.function,
        (index, 0, upper - 1),
    )
    expected_inner = sp.Mul(
        -1,
        sp.floor(substituted_sum, evaluate=False),
        evaluate=False,
    )
    expected = sp.Max(0, expected_inner, evaluate=False)
    generic_evaluation = Mock(
        side_effect=AssertionError("negative huge Sum used generic evaluation")
    )
    monkeypatch.setattr(sp.Sum, "doit", generic_evaluation)

    result = expressions_module._substitute_resource_expr(
        expression,
        {bound: upper},
    )

    assert result == expected
    assert result.is_nonnegative is True
    generic_evaluation.assert_not_called()


def test_additive_huge_sum_gets_lazy_nonnegative_clamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An additive expression cannot hide a negative large-sum resource."""
    index = sp.Dummy("index", integer=True, nonnegative=True)
    bound = sp.Symbol("bound", integer=True, nonnegative=True)
    upper = sp.Integer(2_000)
    summation = sp.Sum(index**2 + 1, (index, 0, bound))
    expression = sp.Add(5, -summation, evaluate=False)
    substituted_sum = sp.Sum(index**2 + 1, (index, 0, upper))
    substituted = sp.Add(5, -substituted_sum, evaluate=False)
    expected = sp.Max(0, substituted, evaluate=False)
    generic_evaluation = Mock(
        side_effect=AssertionError("additive huge Sum used generic evaluation")
    )
    monkeypatch.setattr(sp.Sum, "doit", generic_evaluation)

    result = expressions_module._substitute_resource_expr(
        expression,
        {bound: upper},
    )

    assert result == expected
    assert result.is_nonnegative is True
    generic_evaluation.assert_not_called()
