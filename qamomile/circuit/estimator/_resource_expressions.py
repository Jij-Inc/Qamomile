"""Re-export symbolic resource-expression helpers from their owner modules.

This private compatibility facade preserves established estimator imports while
the implementation remains split into acyclic responsibility modules.
"""

from __future__ import annotations

from qamomile.circuit.estimator._resource_bounds import (
    _condition_restricts_to_finite_points as _condition_restricts_to_finite_points,
    _conservative_maximum_sum_bound as _conservative_maximum_sum_bound,
    _endpoint_bounded_expression as _endpoint_bounded_expression,
    _guarded_range_candidate as _guarded_range_candidate,
    _guarded_resource_extension as _guarded_resource_extension,
    _is_structurally_less_equal as _is_structurally_less_equal,
    _is_structurally_nonnegative as _is_structurally_nonnegative,
    _maximum_expr_over_range as _maximum_expr_over_range,
    _maximum_piecewise_over_range as _maximum_piecewise_over_range,
    _resource_max as _resource_max,
    _resource_max_many as _resource_max_many,
)
from qamomile.circuit.estimator._resource_conditions import (
    _RANGE_ANY_REPLAY_LIMIT as _RANGE_ANY_REPLAY_LIMIT,
    _activation_over_range as _activation_over_range,
    _and_conditions as _and_conditions,
    _at_least_two_activations_over_range as _at_least_two_activations_over_range,
    _boolean_condition as _boolean_condition,
    _ConditionIndicator as _ConditionIndicator,
    _linear_condition_boundaries as _linear_condition_boundaries,
    _piecewise as _piecewise,
    _RangeAny as _RangeAny,
    _RangeAtLeastTwo as _RangeAtLeastTwo,
    _resolve_large_affine_range_any as _resolve_large_affine_range_any,
    _resource_activity_condition as _resource_activity_condition,
    _rewrite_condition as _rewrite_condition,
    _unresolved_condition_guard as _unresolved_condition_guard,
)
from qamomile.circuit.estimator._resource_scalars import (
    _add_maps as _add_maps,
    _expr as _expr,
    _resource_expr as _resource_expr,
    _safe_constraint_substitute as _safe_constraint_substitute,
    _safe_simplify as _safe_simplify,
    _substitute_basic_lazily as _substitute_basic_lazily,
    _substitute_resource_expr as _substitute_resource_expr,
)
from qamomile.circuit.estimator._resource_sums import (
    _evaluate_constant_piecewise_sum as _evaluate_constant_piecewise_sum,
    _finite_integer_set_cardinality as _finite_integer_set_cardinality,
    _finite_sum_is_structurally_nonnegative as _finite_sum_is_structurally_nonnegative,
    _has_large_concrete_sum as _has_large_concrete_sum,
    _large_sum_expression_is_structurally_negative as _large_sum_expression_is_structurally_negative,
    _large_sum_expression_is_structurally_nonnegative as _large_sum_expression_is_structurally_nonnegative,
    _simplify_sum_range_guards as _simplify_sum_range_guards,
    _sum_expr as _sum_expr,
)
