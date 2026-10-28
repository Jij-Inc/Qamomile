"""Tests for condition-aware resource metadata and call summaries."""

from __future__ import annotations

import pytest
import sympy as sp
from sympy.logic.boolalg import Boolean

import qamomile.circuit as qmc
import qamomile.circuit.estimator._estimate_rewrite as estimate_rewrite_module
from qamomile.circuit.estimator._resource_expressions import _RangeAny

_RANGE_GUARD_SYMBOL = sp.Symbol("range_guard", integer=True)


def test_resource_metadata_axes_remain_independent_through_composition() -> None:
    """Derivation, count quality, and approximation compose independently."""
    modeled_conservative = qmc.ResourceEstimate(
        derivation=qmc.EstimateDerivation.MODELED,
        quality=qmc.EstimateQuality.CONSERVATIVE,
    )
    approximate = qmc.ResourceEstimate(
        approximation=qmc.ApproximationStatus.APPROXIMATE,
    )

    combined = modeled_conservative.seq(approximate)

    assert combined.derivation is qmc.EstimateDerivation.MODELED
    assert combined.quality is qmc.EstimateQuality.CONSERVATIVE
    assert combined.approximation is qmc.ApproximationStatus.APPROXIMATE
    assert (
        qmc.ResourceEstimate(
            quality=qmc.EstimateQuality.CONSERVATIVE,
        ).approximation
        is qmc.ApproximationStatus.EXACT
    )
    assert (
        qmc.ResourceEstimate(
            approximation=qmc.ApproximationStatus.APPROXIMATE,
        ).quality
        is qmc.EstimateQuality.EXACT
    )
    assert (
        combined.seq(qmc.ResourceEstimate(quality=qmc.EstimateQuality.UNKNOWN)).quality
        is qmc.EstimateQuality.UNKNOWN
    )


def test_conditional_substitution_restores_each_metadata_axis() -> None:
    """Specialization prunes guarded derivation and quality independently."""
    flag = sp.Symbol("flag", integer=True, nonnegative=True)
    modeled_conservative = qmc.ResourceEstimate(
        derivation=qmc.EstimateDerivation.MODELED,
        quality=qmc.EstimateQuality.CONSERVATIVE,
    )
    symbolic = modeled_conservative.conditional(
        qmc.ResourceEstimate.zero(),
        sp.Eq(flag, 1),
    )

    active = symbolic.substitute(flag=1)
    inactive = symbolic.substitute(flag=0)

    assert active.derivation is qmc.EstimateDerivation.MODELED
    assert active.quality is qmc.EstimateQuality.CONSERVATIVE
    assert inactive.derivation is qmc.EstimateDerivation.STRUCTURAL
    assert inactive.quality is qmc.EstimateQuality.EXACT
    assert symbolic.to_dict()["derivation"] == "modeled"
    assert symbolic.to_dict()["quality"] == "conservative"


def test_substitute_keeps_boolean_guards_out_of_numeric_rewriter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata guards use the non-clamping structural substitution path."""
    original = estimate_rewrite_module._substitute_resource_expr

    def reject_boolean(
        expression: sp.Basic,
        substitutions: dict[sp.Symbol, sp.Expr],
    ) -> sp.Basic:
        """Reject Boolean input to the numeric resource-expression helper.

        Args:
            expression (sp.Basic): Expression selected for substitution.
            substitutions (dict[sp.Symbol, sp.Expr]): Concrete replacements.

        Returns:
            sp.Basic: Substituted numeric expression.
        """
        assert not isinstance(expression, Boolean)
        return original(expression, substitutions)

    monkeypatch.setattr(
        estimate_rewrite_module,
        "_substitute_resource_expr",
        reject_boolean,
    )
    flag = sp.Symbol("flag", integer=True, nonnegative=True)
    symbolic = qmc.ResourceEstimate(
        quality=qmc.EstimateQuality.CONSERVATIVE,
    ).conditional(qmc.ResourceEstimate.zero(), sp.Eq(flag, 1))

    active = symbolic.substitute(flag=1)
    inactive = symbolic.substitute(flag=0)

    assert active.quality is qmc.EstimateQuality.CONSERVATIVE
    assert inactive.quality is qmc.EstimateQuality.EXACT


@pytest.mark.parametrize("uses_callback", [False, True])
def test_opaque_cost_preserves_declared_quality_and_approximation(
    uses_callback: bool,
) -> None:
    """Opaque boundaries mark derivation without replacing other axes."""
    declared = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=1, single_qubit=1),
        quality=qmc.EstimateQuality.CONSERVATIVE,
        approximation=qmc.ApproximationStatus.APPROXIMATE,
    )

    def cost(_ctx: qmc.OpaqueCostContext) -> qmc.ResourceEstimate:
        """Return the same definition-level cost for callback coverage.

        Args:
            _ctx (qmc.OpaqueCostContext): Opaque call context.

        Returns:
            qmc.ResourceEstimate: Declared definition-level cost.
        """
        return declared

    oracle = qmc.opaque(
        "axis_preserving_oracle",
        num_qubits=1,
        cost=cost if uses_callback else declared,
    )

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        """Invoke the modeled Oracle once."""
        (target,) = oracle(qmc.qubit("target"))
        return target

    estimate = circuit.estimate_resources()

    assert estimate.derivation is qmc.EstimateDerivation.MODELED
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert estimate.approximation is qmc.ApproximationStatus.APPROXIMATE


def test_conditional_algebra_prunes_inactive_metadata_and_calls() -> None:
    """Concrete substitution keeps metadata from only the selected branch."""
    flag = sp.Symbol("flag", integer=True, nonnegative=True)
    note = qmc.ResourceAssumption("fallback cost is modeled", source="untaken")
    taken = qmc.ResourceEstimate(
        calls=qmc.CallResources(calls_by_name={"taken": sp.Integer(1)}),
    )
    untaken = qmc.ResourceEstimate(
        calls=qmc.CallResources(calls_by_name={"untaken": sp.Integer(1)}),
        assumptions=(note,),
        derivation=qmc.EstimateDerivation.MODELED,
    )

    symbolic = taken.conditional(untaken, sp.Eq(flag, 1))
    selected_taken = symbolic.substitute(flag=1)
    selected_untaken = symbolic.substitute(flag=0)

    assert symbolic.derivation is qmc.EstimateDerivation.MODELED
    assert symbolic.assumptions == (note,)
    assert selected_taken.calls.calls_by_name == {"taken": 1}
    assert selected_taken.assumptions == ()
    assert selected_taken.derivation is qmc.EstimateDerivation.STRUCTURAL
    assert selected_taken.quality is qmc.EstimateQuality.EXACT
    assert selected_untaken.calls.calls_by_name == {"untaken": 1}
    assert selected_untaken.assumptions == (note,)
    assert selected_untaken.derivation is qmc.EstimateDerivation.MODELED
    assert selected_untaken.quality is qmc.EstimateQuality.EXACT


def _conditional_opaque_kernel() -> qmc.QKernel:
    """Build a symbolic branch with an exact and a bodyless alternative.

    Returns:
        qmc.QKernel: Kernel selecting Hadamard or an opaque callable.
    """
    oracle = qmc.opaque("conditional_oracle", num_qubits=1)

    @qmc.qkernel
    def circuit(flag: qmc.UInt) -> qmc.Qubit:
        """Select an exact gate or a bodyless oracle."""
        target = qmc.qubit("target")
        if flag:
            target = qmc.h(target)
        else:
            (target,) = oracle(target)
        return target

    return circuit


def test_qkernel_branch_prunes_inactive_opaque_call_and_trace() -> None:
    """Selecting the exact branch removes opaque provenance completely."""
    circuit = _conditional_opaque_kernel()
    symbolic = circuit.estimate_resources(
        unknown_policy=qmc.UnknownResourcePolicy.OPAQUE_CALL,
        trace=True,
    )

    taken = symbolic.substitute(flag=1)
    untaken = symbolic.substitute(flag=0)

    assert taken.calls.calls_by_name == {}
    assert taken.quality is qmc.EstimateQuality.EXACT
    assert "conditional_oracle" not in taken.explain()
    assert untaken.calls.calls_by_name == {"conditional_oracle": 1}
    assert untaken.derivation is qmc.EstimateDerivation.MODELED
    assert untaken.quality is qmc.EstimateQuality.UNKNOWN
    assert "conditional_oracle" in untaken.explain()


def test_qkernel_branch_prunes_inactive_zero_policy_warning() -> None:
    """An untaken zero-cost fallback does not leave a warning or modeled tag."""
    circuit = _conditional_opaque_kernel()
    symbolic = circuit.estimate_resources(
        unknown_policy=qmc.UnknownResourcePolicy.ZERO_WITH_WARNING,
    )

    taken = symbolic.substitute(flag=1)
    untaken = symbolic.substitute(flag=0)

    assert taken.assumptions == ()
    assert taken.quality is qmc.EstimateQuality.EXACT
    assert len(untaken.assumptions) == 1
    assert untaken.derivation is qmc.EstimateDerivation.MODELED
    assert untaken.quality is qmc.EstimateQuality.UNKNOWN


def test_call_resources_simplify_prunes_exact_zero_entries() -> None:
    """Zero-valued call names do not remain in user-facing dictionaries."""
    calls = qmc.CallResources(
        calls_by_name={"zero": sp.Integer(0), "one": sp.Integer(1)},
        queries_by_name={"zero": sp.Integer(0)},
    ).simplify()

    assert calls.calls_by_name == {"one": 1}
    assert calls.queries_by_name == {}


def test_large_concrete_range_prunes_unreachable_guarded_metadata() -> None:
    """An affine guard false over a large range leaves no modeled metadata."""
    iteration = sp.Symbol("iteration", integer=True)
    note = qmc.ResourceAssumption("large-range modeled branch")
    modeled = qmc.ResourceEstimate(
        assumptions=(note,),
        derivation=qmc.EstimateDerivation.MODELED,
    )

    estimate = modeled.conditional(
        qmc.ResourceEstimate.zero(),
        sp.Gt(iteration, 10_000),
    ).sum_over(
        iteration,
        sp.Integer(0),
        sp.Integer(5_000),
    )

    assert estimate.assumptions == ()
    assert estimate.derivation is qmc.EstimateDerivation.STRUCTURAL
    assert estimate.quality is qmc.EstimateQuality.EXACT


def test_large_concrete_range_keeps_reachable_guarded_metadata() -> None:
    """An affine guard reached through a large descending range stays active."""
    iteration = sp.Symbol("iteration", integer=True)
    note = qmc.ResourceAssumption("large-range modeled branch")
    modeled = qmc.ResourceEstimate(
        assumptions=(note,),
        derivation=qmc.EstimateDerivation.MODELED,
    )

    estimate = modeled.conditional(
        qmc.ResourceEstimate.zero(),
        sp.And(
            sp.Ge(iteration, -4_000),
            sp.Lt(iteration, -3_990),
        ),
    ).sum_over(
        iteration,
        sp.Integer(0),
        sp.Integer(-5_000),
        sp.Integer(-1),
    )

    assert estimate.assumptions == (note,)
    assert estimate.derivation is qmc.EstimateDerivation.MODELED
    assert estimate._guarded_derivations
    assert estimate._guarded_derivations[0].active_when is sp.true


@pytest.mark.parametrize(
    ("condition", "start", "step", "iterations", "expected"),
    (
        (sp.Gt(_RANGE_GUARD_SYMBOL, 4_999), 0, 1, 5_000, 0),
        (sp.Ge(_RANGE_GUARD_SYMBOL, 4_999), 0, 1, 5_000, 1),
        (sp.Lt(_RANGE_GUARD_SYMBOL, 0), 0, 1, 5_000, 0),
        (sp.Le(_RANGE_GUARD_SYMBOL, 0), 0, 1, 5_000, 1),
        (sp.Eq(_RANGE_GUARD_SYMBOL, 4_999), 0, 1, 5_000, 1),
        (sp.Eq(_RANGE_GUARD_SYMBOL, 5_000), 0, 1, 5_000, 0),
        (sp.Ne(_RANGE_GUARD_SYMBOL, 0), 0, 1, 5_000, 1),
        (sp.Eq(_RANGE_GUARD_SYMBOL, 3_999), 1, 2, 5_000, 1),
        (sp.Eq(_RANGE_GUARD_SYMBOL, 4_000), 1, 2, 5_000, 0),
        (sp.Eq(_RANGE_GUARD_SYMBOL, -4_997), 10_000, -3, 5_000, 1),
        (sp.Lt(_RANGE_GUARD_SYMBOL, -4_999), 0, -1, 5_000, 0),
        (
            sp.Or(
                sp.And(
                    sp.Gt(_RANGE_GUARD_SYMBOL, 10_000),
                    sp.Lt(_RANGE_GUARD_SYMBOL, 11_000),
                ),
                sp.Not(sp.Ge(_RANGE_GUARD_SYMBOL, 0)),
            ),
            0,
            1,
            5_000,
            0,
        ),
    ),
)
def test_large_affine_range_any_resolves_integer_boundaries(
    condition: sp.Basic,
    start: int,
    step: int,
    iterations: int,
    expected: int,
) -> None:
    """Large affine guards resolve exactly at integer relation boundaries."""
    result = _RangeAny(
        sp.Lambda(_RANGE_GUARD_SYMBOL, condition),
        sp.Integer(start),
        sp.Integer(step),
        sp.Integer(iterations),
    )

    assert result == expected


def test_empty_range_prunes_guarded_metadata() -> None:
    """A zero-trip range removes guarded metadata and assumptions exactly."""
    iteration = sp.Symbol("iteration", integer=True)
    note = qmc.ResourceAssumption("empty-range modeled branch")
    modeled = qmc.ResourceEstimate(
        assumptions=(note,),
        derivation=qmc.EstimateDerivation.MODELED,
    )

    estimate = modeled.conditional(
        qmc.ResourceEstimate.zero(),
        sp.Eq(iteration, 5),
    ).sum_over(
        iteration,
        sp.Integer(5),
        sp.Integer(5),
    )

    assert estimate.assumptions == ()
    assert estimate.derivation is qmc.EstimateDerivation.STRUCTURAL
    assert estimate.quality is qmc.EstimateQuality.EXACT


def test_large_range_keeps_unsupported_guards_symbolic() -> None:
    """External, nonlinear, and Piecewise predicates remain conservative."""
    threshold = sp.Symbol("threshold", integer=True)
    external = _RangeAny(
        sp.Lambda(
            _RANGE_GUARD_SYMBOL,
            sp.Gt(_RANGE_GUARD_SYMBOL, threshold),
        ),
        sp.Integer(0),
        sp.Integer(1),
        sp.Integer(5_000),
    )
    piecewise = _RangeAny(
        sp.Lambda(
            _RANGE_GUARD_SYMBOL,
            sp.Gt(
                sp.Piecewise(
                    (sp.Integer(1), sp.Lt(_RANGE_GUARD_SYMBOL, 2_500)),
                    (sp.Integer(0), True),
                ),
                0,
            ),
        ),
        sp.Integer(0),
        sp.Integer(1),
        sp.Integer(5_000),
    )
    nonlinear = _RangeAny(
        sp.Lambda(
            _RANGE_GUARD_SYMBOL,
            sp.Eq(_RANGE_GUARD_SYMBOL**2, 4),
        ),
        sp.Integer(0),
        sp.Integer(1),
        sp.Integer(5_000),
    )

    assert isinstance(external, _RangeAny)
    assert external.subs(threshold, 6_000) == 0
    assert external.subs(threshold, 4_000) == 1
    assert isinstance(piecewise, _RangeAny)
    assert isinstance(nonlinear, _RangeAny)


def test_large_affine_range_any_cost_is_independent_of_trip_count() -> None:
    """An enormous range resolves from affine boundaries without replay."""
    iterations = 10**30
    result = _RangeAny(
        sp.Lambda(
            _RANGE_GUARD_SYMBOL,
            sp.Eq(_RANGE_GUARD_SYMBOL, iterations - 1),
        ),
        sp.Integer(0),
        sp.Integer(1),
        sp.Integer(iterations),
    )

    assert result == 1
