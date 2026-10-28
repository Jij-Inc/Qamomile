"""Profile controlled work for shared-ladder decomposition."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._constants import (
    _ZERO,
)
from qamomile.circuit.estimator._control_decomposition import (
    CLEAN_ANCILLA_BATCH_MIN_WORK,
)
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _ConditionIndicator,
    _expr,
)
from qamomile.circuit.estimator._symbolic import (
    _CappedRangeSum,
)


@dataclasses.dataclass(frozen=True, slots=True)
class _EstimatorControlBatchProfile:
    """Track symbolic work needed to choose a shared control carrier.

    ``work`` is capped at two because the fixed resource model only
    distinguishes empty, singleton, and multi-operation bodies.

    Args:
        work (ResourceExpr | int): Symbolic controlled work, capped at two.
    """

    work: ResourceExpr | int = _ZERO

    def __post_init__(self) -> None:
        """Normalize the work expression."""
        object.__setattr__(self, "work", _expr(self.work))

    @property
    def active(self) -> Boolean:
        """Return whether this profile contains any controlled work.

        Returns:
            sp.Basic: Symbolic nonempty-work predicate.
        """
        work = cast(ResourceExpr, self.work)
        if work == _ZERO:
            return sp.false
        if work.is_positive is True:
            return sp.true
        if isinstance(work, _ConditionIndicator):
            return _boolean_condition(work.args[0])
        coefficient, remainder = work.as_coeff_Mul()
        if coefficient.is_positive and isinstance(remainder, _ConditionIndicator):
            return _boolean_condition(remainder.args[0])
        return _boolean_condition(sp.Gt(work, _ZERO))

    @property
    def has_multiple_work(self) -> Boolean:
        """Return whether the profile contains at least two work units.

        Returns:
            sp.Basic: Symbolic batching-threshold predicate.
        """
        work = cast(ResourceExpr, self.work)
        if work.is_number:
            return sp.true if work >= 2 else sp.false
        coefficient, remainder = work.as_coeff_Mul()
        if coefficient >= 2 and isinstance(remainder, _ConditionIndicator):
            return _boolean_condition(remainder.args[0])
        if isinstance(work, sp.Min) and _expr(2) in work.args:
            remaining = [arg for arg in work.args if arg != _expr(2)]
            if len(remaining) == 1:
                return _boolean_condition(sp.Ge(remaining[0], 2))
        return _boolean_condition(sp.Ge(work, 2))

    def when(self, condition: sp.Basic) -> _EstimatorControlBatchProfile:
        """Guard this profile by one symbolic activation condition.

        Args:
            condition (sp.Basic): Condition under which the work executes.

        Returns:
            _EstimatorControlBatchProfile: Conditionally active profile.
        """
        predicate = _boolean_condition(condition)
        indicator = cast(ResourceExpr, _ConditionIndicator(predicate))
        return _EstimatorControlBatchProfile(
            work=cast(ResourceExpr, self.work) * indicator,
        )

    def as_shared_leaf(self) -> _EstimatorControlBatchProfile:
        """Promote any active work to one independently batch-worthy leaf.

        Nested controlled calls, SELECT cases, and Pauli evolution already
        contain an intrinsic control boundary. One active instance therefore
        justifies composing that boundary with an outer shared carrier.

        Returns:
            _EstimatorControlBatchProfile: Heavy profile guarded by activity.
        """
        active = self.active
        indicator = cast(ResourceExpr, _ConditionIndicator(active))
        return _EstimatorControlBatchProfile(
            work=2 * indicator,
        )

    def conditional(
        self,
        other: _EstimatorControlBatchProfile,
        condition: sp.Basic,
    ) -> _EstimatorControlBatchProfile:
        """Select between two profiles with a symbolic condition.

        Args:
            other (_EstimatorControlBatchProfile): False-branch profile.
            condition (sp.Basic): Predicate selecting ``self`` when true.

        Returns:
            _EstimatorControlBatchProfile: Exact symbolic branch profile.
        """
        predicate = _boolean_condition(condition)
        true_indicator = cast(ResourceExpr, _ConditionIndicator(predicate))
        false_indicator = cast(
            ResourceExpr,
            _ConditionIndicator(sp.Not(predicate)),
        )
        return _EstimatorControlBatchProfile(
            work=(
                cast(ResourceExpr, self.work) * true_indicator
                + cast(ResourceExpr, other.work) * false_indicator
            ),
        )

    def choice(
        self,
        other: _EstimatorControlBatchProfile,
    ) -> _EstimatorControlBatchProfile:
        """Take the conservative work maximum of two possible branches.

        Args:
            other (_EstimatorControlBatchProfile): Alternative branch profile.

        Returns:
            _EstimatorControlBatchProfile: Field-wise branch maximum, capped
                at the shared-ladder threshold.
        """
        return _EstimatorControlBatchProfile(
            work=sp.Min(2, sp.Max(self.work, other.work)),
        )

    def sum_over(
        self,
        loop_symbol: sp.Symbol,
        start: ResourceExpr,
        step: ResourceExpr,
        iterations: ResourceExpr,
    ) -> _EstimatorControlBatchProfile:
        """Accumulate a profile over Python ``range`` semantics.

        Args:
            loop_symbol (sp.Symbol): Symbolic loop induction variable.
            start (ResourceExpr): First loop value.
            step (ResourceExpr): Loop step.
            iterations (ResourceExpr): Number of executed iterations.

        Returns:
            _EstimatorControlBatchProfile: Capped aggregate loop profile.
        """
        work = _CappedRangeSum(
            sp.Lambda(loop_symbol, cast(ResourceExpr, self.work)),
            start,
            step,
            iterations,
        )
        return _EstimatorControlBatchProfile(work=work)

    @classmethod
    def combine(
        cls,
        profiles: Iterable[_EstimatorControlBatchProfile],
    ) -> _EstimatorControlBatchProfile:
        """Combine sequential operation profiles with capped work.

        Args:
            profiles (Iterable[_EstimatorControlBatchProfile]): Profiles in
                program order.

        Returns:
            _EstimatorControlBatchProfile: Combined symbolic body profile.
        """
        work_items: list[ResourceExpr] = []
        for profile in profiles:
            profile_work = cast(ResourceExpr, profile.work)
            if profile_work != _ZERO:
                work_items.append(profile_work)
        if not work_items:
            work = _ZERO
        elif len(work_items) == 1:
            work = work_items[0]
        else:
            work = cast(ResourceExpr, sp.Min(2, sp.Add(*work_items)))
        return cls(work=work)


def _clean_ancilla_shared_ladder_condition(
    controls: ResourceExpr,
    profile: _EstimatorControlBatchProfile,
) -> sp.Basic:
    """Return when the fixed model shares one control-condition ladder.

    Both body-backed calls and aggregate opaque costs must use this exact
    threshold so an invocation boundary cannot change the selected resource
    recipe.

    Args:
        controls (ResourceExpr): Number of surrounding coherent controls.
        profile (_EstimatorControlBatchProfile): Capped controlled work.

    Returns:
        sp.Basic: Symbolic shared-ladder selection predicate.
    """
    return _boolean_condition(
        sp.And(
            sp.Ge(controls, CLEAN_ANCILLA_BATCH_MIN_WORK),
            profile.has_multiple_work,
        )
    )
