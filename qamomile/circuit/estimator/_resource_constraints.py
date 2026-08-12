"""Represent and validate symbolic resource constraints."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence
from typing import Any, cast

import sympy as sp
from sympy.calculus.util import minimum as calculus_minimum
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._domain_affine import (
    _extract_affine_domain_expression,
)
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
    _is_concrete_integer,
    _symbol_display_name,
)
from qamomile.circuit.estimator._resource_expressions import (
    _and_conditions,
    _boolean_condition,
    _expr,
    _finite_integer_set_cardinality,
    _resource_expr,
    _safe_constraint_substitute,
    _safe_simplify,
    _substitute_basic_lazily,
)


class _ConstraintOrigin(enum.StrEnum):
    """Classify the semantic producer of a resource constraint.

    Values:
        INTERNAL: Internal structural requirement with no public input-domain
            contract.
        QKERNEL_INPUT: Scalar type or quantum-array shape declared by the root
            qkernel interface.
        ARRAY_ACCESS: Bounds requirement created by an array element access.
        ARRAY_VIEW: Shape or coverage requirement created by an array view.
        MODEL_CONTRACT: Opaque or modeled resource contract.
    """

    INTERNAL = "internal"
    QKERNEL_INPUT = "qkernel_input"
    ARRAY_ACCESS = "array_access"
    ARRAY_VIEW = "array_view"
    MODEL_CONTRACT = "model_contract"


_DOMAIN_ELIGIBLE_ORIGINS = frozenset(
    {
        _ConstraintOrigin.QKERNEL_INPUT,
        _ConstraintOrigin.ARRAY_ACCESS,
        _ConstraintOrigin.ARRAY_VIEW,
    }
)


@dataclasses.dataclass(frozen=True)
class _ConstraintProvenance:
    """Retain typed input-domain lineage for one structural constraint.

    Args:
        origin (_ConstraintOrigin): Semantic producer classification. Defaults
            to an internal, non-eligible requirement.
        source_expressions (tuple[sp.Basic, ...]): Exact resolved expressions
            from which the requirement was derived. Defaults to an empty
            tuple.
        root_formal_names (tuple[str, ...]): Stable root-qkernel formal names
            verified to cover every symbol in the constraint payload. An empty
            tuple means the constraint has not crossed that trust boundary.
    """

    origin: _ConstraintOrigin = _ConstraintOrigin.INTERNAL
    source_expressions: tuple[sp.Basic, ...] = ()
    root_formal_names: tuple[str, ...] = ()

    @property
    def domain_eligible(self) -> bool:
        """Return whether typed provenance reached the root-domain boundary.

        Returns:
            bool: Whether the origin is supported and at least one verified
                root formal contributes to the constraint.
        """
        return self.origin in _DOMAIN_ELIGIBLE_ORIGINS and bool(self.root_formal_names)

    def mapped(self, fn: Any) -> _ConstraintProvenance:
        """Rewrite the recorded source expressions.

        Args:
            fn (Any): Callable that accepts and returns a SymPy expression.

        Returns:
            _ConstraintProvenance: Provenance with mapped source expressions
                and unchanged stable formal names.
        """
        return dataclasses.replace(
            self,
            source_expressions=tuple(
                cast(sp.Basic, fn(expression)) for expression in self.source_expressions
            ),
        )


@dataclasses.dataclass(frozen=True)
class _ConstraintRange:
    """Describe one quantified loop range for a structural requirement.

    Args:
        symbol (sp.Symbol): Internal loop variable.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.
    """

    symbol: sp.Symbol
    start: ResourceExpr
    step: ResourceExpr
    iterations: ResourceExpr

    def mapped(self, fn: Any) -> _ConstraintRange:
        """Rewrite external expressions in this loop range.

        Args:
            fn (Any): Callable that accepts and returns a SymPy expression.

        Returns:
            _ConstraintRange: Rewritten quantified range.
        """
        return dataclasses.replace(
            self,
            start=fn(self.start),
            step=fn(self.step),
            iterations=fn(self.iterations),
        )


@dataclasses.dataclass(frozen=True)
class _ResourceConstraint:
    """Retain a structural requirement across symbolic estimation.

    Resource metrics alone cannot preserve every legality condition. For
    example, a zero control count or SELECT address width may simplify gate
    counts to zero even though the underlying operation is invalid. These
    constraints travel with an estimate and are rechecked whenever symbolic
    expressions are rewritten.

    Args:
        expression (ResourceExpr): Symbolic value constrained by the IR
            operation.
        minimum (int | None): Lower bound, or ``None`` when only an equality
            is required.
        label (str): User-facing name of the constrained value.
        unit (str): Optional singular unit appended to diagnostics. Defaults
            to an empty string.
        integer (bool): Whether concrete values must be integers. Defaults to
            ``True``.
        minimum_inclusive (bool): Whether ``minimum`` itself is accepted.
            Defaults to ``True``.
        finite (bool): Whether concrete values must be finite. Defaults to
            ``False``.
        expected (ResourceExpr | None): Required exact value. Defaults to
            ``None``.
        ranges (tuple[_ConstraintRange, ...]): Outer-to-inner loop ranges that
            quantify internal symbols in ``expression``. Defaults to an empty
            tuple.
        active_when (Boolean): Predicate under which the requirement applies.
            Defaults to true.
        provenance (_ConstraintProvenance): Typed semantic origin and resolved
            input lineage. Defaults to a non-eligible internal origin.
    """

    expression: ResourceExpr
    minimum: int | None
    label: str
    unit: str = ""
    integer: bool = True
    minimum_inclusive: bool = True
    finite: bool = False
    expected: ResourceExpr | None = None
    ranges: tuple[_ConstraintRange, ...] = ()
    active_when: Boolean = sp.true
    provenance: _ConstraintProvenance = dataclasses.field(
        default_factory=_ConstraintProvenance
    )

    @property
    def domain_eligible(self) -> bool:
        """Return whether this requirement can be offered to the domain prover.

        Returns:
            bool: Whether provenance is trusted and the requirement is
                unguarded and range-free.
        """
        return _constraint_is_domain_eligible(self)

    @property
    def source_formals(self) -> tuple[str, ...]:
        """Return stable root-qkernel formal names for this requirement.

        Returns:
            tuple[str, ...]: Verified source formal names in root-interface
                order, or an empty tuple for a non-domain constraint.
        """
        return self.provenance.root_formal_names

    def domain_predicate(self) -> Boolean | None:
        """Return the supported Boolean input-domain predicate.

        Returns:
            Boolean | None: Exact equality or lower-bound predicate, or
                ``None`` when this requirement must remain validation-only.
        """
        return _constraint_predicate(self)

    def mapped(self, fn: Any) -> _ResourceConstraint:
        """Rewrite and validate the constrained expression.

        Args:
            fn (Any): Callable that accepts and returns a SymPy expression.

        Returns:
            _ResourceConstraint: Rewritten structural constraint.

        Raises:
            ValueError: If rewriting resolves the expression to an invalid
                concrete value.
        """
        mapped = dataclasses.replace(
            self,
            expression=fn(self.expression),
            expected=(fn(self.expected) if self.expected is not None else None),
            ranges=tuple(loop_range.mapped(fn) for loop_range in self.ranges),
            active_when=_boolean_condition(fn(self.active_when)),
            provenance=self.provenance.mapped(fn),
        )
        mapped.validate()
        return mapped

    def when(self, condition: sp.Basic) -> _ResourceConstraint:
        """Make this requirement vacuous outside one symbolic branch.

        Args:
            condition (sp.Basic): Boolean condition selecting the branch that
                owns this requirement.

        Returns:
            _ResourceConstraint: Conditionally active requirement.
        """
        return dataclasses.replace(
            self,
            active_when=_and_conditions(self.active_when, condition),
        )

    def validate(self) -> None:
        """Reject an invalid concrete structural value.

        Raises:
            ValueError: If the expression is concrete and violates its
                integer or lower-bound requirement.
        """
        self._validate_ranges(0, {}, [4096])

    def _valid_fallback(self) -> sp.Integer:
        """Return one concrete value satisfying this lower-bound requirement.

        Conditional constraints use this value outside their active branch.
        Equality requirements replace it separately with zero.

        Returns:
            sp.Integer: Finite integer that satisfies the lower bound.
        """
        if self.minimum is None:
            return _ZERO
        offset = 0 if self.minimum_inclusive else 1
        return sp.Integer(self.minimum + offset)

    def _validate_ranges(
        self,
        range_index: int,
        substitutions: Mapping[sp.Symbol, sp.Expr],
        budget: list[int],
    ) -> None:
        """Validate all concrete points in quantified loop ranges.

        Args:
            range_index (int): Current range position.
            substitutions (Mapping[sp.Symbol, sp.Expr]): Values already bound
                by enclosing ranges.
            budget (list[int]): Remaining exhaustive validation points, stored
                in a mutable single-item list across recursive calls.

        Raises:
            ValueError: If a concrete range or constrained value is invalid.
        """
        active = _boolean_condition(
            _substitute_basic_lazily(self.active_when, substitutions)
        )
        if active is sp.false:
            return
        if range_index == len(self.ranges):
            if active is not sp.true:
                return
            resolved = _safe_constraint_substitute(self.expression, substitutions)
            if resolved.is_number:
                self._validate_value(resolved, substitutions)
                budget[0] -= 1
            return

        loop_range = self.ranges[range_index]
        start = _safe_constraint_substitute(loop_range.start, substitutions)
        step = _safe_constraint_substitute(loop_range.step, substitutions)
        iterations = _safe_constraint_substitute(
            loop_range.iterations,
            substitutions,
        )
        if not all(value.is_number for value in (start, step, iterations)):
            return
        if not all(value.is_integer is True for value in (start, step, iterations)):
            raise ValueError(
                f"Cannot validate {self.label}: its quantified loop range "
                "must resolve to integer values."
            )
        count = int(iterations)
        if count < 0:
            raise ValueError(
                f"Cannot validate {self.label}: loop iterations must be "
                f"nonnegative; got {count}."
            )
        if count > budget[0]:
            if range_index == len(self.ranges) - 1 and self._validate_large_range(
                loop_range,
                start,
                step,
                count,
                substitutions,
                budget,
            ):
                return
            raise ValueError(
                f"Cannot validate {self.label} exhaustively across {count} "
                "loop iterations; simplify the constrained expression or "
                "estimate with a smaller bound."
            )
        for offset in range(count):
            value = start + step * offset
            self._validate_ranges(
                range_index + 1,
                {**substitutions, loop_range.symbol: value},
                budget,
            )

    def _validate_large_range(
        self,
        loop_range: _ConstraintRange,
        start: sp.Expr,
        step: sp.Expr,
        count: int,
        substitutions: Mapping[sp.Symbol, sp.Expr],
        budget: list[int],
    ) -> bool:
        """Prove a large one-dimensional quantified requirement analytically.

        Args:
            loop_range (_ConstraintRange): Final quantified range.
            start (sp.Expr): Concrete first loop value.
            step (sp.Expr): Concrete loop step.
            count (int): Concrete positive iteration count.
            substitutions (Mapping[sp.Symbol, sp.Expr]): Outer loop values.
            budget (list[int]): Remaining point-validation budget shared by
                enclosing quantified ranges.

        Returns:
            bool: Whether integrality and the lower bound were proven without
            exhaustive enumeration.

        Raises:
            ValueError: If an analytically selected integer point violates the
                requirement.
        """
        expression = _safe_constraint_substitute(self.expression, substitutions)
        active_when = _boolean_condition(
            _substitute_basic_lazily(self.active_when, substitutions)
        )
        if active_when is sp.false:
            return True
        expected = (
            _safe_constraint_substitute(self.expected, substitutions)
            if self.expected is not None
            else None
        )
        external_symbols = expression.free_symbols - {loop_range.symbol}
        external_symbols.update(active_when.free_symbols - {loop_range.symbol})
        if expected is not None:
            external_symbols.update(expected.free_symbols - {loop_range.symbol})
        if external_symbols:
            return True
        if active_when is not sp.true:
            return self._validate_guarded_large_range(
                expression,
                active_when,
                loop_range,
                start,
                step,
                count,
                substitutions,
                budget,
            )

        index = sp.Dummy("constraint_index", integer=True, nonnegative=True)
        indexed = cast(
            ResourceExpr,
            expression.subs(loop_range.symbol, start + step * index),
        )
        indexed_expected = (
            cast(
                ResourceExpr,
                expected.subs(loop_range.symbol, start + step * index),
            )
            if expected is not None
            else None
        )
        integer_proven = not self.integer or indexed.is_integer is True
        if self.integer and not integer_proven:
            try:
                polynomial = sp.Poly(indexed, index)
            except sp.PolynomialError:
                polynomial = None
            integer_proven = polynomial is not None and all(
                coefficient.is_integer is True
                for coefficient in polynomial.all_coeffs()
            )
        finite_proven = not self.finite or indexed.is_finite is True
        real_proven = self.minimum is None or indexed.is_real is True

        if indexed_expected is not None:
            difference = _safe_simplify(cast(ResourceExpr, indexed - indexed_expected))
            if difference == _ZERO and integer_proven and finite_proven and real_proven:
                return True
            for offset in {0, count - 1}:
                resolved = _safe_constraint_substitute(
                    indexed,
                    {index: sp.Integer(offset)},
                )
                self._validate_value(
                    resolved,
                    {
                        **substitutions,
                        loop_range.symbol: start + step * offset,
                    },
                )
            return False

        domain = sp.Interval(_ZERO, sp.Integer(count - 1))
        try:
            lower_bound = calculus_minimum(indexed, index, domain)
        except (NotImplementedError, RecursionError, TypeError, ValueError):
            lower_bound = None
        if (
            integer_proven
            and finite_proven
            and real_proven
            and self.minimum is not None
            and isinstance(lower_bound, sp.Expr)
            and lower_bound.is_number
            and self._minimum_relation(lower_bound) is sp.true
        ):
            return True

        try:
            polynomial = sp.Poly(indexed, index)
        except sp.PolynomialError:
            return False
        stationary = sp.solveset(
            sp.diff(indexed, index),
            index,
            domain=domain,
        )
        if not isinstance(stationary, sp.FiniteSet):
            return False
        candidates = {0, count - 1}
        for point in stationary:
            if point.is_real is not True:
                continue
            try:
                neighbors = (int(sp.floor(point)), int(sp.ceiling(point)))
            except TypeError:
                return False
            candidates.update(
                neighbor for neighbor in neighbors if 0 <= neighbor < count
            )
        for offset in candidates:
            resolved = _safe_constraint_substitute(
                indexed,
                {index: sp.Integer(offset)},
            )
            if not resolved.is_number:
                return False
            self._validate_value(
                resolved,
                {
                    **substitutions,
                    loop_range.symbol: start + step * offset,
                },
            )
        return integer_proven and finite_proven and real_proven

    def _validate_guarded_large_range(
        self,
        expression: ResourceExpr,
        active_when: Boolean,
        loop_range: _ConstraintRange,
        start: sp.Expr,
        step: sp.Expr,
        count: int,
        substitutions: Mapping[sp.Symbol, sp.Expr],
        budget: list[int],
    ) -> bool:
        """Validate only the active subset of a large guarded range.

        The guard is converted to an integer offset set within the concrete
        loop range. Small sets are checked point by point under the shared
        validation budget. Large arithmetic ranges reuse the same analytic
        proof as an unguarded loop, so inequalities, descending ranges, and
        disjoint unions do not need separate ad-hoc boundary rules.

        Args:
            expression (ResourceExpr): Constraint expression before indexing.
            active_when (Boolean): Loop-index-dependent activation predicate.
            loop_range (_ConstraintRange): Quantified loop range.
            start (sp.Expr): Concrete first loop value.
            step (sp.Expr): Concrete loop step.
            count (int): Concrete positive iteration count.
            substitutions (Mapping[sp.Symbol, sp.Expr]): Concrete outer-range
                substitutions.
            budget (list[int]): Remaining point-validation budget shared by
                enclosing quantified ranges.

        Returns:
            bool: Whether every active point was proven and validated.

        Raises:
            ValueError: If an active point violates the requirement.
        """
        index = sp.Dummy("constraint_index", integer=True, nonnegative=True)
        indexed_guard = _boolean_condition(
            _substitute_basic_lazily(
                active_when,
                {loop_range.symbol: start + step * index},
            )
        )
        if indexed_guard.free_symbols - {index}:
            return False
        try:
            active_values = cast(
                sp.Set,
                sp.Intersection(
                    sp.Range(_ZERO, sp.Integer(count)),
                    indexed_guard.as_set(),
                ),
            )
        except (
            ArithmeticError,
            AttributeError,
            NotImplementedError,
            RecursionError,
            TypeError,
            ValueError,
        ):
            return False

        cardinality = _finite_integer_set_cardinality(active_values)
        if cardinality is None or not _is_concrete_integer(cardinality):
            return False
        active_count = int(cardinality)
        if active_count <= 0:
            return True

        expression_is_constant = loop_range.symbol not in expression.free_symbols
        expected_is_constant = (
            self.expected is None or loop_range.symbol not in self.expected.free_symbols
        )
        if expression_is_constant and expected_is_constant and expression.is_number:
            self._validate_value(expression, substitutions)
            budget[0] -= 1
            return True

        def validate_offset(offset: sp.Expr) -> bool:
            """Validate one active integer loop offset.

            Args:
                offset (sp.Expr): Zero-based concrete loop offset.

            Returns:
                bool: Whether the offset resolved to a concrete valid value.

            Raises:
                ValueError: If the active constraint is violated.
            """
            if not offset.is_number or not _is_concrete_integer(offset):
                return False
            loop_value = start + step * offset
            point_substitutions = {
                **substitutions,
                loop_range.symbol: loop_value,
            }
            resolved = _safe_constraint_substitute(expression, point_substitutions)
            if not resolved.is_number:
                return False
            self._validate_value(resolved, point_substitutions)
            budget[0] -= 1
            return True

        if active_count <= budget[0]:
            try:
                offsets = tuple(
                    cast(sp.Expr, value) for value in cast(Any, active_values)
                )
            except (NotImplementedError, TypeError, ValueError):
                return False
            return len(offsets) == active_count and all(
                validate_offset(offset) for offset in offsets
            )

        def validate_set(values: sp.Set) -> bool:
            """Validate one normalized component of the active offset set.

            Args:
                values (sp.Set): Finite integer offsets or arithmetic range.

            Returns:
                bool: Whether the component was validated analytically or
                within the remaining point budget.
            """
            component_count = _finite_integer_set_cardinality(values)
            if component_count is None or not _is_concrete_integer(component_count):
                return False
            size = int(component_count)
            if size <= 0:
                return True
            if size <= budget[0]:
                try:
                    offsets = tuple(cast(sp.Expr, value) for value in cast(Any, values))
                except (NotImplementedError, TypeError, ValueError):
                    return False
                return len(offsets) == size and all(
                    validate_offset(offset) for offset in offsets
                )
            if isinstance(values, sp.Range):
                range_size = cast(sp.Expr, values.size)
                if not _is_concrete_integer(range_size):
                    return False
                unguarded = dataclasses.replace(self, active_when=sp.true)
                return unguarded._validate_large_range(
                    loop_range,
                    start + step * cast(sp.Expr, values.start),
                    step * cast(sp.Expr, values.step),
                    int(range_size),
                    substitutions,
                    budget,
                )
            if isinstance(values, sp.Union):
                return all(validate_set(cast(sp.Set, subset)) for subset in values.args)
            return False

        return validate_set(active_values)

    def _validate_value(
        self,
        resolved: sp.Expr,
        substitutions: Mapping[sp.Symbol, sp.Expr],
    ) -> None:
        """Validate one concrete constrained value.

        Args:
            resolved (sp.Expr): Concrete value to validate.
            substitutions (Mapping[sp.Symbol, sp.Expr]): Quantified loop values
                used to resolve it.

        Raises:
            ValueError: If the value is non-finite, non-integral, or outside
                the lower bound.
        """
        location = ""
        if substitutions:
            assignments = ", ".join(
                f"{_symbol_display_name(symbol)}={value}"
                for symbol, value in substitutions.items()
            )
            location = f" at {assignments}"
        if self.finite and resolved.is_finite is not True:
            raise ValueError(f"{self.label} must be finite; got {resolved}{location}.")
        if self.minimum is not None and resolved.is_real is not True:
            raise ValueError(f"{self.label} must be real; got {resolved}{location}.")
        if self.integer and not _is_concrete_integer(resolved):
            raise ValueError(
                f"{self.label} must be an integer; got {resolved}{location}."
            )
        if self.expected is not None:
            expected = _safe_constraint_substitute(
                self.expected,
                substitutions,
            )
            if expected.is_number and sp.Ne(resolved, expected) is sp.true:
                raise ValueError(
                    f"{self.label} must equal {expected}; got {resolved}{location}."
                )
        if self.minimum is not None and self._minimum_violation(resolved) is sp.true:
            unit = ""
            if self.unit:
                plural = "" if self.minimum == 1 else "s"
                unit = f" {self.unit}{plural}"
            comparison = (
                f"greater than {self.minimum}"
                if not self.minimum_inclusive
                else f"at least {self.minimum}"
            )
            raise ValueError(
                f"{self.label} must be {comparison}{unit}; got {resolved}{location}."
            )

    def _minimum_relation(self, value: sp.Expr) -> Boolean:
        """Return the predicate that proves ``value`` satisfies the bound.

        Args:
            value (sp.Expr): Value or analytic lower bound to compare.

        Returns:
            Boolean: Inclusive or exclusive lower-bound predicate.
        """
        assert self.minimum is not None
        relation = sp.Ge if self.minimum_inclusive else sp.Gt
        return cast(Boolean, relation(value, self.minimum))

    def _minimum_violation(self, value: sp.Expr) -> Boolean:
        """Return the predicate that proves ``value`` violates the bound.

        Args:
            value (sp.Expr): Concrete constrained value.

        Returns:
            Boolean: Inclusive or exclusive lower-bound violation predicate.
        """
        assert self.minimum is not None
        relation = sp.Lt if self.minimum_inclusive else sp.Le
        return cast(Boolean, relation(value, self.minimum))

    def bound_over(
        self,
        loop_symbol: sp.Symbol,
        start: ResourceExpr,
        step: ResourceExpr,
        iterations: ResourceExpr,
    ) -> _ResourceConstraint:
        """Quantify a requirement over a repeated loop body.

        Affine constrained expressions attain their minimum at one endpoint,
        so they can be reduced to an external expression and remain
        checkable after later input substitution. Non-affine expressions keep
        the loop symbol as an internal bound variable instead of exposing it
        as a user parameter.

        Args:
            loop_symbol (sp.Symbol): Internal loop variable to bind.
            start (ResourceExpr): First loop value.
            step (ResourceExpr): Loop step.
            iterations (ResourceExpr): Number of executed iterations.

        Returns:
            _ResourceConstraint: Constraint with the loop variable reduced or
            marked as internally bound.

        Raises:
            ValueError: If a concrete nonempty range violates the constraint.
        """
        if any(
            loop_symbol in expression.free_symbols
            for expression in self.provenance.source_expressions
        ):
            self = dataclasses.replace(
                self,
                provenance=_ConstraintProvenance(),
            )
        range_expressions = (
            expression
            for loop_range in self.ranges
            for expression in (
                loop_range.start,
                loop_range.step,
                loop_range.iterations,
            )
        )
        appears_in_nested_range = any(
            loop_symbol in expression.free_symbols for expression in range_expressions
        )
        appears_in_expected = (
            self.expected is not None and loop_symbol in self.expected.free_symbols
        )
        appears_in_activation = loop_symbol in self.active_when.free_symbols
        if (
            loop_symbol not in self.expression.free_symbols
            and not appears_in_expected
            and not appears_in_nested_range
            and not appears_in_activation
        ):
            return self
        quantified_range = _ConstraintRange(
            symbol=loop_symbol,
            start=_expr(start),
            step=_expr(step),
            iterations=_expr(iterations),
        )
        if (
            self.expected is not None
            or self.ranges
            or loop_symbol not in self.expression.free_symbols
            or appears_in_activation
        ):
            bound = dataclasses.replace(
                self,
                ranges=(quantified_range, *self.ranges),
            )
            bound.validate()
            return bound
        try:
            polynomial = sp.Poly(self.expression, loop_symbol)
        except sp.PolynomialError:
            polynomial = None
        if polynomial is None or polynomial.degree() > 1:
            bound = dataclasses.replace(
                self,
                ranges=(quantified_range,),
            )
            bound.validate()
            return bound

        first = cast(sp.Expr, self.expression.subs(loop_symbol, start))
        last_value = start + step * (iterations - _ONE)
        last = cast(sp.Expr, self.expression.subs(loop_symbol, last_value))
        direction = _safe_simplify(
            cast(
                ResourceExpr,
                sp.diff(self.expression, loop_symbol) * step,
            )
        )
        if direction.is_nonnegative:
            range_minimum = first
        elif direction.is_nonpositive:
            range_minimum = last
        else:
            range_minimum = sp.Min(first, last)
        expression = _resource_expr(
            sp.Piecewise(
                (range_minimum, sp.Gt(iterations, _ZERO)),
                (self._valid_fallback(), True),
            )
        )
        bound = dataclasses.replace(
            self,
            expression=expression,
        )
        bound.validate()
        return bound


def _mark_root_domain_constraints(
    constraints: Sequence[_ResourceConstraint],
    root_formals: Mapping[sp.Symbol, str],
) -> tuple[_ResourceConstraint, ...]:
    """Mark typed constraints whose complete lineage belongs to root inputs.

    SymPy symbol equality, rather than display spelling, defines membership in
    the root interface. Equal non-``Dummy`` symbols represent the same SymPy
    variable even when an IR view reconstructed the Python object. A symbol
    with different assumptions and every identity-distinct ``Dummy`` remain
    separate and cannot be promoted merely because their names match.
    Quantified requirements remain unmarked because their bound symbols are
    intentionally outside the root interface.

    Args:
        constraints (Sequence[_ResourceConstraint]): Structural requirements
            after call and view mapping and before user input substitution.
        root_formals (Mapping[sp.Symbol, str]): Root-interface symbols mapped
            to their stable public formal names in interface order.

    Returns:
        tuple[_ResourceConstraint, ...]: Constraints with verified source
            formal names attached only to supported typed origins.
    """
    formal_entries = tuple(root_formals.items())
    marked: list[_ResourceConstraint] = []
    for constraint in constraints:
        provenance = dataclasses.replace(
            constraint.provenance,
            root_formal_names=(),
        )
        if provenance.origin not in _DOMAIN_ELIGIBLE_ORIGINS or constraint.ranges:
            marked.append(dataclasses.replace(constraint, provenance=provenance))
            continue

        payload: list[sp.Basic] = [
            constraint.expression,
            constraint.active_when,
            *provenance.source_expressions,
        ]
        if constraint.expected is not None:
            payload.append(constraint.expected)
        symbols = set().union(*(expression.free_symbols for expression in payload))
        if (
            not symbols
            or any(isinstance(symbol, sp.Dummy) for symbol in symbols)
            or any(symbol not in root_formals for symbol in symbols)
        ):
            marked.append(dataclasses.replace(constraint, provenance=provenance))
            continue

        names = tuple(name for symbol, name in formal_entries if symbol in symbols)
        marked.append(
            dataclasses.replace(
                constraint,
                provenance=dataclasses.replace(
                    provenance,
                    root_formal_names=names,
                ),
            )
        )
    return tuple(marked)


def _constraint_predicate(
    constraint: _ResourceConstraint,
) -> Boolean | None:
    """Build the supported domain predicate for one trusted requirement.

    Args:
        constraint (_ResourceConstraint): Constraint whose typed root
            provenance, activation, range, and scalar facts are inspected.

    Returns:
        Boolean | None: Exact equality or lower-bound predicate, or ``None``
            when the constraint must remain validation-only.
    """
    if not constraint.domain_eligible:
        return None
    expression = constraint.expression
    if constraint.expected is not None:
        return cast(Boolean, sp.Eq(expression, constraint.expected))
    return constraint._minimum_relation(expression)


def _constraint_is_domain_eligible(constraint: _ResourceConstraint) -> bool:
    """Return whether a constraint satisfies the first-pass source grammar.

    Args:
        constraint (_ResourceConstraint): Typed structural requirement to
            inspect before constructing its Boolean predicate.

    Returns:
        bool: Whether the requirement is trusted, unguarded, range-free,
            finite, integral when required, and affine.
    """
    if (
        not constraint.provenance.domain_eligible
        or constraint.ranges
        or constraint.active_when is not sp.true
    ):
        return False
    expression = constraint.expression
    if not _is_affine_domain_expression(expression):
        return False
    if (
        constraint.integer and expression.is_integer is not True
    ) or expression.is_finite is not True:
        return False
    if constraint.expected is not None:
        if not _is_affine_domain_expression(constraint.expected):
            return False
        return (
            constraint.minimum is None
            and (not constraint.integer or constraint.expected.is_integer is True)
            and constraint.expected.is_finite is True
        )
    return constraint.minimum is not None and (
        constraint.integer or expression.is_real is True
    )


def _is_affine_domain_expression(expression: sp.Expr) -> bool:
    """Return whether an expression is affine in all of its free symbols.

    Args:
        expression (sp.Expr): Candidate source side for a domain relation.

    Returns:
        bool: Whether bounded structural inspection proves an affine form.
    """
    return _extract_affine_domain_expression(expression) is not None
