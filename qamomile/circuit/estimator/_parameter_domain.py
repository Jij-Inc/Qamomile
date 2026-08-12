"""Rewrite public resource formulas over proven qkernel input domains."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable
from typing import Any, cast

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._domain_affine import (
    _constant_is_zero,
    _extract_affine_domain_expression,
)
from qamomile.circuit.estimator._resource_base import ResourceExpr

_DOMAIN_RAW_CONSTRAINT_LIMIT = 4096
_DOMAIN_SOURCE_LIMIT = 128
_DOMAIN_SYMBOL_LIMIT = 64
_DOMAIN_EXPRESSION_NODE_LIMIT = 512
_DOMAIN_PREDICATE_NODE_LIMIT = 1024
_DOMAIN_COMPARISON_LIMIT = 4096
_DOMAIN_PROOF_CACHE_LIMIT = 2048

_PROTECTED_NODE_TYPES = (
    sp.Derivative,
    sp.Integral,
    sp.Lambda,
    sp.Limit,
    sp.Product,
    sp.Sum,
)
_ARITHMETIC_NODE_TYPES = (sp.Add, sp.Mul, sp.Pow, sp.Max, sp.Min, sp.Piecewise)
_BOOLEAN_NODE_TYPES = (
    sp.logic.boolalg.And,
    sp.logic.boolalg.Not,
    sp.logic.boolalg.Or,
    Relational,
)


class _DomainRewritePolicy(enum.StrEnum):
    """Control whether an estimate may use qkernel input-domain facts.

    Values:
        INHERITED: Let a containing estimator or algebra operation choose.
        ENABLED: Restore and run domain-aware simplification at boundaries.
        DISABLED: Preserve domain-independent formulas until explicitly enabled.
    """

    INHERITED = "inherited"
    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclasses.dataclass(frozen=True)
class _ConsumedDomainRequirement:
    """Record one qkernel input-domain predicate used by a public rewrite.

    Args:
        predicate (Boolean): Exact predicate required for the rewrite.
        source_formals (tuple[str, ...]): Stable root-qkernel formal names that
            produced the predicate.
        labels (tuple[str, ...]): Structural constraint labels supporting the
            predicate.
    """

    predicate: Boolean
    source_formals: tuple[str, ...]
    labels: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _DomainSource:
    """Keep a projected domain fact together with its public evidence.

    Args:
        requirement (_ConsumedDomainRequirement): Structured public evidence
            for this source predicate.
        facts (tuple[Boolean, ...]): Conjunctive atomic facts available to the
            bounded prover.
    """

    requirement: _ConsumedDomainRequirement
    facts: tuple[Boolean, ...]


@dataclasses.dataclass
class _ProofEnvironment:
    """Hold bounded proof sources, caches, and the remaining comparison budget.

    Args:
        sources (tuple[_DomainSource, ...]): Eligible projected domain sources.
        remaining_comparisons (int): Maximum remaining relational comparisons.
        cache (dict[tuple[Boolean, tuple[Boolean, ...]], tuple[int, ...] | None]):
            Bounded proof-result cache.
        exhausted (bool): Whether any hard proof budget has been exceeded.
    """

    sources: tuple[_DomainSource, ...]
    remaining_comparisons: int = dataclasses.field(
        default_factory=lambda: _DOMAIN_COMPARISON_LIMIT
    )
    cache: dict[
        tuple[Boolean, tuple[Boolean, ...]],
        tuple[int, ...] | None,
    ] = dataclasses.field(default_factory=dict)
    exhausted: bool = False

    def is_exhausted(self) -> bool:
        """Return whether any shared proof budget was exhausted.

        Returns:
            bool: Whether the environment must fail closed.
        """
        return self.exhausted

    def spend_comparison(self) -> bool:
        """Consume one bounded relational comparison.

        Returns:
            bool: Whether the comparison remains within the proof budget.
        """
        if self.exhausted:
            return False
        self.remaining_comparisons -= 1
        if self.remaining_comparisons < 0:
            self.exhausted = True
            return False
        return True

    def remember(
        self,
        key: tuple[Boolean, tuple[Boolean, ...]],
        result: tuple[int, ...] | None,
    ) -> None:
        """Store one proof result without exceeding the cache budget.

        Args:
            key (tuple[Boolean, tuple[Boolean, ...]]): Predicate and branch
                context used for the proof.
            result (tuple[int, ...] | None): Supporting source indices, or None
                when the predicate was not proven.
        """
        if key in self.cache:
            self.cache[key] = result
            return
        if len(self.cache) >= _DOMAIN_PROOF_CACHE_LIMIT:
            self.exhausted = True
            return
        self.cache[key] = result


def _rewrite_expression_over_domain(
    expression: ResourceExpr,
    constraints: Iterable[Any],
) -> tuple[ResourceExpr, tuple[_ConsumedDomainRequirement, ...]]:
    """Rewrite one public resource expression over eligible input constraints.

    The function is intentionally fail-closed. Unsupported symbolic nodes,
    contradictory sources, malformed provenance, or exhausted proof budgets
    return the original expression without partially consumed requirements.

    Args:
        expression (ResourceExpr): Domain-independent public metric formula.
        constraints (Iterable[Any]): Resource constraints exposing
            ``domain_eligible``, ``source_formals``, ``label``, ``ranges``,
            ``active_when``, and ``domain_predicate()``.

    Returns:
        tuple[ResourceExpr, tuple[_ConsumedDomainRequirement, ...]]: Rewritten
        formula and the exact projected predicates it consumed. The original
        formula and an empty tuple are returned when the bounded pass cannot
        prove a complete safe rewrite.
    """
    normalized = cast(ResourceExpr, sp.sympify(expression))
    environment = _prepare_domain_proof_environment(constraints)
    if environment is None:
        return normalized, ()
    return _rewrite_expression_in_domain(normalized, environment)


def _prepare_domain_proof_environment(
    constraints: Iterable[Any],
) -> _ProofEnvironment | None:
    """Project and validate input-domain sources once for a rewrite batch.

    Args:
        constraints (Iterable[Any]): Candidate resource constraints.

    Returns:
        _ProofEnvironment | None: Shared bounded proof environment, or
        ``None`` when no usable domain exists or validation fails closed.
    """
    sources = _project_domain_sources(constraints)
    if sources is None or not sources:
        return None
    environment = _ProofEnvironment(sources)
    if _sources_are_contradictory(environment) or environment.exhausted:
        return None
    return environment


def _rewrite_expression_in_domain(
    expression: ResourceExpr,
    environment: _ProofEnvironment,
) -> tuple[ResourceExpr, tuple[_ConsumedDomainRequirement, ...]]:
    """Rewrite one expression using an already validated shared domain.

    Args:
        expression (ResourceExpr): Public metric formula.
        environment (_ProofEnvironment): Shared proof sources, cache, and
            aggregate comparison budget for the containing estimate.

    Returns:
        tuple[ResourceExpr, tuple[_ConsumedDomainRequirement, ...]]: Rewritten
        formula and exact consumed predicates, or the original formula with no
        evidence when unsupported or exhausted.
    """
    normalized = cast(ResourceExpr, sp.sympify(expression))
    if environment.is_exhausted() or not _expression_is_supported(normalized):
        return normalized, ()

    rewritten, evidence = _rewrite_node(normalized, (), environment)
    if environment.is_exhausted():
        return normalized, ()
    if rewritten == normalized:
        return normalized, ()
    requirements = tuple(
        environment.sources[index].requirement for index in sorted(evidence)
    )
    return rewritten, requirements


def _expression_is_supported(expression: sp.Basic) -> bool:
    """Return whether the whole expression belongs to the bounded allowlist.

    Args:
        expression (sp.Basic): Public metric expression to inspect.

    Returns:
        bool: Whether every node is binder-free and supported by the walker.
    """
    for index, node in enumerate(sp.preorder_traversal(expression), start=1):
        if index > _DOMAIN_EXPRESSION_NODE_LIMIT:
            return False
        if isinstance(node, _PROTECTED_NODE_TYPES):
            return False
        if isinstance(node, AppliedUndef):
            return False
        if isinstance(node, sp.Dummy):
            return False
        if isinstance(node, sp.Function) and not isinstance(
            node,
            _ARITHMETIC_NODE_TYPES,
        ):
            return False
        if node.is_Atom:
            if isinstance(node, (sp.Number, sp.Symbol)) or node in (sp.true, sp.false):
                continue
            return False
        if not isinstance(
            node,
            (*_ARITHMETIC_NODE_TYPES, *_BOOLEAN_NODE_TYPES, ExprCondPair),
        ):
            return False
    return True


def _project_domain_sources(
    constraints: Iterable[Any],
) -> tuple[_DomainSource, ...] | None:
    """Project audited resource constraints into deduplicated proof sources.

    Args:
        constraints (Iterable[Any]): Candidate resource constraints.

    Returns:
        tuple[_DomainSource, ...] | None: Eligible deduplicated sources, or None
        when a hard source budget or malformed eligible predicate fails closed.
    """
    projected: dict[Boolean, _ConsumedDomainRequirement] = {}
    raw_count = 0
    predicate_nodes = 0
    symbols: set[sp.Symbol] = set()
    for constraint in constraints:
        raw_count += 1
        if raw_count > _DOMAIN_RAW_CONSTRAINT_LIMIT:
            return None
        if not getattr(constraint, "domain_eligible", False):
            continue
        source_formals = tuple(getattr(constraint, "source_formals", ()))
        if not source_formals:
            continue
        if tuple(getattr(constraint, "ranges", ())):
            continue
        if getattr(constraint, "active_when", sp.false) is not sp.true:
            continue
        predicate_factory = getattr(constraint, "domain_predicate", None)
        if not callable(predicate_factory):
            return None
        predicate = predicate_factory()
        if not isinstance(predicate, Boolean) or predicate is sp.false:
            return None
        if predicate is sp.true:
            continue
        if not _predicate_is_supported(predicate):
            return None
        predicate_nodes += sum(1 for _ in sp.preorder_traversal(predicate))
        if predicate_nodes > _DOMAIN_PREDICATE_NODE_LIMIT:
            return None
        symbols.update(cast(set[sp.Symbol], predicate.free_symbols))
        if len(symbols) > _DOMAIN_SYMBOL_LIMIT:
            return None

        labels = (str(getattr(constraint, "label", "qkernel input")),)
        requirement = projected.get(predicate)
        if requirement is None:
            projected[predicate] = _ConsumedDomainRequirement(
                predicate=predicate,
                source_formals=tuple(dict.fromkeys(map(str, source_formals))),
                labels=labels,
            )
        else:
            projected[predicate] = dataclasses.replace(
                requirement,
                source_formals=tuple(
                    dict.fromkeys(
                        (*requirement.source_formals, *map(str, source_formals))
                    )
                ),
                labels=tuple(dict.fromkeys((*requirement.labels, *labels))),
            )
        if len(projected) > _DOMAIN_SOURCE_LIMIT:
            return None

    return tuple(
        _DomainSource(requirement, _flatten_conjunction(requirement.predicate))
        for requirement in projected.values()
    )


def _predicate_is_supported(predicate: Boolean) -> bool:
    """Return whether a domain predicate uses supported binder-free relations.

    Args:
        predicate (Boolean): Projected structural requirement predicate.

    Returns:
        bool: Whether the bounded relation prover can inspect the predicate.
    """
    for node in sp.preorder_traversal(predicate):
        if isinstance(node, (sp.Dummy, *_PROTECTED_NODE_TYPES, AppliedUndef)):
            return False
        if isinstance(node, sp.Function):
            return False
        if node.is_Atom:
            if isinstance(node, (sp.Number, sp.Symbol)) or node in (sp.true, sp.false):
                continue
            return False
        if isinstance(node, (*_BOOLEAN_NODE_TYPES, sp.Add, sp.Mul, sp.Pow)):
            continue
        return False
    return True


def _flatten_conjunction(predicate: Boolean) -> tuple[Boolean, ...]:
    """Flatten one conjunction into atomic proof facts.

    Args:
        predicate (Boolean): Source or branch predicate to flatten.

    Returns:
        tuple[Boolean, ...]: Ordered atomic facts.
    """
    if isinstance(predicate, sp.And):
        facts: list[Boolean] = []
        for argument in predicate.args:
            facts.extend(_flatten_conjunction(cast(Boolean, argument)))
        return tuple(facts)
    return (predicate,)


def _sources_are_contradictory(environment: _ProofEnvironment) -> bool:
    """Detect directly contradictory projected source facts.

    Args:
        environment (_ProofEnvironment): Domain proof environment.

    Returns:
        bool: Whether one atomic source fact proves another source fact false.
    """
    facts: list[tuple[Boolean, int]] = []
    for source_index, source in enumerate(environment.sources):
        facts.extend((fact, source_index) for fact in source.facts)
    for index, (fact, _) in enumerate(facts):
        other_facts = (*facts[:index], *facts[index + 1 :])
        evidence = _prove_from_facts(
            cast(Boolean, sp.Not(fact)),
            other_facts,
            environment,
        )
        if evidence is not None:
            return True
        if environment.exhausted:
            return True
    equalities = tuple(
        fact for fact, _source_index in facts if isinstance(fact, sp.Equality)
    )
    for index, left in enumerate(equalities):
        left_margin = _affine_expression(_relation_difference(left))
        if left_margin is None:
            continue
        for right in equalities[index + 1 :]:
            if not environment.spend_comparison():
                return True
            right_margin = _affine_expression(_relation_difference(right))
            if right_margin is None:
                continue
            if _equalities_are_inconsistent(left_margin, right_margin):
                return True
    return False


def _equalities_are_inconsistent(
    left_margin: ResourceExpr,
    right_margin: ResourceExpr,
) -> bool:
    """Return whether two affine zero margins have no common solution.

    Args:
        left_margin (ResourceExpr): First affine expression constrained to zero.
        right_margin (ResourceExpr): Second affine expression constrained to zero.

    Returns:
        bool: Whether proportional symbolic coefficients imply distinct
            constants, making the two equalities inconsistent.
    """
    left_form = _extract_affine_domain_expression(left_margin)
    right_form = _extract_affine_domain_expression(right_margin)
    if left_form is None or right_form is None:
        return False
    left_coefficients = dict(left_form.coefficients)
    right_coefficients = dict(right_form.coefficients)
    symbols = tuple(
        sorted(
            left_coefficients.keys() | right_coefficients.keys(),
            key=sp.default_sort_key,
        )
    )
    if not symbols:
        return not (
            _constant_is_zero(left_form.constant)
            and _constant_is_zero(right_form.constant)
        )
    scale: sp.Expr | None = None
    for symbol in symbols:
        left_coefficient = left_coefficients.get(symbol, sp.S.Zero)
        right_coefficient = right_coefficients.get(symbol, sp.S.Zero)
        left_is_zero = _constant_is_zero(left_coefficient)
        right_is_zero = _constant_is_zero(right_coefficient)
        if left_is_zero and right_is_zero:
            continue
        if left_is_zero or right_is_zero:
            return False
        if left_coefficient.is_zero is not False:
            return False
        candidate = right_coefficient / left_coefficient
        if scale is None:
            scale = candidate
        elif not _constant_is_zero(right_coefficient - scale * left_coefficient):
            return False
    if scale is None:
        return False
    constant_difference = right_form.constant - scale * left_form.constant
    return constant_difference.is_zero is False


def _rewrite_node(
    expression: ResourceExpr,
    context: tuple[Boolean, ...],
    environment: _ProofEnvironment,
) -> tuple[ResourceExpr, frozenset[int]]:
    """Rewrite one supported arithmetic node under a branch context.

    Args:
        expression (ResourceExpr): Current supported expression node.
        context (tuple[Boolean, ...]): Ordered Piecewise reachability facts.
        environment (_ProofEnvironment): Bounded proof state.

    Returns:
        tuple[ResourceExpr, frozenset[int]]: Rewritten expression and exact
        source indices consumed below this node.
    """
    if expression.is_Atom:
        return expression, frozenset()
    if isinstance(expression, sp.Piecewise):
        return _rewrite_piecewise(expression, context, environment)

    rewritten_arguments: list[sp.Basic] = []
    evidence: set[int] = set()
    for argument in expression.args:
        rewritten, used = _rewrite_node(
            cast(ResourceExpr, argument),
            context,
            environment,
        )
        rewritten_arguments.append(rewritten)
        evidence.update(used)
    if environment.exhausted:
        return expression, frozenset()

    if isinstance(expression, (sp.Max, sp.Min)):
        return _rewrite_extremum(
            expression,
            tuple(cast(ResourceExpr, value) for value in rewritten_arguments),
            context,
            environment,
            evidence,
        )
    rebuilt = cast(ResourceExpr, expression.func(*rewritten_arguments))
    return rebuilt, frozenset(evidence)


def _rewrite_extremum(
    original: sp.Max | sp.Min,
    arguments: tuple[ResourceExpr, ...],
    context: tuple[Boolean, ...],
    environment: _ProofEnvironment,
    inherited_evidence: set[int],
) -> tuple[ResourceExpr, frozenset[int]]:
    """Select a proven dominating candidate from an n-ary extremum.

    Args:
        original (sp.Max | sp.Min): Original extremum node.
        arguments (tuple[ResourceExpr, ...]): Recursively rewritten candidates.
        context (tuple[Boolean, ...]): Active Piecewise branch facts.
        environment (_ProofEnvironment): Bounded proof state.
        inherited_evidence (set[int]): Sources consumed by child rewrites.

    Returns:
        tuple[ResourceExpr, frozenset[int]]: Selected candidate when proven,
        otherwise a rebuilt extremum, plus all consumed source indices.
    """
    for candidate in arguments:
        candidate_evidence: set[int] = set()
        for other in arguments:
            if candidate == other:
                continue
            comparison = (
                sp.Ge(candidate, other)
                if isinstance(original, sp.Max)
                else sp.Le(candidate, other)
            )
            proof = _prove_condition(cast(Boolean, comparison), context, environment)
            if proof is None:
                break
            candidate_evidence.update(proof)
        else:
            return candidate, frozenset((*inherited_evidence, *candidate_evidence))
        if environment.exhausted:
            return cast(ResourceExpr, original), frozenset()
    rebuilt = cast(ResourceExpr, original.func(*arguments, evaluate=False))
    return rebuilt, frozenset(inherited_evidence)


def _rewrite_piecewise(
    expression: sp.Piecewise,
    context: tuple[Boolean, ...],
    environment: _ProofEnvironment,
) -> tuple[ResourceExpr, frozenset[int]]:
    """Rewrite an ordered Piecewise without conflating guard and value context.

    Args:
        expression (sp.Piecewise): Ordered conditional resource expression.
        context (tuple[Boolean, ...]): Reachability facts from outer branches.
        environment (_ProofEnvironment): Bounded proof state.

    Returns:
        tuple[ResourceExpr, frozenset[int]]: Rewritten ordered expression and
        exact source indices consumed by branch or value rewrites.
    """
    branches: list[tuple[ResourceExpr, Boolean]] = []
    prior_false: list[Boolean] = []
    evidence: set[int] = set()
    for pair in cast(tuple[ExprCondPair, ...], expression.args):
        raw_value, raw_condition = pair.args
        condition = cast(Boolean, raw_condition)
        branch_context = (*context, *prior_false)
        proof_true = _prove_condition(condition, branch_context, environment)
        if proof_true is not None:
            rewritten_value, value_evidence = _rewrite_node(
                cast(ResourceExpr, raw_value),
                (*branch_context, condition),
                environment,
            )
            evidence.update(proof_true)
            evidence.update(value_evidence)
            branches.append((rewritten_value, sp.true))
            break

        proof_false = _prove_condition(
            cast(Boolean, sp.Not(condition)),
            branch_context,
            environment,
        )
        if proof_false is not None:
            evidence.update(proof_false)
            prior_false.append(cast(Boolean, sp.Not(condition)))
            continue

        rewritten_value, value_evidence = _rewrite_node(
            cast(ResourceExpr, raw_value),
            (*branch_context, condition),
            environment,
        )
        evidence.update(value_evidence)
        branches.append((rewritten_value, condition))
        prior_false.append(cast(Boolean, sp.Not(condition)))
        if environment.exhausted:
            return cast(ResourceExpr, expression), frozenset()

    if not branches:
        return cast(ResourceExpr, expression), frozenset()
    rebuilt = cast(ResourceExpr, sp.Piecewise(*branches, evaluate=False))
    return rebuilt, frozenset(evidence)


def _prove_condition(
    condition: Boolean,
    context: tuple[Boolean, ...],
    environment: _ProofEnvironment,
) -> tuple[int, ...] | None:
    """Prove one Boolean condition from branch facts and projected sources.

    Args:
        condition (Boolean): Target predicate to prove true.
        context (tuple[Boolean, ...]): Assumed branch reachability facts.
        environment (_ProofEnvironment): Bounded proof state and cache.

    Returns:
        tuple[int, ...] | None: Supporting domain-source indices, an empty tuple
        for a context/type proof, or None when the condition is unresolved.
    """
    if condition is sp.true:
        return ()
    if condition is sp.false or environment.exhausted:
        return None
    key = (condition, context)
    if key in environment.cache:
        return environment.cache[key]

    contextual_facts: list[tuple[Boolean, int | None]] = []
    for predicate in context:
        contextual_facts.extend(
            (fact, None) for fact in _flatten_conjunction(predicate)
        )
    for source_index, source in enumerate(environment.sources):
        contextual_facts.extend((fact, source_index) for fact in source.facts)
    result = _prove_from_facts(condition, tuple(contextual_facts), environment)
    environment.remember(key, result)
    return result


def _prove_from_facts(
    condition: Boolean,
    facts: tuple[tuple[Boolean, int | None], ...],
    environment: _ProofEnvironment,
) -> tuple[int, ...] | None:
    """Prove one target from explicit atomic facts without ex-falso reasoning.

    Args:
        condition (Boolean): Target predicate.
        facts (tuple[tuple[Boolean, int | None], ...]): Atomic facts paired with
            their optional projected-source index.
        environment (_ProofEnvironment): Remaining bounded comparison budget.

    Returns:
        tuple[int, ...] | None: Exact supporting source indices, or None when
        the target is not proven.
    """
    if condition is sp.true:
        return ()
    if condition is sp.false:
        return None
    if isinstance(condition, sp.And):
        evidence: set[int] = set()
        for argument in condition.args:
            proof = _prove_from_facts(cast(Boolean, argument), facts, environment)
            if proof is None:
                return None
            evidence.update(proof)
        return tuple(sorted(evidence))
    if isinstance(condition, sp.Or):
        for argument in condition.args:
            proof = _prove_from_facts(cast(Boolean, argument), facts, environment)
            if proof is not None:
                return proof
        return None
    if isinstance(condition, sp.Not):
        argument = cast(Boolean, condition.args[0])
        if isinstance(argument, Relational):
            complemented = cast(Boolean, sp.Not(argument))
            if complemented != condition:
                return _prove_from_facts(complemented, facts, environment)

    for fact, source_index in facts:
        if condition == fact:
            return () if source_index is None else (source_index,)
        if not environment.spend_comparison():
            return None
        if isinstance(condition, Relational) and isinstance(fact, Relational):
            if _relation_implies(fact, condition):
                return () if source_index is None else (source_index,)

    if isinstance(condition, Relational) and _relation_is_type_proven(condition):
        return ()
    return None


def _relation_implies(source: Relational, target: Relational) -> bool:
    """Return whether one supported affine relation implies another.

    Args:
        source (Relational): One assumed affine relation.
        target (Relational): Affine relation to prove.

    Returns:
        bool: Whether the source margin dominates the target margin exactly.
    """
    if source == target:
        return True
    if isinstance(target, sp.Equality):
        if not isinstance(source, sp.Equality):
            return False
        source_margin = _affine_expression(_relation_difference(source))
        target_margin = _affine_expression(_relation_difference(target))
        return (
            source_margin is not None
            and target_margin is not None
            and (
                _provably_zero(target_margin - source_margin)
                or _provably_zero(target_margin + source_margin)
            )
        )
    if isinstance(target, sp.Unequality):
        difference = _affine_expression(_relation_difference(target))
        if difference is None:
            return False
        positive = _lower_bound_margin(sp.Gt(difference, 0))
        negative = _lower_bound_margin(sp.Gt(-difference, 0))
        source_bound = _lower_bound_margin(source)
        return source_bound is not None and any(
            bound is not None and _margin_dominates(source_bound, bound)
            for bound in (positive, negative)
        )
    source_bound = _lower_bound_margin(source)
    target_bound = _lower_bound_margin(target)
    return (
        source_bound is not None
        and target_bound is not None
        and _margin_dominates(source_bound, target_bound)
    )


def _lower_bound_margin(relation: Relational) -> ResourceExpr | None:
    """Normalize one affine ordering relation to an inclusive zero margin.

    Args:
        relation (Relational): Equality or ordering relation to normalize.

    Returns:
        ResourceExpr | None: Affine expression known to be nonnegative, or None
        when strictness cannot safely be shifted over proven integers.
    """
    if isinstance(relation, sp.GreaterThan):
        margin = _relation_difference(relation)
    elif isinstance(relation, sp.LessThan):
        margin = -_relation_difference(relation)
    elif isinstance(relation, sp.StrictGreaterThan):
        raw = _relation_difference(relation)
        if raw.is_integer is not True:
            return None
        margin = raw - 1
    elif isinstance(relation, sp.StrictLessThan):
        raw = -_relation_difference(relation)
        if raw.is_integer is not True:
            return None
        margin = raw - 1
    else:
        return None
    return _affine_expression(margin)


def _margin_dominates(
    source_margin: ResourceExpr,
    target_margin: ResourceExpr,
) -> bool:
    """Return whether nonnegativity of one margin proves another margin.

    Args:
        source_margin (ResourceExpr): Affine margin assumed nonnegative.
        target_margin (ResourceExpr): Affine margin requested nonnegative.

    Returns:
        bool: Whether their exact difference is provably nonnegative.
    """
    return _provably_nonnegative(target_margin - source_margin)


def _relation_difference(relation: Relational) -> ResourceExpr:
    """Return the typed left-minus-right difference of one relation.

    Args:
        relation (Relational): Relation whose scalar sides are subtracted.

    Returns:
        ResourceExpr: Exact left-minus-right arithmetic expression.
    """
    left = cast(ResourceExpr, relation.lhs)
    right = cast(ResourceExpr, relation.rhs)
    return cast(ResourceExpr, left - right)


def _affine_expression(expression: sp.Expr) -> ResourceExpr | None:
    """Normalize a bounded expression only when it is affine in free symbols.

    Args:
        expression (sp.Expr): Candidate relation margin.

    Returns:
        ResourceExpr | None: Canonical affine expression, or None for nonlinear
        or unsupported expressions.
    """
    if not _expression_is_supported(expression):
        return None
    affine = _extract_affine_domain_expression(expression)
    if affine is None:
        return None
    return affine.as_expression()


def _relation_is_type_proven(relation: Relational) -> bool:
    """Return whether SymPy type facts alone settle one relation.

    Args:
        relation (Relational): Relation to inspect without domain sources.

    Returns:
        bool: Whether expression facts prove the requested relation.
    """
    if isinstance(relation, sp.Equality):
        return _provably_zero(_relation_difference(relation))
    if isinstance(relation, sp.Unequality):
        difference = _relation_difference(relation)
        return difference.is_zero is False
    margin = _lower_bound_margin(relation)
    return margin is not None and _provably_nonnegative(margin)


def _provably_nonnegative(expression: sp.Expr) -> bool:
    """Return whether bounded local simplification proves a nonnegative sign.

    Args:
        expression (sp.Expr): Exact arithmetic difference to inspect.

    Returns:
        bool: Whether SymPy's local facts prove the expression nonnegative.
    """
    affine = _extract_affine_domain_expression(expression)
    return affine is not None and affine.as_expression().is_nonnegative is True


def _provably_zero(expression: sp.Expr) -> bool:
    """Return whether bounded local simplification proves exact zero.

    Args:
        expression (sp.Expr): Exact arithmetic difference to inspect.

    Returns:
        bool: Whether bounded affine normalization proves exact zero.
    """
    if expression == 0:
        return True
    affine = _extract_affine_domain_expression(expression)
    if affine is None or not _constant_is_zero(affine.constant):
        return False
    return all(
        _constant_is_zero(coefficient) for _symbol, coefficient in affine.coefficients
    )
