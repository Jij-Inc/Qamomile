"""Hold mutable state for IR resource interpretation."""

from __future__ import annotations

import itertools
import math
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._array_state import _ArrayReferenceState
from qamomile.circuit.estimator._classical_facts import (
    _fact_from_expression,
    _ResolvedClassicalFact,
)
from qamomile.circuit.estimator._classical_provenance import (
    _OBSERVATION_SOURCE_PREFIX,
    _direct_observation_result_uuids,
    _operation_classical_dependency_inputs,
    _resolved_classical_facts,
    _structural_value_ancestry,
    _value_taint_condition,
)
from qamomile.circuit.estimator._config import (
    _ResourceEstimatorConfig,
)
from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._interpreter_contract import _InterpreterContract
from qamomile.circuit.estimator._measurement_provenance import (
    _merge_measurement_taint_conditions,
)
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resolver_indices import _resolve_concrete_array_payload
from qamomile.circuit.estimator._resource_base import (
    _is_concrete_integer,
    _symbol_display_name,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ResourceConstraint,
)
from qamomile.circuit.estimator._resource_expressions import (
    _and_conditions,
)
from qamomile.circuit.estimator._scopes import _typed_value_symbol
from qamomile.circuit.estimator._shot_context import _ShotContext
from qamomile.circuit.estimator._symbolic import _normalize_resource_scalar
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, UnaryMathOp
from qamomile.circuit.ir.operation.callable import (
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
    WhileOperation,
)
from qamomile.circuit.ir.operation.operation import (
    Operation,
)
from qamomile.circuit.ir.types.primitives import (
    BitType,
    UIntType,
)
from qamomile.circuit.ir.value import (
    Value,
)

_MAX_RUNTIME_DOMAIN_ALTERNATIVES = 16


@dataclass(slots=True)
class _InterpreterRunState:
    """Own mutable state for one resource interpretation run.

    Args:
        condition_values (Mapping[str, sp.Expr]): Numeric scalar input values
            available to specialize the current root block.
    """

    condition_values: Mapping[str, sp.Expr]
    branch_condition_names: set[str] = field(default_factory=set)
    reported_undecidable: set[str] = field(default_factory=set)
    measurement_taint_conditions: dict[str, Boolean] = field(default_factory=dict)
    observation_occurrence_path: tuple[tuple[str, int, int], ...] = ()
    runtime_observation_symbols: set[sp.Symbol] = field(default_factory=set)
    runtime_value_domains: dict[
        sp.Symbol,
        tuple[sp.Expr, ...] | None,
    ] = field(default_factory=dict)
    unresolved_resource_symbols: set[sp.Symbol] = field(default_factory=set)
    shot_context: _ShotContext = field(default_factory=_ShotContext)
    constraint_scope_condition: Boolean = sp.true
    array_constraint_proven: dict[_ResourceConstraint, bool] = field(
        default_factory=dict
    )
    runtime_observation_cache: dict[
        int,
        tuple[Block, frozenset[int], bool],
    ] = field(default_factory=dict)
    allocation_owners_by_uuid: dict[str, str] = field(default_factory=dict)
    dependency_owner_aliases: dict[str, frozenset[str]] = field(default_factory=dict)
    active_call_states: dict[int, list[tuple[sp.Expr, ...]]] = field(
        default_factory=dict
    )
    active_batch_profile_states: dict[int, list[tuple[sp.Expr, ...]]] = field(
        default_factory=dict
    )
    opaque_definition_cost_cache: dict[
        tuple[Any, ...],
        tuple[InvokeOperation, ResourceEstimate],
    ] = field(default_factory=dict)
    while_trip_count_names: dict[
        tuple[tuple[tuple[int, int], ...], int],
        tuple[WhileOperation, str],
    ] = field(default_factory=dict)
    while_name_scan_operations: dict[
        tuple[tuple[tuple[int, int], ...], int],
        Operation,
    ] = field(default_factory=dict)
    loop_validation_pruned_blocks: dict[int, Block] = field(default_factory=dict)


class _InterpreterState(_InterpreterContract):
    """Own mutable facts and scoped state shared by interpretation."""

    config: _ResourceEstimatorConfig
    bindings: Mapping[str, Any]

    def __init__(
        self,
        *,
        config: _ResourceEstimatorConfig,
        bindings: Mapping[str, Any],
        condition_values: Mapping[str, sp.Expr] | None = None,
        shot_bindings: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize an interpreter.

        Args:
            config (_ResourceEstimatorConfig): Estimator configuration.
            bindings (Mapping[str, Any]): Concrete user bindings.
            condition_values (Mapping[str, sp.Expr] | None): Numeric scalar
                input values used to decide compile-time ``if`` branches and
                resolve physical dependency indices. Defaults to ``None``.
            shot_bindings (Mapping[str, Any] | None): Explicit array inputs
                available only for shot-dependent scalar projection. Defaults
                to ``None``.
        """
        self.config = config
        self.bindings = bindings
        self._initial_condition_values = condition_values or {}
        self._shot_bindings = shot_bindings or {}
        # Keep an ad-hoc state available for low-level handler tests that call
        # services directly without entering ``estimate``.
        self._run_state = self._new_run_state()

    def _new_run_state(self) -> _InterpreterRunState:
        """Create fresh mutable state for one interpretation.

        Returns:
            _InterpreterRunState: State initialized from interpreter-lifetime
            scalar condition values.
        """
        state = _InterpreterRunState(
            condition_values=self._initial_condition_values,
        )
        state.unresolved_resource_symbols.add(state.shot_context.current)
        state.runtime_value_domains[state.shot_context.current] = None
        return state

    @property
    def condition_values(self) -> Mapping[str, sp.Expr]:
        """Return scalar condition values active for the current run.

        Returns:
            Mapping[str, sp.Expr]: Current scalar condition values.
        """
        return self._run_state.condition_values

    @property
    def branch_condition_names(self) -> set[str]:
        """Return classical input names consumed by the current run.

        Returns:
            set[str]: Consumed classical input names.
        """
        return self._run_state.branch_condition_names

    @property
    def _runtime_observation_cache(
        self,
    ) -> dict[int, tuple[Block, frozenset[int], bool]]:
        """Return callable observation summaries for the current run.

        Returns:
            dict[int, tuple[Block, frozenset[int], bool]]: Observation cache.
        """
        return self._run_state.runtime_observation_cache

    def _apply_condition_values(
        self,
        expression: sp.Expr,
        *,
        record_usage: bool = True,
    ) -> sp.Expr:
        """Substitute supplied scalar values into an expression.

        Args:
            expression (sp.Expr): Expression to specialize.
            record_usage (bool): Whether matching input names should be marked
                as consumed by a scheduling or branch decision. Defaults to
                ``True``.

        Returns:
            sp.Expr: Specialized expression.
        """
        condition_inputs = {
            symbol: self._run_state.condition_values[_symbol_display_name(symbol)]
            for symbol in expression.free_symbols
            if not isinstance(symbol, sp.Dummy)
            and _symbol_display_name(symbol) in self._run_state.condition_values
        }
        if not condition_inputs:
            return expression
        if record_usage:
            self._run_state.branch_condition_names.update(
                _symbol_display_name(symbol) for symbol in condition_inputs
            )
        return cast(
            sp.Expr,
            expression.subs(list(condition_inputs.items()), simultaneous=True).doit(),
        )

    @staticmethod
    def _concrete_scalar(expression: sp.Expr) -> int | None:
        """Resolve an integer expression after any requested specialization.

        Args:
            expression (sp.Expr): Symbolic scalar expression.

        Returns:
            int | None: Concrete integer, or ``None`` when unresolved.
        """
        if expression.is_number and _is_concrete_integer(expression):
            return int(expression)
        return None

    @contextmanager
    def _guarded_constraint_scope(
        self,
        active_when: sp.Basic,
    ) -> Generator[None, None, None]:
        """Conjoin one activation guard while nested constraints are built.

        Args:
            active_when (sp.Basic): Predicate under which the nested scope can
                execute.

        Yields:
            None: Control returns to the caller while the guard is active.
        """
        previous = self._run_state.constraint_scope_condition
        self._run_state.constraint_scope_condition = _and_conditions(
            previous, active_when
        )
        try:
            yield
        finally:
            self._run_state.constraint_scope_condition = previous

    @contextmanager
    def _measurement_taint_scope(
        self,
        additions: Mapping[str, Boolean],
    ) -> Generator[None, None, None]:
        """Expose temporary guarded observation provenance in a nested scope.

        Args:
            additions (Mapping[str, Boolean]): UUID-keyed provenance to merge
                for the nested evaluation.

        Yields:
            None: Control returns while the temporary mapping is visible.
        """
        previous = self._run_state.measurement_taint_conditions
        self._run_state.measurement_taint_conditions = (
            _merge_measurement_taint_conditions(
                previous,
                additions,
            )
        )
        try:
            yield
        finally:
            self._run_state.measurement_taint_conditions = previous

    @contextmanager
    def _observation_occurrence_scope(
        self,
        kind: str,
        operation: Operation,
        ordinal: int,
    ) -> Generator[None, None, None]:
        """Qualify direct observation tokens within one repeated body visit.

        Args:
            kind (str): Stable loop-family label such as ``"range"``.
            operation (Operation): Repeated operation owning the body.
            ordinal (int): Concrete visit ordinal, or ``-1`` for one symbolic
                occurrence family.

        Yields:
            None: Control returns while newly published observation tokens are
            qualified by this occurrence path.
        """
        previous = self._run_state.observation_occurrence_path
        self._run_state.observation_occurrence_path = (
            *previous,
            (kind, id(operation), ordinal),
        )
        try:
            yield
        finally:
            self._run_state.observation_occurrence_path = previous

    def _observation_source_token(
        self,
        operation: Operation,
        result_uuid: str,
        result_slot: int,
        resolver: ExprResolver,
    ) -> str:
        """Return one identity-qualified scheduler token for an observation.

        Args:
            operation (Operation): Direct observation operation.
            result_uuid (str): UUID of the observed classical result.
            result_slot (int): Position in the operation's direct results.
            resolver (ExprResolver): Resolver carrying the callable path.

        Returns:
            str: Token unique to the call path, repeated occurrence, and slot.
        """
        call_scope = "/".join(
            f"{call_id:x}:{body_id:x}" for call_id, body_id in resolver.structural_scope
        )
        occurrence = "/".join(
            f"{kind}:{owner_id:x}:{ordinal}"
            for kind, owner_id, ordinal in self._run_state.observation_occurrence_path
        )
        scope = "/".join(part for part in (call_scope, occurrence) if part)
        identity = (
            f"{result_uuid}@{scope}#{result_slot}"
            if scope
            else f"{result_uuid}#{result_slot}"
        )
        return f"{_OBSERVATION_SOURCE_PREFIX}:{identity}"

    @contextmanager
    def _isolated_loop_taint_probe_state(self) -> Generator[None, None, None]:
        """Prevent discarded loop probes from mutating interpretation results.

        Definition-cost and observation-summary caches remain shared because
        they are identity-keyed pure memoization and prevent user callbacks from
        being executed repeatedly. Runtime-observation symbol identities also
        remain shared because the probe resolver expressions are reused by the
        recurrence analysis after this scope exits. Reporting, constraint, and
        owner state is restored so extra provenance probes cannot suppress
        assumptions, claim input usage, or change final scheduling metadata.

        Yields:
            None: Control returns while disposable probe mutations are isolated.
        """
        branch_condition_names = set(self._run_state.branch_condition_names)
        reported_undecidable = set(self._run_state.reported_undecidable)
        constraint_scope_condition = self._run_state.constraint_scope_condition
        array_constraint_proven = dict(self._run_state.array_constraint_proven)
        allocation_owners_by_uuid = dict(self._run_state.allocation_owners_by_uuid)
        dependency_owner_aliases = dict(self._run_state.dependency_owner_aliases)
        try:
            yield
        finally:
            self._run_state.branch_condition_names.clear()
            self._run_state.branch_condition_names.update(branch_condition_names)
            self._run_state.reported_undecidable.clear()
            self._run_state.reported_undecidable.update(reported_undecidable)
            self._run_state.constraint_scope_condition = constraint_scope_condition
            self._run_state.array_constraint_proven.clear()
            self._run_state.array_constraint_proven.update(array_constraint_proven)
            self._run_state.allocation_owners_by_uuid.clear()
            self._run_state.allocation_owners_by_uuid.update(allocation_owners_by_uuid)
            self._run_state.dependency_owner_aliases.clear()
            self._run_state.dependency_owner_aliases.update(dependency_owner_aliases)

    def _record_runtime_value_symbols(
        self,
        operation: Operation,
        resolver: ExprResolver,
    ) -> None:
        """Classify resolver fallbacks for observation-derived IR values.

        Ordinary expressions can mix runtime state with public inputs, for
        example ``measured_choice + n``.  Their free symbols must therefore
        remain classified independently.  Only the resolver-owned fallback
        for a value that cannot otherwise be expressed is newly internalized.
        This covers direct observations and array contents selected through a
        runtime-derived index or view without mistaking ``n`` for an outcome.

        Args:
            operation (Operation): Recently evaluated operation and its inputs.
            resolver (ExprResolver): Resolver that names scalar values.
        """
        candidates = (
            value
            for root in (*operation.all_input_values(), *operation.results)
            for value in _structural_value_ancestry(root)
            if isinstance(value, Value) and not value.type.is_quantum()
        )
        for value in candidates:
            if (
                _value_taint_condition(
                    value,
                    self._run_state.measurement_taint_conditions,
                )
                is sp.false
            ):
                continue
            symbol = resolver.unresolved_fallback_symbol(value)
            if symbol is None:
                continue
            self._run_state.runtime_observation_symbols.add(symbol)
            if symbol in self._run_state.runtime_value_domains:
                continue
            self._run_state.runtime_value_domains[symbol] = (
                (_ZERO, _ONE) if isinstance(value.type, BitType) else None
            )

    def _publish_operation_classical_facts(
        self,
        operation: Operation,
        resolver: ExprResolver,
        input_sources: Mapping[str, Boolean],
    ) -> None:
        """Publish one operation's classical values into the shared fact model.

        Direct observations create readiness-source tokens. Ordinary
        classical results retain the guarded sources of their semantic inputs.
        Nested operations publish their own boundary results because their
        branch, loop, or callable structure determines more precise facts.

        Args:
            operation (Operation): Operation whose results were just evaluated.
            resolver (ExprResolver): Resolver updated for subsequent work.
            input_sources (Mapping[str, Boolean]): Observation sources consumed
                by the operation before it was evaluated.
        """
        direct_sources = _direct_observation_result_uuids(operation)
        if direct_sources:
            direct_slots = {
                result_uuid: slot for slot, result_uuid in enumerate(direct_sources)
            }
            for result in operation.results:
                if not isinstance(result, Value) or result.uuid not in direct_slots:
                    continue
                source_token = self._observation_source_token(
                    operation,
                    result.uuid,
                    direct_slots[result.uuid],
                    resolver,
                )
                resolver.bind_classical_fact(
                    result,
                    _ResolvedClassicalFact.create(
                        resolver.resolve(result),
                        {source_token: sp.true},
                    ),
                )
            return
        if isinstance(operation, (HasNestedOps, InvokeOperation)):
            return
        source_facts = _resolved_classical_facts(
            _operation_classical_dependency_inputs(operation),
            resolver,
        )
        for result in operation.results:
            if not isinstance(result, Value) or result.type.is_quantum():
                continue
            resolved = resolver.resolve(result)
            if resolver.unresolved_fallback_symbol(result) is not None:
                fact = _ResolvedClassicalFact.create(resolved, input_sources)
            else:
                fact = _fact_from_expression(resolved, *source_facts)
            resolver.bind_classical_fact(
                result,
                fact,
            )

    def _require_nonnegative_shot_uint_candidates(
        self,
        expression: sp.Expr,
    ) -> None:
        """Reject finite context values outside the supported UInt domain.

        Args:
            expression (sp.Expr): Context-derived UInt value to check before
                further arithmetic or resource use.

        Raises:
            NotImplementedError: If a candidate is negative or its
                nonnegativity cannot be proved within the bounded analysis.
        """
        normalized = self._run_state.shot_context.normalize_remainders(
            expression,
            self._run_state.runtime_value_domains,
            limit=_MAX_RUNTIME_DOMAIN_ALTERNATIVES,
        )
        alternatives = self._finite_runtime_alternatives(normalized)
        if alternatives is None:
            alternatives = (normalized,)
        if any(
            self._apply_condition_values(candidate, record_usage=False).is_nonnegative
            is not True
            for candidate in alternatives
        ):
            raise NotImplementedError(
                "Resource estimation cannot bound shot-dependent UInt arithmetic "
                "that may become negative."
            )

    def _prepare_shot_values(
        self,
        operation: Operation,
        resolver: ExprResolver,
    ) -> None:
        """Resolve bounded context arithmetic and selected scalar array values.

        Args:
            operation (Operation): Operation about to consume classical data.
            resolver (ExprResolver): Resolver for the active scope.
        """
        shot = self._run_state.shot_context
        if not shot.used:
            return
        visited: set[str] = set()

        def visit(value: Value) -> None:
            """Prepare indices before their enclosing scalar value.

            Args:
                value (Value): Scalar whose symbolic expression is prepared.
            """
            if value.uuid in visited or value.type.is_quantum():
                return
            visited.add(value.uuid)
            for index in value.element_indices:
                visit(index)
            fact = resolver.resolve_classical_fact(value)
            expression = cast(sp.Expr, fact.value)
            normalized = shot.normalize_remainders(
                expression,
                self._run_state.runtime_value_domains,
                limit=_MAX_RUNTIME_DOMAIN_ALTERNATIVES,
            )
            if isinstance(value.type, UIntType) and shot.contains(normalized):
                self._require_nonnegative_shot_uint_candidates(normalized)
            if normalized != expression:
                self._run_state.unresolved_resource_symbols.update(
                    symbol
                    for symbol in shot.derived
                    if symbol in normalized.free_symbols
                )
                resolver.bind_classical_fact(
                    value,
                    _ResolvedClassicalFact.create(normalized, fact.dependencies),
                )
            if not value.is_array_element() or not any(
                shot.contains(resolver.resolve(index))
                for index in value.element_indices
            ):
                return
            if expression.free_symbols & shot.derived:
                return
            if resolver._resolve_array_element(value, concrete=False) is not None:
                return
            alternatives = [
                self._finite_runtime_alternatives(resolver.resolve(index))
                for index in value.element_indices
            ]
            selected: list[sp.Expr] = []
            parent = value.parent_array
            assert parent is not None
            state = resolver.snapshot_array_state(parent)
            payload = (
                _resolve_concrete_array_payload(
                    state.array,
                    self._shot_bindings,
                    resolve_expression=resolver.resolve,
                    specialize=lambda expression: self._apply_condition_values(
                        expression, record_usage=False
                    ),
                    source="shot-dependent array",
                )
                if isinstance(state, _ArrayReferenceState)
                else None
            )
            if (
                all(domain is not None for domain in alternatives)
                and math.prod(
                    len(domain) for domain in alternatives if domain is not None
                )
                <= _MAX_RUNTIME_DOMAIN_ALTERNATIVES
            ):
                for indices in itertools.product(
                    *(domain for domain in alternatives if domain is not None)
                ):
                    child = resolver.child_scope(None)
                    for index, candidate in zip(
                        value.element_indices, indices, strict=True
                    ):
                        child.bind_classical_fact(
                            index,
                            _ResolvedClassicalFact.create(candidate),
                        )
                    projected = child._resolve_array_element(value, concrete=False)
                    if (
                        projected is None
                        and payload is not None
                        and all(
                            isinstance(candidate, sp.Integer) for candidate in indices
                        )
                    ):
                        item = payload
                        try:
                            for candidate in indices:
                                if int(candidate) < 0:
                                    raise IndexError("negative array index")
                                item = item[int(candidate)]
                        except (IndexError, KeyError, TypeError) as error:
                            raise ValueError(
                                "Shot-dependent array index is out of bounds."
                            ) from error
                        projected = _normalize_resource_scalar(
                            item,
                            label="shot-dependent array element",
                            allow_symbolic=False,
                            allow_bool=True,
                        )
                    if projected is None:
                        selected.clear()
                        break
                    if projected not in selected:
                        selected.append(projected)
            symbol = _typed_value_symbol(value, "shot_array_value", fresh=True)
            shot.derived.add(symbol)
            self._run_state.unresolved_resource_symbols.add(symbol)
            self._run_state.runtime_value_domains[symbol] = (
                tuple(selected) if selected else None
            )
            if isinstance(value.type, UIntType):
                self._require_nonnegative_shot_uint_candidates(symbol)
            resolver.bind_classical_fact(
                value,
                _ResolvedClassicalFact.create(symbol, fact.dependencies),
            )

        for root in operation.all_input_values():
            for value in _structural_value_ancestry(root):
                if isinstance(value, Value):
                    visit(value)

    def _record_shot_results(
        self,
        operation: Operation,
        resolver: ExprResolver,
    ) -> None:
        """Keep unresolved context-derived results private across boundaries.

        Args:
            operation (Operation): Recently evaluated operation.
            resolver (ExprResolver): Resolver carrying its result expressions.
        """
        shot = self._run_state.shot_context
        if not shot.used:
            return
        has_shot_inputs = any(
            isinstance(value, Value)
            and not value.type.is_quantum()
            and shot.contains(resolver.resolve(value))
            for root in operation.all_input_values()
            for value in _structural_value_ancestry(root)
        )
        if (
            has_shot_inputs
            and isinstance(operation, (BinOp, UnaryMathOp))
            and any(isinstance(result.type, UIntType) for result in operation.results)
        ):
            shot.used_uint_arithmetic = True
        for result in operation.results:
            if not isinstance(result, Value) or result.type.is_quantum():
                continue
            expression = resolver.resolve(result)
            if isinstance(result.type, UIntType) and (
                has_shot_inputs or expression.is_negative is True
            ):
                self._require_nonnegative_shot_uint_candidates(expression)
            if not has_shot_inputs:
                continue
            fallback = resolver.unresolved_fallback_symbol(result)
            if fallback is not None:
                self._run_state.unresolved_resource_symbols.add(fallback)
            shot.derived.update(
                symbol
                for symbol in self._run_state.unresolved_resource_symbols
                if symbol in expression.free_symbols
            )

    def _finite_runtime_alternatives(
        self,
        expression: sp.Expr,
    ) -> tuple[sp.Expr, ...] | None:
        """Expand an expression over registered finite runtime domains.

        Args:
            expression (sp.Expr): Runtime or compile-time scalar expression.

        Returns:
            tuple[sp.Expr, ...] | None: Structural alternatives, or ``None``
                when a participating runtime value is unbounded or the fixed
                expansion limit would be exceeded.
        """
        runtime_symbols = sorted(
            (
                symbol
                for symbol in expression.free_symbols
                if isinstance(symbol, sp.Symbol)
                and symbol in self._run_state.runtime_value_domains
            ),
            key=sp.default_sort_key,
        )
        if not runtime_symbols:
            return (expression,)
        domains = [
            self._run_state.runtime_value_domains[symbol] for symbol in runtime_symbols
        ]
        if any(domain is None for domain in domains):
            return None
        finite_domains = cast(list[tuple[sp.Expr, ...]], domains)
        alternative_count = math.prod(len(domain) for domain in finite_domains)
        if alternative_count > _MAX_RUNTIME_DOMAIN_ALTERNATIVES:
            return None
        alternatives: list[sp.Expr] = []
        for values in itertools.product(*finite_domains):
            replacement: dict[sp.Symbol, sp.Expr] = dict(
                zip(runtime_symbols, values, strict=True)
            )
            candidate = cast(
                sp.Expr,
                expression.subs(list(replacement.items()), simultaneous=True),
            )
            if candidate not in alternatives:
                alternatives.append(candidate)
        return tuple(alternatives)

    def _register_runtime_value_domain(
        self,
        symbol: sp.Symbol,
        *sources: sp.Expr,
    ) -> tuple[sp.Expr, ...] | None:
        """Register the finite union of runtime branch-source alternatives.

        Args:
            symbol (sp.Symbol): Fresh internal runtime merge symbol.
            *sources (sp.Expr): Values selected by the runtime branches.

        Returns:
            tuple[sp.Expr, ...] | None: Registered finite alternatives, or
                ``None`` when the runtime domain cannot be bounded safely.
        """
        alternatives: list[sp.Expr] = []
        for source in sources:
            expanded = self._finite_runtime_alternatives(source)
            if expanded is None:
                self._run_state.runtime_value_domains[symbol] = None
                return None
            for candidate in expanded:
                if candidate not in alternatives:
                    alternatives.append(candidate)
            if len(alternatives) > _MAX_RUNTIME_DOMAIN_ALTERNATIVES:
                self._run_state.runtime_value_domains[symbol] = None
                return None
        domain = tuple(alternatives)
        self._run_state.runtime_value_domains[symbol] = domain
        return domain
