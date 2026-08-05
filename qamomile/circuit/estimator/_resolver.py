"""Unified value resolution for resource estimation.

ExprResolver converts IR Values to SymPy expressions, providing a single
source of truth for all estimators (gate counting, qubits).

Two-mode API:
  resolve()          — symbolic; unbound parameters → sp.Symbol
  resolve_concrete() — concrete; must return int, raises on symbolic
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._metrics import _activation_over_range
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.dataflow import walk_operations
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
    UnaryMathOp,
    UnaryMathOpKind,
)
from qamomile.circuit.ir.operation.callable import CallTransform
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands

from ._utils import BINOP_TO_SYMPY, UNARY_MATH_TO_SYMPY


@dataclasses.dataclass(frozen=True)
class _ResolvedClassicalFact:
    """Carry one resolved value and guarded scheduler dependencies.

    Dependencies are stored as a deterministic immutable tuple so facts can be
    compared structurally. Each pair maps one source token to the condition
    under which the resolved value depends on that source. Most tokens denote
    runtime observations; loop-array summaries also use internal readiness
    tokens that must not be mistaken for measurement feed-forward.

    Args:
        value (sp.Basic): Resolved symbolic or concrete classical value.
        source_guards (tuple[tuple[str, Boolean], ...]): Sorted source-token
            guards. False guards are omitted.
    """

    value: sp.Basic
    source_guards: tuple[tuple[str, Boolean], ...] = ()

    @classmethod
    def create(
        cls,
        value: sp.Basic | int | float | bool,
        source_guards: Mapping[str, sp.Basic] | None = None,
    ) -> _ResolvedClassicalFact:
        """Create a normalized immutable classical fact.

        Args:
            value (sp.Basic | int | float | bool): Resolved classical value.
            source_guards (Mapping[str, sp.Basic] | None): Optional source
                tokens and their activation guards. Defaults to ``None``.

        Returns:
            _ResolvedClassicalFact: Fact with deterministic nonfalse guards.
        """
        return cls(
            sp.sympify(value),
            _normalized_source_guards(source_guards or {}),
        )

    @property
    def dependencies(self) -> dict[str, Boolean]:
        """Return a mutable copy of the source-token guards.

        Returns:
            dict[str, Boolean]: Source tokens mapped to activation guards.
        """
        return dict(self.source_guards)


class _ArrayState:
    """Marker base for immutable classical-array estimator state."""


@dataclasses.dataclass(frozen=True)
class _ArrayReferenceState(_ArrayState):
    """Reference an IR array whose provenance remains in the current scope.

    Args:
        array (ArrayValue): IR array projected when an element is requested.
    """

    array: ArrayValue


@dataclasses.dataclass(frozen=True)
class _ArrayConstantState(_ArrayState):
    """Hold immutable nested constant-array contents.

    Args:
        contents (Any): Frozen nested array payload.
    """

    contents: Any


@dataclasses.dataclass(frozen=True)
class _ArrayStoreState(_ArrayState):
    """Represent one immutable element update.

    Args:
        previous (_ArrayState): State before the update.
        stored (_ResolvedClassicalFact): Call-scoped stored scalar fact.
        indices (tuple[_ResolvedClassicalFact, ...]): Call-scoped store-index
            facts.
    """

    previous: _ArrayState
    stored: _ResolvedClassicalFact
    indices: tuple[_ResolvedClassicalFact, ...]


@dataclasses.dataclass(frozen=True)
class _ArraySliceState(_ArrayState):
    """Represent a one-dimensional affine array view.

    Args:
        source (_ArrayState): State viewed by the slice.
        start (_ResolvedClassicalFact): Source-space first-index fact.
        step (_ResolvedClassicalFact): Source-space stride fact.
    """

    source: _ArrayState
    start: _ResolvedClassicalFact
    step: _ResolvedClassicalFact


@dataclasses.dataclass(frozen=True)
class _ArrayChoiceState(_ArrayState):
    """Represent a predicate-selected immutable array state.

    Args:
        when_true (_ArrayState): State selected when the predicate is true.
        when_false (_ArrayState): State selected when the predicate is false.
        condition (_ResolvedClassicalFact): Selection-predicate fact.
    """

    when_true: _ArrayState
    when_false: _ArrayState
    condition: _ResolvedClassicalFact


@dataclasses.dataclass(frozen=True)
class _ArrayLoopSummaryState(_ArrayState):
    """Summarize one traced array update over a symbolic iteration range.

    Args:
        initial (_ArrayState): State before the loop executes.
        iteration (_ArrayState): State after one symbolic body execution.
        loop_symbol (sp.Symbol): Symbol used by the traced body index.
        start (sp.Expr): First Python-range value.
        step (sp.Expr): Python-range stride.
        iterations (sp.Expr): Number of reachable iterations.
        fallback (sp.Symbol): Internal value used only when the final element
            value cannot be represented without a loop-local symbol.
        uncertainty_token (str): Boundary readiness token used for that
            conservative fallback.
    """

    initial: _ArrayState
    iteration: _ArrayState
    loop_symbol: sp.Symbol
    start: sp.Expr
    step: sp.Expr
    iterations: sp.Expr
    fallback: sp.Symbol
    uncertainty_token: str


@dataclasses.dataclass(frozen=True)
class _ArrayUnknownLoopSummaryState(_ArrayState):
    """Summarize an array after iterations with unknown keys or values.

    Args:
        initial (_ArrayState): State selected when the loop is empty.
        iterations (sp.Expr): Number of loop iterations.
        fallback (sp.Symbol): Internal unresolved element value.
        uncertainty_token (str): Loop-boundary readiness token.
    """

    initial: _ArrayState
    iterations: sp.Expr
    fallback: sp.Symbol
    uncertainty_token: str


def input_shape_dimension_aliases(block: Block) -> dict[str, str]:
    """Return collision-free public aliases for root array dimensions.

    Frontend-generated dimension labels can collide with an ordinary
    classical argument (for example, a vector ``signal`` and a UInt argument
    named ``signal_dim0``). Reusing that spelling for both SymPy symbols would
    make one resource input specialize two semantically different values.

    Args:
        block (Block): Root block whose input dimensions should be named.

    Returns:
        dict[str, str]: Dimension UUID to deterministic, unique input alias.
    """
    occupied = {
        *block.label_args,
        *(slot.name for slot in block.param_slots),
        *block.parameters,
    }
    aliases: dict[str, str] = {}
    for input_value in block.input_values:
        if not isinstance(input_value, ArrayValue):
            continue
        for dimension in input_value.shape:
            alias = dimension.name
            if not alias:
                alias = f"array_dim_{len(aliases)}"
            if alias in occupied:
                base = f"{alias}__shape"
                alias = base
                suffix = 2
                while alias in occupied:
                    alias = f"{base}_{suffix}"
                    suffix += 1
            aliases[dimension.uuid] = alias
            occupied.add(alias)
    return aliases


class UnresolvedValueError(Exception):
    """A value cannot be concretized during resource estimation."""

    def __init__(self, uuid: str, message: str = ""):
        self.uuid = uuid
        super().__init__(message or f"Cannot resolve value {uuid} to concrete int")


class ExprResolver:
    """Single source of truth for converting IR Values to SymPy expressions.

    Resolution strategy (deterministic, single path):
      1. Already sp.Basic                        → return as-is
      2. Not a Value (int, float, bool)           → direct conversion
      3. UUID in context (call context / expression) → return mapped expression
      4. Constant value                           → sp.Integer / sp.Float
      5. Unbound parameter                        → sp.Symbol (symbolic) or raise (concrete)
      6. Arithmetic/comparison result              → trace in block operations
      7. Search parent blocks                     → trace in ancestors
      8. Fallback                                 → identity-qualified symbol or raise
    """

    __slots__ = (
        "_array_context",
        "_block",
        "_classical_fact_context",
        "_context",
        "_input_shape_alias_maps",
        "_loop_var_names",
        "_parent_blocks",
        "_producer_maps",
        "_structural_scope",
    )

    def __init__(
        self,
        block: Any = None,
        context: dict[str, sp.Expr] | None = None,
        loop_var_names: dict[str, sp.Expr] | None = None,
        parent_blocks: list[Any] | None = None,
        producer_maps: dict[int, tuple[Any, dict[str, Operation]]] | None = None,
        input_shape_alias_maps: (dict[int, tuple[Block, dict[str, str]]] | None) = None,
        structural_scope: tuple[tuple[int, int], ...] | None = None,
        array_context: dict[str, _ArrayState] | None = None,
        classical_fact_context: dict[str, _ResolvedClassicalFact] | None = None,
    ):
        """Initialise an ExprResolver.

        Args:
            block (Any): The current block (Block or _LocalBlock)
                whose operations are searched for classical expression traces.
            context (dict[str, sp.Expr] | None): UUID → resolved expression
                mapping for values passed across scope boundaries (e.g.
                call arguments, composite-gate operands).
            loop_var_names (dict[str, sp.Expr] | None): Value name → SymPy
                expression mapping for loop variables in scope.
            parent_blocks (list[Any] | None): Ancestor blocks to search
                when tracing fails in the current block.
            producer_maps (dict[int, tuple[Any, dict[str, Operation]]] | None):
                Shared block-identity index containing a strong block reference
                and its result UUID to producer-operation map. Child resolvers
                reuse it so every block is indexed at most once. Defaults to
                ``None``.
            input_shape_alias_maps (dict[int, tuple[Block, dict[str, str]]] | None):
                Shared block-identity index containing a strong block reference
                and its input-dimension UUID to public alias map. Child
                resolvers reuse it so every block interface is scanned at most
                once. Defaults to ``None``.
            structural_scope (tuple[tuple[int, int], ...] | None): Stable
                call-site path used by structural resource symbols. Each item
                contains the invocation and selected-body identities. Defaults
                to ``None`` for a root scope.
            array_context (dict[str, _ArrayState] | None): Array-result UUID to
                immutable element-wise state. Ordinary child regions share the
                mapping, while callable scopes copy and explicitly publish
                output snapshots. Defaults to ``None``.
            classical_fact_context (dict[str, _ResolvedClassicalFact] | None):
                Scalar or whole-array UUID to its resolved value and guarded
                scheduler dependencies. Defaults to ``None``.
        """
        self._array_context = array_context if array_context is not None else {}
        self._block = block
        self._classical_fact_context = dict(classical_fact_context or {})
        self._context: dict[str, sp.Expr] = dict(context or {})
        self._loop_var_names: dict[str, sp.Expr] = dict(loop_var_names or {})
        self._parent_blocks: list[Any] = list(parent_blocks or [])
        self._producer_maps = producer_maps if producer_maps is not None else {}
        self._input_shape_alias_maps = (
            input_shape_alias_maps if input_shape_alias_maps is not None else {}
        )
        self._structural_scope = structural_scope or ()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def resolve(self, v: Any) -> sp.Expr:
        """Convert IR Value to SymPy expression (symbolic mode).

        Unbound parameters become ``sp.Symbol``.  Never raises for valid IR.

        Args:
            v (Any): IR Value, primitive Python type, or ``sp.Basic``.

        Returns:
            sp.Expr: Resolved SymPy expression.
        """
        return self._resolve(v, concrete=False)

    def resolve_concrete(self, v: Any) -> int:
        """Convert IR Value to concrete ``int``.

        Args:
            v (Any): IR Value, primitive Python type, or ``sp.Basic``.

        Returns:
            int: The resolved concrete integer.

        Raises:
            UnresolvedValueError: If the value is symbolic.
        """
        expr = self._resolve(v, concrete=True)
        if isinstance(expr, sp.Integer):
            return int(expr)
        if expr.is_number and expr.is_integer:
            return int(expr)
        raise UnresolvedValueError(
            getattr(v, "uuid", "?"),
            f"Expected concrete int, got {expr}",
        )

    def resolve_classical_fact(self, value: Any) -> _ResolvedClassicalFact:
        """Resolve a classical value together with scheduler dependencies.

        Precise array-element state takes precedence over a conservative
        whole-array fact. The whole-array dependency is used only when the
        persistent state cannot project the requested element.

        Args:
            value (Any): IR value, primitive Python value, or SymPy value to
                resolve.

        Returns:
            _ResolvedClassicalFact: Resolved value and guarded source tokens.
        """
        return self._resolve_classical_fact(value, concrete=False)

    def bind_classical_fact(
        self,
        value: Value,
        fact: _ResolvedClassicalFact,
    ) -> None:
        """Bind one scalar or whole-array provenance fact by SSA identity.

        Args:
            value (Value): IR value receiving the fact.
            fact (_ResolvedClassicalFact): Resolved value and dependencies.
        """
        self._classical_fact_context[value.uuid] = fact
        self._context[value.uuid] = cast(sp.Expr, fact.value)

    def bind_classical_selection(
        self,
        result: Value,
        when_true: _ResolvedClassicalFact,
        when_false: _ResolvedClassicalFact,
        selector: _ResolvedClassicalFact,
        value_override: sp.Basic | int | float | bool | None = None,
    ) -> None:
        """Bind a branch-selected scalar fact to one SSA result.

        Branch dependencies use the same guarded choice semantics as array
        element projection. A caller may supply a separately derived value
        expression without changing those dependency guards.

        Args:
            result (Value): Scalar SSA result receiving the selected fact.
            when_true (_ResolvedClassicalFact): True-branch value and sources.
            when_false (_ResolvedClassicalFact): False-branch value and
                sources.
            selector (_ResolvedClassicalFact): Branch selector and its source
                dependencies.
            value_override (sp.Basic | int | float | bool | None): Optional
                result expression to use instead of the selected Piecewise
                value. Defaults to ``None``.
        """
        selected = _choice_classical_fact(when_true, when_false, selector)
        if value_override is not None:
            selected = _ResolvedClassicalFact.create(
                value_override,
                selected.dependencies,
            )
        self.bind_classical_fact(result, selected)

    def unresolved_fallback_symbol(self, value: Value) -> sp.Symbol | None:
        """Return the private fallback symbol used for an unresolved value.

        A resolved expression can contain both runtime-derived state and
        ordinary public inputs.  Callers that classify the former must not
        infer provenance from every free symbol in the expression.  This
        method distinguishes the one identity-qualified symbol introduced by
        the resolver itself when no binding, constant, parameter, supported
        producer, or public input-shape alias can explain ``value``.

        Args:
            value (Value): IR scalar value to classify.

        Returns:
            sp.Symbol | None: The resolver-owned fallback symbol, or ``None``
                when ``value`` has an ordinary symbolic resolution.
        """
        if value.uuid in self._context or value.is_constant() or value.is_parameter():
            return None
        if (
            value.is_array_element()
            and self._resolve_array_element(
                value,
                concrete=False,
            )
            is not None
        ):
            return None
        if self._block is not None:
            traced = self._trace(value, self._block, set(), concrete=False)
            if traced is not None:
                return None
        for parent_block in reversed(self._parent_blocks):
            traced = self._trace(value, parent_block, set(), concrete=False)
            if traced is not None:
                return None
        if self._input_shape_dimension_alias(value) is not None:
            return None
        return _fallback_symbol(value)

    @property
    def structural_scope(self) -> tuple[tuple[int, int], ...]:
        """Return the nested callable path for structural resource symbols.

        Returns:
            tuple[tuple[int, int], ...]: Invocation/body identity pairs from
                the root block to this resolver scope.
        """
        return self._structural_scope

    def call_structural_scope(
        self,
        call_op: Any,
        called_block: Block,
    ) -> tuple[tuple[int, int], ...]:
        """Return the structural scope of one selected callable body.

        Args:
            call_op (Any): Invocation-like operation defining the call site.
            called_block (Block): Selected body entered at that call site.

        Returns:
            tuple[tuple[int, int], ...]: Parent path extended by this call
                site and selected body.
        """
        return (*self._structural_scope, (id(call_op), id(called_block)))

    def child_scope(
        self,
        inner_block: Any,
        extra_context: dict[str, sp.Expr] | None = None,
        extra_loop_vars: dict[str, sp.Expr] | None = None,
    ) -> ExprResolver:
        """Create a child resolver for an inner scope (loop body, branch).

        Propagates parent_blocks so values from outer scopes remain
        traceable.  For callee invocation scopes, use
        :meth:`call_child_scope` instead — callees get a fresh scope.

        Args:
            inner_block (Any): The block for the child scope.
            extra_context (dict[str, sp.Expr] | None): Additional UUID →
                expression mappings to merge into the child context.
            extra_loop_vars (dict[str, sp.Expr] | None): Additional loop
                variable name → expression mappings.

        Returns:
            ExprResolver: A new resolver scoped to *inner_block* with
                parent blocks propagated from the current resolver.
        """
        ctx = self._context.copy()
        if extra_context:
            ctx.update(extra_context)
        lvn = self._loop_var_names.copy()
        if extra_loop_vars:
            lvn.update(extra_loop_vars)
        # Propagate parent chain: existing parents + current block
        new_parents = list(self._parent_blocks)
        if self._block is not None:
            new_parents.append(self._block)
        return ExprResolver(
            block=inner_block,
            context=ctx,
            loop_var_names=lvn,
            parent_blocks=new_parents,
            producer_maps=self._producer_maps,
            input_shape_alias_maps=self._input_shape_alias_maps,
            structural_scope=self._structural_scope,
            array_context=self._array_context,
            classical_fact_context=self._classical_fact_context,
        )

    def isolated_scope(
        self,
        inner_block: Any,
        extra_context: dict[str, sp.Expr] | None = None,
        structural_scope: tuple[tuple[int, int], ...] | None = None,
    ) -> ExprResolver:
        """Create a resolver scope isolated from caller block visibility.

        Callable bodies receive only values mapped explicitly through their
        operands, but immutable block indexes remain safe to share across the
        resolver tree.

        Args:
            inner_block (Any): Callable block for the isolated scope.
            extra_context (dict[str, sp.Expr] | None): Additional UUID to
                expression mappings for formal inputs. Defaults to ``None``.
            structural_scope (tuple[tuple[int, int], ...] | None): Explicit
                call-site path for the isolated scope. Defaults to the current
                path.

        Returns:
            ExprResolver: Resolver with no parent blocks and shared block
                indexes.
        """
        context = dict(extra_context or {})
        return ExprResolver(
            block=inner_block,
            context=context,
            loop_var_names=self._loop_var_names.copy(),
            parent_blocks=[],
            producer_maps=self._producer_maps,
            input_shape_alias_maps=self._input_shape_alias_maps,
            structural_scope=(
                self._structural_scope if structural_scope is None else structural_scope
            ),
            array_context={},
            classical_fact_context={},
        )

    def call_child_scope(
        self,
        call_op: Any,
        *,
        called_block: Block | None = None,
        body_implements_transform: bool = False,
        actual_operands: Sequence[Any] | None = None,
    ) -> ExprResolver:
        """Create a child resolver for an inline callable invocation.

        Maps formal parameter UUIDs → resolved actual arguments,
        **including array shape dimension UUIDs** (critical for
        resolving e.g. ``kernel.shape[0]`` inside the callee).

        Parent blocks are intentionally reset — the callee only sees
        its own scope plus values propagated through ``call_context``.

        Args:
            call_op (Any): An invocation carrying either a legacy
                ``block`` field, an ``InvokeOperation.effective_body()``
                method, or an ``InvokeOperation.body`` field, plus
                ``operands`` containing actual arguments.
            called_block (Block | None): Already-selected callable body.
                Pass this when another resolver has selected a backend- or
                strategy-specific implementation. Defaults to ``None``.
            body_implements_transform (bool): Whether ``called_block`` is a
                transform-specific implementation whose formal inputs include
                control operands. Defaults to ``False`` for a direct body that
                the compiler transforms structurally.
            actual_operands (Sequence[Any] | None): Call-site operands already
                aligned to ``called_block``. When omitted, the resolver derives
                the alignment from the invocation metadata. Defaults to
                ``None``.

        Returns:
            ExprResolver: A new resolver scoped to the callee block with
                formal→actual bindings in context and empty parent blocks.
        """
        if called_block is None:
            called_block = getattr(call_op, "block", None)
        if not isinstance(called_block, Block):
            effective_body = getattr(call_op, "effective_body", None)
            if callable(effective_body):
                called_block = effective_body()
        if not isinstance(called_block, Block):
            called_block = getattr(call_op, "body", None)
        if not isinstance(called_block, Block):
            # Not a nested Block input — use child_scope as fallback
            return self.child_scope(called_block)

        if actual_operands is None:
            actual_operands = call_op.operands
            if (
                getattr(call_op, "transform", CallTransform.DIRECT).is_controlled
                and not body_implements_transform
            ):
                control_count = getattr(
                    call_op,
                    "num_body_external_control_qubits",
                    call_op.num_control_qubits,
                )
                actual_operands = actual_operands[control_count:]

        extra: dict[str, sp.Expr] = {}
        array_inputs: list[tuple[ArrayValue, ArrayValue]] = []
        classical_inputs: list[tuple[Value, Value]] = []
        for formal, actual in pair_block_operands(called_block, actual_operands):
            extra[formal.uuid] = self.resolve(actual)
            if (
                isinstance(formal, Value)
                and isinstance(actual, Value)
                and not isinstance(formal, ArrayValue)
                and not isinstance(actual, ArrayValue)
                and not formal.type.is_quantum()
            ):
                classical_inputs.append((formal, actual))
            # Map array shape dimension UUIDs
            if isinstance(actual, ArrayValue) and isinstance(formal, ArrayValue):
                if not formal.type.is_quantum():
                    array_inputs.append((formal, actual))
                for df, da in zip(formal.shape, actual.shape):
                    extra[df.uuid] = self.resolve(da)

        # Callee gets fresh scope — no parent blocks from caller.
        child = self.isolated_scope(
            called_block,
            extra,
            structural_scope=self.call_structural_scope(call_op, called_block),
        )
        for formal, actual in classical_inputs:
            child.bind_classical_fact(
                formal,
                self.resolve_classical_fact(actual),
            )
        for formal, actual in array_inputs:
            child.bind_call_array_input(
                called_block,
                formal,
                self.snapshot_array_state(actual),
            )
        return child

    def bind(self, value: Value, expression: sp.Expr) -> None:
        """Bind an IR value to an expression in this resolver scope.

        This is used for SSA results whose value is established while walking
        operations in program order, notably the final results of loop region
        arguments. Child scopes still receive a copy, so a binding cannot leak
        backwards into an already-created sibling scope.

        Args:
            value (Value): IR value whose UUID identifies the binding.
            expression (sp.Expr): Symbolic or concrete expression represented by
                the value.
        """
        self._context[value.uuid] = expression
        self._classical_fact_context[value.uuid] = _ResolvedClassicalFact.create(
            expression
        )

    def bind_array_selection(
        self,
        result: ArrayValue,
        when_true: ArrayValue,
        when_false: ArrayValue,
        condition: sp.Basic | _ResolvedClassicalFact,
    ) -> None:
        """Bind an array result to branch-selected source array states.

        Array contents are not scalar SymPy expressions. Keeping the selected
        source arrays structurally lets a later element read project the same
        index from the selected state without inventing a public array symbol.

        Args:
            result (ArrayValue): Array SSA version visible after selection.
            when_true (ArrayValue): Source array selected when ``condition``
                is true.
            when_false (ArrayValue): Source array selected when ``condition``
                is false.
            condition (sp.Basic | _ResolvedClassicalFact): Predicate selecting
                the source array, optionally with source-token provenance.
        """
        self._array_context[result.uuid] = _ArrayChoiceState(
            self.snapshot_array_state(
                when_true,
                ignore_binding=result.uuid if when_true.uuid == result.uuid else None,
            ),
            self.snapshot_array_state(
                when_false,
                ignore_binding=result.uuid if when_false.uuid == result.uuid else None,
            ),
            _coerce_classical_fact(condition),
        )

    def bind_array_state(self, result: ArrayValue, state: _ArrayState) -> None:
        """Bind a caller-visible array result to an immutable state snapshot.

        Args:
            result (ArrayValue): Array SSA value receiving the snapshot.
            state (_ArrayState): Frozen state resolved in the producing scope.
        """
        self._array_context[result.uuid] = state

    def bind_array_state_selection(
        self,
        result: ArrayValue,
        when_true: _ArrayState,
        when_false: _ArrayState,
        selector: _ResolvedClassicalFact,
    ) -> None:
        """Bind detached branch snapshots to one selected array result.

        The snapshots may come from independently forked branch resolvers.
        Keeping the choice structural lets each later element projection drop
        selector dependencies when that element's two facts are identical.

        Args:
            result (ArrayValue): Array SSA result receiving the selected state.
            when_true (_ArrayState): Detached true-branch snapshot.
            when_false (_ArrayState): Detached false-branch snapshot.
            selector (_ResolvedClassicalFact): Branch selector and its source
                dependencies.
        """
        state = (
            when_true
            if when_true == when_false
            else _ArrayChoiceState(when_true, when_false, selector)
        )
        self.bind_array_state(result, state)

    def bind_loop_array_summary(
        self,
        result: ArrayValue,
        *,
        initial: _ArrayState,
        iteration: _ArrayState,
        loop_symbol: sp.Symbol,
        start: sp.Expr,
        step: sp.Expr,
        iterations: sp.Expr,
        fallback: sp.Symbol,
        uncertainty_token: str,
    ) -> None:
        """Bind one range-quantified loop-exit array state.

        Args:
            result (ArrayValue): Caller-visible array SSA result.
            initial (_ArrayState): State selected by a zero-trip loop.
            iteration (_ArrayState): State after one symbolic body execution.
            loop_symbol (sp.Symbol): Body-local induction symbol.
            start (sp.Expr): First range value.
            step (sp.Expr): Range stride.
            iterations (sp.Expr): Number of loop iterations.
            fallback (sp.Symbol): Internal unresolved element placeholder.
            uncertainty_token (str): Boundary token for conservative element
                projection.
        """
        self.bind_array_state(
            result,
            self.loop_array_summary_state(
                initial=initial,
                iteration=iteration,
                loop_symbol=loop_symbol,
                start=start,
                step=step,
                iterations=iterations,
                fallback=fallback,
                uncertainty_token=uncertainty_token,
            ),
        )

    def loop_array_summary_state(
        self,
        *,
        initial: _ArrayState,
        iteration: _ArrayState,
        loop_symbol: sp.Symbol,
        start: sp.Expr,
        step: sp.Expr,
        iterations: sp.Expr,
        fallback: sp.Symbol,
        uncertainty_token: str,
    ) -> _ArrayState:
        """Create one range-quantified array state without binding it.

        Args:
            initial (_ArrayState): State before the loop executes.
            iteration (_ArrayState): One symbolic body transition.
            loop_symbol (sp.Symbol): Body-local induction symbol.
            start (sp.Expr): First range value.
            step (sp.Expr): Range stride.
            iterations (sp.Expr): Number of loop iterations.
            fallback (sp.Symbol): Internal unresolved element placeholder.
            uncertainty_token (str): Boundary token for conservative element
                projection.

        Returns:
            _ArrayState: Immutable symbolic loop summary.
        """
        return _ArrayLoopSummaryState(
            initial,
            iteration,
            loop_symbol,
            start,
            step,
            iterations,
            fallback,
            uncertainty_token,
        )

    def unknown_loop_array_summary_state(
        self,
        *,
        initial: _ArrayState,
        iterations: sp.Expr,
        fallback: sp.Symbol,
        uncertainty_token: str,
    ) -> _ArrayState:
        """Create a conservative loop-exit state for unknown item updates.

        Args:
            initial (_ArrayState): State selected when the loop is empty.
            iterations (sp.Expr): Number of loop iterations.
            fallback (sp.Symbol): Internal unresolved element value.
            uncertainty_token (str): Loop-boundary readiness token.

        Returns:
            _ArrayState: Immutable unknown loop summary.
        """
        return _ArrayUnknownLoopSummaryState(
            initial,
            iterations,
            fallback,
            uncertainty_token,
        )

    def copy_array_context(self) -> None:
        """Detach this resolver from a shared mutable array-context mapping."""
        self._array_context = dict(self._array_context)

    def fork_array_context(self) -> dict[str, _ArrayState]:
        """Return a detached shallow copy for a child resolver scope.

        The state nodes are immutable, so copying only the UUID map is enough
        to isolate later bindings while retaining structural sharing.

        Returns:
            dict[str, _ArrayState]: Detached array-state mapping.
        """
        return dict(self._array_context)

    def export_array_context(
        self,
        arrays: Sequence[ArrayValue] | None = None,
    ) -> dict[str, _ArrayState]:
        """Export all or selected persistent array-state bindings.

        Args:
            arrays (Sequence[ArrayValue] | None): Optional array SSA values to
                export. Defaults to ``None``, which exports every binding.

        Returns:
            dict[str, _ArrayState]: Detached mapping safe to import elsewhere.
        """
        if arrays is None:
            return self.fork_array_context()
        return {
            array.uuid: self._array_context[array.uuid]
            for array in arrays
            if array.uuid in self._array_context
        }

    def import_array_context(
        self,
        context: Mapping[str, _ArrayState],
        *,
        replace: bool = False,
    ) -> None:
        """Import persistent array-state bindings into this resolver.

        Args:
            context (Mapping[str, _ArrayState]): Exported UUID-to-state map.
            replace (bool): Whether to replace every existing binding before
                importing. Defaults to ``False``, which overlays the supplied
                bindings.
        """
        imported = dict(context)
        if replace:
            self._array_context = imported
            return
        self._array_context = {**self._array_context, **imported}

    def bind_call_array_input(
        self,
        block: Block,
        formal: ArrayValue,
        state: _ArrayState,
    ) -> None:
        """Bind the entry lineage of one callable array formal.

        Callable tracing mutates ``Block.input_values`` to the latest SSA
        version. A helper that stores into an input can therefore expose its
        produced output version as the formal interface value. The formal is
        nevertheless the value visible at call entry, so bind it before body
        evaluation; also bind unproduced aliases in the same lineage without
        overwriting produced intermediate versions.

        Args:
            block (Block): Selected callable body.
            formal (ArrayValue): Array value paired with the call operand.
            state (_ArrayState): Caller state captured at invocation time.
        """
        produced = {
            result.uuid
            for operation in walk_operations(block.operations)
            for result in operation.results
            if isinstance(result, ArrayValue)
        }
        self._array_context[formal.uuid] = state
        candidates: list[ArrayValue] = []
        for operation in walk_operations(block.operations):
            candidates.extend(
                operand
                for operand in operation.operands
                if isinstance(operand, ArrayValue)
                and operand.logical_id == formal.logical_id
            )
        for candidate in candidates:
            if candidate.uuid != formal.uuid and candidate.uuid not in produced:
                self._array_context[candidate.uuid] = state

    def bind_loop_array_input(
        self,
        operations: Sequence[Operation],
        entry: ArrayValue,
        state: _ArrayState,
    ) -> None:
        """Bind one carried array snapshot to a loop body's entry SSA values.

        A traced loop body reuses the same SSA graph for every iteration. The
        explicit entry always receives the previous iteration's snapshot,
        even when tracing reused its UUID for a later result. Other aliases are
        rebound only when unproduced, so within-iteration updates stay intact.

        Args:
            operations (Sequence[Operation]): Loop-body operations.
            entry (ArrayValue): Pre-loop array value naming the carried lineage.
            state (_ArrayState): Snapshot produced by the previous iteration.
        """
        produced = {
            result.uuid
            for operation in walk_operations(operations)
            for result in operation.results
            if isinstance(result, ArrayValue)
        }
        self._array_context[entry.uuid] = state
        candidates: list[ArrayValue] = []
        for operation in walk_operations(operations):
            candidates.extend(
                operand
                for operand in operation.operands
                if isinstance(operand, ArrayValue)
                and operand.logical_id == entry.logical_id
            )
        for candidate in candidates:
            if candidate.uuid != entry.uuid and candidate.uuid not in produced:
                self._array_context[candidate.uuid] = state

    def instantiate_array_state(
        self,
        state: _ArrayState,
        *,
        state_replacements: Sequence[tuple[_ArrayState, _ArrayState]] = (),
        substitutions: Mapping[sp.Basic, sp.Basic] | None = None,
    ) -> _ArrayState:
        """Instantiate one persistent array transition without reevaluating IR.

        Identity-based state replacements splice a previous iteration's exit
        snapshot into the traced transition. Scalar substitutions then bind
        the loop index in stored values, indices, and source guards.

        Args:
            state (_ArrayState): Persistent transition state to instantiate.
            state_replacements (Sequence[tuple[_ArrayState, _ArrayState]]):
                Identity anchors and their replacement snapshots. Defaults to
                an empty sequence.
            substitutions (Mapping[sp.Basic, sp.Basic] | None): Simultaneous
                scalar substitutions. Defaults to ``None``.

        Returns:
            _ArrayState: Instantiated immutable state tree.
        """
        for anchor, replacement in state_replacements:
            if state is anchor:
                return replacement
        scalar_substitutions = dict(substitutions or {})

        def fact(value: _ResolvedClassicalFact) -> _ResolvedClassicalFact:
            """Instantiate one scalar fact.

            Args:
                value (_ResolvedClassicalFact): Fact to specialize.

            Returns:
                _ResolvedClassicalFact: Specialized value and source guards.
            """
            if not scalar_substitutions:
                return value
            return _ResolvedClassicalFact.create(
                value.value.subs(
                    cast(Any, scalar_substitutions),
                    simultaneous=True,
                ),
                {
                    source: guard.subs(
                        cast(Any, scalar_substitutions),
                        simultaneous=True,
                    )
                    for source, guard in value.dependencies.items()
                },
            )

        if isinstance(state, (_ArrayReferenceState, _ArrayConstantState)):
            return state
        if isinstance(state, _ArrayStoreState):
            return _ArrayStoreState(
                self.instantiate_array_state(
                    state.previous,
                    state_replacements=state_replacements,
                    substitutions=scalar_substitutions,
                ),
                fact(state.stored),
                tuple(fact(index) for index in state.indices),
            )
        if isinstance(state, _ArraySliceState):
            return _ArraySliceState(
                self.instantiate_array_state(
                    state.source,
                    state_replacements=state_replacements,
                    substitutions=scalar_substitutions,
                ),
                fact(state.start),
                fact(state.step),
            )
        if isinstance(state, _ArrayChoiceState):
            condition = fact(state.condition)
            predicate = _as_boolean(condition.value)
            when_true = self.instantiate_array_state(
                state.when_true,
                state_replacements=state_replacements,
                substitutions=scalar_substitutions,
            )
            when_false = self.instantiate_array_state(
                state.when_false,
                state_replacements=state_replacements,
                substitutions=scalar_substitutions,
            )
            if predicate is sp.true:
                return when_true
            if predicate is sp.false:
                return when_false
            return _ArrayChoiceState(when_true, when_false, condition)
        if isinstance(state, _ArrayLoopSummaryState):
            return _ArrayLoopSummaryState(
                self.instantiate_array_state(
                    state.initial,
                    state_replacements=state_replacements,
                    substitutions=scalar_substitutions,
                ),
                self.instantiate_array_state(
                    state.iteration,
                    state_replacements=state_replacements,
                    substitutions=scalar_substitutions,
                ),
                state.loop_symbol,
                cast(
                    sp.Expr,
                    state.start.subs(
                        cast(Any, scalar_substitutions),
                        simultaneous=True,
                    ),
                ),
                cast(
                    sp.Expr,
                    state.step.subs(
                        cast(Any, scalar_substitutions),
                        simultaneous=True,
                    ),
                ),
                cast(
                    sp.Expr,
                    state.iterations.subs(
                        cast(Any, scalar_substitutions),
                        simultaneous=True,
                    ),
                ),
                state.fallback,
                state.uncertainty_token,
            )
        if isinstance(state, _ArrayUnknownLoopSummaryState):
            return _ArrayUnknownLoopSummaryState(
                self.instantiate_array_state(
                    state.initial,
                    state_replacements=state_replacements,
                    substitutions=scalar_substitutions,
                ),
                cast(
                    sp.Expr,
                    state.iterations.subs(
                        cast(Any, scalar_substitutions),
                        simultaneous=True,
                    ),
                ),
                state.fallback,
                state.uncertainty_token,
            )
        return state

    def snapshot_array_state(
        self,
        array: ArrayValue,
        *,
        ignore_binding: str | None = None,
        visited: set[str] | None = None,
    ) -> _ArrayState:
        """Capture call-scoped immutable state for an array SSA value.

        Scalar store operands, indices, and selection predicates are resolved
        immediately, so a later invocation reusing the callee's formal UUIDs
        cannot change an earlier caller result.

        Args:
            array (ArrayValue): Array whose current state is captured.
            ignore_binding (str | None): Array binding bypassed for one raw
                producer lookup. Defaults to ``None``.
            visited (set[str] | None): Array UUIDs already visited on this
                capture path. Defaults to ``None``.

        Returns:
            _ArrayState: Immutable state tree rooted at ``array``.
        """
        if visited is None:
            visited = set()
        binding = self._array_context.get(array.uuid)
        if binding is not None and ignore_binding != array.uuid:
            return binding
        if array.uuid in visited:
            return _ArrayReferenceState(array)
        visited.add(array.uuid)

        if array.is_slice() and array.slice_of is not None:
            if array.slice_start is not None and array.slice_step is not None:
                return _ArraySliceState(
                    self.snapshot_array_state(
                        array.slice_of,
                        visited=set(visited),
                    ),
                    self.resolve_classical_fact(array.slice_start),
                    self.resolve_classical_fact(array.slice_step),
                )
        producer = self._array_producer(array)
        if isinstance(producer, StoreArrayElementOperation):
            return _ArrayStoreState(
                self.snapshot_array_state(
                    producer.array,
                    visited=set(visited),
                ),
                self.resolve_classical_fact(producer.stored_value),
                tuple(
                    self.resolve_classical_fact(index)
                    for index in producer.index_values
                ),
            )
        if isinstance(producer, IfOperation):
            merge = next(
                (
                    candidate
                    for candidate in producer.iter_merges()
                    if candidate.result.uuid == array.uuid
                    and isinstance(candidate.true_value, ArrayValue)
                    and isinstance(candidate.false_value, ArrayValue)
                ),
                None,
            )
            if merge is not None:
                return _ArrayChoiceState(
                    self.snapshot_array_state(
                        cast(ArrayValue, merge.true_value),
                        visited=set(visited),
                    ),
                    self.snapshot_array_state(
                        cast(ArrayValue, merge.false_value),
                        visited=set(visited),
                    ),
                    self.resolve_classical_fact(producer.condition),
                )
        constant = array.get_const_array()
        if constant is not None:
            return _ArrayConstantState(constant)
        return _ArrayReferenceState(array)

    def array_state_dependencies(self, array: ArrayValue) -> dict[str, Boolean]:
        """Return every guarded observation source retained by an array state.

        This whole-state summary is used only at aggregate boundaries such as
        loops and calls. Element reads remain precise through
        :meth:`resolve_classical_fact`; publishing the union here ensures each
        retained element token becomes ready no earlier than the boundary.

        Args:
            array (ArrayValue): Array whose immutable state is summarized.

        Returns:
            dict[str, Boolean]: Retained source tokens and activation guards.
        """

        def visit(state: _ArrayState) -> dict[str, Boolean]:
            """Collect dependencies from one persistent state node.

            Args:
                state (_ArrayState): State node to inspect.

            Returns:
                dict[str, Boolean]: Source guards reachable from the node.
            """
            if isinstance(state, (_ArrayReferenceState, _ArrayConstantState)):
                return {}
            if isinstance(state, _ArrayStoreState):
                return _merge_source_guard_maps(
                    visit(state.previous),
                    state.stored.dependencies,
                    *(index.dependencies for index in state.indices),
                )
            if isinstance(state, _ArraySliceState):
                return _merge_source_guard_maps(
                    visit(state.source),
                    state.start.dependencies,
                    state.step.dependencies,
                )
            if isinstance(state, _ArrayChoiceState):
                predicate = _as_boolean(state.condition.value)
                return _merge_source_guard_maps(
                    state.condition.dependencies,
                    _guard_source_guards(visit(state.when_true), predicate),
                    _guard_source_guards(
                        visit(state.when_false),
                        sp.Not(predicate),
                    ),
                )
            if isinstance(state, _ArrayLoopSummaryState):
                active = _as_boolean(sp.Gt(state.iterations, 0))
                iteration_dependencies = visit(state.iteration)
                summarized = {
                    source: _as_boolean(
                        _activation_over_range(
                            guard,
                            state.loop_symbol,
                            state.start,
                            state.step,
                            state.iterations,
                        )
                    )
                    for source, guard in iteration_dependencies.items()
                }
                return _merge_source_guard_maps(
                    _guard_source_guards(visit(state.initial), sp.Not(active)),
                    summarized,
                    {state.uncertainty_token: active},
                )
            if isinstance(state, _ArrayUnknownLoopSummaryState):
                active = _as_boolean(sp.Gt(state.iterations, 0))
                return _merge_source_guard_maps(
                    _guard_source_guards(visit(state.initial), sp.Not(active)),
                    {state.uncertainty_token: active},
                )
            return {}

        return visit(self.snapshot_array_state(array))

    def guard_array_update(
        self,
        result: ArrayValue,
        previous: ArrayValue,
        condition: sp.Basic | _ResolvedClassicalFact,
    ) -> None:
        """Guard one body-local array update by an enclosing execution path.

        Nested loops encounter the same Store result from the inside out. When
        an inner loop already guarded that result, conjoin the outer reachability
        condition instead of replacing the more specific inner condition.

        Args:
            result (ArrayValue): Array SSA version produced by the store.
            previous (ArrayValue): Array version read by the store.
            condition (sp.Basic | _ResolvedClassicalFact): Predicate that the
                enclosing region executes, optionally with provenance.
        """
        current = self._array_context.get(result.uuid)
        if current is None:
            current = self.snapshot_array_state(
                result,
                ignore_binding=result.uuid,
            )
        self._array_context[result.uuid] = _ArrayChoiceState(
            current,
            self.snapshot_array_state(previous),
            _coerce_classical_fact(condition),
        )

    def record_array_store(self, operation: StoreArrayElementOperation) -> None:
        """Record one Store result in program order.

        Inlining can intentionally reuse the actual operand UUID for a
        callee's returned array. A block-wide producer index then cannot
        distinguish the pre-call and post-call state. Recording each Store as
        it executes preserves the earlier snapshot for untouched slots.

        Args:
            operation (StoreArrayElementOperation): Store just encountered by
                the estimator's sequential interpreter.
        """
        result = operation.results[0]
        if not isinstance(result, ArrayValue):
            return
        previous = self._array_context.get(operation.array.uuid)
        if previous is None:
            # InlinePass can retain the callee formal as the Store operand
            # while substituting the returned result UUID with the caller's
            # current array version. The result binding is then the precise
            # pre-call state that untouched slots must retain.
            previous = self._array_context.get(result.uuid)
        if previous is None and result.is_slice():
            # An inlined Store result can itself retain the caller view's
            # affine lineage even when no prior binding was materialized.
            previous = self.snapshot_array_state(
                result,
                ignore_binding=result.uuid,
            )
        if previous is None:
            if operation.array.uuid == result.uuid:
                constant = operation.array.get_const_array()
                previous = (
                    _ArrayConstantState(constant)
                    if constant is not None
                    else _ArrayReferenceState(operation.array)
                )
            else:
                previous = self.snapshot_array_state(operation.array)
        self._array_context[result.uuid] = _ArrayStoreState(
            previous,
            self.resolve_classical_fact(operation.stored_value),
            tuple(
                self.resolve_classical_fact(index) for index in operation.index_values
            ),
        )

    # Read-only accessors for engine / accumulator use

    @property
    def context(self) -> dict[str, sp.Expr]:
        """Copy of the UUID → expression context mapping."""
        return self._context.copy()

    @property
    def loop_var_names(self) -> dict[str, sp.Expr]:
        """Copy of the loop variable name → expression mapping."""
        return self._loop_var_names.copy()

    @property
    def block(self) -> Any:
        """The current block being resolved against."""
        return self._block

    # ------------------------------------------------------------------ #
    #  Internal resolution                                                #
    # ------------------------------------------------------------------ #

    def _resolve_classical_fact(
        self,
        value: Any,
        *,
        concrete: bool,
    ) -> _ResolvedClassicalFact:
        """Resolve one value and its guarded observation dependencies.

        Args:
            value (Any): IR, Python, or SymPy value to resolve.
            concrete (bool): Whether unresolved symbolic values are rejected.

        Returns:
            _ResolvedClassicalFact: Resolved value and source-token guards.

        Raises:
            UnresolvedValueError: If ``concrete`` is true and the value cannot
                be resolved concretely.
        """
        if isinstance(value, Value):
            direct = self._classical_fact_context.get(value.uuid)
            if direct is not None:
                return direct
            if value.is_array_element():
                projected = self._resolve_array_element_fact(
                    value,
                    concrete=concrete,
                )
                if projected is not None:
                    # A proven element state is strictly more precise than an
                    # aggregate array fallback. In particular, a strong Store
                    # overwrites old dependencies instead of unioning them.
                    return projected
                fallback = _ResolvedClassicalFact.create(
                    self._resolve(value, concrete),
                )
                parent = value.parent_array
                inherited: dict[str, Boolean] = {}
                if parent is not None:
                    parent_fact = self._classical_fact_context.get(parent.uuid)
                    if parent_fact is not None:
                        inherited.update(parent_fact.dependencies)
                for index in value.element_indices:
                    index_fact = self._resolve_classical_fact(
                        index,
                        concrete=concrete,
                    )
                    inherited = _merge_source_guard_maps(
                        inherited,
                        index_fact.dependencies,
                    )
                return _fact_with_dependencies(fallback, inherited)
        return _ResolvedClassicalFact.create(self._resolve(value, concrete))

    def _resolve(self, v: Any, concrete: bool) -> sp.Expr:
        """Core resolution dispatcher (9-step priority chain).

        Args:
            v (Any): The value to resolve.
            concrete (bool): If ``True``, raise on symbolic results;
                if ``False``, produce ``sp.Symbol`` fallbacks.

        Returns:
            sp.Expr: Resolved expression.

        Raises:
            UnresolvedValueError: If *concrete* is ``True`` and the value
                cannot be resolved to a concrete integer.
        """
        # 1. Already SymPy
        if isinstance(v, sp.Basic):
            return cast(sp.Expr, v)

        # 2. Primitive Python types
        if not isinstance(v, Value):
            if isinstance(v, bool):
                return sp.Integer(1 if v else 0)
            if isinstance(v, int):
                return sp.Integer(v)
            if isinstance(v, float):
                return cast(sp.Expr, sp.Float(v))
            if concrete:
                raise UnresolvedValueError("?", f"Non-Value type: {type(v).__name__}")
            return sp.Symbol(str(v), integer=True, positive=True)

        # 3. UUID lookup in context
        if v.uuid in self._context:
            return self._resolve(self._context[v.uuid], concrete)

        # 4. Constant
        if v.is_constant():
            c = v.get_const()
            if c is not None:
                if isinstance(c, bool):
                    return sp.Integer(1 if c else 0)
                if isinstance(c, float):
                    return cast(sp.Expr, sp.Float(c))
                return sp.Integer(int(c))

        # Array element loads are encoded as Value provenance rather than a
        # dedicated operation. Try the parent array state before accepting an
        # element-shaped parameter label: inlining can retain the callee's
        # parameter metadata after substituting a concrete caller array.
        if v.is_array_element():
            element = self._resolve_array_element(v, concrete=concrete)
            if element is not None:
                return element

        # 5. Unbound parameter
        if v.is_parameter():
            pname = v.parameter_name()
            if pname is not None:
                if concrete:
                    raise UnresolvedValueError(v.uuid, f"Symbolic parameter '{pname}'")
                return _parameter_symbol(v, pname)

        # 6. Trace BinOp / CompOp in current block
        if self._block is not None:
            traced = self._trace(v, self._block, set(), concrete)
            if traced is not None:
                return traced

        # 7. Parent blocks
        for pb in reversed(self._parent_blocks):
            traced = self._trace(v, pb, set(), concrete)
            if traced is not None:
                return traced

        # 8. Public input shapes retain their stable names. Other unresolved
        # values use the complete UUID because display names are not identity
        # and canonical UUIDs commonly share long prefixes.
        if concrete:
            raise UnresolvedValueError(v.uuid, f"Unresolvable: '{v.name}'")
        shape_alias = self._input_shape_dimension_alias(v)
        if shape_alias is not None:
            return sp.Symbol(shape_alias, integer=True, nonnegative=True)
        return _fallback_symbol(v)

    def _resolve_array_element(
        self,
        value: Value,
        *,
        concrete: bool,
    ) -> sp.Expr | None:
        """Resolve one scalar element from its parent array state.

        Args:
            value (Value): Scalar value carrying ``parent_array`` and
                ``element_indices`` provenance.
            concrete (bool): Whether index/value resolution requires concrete
                expressions.

        Returns:
            sp.Expr | None: Stored or initialized element expression, or
                ``None`` when the array state cannot be proven.
        """
        fact = self._resolve_array_element_fact(value, concrete=concrete)
        return cast(sp.Expr, fact.value) if fact is not None else None

    def _resolve_array_element_fact(
        self,
        value: Value,
        *,
        concrete: bool,
    ) -> _ResolvedClassicalFact | None:
        """Resolve one scalar array element as a provenance fact.

        Args:
            value (Value): Scalar value carrying parent-array provenance.
            concrete (bool): Whether index and value resolution is concrete.

        Returns:
            _ResolvedClassicalFact | None: Precise projected fact, or ``None``
                when the persistent array state cannot explain the element.
        """
        parent = value.parent_array
        if parent is None or not value.element_indices:
            return None
        indices = tuple(
            self._resolve_classical_fact(index, concrete=concrete)
            for index in value.element_indices
        )
        return self._resolve_array_state_element_fact(
            parent,
            indices,
            concrete=concrete,
            visited=set(),
        )

    def _resolve_array_state_element(
        self,
        array: ArrayValue,
        indices: tuple[sp.Expr, ...],
        *,
        concrete: bool,
        visited: set[str],
        ignore_selection: str | None = None,
    ) -> sp.Expr | None:
        """Project one element from an immutable array SSA state.

        Args:
            array (ArrayValue): Array version whose contents are inspected.
            indices (tuple[sp.Expr, ...]): Resolved local element indices.
            concrete (bool): Whether nested scalar resolution is concrete.
            visited (set[str]): Array UUIDs already followed on this path.
            ignore_selection (str | None): Selection UUID to bypass once when
                its true source is the result array itself. Defaults to
                ``None``.

        Returns:
            sp.Expr | None: Proven element value, or ``None`` when no supported
                immutable source explains it.
        """
        facts = tuple(_ResolvedClassicalFact.create(index) for index in indices)
        fact = self._resolve_array_state_element_fact(
            array,
            facts,
            concrete=concrete,
            visited=visited,
            ignore_selection=ignore_selection,
        )
        return cast(sp.Expr, fact.value) if fact is not None else None

    def _resolve_array_state_element_fact(
        self,
        array: ArrayValue,
        indices: tuple[_ResolvedClassicalFact, ...],
        *,
        concrete: bool,
        visited: set[str],
        ignore_selection: str | None = None,
    ) -> _ResolvedClassicalFact | None:
        """Project one provenance fact from an immutable array SSA state.

        Args:
            array (ArrayValue): Array version whose contents are inspected.
            indices (tuple[_ResolvedClassicalFact, ...]): Local element-index
                facts.
            concrete (bool): Whether nested scalar resolution is concrete.
            visited (set[str]): Array UUIDs already followed on this path.
            ignore_selection (str | None): Binding UUID to bypass once for a
                raw reference. Defaults to ``None``.

        Returns:
            _ResolvedClassicalFact | None: Projected fact, or ``None`` when no
                supported immutable source explains the element.
        """
        state = self._array_context.get(array.uuid)
        if state is not None and ignore_selection != array.uuid:
            return self._project_array_state_fact(
                state,
                indices,
                concrete=concrete,
                visited=visited,
            )

        if array.uuid in visited:
            return None
        visited.add(array.uuid)

        if array.is_slice():
            if (
                len(indices) != 1
                or array.slice_of is None
                or array.slice_start is None
                or array.slice_step is None
            ):
                return None
            start = self._resolve_classical_fact(
                array.slice_start,
                concrete=concrete,
            )
            step = self._resolve_classical_fact(
                array.slice_step,
                concrete=concrete,
            )
            root_index = _fact_from_expression(
                cast(sp.Expr, start.value)
                + cast(sp.Expr, step.value) * cast(sp.Expr, indices[0].value),
                start,
                step,
                indices[0],
            )
            return self._resolve_array_state_element_fact(
                array.slice_of,
                (root_index,),
                concrete=concrete,
                visited=visited,
            )

        producer = self._array_producer(array)
        if isinstance(producer, IfOperation):
            merge = next(
                (
                    candidate
                    for candidate in producer.iter_merges()
                    if candidate.result.uuid == array.uuid
                    and isinstance(candidate.true_value, ArrayValue)
                    and isinstance(candidate.false_value, ArrayValue)
                ),
                None,
            )
            if merge is None:
                return None
            predicate = self._resolve_classical_fact(
                producer.condition,
                concrete=concrete,
            )
            true_value = self._resolve_array_state_element_fact(
                cast(ArrayValue, merge.true_value),
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            false_value = self._resolve_array_state_element_fact(
                cast(ArrayValue, merge.false_value),
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            if true_value is None or false_value is None:
                return None
            return _choice_classical_fact(true_value, false_value, predicate)
        if isinstance(producer, StoreArrayElementOperation):
            return self._project_array_state_fact(
                _ArrayStoreState(
                    _ArrayReferenceState(producer.array),
                    self._resolve_classical_fact(
                        producer.stored_value,
                        concrete=concrete,
                    ),
                    tuple(
                        self._resolve_classical_fact(index, concrete=concrete)
                        for index in producer.index_values
                    ),
                ),
                indices,
                concrete=concrete,
                visited=visited,
            )

        constant = array.get_const_array()
        if constant is None:
            return None
        constant_element = self._constant_array_element(
            constant,
            tuple(cast(sp.Expr, index.value) for index in indices),
        )
        if constant_element is None:
            return None
        return self._resolve_classical_fact(
            constant_element,
            concrete=concrete,
        )

    def _project_array_state(
        self,
        state: _ArrayState,
        indices: tuple[sp.Expr, ...],
        *,
        concrete: bool,
        visited: set[str],
    ) -> sp.Expr | None:
        """Project one scalar element from an immutable array state tree.

        Args:
            state (_ArrayState): Frozen state captured in its producing scope.
            indices (tuple[sp.Expr, ...]): Local element indices to project.
            concrete (bool): Whether nested scalar resolution is concrete.
            visited (set[str]): IR reference UUIDs already followed.

        Returns:
            sp.Expr | None: Projected scalar expression, or ``None`` when the
                reference state cannot prove a value.
        """
        facts = tuple(_ResolvedClassicalFact.create(index) for index in indices)
        fact = self._project_array_state_fact(
            state,
            facts,
            concrete=concrete,
            visited=visited,
        )
        return cast(sp.Expr, fact.value) if fact is not None else None

    def _project_array_state_fact(
        self,
        state: _ArrayState,
        indices: tuple[_ResolvedClassicalFact, ...],
        *,
        concrete: bool,
        visited: set[str],
    ) -> _ResolvedClassicalFact | None:
        """Project one scalar provenance fact from a persistent array state.

        Args:
            state (_ArrayState): Frozen state captured in its producing scope.
            indices (tuple[_ResolvedClassicalFact, ...]): Local element-index
                facts.
            concrete (bool): Whether nested scalar resolution is concrete.
            visited (set[str]): IR reference UUIDs already followed.

        Returns:
            _ResolvedClassicalFact | None: Projected fact, or ``None`` when a
                reference state cannot prove a value.
        """
        if isinstance(state, _ArrayReferenceState):
            return self._resolve_array_state_element_fact(
                state.array,
                indices,
                concrete=concrete,
                visited=visited,
                ignore_selection=state.array.uuid,
            )
        if isinstance(state, _ArrayConstantState):
            element = self._constant_array_element(
                state.contents,
                tuple(cast(sp.Expr, index.value) for index in indices),
            )
            if element is None:
                return None
            return self._resolve_classical_fact(element, concrete=concrete)
        if isinstance(state, _ArraySliceState):
            if len(indices) != 1:
                return None
            source_index = _fact_from_expression(
                cast(sp.Expr, state.start.value)
                + cast(sp.Expr, state.step.value) * cast(sp.Expr, indices[0].value),
                state.start,
                state.step,
                indices[0],
            )
            return self._project_array_state_fact(
                state.source,
                (source_index,),
                concrete=concrete,
                visited=visited,
            )
        if isinstance(state, _ArrayChoiceState):
            predicate = _as_boolean(state.condition.value)
            if predicate is sp.true:
                return self._project_array_state_fact(
                    state.when_true,
                    indices,
                    concrete=concrete,
                    visited=visited,
                )
            if predicate is sp.false:
                return self._project_array_state_fact(
                    state.when_false,
                    indices,
                    concrete=concrete,
                    visited=visited,
                )
            true_value = self._project_array_state_fact(
                state.when_true,
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            false_value = self._project_array_state_fact(
                state.when_false,
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            if true_value is None or false_value is None:
                return None
            return _choice_classical_fact(
                true_value,
                false_value,
                state.condition,
            )
        if isinstance(state, _ArrayLoopSummaryState):
            initial = self._project_array_state_fact(
                state.initial,
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            update = self._array_state_update_fact(
                state.iteration,
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            active_value = _as_boolean(
                _activation_over_range(
                    update.value,
                    state.loop_symbol,
                    state.start,
                    state.step,
                    state.iterations,
                )
            )
            if active_value is sp.false:
                return initial
            per_iteration = self._project_array_state_fact(
                state.iteration,
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            summarized_sources = {
                source: _as_boolean(
                    _activation_over_range(
                        guard,
                        state.loop_symbol,
                        state.start,
                        state.step,
                        state.iterations,
                    )
                )
                for source, guard in (
                    per_iteration.dependencies.items()
                    if per_iteration is not None
                    else ()
                )
            }
            update_sources = {
                source: _as_boolean(
                    _activation_over_range(
                        guard,
                        state.loop_symbol,
                        state.start,
                        state.step,
                        state.iterations,
                    )
                )
                for source, guard in update.dependencies.items()
            }
            selector = _ResolvedClassicalFact.create(
                active_value,
                update_sources,
            )
            active_fact = _ResolvedClassicalFact.create(
                state.fallback,
                _merge_source_guard_maps(
                    summarized_sources,
                    update_sources,
                    {state.uncertainty_token: active_value},
                ),
            )
            if initial is None:
                return active_fact
            return _choice_classical_fact(active_fact, initial, selector)
        if isinstance(state, _ArrayUnknownLoopSummaryState):
            active = _as_boolean(sp.Gt(state.iterations, 0))
            initial = self._project_array_state_fact(
                state.initial,
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            active_fact = _ResolvedClassicalFact.create(
                state.fallback,
                {state.uncertainty_token: active},
            )
            if initial is None:
                return active_fact
            return _choice_classical_fact(
                active_fact,
                initial,
                _ResolvedClassicalFact.create(active),
            )
        if not isinstance(state, _ArrayStoreState):
            return None
        if len(state.indices) != len(indices):
            return None
        guard = sp.And(
            *(
                sp.Eq(load_index.value, store_index.value)
                for load_index, store_index in zip(
                    indices,
                    state.indices,
                    strict=True,
                )
            )
        )
        if guard is sp.true:
            return state.stored
        previous = self._project_array_state_fact(
            state.previous,
            indices,
            concrete=concrete,
            visited=visited,
        )
        if guard is sp.false:
            return previous
        if previous is None:
            return None
        selector = _fact_from_expression(
            guard,
            *indices,
            *state.indices,
        )
        return _choice_classical_fact(state.stored, previous, selector)

    def _array_state_update_fact(
        self,
        state: _ArrayState,
        indices: tuple[_ResolvedClassicalFact, ...],
        *,
        concrete: bool,
        visited: set[str],
    ) -> _ResolvedClassicalFact:
        """Return when one body transition may update a projected element.

        Args:
            state (_ArrayState): One-iteration persistent state tree.
            indices (tuple[_ResolvedClassicalFact, ...]): Projected indices.
            concrete (bool): Whether nested resolution requires concrete data.
            visited (set[str]): Reference UUIDs already followed.

        Returns:
            _ResolvedClassicalFact: Boolean update guard and its observation
            dependencies.
        """
        if isinstance(state, (_ArrayReferenceState, _ArrayConstantState)):
            return _ResolvedClassicalFact.create(sp.false)
        if isinstance(state, _ArraySliceState):
            if len(indices) != 1:
                return _ResolvedClassicalFact.create(sp.true)
            source_index = _fact_from_expression(
                cast(sp.Expr, state.start.value)
                + cast(sp.Expr, state.step.value) * cast(sp.Expr, indices[0].value),
                state.start,
                state.step,
                indices[0],
            )
            return self._array_state_update_fact(
                state.source,
                (source_index,),
                concrete=concrete,
                visited=visited,
            )
        if isinstance(state, _ArrayChoiceState):
            predicate = _as_boolean(state.condition.value)
            true_update = self._array_state_update_fact(
                state.when_true,
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            false_update = self._array_state_update_fact(
                state.when_false,
                indices,
                concrete=concrete,
                visited=set(visited),
            )
            return _fact_from_expression(
                sp.Or(
                    sp.And(predicate, _as_boolean(true_update.value)),
                    sp.And(sp.Not(predicate), _as_boolean(false_update.value)),
                ),
                state.condition,
                true_update,
                false_update,
            )
        if isinstance(state, _ArrayLoopSummaryState):
            return _ResolvedClassicalFact.create(
                sp.Gt(state.iterations, 0),
                {state.uncertainty_token: sp.Gt(state.iterations, 0)},
            )
        if isinstance(state, _ArrayUnknownLoopSummaryState):
            active = sp.Gt(state.iterations, 0)
            return _ResolvedClassicalFact.create(
                active,
                {state.uncertainty_token: active},
            )
        if not isinstance(state, _ArrayStoreState):
            return _ResolvedClassicalFact.create(sp.true)
        previous = self._array_state_update_fact(
            state.previous,
            indices,
            concrete=concrete,
            visited=visited,
        )
        if len(state.indices) != len(indices):
            return _fact_from_expression(sp.true, previous, *state.indices, *indices)
        alias = sp.And(
            *(
                sp.Eq(load.value, store.value)
                for load, store in zip(indices, state.indices, strict=True)
            )
        )
        return _fact_from_expression(
            sp.Or(alias, _as_boolean(previous.value)),
            previous,
            *state.indices,
            *indices,
        )

    def _constant_array_element(
        self,
        contents: Any,
        indices: tuple[sp.Expr, ...],
    ) -> Any | None:
        """Return a proven element from immutable constant-array contents.

        A symbolic index is accepted only when every candidate at that axis is
        equal. This proves the result without choosing or guessing an index.

        Args:
            contents (Any): Frozen nested array payload.
            indices (tuple[sp.Expr, ...]): Resolved local indices.

        Returns:
            Any | None: Proven scalar element, or ``None`` when a symbolic
                index can select distinct values or the payload is malformed.
        """
        current = contents
        for index in indices:
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
                return None
            if isinstance(index, sp.Integer):
                position = int(index)
                if position < 0 or position >= len(current):
                    return None
                current = current[position]
                continue
            if not current:
                return None
            first = current[0]
            if any(candidate != first for candidate in current[1:]):
                return None
            current = first
        return current

    def _array_producer(self, array: ArrayValue) -> Operation | None:
        """Return the operation producing one array SSA version.

        Args:
            array (ArrayValue): Array value whose producer is requested.

        Returns:
            Operation | None: Producing operation in the current or an
                enclosing block, or ``None`` when the array is an input or
                initializer.
        """
        for block in (self._block, *reversed(self._parent_blocks)):
            if block is None:
                continue
            producer = self._producer_map(block).get(array.uuid)
            if producer is not None:
                return producer
            for nested in walk_operations(block.operations):
                if any(result.uuid == array.uuid for result in nested.results):
                    return nested
        return None

    def _input_shape_dimension_alias(self, value: Value) -> str | None:
        """Return the collision-free alias for an input-array dimension.

        Args:
            value (Value): Unresolved value considered for symbolic fallback.

        Returns:
            str | None: Stable input alias when ``value`` is an input-array
                dimension in the current or an enclosing block.
        """
        for block in (self._block, *reversed(self._parent_blocks)):
            if not isinstance(block, Block):
                continue
            alias = self._input_shape_alias_map(block).get(value.uuid)
            if alias is not None:
                return alias
        return None

    def _input_shape_alias_map(self, block: Block) -> dict[str, str]:
        """Return the cached input-dimension alias index for one block.

        Args:
            block (Block): Block whose immutable interface is indexed.

        Returns:
            dict[str, str]: Input-dimension UUID to collision-free public
                alias.
        """
        block_id = id(block)
        cached = self._input_shape_alias_maps.get(block_id)
        if cached is None or cached[0] is not block:
            aliases = input_shape_dimension_aliases(block)
            self._input_shape_alias_maps[block_id] = (block, aliases)
            return aliases
        return cached[1]

    def _trace(
        self, v: Value, block: Any, visited: set[int], concrete: bool
    ) -> sp.Expr | None:
        """Trace backward through *block* operations for the op producing *v*.

        Args:
            v (Value): The value whose defining operation is sought.
            block (Any): Block whose operations are scanned.
            visited (set[int]): ``id()``-based visited set for cycle
                prevention.
            concrete (bool): Passed through to :meth:`_resolve`.

        Returns:
            sp.Expr | None: Resolved expression if a supported defining
                classical operation was found; ``None`` otherwise.
        """
        vid = id(v)
        if vid in visited:
            return None
        visited.add(vid)

        op = self._producer_map(block).get(v.uuid)
        if isinstance(op, BinOp):
            left = self._resolve(op.operands[0], concrete)
            right = self._resolve(op.operands[1], concrete)
            assert op.kind is not None
            return _apply_binop(op.kind, left, right)

        if isinstance(op, CompOp):
            left = self._resolve(op.operands[0], concrete)
            right = self._resolve(op.operands[1], concrete)
            assert op.kind is not None
            return _apply_compop(op.kind, left, right)

        if isinstance(op, CondOp):
            left = self._resolve(op.operands[0], concrete)
            right = self._resolve(op.operands[1], concrete)
            assert op.kind is not None
            return _apply_condop(  # type: ignore[return-value]
                op.kind,
                left,
                right,
            )

        if isinstance(op, NotOp):
            operand = self._resolve(op.input, concrete)
            return sp.Not(_as_boolean(operand))  # type: ignore[return-value]

        if isinstance(op, UnaryMathOp):
            operand = self._resolve(op.input, concrete)
            assert op.kind is not None
            return _apply_unary_math(op.kind, operand)

        return None

    def _producer_map(self, block: Any) -> dict[str, Operation]:
        """Return the cached producer index for one block.

        Args:
            block (Any): Block-like object exposing an ``operations`` list.

        Returns:
            dict[str, Operation]: Result UUID to defining operation. All
                operation results are indexed, not only the first result.
        """
        block_id = id(block)
        cached = self._producer_maps.get(block_id)
        if cached is None or cached[0] is not block:
            producers = {
                result.uuid: operation
                for operation in block.operations
                for result in operation.results
            }
            self._producer_maps[block_id] = (block, producers)
            return producers
        return cached[1]


# ------------------------------------------------------------------ #
#  Module-level helpers                                               #
# ------------------------------------------------------------------ #


def _parameter_symbol(value: Value, name: str) -> sp.Symbol:
    """Create a symbol matching an IR parameter's scalar domain.

    Args:
        value (Value): Parameter value whose IR type defines assumptions.
        name (str): Public parameter name used for the symbol.

    Returns:
        sp.Symbol: A nonnegative integer for UInt/Bit, a real symbol for
            Float, or an unconstrained symbol for other value types.
    """
    if isinstance(value.type, FloatType):
        return sp.Symbol(name, real=True)
    if isinstance(value.type, (BitType, UIntType)):
        # Zero is a valid UInt/Bit value. Assuming strict positivity lets SymPy
        # erase ``value == 0`` branches and zero-trip width guards before a
        # later substitution can recover them.
        return sp.Symbol(name, integer=True, nonnegative=True)
    return sp.Symbol(name)


def _normalized_source_guards(
    source_guards: Mapping[str, sp.Basic],
) -> tuple[tuple[str, Boolean], ...]:
    """Normalize source guards into deterministic immutable storage.

    Args:
        source_guards (Mapping[str, sp.Basic]): Source-token activation guards.

    Returns:
        tuple[tuple[str, Boolean], ...]: Sorted nonfalse source guards.
    """
    normalized: list[tuple[str, Boolean]] = []
    for source, raw_guard in source_guards.items():
        guard = _as_boolean(raw_guard)
        if guard is not sp.false:
            normalized.append((source, guard))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _merge_source_guard_maps(
    *source_maps: Mapping[str, sp.Basic],
) -> dict[str, Boolean]:
    """Union guarded source dependencies by source-token identity.

    Args:
        *source_maps (Mapping[str, sp.Basic]): Source-token maps to combine.

    Returns:
        dict[str, Boolean]: Combined source guards using logical OR.
    """
    combined: dict[str, Boolean] = {}
    for source_map in source_maps:
        for source, raw_guard in source_map.items():
            guard = _as_boolean(raw_guard)
            combined[source] = _as_boolean(sp.Or(combined.get(source, sp.false), guard))
    return {
        source: guard for source, guard in combined.items() if guard is not sp.false
    }


def _guard_source_guards(
    source_guards: Mapping[str, sp.Basic],
    condition: sp.Basic,
) -> dict[str, Boolean]:
    """Conjoin one path condition with every source dependency.

    Args:
        source_guards (Mapping[str, sp.Basic]): Dependencies to guard.
        condition (sp.Basic): Path condition selecting those dependencies.

    Returns:
        dict[str, Boolean]: Dependencies active only under ``condition``.
    """
    predicate = _as_boolean(condition)
    return {
        source: _as_boolean(sp.And(predicate, _as_boolean(guard)))
        for source, guard in source_guards.items()
        if sp.And(predicate, _as_boolean(guard)) is not sp.false
    }


def _fact_with_dependencies(
    fact: _ResolvedClassicalFact,
    additions: Mapping[str, sp.Basic],
) -> _ResolvedClassicalFact:
    """Return one fact with additional source dependencies.

    Args:
        fact (_ResolvedClassicalFact): Existing resolved fact.
        additions (Mapping[str, sp.Basic]): Additional guarded sources.

    Returns:
        _ResolvedClassicalFact: Fact with merged immutable dependencies.
    """
    return _ResolvedClassicalFact.create(
        fact.value,
        _merge_source_guard_maps(fact.dependencies, additions),
    )


def _fact_from_expression(
    expression: sp.Basic,
    *sources: _ResolvedClassicalFact,
) -> _ResolvedClassicalFact:
    """Create a fact from sources that remain in the simplified expression.

    SymPy eagerly applies identities such as ``x & False == False`` and
    ``x * 0 == 0``. A source eliminated by such an identity no longer delays
    the result at runtime, so its readiness token must be removed as well.

    Args:
        expression (sp.Basic): Derived symbolic value.
        *sources (_ResolvedClassicalFact): Input facts used by the expression.

    Returns:
        _ResolvedClassicalFact: Expression with dependencies from inputs that
            still affect it.
    """
    expression_symbols = expression.free_symbols
    relevant_sources = (
        source
        for source in sources
        if source.value == expression
        or bool(source.value.free_symbols & expression_symbols)
    )
    return _ResolvedClassicalFact.create(
        expression,
        _merge_source_guard_maps(*(source.dependencies for source in relevant_sources)),
    )


def _coerce_classical_fact(
    value: sp.Basic | _ResolvedClassicalFact,
) -> _ResolvedClassicalFact:
    """Return an existing fact or wrap a dependency-free value.

    Args:
        value (sp.Basic | _ResolvedClassicalFact): Value or fact to normalize.

    Returns:
        _ResolvedClassicalFact: Normalized fact.
    """
    if isinstance(value, _ResolvedClassicalFact):
        return value
    return _ResolvedClassicalFact.create(value)


def _choice_classical_fact(
    when_true: _ResolvedClassicalFact,
    when_false: _ResolvedClassicalFact,
    selector: _ResolvedClassicalFact,
) -> _ResolvedClassicalFact:
    """Select between facts while preserving exact guarded dependencies.

    A selector does not affect the result when both projected facts are
    structurally identical. Otherwise branch dependencies are guarded by the
    selector and its negation, and the selector's own dependencies remain.

    Args:
        when_true (_ResolvedClassicalFact): Fact selected on a true predicate.
        when_false (_ResolvedClassicalFact): Fact selected on a false predicate.
        selector (_ResolvedClassicalFact): Predicate value and its sources.

    Returns:
        _ResolvedClassicalFact: Selected value with guarded dependencies.
    """
    if when_true == when_false:
        return when_true
    predicate = _as_boolean(selector.value)
    if predicate is sp.true:
        return _fact_with_dependencies(when_true, selector.dependencies)
    if predicate is sp.false:
        return _fact_with_dependencies(when_false, selector.dependencies)
    dependencies = _merge_source_guard_maps(
        _guard_source_guards(when_true.dependencies, predicate),
        _guard_source_guards(when_false.dependencies, sp.Not(predicate)),
        selector.dependencies,
    )
    value = sp.Piecewise(
        (when_true.value, predicate),
        (when_false.value, True),
    )
    return _ResolvedClassicalFact.create(value, dependencies)


def _fallback_symbol(value: Value) -> sp.Symbol:
    """Create the identity-qualified symbol for one unresolved IR value.

    Args:
        value (Value): Unresolved scalar IR value.

    Returns:
        sp.Symbol: Typed private symbol whose spelling includes the value UUID.
    """
    fallback_name = f"{value.name}_{value.uuid}"
    if isinstance(value.type, FloatType):
        return sp.Symbol(fallback_name, real=True)
    if isinstance(value.type, (BitType, UIntType)):
        return sp.Symbol(fallback_name, integer=True, nonnegative=True)
    return sp.Symbol(fallback_name)


_COMPOP_MAP = {
    CompOpKind.EQ: sp.Eq,
    CompOpKind.NEQ: sp.Ne,
    CompOpKind.LT: sp.Lt,
    CompOpKind.LE: sp.Le,
    CompOpKind.GT: sp.Gt,
    CompOpKind.GE: sp.Ge,
}


def _as_boolean(expression: sp.Basic) -> Boolean:
    """Convert a numeric or predicate expression to logical truthiness.

    Qamomile predicates follow Python scalar truthiness for compile-time
    numeric values. Symbolically, that means a non-Boolean expression is true
    exactly when it is nonzero.

    Args:
        expression (sp.Basic): Numeric or Boolean SymPy expression.

    Returns:
        Boolean: Boolean expression with the same truthiness.
    """
    if isinstance(expression, Boolean):
        return expression
    return sp.Ne(expression, 0)


def _apply_binop(kind: BinOpKind, left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """Apply binary arithmetic.

    Args:
        kind (BinOpKind): The arithmetic operation kind.
        left (sp.Expr): Left operand.
        right (sp.Expr): Right operand.

    Returns:
        sp.Expr: Result of applying the operation.

    Raises:
        ValueError: If *kind* is not in ``BINOP_TO_SYMPY``.
    """
    fn = BINOP_TO_SYMPY.get(kind)
    if fn is None:
        raise ValueError(f"Unknown BinOpKind: {kind}")
    return fn(left, right)


def _apply_unary_math(
    kind: UnaryMathOpKind,
    operand: sp.Expr,
) -> sp.Expr:
    """Apply one exact symbolic unary mathematical operation.

    Args:
        kind (UnaryMathOpKind): Mathematical operation kind.
        operand (sp.Expr): Symbolic numeric operand.

    Returns:
        sp.Expr: Exact SymPy expression.

    Raises:
        ValueError: If ``kind`` has no symbolic implementation.
    """
    fn = UNARY_MATH_TO_SYMPY.get(kind)
    if fn is None:
        raise ValueError(f"Unknown UnaryMathOpKind: {kind}")
    return fn(operand)


def _apply_compop(kind: CompOpKind, left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """Apply comparison operation.

    Args:
        kind (CompOpKind): The comparison operation kind.
        left (sp.Expr): Left operand.
        right (sp.Expr): Right operand.

    Returns:
        sp.Expr: SymPy relational expression (e.g. ``sp.Eq``, ``sp.Lt``).

    Raises:
        ValueError: If *kind* is not in ``_COMPOP_MAP``.
    """
    fn = _COMPOP_MAP.get(kind)
    if fn is None:
        raise ValueError(f"Unknown CompOpKind: {kind}")
    return fn(left, right)  # type: ignore[return-value]


def _apply_condop(
    kind: CondOpKind,
    left: sp.Basic,
    right: sp.Basic,
) -> Boolean:
    """Apply a symbolic logical AND or OR operation.

    Args:
        kind (CondOpKind): Logical operation kind.
        left (sp.Basic): Left numeric or Boolean operand.
        right (sp.Basic): Right numeric or Boolean operand.

    Returns:
        Boolean: SymPy Boolean expression.

    Raises:
        ValueError: If ``kind`` has no symbolic implementation.
    """
    match kind:
        case CondOpKind.AND:
            return sp.And(_as_boolean(left), _as_boolean(right))
        case CondOpKind.OR:
            return sp.Or(_as_boolean(left), _as_boolean(right))
        case _:
            raise ValueError(f"Unknown CondOpKind: {kind}")
