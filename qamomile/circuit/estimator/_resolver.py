"""Resolve scoped IR values into symbolic resource expressions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._array_context import _ArrayContext
from qamomile.circuit.estimator._array_projection import _ArrayProjector
from qamomile.circuit.estimator._array_state import (
    _ArrayState as _ArrayState,
)
from qamomile.circuit.estimator._classical_expression import (
    _fallback_symbol as _fallback_symbol,
    _parameter_symbol as _parameter_symbol,
)
from qamomile.circuit.estimator._classical_facts import (
    _choice_classical_fact as _choice_classical_fact,
    _fact_with_dependencies as _fact_with_dependencies,
    _merge_source_guard_maps as _merge_source_guard_maps,
    _ResolvedClassicalFact as _ResolvedClassicalFact,
)
from qamomile.circuit.estimator._classical_trace import _trace_classical_value
from qamomile.circuit.estimator._resolver_indices import (
    _compute_input_shape_dimension_aliases,
    _ResolverBlockIndex,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import CallTransform
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands


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
    return _compute_input_shape_dimension_aliases(block)


class UnresolvedValueError(Exception):
    """A value cannot be concretized during resource estimation."""

    def __init__(self, uuid: str, message: str = "") -> None:
        """Initialize an unresolved-value diagnostic.

        Args:
            uuid (str): Identity of the IR value that could not be resolved.
            message (str): Optional diagnostic message. Defaults to a message
                containing ``uuid``.
        """
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
        "_array_projector",
        "_block",
        "_block_index",
        "_classical_fact_context",
        "_context",
        "_loop_var_names",
        "_parent_blocks",
        "_structural_scope",
    )

    def __init__(
        self,
        block: Any = None,
        context: dict[str, sp.Expr] | None = None,
        loop_var_names: dict[str, sp.Expr] | None = None,
        parent_blocks: list[Any] | None = None,
        block_index: _ResolverBlockIndex | None = None,
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
            block_index (_ResolverBlockIndex | None): Shared immutable-block
                index for producer and input-shape lookups. Child resolvers
                reuse one owner so every block is indexed at most once.
                Defaults to ``None``, which creates a new index owner.
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
        self._block = block
        self._block_index = block_index or _ResolverBlockIndex()
        self._classical_fact_context = dict(classical_fact_context or {})
        self._context: dict[str, sp.Expr] = dict(context or {})
        self._loop_var_names: dict[str, sp.Expr] = dict(loop_var_names or {})
        self._parent_blocks: list[Any] = list(parent_blocks or [])
        self._structural_scope = structural_scope or ()
        self._array_context = _ArrayContext(
            self.resolve_classical_fact,
            self._array_producer,
            array_context,
        )
        self._array_projector = _ArrayProjector(
            self._array_context.get,
            lambda value, concrete: self._resolve_classical_fact(
                value,
                concrete=concrete,
            ),
            self._array_producer,
        )

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
            block_index=self._block_index,
            structural_scope=self._structural_scope,
            array_context=self._array_context.shared_states(),
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
            block_index=self._block_index,
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
        """Delegate a branch-selected array binding to the array owner.

        Args:
            result (ArrayValue): Array SSA version visible after selection.
            when_true (ArrayValue): Source array selected when ``condition``
                is true.
            when_false (ArrayValue): Source array selected when ``condition``
                is false.
            condition (sp.Basic | _ResolvedClassicalFact): Predicate selecting
                the source array, optionally with source-token provenance.
        """
        self._array_context.bind_selection(
            result,
            when_true,
            when_false,
            condition,
        )

    def bind_array_state(self, result: ArrayValue, state: _ArrayState) -> None:
        """Delegate an immutable array-state binding to the array owner.

        Args:
            result (ArrayValue): Array SSA value receiving the snapshot.
            state (_ArrayState): Frozen state resolved in the producing scope.
        """
        self._array_context.bind_state(result, state)

    def bind_array_state_selection(
        self,
        result: ArrayValue,
        when_true: _ArrayState,
        when_false: _ArrayState,
        selector: _ResolvedClassicalFact,
    ) -> None:
        """Delegate detached branch snapshots to the array owner.

        Args:
            result (ArrayValue): Array SSA result receiving the selected state.
            when_true (_ArrayState): Detached true-branch snapshot.
            when_false (_ArrayState): Detached false-branch snapshot.
            selector (_ResolvedClassicalFact): Branch selector and its source
                dependencies.
        """
        self._array_context.bind_state_selection(
            result,
            when_true,
            when_false,
            selector,
        )

    def copy_array_context(self) -> None:
        """Detach this resolver from a shared mutable array-context mapping."""
        self._array_context.detach()

    def fork_array_context(self) -> dict[str, _ArrayState]:
        """Return a detached shallow copy for a child resolver scope.

        The state nodes are immutable, so copying only the UUID map is enough
        to isolate later bindings while retaining structural sharing.

        Returns:
            dict[str, _ArrayState]: Detached array-state mapping.
        """
        return self._array_context.fork()

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
        return self._array_context.export(arrays)

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
        self._array_context.import_states(context, replace=replace)

    def bind_call_array_input(
        self,
        block: Block,
        formal: ArrayValue,
        state: _ArrayState,
    ) -> None:
        """Delegate callable-entry array aliasing to the array owner.

        Args:
            block (Block): Selected callable body.
            formal (ArrayValue): Array value paired with the call operand.
            state (_ArrayState): Caller state captured at invocation time.
        """
        self._array_context.bind_call_input(block, formal, state)

    def bind_loop_array_input(
        self,
        operations: Sequence[Operation],
        entry: ArrayValue,
        state: _ArrayState,
    ) -> None:
        """Delegate loop-entry array aliasing to the array owner.

        Args:
            operations (Sequence[Operation]): Loop-body operations.
            entry (ArrayValue): Pre-loop array value naming the carried lineage.
            state (_ArrayState): Snapshot produced by the previous iteration.
        """
        self._array_context.bind_loop_input(operations, entry, state)

    def snapshot_array_state(
        self,
        array: ArrayValue,
        *,
        ignore_binding: str | None = None,
        visited: set[str] | None = None,
    ) -> _ArrayState:
        """Delegate immutable array-state capture to the array owner.

        Args:
            array (ArrayValue): Array whose current state is captured.
            ignore_binding (str | None): Array binding bypassed for one raw
                producer lookup. Defaults to ``None``.
            visited (set[str] | None): Array UUIDs already visited on this
                capture path. Defaults to ``None``.

        Returns:
            _ArrayState: Immutable state tree rooted at ``array``.
        """
        return self._array_context.snapshot(
            array,
            ignore_binding=ignore_binding,
            visited=visited,
        )

    def array_state_dependencies(self, array: ArrayValue) -> dict[str, Boolean]:
        """Delegate whole-array dependency summarization to the array owner.

        Args:
            array (ArrayValue): Array whose immutable state is summarized.

        Returns:
            dict[str, Boolean]: Retained source tokens and activation guards.
        """

        return self._array_context.dependencies(array)

    def guard_array_update(
        self,
        result: ArrayValue,
        previous: ArrayValue,
        condition: sp.Basic | _ResolvedClassicalFact,
    ) -> None:
        """Delegate one execution-guarded update to the array owner.

        Args:
            result (ArrayValue): Array SSA version produced by the store.
            previous (ArrayValue): Array version read by the store.
            condition (sp.Basic | _ResolvedClassicalFact): Predicate that the
                enclosing region executes, optionally with provenance.
        """
        self._array_context.guard_update(result, previous, condition)

    def record_array_store(self, operation: StoreArrayElementOperation) -> None:
        """Delegate one sequential store record to the array owner.

        Args:
            operation (StoreArrayElementOperation): Store just encountered by
                the estimator's sequential interpreter.
        """
        self._array_context.record_store(operation)

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
        """Resolve one scalar element through the array projector.

        Args:
            value (Value): Scalar value carrying parent-array provenance.
            concrete (bool): Whether index and value resolution is concrete.

        Returns:
            sp.Expr | None: Projected expression, or ``None`` when the
                persistent array state cannot explain the element.
        """
        return self._array_projector.resolve_element(value, concrete=concrete)

    def _resolve_array_element_fact(
        self,
        value: Value,
        *,
        concrete: bool,
    ) -> _ResolvedClassicalFact | None:
        """Resolve one scalar array element through the array projector.

        Args:
            value (Value): Scalar value carrying parent-array provenance.
            concrete (bool): Whether index and value resolution is concrete.

        Returns:
            _ResolvedClassicalFact | None: Precise projected fact, or ``None``
                when the persistent array state cannot explain the element.
        """
        return self._array_projector.resolve_element_fact(
            value,
            concrete=concrete,
        )

    def _array_producer(self, array: ArrayValue) -> Operation | None:
        """Return the operation producing one array SSA version.

        Args:
            array (ArrayValue): Array value whose producer is requested.

        Returns:
            Operation | None: Producing operation in the current or an
                enclosing block, or ``None`` when the array is an input or
                initializer.
        """
        return self._block_index.array_producer(
            array,
            (self._block, *reversed(self._parent_blocks)),
        )

    def _input_shape_dimension_alias(self, value: Value) -> str | None:
        """Return the collision-free alias for an input-array dimension.

        Args:
            value (Value): Unresolved value considered for symbolic fallback.

        Returns:
            str | None: Stable input alias when ``value`` is an input-array
                dimension in the current or an enclosing block.
        """
        return self._block_index.input_shape_dimension_alias(
            value,
            (self._block, *reversed(self._parent_blocks)),
        )

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
        return _trace_classical_value(
            v,
            block,
            visited,
            concrete,
            resolve=self._resolve,
            producer_map=self._block_index.producer_map,
        )
