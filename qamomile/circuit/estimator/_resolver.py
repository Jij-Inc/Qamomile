"""Unified value resolution for resource estimation.

ExprResolver converts IR Values to SymPy expressions, providing a single
source of truth for all estimators (gate counting, qubits).

Two-mode API:
  resolve()          — symbolic; unbound parameters → sp.Symbol
  resolve_concrete() — concrete; must return int, raises on symbolic
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.ir.block import Block
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
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands

from ._utils import BINOP_TO_SYMPY, UNARY_MATH_TO_SYMPY


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
        "_block",
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
        """
        self._block = block
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
        context = self._context.copy()
        if extra_context:
            context.update(extra_context)
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
        for formal, actual in pair_block_operands(called_block, actual_operands):
            extra[formal.uuid] = self.resolve(actual)
            # Map array shape dimension UUIDs
            if isinstance(actual, ArrayValue) and isinstance(formal, ArrayValue):
                for df, da in zip(formal.shape, actual.shape):
                    extra[df.uuid] = self.resolve(da)

        # Callee gets fresh scope — no parent blocks from caller.
        return self.isolated_scope(
            called_block,
            extra,
            structural_scope=self.call_structural_scope(call_op, called_block),
        )

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
            return v  # type: ignore[return-value]

        # 2. Primitive Python types
        if not isinstance(v, Value):
            if isinstance(v, bool):
                return sp.Integer(1 if v else 0)
            if isinstance(v, int):
                return sp.Integer(v)
            if isinstance(v, float):
                return sp.Float(v)
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
                    return sp.Float(c)
                return sp.Integer(int(c))

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
        fallback_name = f"{v.name}_{v.uuid}"
        if isinstance(v.type, FloatType):
            return sp.Symbol(fallback_name, real=True)
        if isinstance(v.type, (BitType, UIntType)):
            return sp.Symbol(fallback_name, integer=True, nonnegative=True)
        return sp.Symbol(fallback_name)

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
