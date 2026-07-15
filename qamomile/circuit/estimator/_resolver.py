"""Unified value resolution for resource estimation.

ExprResolver converts IR Values to SymPy expressions, providing a single
source of truth for all estimators (gate counting, qubits).

Two-mode API:
  resolve()          — symbolic; unbound parameters → sp.Symbol
  resolve_concrete() — concrete; must return int, raises on symbolic
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
)
from qamomile.circuit.ir.operation.callable import CallTransform
from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands

from ._utils import BINOP_TO_SYMPY


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
      3. UUID in context (call_context / BinOp)   → return mapped expression
      4. Constant value                           → sp.Integer / sp.Float
      5. Unbound parameter                        → sp.Symbol (symbolic) or raise (concrete)
      6. BinOp/CompOp result                      → trace in block operations
      7. Search parent blocks                     → trace in ancestors
      8. Fallback                                 → identity-qualified symbol or raise
    """

    __slots__ = ("_block", "_context", "_loop_var_names", "_parent_blocks")

    def __init__(
        self,
        block: Any = None,
        context: dict[str, sp.Expr] | None = None,
        loop_var_names: dict[str, sp.Expr] | None = None,
        parent_blocks: list[Any] | None = None,
    ):
        """Initialise an ExprResolver.

        Args:
            block (Any): The current block (Block or _LocalBlock)
                whose operations are searched for BinOp/CompOp traces.
            context (dict[str, sp.Expr] | None): UUID → resolved expression
                mapping for values passed across scope boundaries (e.g.
                call arguments, composite-gate operands).
            loop_var_names (dict[str, sp.Expr] | None): Value name → SymPy
                expression mapping for loop variables in scope.
            parent_blocks (list[Any] | None): Ancestor blocks to search
                when tracing fails in the current block.
        """
        self._block = block
        self._context: dict[str, sp.Expr] = dict(context or {})
        self._loop_var_names: dict[str, sp.Expr] = dict(loop_var_names or {})
        self._parent_blocks: list[Any] = list(parent_blocks or [])

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
        )

    def call_child_scope(
        self,
        call_op: Any,
        *,
        called_block: Block | None = None,
        body_implements_transform: bool = False,
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

        actual_operands = call_op.operands
        if (
            getattr(call_op, "transform", None) is CallTransform.CONTROLLED
            and not body_implements_transform
        ):
            actual_operands = actual_operands[call_op.num_control_qubits :]

        extra: dict[str, sp.Expr] = {}
        for formal, actual in pair_block_operands(called_block, actual_operands):
            extra[formal.uuid] = self.resolve(actual)
            # Map array shape dimension UUIDs
            if isinstance(actual, ArrayValue) and isinstance(formal, ArrayValue):
                for df, da in zip(formal.shape, actual.shape):
                    extra[df.uuid] = self.resolve(da)

        # Callee gets fresh scope — no parent blocks from caller
        ctx = self._context.copy()
        ctx.update(extra)
        return ExprResolver(
            block=called_block,
            context=ctx,
            loop_var_names=self._loop_var_names.copy(),
            parent_blocks=[],
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
        if self._is_input_shape_dimension(v):
            return sp.Symbol(v.name, integer=True, nonnegative=True)
        fallback_name = f"{v.name}_{v.uuid}"
        if isinstance(v.type, FloatType):
            return sp.Symbol(fallback_name, real=True)
        if isinstance(v.type, (BitType, UIntType)):
            return sp.Symbol(fallback_name, integer=True, nonnegative=True)
        return sp.Symbol(fallback_name)

    def _is_input_shape_dimension(self, value: Value) -> bool:
        """Return whether a value is a public input-array dimension.

        Args:
            value (Value): Unresolved value considered for symbolic fallback.

        Returns:
            bool: Whether ``value`` appears in an input array's shape in the
                current or an enclosing block.
        """
        for block in (self._block, *reversed(self._parent_blocks)):
            if not isinstance(block, Block):
                continue
            for input_value in block.input_values:
                if isinstance(input_value, ArrayValue) and any(
                    dimension.uuid == value.uuid for dimension in input_value.shape
                ):
                    return True
        return False

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
            sp.Expr | None: Resolved expression if a defining BinOp or
                CompOp was found; ``None`` otherwise.
        """
        vid = id(v)
        if vid in visited:
            return None
        visited.add(vid)

        for op in block.operations:
            if not hasattr(op, "results") or not op.results:
                continue
            if op.results[0] != v:
                continue

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

        return None


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
