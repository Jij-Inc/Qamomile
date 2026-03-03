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

from qamomile.circuit.ir.block_value import BlockValue
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
)
from qamomile.circuit.ir.value import ArrayValue, Value

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
      6. Loop variable (name-based)               → return mapped symbol
      7. BinOp/CompOp result                      → trace in block operations
      8. Search parent blocks                     → trace in ancestors
      9. Fallback                                 → sp.Symbol (symbolic) or raise (concrete)
    """

    __slots__ = ("_block", "_context", "_loop_var_names", "_parent_blocks")

    def __init__(
        self,
        block: Any = None,
        context: dict[str, sp.Expr] | None = None,
        loop_var_names: dict[str, sp.Symbol] | None = None,
        parent_blocks: list[Any] | None = None,
    ):
        self._block = block
        self._context: dict[str, sp.Expr] = dict(context or {})
        self._loop_var_names: dict[str, sp.Symbol] = dict(loop_var_names or {})
        self._parent_blocks: list[Any] = list(parent_blocks or [])

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def resolve(self, v: Any) -> sp.Expr:
        """Convert IR Value to SymPy expression (symbolic mode).

        Unbound parameters become ``sp.Symbol``.  Never raises for valid IR.
        """
        return self._resolve(v, concrete=False)

    def resolve_concrete(self, v: Any) -> int:
        """Convert IR Value to concrete ``int``.

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
        extra_loop_vars: dict[str, sp.Symbol] | None = None,
    ) -> ExprResolver:
        """Create a child resolver for an inner scope (loop body, branch).

        Propagates parent_blocks so values from outer scopes remain
        traceable.  For *callee* scopes (CallBlockOperation), use
        :meth:`call_child_scope` instead — callees get a fresh scope.
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

    def call_child_scope(self, call_op: Any) -> ExprResolver:
        """Create a child resolver for a ``CallBlockOperation``.

        Maps formal parameter UUIDs → resolved actual arguments,
        **including array shape dimension UUIDs** (critical for
        resolving e.g. ``kernel.shape[0]`` inside the callee).

        Parent blocks are intentionally reset — the callee only sees
        its own scope plus values propagated through ``call_context``.
        """
        called_block = call_op.operands[0]
        if not isinstance(called_block, BlockValue):
            # Not a BlockValue — use child_scope as fallback
            return self.child_scope(called_block)

        extra: dict[str, sp.Expr] = {}
        for i, formal in enumerate(called_block.input_values):
            if i + 1 >= len(call_op.operands):
                break
            actual = call_op.operands[i + 1]
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

    # Read-only accessors for engine / accumulator use

    @property
    def context(self) -> dict[str, sp.Expr]:
        return self._context.copy()

    @property
    def loop_var_names(self) -> dict[str, sp.Symbol]:
        return self._loop_var_names.copy()

    @property
    def block(self) -> Any:
        return self._block

    # ------------------------------------------------------------------ #
    #  Internal resolution                                                #
    # ------------------------------------------------------------------ #

    def _resolve(self, v: Any, concrete: bool) -> sp.Expr:
        # 1. Already SymPy
        if isinstance(v, sp.Basic):
            return v

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
                return sp.Symbol(pname, integer=True, positive=True)

        # 6. Loop variable (name-based lookup)
        if v.name in self._loop_var_names:
            return self._loop_var_names[v.name]

        # 7. Trace BinOp / CompOp in current block
        if self._block is not None:
            traced = self._trace(v, self._block, set(), concrete)
            if traced is not None:
                return traced

        # 8. Parent blocks
        for pb in reversed(self._parent_blocks):
            traced = self._trace(v, pb, set(), concrete)
            if traced is not None:
                return traced

        # 9. Fallback
        if concrete:
            raise UnresolvedValueError(v.uuid, f"Unresolvable: '{v.name}'")
        return sp.Symbol(v.name, integer=True, positive=True)

    def _trace(
        self, v: Value, block: Any, visited: set[int], concrete: bool
    ) -> sp.Expr | None:
        """Trace backward through *block* operations for the op producing *v*."""
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
                return _apply_binop(op.kind, left, right)

            if isinstance(op, CompOp):
                left = self._resolve(op.operands[0], concrete)
                right = self._resolve(op.operands[1], concrete)
                return _apply_compop(op.kind, left, right)

        return None


# ------------------------------------------------------------------ #
#  Module-level helpers                                               #
# ------------------------------------------------------------------ #

_COMPOP_MAP = {
    CompOpKind.EQ: sp.Eq,
    CompOpKind.NEQ: sp.Ne,
    CompOpKind.LT: sp.Lt,
    CompOpKind.LE: sp.Le,
    CompOpKind.GT: sp.Gt,
    CompOpKind.GE: sp.Ge,
}


def _apply_binop(kind: BinOpKind, left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """Apply binary arithmetic with smart FLOORDIV.

    For FLOORDIV: avoids ``sp.floor()`` when the quotient simplifies to
    an obviously-integer form (Integer, Symbol, Pow with non-negative
    exponent), enabling cleaner symbolic expressions like
    ``2**m / 2**i = 2**(m-i)``.
    """
    if kind == BinOpKind.FLOORDIV:
        quotient = sp.simplify(left / right)
        if isinstance(quotient, (sp.Integer, sp.Symbol)):
            return quotient
        if isinstance(quotient, sp.Pow):
            # Only skip floor for Pow with non-negative exponent
            # e.g., 2**(m-i) is fine, but n**(-1) = 1/n is not integer
            base, exp = quotient.as_base_exp()
            if exp.is_nonnegative is not False:
                return quotient
        return sp.floor(left / right)
    fn = BINOP_TO_SYMPY.get(kind)
    if fn is None:
        raise ValueError(f"Unknown BinOpKind: {kind}")
    return fn(left, right)


def _apply_compop(kind: CompOpKind, left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """Apply comparison operation."""
    fn = _COMPOP_MAP.get(kind)
    if fn is None:
        raise ValueError(f"Unknown CompOpKind: {kind}")
    return fn(left, right)
