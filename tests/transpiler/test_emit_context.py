"""Tests for ``EmitContext`` — typed bindings container.

The pre-EmitContext design used a single ``bindings: dict[str, Any]`` for
five distinct semantic categories (params, loop vars, intermediates, phi
aliases, runtime exprs). This caused name-collision bugs (e.g.
``"bit_tmp"`` overwrites between chained predicates). ``EmitContext`` is
a ``dict`` subclass that adds typed methods and slot tracking so writers
can express intent and so structural inspection (``ctx._values`` etc.)
becomes possible without parsing the flat dict.

These tests verify:
- Backward dict-compat (``ctx[k] = v`` and ``ctx[k]`` work).
- Typed writers (``bind_param``, ``push_loop_var``, ``set_value``,
  ``set_runtime_expr``) update both the slot and the flat view.
- ``copy()`` preserves all slots (loop unrollers depend on this).
"""

from __future__ import annotations

from qamomile.circuit.transpiler.emit_context import EmitContext


class TestDictCompat:
    """EmitContext must work as a drop-in for ``dict[str, Any]``."""

    def test_subclass_of_dict(self):
        ctx = EmitContext()
        assert isinstance(ctx, dict)

    def test_flat_writes_and_reads(self):
        ctx = EmitContext()
        ctx["theta"] = 0.5
        assert ctx["theta"] == 0.5
        assert "theta" in ctx
        assert list(ctx.keys()) == ["theta"]

    def test_from_user_bindings_seeds_params(self):
        ctx = EmitContext.from_user_bindings({"theta": 0.5, "n": 3})
        # Flat access works
        assert ctx["theta"] == 0.5
        assert ctx["n"] == 3
        # And the params slot is populated
        assert ctx._params == {"theta": 0.5, "n": 3}

    def test_from_user_bindings_none_yields_empty(self):
        ctx = EmitContext.from_user_bindings(None)
        assert len(ctx) == 0
        assert ctx._params == {}


class TestTypedWriters:
    """Each typed writer updates both its slot and the flat dict view."""

    def test_bind_param(self):
        ctx = EmitContext()
        ctx.bind_param("phi", 1.5)
        assert ctx["phi"] == 1.5
        assert ctx._params == {"phi": 1.5}
        # Other slots untouched
        assert ctx._loop_vars == {}
        assert ctx._values == {}

    def test_bind_params_multiple(self):
        ctx = EmitContext()
        ctx.bind_params({"a": 1, "b": 2})
        assert ctx._params == {"a": 1, "b": 2}
        assert ctx["a"] == 1 and ctx["b"] == 2

    def test_push_loop_var(self):
        ctx = EmitContext()
        ctx.push_loop_var("j", 3)
        assert ctx["j"] == 3
        assert ctx._loop_vars == {"j": 3}
        assert ctx._params == {}

    def test_set_value_uuid_keyed(self):
        ctx = EmitContext()
        uuid = "abc-123"
        ctx.set_value(uuid, 42)
        assert ctx[uuid] == 42
        assert ctx._values == {uuid: 42}

    def test_set_runtime_expr(self):
        ctx = EmitContext()
        uuid = "expr-uuid"

        # A stand-in for a backend Expr object (any non-scalar Python obj works).
        class _FakeExpr:
            pass

        expr = _FakeExpr()
        ctx.set_runtime_expr(uuid, expr)
        assert ctx[uuid] is expr
        assert ctx._runtime_exprs == {uuid: expr}


class TestCopySemantics:
    """``copy()`` is critical for loop unrollers — must preserve slots."""

    def test_copy_returns_emit_context(self):
        ctx = EmitContext()
        ctx.bind_param("a", 1)
        c = ctx.copy()
        assert isinstance(c, EmitContext)

    def test_copy_preserves_all_slots(self):
        ctx = EmitContext()
        ctx.bind_param("a", 1)
        ctx.push_loop_var("j", 5)
        ctx.set_value("uuid-1", 42)
        ctx.set_runtime_expr("uuid-2", object())

        c = ctx.copy()
        assert c._params == ctx._params
        assert c._loop_vars == ctx._loop_vars
        assert c._values == ctx._values
        assert c._runtime_exprs == ctx._runtime_exprs

    def test_copy_is_independent(self):
        """Mutations on the copy must not bleed back to the parent
        (loop unrollers rely on this — each iteration has its own scope)."""
        ctx = EmitContext()
        ctx.bind_param("a", 1)
        c = ctx.copy()
        c.push_loop_var("j", 7)
        c.bind_param("a", 99)  # override

        assert ctx._loop_vars == {}  # parent unchanged
        assert ctx._params == {"a": 1}  # parent unchanged
        assert c._params == {"a": 99}  # copy reflects override


class TestDescribe:
    """``describe()`` is for debugging — exists and produces something."""

    def test_describe_includes_all_slots(self):
        ctx = EmitContext()
        ctx.bind_param("theta", 0.5)
        ctx.set_value("uuid-1", 42)
        out = ctx.describe()
        assert "params" in out
        assert "values" in out
        assert "theta" in out
