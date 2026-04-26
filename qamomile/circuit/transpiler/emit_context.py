"""Typed bindings container for the emit pipeline.

Background — what was wrong with the bare ``dict[str, Any]``:

The pre-EmitContext design used a single ``bindings: dict[str, Any]``
threaded through every emit-pipeline function. That dict served at least
five distinct semantic purposes simultaneously:

1. User-supplied kernel parameters (keyed by parameter name).
2. Loop iteration variables (keyed by loop_var name; pushed on entry,
   restored on exit).
3. Emit-time-computed intermediates — ``BinOp`` / ``CompOp`` /
   ``CondOp`` / ``NotOp`` results (keyed by Value UUID after Fix B;
   originally also keyed by Value name, which collided across tmps).
4. Phi-output aliases (keyed by phi-output UUID; written by
   ``register_classical_phi_aliases``).
5. Backend runtime expressions (e.g. ``qiskit.circuit.classical.expr.Expr``
   for compound runtime if-conditions).

This overloading was the structural cause of two bugs landed earlier in
this session: the ``"bit_tmp"`` name-collision in chained predicates
and the ``j_phi_4`` phi-alias that emit-time loop unrolling could not
bind. Both were patched, but the underlying mess (one dict, five roles)
remained.

What ``EmitContext`` does:

``EmitContext`` is a ``dict`` subclass — flat ``[key]`` access still
works, so existing emit-pipeline code (which expects a dict) is
unchanged. On top of dict semantics, ``EmitContext`` keeps separate
semantic slots (``_params``, ``_loop_vars``, ``_values``,
``_runtime_exprs``) and exposes typed methods (``bind_param``,
``bind_params``, ``push_loop_var``, ``set_value``,
``set_runtime_expr``) that update both the slot and the flat view.

Migration is incremental: new writers use the typed methods (clearer
intent, slot-tagged data is easier to inspect when debugging); existing
writers stay on ``ctx[key] = value`` for now and can be migrated PR by
PR.
"""

from __future__ import annotations

from typing import Any, Iterator


class EmitContext(dict):
    """Bindings container with semantic slots, dict-compatible.

    All emit-pipeline functions that take ``bindings: dict[str, Any]``
    accept an ``EmitContext`` unchanged because it inherits from ``dict``.

    Use the typed methods (``bind_param``, ``set_value``, etc.) when
    writing new code so the slot tracking stays accurate; existing
    ``ctx[key] = value`` writes still work but bypass the slots.

    Slots:
        _params: User-supplied kernel parameters, keyed by parameter
            name. Stable across the run.
        _loop_vars: Currently-bound loop iteration variables, keyed by
            ``loop_var``/``key_vars``/``value_var`` name. Pushed on
            loop entry, restored on exit.
        _values: Emit-time-computed intermediate values (``BinOp``
            results, ``CompOp``/``CondOp``/``NotOp`` results, phi
            aliases), keyed by Value UUID.
        _runtime_exprs: Backend runtime-expression objects (e.g. Qiskit
            ``expr.Expr`` for compound classical conditions), keyed by
            Value UUID.

    Example:
        >>> ctx = EmitContext.from_user_bindings({"theta": 0.5, "n": 3})
        >>> ctx["theta"]  # dict-style read still works
        0.5
        >>> ctx.bind_param("phi", 1.5)
        >>> "phi" in ctx and ctx["phi"] == 1.5
        True
        >>> ctx.set_value(some_uuid, 42)
        >>> ctx[some_uuid] == 42 and some_uuid in ctx._values
        True
    """

    __slots__ = ("_params", "_loop_vars", "_values", "_runtime_exprs")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._params: dict[str, Any] = {}
        self._loop_vars: dict[str, Any] = {}
        self._values: dict[str, Any] = {}
        self._runtime_exprs: dict[str, Any] = {}

    # -- Construction ------------------------------------------------------

    @classmethod
    def from_user_bindings(
        cls, user_bindings: dict[str, Any] | None
    ) -> "EmitContext":
        """Build an EmitContext seeded with user-supplied parameters.

        Args:
            user_bindings: The dict passed by the user to ``transpile()``;
                ``None`` is treated as empty.

        Returns:
            A fresh EmitContext with all entries registered as parameters.
        """
        ctx = cls()
        if user_bindings:
            ctx.bind_params(user_bindings)
        return ctx

    # -- Typed writers -----------------------------------------------------

    def bind_param(self, name: str, value: Any) -> None:
        """Register a kernel parameter binding (by name)."""
        self._params[name] = value
        self[name] = value

    def bind_params(self, params: dict[str, Any]) -> None:
        """Register multiple kernel parameter bindings."""
        for name, value in params.items():
            self.bind_param(name, value)

    def push_loop_var(self, name: str, value: Any) -> None:
        """Bind a loop iteration variable (by name).

        Note: this *adds* a binding to the existing context. Loop
        unrollers typically copy the parent context first so the binding
        is local to one iteration; this method does not copy.
        """
        self._loop_vars[name] = value
        self[name] = value

    def set_value(self, uuid: str, value: Any) -> None:
        """Bind an emit-time-computed intermediate by Value UUID.

        Use for ``BinOp`` / ``CompOp`` / ``CondOp`` / ``NotOp`` results,
        phi aliases, and other UUID-identified intermediates.
        """
        self._values[uuid] = value
        self[uuid] = value

    def set_runtime_expr(self, uuid: str, expr: Any) -> None:
        """Bind a backend runtime expression by Value UUID.

        Backends (e.g. Qiskit) call this when they construct a
        runtime-evaluable expression for a classical predicate that
        wasn't compile-time-foldable. ``_emit_if`` / ``_emit_while``
        consult the runtime-expr slot first when resolving conditions.
        """
        self._runtime_exprs[uuid] = expr
        self[uuid] = expr

    # -- Inspection helpers (debugging) -----------------------------------

    def describe(self) -> str:
        """Return a multi-line summary suitable for debug printing."""
        lines = [
            f"EmitContext(params={len(self._params)}, "
            f"loop_vars={len(self._loop_vars)}, "
            f"values={len(self._values)}, "
            f"runtime_exprs={len(self._runtime_exprs)})"
        ]
        for label, slot in [
            ("params", self._params),
            ("loop_vars", self._loop_vars),
            ("values", self._values),
            ("runtime_exprs", self._runtime_exprs),
        ]:
            for k, v in slot.items():
                short_v = repr(v)
                if len(short_v) > 60:
                    short_v = short_v[:57] + "..."
                lines.append(f"  {label}: {k!r} -> {short_v}")
        return "\n".join(lines)

    # -- Iteration over slots --------------------------------------------

    def iter_values(self) -> Iterator[tuple[str, Any]]:
        """Iterate over UUID-keyed emit-time intermediates only."""
        return iter(self._values.items())

    # -- Copy semantics ---------------------------------------------------

    def copy(self) -> "EmitContext":
        """Return a shallow copy preserving all semantic slots.

        The dict baseclass ``copy()`` returns a plain ``dict``, dropping
        the slot-tracking metadata. Loop unrollers call ``bindings.copy()``
        to make a per-iteration child scope; without this override the
        child would lose the params/loop_vars/values/runtime_exprs
        partitioning and become a flat dict, defeating the whole point of
        ``EmitContext``. We override to return an ``EmitContext`` with
        slot dicts independently copied so child mutations (e.g. pushing
        a new loop var) don't bleed back to the parent.
        """
        new = EmitContext(self)
        new._params = self._params.copy()
        new._loop_vars = self._loop_vars.copy()
        new._values = self._values.copy()
        new._runtime_exprs = self._runtime_exprs.copy()
        return new
