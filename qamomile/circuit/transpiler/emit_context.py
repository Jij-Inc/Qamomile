"""Typed bindings container for the emit pipeline.

Background — what was wrong with the bare ``dict[str, Any]``:

The pre-EmitContext design used a single ``bindings: dict[str, Any]``
threaded through every emit-pipeline function. That dict served at least
seven distinct semantic purposes simultaneously:

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
6. Array data (keyed by array name; bound iterables passed by user).
7. Dict data (keyed by dict name; bound iterables passed by user).
8. Pauli observables (keyed by observable name).

This overloading was the structural cause of every name-collision bug
class seen in this codebase: ``"bit_tmp"`` chained predicates,
``j_phi_4`` phi aliases, the inline-pass ``DictValue`` drop, and the
type-blind ``bool(...)`` coercion in ``resolve_operand``. Each was
patched locally; the structural overloading remained.

What ``EmitContext`` does (root-cause fix):

``EmitContext`` is a ``dict`` subclass — flat ``[key]`` access still
works for migration compatibility. On top of dict semantics, every
binding kind has a **separate, semantically-typed slot** with the
appropriate identity key:

- ``_params`` — user parameters, keyed by **name** (user-facing).
- ``_loop_vars`` — loop iteration variables, keyed by **Value UUID**.
- ``_values`` — emit-time intermediates, keyed by **UUID**.
- ``_runtime_exprs`` — backend Expr objects, keyed by **UUID**.
- ``_array_data`` — array bindings, keyed by ``ArrayValue.uuid``.
- ``_dict_data`` — dict bindings, keyed by ``DictValue.uuid``.
- ``_observables`` — Pauli observables, keyed by ``Value.uuid``.

The key invariant: **after the migration, the dict-baseclass writes
disappear**. All writers go through typed setters (``push_loop_var``,
``set_array_data``, etc.); all readers go through typed getters.
``EmitContext`` retains dict-protocol read-compat for legacy callers
during migration, but new code should never touch ``ctx[key]``.

Identity policy:

- **UUID**: everything compiler-internal (loop vars, intermediates,
  runtime exprs, array/dict/observable bindings).
- **Name**: only at the user-API boundary (parameter names supplied by
  ``transpile(bindings={...})``).

This eliminates the name-collision bug class entirely: empty/duplicate
names cannot resolve to anything because lookups never go through the
name path.
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
            name. Stable across the run. **Name-keyed** because the user
            supplies parameters by name at the public API boundary.
        _loop_vars: Currently-bound loop iteration variables, keyed by
            ``ForOperation.loop_var_value.uuid`` /
            ``ForItemsOperation.value_var_value.uuid`` etc. Pushed on
            loop entry, restored on exit. **UUID-keyed** so identical
            user-chosen variable names in nested or sibling loops never
            collide.
        _values: Emit-time-computed intermediate values (``BinOp``
            results, ``CompOp``/``CondOp``/``NotOp`` results, phi
            aliases), keyed by Value UUID.
        _runtime_exprs: Backend runtime-expression objects (e.g. Qiskit
            ``expr.Expr`` for compound classical conditions), keyed by
            Value UUID.
        _array_data: Bound array data (e.g. ``Vector[Float]`` parameter
            values), keyed by ``ArrayValue.uuid``.
        _dict_data: Bound dict data (e.g. ``Dict[Tuple[UInt, UInt], Float]``
            ising coefficients), keyed by ``DictValue.uuid``.
        _observables: Bound Pauli observables (used by ``PauliEvolveOp``
            and gate counting), keyed by observable Value UUID.

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

    __slots__ = (
        "_params",
        "_loop_vars",
        "_values",
        "_runtime_exprs",
        "_array_data",
        "_dict_data",
        "_observables",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._params: dict[str, Any] = {}
        self._loop_vars: dict[str, Any] = {}
        self._values: dict[str, Any] = {}
        self._runtime_exprs: dict[str, Any] = {}
        self._array_data: dict[str, Any] = {}
        self._dict_data: dict[str, Any] = {}
        self._observables: dict[str, Any] = {}

    # -- Construction ------------------------------------------------------

    @classmethod
    def from_user_bindings(cls, user_bindings: dict[str, Any] | None) -> "EmitContext":
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

    def push_loop_var(
        self,
        uuid: str,
        value: Any,
        display_name: str | None = None,
    ) -> None:
        """Bind a loop iteration variable, keyed by Value UUID.

        Args:
            uuid: ``loop_var_value.uuid`` (or per-key/value UUID for
                ``ForItemsOperation``). Different loops with identical
                user-chosen names (e.g. nested ``for i``) get distinct
                UUIDs and therefore never collide here.
            value: The bound iteration value (int / Hamiltonian item / etc.).
            display_name: Optional name for debug printing. Also written
                to the flat-dict view for legacy name-fallback readers
                during the migration period; remove once all readers
                use UUID lookup.

        Note: this *adds* a binding to the existing context. Loop
        unrollers typically copy the parent context first so the binding
        is local to one iteration; this method does not copy.
        """
        self._loop_vars[uuid] = value
        self[uuid] = value
        if display_name:
            # Migration shim: legacy readers still look up by name.
            # Remove once Phase 3 of #7 lands and all readers use UUID.
            self[display_name] = value

    def get_loop_var(self, uuid: str) -> Any:
        """Get a loop variable binding by Value UUID, or None if absent."""
        return self._loop_vars.get(uuid)

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

    def get_runtime_expr(self, uuid: str) -> Any:
        """Get a backend runtime expression by Value UUID, or None."""
        return self._runtime_exprs.get(uuid)

    def set_array_data(
        self,
        uuid: str,
        data: Any,
        display_name: str | None = None,
    ) -> None:
        """Bind array data by ``ArrayValue.uuid``.

        Args:
            uuid: The array Value's UUID.
            data: The bound iterable / sequence / Vector handle.
            display_name: Migration shim — also writes the flat-dict view
                under the array's user-facing name. Remove once Phase 3
                of #7 lands.
        """
        self._array_data[uuid] = data
        self[uuid] = data
        if display_name:
            self[display_name] = data

    def get_array_data(self, uuid: str) -> Any:
        """Get array data by ``ArrayValue.uuid``, or None."""
        return self._array_data.get(uuid)

    def set_dict_data(
        self,
        uuid: str,
        data: Any,
        display_name: str | None = None,
    ) -> None:
        """Bind dict data by ``DictValue.uuid``.

        Args:
            uuid: The dict Value's UUID.
            data: The bound dict / iterable.
            display_name: Migration shim — see ``set_array_data``.
        """
        self._dict_data[uuid] = data
        self[uuid] = data
        if display_name:
            self[display_name] = data

    def get_dict_data(self, uuid: str) -> Any:
        """Get dict data by ``DictValue.uuid``, or None."""
        return self._dict_data.get(uuid)

    def set_observable(
        self,
        uuid: str,
        observable: Any,
        display_name: str | None = None,
    ) -> None:
        """Bind a Pauli observable by Value UUID.

        Args:
            uuid: The observable Value's UUID.
            observable: A ``qm_o.Hamiltonian`` (or backend-equivalent).
            display_name: Migration shim — see ``set_array_data``.
        """
        self._observables[uuid] = observable
        self[uuid] = observable
        if display_name:
            self[display_name] = observable

    def get_observable(self, uuid: str) -> Any:
        """Get a Pauli observable by Value UUID, or None."""
        return self._observables.get(uuid)

    # -- Inspection helpers (debugging) -----------------------------------

    def describe(self) -> str:
        """Return a multi-line summary suitable for debug printing."""
        lines = [
            f"EmitContext(params={len(self._params)}, "
            f"loop_vars={len(self._loop_vars)}, "
            f"values={len(self._values)}, "
            f"runtime_exprs={len(self._runtime_exprs)}, "
            f"array_data={len(self._array_data)}, "
            f"dict_data={len(self._dict_data)}, "
            f"observables={len(self._observables)})"
        ]
        for label, slot in [
            ("params", self._params),
            ("loop_vars", self._loop_vars),
            ("values", self._values),
            ("runtime_exprs", self._runtime_exprs),
            ("array_data", self._array_data),
            ("dict_data", self._dict_data),
            ("observables", self._observables),
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
        new._array_data = self._array_data.copy()
        new._dict_data = self._dict_data.copy()
        new._observables = self._observables.copy()
        return new
