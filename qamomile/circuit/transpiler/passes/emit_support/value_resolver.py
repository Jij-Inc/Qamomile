"""Value resolution helpers for emission."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qamomile.circuit.transpiler.errors import ResolutionFailureReason
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    QubitAddress,
    QubitMap,
)

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


@dataclass
class QubitResolutionResult:
    """Result of attempting to resolve a qubit index."""

    success: bool
    index: int | None = None
    failure_reason: ResolutionFailureReason | None = None
    failure_details: str = ""


def resolve_qubit_key(qubit: "Value") -> tuple[QubitAddress | None, bool]:
    """Resolve a qubit Value to its allocation key.

    Returns a ``(QubitAddress | None, is_array_element)`` tuple.
    """
    if qubit.parent_array is not None and qubit.element_indices:
        parent_uuid = qubit.parent_array.uuid
        idx_value = qubit.element_indices[0]
        if idx_value.is_constant():
            idx = int(idx_value.get_const())
            return QubitAddress(parent_uuid, idx), True
        return None, True
    return QubitAddress(qubit.uuid), False


class ValueResolver:
    """Resolves Value objects to concrete indices or values."""

    def __init__(self, parameters: set[str] | None = None):
        self.parameters = parameters or set()

    def resolve_qubit_index(
        self,
        v: "Value",
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> int | None:
        result = self.resolve_qubit_index_detailed(v, qubit_map, bindings)
        return result.index if result.success else None

    def resolve_qubit_index_detailed(
        self,
        v: "Value",
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> QubitResolutionResult:
        """Resolve a Value to a physical qubit index with detailed failure info."""
        if v.parent_array is not None and v.element_indices:
            parent_uuid = v.parent_array.uuid
            idx_value = v.element_indices[0]

            idx = None
            if idx_value.is_constant():
                idx = int(idx_value.get_const())
            else:
                raw = self.lookup_in_bindings(idx_value, bindings)
                if raw is not None:
                    idx = self._resolve_numeric_index(raw)
                    if idx is None:
                        return QubitResolutionResult(
                            success=False,
                            failure_reason=ResolutionFailureReason.INDEX_NOT_NUMERIC,
                            failure_details=(
                                f"Index '{idx_value.name}' (uuid: "
                                f"{idx_value.uuid[:8]}...) resolved to "
                                f"non-numeric type: {type(raw).__name__}"
                            ),
                        )
            if idx is None:
                if idx_value.parent_array is not None:
                    nested_result = self.resolve_classical_value(idx_value, bindings)
                    if nested_result is None:
                        array_name = idx_value.parent_array.name
                        return QubitResolutionResult(
                            success=False,
                            failure_reason=ResolutionFailureReason.NESTED_ARRAY_RESOLUTION_FAILED,
                            failure_details=(
                                f"Nested array access '{array_name}[...]' could not be resolved. "
                                f"Array '{array_name}' may not be in bindings."
                            ),
                        )
                    idx = int(nested_result)
                else:
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.SYMBOLIC_INDEX_NOT_BOUND,
                        failure_details=(
                            f"Index variable '{idx_value.name}' (uuid: "
                            f"{idx_value.uuid[:8]}...) is not bound."
                        ),
                    )

            if idx is not None:
                array_qubit_addr = QubitAddress(parent_uuid, idx)
                if array_qubit_addr in qubit_map:
                    return QubitResolutionResult(
                        success=True, index=qubit_map[array_qubit_addr]
                    )
                return QubitResolutionResult(
                    success=False,
                    failure_reason=ResolutionFailureReason.ARRAY_ELEMENT_NOT_IN_QUBIT_MAP,
                    failure_details=(
                        f"Computed qubit ID '{array_qubit_addr}' not found in qubit_map. "
                        f"Index {idx} may be out of bounds for array '{v.parent_array.name}'."
                    ),
                )

        scalar_addr = QubitAddress(v.uuid)
        if scalar_addr in qubit_map:
            return QubitResolutionResult(success=True, index=qubit_map[scalar_addr])

        return QubitResolutionResult(
            success=False,
            failure_reason=ResolutionFailureReason.DIRECT_UUID_NOT_FOUND,
            failure_details=(
                f"Value uuid '{v.uuid[:8]}...' not found in qubit_map "
                f"and is not an array element."
            ),
        )

    # ------------------------------------------------------------------
    # Unified bindings lookup
    # ------------------------------------------------------------------

    def lookup_in_bindings(
        self,
        value: "Value",
        bindings: dict[str, Any],
        *,
        index_array: bool = False,
    ) -> Any:
        """Canonical resolution chain for a Value against ``bindings``.

        All other resolver methods (``resolve_bound_value``,
        ``resolve_classical_value``, ``resolve_int_value``,
        ``resolve_operand_for_binding``) wrap this single chain. Centralizing
        precedence here prevents the historical drift where one resolver
        checked ``is_parameter`` before UUID and another checked it after,
        which manifested as obscure binding failures when name-keyed writes
        were dropped from the emit pass.

        Resolution order (each step returns immediately on a hit):

        1. ``value`` is already a concrete Python scalar (no ``uuid``).
        2. ``value.is_constant()`` — return ``value.get_const()``.
        3. ``value.is_parameter()`` and its parameter name is in
           ``bindings`` — return that.
        4. ``value.uuid`` is in ``bindings`` — return that. This is where
           emit-time-computed intermediates (``evaluate_binop`` /
           ``evaluate_classical_predicate`` results) and phi aliases live.
        5. ``value.name`` is in ``bindings`` — return that. This is where
           kernel parameters and loop iteration variables live. NOT a
           reliable channel for auto-generated tmp names like
           ``"uint_tmp"`` — those are intentionally written by UUID only.
        6. (When ``index_array=True``) ``value`` is an array element with
           a resolvable parent in ``bindings`` — index into it.

        Args:
            value: The IR Value (or already-concrete Python scalar) to
                resolve.
            bindings: The active bindings dict.
            index_array: When True, also resolve array-element accesses
                via ``parent_array`` indexing. Off by default because not
                all callers want to index into bound containers.

        Returns:
            The resolved Python value, or ``None`` if no step matched.
        """
        # 1. Already concrete (Python scalar that was passed through).
        if not hasattr(value, "uuid"):
            return value
        # 2. IR constant.
        if hasattr(value, "is_constant") and value.is_constant():
            return value.get_const()
        # 3. Kernel parameter — keyed by parameter name (the only
        #    legitimate name-keyed path; user supplies parameters by name
        #    at the public API boundary).
        if hasattr(value, "is_parameter") and value.is_parameter():
            param_name = value.parameter_name()
            if param_name and param_name in bindings:
                return bindings[param_name]
        # 4. UUID-keyed entries (intermediates, loop variables, runtime
        #    exprs, phi aliases — every internally-created binding keys
        #    on Value UUID).
        if value.uuid in bindings:
            return bindings[value.uuid]
        # 5. User-parameter compat path: a non-empty Value name that
        #    matches a binding entry. Reaches here for:
        #      - kernel parameters whose ``is_parameter()`` flag was
        #        dropped during a transform (e.g. some inline/substitute
        #        paths) but whose ``name`` still matches the original
        #        parameter (``theta``, ``n``, ``obs``, …);
        #      - parameter array shape dimensions bound by name
        #        (``indices_dim0``, ``key_dim0``).
        #    Phase 1 anonymous tmp Values (``name=""``) skip this path
        #    naturally — there is no remaining "name collision among
        #    auto-generated tmps" risk. Eliminating this step entirely
        #    requires guaranteeing ``is_parameter()`` survives all IR
        #    transforms, which is a separate, larger refactor.
        if getattr(value, "name", None) and value.name in bindings:
            return bindings[value.name]
        # 6. Array-element access via parent_array (opt-in).
        if (
            index_array
            and getattr(value, "parent_array", None) is not None
            and getattr(value, "element_indices", None)
        ):
            return self._index_into_array(value, bindings)
        return None

    def resolve_operand_for_binding(
        self,
        operand: "Value",
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve an operand to a concrete value for block parameter binding.

        Used when calling a sub-block (e.g. a controlled-U body): each
        param operand at the call site must resolve to a value to seed the
        callee's parameter bindings.
        """
        return self.lookup_in_bindings(operand, bindings)

    def bind_block_params(
        self,
        block_value: Any,
        param_operands: list["Value"],
        bindings: dict[str, Any],
    ) -> dict[str, Any]:
        """Create local bindings by matching block parameter inputs to operands."""
        local_bindings = bindings.copy()
        if not hasattr(block_value, "input_values"):
            return local_bindings
        param_inputs = [
            iv
            for iv in block_value.input_values
            if hasattr(iv, "type") and iv.type.is_classical()
        ]
        for i, operand in enumerate(param_operands):
            if i >= len(param_inputs):
                break
            resolved = self.resolve_operand_for_binding(operand, bindings)
            if resolved is not None:
                local_bindings[param_inputs[i].name] = resolved
        return local_bindings

    def resolve_bound_value(
        self,
        value: "Value",
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve a Value to its raw bound Python object.

        Wraps :meth:`lookup_in_bindings` with ``index_array=True`` so that
        ``arr[i]`` accesses against a bound container resolve to the
        element. Does **not** coerce the result — callers that need a
        numeric scalar should go through :meth:`resolve_classical_value`.
        """
        return self.lookup_in_bindings(value, bindings, index_array=True)

    def resolve_classical_value(
        self,
        value: "Value",
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve a classical Value to a concrete Python value.

        Numeric bindings are normalized to native Python scalars
        regardless of whether they come from a direct binding or from
        array-element indexing, so downstream ``isinstance(x, (int,
        float))`` checks are stable when callers bind ``np.pi/4`` or
        the like. ``bool`` is preserved (not coerced to ``int``).
        Non-numeric values (Hamiltonians, strings, dict values, …)
        pass through unchanged.
        """
        raw = self.resolve_bound_value(value, bindings)
        if raw is None or isinstance(raw, bool):
            return raw
        coerced = self._resolve_numeric_value(raw)
        return coerced if coerced is not None else raw

    def _index_into_array(
        self,
        v: "Value",
        bindings: dict[str, Any],
    ) -> Any:
        """Index into a bound array container at the operand's element indices.

        Refuses to index when the parent array's name is in
        ``self.parameters``. That short-circuit is the same invariant
        ``fold_classical_op(... EMIT_RESPECT_PARAMS)`` enforces at the
        op level: a runtime parameter array's "value" is symbolic, and
        any concrete data the user supplied alongside is a placeholder
        — silently indexing into the placeholder is exactly what
        produced the silent miscompilation in Issue #354 B-series. This
        guard is defense-in-depth: even if a future caller forgets the
        op-level guard, array indexing for runtime parameters returns
        ``None`` here.
        """
        parent = v.parent_array
        if parent is None:
            return None
        if parent.name in self.parameters:
            return None
        container = bindings.get(parent.name)
        if container is None:
            container = bindings.get(parent.uuid)
        if container is None:
            return None

        for idx in v.element_indices:
            i = self.resolve_int_value(idx, bindings)
            if i is None:
                return None
            try:
                container = container[i]
            except (IndexError, KeyError, TypeError):
                return None
        return container

    def resolve_int_value(
        self,
        val: Any,
        bindings: dict[str, Any],
    ) -> int | None:
        """Resolve a value to an integer, or ``None`` when unresolvable.

        Unresolvable symbolic Values **must** return ``None``. The previous
        ``return 0`` fallback caused silent loop elision when parameter
        shape dims (``gamma_dim0``) reached this resolver without being
        folded into constants — downstream loop-bound resolution saw 0 and
        quietly emitted an empty loop. Returning ``None`` propagates the
        failure to ``emit_for_unrolled``, which converts it into a hard
        compile error.
        """
        raw = self.lookup_in_bindings(val, bindings)
        if raw is None:
            return None
        return self._resolve_numeric_index(raw)

    def get_parameter_key(
        self,
        value: "Value",
        bindings: dict[str, Any],
    ) -> str | None:
        """Get parameter key if this value should be a symbolic parameter."""
        if value.name in self.parameters:
            return value.name

        if value.parent_array is not None:
            parent_name = value.parent_array.name
            if parent_name in self.parameters:
                if value.element_indices and len(value.element_indices) > 0:
                    idx_value = value.element_indices[0]
                    idx = self.resolve_int_value(idx_value, bindings)
                    if idx is not None:
                        return f"{parent_name}[{idx}]"

        if value.type.is_classical() and value.name:
            return value.name

        return None

    def _resolve_numeric_index(self, value: Any) -> int | None:
        """Resolve a bound numeric scalar to a Python int."""
        numeric = self._resolve_numeric_value(value)
        if numeric is None:
            return None
        return int(numeric)

    def _resolve_numeric_value(self, value: Any) -> int | float | None:
        """Normalize Python and NumPy numeric scalars to Python scalars."""
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            return float(value)
        if hasattr(value, "item"):
            try:
                item = value.item()
            except (TypeError, ValueError):
                return None
            if isinstance(item, numbers.Integral):
                return int(item)
            if isinstance(item, numbers.Real):
                return float(item)
        return None
