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
            elif idx_value.name in bindings:
                idx = self._resolve_numeric_index(bindings[idx_value.name])
                if idx is None:
                    bound_val = bindings[idx_value.name]
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.INDEX_NOT_NUMERIC,
                        failure_details=(
                            f"Index '{idx_value.name}' resolved to non-numeric type: "
                            f"{type(bound_val).__name__}"
                        ),
                    )
            elif idx_value.uuid in bindings:
                idx = self._resolve_numeric_index(bindings[idx_value.uuid])
                if idx is None:
                    bound_val = bindings[idx_value.uuid]
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.INDEX_NOT_NUMERIC,
                        failure_details=(
                            f"Index (uuid: {idx_value.uuid[:8]}...) resolved to "
                            f"non-numeric type: {type(bound_val).__name__}"
                        ),
                    )
            elif idx_value.parent_array is not None:
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
                        f"Index variable '{idx_value.name}' is not bound. "
                        f"Neither name nor uuid found in bindings."
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

    def resolve_operand_for_binding(
        self,
        operand: "Value",
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve an operand to a concrete value for block parameter binding.

        Resolution order:
        1. Constant value (``operand.get_const()``)
        2. Parameter name lookup in *bindings*
        3. Value name lookup in *bindings*
        """
        if operand.is_constant():
            return operand.get_const()
        if operand.is_parameter():
            outer_name = operand.parameter_name()
            if outer_name and outer_name in bindings:
                return bindings[outer_name]
        if operand.name in bindings:
            return bindings[operand.name]
        return None

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

        Looks up in order: scalar constant, ``bindings[uuid]``
        (emit-time intermediates such as BinOp results and in-kernel
        Observables), ``bindings[name]`` (kernel parameters), then
        parent-array element access when ``value`` is ``arr[i]``.
        Returns ``None`` when no binding matches. Does **not** coerce
        the result — callers that need a numeric scalar should go
        through :meth:`resolve_classical_value`.
        """
        if value.is_constant():
            return value.get_const()
        if value.uuid in bindings:
            return bindings[value.uuid]
        if value.name in bindings:
            return bindings[value.name]
        if value.parent_array is not None and value.element_indices:
            return self._index_into_array(value, bindings)
        return None

    def resolve_classical_value(
        self,
        value: "Value",
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve a classical Value to a concrete Python value.

        Array-element lookups often return a numpy scalar; this
        variant normalizes to a Python scalar. Non-array bindings are
        passed through as-is.
        """
        raw = self.resolve_bound_value(value, bindings)
        if raw is None:
            return None
        if value.parent_array is not None and value.element_indices:
            return self._resolve_numeric_value(raw)
        return raw

    def _index_into_array(
        self,
        v: "Value",
        bindings: dict[str, Any],
    ) -> Any:
        parent = v.parent_array
        if parent is None:
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
        from qamomile.circuit.ir.value import Value

        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, Value):
            if val.is_constant():
                return int(val.get_const())
            if val.is_parameter():
                param_name = val.parameter_name()
                if param_name and param_name in bindings:
                    idx = self._resolve_numeric_index(bindings[param_name])
                    if idx is not None:
                        return idx
                    return None
            if val.uuid in bindings:
                idx = self._resolve_numeric_index(bindings[val.uuid])
                if idx is not None:
                    return idx
                return None
            if val.name in bindings:
                idx = self._resolve_numeric_index(bindings[val.name])
                if idx is not None:
                    return idx
                return None
        return None

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
                    idx = None
                    if idx_value.is_constant():
                        idx = int(idx_value.get_const())
                    elif idx_value.name in bindings:
                        idx = self._resolve_numeric_index(bindings[idx_value.name])
                    elif idx_value.uuid in bindings:
                        idx = self._resolve_numeric_index(bindings[idx_value.uuid])

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
