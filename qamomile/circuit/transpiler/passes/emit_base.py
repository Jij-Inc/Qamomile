"""Base classes for emit pass implementation.

This module provides backend-agnostic helper classes for resource allocation,
value resolution, and loop analysis. These are used by StandardEmitPass to
implement the emission logic without backend-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from qamomile.circuit.transpiler.errors import ResolutionFailureReason

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


@dataclass
class QubitResolutionResult:
    """Result of attempting to resolve a qubit index."""

    success: bool
    index: int | None = None
    failure_reason: ResolutionFailureReason | None = None
    failure_details: str = ""

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    MeasureOperation,
    MeasureVectorOperation,
    MeasureQFixedOperation,
    ControlledUOperation,
)
from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp


class ResourceAllocator:
    """Allocates qubit and classical bit indices from operations.

    This class handles the first pass of circuit emission: determining
    how many physical qubits and classical bits are needed and mapping
    Value UUIDs to their physical indices.
    """

    def allocate(
        self,
        operations: list[Operation],
        bindings: dict[str, Any] | None = None,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Allocate qubit and clbit indices for all operations.

        Args:
            operations: List of operations to allocate resources for
            bindings: Optional variable bindings for resolving dynamic sizes

        Returns:
            Tuple of (qubit_map, clbit_map) where each maps UUID to index
        """
        qubit_map: dict[str, int] = {}
        clbit_map: dict[str, int] = {}
        self._allocate_recursive(operations, qubit_map, clbit_map, bindings or {})
        return qubit_map, clbit_map

    def _allocate_recursive(
        self,
        operations: list[Operation],
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Recursively allocate resources from operations."""
        from qamomile.circuit.ir.value import ArrayValue

        for op in operations:
            if isinstance(op, QInitOperation):
                result = op.results[0]
                if isinstance(result, ArrayValue):
                    # Allocate qubits for array elements using {array_uuid}_{i} format
                    if result.shape:
                        size_val = result.shape[0]
                        size = self._resolve_size(size_val, bindings)
                        if size is not None:
                            for i in range(size):
                                qubit_id = f"{result.uuid}_{i}"
                                if qubit_id not in qubit_map:
                                    qubit_map[qubit_id] = len(qubit_map)
                    continue
                if result.uuid not in qubit_map:
                    qubit_map[result.uuid] = len(qubit_map)

            elif isinstance(op, MeasureOperation):
                result = op.results[0]
                if result.uuid not in clbit_map:
                    clbit_map[result.uuid] = len(clbit_map)

            elif isinstance(op, MeasureVectorOperation):
                result = op.results[0]
                if isinstance(result, ArrayValue) and result.shape:
                    size_val = result.shape[0]
                    size = self._resolve_size(size_val, bindings)
                    if size is not None:
                        for i in range(size):
                            clbit_id = f"{result.uuid}_{i}"
                            if clbit_id not in clbit_map:
                                clbit_map[clbit_id] = len(clbit_map)

            elif isinstance(op, MeasureQFixedOperation):
                qfixed = op.operands[0]
                qubit_uuids = qfixed.params.get("qubit_values", [])
                result = op.results[0]
                for i, qubit_uuid in enumerate(qubit_uuids):
                    clbit_id = f"{result.uuid}_{i}"
                    if clbit_id not in clbit_map:
                        clbit_map[clbit_id] = len(clbit_map)

            elif isinstance(op, GateOperation):
                self._allocate_gate(op, qubit_map)

            elif isinstance(op, ForOperation):
                self._allocate_recursive(op.operations, qubit_map, clbit_map, bindings)

            elif isinstance(op, IfOperation):
                self._allocate_recursive(op.true_operations, qubit_map, clbit_map, bindings)
                self._allocate_recursive(op.false_operations, qubit_map, clbit_map, bindings)

            elif isinstance(op, WhileOperation):
                self._allocate_recursive(op.operations, qubit_map, clbit_map, bindings)

            elif isinstance(op, CompositeGateOperation):
                self._allocate_composite(op, qubit_map)

            elif isinstance(op, ControlledUOperation):
                self._allocate_controlled_u(op, qubit_map)

            elif isinstance(op, CastOperation):
                self._allocate_cast(op, qubit_map)

    def _resolve_size(
        self,
        size_val: Any,
        bindings: dict[str, Any],
    ) -> int | None:
        """Resolve a size value to a concrete integer.

        Args:
            size_val: The size value (may be constant or symbolic)
            bindings: Variable bindings for resolution

        Returns:
            Resolved size as int, or None if cannot resolve
        """
        import re

        # Check for constant in params
        if hasattr(size_val, "params") and "const" in size_val.params:
            return int(size_val.params["const"])

        # Check if it's a Value with parent_array (e.g., hi.shape[0])
        if hasattr(size_val, "parent_array") and size_val.parent_array is not None:
            array_name = size_val.parent_array.name
            if array_name in bindings:
                array_data = bindings[array_name]
                # Get the shape/length of the bound array
                if hasattr(array_data, "__len__"):
                    return len(array_data)

        # Check by name in bindings
        if hasattr(size_val, "name") and size_val.name in bindings:
            bound = bindings[size_val.name]
            if isinstance(bound, (int, float)):
                return int(bound)
            if hasattr(bound, "__len__"):
                return len(bound)

        # Check by uuid in bindings
        if hasattr(size_val, "uuid") and size_val.uuid in bindings:
            bound = bindings[size_val.uuid]
            if isinstance(bound, (int, float)):
                return int(bound)

        # Check for dimension naming pattern (e.g., "hi_dim0" -> array "hi", dimension 0)
        # This handles cases where parent_array is None after inlining
        if hasattr(size_val, "name") and size_val.name:
            match = re.match(r"^(.+)_dim(\d+)$", size_val.name)
            if match:
                array_name = match.group(1)
                dim_index = int(match.group(2))
                if array_name in bindings:
                    array_data = bindings[array_name]
                    # Get shape at specified dimension
                    if hasattr(array_data, "shape"):
                        # numpy array or similar
                        if dim_index < len(array_data.shape):
                            return int(array_data.shape[dim_index])
                    elif dim_index == 0 and hasattr(array_data, "__len__"):
                        # For 1D sequences, dim0 is length
                        return len(array_data)

        return None

    def _allocate_gate(
        self,
        op: GateOperation,
        qubit_map: dict[str, int],
    ) -> None:
        """Allocate resources for a GateOperation."""
        for operand in op.operands:
            if operand.uuid not in qubit_map:
                if operand.parent_array is None:
                    qubit_map[operand.uuid] = len(qubit_map)

        for result in op.results:
            if result.uuid not in qubit_map:
                if result.parent_array is None and op.operands:
                    first_operand = op.operands[0]
                    if first_operand.parent_array is None:
                        qubit_map[result.uuid] = qubit_map.get(
                            first_operand.uuid, len(qubit_map)
                        )

    def _allocate_composite(
        self,
        op: CompositeGateOperation,
        qubit_map: dict[str, int],
    ) -> None:
        """Allocate resources for a CompositeGateOperation."""
        all_qubits = op.control_qubits + op.target_qubits

        def resolve_qubit_key(qubit: Value) -> tuple[str | None, bool]:
            if qubit.parent_array is not None and qubit.element_indices:
                parent_uuid = qubit.parent_array.uuid
                idx_value = qubit.element_indices[0]
                if idx_value.is_constant():
                    idx = int(idx_value.get_const())
                    return f"{parent_uuid}_{idx}", True
                return None, True
            return qubit.uuid, False

        for qubit in all_qubits:
            qubit_key, is_array = resolve_qubit_key(qubit)
            if qubit_key is not None:
                if qubit_key not in qubit_map:
                    qubit_map[qubit_key] = len(qubit_map)
                if qubit.uuid not in qubit_map:
                    qubit_map[qubit.uuid] = qubit_map[qubit_key]

        for i, result in enumerate(op.results):
            if result.uuid not in qubit_map and i < len(all_qubits):
                qubit_key, is_array = resolve_qubit_key(all_qubits[i])
                if qubit_key is not None:
                    qubit_map[result.uuid] = qubit_map.get(qubit_key, len(qubit_map))

    def _allocate_controlled_u(
        self,
        op: ControlledUOperation,
        qubit_map: dict[str, int],
    ) -> None:
        """Allocate resources for a ControlledUOperation."""
        control_qubits = list(op.control_operands)
        target_qubits = [
            v for v in op.target_operands if hasattr(v, "type") and v.type.is_quantum()
        ]
        all_qubits = control_qubits + target_qubits

        def resolve_qubit_key(qubit: Value) -> tuple[str | None, bool]:
            if qubit.parent_array is not None and qubit.element_indices:
                parent_uuid = qubit.parent_array.uuid
                idx_value = qubit.element_indices[0]
                if idx_value.is_constant():
                    idx = int(idx_value.get_const())
                    return f"{parent_uuid}_{idx}", True
                return None, True
            return qubit.uuid, False

        for qubit in all_qubits:
            qubit_key, is_array = resolve_qubit_key(qubit)
            if qubit_key is not None:
                if qubit_key not in qubit_map:
                    qubit_map[qubit_key] = len(qubit_map)
                if qubit.uuid not in qubit_map:
                    qubit_map[qubit.uuid] = qubit_map[qubit_key]

        for i, result in enumerate(op.results):
            if result.uuid not in qubit_map and i < len(all_qubits):
                qubit_key, is_array = resolve_qubit_key(all_qubits[i])
                if qubit_key is not None:
                    qubit_map[result.uuid] = qubit_map.get(qubit_key, len(qubit_map))

    def _allocate_cast(
        self,
        op: CastOperation,
        qubit_map: dict[str, int],
    ) -> None:
        """Allocate resources for a CastOperation (no-op)."""
        for i, qubit_uuid in enumerate(op.qubit_mapping):
            if qubit_uuid in qubit_map:
                result_element_id = f"{op.results[0].uuid}_{i}"
                qubit_map[result_element_id] = qubit_map[qubit_uuid]
                if op.results[0].uuid not in qubit_map:
                    qubit_map[op.results[0].uuid] = qubit_map[qubit_uuid]


class ValueResolver:
    """Resolves Value objects to concrete indices or values.

    Handles the mapping from IR Value objects to physical qubit indices,
    classical values, and parameter names during emission.
    """

    def __init__(self, parameters: set[str] | None = None):
        """Initialize the resolver.

        Args:
            parameters: Set of parameter names to preserve as symbolic
        """
        self.parameters = parameters or set()

    def resolve_qubit_index(
        self,
        v: "Value",
        qubit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> int | None:
        """Resolve a Value to a physical qubit index.

        Args:
            v: The Value to resolve
            qubit_map: Mapping from UUID to qubit index
            bindings: Current variable bindings

        Returns:
            Physical qubit index, or None if cannot resolve
        """
        result = self.resolve_qubit_index_detailed(v, qubit_map, bindings)
        return result.index if result.success else None

    def resolve_qubit_index_detailed(
        self,
        v: "Value",
        qubit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> QubitResolutionResult:
        """Resolve a Value to a physical qubit index with detailed failure info.

        Args:
            v: The Value to resolve
            qubit_map: Mapping from UUID to qubit index
            bindings: Current variable bindings

        Returns:
            QubitResolutionResult with success status and either index or failure details
        """
        # Check array element first
        if v.parent_array is not None and v.element_indices:
            parent_uuid = v.parent_array.uuid
            idx_value = v.element_indices[0]

            idx = None
            if idx_value.is_constant():
                idx = int(idx_value.get_const())
            elif idx_value.name in bindings:
                bound_val = bindings[idx_value.name]
                if isinstance(bound_val, (int, float)):
                    idx = int(bound_val)
                else:
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.INDEX_NOT_NUMERIC,
                        failure_details=(
                            f"Index '{idx_value.name}' resolved to non-numeric type: "
                            f"{type(bound_val).__name__}"
                        ),
                    )
            elif idx_value.uuid in bindings:
                bound_val = bindings[idx_value.uuid]
                if isinstance(bound_val, (int, float)):
                    idx = int(bound_val)
                else:
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.INDEX_NOT_NUMERIC,
                        failure_details=(
                            f"Index (uuid: {idx_value.uuid[:8]}...) resolved to "
                            f"non-numeric type: {type(bound_val).__name__}"
                        ),
                    )
            elif idx_value.parent_array is not None:
                # Nested array access (e.g., edges[e, 0])
                nested_result = self._resolve_array_element_value(idx_value, bindings)
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
                array_qubit_id = f"{parent_uuid}_{idx}"
                if array_qubit_id in qubit_map:
                    return QubitResolutionResult(
                        success=True, index=qubit_map[array_qubit_id]
                    )
                else:
                    return QubitResolutionResult(
                        success=False,
                        failure_reason=ResolutionFailureReason.ARRAY_ELEMENT_NOT_IN_QUBIT_MAP,
                        failure_details=(
                            f"Computed qubit ID '{array_qubit_id}' not found in qubit_map. "
                            f"Index {idx} may be out of bounds for array '{v.parent_array.name}'."
                        ),
                    )

        # Direct UUID lookup
        if v.uuid in qubit_map:
            return QubitResolutionResult(success=True, index=qubit_map[v.uuid])

        return QubitResolutionResult(
            success=False,
            failure_reason=ResolutionFailureReason.DIRECT_UUID_NOT_FOUND,
            failure_details=(
                f"Value uuid '{v.uuid[:8]}...' not found in qubit_map "
                f"and is not an array element."
            ),
        )

    def resolve_classical_value(
        self,
        value: "Value",
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve a classical Value to a concrete Python value.

        Args:
            value: The Value to resolve
            bindings: Current variable bindings

        Returns:
            Resolved value (int, float, etc.), or None if cannot resolve
        """
        if value.is_constant():
            return value.get_const()

        if value.uuid in bindings:
            return bindings[value.uuid]

        if value.name in bindings:
            return bindings[value.name]

        if value.params and "const" in value.params:
            return value.params["const"]

        if value.parent_array is not None and value.element_indices:
            return self._resolve_array_element_value(value, bindings)

        return None

    def _resolve_array_element_value(
        self,
        v: "Value",
        bindings: dict[str, Any],
    ) -> int | float | None:
        """Resolve an array element to a concrete value.

        Args:
            v: The array element Value
            bindings: Current variable bindings

        Returns:
            Resolved numeric value, or None if cannot resolve
        """
        if v.parent_array is None or not v.element_indices:
            return None

        array_name = v.parent_array.name
        if array_name not in bindings:
            return None

        array_data = bindings[array_name]

        indices = []
        for idx in v.element_indices:
            idx_val = self.resolve_int_value(idx, bindings)
            if idx_val is None:
                return None
            indices.append(idx_val)

        try:
            import numbers

            result = array_data
            for idx in indices:
                result = result[idx]
            # Check for numeric types including numpy integers/floats
            if isinstance(result, numbers.Real):
                return result
            return None
        except (IndexError, TypeError, KeyError):
            return None

    def resolve_int_value(
        self,
        val: Any,
        bindings: dict[str, Any],
    ) -> int | None:
        """Resolve a value to an integer (for loop bounds, etc.).

        Args:
            val: The value to resolve
            bindings: Current variable bindings

        Returns:
            Integer value, or None if cannot resolve
        """
        from qamomile.circuit.ir.value import Value

        if isinstance(val, (int, float)):
            return int(val)
        elif isinstance(val, Value):
            if val.is_constant():
                return int(val.get_const())
            elif val.is_parameter():
                param_name = val.parameter_name()
                if param_name and param_name in bindings:
                    bound_val = bindings[param_name]
                    if isinstance(bound_val, (int, float)):
                        return int(bound_val)
                    return None
            elif val.name in bindings:
                bound_val = bindings[val.name]
                if isinstance(bound_val, (int, float)):
                    return int(bound_val)
                return None
        return 0

    def get_parameter_key(
        self,
        value: "Value",
        bindings: dict[str, Any],
    ) -> str | None:
        """Get parameter key if this value should be a symbolic parameter.

        Args:
            value: The Value to check
            bindings: Current bindings (for resolving array indices)

        Returns:
            Parameter key (e.g., "gammas[0]") if symbolic, None otherwise
        """
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
                        bound_val = bindings[idx_value.name]
                        if isinstance(bound_val, (int, float)):
                            idx = int(bound_val)
                    elif idx_value.uuid in bindings:
                        bound_val = bindings[idx_value.uuid]
                        if isinstance(bound_val, (int, float)):
                            idx = int(bound_val)

                    if idx is not None:
                        return f"{parent_name}[{idx}]"

        return None


class LoopAnalyzer:
    """Analyzes loop structures to determine emission strategy.

    Determines whether loops should use native backend control flow
    or be unrolled at emission time.
    """

    def should_unroll(
        self,
        op: ForOperation,
        bindings: dict[str, Any],
    ) -> bool:
        """Determine if a ForOperation should be unrolled.

        Args:
            op: The ForOperation to analyze
            bindings: Current variable bindings

        Returns:
            True if loop should be unrolled, False for native emission
        """
        # Check for dynamic nested loops
        if self._has_dynamic_nested_loop(op.operations, bindings, op.loop_var):
            return True

        # Check for array element access
        if self._has_array_element_access(op.operations, op.loop_var):
            return True

        return False

    def _has_dynamic_nested_loop(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
        parent_loop_var: str,
    ) -> bool:
        """Check if operations contain nested loops depending on parent loop variable."""
        for op in operations:
            if isinstance(op, ForOperation):
                for bound_val in op.operands[:3]:
                    if hasattr(bound_val, "name"):
                        if bound_val.name == parent_loop_var:
                            return True
                        if bound_val.name in bindings:
                            bound = bindings[bound_val.name]
                            if not isinstance(bound, (int, float)):
                                return True
                if self._has_dynamic_nested_loop(
                    op.operations, bindings, parent_loop_var
                ):
                    return True
        return False

    def _has_array_element_access(
        self,
        operations: list[Operation],
        loop_var: str,
    ) -> bool:
        """Check if operations access array elements using loop variable."""
        for op in operations:
            if isinstance(op, GateOperation):
                for v in op.operands:
                    if v.parent_array is not None and v.element_indices:
                        for idx in v.element_indices:
                            if self._index_depends_on_loop_var(idx, loop_var):
                                return True

                if hasattr(op, "theta") and op.theta is not None:
                    theta = op.theta
                    if (
                        hasattr(theta, "parent_array")
                        and theta.parent_array is not None
                    ):
                        if hasattr(theta, "element_indices") and theta.element_indices:
                            for idx in theta.element_indices:
                                if self._index_depends_on_loop_var(idx, loop_var):
                                    return True

            elif isinstance(op, BinOp):
                for operand in [op.lhs, op.rhs]:
                    if operand.parent_array is not None and operand.element_indices:
                        for idx in operand.element_indices:
                            if self._index_depends_on_loop_var(idx, loop_var):
                                return True

            elif isinstance(op, ControlledUOperation):
                for v in op.operands:
                    if hasattr(v, "parent_array") and v.parent_array is not None:
                        if hasattr(v, "element_indices") and v.element_indices:
                            for idx in v.element_indices:
                                if self._index_depends_on_loop_var(idx, loop_var):
                                    return True

            elif isinstance(op, ForOperation):
                if self._has_array_element_access(op.operations, loop_var):
                    return True

        return False

    def _index_depends_on_loop_var(self, idx: "Value", loop_var: str) -> bool:
        """Check if an index depends on the loop variable."""
        if idx.name == loop_var:
            return True

        if idx.parent_array is not None and idx.element_indices:
            for sub_idx in idx.element_indices:
                if self._index_depends_on_loop_var(sub_idx, loop_var):
                    return True

        return False


class CompositeDecomposer:
    """Decomposes composite gates into primitive operations.

    Provides algorithms for QFT, IQFT, and QPE decomposition that
    backends can use for fallback when native implementations are
    unavailable.
    """

    @staticmethod
    def qft_structure(n: int) -> list[tuple[str, tuple[int, ...], float | None]]:
        """Generate QFT gate sequence.

        Args:
            n: Number of qubits

        Returns:
            List of (gate_name, qubit_indices, angle) tuples
        """
        import math

        gates = []
        for i in range(n):
            gates.append(("h", (i,), None))
            for j in range(i + 1, n):
                k = j - i
                angle = math.pi / (2**k)
                gates.append(("cp", (j, i), angle))

        # Swaps for bit order reversal
        for i in range(n // 2):
            gates.append(("swap", (i, n - 1 - i), None))

        return gates

    @staticmethod
    def iqft_structure(n: int) -> list[tuple[str, tuple[int, ...], float | None]]:
        """Generate inverse QFT gate sequence.

        Args:
            n: Number of qubits

        Returns:
            List of (gate_name, qubit_indices, angle) tuples
        """
        import math

        gates = []
        for i in range(n - 1, -1, -1):
            gates.append(("h", (i,), None))
            for j in range(i - 1, -1, -1):
                k = i - j
                angle = -math.pi / (2**k)
                gates.append(("cp", (j, i), angle))

        # Swaps for bit order reversal
        for i in range(n // 2):
            gates.append(("swap", (i, n - 1 - i), None))

        return gates
