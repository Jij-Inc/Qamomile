"""Value mapping utilities for IR transformations."""

from __future__ import annotations

import dataclasses
import uuid as uuid_module
from typing import cast

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import GateOperation
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
)


class UUIDRemapper:
    """Clones values and operations with fresh UUIDs and logical_ids.

    Used during inlining to create unique identities for values
    when a block is called multiple times.
    """

    def __init__(self):
        self._uuid_remap: dict[str, str] = {}
        self._logical_id_remap: dict[str, str] = {}
        # Cache for all value types (uses ValueBase for unified handling)
        self._value_cache: dict[str, ValueBase] = {}

    @property
    def uuid_remap(self) -> dict[str, str]:
        """Get the mapping from old UUIDs to new UUIDs."""
        return self._uuid_remap

    @property
    def logical_id_remap(self) -> dict[str, str]:
        """Get the mapping from old logical_ids to new logical_ids."""
        return self._logical_id_remap

    def clone_operations(self, operations: list[Operation]) -> list[Operation]:
        """Clone a list of operations with fresh UUIDs."""
        return [self.clone_operation(op) for op in operations]

    def clone_operation(self, op: Operation) -> Operation:
        """Clone an operation with fresh UUIDs for all values."""
        # Clone operands (handles all ValueBase types)
        new_operands = []
        for v in op.operands:
            if isinstance(v, ValueBase):
                new_operands.append(self.clone_value(v))
            else:
                # BlockValue in CallBlockOperation - don't clone
                new_operands.append(v)

        # Clone results (handles all ValueBase types)
        new_results = []
        for v in op.results:
            if isinstance(v, ValueBase):
                new_results.append(self.clone_value(v))
            else:
                new_results.append(v)

        # Handle control flow operations
        if isinstance(op, ForOperation):
            cloned_body = self.clone_operations(op.operations)
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                operations=cloned_body,
            )
        elif isinstance(op, ForItemsOperation):
            cloned_body = self.clone_operations(op.operations)
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                operations=cloned_body,
            )
        elif isinstance(op, IfOperation):
            cloned_true = self.clone_operations(op.true_operations)
            cloned_false = self.clone_operations(op.false_operations)
            cloned_phi = [self.clone_operation(p) for p in op.phi_ops]
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                true_operations=cloned_true,
                false_operations=cloned_false,
                phi_ops=cloned_phi,
            )
        elif isinstance(op, WhileOperation):
            cloned_body = self.clone_operations(op.operations)
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                operations=cloned_body,
            )
        else:
            cloned = dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
            )
            # Handle GateOperation.theta (a Value stored outside operands)
            if isinstance(cloned, GateOperation) and isinstance(cloned.theta, Value):
                cloned = dataclasses.replace(
                    cloned, theta=cast(Value, self.clone_value(cloned.theta))
                )
            return cloned

    def clone_value(self, value: ValueBase) -> ValueBase:
        """Clone any value type with a fresh UUID and logical_id.

        Handles Value, ArrayValue, TupleValue, and DictValue through
        the unified ValueBase protocol.
        """
        old_uuid = value.uuid

        # Check cache first - uuid is unique per value instance
        if old_uuid in self._value_cache:
            return self._value_cache[old_uuid]

        # Generate new UUID for this specific value
        new_uuid = str(uuid_module.uuid4())
        self._uuid_remap[old_uuid] = new_uuid

        # Generate new logical_id (same for values with same physical identity)
        old_logical_id = value.logical_id
        if old_logical_id not in self._logical_id_remap:
            self._logical_id_remap[old_logical_id] = str(uuid_module.uuid4())
        new_logical_id = self._logical_id_remap[old_logical_id]

        # Handle different value types
        if isinstance(value, TupleValue):
            # Clone tuple elements
            new_elements = tuple(
                cast(Value, self.clone_value(e)) for e in value.elements
            )
            cloned = dataclasses.replace(
                value,
                uuid=new_uuid,
                logical_id=new_logical_id,
                elements=new_elements,
            )
        elif isinstance(value, DictValue):
            # Clone dict entries (both keys and values)
            new_entries: list[tuple[TupleValue | Value, Value]] = []
            for k, v in value.entries:
                if isinstance(k, TupleValue):
                    new_key = cast(TupleValue, self.clone_value(k))
                else:
                    new_key = cast(Value, self.clone_value(k))
                new_val = cast(Value, self.clone_value(v))
                new_entries.append((new_key, new_val))
            cloned = dataclasses.replace(
                value,
                uuid=new_uuid,
                logical_id=new_logical_id,
                entries=new_entries,
            )
        elif isinstance(value, ArrayValue):
            # ArrayValue: clone parent_array, element_indices, AND shape
            new_parent_array: ArrayValue | None = None
            if value.parent_array is not None:
                new_parent_array = cast(
                    ArrayValue, self.clone_value(value.parent_array)
                )

            new_element_indices: tuple[Value, ...] | None = None
            if value.element_indices:
                new_element_indices = tuple(
                    cast(Value, self.clone_value(idx)) for idx in value.element_indices
                )

            # Clone shape values so sub-kernel dimension parameters
            # get fresh UUIDs that can be mapped during inlining.
            new_shape: tuple[Value, ...] = ()
            if value.shape:
                new_shape = tuple(
                    cast(Value, self.clone_value(dim)) for dim in value.shape
                )

            cloned = dataclasses.replace(
                value,
                uuid=new_uuid,
                logical_id=new_logical_id,
                parent_array=new_parent_array,
                element_indices=new_element_indices if new_element_indices else (),
                shape=new_shape,
            )
        elif isinstance(value, Value):
            # Regular Value (not ArrayValue)
            new_parent_array_v: ArrayValue | None = None
            if value.parent_array is not None:
                new_parent_array_v = cast(
                    ArrayValue, self.clone_value(value.parent_array)
                )

            new_element_indices_v: tuple[Value, ...] | None = None
            if value.element_indices:
                new_element_indices_v = tuple(
                    cast(Value, self.clone_value(idx)) for idx in value.element_indices
                )

            cloned = dataclasses.replace(
                value,
                uuid=new_uuid,
                logical_id=new_logical_id,
                parent_array=new_parent_array_v,
                element_indices=new_element_indices_v if new_element_indices_v else (),
            )
        else:
            # Fallback for any other ValueBase type
            cloned = dataclasses.replace(
                value,
                uuid=new_uuid,
                logical_id=new_logical_id,
            )

        self._value_cache[old_uuid] = cloned
        return cloned


class ValueSubstitutor:
    """Substitutes values in operations using a mapping.

    Used during inlining to replace block parameters with caller arguments.
    """

    def __init__(self, value_map: dict[str, ValueBase]):
        self._value_map = value_map

    def substitute_operation(self, op: Operation) -> Operation:
        """Substitute values in an operation using the value map.

        Handles control-flow operations (IfOperation, ForOperation, etc.)
        by recursing into their nested operation lists and phi_ops.
        """
        new_operands = []
        for v in op.operands:
            if isinstance(v, ValueBase):
                new_operands.append(self.substitute_value(v))
            else:
                new_operands.append(v)

        new_results = []
        for v in op.results:
            if isinstance(v, ValueBase):
                new_results.append(self.substitute_value(v))
            else:
                new_results.append(v)

        # Handle IfOperation: also substitute inside phi_ops
        if isinstance(op, IfOperation):
            new_phi_ops = [self.substitute_operation(p) for p in op.phi_ops]
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                phi_ops=new_phi_ops,
            )

        substituted = dataclasses.replace(
            op,
            operands=new_operands,
            results=new_results,
        )

        # Handle GateOperation.theta (a Value stored outside operands)
        if isinstance(substituted, GateOperation) and isinstance(
            substituted.theta, Value
        ):
            new_theta = self.substitute_value(substituted.theta)
            if isinstance(new_theta, Value) and new_theta is not substituted.theta:
                substituted = dataclasses.replace(substituted, theta=new_theta)

        return substituted

    def substitute_value(self, v: ValueBase) -> ValueBase:
        """Substitute a single value using the value map.

        Handles all value types and array elements by substituting
        their parent_array if needed.
        """
        # First try direct lookup
        if v.uuid in self._value_map:
            return self._value_map[v.uuid]

        # Handle TupleValue - substitute elements
        if isinstance(v, TupleValue):
            new_elements = []
            changed = False
            for elem in v.elements:
                if elem.uuid in self._value_map:
                    substituted = self._value_map[elem.uuid]
                    if isinstance(substituted, Value):
                        new_elements.append(substituted)
                        changed = True
                    else:
                        new_elements.append(elem)
                else:
                    new_elements.append(elem)
            if changed:
                return dataclasses.replace(v, elements=tuple(new_elements))
            return v

        # Handle DictValue - substitute entries
        if isinstance(v, DictValue):
            new_entries: list[tuple[TupleValue | Value, Value]] = []
            changed = False
            for k, val in v.entries:
                new_k = k
                new_v = val
                if isinstance(k, (TupleValue, Value)) and k.uuid in self._value_map:
                    sub_k = self._value_map[k.uuid]
                    if isinstance(sub_k, (TupleValue, Value)):
                        new_k = sub_k  # type: ignore
                        changed = True
                if val.uuid in self._value_map:
                    sub_v = self._value_map[val.uuid]
                    if isinstance(sub_v, Value):
                        new_v = sub_v
                        changed = True
                new_entries.append((new_k, new_v))
            if changed:
                return dataclasses.replace(v, entries=new_entries)
            return v

        # Handle regular Value/ArrayValue
        if isinstance(v, Value):
            replacements: dict[str, object] = {}

            # Check if this is an array element whose parent_array should be substituted
            if v.parent_array is not None and v.parent_array.uuid in self._value_map:
                new_parent = self._value_map[v.parent_array.uuid]
                if isinstance(new_parent, ArrayValue):
                    replacements["parent_array"] = new_parent

            # Also check element_indices for substitution
            if v.element_indices:
                new_indices = []
                indices_changed = False
                for idx in v.element_indices:
                    if idx.uuid in self._value_map:
                        sub_idx = self._value_map[idx.uuid]
                        if isinstance(sub_idx, Value):
                            new_indices.append(sub_idx)
                            indices_changed = True
                        else:
                            new_indices.append(idx)
                    else:
                        new_indices.append(idx)
                if indices_changed:
                    replacements["element_indices"] = tuple(new_indices)

            # Substitute ArrayValue.shape dimensions so sub-kernel
            # dimension parameters are resolved to caller arguments.
            if isinstance(v, ArrayValue) and v.shape:
                new_shape_dims: list[Value] = []
                shape_changed = False
                for dim in v.shape:
                    if dim.uuid in self._value_map:
                        sub_dim = self._value_map[dim.uuid]
                        if isinstance(sub_dim, Value):
                            new_shape_dims.append(sub_dim)
                            shape_changed = True
                        else:
                            new_shape_dims.append(dim)
                    else:
                        new_shape_dims.append(dim)
                if shape_changed:
                    replacements["shape"] = tuple(new_shape_dims)

            if replacements:
                return dataclasses.replace(v, **replacements)

        return v
