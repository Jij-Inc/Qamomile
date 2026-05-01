"""Value mapping utilities for IR transformations."""

from __future__ import annotations

import dataclasses
import uuid as uuid_module
from typing import Any, cast

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
    IfOperation,
)
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
        """Clone an operation with fresh UUIDs for all values.

        Cloning goes through the ``Operation.all_input_values()`` /
        ``Operation.replace_values()`` protocol so every Value-typed
        field — including subclass extras (``ControlledUOperation.power``,
        ``ForOperation.loop_var_value``, ``ForItemsOperation.key_var_values``
        etc.) — is cloned consistently with the body references that
        point to it. Without this, a subclass field could keep an old
        UUID while body operands referencing the same logical Value got
        fresh UUIDs, breaking identity-by-UUID lookups at emit time.
        """
        # Build a uuid -> cloned_value substitution map covering every
        # Value the operation owns (operands, results, plus subclass
        # extras exposed via all_input_values).
        sub_map: dict[str, ValueBase] = {}
        for v in op.all_input_values():
            sub_map[v.uuid] = self.clone_value(v)
        for v in op.results:
            if isinstance(v, ValueBase) and v.uuid not in sub_map:
                sub_map[v.uuid] = self.clone_value(v)

        new_op = op.replace_values(sub_map) if sub_map else op

        # Recursively clone nested ops for control flow ops. Nested
        # bodies see the same UUIDRemapper, so their references to a
        # cloned outer Value (e.g. a parent ForOperation's loop_var_value)
        # resolve through ``self._value_cache`` and stay consistent with
        # the parent op's field.
        if isinstance(new_op, HasNestedOps):
            cloned_lists = [
                self.clone_operations(op_list) for op_list in new_op.nested_op_lists()
            ]
            new_op = new_op.rebuild_nested(cloned_lists)

        return new_op

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
                entries=tuple(new_entries),
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

            # Clone slice metadata so sliced ArrayValues that flow
            # through inlining keep their chain intact with fresh
            # uuids ready to be substituted to caller-side values.
            new_slice_of: ArrayValue | None = None
            if value.slice_of is not None:
                new_slice_of = cast(ArrayValue, self.clone_value(value.slice_of))
            new_slice_start: Value | None = None
            if value.slice_start is not None:
                new_slice_start = cast(Value, self.clone_value(value.slice_start))
            new_slice_step: Value | None = None
            if value.slice_step is not None:
                new_slice_step = cast(Value, self.clone_value(value.slice_step))

            cloned = dataclasses.replace(
                value,
                uuid=new_uuid,
                logical_id=new_logical_id,
                parent_array=new_parent_array,
                element_indices=new_element_indices if new_element_indices else (),
                shape=new_shape,
                slice_of=new_slice_of,
                slice_start=new_slice_start,
                slice_step=new_slice_step,
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
                cast(Any, value),
                uuid=new_uuid,
                logical_id=new_logical_id,
            )

        self._value_cache[old_uuid] = cloned
        return cloned


class ValueSubstitutor:
    """Substitutes values in operations using a mapping.

    Used during inlining to replace block parameters with caller arguments.

    When ``transitive`` is True, substitute_value chases transitive chains
    (A -> B -> C) to terminal values with cycle detection, which is needed
    for phi substitution during compile-time if lowering.
    """

    def __init__(self, value_map: dict[str, ValueBase], transitive: bool = False):
        self._value_map = value_map
        self._transitive = transitive

    def substitute_operation(self, op: Operation) -> Operation:
        """Substitute values in an operation using the value map.

        Uses ``Operation.replace_values()`` to handle operands, results,
        and any subclass-specific Value fields (e.g. ControlledUOperation.power).
        Also handles IfOperation phi_ops recursion.
        """
        # Build a substitution mapping (uuid -> substituted ValueBase)
        sub_map: dict[str, ValueBase] = {}
        for v in op.all_input_values():
            substituted = self.substitute_value(v)
            if substituted is not v:
                sub_map[v.uuid] = substituted
        for v in op.results:
            if isinstance(v, ValueBase):
                substituted = self.substitute_value(v)
                if substituted is not v:
                    sub_map[v.uuid] = substituted

        result = op.replace_values(sub_map) if sub_map else op

        # Handle IfOperation: also substitute inside phi_ops
        if isinstance(result, IfOperation):
            from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp

            new_phi_ops = cast(
                list[PhiOp],
                [self.substitute_operation(p) for p in result.phi_ops],
            )
            result = dataclasses.replace(result, phi_ops=new_phi_ops)

        return result

    def _chase_transitive(self, v: ValueBase) -> ValueBase:
        """Chase transitive chains in the value map with cycle detection."""
        result = self._value_map[v.uuid]
        if self._transitive:
            seen: set[str] = {v.uuid}
            while result.uuid in self._value_map and result.uuid not in seen:
                seen.add(result.uuid)
                result = self._value_map[result.uuid]
        return result

    def _has_referenced_field(self, v: "Value") -> bool:
        """Return True if ``v`` holds a field whose UUID is in the map.

        Used after a direct UUID hit to decide whether the mapped
        Value's own fields (``parent_array`` / ``element_indices`` /
        ``slice_of`` / ``slice_start`` / ``slice_step``) still need to
        be substituted — which happens when a callee's dummy is mapped
        to a value produced in a prior inlining level whose parent_array
        also points at another dummy.

        Args:
            v: The mapped Value returned by a direct lookup.

        Returns:
            ``True`` iff at least one field references a UUID present
            in the substitution map.
        """
        if v.parent_array is not None and v.parent_array.uuid in self._value_map:
            return True
        if v.element_indices:
            for idx in v.element_indices:
                if idx.uuid in self._value_map:
                    return True
        if isinstance(v, ArrayValue):
            for dim in v.shape:
                if dim.uuid in self._value_map:
                    return True
            if v.slice_of is not None and v.slice_of.uuid in self._value_map:
                return True
            if v.slice_start is not None and v.slice_start.uuid in self._value_map:
                return True
            if v.slice_step is not None and v.slice_step.uuid in self._value_map:
                return True
        return False

    def _resubstitute_fields(self, v: "Value") -> "Value":
        """Rebuild ``v`` with its fields resolved through the map.

        Args:
            v: A Value returned by the direct-lookup branch whose own
                fields still reference mappable UUIDs.

        Returns:
            A new Value with ``parent_array`` / ``element_indices`` /
            ``shape`` / slice metadata substituted.  Non-referenced
            fields are left unchanged.
        """
        new_parent_array = v.parent_array
        if v.parent_array is not None and v.parent_array.uuid in self._value_map:
            sub_pa = self._chase_transitive(v.parent_array)
            if isinstance(sub_pa, ArrayValue):
                new_parent_array = sub_pa

        new_element_indices = v.element_indices
        if v.element_indices:
            new_indices_list: list[Value] = []
            for idx in v.element_indices:
                if idx.uuid in self._value_map:
                    sub_idx = self._chase_transitive(idx)
                    if isinstance(sub_idx, Value):
                        new_indices_list.append(sub_idx)
                        continue
                new_indices_list.append(idx)
            new_element_indices = tuple(new_indices_list)

        if isinstance(v, ArrayValue):
            new_shape: tuple[Value, ...] = v.shape
            if v.shape:
                new_shape_list: list[Value] = []
                for dim in v.shape:
                    if dim.uuid in self._value_map:
                        sub_dim = self._chase_transitive(dim)
                        if isinstance(sub_dim, Value):
                            new_shape_list.append(sub_dim)
                            continue
                    new_shape_list.append(dim)
                new_shape = tuple(new_shape_list)

            new_slice_of = v.slice_of
            if v.slice_of is not None and v.slice_of.uuid in self._value_map:
                sub_slice_of = self._chase_transitive(v.slice_of)
                if isinstance(sub_slice_of, ArrayValue):
                    new_slice_of = sub_slice_of
            new_slice_start = v.slice_start
            if v.slice_start is not None and v.slice_start.uuid in self._value_map:
                sub_slice_start = self._chase_transitive(v.slice_start)
                if isinstance(sub_slice_start, Value):
                    new_slice_start = sub_slice_start
            new_slice_step = v.slice_step
            if v.slice_step is not None and v.slice_step.uuid in self._value_map:
                sub_slice_step = self._chase_transitive(v.slice_step)
                if isinstance(sub_slice_step, Value):
                    new_slice_step = sub_slice_step

            return dataclasses.replace(
                v,
                parent_array=new_parent_array,
                element_indices=new_element_indices,
                shape=new_shape,
                slice_of=new_slice_of,
                slice_start=new_slice_start,
                slice_step=new_slice_step,
            )

        return dataclasses.replace(
            v,
            parent_array=new_parent_array,
            element_indices=new_element_indices,
        )

    def substitute_value(self, v: ValueBase) -> ValueBase:
        """Substitute a single value using the value map.

        Handles all value types and array elements by substituting
        their parent_array if needed.  When ``transitive`` is enabled,
        chases transitive chains (A -> B -> C) to terminal values.

        If the direct lookup resolves to a value whose own fields
        (``parent_array`` / ``element_indices`` / ``shape`` / ``slice_of``)
        may themselves be subject to further substitution — typically
        when a callee's dummy scalar input gets mapped to a caller
        array element whose parent_array is itself a mapped dummy —
        the result is re-substituted field-wise so the whole transitive
        chain is resolved in one call.  This is what allows patterns
        like nested sub-kernels receiving ``q[i]`` from the caller's
        array to resolve through two levels of inlining.
        """
        # First try direct lookup
        if v.uuid in self._value_map:
            mapped = self._chase_transitive(v)
            # If the mapped value is a Value with unresolved fields
            # (parent_array / slice metadata whose uuids are also in
            # the map), dig in and substitute those as well.  Without
            # this, element values forwarded across a sub-kernel call
            # would keep the inner dummy's parent_array and fail to
            # resolve at emit.
            if (
                isinstance(mapped, Value)
                and mapped is not v
                and self._has_referenced_field(mapped)
            ):
                return self._resubstitute_fields(mapped)
            return mapped

        # Handle TupleValue - substitute elements
        if isinstance(v, TupleValue):
            new_elements = []
            changed = False
            for elem in v.elements:
                if elem.uuid in self._value_map:
                    substituted = self._chase_transitive(elem)
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
                    sub_k = self._chase_transitive(k)
                    if isinstance(sub_k, (TupleValue, Value)):
                        new_k = sub_k  # type: ignore
                        changed = True
                if val.uuid in self._value_map:
                    sub_v = self._chase_transitive(val)
                    if isinstance(sub_v, Value):
                        new_v = sub_v
                        changed = True
                new_entries.append((new_k, new_v))
            if changed:
                return dataclasses.replace(v, entries=tuple(new_entries))
            return v

        # Handle regular Value/ArrayValue
        if isinstance(v, Value):
            new_parent_array = v.parent_array
            new_element_indices = v.element_indices
            new_shape: tuple[Value, ...] | None = None
            changed = False

            # Check if this is an array element whose parent_array should be substituted
            if v.parent_array is not None and v.parent_array.uuid in self._value_map:
                new_parent = self._chase_transitive(v.parent_array)
                if isinstance(new_parent, ArrayValue):
                    new_parent_array = new_parent
                    changed = True

            # Also check element_indices for substitution
            if v.element_indices:
                new_indices: list[Value] = []
                indices_changed = False
                for idx in v.element_indices:
                    if idx.uuid in self._value_map:
                        sub_idx = self._chase_transitive(idx)
                        if isinstance(sub_idx, Value):
                            new_indices.append(sub_idx)
                            indices_changed = True
                        else:
                            new_indices.append(idx)
                    else:
                        new_indices.append(idx)
                if indices_changed:
                    new_element_indices = tuple(new_indices)
                    changed = True

            # Substitute ArrayValue.shape dimensions so sub-kernel
            # dimension parameters are resolved to caller arguments.
            if isinstance(v, ArrayValue) and v.shape:
                new_shape_dims: list[Value] = []
                shape_changed = False
                for dim in v.shape:
                    if dim.uuid in self._value_map:
                        sub_dim = self._chase_transitive(dim)
                        if isinstance(sub_dim, Value):
                            new_shape_dims.append(sub_dim)
                            shape_changed = True
                        else:
                            new_shape_dims.append(dim)
                    else:
                        new_shape_dims.append(dim)
                if shape_changed:
                    new_shape = tuple(new_shape_dims)
                    changed = True

            # Substitute slice_of / slice_start / slice_step on sliced
            # ArrayValues so that view metadata flows through inlining
            # alongside parent_array.  When a callee receives a sliced
            # argument, the dummy input's slice_of (if any) and the
            # caller-side sliced value both need to be re-pointed to
            # the caller's actual values.
            new_slice_of_v: "ArrayValue | None" = None
            new_slice_start_v: "Value | None" = None
            new_slice_step_v: "Value | None" = None
            slice_meta_changed = False
            if isinstance(v, ArrayValue):
                new_slice_of_v = v.slice_of
                new_slice_start_v = v.slice_start
                new_slice_step_v = v.slice_step

                if v.slice_of is not None and v.slice_of.uuid in self._value_map:
                    sub_slice_of = self._chase_transitive(v.slice_of)
                    if isinstance(sub_slice_of, ArrayValue):
                        new_slice_of_v = sub_slice_of
                        slice_meta_changed = True

                if v.slice_start is not None and v.slice_start.uuid in self._value_map:
                    sub_slice_start = self._chase_transitive(v.slice_start)
                    if isinstance(sub_slice_start, Value):
                        new_slice_start_v = sub_slice_start
                        slice_meta_changed = True

                if v.slice_step is not None and v.slice_step.uuid in self._value_map:
                    sub_slice_step = self._chase_transitive(v.slice_step)
                    if isinstance(sub_slice_step, Value):
                        new_slice_step_v = sub_slice_step
                        slice_meta_changed = True

                if slice_meta_changed:
                    changed = True

            if changed:
                if isinstance(v, ArrayValue):
                    replace_kwargs: dict[str, Any] = dict(
                        parent_array=new_parent_array,
                        element_indices=new_element_indices,
                    )
                    if new_shape is not None:
                        replace_kwargs["shape"] = new_shape
                    if slice_meta_changed:
                        replace_kwargs["slice_of"] = new_slice_of_v
                        replace_kwargs["slice_start"] = new_slice_start_v
                        replace_kwargs["slice_step"] = new_slice_step_v
                    return dataclasses.replace(v, **replace_kwargs)
                return dataclasses.replace(
                    v,
                    parent_array=new_parent_array,
                    element_indices=new_element_indices,
                )

        return v
