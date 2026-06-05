"""Value mapping utilities for IR transformations."""

from __future__ import annotations

import dataclasses
import uuid as uuid_module
from typing import Any, Mapping, cast

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
    IfOperation,
)
from qamomile.circuit.ir.value import (
    ArrayRuntimeMetadata,
    ArrayValue,
    CastMetadata,
    DictValue,
    QFixedMetadata,
    TupleValue,
    Value,
    ValueBase,
    ValueMetadata,
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

    def _clone_metadata(self, metadata: ValueMetadata) -> ValueMetadata:
        """Clone metadata UUID references through this remapper.

        Args:
            metadata (ValueMetadata): Metadata bundle attached to the
                value being cloned.

        Returns:
            ValueMetadata: Metadata with UUID-bearing metadata and
                logical-id references rewritten where clone mappings
                already exist.
        """
        new_cast = metadata.cast
        if new_cast is not None:
            new_cast = CastMetadata(
                source_uuid=self._uuid_remap.get(
                    new_cast.source_uuid, new_cast.source_uuid
                ),
                qubit_uuids=tuple(
                    self._uuid_remap.get(uuid, uuid) for uuid in new_cast.qubit_uuids
                ),
                source_logical_id=(
                    self._logical_id_remap.get(
                        new_cast.source_logical_id,
                        new_cast.source_logical_id,
                    )
                    if new_cast.source_logical_id is not None
                    else None
                ),
                qubit_logical_ids=tuple(
                    self._logical_id_remap.get(logical_id, logical_id)
                    for logical_id in new_cast.qubit_logical_ids
                ),
            )

        new_qfixed = metadata.qfixed
        if new_qfixed is not None:
            new_qfixed = QFixedMetadata(
                qubit_uuids=tuple(
                    self._uuid_remap.get(uuid, uuid) for uuid in new_qfixed.qubit_uuids
                ),
                num_bits=new_qfixed.num_bits,
                int_bits=new_qfixed.int_bits,
            )

        new_array_rt = metadata.array_runtime
        if new_array_rt is not None:
            new_array_rt = ArrayRuntimeMetadata(
                const_array=new_array_rt.const_array,
                element_uuids=tuple(
                    self._uuid_remap.get(uuid, uuid)
                    for uuid in new_array_rt.element_uuids
                ),
                element_logical_ids=tuple(
                    self._logical_id_remap.get(logical_id, logical_id)
                    for logical_id in new_array_rt.element_logical_ids
                ),
                element_parent_uuids=tuple(
                    self._uuid_remap.get(uuid, uuid) if uuid else uuid
                    for uuid in new_array_rt.element_parent_uuids
                ),
                element_parent_indices=new_array_rt.element_parent_indices,
            )

        if (
            new_cast is metadata.cast
            and new_qfixed is metadata.qfixed
            and new_array_rt is metadata.array_runtime
        ):
            return metadata

        return dataclasses.replace(
            metadata,
            cast=new_cast,
            qfixed=new_qfixed,
            array_runtime=new_array_rt,
        )

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

        new_metadata = self._clone_metadata(value.metadata)
        if new_metadata is not value.metadata:
            cloned = dataclasses.replace(cast(Any, cloned), metadata=new_metadata)

        self._value_cache[old_uuid] = cloned
        return cloned


class ValueSubstitutor:
    """Substitutes values in operations using a mapping.

    Used during inlining to replace block parameters with caller arguments.

    When ``transitive`` is True, substitute_value chases transitive chains
    (A -> B -> C) to terminal values with cycle detection, which is needed
    for phi substitution during compile-time if lowering.
    """

    def __init__(self, value_map: Mapping[str, ValueBase], transitive: bool = False):
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

    def _mapped_value_for_uuid(self, uuid: str) -> ValueBase | None:
        """Return the mapped value for a UUID, following transitive chains.

        Args:
            uuid (str): UUID to resolve through this substitutor's value
                map.

        Returns:
            ValueBase | None: The mapped terminal value, or ``None`` when
                the UUID is not present in the map.
        """
        result = self._value_map.get(uuid)
        if result is None:
            return None
        if self._transitive:
            seen: set[str] = {uuid}
            while result.uuid in self._value_map and result.uuid not in seen:
                seen.add(result.uuid)
                result = self._value_map[result.uuid]
        return result

    def _const_int(self, v: Value | None) -> int | None:
        """Resolve a constant integer value used in array metadata.

        Args:
            v (Value | None): Candidate integer value.

        Returns:
            int | None: Integer payload when ``v`` is constant; otherwise
                ``None``.
        """
        if v is None or not v.is_constant():
            return None
        const = v.get_const()
        if const is None:
            return None
        return int(const)

    def _resolve_array_runtime_parent(
        self,
        parent: ArrayValue,
        index: int,
    ) -> tuple[str, int]:
        """Resolve a parent/index pair through substituted slice metadata.

        Args:
            parent (ArrayValue): Parent array value for an element packed
                into ``ArrayRuntimeMetadata``.
            index (int): Element index relative to ``parent``.

        Returns:
            tuple[str, int]: Root array UUID and root-space index when
                the slice chain is constant; otherwise the deepest
                resolvable parent UUID and index.
        """
        current = parent
        resolved_index = index

        while current.slice_of is not None:
            start = self._const_int(current.slice_start)
            step = self._const_int(current.slice_step)
            if start is None or step is None:
                return current.uuid, resolved_index

            next_parent = self.substitute_value(current.slice_of)
            if not isinstance(next_parent, ArrayValue):
                next_parent = current.slice_of

            resolved_index = start + step * resolved_index
            current = next_parent

        return current.uuid, resolved_index

    def _substitute_array_runtime_metadata(
        self,
        metadata: ValueMetadata,
    ) -> tuple[ValueMetadata, bool]:
        """Rewrite array-runtime UUID references through the value map.

        Args:
            metadata (ValueMetadata): Metadata bundle owned by the value
                being substituted.

        Returns:
            tuple[ValueMetadata, bool]: The rewritten metadata and a flag
                indicating whether any UUID/index reference changed.
        """
        array_rt = metadata.array_runtime
        if array_rt is None:
            return metadata, False

        changed = False
        element_uuids = list(array_rt.element_uuids)
        element_logical_ids = list(array_rt.element_logical_ids)
        for i, element_uuid in enumerate(element_uuids):
            mapped = self._mapped_value_for_uuid(element_uuid)
            if mapped is not None and mapped.uuid != element_uuid:
                element_uuids[i] = mapped.uuid
                if i < len(element_logical_ids):
                    element_logical_ids[i] = mapped.logical_id
                changed = True

        element_parent_uuids = list(array_rt.element_parent_uuids)
        element_parent_indices = list(array_rt.element_parent_indices)
        for i, parent_uuid in enumerate(element_parent_uuids):
            if (
                not parent_uuid
                or i >= len(element_parent_indices)
                or element_parent_indices[i] < 0
            ):
                continue

            mapped_parent = self._mapped_value_for_uuid(parent_uuid)
            if not isinstance(mapped_parent, ArrayValue):
                continue

            substituted_parent = self.substitute_value(mapped_parent)
            if isinstance(substituted_parent, ArrayValue):
                mapped_parent = substituted_parent

            root_uuid, root_index = self._resolve_array_runtime_parent(
                mapped_parent,
                int(element_parent_indices[i]),
            )
            if root_uuid != parent_uuid or root_index != element_parent_indices[i]:
                element_parent_uuids[i] = root_uuid
                element_parent_indices[i] = root_index
                changed = True

        if not changed:
            return metadata, False

        return (
            dataclasses.replace(
                metadata,
                array_runtime=ArrayRuntimeMetadata(
                    const_array=array_rt.const_array,
                    element_uuids=tuple(element_uuids),
                    element_logical_ids=tuple(element_logical_ids),
                    element_parent_uuids=tuple(element_parent_uuids),
                    element_parent_indices=tuple(element_parent_indices),
                ),
            ),
            True,
        )

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
        array_rt = v.metadata.array_runtime
        if array_rt is not None:
            for element_uuid in array_rt.element_uuids:
                if element_uuid in self._value_map:
                    return True
            for parent_uuid in array_rt.element_parent_uuids:
                if parent_uuid and parent_uuid in self._value_map:
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
        if v.parent_array is not None:
            sub_pa = self.substitute_value(v.parent_array)
            if isinstance(sub_pa, ArrayValue) and sub_pa is not v.parent_array:
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

        new_metadata, metadata_changed = self._substitute_array_runtime_metadata(
            v.metadata
        )

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
            if v.slice_of is not None:
                sub_slice_of = self.substitute_value(v.slice_of)
                if (
                    isinstance(sub_slice_of, ArrayValue)
                    and sub_slice_of is not v.slice_of
                ):
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
                metadata=new_metadata if metadata_changed else v.metadata,
            )

        return dataclasses.replace(
            v,
            parent_array=new_parent_array,
            element_indices=new_element_indices,
            metadata=new_metadata if metadata_changed else v.metadata,
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
            new_metadata = v.metadata
            changed = False

            # Substitute the parent_array.  ``v.parent_array`` itself may
            # not be in the map (it can be a freshly-cloned slice-view
            # ``ArrayValue`` whose own ``slice_of`` is the callee dummy
            # being substituted), so a direct ``uuid in self._value_map``
            # check would leave the parent_array's stale internal fields
            # in place.  Recursing through ``substitute_value`` walks the
            # parent_array's fields so the slice chain is rewritten
            # consistently when a sub-kernel containing
            # ``qs[a:b] = qmc.h(qs[a:b])`` is inlined.
            if v.parent_array is not None:
                sub_parent = self.substitute_value(v.parent_array)
                if (
                    isinstance(sub_parent, ArrayValue)
                    and sub_parent is not v.parent_array
                ):
                    new_parent_array = sub_parent
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

                # ``slice_of`` may also point at a freshly-cloned sliced
                # ``ArrayValue`` whose own slice metadata still references
                # the callee dummy.  Recurse through ``substitute_value``
                # so nested slices (slice-of-slice) get rewritten too.
                if v.slice_of is not None:
                    sub_slice_of = self.substitute_value(v.slice_of)
                    if (
                        isinstance(sub_slice_of, ArrayValue)
                        and sub_slice_of is not v.slice_of
                    ):
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

            new_metadata, metadata_changed = self._substitute_array_runtime_metadata(
                v.metadata
            )
            if metadata_changed:
                changed = True

            if changed:
                if isinstance(v, ArrayValue):
                    replace_kwargs: dict[str, Any] = dict(
                        parent_array=new_parent_array,
                        element_indices=new_element_indices,
                        metadata=new_metadata,
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
                    metadata=new_metadata,
                )

        return v
