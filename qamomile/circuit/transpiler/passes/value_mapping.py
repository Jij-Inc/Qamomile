"""Value mapping utilities for IR transformations."""

from __future__ import annotations

import dataclasses
import uuid as uuid_module
from typing import Any, Mapping, cast

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.cast import CastOperation
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
    remap_indexed_identifier,
    remap_value_metadata_references,
    resolve_root_array_index,
    split_indexed_identifier,
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

        # Clone nested bodies BEFORE results. IfOperation results are phi
        # outputs whose metadata may reference values whose first (and only)
        # appearance is inside the branch bodies — e.g. QFixed carrier keys
        # pointing at an array that is cast inside the ``if``. Cloning the
        # bodies first fills the remap tables so ``_clone_metadata`` can
        # resolve those references; the phi outputs themselves are cloned
        # while cloning ``phi_ops`` (the last nested list) and the results
        # loop below then reuses the cached clones.
        cloned_lists: list[list[Operation]] | None = None
        if isinstance(op, HasNestedOps):
            cloned_lists = [
                self.clone_operations(op_list) for op_list in op.nested_op_lists()
            ]

        for v in op.results:
            if isinstance(v, ValueBase) and v.uuid not in sub_map:
                sub_map[v.uuid] = self.clone_value(v)

        new_op = op.replace_values(sub_map) if sub_map else op

        # Nested bodies see the same UUIDRemapper, so their references to a
        # cloned outer Value (e.g. a parent ForOperation's loop_var_value)
        # resolve through ``self._value_cache`` and stay consistent with
        # the parent op's field.
        if cloned_lists is not None:
            new_op = cast(HasNestedOps, new_op).rebuild_nested(cloned_lists)

        if isinstance(new_op, CastOperation) and new_op.qubit_mapping:
            new_op = dataclasses.replace(
                new_op,
                qubit_mapping=[
                    remap_indexed_identifier(
                        qubit_uuid,
                        lambda uuid: self._uuid_remap.get(uuid, uuid),
                    )
                    for qubit_uuid in new_op.qubit_mapping
                ],
            )

        return new_op

    def _clone_metadata(self, metadata: ValueMetadata) -> ValueMetadata:
        """Clone metadata references through the current UUID maps.

        Args:
            metadata (ValueMetadata): Metadata from the source value.

        Returns:
            ValueMetadata: Metadata whose embedded UUID and logical-id
                references point at cloned values when those values are known.
        """
        return remap_value_metadata_references(
            metadata,
            lambda uuid: self._uuid_remap.get(uuid, uuid),
            lambda logical_id: self._logical_id_remap.get(logical_id, logical_id),
        )

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
        new_metadata = self._clone_metadata(value.metadata)

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
                metadata=new_metadata,
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
                metadata=new_metadata,
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
                metadata=new_metadata,
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
                metadata=new_metadata,
                parent_array=new_parent_array_v,
                element_indices=new_element_indices_v if new_element_indices_v else (),
            )
        else:
            # Fallback for any other ValueBase type
            cloned = dataclasses.replace(
                cast(Any, value),
                uuid=new_uuid,
                logical_id=new_logical_id,
                metadata=new_metadata,
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

    def __init__(self, value_map: Mapping[str, ValueBase], transitive: bool = False):
        self._value_map = value_map
        self._transitive = transitive

    def _mapped_value_for_uuid(self, uuid: str) -> ValueBase | None:
        """Return the mapped value for a UUID, following transitive mappings.

        Args:
            uuid (str): UUID to resolve through the substitution map.

        Returns:
            ValueBase | None: The terminal mapped value, or ``None`` when the
                UUID is not present in the map.
        """
        if uuid not in self._value_map:
            return None
        result = self._value_map[uuid]
        if self._transitive:
            seen: set[str] = {uuid}
            while result.uuid in self._value_map and result.uuid not in seen:
                seen.add(result.uuid)
                result = self._value_map[result.uuid]
        return result

    def _substitute_uuid(self, uuid: str) -> str:
        """Substitute a scalar UUID reference through the value map.

        Args:
            uuid (str): UUID reference to substitute.

        Returns:
            str: Replacement UUID when mapped, otherwise the original UUID.
        """
        mapped = self._mapped_value_for_uuid(uuid)
        return mapped.uuid if mapped is not None else uuid

    def _resolve_mapped_carrier(
        self,
        base_uuid: str,
        suffix: str,
    ) -> tuple[ValueBase, str] | None:
        """Resolve a carrier-key base through the map, folding view indices.

        Composite carrier keys index into the ROOT array's element space.
        When the base maps to a strided slice view (e.g. an inline pass
        substituting a callee formal with the caller's ``q[1::2]``), keeping
        the index verbatim would silently re-interpret it in view-local
        space, so the index is folded through the view's slice chain into
        the root array's index space instead.

        Args:
            base_uuid (str): Base UUID of a legacy ``"<base>_<index>"`` key.
            suffix (str): Decimal index suffix of the key.

        Returns:
            tuple[ValueBase, str] | None: Mapped value and translated index
                suffix, or ``None`` when the base is not mapped. When the
                mapped value is a constant-bound slice view, the chain root
                and the folded root-space index are returned; otherwise the
                mapped value and the original suffix are returned unchanged.
        """
        mapped = self._mapped_value_for_uuid(base_uuid)
        if mapped is None:
            return None
        if isinstance(mapped, ArrayValue) and mapped.slice_of is not None:
            resolved = resolve_root_array_index(mapped, int(suffix))
            if resolved is not None:
                root, root_index = resolved
                return root, str(root_index)
        return mapped, suffix

    def _substitute_uuid_ref(self, uuid_ref: str) -> str:
        """Substitute a scalar or legacy indexed UUID reference.

        Args:
            uuid_ref (str): Scalar UUID or legacy ``"<uuid>_<index>"`` key.

        Returns:
            str: Substituted UUID reference. Index suffixes are preserved for
                plain remappings and folded into root-array space when the
                base maps to a constant-bound slice view.
        """
        parts = split_indexed_identifier(uuid_ref)
        if parts is None:
            return self._substitute_uuid(uuid_ref)
        base_uuid, suffix = parts
        resolved = self._resolve_mapped_carrier(base_uuid, suffix)
        if resolved is None:
            return uuid_ref
        mapped, new_suffix = resolved
        return f"{mapped.uuid}_{new_suffix}"

    def _substitute_logical_id_for_uuid_ref(
        self,
        logical_id: str,
        uuid_ref: str,
    ) -> str:
        """Substitute a logical-id reference using its parallel UUID key.

        Args:
            logical_id (str): Logical-id reference to substitute.
            uuid_ref (str): Parallel UUID reference from the same metadata
                position.

        Returns:
            str: Replacement logical-id reference when the UUID base is mapped,
                otherwise ``logical_id`` unchanged. The index suffix follows
                the same root-space folding as :meth:`_substitute_uuid_ref` so
                UUID and logical-id keys stay parallel.
        """
        parts = split_indexed_identifier(uuid_ref)
        if parts is None:
            mapped = self._mapped_value_for_uuid(uuid_ref)
            return mapped.logical_id if mapped is not None else logical_id

        base_uuid, suffix = parts
        resolved = self._resolve_mapped_carrier(base_uuid, suffix)
        if resolved is None:
            return logical_id
        mapped, new_suffix = resolved
        return f"{mapped.logical_id}_{new_suffix}"

    def _substitute_parallel_logical_ids(
        self,
        logical_ids: tuple[str, ...],
        uuid_refs: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Substitute logical IDs that are parallel to UUID references.

        Args:
            logical_ids (tuple[str, ...]): Logical-id metadata entries.
            uuid_refs (tuple[str, ...]): UUID entries at the same positions.

        Returns:
            tuple[str, ...]: Logical IDs substituted by looking at the
                corresponding UUID references where available.
        """
        rewritten: list[str] = []
        for i, logical_id in enumerate(logical_ids):
            if i < len(uuid_refs):
                rewritten.append(
                    self._substitute_logical_id_for_uuid_ref(logical_id, uuid_refs[i])
                )
            else:
                rewritten.append(logical_id)
        return tuple(rewritten)

    def _metadata_has_referenced_uuid(self, metadata: ValueMetadata) -> bool:
        """Return whether metadata contains a UUID present in the value map.

        Args:
            metadata (ValueMetadata): Metadata bundle to inspect.

        Returns:
            bool: ``True`` when an embedded scalar or carrier-key base UUID is
                present in ``self._value_map``.
        """

        def has_uuid_ref(uuid_ref: str) -> bool:
            parts = split_indexed_identifier(uuid_ref)
            uuid = parts[0] if parts is not None else uuid_ref
            return uuid in self._value_map

        if metadata.cast is not None:
            if has_uuid_ref(metadata.cast.source_uuid):
                return True
            if any(has_uuid_ref(uuid_ref) for uuid_ref in metadata.cast.qubit_uuids):
                return True
        if metadata.qfixed is not None and any(
            has_uuid_ref(uuid_ref) for uuid_ref in metadata.qfixed.qubit_uuids
        ):
            return True
        if metadata.array_runtime is not None:
            if any(
                has_uuid_ref(uuid_ref)
                for uuid_ref in metadata.array_runtime.element_uuids
            ):
                return True
            if any(
                uuid_ref in self._value_map
                for uuid_ref in metadata.array_runtime.element_parent_uuids
                if uuid_ref
            ):
                return True
        return False

    def _substitute_metadata(self, metadata: ValueMetadata) -> ValueMetadata:
        """Substitute UUID and logical-id references inside metadata.

        Args:
            metadata (ValueMetadata): Metadata bundle to substitute.

        Returns:
            ValueMetadata: Metadata with embedded references rewritten through
                the current value map.
        """
        new_cast = metadata.cast
        if new_cast is not None:
            new_cast = CastMetadata(
                source_uuid=self._substitute_uuid(new_cast.source_uuid),
                qubit_uuids=tuple(
                    self._substitute_uuid_ref(uuid_ref)
                    for uuid_ref in new_cast.qubit_uuids
                ),
                source_logical_id=(
                    self._substitute_logical_id_for_uuid_ref(
                        new_cast.source_logical_id,
                        new_cast.source_uuid,
                    )
                    if new_cast.source_logical_id is not None
                    else None
                ),
                qubit_logical_ids=self._substitute_parallel_logical_ids(
                    new_cast.qubit_logical_ids,
                    new_cast.qubit_uuids,
                ),
            )

        new_qfixed = metadata.qfixed
        if new_qfixed is not None:
            new_qfixed = QFixedMetadata(
                qubit_uuids=tuple(
                    self._substitute_uuid_ref(uuid_ref)
                    for uuid_ref in new_qfixed.qubit_uuids
                ),
                num_bits=new_qfixed.num_bits,
                int_bits=new_qfixed.int_bits,
            )

        new_array_rt = metadata.array_runtime
        if new_array_rt is not None:
            new_array_rt = ArrayRuntimeMetadata(
                const_array=new_array_rt.const_array,
                element_uuids=tuple(
                    self._substitute_uuid_ref(uuid_ref)
                    for uuid_ref in new_array_rt.element_uuids
                ),
                element_logical_ids=self._substitute_parallel_logical_ids(
                    new_array_rt.element_logical_ids,
                    new_array_rt.element_uuids,
                ),
                element_parent_uuids=tuple(
                    self._substitute_uuid(uuid_ref) if uuid_ref else uuid_ref
                    for uuid_ref in new_array_rt.element_parent_uuids
                ),
                element_parent_indices=new_array_rt.element_parent_indices,
            )

        return ValueMetadata(
            scalar=metadata.scalar,
            cast=new_cast,
            qfixed=new_qfixed,
            array_runtime=new_array_rt,
            dict_runtime=metadata.dict_runtime,
        )

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

        if isinstance(result, CastOperation) and result.qubit_mapping:
            new_qubit_mapping = [
                self._substitute_uuid_ref(uuid_ref) for uuid_ref in result.qubit_mapping
            ]
            if new_qubit_mapping != result.qubit_mapping:
                result = dataclasses.replace(result, qubit_mapping=new_qubit_mapping)

        return result

    def _chase_transitive(self, v: ValueBase) -> ValueBase:
        """Chase transitive chains in the value map with cycle detection."""
        result = self._mapped_value_for_uuid(v.uuid)
        if result is None:
            raise KeyError(v.uuid)
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
        if self._metadata_has_referenced_uuid(v.metadata):
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
        new_metadata = self._substitute_metadata(v.metadata)

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
                metadata=new_metadata,
                parent_array=new_parent_array,
                element_indices=new_element_indices,
                shape=new_shape,
                slice_of=new_slice_of,
                slice_start=new_slice_start,
                slice_step=new_slice_step,
            )

        return dataclasses.replace(
            v,
            metadata=new_metadata,
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
            new_metadata = self._substitute_metadata(v.metadata)
            changed = new_metadata != v.metadata
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
                return dataclasses.replace(
                    v,
                    metadata=new_metadata,
                    elements=tuple(new_elements),
                )
            return v

        # Handle DictValue - substitute entries
        if isinstance(v, DictValue):
            new_entries: list[tuple[TupleValue | Value, Value]] = []
            new_metadata = self._substitute_metadata(v.metadata)
            changed = new_metadata != v.metadata
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
                return dataclasses.replace(
                    v,
                    metadata=new_metadata,
                    entries=tuple(new_entries),
                )
            return v

        # Handle regular Value/ArrayValue
        if isinstance(v, Value):
            new_parent_array = v.parent_array
            new_element_indices = v.element_indices
            new_shape: tuple[Value, ...] | None = None
            new_metadata = self._substitute_metadata(v.metadata)
            changed = new_metadata != v.metadata

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

            if changed:
                if isinstance(v, ArrayValue):
                    replace_kwargs: dict[str, Any] = dict(
                        metadata=new_metadata,
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
                    metadata=new_metadata,
                    parent_array=new_parent_array,
                    element_indices=new_element_indices,
                )

        return v
