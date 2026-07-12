"""Provide shared IR value substitution utilities."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, cast

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.value import (
    ArrayRuntimeMetadata,
    ArrayValue,
    CastMetadata,
    DictValue,
    QFixedMetadata,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
    ValueMetadata,
    resolve_root_array_index,
    resolve_root_qubit_address,
    split_indexed_identifier,
)

__all__ = ["ValueSubstitutor"]


class ValueSubstitutor:
    """Substitute IR values in operations using a UUID-keyed mapping.

    Args:
        value_map (Mapping[str, ValueBase]): Mapping from original value UUIDs
            to replacement values.
        transitive (bool): Whether substitutions should chase chains such
            as ``A -> B -> C`` to the terminal value. Defaults to False.
    """

    def __init__(self, value_map: Mapping[str, ValueBase], transitive: bool = False):
        """Initialize the substitutor.

        Args:
            value_map (Mapping[str, ValueBase]): Mapping from original value
                UUIDs to replacement values.
            transitive (bool): Whether substitutions should chase chains
                to their terminal value. Defaults to False.
        """
        self._value_map = value_map
        self._transitive = transitive

    def substitute_operation(self, op: Operation) -> Operation:
        """Substitute values in an operation.

        Args:
            op (Operation): Operation whose operands, results, and
                subclass-specific value fields should be substituted.

        Returns:
            Operation: Operation with all mapped value references replaced.
        """
        sub_map: dict[str, ValueBase] = {}
        for value in op.all_input_values():
            substituted = self.substitute_value(value)
            if substituted is not value:
                sub_map[value.uuid] = substituted
        for value in op.results:
            if isinstance(value, ValueBase):
                substituted = self.substitute_value(value)
                if substituted is not value:
                    sub_map[value.uuid] = substituted

        result = op.replace_values(sub_map) if sub_map else op

        # ``CastOperation.qubit_mapping`` holds carrier keys as a bare
        # operation field, so ``replace_values`` does not reach it; substitute
        # it explicitly to keep it in lockstep with the cast result metadata.
        if isinstance(result, CastOperation) and result.qubit_mapping:
            new_qubit_mapping = [
                self._substitute_uuid_ref(uuid_ref) for uuid_ref in result.qubit_mapping
            ]
            if new_qubit_mapping != result.qubit_mapping:
                result = dataclasses.replace(result, qubit_mapping=new_qubit_mapping)

        return result

    def _chase_transitive(self, value: ValueBase) -> ValueBase:
        """Chase a mapped value through transitive substitutions.

        Args:
            value (ValueBase): Value whose UUID must exist in `value_map`.

        Returns:
            ValueBase: Direct replacement, or terminal replacement when
            transitive substitution is enabled.
        """
        result = self._value_map[value.uuid]
        if self._transitive:
            seen: set[str] = {value.uuid}
            while result.uuid in self._value_map and result.uuid not in seen:
                seen.add(result.uuid)
                result = self._value_map[result.uuid]
        return result

    def _has_referenced_field(self, value: Value) -> bool:
        """Return whether a value field references a mappable UUID.

        Args:
            value (Value): Value to inspect for mappable parent, index,
                shape, slice, or typed metadata references.

        Returns:
            bool: True if any referenced field is present in `value_map`.
        """
        if (
            value.parent_array is not None
            and value.parent_array.uuid in self._value_map
        ):
            return True
        if value.element_indices:
            for idx in value.element_indices:
                if idx.uuid in self._value_map:
                    return True
        if isinstance(value, ArrayValue):
            for dim in value.shape:
                if dim.uuid in self._value_map:
                    return True
            if value.slice_of is not None and value.slice_of.uuid in self._value_map:
                return True
            if (
                value.slice_start is not None
                and value.slice_start.uuid in self._value_map
            ):
                return True
            if (
                value.slice_step is not None
                and value.slice_step.uuid in self._value_map
            ):
                return True
        if self._metadata_has_referenced_field(value.metadata):
            return True
        return False

    def _mapped_value_for_uuid(self, uuid: str) -> ValueBase | None:
        """Return the replacement value for a referenced UUID.

        Args:
            uuid (str): Referenced Value UUID.

        Returns:
            ValueBase | None: Replacement value, or None when `uuid` is not
                present in the value map.
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

        Raises:
            ValueError: If the base maps to a strided slice view whose slice
                bounds are not compile-time constants, so the carrier index
                cannot be folded into root-array space. Emitting the verbatim
                view-local key would leave the carrier unresolvable at emit
                (silently dropping the measurement), so this fails fast,
                mirroring the frontend's rejection of symbolic-bound slice
                views for ``cast()``.
        """
        mapped = self._mapped_value_for_uuid(base_uuid)
        if mapped is None:
            return None
        if isinstance(mapped, ArrayValue) and mapped.slice_of is not None:
            resolved = resolve_root_array_index(mapped, int(suffix))
            if resolved is None:
                raise ValueError(
                    "Cannot remap a QFixed carrier key onto a slice view with "
                    "symbolic slice bounds: the carrier index cannot be folded "
                    "into root-array space, so the measurement would be "
                    "silently dropped at emit. Use literal-bounded slicing for "
                    "cast operands."
                )
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

    def _metadata_has_referenced_field(self, metadata: ValueMetadata) -> bool:
        """Return whether typed metadata references a mappable UUID.

        Args:
            metadata (ValueMetadata): Metadata to inspect.

        Returns:
            bool: True when a metadata UUID reference is present in the
                value map.
        """

        def has_uuid_ref(uuid_ref: str) -> bool:
            """Return True when a scalar UUID or carrier-key base is mapped."""
            parts = split_indexed_identifier(uuid_ref)
            uuid = parts[0] if parts is not None else uuid_ref
            return uuid in self._value_map

        if metadata.cast is not None:
            if has_uuid_ref(metadata.cast.source_uuid):
                return True
            if any(has_uuid_ref(uuid) for uuid in metadata.cast.qubit_uuids):
                return True
        if metadata.qfixed is not None:
            if any(has_uuid_ref(uuid) for uuid in metadata.qfixed.qubit_uuids):
                return True
        if metadata.array_runtime is not None:
            if any(has_uuid_ref(uuid) for uuid in metadata.array_runtime.element_uuids):
                return True
            return any(
                uuid and uuid in self._value_map
                for uuid in metadata.array_runtime.element_parent_uuids
            )
        return False

    def _resolve_replaced_parent_address(
        self,
        parent: ArrayValue,
        index: int,
    ) -> tuple[str, int] | None:
        """Resolve an index through a substituted array parent.

        Args:
            parent (ArrayValue): Replacement array parent, possibly a sliced
                view.
            index (int): Element index in `parent` before slice-chain
                composition.

        Returns:
            tuple[str, int] | None: Resolved parent UUID and composed index.
                The parent is the root array when every slice in the chain
                has concrete bounds; otherwise it is the deepest substituted
                symbolic slice that can still address ``index`` locally.
                Returns None when no parent can be resolved.
        """
        current: ArrayValue | None = parent
        resolved_index = index
        while current is not None and current.slice_of is not None:
            if (
                current.slice_start is None
                or current.slice_step is None
                or not current.slice_start.is_constant()
                or not current.slice_step.is_constant()
            ):
                return current.uuid, resolved_index
            start = int(current.slice_start.get_const())
            step = int(current.slice_step.get_const())
            resolved_index = start + step * resolved_index
            current = current.slice_of
        if current is None:
            return None
        return current.uuid, resolved_index

    def _substitute_metadata(self, metadata: ValueMetadata) -> ValueMetadata:
        """Substitute UUID and logical-id references inside typed metadata.

        Args:
            metadata (ValueMetadata): Metadata bundle to inspect.

        Returns:
            ValueMetadata: Metadata with mapped references rewritten, or the
                original metadata when nothing changes.
        """
        new_cast = metadata.cast
        if new_cast is not None:
            source_replacement = self._mapped_value_for_uuid(new_cast.source_uuid)
            source_uuid = (
                source_replacement.uuid
                if source_replacement is not None
                else new_cast.source_uuid
            )
            source_logical_id = new_cast.source_logical_id
            if source_replacement is not None and source_logical_id is not None:
                source_logical_id = source_replacement.logical_id

            # Carrier qubit keys may be scalar UUIDs or legacy composite
            # ``"<root_uuid>_<index>"`` keys; substitute base UUIDs while
            # folding view-local indices into root-array space.
            qubit_uuids = tuple(
                self._substitute_uuid_ref(uuid_ref) for uuid_ref in new_cast.qubit_uuids
            )
            qubit_logical_ids = self._substitute_parallel_logical_ids(
                new_cast.qubit_logical_ids,
                new_cast.qubit_uuids,
            )
            new_cast = CastMetadata(
                source_uuid=source_uuid,
                qubit_uuids=qubit_uuids,
                source_logical_id=source_logical_id,
                qubit_logical_ids=qubit_logical_ids,
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
            element_replacements = [
                self._mapped_value_for_uuid(uuid) for uuid in new_array_rt.element_uuids
            ]
            element_uuids = tuple(
                self._substitute_uuid_ref(uuid_ref)
                for uuid_ref in new_array_rt.element_uuids
            )
            element_logical_ids = self._substitute_parallel_logical_ids(
                new_array_rt.element_logical_ids,
                new_array_rt.element_uuids,
            )
            element_parent_uuids: tuple[str, ...] = ()
            element_parent_indices: tuple[int, ...] = ()
            parent_uuids: list[str] = []
            parent_indices: list[int] = []
            should_write_parent_metadata = False
            parent_count = max(
                len(element_replacements),
                len(new_array_rt.element_parent_uuids),
            )
            for i in range(parent_count):
                replacement = (
                    element_replacements[i] if i < len(element_replacements) else None
                )
                root_addr = (
                    resolve_root_qubit_address(replacement)
                    if isinstance(replacement, Value)
                    else None
                )
                if root_addr is not None:
                    parent_uuid, parent_index = root_addr
                    parent_uuids.append(parent_uuid)
                    parent_indices.append(parent_index)
                    should_write_parent_metadata = True
                    continue

                old_parent_uuid = (
                    new_array_rt.element_parent_uuids[i]
                    if i < len(new_array_rt.element_parent_uuids)
                    else ""
                )
                old_parent_index = (
                    new_array_rt.element_parent_indices[i]
                    if i < len(new_array_rt.element_parent_indices)
                    else -1
                )
                if old_parent_uuid:
                    parent_replacement = self._mapped_value_for_uuid(old_parent_uuid)
                    if (
                        isinstance(parent_replacement, ArrayValue)
                        and old_parent_index >= 0
                    ):
                        parent_addr = self._resolve_replaced_parent_address(
                            parent_replacement,
                            old_parent_index,
                        )
                        if parent_addr is not None:
                            old_parent_uuid, old_parent_index = parent_addr
                    elif parent_replacement is not None:
                        old_parent_uuid = parent_replacement.uuid
                parent_uuids.append(old_parent_uuid)
                parent_indices.append(old_parent_index)
                should_write_parent_metadata = should_write_parent_metadata or bool(
                    old_parent_uuid
                )

            if should_write_parent_metadata or new_array_rt.element_parent_uuids:
                element_parent_uuids = tuple(parent_uuids)
                element_parent_indices = tuple(parent_indices)
            new_array_rt = ArrayRuntimeMetadata(
                const_array=new_array_rt.const_array,
                element_uuids=element_uuids,
                element_logical_ids=element_logical_ids,
                element_parent_uuids=element_parent_uuids,
                element_parent_indices=element_parent_indices,
            )

        if (
            new_cast == metadata.cast
            and new_qfixed == metadata.qfixed
            and new_array_rt == metadata.array_runtime
        ):
            return metadata

        return dataclasses.replace(
            metadata,
            cast=new_cast,
            qfixed=new_qfixed,
            array_runtime=new_array_rt,
        )

    def _resubstitute_fields(self, value: Value) -> Value:
        """Rebuild a mapped value with its metadata fields substituted.

        Args:
            value (Value): Value returned by a direct lookup whose own
                metadata still references mappable UUIDs.

        Returns:
            Value: Rebuilt value with referenced fields substituted.
        """
        new_parent_array = value.parent_array
        if value.parent_array is not None:
            sub_parent_array = self.substitute_value(value.parent_array)
            if (
                isinstance(sub_parent_array, ArrayValue)
                and sub_parent_array is not value.parent_array
            ):
                new_parent_array = sub_parent_array

        new_element_indices = value.element_indices
        if value.element_indices:
            new_indices: list[Value] = []
            for idx in value.element_indices:
                if idx.uuid in self._value_map:
                    sub_idx = self._chase_transitive(idx)
                    if isinstance(sub_idx, Value):
                        new_indices.append(sub_idx)
                        continue
                new_indices.append(idx)
            new_element_indices = tuple(new_indices)

        new_metadata = self._substitute_metadata(value.metadata)
        metadata_changed = new_metadata is not value.metadata

        if isinstance(value, ArrayValue):
            new_shape = self._substitute_shape(value.shape)

            new_slice_of = value.slice_of
            if value.slice_of is not None:
                sub_slice_of = self.substitute_value(value.slice_of)
                if (
                    isinstance(sub_slice_of, ArrayValue)
                    and sub_slice_of is not value.slice_of
                ):
                    new_slice_of = sub_slice_of
            new_slice_start = value.slice_start
            if (
                value.slice_start is not None
                and value.slice_start.uuid in self._value_map
            ):
                sub_slice_start = self._chase_transitive(value.slice_start)
                if isinstance(sub_slice_start, Value):
                    new_slice_start = sub_slice_start
            new_slice_step = value.slice_step
            if (
                value.slice_step is not None
                and value.slice_step.uuid in self._value_map
            ):
                sub_slice_step = self._chase_transitive(value.slice_step)
                if isinstance(sub_slice_step, Value):
                    new_slice_step = sub_slice_step

            return dataclasses.replace(
                value,
                parent_array=new_parent_array,
                element_indices=new_element_indices,
                shape=new_shape,
                slice_of=new_slice_of,
                slice_start=new_slice_start,
                slice_step=new_slice_step,
                metadata=new_metadata,
            )

        return dataclasses.replace(
            value,
            parent_array=new_parent_array,
            element_indices=new_element_indices,
            metadata=new_metadata if metadata_changed else value.metadata,
        )

    def _substitute_shape(self, shape: tuple[Value, ...]) -> tuple[Value, ...]:
        """Substitute shape dimension values.

        Args:
            shape (tuple[Value, ...]): Shape dimensions to inspect.

        Returns:
            tuple[Value, ...]: Shape dimensions with mapped values replaced.
        """
        new_shape: list[Value] = []
        for dim in shape:
            if dim.uuid in self._value_map:
                sub_dim = self._chase_transitive(dim)
                if isinstance(sub_dim, Value):
                    new_shape.append(sub_dim)
                    continue
            new_shape.append(dim)
        return tuple(new_shape)

    def _substitute_tuple_value(self, value: TupleValue) -> ValueBase:
        """Substitute elements inside a tuple value.

        Args:
            value (TupleValue): Tuple value to inspect.

        Returns:
            ValueBase: Rebuilt tuple when any element changes; otherwise
            the original tuple value.
        """
        new_elements: list[ValueLike] = []
        changed = False
        for elem in value.elements:
            substituted = self.substitute_value(elem)
            new_elements.append(cast(ValueLike, substituted))
            if substituted is not elem:
                changed = True
        if changed:
            return dataclasses.replace(value, elements=tuple(new_elements))
        return value

    def _substitute_dict_value(self, value: DictValue) -> ValueBase:
        """Substitute keys and values inside a dict value.

        Args:
            value (DictValue): Dict value to inspect.

        Returns:
            ValueBase: Rebuilt dict when any entry changes; otherwise the
            original dict value.
        """
        new_entries: list[tuple[TupleValue | Value, Value]] = []
        changed = False
        for key, entry_value in value.entries:
            new_key = key
            new_value = entry_value
            sub_key = self.substitute_value(key)
            if isinstance(sub_key, (TupleValue, Value)):
                new_key = sub_key
                if sub_key is not key:
                    changed = True
            sub_value = self.substitute_value(entry_value)
            if isinstance(sub_value, Value):
                new_value = sub_value
                if sub_value is not entry_value:
                    changed = True
            new_entries.append((new_key, new_value))
        if changed:
            return dataclasses.replace(value, entries=tuple(new_entries))
        return value

    def _substitute_value_fields(self, value: Value) -> ValueBase:
        """Substitute metadata fields on a scalar or array value.

        Args:
            value (Value): Value whose parent, index, shape, and slice
                metadata should be inspected.

        Returns:
            ValueBase: Rebuilt value when any field changes; otherwise the
            original value.
        """
        new_parent_array = value.parent_array
        new_element_indices = value.element_indices
        new_shape: tuple[Value, ...] | None = None
        changed = False

        if value.parent_array is not None:
            sub_parent = self.substitute_value(value.parent_array)
            if (
                isinstance(sub_parent, ArrayValue)
                and sub_parent is not value.parent_array
            ):
                new_parent_array = sub_parent
                changed = True

        if value.element_indices:
            new_indices: list[Value] = []
            indices_changed = False
            for idx in value.element_indices:
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

        if isinstance(value, ArrayValue) and value.shape:
            substituted_shape = self._substitute_shape(value.shape)
            if substituted_shape != value.shape:
                new_shape = substituted_shape
                changed = True

        new_metadata = self._substitute_metadata(value.metadata)
        if new_metadata is not value.metadata:
            changed = True

        new_slice_of: ArrayValue | None = None
        new_slice_start: Value | None = None
        new_slice_step: Value | None = None
        slice_meta_changed = False
        if isinstance(value, ArrayValue):
            new_slice_of = value.slice_of
            new_slice_start = value.slice_start
            new_slice_step = value.slice_step

            if value.slice_of is not None:
                sub_slice_of = self.substitute_value(value.slice_of)
                if (
                    isinstance(sub_slice_of, ArrayValue)
                    and sub_slice_of is not value.slice_of
                ):
                    new_slice_of = sub_slice_of
                    slice_meta_changed = True

            if (
                value.slice_start is not None
                and value.slice_start.uuid in self._value_map
            ):
                sub_slice_start = self._chase_transitive(value.slice_start)
                if isinstance(sub_slice_start, Value):
                    new_slice_start = sub_slice_start
                    slice_meta_changed = True

            if (
                value.slice_step is not None
                and value.slice_step.uuid in self._value_map
            ):
                sub_slice_step = self._chase_transitive(value.slice_step)
                if isinstance(sub_slice_step, Value):
                    new_slice_step = sub_slice_step
                    slice_meta_changed = True

            if slice_meta_changed:
                changed = True

        if changed:
            if isinstance(value, ArrayValue):
                replace_kwargs: dict[str, Any] = dict(
                    parent_array=new_parent_array,
                    element_indices=new_element_indices,
                )
                if new_shape is not None:
                    replace_kwargs["shape"] = new_shape
                if slice_meta_changed:
                    replace_kwargs["slice_of"] = new_slice_of
                    replace_kwargs["slice_start"] = new_slice_start
                    replace_kwargs["slice_step"] = new_slice_step
                replace_kwargs["metadata"] = new_metadata
                return dataclasses.replace(value, **replace_kwargs)
            return dataclasses.replace(
                value,
                parent_array=new_parent_array,
                element_indices=new_element_indices,
                metadata=new_metadata,
            )

        return value

    def substitute_value(self, value: ValueBase) -> ValueBase:
        """Substitute a single value.

        Args:
            value (ValueBase): Value to replace or rebuild.

        Returns:
            ValueBase: Replacement value, rebuilt value with substituted
            metadata, or the original value when nothing maps.
        """
        if value.uuid in self._value_map:
            mapped = self._chase_transitive(value)
            if (
                isinstance(mapped, Value)
                and mapped is not value
                and self._has_referenced_field(mapped)
            ):
                return self._resubstitute_fields(mapped)
            return mapped

        if isinstance(value, TupleValue):
            return self._substitute_tuple_value(value)

        if isinstance(value, DictValue):
            return self._substitute_dict_value(value)

        if isinstance(value, Value):
            return self._substitute_value_fields(value)

        return value
