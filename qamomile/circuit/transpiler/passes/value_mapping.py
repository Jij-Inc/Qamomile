"""Value mapping utilities for IR transformations."""

from __future__ import annotations

import dataclasses
import uuid as uuid_module
from typing import Any, cast

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueMetadata,
    remap_indexed_identifier,
    remap_value_metadata_references,
)
from qamomile.circuit.ir.value_mapping import ValueSubstitutor

__all__ = ["UUIDRemapper", "ValueSubstitutor"]


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
        """Clone a list of operations with fresh UUIDs.

        Args:
            operations (list[Operation]): Operations to clone in order.

        Returns:
            list[Operation]: Cloned operations, each with fresh UUIDs, in the
                same order as ``operations``.
        """
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

        Args:
            op (Operation): Operation to clone.

        Returns:
            Operation: A clone of ``op`` with every owned Value (operands,
                results, subclass-extra fields) and every nested-body Value
                reassigned a fresh UUID / logical_id, and with
                ``CastOperation.qubit_mapping`` carrier keys remapped.
        """
        # Build a uuid -> cloned_value substitution map. Operands are
        # cloned first so nested-body metadata referencing an outer value
        # (e.g. a parent array passed as an operand) resolves through the
        # remap tables while the bodies are cloned below.
        sub_map: dict[str, ValueBase] = {}
        for v in op.operands:
            if isinstance(v, ValueBase):
                sub_map[v.uuid] = self.clone_value(v)

        # Clone nested bodies BEFORE subclass-extra values and results.
        # IfOperation yields and merge outputs carry metadata that may
        # reference values whose first (and only) appearance is inside
        # the branch bodies — e.g. QFixed carrier keys pointing at an
        # array that is cast inside the ``if``. Cloning the bodies first
        # fills the remap tables so ``_clone_metadata`` can resolve those
        # references; the value cache then hands the loops below the same
        # clones the bodies produced.
        cloned_lists: list[list[Operation]] | None = None
        if isinstance(op, HasNestedOps):
            cloned_lists = [
                self.clone_operations(op_list) for op_list in op.nested_op_lists()
            ]

        # Subclass extras (loop_var_value, rebind records, if-merge
        # yields) exposed via all_input_values; already-cloned operands
        # hit the sub_map guard.
        for v in op.all_input_values():
            if v.uuid not in sub_map:
                sub_map[v.uuid] = self.clone_value(v)

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
        the unified ValueBase protocol. Nested values (tuple elements, dict
        entries, ``parent_array`` / ``element_indices`` / ``shape`` / slice
        fields) and embedded metadata references are cloned consistently.
        Results are cached by source UUID so a repeated clone returns the
        same instance.

        Args:
            value (ValueBase): The value to clone.

        Returns:
            ValueBase: The cloned value of the same concrete type, with a
                fresh UUID / logical_id and its nested values and metadata
                references remapped through this remapper.
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
