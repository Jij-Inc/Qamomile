"""Clone IR values and blocks into a fresh identity namespace."""

from __future__ import annotations

import dataclasses
import uuid as uuid_module
from typing import Any, cast

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
)
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
    ValueMetadata,
    collect_value_like_uuids,
    remap_indexed_identifier,
    remap_value_metadata_references,
)

__all__ = ["UUIDRemapper"]


class UUIDRemapper:
    """Clones values and operations with fresh UUIDs and logical_ids.

    Used during inlining to create unique identities for values
    when a block is called multiple times.
    """

    def __init__(self):
        """Initialize empty identity, value, and block remapping caches."""
        self._uuid_remap: dict[str, str] = {}
        self._logical_id_remap: dict[str, str] = {}
        self._source_values_by_uuid: dict[str, ValueBase] = {}
        # Cache for all value types (uses ValueBase for unified handling)
        self._value_cache: dict[str, ValueBase] = {}
        self._block_cache: dict[int, Block] = {}
        self._active_blocks: set[int] = set()
        self._owned_uuid_stack: list[set[str]] = []
        self._clone_owned_blocks = False

    @property
    def uuid_remap(self) -> dict[str, str]:
        """Get the mapping from old UUIDs to new UUIDs.

        Returns:
            dict[str, str]: Mutable remapping accumulated by this instance.
        """
        return self._uuid_remap

    @property
    def logical_id_remap(self) -> dict[str, str]:
        """Get the mapping from old logical IDs to new logical IDs.

        Returns:
            dict[str, str]: Mutable remapping accumulated by this instance.
        """
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

    def clone_block(self, block: Block) -> Block:
        """Clone an executable block namespace with fresh value identities.

        Callable definitions and inverse source blocks are semantic provenance
        shared with the original graph. Transform implementations, controlled
        bodies, and SELECT cases are executable children and are cloned
        recursively.

        Args:
            block (Block): Block whose interface, operations, and executable
                owned blocks should be cloned.

        Returns:
            Block: Structurally equivalent block whose values have fresh UUIDs
                and logical IDs.

        Raises:
            ValueError: If the owned block graph is recursive.
        """
        cached = self._block_cache.get(id(block))
        if cached is not None:
            return cached
        if id(block) in self._active_blocks:
            raise ValueError("Cannot clone a recursive block graph.")

        previous_clone_owned_blocks = self._clone_owned_blocks
        self._clone_owned_blocks = True
        self._active_blocks.add(id(block))
        owned_uuids = self._collect_owned_uuids(block)
        self._owned_uuid_stack.append(owned_uuids)
        try:
            self._reserve_block_identities(block, owned_uuids)
            input_values = [
                cast(ValueLike, self.clone_value(value)) for value in block.input_values
            ]
            parameters = {
                name: cast(Value, self.clone_value(value))
                for name, value in block.parameters.items()
            }
            static_bindings = tuple(
                dataclasses.replace(
                    slot,
                    fields=tuple(
                        dataclasses.replace(
                            field,
                            value=cast(Value, self.clone_value(field.value)),
                        )
                        for field in slot.fields
                    ),
                )
                for slot in block.static_bindings
            )
            operations = self.clone_operations(block.operations)
            output_values = [
                cast(ValueLike, self.clone_value(value))
                for value in block.output_values
            ]
            cloned = Block(
                name=block.name,
                label_args=list(block.label_args),
                input_values=input_values,
                output_values=output_values,
                output_names=list(block.output_names),
                operations=operations,
                kind=block.kind,
                parameters=parameters,
                param_slots=block.param_slots,
                static_bindings=static_bindings,
            )
            self._block_cache[id(block)] = cloned
            return cloned
        finally:
            self._owned_uuid_stack.pop()
            self._active_blocks.remove(id(block))
            self._clone_owned_blocks = previous_clone_owned_blocks

    def _collect_owned_uuids(self, block: Block) -> set[str]:
        """Collect identities defined in one block's lexical scope.

        Args:
            block (Block): Block whose local producers should be collected.

        Returns:
            set[str]: UUIDs that must be alpha-renamed with the block.
        """
        owned: set[str] = set()
        referenced_values: list[ValueBase] = []
        referenced_value_ids: set[int] = set()

        def remember(value: ValueBase) -> None:
            """Remember a reachable value and all structural children.

            Args:
                value (ValueBase): Reachable source value.
            """
            if id(value) in referenced_value_ids:
                return
            referenced_value_ids.add(id(value))
            referenced_values.append(value)
            for child in self._value_children(value):
                remember(child)

        for value in (*block.input_values, *block.parameters.values()):
            owned.update(collect_value_like_uuids(value))
            remember(value)
        for slot in block.static_bindings:
            for field in slot.fields:
                owned.update(collect_value_like_uuids(field.value))
                remember(field.value)

        def collect_operations(operations: list[Operation]) -> None:
            """Collect operation results and lexical binders recursively.

            Args:
                operations (list[Operation]): Operations in producer order.
            """
            for operation in operations:
                for operand in operation.all_input_values():
                    remember(operand)
                    if operand.type.is_quantum():
                        self._collect_result_uuids(cast(ValueLike, operand), owned)
                for result in operation.results:
                    remember(result)
                    self._collect_result_uuids(result, owned)
                if (
                    isinstance(operation, ForOperation)
                    and operation.loop_var_value is not None
                ):
                    remember(operation.loop_var_value)
                    self._collect_result_uuids(operation.loop_var_value, owned)
                if isinstance(operation, ForItemsOperation):
                    for key in operation.key_var_values or ():
                        remember(key)
                        self._collect_result_uuids(key, owned)
                    if operation.value_var_value is not None:
                        remember(operation.value_var_value)
                        self._collect_result_uuids(operation.value_var_value, owned)
                for region_arg in getattr(operation, "region_args", ()):
                    remember(region_arg.block_arg)
                    self._collect_result_uuids(region_arg.block_arg, owned)
                if isinstance(operation, HasNestedOps):
                    for nested in operation.nested_op_lists():
                        collect_operations(nested)

        collect_operations(block.operations)
        for value in block.output_values:
            remember(value)

        changed = True
        while changed:
            changed = False
            for value in referenced_values:
                if value.uuid in owned:
                    continue
                if any(
                    child.uuid in owned or child.uuid in self._uuid_remap
                    for child in self._value_children(value)
                ):
                    owned.add(value.uuid)
                    changed = True
        return owned

    @classmethod
    def _collect_result_uuids(
        cls,
        value: ValueLike,
        owned: set[str],
    ) -> None:
        """Collect a produced value without claiming free dependencies.

        Args:
            value (ValueLike): Operation result or lexical binder.
            owned (set[str]): UUID set updated in place.
        """
        owned.add(value.uuid)
        if isinstance(value, TupleValue):
            for element in value.elements:
                cls._collect_result_uuids(element, owned)
        elif isinstance(value, DictValue):
            for key, entry_value in value.entries:
                cls._collect_result_uuids(key, owned)
                cls._collect_result_uuids(entry_value, owned)

    def _reserve_block_identities(
        self,
        block: Block,
        owned_uuids: set[str],
    ) -> None:
        """Reserve fresh IDs before rebuilding values and their metadata.

        Args:
            block (Block): Block whose reachable values should be scanned.
            owned_uuids (set[str]): Locally defined UUIDs to reserve.
        """
        seen: set[str] = set()

        def reserve(value: ValueBase) -> None:
            """Reserve one owned value and its owned structural children.

            Args:
                value (ValueBase): Reachable value occurrence.
            """
            self._validate_source_value(value)
            if value.uuid in seen:
                return
            seen.add(value.uuid)
            if value.uuid not in owned_uuids:
                return
            self._uuid_remap.setdefault(value.uuid, str(uuid_module.uuid4()))
            self._logical_id_remap.setdefault(
                value.logical_id,
                str(uuid_module.uuid4()),
            )
            for child in self._value_children(value):
                reserve(child)

        def reserve_operations(operations: list[Operation]) -> None:
            """Reserve values in one lexical operation tree.

            Args:
                operations (list[Operation]): Operations to scan.
            """
            for operation in operations:
                for value in operation.all_input_values():
                    reserve(value)
                for value in operation.results:
                    reserve(value)
                if isinstance(operation, HasNestedOps):
                    for nested in operation.nested_op_lists():
                        reserve_operations(nested)

        for value in block.input_values:
            reserve(value)
        for value in block.parameters.values():
            reserve(value)
        for slot in block.static_bindings:
            for field in slot.fields:
                reserve(field.value)
        reserve_operations(block.operations)
        for value in block.output_values:
            reserve(value)

    def _validate_source_value(self, value: ValueBase) -> None:
        """Reject one UUID attached to conflicting source structures.

        Args:
            value (ValueBase): Source value occurrence about to be cloned.

        Raises:
            ValueError: If an earlier occurrence with the same UUID has a
                different serializable structure.
        """
        previous = self._source_values_by_uuid.get(value.uuid)
        if previous is None:
            self._source_values_by_uuid[value.uuid] = value
            return
        from qamomile.circuit.ir.serialize.encode import _value_structures_match

        if not _value_structures_match(previous, value):
            raise ValueError(
                f"Value UUID {value.uuid!r} refers to conflicting structures"
            )

    @staticmethod
    def _value_children(value: ValueBase) -> tuple[ValueBase, ...]:
        """Return dataclass-linked child values for one graph node.

        Args:
            value (ValueBase): Value graph node.

        Returns:
            tuple[ValueBase, ...]: Structural child values.
        """
        if isinstance(value, TupleValue):
            return cast(tuple[ValueBase, ...], value.elements)
        if isinstance(value, DictValue):
            return tuple(
                child
                for key, entry_value in value.entries
                for child in (key, entry_value)
            )
        if not isinstance(value, Value):
            return ()
        children: list[ValueBase] = []
        if value.parent_array is not None:
            children.append(value.parent_array)
        children.extend(value.element_indices)
        if isinstance(value, ArrayValue):
            children.extend(value.shape)
            if value.slice_of is not None:
                children.append(value.slice_of)
            if value.slice_start is not None:
                children.append(value.slice_start)
            if value.slice_step is not None:
                children.append(value.slice_step)
        return tuple(children)

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

        if self._clone_owned_blocks and isinstance(new_op, InverseBlockOperation):
            if (
                new_op.source_block is not None
                and id(new_op.source_block) in self._active_blocks
            ):
                raise ValueError("Cannot clone a recursive inverse source graph.")
            new_op = dataclasses.replace(
                new_op,
                implementation_block=(
                    self.clone_block(new_op.implementation_block)
                    if new_op.implementation_block is not None
                    else None
                ),
            )
        if (
            self._clone_owned_blocks
            and isinstance(new_op, ControlledUOperation)
            and new_op.block is not None
        ):
            new_op = dataclasses.replace(
                new_op,
                block=self.clone_block(new_op.block),
            )
        if self._clone_owned_blocks and isinstance(new_op, SelectOperation):
            new_op = dataclasses.replace(
                new_op,
                case_blocks=[self.clone_block(case) for case in new_op.case_blocks],
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
        self._validate_source_value(value)

        # Check cache first - uuid is unique per value instance
        if old_uuid in self._value_cache:
            return self._value_cache[old_uuid]
        if (
            self._owned_uuid_stack
            and old_uuid not in self._owned_uuid_stack[-1]
            and old_uuid not in self._uuid_remap
        ):
            return value

        # Generate new UUID for this specific value
        new_uuid = self._uuid_remap.setdefault(old_uuid, str(uuid_module.uuid4()))

        # Generate new logical_id (same for values with same physical identity)
        old_logical_id = value.logical_id
        new_logical_id = self._logical_id_remap.setdefault(
            old_logical_id,
            str(uuid_module.uuid4()),
        )
        new_metadata = self._clone_metadata(value.metadata)

        # Handle different value types
        if isinstance(value, TupleValue):
            # Clone tuple elements
            new_elements = tuple(
                cast(ValueLike, self.clone_value(e)) for e in value.elements
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
