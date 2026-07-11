"""Unified block representation for all pipeline stages."""

from __future__ import annotations

import dataclasses
import uuid as uuid_module
from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from qamomile.circuit.ir.parameter import ParamSlot
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
    ValueMetadata,
    remap_value_metadata_references,
)

if TYPE_CHECKING:
    from qamomile.circuit.ir.operation import Operation
    from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation


class BlockKind(Enum):
    """Classification of block structure for pipeline stages."""

    TRACED = auto()  # Direct output of frontend tracing / build()
    HIERARCHICAL = auto()  # May contain CallBlockOperations
    AFFINE = auto()  # No CallBlockOperations, For/If preserved
    ANALYZED = auto()  # Validated and dependency-analyzed


class _CallOutputFreshener:
    """Create caller-local placeholder values for ``Block.call`` results."""

    def __init__(self, formals: list[ValueLike], actuals: list[ValueLike]) -> None:
        """Initialize formal-to-actual input bindings.

        Args:
            formals (list[ValueLike]): Callee-side input values.
            actuals (list[ValueLike]): Caller-side values passed to the call.
        """
        self._input_by_uuid: dict[str, ValueLike] = {}
        self._input_by_logical_id: dict[str, ValueLike] = {}
        self._uuid_remap: dict[str, str] = {}
        self._logical_id_remap: dict[str, str] = {}
        self._fresh_cache: dict[str, ValueLike] = {}
        self._pass_through_cache: dict[str, ValueLike] = {}

        for formal, actual in zip(formals, actuals):
            self._register_input_binding(formal, actual)

    def freshen(self, value: ValueLike) -> ValueLike:
        """Return a caller-local output placeholder for ``value``.

        Args:
            value (ValueLike): Callee-side output value.

        Returns:
            ValueLike: A caller-side placeholder with unique UUIDs. Values
            derived from callee inputs are rebound to the corresponding caller
            inputs while preserving their logical identity.
        """
        actual = self._input_by_uuid.get(value.uuid)
        if actual is None:
            actual = self._input_by_logical_id.get(value.logical_id)
        if actual is not None:
            return self._next_pass_through(value, actual)

        cached = self._fresh_cache.get(value.uuid)
        if cached is not None:
            return cached

        new_uuid = str(uuid_module.uuid4())
        new_logical_id = self._logical_id_for_fresh_output(value)
        self._uuid_remap[value.uuid] = new_uuid

        if isinstance(value, TupleValue):
            new_elements = tuple(
                cast(ValueLike, self.freshen(element)) for element in value.elements
            )
            new_metadata = self._remap_metadata(value)
            cloned = dataclasses.replace(
                value,
                elements=new_elements,
                metadata=new_metadata,
                uuid=new_uuid,
                logical_id=new_logical_id,
            )
        elif isinstance(value, DictValue):
            new_entries: list[tuple[TupleValue | Value, Value]] = []
            for key, entry_value in value.entries:
                new_key = self.freshen(key)
                new_value = self.freshen(entry_value)
                new_entries.append(
                    (cast(TupleValue | Value, new_key), cast(Value, new_value))
                )
            new_metadata = self._remap_metadata(value)
            cloned = dataclasses.replace(
                value,
                entries=tuple(new_entries),
                metadata=new_metadata,
                uuid=new_uuid,
                logical_id=new_logical_id,
            )
        elif isinstance(value, ArrayValue):
            if value.slice_of is not None:
                cloned = dataclasses.replace(
                    value,
                    uuid=new_uuid,
                    logical_id=new_logical_id,
                )
            else:
                new_parent = (
                    cast(ArrayValue, self.freshen(value.parent_array))
                    if value.parent_array is not None
                    else None
                )
                new_indices = tuple(
                    cast(Value, self.freshen(index)) for index in value.element_indices
                )
                new_shape = tuple(cast(Value, self.freshen(dim)) for dim in value.shape)
                new_metadata = self._remap_metadata(value)
                cloned = dataclasses.replace(
                    value,
                    parent_array=new_parent,
                    element_indices=new_indices,
                    shape=new_shape,
                    metadata=new_metadata,
                    uuid=new_uuid,
                    logical_id=new_logical_id,
                )
        else:
            new_parent = (
                cast(ArrayValue, self.freshen(value.parent_array))
                if value.parent_array is not None
                else None
            )
            new_indices = tuple(
                cast(Value, self.freshen(index)) for index in value.element_indices
            )
            new_metadata = self._remap_metadata(value)
            cloned = dataclasses.replace(
                value,
                parent_array=new_parent,
                element_indices=new_indices,
                metadata=new_metadata,
                uuid=new_uuid,
                logical_id=new_logical_id,
            )

        self._fresh_cache[value.uuid] = cloned
        return cloned

    def _register_input_binding(
        self,
        formal: ValueLike,
        actual: ValueLike,
        seen: set[str] | None = None,
    ) -> None:
        """Record formal-to-actual bindings for structural call inputs.

        Args:
            formal (ValueLike): Callee-side input value to match.
            actual (ValueLike): Caller-side value bound to ``formal``.
            seen (set[str] | None): Formal UUIDs already visited while
                descending through structural values. Defaults to None.
        """
        if seen is None:
            seen = set()
        if formal.uuid in seen:
            return
        seen.add(formal.uuid)

        self._input_by_uuid[formal.uuid] = actual
        self._input_by_logical_id[formal.logical_id] = actual

        if isinstance(formal, TupleValue) and isinstance(actual, TupleValue):
            for formal_element, actual_element in zip(
                formal.elements, actual.elements, strict=True
            ):
                self._register_input_binding(formal_element, actual_element, seen)
            return

        if isinstance(formal, DictValue) and isinstance(actual, DictValue):
            for (formal_key, formal_value), (actual_key, actual_value) in zip(
                formal.entries,
                actual.entries,
            ):
                self._register_input_binding(formal_key, actual_key, seen)
                self._register_input_binding(formal_value, actual_value, seen)
            return

        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            for formal_dim, actual_dim in zip(formal.shape, actual.shape):
                self._register_input_binding(formal_dim, actual_dim, seen)
            if formal.slice_of is not None and actual.slice_of is not None:
                self._register_input_binding(formal.slice_of, actual.slice_of, seen)
            if formal.slice_start is not None and actual.slice_start is not None:
                self._register_input_binding(
                    formal.slice_start,
                    actual.slice_start,
                    seen,
                )
            if formal.slice_step is not None and actual.slice_step is not None:
                self._register_input_binding(
                    formal.slice_step,
                    actual.slice_step,
                    seen,
                )

        if isinstance(formal, Value) and isinstance(actual, Value):
            if formal.parent_array is not None and actual.parent_array is not None:
                self._register_input_binding(
                    formal.parent_array,
                    actual.parent_array,
                    seen,
                )
            for formal_index, actual_index in zip(
                formal.element_indices,
                actual.element_indices,
            ):
                self._register_input_binding(formal_index, actual_index, seen)

    def _next_pass_through(self, source: ValueLike, actual: ValueLike) -> ValueLike:
        """Return the next caller-side version for an input-derived output.

        Args:
            source (ValueLike): Callee-side value that flows from an input.
            actual (ValueLike): Caller-side value corresponding to ``source``.

        Returns:
            ValueLike: Caller-side next version preserving logical identity.
        """
        cached = self._pass_through_cache.get(source.uuid)
        if cached is not None:
            return cached

        if isinstance(source, TupleValue) and isinstance(actual, TupleValue):
            elements = tuple(
                cast(ValueLike, self._next_pass_through(source_elem, actual_elem))
                for source_elem, actual_elem in zip(
                    source.elements, actual.elements, strict=True
                )
            )
            result = dataclasses.replace(
                actual,
                elements=elements,
                uuid=str(uuid_module.uuid4()),
            )
        elif isinstance(source, DictValue) and isinstance(actual, DictValue):
            entries: list[tuple[TupleValue | Value, Value]] = []
            for (source_key, source_value), (actual_key, actual_value) in zip(
                source.entries,
                actual.entries,
            ):
                new_key = self._next_pass_through(source_key, actual_key)
                new_value = self._next_pass_through(source_value, actual_value)
                entries.append(
                    (cast(TupleValue | Value, new_key), cast(Value, new_value))
                )
            result = dataclasses.replace(
                actual,
                entries=tuple(entries),
                uuid=str(uuid_module.uuid4()),
            )
        else:
            result = cast(ValueLike, actual.next_version())

        self._uuid_remap[source.uuid] = result.uuid
        self._logical_id_remap[source.logical_id] = result.logical_id
        self._pass_through_cache[source.uuid] = result
        return result

    def _logical_id_for_fresh_output(self, value: ValueLike) -> str:
        """Return the logical ID to use for a non-pass-through output.

        Args:
            value (ValueLike): Callee-side output value.

        Returns:
            str: A fresh logical ID for quantum values that do not originate
            from a caller input; otherwise the original logical ID.
        """
        if self._needs_fresh_logical_id(value):
            return self._logical_id_remap.setdefault(
                value.logical_id,
                str(uuid_module.uuid4()),
            )
        self._logical_id_remap[value.logical_id] = value.logical_id
        return value.logical_id

    def _needs_fresh_logical_id(self, value: ValueLike) -> bool:
        """Return whether ``value`` needs a call-site-local logical ID.

        Args:
            value (ValueLike): Callee-side output value.

        Returns:
            bool: True for quantum values that are not derived from call
            inputs, false for input-lineage values and classical containers.
        """
        return (
            isinstance(value, Value)
            and not isinstance(value, ArrayValue)
            and value.type.is_quantum()
            and not self._has_input_lineage(value)
        )

    def _has_input_lineage(
        self,
        value: ValueLike,
        seen: set[str] | None = None,
    ) -> bool:
        """Return whether ``value`` is derived from a call input.

        Args:
            value (ValueLike): Value to inspect.
            seen (set[str] | None): UUIDs already visited while following
                parent and slice links. Defaults to None.

        Returns:
            bool: True when ``value`` itself or one of its parent/slice
            carriers is bound to a caller input.
        """
        if seen is None:
            seen = set()
        if value.uuid in seen:
            return False
        seen.add(value.uuid)

        if (
            value.uuid in self._input_by_uuid
            or value.logical_id in self._input_by_logical_id
        ):
            return True

        if isinstance(value, ArrayValue) and value.slice_of is not None:
            return self._has_input_lineage(value.slice_of, seen)

        if isinstance(value, Value) and value.parent_array is not None:
            return self._has_input_lineage(value.parent_array, seen)

        return False

    def _remap_metadata(self, value: ValueBase) -> ValueMetadata:
        """Remap metadata references through fresh and call-input values.

        Args:
            value (ValueBase): Value whose metadata should be rewritten.

        Returns:
            ValueMetadata: Metadata with UUID and logical-id references remapped.
        """
        return remap_value_metadata_references(
            value.metadata,
            self._remap_uuid,
            self._remap_logical_id,
        )

    def _remap_uuid(self, old_uuid: str) -> str:
        """Return the caller-side UUID for a callee-side UUID reference.

        Args:
            old_uuid (str): UUID reference from the callee scope.

        Returns:
            str: Freshened or caller-bound UUID when known, otherwise
            ``old_uuid``.
        """
        if old_uuid in self._uuid_remap:
            return self._uuid_remap[old_uuid]
        actual = self._input_by_uuid.get(old_uuid)
        return actual.uuid if actual is not None else old_uuid

    def _remap_logical_id(self, old_logical_id: str) -> str:
        """Return the caller-side logical ID for a callee-side reference.

        Args:
            old_logical_id (str): Logical-id reference from the callee scope.

        Returns:
            str: Caller-bound logical ID when known, otherwise
            ``old_logical_id``.
        """
        if old_logical_id in self._logical_id_remap:
            return self._logical_id_remap[old_logical_id]
        actual = self._input_by_logical_id.get(old_logical_id)
        return actual.logical_id if actual is not None else old_logical_id


@dataclasses.dataclass
class Block:
    """Unified block representation for all pipeline stages.

    Replaces the older traced and callable IR wrappers with a single structure.
    The `kind` field indicates which pipeline stage this block is at.
    """

    name: str = ""
    label_args: list[str] = dataclasses.field(default_factory=list)
    input_values: list[ValueLike] = dataclasses.field(default_factory=list)
    output_values: list[ValueLike] = dataclasses.field(default_factory=list)
    output_names: list[str] = dataclasses.field(default_factory=list)
    operations: list["Operation"] = dataclasses.field(default_factory=list)

    # Pipeline stage indicator
    kind: BlockKind = BlockKind.HIERARCHICAL

    # Parameters (unbound values for circuit parameters)
    parameters: dict[str, Value] = dataclasses.field(default_factory=dict)

    # Per-classical-argument metadata describing the kernel's parameter
    # contract (name, type, runtime-or-bound kind, default, bound_value,
    # differentiability hint). Populated by the frontend (``func_to_block``
    # / ``QKernel.build``) and preserved by every pass. Empty for
    # synthetic blocks that have no Python-level classical interface
    # (e.g., nested composite-gate implementation blocks).
    param_slots: tuple[ParamSlot, ...] = dataclasses.field(default_factory=tuple)

    def __post_init__(self):
        """Validate label_args / input_values agreement and param_slots disjointness.

        Raises:
            ValueError: If ``label_args`` is non-empty and its length
                does not match ``input_values``, or if any
                ``ParamSlot.name`` appears more than once across
                ``param_slots``.
        """
        if self.label_args and len(self.label_args) != len(self.input_values):
            raise ValueError(
                f"label_args length ({len(self.label_args)}) must match "
                f"input_values length ({len(self.input_values)})"
            )
        if self.param_slots:
            seen: set[str] = set()
            for slot in self.param_slots:
                if slot.name in seen:
                    raise ValueError(
                        f"Duplicate ParamSlot name {slot.name!r} in Block.param_slots; "
                        f"every classical kernel argument may appear at most once."
                    )
                seen.add(slot.name)

    def unbound_parameters(self) -> list[str]:
        """Return list of unbound parameter names."""
        return list(self.parameters.keys())

    def is_affine(self) -> bool:
        """Check if block contains no CallBlockOperations."""
        return self.kind in (BlockKind.AFFINE, BlockKind.ANALYZED)

    def call(self, **kwargs: ValueLike) -> "CallBlockOperation":
        """Create a CallBlockOperation against this block."""
        from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation

        inputs = [kwargs[label] for label in self.label_args]
        freshener = _CallOutputFreshener(self.input_values, inputs)

        results = []
        for dummy_return in self.output_values:
            results.append(freshener.freshen(dummy_return))

        return CallBlockOperation(
            block=self,
            operands=cast(list[Value], inputs),
            results=cast(list[Value], results),
        )
