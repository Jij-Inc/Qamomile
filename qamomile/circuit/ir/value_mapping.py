"""Provide shared IR value substitution utilities."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, cast

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
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

        if isinstance(result, IfOperation):
            from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp

            new_phi_ops = cast(
                list[PhiOp],
                [self.substitute_operation(phi_op) for phi_op in result.phi_ops],
            )
            result = dataclasses.replace(result, phi_ops=new_phi_ops)

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
                shape, or slice metadata references.

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
        return False

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
            )

        return dataclasses.replace(
            value,
            parent_array=new_parent_array,
            element_indices=new_element_indices,
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
        new_elements = []
        changed = False
        for elem in value.elements:
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
            if isinstance(key, (TupleValue, Value)) and key.uuid in self._value_map:
                sub_key = self._chase_transitive(key)
                if isinstance(sub_key, (TupleValue, Value)):
                    new_key = sub_key
                    changed = True
            if entry_value.uuid in self._value_map:
                sub_value = self._chase_transitive(entry_value)
                if isinstance(sub_value, Value):
                    new_value = sub_value
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
                return dataclasses.replace(value, **replace_kwargs)
            return dataclasses.replace(
                value,
                parent_array=new_parent_array,
                element_indices=new_element_indices,
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
