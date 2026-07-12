"""Loop analysis helpers for emission.

Identity policy: loop variables are identified by their IR ``Value``
UUID (``ForOperation.loop_var_value.uuid``), **not** by their display
name. Two nested or sibling loops with identical user-chosen names
(e.g. ``for i in range(N): for i in range(M):``) are distinct here
because each ``ForOperation`` carries its own ``loop_var_value`` with
a fresh UUID, and ``UUIDRemapper`` clones that field consistently with
body references via the ``all_input_values`` / ``replace_values``
protocol.

There is **no name fallback**: if ``loop_var_value`` is ``None`` (legacy
IR built before the field existed), analysis conservatively requests
unrolling and the emit entry point rejects the malformed IR. Comparing by
name was the soil for nested-loop name-collision bugs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    HasNestedOps,
)

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


def _is_loop_var(value: "Value", loop_var_uuid: str | None) -> bool:
    """Check whether ``value`` is the loop variable of the surrounding for.

    Pure UUID comparison. ``loop_var_uuid`` comes from
    ``ForOperation.loop_var_value`` and is preserved through inline
    cloning by ``UUIDRemapper`` (which uses the ``all_input_values`` /
    ``replace_values`` protocol). When the IR predates the
    ``loop_var_value`` field (``loop_var_uuid is None``), this returns
    ``False`` rather than falling back to name comparison.

    Args:
        value (Value): Candidate IR value.
        loop_var_uuid (str | None): Enclosing loop-variable UUID, or None for
            legacy IR.

    Returns:
        bool: True when ``value`` is the enclosing loop variable.
    """
    if loop_var_uuid is None:
        return False
    return value.uuid == loop_var_uuid


class LoopAnalyzer:
    """Analyze loop structures to determine an emission strategy."""

    def should_unroll(
        self,
        op: ForOperation,
        bindings: dict[str, object],
    ) -> bool:
        """Determine whether a for-loop requires emit-time unrolling.

        Args:
            op (ForOperation): Loop to analyze.
            bindings (dict[str, object]): Compile-time bindings available to
                nested-bound analysis.

        Returns:
            bool: True when identities, carries, indices, predicates,
                dictionary lookups, or nested bounds require concrete
                per-iteration evaluation.
        """
        if op.loop_var_value is None:
            # Never recover compiler identity from a display label. The emit
            # entry point rejects this legacy/malformed IR, while returning
            # True here prevents standalone analyzer callers from selecting a
            # native path that cannot bind the loop index correctly.
            return True
        if op.region_args:
            # A native backend loop keeps one static body per iteration and
            # cannot thread a classical value between iterations.
            return True
        loop_uuid = op.loop_var_value.uuid
        if self._has_dynamic_nested_loop(op.operations, bindings, loop_uuid):
            return True
        return self._has_loop_var_dependency(op.operations, loop_uuid)

    def _has_loop_var_dependency(
        self,
        operations: list[Operation],
        loop_var_uuid: str | None,
    ) -> bool:
        """Return whether any loop-body input structurally depends on the index.

        The generic ``Operation.all_input_values()`` protocol is the source of
        truth so subclass-specific fields such as ``SymbolicControlledU``
        powers, control counts, and control indices cannot be omitted. Each
        value is then walked recursively through array-element indices, slice
        bounds, slice parents, shapes, tuples, and dictionaries. This covers
        operations such as ``MeasureOperation`` without maintaining a fragile
        operation-type whitelist.

        Args:
            operations (list[Operation]): Loop-body operations to scan.
            loop_var_uuid (str | None): UUID of the enclosing loop variable,
                or None for legacy IR without ``loop_var_value``.

        Returns:
            bool: True if any operation input depends on the loop variable.
        """
        for op in operations:
            if any(
                self._value_depends_on_loop_var(value, loop_var_uuid)
                for value in op.all_input_values()
            ):
                return True
            if isinstance(op, HasNestedOps) and any(
                self._has_loop_var_dependency(op_list, loop_var_uuid)
                for op_list in op.nested_op_lists()
            ):
                return True
        return False

    def _value_depends_on_loop_var(
        self,
        value: object,
        loop_var_uuid: str | None,
        visiting: frozenset[int] = frozenset(),
    ) -> bool:
        """Return whether an IR value structurally reads the loop variable.

        Args:
            value (object): Candidate operation input or nested IR value.
            loop_var_uuid (str | None): UUID of the enclosing loop variable.
            visiting (frozenset[int]): Object identities already visited on
                the current recursive path. Defaults to an empty set.

        Returns:
            bool: True if ``value`` depends on the loop variable.
        """
        from qamomile.circuit.ir.value import (
            ArrayValue,
            DictValue,
            TupleValue,
            Value,
            ValueBase,
        )

        if not isinstance(value, ValueBase):
            return False
        object_id = id(value)
        if object_id in visiting:
            return False
        next_visiting = visiting | {object_id}

        if isinstance(value, Value) and _is_loop_var(value, loop_var_uuid):
            return True

        children: list[object] = []
        if isinstance(value, Value):
            children.extend(value.element_indices)
            if value.parent_array is not None:
                children.append(value.parent_array)
        if isinstance(value, ArrayValue):
            children.extend(value.shape)
            if value.slice_of is not None:
                children.append(value.slice_of)
            if value.slice_start is not None:
                children.append(value.slice_start)
            if value.slice_step is not None:
                children.append(value.slice_step)
        elif isinstance(value, TupleValue):
            children.extend(value.elements)
        elif isinstance(value, DictValue):
            for key, item in value.entries:
                children.extend((key, item))

        return any(
            self._value_depends_on_loop_var(child, loop_var_uuid, next_visiting)
            for child in children
        )

    def _has_dynamic_nested_loop(
        self,
        operations: list[Operation],
        bindings: dict[str, object],
        parent_loop_var_uuid: str | None,
    ) -> bool:
        """Return whether a nested loop needs the parent's concrete index.

        Args:
            operations (list[Operation]): Operations in the parent loop body.
            bindings (dict[str, object]): Compile-time bindings available for
                nested loop bounds.
            parent_loop_var_uuid (str | None): UUID of the parent loop variable,
                or None for legacy IR without ``loop_var_value``.

        Returns:
            bool: True if a nested bound directly uses the parent index or
                resolves to a non-numeric binding.
        """
        from qamomile.circuit.ir.value import Value

        for op in operations:
            if isinstance(op, ForOperation):
                for bound_val in op.operands[:3]:
                    if isinstance(bound_val, Value):
                        if _is_loop_var(bound_val, parent_loop_var_uuid):
                            return True
                        if bound_val.uuid in bindings:
                            bound = bindings[bound_val.uuid]
                            if not isinstance(bound, (int, float)):
                                return True
                        elif bound_val.is_parameter():
                            param_name = bound_val.parameter_name()
                            if param_name and param_name in bindings:
                                bound = bindings[param_name]
                                if not isinstance(bound, (int, float)):
                                    return True
            if isinstance(op, HasNestedOps) and any(
                self._has_dynamic_nested_loop(op_list, bindings, parent_loop_var_uuid)
                for op_list in op.nested_op_lists()
            ):
                return True
        return False
