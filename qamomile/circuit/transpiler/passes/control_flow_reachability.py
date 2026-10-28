"""Resolve statically reachable structured-control regions."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    Region,
)
from qamomile.circuit.ir.value import ArrayValue, DictValue, Value, ValueBase

MAX_STATIC_REPLAY_TRIPS = 10_000


def same_exact_typed_constant(left: Value, right: Value) -> bool:
    """Return whether two scalar Values carry the same exact typed constant.

    Equality requires matching IR and Python types. Floating-point comparison
    preserves the sign of zero and the payload bits of NaNs.

    Args:
        left (Value): First scalar value to compare.
        right (Value): Second scalar value to compare.

    Returns:
        bool: ``True`` only for constants of the same IR type and Python type
            with equal value representations.
    """
    if isinstance(left, ArrayValue) or isinstance(right, ArrayValue):
        return False
    if left.type != right.type or not left.is_constant() or not right.is_constant():
        return False
    left_constant = left.get_const()
    right_constant = right.get_const()
    if type(left_constant) is not type(right_constant):
        return False
    if isinstance(left_constant, float):
        return struct.pack("!d", left_constant) == struct.pack("!d", right_constant)
    return bool(left_constant == right_constant)


def constant_integer(value: ValueBase | None) -> int | None:
    """Return a non-boolean integer constant carried by an IR value.

    Args:
        value (ValueBase | None): Candidate scalar value.

    Returns:
        int | None: Normalized Python integer, or ``None`` when ``value`` is
            absent, symbolic, boolean, or non-integral.
    """
    if not isinstance(value, Value) or not value.is_constant():
        return None
    constant = value.get_const()
    if isinstance(constant, (bool, np.bool_)) or not isinstance(
        constant, (int, np.integer)
    ):
        return None
    return int(constant)


def static_for_range(operation: ForOperation) -> range | None:
    """Resolve the exact iteration range of a statically bounded loop.

    Args:
        operation (ForOperation): Counted loop whose three bounds should be
            inspected.

    Returns:
        range | None: Exact Python range when all bounds are integral
            constants and the step is nonzero, otherwise ``None``.
    """
    if len(operation.operands) < 3:
        return None
    start, stop, step = (
        constant_integer(operation.operands[index]) for index in range(3)
    )
    if start is None or stop is None or step in (None, 0):
        return None
    return range(start, stop, step)


def static_for_items_entries(
    operation: ForItemsOperation,
) -> tuple[tuple[Any, Any], ...] | None:
    """Return compile-time entries iterated by a for-items operation.

    Args:
        operation (ForItemsOperation): Items loop whose iterable should be
            inspected.

    Returns:
        tuple[tuple[Any, Any], ...] | None: Bound key/value entries in
            iteration order, including an empty tuple for a known-empty
            mapping, or ``None`` when the iterable remains symbolic.
    """
    if not operation.operands:
        return None
    iterable = operation.operands[0]
    if not isinstance(iterable, DictValue):
        return None
    if iterable.metadata.dict_runtime is None:
        return None
    return iterable.get_bound_data_items()


def static_loop_trip_count(
    operation: ForOperation | ForItemsOperation,
) -> int | None:
    """Resolve the exact trip count of a compile-time loop.

    Args:
        operation (ForOperation | ForItemsOperation): Counted or items loop
            whose cardinality should be inspected.

    Returns:
        int | None: Exact non-negative trip count, or ``None`` when the loop
            cardinality is unresolved.
    """
    if isinstance(operation, ForOperation):
        iteration_range = static_for_range(operation)
        if iteration_range is None:
            return None
        try:
            return len(iteration_range)
        except OverflowError:
            return None
    entries = static_for_items_entries(operation)
    return None if entries is None else len(entries)


def reachable_nested_regions(operation: HasNestedOps) -> tuple[Region, ...]:
    """Return nested regions that may execute for one control-flow operation.

    Constant conditionals expose only their selected branch. Statically empty
    counted and items loops expose no body region. All unresolved control flow
    remains conservative and exposes every region.

    Args:
        operation (HasNestedOps): Structured-control operation to inspect.

    Returns:
        tuple[Region, ...]: Regions that are reachable under compile-time-known
            control decisions.
    """
    regions = operation.nested_regions()
    if isinstance(operation, IfOperation) and operation.condition.is_constant():
        return (regions[0] if bool(operation.condition.get_const()) else regions[1],)
    if isinstance(operation, (ForOperation, ForItemsOperation)):
        trip_count = static_loop_trip_count(operation)
        if trip_count == 0:
            return ()
    return regions


__all__ = [
    "MAX_STATIC_REPLAY_TRIPS",
    "constant_integer",
    "reachable_nested_regions",
    "same_exact_typed_constant",
    "static_for_items_entries",
    "static_for_range",
    "static_loop_trip_count",
]
