"""Shared zero-cost layout isomorphisms for coherent runtime values."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from .types import ProductType, QuantumType, TypeSpec, logical_width, strip_stage

_Leaf = TypeVar("_Leaf")
_Lane = TypeVar("_Lane")


def reshape_coherent(
    value: object,
    target_type: TypeSpec,
    *,
    is_leaf: Callable[[object], bool],
    unpack_leaf: Callable[[object], Sequence[_Lane]],
    pack_leaf: Callable[[tuple[_Lane, ...]], _Leaf],
    target_leaf_widths: Sequence[int] | None = None,
) -> object:
    """Reassociate one coherent runtime value without changing its lanes.

    Args:
        value (object): Runtime quantum leaf or nested product of leaves.
        target_type (TypeSpec): Desired coherent semantic product shape.
        is_leaf (Callable[[object], bool]): Predicate recognizing one backend's
            quantum leaf value.
        unpack_leaf (Callable[[object], Sequence[_Lane]]): Conversion from a
            leaf to its ordered logical lanes.
        pack_leaf (Callable[[tuple[_Lane, ...]], _Leaf]): Conversion from
            ordered lanes back to one leaf.
        target_leaf_widths (Sequence[int] | None): Concrete widths for target
            quantum leaves whose semantic dimensions remain symbolic.

    Returns:
        object: The same logical lanes associated according to ``target_type``.

    Raises:
        ValueError: If either runtime or target layout is not coherent.
    """

    lanes: list[_Lane] = []
    concrete_widths = None if target_leaf_widths is None else tuple(target_leaf_widths)
    next_width = 0

    def flatten(item: object) -> None:
        """Append lanes from one nested runtime product.

        Args:
            item (object): Backend runtime leaf or nested product.

        Raises:
            ValueError: If ``item`` is neither a leaf nor a tuple product.
        """

        if is_leaf(item):
            lanes.extend(unpack_leaf(item))
            return
        if not isinstance(item, tuple):
            raise ValueError("coherent runtime value is not a product of leaves")
        for component in item:
            flatten(component)

    def build(type_spec: TypeSpec, offset: int) -> tuple[object, int]:
        """Build one target component and return the next lane offset.

        Args:
            type_spec (TypeSpec): Coherent target component type.
            offset (int): First unconsumed logical lane.

        Returns:
            tuple[object, int]: Reassociated value and next lane offset.

        Raises:
            ValueError: If the target component is not a fixed coherent shape.
        """

        nonlocal next_width

        base = strip_stage(type_spec)
        if isinstance(base, QuantumType):
            width = logical_width(base.inner)
            if concrete_widths is not None:
                if next_width >= len(concrete_widths):
                    raise ValueError("coherent target has too few leaf widths")
                concrete_width = concrete_widths[next_width]
                next_width += 1
                if width is None:
                    width = concrete_width
                elif width != concrete_width:
                    raise ValueError("coherent target leaf width does not match")
            if width is None or offset + width > len(lanes):
                raise ValueError("coherent target has an invalid bit-state leaf")
            return pack_leaf(tuple(lanes[offset : offset + width])), offset + width
        if isinstance(base, ProductType):
            parts: list[object] = []
            for element in base.elements:
                part, offset = build(element, offset)
                parts.append(part)
            return tuple(parts), offset
        raise ValueError("coherent target is not a quantum product")

    flatten(value)
    target_base = strip_stage(target_type)
    if (
        isinstance(target_base, QuantumType)
        and logical_width(target_base.inner) is None
    ):
        if concrete_widths is not None and concrete_widths != (len(lanes),):
            raise ValueError("coherent target leaf width does not match")
        return pack_leaf(tuple(lanes))

    result, consumed = build(target_type, 0)
    if consumed != len(lanes):
        raise ValueError("coherent reshape changed the logical width")
    if concrete_widths is not None and next_width != len(concrete_widths):
        raise ValueError("coherent target has too many leaf widths")
    return result
