"""Describe compile-time object dependencies of hierarchical qkernels."""

from __future__ import annotations

import dataclasses

from qamomile.circuit.ir.value import Value


@dataclasses.dataclass(frozen=True)
class StaticBindingField:
    """Reference one scalar field projected from a static binding.

    Args:
        name (str): Registered field name on the bound object.
        value (Value): Symbolic scalar used by the hierarchical IR until the
            binding is materialized.
    """

    name: str
    value: Value


@dataclasses.dataclass(frozen=True)
class StaticBindingSlot:
    """Declare one typed compile-time object required by a qkernel.

    The object itself is not an SSA value and never reaches a backend. Only
    registered scalar projections and deferred callable-member references may
    appear in the hierarchical body. A build must resolve the slot before the
    block advances to a compiler stage.

    Args:
        name (str): QKernel argument name used by ``bindings``.
        type_key (str): Stable key of the registered static-binding adapter.
        fields (tuple[StaticBindingField, ...]): Scalar projections referenced
            while tracing the unbound qkernel.
    """

    name: str
    type_key: str
    fields: tuple[StaticBindingField, ...] = ()


__all__ = ["StaticBindingField", "StaticBindingSlot"]
