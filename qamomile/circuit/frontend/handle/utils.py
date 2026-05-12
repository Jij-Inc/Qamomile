"""Utility helpers for handle types."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .array import Vector
    from .handle import Handle

__all__ = ["get_size"]

_H = TypeVar("_H", bound="Handle")


def get_size(arr: Vector[_H]) -> int:
    """Return the size of a Vector handle as a Python integer.

    Resolves the leading axis of *arr.shape* through three forms a
    ``Vector`` shape entry can take:

    1. A plain Python ``int`` (built-in bound shape; this is what you
       get from ``qmc.qubit_array(N, ...)`` for literal ``N``).
    2. A ``UInt`` handle whose underlying ``Value`` carries a
       compile-time constant (set by ``uint(literal)``,
       ``_create_bound_input``, or partial evaluation).
    3. A ``UInt`` handle whose ``init_value`` is set even though its
       ``Value`` is not yet promoted to a constant.

    The ``init_value`` fallback (form 3) is what lets sub-kernels that
    take ``Vector[Qubit]`` as a parameter trace standalone:
    ``create_dummy_input`` constructs the dummy ``Vector[Qubit]`` with
    a symbolic ``UInt(value=symbolic_value)`` whose dataclass-default
    ``init_value`` is ``0``, and stdlib helpers like ``qft`` /
    ``iqft`` rely on the resulting ``get_size(...) == 0`` to reduce
    their loop body to a no-op for the trace.  Removing the fallback
    breaks every affine-type / re-binding test that exercises a
    ``Vector[Qubit] -> Vector[Qubit]`` sub-kernel without an outer
    concrete shape, so we keep it.

    The corollary is that this helper can return ``0`` for an
    unresolved symbolic ``Vector`` instead of raising — callers that
    must reject silent-zero (e.g.,
    ``amplitude_encoding_from_angles`` validating an angle-vector
    length) need to add their own explicit shape checks.

    Args:
        arr (Vector[Handle]): Vector handle whose first axis size is
            requested.

    Returns:
        int: The first-axis size as a plain Python ``int`` (possibly
            ``0`` for an unresolved symbolic handle — see notes above).

    Raises:
        ValueError: If the shape entry exposes neither
            ``value.is_constant() = True`` nor an ``init_value`` —
            i.e., a genuinely opaque symbolic dimension that did not
            even pass through ``UInt(...)`` wrapping.
    """
    size = arr.shape[0]
    if isinstance(size, int):
        return size
    if hasattr(size, "value") and size.value.is_constant():
        val = size.value.get_const()
        if val is not None:
            return int(val)
    if hasattr(size, "init_value"):
        return int(size.init_value)
    raise ValueError("Array must have fixed size")
