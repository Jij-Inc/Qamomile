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

    Resolves the leading axis of *arr.shape* through two forms a
    ``Vector`` shape entry can take:

    1. A plain Python ``int`` (built-in bound shape; this is what you
       get from ``qmc.qubit_array(N, ...)`` for literal ``N``).
    2. A ``UInt`` handle whose underlying ``Value`` carries a
       compile-time constant (set by ``uint(literal)``,
       ``_create_bound_input``, or partial evaluation).

    A ``UInt`` handle whose underlying ``Value`` is *not* a constant is
    treated as an unresolved symbolic dimension and raises
    ``ValueError`` even when the handle has the dataclass-default
    ``init_value=0``.  Falling back to ``init_value`` for that case
    would silently turn a runtime-symbolic ``Vector[Float]`` parameter
    into a "size 0" array, hiding programming errors.  Callers that
    need to handle symbolic shapes (e.g., to gracefully no-op when the
    size is unknown) must catch the ``ValueError`` themselves.

    Args:
        arr (Vector[Handle]): Vector handle whose first axis size is
            requested.

    Returns:
        int: The first-axis size as a plain Python ``int``.

    Raises:
        TypeError: If *arr* is not a sized (``shape``-bearing) handle —
            e.g., a scalar ``Qubit`` was passed where a ``Vector`` is
            required. This is a clearer signal than the bare
            ``AttributeError`` that ``arr.shape`` would otherwise raise,
            and it guards the stdlib / composite callers that resolve a
            register size through this helper.
        ValueError: If the shape cannot be resolved to a concrete
            integer — e.g., the Vector is a runtime-parametric handle
            without compile-time bindings, or carries a ``UInt``
            dimension whose underlying ``Value`` has not been promoted
            to a constant.
    """
    if not hasattr(arr, "shape"):
        raise TypeError(
            f"get_size expects a sized Vector handle, got "
            f"{type(arr).__name__}, which has no 'shape'. A scalar handle "
            f"(e.g. a single Qubit) has no size; pass a Vector / "
            f"qubit_array instead."
        )
    size = arr.shape[0]
    if isinstance(size, int):
        return size
    if hasattr(size, "value") and size.value.is_constant():
        val = size.value.get_const()
        if val is not None:
            return int(val)
    raise ValueError("Array must have fixed size")
