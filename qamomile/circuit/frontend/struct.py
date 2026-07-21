"""Define lightweight named records for qkernel trace-time state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, dataclass_transform

_T = TypeVar("_T")


@dataclass_transform(eq_default=False)
def struct(cls: type[_T]) -> type[_T]:
    """Decorate a class as an immutable trace-time record.

    Structs group related frontend handles without introducing a new IR value
    or changing a qkernel's backend ABI. They are ordinary Python objects that
    exist only while the frontend traces a kernel body. The record is shallowly
    frozen so a field cannot be rebound in place; quantum operations must build
    a successor record from the handles they return. Affine ownership remains
    attached to the contained quantum handles, just as it does for handles in a
    Python tuple. Copying a struct therefore does not clone or transfer a
    qubit. Equality remains based on object identity because comparing symbolic
    handle fields is not a valid trace-time operation.

    Args:
        cls (type[_T]): Annotated class whose fields define the record.

    Returns:
        type[_T]: Frozen dataclass-compatible class with generated
            initialization and representation.

    Raises:
        TypeError: If ``cls`` cannot be converted to a frozen dataclass, such
            as when it defines incompatible dataclass options.

    Example:
        >>> import qamomile.circuit as qmc
        >>> @qmc.struct
        ... class Registers:
        ...     control: qmc.Qubit
        ...     target: qmc.Qubit
    """
    return dataclass(eq=False, frozen=True)(cls)


__all__ = ["struct"]
