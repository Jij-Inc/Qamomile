"""Define lightweight named records for qkernel trace-time state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, dataclass_transform

_T = TypeVar("_T")


@dataclass_transform(eq_default=False)
def struct(cls: type[_T]) -> type[_T]:
    """Decorate a class as a mutable slotted trace-time record.

    Structs group related frontend handles without introducing a new IR value
    or changing a qkernel's backend ABI. They are ordinary Python objects that
    exist only while the frontend traces a kernel body. Equality remains based
    on object identity because comparing symbolic handle fields is not a valid
    trace-time operation.

    Args:
        cls (type[_T]): Annotated class whose fields define the record.

    Returns:
        type[_T]: Dataclass-compatible class with generated initialization,
            representation, and slots.

    Raises:
        TypeError: If ``cls`` cannot be converted to a slotted dataclass, such
            as when it already defines ``__slots__``.

    Example:
        >>> import qamomile.circuit as qmc
        >>> @qmc.struct
        ... class Registers:
        ...     control: qmc.Qubit
        ...     target: qmc.Qubit
    """
    return dataclass(eq=False, slots=True)(cls)


__all__ = ["struct"]
