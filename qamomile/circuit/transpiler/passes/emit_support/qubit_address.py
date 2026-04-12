"""Typed key for qubit/clbit physical-index maps.

Replaces the ad-hoc ``f"{uuid}_{index}"`` string convention with an
explicit, hashable value object that makes the construction/parsing
contract type-safe.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class QubitAddress:
    """Typed key for qubit/clbit physical-index maps.

    For scalar qubits:  ``QubitAddress(uuid="abc123")``
    For array elements: ``QubitAddress(uuid="abc123", element_index=2)``

    This replaces the ``f"{uuid}_{i}"`` string key pattern throughout
    the emit pipeline, making the key format explicit and preventing
    format-string bugs.
    """

    uuid: str
    element_index: int | None = None

    @property
    def is_array_element(self) -> bool:
        """True if this address refers to an array element."""
        return self.element_index is not None

    def with_element(self, index: int) -> QubitAddress:
        """Create an array-element address from this array's base UUID."""
        return QubitAddress(uuid=self.uuid, element_index=index)

    def matches_array(self, array_uuid: str) -> bool:
        """True if this address belongs to the given array."""
        return self.uuid == array_uuid and self.element_index is not None

    def __str__(self) -> str:
        """Backward-compatible string representation.

        Produces the same format as the legacy ``f"{uuid}_{i}"`` pattern,
        enabling gradual migration and consistent logging/error output.
        """
        if self.element_index is not None:
            return f"{self.uuid}_{self.element_index}"
        return self.uuid

    def __repr__(self) -> str:
        if self.element_index is not None:
            return f"QubitAddress({self.uuid!r}, {self.element_index})"
        return f"QubitAddress({self.uuid!r})"

    @classmethod
    def from_composite_key(cls, key: str) -> QubitAddress:
        """Parse a legacy composite key string into a QubitAddress.

        The frontend stores qubit references as composite strings in
        the format ``"{array_uuid}_{element_index}"`` (e.g., cast
        operation qubit mappings, element UUIDs).  This helper converts
        such strings to proper ``QubitAddress`` instances.

        If the key does not match the composite format (i.e., the
        suffix after the last ``_`` is not a non-negative integer),
        it is treated as a plain scalar UUID.
        """
        last_underscore = key.rfind("_")
        if last_underscore > 0:
            suffix = key[last_underscore + 1:]
            if suffix.isdigit():
                return cls(key[:last_underscore], int(suffix))
        return cls(key)


# Type aliases for the physical-index maps
QubitMap = dict[QubitAddress, int]
ClbitMap = dict[QubitAddress, int]
