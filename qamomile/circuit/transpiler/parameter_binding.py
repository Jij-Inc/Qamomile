"""Parameter binding structures for compiled quantum circuits."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class ParameterInfo:
    """Information about a single unbound parameter in the circuit.

    Attributes:
        name: Full parameter key, e.g., "gammas[0]"
        array_name: Base array name, e.g., "gammas"
        index: Array index if vector parameter, None if scalar
        backend_param: Backend-specific parameter object (e.g., qiskit.circuit.Parameter)
    """

    name: str
    array_name: str
    index: int | None
    backend_param: Any


@dataclasses.dataclass
class ParameterMetadata:
    """Metadata for all parameters in a compiled segment.

    Tracks parameter information for runtime binding.
    """

    parameters: list[ParameterInfo] = dataclasses.field(default_factory=list)

    def get_array_names(self) -> set[str]:
        """Get unique array/scalar parameter names."""
        return {p.array_name for p in self.parameters}

    def get_param_by_name(self, name: str) -> ParameterInfo | None:
        """Get parameter info by full name."""
        for p in self.parameters:
            if p.name == name:
                return p
        return None
