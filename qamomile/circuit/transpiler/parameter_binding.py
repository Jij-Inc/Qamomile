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

    def get_ordered_params(self) -> list[Any]:
        """Get backend parameter objects in definition order.

        Useful for backends that require positional parameter binding
        (e.g., QURI Parts).

        Returns:
            List of backend_param objects in the order they were defined.

        Example:
            # For QURI Parts that uses positional binding:
            param_values = [bindings[p.name] for p in metadata.parameters]
            bound_circuit = circuit.bind_parameters(param_values)
        """
        return [p.backend_param for p in self.parameters]

    def to_binding_dict(self, bindings: dict[str, Any]) -> dict[Any, Any]:
        """Convert indexed bindings to backend parameter bindings.

        Transforms user-provided bindings (with indexed names like "gammas[0]")
        into a dictionary mapping backend parameter objects to values.
        Useful for backends that use dict-based parameter binding (e.g., Qiskit).

        Args:
            bindings: Dictionary mapping parameter names to values.
                      e.g., {"gammas[0]": 0.1, "gammas[1]": 0.2, "theta": 0.5}

        Returns:
            Dictionary mapping backend_param objects to values.

        Example:
            # For Qiskit that uses dict-based binding:
            qiskit_bindings = metadata.to_binding_dict(bindings)
            bound_circuit = circuit.assign_parameters(qiskit_bindings)
        """
        return {
            p.backend_param: bindings[p.name]
            for p in self.parameters
            if p.name in bindings
        }
