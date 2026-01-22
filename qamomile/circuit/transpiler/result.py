"""EmitResult data structures for transpiler output."""

from dataclasses import dataclass
from typing import Generic, TypeVar, Union

T = TypeVar("T")  # Backend circuit type


@dataclass
class QubitMapping:
    """Maps a logical Value to a physical qubit index."""

    value_uuid: str
    value_name: str
    qubit_index: int


@dataclass
class ClassicalMapping:
    """Maps a logical Value to a classical value or measurement result slot."""

    value_uuid: str
    value_name: str
    value: Union[int, float, bool, None]
    clbit_index: Union[int, None] = None  # For measurement results


@dataclass
class OutputMapping:
    """Maps Graph output position to physical resource."""

    output_index: int  # Position in tuple (0 for single return)
    value_uuid: str
    value_name: str
    kind: str  # "qubit" | "classical" | "measurement"
    physical_index: Union[int, None]  # qubit_index or clbit_index
    classical_value: Union[int, float, bool, None] = None


@dataclass
class EmitResult(Generic[T]):
    """Structured result from Transpiler.emit().

    Contains the backend-specific circuit along with mapping information
    that tracks the correspondence between logical Values and physical
    qubit/clbit indices.
    """

    circuit: T
    qubit_mappings: list[QubitMapping]
    classical_mappings: list[ClassicalMapping]
    output_mappings: list[OutputMapping]
    num_qubits: int
    num_clbits: int = 0

    def get_output_qubit_indices(self) -> list[int]:
        """Get qubit indices for all quantum outputs in order."""
        return [
            m.physical_index
            for m in self.output_mappings
            if m.kind == "qubit" and m.physical_index is not None
        ]

    def get_qubit_index_by_name(self, name: str) -> Union[int, None]:
        """Look up qubit index by Value name."""
        for m in self.qubit_mappings:
            if m.value_name == name:
                return m.qubit_index
        return None

    def get_output_mapping_by_index(self, index: int) -> Union[OutputMapping, None]:
        """Get output mapping by tuple index."""
        for m in self.output_mappings:
            if m.output_index == index:
                return m
        return None

    def get_measurement_clbit_indices(self) -> list[int]:
        """Get clbit indices for all measurement outputs in order."""
        return [
            m.physical_index
            for m in self.output_mappings
            if m.kind == "measurement" and m.physical_index is not None
        ]
