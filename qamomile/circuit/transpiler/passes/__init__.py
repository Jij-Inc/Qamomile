"""Base classes for compiler passes."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from qamomile.circuit.transpiler.errors import (
    QamomileCompileError,
    DependencyError,
    ValidationError,
)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


# Re-export error classes for convenience
__all__ = [
    "Pass",
    "QamomileCompileError",
    "ValidationError",
    "DependencyError",
]


class Pass(ABC, Generic[InputT, OutputT]):
    """Base class for all compiler passes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this pass."""
        pass

    @abstractmethod
    def run(self, input: InputT) -> OutputT:
        """Execute the pass transformation."""
        pass
