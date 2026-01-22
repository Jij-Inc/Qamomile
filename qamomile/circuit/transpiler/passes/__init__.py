"""Base classes for compiler passes."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from qamomile.circuit.transpiler.errors import (
    QamomileCompileError,
    DependencyError,
    ValidationError,
    LinearTypeError,
)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


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


# Import submodule passes after Pass is defined to avoid circular imports
from .constant_fold import ConstantFoldingPass  # noqa: E402
from .linear_validate import LinearValidationPass  # noqa: E402
from .control_flow_visitor import (  # noqa: E402
    ControlFlowVisitor,
    OperationTransformer,
    OperationCollector,
    ValueCollector,
)
from .value_mapping import UUIDRemapper, ValueSubstitutor  # noqa: E402

# Re-export error classes for convenience
__all__ = [
    "Pass",
    "QamomileCompileError",
    "ValidationError",
    "DependencyError",
    "LinearTypeError",
    "ConstantFoldingPass",
    "LinearValidationPass",
    "ControlFlowVisitor",
    "OperationTransformer",
    "OperationCollector",
    "ValueCollector",
    "UUIDRemapper",
    "ValueSubstitutor",
]
