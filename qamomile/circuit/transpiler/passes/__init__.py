"""Base classes for compiler passes."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from qamomile.circuit.transpiler.errors import (
    QamomileCompileError,
    DependencyError,
    ValidationError,
    AffineTypeError,
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
from .compile_time_if_lowering import CompileTimeIfLoweringPass  # noqa: E402
from .affine_validate import AffineValidationPass  # noqa: E402
from .control_flow_visitor import (  # noqa: E402
    ControlFlowVisitor,
    OperationTransformer,
    OperationCollector,
    ValueCollector,
)
from .validate_while import ValidateWhileContractPass  # noqa: E402
from .value_mapping import UUIDRemapper, ValueSubstitutor  # noqa: E402

# Re-export error classes for convenience
__all__ = [
    "Pass",
    "QamomileCompileError",
    "ValidationError",
    "DependencyError",
    "AffineTypeError",
    "ConstantFoldingPass",
    "CompileTimeIfLoweringPass",
    "AffineValidationPass",
    "ControlFlowVisitor",
    "OperationTransformer",
    "OperationCollector",
    "ValueCollector",
    "ValidateWhileContractPass",
    "UUIDRemapper",
    "ValueSubstitutor",
]
