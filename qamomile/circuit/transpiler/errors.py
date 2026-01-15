"""Compilation error classes for Qamomile transpiler."""


class QamomileCompileError(Exception):
    """Base class for all Qamomile compilation errors."""

    pass


class InliningError(QamomileCompileError):
    """Error during inline pass (inlining CallBlockOperations)."""

    pass


class ValidationError(QamomileCompileError):
    """Error during validation (e.g., non-classical I/O)."""

    def __init__(self, message: str, value_name: str | None = None):
        self.value_name = value_name
        super().__init__(message)


class DependencyError(QamomileCompileError):
    """Error when quantum operation depends on non-parameter classical value.

    This error indicates that the program requires JIT compilation
    which is not yet supported.
    """

    def __init__(
        self,
        message: str,
        quantum_op: str | None = None,
        classical_value: str | None = None,
    ):
        self.quantum_op = quantum_op
        self.classical_value = classical_value
        super().__init__(message)


class SeparationError(QamomileCompileError):
    """Error during quantum/classical separation."""

    pass


class EmitError(QamomileCompileError):
    """Error during backend code emission."""

    def __init__(self, message: str, operation: str | None = None):
        self.operation = operation
        super().__init__(message)


class ExecutionError(QamomileCompileError):
    """Error during program execution."""

    pass
