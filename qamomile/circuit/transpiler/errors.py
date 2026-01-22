"""Compilation error classes for Qamomile transpiler."""

from dataclasses import dataclass, field
from enum import Enum


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


class ResolutionFailureReason(Enum):
    """Categorizes why qubit index resolution failed."""

    SYMBOLIC_INDEX_NOT_BOUND = "symbolic_index_not_bound"
    ARRAY_ELEMENT_NOT_IN_QUBIT_MAP = "array_element_not_in_qubit_map"
    INDEX_NOT_NUMERIC = "index_not_numeric"
    NESTED_ARRAY_RESOLUTION_FAILED = "nested_array_resolution_failed"
    DIRECT_UUID_NOT_FOUND = "direct_uuid_not_found"
    UNKNOWN = "unknown"


@dataclass
class OperandResolutionInfo:
    """Detailed information about a single operand that failed to resolve."""

    operand_name: str
    operand_uuid: str
    is_array_element: bool
    parent_array_name: str | None
    element_indices_names: list[str]
    failure_reason: ResolutionFailureReason
    failure_details: str


class QubitIndexResolutionError(EmitError):
    """Error when qubit indices cannot be resolved during emission.

    This error provides detailed diagnostic information about why
    qubit index resolution failed and suggests remediation steps.
    """

    def __init__(
        self,
        gate_type: str,
        operand_infos: list[OperandResolutionInfo],
        available_bindings_keys: list[str],
        available_qubit_map_keys: list[str],
    ):
        self.gate_type = gate_type
        self.operand_infos = operand_infos
        self.available_bindings_keys = available_bindings_keys
        self.available_qubit_map_keys = available_qubit_map_keys

        message = self._format_message()
        super().__init__(message, operation=f"GateOperation({gate_type})")

    def _format_message(self) -> str:
        """Format the detailed error message."""
        lines = [
            f"Failed to resolve qubit indices for gate '{self.gate_type}'.",
            "",
            "=== Operand Details ===",
        ]

        for i, info in enumerate(self.operand_infos):
            lines.append(f"  Operand {i}: '{info.operand_name}'")
            if info.is_array_element:
                lines.append(f"    - Array: '{info.parent_array_name}'")
                lines.append(f"    - Indices: {info.element_indices_names}")
            lines.append(f"    - Failure: {info.failure_reason.value}")
            lines.append(f"    - Details: {info.failure_details}")
            lines.append("")

        lines.append("=== Diagnostic Info ===")
        bindings_sample = self.available_bindings_keys[:10]
        lines.append(f"  Bindings keys (sample): {bindings_sample}")
        if len(self.available_bindings_keys) > 10:
            lines.append(
                f"    ... and {len(self.available_bindings_keys) - 10} more"
            )

        qubit_map_sample = self.available_qubit_map_keys[:10]
        lines.append(f"  Qubit map keys (sample): {qubit_map_sample}")
        if len(self.available_qubit_map_keys) > 10:
            lines.append(
                f"    ... and {len(self.available_qubit_map_keys) - 10} more"
            )
        lines.append("")

        lines.append("=== Suggested Fixes ===")
        suggestions = self._generate_suggestions()
        for suggestion in suggestions:
            lines.append(f"  - {suggestion}")

        return "\n".join(lines)

    def _generate_suggestions(self) -> list[str]:
        """Generate context-specific suggestions."""
        suggestions = []

        for info in self.operand_infos:
            if info.failure_reason == ResolutionFailureReason.SYMBOLIC_INDEX_NOT_BOUND:
                suggestions.append(
                    f"Bind the index variable by passing it in bindings "
                    f"or ensure the loop variable is properly propagated."
                )
            elif (
                info.failure_reason
                == ResolutionFailureReason.ARRAY_ELEMENT_NOT_IN_QUBIT_MAP
            ):
                suggestions.append(
                    f"Ensure the qubit array '{info.parent_array_name}' was properly "
                    f"initialized with enough qubits. Check that array indices are within bounds."
                )
            elif (
                info.failure_reason
                == ResolutionFailureReason.NESTED_ARRAY_RESOLUTION_FAILED
            ):
                suggestions.append(
                    f"The index expression involves nested array access. "
                    f"Ensure all intermediate arrays are bound in the bindings dict."
                )
                suggestions.append(
                    f"Example: transpiler.transpile(kernel, "
                    f"bindings={{'edges': np.array([[0,1],[1,2]]), ...}})"
                )
            elif info.failure_reason == ResolutionFailureReason.INDEX_NOT_NUMERIC:
                suggestions.append(
                    f"The resolved index for '{info.operand_name}' is not a number. "
                    f"Ensure your bindings contain numeric values for array indices."
                )

        if not suggestions:
            suggestions.append(
                "Check that all array indices can be resolved to concrete integers "
                "at compile time."
            )
            suggestions.append(
                "If using loop variables as indices, ensure the loop is being "
                "unrolled correctly."
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)

        return unique_suggestions


class ExecutionError(QamomileCompileError):
    """Error during program execution."""

    pass


class LinearTypeError(QamomileCompileError):
    """Base class for linear type violations.

    Linear types enforce that quantum resources (qubits) are used exactly once.
    This prevents common errors like using a consumed qubit or aliasing.
    """

    def __init__(
        self,
        message: str,
        handle_name: str | None = None,
        operation_name: str | None = None,
        first_use_location: str | None = None,
    ):
        self.handle_name = handle_name
        self.operation_name = operation_name
        self.first_use_location = first_use_location
        super().__init__(message)


class QubitConsumedError(LinearTypeError):
    """Qubit handle used after being consumed by a previous operation.

    Each qubit handle can only be used once. After a gate operation,
    you must reassign the result to use the new handle.

    Example of incorrect code:
        q1 = qm.h(q)
        q2 = qm.x(q)  # ERROR: q was already consumed by h()

    Correct code:
        q = qm.h(q)  # Reassign to capture new handle
        q = qm.x(q)  # Use the reassigned handle
    """

    pass


class QubitAliasError(LinearTypeError):
    """Same qubit used multiple times in one operation.

    Operations like cx() require distinct qubits for control and target.
    Using the same qubit in both positions is physically impossible
    and indicates a programming error.

    Example of incorrect code:
        q1, q2 = qm.cx(q, q)  # ERROR: same qubit as control and target

    Correct code:
        q1, q2 = qm.cx(control, target)  # Use distinct qubits
    """

    pass


class UnreturnedBorrowError(LinearTypeError):
    """Borrowed array element not returned before array use.

    When you borrow an element from a qubit array, you must return it
    (write it back) before using other elements or the array itself.

    Example of incorrect code:
        q0 = qubits[0]
        q0 = qm.h(q0)
        q1 = qubits[1]  # ERROR: q0 not returned yet

    Correct code:
        q0 = qubits[0]
        q0 = qm.h(q0)
        qubits[0] = q0  # Return the borrowed element
        q1 = qubits[1]  # Now safe to borrow another
    """

    pass
