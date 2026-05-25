"""Compilation error classes for Qamomile transpiler."""

from dataclasses import dataclass
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


class EntrypointValidationError(ValidationError):
    """Error when a top-level transpilation entrypoint has unsupported I/O."""

    pass


class FrontendTransformError(QamomileCompileError):
    """Error during frontend AST-to-builder lowering."""

    pass


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
            lines.append(f"    ... and {len(self.available_bindings_keys) - 10} more")

        qubit_map_sample = self.available_qubit_map_keys[:10]
        lines.append(f"  Qubit map keys (sample): {qubit_map_sample}")
        if len(self.available_qubit_map_keys) > 10:
            lines.append(f"    ... and {len(self.available_qubit_map_keys) - 10} more")
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
                    "Bind the index variable by passing it in bindings "
                    "or ensure the loop variable is properly propagated."
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
                    "The index expression involves nested array access. "
                    "Ensure all intermediate arrays are bound in the bindings dict."
                )
                suggestions.append(
                    "Example: transpiler.transpile(kernel, "
                    "bindings={'edges': np.array([[0,1],[1,2]]), ...})"
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


class AffineTypeError(QamomileCompileError):
    """Base class for affine type violations.

    Affine types enforce that quantum resources (qubits) are used at most once.
    This prevents common errors such as reusing a consumed qubit or aliasing.
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


class QubitConsumedError(AffineTypeError):
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


class QubitBorrowConflictError(AffineTypeError):
    """Qubit slot inaccessible because another live handle borrows it.

    Raised when a qubit slot cannot be accessed because another live
    handle currently borrows it — a slice view that has not been
    returned, an outstanding element borrow, or any future borrow form
    Qamomile may add.  Unlike :class:`QubitConsumedError`, the slot is
    not destroyed: releasing the borrowing handle (slice assignment,
    element write-back, etc.) restores access.

    Example of incorrect code (overlapping slice views)::

        a = q[0:3]      # q[0..2] now borrowed by ``a``
        b = q[2:5]      # ERROR: q[2] is still borrowed by ``a``

    Correct code::

        a = q[0:3]
        q[0:3] = a      # return ``a`` first
        b = q[2:5]      # now safe

    Example of incorrect code (element borrow not returned before
    borrowing a neighbour)::

        q0 = qubits[0]
        q0 = qmc.h(q0)
        q1 = qubits[1]  # ERROR: q0 is still borrowed

    Correct code::

        q0 = qubits[0]
        q0 = qmc.h(q0)
        qubits[0] = q0  # return the element first
        q1 = qubits[1]  # now safe
    """

    pass


class QubitAliasError(AffineTypeError):
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


class UnreturnedBorrowError(AffineTypeError):
    """Borrowed array element not returned before array use.

    When you borrow an element from a qubit array, you must return it
    (write it back) before using other elements or the array itself.

    Example of incorrect code:
        q0 = qubits[0]
        q0 = qmc.h(q0)
        q1 = qubits[1]  # ERROR: q0 not returned yet

    Correct code:
        q0 = qubits[0]
        q0 = qmc.h(q0)
        qubits[0] = q0  # Return the borrowed element
        q1 = qubits[1]  # Now safe to borrow another
    """

    pass


class SliceBorrowViolationError(AffineTypeError):
    """Aliasing detected between a slice view and a direct parent access.

    Raised by :class:`SliceBorrowCheckPass` at transpile time when
    a parent array slot is simultaneously held by a ``VectorView`` and
    accessed directly, or when two overlapping views cover the same
    slot.  For slices with constant bounds this is normally caught at
    trace time; this error covers the post-fold case when slice bounds
    were symbolic UInt parameters resolved by bindings.

    Example of incorrect code (detected only after bindings resolve
    ``lo``/``hi`` to concrete values)::

        region = q[lo:hi]     # bindings give lo=0, hi=4 → covers {0,1,2,3}
        qa = region[0]        # borrows parent slot 0 via the view
        qb = q[0]             # borrows parent slot 0 directly
        # SliceBorrowViolationError: slot 0 is held by a slice view
    """

    pass


class QubitRebindError(AffineTypeError):
    """Quantum variable reassigned from a different quantum source.

    When a quantum variable is reassigned, the RHS must consume the
    same variable (self-update pattern). Reassigning from a different
    quantum variable would silently discard the original quantum state.

    The check runs at qkernel decoration time as a static AST analysis
    (see ``frontend.ast_transform.collect_quantum_rebind_violations``)
    and raises immediately — the wrapped ``QKernel`` object is never
    constructed when a violation is present. The check is run
    unconditionally for every decorated kernel: kernel-level quantum
    parameters (``Qubit`` / ``Vector[Qubit]``) seed origins from the
    signature, and the analyzer's recognition of internal quantum
    constructors (``qubit(...)`` / ``qubit_array(...)``) seeds further
    origins from inside the body so kernels that derive all of their
    quantum state from internal allocations are also covered.

    Branch-internal rebinds (assignments inside an ``if`` / ``for`` /
    ``while`` body) are NOT flagged at decoration time: compile-time
    conditional branches legitimately rebind quantum names (the
    compile-time-if lowering pass selects one branch and discards the
    other), and the single-pass AST analyzer cannot distinguish
    compile-time from runtime branches. To keep those compile-time
    patterns working, branch-internal violations are suppressed.

    This is a known coverage gap. ``AffineValidationPass`` in the IR
    layer only enforces "consumed at most once" and does NOT detect
    "never consumed" / "silent discard" patterns, so a genuine runtime
    ``if cond: q = qm.qubit("fresh")`` that discards a parameter is
    currently not raised by either layer. A dedicated IR-level
    silent-discard pass (or a flow-sensitive frontend analyzer) would
    be needed to close it; this is tracked as follow-up. Top-level
    (non-branch-internal) bypasses continue to raise at decoration
    time.

    Example of incorrect code:
        a = qm.h(b)  # ERROR: 'a' was quantum, now overwritten from 'b'
        a = b         # ERROR: 'a' was quantum, now overwritten from 'b'

    Correct patterns:
        a = qm.h(a)      # Self-update (OK)
        new = qm.h(b)    # New binding (OK, 'new' wasn't quantum before)
    """

    pass
