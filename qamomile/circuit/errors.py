"""Shared circuit error classes.

These errors live below both the frontend and transpiler layers so that
frontend affine-type enforcement does not need to depend on transpiler modules.
"""


class QamomileCompileError(Exception):
    """Base class for all Qamomile compilation errors."""

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
        q0 = qm.h(q0)
        q1 = qubits[1]  # ERROR: q0 not returned yet

    Correct code:
        q0 = qubits[0]
        q0 = qm.h(q0)
        qubits[0] = q0  # Return the borrowed element
        q1 = qubits[1]  # Now safe to borrow another
    """

    pass


class QubitRebindError(AffineTypeError):
    """Quantum variable reassigned from a different quantum source.

    When a quantum variable is reassigned, the RHS must consume the
    same variable (self-update pattern). Reassigning from a different
    quantum variable silently discards the original quantum state.

    Example of incorrect code:
        a = qm.h(b)  # ERROR: 'a' was quantum, now overwritten from 'b'
        a = b         # ERROR: 'a' was quantum, now overwritten from 'b'

    Correct patterns:
        a = qm.h(a)      # Self-update (OK)
        new = qm.h(b)    # New binding (OK, 'new' wasn't quantum before)
    """

    pass
