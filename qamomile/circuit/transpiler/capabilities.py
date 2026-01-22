"""Backend capability definitions for transpiler optimization."""

import enum
from typing import Protocol, runtime_checkable


class BackendCapability(enum.Flag):
    """Capabilities that a backend may support natively.

    Use these flags to indicate which composite gates and features
    a backend supports, allowing the transpiler to use native
    implementations when available.
    """

    NONE = 0

    # Composite gate support
    NATIVE_QFT = enum.auto()  # Native Quantum Fourier Transform
    NATIVE_IQFT = enum.auto()  # Native Inverse QFT
    NATIVE_QPE = enum.auto()  # Native Quantum Phase Estimation

    # Control flow support
    DYNAMIC_CIRCUITS = enum.auto()  # Mid-circuit measurement + conditionals
    FOR_LOOP = enum.auto()  # Native for loop support
    WHILE_LOOP = enum.auto()  # Native while loop support

    # Classical operations
    CLASSICAL_FEEDFORWARD = enum.auto()  # Classical ops mid-circuit

    # Common combinations
    BASIC_COMPOSITE = NATIVE_QFT | NATIVE_IQFT
    FULL_COMPOSITE = BASIC_COMPOSITE | NATIVE_QPE
    FULL_CONTROL_FLOW = DYNAMIC_CIRCUITS | FOR_LOOP | WHILE_LOOP


@runtime_checkable
class CapableBackend(Protocol):
    """Protocol for backends that declare their capabilities.

    Implement this protocol in backend-specific EmitPass classes
    to enable capability-based optimizations.
    """

    @property
    def capabilities(self) -> BackendCapability:
        """Return the capabilities this backend supports."""
        ...

    def has_capability(self, cap: BackendCapability) -> bool:
        """Check if this backend has a specific capability.

        Args:
            cap: The capability to check for.

        Returns:
            True if the backend has the capability, False otherwise.
        """
        ...
