"""Retain immutable LLVM bitcode at the QIR export boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qamomile._execution_payload import ExecutablePayload
from qamomile.errors import ExecutionError


@dataclass(frozen=True, slots=True)
class QIRExecutable(ExecutablePayload):
    """Own a verified QIR module without retaining mutable LLVM objects.

    Args:
        bitcode (bytes): LLVM-verified QIR bitcode produced by the emitter.
    """

    bitcode: bytes

    def to_bytes(self) -> bytes:
        """Export the retained LLVM bitcode.

        Returns:
            bytes: Portable QIR module bytes.
        """
        return self.bitcode

    def module(self) -> Any:
        """Parse a fresh independently owned LLVM module for inspection.

        Returns:
            Any: PyQIR Module with its own LLVM context.

        Raises:
            ExecutionError: If the bitcode is malformed or fails LLVM verification.
            ImportError: If the optional PyQIR dependency is unavailable.
        """
        import pyqir

        try:
            module = pyqir.Module.from_bitcode(pyqir.Context(), self.bitcode)
            error = module.verify()
            if error is not None:
                raise ValueError(error)
            return module
        except ValueError as error:
            raise ExecutionError(f"Q9005: invalid QIR bitcode: {error}") from error

    @classmethod
    def from_bytes(cls, payload: bytes) -> QIRExecutable:
        """Restore bitcode after the native artifact envelope is verified.

        Args:
            payload (bytes): Exported QIR module bytes.

        Returns:
            QIRExecutable: Immutable owner of the LLVM-verified module.

        Raises:
            ExecutionError: If the bitcode is malformed or fails LLVM verification.
            ImportError: If PyQIR is unavailable.
        """
        owner = cls(payload)
        owner.module()
        return owner
