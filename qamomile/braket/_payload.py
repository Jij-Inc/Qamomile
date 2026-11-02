"""Retain the SDK's OpenQASM program at the Braket execution boundary."""

from __future__ import annotations

from typing import Any

from qamomile._execution_payload import ExecutablePayload
from qamomile.errors import ExecutionError


class BraketExecutable(ExecutablePayload):
    """Own a Braket OpenQASM program without a bytes round trip.

    Args:
        source (str): Emitted OpenQASM 3 source.
    """

    def __init__(self, source: str) -> None:
        """Construct the transfer representation expected by the Braket SDK.

        Args:
            source (str): Verified circuit or generated source retained by this backend.
        """
        from braket.ir.openqasm import Program

        self._program = Program(source=source, inputs={})

    def program(self) -> Any:
        """Return a copy isolated from SDK launch-time input mutation.

        Returns:
            Any: SDK Program with independent launch inputs.
        """
        return self._program.copy(deep=True)

    def to_bytes(self) -> bytes:
        """Encode the source only for explicit persistence.

        Returns:
            bytes: Portable backend data for explicit persistence.
        """
        return self._program.source.encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> BraketExecutable:
        """Restore the SDK program from saved OpenQASM source.

        Args:
            payload (bytes): Portable backend data from an explicit artifact export.

        Returns:
            BraketExecutable: Restored executable with its SDK lifetime owners.
        """
        try:
            return cls(payload.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ExecutionError(
                "Q8404: Braket artifact is not UTF-8 OpenQASM"
            ) from error
