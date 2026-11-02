"""Retain immutable Cirq circuits until an explicit JSON export."""

from __future__ import annotations

from typing import Any

from qamomile._execution_payload import ExecutablePayload
from qamomile.errors import ExecutionError


class CirqExecutable(ExecutablePayload):
    """Own a frozen circuit safe to resolve independently for every launch.

    Args:
        circuit (Any): Emitted Cirq circuit.
    """

    def __init__(self, circuit: Any) -> None:
        """Freeze the emitted circuit without serialization.

        Args:
            circuit (Any): SDK circuit whose lifetime is owned by this payload.
        """
        self.circuit = circuit.freeze()

    def to_bytes(self) -> bytes:
        """Serialize a mutable view in the existing Cirq JSON format.

        Returns:
            bytes: Portable backend data for explicit persistence.
        """
        import cirq

        return cirq.to_json(self.circuit.unfreeze()).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> CirqExecutable:
        """Restore and freeze a saved Cirq circuit once.

        Args:
            payload (bytes): Portable backend data from an explicit artifact export.

        Returns:
            CirqExecutable: Restored executable with its SDK lifetime owners.
        """
        import cirq

        try:
            circuit = cirq.read_json(json_text=payload.decode("utf-8"))
        except Exception as error:
            raise ExecutionError("Q8811: invalid Cirq JSON artifact") from error
        if not isinstance(circuit, (cirq.Circuit, cirq.FrozenCircuit)):
            raise ExecutionError(
                "Q8812: Cirq artifact must contain exactly one circuit"
            )
        return cls(circuit)
