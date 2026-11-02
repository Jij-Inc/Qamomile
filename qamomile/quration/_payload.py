"""Own the PyQret circuit together with its complete native owner graph."""

from __future__ import annotations

import json
from typing import Any

from qamomile._execution_payload import ExecutablePayload
from qamomile.errors import ExecutionError


class QurationExecutable(ExecutablePayload):
    """Keep generated PyQret objects alive for repeated simulation.

    Args:
        circuit (Any): Generated circuit.
        module (Any): Module used for explicit portable export.
        owners (tuple): Context, builder, and generator lifetime owners.
    """

    def __init__(self, circuit: Any, module: Any, owners: tuple[Any, ...]) -> None:
        """Retain the entire owner graph without an SDK serialization step.

        Args:
            circuit (Any): SDK circuit whose lifetime is owned by this payload.
            module (Any): SDK module retained for circuit ownership and explicit export.
            owners (tuple[Any, ...]): SDK objects that must outlive the executable circuit.
        """
        self.circuit = circuit
        self.module = module
        self.owners = owners
        setattr(circuit, "_qamomile_owners", (module, *owners))

    def to_bytes(self) -> bytes:
        """Export deterministic native module JSON on an explicit save.

        Returns:
            bytes: Portable backend data for explicit persistence.
        """
        data = self.module.dumps()
        data["metadata"].pop("created_at", None)
        return json.dumps(
            data, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()

    @classmethod
    def from_bytes(cls, payload: bytes) -> QurationExecutable:
        """Restore a saved module with an independently owned context.

        Args:
            payload (bytes): Portable backend data from an explicit artifact export.

        Returns:
            QurationExecutable: Restored executable with its SDK lifetime owners.
        """
        from pyqret.frontend import Context

        try:
            context = Context()
            module = context.loads(payload.decode("utf-8"))
            return cls(module.get_circuit("main"), module, (context,))
        except (ValueError, RuntimeError, UnicodeDecodeError) as error:
            raise ExecutionError(f"Q8906: invalid Quration module: {error}") from error
