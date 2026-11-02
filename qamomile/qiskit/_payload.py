"""Retain Qiskit circuits independently of their optional QPY representation."""

from __future__ import annotations

import io
from copy import deepcopy
from threading import RLock
from typing import Any

from qamomile._execution_payload import ExecutablePayload
from qamomile.errors import ExecutionError


class QiskitExecutable(ExecutablePayload):
    """Own an unbound circuit and one lazily prepared Aer template.

    Args:
        circuit (Any): Emitted QuantumCircuit transferred to this owner.
    """

    def __init__(self, circuit: Any) -> None:
        """Retain the circuit and initialize an empty endpoint preparation slot.

        Args:
            circuit (Any): SDK circuit whose lifetime is owned by this payload.
        """
        self._circuit = circuit
        self._aer: tuple[tuple[str, int, int], Any] | None = None
        self._lock = RLock()
        self._serialized: bytes | None = None

    def circuit(self) -> Any:
        """Return a fresh circuit copy so callers cannot mutate the template.

        Returns:
            Any: Fresh copy isolated from template mutation.
        """
        return self._circuit.copy()

    def aer_circuit(
        self,
        backend: Any,
        method: str,
        optimization_level: int = 0,
        seed_transpiler: int = 0,
    ) -> Any:
        """Prepare a reusable template for the selected Aer execution method.

        Args:
            backend (Any): Aer backend defining the supported execution basis.
            method (str): Cache identity for the selected simulation method.
            optimization_level (int): Endpoint synthesis optimization level.
            seed_transpiler (int): Reproducible synthesis seed.

        Returns:
            Any: Fresh copy of the endpoint-compatible unbound circuit.
        """
        from qiskit import transpile

        key = (method, optimization_level, seed_transpiler)
        with self._lock:
            if self._aer is None or self._aer[0] != key:
                circuit = transpile(
                    self._circuit,
                    backend=backend,
                    optimization_level=optimization_level,
                    seed_transpiler=seed_transpiler,
                )
                self._aer = (key, circuit)
            return self._aer[1].copy()

    def to_bytes(self) -> bytes:
        """Export a copy in QPY's supported gate vocabulary.

        Returns:
            bytes: Portable backend data for explicit persistence.
        """
        from qiskit import qpy

        with self._lock:
            if self._serialized is None:
                circuit = _qpy_export_circuit(self._circuit)
                output = io.BytesIO()
                qpy.dump(circuit, output)
                self._serialized = output.getvalue()
            return self._serialized

    @classmethod
    def from_bytes(cls, payload: bytes) -> QiskitExecutable:
        """Restore one QPY circuit at the explicit persistence boundary.

        Args:
            payload (bytes): Portable backend data from an explicit artifact export.

        Returns:
            QiskitExecutable: Restored executable with its SDK lifetime owners.
        """
        from qiskit import qpy

        try:
            circuits = qpy.load(io.BytesIO(payload))
        except Exception as error:
            raise ExecutionError("Q8205: invalid QPY artifact payload") from error
        if len(circuits) != 1:
            raise ExecutionError("Q8206: QPY artifact must contain exactly one circuit")
        owner = cls(circuits[0])
        owner._serialized = payload
        return owner


def _qpy_export_circuit(circuit: Any) -> Any:
    """Copy SDK arithmetic definitions into QPY-compatible generic gate records.

    No basis synthesis is performed for persistence. SDK gates supported by QPY
    remain intact, including symbolic PauliEvolution and native controlled gates.

    Args:
        circuit (Any): Retained circuit, which must remain unchanged.

    Returns:
        Any: Export copy with unsupported arithmetic constructors replaced by
        generic gates holding the same exact definitions.
    """
    result = circuit.copy_empty_like()
    for row in circuit.data:
        result.append(_qpy_export_operation(row.operation), row.qubits, row.clbits)
    return result


def _qpy_export_operation(operation: Any) -> Any:
    """Normalize nested arithmetic and annotation bases only for explicit export.

    Args:
        operation (Any): SDK operation whose executable definition stays intact.

    Returns:
        Any: QPY-compatible copy preserving control and inverse modifiers.
    """
    from qiskit.circuit import AnnotatedOperation, ControlFlowOp, ControlledGate, Gate
    from qiskit.circuit.library import (
        BitFlipOracleGate,
        ModularAdderGate,
        PhaseOracleGate,
    )

    from ._deferred import DeferredGate

    if isinstance(operation, AnnotatedOperation):
        return AnnotatedOperation(
            _qpy_export_operation(operation.base_op), operation.modifiers
        )
    if isinstance(
        operation, (ModularAdderGate, BitFlipOracleGate, PhaseOracleGate, DeferredGate)
    ) or (
        isinstance(operation, ControlledGate)
        and isinstance(operation.base_gate, ModularAdderGate)
    ):
        operation = deepcopy(operation)
        replacement = Gate(
            operation.name,
            operation.num_qubits,
            operation.params,
            label=operation.label,
        )
        replacement.definition = _qpy_export_circuit(operation.definition)
        return replacement
    if type(operation) is Gate and operation.definition is not None:
        replacement = operation.copy()
        replacement.definition = _qpy_export_circuit(operation.definition)
        return replacement
    if isinstance(operation, ControlFlowOp):
        return operation.replace_blocks(
            [_qpy_export_circuit(block) for block in operation.blocks]
        )
    return operation
