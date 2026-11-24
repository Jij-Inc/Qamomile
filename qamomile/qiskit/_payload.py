"""Retain Qiskit circuits independently of their optional QPY representation."""

from __future__ import annotations

import io
from collections.abc import Mapping
from copy import deepcopy
from threading import RLock
from typing import TYPE_CHECKING, Any

from qamomile._execution_payload import ExecutablePayload
from qamomile.errors import ExecutionError

if TYPE_CHECKING:
    from qamomile._qamomile_native import NativeShotCircuitTemplate
    from qamomile.compilation import CompiledProgram


def copy_circuit(circuit: Any, bindings: Mapping[Any, Any] | None = None) -> Any:
    """Copy a Qiskit circuit and optionally bind parameters on the copy.

    Args:
        circuit (Any): Source circuit with its retained operations and layout.
        bindings (Mapping[Any, Any] | None): Optional SDK parameter replacements.

    Returns:
        Any: Copy with independent bindings in Qamomile's emitted control-flow blocks.
    """
    if circuit.has_control_flow_op():
        from qiskit.circuit import ControlFlowOp

        # Qiskit 2.3.1's copy() shares mutable gates inside control-flow blocks.
        result = circuit.copy_empty_like()
        for row in circuit.data:
            operation = row.operation
            if isinstance(operation, ControlFlowOp):
                operation = operation.replace_blocks(
                    copy_circuit(block) for block in operation.blocks
                )
                operation.label = row.label
            else:
                operation = operation.copy()
            # Keep block-local wires and the schedule from the empty copy.
            result.data.append(row.replace(operation=operation))
        if bindings is None:
            return result
        circuit = result
    elif bindings is None:
        return circuit.copy()
    # Qiskit assigns a distinct experiment name to each out-of-place binding.
    return circuit.assign_parameters(bindings, inplace=False)


class QiskitExecutable(ExecutablePayload):
    """Own an unbound circuit and one lazily prepared Aer template.

    Args:
        circuit (Any): Emitted QuantumCircuit transferred to this owner.
        shot_template (NativeShotCircuitTemplate | None): Checked native numeric
            launch plan, or None for an ordinary circuit.
    """

    def __init__(
        self, circuit: Any, shot_template: NativeShotCircuitTemplate | None = None
    ) -> None:
        """Retain the circuit and initialize an empty endpoint preparation slot.

        Args:
            circuit (Any): SDK circuit whose lifetime is owned by this payload.
            shot_template (NativeShotCircuitTemplate | None): Immutable native
                numeric launch plan, if present.
        """
        self._circuit = circuit
        self.shot_template = shot_template
        self.hidden_parameters: dict[int, Any] = {}
        self._restore_hidden_parameters()
        self._aer: tuple[tuple[str, int, int], Any] | None = None
        self._lock = RLock()
        self._serialized: bytes | None = None

    def _restore_hidden_parameters(self) -> None:
        """Retain compiler slot positions after SDK parameter pruning.

        Surviving symbols keep their SDK identities. A removed symbol receives
        an unused position holder, preserving native column indices through
        QPY restoration. Consumers bind only symbols still free in the circuit;
        native launch evaluation checks every original slot.
        """
        if self.shot_template is None:
            return
        from qiskit.circuit import Parameter

        by_name = {parameter.name: parameter for parameter in self._circuit.parameters}
        self.hidden_parameters = {}
        for index in range(self.shot_template.slot_count):
            name = f"_qamomile_shot_{index}"
            self.hidden_parameters[index] = (
                by_name[name] if name in by_name else Parameter(name)
            )

    def restore_native_plan(self, artifact: CompiledProgram) -> None:
        """Restore persisted native rows against the enclosing artifact identity.

        Args:
            artifact (CompiledProgram): Native-validated immutable compiled artifact.

        Raises:
            ExecutionError: If a persisted plan or its artifact identity disagrees.
        """
        from qamomile._native import compiler as native
        from qamomile.effects import Requirement

        payload = dict(artifact.metadata.entries).get(
            "qamomile.native_shot_template.v1"
        )
        required = (
            Requirement.LOGICAL_SHOT_CONTEXT in artifact.execution_contract.requirements
        )
        if payload is None:
            if required:
                raise ExecutionError(
                    "Q4481: indexed QPY artifact is missing its native template plan"
                )
            return
        if not isinstance(payload, bytes) or not required:
            raise ExecutionError(
                "Q4481: native shot template metadata disagrees with execution contract"
            )
        try:
            self.shot_template = native.NativeShotCircuitTemplate.from_bytes(
                payload,
                artifact.parameter_abi._to_bytes(),
                artifact.result_abi._to_bytes(),
                artifact.target_fingerprint,
                artifact.source_digest.hex(),
            )
            self._restore_hidden_parameters()
        except ValueError as error:
            raise ExecutionError(str(error)) from error

    def circuit(self) -> Any:
        """Return a fresh circuit copy so callers cannot mutate the template.

        Returns:
            Any: Fresh copy isolated from template mutation.
        """
        return copy_circuit(self._circuit)

    @property
    def indexed_preparation_work(self) -> int:
        """Return retained SDK structure units without copying or exporting QPY.

        Returns:
            int: Circuit, wire, instruction, and parameter storage units.
                These bound preparation growth rather than process memory usage.
        """
        return circuit_preparation_work(self._circuit)

    @property
    def indexed_has_qubits(self) -> bool:
        """Return whether indexed execution must submit this retained circuit.

        Returns:
            bool: Whether the retained SDK circuit has any quantum wires.
        """
        return bool(self._circuit.num_qubits)

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
            return copy_circuit(self._aer[1])

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


def circuit_preparation_work(circuit: Any) -> int:
    """Count retained SDK structure without expanding lazy gate definitions.

    Args:
        circuit (Any): SDK circuit whose existing object graph is inspected.

    Returns:
        int: Preparation units, including one unit per retained circuit object
        and one unit per wire, instruction operand, and numeric parameter cell.
        Shared nested circuit objects are counted once within this graph.
    """
    pending = [circuit]
    seen = set()
    work = 0
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        work += 1 + current.num_qubits + current.num_clbits
        work += len(current.parameters)
        for instruction in current.data:
            operation = instruction.operation
            work += 1 + len(instruction.qubits) + len(instruction.clbits)
            for value in operation.params:
                size = getattr(value, "size", None)
                work += size if type(size) is int else 1
            pending.extend(getattr(operation, "blocks", ()))
            definition = getattr(operation, "_definition", None)
            if definition is not None:
                pending.append(definition)
    return work


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
