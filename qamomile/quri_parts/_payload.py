"""Retain SDK gate templates and typed deferred angle expressions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from threading import RLock
from typing import Any, cast

from qamomile._execution_payload import ExecutablePayload
from qamomile.circuit import (
    CircuitInstruction,
    CircuitProgram,
    CircuitValue,
    ClassicalExpr,
    ParameterValue,
    _block_to_data,
    _value_to_data,
)
from qamomile.errors import ExecutionError
from qamomile.ir import FrozenMapping, thaw_value
from qamomile.operations import evaluate_classical


def _runtime(value: CircuitValue) -> bool:
    """Identify angle expressions which need launch-time parameter binding.

    Args:
        value (CircuitValue): Compiler-projected angle expression.

    Returns:
        bool: Whether this expression requires launch-time binding.
    """
    if isinstance(value, ParameterValue):
        return True
    if isinstance(value, ClassicalExpr):
        return any(_runtime(item) for item in value.operands)
    if isinstance(value, tuple):
        return any(_runtime(item) for item in value)
    return False


def _resolve(value: CircuitValue, parameters: Mapping[str, object]) -> object:
    """Bind SDK angle expressions using the native classical evaluator.

    Args:
        value (CircuitValue): Compiler-projected angle expression.
        parameters (Mapping): Validated launch values.

    Returns:
        object: Concrete SDK argument.
    """
    if isinstance(value, ParameterValue):
        current = parameters[value.name]
        for component in value.path:
            current = current[component]  # type: ignore[index]
        return current
    if isinstance(value, ClassicalExpr):
        return evaluate_classical(
            value.opcode,
            tuple(_resolve(item, parameters) for item in value.operands),
            attributes={
                name: thaw_value(item) for name, item in value.attributes.entries
            },
        )
    if isinstance(value, tuple):
        return tuple(_resolve(item, parameters) for item in value)
    return value


def _restore_value(value: Any) -> CircuitValue:
    """Decode saved SDK angle expressions at the explicit import boundary.

    Args:
        value (Any): Compiler-projected angle expression.

    Returns:
        CircuitValue: Typed SDK angle expression.
    """
    if not isinstance(value, dict):
        if value is None or isinstance(value, (bool, int, float)):
            return value
        raise ExecutionError("Q8709: malformed circuit expression")
    if "parameter" in value:
        return ParameterValue(value["parameter"], tuple(value.get("path", [])))
    if "expr" in value:
        return ClassicalExpr(
            value["expr"],
            tuple(_restore_value(item) for item in value["operands"]),
            FrozenMapping.from_mapping(value.get("attributes", {})),
        )
    if "tuple" in value:
        return tuple(_restore_value(item) for item in value["tuple"])
    raise ExecutionError("Q8709: unsupported circuit expression")


class QuriExecutable(ExecutablePayload):
    """Own constant SDK gates and typed expressions for deferred angles.

    Args:
        source (CircuitProgram | bytes): Verified circuit or explicitly saved data.
        width (int): Logical circuit width.
        instructions (tuple): Checked instructions in execution order.
        data (dict | None): Already decoded persistence data, if imported.
    """

    def __init__(
        self,
        source: CircuitProgram | bytes,
        width: int,
        instructions: tuple[CircuitInstruction, ...],
        data: dict[str, Any] | None = None,
    ) -> None:
        """Build constant SDK gates once and retain only dynamic angle recipes.

        Args:
            source (CircuitProgram | bytes): Verified circuit or generated source retained by this backend.
            width (int): Compiler-derived logical qubit count.
            instructions (tuple[CircuitInstruction, ...]): Verified instructions in execution order.
            data (dict[str, Any] | None): Already decoded persistence data, when restoring a saved circuit.
        """
        from quri_parts.circuit import QuantumCircuit

        self.source = source
        self.width = width
        self._data = data
        self._instructions = instructions
        self._public_parameters: dict[tuple[str, tuple[int | str, ...]], Any] = {}
        self._qulacs: Any = None
        self._qulacs_mapper: Any = None
        self._qulacs_fallback = False
        self.measurements: dict[int, int] = {}
        self._steps: list[Any | CircuitInstruction] = []
        self._public: Any = None
        self._lock = RLock()
        for instruction in instructions:
            if instruction.opcode == "measure":
                self.measurements[int(instruction.parameters[0])] = instruction.qubits[
                    0
                ]
            elif any(_runtime(value) for value in instruction.parameters):
                self._steps.append(instruction)
            else:
                circuit = QuantumCircuit(width)
                self._emit(circuit, instruction, {})
                self._steps.extend(circuit.gates)
        self._constant = None
        if not any(isinstance(step, CircuitInstruction) for step in self._steps):
            circuit = QuantumCircuit(width)
            for gate in self._steps:
                circuit.add_gate(cast(Any, gate))
            self._constant = circuit.freeze()

    @classmethod
    def from_circuit(cls, logical: CircuitProgram) -> QuriExecutable:
        """Capture a verified circuit without converting it to JSON.

        Args:
            logical (CircuitProgram): Verified target-neutral circuit and native-derived ABI.

        Returns:
            QuriExecutable: Executable owning the SDK gate templates.
        """
        return cls(logical, logical.width, logical.body.instructions)

    def _emit(
        self,
        circuit: Any,
        instruction: CircuitInstruction,
        parameters: Mapping[str, object],
    ) -> None:
        """Emit one deferred gate after evaluating its typed angle expressions.

        Args:
            circuit (Any): SDK circuit whose lifetime is owned by this payload.
            instruction (CircuitInstruction): One verified instruction with possibly deferred angles.
            parameters (Mapping[str, object]): Validated launch values keyed by parameter name.
        """
        from ._materialization import (
            _add_controlled_instruction,
            _add_native_instruction,
        )

        try:
            values = tuple(
                _resolve(value, parameters) for value in instruction.parameters
            )
        except (ArithmeticError, IndexError, TypeError, ValueError) as error:
            raise ExecutionError(
                f"Q8711: cannot evaluate circuit parameters: {error}"
            ) from error
        if instruction.controls:
            _add_controlled_instruction(
                circuit,
                instruction.opcode,
                list(instruction.controls),
                list(instruction.qubits),
                values,
            )
        else:
            _add_native_instruction(
                circuit, instruction.opcode, list(instruction.qubits), values
            )

    def bind(self, parameters: Mapping[str, object]) -> Any:
        """Bind only deferred angles into an independent immutable SDK circuit.

        Args:
            parameters (Mapping[str, object]): Validated launch values keyed by parameter name.

        Returns:
            Any: Immutable circuit for this invocation.
        """
        if self._constant is not None:
            return self._constant
        from quri_parts.circuit import QuantumCircuit

        circuit = QuantumCircuit(self.width)
        for step in self._steps:
            if isinstance(step, CircuitInstruction):
                self._emit(circuit, step, parameters)
            else:
                circuit.add_gate(step)
        return circuit.freeze()

    def qulacs_circuit(self, parameters: Mapping[str, object]) -> Any:
        """Bind a private copy of a retained native Qulacs template.

        Args:
            parameters (Mapping[str, object]): Validated invocation values.

        Returns:
            Any: Independent native circuit; quantum state is never retained.

        Raises:
            ExecutionError: If a deferred angle cannot be evaluated.
        """
        from quri_parts.qulacs.circuit import (
            convert_circuit,
            convert_parametric_circuit,
        )

        with self._lock:
            if self._qulacs is None and not self._qulacs_fallback:
                if self._constant is not None:
                    self._qulacs = convert_circuit(self._constant)
                else:
                    try:
                        public = self.public_circuit()
                    except ExecutionError as error:
                        if error.diagnostic.code not in {"Q8721", "Q8722"}:
                            raise
                        self._qulacs_fallback = True
                    else:
                        self._qulacs, self._qulacs_mapper = convert_parametric_circuit(
                            public
                        )
            if self._qulacs_fallback:
                return convert_circuit(self.bind(parameters))
            circuit = self._qulacs.copy()
            if self._qulacs_mapper is not None:
                values = [
                    _resolve(ParameterValue(name, path), parameters)
                    for name, path in self._public_parameters
                ]
                for index, value in enumerate(self._qulacs_mapper(values)):
                    circuit.set_parameter(index, value)
            return circuit

    def data(self) -> dict[str, Any]:
        """Project inspection/export data without a serialize-and-parse cycle.

        Returns:
            dict[str, Any]: Inspection and export data derived from the retained circuit.
        """
        if self._data is not None:
            return self._data
        assert isinstance(self.source, CircuitProgram)
        return {
            "schema": 2,
            "width": self.width,
            "measurement_count": self.source.measurement_count,
            "body": _block_to_data(self.source.body),
            "result": _value_to_data(self.source.result_value),
        }

    def public_circuit(self) -> Any:
        """Return the cached immutable affine SDK projection when representable.

        Returns:
            Any: Immutable SDK affine circuit suitable for public estimators.
        """
        from quri_parts.circuit import LinearMappedParametricQuantumCircuit

        from ._materialization import _materialize_parametric_instructions

        with self._lock:
            if self._public is None:
                circuit = LinearMappedParametricQuantumCircuit(self.width)
                parameters: dict[tuple[str, tuple[int | str, ...]], Any] = {}
                _materialize_parametric_instructions(
                    circuit, self._instructions, parameters
                )
                self._public = circuit.freeze()
                self._public_parameters = parameters
            return self._public

    def to_bytes(self) -> bytes:
        """Export the logical format only when persistence is requested.

        Returns:
            bytes: Portable backend data for explicit persistence.
        """
        return self.source if isinstance(self.source, bytes) else self.source.to_bytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> QuriExecutable:
        """Restore saved expressions and construct SDK templates once.

        Args:
            payload (bytes): Portable backend data from an explicit artifact export.

        Returns:
            QuriExecutable: Restored executable with its SDK lifetime owners.
        """
        try:
            data = json.loads(payload)
            if (
                data["schema"] != 2
                or type(data["width"]) is not int
                or data["width"] < 0
            ):
                raise ValueError("invalid circuit header")
            if not isinstance(data["body"], list):
                raise ValueError("invalid instruction block")
            for row in data["body"]:
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("op"), str)
                    or not isinstance(row.get("qubits"), list)
                    or not isinstance(row.get("controls"), list)
                    or not isinstance(row.get("parameters"), list)
                    or row.get("condition") is not None
                    or row.get("regions")
                    or any(
                        type(wire) is not int or not 0 <= wire < data["width"]
                        for wire in [*row["qubits"], *row["controls"]]
                    )
                ):
                    raise ValueError("invalid static instruction")
            instructions = tuple(
                CircuitInstruction(
                    row["op"],
                    tuple(row["qubits"]),
                    tuple(_restore_value(value) for value in row["parameters"]),
                    controls=tuple(row["controls"]),
                )
                for row in data["body"]
            )
            return cls(payload, data["width"], instructions, data)
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            RecursionError,
            OverflowError,
        ) as error:
            raise ExecutionError("Q8703: invalid QURI Parts logical circuit") from error
