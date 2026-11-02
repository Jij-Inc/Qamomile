"""Emit target-neutral circuit projections as Qiskit SDK objects."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import replace
from typing import Any, cast

from qamomile.circuit import (
    BooleanPlan,
    CircuitBlock,
    CircuitInstruction,
    CircuitProgram,
    CircuitValue,
    ClassicalExpr,
    DeferredDefinition,
    LoopIndex,
    MeasurementBit,
    ParameterValue,
    scalar_parameters,
)
from qamomile.errors import CapabilityError
from qamomile.observable import Hamiltonian


class _EmissionParameters(dict[ParameterValue | LoopIndex, Any]):
    """Own scalar bindings and the bounded, per-emission definition cache.

    Args:
        values (dict[ParameterValue | LoopIndex, Any]): SDK scalar bindings.
        gates (dict[Any, Any] | None): Shared definitions
            for nested loop scopes within this compilation only.
    """

    def __init__(
        self,
        values: dict[ParameterValue | LoopIndex, Any],
        gates: dict[Any, Any] | None = None,
    ) -> None:
        """Initialize one emission scope without retaining process-global state.

        Args:
            values (dict[ParameterValue | LoopIndex, Any]): Scalar bindings.
            gates (dict[Any, Any] | None): Existing cache.
        """
        super().__init__(values)
        self.gates = {} if gates is None else gates


def emit_circuit(program: CircuitProgram) -> Any:
    """Materialize target-neutral circuit instructions as a QuantumCircuit.

    Args:
        program (CircuitProgram): Verified projection with its native ABI.

    Returns:
        Any: SDK circuit retaining supported composite operations.

    Raises:
        CapabilityError: If an instruction has no supported SDK representation.
    """

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    circuit = QuantumCircuit(
        program.width,
        program.measurement_count,
        name=program.name,
    )
    parameters: dict[ParameterValue | LoopIndex, Any] = {
        leaf.value: Parameter(
            f"p{index}",
            uuid=uuid.UUID(
                bytes=hashlib.sha256(
                    (
                        f"qamomile:{program.semantic_id}:{leaf.value.name}:"
                        f"{leaf.value.path!r}"
                    ).encode()
                ).digest()[:16]
            ),
        )
        for index, leaf in enumerate(scalar_parameters(program.parameter_abi))
    }
    _emit_block(circuit, program.body, _EmissionParameters(parameters))
    return circuit


def _emit_block(
    circuit: Any,
    block: CircuitBlock,
    parameters: dict[ParameterValue | LoopIndex, Any],
) -> None:
    """Emit one structured circuit block into Qiskit.

    Args:
        circuit (Any): Destination SDK circuit.
        block (CircuitBlock): Verified ordered instructions.
        parameters (dict[ParameterValue | LoopIndex, Any]): Scoped SDK bindings.

    Raises:
        CapabilityError: If a contained instruction cannot be emitted.
    """

    for instruction in block.instructions:
        _emit_instruction(circuit, instruction, parameters)


def _emit_instruction(
    circuit: Any,
    instruction: CircuitInstruction,
    parameters: dict[ParameterValue | LoopIndex, Any],
) -> None:
    """Emit one target-neutral circuit instruction into Qiskit.

    Args:
        circuit (Any): Destination SDK circuit.
        instruction (CircuitInstruction): Verified instruction and operand layout.
        parameters (dict[ParameterValue | LoopIndex, Any]): Scoped SDK bindings.

    Raises:
        CapabilityError: If the instruction is malformed or unsupported.
    """

    op = instruction.opcode
    qubits = instruction.qubits
    if op == "store_bool":
        destination, value = instruction.parameters
        if type(destination) is not int or instruction.regions or instruction.controls:
            raise CapabilityError("Q8235: malformed Boolean store")
        circuit.store(
            circuit.clbits[destination], _emit_condition(circuit, value, parameters)
        )
        return
    if op in {
        "phase_oracle",
        "xor_oracle",
        "linear_function",
        "linear_function_inverse",
    }:
        _emit_boolean(circuit, instruction)
        return
    if op in {"add_into", "subtract_into"}:
        _emit_addition(circuit, instruction)
        return
    if op == "pauli_evolution":
        _emit_pauli_evolution(circuit, instruction, parameters)
        return
    if op in {"deferred", "inverse:deferred"} or op.startswith(
        ("box:", "inverse:box:")
    ):
        _emit_named_box(circuit, instruction, parameters)
        return
    if instruction.controls:
        _emit_controlled_instruction(circuit, instruction, parameters)
        return
    if op == "global_phase":
        from qiskit.circuit.library import GlobalPhaseGate

        circuit.append(
            GlobalPhaseGate(_emit_value(instruction.parameters[0], parameters)), []
        )
        return
    if op in {"h", "x", "y", "z", "s", "t", "sdg", "tdg", "reset"}:
        getattr(circuit, op)(qubits[0])
        return
    if op in {"p", "rx", "ry", "rz"}:
        angle = _emit_value(instruction.parameters[0], parameters)
        getattr(circuit, op)(angle, qubits[0])
        return
    if op in {"cp", "rzz"}:
        angle = _emit_value(instruction.parameters[0], parameters)
        getattr(circuit, op)(angle, *qubits)
        return
    if op == "ccx":
        circuit.ccx(*qubits)
        return
    if op in {"cx", "cz", "swap"}:
        getattr(circuit, op)(*qubits)
        return
    if op in {"measure_x", "measure_y", "measure_z"}:
        from qiskit.circuit.library import PauliProductMeasurement
        from qiskit.quantum_info import Pauli

        circuit.append(
            PauliProductMeasurement(Pauli(op[-1].upper())),
            qubits,
            instruction.parameters,
        )
        return
    if op == "measure":
        classical_index = instruction.parameters[0]
        if not isinstance(classical_index, int):
            raise CapabilityError("Q8207: measurement index must be static")
        circuit.measure(qubits[0], classical_index)
        return
    if op == "if":
        condition = instruction.condition
        if condition is None or len(instruction.regions) != 2:
            raise CapabilityError("Q8208: malformed device if instruction")
        predicate = _emit_condition(circuit, condition, parameters)
        with circuit.if_test(predicate) as else_:
            _emit_block(circuit, instruction.regions[0], parameters)
        with else_:
            _emit_block(circuit, instruction.regions[1], parameters)
        return
    if op == "for":
        _emit_for(circuit, instruction, parameters)
        return
    if op.startswith("box:"):
        _emit_named_box(circuit, instruction, parameters)
        return
    raise CapabilityError(f"Q8209: Qiskit cannot emit logical instruction {op!r}")


def _boolean_expression(terms: tuple[tuple[int, ...], ...], width: int) -> str:
    """Render checked ANF as a balanced SDK expression without evaluating it.

    Args:
        terms (tuple[tuple[int, ...], ...]): Verified conjunctions XORed together.
        width (int): Number of ordered source variables, including unused ones.

    Returns:
        str: Boolean expression whose parser depth is logarithmic in term count.
    """
    expressions = [
        "(" + " & ".join(f"x{bit}" for bit in term) + ")" if term else "(x0 | ~x0)"
        for term in terms
    ]
    expressions.extend(f"(x{bit} & ~x{bit})" for bit in range(width))
    if not expressions:
        return "False"
    while len(expressions) > 1:
        expressions = [
            f"({expressions[index]} ^ {expressions[index + 1]})"
            if index + 1 < len(expressions)
            else expressions[index]
            for index in range(0, len(expressions), 2)
        ]
    return expressions[0]


def _emit_boolean(circuit: Any, instruction: CircuitInstruction) -> None:
    """Emit native Boolean operations from compiler-verified polynomials.

    SDK truth-table synthesis is limited to eight source bits. Larger inputs
    use the same checked ANF as a polynomial-size body-backed SDK gate.

    Args:
        circuit (Any): Destination SDK circuit.
        instruction (CircuitInstruction): Checked phase, XOR, or linear map.

    Raises:
        CapabilityError: If the projection's Boolean operand layout is invalid.
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import (
        BitFlipOracleGate,
        LinearFunction,
        PhaseOracleGate,
        XGate,
        ZGate,
    )

    if len(instruction.parameters) != 1 or not isinstance(
        instruction.parameters[0], BooleanPlan
    ):
        raise CapabilityError("Q8238: malformed Boolean realization")
    plan = instruction.parameters[0]
    width = plan.input_width
    phase = instruction.opcode == "phase_oracle"
    linear = instruction.opcode.startswith("linear_function")
    expected = width if phase or linear else width + len(plan.outputs)
    if len(instruction.qubits) != expected or (phase and len(plan.outputs) != 1):
        raise CapabilityError("Q8238: Boolean realization width mismatch")
    gate: Any
    if linear:
        # Matrix storage is bounded; larger maps retain their polynomial circuit.
        if width > 1024:
            raise CapabilityError("Q8238: linear SDK realization exceeds 1024 wires")
        matrix = [[False] * width for _ in range(width)]
        for row, terms in enumerate(plan.outputs):
            for term in terms:
                if len(term) != 1:
                    raise CapabilityError(
                        "Q8238: non-linear term in a verified linear map"
                    )
                matrix[row][term[0]] = True
        gate = LinearFunction(matrix)
        if instruction.opcode == "linear_function_inverse":
            gate = gate.inverse(annotated=True)
    elif (
        0 < width <= 8
        and len(plan.outputs) == 1
        and len(plan.outputs[0]) <= 256
        and any(plan.outputs[0])
    ):
        expression = _boolean_expression(plan.outputs[0], width)
        factory = PhaseOracleGate if phase else BitFlipOracleGate
        gate = factory(expression, var_order=[f"x{bit}" for bit in range(width)])
    else:
        definition = QuantumCircuit(expected)
        for output, terms in enumerate(plan.outputs):
            for term in terms:
                if phase and not term:
                    definition.global_phase += np.pi
                    continue
                target = term[-1] if phase else width + output
                controls = term[:-1] if phase else term
                primitive: Any = ZGate() if phase else XGate()
                if controls:
                    primitive = primitive.control(len(controls), annotated=True)
                definition.append(primitive, (*controls, target))
        gate = definition.to_gate(label=instruction.opcode)
    if instruction.controls:
        gate = gate.control(len(instruction.controls), annotated=True)
    circuit.append(gate, (*instruction.controls, *instruction.qubits))


def _emit_addition(circuit: Any, instruction: CircuitInstruction) -> None:
    """Preserve a verified wrapping addition as an SDK arithmetic gate.

    Args:
        circuit (Any): Qiskit circuit receiving the operation.
        instruction (CircuitInstruction): Equal little-endian operand halves.

    Raises:
        CapabilityError: If the target-neutral projection has malformed shape.
    """

    from qiskit.circuit.library import ModularAdderGate

    if not instruction.qubits or len(instruction.qubits) % 2 or instruction.parameters:
        raise CapabilityError("Q8236: malformed wrapping addition instruction")
    if len(instruction.qubits) > 128:
        raise CapabilityError(
            "Q8237: Qiskit addition synthesis exceeds the 128-wire emission budget; select a portable realization"
        )
    gate: Any = ModularAdderGate(len(instruction.qubits) // 2)
    if instruction.opcode == "subtract_into":
        gate = gate.inverse(annotated=True)
    if instruction.controls:
        gate = gate.control(len(instruction.controls), annotated=True)
        _append_controlled_gate(
            circuit, gate, (*instruction.controls, *instruction.qubits)
        )
    else:
        circuit.append(gate, instruction.qubits)
        metadata = dict(circuit.metadata or {})
        named_calls = metadata.setdefault("qamomile_named_calls", [])
        if not isinstance(named_calls, list):
            raise CapabilityError("Q8236: addition metadata is malformed")
        if gate.name not in named_calls:
            named_calls.append(gate.name)
        circuit.metadata = metadata


def _emit_pauli_evolution(
    circuit: Any,
    instruction: CircuitInstruction,
    parameters: dict[ParameterValue | LoopIndex, Any],
) -> None:
    """Emit one abstract QCore evolution as Qiskit's native composite gate.

    Args:
        circuit (Any): Qiskit circuit receiving the operation.
        instruction (CircuitInstruction): Verified evolution containing one Hamiltonian
            and one time expression.
        parameters (dict[ParameterValue | LoopIndex, Any]): Runtime SDK parameter mapping.

    Raises:
        CapabilityError: If the native projection is malformed.
    """

    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    if (
        instruction.condition is not None
        or instruction.regions
        or len(instruction.parameters) != 2
    ):
        raise CapabilityError("Q8235: malformed Pauli evolution instruction")
    hamiltonian, raw_time = instruction.parameters
    if not isinstance(hamiltonian, Hamiltonian):
        raise CapabilityError("Q8235: Pauli evolution Hamiltonian is malformed")
    width = len(instruction.qubits)
    if hamiltonian.num_qubits > width:
        raise CapabilityError("Q8235: Pauli evolution width is malformed")
    rows: list[tuple[str, complex]] = []
    if hamiltonian.constant:
        rows.append(("I" * width, hamiltonian.constant))
    for word, coefficient in hamiltonian.terms:
        label = ["I"] * width
        for factor in word.factors:
            label[width - factor.index - 1] = factor.pauli.value
        rows.append(("".join(label), coefficient))
    if not rows:
        rows.append(("I" * width, 0j))
    operator = SparsePauliOp.from_list(rows)
    time = _emit_value(raw_time, parameters)
    gate = PauliEvolutionGate(operator, time=time)
    if instruction.controls:
        gate = gate.control(len(instruction.controls), annotated=True)
        _append_controlled_gate(
            circuit, gate, (*instruction.controls, *instruction.qubits)
        )
        return
    circuit.append(gate, instruction.qubits)
    metadata = dict(circuit.metadata or {})
    named_calls = metadata.setdefault("qamomile_named_calls", [])
    if not isinstance(named_calls, list):
        raise CapabilityError("Q8235: Pauli evolution metadata is malformed")
    if gate.name not in named_calls:
        named_calls.append(gate.name)
    circuit.metadata = metadata


def _remap_block(block: CircuitBlock, wires: dict[int, int]) -> CircuitBlock:
    """Project global wire IDs onto a composite's ordered local operands.

    Args:
        block (CircuitBlock): Verified coherent body.
        wires (dict[int, int]): Complete global-to-local wire mapping.

    Returns:
        CircuitBlock: Equivalent body with local wire and control indices.
    """
    return replace(
        block,
        instructions=tuple(
            replace(
                item,
                qubits=tuple(wires[index] for index in item.qubits),
                controls=tuple(wires[index] for index in item.controls),
                regions=tuple(_remap_block(region, wires) for region in item.regions),
            )
            for item in block.instructions
        ),
    )


def _emit_named_box(
    circuit: Any,
    instruction: CircuitInstruction,
    parameters: dict[ParameterValue | LoopIndex, Any],
) -> None:
    """Emit a verified body on exactly its operands, with lazy SDK modifiers.

    Args:
        circuit (Any): Destination Qiskit circuit.
        instruction (CircuitInstruction): Coherent region with ordered operands.
        parameters (dict[ParameterValue | LoopIndex, Any]): SDK scalar parameter bindings.

    Raises:
        CapabilityError: If the region is malformed or cannot become a gate.
    """
    if instruction.parameters and isinstance(
        instruction.parameters[0], DeferredDefinition
    ):
        _emit_deferred_definition(circuit, instruction, parameters)
        return
    if (
        instruction.condition is not None
        or instruction.parameters
        or len(instruction.regions) != 1
    ):
        raise CapabilityError("Q8233: malformed coherent callable instruction")
    inverse = instruction.opcode.startswith("inverse:")
    label = instruction.opcode.removeprefix("inverse:").removeprefix("box:")
    if not label:
        raise CapabilityError("Q8233: named callable label is empty")
    from qiskit import QuantumCircuit

    body = _remap_block(
        instruction.regions[0],
        dict(zip(instruction.qubits, range(len(instruction.qubits)), strict=True)),
    )
    key = (label, body)
    gate: Any = (
        parameters.gates.get(key)
        if isinstance(parameters, _EmissionParameters)
        else None
    )
    if gate is None:
        definition = QuantumCircuit(
            len(instruction.qubits), name=f"qamomile_{uuid.uuid4().hex}"
        )
        _emit_block(definition, body, parameters)
        try:
            gate = definition.to_gate(label=label)
        except Exception as error:
            raise CapabilityError(
                "Q8233: callable body is not a coherent Qiskit gate"
            ) from error
        if isinstance(parameters, _EmissionParameters):
            parameters.gates[key] = gate
    if inverse:
        gate = gate.inverse(annotated=True)
    if instruction.controls:
        gate = gate.control(len(instruction.controls), annotated=True)
    circuit.append(gate, (*instruction.controls, *instruction.qubits))
    metadata = dict(circuit.metadata or {})
    named_calls = metadata.setdefault("qamomile_named_calls", [])
    if not isinstance(named_calls, list):
        raise CapabilityError("Q8233: named callable metadata is malformed")
    if label not in named_calls:
        named_calls.append(label)
    circuit.metadata = metadata


def _emit_deferred_definition(
    circuit: Any,
    instruction: CircuitInstruction,
    parameters: dict[ParameterValue | LoopIndex, Any],
) -> None:
    """Attach a shared compiler definition without requesting its gate body.

    Args:
        circuit (Any): Destination SDK circuit.
        instruction (CircuitInstruction): Checked local definition and operands.
        parameters (dict[ParameterValue | LoopIndex, Any]): Current scalar scope.
    """
    from ._deferred import DeferredGate

    definition = cast(DeferredDefinition, instruction.parameters[0])
    scope = {
        value: parameters[value]
        for value in definition.scalars
        if isinstance(value, (ParameterValue, LoopIndex))
    }
    label = instruction.opcode.removeprefix("inverse:").removeprefix("box:")
    key = (definition.identity, tuple(scope.items()))
    gates = parameters.gates if isinstance(parameters, _EmissionParameters) else {}
    gate: Any = gates.get(key)
    if gate is None:
        gate = DeferredGate(label, definition, scope)
        gates[key] = gate
    if instruction.opcode.startswith("inverse:"):
        gate = gate.inverse(annotated=True)
    if instruction.controls:
        gate = gate.control(len(instruction.controls), annotated=True)
    circuit.append(gate, (*instruction.controls, *instruction.qubits))
    if label != "deferred":
        metadata = dict(circuit.metadata or {})
        labels = metadata.setdefault("qamomile_named_calls", [])
        if label not in labels:
            labels.append(label)
        circuit.metadata = metadata


def _emit_for(
    circuit: Any,
    instruction: CircuitInstruction,
    parameters: dict[ParameterValue | LoopIndex, Any],
) -> None:
    """Emit a structured loop with an optional scoped SDK index parameter.

    Args:
        circuit (Any): Qiskit circuit receiving the loop.
        instruction (CircuitInstruction): Verified target-neutral loop instruction.
        parameters (dict[ParameterValue | LoopIndex, Any]): Scoped SDK bindings.

    Raises:
        CapabilityError: If the native-loop projection is malformed.
    """

    if (
        instruction.controls
        or instruction.condition is not None
        or len(instruction.parameters) not in {3, 4}
        or len(instruction.regions) != 1
    ):
        raise CapabilityError("Q8234: malformed native for instruction")
    bounds = tuple(
        _emit_value(value, parameters) for value in instruction.parameters[:3]
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in bounds):
        raise CapabilityError("Q8234: native for bounds must be integers")
    lower, upper, stride = bounds
    if stride == 0:
        raise CapabilityError("Q8234: native for stride must be nonzero")
    with circuit.for_loop(range(lower, upper, stride)) as index:
        scoped = _EmissionParameters(
            parameters,
            parameters.gates if isinstance(parameters, _EmissionParameters) else None,
        )
        if len(instruction.parameters) == 4:
            logical_index = instruction.parameters[3]
            if not isinstance(logical_index, LoopIndex):
                raise CapabilityError("Q8234: native for index must be scoped")
            scoped[logical_index] = index
        _emit_block(circuit, instruction.regions[0], scoped)


def _emit_controlled_instruction(
    circuit: Any,
    instruction: CircuitInstruction,
    parameters: dict[ParameterValue | LoopIndex, Any],
) -> None:
    """Emit an exact outer control without eagerly synthesizing its definition.

    Args:
        circuit (Any): Destination SDK circuit.
        instruction (CircuitInstruction): Unitary gate with positive controls.
        parameters (dict[ParameterValue | LoopIndex, Any]): Scoped SDK bindings.

    Raises:
        CapabilityError: If the opcode has no controlled SDK representation.
    """

    from qiskit.circuit.library import (
        CCXGate,
        CPhaseGate,
        CXGate,
        CZGate,
        HGate,
        PhaseGate,
        RXGate,
        RYGate,
        RZGate,
        RZZGate,
        SdgGate,
        SGate,
        SwapGate,
        TdgGate,
        TGate,
        XGate,
        YGate,
        ZGate,
    )

    op = instruction.opcode
    values = tuple(_emit_value(item, parameters) for item in instruction.parameters)
    if op == "global_phase":
        angle = values[0]
        controls = instruction.controls
        gate = PhaseGate(angle)
        if len(controls) == 1:
            circuit.append(gate, controls)
        else:
            _append_controlled_gate(
                circuit,
                gate.control(len(controls) - 1, annotated=True),
                controls,
            )
        return
    factories: dict[str, Any] = {
        "h": HGate,
        "x": XGate,
        "y": YGate,
        "z": ZGate,
        "s": SGate,
        "sdg": SdgGate,
        "t": TGate,
        "tdg": TdgGate,
        "p": lambda: PhaseGate(values[0]),
        "rx": lambda: RXGate(values[0]),
        "ry": lambda: RYGate(values[0]),
        "rz": lambda: RZGate(values[0]),
        "cx": CXGate,
        "cz": CZGate,
        "ccx": CCXGate,
        "cp": lambda: CPhaseGate(values[0]),
        "rzz": lambda: RZZGate(values[0]),
        "swap": SwapGate,
    }
    factory = factories.get(op)
    if factory is None:
        raise CapabilityError(f"Q8213: Qiskit cannot coherently control {op!r}")
    gate = factory()
    _append_controlled_gate(
        circuit,
        gate.control(len(instruction.controls), annotated=True),
        (*instruction.controls, *instruction.qubits),
    )


def _append_controlled_gate(circuit: Any, gate: Any, qubits: tuple[int, ...]) -> None:
    """Retain the SDK gate until an endpoint selects its supported basis.

    Args:
        circuit (Any): Destination SDK circuit.
        gate (Any): Exact Qiskit controlled gate, including its definition.
        qubits (tuple[int, ...]): Ordered control and target operands.
    """
    circuit.append(gate, qubits)


def _emit_condition(
    circuit: Any,
    value: CircuitValue,
    parameters: dict[ParameterValue | LoopIndex, Any],
) -> Any:
    """Lower a Boolean measurement expression to Qiskit's classical AST.

    Args:
        circuit (Any): SDK circuit owning the measurement bits.
        value (CircuitValue): Verified Boolean condition.
        parameters (dict[ParameterValue | LoopIndex, Any]): Scoped SDK bindings.

    Returns:
        Any: Exact SDK Boolean expression, including XOR and Boolean selection.

    Raises:
        CapabilityError: If the expression needs unsupported device arithmetic.
    """

    from qiskit.circuit.classical import expr

    if isinstance(value, MeasurementBit):
        return expr.lift(circuit.clbits[value.index])
    if isinstance(value, bool):
        return expr.lift(value)
    if isinstance(value, ClassicalExpr):
        operands = tuple(
            _emit_condition(circuit, item, parameters) for item in value.operands
        )
        opcode = value.opcode.removeprefix("classical.")
        if opcode == "cast" and dict(value.attributes.entries).get("target") == "Bool":
            return operands[0]
        if opcode == "not":
            return expr.logic_not(operands[0])
        if opcode == "and":
            result = operands[0]
            for operand in operands[1:]:
                result = expr.logic_and(result, operand)
            return result
        if opcode == "or":
            result = operands[0]
            for operand in operands[1:]:
                result = expr.logic_or(result, operand)
            return result
        if opcode == "xor":
            return expr.bit_xor(operands[0], operands[1])
        if opcode == "select":
            return expr.logic_or(
                expr.logic_and(operands[0], operands[1]),
                expr.logic_and(expr.logic_not(operands[0]), operands[2]),
            )
        if opcode == "compare_chain":
            names = cast(
                tuple[str, ...],
                dict(value.attributes.entries).get("operators", ()),
            )
            constructors = {
                "eq": expr.equal,
                "ne": expr.not_equal,
                "lt": expr.less,
                "le": expr.less_equal,
                "gt": expr.greater,
                "ge": expr.greater_equal,
            }
            comparisons = tuple(
                constructors[name](left, right)
                for name, left, right in zip(
                    names, operands[:-1], operands[1:], strict=True
                )
            )
            result = comparisons[0]
            for comparison in comparisons[1:]:
                result = expr.logic_and(result, comparison)
            return result
    raise CapabilityError(f"Q8212: unsupported Qiskit condition {value!r}")


def _emit_value(
    value: CircuitValue, parameters: dict[ParameterValue | LoopIndex, Any]
) -> Any:
    """Lower a circuit-family scalar to a Qiskit parameter expression.

    Args:
        value (CircuitValue): Verified scalar expression to emit.
        parameters (dict[ParameterValue | LoopIndex, Any]): Qiskit parameters keyed by
            their target-neutral ABI leaves.

    Returns:
        Any: A Qiskit parameter, parameter expression, or scalar literal.

    Raises:
        CapabilityError: If the circuit projection contains an unsupported
            scalar expression.
    """

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (ParameterValue, LoopIndex)):
        return parameters[value]
    if isinstance(value, ClassicalExpr):
        operands = tuple(_emit_value(item, parameters) for item in value.operands)
        op = value.opcode.removeprefix("classical.")
        if op == "neg":
            return -operands[0]
        if op == "add":
            return operands[0] + operands[1]
        if op == "sub":
            return operands[0] - operands[1]
        if op == "mul":
            # Qiskit 2.3.1 cannot read its own QPY after identity arithmetic
            # produces a ParameterExpression with no actual operation left.
            if type(operands[0]) in (int, float) and operands[0] == 1:
                return operands[1]
            if type(operands[1]) in (int, float) and operands[1] == 1:
                return operands[0]
            return operands[0] * operands[1]
        if op in {"div", "truediv"}:
            if type(operands[1]) in (int, float) and operands[1] == 1:
                return operands[0]
            return operands[0] / operands[1]
        if op == "pow":
            return operands[0] ** operands[1]
    raise CapabilityError(f"Q8210: unsupported Qiskit scalar expression {value!r}")
