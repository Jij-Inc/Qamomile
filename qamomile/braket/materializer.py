"""Materialize backend-neutral circuit programs as Amazon Braket circuits."""

from __future__ import annotations

import dataclasses
import re
from typing import Any, cast

from qamomile.circuit.transpiler.circuit_ir import (
    ALL_PRIMITIVE_GATES,
    BarrierInstruction,
    BinaryExpr,
    BinaryOperator,
    CallControlMode,
    CallInstruction,
    CallPhaseMode,
    CallTransformCapabilities,
    CircuitCapabilities,
    CircuitInstruction,
    CircuitProgram,
    ClassicalBitExpr,
    ForInstruction,
    GateInstruction,
    IfInstruction,
    LiteralExpr,
    LoopVariableExpr,
    MaterializedCircuit,
    MeasureInstruction,
    MeasureVectorInstruction,
    ParameterExpr,
    PauliEvolutionInstruction,
    PauliEvolutionRealization,
    ResetInstruction,
    ScalarAtom,
    ScalarCapabilities,
    ScalarExpr,
    ScalarExpressionForm,
    UnaryExpr,
    UnaryOperator,
    WhileInstruction,
    WireId,
    has_mid_circuit_measurement,
    verify_circuit,
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.gate_emitter import GateKind
from qamomile.observable.hamiltonian import PAULI_TERM_ZERO_ATOL

_OPENQASM_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENCODED_PARAMETER_PREFIX = "_qamomile_parameter_"


class BraketMaterializer:
    """Convert verified circuit IR to an Amazon Braket circuit."""

    @property
    def capabilities(self) -> CircuitCapabilities:
        """Declare the Amazon Braket circuit capabilities.

        Returns:
            CircuitCapabilities: Immutable Braket target declaration.
        """
        numeric = ScalarCapabilities(
            atoms=frozenset(
                {ScalarAtom.LITERAL, ScalarAtom.PARAMETER, ScalarAtom.LOOP_VARIABLE}
            ),
            unary_operators=frozenset({UnaryOperator.NEG}),
            binary_operators=frozenset(
                {
                    BinaryOperator.ADD,
                    BinaryOperator.SUB,
                    BinaryOperator.MUL,
                    BinaryOperator.DIV,
                    BinaryOperator.POW,
                }
            ),
            parameter_form=ScalarExpressionForm.ARBITRARY,
        )
        return CircuitCapabilities(
            name="braket",
            primitive_gates=ALL_PRIMITIVE_GATES,
            native_semantic_ops=(),
            gate_parameters=numeric,
            predicates=ScalarCapabilities(
                atoms=frozenset(),
                unary_operators=frozenset(),
                binary_operators=frozenset(),
                parameter_form=ScalarExpressionForm.CONCRETE_ONLY,
            ),
            pauli_time=numeric,
            global_phase=numeric,
            generic_calls=CallTransformCapabilities(
                supports_power=True,
                supports_inverse=True,
                max_controls=None,
                supports_barrier_body=True,
                control_mode=CallControlMode.DISTRIBUTE,
                controlled_gate_kinds=ALL_PRIMITIVE_GATES,
                controlled_pauli_time=numeric,
                phase_mode=CallPhaseMode.NATIVE_BODY,
                controlled_phase_scalars=numeric,
            ),
            supports_dynamic_if=False,
            supports_dynamic_while=False,
            supports_reset=False,
            pauli_realizations=frozenset({PauliEvolutionRealization.GADGET}),
        )

    def materialize(
        self,
        program: CircuitProgram,
        parameter_names: tuple[str, ...] = (),
    ) -> MaterializedCircuit[Any]:
        """Build a Braket circuit and static-measurement metadata.

        Args:
            program (CircuitProgram): Verified target-legal circuit program.
            parameter_names (tuple[str, ...]): Public parameter ABI names.
                Defaults to an empty tuple.

        Returns:
            MaterializedCircuit[Any]: Braket circuit and binding metadata.

        Raises:
            EmitError: If runtime control, reset, or mid-circuit measurement
                remains in the program.
            ValueError: If structural circuit verification fails.
        """
        from braket.circuits import (  # type: ignore[import-not-found]
            Circuit,
            FreeParameter,
            Gate,
            Instruction,
        )

        gate_api = cast(Any, Gate)

        verify_circuit(program)
        if has_mid_circuit_measurement(program.operations):
            raise EmitError(
                "Amazon Braket static sampling cannot represent a mid-circuit "
                "measurement whose qubit is used again"
            )
        circuit = Circuit()
        wires = {wire: index for index, wire in enumerate(program.input_wires)}
        parameters: dict[str, Any] = {}
        for name in parameter_names:
            _create_parameter(name, parameters, FreeParameter)
        measurements: dict[int, int] = {}
        phase_ancillas = tuple(
            range(
                program.num_qubits,
                program.num_qubits
                + _phase_ancilla_demand(program.operations, inherited_controls=0),
            )
        )
        phase = _materialize_scalar(program.global_phase, parameters, {})
        if not _is_zero(program.global_phase):
            circuit.add_instruction(Instruction(gate_api.GPhase(phase), target=[]))
        _emit_region(
            program.operations,
            circuit,
            wires,
            parameters,
            measurements,
            {},
            (),
            phase_ancillas,
            False,
        )
        present = {int(qubit) for qubit in circuit.qubits}
        for index in range(program.num_qubits):
            if index not in present:
                circuit.add_instruction(Instruction(gate_api.I(), index))
        return MaterializedCircuit(
            artifact=circuit,
            parameters=parameters,
            measurement_qubit_map=measurements,
            implicit_output_qubit_indices=tuple(range(program.num_qubits)),
        )


def _phase_ancilla_demand(
    operations: tuple[CircuitInstruction, ...],
    inherited_controls: int,
) -> int:
    """Compute the maximum clean-work-qubit demand for controlled phases.

    Args:
        operations (tuple[CircuitInstruction, ...]): Operations to inspect.
        inherited_controls (int): Number of controls inherited by this region.

    Returns:
        int: Maximum simultaneously required clean ancilla count.
    """
    demand = 0
    for operation in operations:
        if isinstance(operation, CallInstruction):
            controls = inherited_controls + operation.callee.controls
            if not _is_zero(operation.callee.body.global_phase):
                demand = max(demand, controls - 2)
            demand = max(
                demand,
                _phase_ancilla_demand(operation.callee.body.operations, controls),
            )
        elif isinstance(operation, ForInstruction):
            demand = max(
                demand,
                _phase_ancilla_demand(operation.body, inherited_controls),
            )
    return demand


def _emit_region(
    operations: tuple[CircuitInstruction, ...],
    circuit: Any,
    input_wires: dict[WireId, int],
    parameters: dict[str, Any],
    measurements: dict[int, int],
    loop_variables: dict[str, int],
    inherited_controls: tuple[int, ...],
    phase_ancillas: tuple[int, ...],
    inverse: bool,
) -> dict[WireId, int]:
    """Emit one circuit region with inherited call transforms.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region instructions.
        circuit (Any): Destination Braket circuit.
        input_wires (dict[WireId, int]): Region input wire mapping.
        parameters (dict[str, Any]): Braket parameter cache.
        measurements (dict[int, int]): Static measurement mapping to update.
        loop_variables (dict[str, int]): Concrete loop induction values.
        inherited_controls (tuple[int, ...]): Coherent outer controls.
        phase_ancillas (tuple[int, ...]): Clean work qubits for controlled
            global phases.
        inverse (bool): Whether to reverse and invert the region.

    Returns:
        dict[WireId, int]: Mapping containing produced virtual wires.

    Raises:
        EmitError: If a non-unitary instruction occurs under a transform or
            an unsupported instruction reaches materialization.
    """
    wires = dict(input_wires)
    sequence = reversed(operations) if inverse else operations
    for operation in sequence:
        if isinstance(operation, GateInstruction):
            inputs = operation.outputs if inverse else operation.inputs
            outputs = operation.inputs if inverse else operation.outputs
            qubits = tuple(wires[wire] for wire in inputs)
            angles = tuple(
                _materialize_scalar(value, parameters, loop_variables)
                for value in operation.parameters
            )
            _emit_gate(
                circuit,
                operation.kind,
                qubits,
                angles,
                inherited_controls,
                -1 if inverse else 1,
            )
            _publish(outputs, qubits, wires)
        elif isinstance(operation, MeasureInstruction):
            if inverse or inherited_controls:
                raise EmitError("Cannot transform a measurement instruction")
            slot = wires[operation.input]
            measurements[operation.clbit] = slot
            wires[operation.output] = slot
        elif isinstance(operation, MeasureVectorInstruction):
            if inverse or inherited_controls:
                raise EmitError("Cannot transform a measurement instruction")
            slots = tuple(wires[wire] for wire in operation.inputs)
            measurements.update(zip(operation.clbits, slots, strict=True))
            _publish(operation.outputs, slots, wires)
        elif isinstance(operation, BarrierInstruction):
            if not inherited_controls:
                circuit.barrier([wires[wire] for wire in operation.wires])
        elif isinstance(operation, ResetInstruction):
            raise EmitError(
                "Amazon Braket circuit materialization does not support reset"
            )
        elif isinstance(operation, PauliEvolutionInstruction):
            transformed = operation
            if inverse:
                transformed = dataclasses.replace(
                    operation,
                    time=UnaryExpr(UnaryOperator.NEG, operation.time),
                )
            _emit_pauli_evolution(
                transformed,
                circuit,
                wires,
                parameters,
                loop_variables,
                inherited_controls,
            )
        elif isinstance(operation, ForInstruction):
            inputs = operation.outputs if inverse else operation.inputs
            outputs = operation.inputs if inverse else operation.outputs
            body_inputs = operation.body_outputs if inverse else operation.inputs
            body_outputs = operation.inputs if inverse else operation.body_outputs
            current = [wires[wire] for wire in inputs]
            indices = list(operation.indexset)
            if inverse:
                indices.reverse()
            for index in indices:
                nested_variables = dict(loop_variables)
                nested_variables[operation.loop_variable.name] = index
                nested_wires = _emit_region(
                    operation.body,
                    circuit,
                    dict(zip(body_inputs, current, strict=True)),
                    parameters,
                    measurements,
                    nested_variables,
                    inherited_controls,
                    phase_ancillas,
                    inverse,
                )
                current = [nested_wires[wire] for wire in body_outputs]
            _publish(outputs, current, wires)
        elif isinstance(operation, CallInstruction):
            _emit_call(
                operation,
                circuit,
                wires,
                parameters,
                measurements,
                loop_variables,
                inherited_controls,
                phase_ancillas,
                inverse,
            )
        elif isinstance(operation, (IfInstruction, WhileInstruction)):
            raise EmitError(
                "Amazon Braket does not support generic runtime control flow"
            )
        else:  # pragma: no cover - defensive closed-union guard
            raise EmitError(f"Unsupported Braket instruction: {operation!r}")
    return wires


def _emit_call(
    operation: CallInstruction,
    circuit: Any,
    wires: dict[WireId, int],
    parameters: dict[str, Any],
    measurements: dict[int, int],
    loop_variables: dict[str, int],
    inherited_controls: tuple[int, ...],
    phase_ancillas: tuple[int, ...],
    inherited_inverse: bool,
) -> None:
    """Inline a reusable call while composing controls and inversion.

    Args:
        operation (CallInstruction): Reusable circuit call.
        circuit (Any): Destination Braket circuit.
        wires (dict[WireId, int]): Enclosing wire mapping.
        parameters (dict[str, Any]): Braket parameter cache.
        measurements (dict[int, int]): Static measurement mapping.
        loop_variables (dict[str, int]): Concrete loop values.
        inherited_controls (tuple[int, ...]): Controls from enclosing calls.
        phase_ancillas (tuple[int, ...]): Clean work qubits for controlled
            global phases.
        inherited_inverse (bool): Whether the enclosing region is inverted.
    """
    from braket.circuits import Gate, Instruction  # type: ignore[import-not-found]

    gate_api = cast(Any, Gate)

    callee = operation.callee
    call_inputs = operation.outputs if inherited_inverse else operation.inputs
    call_outputs = operation.inputs if inherited_inverse else operation.outputs
    actual = [wires[wire] for wire in call_inputs]
    own_controls = tuple(actual[: callee.controls])
    targets = actual[callee.controls :]
    controls = (*inherited_controls, *own_controls)
    inverse = inherited_inverse ^ callee.inverse
    phase = _materialize_scalar(callee.body.global_phase, parameters, loop_variables)
    phase_power = -1 if inverse else 1
    for _ in range(callee.power):
        body_inputs = callee.body.output_wires if inverse else callee.body.input_wires
        body_outputs = callee.body.input_wires if inverse else callee.body.output_wires
        body_wires = _emit_region(
            callee.body.operations,
            circuit,
            dict(zip(body_inputs, targets, strict=True)),
            parameters,
            measurements,
            loop_variables,
            controls,
            phase_ancillas,
            inverse,
        )
        targets = [body_wires[wire] for wire in body_outputs]
        if not _is_zero(callee.body.global_phase):
            _emit_global_phase(
                circuit,
                phase * phase_power,
                controls,
                phase_ancillas,
                gate_api,
                Instruction,
            )
    _publish(call_outputs, [*own_controls, *targets], wires)


def _emit_global_phase(
    circuit: Any,
    phase: Any,
    controls: tuple[int, ...],
    ancillas: tuple[int, ...],
    gate_api: Any,
    instruction_type: Any,
) -> None:
    """Emit an unconditional or coherently controlled global phase.

    Args:
        circuit (Any): Destination Braket circuit.
        phase (Any): Numeric or symbolic phase angle.
        controls (tuple[int, ...]): Coherent phase controls.
        ancillas (tuple[int, ...]): Clean work qubits reserved for conjunctions.
        gate_api (Any): Dynamically registered Braket gate namespace.
        instruction_type (Any): Braket ``Instruction`` constructor.

    Raises:
        EmitError: If insufficient work qubits were reserved.
    """
    if not controls:
        instruction = instruction_type(gate_api.GPhase(phase), target=[])
    elif len(controls) == 1:
        instruction = instruction_type(gate_api.PhaseShift(phase), controls[0])
    elif len(controls) == 2:
        instruction = instruction_type(gate_api.CPhaseShift(phase), controls)
    else:
        required = len(controls) - 2
        if len(ancillas) < required:
            raise EmitError(
                "Insufficient Braket work qubits for a controlled global phase"
            )
        control_list = list(controls)
        used = list(ancillas[:required])
        circuit.add_instruction(
            instruction_type(
                gate_api.CCNot(), [control_list[0], control_list[1], used[0]]
            )
        )
        for index in range(2, len(control_list) - 1):
            circuit.add_instruction(
                instruction_type(
                    gate_api.CCNot(),
                    [used[index - 2], control_list[index], used[index - 1]],
                )
            )
        circuit.add_instruction(
            instruction_type(
                gate_api.CPhaseShift(phase),
                [used[-1], control_list[-1]],
            )
        )
        for index in range(len(control_list) - 2, 1, -1):
            circuit.add_instruction(
                instruction_type(
                    gate_api.CCNot(),
                    [used[index - 2], control_list[index], used[index - 1]],
                )
            )
        circuit.add_instruction(
            instruction_type(
                gate_api.CCNot(), [control_list[0], control_list[1], used[0]]
            )
        )
        return
    circuit.add_instruction(instruction)


def _emit_pauli_evolution(
    operation: PauliEvolutionInstruction,
    circuit: Any,
    wires: dict[WireId, int],
    parameters: dict[str, Any],
    loop_variables: dict[str, int],
    controls: tuple[int, ...],
) -> None:
    """Emit Pauli evolution as controlled primitive phase gadgets.

    Args:
        operation (PauliEvolutionInstruction): Legalized Pauli evolution.
        circuit (Any): Destination Braket circuit.
        wires (dict[WireId, int]): Virtual wire mapping.
        parameters (dict[str, Any]): Braket parameter cache.
        loop_variables (dict[str, int]): Concrete loop values.
        controls (tuple[int, ...]): Coherent call controls.

    Raises:
        EmitError: If a non-gadget realization reaches this target.
    """
    import qamomile.observable as qm_o
    from braket.circuits import Gate, Instruction  # type: ignore[import-not-found]

    gate_api = cast(Any, Gate)

    if operation.realization is not PauliEvolutionRealization.GADGET:
        raise EmitError("Amazon Braket Pauli evolution must be gadget-realized")
    time = _materialize_scalar(operation.time, parameters, loop_variables)
    slots = tuple(wires[wire] for wire in operation.inputs)
    for operators, coefficient in operation.hamiltonian:
        if not operators or abs(coefficient) <= PAULI_TERM_ZERO_ATOL:
            continue
        selected = [slots[item.index] for item in operators]
        for item, slot in zip(operators, selected, strict=True):
            if item.pauli is qm_o.Pauli.X:
                circuit.add_instruction(Instruction(gate_api.H(), slot))
            elif item.pauli is qm_o.Pauli.Y:
                circuit.add_instruction(Instruction(gate_api.Si(), slot))
                circuit.add_instruction(Instruction(gate_api.H(), slot))
        for left, right in zip(selected, selected[1:]):
            circuit.add_instruction(Instruction(gate_api.CNot(), [left, right]))
        angle = 2.0 * float(complex(coefficient).real) * time
        circuit.add_instruction(
            Instruction(gate_api.Rz(angle), selected[-1], control=controls)
        )
        for left, right in reversed(list(zip(selected, selected[1:]))):
            circuit.add_instruction(Instruction(gate_api.CNot(), [left, right]))
        for item, slot in reversed(list(zip(operators, selected, strict=True))):
            if item.pauli is qm_o.Pauli.X:
                circuit.add_instruction(Instruction(gate_api.H(), slot))
            elif item.pauli is qm_o.Pauli.Y:
                circuit.add_instruction(Instruction(gate_api.H(), slot))
                circuit.add_instruction(Instruction(gate_api.S(), slot))
    _publish(operation.outputs, slots, wires)


def _materialize_scalar(
    expression: ScalarExpr,
    parameters: dict[str, Any],
    loop_variables: dict[str, int],
) -> Any:
    """Convert a circuit scalar to a Braket numeric expression.

    Args:
        expression (ScalarExpr): Target-neutral scalar expression.
        parameters (dict[str, Any]): Braket parameter cache.
        loop_variables (dict[str, int]): Concrete induction values.

    Returns:
        Any: Numeric value or Braket free-parameter expression.

    Raises:
        EmitError: If the expression contains an unsupported operator or bit.
    """
    from braket.circuits import FreeParameter  # type: ignore[import-not-found]

    if isinstance(expression, LiteralExpr):
        return expression.value
    if isinstance(expression, ParameterExpr):
        return _create_parameter(expression.name, parameters, FreeParameter)
    if isinstance(expression, LoopVariableExpr):
        try:
            return loop_variables[expression.name]
        except KeyError as error:
            raise EmitError(
                f"Unresolved Braket loop variable {expression.name!r}"
            ) from error
    if isinstance(expression, ClassicalBitExpr):
        raise EmitError("Braket gate parameters cannot depend on measurement bits")
    if isinstance(expression, UnaryExpr):
        if expression.operator is UnaryOperator.NEG:
            return -_materialize_scalar(expression.operand, parameters, loop_variables)
        raise EmitError(f"Unsupported Braket unary operator {expression.operator.name}")
    if isinstance(expression, BinaryExpr):
        left = _materialize_scalar(expression.left, parameters, loop_variables)
        right = _materialize_scalar(expression.right, parameters, loop_variables)
        if expression.operator is BinaryOperator.ADD:
            return left + right
        if expression.operator is BinaryOperator.SUB:
            return left - right
        if expression.operator is BinaryOperator.MUL:
            return left * right
        if expression.operator is BinaryOperator.DIV:
            return left / right
        if expression.operator is BinaryOperator.POW:
            return left**right
        raise EmitError(
            f"Unsupported Braket binary operator {expression.operator.name}"
        )
    raise EmitError(f"Unsupported Braket scalar expression {expression!r}")


def _create_parameter(
    name: str,
    parameters: dict[str, Any],
    parameter_type: Any,
) -> Any:
    """Create a Braket parameter while preserving Qamomile's public name.

    Braket serializes free parameters as OpenQASM identifiers. Indexed
    Qamomile names such as ``angles[0]`` would instead be parsed as array
    declarations, and OpenQASM keywords such as ``angle`` are rejected. The
    mapping remains keyed by the original Qamomile name while unsafe backend
    spellings use an injective UTF-8 encoding.

    Args:
        name (str): Public Qamomile parameter name.
        parameters (dict[str, Any]): Parameter mapping to read and update.
        parameter_type (Any): Braket ``FreeParameter`` constructor.

    Returns:
        Any: Existing or newly created Braket free parameter.

    Raises:
        ValueError: If Braket rejects the generated safe name.
    """
    if name in parameters:
        return parameters[name]
    backend_name = name
    if not _OPENQASM_IDENTIFIER.fullmatch(name) or name.startswith(
        _ENCODED_PARAMETER_PREFIX
    ):
        backend_name = _encode_parameter_name(name)
    try:
        parameter = parameter_type(backend_name)
    except ValueError:
        parameter = parameter_type(_encode_parameter_name(name))
    parameters[name] = parameter
    return parameter


def _encode_parameter_name(name: str) -> str:
    """Encode an arbitrary public name as an injective OpenQASM identifier.

    Args:
        name (str): Qamomile parameter name.

    Returns:
        str: Backend-safe identifier that cannot collide with unencoded names.
    """
    return f"{_ENCODED_PARAMETER_PREFIX}{name.encode('utf-8').hex()}"


def _emit_gate(
    circuit: Any,
    kind: GateKind,
    qubits: tuple[int, ...],
    angles: tuple[Any, ...],
    controls: tuple[int, ...],
    power: int,
) -> None:
    """Emit one primitive Braket gate with optional outer controls.

    Args:
        circuit (Any): Destination Braket circuit.
        kind (GateKind): Primitive gate kind.
        qubits (tuple[int, ...]): Physical gate operands.
        angles (tuple[Any, ...]): Materialized gate parameters.
        controls (tuple[int, ...]): Additional coherent controls.
        power (int): Instruction power, either one or negative one.

    Raises:
        EmitError: If the primitive gate shape is unsupported.
    """
    from braket.circuits import Gate, Instruction  # type: ignore[import-not-found]

    gate_api = cast(Any, Gate)

    fixed = {
        GateKind.H: gate_api.H,
        GateKind.X: gate_api.X,
        GateKind.Y: gate_api.Y,
        GateKind.Z: gate_api.Z,
        GateKind.S: gate_api.S,
        GateKind.SDG: gate_api.Si,
        GateKind.T: gate_api.T,
        GateKind.TDG: gate_api.Ti,
        GateKind.CX: gate_api.CNot,
        GateKind.CZ: gate_api.CZ,
        GateKind.SWAP: gate_api.Swap,
        GateKind.RZZ: gate_api.ZZ,
        GateKind.TOFFOLI: gate_api.CCNot,
    }
    rotations = {
        GateKind.RX: gate_api.Rx,
        GateKind.RY: gate_api.Ry,
        GateKind.RZ: gate_api.Rz,
        GateKind.P: gate_api.PhaseShift,
        GateKind.CP: gate_api.CPhaseShift,
    }
    own_controlled = {
        GateKind.CH: gate_api.H,
    }
    if kind in {GateKind.CRX, GateKind.CRY, GateKind.CRZ} and not controls:
        controls = (qubits[0],)
        qubits = (qubits[1],)
        kind = {
            GateKind.CRX: GateKind.RX,
            GateKind.CRY: GateKind.RY,
            GateKind.CRZ: GateKind.RZ,
        }[kind]
    if kind in {GateKind.RX, GateKind.RY, GateKind.RZ} and len(controls) == 1:
        _emit_single_controlled_rotation(
            circuit,
            kind,
            controls[0],
            qubits[0],
            angles[0] * power,
            gate_api,
            Instruction,
        )
        return
    if kind is GateKind.X and len(qubits) == 1 and len(controls) == 1:
        instruction = Instruction(
            gate_api.CNot(),
            [controls[0], qubits[0]],
            power=power,
        )
    elif kind is GateKind.X and len(qubits) == 1 and len(controls) == 2:
        instruction = Instruction(
            gate_api.CCNot(),
            [*controls, qubits[0]],
            power=power,
        )
    elif kind is GateKind.CX and len(qubits) == 2 and len(controls) == 1:
        instruction = Instruction(
            gate_api.CCNot(),
            [controls[0], *qubits],
            power=power,
        )
    elif kind is GateKind.Y and len(qubits) == 1 and len(controls) == 1:
        instruction = Instruction(
            gate_api.CY(),
            [controls[0], qubits[0]],
            power=power,
        )
    elif kind is GateKind.CY and len(qubits) == 2 and not controls:
        instruction = Instruction(gate_api.CY(), qubits, power=power)
    elif kind is GateKind.Z and len(qubits) == 1 and len(controls) == 1:
        instruction = Instruction(
            gate_api.CZ(),
            [controls[0], qubits[0]],
            power=power,
        )
    elif kind is GateKind.SWAP and len(qubits) == 2 and len(controls) == 1:
        instruction = Instruction(
            gate_api.CSwap(),
            [controls[0], *qubits],
            power=power,
        )
    elif kind is GateKind.P and len(controls) == 1 and len(qubits) == 1:
        instruction = Instruction(
            gate_api.CPhaseShift(angles[0]),
            [controls[0], qubits[0]],
            power=power,
        )
    elif kind in own_controlled:
        gate = own_controlled[kind](*angles)
        instruction = Instruction(
            gate,
            qubits[-1],
            control=(*controls, *qubits[:-1]),
            power=power,
        )
    elif kind in fixed:
        gate = fixed[kind](*angles)
        instruction = Instruction(gate, qubits, control=controls, power=power)
    elif kind in rotations and len(angles) == 1:
        gate = rotations[kind](angles[0])
        instruction = Instruction(gate, qubits, control=controls, power=power)
    else:
        raise EmitError(f"Unsupported Braket gate {kind.name}")
    circuit.add_instruction(instruction)


def _emit_single_controlled_rotation(
    circuit: Any,
    kind: GateKind,
    control: int,
    target: int,
    angle: Any,
    gate_api: Any,
    instruction_type: Any,
) -> None:
    """Emit a one-control rotation without Braket's generic modifier.

    The default simulator in Braket SDK 1.125 can mis-handle generic
    ``ctrl @`` rotation modifiers for some nonzero control indices. Native
    CNOT gates and the standard half-angle decomposition avoid that path.

    Args:
        circuit (Any): Destination Braket circuit.
        kind (GateKind): RX, RY, or RZ rotation kind.
        control (int): Physical control qubit.
        target (int): Physical target qubit.
        angle (Any): Effective rotation angle, including inversion.
        gate_api (Any): Dynamically registered Braket gate namespace.
        instruction_type (Any): Braket ``Instruction`` constructor.

    Raises:
        EmitError: If ``kind`` is not a supported rotation.
    """
    if kind is GateKind.RX:
        circuit.add_instruction(instruction_type(gate_api.H(), target))
        _emit_single_controlled_rotation(
            circuit,
            GateKind.RZ,
            control,
            target,
            angle,
            gate_api,
            instruction_type,
        )
        circuit.add_instruction(instruction_type(gate_api.H(), target))
        return
    rotation = {
        GateKind.RY: gate_api.Ry,
        GateKind.RZ: gate_api.Rz,
    }.get(kind)
    if rotation is None:
        raise EmitError(f"Unsupported controlled Braket rotation {kind.name}")
    half_angle = angle / 2
    circuit.add_instruction(instruction_type(rotation(half_angle), target))
    circuit.add_instruction(instruction_type(gate_api.CNot(), [control, target]))
    circuit.add_instruction(instruction_type(rotation(-half_angle), target))
    circuit.add_instruction(instruction_type(gate_api.CNot(), [control, target]))


def _publish(
    outputs: tuple[WireId, ...],
    slots: tuple[int, ...] | list[int],
    wires: dict[WireId, int],
) -> None:
    """Publish virtual output wires on unchanged physical slots.

    Args:
        outputs (tuple[WireId, ...]): Produced virtual wires.
        slots (tuple[int, ...] | list[int]): Corresponding physical slots.
        wires (dict[WireId, int]): Mapping to update.
    """
    for output, slot in zip(outputs, slots, strict=True):
        wires[output] = slot


def _is_zero(expression: ScalarExpr) -> bool:
    """Return whether an expression is a literal numeric zero.

    Args:
        expression (ScalarExpr): Expression to inspect.

    Returns:
        bool: True only for a zero-valued literal.
    """
    return isinstance(expression, LiteralExpr) and not float(expression.value)
