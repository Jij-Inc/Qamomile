"""Materialize circuit-family IR as QURI Parts circuits."""

from __future__ import annotations

import dataclasses
import math
import numbers
from typing import Any

from qamomile.circuit.ir.operation.arithmetic_operations import BinOpKind
from qamomile.circuit.transpiler.circuit_ir import (
    ALL_PRIMITIVE_GATES,
    ARITHMETIC_BINARY_OPERATORS,
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
    GlobalPhaseCapabilities,
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
    verify_circuit,
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.gate_emitter import GateKind
from qamomile.quri_parts.emitter import QuriPartsGateEmitter
from qamomile.quri_parts.exceptions import QamomileQuriPartsTranspileError

_BINARY_KINDS = {
    BinaryOperator.ADD: BinOpKind.ADD,
    BinaryOperator.SUB: BinOpKind.SUB,
    BinaryOperator.MUL: BinOpKind.MUL,
    BinaryOperator.DIV: BinOpKind.DIV,
    BinaryOperator.FLOORDIV: BinOpKind.FLOORDIV,
    BinaryOperator.MOD: BinOpKind.MOD,
    BinaryOperator.POW: BinOpKind.POW,
}


class QuriPartsMaterializer:
    """Convert verified circuit IR to QURI Parts' linear circuit model."""

    @property
    def capabilities(self) -> CircuitCapabilities:
        """Declare QURI Parts' circuit-IR capabilities.

        The load-bearing restriction is the scalar form: QURI Parts'
        ``LinearMappedUnboundParametricQuantumCircuit`` represents angles as
        linear combinations of parameters, so non-linear parameter
        expressions are illegal at the target boundary rather than deep in
        materialization.

        Returns:
            CircuitCapabilities: Immutable capability declaration.
        """
        numeric = ScalarCapabilities(
            atoms=frozenset(
                {ScalarAtom.LITERAL, ScalarAtom.PARAMETER, ScalarAtom.LOOP_VARIABLE}
            ),
            unary_operators=frozenset({UnaryOperator.NEG}),
            binary_operators=ARITHMETIC_BINARY_OPERATORS,
            parameter_form=ScalarExpressionForm.LINEAR,
        )
        controlled_numeric = ScalarCapabilities(
            atoms=frozenset({ScalarAtom.LITERAL, ScalarAtom.LOOP_VARIABLE}),
            unary_operators=frozenset({UnaryOperator.NEG}),
            binary_operators=ARITHMETIC_BINARY_OPERATORS,
            parameter_form=ScalarExpressionForm.CONCRETE_ONLY,
        )
        return CircuitCapabilities(
            name="quri_parts",
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
            global_phase=GlobalPhaseCapabilities(
                scalars=numeric,
            ),
            generic_calls=CallTransformCapabilities(
                supports_power=True,
                supports_inverse=True,
                max_controls=None,
                supports_barrier_body=True,
                control_mode=CallControlMode.DISTRIBUTE,
                controlled_gate_kinds=ALL_PRIMITIVE_GATES
                - {
                    GateKind.CH,
                    GateKind.CRX,
                    GateKind.CRY,
                    GateKind.CRZ,
                },
                controlled_pauli_time=controlled_numeric,
                phase_mode=CallPhaseMode.EXPLICIT_CORRECTION,
                controlled_phase_scalars=numeric,
            ),
            supports_dynamic_if=False,
            supports_dynamic_while=False,
            supports_reset=False,
            pauli_realizations=frozenset(
                {
                    PauliEvolutionRealization.NATIVE,
                    PauliEvolutionRealization.GADGET,
                }
            ),
        )

    def materialize(
        self,
        program: CircuitProgram,
        parameter_names: tuple[str, ...] = (),
    ) -> MaterializedCircuit[Any]:
        """Build a QURI Parts circuit and static-measurement metadata.

        Args:
            program (CircuitProgram): Verified circuit-family program.
            parameter_names (tuple[str, ...]): Public runtime-parameter ABI in
                positional order. Defaults to an empty tuple.

        Returns:
            MaterializedCircuit[Any]: QURI circuit, parameters, and static
                measurement-to-qubit mapping.

        Raises:
            EmitError: If runtime control, reset, or transformed calls remain.
            QamomileQuriPartsTranspileError: If a parameter expression is not
                linear.
        """
        verify_circuit(program)
        ancilla_count = _ancilla_demand(program.operations)
        needs_phase_carrier = _requires_phase_carrier(program)
        phase_carrier = (
            program.num_qubits + ancilla_count if needs_phase_carrier else None
        )
        emitter = QuriPartsGateEmitter(phase_carrier=phase_carrier)
        circuit = emitter.create_circuit(
            program.num_qubits + ancilla_count + int(needs_phase_carrier),
            0,
        )
        wires = {wire: index for index, wire in enumerate(program.input_wires)}
        parameters = {name: emitter.create_parameter(name) for name in parameter_names}
        measurements: dict[int, int] = {}
        phase = _materialize_scalar(
            program.global_phase,
            parameters,
            {},
            emitter,
        )
        if not _is_zero(program.global_phase):
            emitter.emit_global_phase(
                circuit,
                phase,
                carrier=0 if program.num_qubits else None,
            )
        _emit_region(
            program.operations,
            circuit,
            wires,
            parameters,
            measurements,
            {},
            emitter,
            tuple(range(program.num_qubits, program.num_qubits + ancilla_count)),
        )
        return MaterializedCircuit(
            artifact=circuit,
            parameters=parameters,
            measurement_qubit_map=measurements,
            parameter_order=tuple(parameters),
            implicit_output_qubit_indices=tuple(range(program.num_qubits)),
        )


def _emit_region(
    operations: tuple[CircuitInstruction, ...],
    circuit: Any,
    input_wires: dict[WireId, int],
    parameters: dict[str, Any],
    measurements: dict[int, int],
    loop_variables: dict[str, int],
    emitter: QuriPartsGateEmitter,
    ancillas: tuple[int, ...],
) -> dict[WireId, int]:
    """Emit one QURI Parts circuit region.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region instructions.
        circuit (Any): Destination QURI circuit.
        input_wires (dict[WireId, int]): Region input wire slots.
        parameters (dict[str, Any]): QURI parameter cache.
        measurements (dict[int, int]): Static measurement mapping to update.
        loop_variables (dict[str, int]): Concrete induction values.
        emitter (QuriPartsGateEmitter): Primitive gate materializer.
        ancillas (tuple[int, ...]): Clean ancillary qubit slots.

    Returns:
        dict[WireId, int]: Mapping containing produced virtual wires.

    Raises:
        EmitError: If the QURI profile cannot represent an instruction.
    """
    wires = dict(input_wires)
    for operation in operations:
        if isinstance(operation, GateInstruction):
            qubits = tuple(wires[wire] for wire in operation.inputs)
            angles = tuple(
                _materialize_scalar(
                    expression,
                    parameters,
                    loop_variables,
                    emitter,
                )
                for expression in operation.parameters
            )
            _emit_gate(emitter, circuit, operation.kind, qubits, angles)
            _publish(operation.outputs, qubits, wires)
        elif isinstance(operation, MeasureInstruction):
            slot = wires[operation.input]
            measurements[operation.clbit] = slot
            wires[operation.output] = slot
        elif isinstance(operation, MeasureVectorInstruction):
            slots = tuple(wires[wire] for wire in operation.inputs)
            measurements.update(zip(operation.clbits, slots, strict=True))
            _publish(operation.outputs, slots, wires)
        elif isinstance(operation, BarrierInstruction):
            emitter.emit_barrier(circuit, [wires[wire] for wire in operation.wires])
        elif isinstance(operation, ResetInstruction):
            raise EmitError("QURI Parts cannot emit a qubit reset")
        elif isinstance(operation, PauliEvolutionInstruction):
            if operation.realization is PauliEvolutionRealization.NATIVE:
                _emit_pauli_evolution_native(
                    operation,
                    circuit,
                    wires,
                    parameters,
                    loop_variables,
                    emitter,
                )
            elif operation.realization is PauliEvolutionRealization.GADGET:
                _emit_pauli_evolution_gadget(
                    operation,
                    circuit,
                    wires,
                    parameters,
                    loop_variables,
                    emitter,
                )
            else:  # pragma: no cover - target verification owns this invariant
                raise AssertionError("QURI Pauli evolution was not legalized")
        elif isinstance(operation, ForInstruction):
            current = [wires[wire] for wire in operation.inputs]
            for index in operation.indexset:
                body_inputs = dict(zip(operation.inputs, current, strict=True))
                nested_variables = dict(loop_variables)
                nested_variables[operation.loop_variable.name] = index
                body_wires = _emit_region(
                    operation.body,
                    circuit,
                    body_inputs,
                    parameters,
                    measurements,
                    nested_variables,
                    emitter,
                    ancillas,
                )
                current = [body_wires[wire] for wire in operation.body_outputs]
            _publish(operation.outputs, current, wires)
        elif isinstance(operation, CallInstruction):
            _emit_call(
                operation,
                circuit,
                wires,
                parameters,
                measurements,
                loop_variables,
                emitter,
                ancillas,
            )
        elif isinstance(operation, (IfInstruction, WhileInstruction)):
            raise EmitError("QURI Parts does not support runtime circuit control flow")
        else:  # pragma: no cover - defensive closed-union guard
            raise EmitError(f"Unsupported QURI instruction: {operation!r}")
    return wires


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


def _materialize_scalar(
    expression: ScalarExpr,
    parameters: dict[str, Any],
    loop_variables: dict[str, int],
    emitter: QuriPartsGateEmitter,
) -> Any:
    """Convert a circuit scalar to a QURI concrete or linear angle.

    Args:
        expression (ScalarExpr): Target-neutral scalar expression.
        parameters (dict[str, Any]): QURI parameter cache.
        loop_variables (dict[str, int]): Concrete induction values.
        emitter (QuriPartsGateEmitter): Parameter and linear-form helper.

    Returns:
        Any: Concrete scalar, QURI parameter, or linear-form mapping.

    Raises:
        QamomileQuriPartsTranspileError: If the expression is nonlinear or
            references measurement bits.
    """
    if isinstance(expression, LiteralExpr):
        return expression.value
    if isinstance(expression, ParameterExpr):
        parameter = parameters.get(expression.name)
        if parameter is None:
            parameter = emitter.create_parameter(expression.name)
            parameters[expression.name] = parameter
        return parameter
    if isinstance(expression, LoopVariableExpr):
        try:
            return loop_variables[expression.name]
        except KeyError as error:
            raise QamomileQuriPartsTranspileError(
                f"Unresolved QURI loop variable {expression.name!r}"
            ) from error
    if isinstance(expression, ClassicalBitExpr):
        raise QamomileQuriPartsTranspileError(
            "QURI Parts angles cannot depend on measurement bits"
        )
    if isinstance(expression, UnaryExpr):
        if expression.operator is UnaryOperator.NEG:
            operand = _materialize_scalar(
                expression.operand,
                parameters,
                loop_variables,
                emitter,
            )
            if isinstance(operand, numbers.Real):
                return -float(operand)
            return emitter.combine_symbolic(BinOpKind.MUL, -1.0, operand)
        raise QamomileQuriPartsTranspileError(
            "QURI Parts does not support logical NOT in angle expressions"
        )
    if isinstance(expression, BinaryExpr):
        kind = _BINARY_KINDS.get(expression.operator)
        if kind is None:
            raise QamomileQuriPartsTranspileError(
                f"QURI Parts angle cannot contain {expression.operator.name}"
            )
        left = _materialize_scalar(
            expression.left,
            parameters,
            loop_variables,
            emitter,
        )
        right = _materialize_scalar(
            expression.right,
            parameters,
            loop_variables,
            emitter,
        )
        if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
            return _evaluate_concrete_binary(expression.operator, left, right)
        return emitter.combine_symbolic(kind, left, right)
    raise QamomileQuriPartsTranspileError(
        f"Unsupported QURI scalar expression {expression!r}"
    )


def _evaluate_concrete_binary(
    operator: BinaryOperator,
    left: numbers.Real,
    right: numbers.Real,
) -> float:
    """Evaluate a concrete arithmetic expression before QURI emission.

    Args:
        operator (BinaryOperator): Arithmetic operation.
        left (numbers.Real): Left operand.
        right (numbers.Real): Right operand.

    Returns:
        float: Concrete result.

    Raises:
        QamomileQuriPartsTranspileError: If the operator is non-arithmetic.
    """
    left_value = float(left)  # type: ignore[arg-type]
    right_value = float(right)  # type: ignore[arg-type]
    if operator is BinaryOperator.ADD:
        return left_value + right_value
    if operator is BinaryOperator.SUB:
        return left_value - right_value
    if operator is BinaryOperator.MUL:
        return left_value * right_value
    if operator is BinaryOperator.DIV:
        return left_value / right_value
    if operator is BinaryOperator.FLOORDIV:
        return left_value // right_value
    if operator is BinaryOperator.MOD:
        return left_value % right_value
    if operator is BinaryOperator.POW:
        return left_value**right_value
    raise QamomileQuriPartsTranspileError(f"QURI angle cannot contain {operator.name}")


def _emit_call(
    operation: CallInstruction,
    circuit: Any,
    wires: dict[WireId, int],
    parameters: dict[str, Any],
    measurements: dict[int, int],
    loop_variables: dict[str, int],
    emitter: QuriPartsGateEmitter,
    ancillas: tuple[int, ...],
) -> None:
    """Inline a direct reusable call into a QURI circuit.

    Args:
        operation (CallInstruction): Reusable circuit call.
        circuit (Any): Destination QURI circuit.
        wires (dict[WireId, int]): Enclosing virtual wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        measurements (dict[int, int]): Static measurement mapping.
        loop_variables (dict[str, int]): Concrete induction values.
        emitter (QuriPartsGateEmitter): Primitive gate materializer.
        ancillas (tuple[int, ...]): Clean ancillary qubit slots.

    Raises:
        EmitError: If inverse or controlled transforms remain unlegalized.
    """
    _emit_transformed_call(
        operation,
        circuit,
        wires,
        parameters,
        measurements,
        loop_variables,
        emitter,
        ancillas,
        inherited_controls=(),
        inherited_inverse=False,
    )


def _emit_transformed_call(
    operation: CallInstruction,
    circuit: Any,
    wires: dict[WireId, int],
    parameters: dict[str, Any],
    measurements: dict[int, int],
    loop_variables: dict[str, int],
    emitter: QuriPartsGateEmitter,
    ancillas: tuple[int, ...],
    inherited_controls: tuple[int, ...],
    inherited_inverse: bool,
) -> None:
    """Inline a call while composing control and inverse transforms.

    Args:
        operation (CallInstruction): Reusable circuit invocation.
        circuit (Any): Destination QURI circuit.
        wires (dict[WireId, int]): Enclosing virtual wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        measurements (dict[int, int]): Static measurement mapping.
        loop_variables (dict[str, int]): Concrete induction values.
        emitter (QuriPartsGateEmitter): Primitive gate materializer.
        ancillas (tuple[int, ...]): Clean ancillary slots.
        inherited_controls (tuple[int, ...]): Controls from enclosing calls.
        inherited_inverse (bool): Whether the enclosing region is inverted.
    """
    callee = operation.callee
    call_inputs = operation.outputs if inherited_inverse else operation.inputs
    call_outputs = operation.inputs if inherited_inverse else operation.outputs
    actual = [wires[wire] for wire in call_inputs]
    own_controls = tuple(actual[: callee.controls])
    targets = actual[callee.controls :]
    controls = (*inherited_controls, *own_controls)
    inverse = inherited_inverse ^ callee.inverse
    phase_expression: ScalarExpr = callee.body.global_phase
    if inverse and not _is_zero(phase_expression):
        phase_expression = UnaryExpr(UnaryOperator.NEG, phase_expression)
    phase = _materialize_scalar(
        phase_expression,
        parameters,
        loop_variables,
        emitter,
    )
    for _ in range(callee.power):
        body_entry = callee.body.output_wires if inverse else callee.body.input_wires
        body_exit = callee.body.input_wires if inverse else callee.body.output_wires
        body_inputs = dict(zip(body_entry, targets, strict=True))
        body_wires = _emit_transformed_region(
            callee.body.operations,
            circuit,
            body_inputs,
            parameters,
            measurements,
            loop_variables,
            emitter,
            ancillas,
            controls,
            inverse,
        )
        targets = [body_wires[wire] for wire in body_exit]
        if not _is_zero(phase_expression):
            if controls:
                _emit_projector_phase(
                    circuit,
                    phase,
                    controls,
                    emitter,
                    ancillas,
                )
            else:
                emitter.emit_global_phase(
                    circuit,
                    phase,
                    carrier=targets[0] if targets else None,
                )
    _publish(call_outputs, [*own_controls, *targets], wires)


def _emit_projector_phase(
    circuit: Any,
    phase: Any,
    controls: tuple[int, ...],
    emitter: QuriPartsGateEmitter,
    ancillas: tuple[int, ...],
) -> None:
    """Emit a phase conditioned on every active control being one.

    Args:
        circuit (Any): Destination QURI Parts circuit.
        phase (Any): Concrete or linear phase expression.
        controls (tuple[int, ...]): Accumulated coherent control slots.
        emitter (QuriPartsGateEmitter): Primitive gate adapter.
        ancillas (tuple[int, ...]): Available clean ancillary slots.

    Raises:
        EmitError: If no controls are supplied or the ancilla pool is too
            small for a multi-controlled phase.
    """
    if not controls:
        raise EmitError("Projector phase requires at least one control")
    if len(controls) == 1:
        emitter.emit_p(circuit, controls[0], phase)
        return
    outer_controls = tuple(controls[index] for index in range(len(controls) - 1))
    target = controls[len(controls) - 1]
    _emit_controlled_gate(
        emitter,
        circuit,
        GateKind.P,
        outer_controls,
        (target,),
        (phase,),
        ancillas,
    )


def _emit_transformed_region(
    operations: tuple[CircuitInstruction, ...],
    circuit: Any,
    input_wires: dict[WireId, int],
    parameters: dict[str, Any],
    measurements: dict[int, int],
    loop_variables: dict[str, int],
    emitter: QuriPartsGateEmitter,
    ancillas: tuple[int, ...],
    controls: tuple[int, ...],
    inverse: bool,
) -> dict[WireId, int]:
    """Emit a unitary region under composed call transforms.

    Args:
        operations (tuple[CircuitInstruction, ...]): Unitary body operations.
        circuit (Any): Destination QURI circuit.
        input_wires (dict[WireId, int]): Body input mapping.
        parameters (dict[str, Any]): Parameter cache.
        measurements (dict[int, int]): Static measurement mapping.
        loop_variables (dict[str, int]): Concrete induction values.
        emitter (QuriPartsGateEmitter): Primitive gate materializer.
        ancillas (tuple[int, ...]): Clean ancillary slots.
        controls (tuple[int, ...]): Accumulated physical control slots.
        inverse (bool): Whether operation order and gates are inverted.

    Returns:
        dict[WireId, int]: Produced virtual wire mapping.

    Raises:
        EmitError: If a non-unitary or unsupported transformed operation
            occurs.
    """
    if _should_batch_controlled_region(operations, len(controls)):
        required = len(controls) - 1
        used = ancillas[:required]
        if len(used) != required:
            raise EmitError(
                f"Controlled region requires {required} clean ancilla qubits"
            )
        _compute_control_conjunction(emitter, circuit, controls, used)
        try:
            return _emit_transformed_region(
                operations,
                circuit,
                input_wires,
                parameters,
                measurements,
                loop_variables,
                emitter,
                ancillas[required:],
                (used[-1],),
                inverse,
            )
        finally:
            _uncompute_control_conjunction(emitter, circuit, controls, used)

    wires = dict(input_wires)
    sequence = reversed(operations) if inverse else operations
    for operation in sequence:
        if isinstance(operation, GateInstruction):
            gate_inputs = operation.outputs if inverse else operation.inputs
            gate_outputs = operation.inputs if inverse else operation.outputs
            qubits = tuple(wires[wire] for wire in gate_inputs)
            angles = tuple(
                _materialize_scalar(value, parameters, loop_variables, emitter)
                for value in operation.parameters
            )
            kind, angles = _invert_gate(operation.kind, angles, emitter, inverse)
            if controls:
                _emit_controlled_gate(
                    emitter,
                    circuit,
                    kind,
                    controls,
                    qubits,
                    angles,
                    ancillas,
                )
            else:
                _emit_gate(emitter, circuit, kind, qubits, angles)
            _publish(gate_outputs, qubits, wires)
        elif isinstance(operation, ForInstruction):
            loop_inputs = operation.outputs if inverse else operation.inputs
            loop_outputs = operation.inputs if inverse else operation.outputs
            body_entry = operation.body_outputs if inverse else operation.inputs
            body_exit = operation.inputs if inverse else operation.body_outputs
            current = [wires[wire] for wire in loop_inputs]
            indices = list(operation.indexset)
            if inverse:
                indices.reverse()
            for index in indices:
                nested_variables = dict(loop_variables)
                nested_variables[operation.loop_variable.name] = index
                body_inputs = dict(zip(body_entry, current, strict=True))
                body_wires = _emit_transformed_region(
                    operation.body,
                    circuit,
                    body_inputs,
                    parameters,
                    measurements,
                    nested_variables,
                    emitter,
                    ancillas,
                    controls,
                    inverse,
                )
                current = [body_wires[wire] for wire in body_exit]
            _publish(loop_outputs, current, wires)
        elif isinstance(operation, CallInstruction):
            _emit_transformed_call(
                operation,
                circuit,
                wires,
                parameters,
                measurements,
                loop_variables,
                emitter,
                ancillas,
                controls,
                inverse,
            )
        elif isinstance(operation, PauliEvolutionInstruction):
            if controls:
                transformed = operation
                if inverse:
                    transformed = dataclasses.replace(
                        operation,
                        time=UnaryExpr(UnaryOperator.NEG, operation.time),
                    )
                _emit_controlled_pauli_evolution(
                    transformed,
                    circuit,
                    wires,
                    parameters,
                    loop_variables,
                    emitter,
                    controls,
                    ancillas,
                )
                continue
            transformed = operation
            if inverse:
                transformed = dataclasses.replace(
                    operation,
                    time=UnaryExpr(UnaryOperator.NEG, operation.time),
                )
            if transformed.realization is PauliEvolutionRealization.NATIVE:
                _emit_pauli_evolution_native(
                    transformed,
                    circuit,
                    wires,
                    parameters,
                    loop_variables,
                    emitter,
                )
            else:
                _emit_pauli_evolution_gadget(
                    transformed,
                    circuit,
                    wires,
                    parameters,
                    loop_variables,
                    emitter,
                )
        elif isinstance(operation, BarrierInstruction):
            continue
        else:
            raise EmitError(
                f"Cannot apply reusable transform to {type(operation).__name__}"
            )
    return wires


def _emit_pauli_evolution_native(
    operation: PauliEvolutionInstruction,
    circuit: Any,
    wires: dict[WireId, int],
    parameters: dict[str, Any],
    loop_variables: dict[str, int],
    emitter: QuriPartsGateEmitter,
) -> None:
    """Materialize Pauli evolution with native QURI PauliRotation gates.

    Args:
        operation (PauliEvolutionInstruction): Abstract evolution.
        circuit (Any): Destination QURI circuit.
        wires (dict[WireId, int]): Virtual wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        loop_variables (dict[str, int]): Concrete induction values.
        emitter (QuriPartsGateEmitter): Native gate materializer.
    """
    time = _materialize_scalar(
        operation.time,
        parameters,
        loop_variables,
        emitter,
    )
    slots = tuple(wires[wire] for wire in operation.inputs)
    for operators, coefficient in operation.hamiltonian:
        if not operators or abs(coefficient) == 0:
            continue
        selected = [slots[item.index] for item in operators]
        pauli_ids = [int(item.pauli.value) + 1 for item in operators]
        factor = 2.0 * float(coefficient.real)
        angle = (
            factor * float(time)
            if isinstance(time, numbers.Real)
            else emitter.combine_symbolic(BinOpKind.MUL, factor, time)
        )
        emitter.emit_pauli_rotation(circuit, selected, pauli_ids, angle)
    _publish(operation.outputs, slots, wires)


def _emit_pauli_evolution_gadget(
    operation: PauliEvolutionInstruction,
    circuit: Any,
    wires: dict[WireId, int],
    parameters: dict[str, Any],
    loop_variables: dict[str, int],
    emitter: QuriPartsGateEmitter,
) -> None:
    """Materialize abstract Pauli evolution as primitive phase gadgets.

    Args:
        operation (PauliEvolutionInstruction): Abstract evolution.
        circuit (Any): Destination QURI circuit.
        wires (dict[WireId, int]): Virtual wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        loop_variables (dict[str, int]): Concrete induction values.
        emitter (QuriPartsGateEmitter): Primitive gate materializer.
    """
    import qamomile.observable as qm_o

    time = _materialize_scalar(
        operation.time,
        parameters,
        loop_variables,
        emitter,
    )
    slots = tuple(wires[wire] for wire in operation.inputs)
    for operators, coefficient in operation.hamiltonian:
        selected = [slots[item.index] for item in operators]
        for item, slot in zip(operators, selected, strict=True):
            if item.pauli is qm_o.Pauli.X:
                emitter.emit_h(circuit, slot)
            elif item.pauli is qm_o.Pauli.Y:
                emitter.emit_sdg(circuit, slot)
                emitter.emit_h(circuit, slot)
        for left, right in zip(selected, selected[1:]):
            emitter.emit_cx(circuit, left, right)
        if selected:
            factor = 2.0 * float(coefficient.real)
            angle = (
                factor * float(time)
                if isinstance(time, numbers.Real)
                else emitter.combine_symbolic(BinOpKind.MUL, factor, time)
            )
            emitter.emit_rz(circuit, selected[-1], angle)
        for left, right in reversed(list(zip(selected, selected[1:]))):
            emitter.emit_cx(circuit, left, right)
        for item, slot in reversed(list(zip(operators, selected, strict=True))):
            if item.pauli is qm_o.Pauli.X:
                emitter.emit_h(circuit, slot)
            elif item.pauli is qm_o.Pauli.Y:
                emitter.emit_h(circuit, slot)
                emitter.emit_s(circuit, slot)
    _publish(operation.outputs, slots, wires)


def _emit_controlled_pauli_evolution(
    operation: PauliEvolutionInstruction,
    circuit: Any,
    wires: dict[WireId, int],
    parameters: dict[str, Any],
    loop_variables: dict[str, int],
    emitter: QuriPartsGateEmitter,
    controls: tuple[int, ...],
    ancillas: tuple[int, ...],
) -> None:
    """Legalize a controlled Pauli evolution to controlled phase gadgets.

    Args:
        operation (PauliEvolutionInstruction): Abstract evolution.
        circuit (Any): Destination QURI circuit.
        wires (dict[WireId, int]): Virtual wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        loop_variables (dict[str, int]): Concrete induction values.
        emitter (QuriPartsGateEmitter): Primitive gate materializer.
        controls (tuple[int, ...]): Accumulated physical controls.
        ancillas (tuple[int, ...]): Clean ancillary slots.

    Raises:
        EmitError: If the evolution time is runtime-parametric.
    """
    import qamomile.observable as qm_o

    time = _materialize_scalar(
        operation.time,
        parameters,
        loop_variables,
        emitter,
    )
    if not isinstance(time, numbers.Real):
        raise EmitError(
            "Controlled Pauli evolution requires a compile-time-numeric "
            "time on QURI Parts"
        )
    slots = tuple(wires[wire] for wire in operation.inputs)
    for operators, coefficient in operation.hamiltonian:
        if not operators or abs(coefficient) == 0:
            continue
        selected = [slots[item.index] for item in operators]
        for item, slot in zip(operators, selected, strict=True):
            if item.pauli is qm_o.Pauli.X:
                emitter.emit_h(circuit, slot)
            elif item.pauli is qm_o.Pauli.Y:
                emitter.emit_sdg(circuit, slot)
                emitter.emit_h(circuit, slot)
        for left, right in zip(selected, selected[1:]):
            emitter.emit_cx(circuit, left, right)
        angle = 2.0 * float(coefficient.real) * float(time)
        _emit_controlled_gate(
            emitter,
            circuit,
            GateKind.RZ,
            controls,
            (selected[-1],),
            (angle,),
            ancillas,
        )
        for left, right in reversed(list(zip(selected, selected[1:]))):
            emitter.emit_cx(circuit, left, right)
        for item, slot in reversed(list(zip(operators, selected, strict=True))):
            if item.pauli is qm_o.Pauli.X:
                emitter.emit_h(circuit, slot)
            elif item.pauli is qm_o.Pauli.Y:
                emitter.emit_h(circuit, slot)
                emitter.emit_s(circuit, slot)

    _publish(operation.outputs, slots, wires)


def _invert_gate(
    kind: GateKind,
    angles: tuple[Any, ...],
    emitter: QuriPartsGateEmitter,
    inverse: bool,
) -> tuple[GateKind, tuple[Any, ...]]:
    """Return the inverse primitive gate representation when requested.

    Args:
        kind (GateKind): Original gate kind.
        angles (tuple[Any, ...]): Materialized gate angles.
        emitter (QuriPartsGateEmitter): Linear-form helper.
        inverse (bool): Whether inversion is requested.

    Returns:
        tuple[GateKind, tuple[Any, ...]]: Inverted kind and angles.
    """
    if not inverse:
        return kind, angles
    adjoints = {
        GateKind.S: GateKind.SDG,
        GateKind.SDG: GateKind.S,
        GateKind.T: GateKind.TDG,
        GateKind.TDG: GateKind.T,
    }
    if kind in adjoints:
        return adjoints[kind], angles
    if angles:
        negated = tuple(
            -float(angle)
            if isinstance(angle, numbers.Real)
            else emitter.combine_symbolic(BinOpKind.MUL, -1.0, angle)
            for angle in angles
        )
        return kind, negated
    return kind, angles


def _emit_controlled_gate(
    emitter: QuriPartsGateEmitter,
    circuit: Any,
    kind: GateKind,
    controls: tuple[int, ...],
    qubits: tuple[int, ...],
    angles: tuple[Any, ...],
    ancillas: tuple[int, ...],
) -> None:
    """Legalize one primitive gate under accumulated controls.

    Args:
        emitter (QuriPartsGateEmitter): Primitive gate adapter.
        circuit (Any): Destination QURI circuit.
        kind (GateKind): Primitive body gate.
        controls (tuple[int, ...]): Accumulated outer controls.
        qubits (tuple[int, ...]): Body gate operands.
        angles (tuple[Any, ...]): Materialized angles.
        ancillas (tuple[int, ...]): Clean ancillary slots.

    Raises:
        EmitError: If the controlled form is not implemented.
    """
    if not controls:
        raise EmitError("Controlled gate legalization requires a control qubit")
    if kind is GateKind.X:
        _emit_multi_controlled_x(emitter, circuit, controls, qubits[0], ancillas)
        return
    if kind is GateKind.CX:
        _emit_multi_controlled_x(
            emitter,
            circuit,
            (*controls, qubits[0]),
            qubits[1],
            ancillas,
        )
        return
    if kind is GateKind.TOFFOLI:
        _emit_multi_controlled_x(
            emitter,
            circuit,
            (*controls, qubits[0], qubits[1]),
            qubits[2],
            ancillas,
        )
        return
    if kind in {GateKind.Z, GateKind.CZ}:
        intrinsic_controls = qubits[:-1] if kind is GateKind.CZ else ()
        target = qubits[-1]
        emitter.emit_h(circuit, target)
        _emit_multi_controlled_x(
            emitter,
            circuit,
            (*controls, *intrinsic_controls),
            target,
            ancillas,
        )
        emitter.emit_h(circuit, target)
        return
    if kind in {GateKind.Y, GateKind.CY}:
        intrinsic_controls = qubits[:-1] if kind is GateKind.CY else ()
        target = qubits[-1]
        emitter.emit_s(circuit, target)
        _emit_multi_controlled_x(
            emitter,
            circuit,
            (*controls, *intrinsic_controls),
            target,
            ancillas,
        )
        emitter.emit_sdg(circuit, target)
        return
    if len(controls) > 1:
        required = len(controls) - 1
        if len(ancillas) < required:
            raise EmitError(
                f"{len(controls)}-controlled {kind.name} requires "
                f"{required} clean ancilla qubits"
            )
        used = ancillas[:required]
        _compute_control_conjunction(emitter, circuit, controls, used)
        try:
            _emit_controlled_gate(
                emitter,
                circuit,
                kind,
                (used[-1],),
                qubits,
                angles,
                ancillas[required:],
            )
        finally:
            _uncompute_control_conjunction(emitter, circuit, controls, used)
        return
    control = next(iter(controls))
    one_control = {
        GateKind.H: emitter.emit_ch,
        GateKind.RX: emitter.emit_crx,
        GateKind.RY: emitter.emit_cry,
        GateKind.RZ: emitter.emit_crz,
        GateKind.P: emitter.emit_cp,
    }
    if kind in one_control:
        if angles:
            one_control[kind](circuit, control, qubits[0], angles[0])  # type: ignore[call-arg]
        else:
            one_control[kind](circuit, control, qubits[0])  # type: ignore[call-arg]
        return
    if kind in {GateKind.S, GateKind.SDG, GateKind.T, GateKind.TDG}:
        phase = {
            GateKind.S: math.pi / 2,
            GateKind.SDG: -math.pi / 2,
            GateKind.T: math.pi / 4,
            GateKind.TDG: -math.pi / 4,
        }[kind]
        emitter.emit_cp(circuit, control, qubits[0], phase)
        return
    if kind is GateKind.CP:
        outer, inner, target = control, qubits[0], qubits[1]
        angle = angles[0]
        half = _scale_angle(emitter, angle, 0.5)
        emitter.emit_cp(circuit, outer, target, half)
        emitter.emit_cx(circuit, inner, outer)
        emitter.emit_cp(circuit, outer, target, _scale_angle(emitter, angle, -0.5))
        emitter.emit_cx(circuit, inner, outer)
        emitter.emit_cp(circuit, inner, target, half)
        return
    if kind is GateKind.RZZ:
        left, right = qubits
        emitter.emit_toffoli(circuit, control, left, right)
        emitter.emit_crz(circuit, control, right, angles[0])
        emitter.emit_toffoli(circuit, control, left, right)
        return
    if kind is GateKind.SWAP:
        left, right = qubits
        emitter.emit_cx(circuit, left, right)
        emitter.emit_toffoli(circuit, control, right, left)
        emitter.emit_cx(circuit, left, right)
        return
    raise EmitError(f"QURI Parts cannot legalize controlled {kind.name}")


def _compute_control_conjunction(
    emitter: QuriPartsGateEmitter,
    circuit: Any,
    controls: tuple[int, ...],
    ancillas: tuple[int, ...],
) -> None:
    """Compute the conjunction of controls into the final clean ancilla.

    Args:
        emitter (QuriPartsGateEmitter): Primitive gate adapter.
        circuit (Any): Destination QURI circuit.
        controls (tuple[int, ...]): Two or more physical controls.
        ancillas (tuple[int, ...]): Exactly ``len(controls) - 1`` clean slots.
    """
    emitter.emit_toffoli(circuit, controls[0], controls[1], ancillas[0])
    for index in range(2, len(controls)):
        emitter.emit_toffoli(
            circuit,
            controls[index],
            ancillas[index - 2],
            ancillas[index - 1],
        )


def _uncompute_control_conjunction(
    emitter: QuriPartsGateEmitter,
    circuit: Any,
    controls: tuple[int, ...],
    ancillas: tuple[int, ...],
) -> None:
    """Restore ancillas used for a control conjunction to zero.

    Args:
        emitter (QuriPartsGateEmitter): Primitive gate adapter.
        circuit (Any): Destination QURI circuit.
        controls (tuple[int, ...]): Two or more physical controls.
        ancillas (tuple[int, ...]): Conjunction slots to uncompute.
    """
    for index in range(len(controls) - 1, 1, -1):
        emitter.emit_toffoli(
            circuit,
            controls[index],
            ancillas[index - 2],
            ancillas[index - 1],
        )
    emitter.emit_toffoli(circuit, controls[0], controls[1], ancillas[0])


def _scale_angle(
    emitter: QuriPartsGateEmitter,
    angle: Any,
    factor: float,
) -> Any:
    """Scale a concrete or linear QURI angle.

    Args:
        emitter (QuriPartsGateEmitter): Linear-form helper.
        angle (Any): Concrete or parameterized angle.
        factor (float): Numeric scale.

    Returns:
        Any: Scaled angle representation.
    """
    if isinstance(angle, numbers.Real):
        return float(angle) * factor
    return emitter.combine_symbolic(BinOpKind.MUL, factor, angle)


def _emit_multi_controlled_x(
    emitter: QuriPartsGateEmitter,
    circuit: Any,
    controls: tuple[int, ...],
    target: int,
    ancillas: tuple[int, ...],
) -> None:
    """Emit a clean-ancilla multi-controlled X cascade.

    Args:
        emitter (QuriPartsGateEmitter): Primitive gate adapter.
        circuit (Any): Destination QURI circuit.
        controls (tuple[int, ...]): Physical control slots.
        target (int): Physical target slot.
        ancillas (tuple[int, ...]): Available clean ancillary slots.

    Raises:
        EmitError: If controls alias or the ancilla pool is too small.
    """
    if len(set((*controls, target))) != len(controls) + 1:
        raise EmitError("Controlled gate operands alias the same physical qubit")
    if len(controls) == 1:
        emitter.emit_cx(circuit, controls[0], target)
        return
    if len(controls) == 2:
        emitter.emit_toffoli(circuit, controls[0], controls[1], target)
        return
    required = len(controls) - 2
    if len(ancillas) < required:
        raise EmitError(f"Multi-controlled X requires {required} clean ancilla qubits")
    used = ancillas[:required]
    emitter.emit_toffoli(circuit, controls[0], controls[1], used[0])
    for index in range(2, len(controls) - 1):
        emitter.emit_toffoli(
            circuit,
            controls[index],
            used[index - 2],
            used[index - 1],
        )
    emitter.emit_toffoli(circuit, used[-1], controls[-1], target)
    for index in range(len(controls) - 2, 1, -1):
        emitter.emit_toffoli(
            circuit,
            controls[index],
            used[index - 2],
            used[index - 1],
        )
    emitter.emit_toffoli(circuit, controls[0], controls[1], used[0])


def _emit_gate(
    emitter: QuriPartsGateEmitter,
    circuit: Any,
    kind: GateKind,
    qubits: tuple[int, ...],
    angles: tuple[Any, ...],
) -> None:
    """Emit one primitive QURI Parts gate.

    Args:
        emitter (QuriPartsGateEmitter): Primitive gate adapter.
        circuit (Any): Destination circuit.
        kind (GateKind): Primitive gate kind.
        qubits (tuple[int, ...]): Physical qubit slots.
        angles (tuple[Any, ...]): Concrete or linear angle values.

    Raises:
        EmitError: If the gate kind is unsupported.
    """
    fixed = {
        GateKind.H: emitter.emit_h,
        GateKind.X: emitter.emit_x,
        GateKind.Y: emitter.emit_y,
        GateKind.Z: emitter.emit_z,
        GateKind.S: emitter.emit_s,
        GateKind.SDG: emitter.emit_sdg,
        GateKind.T: emitter.emit_t,
        GateKind.TDG: emitter.emit_tdg,
        GateKind.CX: emitter.emit_cx,
        GateKind.CZ: emitter.emit_cz,
        GateKind.SWAP: emitter.emit_swap,
        GateKind.CH: emitter.emit_ch,
        GateKind.CY: emitter.emit_cy,
        GateKind.TOFFOLI: emitter.emit_toffoli,
    }
    if kind in fixed:
        fixed[kind](circuit, *qubits)
        return
    rotations = {
        GateKind.RX: emitter.emit_rx,
        GateKind.RY: emitter.emit_ry,
        GateKind.RZ: emitter.emit_rz,
        GateKind.P: emitter.emit_p,
        GateKind.CP: emitter.emit_cp,
        GateKind.RZZ: emitter.emit_rzz,
        GateKind.CRX: emitter.emit_crx,
        GateKind.CRY: emitter.emit_cry,
        GateKind.CRZ: emitter.emit_crz,
    }
    if kind in rotations and len(angles) == 1:
        rotations[kind](circuit, *qubits, angles[0])  # type: ignore[call-arg]
        return
    raise EmitError(f"Unsupported QURI gate {kind.name}")


def _is_zero(expression: ScalarExpr) -> bool:
    """Return whether an expression is the literal numeric zero.

    Args:
        expression (ScalarExpr): Expression to inspect.

    Returns:
        bool: True only for a zero literal.
    """
    return isinstance(expression, LiteralExpr) and not float(expression.value)


def _contains_runtime_parameter(expression: ScalarExpr) -> bool:
    """Return whether a scalar expression contains a runtime parameter.

    Loop variables are intentionally concrete here because QURI Parts unrolls
    every accepted ``ForInstruction`` before gate emission.

    Args:
        expression (ScalarExpr): Expression to inspect recursively.

    Returns:
        bool: True when a ``ParameterExpr`` occurs in the expression tree.
    """
    if isinstance(expression, ParameterExpr):
        return True
    if isinstance(expression, UnaryExpr):
        return _contains_runtime_parameter(expression.operand)
    if isinstance(expression, BinaryExpr):
        return _contains_runtime_parameter(
            expression.left
        ) or _contains_runtime_parameter(expression.right)
    return False


def _requires_phase_carrier(program: CircuitProgram) -> bool:
    """Return whether QURI materialization can emit a scalar phase.

    Args:
        program (CircuitProgram): Verified circuit program to inspect.

    Returns:
        bool: True when a dedicated clean phase-carrier qubit is required.
    """
    root_requires_carrier = not _is_zero(program.global_phase) and (
        program.num_qubits == 0 or _contains_runtime_parameter(program.global_phase)
    )
    return root_requires_carrier or _region_requires_phase_carrier(
        program.operations,
        inherited_controls=0,
    )


def _region_requires_phase_carrier(
    operations: tuple[CircuitInstruction, ...],
    inherited_controls: int,
) -> bool:
    """Inspect one region for paths that call ``emit_global_phase``.

    Args:
        operations (tuple[CircuitInstruction, ...]): Instructions to inspect.
        inherited_controls (int): Coherent controls inherited from callers.

    Returns:
        bool: True when any executable path requires the clean carrier.
    """
    for operation in operations:
        if isinstance(operation, GateInstruction):
            if operation.kind in {GateKind.P, GateKind.CP} and any(
                _contains_runtime_parameter(parameter)
                for parameter in operation.parameters
            ):
                return True
        elif isinstance(operation, CallInstruction):
            effective_controls = inherited_controls + operation.callee.controls
            phase = operation.callee.body.global_phase
            if not _is_zero(phase) and (
                _contains_runtime_parameter(phase)
                or (effective_controls == 0 and operation.callee.body.num_qubits == 0)
            ):
                return True
            if _region_requires_phase_carrier(
                operation.callee.body.operations,
                effective_controls,
            ):
                return True
        elif isinstance(operation, ForInstruction):
            if operation.indexset and _region_requires_phase_carrier(
                operation.body,
                inherited_controls,
            ):
                return True
        elif isinstance(operation, IfInstruction):
            if _region_requires_phase_carrier(
                operation.true_body,
                inherited_controls,
            ) or _region_requires_phase_carrier(
                operation.false_body,
                inherited_controls,
            ):
                return True
        elif isinstance(operation, WhileInstruction) and _region_requires_phase_carrier(
            operation.body,
            inherited_controls,
        ):
            return True
    return False


def _ancilla_demand(
    operations: tuple[CircuitInstruction, ...],
    inherited_controls: int = 0,
) -> int:
    """Compute peak clean-ancilla demand for controlled X-family gates.

    Args:
        operations (tuple[CircuitInstruction, ...]): Instructions to inspect.
        inherited_controls (int): Controls accumulated from enclosing calls.
            Defaults to zero.

    Returns:
        int: Maximum simultaneously required clean ancilla count.
    """
    if _should_batch_controlled_region(operations, inherited_controls):
        shared = inherited_controls - 1
        return shared + _ancilla_demand(operations, 1)
    demand = 0
    for operation in operations:
        if isinstance(operation, GateInstruction):
            intrinsic = {
                GateKind.X: 0,
                GateKind.Y: 0,
                GateKind.Z: 0,
                GateKind.CX: 1,
                GateKind.CY: 1,
                GateKind.CZ: 1,
                GateKind.TOFFOLI: 2,
            }.get(operation.kind)
            if intrinsic is not None:
                demand = max(demand, inherited_controls + intrinsic - 2)
            elif inherited_controls > 1:
                demand = max(demand, inherited_controls - 1)
        elif isinstance(operation, PauliEvolutionInstruction):
            if inherited_controls > 1:
                demand = max(demand, inherited_controls - 1)
        elif isinstance(operation, CallInstruction):
            effective_controls = inherited_controls + operation.callee.controls
            phase_demand = (
                max(0, effective_controls - 2)
                if not _is_zero(operation.callee.body.global_phase)
                else 0
            )
            demand = max(
                demand,
                phase_demand,
                _ancilla_demand(
                    operation.callee.body.operations,
                    effective_controls,
                ),
            )
        elif isinstance(operation, ForInstruction):
            demand = max(
                demand,
                _ancilla_demand(operation.body, inherited_controls),
            )
        elif isinstance(operation, IfInstruction):
            demand = max(
                demand,
                _ancilla_demand(operation.true_body, inherited_controls),
                _ancilla_demand(operation.false_body, inherited_controls),
            )
        elif isinstance(operation, WhileInstruction):
            demand = max(
                demand,
                _ancilla_demand(operation.body, inherited_controls),
            )
    return max(0, demand)


def _should_batch_controlled_region(
    operations: tuple[CircuitInstruction, ...],
    control_count: int,
) -> bool:
    """Return whether one shared control conjunction should wrap a region.

    Args:
        operations (tuple[CircuitInstruction, ...]): Controlled body.
        control_count (int): Number of accumulated controls.

    Returns:
        bool: True when sharing a clean-ancilla AND ladder reduces work.
    """
    if control_count < 2 or _controlled_region_weight(operations) < 2:
        return False
    if control_count == 2 and _contains_only_two_control_native_work(operations):
        return False
    return True


def _controlled_region_weight(
    operations: tuple[CircuitInstruction, ...],
) -> int:
    """Estimate controlled work in a region, capped at two.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region to inspect.

    Returns:
        int: Zero, one, or two, where two means batching can amortize.
    """
    weight = 0
    for operation in operations:
        if isinstance(operation, GateInstruction):
            weight += 1
        elif isinstance(operation, PauliEvolutionInstruction):
            weight += 2
        elif isinstance(operation, ForInstruction):
            body_weight = _controlled_region_weight(operation.body)
            weight += min(2, len(operation.indexset) * body_weight)
        elif isinstance(operation, CallInstruction):
            weight += 1
        if weight >= 2:
            return 2
    return weight


def _contains_only_two_control_native_work(
    operations: tuple[CircuitInstruction, ...],
) -> bool:
    """Return whether two controls need no shared conjunction ancilla.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region to inspect.

    Returns:
        bool: True for bodies composed solely of cheap X/Z-family gates.
    """
    native = {
        GateKind.X,
        GateKind.Z,
        GateKind.CX,
        GateKind.CZ,
        GateKind.TOFFOLI,
        GateKind.RZZ,
    }
    for operation in operations:
        if isinstance(operation, GateInstruction):
            if operation.kind not in native:
                return False
        elif isinstance(operation, ForInstruction):
            if not _contains_only_two_control_native_work(operation.body):
                return False
        else:
            return False
    return True
