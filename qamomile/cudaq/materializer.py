"""Materialize circuit-family IR as CUDA-Q decorator kernels."""

from __future__ import annotations

import dataclasses
from typing import Any

from qamomile.circuit.transpiler.circuit_ir import (
    ALL_BINARY_OPERATORS,
    ALL_PRIMITIVE_GATES,
    ALL_UNARY_OPERATORS,
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
    StandalonePhaseMode,
    UnaryExpr,
    UnaryOperator,
    WhileInstruction,
    WireId,
    verify_circuit,
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.gate_emitter import GateKind, MeasurementMode

from .emitter import CudaqKernelEmitter, ExecutionMode

_CUDAQ_CONTROLLED_GATE_KINDS = frozenset(
    {
        GateKind.CX,
        GateKind.CZ,
        GateKind.CP,
        GateKind.TOFFOLI,
        GateKind.CH,
        GateKind.CY,
        GateKind.CRX,
        GateKind.CRY,
        GateKind.CRZ,
    }
)
_SELF_INVERSE_GATE_KINDS = frozenset(
    {
        GateKind.H,
        GateKind.X,
        GateKind.Y,
        GateKind.Z,
        GateKind.CX,
        GateKind.CZ,
        GateKind.SWAP,
        GateKind.TOFFOLI,
        GateKind.CH,
        GateKind.CY,
    }
)
_SWAPPED_INVERSE_GATE_KINDS = {
    GateKind.S: GateKind.SDG,
    GateKind.SDG: GateKind.S,
    GateKind.T: GateKind.TDG,
    GateKind.TDG: GateKind.T,
}
_NEGATED_PARAMETER_GATE_KINDS = frozenset(
    {
        GateKind.RX,
        GateKind.RY,
        GateKind.RZ,
        GateKind.P,
        GateKind.CP,
        GateKind.RZZ,
        GateKind.CRX,
        GateKind.CRY,
        GateKind.CRZ,
    }
)


class CudaqMaterializer:
    """Convert verified circuit IR to a CUDA-Q source-built kernel."""

    @property
    def capabilities(self) -> CircuitCapabilities:
        """Declare CUDA-Q's circuit-IR capabilities.

        Measurement-conditioned control flow is legal because the
        materializer selects the RUNNABLE execution mode when it appears;
        the static/runnable split is an internal realization detail, not a
        capability boundary.

        Returns:
            CircuitCapabilities: Immutable capability declaration.
        """
        numeric = ScalarCapabilities(
            atoms=frozenset(
                {ScalarAtom.LITERAL, ScalarAtom.PARAMETER, ScalarAtom.LOOP_VARIABLE}
            ),
            unary_operators=frozenset({UnaryOperator.NEG}),
            binary_operators=ARITHMETIC_BINARY_OPERATORS,
            parameter_form=ScalarExpressionForm.ARBITRARY,
        )
        return CircuitCapabilities(
            name="cudaq",
            primitive_gates=ALL_PRIMITIVE_GATES,
            native_semantic_ops=(),
            gate_parameters=numeric,
            predicates=ScalarCapabilities(
                atoms=frozenset(ScalarAtom),
                unary_operators=ALL_UNARY_OPERATORS,
                binary_operators=ALL_BINARY_OPERATORS,
                parameter_form=ScalarExpressionForm.ARBITRARY,
            ),
            pauli_time=numeric,
            global_phase=GlobalPhaseCapabilities(
                scalars=numeric,
                standalone_mode=StandalonePhaseMode.PRESERVE,
                min_qubits=1,
            ),
            generic_calls=CallTransformCapabilities(
                supports_power=True,
                supports_inverse=True,
                max_controls=None,
                supports_barrier_body=True,
                control_mode=CallControlMode.WHOLE_CALL,
                phase_mode=CallPhaseMode.EXPLICIT_CORRECTION,
                controlled_phase_scalars=numeric,
            ),
            supports_dynamic_if=True,
            supports_dynamic_while=True,
            supports_reset=True,
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
        """Build a CUDA-Q kernel and its execution metadata.

        Args:
            program (CircuitProgram): Verified circuit-family program.
            parameter_names (tuple[str, ...]): Public runtime-parameter ABI in
                positional order. Defaults to an empty tuple.

        Returns:
            MaterializedCircuit[Any]: CUDA-Q artifact, runtime parameters, and
                static measurement mapping.

        Raises:
            EmitError: If CUDA-Q cannot represent an instruction.
        """
        verify_circuit(program)
        runtime = _has_runtime_control(program.operations)
        mode = ExecutionMode.RUNNABLE if runtime else ExecutionMode.STATIC
        emitter = CudaqKernelEmitter()
        emitter.measurement_mode = (
            MeasurementMode.RUNNABLE if runtime else MeasurementMode.STATIC
        )
        if runtime:
            emitter.configure_runtime_clbits(program.num_clbits)
        artifact = emitter.create_circuit(program.num_qubits, program.num_clbits)
        wires = {wire: index for index, wire in enumerate(program.input_wires)}
        parameters = {name: emitter.create_parameter(name) for name in parameter_names}
        measurements: dict[int, int] = {}
        phase = _program_global_phase(program, parameters, {}, emitter)
        if not _is_materialized_zero(phase):
            if program.num_qubits < 1:
                raise EmitError(
                    "CUDA-Q requires at least 1 qubit to preserve a nonzero "
                    "standalone global phase"
                )
            emitter.emit_global_phase(artifact, 0, phase)
        _emit_region(
            program.operations,
            artifact,
            wires,
            parameters,
            measurements,
            {},
            emitter,
        )
        emitter.set_parametric(bool(parameters))
        artifact = emitter.finalize(artifact, mode)
        return MaterializedCircuit(
            artifact=artifact,
            parameters=parameters,
            measurement_qubit_map={} if runtime else measurements,
            parameter_order=tuple(parameters),
        )


def _emit_region(
    operations: tuple[CircuitInstruction, ...],
    circuit: Any,
    input_wires: dict[WireId, int],
    parameters: dict[str, Any],
    measurements: dict[int, int],
    loop_variables: dict[str, int],
    emitter: CudaqKernelEmitter,
) -> dict[WireId, int]:
    """Emit one structured CUDA-Q region.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region instructions.
        circuit (Any): Destination CUDA-Q artifact.
        input_wires (dict[WireId, int]): Region input slots.
        parameters (dict[str, Any]): Runtime parameter cache.
        measurements (dict[int, int]): Static measurement mapping.
        loop_variables (dict[str, int]): Concrete loop induction values.
        emitter (CudaqKernelEmitter): CUDA-Q source builder.

    Returns:
        dict[WireId, int]: Mapping containing produced virtual wires.
    """
    wires = dict(input_wires)
    for operation in operations:
        if isinstance(operation, GateInstruction):
            slots = tuple(wires[wire] for wire in operation.inputs)
            angles = tuple(
                _scalar(value, parameters, loop_variables, emitter)
                for value in operation.parameters
            )
            _emit_gate(emitter, circuit, operation.kind, slots, angles)
            _publish(operation.outputs, slots, wires)
        elif isinstance(operation, MeasureInstruction):
            slot = wires[operation.input]
            emitter.emit_measure(circuit, slot, operation.clbit)
            measurements[operation.clbit] = slot
            wires[operation.output] = slot
        elif isinstance(operation, MeasureVectorInstruction):
            slots = tuple(wires[wire] for wire in operation.inputs)
            for slot, clbit in zip(slots, operation.clbits, strict=True):
                emitter.emit_measure(circuit, slot, clbit)
                measurements[clbit] = slot
            _publish(operation.outputs, slots, wires)
        elif isinstance(operation, ResetInstruction):
            slot = wires[operation.input]
            emitter.emit_reset(circuit, slot)
            wires[operation.output] = slot
        elif isinstance(operation, BarrierInstruction):
            emitter.emit_barrier(circuit, [wires[wire] for wire in operation.wires])
        elif isinstance(operation, PauliEvolutionInstruction):
            if operation.realization is PauliEvolutionRealization.ABSTRACT:
                raise AssertionError("CUDA-Q Pauli evolution was not legalized")
            _emit_pauli(
                operation,
                circuit,
                wires,
                parameters,
                loop_variables,
                emitter,
                use_native=operation.realization is PauliEvolutionRealization.NATIVE,
            )
        elif isinstance(operation, ForInstruction):
            current = [wires[wire] for wire in operation.inputs]
            for index in operation.indexset:
                nested = dict(loop_variables)
                nested[operation.loop_variable.name] = index
                body_wires = _emit_region(
                    operation.body,
                    circuit,
                    dict(zip(operation.inputs, current, strict=True)),
                    parameters,
                    measurements,
                    nested,
                    emitter,
                )
                current = [body_wires[wire] for wire in operation.body_outputs]
            _publish(operation.outputs, current, wires)
        elif isinstance(operation, IfInstruction):
            branch_inputs = {wire: wires[wire] for wire in operation.inputs}
            branch_start = emitter.begin_source_block(
                f"if {_predicate(operation.condition, parameters, loop_variables, emitter)}:"
            )
            true_wires = _emit_region(
                operation.true_body,
                circuit,
                branch_inputs,
                parameters,
                measurements,
                loop_variables,
                emitter,
            )
            emitter.end_source_block(branch_start)
            branch_start = emitter.begin_source_block("else:")
            false_wires = _emit_region(
                operation.false_body,
                circuit,
                branch_inputs,
                parameters,
                measurements,
                loop_variables,
                emitter,
            )
            emitter.end_source_block(branch_start)
            true_slots = [true_wires[wire] for wire in operation.true_outputs]
            false_slots = [false_wires[wire] for wire in operation.false_outputs]
            if true_slots != false_slots:
                raise EmitError("CUDA-Q branches change physical wire layout")
            _publish(operation.outputs, true_slots, wires)
        elif isinstance(operation, WhileInstruction):
            body_start = emitter.begin_source_block(
                f"while {_predicate(operation.condition, parameters, loop_variables, emitter)}:"
            )
            body_wires = _emit_region(
                operation.body,
                circuit,
                {wire: wires[wire] for wire in operation.inputs},
                parameters,
                measurements,
                loop_variables,
                emitter,
            )
            emitter.end_source_block(body_start)
            _publish(
                operation.outputs,
                [body_wires[wire] for wire in operation.body_outputs],
                wires,
            )
        elif isinstance(operation, CallInstruction):
            _emit_call(
                operation,
                circuit,
                wires,
                parameters,
                measurements,
                loop_variables,
                emitter,
                (),
                False,
            )
        else:  # pragma: no cover - closed instruction union
            raise EmitError(f"Unsupported CUDA-Q instruction: {operation!r}")
    return wires


def _emit_call(
    operation: CallInstruction,
    circuit: Any,
    wires: dict[WireId, int],
    parameters: dict[str, Any],
    measurements: dict[int, int],
    loop_variables: dict[str, int],
    emitter: CudaqKernelEmitter,
    inherited_controls: tuple[int, ...],
    inherited_inverse: bool,
) -> None:
    """Emit a reusable call through CUDA-Q's named-kernel operations.

    Args:
        operation (CallInstruction): Reusable circuit invocation.
        circuit (Any): Destination CUDA-Q artifact.
        wires (dict[WireId, int]): Enclosing wire mapping.
        parameters (dict[str, Any]): Runtime parameter cache.
        measurements (dict[int, int]): Static measurement mapping.
        loop_variables (dict[str, int]): Concrete loop values.
        emitter (CudaqKernelEmitter): CUDA-Q source builder.
        inherited_controls (tuple[int, ...]): Enclosing control slots.
        inherited_inverse (bool): Whether the enclosing body is inverted.
    """
    callee = operation.callee
    call_inputs = operation.outputs if inherited_inverse else operation.inputs
    call_outputs = operation.inputs if inherited_inverse else operation.outputs
    actual = [wires[wire] for wire in call_inputs]
    own_controls = tuple(actual[: callee.controls])
    targets = actual[callee.controls :]
    controls = (*inherited_controls, *own_controls)
    inverse = inherited_inverse ^ callee.inverse

    def emit_helper_body() -> None:
        """Materialize the target-neutral fallback as one CUDA-Q helper."""
        helper_wires = {
            wire: index for index, wire in enumerate(callee.body.input_wires)
        }
        _emit_region(
            callee.body.operations,
            circuit,
            helper_wires,
            parameters,
            {},
            {},
            emitter,
        )

    try:
        try:
            hash(callee.body)
            definition_key = (callee.name, callee.body)
        except TypeError:
            definition_key = (callee.name, id(callee.body))
        helper_name = emitter.define_helper(
            definition_key,
            callee.name,
            callee.body.num_qubits,
            emit_helper_body,
        )
    except ValueError as error:
        raise EmitError(str(error)) from error
    if inverse:
        if _requires_explicit_inverse(callee.body.operations):

            def emit_inverse_helper_body() -> None:
                """Materialize an SDK-safe explicit inverse helper."""
                helper_wires = {
                    wire: index for index, wire in enumerate(callee.body.output_wires)
                }
                _emit_inverse_region(
                    callee.body.operations,
                    circuit,
                    helper_wires,
                    parameters,
                    {},
                    {},
                    emitter,
                )

            try:
                helper_name = emitter.define_helper(
                    ("explicit-inverse", definition_key),
                    f"{callee.name}_inverse",
                    callee.body.num_qubits,
                    emit_inverse_helper_body,
                )
            except ValueError as error:
                raise EmitError(str(error)) from error
        else:
            helper_name = emitter.define_adjoint_helper(
                helper_name,
                callee.body.num_qubits,
            )
    relative_phase = _program_global_phase(
        callee.body,
        parameters,
        loop_variables,
        emitter,
    )
    if inverse:
        relative_phase = -relative_phase
    for _ in range(callee.power):
        emitter.emit_reusable_call(
            helper_name,
            tuple(targets),
            controls,
        )
        if controls and not _is_materialized_zero(relative_phase):
            phase_controls = list(controls)
            phase_target = phase_controls.pop()
            emitter.emit_multi_controlled_p(
                circuit,
                phase_controls,
                phase_target,
                relative_phase,
            )
    _publish(call_outputs, [*own_controls, *targets], wires)


def _requires_explicit_inverse(
    operations: tuple[CircuitInstruction, ...],
) -> bool:
    """Return whether CUDA-Q 0.14 requires inverse-body expansion.

    CUDA-Q's decorator frontend currently accepts adjoint kernels containing
    controlled operations or another adjoint call, but aborts when executing
    them. Ordinary helpers continue to use native ``cudaq.adjoint``; only
    these unsupported shapes take the late explicit-inverse fallback.

    Args:
        operations (tuple[CircuitInstruction, ...]): Helper region to inspect.

    Returns:
        bool: Whether the helper must be inverted explicitly.
    """
    for operation in operations:
        if (
            isinstance(operation, GateInstruction)
            and operation.kind in _CUDAQ_CONTROLLED_GATE_KINDS
        ):
            return True
        if isinstance(operation, CallInstruction):
            if operation.callee.inverse or operation.callee.controls:
                return True
            if _requires_explicit_inverse(operation.callee.body.operations):
                return True
        if isinstance(operation, ForInstruction) and _requires_explicit_inverse(
            operation.body
        ):
            return True
    return False


def _emit_inverse_region(
    operations: tuple[CircuitInstruction, ...],
    circuit: Any,
    input_wires: dict[WireId, int],
    parameters: dict[str, Any],
    measurements: dict[int, int],
    loop_variables: dict[str, int],
    emitter: CudaqKernelEmitter,
) -> dict[WireId, int]:
    """Emit an explicit inverse only for CUDA-Q-unsupported adjoint shapes.

    Args:
        operations (tuple[CircuitInstruction, ...]): Forward region.
        circuit (Any): Destination CUDA-Q artifact.
        input_wires (dict[WireId, int]): Mapping from forward output wires to
            physical slots.
        parameters (dict[str, Any]): Runtime parameter cache.
        measurements (dict[int, int]): Static measurement mapping.
        loop_variables (dict[str, int]): Active concrete loop values.
        emitter (CudaqKernelEmitter): CUDA-Q source builder.

    Returns:
        dict[WireId, int]: Mapping containing the recovered forward inputs.

    Raises:
        EmitError: If a nonunitary or dynamic construct reaches the fallback.
    """
    wires = dict(input_wires)
    for operation in reversed(operations):
        if isinstance(operation, GateInstruction):
            slots = tuple(wires[wire] for wire in operation.outputs)
            if operation.kind in _SELF_INVERSE_GATE_KINDS:
                kind = operation.kind
                parameters_for_gate = operation.parameters
            elif operation.kind in _SWAPPED_INVERSE_GATE_KINDS:
                kind = _SWAPPED_INVERSE_GATE_KINDS[operation.kind]
                parameters_for_gate = operation.parameters
            elif operation.kind in _NEGATED_PARAMETER_GATE_KINDS:
                kind = operation.kind
                parameters_for_gate = tuple(
                    UnaryExpr(UnaryOperator.NEG, value)
                    for value in operation.parameters
                )
            else:
                raise EmitError(
                    f"CUDA-Q cannot explicitly invert gate {operation.kind.name}"
                )
            angles = tuple(
                _scalar(value, parameters, loop_variables, emitter)
                for value in parameters_for_gate
            )
            _emit_gate(emitter, circuit, kind, slots, angles)
            _publish(operation.inputs, slots, wires)
        elif isinstance(operation, PauliEvolutionInstruction):
            transformed = dataclasses.replace(
                operation,
                time=UnaryExpr(UnaryOperator.NEG, operation.time),
                inputs=operation.outputs,
                outputs=operation.inputs,
            )
            _emit_pauli(
                transformed,
                circuit,
                wires,
                parameters,
                loop_variables,
                emitter,
                use_native=(
                    transformed.realization is PauliEvolutionRealization.NATIVE
                ),
            )
        elif isinstance(operation, CallInstruction):
            _emit_call(
                operation,
                circuit,
                wires,
                parameters,
                measurements,
                loop_variables,
                emitter,
                (),
                True,
            )
        elif isinstance(operation, ForInstruction):
            current = [wires[wire] for wire in operation.outputs]
            for index in reversed(operation.indexset):
                nested_variables = dict(loop_variables)
                nested_variables[operation.loop_variable.name] = index
                body_wires = _emit_inverse_region(
                    operation.body,
                    circuit,
                    dict(zip(operation.body_outputs, current, strict=True)),
                    parameters,
                    measurements,
                    nested_variables,
                    emitter,
                )
                current = [body_wires[wire] for wire in operation.inputs]
            _publish(operation.inputs, current, wires)
        elif isinstance(operation, BarrierInstruction):
            emitter.emit_barrier(circuit, [wires[wire] for wire in operation.wires])
        else:
            raise EmitError(
                "CUDA-Q explicit inverse fallback received a nonunitary or "
                f"dynamic instruction: {operation!r}"
            )
    return wires


def _program_global_phase(
    program: CircuitProgram,
    parameters: dict[str, Any],
    loop_variables: dict[str, int],
    emitter: CudaqKernelEmitter,
) -> Any:
    """Evaluate the phase omitted from one CUDA-Q helper definition.

    Pauli identity terms, static-loop phase, and uncontrolled nested-call
    phase are canonicalized into ``program.global_phase`` before target
    legalization. CUDA-Q can therefore retain a high-level ``cudaq.control``
    call and add exactly this one correction to its controls.

    Args:
        program (CircuitProgram): Reusable helper body.
        parameters (dict[str, Any]): Runtime parameter cache.
        loop_variables (dict[str, int]): Active concrete loop values.
        emitter (CudaqKernelEmitter): CUDA-Q scalar-expression builder.

    Returns:
        Any: Concrete or source-level phase angle omitted by the helper.
    """
    return _scalar(program.global_phase, parameters, loop_variables, emitter)


def _is_materialized_zero(value: Any) -> bool:
    """Return whether a materialized scalar is concretely zero.

    Args:
        value (Any): Concrete number or CUDA-Q source expression.

    Returns:
        bool: True only for a concrete numeric zero.
    """
    return isinstance(value, (int, float)) and not float(value)


def _scalar(
    expression: ScalarExpr,
    parameters: dict[str, Any],
    loop_variables: dict[str, int],
    emitter: CudaqKernelEmitter,
) -> Any:
    """Materialize a scalar as a Python value or CUDA-Q source expression.

    Args:
        expression (ScalarExpr): Target-neutral expression.
        parameters (dict[str, Any]): Runtime parameter cache.
        loop_variables (dict[str, int]): Concrete loop values.
        emitter (CudaqKernelEmitter): CUDA-Q parameter factory.

    Returns:
        Any: Concrete value or ``CudaqExpr``.

    Raises:
        EmitError: If a measurement bit appears outside a predicate.
    """
    if isinstance(expression, LiteralExpr):
        return expression.value
    if isinstance(expression, ParameterExpr):
        value = parameters.get(expression.name)
        if value is None:
            value = emitter.create_parameter(expression.name)
            parameters[expression.name] = value
        return value
    if isinstance(expression, LoopVariableExpr):
        try:
            return loop_variables[expression.name]
        except KeyError as error:
            raise EmitError(
                f"Unresolved CUDA-Q loop variable {expression.name!r}"
            ) from error
    if isinstance(expression, ClassicalBitExpr):
        raise EmitError("Measurement bits are only valid in CUDA-Q predicates")
    if isinstance(expression, UnaryExpr):
        operand = _scalar(expression.operand, parameters, loop_variables, emitter)
        if expression.operator is UnaryOperator.NEG:
            return -operand
        raise EmitError("Logical NOT is only valid in CUDA-Q predicates")
    if isinstance(expression, BinaryExpr):
        left = _scalar(expression.left, parameters, loop_variables, emitter)
        right = _scalar(expression.right, parameters, loop_variables, emitter)
        return _binary(expression.operator, left, right)
    raise EmitError(f"Unsupported CUDA-Q scalar expression: {expression!r}")


def _predicate(
    expression: ScalarExpr,
    parameters: dict[str, Any],
    loop_variables: dict[str, int],
    emitter: CudaqKernelEmitter,
) -> str:
    """Render a runtime predicate as CUDA-Q kernel source.

    Args:
        expression (ScalarExpr): Predicate expression.
        parameters (dict[str, Any]): Runtime parameter cache.
        loop_variables (dict[str, int]): Concrete loop values.
        emitter (CudaqKernelEmitter): CUDA-Q source builder.

    Returns:
        str: Python/CUDA-Q predicate source.
    """
    if isinstance(expression, ClassicalBitExpr):
        return emitter.clbit_ref(expression.index)
    if isinstance(expression, LiteralExpr):
        return repr(expression.value)
    if isinstance(expression, ParameterExpr):
        return str(_scalar(expression, parameters, loop_variables, emitter))
    if isinstance(expression, LoopVariableExpr):
        return repr(_scalar(expression, parameters, loop_variables, emitter))
    if isinstance(expression, UnaryExpr):
        if expression.operator is UnaryOperator.NOT:
            return f"not ({_predicate(expression.operand, parameters, loop_variables, emitter)})"
        return (
            f"-({_predicate(expression.operand, parameters, loop_variables, emitter)})"
        )
    if isinstance(expression, BinaryExpr):
        symbols = {
            BinaryOperator.ADD: "+",
            BinaryOperator.SUB: "-",
            BinaryOperator.MUL: "*",
            BinaryOperator.DIV: "/",
            BinaryOperator.FLOORDIV: "//",
            BinaryOperator.MOD: "%",
            BinaryOperator.POW: "**",
            BinaryOperator.EQ: "==",
            BinaryOperator.NEQ: "!=",
            BinaryOperator.LT: "<",
            BinaryOperator.LE: "<=",
            BinaryOperator.GT: ">",
            BinaryOperator.GE: ">=",
            BinaryOperator.AND: "and",
            BinaryOperator.OR: "or",
        }
        left = _predicate(expression.left, parameters, loop_variables, emitter)
        right = _predicate(expression.right, parameters, loop_variables, emitter)
        return f"({left}) {symbols[expression.operator]} ({right})"
    raise EmitError(f"Unsupported CUDA-Q predicate: {expression!r}")


def _binary(operator: BinaryOperator, left: Any, right: Any) -> Any:
    """Evaluate or compose one arithmetic binary operation.

    Args:
        operator (BinaryOperator): Arithmetic operation.
        left (Any): Left operand.
        right (Any): Right operand.

    Returns:
        Any: Concrete or symbolic result.

    Raises:
        EmitError: If the operator is not arithmetic.
    """
    functions = {
        BinaryOperator.ADD: lambda: left + right,
        BinaryOperator.SUB: lambda: left - right,
        BinaryOperator.MUL: lambda: left * right,
        BinaryOperator.DIV: lambda: left / right,
        BinaryOperator.FLOORDIV: lambda: left // right,
        BinaryOperator.MOD: lambda: left % right,
        BinaryOperator.POW: lambda: left**right,
    }
    try:
        return functions[operator]()
    except KeyError as error:
        raise EmitError(f"CUDA-Q angle cannot contain {operator.name}") from error


def _emit_gate(
    emitter: CudaqKernelEmitter,
    circuit: Any,
    kind: GateKind,
    slots: tuple[int, ...],
    angles: tuple[Any, ...],
) -> None:
    """Emit one primitive CUDA-Q gate.

    Args:
        emitter (CudaqKernelEmitter): CUDA-Q gate adapter.
        circuit (Any): Destination artifact.
        kind (GateKind): Primitive gate kind.
        slots (tuple[int, ...]): Physical qubit slots.
        angles (tuple[Any, ...]): Materialized angles.
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
        fixed[kind](circuit, *slots)
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
        rotations[kind](circuit, *slots, angles[0])  # type: ignore[call-arg]
        return
    raise EmitError(f"Unsupported CUDA-Q gate {kind.name}")


def _emit_pauli(
    operation: PauliEvolutionInstruction,
    circuit: Any,
    wires: dict[WireId, int],
    parameters: dict[str, Any],
    loop_variables: dict[str, int],
    emitter: CudaqKernelEmitter,
    controls: tuple[int, ...] = (),
    use_native: bool = False,
) -> None:
    """Emit Pauli evolution as native or controlled phase gadgets.

    Args:
        operation (PauliEvolutionInstruction): Abstract evolution.
        circuit (Any): Destination artifact.
        wires (dict[WireId, int]): Virtual wire mapping.
        parameters (dict[str, Any]): Runtime parameter cache.
        loop_variables (dict[str, int]): Concrete loop values.
        emitter (CudaqKernelEmitter): CUDA-Q source builder.
        controls (tuple[int, ...]): Optional accumulated controls.
        use_native (bool): Whether direct terms use CUDA-Q ``exp_pauli``.
            Defaults to ``False``.
    """
    import qamomile.observable as qm_o

    time = _scalar(operation.time, parameters, loop_variables, emitter)
    slots = tuple(wires[wire] for wire in operation.inputs)
    for operators, coefficient in operation.hamiltonian:
        if not operators or abs(coefficient) == 0:
            continue
        selected = [slots[item.index] for item in operators]
        angle = 2.0 * float(coefficient.real) * time
        if not controls and use_native:
            word = "".join(item.pauli.name for item in operators)
            emitter.emit_exp_pauli(circuit, selected, word, -0.5 * angle)
            continue
        for item, slot in zip(operators, selected, strict=True):
            if item.pauli is qm_o.Pauli.X:
                emitter.emit_h(circuit, slot)
            elif item.pauli is qm_o.Pauli.Y:
                emitter.emit_sdg(circuit, slot)
                emitter.emit_h(circuit, slot)
        for left, right in zip(selected, selected[1:]):
            emitter.emit_cx(circuit, left, right)
        emitter.emit_multi_controlled_rz(circuit, list(controls), selected[-1], angle)
        for left, right in reversed(list(zip(selected, selected[1:]))):
            emitter.emit_cx(circuit, left, right)
        for item, slot in reversed(list(zip(operators, selected, strict=True))):
            if item.pauli is qm_o.Pauli.X:
                emitter.emit_h(circuit, slot)
            elif item.pauli is qm_o.Pauli.Y:
                emitter.emit_h(circuit, slot)
                emitter.emit_s(circuit, slot)
    _publish(operation.outputs, slots, wires)


def _publish(
    outputs: tuple[WireId, ...],
    slots: tuple[int, ...] | list[int],
    wires: dict[WireId, int],
) -> None:
    """Publish virtual outputs on unchanged physical slots.

    Args:
        outputs (tuple[WireId, ...]): Produced wires.
        slots (tuple[int, ...] | list[int]): Corresponding physical slots.
        wires (dict[WireId, int]): Mapping to update.
    """
    for output, slot in zip(outputs, slots, strict=True):
        wires[output] = slot


def _has_runtime_control(operations: tuple[CircuitInstruction, ...]) -> bool:
    """Return whether a region requires CUDA-Q runnable mode.

    Args:
        operations (tuple[CircuitInstruction, ...]): Instructions to scan.

    Returns:
        bool: True when runtime if/while control is present.
    """
    for operation in operations:
        if isinstance(operation, (IfInstruction, WhileInstruction)):
            return True
        if isinstance(operation, ForInstruction) and _has_runtime_control(
            operation.body
        ):
            return True
    return False
