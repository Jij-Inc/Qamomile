"""Materialize backend-neutral circuit programs as Qiskit circuits."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.transpiler.circuit_ir import (
    ALL_PRIMITIVE_GATES,
    BarrierInstruction,
    BinaryExpr,
    BinaryOperator,
    CallInstruction,
    CallTransformCapabilities,
    CircuitCapabilities,
    CircuitInstruction,
    CircuitIntrinsic,
    CircuitProgram,
    ClassicalBitExpr,
    ForInstruction,
    GateInstruction,
    IfInstruction,
    LiteralExpr,
    LoopVariableExpr,
    MaterializedCircuit,
    MeasureInstruction,
    NativeIntrinsicCapabilities,
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


class QiskitMaterializer:
    """Convert target-legal circuit code-generation IR to ``QuantumCircuit``."""

    @property
    def capabilities(self) -> CircuitCapabilities:
        """Declare Qiskit's circuit-IR capabilities.

        Native intrinsic support is probed against the installed Qiskit at
        declaration time: ``QFTGate`` is only available on recent releases,
        so the declaration — not a materialize-time failure — tells
        legalization whether intrinsic calls may stay native.

        Returns:
            CircuitCapabilities: Immutable capability declaration.
        """
        native_intrinsics: tuple[NativeIntrinsicCapabilities, ...] = ()
        call_transforms = CallTransformCapabilities(
            supports_power=True,
            supports_inverse=True,
            max_controls=None,
        )
        try:
            from qiskit.circuit.library import QFTGate  # noqa: F401

            native_intrinsics = tuple(
                NativeIntrinsicCapabilities(intrinsic, call_transforms)
                for intrinsic in (CircuitIntrinsic.QFT, CircuitIntrinsic.IQFT)
            )
        except ImportError:
            pass
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
        predicates = ScalarCapabilities(
            atoms=frozenset({ScalarAtom.LITERAL, ScalarAtom.CLASSICAL_BIT}),
            unary_operators=frozenset({UnaryOperator.NOT}),
            binary_operators=frozenset(
                {
                    BinaryOperator.ADD,
                    BinaryOperator.SUB,
                    BinaryOperator.MUL,
                    BinaryOperator.DIV,
                    BinaryOperator.EQ,
                    BinaryOperator.NEQ,
                    BinaryOperator.LT,
                    BinaryOperator.LE,
                    BinaryOperator.GT,
                    BinaryOperator.GE,
                    BinaryOperator.AND,
                    BinaryOperator.OR,
                }
            ),
            parameter_form=ScalarExpressionForm.ARBITRARY,
        )
        return CircuitCapabilities(
            name="qiskit",
            primitive_gates=ALL_PRIMITIVE_GATES,
            native_intrinsics=native_intrinsics,
            gate_parameters=numeric,
            predicates=predicates,
            pauli_time=numeric,
            global_phase=numeric,
            generic_calls=call_transforms,
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

    def materialize(self, program: CircuitProgram) -> MaterializedCircuit[Any]:
        """Build and return a Qiskit circuit.

        Args:
            program (CircuitProgram): Verified circuit-family program.

        Returns:
            MaterializedCircuit[Any]: Qiskit circuit and parameter mapping.

        Raises:
            EmitError: If an instruction cannot be represented by Qiskit.
            ValueError: If circuit IR verification fails.
        """
        verify_circuit(program)
        artifact = self._materialize(program, preserve_control_flow=True)
        return MaterializedCircuit(
            artifact=artifact,
            parameters={parameter.name: parameter for parameter in artifact.parameters},
        )

    def _materialize(
        self,
        program: CircuitProgram,
        preserve_control_flow: bool,
    ) -> Any:
        """Build a Qiskit circuit with native or unrolled loop regions.

        Args:
            program (CircuitProgram): Verified circuit program.
            preserve_control_flow (bool): Whether concrete loops remain native.
                Reusable gate bodies disable this because Qiskit gates cannot
                contain control-flow instructions.

        Returns:
            Any: Materialized ``QuantumCircuit``.
        """
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(
            program.num_qubits,
            program.num_clbits,
            name=program.name,
        )
        wires = {
            wire: circuit.qubits[index]
            for index, wire in enumerate(program.input_wires)
        }
        parameters: dict[str, Any] = {}
        _emit_region(
            program.operations,
            circuit,
            wires,
            parameters,
            {},
            preserve_control_flow,
        )
        circuit.global_phase += _materialize_scalar(
            program.global_phase,
            circuit,
            parameters,
            {},
        )
        return circuit


def _emit_region(
    operations: tuple[CircuitInstruction, ...],
    circuit: Any,
    input_wires: dict[WireId, Any],
    parameters: dict[str, Any],
    loop_variables: dict[str, Any],
    preserve_control_flow: bool,
) -> dict[WireId, Any]:
    """Emit one structured circuit region.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region instructions.
        circuit (Any): Destination ``QuantumCircuit`` or control-flow scope.
        input_wires (dict[WireId, Any]): Region input wire mapping.
        parameters (dict[str, Any]): Shared Qiskit parameter cache.
        loop_variables (dict[str, Any]): Active induction-variable mapping.
        preserve_control_flow (bool): Whether concrete loops remain native.

    Returns:
        dict[WireId, Any]: Mapping containing every produced virtual wire.

    Raises:
        EmitError: If an instruction or expression is unsupported.
    """
    wires = dict(input_wires)
    for operation in operations:
        if isinstance(operation, GateInstruction):
            qubits = tuple(wires[wire] for wire in operation.inputs)
            arguments = tuple(
                _materialize_scalar(
                    parameter,
                    circuit,
                    parameters,
                    loop_variables,
                )
                for parameter in operation.parameters
            )
            _emit_gate(circuit, operation.kind, qubits, arguments)
            _publish_wires(operation.outputs, qubits, wires)
        elif isinstance(operation, MeasureInstruction):
            qubit = wires[operation.input]
            circuit.measure(qubit, circuit.clbits[operation.clbit])
            wires[operation.output] = qubit
        elif isinstance(operation, ResetInstruction):
            qubit = wires[operation.input]
            try:
                circuit.reset(qubit)
            except (NotImplementedError, TypeError, ValueError) as error:
                raise EmitError(
                    "This backend cannot emit a qubit reset",
                    operation="ResetInstruction",
                ) from error
            wires[operation.output] = qubit
        elif isinstance(operation, BarrierInstruction):
            circuit.barrier(*(wires[wire] for wire in operation.wires))
        elif isinstance(operation, PauliEvolutionInstruction):
            if operation.realization is PauliEvolutionRealization.NATIVE:
                _emit_pauli_evolution(
                    operation,
                    circuit,
                    wires,
                    parameters,
                    loop_variables,
                )
            elif operation.realization is PauliEvolutionRealization.GADGET:
                _emit_pauli_evolution_gadget(
                    operation,
                    circuit,
                    wires,
                    parameters,
                    loop_variables,
                )
            else:  # pragma: no cover - target verification owns this invariant
                raise AssertionError("Pauli evolution was not legalized")
        elif isinstance(operation, CallInstruction):
            _emit_call(
                operation,
                circuit,
                wires,
                parameters,
            )
        elif isinstance(operation, ForInstruction):
            _emit_for(
                operation,
                circuit,
                wires,
                parameters,
                loop_variables,
                preserve_control_flow,
            )
        elif isinstance(operation, IfInstruction):
            _emit_if(
                operation,
                circuit,
                wires,
                parameters,
                loop_variables,
                preserve_control_flow,
            )
        elif isinstance(operation, WhileInstruction):
            _emit_while(
                operation,
                circuit,
                wires,
                parameters,
                loop_variables,
                preserve_control_flow,
            )
        else:  # pragma: no cover - defensive guard for a closed union
            raise EmitError(f"Unsupported Qiskit circuit instruction: {operation!r}")
    return wires


def _publish_wires(
    outputs: tuple[WireId, ...],
    qubits: tuple[Any, ...] | list[Any],
    wires: dict[WireId, Any],
) -> None:
    """Bind produced virtual wires to their physical Qiskit qubits.

    Args:
        outputs (tuple[WireId, ...]): Produced virtual wires.
        qubits (tuple[Any, ...] | list[Any]): Corresponding Qiskit qubits.
        wires (dict[WireId, Any]): Mapping to update.
    """
    for output, qubit in zip(outputs, qubits, strict=True):
        wires[output] = qubit


def _materialize_scalar(
    expression: ScalarExpr,
    circuit: Any,
    parameters: dict[str, Any],
    loop_variables: dict[str, Any],
) -> Any:
    """Convert a target-neutral scalar expression to a Qiskit value.

    Args:
        expression (ScalarExpr): Expression to materialize.
        circuit (Any): Circuit owning referenced classical bits.
        parameters (dict[str, Any]): Parameter cache keyed by public name.
        loop_variables (dict[str, Any]): Active induction-variable mapping.

    Returns:
        Any: Python literal, Qiskit parameter expression, loop parameter, or
            Qiskit classical expression.

    Raises:
        EmitError: If a loop variable is unresolved or an operation has no
            Qiskit representation.
    """
    if isinstance(expression, LiteralExpr):
        return expression.value
    if isinstance(expression, ParameterExpr):
        from qiskit.circuit import Parameter

        return parameters.setdefault(expression.name, Parameter(expression.name))
    if isinstance(expression, ClassicalBitExpr):
        return circuit.clbits[expression.index]
    if isinstance(expression, LoopVariableExpr):
        try:
            return loop_variables[expression.name]
        except KeyError as error:
            raise EmitError(
                f"Unresolved Qiskit loop variable {expression.name!r}"
            ) from error
    if isinstance(expression, UnaryExpr):
        operand = _materialize_scalar(
            expression.operand,
            circuit,
            parameters,
            loop_variables,
        )
        if expression.operator is UnaryOperator.NEG:
            return -operand
        if expression.operator is UnaryOperator.NOT:
            from qiskit.circuit.classical import expr

            return expr.logic_not(operand)
    if isinstance(expression, BinaryExpr):
        left = _materialize_scalar(
            expression.left,
            circuit,
            parameters,
            loop_variables,
        )
        right = _materialize_scalar(
            expression.right,
            circuit,
            parameters,
            loop_variables,
        )
        return _materialize_binary(expression.operator, left, right)
    raise EmitError(f"Unsupported Qiskit scalar expression: {expression!r}")


def _materialize_binary(operator: BinaryOperator, left: Any, right: Any) -> Any:
    """Apply one scalar operator using Qiskit or Python arithmetic.

    Args:
        operator (BinaryOperator): Scalar operation kind.
        left (Any): Materialized left operand.
        right (Any): Materialized right operand.

    Returns:
        Any: Materialized result expression.

    Raises:
        EmitError: If Qiskit has no corresponding runtime operation.
    """
    from qiskit.circuit.classical import expr

    functions = {
        BinaryOperator.ADD: expr.add,
        BinaryOperator.SUB: expr.sub,
        BinaryOperator.MUL: expr.mul,
        BinaryOperator.DIV: expr.div,
        BinaryOperator.EQ: expr.equal,
        BinaryOperator.NEQ: expr.not_equal,
        BinaryOperator.LT: expr.less,
        BinaryOperator.LE: expr.less_equal,
        BinaryOperator.GT: expr.greater,
        BinaryOperator.GE: expr.greater_equal,
        BinaryOperator.AND: expr.logic_and,
        BinaryOperator.OR: expr.logic_or,
    }
    if operator in functions:
        try:
            return functions[operator](left, right)
        except (TypeError, ValueError):
            pass

    python_functions = {
        BinaryOperator.ADD: lambda: left + right,
        BinaryOperator.SUB: lambda: left - right,
        BinaryOperator.MUL: lambda: left * right,
        BinaryOperator.DIV: lambda: left / right,
        BinaryOperator.FLOORDIV: lambda: left // right,
        BinaryOperator.MOD: lambda: left % right,
        BinaryOperator.POW: lambda: left**right,
    }
    try:
        return python_functions[operator]()
    except (KeyError, TypeError, ValueError) as error:
        raise EmitError(
            f"Unsupported Qiskit binary operator: {operator.value}"
        ) from error


def _condition(
    expression: ScalarExpr,
    circuit: Any,
    parameters: dict[str, Any],
    loop_variables: dict[str, Any],
) -> Any:
    """Build a condition accepted by Qiskit control-flow contexts.

    Args:
        expression (ScalarExpr): Target-neutral predicate.
        circuit (Any): Circuit owning referenced classical bits.
        parameters (dict[str, Any]): Parameter cache.
        loop_variables (dict[str, Any]): Active loop variables.

    Returns:
        Any: Qiskit condition tuple or classical expression.
    """
    if isinstance(expression, ClassicalBitExpr):
        return (circuit.clbits[expression.index], 1)
    return _materialize_scalar(expression, circuit, parameters, loop_variables)


def _emit_for(
    operation: ForInstruction,
    circuit: Any,
    wires: dict[WireId, Any],
    parameters: dict[str, Any],
    loop_variables: dict[str, Any],
    preserve_control_flow: bool,
) -> None:
    """Materialize a structured Qiskit for-loop.

    Args:
        operation (ForInstruction): Loop instruction.
        circuit (Any): Destination circuit.
        wires (dict[WireId, Any]): Enclosing wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        loop_variables (dict[str, Any]): Active loop-variable mapping.
        preserve_control_flow (bool): Whether to retain native loop structure.
    """
    current = [wires[wire] for wire in operation.inputs]
    if preserve_control_flow:
        with circuit.for_loop(operation.indexset) as loop_parameter:
            nested_variables = dict(loop_variables)
            nested_variables[operation.loop_variable.name] = loop_parameter
            body_inputs = dict(zip(operation.inputs, current, strict=True))
            body_wires = _emit_region(
                operation.body,
                circuit,
                body_inputs,
                parameters,
                nested_variables,
                preserve_control_flow,
            )
        current = [body_wires[wire] for wire in operation.body_outputs]
    else:
        for index in operation.indexset:
            nested_variables = dict(loop_variables)
            nested_variables[operation.loop_variable.name] = index
            body_inputs = dict(zip(operation.inputs, current, strict=True))
            body_wires = _emit_region(
                operation.body,
                circuit,
                body_inputs,
                parameters,
                nested_variables,
                preserve_control_flow,
            )
            current = [body_wires[wire] for wire in operation.body_outputs]
    _publish_wires(
        operation.outputs,
        current,
        wires,
    )


def _emit_if(
    operation: IfInstruction,
    circuit: Any,
    wires: dict[WireId, Any],
    parameters: dict[str, Any],
    loop_variables: dict[str, Any],
    preserve_control_flow: bool,
) -> None:
    """Materialize a structured Qiskit conditional.

    Args:
        operation (IfInstruction): Conditional instruction.
        circuit (Any): Destination circuit.
        wires (dict[WireId, Any]): Enclosing wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        loop_variables (dict[str, Any]): Active loop variables.
        preserve_control_flow (bool): Whether nested loops remain native.
    """
    condition = _condition(
        operation.condition,
        circuit,
        parameters,
        loop_variables,
    )
    branch_inputs = {wire: wires[wire] for wire in operation.inputs}
    with circuit.if_test(condition) as else_context:
        true_wires = _emit_region(
            operation.true_body,
            circuit,
            branch_inputs,
            parameters,
            loop_variables,
            preserve_control_flow,
        )
    with else_context:
        false_wires = _emit_region(
            operation.false_body,
            circuit,
            branch_inputs,
            parameters,
            loop_variables,
            preserve_control_flow,
        )
    true_qubits = [true_wires[wire] for wire in operation.true_outputs]
    false_qubits = [false_wires[wire] for wire in operation.false_outputs]
    if true_qubits != false_qubits:
        raise EmitError("Qiskit conditional branches change physical wire layout")
    _publish_wires(operation.outputs, true_qubits, wires)


def _emit_while(
    operation: WhileInstruction,
    circuit: Any,
    wires: dict[WireId, Any],
    parameters: dict[str, Any],
    loop_variables: dict[str, Any],
    preserve_control_flow: bool,
) -> None:
    """Materialize a structured Qiskit while-loop.

    Args:
        operation (WhileInstruction): While-loop instruction.
        circuit (Any): Destination circuit.
        wires (dict[WireId, Any]): Enclosing wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        loop_variables (dict[str, Any]): Active loop variables.
        preserve_control_flow (bool): Whether nested loops remain native.
    """
    condition = _condition(
        operation.condition,
        circuit,
        parameters,
        loop_variables,
    )
    body_inputs = {wire: wires[wire] for wire in operation.inputs}
    with circuit.while_loop(condition):
        body_wires = _emit_region(
            operation.body,
            circuit,
            body_inputs,
            parameters,
            loop_variables,
            preserve_control_flow,
        )
    _publish_wires(
        operation.outputs,
        [body_wires[wire] for wire in operation.body_outputs],
        wires,
    )


def _emit_call(
    operation: CallInstruction,
    circuit: Any,
    wires: dict[WireId, Any],
    parameters: dict[str, Any],
) -> None:
    """Materialize a reusable circuit call as a Qiskit gate.

    Args:
        operation (CallInstruction): Reusable circuit invocation.
        circuit (Any): Destination circuit.
        wires (dict[WireId, Any]): Enclosing wire mapping.
        parameters (dict[str, Any]): Shared parameter cache.

    Raises:
        EmitError: If the reusable body is not unitary or has incompatible
            arity.
    """
    callee = operation.callee
    identity = callee.identity
    if identity is not None and identity.intrinsic is not None:
        _emit_native_intrinsic(operation, circuit, wires)
        return
    if callee.body.num_clbits:
        raise EmitError("A measured circuit cannot be materialized as a Qiskit gate")
    nested = QiskitMaterializer()._materialize(
        callee.body,
        preserve_control_flow=False,
    )
    replacements: dict[Any, Any] = {}
    for parameter in nested.parameters:
        shared = parameters.setdefault(parameter.name, parameter)
        if shared is not parameter:
            replacements[parameter] = shared
    if replacements:
        nested = nested.assign_parameters(replacements)
    try:
        gate = nested.to_gate(label=callee.name)
        if callee.power != 1:
            gate = gate.power(callee.power)
        if callee.inverse:
            gate = gate.inverse()
        if callee.controls:
            gate = gate.control(callee.controls)
    except Exception as error:
        raise EmitError(
            f"Reusable circuit {callee.name!r} cannot become a Qiskit gate"
        ) from error
    qubits = [wires[wire] for wire in operation.inputs]
    if gate.num_qubits != len(qubits):
        raise EmitError(
            f"Reusable circuit {callee.name!r} expects {gate.num_qubits} "
            f"qubits but received {len(qubits)}"
        )
    circuit.append(gate, qubits)
    _publish_wires(operation.outputs, qubits, wires)


def _emit_native_intrinsic(
    operation: CallInstruction,
    circuit: Any,
    wires: dict[WireId, Any],
) -> None:
    """Materialize an intrinsic-tagged call with Qiskit's native library.

    Target verification guarantees the intrinsic is declared native before
    this runs, so the body is intentionally ignored: the intrinsic identity
    is the meaning, and the native gate is Qiskit's realization of it.

    Args:
        operation (CallInstruction): Intrinsic-tagged reusable call.
        circuit (Any): Destination Qiskit circuit.
        wires (dict[WireId, Any]): Virtual-to-Qiskit wire mapping.

    Raises:
        EmitError: If the intrinsic kind has no native Qiskit realization or
            the requested transforms cannot be applied.
    """
    from qiskit.circuit.library import QFTGate

    callee = operation.callee
    identity = callee.identity
    assert identity is not None and identity.intrinsic is not None
    if identity.intrinsic not in (CircuitIntrinsic.QFT, CircuitIntrinsic.IQFT):
        raise EmitError(
            f"Qiskit has no native realization for intrinsic {identity.intrinsic.name}"
        )
    gate: Any = QFTGate(callee.body.num_qubits)
    if identity.intrinsic is CircuitIntrinsic.IQFT:
        gate = gate.inverse(annotated=True)
    try:
        if callee.power != 1:
            gate = gate.power(callee.power)
        if callee.inverse:
            gate = gate.inverse()
        if callee.controls:
            gate = gate.control(callee.controls)
    except Exception as error:
        raise EmitError(
            f"Native intrinsic {identity.intrinsic.name} cannot apply the "
            f"requested transforms (power={callee.power}, "
            f"inverse={callee.inverse}, controls={callee.controls})"
        ) from error
    qubits = [wires[wire] for wire in operation.inputs]
    circuit.append(gate, qubits)
    _publish_wires(operation.outputs, qubits, wires)


def _emit_pauli_evolution(
    operation: PauliEvolutionInstruction,
    circuit: Any,
    wires: dict[WireId, Any],
    parameters: dict[str, Any],
    loop_variables: dict[str, Any],
) -> None:
    """Materialize abstract Pauli evolution with Qiskit's native gate.

    Args:
        operation (PauliEvolutionInstruction): Evolution instruction.
        circuit (Any): Destination Qiskit circuit.
        wires (dict[WireId, Any]): Virtual-to-Qiskit wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        loop_variables (dict[str, Any]): Active induction variables.
    """
    from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op
    from qiskit.circuit.library import PauliEvolutionGate

    time = _materialize_scalar(
        operation.time,
        circuit,
        parameters,
        loop_variables,
    )
    qubits = tuple(wires[wire] for wire in operation.inputs)
    if not qubits:
        circuit.global_phase += -float(operation.hamiltonian.constant.real) * time
        return
    observable = hamiltonian_to_sparse_pauli_op(operation.hamiltonian)
    circuit.append(PauliEvolutionGate(observable, time=time), qubits)
    _publish_wires(operation.outputs, qubits, wires)


def _emit_pauli_evolution_gadget(
    operation: PauliEvolutionInstruction,
    circuit: Any,
    wires: dict[WireId, Any],
    parameters: dict[str, Any],
    loop_variables: dict[str, Any],
) -> None:
    """Legalize Pauli evolution to basis changes and phase gadgets.

    Args:
        operation (PauliEvolutionInstruction): Abstract evolution instruction.
        circuit (Any): Destination Qiskit circuit.
        wires (dict[WireId, Any]): Virtual-to-Qiskit wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        loop_variables (dict[str, Any]): Active induction variables.
    """
    import qamomile.observable as qm_o

    time = _materialize_scalar(
        operation.time,
        circuit,
        parameters,
        loop_variables,
    )
    qubits = tuple(wires[wire] for wire in operation.inputs)
    for operators, coefficient in operation.hamiltonian:
        selected = [qubits[item.index] for item in operators]
        for item, qubit in zip(operators, selected, strict=True):
            if item.pauli is qm_o.Pauli.X:
                circuit.h(qubit)
            elif item.pauli is qm_o.Pauli.Y:
                circuit.sdg(qubit)
                circuit.h(qubit)
        for left, right in zip(selected, selected[1:]):
            circuit.cx(left, right)
        if selected:
            circuit.rz(2.0 * float(coefficient.real) * time, selected[-1])
        for left, right in reversed(list(zip(selected, selected[1:]))):
            circuit.cx(left, right)
        for item, qubit in reversed(list(zip(operators, selected, strict=True))):
            if item.pauli is qm_o.Pauli.X:
                circuit.h(qubit)
            elif item.pauli is qm_o.Pauli.Y:
                circuit.h(qubit)
                circuit.s(qubit)
    circuit.global_phase += -float(operation.hamiltonian.constant.real) * time
    _publish_wires(operation.outputs, qubits, wires)


def _emit_gate(
    circuit: Any,
    kind: GateKind,
    qubits: tuple[Any, ...],
    parameters: tuple[Any, ...],
) -> None:
    """Append one primitive Qiskit gate.

    Args:
        circuit (Any): Destination circuit.
        kind (GateKind): Primitive operation kind.
        qubits (tuple[Any, ...]): Participating Qiskit qubits.
        parameters (tuple[Any, ...]): Materialized gate parameters.

    Raises:
        EmitError: If ``kind`` is unknown.
    """
    fixed = {
        GateKind.H: circuit.h,
        GateKind.X: circuit.x,
        GateKind.Y: circuit.y,
        GateKind.Z: circuit.z,
        GateKind.S: circuit.s,
        GateKind.SDG: circuit.sdg,
        GateKind.T: circuit.t,
        GateKind.TDG: circuit.tdg,
        GateKind.CX: circuit.cx,
        GateKind.CY: circuit.cy,
        GateKind.CZ: circuit.cz,
        GateKind.SWAP: circuit.swap,
        GateKind.CH: circuit.ch,
        GateKind.TOFFOLI: circuit.ccx,
    }
    if kind in fixed:
        fixed[kind](*qubits)
        return
    rotations = {
        GateKind.RX: circuit.rx,
        GateKind.RY: circuit.ry,
        GateKind.RZ: circuit.rz,
        GateKind.P: circuit.p,
        GateKind.CP: circuit.cp,
        GateKind.RZZ: circuit.rzz,
        GateKind.CRX: circuit.crx,
        GateKind.CRY: circuit.cry,
        GateKind.CRZ: circuit.crz,
    }
    if kind in rotations and len(parameters) == 1:
        rotations[kind](parameters[0], *qubits)
        return
    raise EmitError(
        f"Unsupported Qiskit gate instruction {kind.name} with "
        f"{len(parameters)} parameters"
    )
