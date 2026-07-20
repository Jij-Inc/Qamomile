"""Materialize backend-neutral circuit programs as Qiskit circuits."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.transpiler.circuit_ir import (
    ALL_PRIMITIVE_GATES,
    IQFT_SEMANTIC_KEY,
    MULTI_CONTROLLED_X_SEMANTIC_KEY,
    QFT_SEMANTIC_KEY,
    RIPPLE_CARRY_ADD_SEMANTIC_KEY,
    STATE_PREPARATION_SEMANTIC_KEY,
    BarrierInstruction,
    BinaryExpr,
    BinaryOperator,
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
    NativeSemanticOpCapabilities,
    ParameterExpr,
    PauliEvolutionInstruction,
    PauliEvolutionRealization,
    ResetInstruction,
    ReusableCircuit,
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
    """Convert target-legal circuit code-generation IR to ``QuantumCircuit``.

    Root and runtime-control-flow phases are retained in Qiskit's native
    ``QuantumCircuit.global_phase`` metadata on the corresponding circuit or
    block.
    """

    @property
    def capabilities(self) -> CircuitCapabilities:
        """Declare Qiskit's circuit-IR capabilities.

        Native semantic-operation support is probed against the installed
        Qiskit at declaration time, so the declaration rather than a late
        materialization failure determines whether calls may stay native.

        Returns:
            CircuitCapabilities: Immutable capability declaration.
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
        native_semantic_ops: tuple[NativeSemanticOpCapabilities, ...] = ()
        call_transforms = CallTransformCapabilities(
            supports_power=True,
            supports_inverse=True,
            max_controls=None,
            phase_mode=CallPhaseMode.NATIVE_BODY,
            controlled_phase_scalars=numeric,
        )
        native_call_transforms = CallTransformCapabilities(
            supports_power=True,
            supports_inverse=True,
            max_controls=None,
        )
        direct_only = CallTransformCapabilities(
            supports_power=False,
            supports_inverse=False,
            max_controls=0,
        )
        try:
            from qiskit.circuit.library import (  # noqa: F401
                FullAdderGate,
                MCXGate,
                QFTGate,
                StatePreparation,
            )

            native_semantic_ops = (
                NativeSemanticOpCapabilities(
                    QFT_SEMANTIC_KEY,
                    "qiskit.qft",
                    native_call_transforms,
                    operand_widths=(None,),
                    min_qubits=1,
                ),
                NativeSemanticOpCapabilities(
                    IQFT_SEMANTIC_KEY,
                    "qiskit.iqft",
                    native_call_transforms,
                    operand_widths=(None,),
                    min_qubits=1,
                ),
                NativeSemanticOpCapabilities(
                    STATE_PREPARATION_SEMANTIC_KEY,
                    "qiskit.state_preparation",
                    direct_only,
                    operand_widths=(None,),
                    min_qubits=1,
                    required_arguments=frozenset({"amplitudes"}),
                ),
                NativeSemanticOpCapabilities(
                    RIPPLE_CARRY_ADD_SEMANTIC_KEY,
                    "qiskit.ripple_carry_add",
                    direct_only,
                    operand_widths=(None, None, 1, 1),
                    min_qubits=4,
                    matching_operand_widths=((0, 1),),
                ),
                NativeSemanticOpCapabilities(
                    MULTI_CONTROLLED_X_SEMANTIC_KEY,
                    "qiskit.multi_controlled_x",
                    direct_only,
                    operand_widths=(None, 1),
                    min_qubits=2,
                ),
            )
        except ImportError:
            pass
        return CircuitCapabilities(
            name="qiskit",
            primitive_gates=ALL_PRIMITIVE_GATES,
            native_semantic_ops=native_semantic_ops,
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

    def materialize(
        self,
        program: CircuitProgram,
        parameter_names: tuple[str, ...] = (),
    ) -> MaterializedCircuit[Any]:
        """Build and return a Qiskit circuit with scoped phase metadata.

        Args:
            program (CircuitProgram): Verified circuit-family program.
            parameter_names (tuple[str, ...]): Public parameter ABI. Qiskit
                binds by name, so this is used for boundary validation only.
                Defaults to an empty tuple.

        Returns:
            MaterializedCircuit[Any]: Qiskit circuit and parameter mapping.

        Raises:
            EmitError: If an instruction cannot be represented by Qiskit.
            ValueError: If circuit IR verification fails.
        """
        verify_circuit(program)
        artifact = self._materialize(program, preserve_control_flow=True)
        # Qiskit represents a native for-loop induction value as an internal
        # ``Parameter`` (for example ``_loop_i_0``). It is part of
        # ``artifact.parameters`` but not of Qamomile's public runtime ABI and
        # must never be exposed as an executor binding. Keep only names the
        # compiled segment explicitly declared.
        parameters = {
            parameter.name: parameter
            for parameter in artifact.parameters
            if parameter.name in parameter_names
        }
        if missing_names := set(parameter_names) - parameters.keys():
            from qiskit.circuit import Parameter

            for name in parameter_names:
                if name not in missing_names:
                    continue
                parameter = Parameter(name)
                # Qiskit has no standalone parameter declaration. A zero
                # global-phase coefficient preserves the public ABI without
                # changing the circuit's exact unitary.
                artifact.global_phase += 0 * parameter
                parameters[name] = parameter
        return MaterializedCircuit(
            artifact=artifact,
            parameters=parameters,
        )

    def _materialize(
        self,
        program: CircuitProgram,
        preserve_control_flow: bool,
        parameters: dict[str, Any] | None = None,
        reusable_gates: dict[ReusableCircuit, Any] | None = None,
    ) -> Any:
        """Build a Qiskit circuit with native or unrolled loop regions.

        Args:
            program (CircuitProgram): Verified circuit program.
            preserve_control_flow (bool): Whether concrete loops remain native.
                Reusable gate bodies disable this because Qiskit gates cannot
                contain control-flow instructions.
            parameters (dict[str, Any] | None): Parameter cache shared with an
                enclosing reusable circuit. Reusing the same Qiskit Parameter
                identities through every nested definition keeps recursive
                ``assign_parameters`` reliable. Defaults to a fresh cache.
            reusable_gates (dict[ReusableCircuit, Any] | None): Gate cache
                shared across one root materialization. Defaults to a fresh
                cache.

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
        parameter_cache = {} if parameters is None else parameters
        reusable_gate_cache = {} if reusable_gates is None else reusable_gates
        _emit_region(
            program.operations,
            circuit,
            wires,
            parameter_cache,
            {},
            preserve_control_flow,
            reusable_gate_cache,
        )
        circuit.global_phase += _materialize_scalar(
            program.global_phase,
            circuit,
            parameter_cache,
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
    reusable_gates: dict[ReusableCircuit, Any],
) -> dict[WireId, Any]:
    """Emit one structured circuit region.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region instructions.
        circuit (Any): Destination ``QuantumCircuit`` or control-flow scope.
        input_wires (dict[WireId, Any]): Region input wire mapping.
        parameters (dict[str, Any]): Shared Qiskit parameter cache.
        loop_variables (dict[str, Any]): Active induction-variable mapping.
        preserve_control_flow (bool): Whether concrete loops remain native.
        reusable_gates (dict[ReusableCircuit, Any]): Materialized gate cache
            scoped to the root circuit.

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
        elif isinstance(operation, MeasureVectorInstruction):
            qubits = tuple(wires[wire] for wire in operation.inputs)
            clbits = tuple(circuit.clbits[index] for index in operation.clbits)
            circuit.measure(qubits, clbits)
            _publish_wires(operation.outputs, qubits, wires)
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
                reusable_gates,
            )
        elif isinstance(operation, ForInstruction):
            _emit_for(
                operation,
                circuit,
                wires,
                parameters,
                loop_variables,
                preserve_control_flow,
                reusable_gates,
            )
        elif isinstance(operation, IfInstruction):
            _emit_if(
                operation,
                circuit,
                wires,
                parameters,
                loop_variables,
                preserve_control_flow,
                reusable_gates,
            )
        elif isinstance(operation, WhileInstruction):
            _emit_while(
                operation,
                circuit,
                wires,
                parameters,
                loop_variables,
                preserve_control_flow,
                reusable_gates,
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
    python_functions = {
        BinaryOperator.ADD: lambda: left + right,
        BinaryOperator.SUB: lambda: left - right,
        BinaryOperator.MUL: lambda: left * right,
        BinaryOperator.DIV: lambda: left / right,
        BinaryOperator.FLOORDIV: lambda: left // right,
        BinaryOperator.MOD: lambda: left % right,
        BinaryOperator.POW: lambda: left**right,
        BinaryOperator.EQ: lambda: left == right,
        BinaryOperator.NEQ: lambda: left != right,
        BinaryOperator.LT: lambda: left < right,
        BinaryOperator.LE: lambda: left <= right,
        BinaryOperator.GT: lambda: left > right,
        BinaryOperator.GE: lambda: left >= right,
        BinaryOperator.AND: lambda: bool(left) and bool(right),
        BinaryOperator.OR: lambda: bool(left) or bool(right),
    }
    if isinstance(left, (bool, int, float)) and isinstance(
        right,
        (bool, int, float),
    ):
        try:
            return python_functions[operator]()
        except (KeyError, TypeError, ValueError) as error:
            raise EmitError(
                f"Unsupported Qiskit binary operator: {operator.value}"
            ) from error

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
            mixed_comparison = _materialize_bool_uint_comparison(
                operator,
                left,
                right,
            )
            if mixed_comparison is not None:
                return mixed_comparison

    try:
        return python_functions[operator]()
    except (KeyError, TypeError, ValueError) as error:
        raise EmitError(
            f"Unsupported Qiskit binary operator: {operator.value}"
        ) from error


def _materialize_bool_uint_comparison(
    operator: BinaryOperator,
    left: Any,
    right: Any,
) -> Any | None:
    """Materialize mixed Qiskit Bool and Uint equality at the target boundary.

    Qiskit does not directly compare its classical ``Bool`` and ``Uint``
    types. Preserve the single abstract comparison through Qamomile's shared
    circuit IR, then express equality using the Boolean value's numeric domain
    only when Qiskit rejects the direct operation.

    Args:
        operator (BinaryOperator): Requested equality operation.
        left (Any): Materialized left operand.
        right (Any): Materialized right operand.

    Returns:
        Any | None: Qiskit classical expression for mixed equality, or
            ``None`` when the operands or operator do not match this case.
    """
    if operator not in {BinaryOperator.EQ, BinaryOperator.NEQ}:
        return None

    from qiskit.circuit.classical import expr, types

    try:
        left_expression = expr.lift(left)
        right_expression = expr.lift(right)
    except (TypeError, ValueError):
        return None
    if isinstance(left_expression.type, types.Bool) and isinstance(
        right_expression.type, types.Uint
    ):
        bit_expression = left_expression
        uint_expression = right_expression
    elif isinstance(left_expression.type, types.Uint) and isinstance(
        right_expression.type, types.Bool
    ):
        uint_expression = left_expression
        bit_expression = right_expression
    else:
        return None

    equality = expr.logic_or(
        expr.logic_and(
            expr.logic_not(bit_expression),
            expr.equal(uint_expression, 0),
        ),
        expr.logic_and(
            bit_expression,
            expr.equal(uint_expression, 1),
        ),
    )
    if operator is BinaryOperator.NEQ:
        return expr.logic_not(equality)
    return equality


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
    reusable_gates: dict[ReusableCircuit, Any],
) -> None:
    """Materialize a structured Qiskit for-loop.

    Args:
        operation (ForInstruction): Loop instruction.
        circuit (Any): Destination circuit.
        wires (dict[WireId, Any]): Enclosing wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        loop_variables (dict[str, Any]): Active loop-variable mapping.
        preserve_control_flow (bool): Whether to retain native loop structure.
        reusable_gates (dict[ReusableCircuit, Any]): Materialized gate cache.
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
                reusable_gates,
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
                reusable_gates,
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
    reusable_gates: dict[ReusableCircuit, Any],
) -> None:
    """Materialize a structured Qiskit conditional.

    Args:
        operation (IfInstruction): Conditional instruction.
        circuit (Any): Destination circuit.
        wires (dict[WireId, Any]): Enclosing wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        loop_variables (dict[str, Any]): Active loop variables.
        preserve_control_flow (bool): Whether nested loops remain native.
        reusable_gates (dict[ReusableCircuit, Any]): Materialized gate cache.
    """
    condition = _condition(
        operation.condition,
        circuit,
        parameters,
        loop_variables,
    )
    branch_inputs = {wire: wires[wire] for wire in operation.inputs}
    with circuit.if_test(condition) as else_context:
        circuit.global_phase += _materialize_scalar(
            operation.true_global_phase,
            circuit,
            parameters,
            loop_variables,
        )
        true_wires = _emit_region(
            operation.true_body,
            circuit,
            branch_inputs,
            parameters,
            loop_variables,
            preserve_control_flow,
            reusable_gates,
        )
    with else_context:
        circuit.global_phase += _materialize_scalar(
            operation.false_global_phase,
            circuit,
            parameters,
            loop_variables,
        )
        false_wires = _emit_region(
            operation.false_body,
            circuit,
            branch_inputs,
            parameters,
            loop_variables,
            preserve_control_flow,
            reusable_gates,
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
    reusable_gates: dict[ReusableCircuit, Any],
) -> None:
    """Materialize a structured Qiskit while-loop.

    Args:
        operation (WhileInstruction): While-loop instruction.
        circuit (Any): Destination circuit.
        wires (dict[WireId, Any]): Enclosing wire mapping.
        parameters (dict[str, Any]): Parameter cache.
        loop_variables (dict[str, Any]): Active loop variables.
        preserve_control_flow (bool): Whether nested loops remain native.
        reusable_gates (dict[ReusableCircuit, Any]): Materialized gate cache.
    """
    condition = _condition(
        operation.condition,
        circuit,
        parameters,
        loop_variables,
    )
    body_inputs = {wire: wires[wire] for wire in operation.inputs}
    with circuit.while_loop(condition):
        circuit.global_phase += _materialize_scalar(
            operation.body_global_phase,
            circuit,
            parameters,
            loop_variables,
        )
        body_wires = _emit_region(
            operation.body,
            circuit,
            body_inputs,
            parameters,
            loop_variables,
            preserve_control_flow,
            reusable_gates,
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
    reusable_gates: dict[ReusableCircuit, Any],
) -> None:
    """Materialize a reusable circuit call as a Qiskit gate.

    Args:
        operation (CallInstruction): Reusable circuit invocation.
        circuit (Any): Destination circuit.
        wires (dict[WireId, Any]): Enclosing wire mapping.
        parameters (dict[str, Any]): Shared parameter cache.
        reusable_gates (dict[ReusableCircuit, Any]): Materialized gate cache.

    Raises:
        EmitError: If the reusable body is not unitary or has incompatible
            arity.
    """
    callee = operation.callee
    if callee.native_realization is not None:
        _emit_native_semantic_op(operation, circuit, wires)
        return
    if callee.body.num_clbits:
        raise EmitError("A measured circuit cannot be materialized as a Qiskit gate")
    try:
        gate = reusable_gates.get(callee)
        cacheable = True
    except TypeError:
        # Some otherwise immutable circuit instructions carry third-party
        # semantic payloads, such as Hamiltonian, that deliberately do not
        # implement hashing. They remain valid reusable bodies but cannot be
        # structural dictionary keys, so materialize them without caching.
        gate = None
        cacheable = False
    if gate is None:
        nested = QiskitMaterializer()._materialize(
            callee.body,
            preserve_control_flow=False,
            parameters=parameters,
            reusable_gates=reusable_gates,
        )
        from qiskit.exceptions import QiskitError

        try:
            gate = nested.to_gate(label=callee.name)
            if callee.inverse:
                gate = gate.inverse()
            if callee.controls:
                gate = gate.control(callee.controls)
            if callee.power != 1:
                gate = gate.power(callee.power, annotated=True)
        except (QiskitError, TypeError, ValueError) as error:
            raise EmitError(
                f"Reusable circuit {callee.name!r} cannot become a Qiskit gate"
            ) from error
        if cacheable:
            reusable_gates[callee] = gate
    qubits = [wires[wire] for wire in operation.inputs]
    if gate.num_qubits != len(qubits):
        raise EmitError(
            f"Reusable circuit {callee.name!r} expects {gate.num_qubits} "
            f"qubits but received {len(qubits)}"
        )
    circuit.append(gate, qubits)
    _publish_wires(operation.outputs, qubits, wires)


def _emit_native_semantic_op(
    operation: CallInstruction,
    circuit: Any,
    wires: dict[WireId, Any],
) -> None:
    """Materialize a semantic-tagged call with Qiskit's native library.

    Target verification guarantees the semantic operation is declared native
    before this runs, so the body is intentionally ignored: the semantic
    identity is the meaning, and the native gate is Qiskit's realization of it.

    Args:
        operation (CallInstruction): Semantic-tagged reusable call.
        circuit (Any): Destination Qiskit circuit.
        wires (dict[WireId, Any]): Virtual-to-Qiskit wire mapping.

    Raises:
        EmitError: If the semantic operation has no native Qiskit realization
            or the requested transforms cannot be applied.
    """
    from qiskit.circuit.library import (
        FullAdderGate,
        MCXGate,
        QFTGate,
        StatePreparation,
    )
    from qiskit.exceptions import QiskitError

    callee = operation.callee
    identity = callee.identity
    assert identity is not None and callee.native_realization is not None
    realization = callee.native_realization
    if realization in {"qiskit.qft", "qiskit.iqft"}:
        gate: Any = QFTGate(callee.body.num_qubits)
        if realization == "qiskit.iqft":
            gate = gate.inverse(annotated=True)
        qubits = [wires[wire] for wire in operation.inputs]
    elif realization == "qiskit.state_preparation":
        encoded = identity.arguments.get("amplitudes")
        amplitudes = _decode_state_preparation_amplitudes(encoded)
        gate = StatePreparation(amplitudes, normalize=False)
        qubits = [wires[wire] for wire in operation.inputs]
    elif realization == "qiskit.ripple_carry_add":
        left_width, right_width, carry_width, overflow_width = callee.operand_widths
        if left_width != right_width or (carry_width, overflow_width) != (1, 1):
            raise EmitError("Ripple-carry adder operand grouping is malformed")
        gate = FullAdderGate(left_width)
        original = [wires[wire] for wire in operation.inputs]
        left_end = left_width
        right_end = left_end + right_width
        carry = original[right_end]
        overflow = original[right_end + 1]
        qubits = [carry, *original[:left_end], *original[left_end:right_end], overflow]
    elif realization == "qiskit.multi_controlled_x":
        control_width, target_width = callee.operand_widths
        if target_width != 1:
            raise EmitError("Multi-controlled X requires one target qubit")
        gate = MCXGate(control_width)
        qubits = [wires[wire] for wire in operation.inputs]
    else:
        raise EmitError(
            f"Qiskit has no native realization for semantic operation "
            f"{identity.key.namespace}.{identity.key.name}"
        )
    try:
        if callee.inverse:
            gate = gate.inverse()
        if callee.controls:
            gate = gate.control(callee.controls)
        if callee.power != 1:
            gate = gate.power(callee.power, annotated=True)
    except (QiskitError, TypeError, ValueError) as error:
        raise EmitError(
            f"Native semantic operation {identity.key.name} cannot apply the "
            f"requested transforms (power={callee.power}, "
            f"inverse={callee.inverse}, controls={callee.controls})"
        ) from error
    circuit.append(gate, qubits)
    original_qubits = [wires[wire] for wire in operation.inputs]
    _publish_wires(operation.outputs, original_qubits, wires)


def _decode_state_preparation_amplitudes(value: Any) -> list[complex]:
    """Decode immutable real-imaginary pairs for Qiskit state preparation.

    Args:
        value (Any): Semantic argument expected to contain ``(real, imag)``
            pairs.

    Returns:
        list[complex]: Concrete normalized amplitudes in basis-state order.

    Raises:
        EmitError: If the payload is absent or malformed.
    """
    if not isinstance(value, tuple):
        raise EmitError("State preparation requires encoded amplitudes")
    amplitudes: list[complex] = []
    for pair in value:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], (int, float))
            or not isinstance(pair[1], (int, float))
        ):
            raise EmitError("State preparation amplitudes are malformed")
        amplitudes.append(complex(float(pair[0]), float(pair[1])))
    return amplitudes


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
