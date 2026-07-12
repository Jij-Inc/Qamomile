"""Lower circuit-family execution plans into backend-neutral circuit IR."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.operation.arithmetic_operations import (
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
    RuntimeClassicalExpr,
    RuntimeOpKind,
)
from qamomile.circuit.ir.operation.callable import (
    CompositeGateType,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import IfOperation, WhileOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.circuit_ir.emitter import CircuitGateEmitter
from qamomile.circuit.transpiler.circuit_ir.model import (
    BinaryExpr,
    BinaryOperator,
    CallableIdentity,
    CircuitBuilder,
    CircuitProgram,
    ClassicalBitExpr,
    LiteralExpr,
    LoopVariableExpr,
    ParameterExpr,
    ReusableCircuit,
    ScalarExpr,
    SemanticArguments,
    SemanticOpKey,
    UnaryExpr,
    UnaryOperator,
)
from qamomile.circuit.transpiler.circuit_ir.verify import verify_circuit
from qamomile.circuit.transpiler.compiled_segments import CompiledQuantumSegment
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.passes.emit_support import ClbitMap, QubitMap
from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
    register_classical_merge_aliases,
    register_merge_outputs,
    resolve_condition_address,
    resolve_if_condition,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import QubitAddress
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.segments import ProgramPlan

_RUNTIME_BINARY_OPERATORS = {
    RuntimeOpKind.EQ: BinaryOperator.EQ,
    RuntimeOpKind.NEQ: BinaryOperator.NEQ,
    RuntimeOpKind.LT: BinaryOperator.LT,
    RuntimeOpKind.LE: BinaryOperator.LE,
    RuntimeOpKind.GT: BinaryOperator.GT,
    RuntimeOpKind.GE: BinaryOperator.GE,
    RuntimeOpKind.AND: BinaryOperator.AND,
    RuntimeOpKind.OR: BinaryOperator.OR,
    RuntimeOpKind.ADD: BinaryOperator.ADD,
    RuntimeOpKind.SUB: BinaryOperator.SUB,
    RuntimeOpKind.MUL: BinaryOperator.MUL,
    RuntimeOpKind.DIV: BinaryOperator.DIV,
    RuntimeOpKind.FLOORDIV: BinaryOperator.FLOORDIV,
    RuntimeOpKind.MOD: BinaryOperator.MOD,
    RuntimeOpKind.POW: BinaryOperator.POW,
}


class CircuitLoweringPass(StandardEmitPass[CircuitBuilder]):
    """Lower a segmented circuit program into target-neutral builders.

    Args:
        bindings (dict[str, Any] | None): Compile-time parameter bindings.
            Defaults to ``None``.
        parameters (list[str] | None): Runtime parameter names. Defaults to
            ``None``.
    """

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> None:
        """Initialize circuit-IR lowering.

        Args:
            bindings (dict[str, Any] | None): Compile-time parameter bindings.
                Defaults to ``None``.
            parameters (list[str] | None): Runtime parameter names. Defaults
                to ``None``.
        """
        super().__init__(
            CircuitGateEmitter(),
            bindings=bindings,
            parameters=parameters,
            backend_name="circuit_ir",
        )
        self._composite_emitters.append(_SemanticCompositeEmitter(self))

    def _runtime_operand(
        self,
        value: Value,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> ScalarExpr:
        """Resolve a semantic runtime operand to a circuit scalar expression.

        Args:
            value (Value): Semantic classical operand.
            clbit_map (ClbitMap): Measurement-result allocation map.
            bindings (dict[str, Any]): Typed emit context.

        Returns:
            ScalarExpr: Target-neutral runtime expression operand.

        Raises:
            EmitError: If the value cannot be represented at circuit level.
        """
        get_runtime_expr = getattr(bindings, "get_runtime_expr", None)
        if callable(get_runtime_expr):
            stored = get_runtime_expr(value.uuid)
            if isinstance(stored, _SCALAR_EXPR_TYPES):
                return stored
        stored = bindings.get(value.uuid)
        if isinstance(stored, _SCALAR_EXPR_TYPES):
            return stored
        if value.is_constant():
            concrete = value.get_const()
            if isinstance(concrete, (bool, int, float)):
                return LiteralExpr(concrete)
        address = resolve_condition_address(value, bindings, self._resolver)
        if address in clbit_map:
            return ClassicalBitExpr(clbit_map[address])
        concrete = self._resolver.resolve_classical_value(value, bindings)
        if isinstance(concrete, (bool, int, float)):
            return LiteralExpr(concrete)
        parameter_key = self._resolver.get_parameter_key(value, bindings)
        if parameter_key:
            parameter = self._get_or_create_parameter(parameter_key, value.uuid)
            if isinstance(parameter, _SCALAR_EXPR_TYPES):
                return parameter
        get_loop_var = getattr(bindings, "get_loop_var", None)
        if callable(get_loop_var):
            loop_value = get_loop_var(value.uuid)
            if isinstance(loop_value, _SCALAR_EXPR_TYPES):
                return loop_value
            if isinstance(loop_value, (bool, int, float)):
                return LiteralExpr(loop_value)
        raise EmitError(
            f"Cannot lower runtime classical value {value.name!r} to CircuitProgram"
        )

    def _emit_runtime_classical_expr(
        self,
        circuit: CircuitBuilder,
        op: RuntimeClassicalExpr,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Normalize a runtime classical operation into circuit scalar IR.

        Args:
            circuit (CircuitBuilder): Current circuit builder.
            op (RuntimeClassicalExpr): Semantic runtime expression.
            clbit_map (ClbitMap): Measurement-result allocation map.
            bindings (dict[str, Any]): Typed emit context to update.

        Raises:
            EmitError: If an operand or operation kind is unsupported.
        """
        del circuit
        if op.kind is RuntimeOpKind.NOT:
            result: ScalarExpr = UnaryExpr(
                UnaryOperator.NOT,
                self._runtime_operand(op.operands[0], clbit_map, bindings),
            )
        elif op.kind is RuntimeOpKind.SELECT:
            if len(op.operands) != 3:
                raise EmitError(
                    "Circuit runtime SELECT requires condition, true, and false "
                    "operands"
                )
            condition, true_value, false_value = (
                self._runtime_operand(operand, clbit_map, bindings)
                for operand in op.operands
            )
            result = BinaryExpr(
                BinaryOperator.OR,
                BinaryExpr(BinaryOperator.AND, condition, true_value),
                BinaryExpr(
                    BinaryOperator.AND,
                    UnaryExpr(UnaryOperator.NOT, condition),
                    false_value,
                ),
            )
        else:
            if op.kind is None:
                raise EmitError("Circuit runtime operation has no operation kind")
            operator = _RUNTIME_BINARY_OPERATORS.get(op.kind)
            if operator is None:
                raise EmitError(f"Unsupported circuit runtime operation: {op.kind}")
            result = BinaryExpr(
                operator,
                self._runtime_operand(op.operands[0], clbit_map, bindings),
                self._runtime_operand(op.operands[1], clbit_map, bindings),
            )
        set_runtime_expr = getattr(bindings, "set_runtime_expr", None)
        if callable(set_runtime_expr):
            set_runtime_expr(op.results[0].uuid, result)
        else:
            bindings[op.results[0].uuid] = result

    def _build_runtime_predicate_expr(
        self,
        circuit: CircuitBuilder,
        op: CompOp | CondOp | NotOp,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> ScalarExpr | None:
        """Normalize a residual predicate not rewritten by classical lowering.

        Args:
            circuit (CircuitBuilder): Current circuit builder.
            op (CompOp | CondOp | NotOp): Residual semantic predicate.
            clbit_map (ClbitMap): Measurement-result allocation map.
            bindings (dict[str, Any]): Typed emit context.

        Returns:
            ScalarExpr | None: Normalized predicate, or ``None`` when its kind
                is outside the circuit scalar vocabulary.
        """
        del circuit
        if isinstance(op, NotOp):
            return UnaryExpr(
                UnaryOperator.NOT,
                self._runtime_operand(op.operands[0], clbit_map, bindings),
            )
        operator = None
        if isinstance(op, CompOp):
            if op.kind is None:
                return None
            operator = {
                CompOpKind.EQ: BinaryOperator.EQ,
                CompOpKind.NEQ: BinaryOperator.NEQ,
                CompOpKind.LT: BinaryOperator.LT,
                CompOpKind.LE: BinaryOperator.LE,
                CompOpKind.GT: BinaryOperator.GT,
                CompOpKind.GE: BinaryOperator.GE,
            }.get(op.kind)
        elif isinstance(op, CondOp):
            if op.kind is None:
                return None
            operator = {
                CondOpKind.AND: BinaryOperator.AND,
                CondOpKind.OR: BinaryOperator.OR,
            }.get(op.kind)
        if operator is None:
            return None
        return BinaryExpr(
            operator,
            self._runtime_operand(op.operands[0], clbit_map, bindings),
            self._runtime_operand(op.operands[1], clbit_map, bindings),
        )

    def _condition_expression(
        self,
        condition: Value,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> ScalarExpr:
        """Resolve an if/while condition to circuit scalar IR.

        Args:
            condition (Value): Semantic predicate value.
            clbit_map (ClbitMap): Measurement-result allocation map.
            bindings (dict[str, Any]): Typed emit context.

        Returns:
            ScalarExpr: Runtime predicate expression.
        """
        expression = self._runtime_operand(condition, clbit_map, bindings)
        if not _contains_classical_bit(expression):
            raise EmitError(
                "Runtime circuit conditions must come from measurement results; "
                "non-measurement parameters must be supplied through bindings"
            )
        return expression

    def _emit_if(
        self,
        circuit: CircuitBuilder,
        op: IfOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Lower runtime conditionals as structured circuit regions.

        Args:
            circuit (CircuitBuilder): Current circuit builder.
            op (IfOperation): Semantic conditional.
            qubit_map (QubitMap): Quantum allocation map.
            clbit_map (ClbitMap): Measurement-result allocation map.
            bindings (dict[str, Any]): Typed emit context.
        """
        if resolve_if_condition(op.condition, bindings) is not None:
            super()._emit_if(circuit, op, qubit_map, clbit_map, bindings)
            return
        condition = self._condition_expression(op.condition, clbit_map, bindings)
        context = circuit.begin_if(condition)
        self._emit_operations(
            circuit,
            op.true_operations,
            qubit_map,
            clbit_map,
            bindings,
            emit_qinit_reset=True,
        )
        if op.false_operations:
            circuit.begin_else(context)
            self._emit_operations(
                circuit,
                op.false_operations,
                qubit_map,
                clbit_map,
                bindings,
                emit_qinit_reset=True,
            )
        circuit.end_if(context)
        register_merge_outputs(self, op, qubit_map, clbit_map, bindings)
        register_classical_merge_aliases(self, op, bindings, None)

    def _emit_while(
        self,
        circuit: CircuitBuilder,
        op: WhileOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Lower runtime while loops as structured circuit regions.

        Args:
            circuit (CircuitBuilder): Current circuit builder.
            op (WhileOperation): Semantic while loop.
            qubit_map (QubitMap): Quantum allocation map.
            clbit_map (ClbitMap): Measurement-result allocation map.
            bindings (dict[str, Any]): Typed emit context.

        Raises:
            EmitError: If the loop has no condition operand.
        """
        if not op.operands:
            raise EmitError("WhileOperation requires a condition operand")
        condition = op.operands[0]
        if not isinstance(condition, Value):
            raise EmitError("Circuit while condition must be a semantic Value")
        context = circuit.begin_while(
            self._condition_expression(condition, clbit_map, bindings)
        )
        self._emit_operations(
            circuit,
            op.operations,
            qubit_map,
            clbit_map,
            bindings,
            emit_qinit_reset=True,
        )
        circuit.end_while(context)

    def _emit_pauli_evolve(
        self,
        circuit: CircuitBuilder,
        op: PauliEvolveOp,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Lower Pauli terms and retain the Hamiltonian constant as phase.

        Args:
            circuit (CircuitBuilder): Current circuit builder.
            op (PauliEvolveOp): Semantic Pauli evolution.
            qubit_map (QubitMap): Quantum allocation map.
            bindings (dict[str, Any]): Typed emit context.

        Raises:
            EmitError: If the Hamiltonian, evolution time, or Hermiticity
                contract cannot be resolved.
        """
        import qamomile.observable as qm_o
        from qamomile.circuit.transpiler.passes.emit_support.pauli_evolve_emission import (
            _resolve_gamma,
            validate_hamiltonian_within_register,
        )
        from qamomile.observable.hamiltonian import HERMITIAN_IMAG_ATOL

        hamiltonian = self._resolver.resolve_bound_value(op.observable, bindings)
        if not isinstance(hamiltonian, qm_o.Hamiltonian):
            raise EmitError("PauliEvolveOp requires a Hamiltonian binding")
        if abs(hamiltonian.constant.imag) > HERMITIAN_IMAG_ATOL:
            raise EmitError(
                "PauliEvolveOp requires a real Hamiltonian constant; "
                "a complex constant is non-Hermitian",
                operation="PauliEvolveOp",
            )
        gamma = _resolve_gamma(self, op, bindings)
        if gamma is None:
            raise EmitError("Cannot resolve Pauli evolution time")
        for operators, coefficient in hamiltonian:
            if abs(coefficient.imag) > HERMITIAN_IMAG_ATOL:
                raise EmitError(
                    f"PauliEvolveOp requires a Hermitian Hamiltonian, but "
                    f"coefficient {coefficient} on term {operators} is non-real",
                    operation="PauliEvolveOp",
                )

        input_array = op.qubits
        if not isinstance(input_array, ArrayValue):
            raise EmitError("PauliEvolveOp requires an array of qubits")
        if input_array.shape:
            register_size = self._resolver.resolve_int_value(
                input_array.shape[0],
                bindings,
            )
            if register_size is not None:
                validate_hamiltonian_within_register(
                    hamiltonian.num_qubits,
                    register_size,
                )
        root, start, step = self._resolver.resolve_slice_chain(
            input_array,
            bindings,
            operation="PauliEvolveOp",
        )
        qubit_indices = []
        for index in range(hamiltonian.num_qubits):
            address = QubitAddress(root.uuid, start + step * index)
            try:
                qubit_indices.append(qubit_map[address])
            except KeyError as error:
                raise EmitError(
                    f"Cannot resolve qubit {index} for PauliEvolveOp",
                    operation="PauliEvolveOp",
                ) from error
        circuit.append_pauli_evolution(
            tuple(qubit_indices),
            hamiltonian,
            gamma,
        )

        result_array = op.evolved_qubits
        if not isinstance(result_array, ArrayValue):
            raise EmitError("PauliEvolveOp result must be an array of qubits")
        result_root, result_start, result_step = self._resolver.resolve_slice_chain(
            result_array,
            bindings,
            operation="PauliEvolveOp",
        )
        for index, physical in enumerate(qubit_indices):
            qubit_map.setdefault(QubitAddress(result_array.uuid, index), physical)
            qubit_map.setdefault(
                QubitAddress(
                    result_root.uuid,
                    result_start + result_step * index,
                ),
                physical,
            )


class _SemanticCompositeEmitter:
    """Preserve every executable composite as an abstract callable.

    The fallback implementation is built once as a reusable circuit, while
    the callable reference and exact strategy variant remain available to
    target legalization. This is intentionally not limited to a closed list
    of standard-library operations.

    Args:
        lowering_pass (CircuitLoweringPass): Owning lowering pass whose
            emitter receives the fallback body gates.
    """

    def __init__(self, lowering_pass: "CircuitLoweringPass") -> None:
        """Initialize the semantic-callable preserving emitter.

        Args:
            lowering_pass (CircuitLoweringPass): Owning lowering pass.
        """
        self._pass = lowering_pass

    def can_emit(self, gate_type: CompositeGateType) -> bool:
        """Report whether a composite kind can retain a callable boundary.

        Args:
            gate_type (CompositeGateType): Composite kind to check.

        Returns:
            bool: Always ``True``; executability is checked by :meth:`emit`.
        """
        del gate_type
        return True

    def emit(
        self,
        circuit: CircuitBuilder,
        op: InvokeOperation,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> bool:
        """Box an invocation with its semantic key and fallback body.

        Args:
            circuit (CircuitBuilder): Builder receiving the call.
            op (InvokeOperation): Semantic callable invocation.
            qubit_indices (list[int]): Physical qubit slots in operand order.
            bindings (dict[str, Any]): Active emit bindings used while
                constructing the fallback body.

        Returns:
            bool: True when the invocation was boxed; ``False`` when no
                executable fallback exists.
        """
        from qamomile.circuit.transpiler.passes.emit_support.composite_gate_emission import (
            emit_composite_fallback,
        )

        if not qubit_indices:
            return False
        if op.gate_type is CompositeGateType.CUSTOM and op.effective_body() is None:
            return False
        body_builder = CircuitBuilder(
            len(qubit_indices),
            0,
            name=op.target.name,
        )
        emit_composite_fallback(
            self._pass,
            body_builder,
            op,
            list(range(len(qubit_indices))),
            bindings,
        )
        circuit.append_call(
            ReusableCircuit(
                body=body_builder.freeze(),
                name=op.custom_name,
                operand_widths=_semantic_operand_widths(op, len(qubit_indices)),
                identity=CallableIdentity(
                    key=SemanticOpKey(
                        namespace=op.target.namespace,
                        name=op.target.name,
                        version=op.target.version,
                        variant=op.strategy_name,
                    ),
                    symbol=op.target.name,
                    arguments=SemanticArguments.from_mapping(
                        op.attrs.get("semantic_arguments")
                    ),
                ),
            ),
            tuple(qubit_indices),
        )
        return True


def _semantic_operand_widths(
    operation: InvokeOperation,
    flattened_width: int,
) -> tuple[int, ...]:
    """Recover semantic quantum-operand grouping before it is flattened.

    Args:
        operation (InvokeOperation): Callable invocation whose controls and
            targets still retain scalar-versus-vector value types.
        flattened_width (int): Total number of resolved physical qubits.

    Returns:
        tuple[int, ...]: Width per semantic operand, or an empty tuple when a
        dynamic shape cannot be recovered without guessing.
    """
    widths: list[int] = []
    for operand in operation.control_qubits + operation.target_qubits:
        if not isinstance(operand, ArrayValue):
            widths.append(1)
            continue
        if len(operand.shape) != 1 or not operand.shape[0].is_constant():
            return ()
        width = int(operand.shape[0].get_const())
        if width <= 0:
            return ()
        widths.append(width)
    return tuple(widths) if sum(widths) == flattened_width else ()


_SCALAR_EXPR_TYPES = (
    LiteralExpr,
    ParameterExpr,
    ClassicalBitExpr,
    LoopVariableExpr,
    BinaryExpr,
    UnaryExpr,
)


def _contains_classical_bit(expression: ScalarExpr) -> bool:
    """Return whether an expression depends on a measured classical bit.

    Args:
        expression (ScalarExpr): Expression to inspect.

    Returns:
        bool: True when a :class:`ClassicalBitExpr` occurs recursively.
    """
    if isinstance(expression, ClassicalBitExpr):
        return True
    if isinstance(expression, BinaryExpr):
        return _contains_classical_bit(expression.left) or _contains_classical_bit(
            expression.right
        )
    if isinstance(expression, UnaryExpr):
        return _contains_classical_bit(expression.operand)
    return False


def lower_circuit_plan(
    plan: ProgramPlan,
    bindings: dict[str, Any] | None = None,
    parameters: list[str] | None = None,
) -> ExecutableProgram[CircuitProgram]:
    """Lower every quantum segment in a plan to immutable circuit IR.

    Classical and expectation-value orchestration metadata remains in the
    returned executable container. Only backend-native quantum artifacts are
    replaced with verified :class:`CircuitProgram` objects.

    Args:
        plan (ProgramPlan): Circuit-family C-to-Q-to-C execution plan.
        bindings (dict[str, Any] | None): Compile-time parameter bindings.
            Defaults to ``None``.
        parameters (list[str] | None): Runtime parameter names. Defaults to
            ``None``.

    Returns:
        ExecutableProgram[CircuitProgram]: Execution structure containing
            immutable backend-neutral quantum programs.

    Raises:
        EmitError: If the semantic operations cannot be lowered to the
            circuit-family instruction set.
        ValueError: If structural verification rejects a lowered circuit.
    """
    lowered = CircuitLoweringPass(bindings, parameters).run(plan)
    quantum_segments: list[CompiledQuantumSegment[CircuitProgram]] = []
    for segment in lowered.compiled_quantum:
        program = segment.circuit.freeze()
        verify_circuit(program)
        quantum_segments.append(
            CompiledQuantumSegment(
                segment=segment.segment,
                circuit=program,
                qubit_map=segment.qubit_map,
                clbit_map=segment.clbit_map,
                measurement_qubit_map=segment.measurement_qubit_map,
                parameter_metadata=segment.parameter_metadata,
            )
        )
    return ExecutableProgram(
        plan=lowered.plan,
        compiled_quantum=quantum_segments,
        compiled_classical=lowered.compiled_classical,
        compiled_expval=lowered.compiled_expval,
        output_values=list(lowered.output_values),
    )
