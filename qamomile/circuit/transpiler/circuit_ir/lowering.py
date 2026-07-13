"""Lower circuit-family execution plans into backend-neutral circuit IR."""

from __future__ import annotations

import dataclasses
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
    CallTransform,
    CompositeGateType,
    InlineRegionBoundary,
    InlineRegionBoundaryOperation,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import IfOperation, WhileOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.block_parameter_binding import (
    pair_block_parameter_operands,
)
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
    as_scalar_expr,
)
from qamomile.circuit.transpiler.circuit_ir.trace import (
    CircuitProgramTrace,
    ItemsLoopOrigin,
    RangeLoopOrigin,
    SpecializedLoopKind,
    TracingCircuitBuilder,
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
        preserve_semantic_call_names (bool): Whether reusable calls retain
            source block names for structural consumers. Defaults to ``False``.
        capture_draw_trace (bool): Whether lowering retains drawing-only
            source regions beside the flat program. Defaults to ``False``.
        allow_opaque_semantic_calls (bool): Whether bodyless semantic calls
            may be retained as explicit opaque boxes. Defaults to ``False``.
    """

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
        *,
        preserve_semantic_call_names: bool = False,
        capture_draw_trace: bool = False,
        allow_opaque_semantic_calls: bool = False,
    ) -> None:
        """Initialize circuit-IR lowering.

        Args:
            bindings (dict[str, Any] | None): Compile-time parameter bindings.
                Defaults to ``None``.
            parameters (list[str] | None): Runtime parameter names. Defaults
                to ``None``.
            preserve_semantic_call_names (bool): Whether reusable calls retain
                source block names. Defaults to ``False`` so SDK materializer
                output remains unchanged.
            capture_draw_trace (bool): Whether builders retain drawing-only
                source provenance. Defaults to ``False``.
            allow_opaque_semantic_calls (bool): Whether bodyless semantic
                calls may remain as explicit non-executable boxes. Defaults
                to ``False``.
        """
        super().__init__(
            CircuitGateEmitter(capture_draw_trace=capture_draw_trace),
            bindings=bindings,
            parameters=parameters,
            backend_name="circuit_ir",
        )
        self._preserve_semantic_call_names = preserve_semantic_call_names
        self._capture_draw_trace = capture_draw_trace
        self._allow_opaque_semantic_calls = allow_opaque_semantic_calls
        self._composite_emitters.append(_SemanticCompositeEmitter(self))

    def _begin_specialized_loop_trace(
        self,
        circuit: CircuitBuilder,
        kind: str,
        origin: Any,
    ) -> Any:
        """Open a specialized-loop trace on a drawing builder.

        Args:
            circuit (CircuitBuilder): Circuit currently being emitted.
            kind (str): Stable loop category name.
            origin (Any): Concrete source loop metadata.

        Returns:
            Any: Opaque trace context, or ``None`` outside drawing lowering.

        Raises:
            ValueError: If drawing lowering receives malformed loop metadata.
        """
        if not isinstance(circuit, TracingCircuitBuilder):
            return None
        if kind == SpecializedLoopKind.RANGE.value:
            if (
                not isinstance(origin, tuple)
                or len(origin) != 2
                or not isinstance(origin[0], str)
                or not isinstance(origin[1], range)
            ):
                raise ValueError("Malformed specialized range-loop metadata")
            trace_kind = SpecializedLoopKind.RANGE
            trace_origin = RangeLoopOrigin(origin[0], origin[1])
        elif kind == SpecializedLoopKind.ITEMS.value:
            if (
                not isinstance(origin, tuple)
                or len(origin) != 2
                or not isinstance(origin[0], tuple)
                or not isinstance(origin[1], str)
            ):
                raise ValueError("Malformed specialized items-loop metadata")
            trace_kind = SpecializedLoopKind.ITEMS
            trace_origin = ItemsLoopOrigin(origin[0], origin[1])
        else:
            raise ValueError(f"Unknown specialized-loop trace kind: {kind!r}")
        return circuit.begin_specialized_loop(trace_kind, trace_origin)

    def _begin_specialized_iteration_trace(
        self,
        circuit: CircuitBuilder,
        context: Any,
        value_label: str,
    ) -> None:
        """Open one exact specialized iteration on a drawing builder.

        Args:
            circuit (CircuitBuilder): Circuit currently being emitted.
            context (Any): Opaque specialized-loop context.
            value_label (str): Deterministic iteration-value label.
        """
        if isinstance(circuit, TracingCircuitBuilder):
            circuit.begin_specialized_iteration(context, value_label)

    def _end_specialized_iteration_trace(
        self,
        circuit: CircuitBuilder,
        context: Any,
    ) -> None:
        """Close one exact specialized iteration on a drawing builder.

        Args:
            circuit (CircuitBuilder): Circuit currently being emitted.
            context (Any): Opaque specialized-loop context.
        """
        if isinstance(circuit, TracingCircuitBuilder):
            circuit.end_specialized_iteration(context)

    def _end_specialized_loop_trace(
        self,
        circuit: CircuitBuilder,
        context: Any,
    ) -> None:
        """Close one specialized-loop trace on a drawing builder.

        Args:
            circuit (CircuitBuilder): Circuit currently being emitted.
            context (Any): Opaque specialized-loop context.
        """
        if isinstance(circuit, TracingCircuitBuilder):
            circuit.end_specialized_loop(context)

    def _emit_inline_region_boundary(
        self,
        circuit: CircuitBuilder,
        operation: InlineRegionBoundaryOperation,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Capture a drawing-only inlined qkernel boundary as provenance.

        Args:
            circuit (CircuitBuilder): Circuit currently being emitted.
            operation (InlineRegionBoundaryOperation): Source boundary marker.
            qubit_map (QubitMap): Current semantic-to-physical qubit map.
            bindings (dict[str, Any]): Active emit-time bindings.

        Raises:
            EmitError: If a retained scalar call argument cannot be represented
                by target-neutral circuit scalar IR.
        """
        if not isinstance(circuit, TracingCircuitBuilder):
            return
        quantum_slots = self._resolve_inline_quantum_slots(
            operation,
            qubit_map,
            bindings,
        )
        if operation.boundary is InlineRegionBoundary.START:
            arguments = self._resolve_inline_call_arguments(operation, bindings)
            circuit.begin_inline_region(
                operation.label,
                operation.region_id,
                arguments,
                quantum_slots,
            )
            return
        circuit.end_inline_region(
            operation.label,
            operation.region_id,
            quantum_slots,
        )

    def _resolve_inline_quantum_slots(
        self,
        operation: InlineRegionBoundaryOperation,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> tuple[int, ...]:
        """Resolve a source boundary's exact scalar/vector quantum interface.

        Args:
            operation (InlineRegionBoundaryOperation): Boundary marker carrying
                substituted source-call inputs or outputs.
            qubit_map (QubitMap): Current semantic-to-physical qubit map.
            bindings (dict[str, Any]): Active emit-time bindings.

        Returns:
            tuple[int, ...]: Exact physical slots in source operand order.

        Raises:
            EmitError: If any retained scalar, array, slice, or indexed value
                cannot be resolved without guessing.
        """
        from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
            _expand_quantum_operands_to_phys,
        )

        slots: list[int] = []
        for value in operation.quantum_values:
            slots.extend(
                _expand_quantum_operands_to_phys(
                    self,
                    value,
                    qubit_map,
                    bindings,
                    operation="InlineRegionBoundaryOperation",
                )
            )
        return tuple(slots)

    def _resolve_inline_call_arguments(
        self,
        operation: InlineRegionBoundaryOperation,
        bindings: dict[str, Any],
    ) -> tuple[tuple[str, ScalarExpr], ...]:
        """Resolve source marker scalar actuals using the emit-time resolver.

        Args:
            operation (InlineRegionBoundaryOperation): Start marker whose names
                and operands are positionally aligned.
            bindings (dict[str, Any]): Active emit-time bindings.

        Returns:
            tuple[tuple[str, ScalarExpr], ...]: Ordered formal-name and scalar
                expression pairs.

        Raises:
            EmitError: If a retained operand has no scalar representation.
        """
        if len(operation.argument_names) != len(operation.argument_values):
            raise EmitError("Inline source-call argument metadata is malformed")
        result: list[tuple[str, ScalarExpr]] = []
        for name, actual in zip(
            operation.argument_names,
            operation.argument_values,
            strict=True,
        ):
            resolved = self._resolver.resolve_operand_for_binding(actual, bindings)
            if resolved is None:
                parameter_name = (
                    actual.parameter_name() if actual.is_parameter() else actual.name
                )
                if parameter_name:
                    resolved = ParameterExpr(parameter_name)
            if resolved is None:
                raise EmitError(f"Cannot resolve inline source-call argument {name!r}")
            try:
                expression = as_scalar_expr(resolved)
            except TypeError as error:
                raise EmitError(
                    f"Inline source-call argument {name!r} is not scalar"
                ) from error
            result.append((name, expression))
        return tuple(result)

    def _blockvalue_to_gate(
        self,
        block_value: Any,
        num_qubits: int,
        bindings: dict[str, Any],
        input_operands: list[Any] | None = None,
        operation_name: str = "ControlledUOperation",
    ) -> Any:
        """Preserve a nested block name in target-neutral reusable calls.

        Target emitters keep their established helper naming, while the
        target-neutral circuit model retains the semantic name needed by
        visualization and other structural consumers.

        Args:
            block_value (Any): Block-like object converted to a reusable call.
            num_qubits (int): Number of qubits in the nested circuit.
            bindings (dict[str, Any]): Active emit bindings.
            input_operands (list[Any] | None): Optional call-site operands.
                Defaults to ``None``.
            operation_name (str): Operation name used in diagnostics. Defaults
                to ``"ControlledUOperation"``.

        Returns:
            Any: Reusable circuit with a semantic name when one is available.
        """
        if input_operands is not None or operation_name != "ControlledUOperation":
            gate = super()._blockvalue_to_gate(
                block_value,
                num_qubits,
                bindings,
                input_operands=input_operands,
                operation_name=operation_name,
            )
        else:
            # Preserve the historical hook contract for the default
            # ControlledU path.
            # Third-party subclasses and tests may override this method with
            # the original three-positional-argument signature. Invoke and
            # inverse calls still use the extended arguments because their
            # nested classical parameters must be bound at the call site.
            gate = super()._blockvalue_to_gate(
                block_value,
                num_qubits,
                bindings,
            )
        if not self._preserve_semantic_call_names or not isinstance(
            gate, ReusableCircuit
        ):
            return gate
        semantic_name = getattr(block_value, "name", None)
        updates: dict[str, Any] = {
            "call_arguments": self._call_arguments(
                block_value,
                input_operands,
                bindings,
            )
        }
        if isinstance(semantic_name, str) and semantic_name:
            updates["name"] = semantic_name
        return dataclasses.replace(gate, **updates)

    def _call_arguments(
        self,
        block_value: Any,
        input_operands: list[Any] | None,
        bindings: dict[str, Any],
    ) -> tuple[tuple[str, ScalarExpr], ...]:
        """Resolve ordered scalar call arguments for structural consumers.

        Args:
            block_value (Any): Nested callable body exposing formal inputs.
            input_operands (list[Any] | None): Call-site actual arguments.
            bindings (dict[str, Any]): Active emit-time bindings.

        Returns:
            tuple[tuple[str, ScalarExpr], ...]: Ordered formal-name and
            target-neutral scalar-expression pairs. Unsupported object and
            container parameters are omitted.
        """
        if not hasattr(block_value, "input_values"):
            return ()
        parameter_inputs = [
            formal
            for formal in block_value.input_values
            if formal.type.is_classical() or formal.type.is_object()
        ]
        if input_operands is None:
            pairs = [(formal, formal) for formal in parameter_inputs]
        else:
            parameter_operands = [
                operand
                for operand in input_operands
                if hasattr(operand, "type")
                and (operand.type.is_classical() or operand.type.is_object())
            ]
            pairs = pair_block_parameter_operands(
                block_value,
                parameter_operands,
            )
        result: list[tuple[str, ScalarExpr]] = []
        for formal, actual in pairs:
            if not formal.type.is_classical() or isinstance(formal, ArrayValue):
                continue
            resolved = self._resolver.resolve_operand_for_binding(actual, bindings)
            if resolved is None and isinstance(actual, Value):
                parameter_name = (
                    actual.parameter_name() if actual.is_parameter() else actual.name
                )
                if parameter_name:
                    resolved = ParameterExpr(parameter_name)
            if resolved is None:
                continue
            try:
                expression = as_scalar_expr(resolved)
            except TypeError:
                continue
            if not isinstance(expression, _SCALAR_EXPR_TYPES):
                continue
            result.append((formal.name or "?", expression))
        return tuple(result)

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
        effective_body = op.effective_body()
        opaque = op.gate_type is CompositeGateType.CUSTOM and effective_body is None
        if opaque and not self._pass._allow_opaque_semantic_calls:
            return False
        if opaque:
            control_width, operand_widths = _opaque_call_shape(
                op,
                len(qubit_indices),
            )
        else:
            control_width = 0
            operand_widths = _semantic_operand_widths(op, len(qubit_indices))
        call_arguments = self._pass._call_arguments(
            effective_body,
            op.parameters,
            bindings,
        )
        body_builder = CircuitBuilder(
            len(qubit_indices) - control_width,
            0,
            name=op.target.name,
        )
        if not opaque:
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
                controls=control_width,
                inverse=opaque and op.transform is CallTransform.INVERSE,
                operand_widths=operand_widths,
                call_arguments=call_arguments,
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
                opaque=opaque,
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


def _opaque_call_shape(
    operation: InvokeOperation,
    flattened_width: int,
) -> tuple[int, tuple[int, ...]]:
    """Recover an opaque call's control width and target operand grouping.

    Args:
        operation (InvokeOperation): Bodyless semantic invocation.
        flattened_width (int): Total resolved physical width.

    Returns:
        tuple[int, tuple[int, ...]]: Flattened leading control count and exact
            target operand widths when statically recoverable.

    Raises:
        EmitError: If a dynamic aggregate control prevents an exact split
            between control and target slots.
    """
    all_widths = _semantic_operand_widths(operation, flattened_width)
    control_operands = operation.control_qubits
    if all_widths:
        control_count = len(control_operands)
        return sum(all_widths[:control_count]), all_widths[control_count:]
    if any(isinstance(operand, ArrayValue) for operand in control_operands):
        raise EmitError(
            "Opaque callable control width cannot be resolved exactly",
            operation=f"InvokeOperation[{operation.target.name}]",
        )
    control_width = len(control_operands)
    if control_width > flattened_width:
        raise EmitError(
            "Opaque callable control width exceeds its resolved arity",
            operation=f"InvokeOperation[{operation.target.name}]",
        )
    return control_width, ()


_SCALAR_EXPR_TYPES = (
    LiteralExpr,
    ParameterExpr,
    ClassicalBitExpr,
    LoopVariableExpr,
    BinaryExpr,
    UnaryExpr,
)


def resolve_expval_qubit_slots(
    qubits_value: Value | None,
    quantum_segment_index: int,
    compiled_quantum: list[CompiledQuantumSegment[CircuitProgram]],
    bindings: dict[str, Any] | None = None,
    parameters: list[str] | None = None,
) -> tuple[int, ...]:
    """Resolve an expectation-value operand to exact physical slots.

    This exposes the same slice/root-address logic used by normal backend
    emission without requiring a concrete Hamiltonian. Structural consumers
    such as circuit drawing can therefore retain a symbolic ``<H>`` terminal
    operation while still rejecting incomplete qubit mappings.

    Args:
        qubits_value (Value | None): Scalar or aggregate quantum operand.
        quantum_segment_index (int): Compiled quantum segment providing the
            state.
        compiled_quantum (list[CompiledQuantumSegment[CircuitProgram]]):
            Verified target-neutral quantum segments.
        bindings (dict[str, Any] | None): Compile-time structural bindings.
            Defaults to ``None``.
        parameters (list[str] | None): Runtime parameter names. Defaults to
            ``None``.

    Returns:
        tuple[int, ...]: Physical slots in observable operand order.

    Raises:
        EmitError: If the operand width or any physical slot cannot be
            resolved exactly.
        ValueError: If bindings and runtime parameters overlap.
    """
    if qubits_value is None:
        raise EmitError(
            "Expectation-value drawing has no quantum operand.",
            operation="ExpvalSegment",
        )

    lowering_pass = CircuitLoweringPass(bindings, parameters)
    mapping = lowering_pass._build_qubit_map(
        qubits_value,
        quantum_segment_index,
        compiled_quantum,
    )
    if isinstance(qubits_value, ArrayValue):
        width = len(qubits_value.get_element_uuids())
        if not width and qubits_value.shape:
            resolved = lowering_pass._resolver.resolve_int_value(
                qubits_value.shape[0],
                lowering_pass.bindings,
            )
            width = resolved if resolved is not None else 0
    else:
        width = 1
    expected_indices = list(range(width))
    if width <= 0 or sorted(mapping) != expected_indices:
        raise EmitError(
            "Expectation-value qubits could not be mapped to exact physical "
            f"slots (expected logical indices {expected_indices}, resolved "
            f"{sorted(mapping)}).",
            operation="ExpvalSegment",
        )
    return tuple(mapping[index] for index in expected_indices)


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
    *,
    preserve_semantic_call_names: bool = False,
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
        preserve_semantic_call_names (bool): Whether reusable calls retain
            source block names for structural consumers. Defaults to ``False``.

    Returns:
        ExecutableProgram[CircuitProgram]: Execution structure containing
            immutable backend-neutral quantum programs.

    Raises:
        EmitError: If the semantic operations cannot be lowered to the
            circuit-family instruction set.
        ValueError: If structural verification rejects a lowered circuit.
    """
    lowered = CircuitLoweringPass(
        bindings,
        parameters,
        preserve_semantic_call_names=preserve_semantic_call_names,
    ).run(plan)
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


def lower_circuit_plan_with_trace(
    plan: ProgramPlan,
    bindings: dict[str, Any] | None = None,
    parameters: list[str] | None = None,
    *,
    preserve_semantic_call_names: bool = False,
) -> tuple[
    ExecutableProgram[CircuitProgram],
    tuple[CircuitProgramTrace, ...],
]:
    """Lower a plan and retain lossless drawing-only source provenance.

    This entry point is intentionally separate from :func:`lower_circuit_plan`:
    SDK compilation keeps its existing builders, programs, and materializers,
    while structural consumers opt into the immutable sidecar explicitly.

    Args:
        plan (ProgramPlan): Circuit-family execution plan.
        bindings (dict[str, Any] | None): Compile-time parameter bindings.
            Defaults to ``None``.
        parameters (list[str] | None): Runtime parameter names. Defaults to
            ``None``.
        preserve_semantic_call_names (bool): Whether reusable calls retain
            source names. Defaults to ``False``.

    Returns:
        tuple[ExecutableProgram[CircuitProgram], tuple[CircuitProgramTrace, ...]]:
            Verified flat executable and one aligned immutable trace per
            quantum segment.

    Raises:
        EmitError: If semantic operations cannot be lowered exactly.
        RuntimeError: If trace region construction remains unbalanced.
        ValueError: If circuit or trace structural verification fails.
    """
    lowered = CircuitLoweringPass(
        bindings,
        parameters,
        preserve_semantic_call_names=preserve_semantic_call_names,
        capture_draw_trace=True,
        allow_opaque_semantic_calls=True,
    ).run(plan)
    quantum_segments: list[CompiledQuantumSegment[CircuitProgram]] = []
    traces: list[CircuitProgramTrace] = []
    for segment in lowered.compiled_quantum:
        if not isinstance(segment.circuit, TracingCircuitBuilder):
            raise RuntimeError("Drawing lowering did not create a tracing builder")
        program = segment.circuit.freeze()
        verify_circuit(program)
        trace = segment.circuit.freeze_trace(program)
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
        traces.append(trace)
    executable = ExecutableProgram(
        plan=lowered.plan,
        compiled_quantum=quantum_segments,
        compiled_classical=lowered.compiled_classical,
        compiled_expval=lowered.compiled_expval,
        output_values=list(lowered.output_values),
    )
    return executable, tuple(traces)
