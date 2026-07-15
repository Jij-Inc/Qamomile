"""Materialize backend-neutral circuit IR through the PyQret builder API."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from qamomile.circuit.transpiler.circuit_ir import (
    ALL_BINARY_OPERATORS,
    ALL_PRIMITIVE_GATES,
    ALL_UNARY_OPERATORS,
    MULTI_CONTROLLED_X_SEMANTIC_KEY,
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


def _require_pyqret() -> tuple[Any, Any]:
    """Import the optional PyQret frontend and intrinsic gate modules.

    Returns:
        tuple[Any, Any]: The ``pyqret.frontend`` module and its intrinsic gate
            module.

    Raises:
        ImportError: If Quration's ``pyqret`` package is not installed.
    """
    try:
        import pyqret.frontend as frontend  # type: ignore[import-not-found]
        import pyqret.frontend.gate.intrinsic as intrinsic  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError(
            "Quration support requires the optional 'pyqret' package. "
            "Build and install Quration's Python bindings before using "
            "QurationTranspiler."
        ) from error
    return frontend, intrinsic


def evaluate_scalar(
    expression: ScalarExpr,
    loop_values: dict[str, int] | None = None,
) -> bool | int | float:
    """Evaluate a concrete circuit scalar expression for PyQret.

    PyQret rotation constructors currently require concrete Python floats.
    Runtime parameters and measured-bit expressions therefore remain illegal
    at this materialization boundary.

    Args:
        expression (ScalarExpr): Target-neutral expression to evaluate.
        loop_values (dict[str, int] | None): Active structured-loop induction
            values keyed by loop-variable name. Defaults to an empty mapping.

    Returns:
        bool | int | float: Concrete scalar value.

    Raises:
        EmitError: If the expression contains a runtime parameter, measured
            bit, missing loop variable, or unsupported operation.
    """
    loops = loop_values or {}
    if isinstance(expression, LiteralExpr):
        return expression.value
    if isinstance(expression, LoopVariableExpr):
        if expression.name not in loops:
            raise EmitError(f"Unbound circuit loop variable {expression.name!r}")
        return loops[expression.name]
    if isinstance(expression, ParameterExpr):
        raise EmitError(
            f"Quration requires concrete rotation angles; runtime parameter "
            f"{expression.name!r} must be supplied through bindings."
        )
    if isinstance(expression, ClassicalBitExpr):
        raise EmitError(
            "Quration cannot use a runtime measurement bit as a gate parameter"
        )
    if isinstance(expression, UnaryExpr):
        value = evaluate_scalar(expression.operand, loops)
        if expression.operator is UnaryOperator.NOT:
            return not bool(value)
        if expression.operator is UnaryOperator.NEG:
            return -value
    if isinstance(expression, BinaryExpr):
        left = evaluate_scalar(expression.left, loops)
        right = evaluate_scalar(expression.right, loops)
        return _evaluate_binary(expression.operator, left, right)
    raise EmitError(f"Unsupported Quration scalar expression: {expression!r}")


def _evaluate_binary(
    operator: BinaryOperator,
    left: bool | int | float,
    right: bool | int | float,
) -> bool | int | float:
    """Evaluate one concrete binary scalar operation.

    Args:
        operator (BinaryOperator): Operation to evaluate.
        left (bool | int | float): Left operand.
        right (bool | int | float): Right operand.

    Returns:
        bool | int | float: Concrete operation result.

    Raises:
        EmitError: If ``operator`` is not supported by the circuit scalar
            evaluator.
    """
    match operator:
        case BinaryOperator.ADD:
            return left + right
        case BinaryOperator.SUB:
            return left - right
        case BinaryOperator.MUL:
            return left * right
        case BinaryOperator.DIV:
            return left / right
        case BinaryOperator.FLOORDIV:
            return left // right
        case BinaryOperator.MOD:
            return left % right
        case BinaryOperator.POW:
            return left**right
        case BinaryOperator.EQ:
            return left == right
        case BinaryOperator.NEQ:
            return left != right
        case BinaryOperator.LT:
            return left < right
        case BinaryOperator.LE:
            return left <= right
        case BinaryOperator.GT:
            return left > right
        case BinaryOperator.GE:
            return left >= right
        case BinaryOperator.AND:
            return bool(left) and bool(right)
        case BinaryOperator.OR:
            return bool(left) or bool(right)
    raise EmitError(f"Unsupported binary scalar operator: {operator}")


class PyQretMaterializer:
    """Convert a verified circuit program to ``pyqret.frontend.Circuit``.

    Args:
        rotation_precision (float): Absolute synthesis precision forwarded to
            PyQret rotation operations. Defaults to ``1e-10``.
    """

    def __init__(self, rotation_precision: float = 1e-10) -> None:
        """Initialize the PyQret materializer.

        Args:
            rotation_precision (float): Positive rotation synthesis precision.
                Defaults to ``1e-10``.

        Raises:
            ValueError: If ``rotation_precision`` is not positive.
        """
        if rotation_precision <= 0:
            raise ValueError("rotation_precision must be positive")
        self.rotation_precision = rotation_precision

    @property
    def capabilities(self) -> CircuitCapabilities:
        """Declare Quration's circuit-IR capabilities.

        PyQret builds static DAG circuits with concrete rotation angles, so
        runtime parameters and measurement-conditioned control flow are
        rejected at the target boundary before materialization starts.

        Returns:
            CircuitCapabilities: Immutable capability declaration.
        """
        numeric = ScalarCapabilities(
            atoms=frozenset({ScalarAtom.LITERAL, ScalarAtom.LOOP_VARIABLE}),
            unary_operators=ALL_UNARY_OPERATORS,
            binary_operators=ALL_BINARY_OPERATORS,
            parameter_form=ScalarExpressionForm.CONCRETE_ONLY,
        )
        generic_call = CallTransformCapabilities(
            supports_power=True,
            supports_inverse=False,
            max_controls=0,
            supports_barrier_body=True,
            control_mode=CallControlMode.UNSUPPORTED,
            phase_mode=CallPhaseMode.NATIVE_BODY,
        )
        native_direct_call = dataclasses.replace(
            generic_call,
            supports_power=False,
            phase_mode=CallPhaseMode.UNSUPPORTED,
            controlled_phase_scalars=None,
        )
        return CircuitCapabilities(
            name="quration",
            primitive_gates=ALL_PRIMITIVE_GATES,
            native_semantic_ops=(
                NativeSemanticOpCapabilities(
                    MULTI_CONTROLLED_X_SEMANTIC_KEY,
                    "pyqret.multi_controlled_x",
                    native_direct_call,
                    operand_widths=(None, 1),
                    min_qubits=2,
                    max_qubits=3,
                ),
            ),
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
            generic_calls=generic_call,
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
        """Materialize one circuit program through PyQret's definition context.

        Args:
            program (CircuitProgram): Verified backend-neutral circuit program.
            parameter_names (tuple[str, ...]): Public runtime-parameter ABI.
                Quration currently requires this to be empty. Defaults to an
                empty tuple.

        Returns:
            MaterializedCircuit[Any]: PyQret circuit without runtime
                parameter metadata.

        Raises:
            ImportError: If PyQret is unavailable.
            EmitError: If the circuit uses a construct outside the supported
                Quration profile.
            ValueError: If circuit verification fails.
        """
        if parameter_names:
            raise EmitError("Quration requires all circuit parameters to be bound")
        verify_circuit(program)
        frontend, intrinsic = _require_pyqret()
        context = frontend.Context()
        module = frontend.Module(program.name or "qamomile", context)
        builder = frontend.CircuitBuilder(module)
        precision = self.rotation_precision
        reusable_circuits, reusable_generators = _build_reusable_circuits(
            program,
            frontend,
            intrinsic,
            builder,
            precision,
        )
        context_data = _PyQretContext(
            frontend=frontend,
            intrinsic=intrinsic,
            builder=builder,
            precision=precision,
            reusable_circuits=reusable_circuits,
        )

        class GeneratedCircuit(frontend.CircuitGenerator):  # type: ignore[misc]
            """Define one PyQret circuit from an immutable circuit program."""

            def name(self) -> str:
                """Return the generated circuit name.

                Returns:
                    str: Stable circuit name.
                """
                return program.name or "main"

            def arg(self) -> Any:
                """Declare PyQret qubit and measurement-register arguments.

                Returns:
                    Any: ``pyqret.frontend.Argument`` declaration.
                """
                argument = frontend.Argument()
                if program.num_qubits:
                    argument.add_operates("q", program.num_qubits)
                if program.num_clbits:
                    argument.add_outputs("c", program.num_clbits)
                return argument

            def logic(self, arg: Any) -> None:
                """Emit program instructions inside PyQret's definition context.

                Args:
                    arg (Any): PyQret circuit argument view.

                Raises:
                    EmitError: If materialization encounters an unsupported
                        instruction or scalar expression.
                """
                qubits = arg["q"] if program.num_qubits else ()
                registers = arg["c"] if program.num_clbits else ()
                wires = {
                    wire: qubits[index]
                    for index, wire in enumerate(program.input_wires)
                }
                _emit_global_phase(program, context_data)
                _emit_region(
                    program.operations,
                    wires,
                    registers,
                    context_data,
                    {},
                )

        generator = GeneratedCircuit(builder)
        circuit = generator.generate()
        # PyQret's native Circuit keeps non-owning references into the
        # definition context. Preserve the complete builder ownership graph
        # for simulation and backend compilation.
        circuit._qamomile_owners = (
            context,
            module,
            builder,
            generator,
            *reusable_generators,
            *reusable_circuits.values(),
        )
        return MaterializedCircuit(artifact=circuit)


@dataclasses.dataclass(frozen=True)
class _PyQretContext:
    """Hold PyQret objects shared by nested materialization.

    Args:
        frontend (Any): Imported ``pyqret.frontend`` module.
        intrinsic (Any): Imported PyQret intrinsic gate module.
        builder (Any): Circuit builder owning every generated definition.
        precision (float): Rotation synthesis precision.
        reusable_circuits (dict[int, Any]): Reusable-circuit object identity to
            generated PyQret circuit.
    """

    frontend: Any
    intrinsic: Any
    builder: Any
    precision: float
    reusable_circuits: dict[int, Any]


def _build_reusable_circuits(
    program: CircuitProgram,
    frontend: Any,
    intrinsic: Any,
    builder: Any,
    precision: float,
) -> tuple[dict[int, Any], list[Any]]:
    """Predeclare reusable bodies as native PyQret circuits.

    Definitions are generated bottom-up so a parent body can emit native
    calls to every child definition while its own definition context is
    active.

    Args:
        program (CircuitProgram): Root circuit program to inspect.
        frontend (Any): Imported ``pyqret.frontend`` module.
        intrinsic (Any): Imported PyQret intrinsic gate module.
        builder (Any): Shared PyQret circuit builder.
        precision (float): Rotation synthesis precision.

    Returns:
        tuple[dict[int, Any], list[Any]]: Mapping from reusable-circuit object
        identity to native PyQret circuit, plus generators retained for
        ownership.
    """
    circuits: dict[int, Any] = {}
    generators: list[Any] = []
    definitions: dict[tuple[str, Any], Any] = {}

    def visit_operations(operations: tuple[CircuitInstruction, ...]) -> None:
        """Generate definitions reachable from one instruction region.

        Args:
            operations (tuple[CircuitInstruction, ...]): Region to traverse.
        """
        for operation in operations:
            if isinstance(operation, CallInstruction):
                visit_callee(operation.callee)
            elif isinstance(operation, ForInstruction):
                visit_operations(operation.body)
            elif isinstance(operation, IfInstruction):
                visit_operations(operation.true_body)
                visit_operations(operation.false_body)
            elif isinstance(operation, WhileInstruction):
                visit_operations(operation.body)

    def visit_callee(callee: ReusableCircuit) -> None:
        """Generate one reusable definition after its dependencies.

        Args:
            callee (ReusableCircuit): Reusable body to materialize.
        """
        if callee.native_realization is not None:
            return
        cache_key = id(callee)
        if cache_key in circuits:
            return
        try:
            hash(callee.body)
            definition_key = (callee.name, callee.body)
        except TypeError:
            definition_key = (callee.name, id(callee.body))
        existing = definitions.get(definition_key)
        if existing is not None:
            circuits[cache_key] = existing
            return
        visit_operations(callee.body.operations)
        definition_index = len(definitions)

        class GeneratedReusableCircuit(frontend.CircuitGenerator):  # type: ignore[misc]
            """Define one reusable Qamomile body as a PyQret circuit."""

            def name(self) -> str:
                """Return a unique readable PyQret definition name.

                Returns:
                    str: Definition name unique within the shared module.
                """
                safe_name = "".join(
                    character if character.isalnum() or character == "_" else "_"
                    for character in callee.name
                )
                return f"{safe_name or 'callable'}__{definition_index}"

            def arg(self) -> Any:
                """Declare one scalar qubit argument per body input.

                Returns:
                    Any: PyQret argument declaration.
                """
                argument = frontend.Argument()
                for index in range(callee.body.num_qubits):
                    argument.add_operate(f"q{index}")
                return argument

            def logic(self, arg: Any) -> None:
                """Emit the reusable body inside its definition context.

                Args:
                    arg (Any): PyQret argument view.
                """
                context = _PyQretContext(
                    frontend=frontend,
                    intrinsic=intrinsic,
                    builder=builder,
                    precision=precision,
                    reusable_circuits=circuits,
                )
                wires = {
                    wire: arg[f"q{index}"]
                    for index, wire in enumerate(callee.body.input_wires)
                }
                _emit_global_phase(callee.body, context)
                _emit_region(callee.body.operations, wires, (), context, {})

        generator = GeneratedReusableCircuit(builder)
        native_circuit = generator.generate()
        generators.append(generator)
        circuits[cache_key] = native_circuit
        definitions[definition_key] = native_circuit

    visit_operations(program.operations)
    return circuits, generators


def _emit_global_phase(
    program: CircuitProgram,
    context: _PyQretContext,
) -> None:
    """Emit a concrete program-level global phase through PyQret.

    Args:
        program (CircuitProgram): Program whose global phase is active.
        context (_PyQretContext): Shared PyQret materialization context.

    Raises:
        EmitError: If the global phase is not concrete.
    """
    phase = float(evaluate_scalar(program.global_phase))
    if not phase:
        return
    context.intrinsic.global_phase(
        context.builder,
        phase,
        context.precision,
    )


def _emit_region(
    operations: tuple[CircuitInstruction, ...],
    wires: dict[WireId, Any],
    registers: Any,
    context: _PyQretContext,
    loop_values: dict[str, int],
) -> dict[WireId, Any]:
    """Emit one circuit region and return its wire-to-qubit mapping.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region instructions.
        wires (dict[WireId, Any]): Input virtual-wire to PyQret qubit mapping.
        registers (Any): PyQret output-register array.
        context (_PyQretContext): Shared PyQret materialization context.
        loop_values (dict[str, int]): Active loop-variable bindings.

    Returns:
        dict[WireId, Any]: Mapping extended with region output wires.

    Raises:
        EmitError: If the region contains unsupported dynamic control, reset,
            or transformed reusable calls.
    """
    intrinsic = context.intrinsic
    precision = context.precision
    environment = dict(wires)
    for operation in operations:
        if isinstance(operation, GateInstruction):
            qubits = tuple(environment[wire] for wire in operation.inputs)
            _emit_gate(
                operation,
                qubits,
                intrinsic,
                precision,
                loop_values,
                context.builder,
            )
            for output, qubit in zip(operation.outputs, qubits, strict=True):
                environment[output] = qubit
        elif isinstance(operation, MeasureInstruction):
            qubit = environment[operation.input]
            intrinsic.measure(qubit, registers[operation.clbit])
            environment[operation.output] = qubit
        elif isinstance(operation, MeasureVectorInstruction):
            qubits = tuple(environment[wire] for wire in operation.inputs)
            for qubit, clbit in zip(qubits, operation.clbits, strict=True):
                intrinsic.measure(qubit, registers[clbit])
            for output, qubit in zip(operation.outputs, qubits, strict=True):
                environment[output] = qubit
        elif isinstance(operation, BarrierInstruction):
            continue
        elif isinstance(operation, PauliEvolutionInstruction):
            if operation.realization is not PauliEvolutionRealization.GADGET:
                raise AssertionError(
                    "Quration Pauli evolution was not gadget-legalized"
                )
            qubits = tuple(environment[wire] for wire in operation.inputs)
            _emit_pauli_evolution(
                operation,
                qubits,
                intrinsic,
                precision,
                loop_values,
            )
            for output, qubit in zip(operation.outputs, qubits, strict=True):
                environment[output] = qubit
        elif isinstance(operation, ResetInstruction):
            raise EmitError("PyQret does not expose a reset intrinsic")
        elif isinstance(operation, ForInstruction):
            current = [environment[wire] for wire in operation.inputs]
            for index in operation.indexset:
                body_inputs = dict(zip(operation.inputs, current, strict=True))
                nested_loops = dict(loop_values)
                nested_loops[operation.loop_variable.name] = index
                body_environment = _emit_region(
                    operation.body,
                    body_inputs,
                    registers,
                    context,
                    nested_loops,
                )
                current = [body_environment[wire] for wire in operation.body_outputs]
            for output, qubit in zip(operation.outputs, current, strict=True):
                environment[output] = qubit
        elif isinstance(operation, CallInstruction):
            _emit_call(
                operation,
                environment,
                registers,
                context,
                loop_values,
            )
        elif isinstance(operation, (IfInstruction, WhileInstruction)):
            raise EmitError(
                "The initial Quration profile does not support runtime "
                f"{type(operation).__name__} control flow"
            )
        else:  # pragma: no cover - closed union defensive guard
            raise EmitError(f"Unsupported Quration instruction: {operation!r}")
    return environment


def _emit_call(
    operation: CallInstruction,
    environment: dict[WireId, Any],
    registers: Any,
    context: _PyQretContext,
    loop_values: dict[str, int],
) -> None:
    """Emit an untransformed reusable circuit as a native PyQret call.

    Args:
        operation (CallInstruction): Reusable circuit call.
        environment (dict[WireId, Any]): Enclosing wire mapping to update.
        registers (Any): PyQret output-register array.
        context (_PyQretContext): Shared PyQret materialization context.
        loop_values (dict[str, int]): Active loop-variable bindings.

    Raises:
        EmitError: If the reusable circuit has control or inverse transforms.
    """
    callee = operation.callee
    if callee.native_realization == "pyqret.multi_controlled_x":
        current = [environment[wire] for wire in operation.inputs]
        control_width, target_width = callee.operand_widths
        if target_width != 1 or control_width not in (1, 2):
            raise EmitError("PyQret supports one or two controls for native MCX")
        if control_width == 1:
            context.intrinsic.cx(current[0], current[1])
        else:
            context.intrinsic.ccx(current[0], current[1], current[2])
        for output, qubit in zip(operation.outputs, current, strict=True):
            environment[output] = qubit
        return
    if callee.controls or callee.inverse:
        raise EmitError(
            "Quration materialization requires controlled and inverse calls "
            "to be legalized before materialization"
        )
    current = [environment[wire] for wire in operation.inputs]
    native_circuit = context.reusable_circuits.get(id(callee))
    if native_circuit is None:
        raise EmitError(f"Reusable PyQret circuit {callee.name!r} was not predeclared")
    for _ in range(callee.power):
        native_circuit(*current)
    for output, qubit in zip(operation.outputs, current, strict=True):
        environment[output] = qubit


def _emit_pauli_evolution(
    operation: PauliEvolutionInstruction,
    qubits: tuple[Any, ...],
    intrinsic: Any,
    precision: float,
    loop_values: dict[str, int],
) -> None:
    """Legalize an abstract Pauli evolution to PyQret gate gadgets.

    Args:
        operation (PauliEvolutionInstruction): Abstract evolution.
        qubits (tuple[Any, ...]): Participating PyQret qubits.
        intrinsic (Any): PyQret intrinsic gate module.
        precision (float): Rotation synthesis precision.
        loop_values (dict[str, int]): Active induction values.

    Raises:
        EmitError: If the evolution time is not concrete.
    """
    import qamomile.observable as qm_o

    time = float(evaluate_scalar(operation.time, loop_values))
    for operators, coefficient in operation.hamiltonian:
        selected = [qubits[item.index] for item in operators]
        for item, qubit in zip(operators, selected, strict=True):
            if item.pauli is qm_o.Pauli.X:
                intrinsic.h(qubit)
            elif item.pauli is qm_o.Pauli.Y:
                intrinsic.sdag(qubit)
                intrinsic.h(qubit)
        for left, right in zip(selected, selected[1:]):
            intrinsic.cx(right, left)
        if selected:
            intrinsic.rz(
                selected[-1],
                2.0 * float(coefficient.real) * time,
                precision,
            )
        for left, right in reversed(list(zip(selected, selected[1:]))):
            intrinsic.cx(right, left)
        for item, qubit in reversed(list(zip(operators, selected, strict=True))):
            if item.pauli is qm_o.Pauli.X:
                intrinsic.h(qubit)
            elif item.pauli is qm_o.Pauli.Y:
                intrinsic.h(qubit)
                intrinsic.s(qubit)


def _emit_gate(
    operation: GateInstruction,
    qubits: tuple[Any, ...],
    intrinsic: Any,
    precision: float,
    loop_values: dict[str, int],
    builder: Any,
) -> None:
    """Emit or decompose one primitive circuit gate for PyQret.

    Args:
        operation (GateInstruction): Primitive gate instruction.
        qubits (tuple[Any, ...]): PyQret qubit operands in circuit order.
        intrinsic (Any): PyQret intrinsic gate module.
        precision (float): Rotation synthesis precision.
        loop_values (dict[str, int]): Active loop-variable bindings.
        builder (Any): PyQret circuit builder that owns zero-qubit phase
            operations.

    Raises:
        EmitError: If the gate kind has no Quration materialization rule.
    """
    kind = operation.kind
    angles = [
        float(evaluate_scalar(parameter, loop_values))
        for parameter in operation.parameters
    ]
    unary = {
        GateKind.H: intrinsic.h,
        GateKind.X: intrinsic.x,
        GateKind.Y: intrinsic.y,
        GateKind.Z: intrinsic.z,
        GateKind.S: intrinsic.s,
        GateKind.SDG: intrinsic.sdag,
        GateKind.T: intrinsic.t,
        GateKind.TDG: intrinsic.tdag,
    }
    if kind in unary:
        unary[kind](qubits[0])
    elif kind is GateKind.RX:
        intrinsic.rx(qubits[0], angles[0], precision)
    elif kind is GateKind.RY:
        intrinsic.ry(qubits[0], angles[0], precision)
    elif kind is GateKind.RZ:
        intrinsic.rz(qubits[0], angles[0], precision)
    elif kind is GateKind.P:
        intrinsic.global_phase(builder, angles[0] / 2.0, precision)
        intrinsic.rz(qubits[0], angles[0], precision)
    elif kind is GateKind.CX:
        intrinsic.cx(qubits[1], qubits[0])
    elif kind is GateKind.CY:
        intrinsic.cy(qubits[1], qubits[0])
    elif kind is GateKind.CZ:
        intrinsic.cz(qubits[1], qubits[0])
    elif kind is GateKind.TOFFOLI:
        intrinsic.ccx(qubits[2], qubits[0], qubits[1])
    elif kind is GateKind.SWAP:
        intrinsic.cx(qubits[1], qubits[0])
        intrinsic.cx(qubits[0], qubits[1])
        intrinsic.cx(qubits[1], qubits[0])
    elif kind is GateKind.RZZ:
        intrinsic.cx(qubits[1], qubits[0])
        intrinsic.rz(qubits[1], angles[0], precision)
        intrinsic.cx(qubits[1], qubits[0])
    elif kind is GateKind.CP:
        half = angles[0] / 2
        intrinsic.global_phase(builder, angles[0] / 4.0, precision)
        intrinsic.rz(qubits[0], half, precision)
        intrinsic.cx(qubits[1], qubits[0])
        intrinsic.rz(qubits[1], -half, precision)
        intrinsic.cx(qubits[1], qubits[0])
        intrinsic.rz(qubits[1], half, precision)
    elif kind is GateKind.CRX:
        _emit_controlled_rotation(
            intrinsic,
            "x",
            qubits[0],
            qubits[1],
            angles[0],
            precision,
        )
    elif kind is GateKind.CRY:
        _emit_controlled_rotation(
            intrinsic,
            "y",
            qubits[0],
            qubits[1],
            angles[0],
            precision,
        )
    elif kind is GateKind.CRZ:
        _emit_controlled_rotation(
            intrinsic,
            "z",
            qubits[0],
            qubits[1],
            angles[0],
            precision,
        )
    elif kind is GateKind.CH:
        intrinsic.s(qubits[1])
        intrinsic.h(qubits[1])
        intrinsic.t(qubits[1])
        intrinsic.cx(qubits[1], qubits[0])
        intrinsic.tdag(qubits[1])
        intrinsic.h(qubits[1])
        intrinsic.sdag(qubits[1])
    else:
        raise EmitError(f"Unsupported Quration gate kind: {kind}")


def _emit_controlled_rotation(
    intrinsic: Any,
    axis: str,
    control: Any,
    target: Any,
    angle: float,
    precision: float,
) -> None:
    """Decompose a controlled Pauli rotation into PyQret intrinsics.

    Args:
        intrinsic (Any): PyQret intrinsic gate module.
        axis (str): Rotation axis, one of ``"x"``, ``"y"``, or ``"z"``.
        control (Any): PyQret control qubit.
        target (Any): PyQret target qubit.
        angle (float): Rotation angle in radians.
        precision (float): Rotation synthesis precision.

    Raises:
        EmitError: If ``axis`` is not supported.
    """
    if axis == "x":
        intrinsic.rz(target, math.pi / 2, precision)
        _emit_controlled_rotation(intrinsic, "y", control, target, angle, precision)
        intrinsic.rz(target, -math.pi / 2, precision)
        return
    rotation = intrinsic.ry if axis == "y" else intrinsic.rz if axis == "z" else None
    if rotation is None:
        raise EmitError(f"Unsupported controlled rotation axis: {axis!r}")
    rotation(target, angle / 2, precision)
    intrinsic.cx(target, control)
    rotation(target, -angle / 2, precision)
    intrinsic.cx(target, control)
