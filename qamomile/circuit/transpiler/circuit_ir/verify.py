"""Structural verifier for backend-neutral circuit programs."""

from __future__ import annotations

from qamomile._utils import is_close_zero
from qamomile.circuit.transpiler.circuit_ir.model import (
    BarrierInstruction,
    BinaryExpr,
    CallInstruction,
    CircuitInstruction,
    CircuitProgram,
    ClassicalBitExpr,
    ForInstruction,
    GateInstruction,
    IfInstruction,
    LiteralExpr,
    LoopVariableExpr,
    MeasureInstruction,
    MeasureVectorInstruction,
    ParameterExpr,
    PauliEvolutionInstruction,
    ResetInstruction,
    ScalarExpr,
    UnaryExpr,
    WhileInstruction,
    WireId,
)
from qamomile.circuit.transpiler.gate_emitter import GATE_SPECS, GateKind


def verify_circuit(program: CircuitProgram) -> None:
    """Verify wire linearity, regions, expressions, and slot bounds.

    Args:
        program (CircuitProgram): Immutable circuit program to verify.

    Raises:
        ValueError: If the program contains duplicate wire definitions,
            consumes a non-live wire, has malformed structured-region yields,
            references an invalid classical bit, or reports incorrect outputs.
    """
    if program.num_qubits < 0 or program.num_clbits < 0:
        raise ValueError("Circuit slot counts must be non-negative")
    if len(program.input_wires) != program.num_qubits:
        raise ValueError("Input wire count does not match num_qubits")
    if len(program.output_wires) != program.num_qubits:
        raise ValueError("Output wire count does not match num_qubits")
    if len(set(program.input_wires)) != len(program.input_wires):
        raise ValueError("Circuit input wires must be unique")
    _verify_expression(program.global_phase, program.num_clbits)

    definitions = set(program.input_wires)
    final = _verify_region(
        program.operations,
        program.input_wires,
        definitions,
        program.num_clbits,
    )
    if final != program.output_wires:
        raise ValueError("Circuit output wires do not match the final region state")


def _verify_expression(expression: ScalarExpr, num_clbits: int) -> None:
    """Verify all classical-bit references in a scalar expression.

    Args:
        expression (ScalarExpr): Expression to inspect recursively.
        num_clbits (int): Number of allocated classical bit slots.

    Raises:
        ValueError: If a classical-bit reference is outside the allocated
            circuit range or ``expression`` is not a supported scalar node.
    """
    if isinstance(expression, ClassicalBitExpr):
        if expression.index < 0 or expression.index >= num_clbits:
            raise ValueError(
                f"Classical bit expression {expression.index} is out of range"
            )
    elif isinstance(expression, BinaryExpr):
        _verify_expression(expression.left, num_clbits)
        _verify_expression(expression.right, num_clbits)
    elif isinstance(expression, UnaryExpr):
        _verify_expression(expression.operand, num_clbits)
    elif not isinstance(expression, (LiteralExpr, ParameterExpr, LoopVariableExpr)):
        raise ValueError(f"Unsupported scalar expression: {type(expression).__name__}")


def _define_wires(wires: tuple[WireId, ...], definitions: set[WireId]) -> None:
    """Register fresh wire definitions.

    Args:
        wires (tuple[WireId, ...]): Wire identifiers produced by an operation.
        definitions (set[WireId]): Program-global definition registry.

    Raises:
        ValueError: If an output is duplicated locally or was already defined.
    """
    if len(set(wires)) != len(wires):
        raise ValueError("An instruction defines the same wire more than once")
    duplicate = set(wires) & definitions
    if duplicate:
        raise ValueError(f"Wire definitions are not unique: {sorted(duplicate)}")
    definitions.update(wires)


def _advance(
    live: tuple[WireId, ...],
    inputs: tuple[WireId, ...],
    outputs: tuple[WireId, ...],
    definitions: set[WireId],
) -> tuple[WireId, ...]:
    """Consume live inputs and publish fresh output wires.

    Args:
        live (tuple[WireId, ...]): Current wire state in physical-slot order.
        inputs (tuple[WireId, ...]): Wires consumed by an instruction.
        outputs (tuple[WireId, ...]): Wires produced by an instruction.
        definitions (set[WireId]): Program-global definition registry.

    Returns:
        tuple[WireId, ...]: Updated live wires in physical-slot order.

    Raises:
        ValueError: If an input is not live, input/output arities differ, or
            an output wire is not fresh.
    """
    if len(inputs) != len(outputs):
        raise ValueError("Quantum instruction input/output arities must match")
    if len(set(inputs)) != len(inputs):
        raise ValueError("Quantum instruction consumes the same wire twice")
    missing = set(inputs) - set(live)
    if missing:
        raise ValueError(f"Quantum instruction consumes non-live wires: {missing}")
    _define_wires(outputs, definitions)
    replacements = dict(zip(inputs, outputs, strict=True))
    return tuple(replacements.get(wire, wire) for wire in live)


def _verify_control_inputs(
    inputs: tuple[WireId, ...],
    live: tuple[WireId, ...],
) -> None:
    """Verify a structured region receives the complete live wire state.

    Args:
        inputs (tuple[WireId, ...]): Structured operation input wires.
        live (tuple[WireId, ...]): Current enclosing-region wire state in
            physical-slot order.

    Raises:
        ValueError: If the structured operation omits, duplicates, or adds a
            wire relative to the enclosing region.
    """
    if inputs != live:
        raise ValueError(
            "Structured control inputs must equal the ordered live wire state"
        )


def _verify_region(
    operations: tuple[CircuitInstruction, ...],
    initial_live: tuple[WireId, ...],
    definitions: set[WireId],
    num_clbits: int,
) -> tuple[WireId, ...]:
    """Verify one structured region and return its yielded wire state.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region instructions.
        initial_live (tuple[WireId, ...]): Wires available at region entry in
            physical-slot order.
        definitions (set[WireId]): Program-global wire definitions.
        num_clbits (int): Number of allocated classical bit slots.

    Returns:
        tuple[WireId, ...]: Live wires yielded in physical-slot order.

    Raises:
        ValueError: If any nested instruction violates circuit invariants.
    """
    live = initial_live
    for operation in operations:
        if isinstance(operation, GateInstruction):
            if operation.kind is GateKind.MEASURE:
                raise ValueError(
                    "GateKind.MEASURE must use MeasureInstruction in circuit IR"
                )
            specification = GATE_SPECS[operation.kind]
            if len(operation.inputs) != specification.num_qubits:
                raise ValueError(
                    f"{operation.kind.name} gate expects "
                    f"{specification.num_qubits} qubits, received "
                    f"{len(operation.inputs)}"
                )
            expected_parameters = 1 if specification.has_angle else 0
            if len(operation.parameters) != expected_parameters:
                raise ValueError(
                    f"{operation.kind.name} gate expects {expected_parameters} "
                    f"parameters, received {len(operation.parameters)}"
                )
            for parameter in operation.parameters:
                _verify_expression(parameter, num_clbits)
            live = _advance(live, operation.inputs, operation.outputs, definitions)
        elif isinstance(operation, MeasureInstruction):
            if operation.clbit < 0 or operation.clbit >= num_clbits:
                raise ValueError(f"Measurement clbit {operation.clbit} is out of range")
            live = _advance(
                live,
                (operation.input,),
                (operation.output,),
                definitions,
            )
        elif isinstance(operation, MeasureVectorInstruction):
            if len(operation.inputs) != len(operation.clbits):
                raise ValueError("Vector measurement wire/clbit arities must match")
            if len(set(operation.clbits)) != len(operation.clbits):
                raise ValueError("Vector measurement classical bits must be unique")
            if any(clbit < 0 or clbit >= num_clbits for clbit in operation.clbits):
                raise ValueError("Vector measurement clbit is out of range")
            live = _advance(
                live,
                operation.inputs,
                operation.outputs,
                definitions,
            )
        elif isinstance(operation, ResetInstruction):
            live = _advance(
                live,
                (operation.input,),
                (operation.output,),
                definitions,
            )
        elif isinstance(operation, PauliEvolutionInstruction):
            _verify_expression(operation.time, num_clbits)
            live = _advance(live, operation.inputs, operation.outputs, definitions)
        elif isinstance(operation, BarrierInstruction):
            if set(operation.wires) - set(live):
                raise ValueError("Barrier references a non-live wire")
        elif isinstance(operation, CallInstruction):
            if operation.callee.power <= 0:
                raise ValueError("Reusable call power must be positive")
            if operation.callee.controls < 0:
                raise ValueError("Reusable call control count must be non-negative")
            if any(width <= 0 for width in operation.callee.operand_widths):
                raise ValueError("Reusable call operand widths must be positive")
            if (
                operation.callee.operand_widths
                and sum(operation.callee.operand_widths)
                != operation.callee.body.num_qubits
            ):
                raise ValueError(
                    "Reusable call operand widths must cover the fallback body"
                )
            argument_names: set[str] = set()
            for name, expression in operation.callee.call_arguments:
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        "Reusable call argument names must be non-empty strings"
                    )
                if name in argument_names:
                    raise ValueError("Reusable call argument names must be unique")
                argument_names.add(name)
                _verify_expression(expression, num_clbits)
            if operation.callee.opaque:
                if operation.callee.identity is None:
                    raise ValueError("Opaque reusable call requires semantic identity")
                if (
                    operation.callee.body.operations
                    or operation.callee.body.num_clbits != 0
                    or not isinstance(
                        operation.callee.body.global_phase,
                        LiteralExpr,
                    )
                    or not is_close_zero(
                        float(operation.callee.body.global_phase.value)
                    )
                ):
                    raise ValueError(
                        "Opaque reusable call body must be an empty arity placeholder"
                    )
            verify_circuit(operation.callee.body)
            if len(operation.inputs) != operation.callee.num_qubits:
                raise ValueError("Reusable call input arity does not match its callee")
            live = _advance(live, operation.inputs, operation.outputs, definitions)
        elif isinstance(operation, IfInstruction):
            _verify_expression(operation.condition, num_clbits)
            _verify_control_inputs(operation.inputs, live)
            true_live = _verify_region(
                operation.true_body,
                operation.inputs,
                definitions,
                num_clbits,
            )
            false_live = _verify_region(
                operation.false_body,
                operation.inputs,
                definitions,
                num_clbits,
            )
            if true_live != operation.true_outputs:
                raise ValueError("True branch outputs do not match its final wires")
            if false_live != operation.false_outputs:
                raise ValueError("False branch outputs do not match its final wires")
            live = _advance(live, operation.inputs, operation.outputs, definitions)
        elif isinstance(operation, ForInstruction):
            _verify_control_inputs(operation.inputs, live)
            body_live = _verify_region(
                operation.body,
                operation.inputs,
                definitions,
                num_clbits,
            )
            if body_live != operation.body_outputs:
                raise ValueError("For-loop body outputs do not match its final wires")
            live = _advance(live, operation.inputs, operation.outputs, definitions)
        elif isinstance(operation, WhileInstruction):
            _verify_expression(operation.condition, num_clbits)
            _verify_control_inputs(operation.inputs, live)
            body_live = _verify_region(
                operation.body,
                operation.inputs,
                definitions,
                num_clbits,
            )
            if body_live != operation.body_outputs:
                raise ValueError("While-loop body outputs do not match its final wires")
            live = _advance(live, operation.inputs, operation.outputs, definitions)
        else:  # pragma: no cover - closed union defensive guard
            raise ValueError(f"Unknown circuit instruction: {type(operation).__name__}")
    return live
