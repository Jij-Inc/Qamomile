"""Reconcile runtime parameter metadata with immutable circuit IR usage."""

from __future__ import annotations

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
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata


def collect_scalar_parameter_names(expression: ScalarExpr) -> set[str]:
    """Collect runtime parameter names from one scalar expression.

    Args:
        expression (ScalarExpr): Closed CircuitIR scalar expression to inspect.

    Returns:
        set[str]: Names referenced by the expression.

    Raises:
        EmitError: If an unknown scalar-expression node reaches CircuitIR.
    """
    if isinstance(expression, ParameterExpr):
        return {expression.name}
    if isinstance(expression, BinaryExpr):
        return collect_scalar_parameter_names(
            expression.left
        ) | collect_scalar_parameter_names(expression.right)
    if isinstance(expression, UnaryExpr):
        return collect_scalar_parameter_names(expression.operand)
    if isinstance(
        expression,
        (LiteralExpr, ClassicalBitExpr, LoopVariableExpr),
    ):
        return set()
    raise EmitError(
        "Unknown CircuitIR scalar expression while collecting runtime "
        f"parameter usage: {type(expression).__name__}"
    )


def collect_program_parameter_names(program: CircuitProgram) -> set[str]:
    """Collect runtime parameter names used by one complete circuit program.

    Shared reusable bodies are scanned once. An active-call guard rejects a
    malformed cyclic reusable-call graph instead of recursing indefinitely.

    Args:
        program (CircuitProgram): Immutable circuit program to inspect.

    Returns:
        set[str]: Names referenced anywhere in the program or nested bodies.

    Raises:
        EmitError: If the program contains an unknown node or a cyclic
            reusable-call graph.
    """
    return set(_collect_program_parameter_names(program, {}, set()))


def reconcile_parameter_metadata(
    program: CircuitProgram,
    metadata: ParameterMetadata,
) -> ParameterMetadata:
    """Filter provisional runtime metadata to parameters used by CircuitIR.

    Lowering may resolve formal runtime arguments before it knows whether the
    callee body uses them. The immutable circuit program is the authoritative
    record of actual use, while the provisional metadata retains ABI ordering,
    source references, container kinds, and engine parameter placeholders.

    Args:
        program (CircuitProgram): Verified immutable circuit program.
        metadata (ParameterMetadata): Provisional segment parameter metadata.

    Returns:
        ParameterMetadata: Metadata containing exactly the used parameter
            slots, in their original ABI order.

    Raises:
        EmitError: If CircuitIR references a runtime parameter absent from the
            provisional metadata, or the CircuitIR graph is malformed.
        ValueError: If retained array slots have inconsistent ranks.
    """
    used_names = collect_program_parameter_names(program)
    candidate_names = {parameter.name for parameter in metadata.parameters}
    missing = used_names - candidate_names
    if missing:
        raise EmitError(
            "CircuitIR references runtime parameters missing from the compiled "
            f"ABI metadata: {sorted(missing)}"
        )

    filtered = ParameterMetadata(
        parameters=[
            parameter
            for parameter in metadata.parameters
            if parameter.name in used_names
        ]
    )
    filtered.arrays = {
        name: metadata.arrays.get(name, derived)
        for name, derived in filtered.arrays.items()
    }
    return filtered


def _collect_program_parameter_names(
    program: CircuitProgram,
    memo: dict[int, tuple[CircuitProgram, frozenset[str]]],
    active: set[int],
) -> frozenset[str]:
    """Collect one program while sharing nested-body results.

    Args:
        program (CircuitProgram): Program or reusable body to inspect.
        memo (dict[int, tuple[CircuitProgram, frozenset[str]]]): Completed
            body results keyed by object identity and retaining each object.
        active (set[int]): Program identities active on the current call path.

    Returns:
        frozenset[str]: Runtime parameter names used by ``program``.

    Raises:
        EmitError: If a reusable-call cycle, identity collision, or unknown
            CircuitIR node is encountered.
    """
    identity = id(program)
    if identity in active:
        raise EmitError(
            "CircuitIR reusable-call graph contains a cycle while collecting "
            f"runtime parameter usage for {program.name!r}"
        )
    cached = memo.get(identity)
    if cached is not None:
        if cached[0] is not program:  # pragma: no cover - defensive invariant
            raise EmitError(
                "CircuitIR parameter-use cache observed an object identity collision"
            )
        return cached[1]

    active.add(identity)
    try:
        names = collect_scalar_parameter_names(program.global_phase)
        names.update(_collect_region_parameter_names(program.operations, memo, active))
    finally:
        active.remove(identity)
    result = frozenset(names)
    memo[identity] = (program, result)
    return result


def _collect_region_parameter_names(
    operations: tuple[CircuitInstruction, ...],
    memo: dict[int, tuple[CircuitProgram, frozenset[str]]],
    active: set[int],
) -> set[str]:
    """Collect runtime parameter names from one structured region.

    Args:
        operations (tuple[CircuitInstruction, ...]): Region instructions.
        memo (dict[int, tuple[CircuitProgram, frozenset[str]]]): Completed
            reusable-body results.
        active (set[int]): Program identities active on the current call path.

    Returns:
        set[str]: Runtime parameter names used by the region.

    Raises:
        EmitError: If an unknown instruction or malformed nested graph is
            encountered.
    """
    names: set[str] = set()
    for operation in operations:
        names.update(_collect_instruction_parameter_names(operation, memo, active))
    return names


def _collect_instruction_parameter_names(
    operation: CircuitInstruction,
    memo: dict[int, tuple[CircuitProgram, frozenset[str]]],
    active: set[int],
) -> set[str]:
    """Collect runtime parameter names from one CircuitIR instruction.

    Args:
        operation (CircuitInstruction): Instruction to inspect.
        memo (dict[int, tuple[CircuitProgram, frozenset[str]]]): Completed
            reusable-body results.
        active (set[int]): Program identities active on the current call path.

    Returns:
        set[str]: Runtime parameter names used by the instruction.

    Raises:
        EmitError: If an unknown instruction or malformed nested graph is
            encountered.
    """
    if isinstance(operation, GateInstruction):
        names: set[str] = set()
        for parameter in operation.parameters:
            names.update(collect_scalar_parameter_names(parameter))
        return names
    if isinstance(operation, PauliEvolutionInstruction):
        return collect_scalar_parameter_names(operation.time)
    if isinstance(operation, CallInstruction):
        return set(
            _collect_program_parameter_names(operation.callee.body, memo, active)
        )
    if isinstance(operation, ForInstruction):
        if not operation.indexset:
            return set()
        return _collect_region_parameter_names(operation.body, memo, active)
    if isinstance(operation, IfInstruction):
        names = collect_scalar_parameter_names(operation.condition)
        names.update(collect_scalar_parameter_names(operation.true_global_phase))
        names.update(collect_scalar_parameter_names(operation.false_global_phase))
        names.update(_collect_region_parameter_names(operation.true_body, memo, active))
        names.update(
            _collect_region_parameter_names(operation.false_body, memo, active)
        )
        return names
    if isinstance(operation, WhileInstruction):
        names = collect_scalar_parameter_names(operation.condition)
        names.update(collect_scalar_parameter_names(operation.body_global_phase))
        names.update(_collect_region_parameter_names(operation.body, memo, active))
        return names
    if isinstance(
        operation,
        (
            MeasureInstruction,
            MeasureVectorInstruction,
            ResetInstruction,
            BarrierInstruction,
        ),
    ):
        return set()
    raise EmitError(
        "Unknown CircuitIR instruction while collecting runtime parameter "
        f"usage: {type(operation).__name__}"
    )
