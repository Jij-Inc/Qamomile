"""Direct lowering from prepared Qamomile semantics to Guppy-compatible HUGR."""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    NotOp,
)
from qamomile.circuit.ir.operation.callable import (
    CallableRef,
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    MeasureOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import CInitOperation, QInitOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.types import BitType, FloatType, QubitType, UIntType
from qamomile.circuit.ir.types.primitives import ValueType
from qamomile.circuit.ir.value import (
    ArrayValue,
    Value,
    resolve_root_array_index,
    resolve_root_qubit_address,
)
from qamomile.circuit.transpiler.artifact import (
    CompilationMetadata,
    CompiledProgram,
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.prepared import PreparedModule


@dataclasses.dataclass(frozen=True)
class HugrCompilationPlan:
    """Describe the callable symbols emitted into one HUGR module.

    Args:
        definitions (tuple[CallableRef, ...]): Reachable body-backed callable
            definitions emitted as HUGR functions.
    """

    definitions: tuple[CallableRef, ...]


class HugrTarget:
    """Plan, lower, package, and validate a Guppy-compatible HUGR target."""

    @property
    def name(self) -> str:
        """Return the stable compilation target name.

        Returns:
            str: ``"hugr"``.
        """
        return "hugr"

    def plan(self, program: PreparedModule) -> HugrCompilationPlan:
        """Select reachable body-backed callables for HUGR functions.

        Args:
            program (PreparedModule): Prepared hierarchical semantic program.

        Returns:
            HugrCompilationPlan: Stable callable-definition order.
        """
        definitions = tuple(
            ref
            for ref, definition in program.definitions.items()
            if definition.body is not None
        )
        return HugrCompilationPlan(definitions=definitions)

    def compile(
        self,
        program: PreparedModule,
        plan: HugrCompilationPlan,
    ) -> CompiledProgram[Any]:
        """Lower a prepared semantic module directly to a HUGR package.

        Args:
            program (PreparedModule): Prepared hierarchical semantic program.
            plan (HugrCompilationPlan): Callable emission plan.

        Returns:
            CompiledProgram[Any]: HUGR package and target metadata.

        Raises:
            ImportError: If HUGR or TKET extension packages are unavailable.
            EmitError: If a semantic operation has no HUGR lowering yet.
        """
        package = _lower_module(program, plan)
        return CompiledProgram(
            artifact=package,
            abi=program.abi,
            metadata=CompilationMetadata(
                target=self.name,
                pipeline="program_graph",
                properties={"extension_family": "tket"},
            ),
        )

    def validate(self, artifact: Any) -> None:
        """Validate a HUGR package with the native Rust-backed validator.

        Args:
            artifact (Any): ``hugr.package.Package`` to validate.

        Raises:
            ImportError: If the HUGR package is unavailable.
            HugrCliError: If HUGR validation rejects the package.
        """
        try:
            from hugr.cli import validate
        except ImportError as error:
            raise ImportError(
                "HUGR support requires the optional 'hugr' and 'tket-exts' packages."
            ) from error
        validate(artifact.to_bytes())


def _require_hugr() -> tuple[Any, Any, Any, Any, Any]:
    """Import HUGR builders, types, values, and TKET extensions.

    Returns:
        tuple[Any, Any, Any, Any, Any]: HUGR build module, type module, value
            module, package class, and ``tket_exts`` module.

    Raises:
        ImportError: If HUGR or TKET extensions are unavailable.
    """
    try:
        import tket_exts

        import hugr.build as build
        import hugr.tys as tys
        import hugr.val as val
        from hugr.package import Package
    except ImportError as error:
        raise ImportError(
            "HUGR support requires optional 'hugr' and 'tket-exts' packages."
        ) from error
    return build, tys, val, Package, tket_exts


def _lower_module(program: PreparedModule, plan: HugrCompilationPlan) -> Any:
    """Build one HUGR package from a prepared Qamomile module.

    Args:
        program (PreparedModule): Prepared hierarchical semantic program.
        plan (HugrCompilationPlan): Callable emission plan.

    Returns:
        Any: ``hugr.package.Package`` containing one module and its TKET
            extension definitions.

    Raises:
        EmitError: If type or operation lowering is unsupported.
        ImportError: If HUGR dependencies are unavailable.
    """
    build, _, _, Package, tket_exts = _require_hugr()
    module = build.Module()
    functions: dict[CallableRef, Any] = {}

    for ref in plan.definitions:
        body = program.body(ref)
        functions[ref] = module.define_function(
            _symbol_name(ref),
            [_lower_value_type(value) for value in body.input_values],
            [_lower_value_type(value) for value in body.output_values],
        )

    entry_inputs = _entry_inputs(program.entrypoint)
    main = module.define_function(
        "main",
        [_lower_value_type(value) for value in entry_inputs],
        [_lower_value_type(value) for value in program.entrypoint.output_values],
        visibility="Public",
    )

    for ref in plan.definitions:
        _lower_block(program.body(ref), functions[ref], functions, None)
    _lower_block(program.entrypoint, main, functions, entry_inputs)

    extensions = [tket_exts.quantum(), tket_exts.rotation(), tket_exts.bool()]
    return Package([module.hugr], extensions)


def _symbol_name(ref: CallableRef) -> str:
    """Convert a callable reference to a valid stable HUGR symbol.

    Args:
        ref (CallableRef): Qamomile callable identity.

    Returns:
        str: Sanitized namespace, name, and version.
    """
    raw = f"{ref.namespace}__{ref.name}__v{ref.version}"
    return re.sub(r"[^A-Za-z0-9_]", "_", raw)


def _entry_inputs(block: Block) -> list[Value]:
    """Return runtime parameter values exposed by the HUGR entrypoint.

    Args:
        block (Block): Prepared top-level semantic block.

    Returns:
        list[Value]: Runtime parameter values in block input order.
    """
    parameter_uuids = {value.uuid for value in block.parameters.values()}
    return [value for value in block.input_values if value.uuid in parameter_uuids]


def _lower_type(value_type: ValueType) -> Any:
    """Map a scalar Qamomile type to its HUGR counterpart.

    Args:
        value_type (ValueType): Qamomile semantic value type.

    Returns:
        Any: HUGR type object.

    Raises:
        EmitError: If the type has no HUGR lowering.
    """
    _, tys, _, _, _ = _require_hugr()
    if isinstance(value_type, QubitType):
        return tys.Qubit
    if isinstance(value_type, BitType):
        return tys.Bool
    if isinstance(value_type, FloatType):
        from hugr.std.float import FLOAT_T

        return FLOAT_T
    if isinstance(value_type, UIntType):
        from hugr.std.int import INT_T

        return INT_T
    raise EmitError(f"Unsupported HUGR value type: {value_type.label()}")


def _lower_value_type(value: Value) -> Any:
    """Map a scalar or fixed-size array value to a HUGR carrier type.

    Args:
        value (Value): Qamomile value including shape metadata.

    Returns:
        Any: Scalar type or fixed-length HUGR tuple type.
    """
    if isinstance(value, ArrayValue):
        _, tys, _, _, _ = _require_hugr()
        return tys.Tuple(*[_lower_type(value.type) for _ in range(_array_size(value))])
    return _lower_type(value.type)


def _array_size(value: ArrayValue) -> int:
    """Return a statically known one-dimensional array size.

    Args:
        value (ArrayValue): Array value to inspect.

    Returns:
        int: Non-negative fixed size.

    Raises:
        EmitError: If shape is missing, dynamic, or multidimensional.
    """
    if len(value.shape) != 1 or not value.shape[0].is_constant():
        raise EmitError(f"HUGR array {value.name!r} requires one static dimension")
    size = int(value.shape[0].get_const())
    if size < 0:
        raise EmitError(f"HUGR array {value.name!r} has negative size")
    return size


def _lower_block(
    block: Block,
    builder: Any,
    functions: dict[CallableRef, Any],
    explicit_inputs: list[Value] | None,
) -> None:
    """Lower one Qamomile block into a HUGR function dataflow graph.

    Args:
        block (Block): Semantic callable body.
        builder (Any): ``hugr.build.Function`` receiving operations.
        functions (dict[CallableRef, Any]): Predeclared callable functions.
        explicit_inputs (list[Value] | None): Input values corresponding to
            HUGR function ports. ``None`` uses every block input.

    Raises:
        EmitError: If an operation or value is unsupported or unresolved.
    """
    inputs = block.input_values if explicit_inputs is None else explicit_inputs
    environment: dict[str, Any] = {}
    for value, wire in zip(inputs, builder.inputs(), strict=True):
        if isinstance(value, ArrayValue):
            from hugr import ops

            element_types = [_lower_type(value.type) for _ in range(_array_size(value))]
            environment[value.uuid] = list(
                builder.add_op(ops.UnpackTuple(element_types), wire)
            )
        else:
            environment[value.uuid] = wire
    live_qubits: dict[str, Any] = {}
    for value in inputs:
        if not value.type.is_quantum():
            continue
        if isinstance(value, ArrayValue):
            live_qubits.update(
                {
                    f"{value.uuid}:{index}": wire
                    for index, wire in enumerate(environment[value.uuid])
                }
            )
        else:
            live_qubits[value.uuid] = environment[value.uuid]

    for operation in block.operations:
        _lower_operation(operation, builder, environment, live_qubits, functions)

    outputs = []
    for value in block.output_values:
        if value.uuid not in environment:
            raise EmitError(
                f"HUGR output value {value.name!r} ({value.uuid}) is unresolved"
            )
        outputs.append(_pack_value(value, environment, builder))

    output_qubits: set[str] = set()
    for value in block.output_values:
        if not value.type.is_quantum():
            continue
        if isinstance(value, ArrayValue):
            output_qubits.update(
                f"{value.uuid}:{index}" for index in range(_array_size(value))
            )
        else:
            output_qubits.add(value.uuid)
    from tket_exts import quantum

    for uuid, wire in live_qubits.items():
        if uuid not in output_qubits:
            builder.add_op(quantum.qFree, wire)
    builder.set_outputs(*outputs)


def _pack_value(value: Value, environment: dict[str, Any], builder: Any) -> Any:
    """Return a scalar wire or pack array elements into a HUGR tuple.

    Args:
        value (Value): Value to materialize at a function boundary.
        environment (dict[str, Any]): UUID-to-wire mapping.
        builder (Any): HUGR dataflow builder.

    Returns:
        Any: Scalar or tuple wire.

    Raises:
        EmitError: If the value is unresolved.
    """
    try:
        resolved = environment[value.uuid]
    except KeyError as error:
        raise EmitError(f"Unresolved HUGR value {value.name!r}") from error
    if not isinstance(value, ArrayValue):
        return resolved
    from hugr import ops

    element_types = [_lower_type(value.type) for _ in range(_array_size(value))]
    [packed] = builder.add_op(ops.MakeTuple(element_types), *resolved)
    return packed


def _lower_operation(
    operation: Operation,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
    functions: dict[CallableRef, Any],
) -> None:
    """Lower one semantic operation into a HUGR function builder.

    Args:
        operation (Operation): Qamomile semantic operation.
        builder (Any): HUGR function or region builder.
        environment (dict[str, Any]): Qamomile UUID to HUGR wire mapping.
        live_qubits (dict[str, Any]): Current live quantum values.
        functions (dict[CallableRef, Any]): Predeclared HUGR functions.

    Raises:
        EmitError: If the operation has no direct program-graph lowering.
    """
    match operation:
        case CInitOperation():
            _lower_cinit(operation, builder, environment)
        case QInitOperation():
            _lower_qinit(operation, builder, environment, live_qubits)
        case GateOperation():
            _lower_gate(operation, builder, environment, live_qubits)
        case ControlledUOperation() | InverseBlockOperation():
            _lower_transformed_call(
                operation, builder, environment, live_qubits, functions
            )
        case MeasureOperation() | MeasureVectorOperation():
            _lower_measure(operation, builder, environment, live_qubits)
        case ProjectOperation():
            _lower_project(operation, builder, environment, live_qubits)
        case ResetOperation():
            _lower_reset(operation, builder, environment, live_qubits)
        case InvokeOperation():
            _lower_call(operation, builder, environment, live_qubits, functions)
        case ForOperation():
            _lower_for(operation, builder, environment, live_qubits, functions)
        case IfOperation():
            _lower_if(operation, builder, environment, live_qubits, functions)
        case WhileOperation():
            _lower_while(operation, builder, environment, live_qubits, functions)
        case BinOp():
            _lower_binop(operation, builder, environment)
        case CompOp():
            _lower_compop(operation, builder, environment)
        case NotOp():
            from hugr.std.logic import Not

            [result] = builder.add(Not(environment[operation.operands[0].uuid]))
            environment[operation.results[0].uuid] = result
        case ReturnOperation():
            return
        case _:
            raise EmitError(
                f"Unsupported direct HUGR lowering for "
                f"{type(operation).__name__}. Circuit segmentation is "
                f"intentionally not used for this target."
            )


def _lower_for(
    operation: ForOperation,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
    functions: dict[CallableRef, Any],
) -> None:
    """Lower a for-loop with static unrolling or a native runtime TailLoop.

    Args:
        operation (ForOperation): Prepared loop operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.
        live_qubits (dict[str, Any]): Live quantum mapping.
        functions (dict[CallableRef, Any]): Predeclared HUGR functions.

    Raises:
        EmitError: If runtime bounds or the step cannot be represented.
    """
    if len(operation.operands) < 3:
        raise EmitError("HUGR for-loop requires start, stop, and step operands")
    if not all(value.is_constant() for value in operation.operands[:3]):
        _lower_runtime_for(
            operation,
            builder,
            environment,
            live_qubits,
            functions,
        )
        return
    start, stop, step = (int(value.get_const()) for value in operation.operands[:3])
    if step == 0:
        raise EmitError("HUGR for-loop step cannot be zero")
    carried = {
        region.block_arg.uuid: environment[region.init.uuid]
        for region in operation.region_args
    }
    for index in range(start, stop, step):
        if operation.loop_var_value is not None:
            from hugr.std.int import IntVal

            [loop_wire] = builder.load(IntVal(index))
            environment[operation.loop_var_value.uuid] = loop_wire
            environment[f"__index__:{operation.loop_var_value.uuid}"] = index
        environment.update(carried)
        for nested in operation.operations:
            _lower_operation(
                nested,
                builder,
                environment,
                live_qubits,
                functions,
            )
        carried = {
            region.block_arg.uuid: environment[region.yielded.uuid]
            for region in operation.region_args
        }
    for region in operation.region_args:
        wire = carried.get(region.block_arg.uuid, environment[region.init.uuid])
        environment[region.result.uuid] = wire
        if region.result.type.is_quantum():
            live_qubits.pop(region.init.uuid, None)
            live_qubits[region.result.uuid] = wire


def _lower_runtime_for(
    operation: ForOperation,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
    functions: dict[CallableRef, Any],
) -> None:
    """Lower a runtime-bounded range to a native HUGR TailLoop.

    Args:
        operation (ForOperation): Loop with at least one runtime bound.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.
        live_qubits (dict[str, Any]): Live quantum mapping.
        functions (dict[CallableRef, Any]): Predeclared HUGR functions.

    Raises:
        EmitError: If the step is dynamic, zero, or a bound is unresolved.
    """
    start, stop, step = operation.operands[:3]
    if not step.is_constant():
        raise EmitError("HUGR runtime for-loop step must be compile-time constant")
    step_value = int(step.get_const())
    if step_value == 0:
        raise EmitError("HUGR for-loop step cannot be zero")

    start_wire = _integer_value_wire(start, builder, environment)
    stop_wire = _integer_value_wire(stop, builder, environment)
    captured_ids = _for_capture_ids(operation, environment)
    captured_wires = [environment[uuid] for uuid in captured_ids]
    loop = builder.add_tail_loop([], [start_wire, stop_wire, *captured_wires])
    loop_inputs = loop.inputs()
    index_wire, current_stop, *current_captured = loop_inputs

    from hugr import ops, tys
    from hugr.std.int import INT_OPS_EXTENSION, INT_T, IntVal

    comparison_name = "ilt_s" if step_value > 0 else "igt_s"
    comparison = INT_OPS_EXTENSION.get_op(comparison_name).instantiate(
        [tys.BoundedNatArg(5)],
        concrete_signature=tys.FunctionType([INT_T, INT_T], [tys.Bool]),
    )
    [condition] = loop.add_op(comparison, index_wire, current_stop)
    branch = loop.add_if(condition, *loop_inputs)
    branch_index, branch_stop, *branch_captured = branch.inputs()
    branch_environment = dict(environment)
    branch_environment.update(zip(captured_ids, branch_captured, strict=True))
    if operation.loop_var_value is not None:
        branch_environment[operation.loop_var_value.uuid] = branch_index
    branch_live = dict(live_qubits)
    for nested in operation.operations:
        _lower_operation(
            nested,
            branch,
            branch_environment,
            branch_live,
            functions,
        )

    latest = _latest_quantum_wires(
        operation.operations,
        captured_ids,
        branch_environment,
    )
    [step_wire] = branch.load(IntVal(step_value))
    addition = INT_OPS_EXTENSION.get_op("iadd").instantiate(
        [tys.BoundedNatArg(5)],
        concrete_signature=tys.FunctionType([INT_T, INT_T], [INT_T]),
    )
    [next_index] = branch.add_op(addition, branch_index, step_wire)
    control_type = tys.Sum([[], []])
    [continue_wire] = branch.add_op(ops.Tag(0, control_type))
    branch.set_outputs(
        continue_wire,
        next_index,
        branch_stop,
        *(latest.get(uuid, branch_environment[uuid]) for uuid in captured_ids),
    )

    false_branch = branch.add_else()
    [break_wire] = false_branch.add_op(ops.Tag(1, control_type))
    false_branch.set_outputs(break_wire, *false_branch.inputs())
    conditional_outputs = [
        branch.conditional_node.out(index) for index in range(3 + len(captured_ids))
    ]
    loop.set_loop_outputs(*conditional_outputs)

    loop_outputs = [
        loop.parent_node.out(index) for index in range(2 + len(captured_ids))
    ]
    origins = _quantum_origins(operation.operations, captured_ids)
    for uuid, wire in zip(captured_ids, loop_outputs[2:], strict=True):
        previous_wire = environment[uuid]
        for alias, candidate in list(environment.items()):
            if candidate == previous_wire:
                environment[alias] = wire
        environment[uuid] = wire
        result_ids = [
            result_uuid
            for result_uuid, origin_uuid in origins.items()
            if origin_uuid == uuid
        ]
        for result_uuid in result_ids:
            environment[result_uuid] = wire
        if uuid in live_qubits:
            live_qubits.pop(uuid)
            live_qubits[result_ids[-1] if result_ids else uuid] = wire


def _integer_value_wire(
    value: Value,
    builder: Any,
    environment: dict[str, Any],
) -> Any:
    """Resolve or load an integer loop-bound wire.

    Args:
        value (Value): Constant or runtime UInt value.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.

    Returns:
        Any: HUGR integer wire.

    Raises:
        EmitError: If a runtime value is unresolved.
    """
    if value.is_constant():
        from hugr.std.int import IntVal

        [wire] = builder.load(IntVal(int(value.get_const())))
        return wire
    try:
        return environment[value.uuid]
    except KeyError as error:
        raise EmitError(f"Unresolved HUGR loop bound {value.name!r}") from error


def _for_capture_ids(
    operation: ForOperation,
    environment: dict[str, Any],
) -> list[str]:
    """Collect values captured by a runtime for-loop body.

    Args:
        operation (ForOperation): Runtime loop to inspect.
        environment (dict[str, Any]): Available parent-region wires.

    Returns:
        list[str]: Stable UUID order for captured values.
    """
    captured: set[str] = set()
    for nested in operation.operations:
        captured.update(_captured_operand_ids(nested, environment))
    return [uuid for uuid in environment if uuid in captured]


def _lower_if(
    operation: IfOperation,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
    functions: dict[CallableRef, Any],
) -> None:
    """Lower an SSA-merging conditional to a native HUGR Conditional node.

    Args:
        operation (IfOperation): Prepared conditional operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.
        live_qubits (dict[str, Any]): Live quantum mapping.
        functions (dict[CallableRef, Any]): Predeclared HUGR functions.

    Raises:
        EmitError: If a branch yield is unresolved.
    """
    condition = environment[operation.condition.uuid]
    captured_ids = _branch_capture_ids(operation, environment)
    captured = _flatten_environment_values(captured_ids, environment)
    true_builder = builder.add_if(condition, *captured)
    true_environment = dict(environment)
    _bind_region_inputs(
        captured_ids,
        list(true_builder.inputs()),
        true_environment,
    )
    true_live = dict(live_qubits)
    for nested in operation.true_operations:
        _lower_operation(
            nested,
            true_builder,
            true_environment,
            true_live,
            functions,
        )
    true_outputs = _flatten_region_yields(
        operation.true_yields, true_environment, "true"
    )
    true_builder.set_outputs(*true_outputs)

    false_builder = true_builder.add_else()
    false_environment = dict(environment)
    _bind_region_inputs(
        captured_ids,
        list(false_builder.inputs()),
        false_environment,
    )
    false_live = dict(live_qubits)
    for nested in operation.false_operations:
        _lower_operation(
            nested,
            false_builder,
            false_environment,
            false_live,
            functions,
        )
    false_outputs = _flatten_region_yields(
        operation.false_yields, false_environment, "false"
    )
    false_builder.set_outputs(*false_outputs)

    true_origins = _quantum_origins(operation.true_operations, captured_ids)
    false_origins = _quantum_origins(operation.false_operations, captured_ids)

    output_count = sum(
        _array_size(result) if isinstance(result, ArrayValue) else 1
        for result in operation.results
    )
    outputs = [
        true_builder.conditional_node.out(index) for index in range(output_count)
    ]
    for uuid in captured_ids:
        live_qubits.pop(uuid, None)
    output_index = 0
    for merge in operation.iter_merges():
        width = _array_size(merge.result) if isinstance(merge.result, ArrayValue) else 1
        result_wires = outputs[output_index : output_index + width]
        output_index += width
        if isinstance(merge.result, ArrayValue):
            environment[merge.result.uuid] = result_wires
            for value in (merge.true_value, merge.false_value):
                if isinstance(value, ArrayValue):
                    environment[value.uuid] = result_wires
                    for index in range(width):
                        live_qubits.pop(f"{value.uuid}:{index}", None)
            for index, wire in enumerate(result_wires):
                live_qubits[f"{merge.result.uuid}:{index}"] = wire
            continue
        [wire] = result_wires
        if merge.true_value.type.is_quantum():
            live_qubits.pop(merge.true_value.uuid, None)
        if merge.false_value.type.is_quantum():
            live_qubits.pop(merge.false_value.uuid, None)
        environment[merge.result.uuid] = wire
        if merge.result.type.is_quantum():
            live_qubits[merge.result.uuid] = wire
            true_origin = true_origins.get(merge.true_value.uuid)
            false_origin = false_origins.get(merge.false_value.uuid)
            if true_origin is not None and true_origin == false_origin:
                previous_wire = environment[true_origin]
                for uuid, candidate in list(environment.items()):
                    if not isinstance(candidate, list) and candidate == previous_wire:
                        environment[uuid] = wire
                environment[true_origin] = wire
                address = _array_address(merge.result, environment)
                if address is not None:
                    root_uuid, index = address
                    environment[root_uuid][index] = wire


def _quantum_origins(
    operations: list[Operation],
    input_ids: list[str],
) -> dict[str, str]:
    """Trace quantum result values back to captured region inputs.

    Args:
        operations (list[Operation]): Region operations in execution order.
        input_ids (list[str]): UUIDs of values captured from the parent region.

    Returns:
        dict[str, str]: Quantum value UUID to captured input UUID provenance.
    """
    origins = {uuid: uuid for uuid in input_ids}
    for operation in operations:
        quantum_operands = [
            value for value in operation.operands if value.type.is_quantum()
        ]
        quantum_results = [
            value for value in operation.results if value.type.is_quantum()
        ]
        for operand, result in zip(quantum_operands, quantum_results, strict=False):
            origin = origins.get(operand.uuid)
            if origin is not None:
                origins[result.uuid] = origin
    return origins


def _latest_quantum_wires(
    operations: list[Operation],
    input_ids: list[str],
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the final wire produced for each captured quantum input.

    Args:
        operations (list[Operation]): Region operations in execution order.
        input_ids (list[str]): Captured input UUIDs to track.
        environment (dict[str, Any]): Lowered region environment.

    Returns:
        dict[str, Any]: Captured input UUID to its final linear wire.
    """
    origins = _quantum_origins(operations, input_ids)
    latest = {
        input_id: environment[input_id]
        for input_id in input_ids
        if input_id in environment
    }
    for value_uuid, input_id in origins.items():
        if value_uuid in environment:
            latest[input_id] = environment[value_uuid]
    return latest


def _lower_while(
    operation: WhileOperation,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
    functions: dict[CallableRef, Any],
) -> None:
    """Lower measurement-controlled repetition to a native HUGR TailLoop.

    Args:
        operation (WhileOperation): Measurement-backed Qamomile while loop.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.
        live_qubits (dict[str, Any]): Live quantum mapping.
        functions (dict[CallableRef, Any]): Predeclared HUGR functions.

    Raises:
        EmitError: If the condition or a captured body value is unresolved.
    """
    if not operation.operands:
        raise EmitError("HUGR while loop requires a condition operand")
    condition = operation.operands[0]
    if condition.uuid not in environment:
        raise EmitError("HUGR while-loop condition is unresolved")

    captured_ids = _loop_capture_ids(operation, environment)
    rest_wires = [environment[uuid] for uuid in captured_ids]
    loop = builder.add_tail_loop([], rest_wires)
    loop_inputs = loop.inputs()
    loop_environment = dict(environment)
    loop_environment.update(zip(captured_ids, loop_inputs, strict=True))

    condition_wire = loop_environment[condition.uuid]
    branch = loop.add_if(condition_wire, *loop_inputs)
    true_environment = dict(loop_environment)
    true_environment.update(zip(captured_ids, branch.inputs(), strict=True))
    true_live = dict(live_qubits)
    for nested in operation.operations:
        _lower_operation(nested, branch, true_environment, true_live, functions)

    next_wires = dict(
        _latest_quantum_wires(operation.operations, captured_ids, true_environment)
    )
    if len(operation.operands) > 1:
        updated_condition = operation.operands[1]
        next_wires[condition.uuid] = _require_wire(
            updated_condition, true_environment, "while"
        )
    from hugr import ops, tys

    control_type = tys.Sum([[], []])
    [continue_wire] = branch.add_op(ops.Tag(0, control_type))
    branch.set_outputs(
        continue_wire,
        *(next_wires.get(uuid, true_environment[uuid]) for uuid in captured_ids),
    )

    false_branch = branch.add_else()
    [break_wire] = false_branch.add_op(ops.Tag(1, control_type))
    false_branch.set_outputs(break_wire, *false_branch.inputs())

    conditional_outputs = [
        branch.conditional_node.out(index) for index in range(1 + len(captured_ids))
    ]
    loop.set_loop_outputs(*conditional_outputs)
    loop_outputs = [loop.parent_node.out(index) for index in range(len(captured_ids))]
    for uuid, wire in zip(captured_ids, loop_outputs, strict=True):
        environment[uuid] = wire
        if uuid in live_qubits:
            live_qubits[uuid] = wire
    if len(operation.operands) > 1:
        environment[operation.operands[1].uuid] = environment[condition.uuid]


def _loop_capture_ids(
    operation: WhileOperation,
    environment: dict[str, Any],
) -> list[str]:
    """Collect values that cross a while-loop region boundary.

    Args:
        operation (WhileOperation): While loop to inspect.
        environment (dict[str, Any]): Available parent-region wires.

    Returns:
        list[str]: Stable UUID order for TailLoop carried values.
    """
    captured = {operation.operands[0].uuid}
    for nested in operation.operations:
        captured.update(_captured_operand_ids(nested, environment))
    return [uuid for uuid in environment if uuid in captured]


def _branch_capture_ids(
    operation: IfOperation,
    environment: dict[str, Any],
) -> list[str]:
    """Collect outer values captured by either conditional branch.

    Args:
        operation (IfOperation): Conditional to inspect.
        environment (dict[str, Any]): Available outer wires.

    Returns:
        list[str]: Stable UUID order for HUGR conditional inputs.
    """
    captured: set[str] = set()
    for nested in [*operation.true_operations, *operation.false_operations]:
        captured.update(_captured_operand_ids(nested, environment))
    captured.update(
        value.uuid
        for value in [*operation.true_yields, *operation.false_yields]
        if value.uuid in environment
    )
    captured.discard(operation.condition.uuid)
    return [uuid for uuid in environment if uuid in captured]


def _captured_operand_ids(
    operation: Operation,
    environment: dict[str, Any],
) -> set[str]:
    """Collect parent-environment operands used by an operation tree.

    Args:
        operation (Operation): Root operation to inspect recursively.
        environment (dict[str, Any]): Available parent-region values.

    Returns:
        set[str]: UUIDs captured from the parent environment.
    """
    captured: set[str] = set()
    for value in operation.operands:
        if value.uuid in environment:
            captured.add(value.uuid)
            continue
        address = _array_address(value, environment)
        if address is None:
            continue
        root_uuid, _ = address
        if root_uuid in environment:
            captured.add(root_uuid)
    nested_lists = getattr(operation, "nested_op_lists", None)
    if callable(nested_lists):
        for nested_operations in nested_lists():
            for nested in nested_operations:
                captured.update(_captured_operand_ids(nested, environment))
    return captured


def _flatten_environment_values(
    value_ids: list[str],
    environment: dict[str, Any],
) -> list[Any]:
    """Flatten scalar and fixed-array environment values into region wires.

    Args:
        value_ids (list[str]): Environment UUIDs in stable capture order.
        environment (dict[str, Any]): UUID-to-wire or UUID-to-wire-list mapping.

    Returns:
        list[Any]: Scalar HUGR wires suitable for a region boundary.
    """
    wires: list[Any] = []
    for uuid in value_ids:
        value = environment[uuid]
        wires.extend(value if isinstance(value, list) else [value])
    return wires


def _bind_region_inputs(
    value_ids: list[str],
    inputs: list[Any],
    environment: dict[str, Any],
) -> None:
    """Bind flattened region inputs back to scalar and fixed-array values.

    Args:
        value_ids (list[str]): Captured UUIDs in stable order.
        inputs (list[Any]): Flattened HUGR region input wires.
        environment (dict[str, Any]): Parent-shaped environment to update.

    Raises:
        EmitError: If the flattened input arity does not match the environment.
    """
    offset = 0
    for uuid in value_ids:
        previous = environment[uuid]
        width = len(previous) if isinstance(previous, list) else 1
        replacement = inputs[offset : offset + width]
        if len(replacement) != width:
            raise EmitError("HUGR region input arity mismatch")
        environment[uuid] = (
            replacement if isinstance(previous, list) else replacement[0]
        )
        offset += width
    if offset != len(inputs):
        raise EmitError("HUGR region input arity mismatch")


def _flatten_region_yields(
    values: list[Value],
    environment: dict[str, Any],
    branch: str,
) -> list[Any]:
    """Flatten scalar and fixed-array region yields into HUGR wires.

    Args:
        values (list[Value]): Semantic values yielded by a region.
        environment (dict[str, Any]): Region-local value environment.
        branch (str): Region name used in diagnostics.

    Returns:
        list[Any]: Flattened HUGR output wires.

    Raises:
        EmitError: If a yielded value is unresolved.
    """
    wires: list[Any] = []
    for value in values:
        resolved = _require_wire(value, environment, branch)
        wires.extend(resolved if isinstance(resolved, list) else [resolved])
    return wires


def _require_wire(
    value: Value,
    environment: dict[str, Any],
    branch: str,
) -> Any:
    """Return a resolved conditional yield wire.

    Args:
        value (Value): Branch yield value.
        environment (dict[str, Any]): Branch-local environment.
        branch (str): Branch name for diagnostics.

    Returns:
        Any: Resolved HUGR wire.

    Raises:
        EmitError: If the yield is unresolved.
    """
    try:
        return environment[value.uuid]
    except KeyError as error:
        raise EmitError(
            f"HUGR {branch} branch yield {value.name!r} is unresolved"
        ) from error


def _lower_cinit(
    operation: CInitOperation,
    builder: Any,
    environment: dict[str, Any],
) -> None:
    """Lower scalar constants while preserving pre-mapped parameters.

    Args:
        operation (CInitOperation): Classical initialization operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping to update.

    Raises:
        EmitError: If an unbound non-parameter scalar reaches lowering.
    """
    for result in operation.results:
        if result.uuid in environment:
            continue
        if isinstance(result, ArrayValue):
            raise EmitError("HUGR array initialization is not implemented yet")
        if not result.is_constant():
            raise EmitError(f"Unresolved HUGR classical value {result.name!r}")
        [wire] = builder.load(_constant_value(result))
        environment[result.uuid] = wire


def _constant_value(value: Value) -> Any:
    """Convert a constant Qamomile scalar to a HUGR value object.

    Args:
        value (Value): Constant scalar value.

    Returns:
        Any: HUGR value object.

    Raises:
        EmitError: If ``value`` is non-constant or has an unsupported type.
    """
    _, _, val, _, _ = _require_hugr()
    if not value.is_constant():
        raise EmitError(f"Value {value.name!r} is not constant")
    concrete = value.get_const()
    if isinstance(value.type, BitType):
        return val.TRUE if bool(concrete) else val.FALSE
    if isinstance(value.type, FloatType):
        from hugr.std.float import FloatVal

        assert concrete is not None
        return FloatVal(float(concrete))
    if isinstance(value.type, UIntType):
        from hugr.std.int import IntVal

        return IntVal(int(concrete))
    raise EmitError(f"Unsupported HUGR constant type: {value.type.label()}")


def _lower_qinit(
    operation: QInitOperation,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
) -> None:
    """Lower scalar qubit allocation.

    Args:
        operation (QInitOperation): Qubit initialization operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping to update.
        live_qubits (dict[str, Any]): Live quantum mapping to update.

    Raises:
        EmitError: If vector allocation reaches the scalar lowering path.
    """
    from tket_exts import quantum

    for result in operation.results:
        if isinstance(result, ArrayValue):
            wires = []
            for index in range(_array_size(result)):
                [wire] = builder.add_op(quantum.qAlloc)
                wires.append(wire)
                live_qubits[f"{result.uuid}:{index}"] = wire
            environment[result.uuid] = wires
        else:
            [wire] = builder.add_op(quantum.qAlloc)
            environment[result.uuid] = wire
            live_qubits[result.uuid] = wire


def _resolve_wire(value: Value, environment: dict[str, Any]) -> Any:
    """Resolve a scalar or statically indexed array-element wire.

    Args:
        value (Value): Qamomile value to resolve.
        environment (dict[str, Any]): UUID-to-wire mapping.

    Returns:
        Any: HUGR wire.

    Raises:
        EmitError: If the value or its static array address is unresolved.
    """
    if value.uuid in environment:
        return environment[value.uuid]
    address = _array_address(value, environment)
    if address is not None:
        root_uuid, index = address
        try:
            return environment[root_uuid][index]
        except (KeyError, IndexError) as error:
            raise EmitError(f"Unresolved HUGR array element {value.name!r}") from error
    raise EmitError(f"Unresolved HUGR value {value.name!r}")


def _array_address(
    value: Value,
    environment: dict[str, Any],
) -> tuple[str, int] | None:
    """Resolve a constant or statically unrolled array-element address.

    Args:
        value (Value): Potential array-element value.
        environment (dict[str, Any]): Environment containing loop indices.

    Returns:
        tuple[str, int] | None: Root array UUID and element index.
    """
    address = resolve_root_qubit_address(value)
    if address is not None:
        return address
    if value.parent_array is None or len(value.element_indices) != 1:
        return None
    index_value = value.element_indices[0]
    key = f"__index__:{index_value.uuid}"
    if key not in environment:
        return None
    resolved = resolve_root_array_index(value.parent_array, int(environment[key]))
    if resolved is None:
        return None
    root, index = resolved
    return root.uuid, index


def _quantum_key(value: Value, environment: dict[str, Any]) -> str:
    """Return the live-resource key for a scalar or array element.

    Args:
        value (Value): Quantum value.
        environment (dict[str, Any]): Environment containing loop indices.

    Returns:
        str: Stable live-resource key.
    """
    address = _array_address(value, environment)
    if address is None:
        return value.uuid
    return f"{address[0]}:{address[1]}"


def _replace_quantum_results(
    operands: list[Value],
    results: list[Value],
    wires: list[Any],
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
) -> None:
    """Advance Qamomile quantum SSA values to produced HUGR wires.

    Args:
        operands (list[Value]): Consumed quantum values.
        results (list[Value]): Produced quantum values.
        wires (list[Any]): HUGR output wires corresponding to ``results``.
        environment (dict[str, Any]): UUID-to-wire mapping to update.
        live_qubits (dict[str, Any]): Live quantum mapping to update.

    Raises:
        EmitError: If result and wire arities differ.
    """
    if len(results) != len(wires):
        raise EmitError("HUGR quantum result arity mismatch")
    previous_wires = [_resolve_wire(operand, environment) for operand in operands]
    for operand in operands:
        live_qubits.pop(_quantum_key(operand, environment), None)
    for operand, result, wire in zip(operands, results, wires, strict=True):
        environment[result.uuid] = wire
        key = _quantum_key(result, environment)
        live_qubits[key] = wire
        address = _array_address(result, environment)
        if address is not None:
            root_uuid, index = address
            environment[root_uuid][index] = wire
    # Static loop bodies are traced once and therefore reuse their operand
    # UUIDs on every unrolled iteration. Keep those identities pointed at the
    # current linear wire after consumption.
    for operand, wire, previous in zip(operands, wires, previous_wires, strict=True):
        if operand.parent_array is None:
            for uuid, candidate in list(environment.items()):
                if not isinstance(candidate, list) and candidate == previous:
                    environment[uuid] = wire
            environment[operand.uuid] = wire


def _lower_gate(
    operation: GateOperation,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
    inverse: bool = False,
) -> None:
    """Lower a primitive gate through TKET's HUGR quantum extension.

    Args:
        operation (GateOperation): Qamomile gate operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.
        live_qubits (dict[str, Any]): Live quantum mapping.
        inverse (bool): Whether to emit the adjoint primitive. Defaults to
            false.

    Raises:
        EmitError: If vector operands or an unknown gate reach lowering.
    """
    from tket_exts import quantum

    operands = operation.qubit_operands
    gate_type = operation.gate_type
    if gate_type is None:
        raise EmitError("HUGR gate operation has no gate type")
    if inverse:
        gate_type = {
            GateOperationType.S: GateOperationType.SDG,
            GateOperationType.SDG: GateOperationType.S,
            GateOperationType.T: GateOperationType.TDG,
            GateOperationType.TDG: GateOperationType.T,
        }.get(gate_type, gate_type)
    if any(isinstance(value, ArrayValue) for value in operands):
        if (
            len(operands) != 1
            or len(operation.results) != 1
            or not isinstance(operands[0], ArrayValue)
            or not isinstance(operation.results[0], ArrayValue)
            or gate_type
            not in {
                GateOperationType.H,
                GateOperationType.X,
                GateOperationType.Y,
                GateOperationType.Z,
                GateOperationType.S,
                GateOperationType.SDG,
                GateOperationType.T,
                GateOperationType.TDG,
            }
        ):
            raise EmitError("HUGR supports only unary broadcast array gates")
        source = environment[operands[0].uuid]
        fixed_array = {
            GateOperationType.H: quantum.H,
            GateOperationType.X: quantum.X,
            GateOperationType.Y: quantum.Y,
            GateOperationType.Z: quantum.Z,
            GateOperationType.S: quantum.S,
            GateOperationType.SDG: quantum.Sdg,
            GateOperationType.T: quantum.T,
            GateOperationType.TDG: quantum.Tdg,
        }
        outputs = [builder.add_op(fixed_array[gate_type], wire)[0] for wire in source]
        environment[operation.results[0].uuid] = outputs
        environment[operands[0].uuid] = outputs
        for index, wire in enumerate(outputs):
            live_qubits.pop(f"{operands[0].uuid}:{index}", None)
            live_qubits[f"{operation.results[0].uuid}:{index}"] = wire
        return
    qubits = [_resolve_wire(value, environment) for value in operands]
    fixed = {
        GateOperationType.H: quantum.H,
        GateOperationType.X: quantum.X,
        GateOperationType.Y: quantum.Y,
        GateOperationType.Z: quantum.Z,
        GateOperationType.S: quantum.S,
        GateOperationType.SDG: quantum.Sdg,
        GateOperationType.T: quantum.T,
        GateOperationType.TDG: quantum.Tdg,
        GateOperationType.CX: quantum.CX,
        GateOperationType.CZ: quantum.CZ,
        GateOperationType.TOFFOLI: quantum.toffoli,
    }
    if gate_type in fixed:
        outputs = list(builder.add_op(fixed[gate_type], *qubits))
    elif gate_type in {
        GateOperationType.RX,
        GateOperationType.RY,
        GateOperationType.RZ,
        GateOperationType.P,
    }:
        assert operation.theta is not None
        rotation = _rotation_wire(
            builder,
            operation.theta,
            environment,
            scale=-1.0 if inverse else 1.0,
        )
        rotation_op = {
            GateOperationType.RX: quantum.Rx,
            GateOperationType.RY: quantum.Ry,
            GateOperationType.RZ: quantum.Rz,
            GateOperationType.P: quantum.Rz,
        }[gate_type]
        outputs = list(builder.add_op(rotation_op, qubits[0], rotation))
    elif gate_type is GateOperationType.SWAP:
        left, right = qubits
        left, right = builder.add_op(quantum.CX, left, right)
        right, left = builder.add_op(quantum.CX, right, left)
        left, right = builder.add_op(quantum.CX, left, right)
        outputs = [left, right]
    elif gate_type is GateOperationType.RZZ:
        assert operation.theta is not None
        left, right = builder.add_op(quantum.CX, *qubits)
        rotation = _rotation_wire(
            builder,
            operation.theta,
            environment,
            scale=-1.0 if inverse else 1.0,
        )
        [right] = builder.add_op(quantum.Rz, right, rotation)
        left, right = builder.add_op(quantum.CX, left, right)
        outputs = [left, right]
    elif gate_type is GateOperationType.CP:
        assert operation.theta is not None
        control, target = qubits
        direction = -1.0 if inverse else 1.0
        half = _rotation_wire(
            builder,
            operation.theta,
            environment,
            scale=0.5 * direction,
        )
        [control] = builder.add_op(quantum.Rz, control, half)
        full = _rotation_wire(
            builder,
            operation.theta,
            environment,
            scale=direction,
        )
        control, target = builder.add_op(quantum.CRz, control, target, full)
        outputs = [control, target]
    else:
        raise EmitError(f"Unsupported HUGR gate: {gate_type}")
    _replace_quantum_results(
        operands,
        operation.results,
        outputs,
        environment,
        live_qubits,
    )


def _rotation_wire(
    builder: Any,
    theta: Value,
    environment: dict[str, Any],
    scale: float = 1.0,
) -> Any:
    """Convert a radian-valued Qamomile angle to TKET's rotation type.

    Args:
        builder (Any): HUGR dataflow builder.
        theta (Value): Constant or runtime radian-valued angle.
        environment (dict[str, Any]): UUID-to-wire mapping.
        scale (float): Additional numeric scale before conversion. Defaults
            to one.

    Returns:
        Any: HUGR wire of ``tket.rotation.rotation`` type.

    Raises:
        EmitError: If a runtime theta wire is unresolved.
    """
    from tket_exts import rotation

    from hugr.std.float import FLOAT_OPS_EXTENSION, FloatVal

    if theta.is_constant():
        concrete = theta.get_const()
        assert concrete is not None
        [halfturns] = builder.load(FloatVal(float(concrete) * scale / math.pi))
    else:
        if theta.uuid not in environment:
            raise EmitError(f"Unresolved HUGR rotation value {theta.name!r}")
        [factor] = builder.load(FloatVal(scale / math.pi))
        multiply = FLOAT_OPS_EXTENSION.get_op("fmul").instantiate()
        [halfturns] = builder.add_op(multiply, environment[theta.uuid], factor)
    [result] = builder.add_op(rotation.from_halfturns_unchecked, halfturns)
    return result


def _lower_measure(
    operation: MeasureOperation | MeasureVectorOperation,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
) -> None:
    """Lower destructive Qamomile measurement and free the HUGR qubit.

    Args:
        operation (MeasureOperation | MeasureVectorOperation): Measurement
            operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping to update.
        live_qubits (dict[str, Any]): Live quantum mapping to update.

    Raises:
        EmitError: If vector measurement reaches scalar lowering.
    """
    from tket_exts import quantum

    operand = operation.operands[0]
    if isinstance(operand, ArrayValue):
        if not isinstance(operation.results[0], ArrayValue):
            raise EmitError("HUGR vector measurement requires an array result")
        bits = []
        for index, wire in enumerate(environment[operand.uuid]):
            qubit, bit = builder.add_op(quantum.measure, wire)
            builder.add_op(quantum.qFree, qubit)
            live_qubits.pop(f"{operand.uuid}:{index}", None)
            bits.append(bit)
        environment[operation.results[0].uuid] = bits
    else:
        qubit, bit = builder.add_op(
            quantum.measure, _resolve_wire(operand, environment)
        )
        builder.add_op(quantum.qFree, qubit)
        live_qubits.pop(_quantum_key(operand, environment), None)
        environment[operation.results[0].uuid] = bit


def _lower_project(
    operation: ProjectOperation,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
) -> None:
    """Lower Pauli-basis projection while retaining the projected qubit.

    Args:
        operation (ProjectOperation): Projection operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.
        live_qubits (dict[str, Any]): Live quantum mapping.
    """
    from tket_exts import quantum

    operand = operation.operands[0]
    qubit = _resolve_wire(operand, environment)
    if operation.axis == "x":
        [qubit] = builder.add_op(quantum.H, qubit)
    elif operation.axis == "y":
        [qubit] = builder.add_op(quantum.Sdg, qubit)
        [qubit] = builder.add_op(quantum.H, qubit)
    qubit, bit = builder.add_op(quantum.measure, qubit)
    if operation.axis == "x":
        [qubit] = builder.add_op(quantum.H, qubit)
    elif operation.axis == "y":
        [qubit] = builder.add_op(quantum.H, qubit)
        [qubit] = builder.add_op(quantum.S, qubit)
    _replace_quantum_results(
        [operand],
        [operation.results[0]],
        [qubit],
        environment,
        live_qubits,
    )
    environment[operation.results[1].uuid] = bit


def _lower_reset(
    operation: ResetOperation,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
) -> None:
    """Lower qubit reset through TKET's quantum extension.

    Args:
        operation (ResetOperation): Qubit reset operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.
        live_qubits (dict[str, Any]): Live quantum mapping.
    """
    from tket_exts import quantum

    operand = operation.operands[0]
    [wire] = builder.add_op(quantum.reset, _resolve_wire(operand, environment))
    _replace_quantum_results(
        [operand], operation.results, [wire], environment, live_qubits
    )


def _lower_call(
    operation: InvokeOperation,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
    functions: dict[CallableRef, Any],
) -> None:
    """Preserve a direct callable invocation as a HUGR function call.

    Args:
        operation (InvokeOperation): Semantic callable invocation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.
        live_qubits (dict[str, Any]): Live quantum mapping.
        functions (dict[CallableRef, Any]): Predeclared HUGR functions.

    Raises:
        EmitError: If the call is transformed, opaque, or unresolved.
    """
    if operation.transform is not CallTransform.DIRECT:
        _lower_transformed_call(
            operation,
            builder,
            environment,
            live_qubits,
            functions,
        )
        return
    if operation.target not in functions:
        raise EmitError(f"Opaque HUGR callable {operation.target.name!r}")
    operands = [environment[value.uuid] for value in operation.operands]
    outputs = list(builder.call(functions[operation.target].parent_node, *operands))
    if len(outputs) != len(operation.results):
        raise EmitError("HUGR callable result arity mismatch")
    for operand in operation.operands:
        if operand.type.is_quantum():
            live_qubits.pop(operand.uuid, None)
    for result, wire in zip(operation.results, outputs, strict=True):
        environment[result.uuid] = wire
        if result.type.is_quantum():
            live_qubits[result.uuid] = wire


def _lower_transformed_call(
    operation: InvokeOperation | ControlledUOperation | InverseBlockOperation,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
    functions: dict[CallableRef, Any],
) -> None:
    """Inline a transformed callable at the target legalization boundary.

    Args:
        operation (InvokeOperation | ControlledUOperation |
            InverseBlockOperation): Transformed invocation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.
        live_qubits (dict[str, Any]): Live quantum mapping.
        functions (dict[CallableRef, Any]): Predeclared HUGR functions.

    Raises:
        EmitError: If the callable body or transform profile is unsupported.
    """
    if isinstance(operation, InverseBlockOperation):
        body = operation.source_block
        controls = operation.control_qubits
        targets = operation.target_qubits
        classical = operation.parameters
        inverse = True
        display_name = operation.name
    elif isinstance(operation, ControlledUOperation):
        body = operation.block
        controls = operation.control_operands
        targets = [
            value for value in operation.target_operands if value.type.is_quantum()
        ]
        classical = operation.param_operands
        inverse = False
        display_name = "controlled"
    else:
        body = operation.body
        controls = operation.control_qubits
        targets = (
            operation.target_qubits
            if controls
            else [value for value in operation.operands if value.type.is_quantum()]
        )
        classical = [value for value in operation.operands if value.type.is_classical()]
        inverse = operation.transform is CallTransform.INVERSE
        display_name = operation.target.name
    if body is None:
        raise EmitError(f"Transformed HUGR callable {display_name!r} is opaque")
    local = dict(environment)
    body_quantum_inputs = [
        value for value in body.input_values if value.type.is_quantum()
    ]
    body_classical_inputs = [
        value for value in body.input_values if value.type.is_classical()
    ]
    body_quantum_outputs = [
        value for value in body.output_values if value.type.is_quantum()
    ]
    quantum_entry = body_quantum_outputs if inverse else body_quantum_inputs
    if len(quantum_entry) != len(targets):
        raise EmitError("HUGR transformed call quantum arity mismatch")
    for formal, actual in zip(quantum_entry, targets, strict=True):
        local[formal.uuid] = _resolve_wire(actual, environment)
    if len(body_classical_inputs) != len(classical):
        raise EmitError("HUGR transformed call classical arity mismatch")
    for formal, actual in zip(body_classical_inputs, classical, strict=True):
        local[formal.uuid] = environment[actual.uuid]

    control_wires = [_resolve_wire(value, environment) for value in controls]
    sequence = reversed(body.operations) if inverse else body.operations
    for nested in sequence:
        if isinstance(nested, ReturnOperation):
            continue
        if isinstance(nested, CInitOperation):
            _lower_cinit(nested, builder, local)
            continue
        if not isinstance(nested, GateOperation):
            raise EmitError(
                "HUGR transformed calls currently require primitive unitary bodies"
            )
        if controls:
            control_wires = _lower_controlled_gate(
                nested,
                builder,
                local,
                control_wires,
                inverse,
            )
        else:
            lowered_gate = nested
            if inverse:
                lowered_gate = dataclasses.replace(
                    nested,
                    operands=[
                        *nested.results,
                        *([nested.theta] if nested.theta is not None else []),
                    ],
                    results=list(nested.qubit_operands),
                )
            _lower_gate(lowered_gate, builder, local, {}, inverse=inverse)

    quantum_exit = body_quantum_inputs if inverse else body_quantum_outputs
    result_quantum = [value for value in operation.results if value.type.is_quantum()]
    result_controls = result_quantum[: len(controls)]
    result_targets = result_quantum[len(controls) :]
    for actual in [*controls, *targets]:
        live_qubits.pop(_quantum_key(actual, environment), None)
    for source, result, wire in zip(
        controls, result_controls, control_wires, strict=True
    ):
        _publish_transformed_result(source, result, wire, environment, live_qubits)
    for source, result, formal in zip(
        targets, result_targets, quantum_exit, strict=True
    ):
        wire = local[formal.uuid]
        _publish_transformed_result(source, result, wire, environment, live_qubits)


def _publish_transformed_result(
    source: Value,
    result: Value,
    wire: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
) -> None:
    """Publish a transformed-call result and its containing array version.

    Args:
        source (Value): Quantum value consumed by the transformed call.
        result (Value): Quantum SSA value produced by the transformed call.
        wire (Any): Resulting linear HUGR wire.
        environment (dict[str, Any]): UUID-to-wire mapping to update.
        live_qubits (dict[str, Any]): Live quantum resource mapping to update.

    Raises:
        EmitError: If a result array version cannot be reconstructed.
    """
    environment[result.uuid] = wire
    address = _array_address(result, environment)
    if address is not None:
        root_uuid, index = address
        if root_uuid not in environment:
            source_address = _array_address(source, environment)
            if source_address is None or source_address[0] not in environment:
                raise EmitError(f"Cannot reconstruct HUGR array result {result.name!r}")
            environment[root_uuid] = list(environment[source_address[0]])
        environment[root_uuid][index] = wire
    live_qubits[_quantum_key(result, environment)] = wire


def _lower_controlled_gate(
    operation: GateOperation,
    builder: Any,
    environment: dict[str, Any],
    controls: list[Any],
    inverse: bool,
) -> list[Any]:
    """Legalize one primitive gate under one or two HUGR controls.

    Args:
        operation (GateOperation): Primitive body gate.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): Body-local wire mapping.
        controls (list[Any]): Current control wires.
        inverse (bool): Whether to emit the adjoint gate.

    Returns:
        list[Any]: Updated linear control wires.

    Raises:
        EmitError: If the controlled primitive is outside the target profile.
    """
    from tket_exts import quantum

    if len(operation.qubit_operands) != 1:
        raise EmitError("Nested controlled multi-qubit HUGR gates are unsupported")
    target_value = operation.qubit_operands[0]
    target = _resolve_wire(target_value, environment)
    if operation.gate_type is GateOperationType.X:
        if len(controls) == 1:
            controls[0], target = builder.add_op(quantum.CX, controls[0], target)
        elif len(controls) == 2:
            controls[0], controls[1], target = builder.add_op(
                quantum.toffoli, controls[0], controls[1], target
            )
        else:
            raise EmitError("HUGR controlled X supports one or two controls")
    elif operation.gate_type is GateOperationType.Y and len(controls) == 1:
        controls[0], target = builder.add_op(quantum.CY, controls[0], target)
    elif operation.gate_type is GateOperationType.Z and len(controls) == 1:
        controls[0], target = builder.add_op(quantum.CZ, controls[0], target)
    elif operation.gate_type is GateOperationType.RZ and len(controls) == 1:
        assert operation.theta is not None
        rotation = _rotation_wire(
            builder,
            operation.theta,
            environment,
            scale=-1.0 if inverse else 1.0,
        )
        controls[0], target = builder.add_op(quantum.CRz, controls[0], target, rotation)
    else:
        raise EmitError(f"Unsupported controlled HUGR gate: {operation.gate_type}")
    environment[operation.results[0].uuid] = target
    environment[target_value.uuid] = target
    return controls


def _lower_binop(
    operation: BinOp,
    builder: Any,
    environment: dict[str, Any],
) -> None:
    """Lower supported floating-point arithmetic to HUGR extensions.

    Args:
        operation (BinOp): Qamomile binary arithmetic operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.

    Raises:
        EmitError: If operand types or operation kind are unsupported.
    """
    from hugr.std.float import FLOAT_OPS_EXTENSION

    if not all(isinstance(value.type, FloatType) for value in operation.operands):
        raise EmitError("Only Float HUGR arithmetic is implemented initially")
    names = {
        BinOpKind.ADD: "fadd",
        BinOpKind.SUB: "fsub",
        BinOpKind.MUL: "fmul",
        BinOpKind.DIV: "fdiv",
        BinOpKind.POW: "fpow",
        BinOpKind.MIN: "fmin",
    }
    if operation.kind is None:
        raise EmitError("HUGR floating operation has no operation kind")
    name = names.get(operation.kind)
    if name is None:
        raise EmitError(f"Unsupported HUGR floating operation: {operation.kind}")
    op = FLOAT_OPS_EXTENSION.get_op(name).instantiate()
    [wire] = builder.add_op(
        op,
        environment[operation.operands[0].uuid],
        environment[operation.operands[1].uuid],
    )
    environment[operation.results[0].uuid] = wire


def _lower_compop(
    operation: CompOp,
    builder: Any,
    environment: dict[str, Any],
) -> None:
    """Lower supported floating-point comparisons to HUGR extensions.

    Args:
        operation (CompOp): Qamomile comparison operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.

    Raises:
        EmitError: If operand types or comparison kind are unsupported.
    """
    from hugr.std.float import FLOAT_OPS_EXTENSION

    if not all(isinstance(value.type, FloatType) for value in operation.operands):
        raise EmitError("Only Float HUGR comparisons are implemented initially")
    names = {
        CompOpKind.EQ: "feq",
        CompOpKind.NEQ: "fne",
        CompOpKind.LT: "flt",
        CompOpKind.LE: "fle",
        CompOpKind.GT: "fgt",
        CompOpKind.GE: "fge",
    }
    if operation.kind is None:
        raise EmitError("HUGR comparison has no comparison kind")
    name = names.get(operation.kind)
    if name is None:
        raise EmitError(f"Unsupported HUGR comparison: {operation.kind}")
    op = FLOAT_OPS_EXTENSION.get_op(name).instantiate()
    [wire] = builder.add_op(
        op,
        environment[operation.operands[0].uuid],
        environment[operation.operands[1].uuid],
    )
    environment[operation.results[0].uuid] = wire
