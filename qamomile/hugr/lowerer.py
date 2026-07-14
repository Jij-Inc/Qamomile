"""Direct lowering from prepared Qamomile semantics to Guppy-compatible HUGR."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from typing import Any, TypeAlias, cast

from qamomile._utils import is_close_zero
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
    CallPolicy,
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    LoopCarriedRebind,
    WhileOperation,
    validate_region_args,
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
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import CInitOperation, QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.types import (
    BitType,
    FloatType,
    ObservableType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.types.primitives import ValueType
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
    array_physical_region,
    resolve_root_array_index,
    resolve_root_qubit_address,
)
from qamomile.circuit.transpiler.artifact import (
    CompilationMetadata,
    CompiledProgram,
)
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands
from qamomile.circuit.transpiler.errors import (
    CallableDefinitionConflictError,
    EmitError,
)
from qamomile.circuit.transpiler.passes.analyze import (
    reject_control_flow_quantum_discard,
)
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.circuit.transpiler.passes.validate_while import (
    ValidateWhileContractPass,
)
from qamomile.circuit.transpiler.prepared import PreparedModule

_QuantumOrigin: TypeAlias = tuple[str, int | None]
_QuantumFootprint: TypeAlias = frozenset[_QuantumOrigin]


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

        Raises:
            CallableDefinitionConflictError: If one source callable produced
                multiple specialized bodies that cannot share one HUGR symbol.
        """
        _validate_direct_semantics(program)
        for ref, variants in program.definition_variants.items():
            if len(variants) > 1:
                symbol = f"{ref.namespace}.{ref.name}@{ref.version}"
                raise CallableDefinitionConflictError(symbol)
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


def _validate_direct_semantics(program: PreparedModule) -> None:
    """Validate target-neutral invariants skipped by direct HUGR lowering.

    Circuit-family planning runs these checks as part of partial evaluation,
    analysis, and segmentation. HUGR intentionally preserves the prepared
    program graph, so it invokes only the non-destructive semantic checks
    here instead of importing the circuit segmentation pipeline. INLINE
    callables are expanded only in this validation view so formal values are
    checked with their call-site provenance; emitted HUGR remains hierarchical.

    Args:
        program (PreparedModule): Prepared entrypoint and callable bodies.

    Raises:
        ValidationError: If a while condition is not measurement-backed.
        AffineTypeError: If control flow discards a quantum value.
    """
    inline = InlinePass()
    blocks: list[tuple[Block, Mapping[str, Any]]] = [
        (inline.run(program.entrypoint), program.bindings)
    ]
    blocks.extend(
        (inline.run(definition.body), {})
        for definition in program.definitions.values()
        if definition.body is not None
        and definition.default_policy is not CallPolicy.INLINE
    )
    visited: set[int] = set()
    for block, bindings in blocks:
        if id(block) in visited:
            continue
        visited.add(id(block))
        ValidateWhileContractPass().run(block)
        reject_control_flow_quantum_discard(block.operations, dict(bindings))


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

    entry_inputs = _entry_inputs(program.entrypoint, program.bindings)
    main = module.define_function(
        "main",
        [_lower_value_type(value) for value in entry_inputs],
        [_lower_value_type(value) for value in program.entrypoint.output_values],
        visibility="Public",
    )

    for ref in plan.definitions:
        _lower_block(program.body(ref), functions[ref], functions, None)
    _lower_block(
        program.entrypoint,
        main,
        functions,
        entry_inputs,
        bindings=program.bindings,
    )

    extensions = [
        tket_exts.quantum(),
        tket_exts.rotation(),
        tket_exts.bool(),
        tket_exts.global_phase(),
    ]
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


def _entry_inputs(block: Block, bindings: Mapping[str, Any]) -> list[ValueLike]:
    """Return runtime parameter values exposed by the HUGR entrypoint.

    Args:
        block (Block): Prepared top-level semantic block.
        bindings (Mapping[str, Any]): Compile-time values excluded from the
            public HUGR function ABI.

    Returns:
        list[ValueLike]: Runtime parameter values in block input order.
    """
    parameter_uuids = {
        value.uuid for name, value in block.parameters.items() if name not in bindings
    }
    return [
        value
        for value in block.input_values
        if value.uuid in parameter_uuids and not value.is_constant()
    ]


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


def _lower_value_type(value: ValueLike) -> Any:
    """Map a scalar or fixed-size array value to a HUGR carrier type.

    Args:
        value (ValueLike): Qamomile value including structural metadata.

    Returns:
        Any: Scalar type or fixed-length HUGR tuple type.
    """
    if isinstance(value, (TupleValue, DictValue)):
        raise EmitError(f"HUGR structural value {value.name!r} is not supported yet")
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
    explicit_inputs: list[ValueLike] | None,
    bindings: Mapping[str, Any] | None = None,
) -> None:
    """Lower one Qamomile block into a HUGR function dataflow graph.

    Args:
        block (Block): Semantic callable body.
        builder (Any): ``hugr.build.Function`` receiving operations.
        functions (dict[CallableRef, Any]): Predeclared callable functions.
        explicit_inputs (list[ValueLike] | None): Input values corresponding to
            HUGR function ports. ``None`` uses every block input.
        bindings (Mapping[str, Any] | None): Compile-time entrypoint values
            keyed by public parameter name. Defaults to ``None``.

    Raises:
        EmitError: If an operation or value is unsupported or unresolved.
    """
    inputs = block.input_values if explicit_inputs is None else explicit_inputs
    environment: dict[str, Any] = {}
    for name, concrete in (bindings or {}).items():
        value = block.parameters.get(name)
        if value is None:
            continue
        if isinstance(value.type, ObservableType):
            resolved = concrete
        else:
            scalar = cast(int | float | bool, concrete)
            [resolved] = builder.load(_constant_value(value.with_const(scalar)))
        environment[value.uuid] = resolved
        environment[f"__parameter__:{name}"] = resolved
    for value, wire in zip(inputs, builder.inputs(), strict=True):
        if isinstance(value, ArrayValue):
            from hugr import ops

            element_types = [_lower_type(value.type) for _ in range(_array_size(value))]
            environment[value.uuid] = list(
                builder.add_op(ops.UnpackTuple(element_types), wire)
            )
        else:
            environment[value.uuid] = wire
        parameter_name = value.parameter_name()
        if parameter_name is not None:
            environment[f"__parameter__:{parameter_name}"] = environment[value.uuid]
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


def _pack_value(
    value: ValueLike,
    environment: dict[str, Any],
    builder: Any,
) -> Any:
    """Return a scalar wire or pack array elements into a HUGR tuple.

    Args:
        value (ValueLike): Value to materialize at a function boundary.
        environment (dict[str, Any]): UUID-to-wire mapping.
        builder (Any): HUGR dataflow builder.

    Returns:
        Any: Scalar or tuple wire.

    Raises:
        EmitError: If the value is unresolved.
    """
    if isinstance(value, (TupleValue, DictValue)):
        raise EmitError(f"HUGR structural value {value.name!r} is not supported yet")
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
        case GlobalPhaseOperation():
            _lower_global_phase(operation.phase, builder, environment)
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
        case PauliEvolveOp():
            _lower_pauli_evolution(operation, builder, environment, live_qubits)
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
    try:
        validate_region_args(operation)
    except ValueError as error:
        raise EmitError(str(error), operation="ForOperation") from error
    quantum_rebinds = _unsupported_quantum_rebinds(operation)
    if quantum_rebinds:
        names = ", ".join(rebind.var_name for rebind in quantum_rebinds)
        raise EmitError(
            "HUGR cannot lower loop-carried quantum resource rebinding "
            f"without explicit linear region slots ({names}).",
            operation="ForOperation",
        )
    residual_rebinds = [
        rebind
        for rebind in operation.loop_carried_rebinds
        if not rebind.before.type.is_quantum()
    ]
    if residual_rebinds:
        names = ", ".join(rebind.var_name for rebind in residual_rebinds)
        raise EmitError(
            "HUGR cannot lower loop-carried classical values without explicit "
            f"region arguments ({names}).",
            operation="ForOperation",
        )
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
    captured_ids = _for_capture_ids(operation, environment)
    carried = {
        region.block_arg.uuid: _resolve_classical_argument(
            region.init,
            builder,
            environment,
        )
        for region in operation.region_args
    }
    for index in range(start, stop, step):
        incoming_keys = set(live_qubits)
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
            region.block_arg.uuid: _resolve_classical_argument(
                region.yielded,
                builder,
                environment,
            )
            for region in operation.region_args
        }
        _free_region_local_qubits(
            builder,
            live_qubits,
            incoming_keys,
            [
                *_flatten_environment_values(captured_ids, environment),
                *carried.values(),
            ],
        )
    for region in operation.region_args:
        wire = carried[region.block_arg.uuid]
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
    captured_wires = _flatten_environment_values(captured_ids, environment)
    captured_width = len(captured_wires)
    region_init_wires = [
        _resolve_classical_argument(region.init, builder, environment)
        for region in operation.region_args
    ]
    loop = builder.add_tail_loop(
        [],
        [start_wire, stop_wire, *captured_wires, *region_init_wires],
    )
    loop_inputs = loop.inputs()
    index_wire, current_stop = loop_inputs[:2]

    from hugr import ops, tys
    from hugr.std.int import INT_OPS_EXTENSION, INT_T, IntVal

    comparison_name = "ilt_u" if step_value > 0 else "igt_u"
    comparison = INT_OPS_EXTENSION.get_op(comparison_name).instantiate(
        [tys.BoundedNatArg(5)],
        concrete_signature=tys.FunctionType([INT_T, INT_T], [tys.Bool]),
    )
    [condition] = loop.add_op(comparison, index_wire, current_stop)
    branch = loop.add_if(condition, *loop_inputs)
    branch_index, branch_stop, *branch_state = branch.inputs()
    branch_captured = branch_state[:captured_width]
    branch_region_args = branch_state[captured_width:]
    branch_environment = dict(environment)
    _bind_region_inputs(
        captured_ids,
        list(branch_captured),
        branch_environment,
    )
    branch_environment.update(
        zip(
            (region.block_arg.uuid for region in operation.region_args),
            branch_region_args,
            strict=True,
        )
    )
    if operation.loop_var_value is not None:
        branch_environment[operation.loop_var_value.uuid] = branch_index
    branch_live = dict(live_qubits)
    branch_incoming_keys = set(branch_live)
    for nested in operation.operations:
        _lower_operation(
            nested,
            branch,
            branch_environment,
            branch_live,
            functions,
        )

    [step_wire] = branch.load(IntVal(step_value))
    addition = INT_OPS_EXTENSION.get_op("iadd").instantiate(
        [tys.BoundedNatArg(5)],
        concrete_signature=tys.FunctionType([INT_T, INT_T], [INT_T]),
    )
    [next_index] = branch.add_op(addition, branch_index, step_wire)
    control_type = tys.Sum([[], []])
    [continue_wire] = branch.add_op(ops.Tag(0, control_type))
    next_region_args = [
        _resolve_classical_argument(region.yielded, branch, branch_environment)
        for region in operation.region_args
    ]
    next_captured = _flatten_environment_values(captured_ids, branch_environment)
    _free_region_local_qubits(
        branch,
        branch_live,
        branch_incoming_keys,
        [*next_captured, *next_region_args],
    )
    branch.set_outputs(
        continue_wire,
        next_index,
        branch_stop,
        *next_captured,
        *next_region_args,
    )

    false_branch = branch.add_else()
    [break_wire] = false_branch.add_op(ops.Tag(1, control_type))
    false_branch.set_outputs(break_wire, *false_branch.inputs())
    conditional_outputs = [
        branch.conditional_node.out(index)
        for index in range(3 + captured_width + len(operation.region_args))
    ]
    loop.set_loop_outputs(*conditional_outputs)

    loop_outputs = [
        loop.parent_node.out(index)
        for index in range(2 + captured_width + len(operation.region_args))
    ]
    origins = _quantum_origins(operation.operations, captured_ids)
    capture_outputs = loop_outputs[2 : 2 + captured_width]
    _publish_captured_outputs(
        captured_ids,
        capture_outputs,
        environment,
        live_qubits,
        origins,
    )
    region_outputs = loop_outputs[2 + captured_width :]
    for region, wire in zip(operation.region_args, region_outputs, strict=True):
        environment[region.result.uuid] = wire


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
    # A loop-invariant overwrite can yield an outer value without any body
    # operation reading it. It still crosses the TailLoop boundary.
    return _region_capture_ids(
        operation.operations,
        environment,
        extra_values=[region.yielded for region in operation.region_args],
    )


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
    condition = _resolve_wire(operation.condition, environment)
    captured_ids = _branch_capture_ids(operation, environment)
    true_origins = _quantum_origins(operation.true_operations, captured_ids)
    false_origins = _quantum_origins(operation.false_operations, captured_ids)
    _validate_conditional_quantum_outputs(
        operation,
        true_origins,
        false_origins,
    )
    captured = _flatten_environment_values(captured_ids, environment)
    true_builder = builder.add_if(condition, *captured)
    true_environment = dict(environment)
    _bind_region_inputs(
        captured_ids,
        list(true_builder.inputs()),
        true_environment,
    )
    true_live = dict(live_qubits)
    true_parent_keys = set(true_live)
    true_region_keys = _rebind_captured_live_wires(
        captured_ids,
        environment,
        true_environment,
        true_live,
    )
    true_protected_keys = true_parent_keys - true_region_keys
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
    _validate_live_quantum_yields(
        operation.true_yields,
        true_environment,
        true_live,
        "true",
    )
    _free_region_local_qubits(
        true_builder,
        true_live,
        true_protected_keys,
        true_outputs,
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
    false_parent_keys = set(false_live)
    false_region_keys = _rebind_captured_live_wires(
        captured_ids,
        environment,
        false_environment,
        false_live,
    )
    false_protected_keys = false_parent_keys - false_region_keys
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
    _validate_live_quantum_yields(
        operation.false_yields,
        false_environment,
        false_live,
        "false",
    )
    _free_region_local_qubits(
        false_builder,
        false_live,
        false_protected_keys,
        false_outputs,
    )
    false_builder.set_outputs(*false_outputs)

    output_count = sum(
        _array_size(result) if isinstance(result, ArrayValue) else 1
        for result in operation.results
    )
    outputs = [
        true_builder.conditional_node.out(index) for index in range(output_count)
    ]
    for key, wire in list(live_qubits.items()):
        if any(wire == captured_wire for captured_wire in captured):
            live_qubits.pop(key)
    output_index = 0
    for merge in operation.iter_merges():
        width = _array_size(merge.result) if isinstance(merge.result, ArrayValue) else 1
        result_wires = outputs[output_index : output_index + width]
        output_index += width
        if isinstance(merge.result, ArrayValue):
            common_origin = _common_quantum_origin(
                merge.true_value,
                merge.false_value,
                true_origins,
                false_origins,
            )
            if common_origin is not None:
                previous = _quantum_origin_value(common_origin, environment)
                _replace_environment_aliases(environment, previous, result_wires)
                _set_quantum_origin_value(common_origin, result_wires, environment)
            environment[merge.result.uuid] = result_wires
            for value in (merge.true_value, merge.false_value):
                if isinstance(value, ArrayValue):
                    environment[value.uuid] = result_wires
            if merge.result.type.is_quantum():
                consumed_ids = {
                    merge.true_value.uuid,
                    merge.false_value.uuid,
                }
                if common_origin is not None:
                    consumed_ids.add(common_origin[0])
                for uuid in consumed_ids:
                    for index in range(width):
                        live_qubits.pop(f"{uuid}:{index}", None)
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
            common_origin = _common_quantum_origin(
                merge.true_value,
                merge.false_value,
                true_origins,
                false_origins,
            )
            if common_origin is not None:
                previous_wire = _quantum_origin_value(common_origin, environment)
                for uuid, candidate in list(environment.items()):
                    if not isinstance(candidate, list) and candidate == previous_wire:
                        environment[uuid] = wire
                _set_quantum_origin_value(common_origin, wire, environment)
                address = _array_address(merge.result, environment)
                if address is not None:
                    root_uuid, index = address
                    environment[root_uuid][index] = wire


def _common_quantum_origin(
    true_value: Value,
    false_value: Value,
    true_origins: dict[str, _QuantumOrigin],
    false_origins: dict[str, _QuantumOrigin],
) -> _QuantumOrigin | None:
    """Return the captured origin shared by two quantum branch yields.

    Args:
        true_value (Value): Value yielded by the true branch.
        false_value (Value): Value yielded by the false branch.
        true_origins (dict[str, _QuantumOrigin]): True-region provenance.
        false_origins (dict[str, _QuantumOrigin]): False-region provenance.

    Returns:
        _QuantumOrigin | None: Shared captured resource and optional array
            element index, or ``None`` when the branches diverge.
    """
    if not true_value.type.is_quantum() or not false_value.type.is_quantum():
        return None
    true_origin = _value_quantum_origin(true_value, true_origins)
    false_origin = _value_quantum_origin(false_value, false_origins)
    if true_origin is None or true_origin != false_origin:
        return None
    return true_origin


def _quantum_footprint(
    value: Value,
    origins: dict[str, _QuantumOrigin],
) -> _QuantumFootprint | None:
    """Resolve a quantum value to the captured physical slots it denotes.

    Args:
        value (Value): Quantum scalar, array, slice, or array element.
        origins (dict[str, _QuantumOrigin]): Value provenance for one region.

    Returns:
        _QuantumFootprint | None: Captured physical slots, or ``None`` when
            the value is body-local or its array region is dynamic.
    """
    origin = _value_quantum_origin(value, origins)
    if origin is None:
        return None
    root_uuid, index = origin
    if index is not None:
        return frozenset({origin})
    if not isinstance(value, ArrayValue):
        return frozenset({origin})
    region = array_physical_region(value)
    if region is None:
        return None
    return frozenset((root_uuid, element_index) for element_index in region[1])


def _intrinsic_quantum_footprint(value: Value) -> _QuantumFootprint:
    """Return a branch-local physical footprint without capture provenance.

    Args:
        value (Value): Quantum scalar, array, slice, or array element.

    Returns:
        _QuantumFootprint: Best-effort branch-local physical slots. Dynamic
            shapes and indices conservatively fall back to logical identity.
    """
    if isinstance(value, ArrayValue):
        region = array_physical_region(value)
        if region is not None:
            root_logical_id, indices = region
            return frozenset(
                (root_logical_id, element_index) for element_index in indices
            )
    elif value.parent_array is not None:
        address = resolve_root_qubit_address(value)
        if address is not None:
            array = value.parent_array
            while array.slice_of is not None:
                array = array.slice_of
            return frozenset({(array.logical_id, address[1])})
    return frozenset({(value.logical_id, None)})


def _validate_conditional_quantum_outputs(
    operation: IfOperation,
    true_origins: dict[str, _QuantumOrigin],
    false_origins: dict[str, _QuantumOrigin],
) -> None:
    """Reject conditional quantum output layouts this lowering cannot linearize.

    HUGR conditional ports must carry each physical qubit exactly once. Qamomile
    can express semantic aliases that either select a different resource per
    branch or expose one resource through multiple merge results. Supporting
    those shapes requires coalescing physical ports separately from semantic
    merge values; reject them until that remapping exists instead of emitting
    an invalid graph.

    Args:
        operation (IfOperation): Conditional whose merge outputs are checked.
        true_origins (dict[str, _QuantumOrigin]): True-region provenance.
        false_origins (dict[str, _QuantumOrigin]): False-region provenance.

    Raises:
        EmitError: If resource selection diverges across branches or one branch
            yields an overlapping linear footprint more than once.
    """
    seen_true: set[_QuantumOrigin] = set()
    seen_false: set[_QuantumOrigin] = set()
    known_true: set[_QuantumOrigin] = set()
    known_false: set[_QuantumOrigin] = set()
    for merge in operation.iter_merges():
        if not merge.result.type.is_quantum():
            continue
        true_footprint = _quantum_footprint(merge.true_value, true_origins)
        false_footprint = _quantum_footprint(merge.false_value, false_origins)
        if true_footprint is not None:
            known_true.update(true_footprint)
        if false_footprint is not None:
            known_false.update(false_footprint)
        for value, footprint, seen in (
            (merge.true_value, true_footprint, seen_true),
            (merge.false_value, false_footprint, seen_false),
        ):
            branch_footprint = footprint or _intrinsic_quantum_footprint(value)
            if not branch_footprint.isdisjoint(seen):
                raise EmitError(
                    "HUGR conditional yields the same linear quantum "
                    "resource through multiple merge outputs",
                    operation="IfOperation",
                )
            seen.update(branch_footprint)
    if (known_true or known_false) and known_true != known_false:
        raise EmitError(
            "HUGR cannot lower data-dependent quantum resource selection "
            "without explicit linear merge ports",
            operation="IfOperation",
        )


def _value_quantum_origin(
    value: Value,
    origins: dict[str, _QuantumOrigin],
) -> _QuantumOrigin | None:
    """Resolve a quantum value's captured origin, including array elements.

    Args:
        value (Value): Quantum scalar, array, or array element.
        origins (dict[str, _QuantumOrigin]): Known value provenance.

    Returns:
        _QuantumOrigin | None: Captured UUID and optional element index.
    """
    origin = origins.get(value.uuid)
    if origin is not None:
        return origin
    if isinstance(value, ArrayValue):
        root_uuid = _array_root_uuid(value)
        return origins.get(root_uuid) if root_uuid is not None else None
    address = resolve_root_qubit_address(value)
    if address is None:
        return None
    root_uuid, index = address
    root_origin = origins.get(root_uuid)
    if root_origin is None or root_origin[1] is not None:
        return None
    return root_origin[0], index


def _quantum_origin_value(
    origin: _QuantumOrigin,
    environment: dict[str, Any],
) -> Any:
    """Resolve a whole captured value or one captured array element.

    Args:
        origin (_QuantumOrigin): Captured UUID and optional element index.
        environment (dict[str, Any]): Active HUGR value environment.

    Returns:
        Any: Whole scalar/array value or selected element wire.

    Raises:
        EmitError: If the recorded origin does not match its environment shape.
    """
    uuid, index = origin
    resolved = environment[uuid]
    if index is None:
        return resolved
    if not isinstance(resolved, list) or index >= len(resolved):
        raise EmitError("HUGR quantum provenance has an invalid array element")
    return resolved[index]


def _set_quantum_origin_value(
    origin: _QuantumOrigin,
    replacement: Any,
    environment: dict[str, Any],
) -> None:
    """Advance a whole captured value or one captured array element.

    Args:
        origin (_QuantumOrigin): Captured UUID and optional element index.
        replacement (Any): Replacement wire or flattened array.
        environment (dict[str, Any]): Active HUGR value environment.

    Raises:
        EmitError: If the recorded origin does not match its environment shape.
    """
    uuid, index = origin
    if index is None:
        environment[uuid] = replacement
        return
    resolved = environment[uuid]
    if isinstance(replacement, list) or not isinstance(resolved, list):
        raise EmitError("HUGR quantum provenance changed an array element shape")
    if index >= len(resolved):
        raise EmitError("HUGR quantum provenance has an invalid array element")
    resolved[index] = replacement


def _quantum_origins(
    operations: list[Operation],
    input_ids: list[str],
) -> dict[str, _QuantumOrigin]:
    """Trace quantum result values back to captured region inputs.

    Args:
        operations (list[Operation]): Region operations in execution order.
        input_ids (list[str]): UUIDs of values captured from the parent region.

    Returns:
        dict[str, _QuantumOrigin]: Quantum value UUID to captured resource and
            optional array-element index.
    """

    def trace_region(
        nested_operations: list[Operation],
        incoming: dict[str, _QuantumOrigin],
    ) -> dict[str, _QuantumOrigin]:
        """Propagate captured quantum origins through one nested region.

        Args:
            nested_operations (list[Operation]): Region operations in execution
                order.
            incoming (dict[str, _QuantumOrigin]): Quantum origins visible at
                the region entry.

        Returns:
            dict[str, _QuantumOrigin]: Quantum origins visible at the region
                exit.
        """
        current = dict(incoming)
        for operation in nested_operations:
            if isinstance(operation, IfOperation):
                true_origins = trace_region(operation.true_operations, current)
                false_origins = trace_region(operation.false_operations, current)
                for merge in operation.iter_merges():
                    common_origin = _common_quantum_origin(
                        merge.true_value,
                        merge.false_value,
                        true_origins,
                        false_origins,
                    )
                    if common_origin is not None:
                        current[merge.result.uuid] = common_origin
                continue
            if isinstance(operation, (ForOperation, WhileOperation)):
                # Frontend loop bodies are traced once. Their final quantum
                # handles can therefore escape as post-loop references even
                # when the loop operation has no explicit quantum result.
                current.update(trace_region(operation.operations, current))
                for region in operation.region_args:
                    init_origin = _value_quantum_origin(region.init, current)
                    yielded_origin = _value_quantum_origin(region.yielded, current)
                    if init_origin is not None and init_origin == yielded_origin:
                        current[region.result.uuid] = init_origin
                continue
            quantum_operands = [
                value for value in operation.operands if value.type.is_quantum()
            ]
            quantum_results = [
                value for value in operation.results if value.type.is_quantum()
            ]
            for index, result in enumerate(quantum_results):
                operand = next(
                    (
                        candidate
                        for candidate in quantum_operands
                        if candidate.logical_id == result.logical_id
                    ),
                    None,
                )
                if (
                    operand is None
                    and not isinstance(operation, InvokeOperation)
                    and index < len(quantum_operands)
                ):
                    operand = quantum_operands[index]
                if operand is None:
                    continue
                origin = _value_quantum_origin(operand, current)
                if origin is not None:
                    current[result.uuid] = origin
        return current

    return trace_region(operations, {uuid: (uuid, None) for uuid in input_ids})


def _unsupported_quantum_rebinds(
    operation: ForOperation | WhileOperation,
) -> list[LoopCarriedRebind]:
    """Find loop rebinds that change the underlying quantum resource.

    Quantum SSA UUIDs and logical IDs can both change across gates, calls, and
    structured merges. Trace each body result back to the physical value that
    entered the loop instead of treating either identifier as resource
    identity.

    Args:
        operation (ForOperation | WhileOperation): Loop whose trace-time
            rebind records should be classified.

    Returns:
        list[LoopCarriedRebind]: Quantum rebinds whose outgoing value cannot be
            proven to have the same captured origin as the incoming value.
    """
    rebinds = [
        rebind
        for rebind in operation.loop_carried_rebinds
        if rebind.before.type.is_quantum()
    ]
    input_ids: list[str] = []
    for rebind in rebinds:
        before = cast(Value, rebind.before)
        input_id = _array_root_uuid(before) or before.uuid
        if input_id not in input_ids:
            input_ids.append(input_id)
    origins = _quantum_origins(operation.operations, input_ids)
    unsupported: list[LoopCarriedRebind] = []
    for rebind in rebinds:
        before = _quantum_footprint(cast(Value, rebind.before), origins)
        after = _quantum_footprint(cast(Value, rebind.after), origins)
        if before is None or before != after:
            unsupported.append(rebind)
    return unsupported


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
    try:
        validate_region_args(operation)
    except ValueError as error:
        raise EmitError(str(error), operation="WhileOperation") from error
    quantum_rebinds = _unsupported_quantum_rebinds(operation)
    unsupported_rebinds = []
    for rebind in operation.loop_carried_rebinds:
        if rebind.before.type.is_quantum():
            continue
        if (
            len(operation.operands) >= 2
            and rebind.before.uuid == operation.operands[0].uuid
            and rebind.after.uuid == operation.operands[1].uuid
        ):
            continue
        unsupported_rebinds.append(rebind)
    if quantum_rebinds:
        names = ", ".join(rebind.var_name for rebind in quantum_rebinds)
        raise EmitError(
            "HUGR cannot lower loop-carried quantum resource rebinding in a "
            f"while loop without explicit linear region slots ({names}).",
            operation="WhileOperation",
        )
    if operation.region_args or unsupported_rebinds:
        names = [region.var_name for region in operation.region_args]
        names.extend(rebind.var_name for rebind in unsupported_rebinds)
        raise EmitError(
            "HUGR cannot lower loop-carried classical values in a while loop "
            f"without explicit region arguments ({', '.join(names)}).",
            operation="WhileOperation",
        )
    if not operation.operands:
        raise EmitError("HUGR while loop requires a condition operand")
    condition = operation.operands[0]
    condition_wire = _resolve_wire(condition, environment)

    captured_ids = _loop_capture_ids(operation, environment)
    captured_wires = _flatten_environment_values(captured_ids, environment)
    captured_width = len(captured_wires)
    loop = builder.add_tail_loop([], [condition_wire, *captured_wires])
    loop_inputs = loop.inputs()
    current_condition, *current_captured = loop_inputs
    loop_environment = dict(environment)
    loop_environment[condition.uuid] = current_condition
    _bind_region_inputs(captured_ids, current_captured, loop_environment)

    branch = loop.add_if(current_condition, *loop_inputs)
    branch_condition, *branch_captured = branch.inputs()
    true_environment = dict(loop_environment)
    true_environment[condition.uuid] = branch_condition
    _bind_region_inputs(captured_ids, branch_captured, true_environment)
    true_live = dict(live_qubits)
    true_incoming_keys = set(true_live)
    for nested in operation.operations:
        _lower_operation(nested, branch, true_environment, true_live, functions)

    if len(operation.operands) > 1:
        updated_condition = operation.operands[1]
        next_condition = _resolve_wire(updated_condition, true_environment)
    else:
        next_condition = branch_condition
    from hugr import ops, tys

    control_type = tys.Sum([[], []])
    [continue_wire] = branch.add_op(ops.Tag(0, control_type))
    next_captured = _flatten_environment_values(captured_ids, true_environment)
    _free_region_local_qubits(
        branch,
        true_live,
        true_incoming_keys,
        next_captured,
    )
    branch.set_outputs(
        continue_wire,
        next_condition,
        *next_captured,
    )

    false_branch = branch.add_else()
    [break_wire] = false_branch.add_op(ops.Tag(1, control_type))
    false_branch.set_outputs(break_wire, *false_branch.inputs())

    conditional_outputs = [
        branch.conditional_node.out(index) for index in range(2 + captured_width)
    ]
    loop.set_loop_outputs(*conditional_outputs)
    loop_outputs = [loop.parent_node.out(index) for index in range(1 + captured_width)]
    final_condition, *capture_outputs = loop_outputs
    origins = _quantum_origins(operation.operations, captured_ids)
    _publish_captured_outputs(
        captured_ids,
        capture_outputs,
        environment,
        live_qubits,
        origins,
    )
    if len(operation.operands) > 1:
        environment[operation.operands[1].uuid] = final_condition


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
    return _region_capture_ids(
        operation.operations,
        environment,
        excluded={operation.operands[0].uuid},
    )


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
    operations = [*operation.true_operations, *operation.false_operations]
    yields = [*operation.true_yields, *operation.false_yields]
    return _region_capture_ids(
        operations,
        environment,
        extra_values=yields,
        excluded={operation.condition.uuid},
    )


def _region_capture_ids(
    operations: list[Operation],
    environment: dict[str, Any],
    extra_values: list[Value] | None = None,
    excluded: set[str] | None = None,
) -> list[str]:
    """Collect stable, deduplicated captures for one nested HUGR region.

    Args:
        operations (list[Operation]): Region operations to inspect recursively.
        environment (dict[str, Any]): Available parent-region wires.
        extra_values (list[Value] | None): Region yields that cross the
            boundary without an operation read. Defaults to ``None``.
        excluded (set[str] | None): UUIDs carried by dedicated region state
            rather than generic captures. Defaults to ``None``.

    Returns:
        list[str]: Captured UUIDs in parent-environment order.
    """
    captured: set[str] = set()
    for nested in operations:
        captured.update(_captured_operand_ids(nested, environment))
    extras = extra_values or []
    for value in extras:
        if value.uuid in environment:
            captured.add(value.uuid)
            continue
        root_uuid = _array_root_uuid(value)
        if root_uuid is not None and root_uuid in environment:
            captured.add(root_uuid)
    _deduplicate_quantum_capture_ids(
        operations,
        captured,
        extras,
    )
    captured.difference_update(excluded or ())
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
    for value in operation.all_input_values():
        if value.uuid in environment:
            captured.add(value.uuid)
            continue
        if not isinstance(value, Value):
            continue
        root_uuid = _array_root_uuid(value)
        if root_uuid is not None and root_uuid in environment:
            captured.add(root_uuid)
    nested_lists = getattr(operation, "nested_op_lists", None)
    if callable(nested_lists):
        for nested_operations in nested_lists():
            for nested in nested_operations:
                captured.update(_captured_operand_ids(nested, environment))
    return captured


def _deduplicate_quantum_capture_ids(
    operations: list[Operation],
    captured: set[str],
    extra_values: list[Value] | None = None,
) -> None:
    """Drop scalar element captures already covered by a captured root array.

    Args:
        operations (list[Operation]): Region operations whose values identify
            array-element aliases.
        captured (set[str]): Mutable set of captured environment UUIDs.
        extra_values (list[Value] | None): Structural region yields not owned
            by an operation. Defaults to ``None``.
    """

    def visit_value(value: ValueBase) -> None:
        """Discard a captured element already represented by its root array.

        Args:
            value (ValueBase): Candidate quantum value to deduplicate.
        """
        if not isinstance(value, Value) or not value.type.is_quantum():
            return
        root_uuid = _array_root_uuid(value)
        if (
            root_uuid is not None
            and root_uuid != value.uuid
            and root_uuid in captured
            and value.uuid in captured
        ):
            captured.discard(value.uuid)

    def visit_operations(nested_operations: list[Operation]) -> None:
        """Visit every input value in a nested operation tree.

        Args:
            nested_operations (list[Operation]): Operations to inspect
                recursively.
        """
        for operation in nested_operations:
            for value in operation.all_input_values():
                visit_value(value)
            nested_lists = getattr(operation, "nested_op_lists", None)
            if callable(nested_lists):
                for body in nested_lists():
                    visit_operations(body)

    visit_operations(operations)
    for value in extra_values or ():
        visit_value(value)


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


def _free_region_local_qubits(
    builder: Any,
    live_qubits: dict[str, Any],
    protected_parent_keys: set[str],
    output_wires: list[Any],
) -> None:
    """Free live quantum wires created in a nested region but not yielded.

    Args:
        builder (Any): HUGR region builder receiving ``qFree`` operations.
        live_qubits (dict[str, Any]): Region-local live-resource mapping.
        protected_parent_keys (set[str]): Parent-owned live-resource keys that
            did not cross into this region and therefore cannot be freed here.
        output_wires (list[Any]): Region outputs that must remain live.
    """
    from tket_exts import quantum

    protected = list(output_wires)
    protected.extend(
        wire for key, wire in live_qubits.items() if key in protected_parent_keys
    )
    freed: list[Any] = []
    for key, wire in list(live_qubits.items()):
        if key in protected_parent_keys or any(wire == output for output in protected):
            continue
        live_qubits.pop(key)
        if any(wire == previous for previous in freed):
            continue
        builder.add_op(quantum.qFree, wire)
        freed.append(wire)


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
        resolved = replacement if isinstance(previous, list) else replacement[0]
        _replace_environment_aliases(environment, previous, resolved)
        environment[uuid] = resolved
        offset += width
    if offset != len(inputs):
        raise EmitError("HUGR region input arity mismatch")


def _rebind_captured_live_wires(
    value_ids: list[str],
    parent_environment: dict[str, Any],
    region_environment: dict[str, Any],
    live_qubits: dict[str, Any],
) -> set[str]:
    """Point region live-resource entries at their conditional input wires.

    ``live_qubits`` is copied from the parent so its keys retain semantic
    resource identity, while a HUGR region receives fresh wire handles for the
    captured values. Keeping the parent handles would make a destructive
    branch operation appear live through the old wire and hide invalid quantum
    yields.

    Args:
        value_ids (list[str]): Captured values in region-port order.
        parent_environment (dict[str, Any]): Parent UUID-to-wire mapping.
        region_environment (dict[str, Any]): Region mapping after input binding.
        live_qubits (dict[str, Any]): Mutable region live-resource mapping.

    Returns:
        set[str]: Live-resource keys rebound to region-owned input wires.
    """
    parent_wires = _flatten_environment_values(value_ids, parent_environment)
    region_wires = _flatten_environment_values(value_ids, region_environment)
    rebound_keys: set[str] = set()
    for key, candidate in list(live_qubits.items()):
        for parent_wire, region_wire in zip(
            parent_wires,
            region_wires,
            strict=True,
        ):
            if candidate == parent_wire:
                live_qubits[key] = region_wire
                rebound_keys.add(key)
                break
    return rebound_keys


def _validate_live_quantum_yields(
    values: list[Value],
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
    branch: str,
) -> None:
    """Reject branch yields that reuse a destructively consumed qubit.

    Frontend branch bookkeeping may retain a whole-array identity merge even
    when one of its elements was measured in the branch. HUGR linear wires
    cannot be both consumed by ``measure`` and yielded from the Conditional;
    detecting the stale yield here produces a target-level diagnostic instead
    of leaking a native validator error.

    Args:
        values (list[Value]): Semantic values yielded by one branch.
        environment (dict[str, Any]): Region UUID-to-wire mapping.
        live_qubits (dict[str, Any]): Live quantum wires after branch lowering.
        branch (str): Branch name used in the diagnostic.

    Raises:
        EmitError: If a quantum yield no longer denotes a live branch wire.
    """
    live_wires = list(live_qubits.values())
    quantum_values = [value for value in values if value.type.is_quantum()]
    yielded_wires = _flatten_region_yields(quantum_values, environment, branch)
    if all(
        any(wire == live_wire for live_wire in live_wires) for wire in yielded_wires
    ):
        return
    raise EmitError(
        "HUGR conditional cannot yield a quantum resource after it was "
        f"destructively consumed in the {branch} branch; partial-array "
        "quantum merges require liveness-aware lowering",
        operation="IfOperation",
    )


def _publish_captured_outputs(
    value_ids: list[str],
    outputs: list[Any],
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
    origins: dict[str, _QuantumOrigin] | None = None,
) -> None:
    """Publish flattened loop outputs back into the parent environment.

    Args:
        value_ids (list[str]): Captured parent UUIDs in stable order.
        outputs (list[Any]): Flattened TailLoop output wires.
        environment (dict[str, Any]): Parent environment to update.
        live_qubits (dict[str, Any]): Parent linear-qubit mapping to update.
        origins (dict[str, _QuantumOrigin] | None): Body quantum result UUID
            to captured resource and optional element index. Defaults to no
            result aliases.

    Raises:
        EmitError: If output arity disagrees with the captured value shapes.
    """
    replacements = {uuid: environment[uuid] for uuid in value_ids}
    _bind_region_inputs(value_ids, outputs, replacements)
    result_origins = origins or {}
    for uuid in value_ids:
        previous = environment[uuid]
        replacement = replacements[uuid]
        _replace_environment_aliases(environment, previous, replacement)
        environment[uuid] = replacement
        result_origins_for_capture = [
            (result_uuid, index)
            for result_uuid, (origin_uuid, index) in result_origins.items()
            if origin_uuid == uuid
        ]
        for result_uuid, index in result_origins_for_capture:
            if index is None:
                environment[result_uuid] = replacement
            elif isinstance(replacement, list) and index < len(replacement):
                environment[result_uuid] = replacement[index]
            else:
                raise EmitError("HUGR loop result has invalid array provenance")
        whole_result_ids = [
            result_uuid
            for result_uuid, index in result_origins_for_capture
            if index is None
        ]
        live_result = whole_result_ids[-1] if whole_result_ids else uuid
        if isinstance(replacement, list):
            was_live = any(
                f"{uuid}:{index}" in live_qubits for index in range(len(replacement))
            )
            for index in range(len(replacement)):
                live_qubits.pop(f"{uuid}:{index}", None)
            if was_live:
                live_qubits.update(
                    {
                        f"{live_result}:{index}": wire
                        for index, wire in enumerate(replacement)
                    }
                )
        elif uuid in live_qubits:
            live_qubits.pop(uuid)
            live_qubits[live_result] = replacement


def _replace_environment_aliases(
    environment: dict[str, Any],
    previous: Any,
    replacement: Any,
) -> None:
    """Advance aliases that point at one captured loop value.

    Args:
        environment (dict[str, Any]): UUID and stable-name environment.
        previous (Any): Scalar wire or flattened array before the loop.
        replacement (Any): Matching scalar wire or flattened array after it.

    Raises:
        EmitError: If array replacement widths differ.
    """
    if isinstance(previous, list):
        if not isinstance(replacement, list) or len(previous) != len(replacement):
            raise EmitError("HUGR loop capture changed array width")
        for alias, candidate in list(environment.items()):
            if candidate is previous:
                environment[alias] = replacement
                continue
            if isinstance(candidate, list):
                continue
            for old_wire, new_wire in zip(previous, replacement, strict=True):
                if candidate == old_wire:
                    environment[alias] = new_wire
                    break
        return
    if isinstance(replacement, list):
        raise EmitError("HUGR loop capture changed scalar width")
    for alias, candidate in list(environment.items()):
        if not isinstance(candidate, list) and candidate == previous:
            environment[alias] = replacement


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
        return _resolve_wire(value, environment)
    except EmitError as error:
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
    if value.type.is_quantum() and value.parent_array is not None:
        raise EmitError(
            "HUGR cannot lower dynamic quantum array indexing without an "
            "explicit linear array operation"
        )
    raise EmitError(f"Unresolved HUGR value {value.name!r}")


def _resolve_classical_argument(
    value: Value,
    builder: Any,
    environment: dict[str, Any],
) -> Any:
    """Resolve a classical call argument as a wire or compile-time object.

    Args:
        value (Value): Actual classical call operand.
        builder (Any): HUGR dataflow builder used to load scalar constants.
        environment (dict[str, Any]): Parent environment including stable
            public-parameter aliases.

    Returns:
        Any: HUGR scalar wire or a compile-time semantic object such as a
        Hamiltonian.

    Raises:
        EmitError: If the argument has no constant, UUID, or parameter-name
            resolution.
    """
    if value.uuid in environment:
        return environment[value.uuid]
    address = _array_address(value, environment)
    if address is not None:
        root_uuid, index = address
        try:
            return environment[root_uuid][index]
        except (KeyError, IndexError) as error:
            raise EmitError(
                f"Unresolved HUGR classical array element {value.name!r}"
            ) from error
    parameter_name = value.parameter_name()
    if parameter_name is not None:
        alias = f"__parameter__:{parameter_name}"
        if alias in environment:
            return environment[alias]
    if value.is_constant():
        [wire] = builder.load(_constant_value(value))
        return wire
    raise EmitError(f"Unresolved HUGR classical argument {value.name!r}")


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


def _array_root_uuid(value: Value) -> str | None:
    """Return an array value or element's root UUID without resolving indices.

    Capture analysis only needs to know which parent-region array crosses a
    nested region boundary. Unlike physical element lookup, it must also work
    while a nested loop index is not bound yet.

    Args:
        value (Value): Potential array or array-element value.

    Returns:
        str | None: Root array UUID, or ``None`` for a scalar value.
    """
    array = value if isinstance(value, ArrayValue) else value.parent_array
    if array is None:
        return None
    while array.slice_of is not None:
        array = array.slice_of
    return array.uuid


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
        }[gate_type]
        outputs = list(builder.add_op(rotation_op, qubits[0], rotation))
    elif gate_type is GateOperationType.P:
        assert operation.theta is not None
        outputs = _lower_phase_on_controls(
            operation.theta,
            builder,
            environment,
            qubits,
            direction=-1.0 if inverse else 1.0,
        )
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
        outputs = _lower_phase_on_controls(
            operation.theta,
            builder,
            environment,
            qubits,
            direction=-1.0 if inverse else 1.0,
        )
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
        resolved_theta = environment.get(theta.uuid)
        parameter_name = theta.parameter_name()
        if resolved_theta is None and parameter_name is not None:
            resolved_theta = environment.get(f"__parameter__:{parameter_name}")
        if resolved_theta is None:
            raise EmitError(f"Unresolved HUGR rotation value {theta.name!r}")
        [factor] = builder.load(FloatVal(scale / math.pi))
        multiply = FLOAT_OPS_EXTENSION.get_op("fmul").instantiate()
        [halfturns] = builder.add_op(multiply, resolved_theta, factor)
    [result] = builder.add_op(rotation.from_halfturns_unchecked, halfturns)
    return result


def _lower_global_phase(
    phase: Value,
    builder: Any,
    environment: dict[str, Any],
    scale: float = 1.0,
) -> None:
    """Emit a native zero-qubit TKET global-phase operation.

    Args:
        phase (Value): Radian-valued phase angle.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.
        scale (float): Numeric factor applied before emission. Defaults to
            one.
    """
    import tket_exts

    rotation = _rotation_wire(builder, phase, environment, scale=scale)
    operation = tket_exts.global_phase().get_op("global_phase").instantiate()
    builder.add_op(operation, rotation)


def _lower_phase_on_controls(
    phase: Value,
    builder: Any,
    environment: dict[str, Any],
    controls: list[Any],
    direction: float = 1.0,
) -> list[Any]:
    """Emit an exact phase conditioned on one or two control wires.

    A one-wire projector phase is ``P(θ) = exp(iθ/2) Rz(θ)``. With two
    wires, ``CP(θ) = exp(iθ/4) Rz(θ/2) CRz(θ)``. Keeping both decompositions
    here prevents the zero-qubit factor from diverging between primitive and
    transformed-call lowering.

    Args:
        phase (Value): Radian-valued phase angle.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.
        controls (list[Any]): One or two existing control wires.
        direction (float): Positive for the phase and negative for its
            inverse. Defaults to one.

    Returns:
        list[Any]: Updated linear control wires.

    Raises:
        EmitError: If the current HUGR lowering profile cannot synthesize the
            requested control width.
    """
    from tket_exts import quantum

    if len(controls) == 1:
        _lower_global_phase(
            phase,
            builder,
            environment,
            scale=0.5 * direction,
        )
        rotation = _rotation_wire(
            builder,
            phase,
            environment,
            scale=direction,
        )
        [control] = builder.add_op(quantum.Rz, controls[0], rotation)
        return [control]
    if len(controls) == 2:
        _lower_global_phase(
            phase,
            builder,
            environment,
            scale=0.25 * direction,
        )
        half = _rotation_wire(
            builder,
            phase,
            environment,
            scale=0.5 * direction,
        )
        [left] = builder.add_op(quantum.Rz, controls[0], half)
        full = _rotation_wire(
            builder,
            phase,
            environment,
            scale=direction,
        )
        left, right = builder.add_op(quantum.CRz, left, controls[1], full)
        return [left, right]
    raise EmitError(
        "HUGR phase synthesis supports exactly one or two coherent controls",
        operation="GlobalPhaseOperation",
    )


def _lower_pauli_evolution(
    operation: PauliEvolveOp,
    builder: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
) -> None:
    """Lower Pauli evolution at the HUGR target boundary.

    HUGR's installed TKET extension has no whole-Hamiltonian evolution node,
    so each Pauli term is lowered here, after the semantic operation has
    survived every shared compiler stage.

    Args:
        operation (PauliEvolveOp): Bound Pauli evolution operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping to update.
        live_qubits (dict[str, Any]): Live quantum mapping to update.

    Raises:
        EmitError: If the Hamiltonian is unbound, the qubit operand is not a
            fixed vector, or a Pauli term addresses outside the register.
    """
    from tket_exts import quantum

    import qamomile.observable as qm_o

    qubit_operand = operation.qubits
    result = operation.evolved_qubits
    if not isinstance(qubit_operand, ArrayValue) or not isinstance(result, ArrayValue):
        raise EmitError("HUGR Pauli evolution requires a fixed qubit vector")
    if qubit_operand.uuid not in environment:
        raise EmitError("HUGR Pauli evolution cannot resolve its qubit vector")
    hamiltonian = (
        cast(Any, operation.observable.get_const())
        if operation.observable.is_constant()
        else environment.get(operation.observable.uuid)
    )
    if hamiltonian is None:
        raise EmitError("HUGR Pauli evolution requires a bound Hamiltonian")
    if not isinstance(hamiltonian, qm_o.Hamiltonian):
        raise EmitError("HUGR Pauli evolution binding is not a Hamiltonian")
    _validate_pauli_evolution_hamiltonian(hamiltonian)

    qubits = list(environment[qubit_operand.uuid])
    constant_value = float(hamiltonian.constant.real)
    if constant_value:
        _lower_global_phase(
            operation.gamma,
            builder,
            environment,
            scale=-constant_value,
        )
    for operators, coefficient in hamiltonian:
        if not operators or is_close_zero(abs(coefficient)):
            continue
        selected_indices = [item.index for item in operators]
        if any(index < 0 or index >= len(qubits) for index in selected_indices):
            raise EmitError("HUGR Pauli term addresses outside the qubit vector")
        selected = [qubits[index] for index in selected_indices]
        for item_index, (item, wire) in enumerate(
            zip(operators, selected, strict=True)
        ):
            if item.pauli is qm_o.Pauli.X:
                [wire] = builder.add_op(quantum.H, wire)
            elif item.pauli is qm_o.Pauli.Y:
                [wire] = builder.add_op(quantum.Sdg, wire)
                [wire] = builder.add_op(quantum.H, wire)
            selected[item_index] = wire
        for index in range(len(selected) - 1):
            selected[index], selected[index + 1] = builder.add_op(
                quantum.CX,
                selected[index],
                selected[index + 1],
            )
        rotation = _rotation_wire(
            builder,
            operation.gamma,
            environment,
            scale=2.0 * float(coefficient.real),
        )
        [selected[-1]] = builder.add_op(quantum.Rz, selected[-1], rotation)
        for index in range(len(selected) - 2, -1, -1):
            selected[index], selected[index + 1] = builder.add_op(
                quantum.CX,
                selected[index],
                selected[index + 1],
            )
        for item_index in range(len(operators) - 1, -1, -1):
            item = operators[item_index]
            wire = selected[item_index]
            if item.pauli is qm_o.Pauli.X:
                [wire] = builder.add_op(quantum.H, wire)
            elif item.pauli is qm_o.Pauli.Y:
                [wire] = builder.add_op(quantum.H, wire)
                [wire] = builder.add_op(quantum.S, wire)
            qubits[item.index] = wire

    environment[result.uuid] = qubits
    environment[qubit_operand.uuid] = qubits
    for index in range(len(qubits)):
        live_qubits.pop(f"{qubit_operand.uuid}:{index}", None)
        live_qubits[f"{result.uuid}:{index}"] = qubits[index]


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
    operands = [
        (
            _pack_value(value, environment, builder)
            if isinstance(value, ArrayValue)
            else _resolve_wire(value, environment)
            if value.type.is_quantum()
            else _resolve_classical_argument(value, builder, environment)
        )
        for value in operation.operands
    ]
    outputs = list(builder.call(functions[operation.target].parent_node, *operands))
    if len(outputs) != len(operation.results):
        raise EmitError("HUGR callable result arity mismatch")
    consumed_wires: list[Any] = []
    for operand in operation.operands:
        if not operand.type.is_quantum():
            continue
        resolved = _resolve_wire(operand, environment)
        consumed_wires.extend(resolved if isinstance(resolved, list) else [resolved])
    for key, live_wire in list(live_qubits.items()):
        if any(live_wire == consumed for consumed in consumed_wires):
            live_qubits.pop(key)
    quantum_sources = [value for value in operation.operands if value.type.is_quantum()]
    for result, wire in zip(operation.results, outputs, strict=True):
        if isinstance(result, ArrayValue):
            from hugr import ops

            element_types = [
                _lower_type(result.type) for _ in range(_array_size(result))
            ]
            elements = list(builder.add_op(ops.UnpackTuple(element_types), wire))
            if result.type.is_quantum():
                source = next(
                    (
                        value
                        for value in quantum_sources
                        if value.logical_id == result.logical_id
                    ),
                    None,
                )
                _publish_transformed_result(
                    source,
                    result,
                    elements,
                    environment,
                    live_qubits,
                )
            else:
                environment[result.uuid] = elements
        elif result.type.is_quantum():
            source = next(
                (
                    value
                    for value in quantum_sources
                    if value.logical_id == result.logical_id
                ),
                None,
            )
            _publish_transformed_result(
                source,
                result,
                wire,
                environment,
                live_qubits,
            )
        else:
            environment[result.uuid] = wire


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
        classical = [
            value
            for value in operation.operands
            if value.type.is_classical() or value.type.is_object()
        ]
        inverse = operation.transform is CallTransform.INVERSE
        display_name = operation.target.name
    if body is None:
        raise EmitError(f"Transformed HUGR callable {display_name!r} is opaque")
    body = InlinePass().run(body)
    power = _resolve_transformed_power(operation, environment)
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
    if inverse:
        if len(quantum_entry) != len(targets):
            raise EmitError("HUGR transformed call quantum arity mismatch")
        for formal, actual in zip(quantum_entry, targets, strict=True):
            local[formal.uuid] = _resolve_wire(actual, environment)
        if len(body_classical_inputs) != len(classical):
            raise EmitError("HUGR transformed call classical arity mismatch")
        pairs = zip(body_classical_inputs, classical, strict=True)
    else:
        operand_pairs = pair_block_operands(body, [*targets, *classical])
        if len(operand_pairs) != len(body.input_values):
            raise EmitError("HUGR transformed call operand arity mismatch")
        pairs = (
            (formal, actual)
            for formal, actual in operand_pairs
            if not formal.type.is_quantum()
        )
        for formal, actual in operand_pairs:
            if formal.type.is_quantum():
                local[formal.uuid] = _resolve_wire(cast(Value, actual), environment)
    for formal, actual in pairs:
        resolved = _resolve_classical_argument(
            cast(Value, actual),
            builder,
            environment,
        )
        local[formal.uuid] = resolved
        parameter_name = formal.parameter_name()
        if parameter_name is not None:
            local[f"__parameter__:{parameter_name}"] = resolved

    control_wires = []
    for value in controls:
        resolved = _resolve_wire(value, environment)
        control_wires.extend(resolved if isinstance(resolved, list) else [resolved])
    for _ in range(power):
        sequence = reversed(body.operations) if inverse else body.operations
        for nested in sequence:
            if isinstance(nested, ReturnOperation):
                continue
            if isinstance(nested, CInitOperation):
                _lower_cinit(nested, builder, local)
                continue
            if isinstance(nested, GlobalPhaseOperation):
                direction = -1.0 if inverse else 1.0
                if control_wires:
                    control_wires = _lower_phase_on_controls(
                        nested.phase,
                        builder,
                        local,
                        control_wires,
                        direction,
                    )
                else:
                    _lower_global_phase(
                        nested.phase,
                        builder,
                        local,
                        scale=direction,
                    )
                continue
            if isinstance(nested, PauliEvolveOp):
                if len(control_wires) != 1:
                    raise EmitError(
                        "HUGR transformed Pauli evolution supports exactly one control"
                    )
                control_wires[0] = _lower_controlled_pauli_evolution(
                    nested,
                    builder,
                    local,
                    control_wires[0],
                    inverse,
                )
                continue
            if not isinstance(nested, GateOperation):
                raise EmitError(
                    "HUGR transformed calls currently require primitive unitary "
                    f"bodies; found {type(nested).__name__}"
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
    control_offset = 0
    for source, result in zip(controls, result_controls, strict=True):
        width = _array_size(source) if isinstance(source, ArrayValue) else 1
        selected = control_wires[control_offset : control_offset + width]
        control_offset += width
        wire: Any = selected if isinstance(result, ArrayValue) else selected[0]
        _publish_transformed_result(source, result, wire, environment, live_qubits)
    for source, result, formal in zip(
        targets, result_targets, quantum_exit, strict=True
    ):
        wire = local[formal.uuid]
        _publish_transformed_result(source, result, wire, environment, live_qubits)


def _resolve_transformed_power(
    operation: InvokeOperation | ControlledUOperation | InverseBlockOperation,
    environment: dict[str, Any],
) -> int:
    """Resolve a transformed call's statically known positive power.

    Args:
        operation (InvokeOperation | ControlledUOperation |
            InverseBlockOperation): Transformed call being lowered.
        environment (dict[str, Any]): UUID-to-wire and compile-time value
            mapping.

    Returns:
        int: Positive number of complete body applications.

    Raises:
        EmitError: If a controlled-call power is dynamic or invalid.
    """
    if not isinstance(operation, ControlledUOperation):
        return 1
    power = operation.power
    if isinstance(power, bool):
        resolved: Any = None
    elif isinstance(power, int):
        resolved = power
    elif power.is_constant():
        resolved = power.get_const()
    else:
        resolved = environment.get(f"__index__:{power.uuid}")
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved <= 0:
        raise EmitError(
            "HUGR transformed call power must be a compile-time positive integer",
            operation="ControlledUOperation",
        )
    return resolved


def _publish_transformed_result(
    source: Value | None,
    result: Value,
    wire: Any,
    environment: dict[str, Any],
    live_qubits: dict[str, Any],
) -> None:
    """Publish a transformed-call result and its containing array version.

    Args:
        source (Value | None): Same-resource quantum value consumed by the
            call, or ``None`` when the result has no input lineage.
        result (Value): Quantum SSA value produced by the transformed call.
        wire (Any): Resulting linear HUGR wire.
        environment (dict[str, Any]): UUID-to-wire mapping to update.
        live_qubits (dict[str, Any]): Live quantum resource mapping to update.

    Raises:
        EmitError: If a result array version cannot be reconstructed.
    """
    if isinstance(result, ArrayValue):
        if not isinstance(wire, list):
            raise EmitError(
                f"HUGR array result {result.name!r} did not produce linear elements"
            )
        source_wires: list[Any] = (
            _resolve_wire(source, environment) if isinstance(source, ArrayValue) else []
        )
        if source is not None:
            _replace_environment_aliases(environment, source_wires, wire)
            environment[source.uuid] = wire
        environment[result.uuid] = wire
        for key, live_wire in list(live_qubits.items()):
            if any(live_wire == source_wire for source_wire in source_wires):
                live_qubits.pop(key)
        for index, element in enumerate(wire):
            live_qubits[f"{result.uuid}:{index}"] = element
        return

    previous = _resolve_wire(source, environment) if source is not None else None
    if source is not None:
        _replace_environment_aliases(environment, previous, wire)
        environment[source.uuid] = wire
        source_address = _array_address(source, environment)
        if source_address is not None:
            source_root, source_index = source_address
            environment[source_root][source_index] = wire
        live_qubits.pop(_quantum_key(source, environment), None)
    environment[result.uuid] = wire
    address = _array_address(result, environment)
    if address is not None:
        root_uuid, index = address
        if root_uuid not in environment:
            source_address = (
                _array_address(source, environment) if source is not None else None
            )
            if source_address is None or source_address[0] not in environment:
                raise EmitError(f"Cannot reconstruct HUGR array result {result.name!r}")
            environment[root_uuid] = list(environment[source_address[0]])
        environment[root_uuid][index] = wire
    live_qubits[_quantum_key(result, environment)] = wire


def _lower_controlled_pauli_evolution(
    operation: PauliEvolveOp,
    builder: Any,
    environment: dict[str, Any],
    control: Any,
    inverse: bool,
) -> Any:
    """Lower a one-control Pauli evolution through TKET controlled rotations.

    Basis changes and parity ladders are emitted unconditionally around each
    controlled rotation; they cancel when the control is zero and therefore
    preserve the whole-call semantics without distributing control onto every
    primitive gate.

    Args:
        operation (PauliEvolveOp): Pauli evolution inside a transformed body.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): Body-local UUID-to-wire mapping.
        control (Any): Current linear control-qubit wire.
        inverse (bool): Whether to apply the inverse evolution.

    Returns:
        Any: Updated linear control-qubit wire.

    Raises:
        EmitError: If the Hamiltonian or fixed target vector cannot be
            resolved, or a term addresses outside that vector.
    """
    from tket_exts import quantum

    import qamomile.observable as qm_o

    qubit_operand = operation.qubits
    result = operation.evolved_qubits
    if not isinstance(qubit_operand, ArrayValue) or not isinstance(result, ArrayValue):
        raise EmitError("Controlled HUGR Pauli evolution requires a fixed vector")
    if qubit_operand.uuid not in environment:
        raise EmitError("Controlled HUGR Pauli evolution cannot resolve its vector")
    if operation.observable.is_constant():
        hamiltonian = cast(Any, operation.observable.get_const())
    else:
        hamiltonian = environment.get(operation.observable.uuid)
        parameter_name = operation.observable.parameter_name()
        if hamiltonian is None and parameter_name is not None:
            hamiltonian = environment.get(f"__parameter__:{parameter_name}")
    if not isinstance(hamiltonian, qm_o.Hamiltonian):
        raise EmitError("Controlled HUGR Pauli evolution requires a bound Hamiltonian")
    _validate_pauli_evolution_hamiltonian(hamiltonian)

    qubits = list(environment[qubit_operand.uuid])
    direction = -1.0 if inverse else 1.0
    constant_value = float(hamiltonian.constant.real)
    if constant_value:
        _lower_global_phase(
            operation.gamma,
            builder,
            environment,
            scale=-0.5 * direction * constant_value,
        )
        rotation = _rotation_wire(
            builder,
            operation.gamma,
            environment,
            scale=-direction * constant_value,
        )
        [control] = builder.add_op(quantum.Rz, control, rotation)
    for operators, coefficient in hamiltonian:
        coefficient_value = float(coefficient.real)
        if is_close_zero(abs(coefficient_value)):
            continue
        selected_indices = [item.index for item in operators]
        if any(index < 0 or index >= len(qubits) for index in selected_indices):
            raise EmitError("Controlled HUGR Pauli term exceeds the target vector")
        selected = [qubits[index] for index in selected_indices]
        for item_index, (item, wire) in enumerate(
            zip(operators, selected, strict=True)
        ):
            if item.pauli is qm_o.Pauli.X:
                [wire] = builder.add_op(quantum.H, wire)
            elif item.pauli is qm_o.Pauli.Y:
                [wire] = builder.add_op(quantum.Sdg, wire)
                [wire] = builder.add_op(quantum.H, wire)
            selected[item_index] = wire
        for index in range(len(selected) - 1):
            selected[index], selected[index + 1] = builder.add_op(
                quantum.CX,
                selected[index],
                selected[index + 1],
            )
        rotation = _rotation_wire(
            builder,
            operation.gamma,
            environment,
            scale=direction * 2.0 * coefficient_value,
        )
        control, selected[-1] = builder.add_op(
            quantum.CRz,
            control,
            selected[-1],
            rotation,
        )
        for index in range(len(selected) - 2, -1, -1):
            selected[index], selected[index + 1] = builder.add_op(
                quantum.CX,
                selected[index],
                selected[index + 1],
            )
        for item_index in range(len(operators) - 1, -1, -1):
            item = operators[item_index]
            wire = selected[item_index]
            if item.pauli is qm_o.Pauli.X:
                [wire] = builder.add_op(quantum.H, wire)
            elif item.pauli is qm_o.Pauli.Y:
                [wire] = builder.add_op(quantum.H, wire)
                [wire] = builder.add_op(quantum.S, wire)
            qubits[item.index] = wire

    environment[qubit_operand.uuid] = qubits
    environment[result.uuid] = qubits
    return control


def _validate_pauli_evolution_hamiltonian(hamiltonian: Any) -> None:
    """Reject non-Hermitian coefficients before HUGR gate synthesis.

    Args:
        hamiltonian (Any): Bound Qamomile Hamiltonian.

    Raises:
        EmitError: If the identity or a Pauli coefficient has a material
            imaginary component.
    """
    from qamomile.observable.hamiltonian import HERMITIAN_IMAG_ATOL

    if abs(hamiltonian.constant.imag) > HERMITIAN_IMAG_ATOL:
        raise EmitError(
            "HUGR Pauli evolution requires a Hermitian Hamiltonian; "
            "the identity coefficient is non-real"
        )
    for operators, coefficient in hamiltonian:
        if abs(coefficient.imag) > HERMITIAN_IMAG_ATOL:
            raise EmitError(
                "HUGR Pauli evolution requires a Hermitian Hamiltonian; "
                f"coefficient {coefficient} on term {operators} is non-real"
            )


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
    elif operation.gate_type is GateOperationType.P:
        assert operation.theta is not None
        projector_wires = _lower_phase_on_controls(
            operation.theta,
            builder,
            environment,
            [*controls, target],
            direction=-1.0 if inverse else 1.0,
        )
        controls = projector_wires[:-1]
        target = projector_wires[-1]
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
    """Lower supported floating-point and unsigned arithmetic to HUGR.

    Args:
        operation (BinOp): Qamomile binary arithmetic operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.

    Raises:
        EmitError: If operand types or operation kind are unsupported.
    """
    if operation.kind is None:
        raise EmitError("HUGR arithmetic operation has no operation kind")
    if all(isinstance(value.type, FloatType) for value in operation.operands):
        from hugr import tys
        from hugr.std.float import FLOAT_OPS_EXTENSION, FLOAT_T

        names = {
            BinOpKind.ADD: "fadd",
            BinOpKind.SUB: "fsub",
            BinOpKind.MUL: "fmul",
            BinOpKind.DIV: "fdiv",
            BinOpKind.POW: "fpow",
            BinOpKind.MIN: "fmin",
        }
        extension = FLOAT_OPS_EXTENSION
        arguments = None
        concrete_signature = tys.FunctionType(
            [FLOAT_T, FLOAT_T],
            [FLOAT_T],
        )
    elif all(isinstance(value.type, UIntType) for value in operation.operands):
        from hugr import tys
        from hugr.std.int import INT_OPS_EXTENSION, INT_T, IntVal

        concrete = []
        for value in operation.operands:
            if value.is_constant():
                concrete.append(int(value.get_const()))
                continue
            known = environment.get(f"__index__:{value.uuid}")
            if not isinstance(known, int):
                break
            concrete.append(known)
        if len(concrete) == 2:
            match operation.kind:
                case BinOpKind.ADD:
                    value = concrete[0] + concrete[1]
                case BinOpKind.SUB:
                    value = concrete[0] - concrete[1]
                case BinOpKind.MUL:
                    value = concrete[0] * concrete[1]
                case BinOpKind.POW:
                    value = concrete[0] ** concrete[1]
                case _:
                    raise EmitError(
                        f"Unsupported HUGR arithmetic operation: {operation.kind}"
                    )
            [wire] = builder.load(IntVal(value))
            result = operation.results[0]
            environment[result.uuid] = wire
            environment[f"__index__:{result.uuid}"] = value
            return
        names = {
            BinOpKind.ADD: "iadd",
            BinOpKind.SUB: "isub",
            BinOpKind.MUL: "imul",
            BinOpKind.POW: "ipow",
        }
        extension = INT_OPS_EXTENSION
        arguments = [tys.BoundedNatArg(5)]
        concrete_signature = tys.FunctionType(
            [INT_T, INT_T],
            [INT_T],
        )
    else:
        raise EmitError("HUGR arithmetic operands must share Float or UInt type")
    name = names.get(operation.kind)
    if name is None:
        raise EmitError(f"Unsupported HUGR arithmetic operation: {operation.kind}")
    op = extension.get_op(name).instantiate(
        arguments,
        concrete_signature=concrete_signature,
    )
    [wire] = builder.add_op(
        op,
        _resolve_classical_argument(operation.operands[0], builder, environment),
        _resolve_classical_argument(operation.operands[1], builder, environment),
    )
    result = operation.results[0]
    environment[result.uuid] = wire


def _lower_compop(
    operation: CompOp,
    builder: Any,
    environment: dict[str, Any],
) -> None:
    """Lower Float and UInt comparisons to matching HUGR extensions.

    Args:
        operation (CompOp): Qamomile comparison operation.
        builder (Any): HUGR dataflow builder.
        environment (dict[str, Any]): UUID-to-wire mapping.

    Raises:
        EmitError: If operand types or comparison kind are unsupported.
    """
    if operation.kind is None:
        raise EmitError("HUGR comparison has no comparison kind")
    if all(isinstance(value.type, FloatType) for value in operation.operands):
        from hugr.std.float import FLOAT_OPS_EXTENSION

        names = {
            CompOpKind.EQ: "feq",
            CompOpKind.NEQ: "fne",
            CompOpKind.LT: "flt",
            CompOpKind.LE: "fle",
            CompOpKind.GT: "fgt",
            CompOpKind.GE: "fge",
        }
        extension = FLOAT_OPS_EXTENSION
        arguments: list[Any] = []
        concrete_signature = None
    elif all(isinstance(value.type, UIntType) for value in operation.operands):
        from hugr import tys
        from hugr.std.int import INT_OPS_EXTENSION, INT_T

        names = {
            CompOpKind.EQ: "ieq",
            CompOpKind.NEQ: "ine",
            CompOpKind.LT: "ilt_u",
            CompOpKind.LE: "ile_u",
            CompOpKind.GT: "igt_u",
            CompOpKind.GE: "ige_u",
        }
        extension = INT_OPS_EXTENSION
        arguments = [tys.BoundedNatArg(5)]
        concrete_signature = tys.FunctionType([INT_T, INT_T], [tys.Bool])
    else:
        raise EmitError("HUGR comparison operands must share Float or UInt type")
    name = names.get(operation.kind)
    if name is None:
        raise EmitError(f"Unsupported HUGR comparison: {operation.kind}")
    op = extension.get_op(name).instantiate(
        arguments,
        concrete_signature=concrete_signature,
    )
    [wire] = builder.add_op(
        op,
        _resolve_classical_argument(operation.operands[0], builder, environment),
        _resolve_classical_argument(operation.operands[1], builder, environment),
    )
    environment[operation.results[0].uuid] = wire
