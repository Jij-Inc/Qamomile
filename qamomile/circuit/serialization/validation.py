"""Semantic validation for static qkernel IR at the serialization boundary."""

from __future__ import annotations

import dataclasses
from typing import Iterable, cast

from qamomile._utils import is_plain_int
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import (
    ExpvalOp,
    ForItemsOperation,
    GateOperation,
    GateOperationType,
    GlobalPhaseOperation,
    InverseBlockOperation,
    InvokeOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
    ReturnOperation,
    SelectOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CondOp,
    NotOp,
    RuntimeClassicalExpr,
    RuntimeOpKind,
)
from qamomile.circuit.ir.operation.callable import CallableDef, CallTransform
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import (
    DecodeQFixedOperation,
    DictGetItemOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    HasNestedOps,
    IfOperation,
    RegionArg,
    WhileOperation,
)
from qamomile.circuit.ir.operation.control_value import normalize_control_value
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.operation import (
    CInitOperation,
    Operation,
    QInitOperation,
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.slice_array import (
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.types.hamiltonian import ObservableType
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    QubitType,
    UIntType,
    ValueType,
)
from qamomile.circuit.ir.types.q_register import QFixedType
from qamomile.circuit.ir.value import ArrayValue, DictValue, Value, ValueBase

_SINGLE_QUBIT_GATES = frozenset(
    {
        GateOperationType.H,
        GateOperationType.X,
        GateOperationType.Y,
        GateOperationType.Z,
        GateOperationType.T,
        GateOperationType.TDG,
        GateOperationType.S,
        GateOperationType.SDG,
    }
)
_SINGLE_QUBIT_ROTATIONS = frozenset(
    {
        GateOperationType.P,
        GateOperationType.RX,
        GateOperationType.RY,
        GateOperationType.RZ,
    }
)
_TWO_QUBIT_GATES = frozenset(
    {GateOperationType.CX, GateOperationType.CZ, GateOperationType.SWAP}
)
_TWO_QUBIT_ROTATIONS = frozenset({GateOperationType.CP, GateOperationType.RZZ})


@dataclasses.dataclass
class _ValidationState:
    """Mutable state shared while validating a complete callable graph."""

    seen_blocks: set[int] = dataclasses.field(default_factory=set)
    seen_definitions: set[int] = dataclasses.field(default_factory=set)


def validate_qkernel_ir(block: Block) -> None:
    """Validate every operation and nested region reachable from a qkernel.

    Args:
        block (Block): Root unbound hierarchical qkernel block.

    Raises:
        ValueError: If an operation violates its arity, type, SSA, callable,
            or region contract.
    """
    _validate_block(block, _ValidationState(), "qkernel body")


def _validate_block(block: Block, state: _ValidationState, location: str) -> None:
    """Validate one block and every graph edge reachable from it.

    Args:
        block (Block): Block to validate.
        state (_ValidationState): Shared graph-validation state.
        location (str): Human-readable graph location.

    Raises:
        ValueError: If the block contains malformed values or operations.
    """
    if id(block) in state.seen_blocks:
        return
    state.seen_blocks.add(id(block))
    _require_unique_values(block.input_values, f"{location} inputs")
    _require_unique_values(block.output_values, f"{location} outputs")
    producers: dict[str, str] = {}

    for index, operation in enumerate(block.operations):
        op_location = f"{location} operation {index} ({type(operation).__name__})"
        _validate_operation(operation, state, producers, op_location)


def _validate_operation(
    operation: Operation,
    state: _ValidationState,
    producers: dict[str, str],
    location: str,
) -> None:
    """Validate one operation before following its nested graph edges.

    Args:
        operation (Operation): Operation to validate.
        state (_ValidationState): Shared graph-validation state.
        producers (dict[str, str]): SSA producers in the owning block scope.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the operation is structurally or semantically invalid.
    """
    if not all(isinstance(value, ValueBase) for value in operation.operands):
        raise ValueError(f"{location} has a non-Value operand")
    if not all(isinstance(value, ValueBase) for value in operation.results):
        raise ValueError(f"{location} has a non-Value result")
    _require_unique_values(operation.results, f"{location} results")
    operand_uuids = {value.uuid for value in operation.operands}
    for result in operation.results:
        if result.uuid in operand_uuids:
            raise ValueError(f"{location} reuses an operand UUID as an SSA result")
        previous = producers.get(result.uuid)
        if previous is not None:
            raise ValueError(
                f"{location} produces UUID {result.uuid!r} already produced by "
                f"{previous}"
            )
        producers[result.uuid] = location

    _validate_operation_contract(operation, location)
    _validate_quantum_operand_uniqueness(operation, location)

    if isinstance(operation, HasNestedOps):
        for region_index, operations in enumerate(operation.nested_op_lists()):
            for child_index, child in enumerate(operations):
                _validate_operation(
                    child,
                    state,
                    producers,
                    f"{location} region {region_index} operation {child_index} "
                    f"({type(child).__name__})",
                )
    if isinstance(operation, InvokeOperation) and operation.definition is not None:
        _validate_definition(operation.definition, state, location)
    if isinstance(operation, (ConcreteControlledU, SymbolicControlledU)):
        if operation.block is not None:
            _validate_block(operation.block, state, f"{location} unitary block")
    if isinstance(operation, InverseBlockOperation):
        if operation.source_block is not None:
            _validate_block(operation.source_block, state, f"{location} source block")
        if operation.implementation_block is not None:
            _validate_block(
                operation.implementation_block,
                state,
                f"{location} implementation block",
            )
    if isinstance(operation, SelectOperation):
        for case_index, case_block in enumerate(operation.case_blocks):
            _validate_block(
                case_block,
                state,
                f"{location} case block {case_index}",
            )


def _validate_definition(
    definition: CallableDef,
    state: _ValidationState,
    location: str,
) -> None:
    """Validate semantic blocks owned by one callable definition.

    Args:
        definition (CallableDef): Callable definition to inspect.
        state (_ValidationState): Shared graph-validation state.
        location (str): Location of the invocation that references it.

    Raises:
        ValueError: If a callable body or implementation is malformed.
    """
    if id(definition) in state.seen_definitions:
        return
    state.seen_definitions.add(id(definition))
    if definition.body is not None:
        _validate_block(definition.body, state, f"{location} callable body")
    for index, implementation in enumerate(definition.implementations):
        if implementation.body is not None:
            _validate_block(
                implementation.body,
                state,
                f"{location} callable implementation {index}",
            )


def _validate_operation_contract(operation: Operation, location: str) -> None:
    """Dispatch operation-specific arity, type, and region checks.

    Args:
        operation (Operation): Operation to inspect.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If a concrete operation contract is violated.
    """
    if isinstance(operation, GateOperation):
        _validate_gate(operation, location)
    elif isinstance(operation, MeasureOperation):
        _require_arity(operation, 1, 1, location)
        _require_types(operation.operands, [QubitType()], location, "operand")
        _require_types(operation.results, [BitType()], location, "result")
    elif isinstance(operation, ProjectOperation):
        _require_arity(operation, 1, 2, location)
        _require_types(operation.operands, [QubitType()], location, "operand")
        _require_types(
            operation.results,
            [QubitType(), BitType()],
            location,
            "result",
        )
    elif isinstance(operation, ResetOperation):
        _require_arity(operation, 1, 1, location)
        _require_types(operation.operands, [QubitType()], location, "operand")
        _require_types(operation.results, [QubitType()], location, "result")
    elif isinstance(operation, MeasureVectorOperation):
        _require_arity(operation, 1, 1, location)
        _require_array_type(operation.operands[0], QubitType(), location)
        _require_array_type(operation.results[0], BitType(), location)
    elif isinstance(operation, MeasureQFixedOperation):
        _require_arity(operation, 1, 1, location)
        _validate_fixed_point_layout(
            operation.num_bits,
            operation.int_bits,
            location,
        )
        _require_types(operation.operands, [QFixedType()], location, "operand")
        _require_types(operation.results, [FloatType()], location, "result")
    elif isinstance(operation, DecodeQFixedOperation):
        _require_arity(operation, 1, 1, location)
        _validate_fixed_point_layout(
            operation.num_bits,
            operation.int_bits,
            location,
        )
        _require_array_type(operation.operands[0], BitType(), location)
        _require_types(operation.results, [FloatType()], location, "result")
    elif isinstance(operation, StoreArrayElementOperation):
        _validate_array_store(operation, location)
    elif isinstance(operation, DictGetItemOperation):
        _validate_dict_get(operation, location)
    elif isinstance(operation, CastOperation):
        _require_arity(operation, 1, 1, location)
        if operation.source_type is None or operation.target_type is None:
            raise ValueError(f"{location} requires source_type and target_type")
        _require_types(
            operation.operands,
            [operation.source_type],
            location,
            "operand",
        )
        _require_types(
            operation.results,
            [operation.target_type],
            location,
            "result",
        )
    elif isinstance(operation, (QInitOperation, CInitOperation)):
        _require_arity(operation, 0, 1, location)
        if (
            isinstance(operation, QInitOperation)
            and not operation.results[0].type.is_quantum()
        ):
            raise ValueError(f"{location} must initialize a quantum value")
        if (
            isinstance(operation, CInitOperation)
            and operation.results[0].type.is_quantum()
        ):
            raise ValueError(f"{location} cannot initialize a quantum value")
    elif isinstance(operation, SliceArrayOperation):
        _require_arity(operation, 3, 1, location)
        if not isinstance(operation.operands[0], ArrayValue) or not isinstance(
            operation.results[0], ArrayValue
        ):
            raise ValueError(f"{location} must slice an ArrayValue into an ArrayValue")
        _require_types(
            operation.operands[1:], [UIntType(), UIntType()], location, "operand"
        )
    elif isinstance(operation, ReleaseSliceViewOperation):
        _require_arity(operation, 1, 0, location)
        if not isinstance(operation.operands[0], ArrayValue):
            raise ValueError(f"{location} must release an ArrayValue")
    elif isinstance(operation, ReturnOperation):
        _require_result_count(operation, 0, location)
    elif isinstance(operation, GlobalPhaseOperation):
        _require_arity(operation, 1, 0, location)
        _require_scalar_type(
            operation.operands[0],
            FloatType(),
            f"{location} phase operand",
        )
    elif isinstance(operation, ExpvalOp):
        _require_arity(operation, 2, 1, location)
        if not operation.operands[0].type.is_quantum():
            raise ValueError(f"{location} first operand must be quantum")
        _require_types(operation.operands[1:], [ObservableType()], location, "operand")
        _require_types(operation.results, [FloatType()], location, "result")
    elif isinstance(operation, PauliEvolveOp):
        _require_arity(operation, 3, 1, location)
        if not operation.operands[0].type.is_quantum():
            raise ValueError(f"{location} first operand must be quantum")
        _require_types(
            operation.operands[1:2],
            [ObservableType()],
            location,
            "operand",
        )
        _require_scalar_type(
            operation.operands[2],
            FloatType(),
            f"{location} angle operand",
        )
        if operation.results[0].type != operation.operands[0].type:
            raise ValueError(f"{location} result type must match its quantum input")
    elif isinstance(operation, BinOp):
        _validate_binop(operation, location)
    elif isinstance(operation, CompOp):
        _validate_comparison(operation, location)
    elif isinstance(operation, CondOp):
        _validate_logical(operation, location)
    elif isinstance(operation, NotOp):
        _require_arity(operation, 1, 1, location)
        _require_types(operation.operands, [BitType()], location, "operand")
        _require_types(operation.results, [BitType()], location, "result")
    elif isinstance(operation, RuntimeClassicalExpr):
        _validate_runtime_expression(operation, location)
    elif isinstance(operation, ForOperation):
        _require_operand_count(operation, 3, location)
        _require_types(
            operation.operands,
            [UIntType(), UIntType(), UIntType()],
            location,
            "operand",
        )
        if operation.loop_var_value is None:
            raise ValueError(f"{location} requires a loop_var_value")
        _require_value_type(operation.loop_var_value, UIntType(), location)
        _validate_region_args(operation.region_args, operation, location)
    elif isinstance(operation, ForItemsOperation):
        _require_operand_count(operation, 1, location)
        if not isinstance(operation.operands[0], DictValue):
            raise ValueError(f"{location} iterable operand must be a DictValue")
        if operation.key_var_values is None or operation.value_var_value is None:
            raise ValueError(f"{location} requires key and value region identities")
        _validate_region_args(operation.region_args, operation, location)
    elif isinstance(operation, WhileOperation):
        if len(operation.operands) not in {1, 2}:
            raise ValueError(f"{location} requires one or two condition operands")
        _require_types(
            operation.operands,
            [BitType()] * len(operation.operands),
            location,
            "operand",
        )
        _validate_region_args(operation.region_args, operation, location)
    elif isinstance(operation, IfOperation):
        _validate_if(operation, location)
    elif isinstance(operation, ConcreteControlledU):
        _validate_concrete_controlled(operation, location)
    elif isinstance(operation, SymbolicControlledU):
        _validate_symbolic_controlled(operation, location)
    elif isinstance(operation, InvokeOperation):
        _validate_invoke(operation, location)
    elif isinstance(operation, InverseBlockOperation):
        _validate_inverse_block(operation, location)
    elif isinstance(operation, SelectOperation):
        _validate_select(operation, location)
    else:
        raise ValueError(f"{location} has unsupported operation type")


def _validate_gate(operation: GateOperation, location: str) -> None:
    """Validate a primitive gate's fixed qubit and angle layout.

    Args:
        operation (GateOperation): Gate to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the gate type or operand/result layout is invalid.
    """
    gate_type = operation.gate_type
    if gate_type in _SINGLE_QUBIT_GATES:
        qubits, has_angle = 1, False
    elif gate_type in _SINGLE_QUBIT_ROTATIONS:
        qubits, has_angle = 1, True
    elif gate_type in _TWO_QUBIT_GATES:
        qubits, has_angle = 2, False
    elif gate_type in _TWO_QUBIT_ROTATIONS:
        qubits, has_angle = 2, True
    elif gate_type is GateOperationType.TOFFOLI:
        qubits, has_angle = 3, False
    else:
        raise ValueError(f"{location} has an unknown gate type {gate_type!r}")
    _require_arity(operation, qubits + int(has_angle), qubits, location)
    _require_types(
        operation.operands[:qubits],
        [QubitType()] * qubits,
        location,
        "operand",
    )
    if has_angle:
        _require_scalar_type(
            operation.operands[-1],
            FloatType(),
            f"{location} angle operand",
        )
    _require_types(
        operation.results,
        [QubitType()] * qubits,
        location,
        "result",
    )


def _validate_fixed_point_layout(
    num_bits: int,
    int_bits: int,
    location: str,
) -> None:
    """Validate a measured fixed-point bit layout.

    Args:
        num_bits (int): Total number of encoded bits.
        int_bits (int): Number of integer bits.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If either width is invalid or internally inconsistent.
    """
    if num_bits < 1:
        raise ValueError(f"{location} num_bits must be positive")
    if int_bits < 0 or int_bits > num_bits:
        raise ValueError(f"{location} int_bits must be between 0 and num_bits")


def _validate_binop(operation: BinOp, location: str) -> None:
    """Validate a scalar numeric arithmetic operation.

    Args:
        operation (BinOp): Arithmetic operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If arity or numeric type promotion is invalid.
    """
    _require_arity(operation, 2, 1, location)
    operand_types = [value.type for value in operation.operands]
    if not all(
        isinstance(value_type, (UIntType, FloatType)) for value_type in operand_types
    ):
        raise ValueError(f"{location} operands must be UIntType or FloatType")

    if operation.kind in {BinOpKind.FLOORDIV, BinOpKind.MOD}:
        if not all(isinstance(value_type, UIntType) for value_type in operand_types):
            raise ValueError(f"{location} requires UIntType operands")
        expected_result = UIntType()
    elif operation.kind is BinOpKind.DIV:
        expected_result = FloatType()
    else:
        result_type = operation.results[0].type
        if not isinstance(result_type, (UIntType, FloatType)):
            raise ValueError(f"{location} result must be UIntType or FloatType")
        expected_result = result_type
    _require_types(operation.results, [expected_result], location, "result")


def _validate_comparison(operation: CompOp, location: str) -> None:
    """Validate a scalar numeric comparison.

    Args:
        operation (CompOp): Comparison operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If operands are not numeric scalars or result is not a bit.
    """
    _require_arity(operation, 2, 1, location)
    if not all(
        isinstance(value.type, (UIntType, FloatType)) for value in operation.operands
    ):
        raise ValueError(f"{location} operands must be UIntType or FloatType")
    _require_types(operation.results, [BitType()], location, "result")


def _validate_logical(operation: CondOp, location: str) -> None:
    """Validate a binary bitwise logical operation.

    Args:
        operation (CondOp): Logical operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If operands or result are not bits.
    """
    _require_arity(operation, 2, 1, location)
    _require_types(operation.operands, [BitType(), BitType()], location, "operand")
    _require_types(operation.results, [BitType()], location, "result")


def _validate_runtime_expression(
    operation: RuntimeClassicalExpr,
    location: str,
) -> None:
    """Validate one lowered runtime classical expression.

    Args:
        operation (RuntimeClassicalExpr): Runtime expression to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the expression family, arity, or scalar types disagree.
    """
    if operation.kind is RuntimeOpKind.NOT:
        _require_arity(operation, 1, 1, location)
        _require_types(operation.operands, [BitType()], location, "operand")
        _require_types(operation.results, [BitType()], location, "result")
        return
    if operation.kind is RuntimeOpKind.SELECT:
        _require_arity(operation, 3, 1, location)
        _require_value_type(operation.operands[0], BitType(), location)
        branch_type = operation.operands[1].type
        if not isinstance(branch_type, (BitType, UIntType, FloatType)):
            raise ValueError(f"{location} selected values must be classical scalars")
        _require_types(
            operation.operands[1:],
            [branch_type, branch_type],
            location,
            "selected operand",
        )
        _require_types(operation.results, [branch_type], location, "result")
        return
    if operation.kind in {
        RuntimeOpKind.EQ,
        RuntimeOpKind.NEQ,
        RuntimeOpKind.LT,
        RuntimeOpKind.LE,
        RuntimeOpKind.GT,
        RuntimeOpKind.GE,
    }:
        _require_arity(operation, 2, 1, location)
        if not all(
            isinstance(value.type, (UIntType, FloatType))
            for value in operation.operands
        ):
            raise ValueError(f"{location} operands must be UIntType or FloatType")
        _require_types(operation.results, [BitType()], location, "result")
        return
    if operation.kind in {RuntimeOpKind.AND, RuntimeOpKind.OR}:
        _require_arity(operation, 2, 1, location)
        _require_types(
            operation.operands,
            [BitType(), BitType()],
            location,
            "operand",
        )
        _require_types(operation.results, [BitType()], location, "result")
        return

    _require_arity(operation, 2, 1, location)
    operand_types = [value.type for value in operation.operands]
    if not all(
        isinstance(value_type, (UIntType, FloatType)) for value_type in operand_types
    ):
        raise ValueError(f"{location} operands must be UIntType or FloatType")
    if operation.kind in {RuntimeOpKind.FLOORDIV, RuntimeOpKind.MOD}:
        if not all(isinstance(value_type, UIntType) for value_type in operand_types):
            raise ValueError(f"{location} requires UIntType operands")
        expected_result = UIntType()
    elif operation.kind is RuntimeOpKind.DIV:
        expected_result = FloatType()
    else:
        result_type = operation.results[0].type
        if not isinstance(result_type, (UIntType, FloatType)):
            raise ValueError(f"{location} result must be UIntType or FloatType")
        expected_result = result_type
    _require_types(operation.results, [expected_result], location, "result")


def _validate_array_store(
    operation: StoreArrayElementOperation,
    location: str,
) -> None:
    """Validate one classical array-element SSA rewrite.

    Args:
        operation (StoreArrayElementOperation): Store to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If array, index, stored-value, or result contracts differ.
    """
    if len(operation.operands) < 3:
        raise ValueError(f"{location} requires an array, value, and index")
    _require_result_count(operation, 1, location)
    source = operation.operands[0]
    result = operation.results[0]
    if not isinstance(source, ArrayValue) or not isinstance(result, ArrayValue):
        raise ValueError(f"{location} source and result must be ArrayValue instances")
    if source.type != operation.operands[1].type or result.type != source.type:
        raise ValueError(f"{location} array element and result types disagree")
    _require_types(
        operation.operands[2:],
        [UIntType()] * (len(operation.operands) - 2),
        location,
        "index",
    )


def _validate_dict_get(operation: DictGetItemOperation, location: str) -> None:
    """Validate a dictionary lookup's declared key arity.

    Args:
        operation (DictGetItemOperation): Lookup to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If its dictionary, key, or result layout is invalid.
    """
    if operation.key_arity < 1:
        raise ValueError(f"{location} key_arity must be positive")
    _require_arity(operation, 1 + operation.key_arity, 1, location)
    if not isinstance(operation.operands[0], DictValue):
        raise ValueError(f"{location} first operand must be a DictValue")


def _validate_if(operation: IfOperation, location: str) -> None:
    """Validate a conditional's condition and merge edges.

    Args:
        operation (IfOperation): Conditional to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the condition or any branch merge is inconsistent.
    """
    _require_operand_count(operation, 1, location)
    _require_types(operation.operands, [BitType()], location, "operand")
    try:
        merges = list(operation.iter_merges())
    except RuntimeError as exc:
        raise ValueError(f"{location} has inconsistent branch merges") from exc
    for merge in merges:
        if not (merge.true_value.type == merge.false_value.type == merge.result.type):
            raise ValueError(f"{location} branch merge types disagree")


def _validate_region_args(
    region_args: tuple[RegionArg, ...],
    operation: Operation,
    location: str,
) -> None:
    """Validate explicit loop-carried SSA region arguments.

    Args:
        region_args (tuple[RegionArg, ...]): Region records to validate.
        operation (Operation): Owning loop operation.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If result ownership, types, or identities are inconsistent.
    """
    if len(region_args) != len(operation.results):
        raise ValueError(f"{location} region-arg count must match result count")
    result_uuids = {value.uuid for value in operation.results}
    block_arg_uuids: set[str] = set()
    for region_arg in region_args:
        values = (
            region_arg.init,
            region_arg.block_arg,
            region_arg.yielded,
            region_arg.result,
        )
        if not all(value.type == region_arg.init.type for value in values):
            raise ValueError(f"{location} region-arg types disagree")
        if region_arg.result.uuid not in result_uuids:
            raise ValueError(f"{location} region result is not owned by the loop")
        if region_arg.block_arg.uuid in block_arg_uuids:
            raise ValueError(f"{location} repeats a region block-argument UUID")
        block_arg_uuids.add(region_arg.block_arg.uuid)


def _validate_concrete_controlled(
    operation: ConcreteControlledU,
    location: str,
) -> None:
    """Validate a concrete controlled-unitary layout.

    Args:
        operation (ConcreteControlledU): Operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If controls or quantum results do not match operands.
    """
    if operation.num_controls < 1 or operation.num_controls > len(operation.operands):
        raise ValueError(f"{location} has an invalid num_controls")
    if not all(
        operand.type.is_quantum()
        for operand in operation.operands[: operation.num_controls]
    ):
        raise ValueError(f"{location} controls must be quantum values")
    _validate_control_activation(
        operation.control_value,
        operation.num_controls,
        location,
    )
    _validate_controlled_results(operation, location)


def _validate_control_activation(
    control_value: object,
    num_controls: int,
    location: str,
) -> None:
    """Validate a canonical coherent-control activation value.

    Args:
        control_value (object): Candidate LSB-first activation integer or null.
        num_controls (int): Concrete width of the control register.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the value is malformed, does not fit the control width,
            or uses the non-canonical explicit all-ones representation.
    """
    if num_controls == 0:
        if control_value is not None:
            raise ValueError(f"{location} control_value requires a control qubit")
        return
    try:
        normalized = normalize_control_value(
            cast("int | None", control_value),
            num_controls,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} has an invalid control_value") from exc
    if normalized != control_value:
        raise ValueError(f"{location} has a non-canonical control_value")


def _validate_symbolic_controlled(
    operation: SymbolicControlledU,
    location: str,
) -> None:
    """Validate a symbolic controlled-unitary layout.

    Args:
        operation (SymbolicControlledU): Operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If symbolic controls, indices, or results are invalid.
    """
    _require_value_type(operation.num_controls, UIntType(), location)
    if operation.num_control_args < 1 or operation.num_control_args > len(
        operation.operands
    ):
        raise ValueError(f"{location} has an invalid num_control_args")
    if not all(
        operand.type.is_quantum()
        for operand in operation.operands[: operation.num_control_args]
    ):
        raise ValueError(f"{location} control arguments must be quantum values")
    if operation.control_indices is not None:
        _require_types(
            operation.control_indices,
            [UIntType()] * len(operation.control_indices),
            location,
            "control index",
        )
    _validate_controlled_results(operation, location)


def _validate_controlled_results(
    operation: ConcreteControlledU | SymbolicControlledU,
    location: str,
) -> None:
    """Require one quantum result for every quantum controlled operand.

    Args:
        operation (ConcreteControlledU | SymbolicControlledU): Controlled op.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the result count or result types are invalid.
    """
    quantum_operands = [
        value for value in operation.operands if value.type.is_quantum()
    ]
    if len(operation.results) != len(quantum_operands):
        raise ValueError(f"{location} results must mirror quantum operands")
    if not all(result.type.is_quantum() for result in operation.results):
        raise ValueError(f"{location} results must be quantum values")


def _validate_invoke(operation: InvokeOperation, location: str) -> None:
    """Validate a callable invocation against its explicit signature.

    Args:
        operation (InvokeOperation): Invocation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the target definition or its signature disagrees.
    """
    if operation.definition is None:
        raise ValueError(f"{location} requires a callable definition")
    if operation.definition.ref != operation.target:
        raise ValueError(f"{location} target disagrees with its definition")
    operation_kind = operation.attrs.get("kind")
    definition_kind = operation.definition.attrs.get("kind")
    if operation_kind != definition_kind:
        raise ValueError(f"{location} kind disagrees with its definition")
    control_count = 0
    if operation.transform is CallTransform.CONTROLLED:
        raw_control_count = operation.attrs.get("num_control_qubits")
        if (
            not isinstance(raw_control_count, int)
            or isinstance(raw_control_count, bool)
            or raw_control_count < 1
        ):
            raise ValueError(
                f"{location} controlled transform requires a positive integer "
                "num_control_qubits"
            )
        control_count = raw_control_count
        if control_count > len(operation.operands) or control_count > len(
            operation.results
        ):
            raise ValueError(f"{location} has fewer values than declared controls")
        if not all(
            value.type.is_quantum()
            for value in operation.operands[:control_count]
            + operation.results[:control_count]
        ):
            raise ValueError(f"{location} control inputs and results must be quantum")
        if (
            "control_value" in operation.attrs
            and operation.attrs["control_value"] is None
        ):
            raise ValueError(f"{location} has a non-canonical null control_value")
        _validate_control_activation(
            operation.attrs.get("control_value"),
            control_count,
            location,
        )
    elif "control_value" in operation.attrs:
        raise ValueError(f"{location} has control_value on a non-controlled invocation")
    signature = operation.definition.signature
    if signature is None:
        return
    signature_includes_controls = operation_kind == "oracle"
    signature_offset = 0 if signature_includes_controls else control_count
    operands = operation.operands[signature_offset:]
    results = operation.results[signature_offset:]
    if len(signature.operands) != len(operands) or len(signature.results) != len(
        results
    ):
        raise ValueError(f"{location} arity disagrees with its callable signature")
    for index, (value, hint) in enumerate(
        zip(operands, signature.operands, strict=True)
    ):
        if hint is not None and value.type != hint.type:
            raise ValueError(
                f"{location} operand {index} type disagrees with signature"
            )
    _require_types(
        results,
        [hint.type for hint in signature.results],
        location,
        "result",
    )


def _validate_inverse_block(
    operation: InverseBlockOperation,
    location: str,
) -> None:
    """Validate required semantic bodies of an inverse operation.

    Args:
        operation (InverseBlockOperation): Inverse operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If neither source nor implementation body is available.
    """
    if operation.source_block is None and operation.implementation_block is None:
        raise ValueError(f"{location} requires a source or implementation block")
    _validate_control_activation(
        operation.control_value,
        operation.num_control_qubits,
        location,
    )


def _validate_select(operation: SelectOperation, location: str) -> None:
    """Validate a SELECT operation and every case interface.

    Args:
        operation (SelectOperation): Multiplexer operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the index width, argument grouping, quantum result
            layout, or case block interfaces are inconsistent.
    """
    width = operation.num_index_qubits
    num_index_args = operation.num_index_args
    if num_index_args < 1 or num_index_args >= len(operation.operands):
        raise ValueError(f"{location} has an invalid num_index_args")

    if is_plain_int(width):
        concrete_width = cast(int, width)
        if concrete_width < 1 or num_index_args != concrete_width:
            raise ValueError(f"{location} has an invalid num_index_qubits")
        minimum_width = (len(operation.case_blocks) - 1).bit_length()
        if len(operation.case_blocks) < 2 or concrete_width < minimum_width:
            raise ValueError(f"{location} has an invalid number of case blocks")
    else:
        if not isinstance(width, Value) or isinstance(width, ArrayValue):
            raise ValueError(f"{location} has an invalid num_index_qubits")
        _require_value_type(width, UIntType(), location)
        if len(operation.case_blocks) < 2:
            raise ValueError(f"{location} has an invalid number of case blocks")

    index_operands = operation.operands[:num_index_args]
    if not all(value.type.is_quantum() for value in index_operands):
        raise ValueError(f"{location} index arguments must be quantum values")
    _require_types(
        index_operands,
        [QubitType()] * num_index_args,
        location,
        "index argument",
    )
    if is_plain_int(width):
        if any(isinstance(value, ArrayValue) for value in index_operands):
            raise ValueError(
                f"{location} concrete index operands must be scalar qubits"
            )

    target_operands = [
        value
        for value in operation.operands[num_index_args:]
        if value.type.is_quantum()
    ]
    if not target_operands:
        raise ValueError(f"{location} requires at least one quantum target")
    quantum_operands = [*index_operands, *target_operands]
    if len(operation.results) != len(quantum_operands):
        raise ValueError(f"{location} results must mirror quantum operands")
    if any(
        isinstance(operand, ArrayValue) != isinstance(result, ArrayValue)
        for operand, result in zip(
            quantum_operands,
            operation.results,
            strict=True,
        )
    ):
        raise ValueError(f"{location} results must preserve quantum argument grouping")
    _require_types(
        operation.results,
        [value.type for value in quantum_operands],
        location,
        "result",
    )

    case_inputs = operation.operands[num_index_args:]
    case_outputs = target_operands
    for case_index, case_block in enumerate(operation.case_blocks):
        case_location = f"{location} case block {case_index}"
        _require_types(
            case_block.input_values,
            [value.type for value in case_inputs],
            case_location,
            "input",
        )
        _require_types(
            case_block.output_values,
            [value.type for value in case_outputs],
            case_location,
            "output",
        )


def _validate_quantum_operand_uniqueness(
    operation: Operation,
    location: str,
) -> None:
    """Reject use of one quantum SSA value in multiple operand positions.

    Args:
        operation (Operation): Operation to inspect.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If a quantum operand UUID occurs more than once.
    """
    quantum_uuids = [
        value.uuid for value in operation.operands if value.type.is_quantum()
    ]
    if len(quantum_uuids) != len(set(quantum_uuids)):
        raise ValueError(f"{location} repeats a quantum operand UUID")


def _require_arity(
    operation: Operation,
    operands: int,
    results: int,
    location: str,
) -> None:
    """Require exact operation operand and result counts.

    Args:
        operation (Operation): Operation to inspect.
        operands (int): Required operand count.
        results (int): Required result count.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If either count differs.
    """
    _require_operand_count(operation, operands, location)
    _require_result_count(operation, results, location)


def _require_operand_count(
    operation: Operation,
    expected: int,
    location: str,
) -> None:
    """Require an exact operation operand count.

    Args:
        operation (Operation): Operation to inspect.
        expected (int): Required operand count.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the count differs.
    """
    if len(operation.operands) != expected:
        raise ValueError(
            f"{location} requires {expected} operands, got {len(operation.operands)}"
        )


def _require_result_count(
    operation: Operation,
    expected: int,
    location: str,
) -> None:
    """Require an exact operation result count.

    Args:
        operation (Operation): Operation to inspect.
        expected (int): Required result count.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the count differs.
    """
    if len(operation.results) != expected:
        raise ValueError(
            f"{location} requires {expected} results, got {len(operation.results)}"
        )


def _require_types(
    values: Iterable[ValueBase],
    expected: Iterable[ValueType],
    location: str,
    role: str,
) -> None:
    """Require positional values to have exact IR types.

    Args:
        values (Iterable[ValueBase]): Values to inspect.
        expected (Iterable[ValueType]): Expected IR type objects.
        location (str): Human-readable operation location.
        role (str): Diagnostic role such as ``operand`` or ``result``.

    Raises:
        ValueError: If a positional type differs.
    """
    for index, (value, expected_type) in enumerate(zip(values, expected, strict=True)):
        if value.type != expected_type:
            raise ValueError(
                f"{location} {role} {index} has type {value.type.label()}, "
                f"expected {expected_type.label()}"
            )


def _require_value_type(
    value: ValueBase,
    expected: ValueType,
    location: str,
) -> None:
    """Require one value to have an exact IR type.

    Args:
        value (ValueBase): Value to inspect.
        expected (ValueType): Expected IR type object.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the value's type differs.
    """
    if value.type != expected:
        raise ValueError(
            f"{location} value has type {value.type.label()}, "
            f"expected {expected.label()}"
        )


def _require_scalar_type(
    value: ValueBase,
    expected: ValueType,
    location: str,
) -> None:
    """Require a scalar Value with a specific IR type.

    Args:
        value (ValueBase): Value to inspect.
        expected (ValueType): Expected IR type object.
        location (str): Human-readable value location.

    Raises:
        ValueError: If the value is not a matching scalar.
    """
    if (
        not isinstance(value, Value)
        or isinstance(value, ArrayValue)
        or value.type != expected
    ):
        raise ValueError(f"{location} requires a scalar {expected.label()}")


def _require_array_type(
    value: ValueBase,
    expected: ValueType,
    location: str,
) -> None:
    """Require an ArrayValue with a specific element type.

    Args:
        value (ValueBase): Value to inspect.
        expected (ValueType): Expected element IR type.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the value is not a matching array.
    """
    if not isinstance(value, ArrayValue) or value.type != expected:
        raise ValueError(f"{location} requires an ArrayValue[{expected.label()}]")


def _require_unique_values(values: Iterable[ValueBase], location: str) -> None:
    """Require a value sequence to contain no duplicate UUIDs.

    Args:
        values (Iterable[ValueBase]): Values to inspect.
        location (str): Human-readable sequence location.

    Raises:
        ValueError: If a UUID occurs more than once.
    """
    uuids = [value.uuid for value in values]
    if len(uuids) != len(set(uuids)):
        raise ValueError(f"{location} contain duplicate UUIDs")
