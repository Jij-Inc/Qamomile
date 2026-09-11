"""Resolve runtime parameter dimensions without binding their element values."""

from __future__ import annotations

import dataclasses
import math
import numbers
import operator
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    UnaryMathOp,
    UnaryMathOpKind,
)
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation, HasNestedOps
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.operation.slice_array import SliceArrayOperation
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
)
from qamomile.circuit.ir.value_mapping import ValueSubstitutor
from qamomile.circuit.transpiler import (
    PreparedModule,
    pair_block_operands,
    prepare_module,
)

_DIMENSION_BINARY_OPERATORS: Mapping[
    BinOpKind, Callable[[int | float, int | float], int | float]
] = {
    BinOpKind.ADD: operator.add,
    BinOpKind.SUB: operator.sub,
    BinOpKind.MUL: operator.mul,
    BinOpKind.DIV: operator.truediv,
    BinOpKind.FLOORDIV: operator.floordiv,
    BinOpKind.MOD: operator.mod,
    BinOpKind.POW: operator.pow,
    BinOpKind.MIN: min,
}


def _value_references(
    values: Sequence[ValueBase], *, include_structure: bool
) -> list[ValueBase]:
    """Visit value dependencies while distinguishing structural metadata.

    Args:
        values (Sequence[ValueBase]): Root values to inspect.
        include_structure (bool): Include dimensions, indices, and slice bounds.

    Returns:
        list[ValueBase]: Each reachable value object once.
    """
    pending = list(values)
    seen: set[int] = set()
    result = []
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        result.append(value)
        if isinstance(value, TupleValue):
            pending.extend(value.elements)
        elif isinstance(value, DictValue):
            pending.extend(item for pair in value.entries for item in pair)
        elif isinstance(value, Value):
            if value.parent_array is not None:
                pending.append(value.parent_array)
            if include_structure:
                pending.extend(value.element_indices)
            if isinstance(value, ArrayValue):
                if value.slice_of is not None:
                    pending.append(value.slice_of)
                if include_structure:
                    pending.extend(value.shape)
                    pending.extend(
                        bound
                        for bound in (value.slice_start, value.slice_step)
                        if bound is not None
                    )
    return result


def _structural_inputs(operation: Operation) -> list[Value]:
    """Identify explicit scalar operands that determine target structure.

    Args:
        operation (Operation): Operation whose range or slice bounds are needed.

    Returns:
        list[Value]: Range bounds or slice bounds; empty for other operations.
    """
    if isinstance(operation, ForOperation):
        return operation.operands[:3]
    if isinstance(operation, SliceArrayOperation):
        return operation.operands[1:]
    return []


def _expression_dependencies(
    roots: set[str], expressions: Mapping[str, BinOp | UnaryMathOp]
) -> set[str]:
    """Trace scalar dependencies backward through pure arithmetic producers.

    Args:
        roots (set[str]): UUIDs whose producer dependencies are required.
        expressions (Mapping[str, BinOp | UnaryMathOp]): Result-to-producer map.

    Returns:
        set[str]: Root UUIDs and all arithmetic operand dependencies.
    """
    required = set(roots)
    pending = list(roots)
    while pending:
        expression = expressions.get(pending.pop())
        if expression is None:
            continue
        for operand in expression.operands:
            if operand.uuid not in required:
                required.add(operand.uuid)
                pending.append(operand.uuid)
    return required


class _ShapeSubstitutor(ValueSubstitutor):
    """Specialize array dimensions without changing native scalar arithmetic.

    Args:
        dimensions (Mapping[str, Value]): Mathematical dimension constants.
        expressions (set[str]): Arithmetic result UUIDs.
        removable (set[str]): Results used only by resolved structural expressions.
    """

    def __init__(
        self,
        dimensions: Mapping[str, Value],
        expressions: set[str],
        removable: set[str],
    ) -> None:
        """Keep scalar substitutions only where no native expression consumes them.

        Args:
            dimensions (Mapping[str, Value]): Mathematical dimension constants.
            expressions (set[str]): Arithmetic result UUIDs.
            removable (set[str]): Fully resolved results with no native consumers.
        """
        super().__init__(
            {
                uuid: value
                for uuid, value in dimensions.items()
                if uuid not in expressions or uuid in removable
            }
        )
        self._dimensions = dimensions

    def substitute_value(self, value: ValueBase) -> ValueBase:
        """Rewrite dimension metadata separately from scalar value uses.

        Args:
            value (ValueBase): Scalar or structural value to rewrite.

        Returns:
            ValueBase: Rewritten value preserving native arithmetic results.
        """
        result = super().substitute_value(value)
        if isinstance(result, Value) and result.element_indices:
            indices = tuple(
                self._dimensions.get(index.uuid, index)
                for index in result.element_indices
            )
            if indices != result.element_indices:
                result = dataclasses.replace(result, element_indices=indices)
        if isinstance(result, ArrayValue):
            shape = tuple(self._dimensions.get(dim.uuid, dim) for dim in result.shape)
            start = (
                self._dimensions.get(result.slice_start.uuid, result.slice_start)
                if result.slice_start is not None
                else None
            )
            step = (
                self._dimensions.get(result.slice_step.uuid, result.slice_step)
                if result.slice_step is not None
                else None
            )
            if (
                shape != result.shape
                or start != result.slice_start
                or step != result.slice_step
            ):
                result = dataclasses.replace(
                    result, shape=shape, slice_start=start, slice_step=step
                )
        return result

    def substitute_operation(self, operation: Operation) -> Operation:
        """Resolve explicit structural operands independently of native scalar uses.

        Args:
            operation (Operation): Operation whose values and bounds are rewritten.

        Returns:
            Operation: Rewritten operation with concrete range or slice bounds.
        """
        result = super().substitute_operation(operation)
        bounds = {value.uuid for value in _structural_inputs(result)}
        if bounds:
            operands = [
                self._dimensions.get(value.uuid, value)
                if value.uuid in bounds
                else value
                for value in result.operands
            ]
            if operands != result.operands:
                result = dataclasses.replace(result, operands=operands)
        return result


def _dimension_pairs(
    left: ValueBase, right: ValueBase, symbol: str
) -> list[tuple[Value, Value, str]]:
    """Collect matching array dimensions across one callable ABI boundary.

    Args:
        left (ValueBase): Callee input or output value.
        right (ValueBase): Corresponding caller value.
        symbol (str): Callable name for conflicting-shape diagnostics.

    Returns:
        list[tuple[Value, Value, str]]: Dimension equality constraints.

    Raises:
        ValueError: If corresponding array or structural ranks disagree.
    """
    if isinstance(left, ArrayValue) and isinstance(right, ArrayValue):
        if len(left.shape) != len(right.shape):
            raise ValueError(f"Conflicting HUGR callable array ranks for {symbol!r}")
        return [(a, b, symbol) for a, b in zip(left.shape, right.shape, strict=True)]
    if isinstance(left, TupleValue) and isinstance(right, TupleValue):
        if len(left.elements) != len(right.elements):
            raise ValueError(f"Conflicting HUGR callable tuple shapes for {symbol!r}")
        return [
            pair
            for a, b in zip(left.elements, right.elements, strict=True)
            for pair in _dimension_pairs(a, b, symbol)
        ]
    if isinstance(left, DictValue) and isinstance(right, DictValue):
        if len(left.entries) != len(right.entries):
            raise ValueError(
                f"Conflicting HUGR callable dictionary shapes for {symbol!r}"
            )
        return [
            pair
            for a, b in zip(left.entries, right.entries, strict=True)
            for x, y in zip(a, b, strict=True)
            for pair in _dimension_pairs(x, y, symbol)
        ]
    return []


def _operation_blocks(operation: Operation) -> list[Block]:
    """Collect independent bodies owned by a transform operation.

    Args:
        operation (Operation): Operation whose separate block namespaces are needed.

    Returns:
        list[Block]: SELECT cases, controlled body, or available inverse bodies.
    """
    if isinstance(operation, SelectOperation):
        return operation.case_blocks
    if isinstance(operation, ControlledUOperation):
        return [operation.block] if operation.block is not None else []
    if isinstance(operation, InverseBlockOperation):
        return [
            block
            for block in (operation.source_block, operation.implementation_block)
            if block is not None
        ]
    return []


def _transform_shape_pairs(
    operation: Operation, body: Block
) -> list[tuple[Value, Value, str]]:
    """Connect an operation-owned body's dimensions to its caller arguments.

    Args:
        operation (Operation): SELECT, controlled, or inverse block operation.
        body (Block): Owned body whose callable ABI must match the operation.

    Returns:
        list[tuple[Value, Value, str]]: Input and output dimension equalities.

    Raises:
        ValueError: If the transform's input or output structure is incompatible.
    """
    if isinstance(operation, SelectOperation):
        prefix = operation.num_index_args
    elif isinstance(operation, ControlledUOperation):
        prefix = len(operation.control_operands)
    elif isinstance(operation, InverseBlockOperation):
        prefix = operation.num_control_qubits
        if body is operation.source_block:
            body = dataclasses.replace(
                body,
                input_values=[
                    value for value in body.output_values if value.type.is_quantum()
                ]
                + [value for value in body.input_values if not value.type.is_quantum()],
                output_values=[
                    value for value in body.input_values if value.type.is_quantum()
                ],
            )
    else:
        return []
    pairs = pair_block_operands(body, operation.operands[prefix:])
    pairs.extend(zip(body.output_values, operation.results[prefix:], strict=True))
    return [
        constraint
        for formal, actual in pairs
        for constraint in _dimension_pairs(formal, actual, body.name)
    ]


def _callable_shape_graph(
    program: PreparedModule,
) -> tuple[list[Block], list[CallableDef], list[tuple[Value, Value, str]]]:
    """Collect owned callable bodies and their bidirectional shape constraints.

    Args:
        program (PreparedModule): Target-owned prepared program.

    Returns:
        tuple[list[Block], list[CallableDef], list[tuple[Value, Value, str]]]:
        Unique callable and transform bodies, definitions, and dimension constraints.

    Raises:
        ValueError: If a call has incompatible input or output structure.
    """
    blocks: dict[int, Block] = {}
    definitions: dict[int, CallableDef] = {}
    constraints: list[tuple[Value, Value, str]] = []

    def visit_definition(definition: CallableDef) -> None:
        """Visit each owned definition and all its available bodies once.

        Args:
            definition (CallableDef): Reachable callable definition.

        Raises:
            ValueError: If a nested call has incompatible ABI structure.
        """
        if id(definition) in definitions:
            return
        definitions[id(definition)] = definition
        if definition.body is not None:
            visit_block(definition.body)
        for implementation in definition.implementations:
            if implementation.body is not None:
                visit_block(implementation.body)

    def visit_operations(operations: Sequence[Operation]) -> None:
        """Collect constraints from calls at every structured-control depth.

        Args:
            operations (Sequence[Operation]): Operations in one region.

        Raises:
            ValueError: If a call has incompatible ABI structure.
        """
        for operation in operations:
            if (
                isinstance(operation, InvokeOperation)
                and operation.definition is not None
            ):
                definition = operation.definition
                visit_definition(definition)
                if (
                    operation.transform is CallTransform.DIRECT
                    and definition.body is not None
                ):
                    body = definition.body
                    symbol = f"{definition.ref.namespace}.{definition.ref.name}"
                    for formal, actual in pair_block_operands(body, operation.operands):
                        constraints.extend(_dimension_pairs(formal, actual, symbol))
                    for formal, actual in zip(
                        body.output_values, operation.results, strict=True
                    ):
                        constraints.extend(_dimension_pairs(formal, actual, symbol))
            for block in _operation_blocks(operation):
                visit_block(block)
                constraints.extend(_transform_shape_pairs(operation, block))
            if isinstance(operation, HasNestedOps):
                for region in operation.nested_regions():
                    visit_operations(region.operations)

    def visit_block(block: Block) -> None:
        """Visit a shared body without duplicating its definition identity.

        Args:
            block (Block): Reachable semantic body.

        Raises:
            ValueError: If a contained call has incompatible ABI structure.
        """
        if id(block) in blocks:
            return
        blocks[id(block)] = block
        visit_operations(block.operations)

    visit_block(program.entrypoint)
    for variants in program.definition_variants.values():
        for definition in variants:
            visit_definition(definition)
    return list(blocks.values()), list(definitions.values()), constraints


def _propagate_dimensions(
    substitutions: dict[str, Value],
    constraints: Sequence[tuple[Value, Value, str]],
    blocks: Sequence[Block],
) -> tuple[set[str], set[str]]:
    """Resolve dimension expressions and callable equalities to a fixed point.

    Args:
        substitutions (dict[str, Value]): Dimension-only replacements to extend.
        constraints (Sequence[tuple[Value, Value, str]]): Dimension equalities.
        blocks (Sequence[Block]): Owned bodies containing dimension expressions.

    Returns:
        tuple[set[str], set[str]]: Removable structural-only results and all
            arithmetic result UUIDs that must preserve native scalar uses.

    Raises:
        ValueError: If a callable is used with incompatible fixed dimensions
            or a dimension-derived expression is invalid or inconsistent.
    """
    pending = [operation for block in blocks for operation in block.operations]
    operations = []
    expressions: dict[str, BinOp | UnaryMathOp] = {}
    values: list[ValueBase] = [
        value
        for block in blocks
        for value in (
            *block.input_values,
            *block.output_values,
            *block.parameters.values(),
        )
    ]
    native_values: list[ValueBase] = [
        value for block in blocks for value in block.output_values
    ]
    structural: set[str] = set()
    while pending:
        operation = pending.pop()
        operations.append(operation)
        values.extend(operation.all_input_values())
        values.extend(operation.results)
        structural.update(value.uuid for value in _structural_inputs(operation))
        if isinstance(operation, (BinOp, UnaryMathOp)):
            expressions[operation.results[0].uuid] = operation
        if isinstance(operation, HasNestedOps):
            for region in operation.nested_regions():
                pending.extend(region.operations)
                values.extend((*region.block_args, *region.captures, *region.yields))
                native_values.extend(region.yields)
    for value in _value_references(values, include_structure=True):
        if isinstance(value, ArrayValue):
            structural.update(dimension.uuid for dimension in value.shape)
            structural.update(
                bound.uuid
                for bound in (value.slice_start, value.slice_step)
                if bound is not None
            )
        if isinstance(value, Value):
            structural.update(index.uuid for index in value.element_indices)
    structural = _expression_dependencies(structural, expressions)
    folded: set[str] = set()
    changed = True
    while changed:
        changed = False
        for left, right, symbol in constraints:
            a = substitutions.get(left.uuid, left)
            b = substitutions.get(right.uuid, right)
            if a.is_constant() and b.is_constant():
                if a.get_const() != b.get_const():
                    raise ValueError(
                        f"Conflicting HUGR callable shapes for {symbol!r}: "
                        f"dimensions {a.get_const()} and {b.get_const()} cannot share one function"
                    )
            elif a.is_constant():
                substitutions[right.uuid] = right.with_const(int(a.get_const()))
                changed = True
            elif b.is_constant():
                substitutions[left.uuid] = left.with_const(int(b.get_const()))
                changed = True
        for expression in reversed(tuple(expressions.values())):
            result = expression.results[0]
            if result.uuid in folded or result.uuid not in structural:
                continue
            constant = _dimension_expression(expression, substitutions)
            if constant is None:
                continue
            previous = substitutions.get(result.uuid, result)
            if previous.is_constant() and previous.get_const() != constant:
                raise ValueError("Conflicting HUGR computed array dimensions")
            substitutions[result.uuid] = result.with_const(constant)
            folded.add(result.uuid)
            changed = True
    for operation in operations:
        if (
            isinstance(operation, (BinOp, UnaryMathOp))
            and operation.results[0].uuid in folded
        ):
            continue
        structural_inputs = {value.uuid for value in _structural_inputs(operation)}
        native_values.extend(
            value
            for value in operation.all_input_values()
            if value.uuid not in structural_inputs
        )
    native = _expression_dependencies(
        {
            value.uuid
            for value in _value_references(native_values, include_structure=False)
        },
        expressions,
    )
    return folded - native, set(expressions)


def _dimension_expression(
    operation: BinOp | UnaryMathOp, substitutions: Mapping[str, Value]
) -> int | float | None:
    """Evaluate scalar arithmetic rooted in declared dimensions only.

    Every operand must be a literal or a resolved dimension-derived scalar.
    Runtime arguments and array elements are never read or rebound, even when
    the operation also consumes a known dimension.

    Args:
        operation (BinOp | UnaryMathOp): Pure scalar expression to inspect.
        substitutions (Mapping[str, Value]): Known dimensions and derived scalars.

    Returns:
        int | float | None: Concrete result, or None if any input is unresolved
            or the expression does not depend on a declared dimension.

    Raises:
        ValueError: If a resolved expression has invalid arithmetic or domain.
    """
    if not any(value.uuid in substitutions for value in operation.operands):
        return None
    values: list[int | float] = []
    for operand in operation.operands:
        value = substitutions.get(operand.uuid, operand)
        constant = value.get_const()
        if (
            isinstance(operand, ArrayValue)
            or operand.parent_array is not None
            or not value.is_constant()
            or isinstance(constant, bool)
            or not isinstance(constant, (int, float))
        ):
            return None
        values.append(constant)
    try:
        if isinstance(operation, BinOp):
            if operation.kind is None:
                raise ValueError("Dimension expression has no arithmetic kind")
            result = _DIMENSION_BINARY_OPERATORS[operation.kind](values[0], values[1])
        elif operation.kind is UnaryMathOpKind.LOG2:
            result = math.log2(values[0])
        elif operation.kind is UnaryMathOpKind.CEIL:
            result = math.ceil(values[0])
            if result < 0:
                raise ValueError("Unsigned ceiling must be non-negative")
        else:
            raise ValueError("Unsupported dimension expression")
        if not isinstance(result, (int, float)) or (
            isinstance(result, float) and not math.isfinite(result)
        ):
            raise ValueError("Dimension expression must produce a finite real scalar")
        return result
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid HUGR dimension-derived arithmetic") from error


def _rewrite_operations(
    operations: Sequence[Operation],
    substitutor: ValueSubstitutor,
    folded: set[str],
    blocks: Mapping[int, Block],
) -> list[Operation]:
    """Replace shape references throughout nested regions and operation-owned bodies.

    Args:
        operations (Sequence[Operation]): Operations to copy and rewrite.
        substitutor (ValueSubstitutor): Dimension-only substitution mapping.
        folded (set[str]): Arithmetic results fully replaced by constants.
        blocks (Mapping[int, Block]): Rewritten owned bodies indexed by original identity.

    Returns:
        list[Operation]: Operations with rewritten operands, regions, and body references.

    Raises:
        KeyError: If an operation-owned body is missing from the rewritten graph.
    """
    result = []
    for operation in operations:
        if (
            isinstance(operation, (BinOp, UnaryMathOp))
            and operation.results[0].uuid in folded
        ):
            continue
        rewritten = substitutor.substitute_operation(operation)
        if isinstance(rewritten, HasNestedOps):
            regions = tuple(
                dataclasses.replace(
                    region,
                    operations=tuple(
                        _rewrite_operations(
                            region.operations, substitutor, folded, blocks
                        )
                    ),
                    block_args=tuple(
                        substitutor.substitute_value(value)
                        for value in region.block_args
                    ),
                    captures=tuple(
                        substitutor.substitute_value(value) for value in region.captures
                    ),
                    yields=tuple(
                        substitutor.substitute_value(value) for value in region.yields
                    ),
                )
                for region in rewritten.nested_regions()
            )
            rewritten = rewritten.rebuild_regions(regions)
        if isinstance(rewritten, SelectOperation):
            rewritten = dataclasses.replace(
                rewritten,
                case_blocks=[blocks[id(block)] for block in rewritten.case_blocks],
            )
        elif isinstance(rewritten, ControlledUOperation):
            if rewritten.block is not None:
                rewritten = dataclasses.replace(
                    rewritten, block=blocks[id(rewritten.block)]
                )
        elif isinstance(rewritten, InverseBlockOperation):
            rewritten = dataclasses.replace(
                rewritten,
                source_block=(
                    blocks[id(rewritten.source_block)]
                    if rewritten.source_block is not None
                    else None
                ),
                implementation_block=(
                    blocks[id(rewritten.implementation_block)]
                    if rewritten.implementation_block is not None
                    else None
                ),
            )
        result.append(rewritten)
    return result


def resolve_parameter_shapes(
    program: PreparedModule,
    parameter_shapes: Mapping[str, Sequence[int]] | None,
) -> PreparedModule:
    """Specify fixed runtime array dimensions while retaining native inputs.

    Only dimension metadata is specialized. Array element values remain
    unbound, including array elements used as runtime gate parameters. Fixed
    dimensions propagate through direct callable and transform inputs and outputs
    at every nesting level and through fully resolved scalar arithmetic without changing
    callable identities. Array element values never participate in folding.

    Args:
        program (PreparedModule): Prepared semantic program.
        parameter_shapes (Mapping[str, Sequence[int]] | None): Concrete shapes
            for top-level runtime array parameters; omitted shapes are unchanged.

    Returns:
        PreparedModule: Independent prepared program with resolved dimensions.

    Raises:
        TypeError: If a named input is not an array or a dimension is not integral.
        ValueError: If a name is unknown or compile-time bound, a shape has the
            wrong rank, or a dimension is negative or conflicts with a fixed size
            at the entrypoint or a shared callable.
            Also if dimension-derived arithmetic is invalid or inconsistent.
    """
    if not parameter_shapes:
        return program
    substitutions = {}
    for name, shape in parameter_shapes.items():
        if name not in program.abi.public_inputs:
            raise ValueError(f"Unknown HUGR runtime array parameter {name!r}")
        if name in program.bindings:
            raise ValueError(
                f"HUGR parameter_shapes requires a runtime parameter: {name!r}"
            )
        value = program.abi.public_inputs[name]
        if not isinstance(value, ArrayValue):
            raise TypeError(f"HUGR parameter_shapes input {name!r} is not an array")
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
            raise TypeError(
                f"HUGR parameter shape {name!r} must be an integer sequence"
            )
        if len(shape) != len(value.shape):
            raise ValueError(
                f"HUGR parameter shape {name!r} requires rank {len(value.shape)}, "
                f"got {len(shape)}"
            )
        for dimension, size in zip(value.shape, shape, strict=True):
            if isinstance(size, bool) or not isinstance(
                cast(object, size), numbers.Integral
            ):
                raise TypeError(
                    f"HUGR parameter shape {name!r} requires integer dimensions"
                )
            if size < 0:
                raise ValueError(f"HUGR parameter shape {name!r} must be non-negative")
            if dimension.is_constant() and int(dimension.get_const()) != int(size):
                raise ValueError(
                    f"HUGR parameter shape {name!r} conflicts with its fixed dimension"
                )
            previous = substitutions.get(dimension.uuid)
            if previous is not None and previous.get_const() != int(size):
                raise ValueError(
                    "HUGR parameter shapes assign conflicting shared dimensions"
                )
            substitutions[dimension.uuid] = dimension.with_const(int(size))
    owned = program.owned_snapshot()
    blocks, definitions, constraints = _callable_shape_graph(owned)
    folded, expressions = _propagate_dimensions(substitutions, constraints, blocks)
    substitutor = _ShapeSubstitutor(substitutions, expressions, folded)
    rewritten = {
        id(block): dataclasses.replace(
            block,
            input_values=[
                cast(ValueLike, substitutor.substitute_value(value))
                for value in block.input_values
            ],
            output_values=[
                cast(ValueLike, substitutor.substitute_value(value))
                for value in block.output_values
            ],
            parameters={
                name: cast(Value, substitutor.substitute_value(value))
                for name, value in block.parameters.items()
            },
        )
        for block in blocks
    }
    for block in blocks:
        rewritten[id(block)].operations = _rewrite_operations(
            block.operations, substitutor, folded, rewritten
        )
    for definition in definitions:
        if definition.body is not None:
            definition.body = rewritten[id(definition.body)]
        for implementation in definition.implementations:
            if implementation.body is not None:
                implementation.body = rewritten[id(implementation.body)]
    return prepare_module(rewritten[id(owned.entrypoint)], owned.bindings)
