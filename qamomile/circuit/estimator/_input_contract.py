"""Inspect callable input contracts and concrete array shapes."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import sympy as sp

from qamomile.circuit._array_shape import (
    _ARRAY_PROTOCOL_ERRORS,
    _rectangular_array_shape,
)
from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
    input_shape_dimension_aliases,
)
from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.estimator._resource_expressions import _safe_simplify
from qamomile.circuit.ir._resource_contract import quantum_operand_widths
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.dataflow import walk_operations
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel


def _root_callable_resource_attrs(
    kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
) -> Mapping[str, Any]:
    """Return resource metadata preserved on a root qkernel-like object.

    A plain :class:`Block` intentionally has no callable identity or attrs.
    Serialized qkernels and stdlib descriptors preserve their definition attrs
    on ``_callable_attrs_override``, which lets root estimation enforce the
    same contract as nested invocation operations without importing frontend
    helpers into the estimator core.

    Args:
        kernel (QKernel[Any, Any] | Block | Sequence[Operation]): Estimator
            input before it is coerced to IR.

    Returns:
        Mapping[str, Any]: Preserved callable attrs, or an empty mapping.
    """
    attrs = getattr(kernel, "_callable_attrs_override", None)
    return attrs if isinstance(attrs, Mapping) else {}


def _root_callable_shape_inputs(
    attrs: Mapping[str, Any],
    block_or_ops: Block | Sequence[Operation],
    explicit_inputs: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, int]:
    """Infer one-dimensional root quantum-port widths from exact metadata.

    A callable resource contract describes total scalar widths. That value
    uniquely determines the shape of a one-dimensional quantum vector, but it
    cannot safely choose dimensions for a higher-rank array. Explicit port or
    dimension inputs always take precedence so a conflicting value reaches
    the normal contract validator and produces a useful error.

    Args:
        attrs (Mapping[str, Any]): Root callable definition attributes.
        block_or_ops (Block | Sequence[Operation]): Coerced estimator input.
        explicit_inputs (Mapping[str, Any]): User-supplied specialization
            inputs before build/estimation partitioning.
        source (str): Callable name used in malformed-contract diagnostics.

    Returns:
        dict[str, int]: Inferred one-dimensional quantum-port widths keyed by
            their public argument names.

    Raises:
        ValueError: If present resource metadata is malformed.
    """
    if not isinstance(block_or_ops, Block):
        return {}
    quantum_ports = [
        (name, value)
        for name, value in zip(
            block_or_ops.label_args,
            block_or_ops.input_values,
        )
        if isinstance(value, Value) and value.type.is_quantum()
    ]
    shape_aliases = input_shape_dimension_aliases(block_or_ops)
    inferred: dict[str, int] = {}
    for entry in quantum_operand_widths(attrs, source=source):
        if entry.index >= len(quantum_ports):
            # The constraint builder reports the complete call-shape
            # diagnostic before inference runs.
            continue
        name, value = quantum_ports[entry.index]
        if not isinstance(value, ArrayValue) or len(value.shape) != 1:
            continue
        if value.shape[0].is_constant():
            continue
        dimension_name = shape_aliases.get(value.shape[0].uuid)
        if name in explicit_inputs or (
            dimension_name is not None and dimension_name in explicit_inputs
        ):
            continue
        inferred[name] = entry.width
    return inferred


def _declared_ir_inputs(
    kernel: Any,
) -> dict[str, ValueBase]:
    """Collect public input values recoverable from an IR estimator target.

    A raw ``Block`` is already traced, so supplied classical/object values and
    quantum-array widths can be used directly by the abstract interpreter
    without constructing a larger circuit. Scalar quantum ports remain actual
    wires rather than estimator inputs. A raw operation sequence has no
    explicit argument manifest and therefore falls back to parameter metadata
    carried by operation operands. Unrecognized names are deliberately omitted
    so downstream strict input validation still diagnoses typos.

    Args:
        kernel (Any): Block or raw operation sequence being estimated.

    Returns:
        dict[str, ValueBase]: Public names mapped to their IR values.
    """
    inputs: dict[str, ValueBase] = {}
    if isinstance(kernel, Block):
        inputs.update(
            {
                name: value
                for name, value in zip(kernel.label_args, kernel.input_values)
                if not value.type.is_quantum() or isinstance(value, ArrayValue)
            }
        )
        inputs.update(kernel.parameters)
        operations = kernel.operations
    elif isinstance(kernel, Sequence):
        operations = kernel
    else:
        return {}

    for operation in walk_operations(operations):
        for operand in operation.all_input_values():
            parameter_name = operand.parameter_name()
            if parameter_name is not None and (
                not operand.type.is_quantum() or isinstance(operand, ArrayValue)
            ):
                inputs.setdefault(parameter_name, operand)
    return inputs


def _contract_names(
    block_or_ops: Block | Sequence[Operation],
) -> frozenset[str] | None:
    """Return the declared classical argument names of a built block, if any.

    Only classical parameters (``param_slots``) remain after quantum-array
    inputs have been translated to their dimension symbols. A scalar quantum
    port has fixed width one and is therefore not a valid specialization name.

    Args:
        block_or_ops (Block | Sequence[Operation]): Coerced estimator input.

    Returns:
        frozenset[str] | None: Classical parameter names when the input is a
        ``Block``; ``None`` for a raw operation sequence (which carries no
        user-facing input contract, so substitution stays strict there).
    """
    if not isinstance(block_or_ops, Block):
        return None
    return frozenset(slot.name for slot in block_or_ops.param_slots)


def _scalar_input_types(
    block_or_ops: Block | Sequence[Operation],
) -> dict[str, Any]:
    """Return recoverable scalar qkernel input types by public name.

    Args:
        block_or_ops (Block | Sequence[Operation]): Coerced estimator input.

    Returns:
        dict[str, Any]: Scalar classical input types, or an empty mapping when
            the input has no root interface manifest.
    """
    if not isinstance(block_or_ops, Block):
        return {}
    input_types = {
        slot.name: slot.type for slot in block_or_ops.param_slots if slot.ndim == 0
    }
    for name, value in zip(block_or_ops.label_args, block_or_ops.input_values):
        if not isinstance(value, ArrayValue) and not value.type.is_quantum():
            input_types.setdefault(name, value.type)
    for name, value in block_or_ops.parameters.items():
        input_types.setdefault(name, value.type)
    return input_types


def _expand_array_shape_inputs(
    block_or_ops: Block | Sequence[Operation],
    inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], frozenset[str]]:
    """Expand supplied array shapes into their IR dimension symbols.

    A one-dimensional quantum register also accepts an integer width, avoiding
    the need to construct a dummy Python sequence solely for resource
    estimation. Once expanded, a quantum port name is removed because its
    element values are not estimator inputs; classical arrays retain their
    original value for branch and parameter specialization.

    Args:
        block_or_ops (Block | Sequence[Operation]): Coerced estimator input.
        inputs (Mapping[str, Any]): User-provided estimation inputs.

    Returns:
        tuple[dict[str, Any], frozenset[str]]: Inputs plus one concrete value
            for each matching array dimension symbol, and input names whose
            shape was consumed by the estimate.

    Raises:
        ValueError: If a supplied concrete array rank differs from the qkernel
            input rank.
    """
    expanded = dict(inputs)
    consumed: set[str] = set()
    if not isinstance(block_or_ops, Block):
        return expanded, frozenset()
    shape_aliases = input_shape_dimension_aliases(block_or_ops)
    dimensions_by_uuid = {
        dimension.uuid: dimension
        for value in block_or_ops.input_values
        if isinstance(value, ArrayValue)
        for dimension in value.shape
    }
    for dimension_uuid, dimension_name in shape_aliases.items():
        if dimension_name in inputs:
            dimension_input = inputs[dimension_name]
            if isinstance(dimension_input, bool):
                raise ValueError(
                    f"array dimension input '{dimension_name}' requires a numeric "
                    "integer or symbolic expression; bool is not a dimension."
                )
            if isinstance(dimension_input, (str, bytes)):
                raise TypeError(
                    f"array dimension input '{dimension_name}' requires a numeric "
                    f"integer or explicit SymPy expression, got "
                    f"{type(dimension_input).__name__} ({dimension_input!r})."
                )
        dimension = dimensions_by_uuid[dimension_uuid]
        if dimension_name in inputs and dimension.is_constant():
            expected = dimension.get_const()
            supplied = inputs[dimension_name]
            if not _input_values_equal(supplied, expected):
                raise ValueError(
                    f"array dimension '{dimension_name}' is fixed at {expected}, "
                    f"but inputs specify {supplied!r}."
                )
            expanded.pop(dimension_name, None)
            consumed.add(dimension_name)
    for name, ir_value in zip(block_or_ops.label_args, block_or_ops.input_values):
        if name not in inputs or not isinstance(ir_value, ArrayValue):
            continue
        supplied = inputs[name]
        is_quantum_array = ir_value.type.is_quantum()
        if is_quantum_array and len(ir_value.shape) == 1 and isinstance(supplied, bool):
            raise ValueError(
                f"quantum array input '{name}' requires an integer width or "
                "an array-like value; bool is not a width."
            )
        if (
            is_quantum_array
            and len(ir_value.shape) == 1
            and isinstance(supplied, numbers.Integral)
        ):
            shape = (int(supplied),)
        else:
            shape = _concrete_input_shape(supplied)
            if is_quantum_array and not shape and not _is_array_like_input(supplied):
                raise ValueError(
                    f"quantum array input '{name}' requires an integer width "
                    "or an array-like value; got "
                    f"{type(supplied).__name__} ({supplied!r})."
                )
        if len(shape) != len(ir_value.shape):
            raise ValueError(
                f"array input '{name}' has rank {len(shape)}, but the qkernel "
                f"declares rank {len(ir_value.shape)}."
            )
        consumed.add(name)
        if ir_value.type.is_quantum():
            expanded.pop(name, None)
        for dimension, size in zip(ir_value.shape, shape):
            dimension_name = shape_aliases.get(dimension.uuid)
            if dimension_name:
                if dimension.is_constant():
                    expected = dimension.get_const()
                    if not _input_values_equal(size, expected):
                        raise ValueError(
                            f"array input '{name}' dimension '{dimension_name}' "
                            f"is fixed at {expected}, but the supplied shape "
                            f"has size {size}."
                        )
                    continue
                has_existing = dimension_name in inputs
                existing = inputs.get(dimension_name)
                if has_existing and not _input_values_equal(existing, size):
                    raise ValueError(
                        f"array input '{name}' implies {dimension_name}={size}, "
                        f"but inputs also specify {dimension_name}={existing!r}."
                    )
                expanded[dimension_name] = size
                consumed.add(dimension_name)
    return expanded, frozenset(consumed)


def _input_values_equal(left: Any, right: Any) -> bool:
    """Return whether two scalar estimation inputs are provably equal.

    Args:
        left (Any): Explicit input value.
        right (Any): Shape-derived input value.

    Returns:
        bool: Whether SymPy proves the scalar values equal.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)):
        return False
    if not isinstance(left, (numbers.Number, sp.Basic)) or not isinstance(
        right,
        (numbers.Number, sp.Basic),
    ):
        return False
    try:
        normalized_left = cast(ResourceExpr, sp.sympify(left))
        normalized_right = cast(ResourceExpr, sp.sympify(right))
        difference = normalized_left - normalized_right
    except (TypeError, ValueError, sp.SympifyError):
        return False
    return _safe_simplify(cast(ResourceExpr, difference)) == _ZERO


def _is_array_like_input(value: Any) -> bool:
    """Return whether an estimation input exposes at least one array axis.

    Args:
        value (Any): Candidate array payload.

    Returns:
        bool: Whether ``value`` is a non-scalar array or sequence.
    """
    try:
        shape = getattr(value, "shape", None)
    except _ARRAY_PROTOCOL_ERRORS:
        return False
    if shape is not None:
        try:
            return len(shape) > 0
        except _ARRAY_PROTOCOL_ERRORS:
            return False
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _concrete_input_shape(value: Any) -> tuple[int, ...]:
    """Return the concrete shape of an array-like estimation input.

    Args:
        value (Any): Array-like user input.

    Returns:
        tuple[int, ...]: Concrete dimensions, or an empty tuple when the value
        has no discoverable array shape.

    Raises:
        ValueError: If shape dimensions are not nonnegative integers or nested
            sequences have inconsistent shapes.
    """
    return _rectangular_array_shape(value)


def _opaque_cost_target_shapes(
    operation: InvokeOperation,
    resolver: ExprResolver,
    *,
    added_controls: int,
    declared_controls: int,
) -> dict[str, tuple[ResourceExpr, ...]]:
    """Resolve definition target shapes for an opaque-cost callback.

    Args:
        operation (InvokeOperation): Bodyless invocation being modeled.
        resolver (ExprResolver): Resolver for symbolic shapes.
        added_controls (int): Call-site control prefix excluded from the
            definition.
        declared_controls (int): Definition control prefix excluded from
            target shapes.

    Returns:
        dict[str, tuple[ResourceExpr, ...]]: Target shapes keyed by definition
        formal name. Scalar qubits use an empty tuple.
    """
    base_operands = operation.operands[added_controls:]
    signature = operation.definition.signature if operation.definition else None
    hints = signature.operands if signature is not None else []
    target_operands = base_operands[declared_controls:]
    target_hints = hints[declared_controls:]
    shapes: dict[str, tuple[ResourceExpr, ...]] = {}
    for index, operand in enumerate(target_operands):
        if not operand.type.is_quantum():
            continue
        hint = target_hints[index] if index < len(target_hints) else None
        name = hint.name if hint is not None else operand.name or f"target_{index}"
        shape = (
            tuple(resolver.resolve(dimension) for dimension in operand.shape)
            if isinstance(operand, ArrayValue)
            else ()
        )
        shapes[name] = shape
    return shapes
