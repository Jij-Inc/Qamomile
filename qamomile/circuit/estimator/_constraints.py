"""Derive symbolic resource constraints from quantum IR."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import sympy as sp

from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._quantum_values import _qubit_value_size
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_base import (
    _is_concrete_integer,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ResourceConstraint,
)
from qamomile.circuit.estimator._resource_expressions import (
    _piecewise,
    _safe_simplify,
)
from qamomile.circuit.ir._resource_contract import quantum_operand_widths
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.classical_ops import (
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.operation import (
    Operation,
)
from qamomile.circuit.ir.types.primitives import (
    BitType,
    UIntType,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
)


def _operation_array_constraints(
    operation: Operation,
    resolver: ExprResolver,
    *,
    active_when: sp.Basic = sp.true,
    proven_cache: dict[_ResourceConstraint, bool] | None = None,
) -> tuple[_ResourceConstraint, ...]:
    """Collect unresolved array-access requirements for one operation.

    Embedded element provenance is carried by ``Value.parent_array`` and
    ``Value.element_indices`` rather than by a dedicated load operation. This
    traversal follows those references, tuple/dict members, and array-view
    ancestry so every operation kind receives the same bounds checks. Classical
    stores and deferred quantum returns encode their indices as separate
    operands and are paired with their arrays explicitly below.

    Args:
        operation (Operation): IR operation whose direct values are inspected.
            Nested operation lists are evaluated in their own resolver scopes
            and are therefore not recursively walked here.
        resolver (ExprResolver): Resolver for array dimensions, element
            indices, and view affine-map expressions in the operation's scope.
        active_when (sp.Basic): Predicate under which the operation executes.
            Defaults to true.
        proven_cache (dict[_ResourceConstraint, bool] | None): Optional cache
            of previously validated requirements and whether their symbolic
            assumptions prove them. Defaults to ``None``.

    Returns:
        tuple[_ResourceConstraint, ...]: Deduplicated requirements that are not
            already provable from symbolic type assumptions.

    Raises:
        ValueError: If an access has the wrong number of indices, a view has
            malformed rank/affine metadata, or a concrete access is out of
            bounds.
    """
    constraints = list(
        _collect_array_value_constraints(
            (*operation.all_input_values(), *operation.results),
            resolver,
        )
    )

    if isinstance(operation, StoreArrayElementOperation):
        constraints.extend(
            _array_index_constraints(
                operation.array,
                operation.index_values,
                resolver,
                access_kind="store",
            )
        )
    elif isinstance(operation, ReturnQuantumArrayElementOperation):
        constraints.extend(
            _array_index_constraints(
                operation.array,
                operation.target_indices,
                resolver,
                access_kind="return target",
            )
        )
        constraints.extend(
            _array_index_constraints(
                operation.array,
                operation.source_indices,
                resolver,
                access_kind="return source",
            )
        )

    return _validated_unproven_array_constraints(
        tuple(constraint.when(active_when) for constraint in constraints),
        proven_cache=proven_cache,
    )


def _collect_array_value_constraints(
    values: Iterable[ValueBase],
    resolver: ExprResolver,
) -> tuple[_ResourceConstraint, ...]:
    """Collect unvalidated array requirements embedded in values.

    Args:
        values (Iterable[ValueBase]): Values whose element and view ancestry
            should be inspected.
        resolver (ExprResolver): Resolver for dimensions, indices, and affine
            view metadata.

    Returns:
        tuple[_ResourceConstraint, ...]: Raw, possibly duplicated structural
        requirements.
    """
    constraints: list[_ResourceConstraint] = []
    visited: set[str] = set()

    def visit(value: ValueBase) -> None:
        """Visit one value and its embedded array dependencies.

        Args:
            value (ValueBase): Value, array, tuple, or dictionary reference to
                inspect recursively.
        """
        if value.uuid in visited:
            return
        visited.add(value.uuid)

        if isinstance(value, TupleValue):
            for element in value.elements:
                visit(element)
            return
        if isinstance(value, DictValue):
            for key, entry_value in value.entries:
                visit(key)
                visit(entry_value)
            return
        if isinstance(value, ArrayValue):
            constraints.extend(_array_view_constraints(value, resolver))
            for dimension in value.shape:
                visit(dimension)
            if value.slice_of is not None:
                visit(value.slice_of)
            if value.slice_start is not None:
                visit(value.slice_start)
            if value.slice_step is not None:
                visit(value.slice_step)
            return
        if isinstance(value, Value):
            if value.parent_array is not None:
                constraints.extend(
                    _array_index_constraints(
                        value.parent_array,
                        value.element_indices,
                        resolver,
                        access_kind="element",
                    )
                )
                visit(value.parent_array)
            for index in value.element_indices:
                visit(index)

    for value in values:
        visit(value)
    return tuple(constraints)


def _array_index_constraints(
    array: ArrayValue,
    indices: Sequence[Value],
    resolver: ExprResolver,
    *,
    access_kind: str,
) -> tuple[_ResourceConstraint, ...]:
    """Build per-axis lower and upper bounds for one array access.

    Args:
        array (ArrayValue): Array whose local coordinate space is addressed.
        indices (Sequence[Value]): One index expression per array dimension.
        resolver (ExprResolver): Resolver for the index and shape expressions.
        access_kind (str): User-facing access role such as ``"element"`` or
            ``"store"`` used in diagnostic labels.

    Returns:
        tuple[_ResourceConstraint, ...]: Two constraints per array dimension.

    Raises:
        ValueError: If the number of indices differs from the array rank.
    """
    if len(indices) != len(array.shape):
        display_name = array.name or "array"
        raise ValueError(
            f"Array '{display_name}' {access_kind} access requires "
            f"{len(array.shape)} indices; got {len(indices)}."
        )

    display_name = array.name or "array"
    constraints: list[_ResourceConstraint] = []
    for axis, (index, dimension) in enumerate(zip(indices, array.shape, strict=True)):
        index_expression = resolver.resolve(index)
        dimension_expression = resolver.resolve(dimension)
        label = f"Array '{display_name}' {access_kind} axis {axis} index"
        constraints.extend(
            (
                _ResourceConstraint(
                    expression=index_expression,
                    minimum=0,
                    label=f"{label} lower bound",
                ),
                _ResourceConstraint(
                    expression=dimension_expression - index_expression,
                    minimum=1,
                    label=(
                        f"Array '{display_name}' {access_kind} axis {axis} "
                        "in-bounds margin (dimension - index)"
                    ),
                    unit="element",
                ),
            )
        )
    return tuple(constraints)


def _array_view_constraints(
    view: ArrayValue,
    resolver: ExprResolver,
) -> tuple[_ResourceConstraint, ...]:
    """Build affine-map and full-coverage requirements for one array view.

    A local element bound is insufficient for a view whose declared length
    extends beyond its parent. Requiring the final covered parent coordinate to
    remain in bounds also protects whole-view operations such as vector
    measurement, where no scalar element ``Value`` exists in the IR. Empty
    views make the coverage requirement vacuous because they address no parent
    slot.

    Args:
        view (ArrayValue): Array value that may be a strided one-dimensional
            view over another array.
        resolver (ExprResolver): Resolver for shape, start, and step values.

    Returns:
        tuple[_ResourceConstraint, ...]: Empty for a root array, otherwise the
            view length/start/step and nonempty coverage requirements.

    Raises:
        ValueError: If a sliced array lacks one-dimensional shape metadata or
            either affine-map operand.
    """
    if view.slice_of is None:
        return ()
    if (
        len(view.shape) != 1
        or len(view.slice_of.shape) != 1
        or view.slice_start is None
        or view.slice_step is None
    ):
        raise ValueError(
            f"Array view '{view.name or 'array'}' must have one-dimensional "
            "shape plus slice_start and slice_step metadata."
        )

    length = resolver.resolve(view.shape[0])
    parent_length = resolver.resolve(view.slice_of.shape[0])
    start = resolver.resolve(view.slice_start)
    step = resolver.resolve(view.slice_step)
    final_parent_index = start + step * (length - _ONE)
    coverage_slack = _piecewise(
        parent_length - final_parent_index,
        _ONE,
        sp.Gt(length, _ZERO),
    )
    display_name = view.name or "array"
    return (
        _ResourceConstraint(
            expression=length,
            minimum=0,
            label=f"Array view '{display_name}' length",
            unit="element",
        ),
        _ResourceConstraint(
            expression=start,
            minimum=0,
            label=f"Array view '{display_name}' start",
        ),
        _ResourceConstraint(
            expression=step,
            minimum=1,
            label=f"Array view '{display_name}' step",
        ),
        _ResourceConstraint(
            expression=coverage_slack,
            minimum=1,
            label=f"Array view '{display_name}' parent coverage",
            unit="element",
        ),
    )


def _validated_unproven_array_constraints(
    constraints: Sequence[_ResourceConstraint],
    *,
    proven_cache: dict[_ResourceConstraint, bool] | None = None,
) -> tuple[_ResourceConstraint, ...]:
    """Validate, prune, and deduplicate array structural requirements.

    Args:
        constraints (Sequence[_ResourceConstraint]): Fresh requirements from
            one operation's directly referenced values.
        proven_cache (dict[_ResourceConstraint, bool] | None): Optional cache
            of prior validation and proof results. Defaults to ``None``.

    Returns:
        tuple[_ResourceConstraint, ...]: Requirements that may still fail after
            later input substitution, preserving first-seen order.

    Raises:
        ValueError: If a requirement is already concretely invalid.
    """
    retained: list[_ResourceConstraint] = []
    seen: set[_ResourceConstraint] = set()
    for constraint in constraints:
        proven = proven_cache.get(constraint) if proven_cache is not None else None
        if proven is None:
            constraint.validate()
            proven = _array_constraint_is_proven(constraint)
            if proven_cache is not None:
                proven_cache[constraint] = proven
        if proven or constraint in seen:
            continue
        seen.add(constraint)
        retained.append(constraint)
    return tuple(retained)


def _array_constraint_is_proven(constraint: _ResourceConstraint) -> bool:
    """Return whether symbolic assumptions already prove a requirement.

    Args:
        constraint (_ResourceConstraint): Requirement to inspect after concrete
            validation.

    Returns:
        bool: Whether integrality, equality, and lower bounds are all proven
            without retaining the requirement for later substitution.
    """
    if constraint.ranges:
        return False
    expression = _safe_simplify(constraint.expression)
    integer_proven = (
        not constraint.integer
        or expression.is_integer is True
        or (expression.is_number and _is_concrete_integer(expression))
    )
    if not integer_proven:
        return False
    if constraint.finite and expression.is_finite is not True:
        return False
    if constraint.expected is not None:
        expected = _safe_simplify(constraint.expected)
        if _safe_simplify(expression - expected) != _ZERO:
            return False
    if constraint.minimum is not None:
        return constraint._minimum_relation(expression) is sp.true
    return True


def _quantum_operand_width_constraints(
    attrs: Mapping[str, Any],
    operands: Sequence[ValueBase],
    resolver: ExprResolver,
    *,
    source: str,
) -> tuple[_ResourceConstraint, ...]:
    """Decode exact quantum widths from callable resource metadata.

    Operand indices refer only to quantum operands, so controls added by a
    callable transform and interleaved classical parameters do not change the
    source callable's register contract. Legacy fixed-width opaque callables
    expose only ``num_target_qubits``; when no per-operand contract exists,
    that value constrains the sum of all quantum target operands.

    Args:
        attrs (Mapping[str, Any]): Callable definition or operation attrs.
        operands (Sequence[ValueBase]): Source-call quantum operands, possibly
            mixed with classical values.
        resolver (ExprResolver): Resolver for symbolic operand dimensions.
        source (str): Callable name used in diagnostics.

    Returns:
        tuple[_ResourceConstraint, ...]: Per-operand or legacy total-width
            requirements carried by the callable.

    Raises:
        ValueError: If present resource metadata is malformed or references a
            missing quantum operand.
    """
    quantum_operands = [
        operand
        for operand in operands
        if isinstance(operand, Value) and operand.type.is_quantum()
    ]
    constraints: list[_ResourceConstraint] = []
    operand_widths = quantum_operand_widths(attrs, source=source)
    for entry in operand_widths:
        if entry.index >= len(quantum_operands):
            raise ValueError(
                f"{source} resource contract references quantum operand "
                f"{entry.index}, "
                f"but the call has only {len(quantum_operands)} quantum operands."
            )
        constraints.append(
            _ResourceConstraint(
                expression=_qubit_value_size(
                    quantum_operands[entry.index],
                    resolver,
                ),
                minimum=None,
                expected=sp.Integer(entry.width),
                label=f"{source} {entry.name} register width",
                unit="qubit",
            )
        )
    legacy_total_width = attrs.get("num_target_qubits")
    if (
        not operand_widths
        and type(legacy_total_width) is int
        and legacy_total_width > 0
    ):
        constraints.append(
            _ResourceConstraint(
                expression=sum(
                    (
                        _qubit_value_size(operand, resolver)
                        for operand in quantum_operands
                    ),
                    _ZERO,
                ),
                minimum=None,
                expected=sp.Integer(legacy_total_width),
                label=f"{source} target register width",
                unit="qubit",
            )
        )
    return tuple(constraints)


def _block_input_constraints(
    block: Block,
    resolver: ExprResolver,
) -> tuple[_ResourceConstraint, ...]:
    """Return structural constraints for typed block inputs.

    Args:
        block (Block): Block whose formal quantum dimensions are constrained.
        resolver (ExprResolver): Resolver for symbolic array dimensions.

    Returns:
        tuple[_ResourceConstraint, ...]: Requirements for quantum-array shapes
        and scalar UInt/Bit parameter domains.
    """
    constraints = [
        _ResourceConstraint(
            expression=resolver.resolve(dimension),
            minimum=0,
            label=f"Quantum input '{value.name}' dimension {position}",
            unit="element",
        )
        for value in block.input_values
        if isinstance(value, ArrayValue) and value.type.is_quantum()
        for position, dimension in enumerate(value.shape)
    ]
    values_by_name = {
        value.name: value
        for value in (*block.input_values, *block.parameters.values())
        if isinstance(value, Value)
    }
    for slot in block.param_slots:
        if slot.ndim != 0 or not isinstance(slot.type, (UIntType, BitType)):
            continue
        value = values_by_name.get(slot.name)
        if value is None:
            continue
        expression = (
            (
                sp.Integer(int(slot.bound_value))
                if isinstance(slot.bound_value, bool)
                else cast(sp.Expr, sp.sympify(slot.bound_value))
            )
            if slot.bound_value is not None
            else resolver.resolve(value)
        )
        constraints.append(
            _ResourceConstraint(
                expression=expression,
                minimum=0,
                label=f"Parameter '{slot.name}'",
            )
        )
        if isinstance(slot.type, BitType):
            constraints.append(
                _ResourceConstraint(
                    expression=_ONE - expression,
                    minimum=0,
                    label=f"Bit parameter '{slot.name}' upper bound",
                )
            )
    return tuple(constraints)


def _block_output_constraints(
    block: Block,
    resolver: ExprResolver,
    *,
    active_when: sp.Basic = sp.true,
    proven_cache: dict[_ResourceConstraint, bool] | None = None,
) -> tuple[_ResourceConstraint, ...]:
    """Return array-access requirements carried only by block outputs.

    A returned array element need not appear in any operation operand. The
    output interface must therefore be inspected explicitly so a root return
    such as ``register[index]`` cannot bypass the same bounds validation used
    for ordinary operations and nested callable bodies.

    Args:
        block (Block): Block whose output values are inspected.
        resolver (ExprResolver): Resolver for output indices and array shapes.
        active_when (sp.Basic): Predicate under which the block output is
            reached. Defaults to true.
        proven_cache (dict[_ResourceConstraint, bool] | None): Optional cache
            of previously proved structural requirements. Defaults to
            ``None``.

    Returns:
        tuple[_ResourceConstraint, ...]: Validated requirements not already
            implied by symbolic type assumptions.

    Raises:
        ValueError: If an output access is malformed or concretely out of
            bounds.
    """
    return _validated_unproven_array_constraints(
        tuple(
            constraint.when(active_when)
            for constraint in _collect_array_value_constraints(
                block.output_values,
                resolver,
            )
        ),
        proven_cache=proven_cache,
    )
