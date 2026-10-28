"""Runtime-array shape specialization across operation-owned transform bodies."""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.gate import ControlledUOperation, GateOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.value import ArrayValue
from qamomile.circuit.transpiler import QamomileCompiler

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

from qamomile.hugr import HugrExecutor, HugrTranspiler
from qamomile.hugr._shapes import resolve_parameter_shapes

pytestmark = pytest.mark.hugr


@qmc.qkernel
def _last_angle(qubit: qmc.Qubit, values: qmc.Vector[qmc.Float]) -> qmc.Qubit:
    """Rotate using the last runtime element selected by a dimension expression.

    Args:
        qubit (qmc.Qubit): Target qubit.
        values (qmc.Vector[qmc.Float]): Runtime rotation angles in radians.

    Returns:
        qmc.Qubit: Rotated target.
    """
    return qmc.rz(qubit, values[values.shape[0] - 1])


@qmc.qkernel
def _forward_angles(qubit: qmc.Qubit, values: qmc.Vector[qmc.Float]) -> qmc.Qubit:
    """Preserve a direct call inside an operation-owned transform body.

    Args:
        qubit (qmc.Qubit): Target qubit.
        values (qmc.Vector[qmc.Float]): Runtime rotation angles in radians.

    Returns:
        qmc.Qubit: Rotated target.
    """
    return _last_angle(qubit, values)


@qmc.qkernel
def _inverse_angles(
    values: qmc.Vector[qmc.Float],
) -> tuple[qmc.Bit, qmc.Vector[qmc.Float]]:
    """Invert a transitive helper while retaining runtime input contents.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime rotation angles in radians.

    Returns:
        tuple[qmc.Bit, qmc.Vector[qmc.Float]]: Target measurement and input angles.
    """
    target = qmc.qubit("target")
    target = qmc.h(target)
    target = qmc.inverse(_forward_angles)(target, values)
    return qmc.measure(qmc.h(target)), values


@qmc.qkernel
def _controlled_angles(
    values: qmc.Vector[qmc.Float],
) -> tuple[qmc.Bit, qmc.Vector[qmc.Float]]:
    """Control a transitive helper nested inside structured control flow.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime rotation angles in radians.

    Returns:
        tuple[qmc.Bit, qmc.Vector[qmc.Float]]: Target measurement and input angles.
    """
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    target = qmc.h(target)
    control = qmc.x(control)
    for _ in qmc.range(1):
        control, target = qmc.control(_forward_angles)(control, target, values)
    qmc.measure(control)
    return qmc.measure(qmc.h(target)), values


@qmc.qkernel
def _select_angles(values: qmc.Vector[qmc.Float]) -> tuple[qmc.Bit, qmc.Bit]:
    """Share one helper body between a SELECT case and a transitive direct call.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime rotation angles in radians.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Index and target measurements in semantic IR.
    """
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    target = qmc.h(target)
    control, target = qmc.select([_forward_angles, _last_angle])(
        control, target, values
    )
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _repeat_angles(qubit: qmc.Qubit, values: qmc.Vector[qmc.Float]) -> qmc.Qubit:
    """Use a dimension-derived loop bound inside a controlled body.

    Args:
        qubit (qmc.Qubit): Target qubit.
        values (qmc.Vector[qmc.Float]): Runtime rotation angles in radians.

    Returns:
        qmc.Qubit: Rotated target after the dimension-derived repetition count.
    """
    for _ in qmc.range(values.shape[0] - 1):
        qubit = _forward_angles(qubit, values)
    return qubit


@qmc.qkernel
def _controlled_loop_angles(
    values: qmc.Vector[qmc.Float],
) -> tuple[qmc.Bit, qmc.Vector[qmc.Float]]:
    """Execute a transitive controlled helper with a dimension-derived loop.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime rotation angles in radians.

    Returns:
        tuple[qmc.Bit, qmc.Vector[qmc.Float]]: Target measurement and input angles.
    """
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    target = qmc.h(target)
    control = qmc.x(control)
    control, target = qmc.control(_repeat_angles)(control, target, values)
    qmc.measure(control)
    return qmc.measure(qmc.h(target)), values


@qmc.qkernel
def _conflicting_transform_angles(
    left: qmc.Vector[qmc.Float], right: qmc.Vector[qmc.Float]
) -> qmc.Bit:
    """Call the same helper through controlled and direct ABI boundaries.

    Args:
        left (qmc.Vector[qmc.Float]): Angles passed through the controlled call.
        right (qmc.Vector[qmc.Float]): Angles passed through the direct call.

    Returns:
        qmc.Bit: Target measurement.
    """
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    target = qmc.h(target)
    control, target = qmc.control(_forward_angles)(control, target, left)
    target = _forward_angles(target, right)
    qmc.measure(control)
    return qmc.measure(target)


def _reachable_blocks(block: Block) -> Iterator[Block]:
    """Inspect transform bodies and called helpers while preserving shared identity.

    Args:
        block (Block): Entrypoint whose reachable semantic bodies are inspected.

    Returns:
        Iterator[Block]: Each reachable body once, including inverse fallbacks.
    """
    seen = set()
    pending = [block]
    while pending:
        body = pending.pop()
        if id(body) in seen:
            continue
        seen.add(id(body))
        yield body
        operations = list(body.operations)
        while operations:
            operation = operations.pop()
            if isinstance(operation, InvokeOperation) and operation.body is not None:
                pending.append(operation.body)
            elif isinstance(operation, SelectOperation):
                pending.extend(operation.case_blocks)
            elif isinstance(operation, ControlledUOperation):
                if operation.block is not None:
                    pending.append(operation.block)
            elif isinstance(operation, InverseBlockOperation):
                pending.extend(
                    nested
                    for nested in (
                        operation.source_block,
                        operation.implementation_block,
                    )
                    if nested is not None
                )
            elif isinstance(operation, HasNestedOps):
                for region in operation.nested_regions():
                    operations.extend(region.operations)


@pytest.mark.parametrize(
    "kernel", [_inverse_angles, _controlled_angles, _select_angles]
)
def test_shape_specialization_reaches_transform_bodies_and_transitive_calls(kernel):
    """Specialize each owned body without mutating the source.

    This test inspects semantic IR to isolate shape propagation through owned
    bodies and transitive calls, including preservation of the original program.
    Native SELECT lowering and execution are covered in test_select_execution.py.
    """
    original = QamomileCompiler().prepare(kernel, parameters=["values"])
    for length in (2, 3):
        resolved = resolve_parameter_shapes(original, {"values": (length,)})
        bodies = list(_reachable_blocks(resolved.entrypoint))
        assert len(bodies) >= 3
        for body in bodies:
            arrays = [
                value for value in body.input_values if isinstance(value, ArrayValue)
            ]
            assert arrays
            for value in arrays:
                assert value.shape[0].get_const() == length
                assert not value.is_constant()
            for operation in body.operations:
                if isinstance(operation, GateOperation):
                    for value in operation.operands:
                        if value.element_indices:
                            assert value.element_indices[0].get_const() == length - 1
                            assert not value.is_constant()
    for body in _reachable_blocks(original.entrypoint):
        for value in body.input_values:
            if isinstance(value, ArrayValue):
                assert not value.shape[0].is_constant()


def test_select_shape_specialization_preserves_shared_body_identity():
    """One SELECT case and a direct call still share the same specialized helper."""
    original = QamomileCompiler().prepare(_select_angles, parameters=["values"])
    resolved = resolve_parameter_shapes(original, {"values": (2,)})
    select = next(
        operation
        for operation in resolved.entrypoint.operations
        if isinstance(operation, SelectOperation)
    )
    call = next(
        operation
        for operation in select.case_blocks[0].operations
        if isinstance(operation, InvokeOperation)
    )
    assert call.body is select.case_blocks[1]


@pytest.mark.parametrize(
    "kernel", [_inverse_angles, _controlled_angles, _controlled_loop_angles]
)
def test_transformed_runtime_arrays_execute_with_dimension_derived_structure(kernel):
    """One executable selects runtime tail values through fixed transformed shapes."""
    executable = HugrTranspiler().transpile(
        kernel, parameters=["values"], parameter_shapes={"values": (2,)}
    )
    executor = HugrExecutor()
    for values, expected in [((0.0, math.pi), True), ((math.pi, 0.0), False)]:
        measured, returned_values = executable.run(
            executor, bindings={"values": values}
        ).result()
        assert measured is expected
        assert returned_values == pytest.approx(values, rel=1e-12, abs=1e-12)


def test_conflicting_shapes_across_transform_and_direct_calls_are_rejected():
    """A transformed call cannot silently specialize a shared helper inconsistently."""
    original = QamomileCompiler().prepare(
        _conflicting_transform_angles, parameters=["left", "right"]
    )
    with pytest.raises(ValueError, match="Conflicting HUGR callable shapes"):
        resolve_parameter_shapes(original, {"left": (2,), "right": (3,)})
