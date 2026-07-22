"""Tests for explicit semantic-region capture normalization."""

import dataclasses

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    RegionArg,
)
from qamomile.circuit.ir.types.primitives import BitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes.region_capture import RegionCapturePass
from qamomile.circuit.transpiler.passes.region_validation import RegionValidationPass


def _uint(name: str) -> Value:
    """Create a symbolic UInt value for a test fixture.

    Args:
        name (str): Diagnostic value name.

    Returns:
        Value: Fresh symbolic UInt value.
    """
    return Value(type=UIntType(), name=name)


def _const(value: int, name: str) -> Value:
    """Create a constant UInt value for a test fixture.

    Args:
        value (int): Constant integer payload.
        name (str): Diagnostic value name.

    Returns:
        Value: Fresh constant UInt value.
    """
    return _uint(name).with_const(value)


def _range_loop_with_capture(*, yield_only: bool = False) -> Block:
    """Build a range loop that reads or directly yields an outer value.

    Args:
        yield_only (bool): Yield the outer value without a body read when
            ``True``. Defaults to ``False``.

    Returns:
        Block: Test block containing one range-loop region.
    """
    outer = _uint("outer")
    loop_var = _uint("index")
    block_arg = _uint("carry")
    yielded = outer if yield_only else _uint("next")
    body = (
        []
        if yield_only
        else [
            BinOp(
                kind=BinOpKind.ADD,
                operands=[block_arg, outer],
                results=[yielded],
            )
        ]
    )
    result = _uint("result")
    loop = ForOperation(
        operands=[_const(0, "start"), _const(2, "stop"), _const(1, "step")],
        results=[result],
        loop_var="index",
        loop_var_value=loop_var,
        operations=body,
        region_args=(
            RegionArg(
                var_name="carry",
                init=_const(0, "init"),
                block_arg=block_arg,
                yielded=yielded,
                result=result,
            ),
        ),
    )
    return Block(
        name="capture",
        kind=BlockKind.AFFINE,
        input_values=[outer],
        label_args=["outer"],
        output_values=[result],
        operations=[loop],
    )


def test_loop_body_read_becomes_explicit_capture() -> None:
    """A non-local body read is recorded once in first-use order."""
    normalized = RegionCapturePass().run(_range_loop_with_capture())
    loop = normalized.operations[0]
    assert isinstance(loop, ForOperation)

    assert [value.name for value in loop.captures] == ["outer"]
    (region,) = loop.nested_regions()
    assert region.captures == loop.captures
    assert [value.name for value in region.block_args] == ["index", "carry"]
    assert [value.name for value in region.yields] == ["next"]


def test_loop_invariant_yield_becomes_explicit_capture() -> None:
    """A yield-only outer dependency crosses the region boundary explicitly."""
    normalized = RegionCapturePass().run(_range_loop_with_capture(yield_only=True))
    loop = normalized.operations[0]
    assert isinstance(loop, ForOperation)

    assert [value.name for value in loop.captures] == ["outer"]


def test_branch_captures_are_kept_per_region() -> None:
    """True and false regions retain distinct first-use capture lists."""
    condition = Value(type=BitType(), name="condition")
    true_input = _uint("true_input")
    false_input = _uint("false_input")
    one = _const(1, "one")
    true_yield = _uint("true_yield")
    false_yield = _uint("false_yield")
    result = _uint("result")
    branch = IfOperation(
        operands=[condition],
        true_operations=[
            BinOp(
                kind=BinOpKind.ADD,
                operands=[true_input, one],
                results=[true_yield],
            )
        ],
        false_operations=[
            BinOp(
                kind=BinOpKind.ADD,
                operands=[false_input, one],
                results=[false_yield],
            )
        ],
    )
    branch.add_merge(true_yield, false_yield, result)
    block = Block(
        kind=BlockKind.AFFINE,
        input_values=[condition, true_input, false_input],
        label_args=["condition", "true_input", "false_input"],
        output_values=[result],
        operations=[branch],
    )

    normalized = RegionCapturePass().run(block)
    normalized_branch = normalized.operations[0]
    assert isinstance(normalized_branch, IfOperation)
    assert [value.name for value in normalized_branch.true_captures] == ["true_input"]
    assert [value.name for value in normalized_branch.false_captures] == ["false_input"]


def test_capture_normalization_is_idempotent_and_preserves_block_identity() -> None:
    """Repeated normalization preserves block and capture identity."""
    original = _range_loop_with_capture()
    unannotated_loop = original.operations[0]
    once = RegionCapturePass().run(original)
    twice = RegionCapturePass().run(once)
    once_loop = once.operations[0]
    twice_loop = twice.operations[0]
    assert isinstance(unannotated_loop, ForOperation)
    assert isinstance(once_loop, ForOperation)
    assert isinstance(twice_loop, ForOperation)

    assert once is original
    assert twice is original
    assert unannotated_loop.captures == ()
    assert [value.uuid for value in once_loop.captures] == [
        value.uuid for value in twice_loop.captures
    ]


def test_declared_aggregate_capture_resolves_ambiguous_outer_versions() -> None:
    """A frontend aggregate snapshot disambiguates multiple outer versions."""
    condition = Value(type=BitType(), name="condition")
    original = ArrayValue(type=UIntType(), name="items", shape=(_const(2, "size"),))
    previous = original.next_version()
    captured = previous.next_version()
    index = _const(0, "index")
    element = Value(
        type=UIntType(),
        name="items[0]",
        parent_array=captured,
        element_indices=(index,),
    )
    result = _uint("result")
    branch = IfOperation(
        operands=[condition],
        true_operations=[
            BinOp(
                kind=BinOpKind.ADD,
                operands=[element, _const(1, "one")],
                results=[result],
            )
        ],
        true_captures=(captured,),
    )
    block = Block(
        kind=BlockKind.AFFINE,
        input_values=[condition, original, previous],
        label_args=["condition", "original", "previous"],
        operations=[branch],
    )

    normalized = RegionCapturePass().run(block)
    normalized_branch = normalized.operations[0]
    assert isinstance(normalized_branch, IfOperation)
    assert normalized_branch.true_captures == (captured,)
    RegionValidationPass().run(normalized)


def test_ambiguous_outer_versions_use_latest_dominating_capture() -> None:
    """An undeclared aggregate read captures the latest outer SSA version."""
    condition = Value(type=BitType(), name="condition")
    original = ArrayValue(type=UIntType(), name="items", shape=(_const(2, "size"),))
    latest = original.next_version()
    branch_snapshot = latest.next_version()
    element = Value(
        type=UIntType(),
        name="items[0]",
        parent_array=branch_snapshot,
        element_indices=(_const(0, "index"),),
    )
    branch = IfOperation(
        operands=[condition],
        true_operations=[
            BinOp(
                kind=BinOpKind.ADD,
                operands=[element, _const(1, "one")],
                results=[_uint("result")],
            )
        ],
    )
    block = Block(
        kind=BlockKind.AFFINE,
        input_values=[condition, original, latest],
        label_args=["condition", "original", "latest"],
        operations=[branch],
    )

    normalized = RegionCapturePass().run(block)
    normalized_branch = normalized.operations[0]
    assert isinstance(normalized_branch, IfOperation)
    assert normalized_branch.true_captures == (latest,)
    RegionValidationPass().run(normalized)


def test_explicit_captures_round_trip_through_qkernel_serialization() -> None:
    """The protobuf graph preserves explicit loop capture references."""

    @qmc.qkernel
    def kernel(n: qmc.UInt) -> qmc.UInt:
        """Accumulate a captured parameter through a symbolic range."""
        total = qmc.uint(0)
        for _ in qmc.range(n):
            total = total + n
        return total

    kernel._block = RegionCapturePass().run(kernel.block)
    restored = deserialize(serialize(kernel))
    loop = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, ForOperation)
    )

    assert [value.name for value in loop.captures] == ["n"]


def test_region_verifier_rejects_an_undeclared_outer_read() -> None:
    """A body cannot read an enclosing value absent from captures."""
    block = RegionCapturePass().run(_range_loop_with_capture())
    loop = block.operations[0]
    assert isinstance(loop, ForOperation)
    block.operations[0] = dataclasses.replace(loop, captures=())

    with pytest.raises(ValidationError, match="before it is defined"):
        RegionValidationPass().run(block)


def test_region_verifier_rejects_branch_yield_arity_mismatch() -> None:
    """Both branches must yield one value for every if result."""
    condition = Value(type=BitType(), name="condition")
    result = Value(type=UIntType(), name="result")
    branch = IfOperation(
        operands=[condition],
        results=[result],
        true_yields=[],
        false_yields=[],
    )
    block = Block(input_values=[condition], operations=[branch])

    with pytest.raises(ValidationError, match="yields 0 values for 1 results"):
        RegionValidationPass().run(block)
