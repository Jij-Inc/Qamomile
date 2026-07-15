"""Regression tests for loop RegionArg identity validation."""

from __future__ import annotations

import pytest

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.canonical import canonicalize
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    RegionArg,
    validate_region_args,
)
from qamomile.circuit.ir.serialize import from_dict, to_dict
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import DictValue, Value
from qamomile.circuit.transpiler.classical_executor import ClassicalExecutor
from qamomile.circuit.transpiler.emit_context import EmitContext
from qamomile.circuit.transpiler.errors import (
    EmitError,
    ExecutionError,
    ValidationError,
)
from qamomile.circuit.transpiler.execution_context import ExecutionContext
from qamomile.circuit.transpiler.passes.constant_fold import ConstantFoldingPass
from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
    _seed_region_args,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)
from qamomile.circuit.transpiler.segments import ClassicalSegment


class _StubEmitPass:
    """Provide the resolver surface needed by RegionArg seed validation."""

    def __init__(self) -> None:
        """Create a stub with no runtime parameters."""
        self._resolver = ValueResolver()

    def _get_or_create_parameter(self, key: str, value_uuid: str) -> object:
        """Reject an unexpected symbolic-parameter request.

        Args:
            key (str): Requested parameter key.
            value_uuid (str): IR value identity requesting the parameter.

        Returns:
            object: Never returns.

        Raises:
            AssertionError: Always; these tests use concrete initializers.
        """
        raise AssertionError(f"unexpected parameter {key!r} for {value_uuid}")


def _uint_const(value: int, name: str) -> Value:
    """Create a constant UInt value.

    Args:
        value (int): Constant payload.
        name (str): Display label.

    Returns:
        Value: Constant UInt IR value.
    """
    return Value(type=UIntType(), name=name).with_const(value)


def _range_carry_block(*, collide_with_loop_var: bool) -> Block:
    """Build a range carry block, optionally corrupting its formal identity.

    Args:
        collide_with_loop_var (bool): Reuse the loop-variable UUID as the
            RegionArg block argument when true.

    Returns:
        Block: Affine block containing the requested loop shape.
    """
    loop_var = Value(type=UIntType(), name="i")
    block_arg = (
        loop_var if collide_with_loop_var else Value(type=UIntType(), name="carry")
    )
    yielded = Value(type=UIntType(), name="yielded")
    result = Value(type=UIntType(), name="result")
    body = BinOp(
        kind=BinOpKind.ADD,
        operands=[block_arg, _uint_const(1, "one")],
        results=[yielded],
    )
    loop = ForOperation(
        loop_var="i",
        loop_var_value=loop_var,
        operands=[
            _uint_const(0, "start"),
            _uint_const(2, "stop"),
            _uint_const(1, "step"),
        ],
        operations=[body],
        region_args=(
            RegionArg(
                var_name="carry",
                init=_uint_const(10, "init"),
                block_arg=block_arg,
                yielded=yielded,
                result=result,
            ),
        ),
        results=[result],
    )
    return Block(
        name="region_identity",
        kind=BlockKind.AFFINE,
        operations=[loop],
        output_values=[result],
    )


def test_loop_variable_region_collision_rejected_at_every_consumer() -> None:
    """All public consumers reject a loop-var/RegionArg UUID collision."""
    block = _range_carry_block(collide_with_loop_var=True)
    loop = block.operations[0]
    assert isinstance(loop, ForOperation)

    with pytest.raises(ValueError, match="disjoint from loop-variable"):
        validate_region_args(loop)
    with pytest.raises(ValueError, match="disjoint from loop-variable"):
        to_dict(block)
    with pytest.raises(ValueError, match="disjoint from loop-variable"):
        canonicalize(block)
    with pytest.raises(ValidationError, match="disjoint from loop-variable"):
        ConstantFoldingPass().run(block)
    with pytest.raises(ExecutionError, match="disjoint from loop-variable"):
        ClassicalExecutor().execute(
            ClassicalSegment(operations=[loop]),
            ExecutionContext(),
        )
    with pytest.raises(EmitError, match="disjoint from loop-variable"):
        _seed_region_args(_StubEmitPass(), loop, EmitContext())


def test_decoder_rejects_loop_variable_region_collision() -> None:
    """Wire decoding rejects a RegionArg block ref changed to the loop var."""
    payload = to_dict(_range_carry_block(collide_with_loop_var=False))
    loop_payload = payload["block"]["operations"][0]
    loop_payload["region_args"][0]["block_arg_ref"] = loop_payload["loop_var_value_ref"]

    with pytest.raises(ValueError, match="disjoint from loop-variable"):
        from_dict(payload)


@pytest.mark.parametrize(
    ("formal", "slot"),
    [("key", "block"), ("key", "result"), ("value", "block"), ("value", "result")],
)
def test_for_items_formals_are_disjoint_from_region_slots(
    formal: str,
    slot: str,
) -> None:
    """ForItems key/value UUIDs cannot double as carry definitions."""
    key = Value(type=UIntType(), name="key")
    item_value = Value(type=FloatType(), name="item")
    selected_formal = key if formal == "key" else item_value
    slot_type = selected_formal.type
    init = Value(type=slot_type, name="init").with_const(
        0 if isinstance(slot_type, UIntType) else 0.0
    )
    block_arg = (
        selected_formal if slot == "block" else Value(type=slot_type, name="carry")
    )
    yielded = block_arg
    result = (
        selected_formal if slot == "result" else Value(type=slot_type, name="result")
    )
    loop = ForItemsOperation(
        operands=[DictValue(name="items").with_dict_runtime_metadata({0: 1.0})],
        results=[result],
        key_vars=["key"],
        key_var_values=(key,),
        value_var="item",
        value_var_value=item_value,
        region_args=(RegionArg("carry", init, block_arg, yielded, result),),
    )

    with pytest.raises(ValueError, match="disjoint from loop-variable"):
        validate_region_args(loop)


def test_constant_fold_region_loop_is_wire_idempotent() -> None:
    """Repeated region-loop folding preserves UUID and logical-id identity."""
    block = _range_carry_block(collide_with_loop_var=False)
    folding = ConstantFoldingPass()

    once = folding.run(block)
    twice = folding.run(once)

    assert once.output_values[0].get_const() == 12
    assert to_dict(once) == to_dict(twice)
    assert once.output_values[0].logical_id == twice.output_values[0].logical_id
